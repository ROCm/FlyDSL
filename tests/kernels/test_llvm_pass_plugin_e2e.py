# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end test for the custom-LLVM-pass JIT path with a real pass plugin.

Builds a minimal LLVM new-PM pass plugin (``.so``) that registers a module pass
named ``flydsl-print-tid``.  The pass injects, at the entry of every
``amdgpu_kernel`` function, a device ``printf("...threadIdx.x=%d...", tid)`` call
(``tid`` from ``llvm.amdgcn.workitem.id.x``).  The kernel is then driven through
``@flyc.jit(llvm_pass_pipeline=..., llvm_pass_plugins=[...])`` so the full
``opt --load-pass-plugin`` -> re-codegen -> run chain is exercised, and the
injected device print is observed in the captured output.

Requires a ROCm GPU, ``FLYDSL_COMPILE_LLVM_DIR`` (for ``opt``/``mlir-translate``/
``mlir-opt`` + LLVM headers), and a host C++ compiler; skipped otherwise.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

torch = pytest.importorskip("torch")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.external_llvm import ExternalLLVMError  # noqa: E402

# Plugin sources live as standalone .cpp files (compiled at test time, see
# llvm_pass_plugins/).  print_tid_ir_pass.cpp registers "flydsl-print-tid":
# at each amdgpu_kernel entry it emits FlyDSL's hostcall device-printf
# sequence (__ockl_printf_*) printing threadIdx.x, matching the fx.printf ABI.
_PLUGINS_DIR = Path(__file__).parent / "llvm_pass_plugins"


def _gpu_available() -> bool:
    try:
        from flydsl.runtime.device import get_rocm_device_count

        return get_rocm_device_count() > 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def print_tid_plugin(tmp_path_factory) -> str:
    """Compile the print-tid pass plugin against the LLVM prefix whose ``opt``
    will load it (ABI must match), or skip if the toolchain is unavailable."""
    raw = os.environ.get("FLYDSL_COMPILE_LLVM_DIR", "").strip()
    if not raw:
        pytest.skip("FLYDSL_COMPILE_LLVM_DIR not set; required to build/load an LLVM pass plugin")
    prefix = Path(raw).expanduser().resolve()
    llvm_config = prefix / "bin" / "llvm-config"
    header = prefix / "include" / "llvm" / "Plugins" / "PassPlugin.h"
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not llvm_config.is_file() or not header.is_file() or cxx is None:
        pytest.skip("LLVM headers/llvm-config or a C++ compiler not available for plugin build")

    cxxflags = subprocess.check_output([str(llvm_config), "--cxxflags"], text=True).split()
    src = _PLUGINS_DIR / "print_tid_ir_pass.cpp"
    so = tmp_path_factory.mktemp("llvm_plugin") / "libFlydslPrintTid.so"
    subprocess.run([cxx, "-shared", "-fPIC", *cxxflags, str(src), "-o", str(so)], check=True)
    assert so.is_file()
    return str(so)


@flyc.kernel
def _add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, block_dim: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))
    ca = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    cab = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.copy_atom_call(cab, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(ca, fx.slice(tB, (None, tid)), rB)
    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(ca, rC, fx.slice(tC, (None, tid)))


def _make_add_jit(**jit_kwargs):
    @flyc.jit(**jit_kwargs)
    def add(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        block_dim = 64
        grid_x = (n + block_dim - 1) // block_dim
        _add_kernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)

    return add


@pytest.mark.skipif(not _gpu_available(), reason="requires a ROCm GPU")
def test_print_tid_plugin_injects_device_printf(print_tid_plugin, monkeypatch):
    """Positive: with the plugin loaded and the pipeline naming the plugin pass,
    the printf-injected kernel compiles, links (ockl), and runs correctly.

    The injected device ``printf`` (``flydsl-pass: threadIdx.x=...``, one line per
    lane) is written by the ROCm hostcall consumer to a file descriptor HIP
    cached at init, so pytest's in-process capture does not see it; run with
    ``pytest -s`` to view it on the terminal.  Compile + link + correct execution
    of the injected ``__ockl_printf_*`` IR is what this asserts."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    add = _make_add_jit(llvm_pass_pipeline="default<O0>,flydsl-print-tid", llvm_pass_plugins=[print_tid_plugin])

    n = 64  # one block of 64 lanes -> 64 valid threads
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    add(tA, B, C, n, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    assert torch.allclose(C, A + B)


def test_print_tid_pipeline_without_plugin_fails(print_tid_plugin, monkeypatch):
    """Negative: the same pipeline naming ``flydsl-print-tid`` *without* loading
    the plugin must fail at the ``opt`` step — proving the plugin provides the
    pass.  (Fails during compile, before any GPU execution.)"""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    add = _make_add_jit(llvm_pass_pipeline="default<O0>,flydsl-print-tid")  # no plugins

    n = 64
    A = torch.zeros(n, dtype=torch.float32)
    B = torch.zeros(n, dtype=torch.float32)
    C = torch.zeros(n, dtype=torch.float32)
    if _gpu_available():
        A, B, C = A.cuda(), B.cuda(), C.cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)

    with pytest.raises(ExternalLLVMError) as excinfo:
        add(tA, B, C, n, stream=torch.cuda.Stream() if _gpu_available() else fx.Stream(None))
    assert "flydsl-print-tid" in str(excinfo.value)
