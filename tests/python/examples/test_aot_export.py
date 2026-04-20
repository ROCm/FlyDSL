#!/usr/bin/env python3
"""Test AOT export: dump_to_object, export_to_c, and load_module.

- export_to_c produces .h + .o
- load_module accepts .o or .so
- Multiple .o files can be linked into one .so

Usage:
    PYTHONPATH=./ python tests/python/examples/test_aot_export.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


# ── Shared kernel (must be at module level for AST rewriter) ──

@flyc.kernel
def _copy_kernel(A: fx.Tensor, B: fx.Tensor):
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tA = fx.logical_divide(A, fx.make_layout(64, 1))
    tB = fx.logical_divide(B, fx.make_layout(64, 1))
    RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, fx.block_idx.x)), rA)
    fx.copy_atom_call(copyAtom, rA, fx.slice(tB, (None, fx.block_idx.x)))


@flyc.jit
def _copy_launch(A: fx.Tensor, B, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    grid_x = (n + 63) // 64
    _copy_kernel(A, B).launch(grid=(grid_x, 1, 1), block=[64, 1, 1], stream=stream)


def _compile_and_get_artifact():
    n = 64
    A = torch.rand(n, dtype=torch.float32).cuda()
    B = torch.zeros(n, dtype=torch.float32).cuda()
    _copy_launch(A, B, n, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    last = _copy_launch._last_compiled
    assert last is not None, "No compiled artifact found"
    return last[1]


# ── Tests ──

def test_dump_to_object():
    """dump_to_object produces a valid ELF with prefixed symbols."""
    print("=== test_dump_to_object ===")
    artifact = _compile_and_get_artifact()

    t0 = time.perf_counter()
    obj_bytes = artifact.dump_to_object("test_copy")
    export_ms = (time.perf_counter() - t0) * 1000

    assert obj_bytes[:4] == b"\x7fELF", f"Expected ELF, got {obj_bytes[:4]!r}"
    print(f"  {len(obj_bytes)} bytes, {export_ms:.0f} ms")

    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        f.write(obj_bytes)
        tmp = f.name
    try:
        r = subprocess.run(["readelf", "-sW", tmp], capture_output=True, text=True)
        if r.returncode == 0:
            assert "test_copy_entry_name" in r.stdout, "metadata missing"
            assert "test_copy_kernels" in r.stdout, "prefixed gpu binary missing"
            print("  symbols OK")
    finally:
        os.unlink(tmp)
    print("  PASSED")


def test_export_to_c():
    """export_to_c produces .h + .o."""
    print("\n=== test_export_to_c ===")
    artifact = _compile_and_get_artifact()
    d = tempfile.mkdtemp(prefix="flydsl_export_")
    try:
        artifact.export_to_c(d, "kernel", "copy_fp32")

        h = os.path.join(d, "kernel.h")
        o = os.path.join(d, "kernel.o")
        assert os.path.exists(h), ".h missing"
        assert os.path.exists(o), ".o missing"

        with open(h) as f:
            header = f.read()
        assert "_mlir_copy_fp32_" in header
        assert "copy_fp32_launch" in header
        assert "hip/hip_runtime.h" in header
        print(f"  .h: {len(header)} chars, .o: {os.path.getsize(o)} bytes")
        print("  PASSED")
    finally:
        shutil.rmtree(d)


def test_load_module_from_o():
    """load_module(".o") loads via LLVM JITLink."""
    print("\n=== test_load_module_from_o ===")
    artifact = _compile_and_get_artifact()
    d = tempfile.mkdtemp(prefix="flydsl_load_")
    try:
        obj_bytes = artifact.dump_to_object("load_test")
        o_path = os.path.join(d, "kernel.o")
        with open(o_path, "wb") as f:
            f.write(obj_bytes)

        from flydsl.compiler.export import load_module
        mod = load_module(o_path)

        fn = mod.get_function("load_test")
        fn2 = mod.load_test  # __getattr__
        assert fn2 is fn
        print("  get_function + __getattr__: OK")

        func_name = mod._read_string_global("load_test_entry_name")
        version = mod._read_string_global("load_test_fly_version")
        arch = mod._read_string_global("load_test_gpu_arch")
        print(f"  entry_name: {func_name}")
        print(f"  fly_version: {version}, gpu_arch: {arch}")
        assert func_name and version and arch
        print("  PASSED")
    finally:
        shutil.rmtree(d)


def test_load_module_from_so():
    """load_module(".so") works with pre-linked .so."""
    print("\n=== test_load_module_from_so ===")
    artifact = _compile_and_get_artifact()
    d = tempfile.mkdtemp(prefix="flydsl_so_")
    try:
        # Export .o, then user links to .so
        artifact.export_to_c(d, "kernel", "so_test")
        o_path = os.path.join(d, "kernel.o")

        from flydsl.compiler.jit_executor import _resolve_runtime_libs
        from pathlib import Path
        lib_dir = str(Path(_resolve_runtime_libs()[0]).parent)
        so_path = os.path.join(d, "libkernel.so")
        subprocess.check_call([
            "gcc", "-shared", "-o", so_path, o_path,
            f"-L{lib_dir}", "-lfly_jit_runtime", "-lmlir_c_runner_utils",
            f"-Wl,-rpath,{lib_dir}",
        ])

        from flydsl.compiler.export import load_module
        mod = load_module(so_path)
        fn = mod.get_function("so_test")
        assert fn is not None
        print(f"  .so loaded, function found")
        print("  PASSED")
    finally:
        shutil.rmtree(d)


def test_gemm_aot_roundtrip():
    """AOT round-trip for preshuffle GEMM in subprocess (avoids GPU state pollution)."""
    result = subprocess.run(
        [sys.executable, "-u", "-c", f"""
import sys; sys.path.insert(0, {_REPO_ROOT!r})
import torch
import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.utils import pertoken_quant, shuffle_weight
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

arch = get_rocm_arch()
FP8 = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz
M, N, K = 32, 1024, 2048
a_q, sa = pertoken_quant(torch.rand(M, K, device="cuda"), quant_dtype=FP8)
b_q, sb = pertoken_quant(torch.rand(N, K, device="cuda"), quant_dtype=FP8)
b_shuf = shuffle_weight(b_q, layout=(16, 16))
c = torch.zeros((M, N), dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

launch_fn = compile_preshuffle_gemm_a8(M=M, N=N, K=K, tile_m=32, tile_n=64, tile_k=512, in_dtype="fp8")
flyc.compile(launch_fn, c.view(-1), a_q.view(torch.int8).view(-1),
    b_shuf.view(torch.int8).view(-1), sa.view(-1), sb.view(-1), M, N, stream)

import tempfile, os
artifact = launch_fn._last_compiled[1]
d = tempfile.mkdtemp()
obj = artifact.dump_to_object("gemm_fp8")
assert obj[:4] == b"\\x7fELF", f"Bad ELF: {{obj[:4]!r}}"
o_path = os.path.join(d, "gemm.o")
with open(o_path, "wb") as f: f.write(obj)

from flydsl.compiler.export import load_module
mod = load_module(o_path)
fn = mod.get_function("gemm_fp8")
assert fn is not None
entry = mod._read_string_global("gemm_fp8_entry_name")
ver = mod._read_string_global("gemm_fp8_fly_version")
assert entry and ver
print(f"GEMM AOT: {{len(obj)}} bytes, entry={{entry}}, ver={{ver}}")
import shutil; shutil.rmtree(d)
"""],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "0", "PYTHONPATH": _REPO_ROOT},
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:500]}")
    assert result.returncode == 0, f"Subprocess failed (rc={result.returncode})"


if __name__ == "__main__":
    test_gemm_aot_roundtrip()
    test_dump_to_object()
    test_export_to_c()
    test_load_module_from_o()
    test_load_module_from_so()
    print("\nALL PASSED")
