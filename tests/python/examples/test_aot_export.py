#!/usr/bin/env python3
"""Test AOT export: dump_to_object and load_module."""

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


def _compile_copy_artifact():
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
    print("=== test_dump_to_object ===")
    artifact = _compile_copy_artifact()

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


def test_dump_to_object_with_output_path():
    print("\n=== test_dump_to_object_with_output_path ===")
    artifact = _compile_copy_artifact()
    d = tempfile.mkdtemp(prefix="flydsl_export_")
    try:
        o_path = os.path.join(d, "kernel.o")
        obj_bytes = artifact.dump_to_object("copy_fp32", output_path=o_path)

        assert os.path.exists(o_path), ".o missing"
        assert obj_bytes[:4] == b"\x7fELF", f"Expected ELF, got {obj_bytes[:4]!r}"
        with open(o_path, "rb") as f:
            assert f.read() == obj_bytes, "file content != returned bytes"
        print(f"  {len(obj_bytes)} bytes written to {o_path}")
        print("  PASSED")
    finally:
        shutil.rmtree(d)


def test_load_module_from_o():
    print("\n=== test_load_module_from_o ===")
    artifact = _compile_copy_artifact()
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
    print("\n=== test_load_module_from_so ===")
    artifact = _compile_copy_artifact()
    d = tempfile.mkdtemp(prefix="flydsl_so_")
    try:
        o_path = os.path.join(d, "kernel.o")
        artifact.dump_to_object("so_test", output_path=o_path)

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


def _run_gemm_aot_subprocess():
    """Run GEMM AOT test in subprocess (isolates GPU state from dump_to_object)."""
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

# Data prep (same pattern as test_preshuffle_gemm.py)
a_q, sa = pertoken_quant(torch.rand(M, K, device="cuda"), quant_dtype=FP8)
b_q, sb = pertoken_quant(torch.rand(N, K, device="cuda"), quant_dtype=FP8)
b_shuf = shuffle_weight(b_q, layout=(16, 16))
c = torch.zeros((M, N), dtype=torch.float16, device="cuda")
dummy_bias = torch.empty(0, dtype=torch.float16, device="cuda")
stream = torch.cuda.current_stream()

def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t

launch_fn = compile_preshuffle_gemm_a8(
    M=M, N=N, K=K, tile_m=32, tile_n=64, tile_k=512, in_dtype="fp8")
flyc.compile(launch_fn,
    c.view(-1), _as_i8(a_q).view(-1), _as_i8(b_shuf).view(-1),
    sa.view(-1), sb.view(-1), dummy_bias, M, N, stream)

# dump_to_object round-trip
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
gpu_arch = mod._read_string_global("gemm_fp8_gpu_arch")
assert entry and ver and gpu_arch
print(f"GEMM AOT: {{len(obj)}} bytes, entry={{entry}}, ver={{ver}}, arch={{gpu_arch}}")
import shutil; shutil.rmtree(d)
"""],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "FLYDSL_RUNTIME_ENABLE_CACHE": "0", "PYTHONPATH": _REPO_ROOT},
    )
    return result


def test_gemm_dump_to_object():
    print("\n=== test_gemm_dump_to_object ===")
    result = _run_gemm_aot_subprocess()
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:500]}")
    assert result.returncode == 0, f"Subprocess failed (rc={result.returncode})"
    print("  PASSED")


if __name__ == "__main__":
    test_gemm_dump_to_object()
    test_dump_to_object()
    test_dump_to_object_with_output_path()
    test_load_module_from_o()
    test_load_module_from_so()
    print("\nALL PASSED")
