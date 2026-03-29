#!/usr/bin/env python3
"""Test that TVMFFIDispatcher guards catch compile-affecting arg changes.

Verifies that changing constexpr values or tensor dtypes between calls
produces correct results (guard triggers fallback to recompilation).
"""

import os
import torch

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_TVM_FFI", "1")

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vecAddKernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width
    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))
    copy_bits = vec_width * 32
    RABMemRefTy = fx.MemRefType.get(
        fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
    )
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)
    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vecAdd(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32, const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    vecAddKernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


def check(c, a, b, label):
    torch.cuda.synchronize()
    err = (c - (a + b)).abs().max().item()
    ok = err < 1e-5
    status = "PASS" if ok else "FAIL"
    print(f"  {label}: err={err:.2e} {status}")
    assert ok, f"{label} failed with err={err}"


def test_guard_constexpr_change():
    """Changing a Constexpr value must trigger recompilation, not reuse wrong kernel."""
    N = 1024 * 256
    BLOCK = 256
    VEC = 4

    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    stream = torch.cuda.current_stream()

    # Call 1: compile with BLOCK=256
    vecAdd(a, b, c, N, N, BLOCK, VEC, stream=stream)
    check(c, a, b, "Call 1 (BLOCK=256)")

    has_tvm = vecAdd._tvm_ffi_dispatch is not None
    print(f"  TVM FFI dispatcher active: {has_tvm}")

    # Call 2: same args → fast path
    c.zero_()
    vecAdd(a, b, c, N, N, BLOCK, VEC, stream=stream)
    check(c, a, b, "Call 2 (same args, fast path)")

    # Call 3: BLOCK=128 → constexpr changed, guard must catch
    c.zero_()
    vecAdd(a, b, c, N, N, 128, VEC, stream=stream)
    check(c, a, b, "Call 3 (BLOCK=128, constexpr changed)")

    # Call 4: back to BLOCK=256
    c.zero_()
    vecAdd(a, b, c, N, N, BLOCK, VEC, stream=stream)
    check(c, a, b, "Call 4 (BLOCK=256 again)")

    # Call 5: VEC=2 → another constexpr change
    c.zero_()
    vecAdd(a, b, c, N, N, BLOCK, 2, stream=stream)
    check(c, a, b, "Call 5 (VEC=2, constexpr changed)")


def test_guard_runtime_scalar_change():
    """Changing a runtime scalar (Int32) should NOT trigger recompilation."""
    N1 = 1024 * 128
    N2 = 1024 * 64
    BLOCK = 256
    VEC = 4

    a = torch.randn(N1, device="cuda", dtype=torch.float32)
    b = torch.randn(N1, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    stream = torch.cuda.current_stream()

    # Call with N1
    vecAdd(a, b, c, N1, N1, BLOCK, VEC, stream=stream)
    check(c[:N1], a[:N1], b[:N1], f"N={N1}")

    # Call with N2 (smaller) — same kernel, different runtime value
    c.zero_()
    vecAdd(a[:N2], b[:N2], c[:N2], N2, N2, BLOCK, VEC, stream=stream)
    check(c[:N2], a[:N2], b[:N2], f"N={N2}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: TVMFFIDispatcher guard validation")
    print("=" * 60)

    print("\n--- test_guard_constexpr_change ---")
    test_guard_constexpr_change()

    print("\n--- test_guard_runtime_scalar_change ---")
    test_guard_runtime_scalar_change()

    print("\nAll guard tests passed!")
