#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CPU dispatch overhead benchmark: FlyDSL vs Triton vs PyTorch.

Measures pure CPU-side dispatch time (no GPU sync) for a small vec_add kernel.
This isolates the Python → JIT call overhead from actual GPU execution time.

Usage:
    python tests/pyir/test_launch_overhead.py
"""

import time
import torch

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# ─────────────────────────────────────────────────────────────────
# 1. FlyDSL vec_add
# ─────────────────────────────────────────────────────────────────
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vecAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
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
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    vecAddKernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


# ─────────────────────────────────────────────────────────────────
# 2. Triton vec_add
# ─────────────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl

    @triton.jit
    def triton_vec_add_kernel(a_ptr, b_ptr, c_ptr, n,
                              BLOCK: tl.constexpr = 1024):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def bench_wallclock(fn, n_warmup=20, n_iters=1000):
    """Measure wall-clock time per call (no GPU sync between calls).

    This measures CPU dispatch overhead: the time from Python calling the
    function to the function returning control to Python (kernel queued
    on GPU stream but not necessarily finished).
    """
    # warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # timed loop — no sync between iterations
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_iters * 1e6  # µs


def main():
    SIZE = 1024 * 256  # 256K elements — small enough that GPU time is trivial
    BLOCK = 256
    VEC = 4
    N_ITERS = 2000

    print("=" * 70)
    print("CPU Dispatch Overhead Benchmark (vec_add, no GPU sync per iter)")
    print(f"  SIZE={SIZE}  BLOCK={BLOCK}  VEC={VEC}  N_ITERS={N_ITERS}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    a = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    stream = torch.cuda.current_stream()

    # ── FlyDSL: compile once, then benchmark ──
    compiled = flyc.compile(vecAdd, a, b, c, SIZE, SIZE, BLOCK, VEC, stream)
    torch.cuda.synchronize()

    # verify correctness
    c.zero_()
    compiled(a, b, c, SIZE, SIZE, BLOCK, VEC, stream)
    torch.cuda.synchronize()
    ref = a + b
    err = (c - ref).abs().max().item()
    assert err < 1e-5, f"FlyDSL correctness failed: max_err={err}"

    # ── Bench flyc.compile'd function ──
    compiled_us = bench_wallclock(
        lambda: compiled(a, b, c, SIZE, SIZE, BLOCK, VEC, stream),
        n_iters=N_ITERS,
    )

    # ── Bench @flyc.jit (implicit path) ──
    flydsl_us = bench_wallclock(
        lambda: vecAdd(a, b, c, SIZE, SIZE, BLOCK, VEC, stream),
        n_iters=N_ITERS,
    )

    # ── Bench PyTorch ──
    torch_us = bench_wallclock(
        lambda: torch.add(a, b, out=c),
        n_iters=N_ITERS,
    )

    # ── Bench Triton ──
    triton_us = None
    if HAS_TRITON:
        grid = ((SIZE + 1023) // 1024,)
        # warmup triton
        triton_vec_add_kernel[grid](a, b, c, SIZE)
        torch.cuda.synchronize()

        triton_us = bench_wallclock(
            lambda: triton_vec_add_kernel[grid](a, b, c, SIZE),
            n_iters=N_ITERS,
        )

    # ── Results ──
    print()
    print(f"  {'Framework':<25s} {'Dispatch (µs)':>14s} {'vs PyTorch':>12s}")
    print("  " + "-" * 53)
    print(f"  {'PyTorch (torch.add)':<25s} {torch_us:>14.1f} {'1.00x':>12s}")
    print(f"  {'FlyDSL (flyc.compile)':<25s} {compiled_us:>14.1f} {compiled_us / torch_us:>11.1f}x")
    print(f"  {'FlyDSL (@flyc.jit)':<25s} {flydsl_us:>14.1f} {flydsl_us / torch_us:>11.1f}x")
    if triton_us is not None:
        print(f"  {'Triton (@triton.jit)':<25s} {triton_us:>14.1f} {triton_us / torch_us:>11.1f}x")
    print()


if __name__ == "__main__":
    main()
