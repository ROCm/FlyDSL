# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled MMA (matrix multiply accumulate) example using FlyDSL with JAX.

JAX equivalent of ``03-tiledMma.py``.  Demonstrates a single-tile GEMM
using MFMA instructions on AMD GPUs, running on JAX arrays.

Requirements:
    pip install jax[rocm]
"""

import jax
import jax.numpy as jnp
import numpy as np

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.jax import from_jax, jax_kernel

block_m = 64
block_n = 64
block_k = 8


# ---------- Kernel (identical to 03-tiledMma.py) ----------


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.zipped_divide(A, tileA)
    bB = fx.zipped_divide(B, tileB)
    bC = fx.zipped_divide(C, tileC)

    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    partition_A = thr_mma.partition_A(bA)
    partition_B = thr_mma.partition_B(bB)
    partition_C = thr_mma.partition_C(bC)

    frag_A = thr_mma.make_fragment_A(partition_A)
    frag_B = thr_mma.make_fragment_B(partition_B)
    frag_C = thr_mma.make_fragment_C(partition_C)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


# ---------- JIT launcher ----------


@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


# ---------- Eager ----------


def run_eager():
    M, N, K = block_m, block_n, block_k
    A = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(1), (N, K), dtype=jnp.float32)
    C = jnp.zeros((M, N), dtype=jnp.float32)

    tA = from_jax(A)
    tB = from_jax(B)
    tC = from_jax(C)

    jax.block_until_ready(A)
    jax.block_until_ready(B)
    tiledMma(tA, tB, tC)

    expected = np.asarray(A) @ np.asarray(B).T
    result = np.asarray(C)
    max_diff = np.max(np.abs(result - expected))
    is_correct = max_diff < 1e-5
    print(f"[Eager] Result correct: {is_correct}  (max diff: {max_diff:.2e})")
    if not is_correct:
        print("  Expected[:2,:4]:", expected[:2, :4])
        print("  Got[:2,:4]:     ", result[:2, :4])
    return is_correct


# ---------- jax.jit ----------


tiledMma_jax = jax_kernel(
    tiledMma,
    out_shapes=lambda a, b: [((a.shape[0], b.shape[0]), jnp.float32)],
)


def run_jit():
    M, N, K = block_m, block_n, block_k
    A = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(1), (N, K), dtype=jnp.float32)

    @jax.jit
    def f(a, b):
        (c,) = tiledMma_jax(a, b)
        return c

    C = f(A, B)

    expected = np.asarray(A) @ np.asarray(B).T
    result = np.asarray(C)
    max_diff = np.max(np.abs(result - expected))
    is_correct = max_diff < 1e-5
    print(f"[jax.jit] Result correct: {is_correct}  (max diff: {max_diff:.2e})")
    if not is_correct:
        print("  Expected[:2,:4]:", expected[:2, :4])
        print("  Got[:2,:4]:     ", result[:2, :4])
    return is_correct


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Tiled MMA (Eager)")
    print("=" * 50)
    ok1 = run_eager()

    print()
    print("=" * 50)
    print("Test 2: Tiled MMA (jax.jit)")
    print("=" * 50)
    try:
        ok2 = run_jit()
    except Exception as e:
        print(f"[jax.jit] FAILED: {e}")
        ok2 = False

    print(f"\nAll passed: {ok1 and ok2}")
