# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled copy example using FlyDSL with JAX arrays.

JAX equivalent of ``02-tiledCopy.py``.  Demonstrates tiled copy with
partitioned tensors using the layout algebra DSL, running on JAX arrays.

Requirements:
    pip install jax[rocm]
"""

import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP: JAX not installed")
    sys.exit(0)

import numpy as np

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.jax import from_jax, jax_kernel


# ---------- Kernel (identical to 02-tiledCopy.py) ----------


@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    block_m = 8
    block_n = 24
    tile = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_n, 1)])

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.zipped_divide(A, tile)
    bB = fx.zipped_divide(B, tile)
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))

    thr_layout = fx.make_layout((4, 1), (1, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    layout_thr_val = fx.logical_product(thr_layout, val_layout)
    layout_thr_val = fx.raked_product(thr_layout, val_layout)

    tile_mn = fx.make_tile(4, 8)

    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    partition_src = thr_copy.partition_S(bA)
    partition_dst = thr_copy.partition_D(bB)

    frag = fx.make_fragment_like(partition_src)

    fx.copy(copy_atom, partition_src, frag)
    fx.copy(copy_atom, frag, partition_dst)


# ---------- JIT launcher ----------


@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_kernel(A, B).launch(grid=(15, 1, 1), block=(4, 1, 1), stream=stream)


# ---------- Eager ----------


def run_eager():
    M, N = 8 * 3, 24 * 5
    A = jnp.arange(M * N, dtype=jnp.float32).reshape(M, N)
    B = jnp.zeros((M, N), dtype=jnp.float32)

    tA = from_jax(A)
    tB = from_jax(B)

    jax.block_until_ready(A)
    tiledCopy(tA, tB)

    is_correct = np.allclose(np.asarray(A), np.asarray(B))
    print(f"[Eager] Result correct: {is_correct}")
    if not is_correct:
        print("  A[:2,:8]:", np.asarray(A)[:2, :8])
        print("  B[:2,:8]:", np.asarray(B)[:2, :8])
    return is_correct


# ---------- jax.jit ----------


tiledCopy_jax = jax_kernel(
    tiledCopy,
    out_shapes=lambda a: [(a.shape, a.dtype)],
)


def run_jit():
    M, N = 8 * 3, 24 * 5
    A = jnp.arange(M * N, dtype=jnp.float32).reshape(M, N)

    @jax.jit
    def f(a):
        (b,) = tiledCopy_jax(a)
        return b

    B = f(A)

    is_correct = np.allclose(np.asarray(A), np.asarray(B))
    print(f"[jax.jit] Result correct: {is_correct}")
    if not is_correct:
        print("  A[:2,:8]:", np.asarray(A)[:2, :8])
        print("  B[:2,:8]:", np.asarray(B)[:2, :8])
    return is_correct


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Tiled Copy (Eager)")
    print("=" * 50)
    ok1 = run_eager()

    print()
    print("=" * 50)
    print("Test 2: Tiled Copy (jax.jit)")
    print("=" * 50)
    try:
        ok2 = run_jit()
    except Exception as e:
        print(f"[jax.jit] FAILED: {e}")
        ok2 = False

    print(f"\nAll passed: {ok1 and ok2}")
