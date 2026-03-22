#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Minimal vector-add test using JAX arrays with FlyDSL.

Demonstrates the eager (Level 1) integration: JAX arrays are wrapped
via ``from_jax`` and passed directly to a ``@flyc.jit`` function.
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pytest.skip("JAX not installed", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.jax import from_jax


# ── Kernel (same as tests/kernels/test_vec_add.py) ─────────────────────

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


# ── JIT launcher ────────────────────────────────────────────────────────

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


# ── Test ────────────────────────────────────────────────────────────────

def main():
    BLOCK_DIM = 256
    VEC_WIDTH = 4
    TILE = BLOCK_DIM * VEC_WIDTH
    N = TILE * 100  # 102400 elements, aligned to tile

    print(f"JAX devices: {jax.devices()}")
    print(f"Vector add: N={N}, block={BLOCK_DIM}, vec_width={VEC_WIDTH}")

    # Create JAX arrays on GPU.
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (N,), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(7), (N,), dtype=jnp.float32)
    C = jnp.zeros(N, dtype=jnp.float32)

    # Wrap for FlyDSL.
    tA = from_jax(A).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
    tB = from_jax(B)
    tC = from_jax(C)

    # Ensure JAX computations are done before kernel launch.
    jax.block_until_ready(A)
    jax.block_until_ready(B)

    # Launch FlyDSL kernel.
    print("Compiling and launching kernel...")
    vecAdd(tA, tB, tC, N, N, BLOCK_DIM, VEC_WIDTH)

    # Synchronize (FlyDSL kernel runs on default HIP stream).
    # Use a HIP device sync via JAX.
    jax.block_until_ready(C)

    # Verify.
    expected = np.asarray(A) + np.asarray(B)
    result = np.asarray(C)
    max_err = np.max(np.abs(result - expected))
    print(f"Max error: {max_err:.2e}")

    if max_err < 1e-5:
        print("PASSED")
    else:
        print("FAILED")
        print(f"  A[:8] = {np.asarray(A)[:8]}")
        print(f"  B[:8] = {np.asarray(B)[:8]}")
        print(f"  C[:8] = {result[:8]}")
        print(f"  expected[:8] = {expected[:8]}")
        return False
    return True


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
