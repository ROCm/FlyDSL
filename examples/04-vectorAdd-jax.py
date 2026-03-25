# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector addition example using FlyDSL with JAX arrays.

This is the JAX equivalent of ``01-vectorAdd.py``.  It demonstrates both:

- **Level 1** (eager): wrapping JAX arrays via ``from_jax`` and calling
  a ``@flyc.jit`` function directly.
- **Level 2** (``jax.jit``): wrapping a ``@flyc.jit`` function with
  ``jax_kernel`` so it can be called inside ``jax.jit``.

Requirements:
    pip install jax[rocm]   # ROCm backend for AMD GPUs
"""

import sys

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP: JAX not installed")
    sys.exit(0)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.jax import from_jax, jax_kernel


# ---------- Kernel definition (identical to 01-vectorAdd.py) ----------


@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    fx.printf("[kernel] bid={}, tid={}", bid, tid)

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

    RABMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)

    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    copyAtomBuffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtomBuffer, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


# ---------- JIT launcher (identical to 01-vectorAdd.py) ----------


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    fx.printf("> vectorAdd: n={}, grid_x={}", n, grid_x)

    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream
    )


# ---------- JAX eager execution ----------


def run_eager_jax():
    """Eager-mode execution with JAX arrays."""
    n = 128

    # Create JAX arrays on the GPU.
    key = jax.random.PRNGKey(42)
    A = jax.random.randint(key, (n,), 0, 10).astype(jnp.float32)
    B = jax.random.randint(jax.random.PRNGKey(7), (n,), 0, 10).astype(jnp.float32)
    C = jnp.zeros(n, dtype=jnp.float32)

    # Wrap JAX arrays for FlyDSL.
    tA = from_jax(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    tB = from_jax(B)
    tC = from_jax(C)

    # Ensure JAX computations are complete before launching the kernel.
    jax.block_until_ready(A)
    jax.block_until_ready(B)

    # Launch kernel (uses default HIP stream).
    vectorAdd(tA, tB, tC, n, n + 1)

    # Synchronize and verify.
    # Note: C was written to in-place on the GPU.  We need to read it back.
    # Since FlyDSL wrote to C's device buffer directly, the JAX array C
    # still points to the same buffer.
    expected = A + B
    is_close = jnp.allclose(C, expected)
    print(f"[JAX Eager] Result correct: {is_close}")
    if not is_close:
        print("  A:", A[:16])
        print("  B:", B[:16])
        print("  C:", C[:16])
        print("  expected:", expected[:16])
    return bool(is_close)


# ---------- Level 2: jax.jit integration via jax_kernel ----------


# Wrap the @flyc.jit function so it can be used inside jax.jit.
# - out_shapes: tells JAX the shape and dtype of each output.
# - constexpr_kwargs: compile-time constants (Constexpr parameters).
# - runtime_scalars: non-tensor runtime args baked into the compiled kernel.
#   The scalar 'n' is traced with value 128 during FlyDSL compilation.
vectorAdd_jax = jax_kernel(
    vectorAdd,
    out_shapes=lambda a, b: [
        (a.shape, a.dtype),  # output C has same shape/dtype as A
    ],
    constexpr_kwargs={"const_n": 129},
    runtime_scalars={"n": 128},
)


def run_jit_jax():
    """jax.jit-compiled execution with JAX arrays.

    The FlyDSL kernel is compiled once and registered as an XLA custom call.
    Subsequent calls reuse the compiled kernel with zero Python overhead.
    """
    n = 128

    key = jax.random.PRNGKey(42)
    A = jax.random.randint(key, (n,), 0, 10).astype(jnp.float32)
    B = jax.random.randint(jax.random.PRNGKey(7), (n,), 0, 10).astype(jnp.float32)

    @jax.jit
    def add_vectors(a, b):
        # vectorAdd_jax receives only JAX arrays; scalar args (n) are baked
        # into the compiled kernel via runtime_scalars.
        (c,) = vectorAdd_jax(a, b)
        return c

    C = add_vectors(A, B)
    expected = A + B
    is_close = jnp.allclose(C, expected)
    print(f"[JAX jit] Result correct: {is_close}")
    if not is_close:
        print("  A:", A[:16])
        print("  B:", B[:16])
        print("  C:", C[:16])
        print("  expected:", expected[:16])
    return bool(is_close)


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: FlyDSL + JAX (Eager)")
    print("=" * 50)
    ok1 = run_eager_jax()

    print()
    print("=" * 50)
    print("Test 2: FlyDSL + JAX (jax.jit)")
    print("=" * 50)
    try:
        ok2 = run_jit_jax()
    except Exception as e:
        print(f"[JAX jit] FAILED with exception: {e}")
        ok2 = False

    print(f"\nAll passed: {ok1 and ok2}")
