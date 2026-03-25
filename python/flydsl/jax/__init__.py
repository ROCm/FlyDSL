# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL JAX integration.

Provides two levels of integration:

Level 1 — Eager mode (``from_jax``):
    Wrap JAX arrays as FlyDSL JitArguments so they can be passed directly to
    ``@flyc.jit`` functions.  Requires ``jax.block_until_ready()`` for
    synchronization.

Level 2 — ``jax.jit`` integration (``jax_kernel``):
    Register compiled FlyDSL kernels as JAX primitives via the XLA FFI so they
    compose with ``jax.jit``.

Usage (eager)::

    import jax.numpy as jnp
    import flydsl.compiler as flyc
    from flydsl.jax import from_jax

    a = jnp.ones(1024, dtype=jnp.float32)
    ta = from_jax(a)
    my_jit_func(ta, ...)

Usage (jax.jit)::

    from flydsl.jax import jax_kernel

    wrapped = jax_kernel(
        my_flyc_jit_func,
        out_shapes=lambda a, b: [(a.shape, a.dtype)],
    )

    @jax.jit
    def f(a, b):
        (c,) = wrapped(a, b)
        return c
"""

from flydsl.jax.adapter import JaxTensorAdaptor, from_jax

# Lazy imports: these modules depend on torch (via flydsl.compiler) and
# JAX internal MLIR dialects, so we defer them to avoid hard import-time
# failures in environments where torch is not installed.


def compile_and_register(*args, **kwargs):
    """Lazy wrapper for :func:`flydsl.jax.ffi_bridge.compile_and_register`."""
    from flydsl.jax.ffi_bridge import compile_and_register as _car

    return _car(*args, **kwargs)


def jax_kernel(*args, **kwargs):
    """Lazy wrapper for :func:`flydsl.jax.primitive.jax_kernel`."""
    from flydsl.jax.primitive import jax_kernel as _jk

    return _jk(*args, **kwargs)


__all__ = [
    "compile_and_register",
    "from_jax",
    "JaxTensorAdaptor",
    "jax_kernel",
]
