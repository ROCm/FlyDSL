# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from .jit_argument import JitArgumentRegistry, from_dlpack
from .jit_function import jit
from .kernel_function import kernel

__all__ = [
    "from_dlpack",
    "JitArgumentRegistry",
    "jit",
    "kernel",
]


def from_jax(array, *, assumed_align=None, use_32bit_stride=False):
    """Convenience re-export of :func:`flydsl.jax.from_jax`.

    Available only when JAX is installed.
    """
    from ..jax.adapter import from_jax as _from_jax

    return _from_jax(array, assumed_align=assumed_align, use_32bit_stride=use_32bit_stride)
