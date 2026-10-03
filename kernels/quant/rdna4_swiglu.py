# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Re-export. The SiLU×mul kernel lives in ``kernels.common.gfx120x_swiglu``.

It is an activation, not a quantizer. This path stays so existing imports work.
"""

from kernels.common.gfx120x_swiglu import (  # noqa: F401
    BLOCK,
    KERNEL_NAME,
    VEC_F32,
    VEC_HALF,
    build_silu_mul_module,
    build_swiglu_chunk_module,
)

__all__ = [
    "KERNEL_NAME",
    "BLOCK",
    "VEC_HALF",
    "VEC_F32",
    "build_silu_mul_module",
    "build_swiglu_chunk_module",
]
