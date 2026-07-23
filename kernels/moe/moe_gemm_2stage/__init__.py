# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE 2-stage MFMA kernels (stage1 / stage2 / reduction).

Split from the former monolithic ``moe_gemm_2stage.py``; public API unchanged.

.. deprecated::
    These kernels use the legacy FlyDSL authoring API (``SmemAllocator`` /
    ``SmemPtr`` + raw ``buffer_ops``) and will be deprecated soon. New MoE work
    should use the current ``fx.*`` surface (``make_buffer_tensor`` +
    ``SharedAllocator`` + ``fx.copy`` / ``fx.gemm``); see ``kernels/moe/mxfp_moe``
    for the fused a4w4/a8w4 pipeline and the ``kernel-code-cleanup`` skill for the
    migration map.
"""

from kernels.moe.moe_gemm_2stage.gemm1 import compile_moe_gemm1
from kernels.moe.moe_gemm_2stage.gemm2 import (
    MoeGemm2Mode,
    _MoeGemm2ReduceWrapper,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
)
from kernels.moe.moe_gemm_2stage.reduction import compile_moe_reduction

__all__ = [
    "MoeGemm2Mode",
    "_MoeGemm2ReduceWrapper",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_gemm2_ex",
    "compile_moe_reduction",
]
