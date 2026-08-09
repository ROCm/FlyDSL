# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE 2-stage fp8 MFMA kernels (stage1 gate-up / stage2 down-projection / reduce).

fp8-only layout-API package. Inputs are fp8 (E4M3) on CDNA3 (gfx94*) / CDNA4
(gfx95*); outputs f16/bf16/f32. Stage2 accumulates via atomics or a separate
topk-reduce kernel.
"""

from kernels.moe.moe_gemm_2stage.gemm1 import compile_moe_gemm1
from kernels.moe.moe_gemm_2stage.gemm2 import compile_moe_gemm2
from kernels.moe.moe_gemm_2stage.moe_reduce import compile_moe_reduction

__all__ = [
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
]
