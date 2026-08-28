# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE 2-stage fp8/int8 MFMA kernels (stage1 gate-up / stage2 down-projection / reduce).

Layout-API package. Inputs fp8 (E4M3) / int8 / int8smooth / int4 (W4A8) on
CDNA3/CDNA4 (gfx94*/gfx95*); outputs f16/bf16/f32. int8 variants share the fp8
pipeline (i32 MFMA acc, f32 dequant). Stage2 accumulates via atomics or a
separate topk-reduce kernel.
"""

from kernels.moe.moe_gemm_2stage.gemm1 import compile_moe_gemm1
from kernels.moe.moe_gemm_2stage.gemm2 import compile_moe_gemm2
from kernels.moe.moe_gemm_2stage.moe_reduce import compile_moe_reduction

__all__ = [
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
]
