# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE 2-stage fp8 MFMA kernels (stage1 gate-up / stage2 down-projection / reduce).

fp8-only, layout-API package (make_buffer_tensor + SharedAllocator + tiled
copy/MMA). Inputs are fp8 (E4M3) on CDNA3 (gfx94*) / CDNA4 (gfx95*); outputs
are f16/bf16/f32. Stage2 supports atomic accumulation or a separate reduce
kernel over the topk dimension.
"""

from kernels.moe.moe_gemm_2stage.gemm1 import compile_moe_gemm1
from kernels.moe.moe_gemm_2stage.gemm2 import MoeGemm2Mode, compile_moe_gemm2
from kernels.moe.moe_gemm_2stage.moe_reduce import compile_moe_reduction

__all__ = [
    "MoeGemm2Mode",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
]
