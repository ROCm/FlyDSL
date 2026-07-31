# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4 (bf16 A x mxfp4 W) 2-stage MoE kernels.

Standalone CDNA4 (gfx950) MFMA pipeline, extracted from ``mxfp_moe`` so the
a16w4 arm is no longer mixed with the a4w4/a8w4 (mxfp4-activation) code. bf16 A
(no A-scale), mxfp4 W1/W2 upconverted to bf16 in-kernel via
``cvt_scalef32_pk_bf16_fp4``, non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` stored by sorted position (no requant, no scale).
  - stage2 (:mod:`gemm2`): down-projection GEMM + routing-weighted atomic bf16
    scatter to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Self-contained: shared numeric/layout
helpers are duplicated into :mod:`common` (no import back into ``mxfp_moe``).
"""

from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.host import (
    a16wi4_scale_to_kernel_layout,
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    pick_a16w4_config,
    resolve_a16w4_gemm1_config,
    resolve_a16w4_gemm2_config,
)

__all__ = [
    "compile_gemm1_a16w4_port",
    "gemm1_a16w4_grid",
    "compile_gemm2_a16w4_port",
    "gemm2_a16w4_grid",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "a16wi4_scale_to_kernel_layout",
    "pick_a16w4_config",
    "resolve_a16w4_gemm1_config",
    "resolve_a16w4_gemm2_config",
]
