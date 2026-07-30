# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a4w4 / a8w4 MoE 2-stage kernels (mxfp_moe).

Hand-specialized CDNA4 (gfx950) MFMA pipeline ported back from aiter:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU + on-device fp4 re-quant,
    emitting a sorted fp4 intermediate (``aqout``/``ascaleout``) directly consumed
    by stage2 (no host re-quant between stages). ``a_dtype`` selects the activation:
    "fp4" (a4w4, mxfp4 A) or "fp8" (a8w4, fp8 e4m3 A x mxfp4 W1); W1/W2 are always
    mxfp4 and the intermediate is fp4, so stage2 is identical for both.
  - stage2 (:mod:`gemm2`): down-projection GEMM with atomic / reduce / cshuffle /
    mxfp4-out epilogues.

Distinct from the parametric ``mixed_moe_gemm_2stage`` it replaces: this variant
does device-side re-quant and uses a ``cumsum`` + ``m_indices`` sorting contract.
"""

from kernels.moe.mxfp_moe.gemm1 import compile_gemm1_a4w4_port, gemm1_grid
from kernels.moe.mxfp_moe.gemm2 import compile_gemm2_a4w4_port
from kernels.moe.mxfp_moe.host import flydsl_mxfp4_gemm1, flydsl_mxfp4_gemm2

__all__ = [
    "compile_gemm1_a4w4_port",
    "gemm1_grid",
    "compile_gemm2_a4w4_port",
    "flydsl_mxfp4_gemm1",
    "flydsl_mxfp4_gemm2",
]
