# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused a16w-mix (bf16 A x mxfp4/int4/bf16/fp8 W) 2-stage MoE kernels.

Kernel builders live in :mod:`gemm1` / :mod:`gemm2` (shared helpers in
:mod:`utils`); :mod:`fp8_b32` holds the public fp8 32x32-block-scale layer
launcher. Other host launch/tile-config/CSV glue is a test-side concern and
lives in ``tests/kernels/moe_a16wmix_host.py``.
"""

from kernels.moe.moe_2stage_a16wmix.fp8_b32 import compile_moe_fp8_b32

__all__ = ["compile_moe_fp8_b32"]
