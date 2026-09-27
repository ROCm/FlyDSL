# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused a16w-mix (bf16 A x mxfp4/int4 W) 2-stage MoE kernels.

Kernel builders live in :mod:`gemm1` / :mod:`gemm2` (shared helpers in
:mod:`utils`). Host launch/tile-config/CSV glue is a test-side concern and
lives in ``tests/kernels/moe_a16wmix_host.py``.
"""
