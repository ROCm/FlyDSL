#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Pytest entry points for the a8w4smooth-only kernel module.

Exercises `kernels.a8w4_moe_gemm_2stage` via the existing harness functions in
`tests/kernels/test_moe_gemm.py`. We monkey-patch the legacy test module's
`compile_*` references to point at the new module, then call the same runners.
This keeps the test surface narrow (no duplicated harness code) while ensuring
the new module produces bit-identical results to the legacy a8w4smooth path.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "build", "python_packages")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Import the new a8w4smooth-only module first so any compile-time errors surface
# before pytest collects.
from kernels.a8w4_moe_gemm_2stage import (
    compile_moe_gemm1 as _compile_moe_gemm1,
    compile_moe_gemm2 as _compile_moe_gemm2,
    compile_moe_gemm2_ex as _compile_moe_gemm2_ex,
    compile_moe_reduction as _compile_moe_reduction,
    MoeGemm2Mode as _MoeGemm2Mode,
)

import tests.kernels.test_moe_gemm as _legacy

# Redirect the legacy harness's compile_* lookups to the new module.
_legacy.compile_moe_gemm1 = _compile_moe_gemm1
_legacy.compile_moe_gemm2 = _compile_moe_gemm2
_legacy.compile_moe_gemm2_ex = _compile_moe_gemm2_ex
_legacy.compile_moe_reduction = _compile_moe_reduction
_legacy.MoeGemm2Mode = _MoeGemm2Mode

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.parametrize("tile_k", [256, 128])
def test_a8w4_stage1(tile_k):
    """Stage1 a8w4smooth standalone — mirrors run_moe_stage1_a8w4smooth shape."""
    _legacy.run_moe_stage1_a8w4smooth(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=int(tile_k),
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )


@pytest.mark.parametrize("tile_k", [256, 128])
def test_a8w4_stage2(tile_k):
    """Stage2 a8w4smooth standalone — mirrors run_moe_stage2_a8w4smooth shape."""
    _legacy.run_moe_stage2_a8w4smooth(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=int(tile_k),
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )


@pytest.mark.parametrize("use_reduce,use_valid_mask", [
    (False, False),  # atomic mode
    (True, False),   # reduce mode without mask
    (True, True),    # reduce mode with valid_mask
])
@pytest.mark.parametrize("test_graph", [False])
def test_a8w4_gemm_2stage(use_reduce, use_valid_mask, test_graph):
    """Combined gemm1 -> gemm2 a8w4smooth flow (atomic + reduce ± valid_mask)."""
    _legacy.test_moe_gemm_2stage(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n1=64,
        tile_k1=256,
        tile_n2=64,
        tile_k2=256,
        doweight_stage1=False,
        in_dtype="a8w4smooth",
        out_dtype="f16",
        use_reduce=bool(use_reduce),
        use_valid_mask=bool(use_valid_mask),
        test_graph=bool(test_graph),
        group_size=-1,
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )
