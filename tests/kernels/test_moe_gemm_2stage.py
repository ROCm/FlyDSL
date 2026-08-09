#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Basic smoke tests for the ``moe_gemm_2stage`` fp8 2-stage kernels.

These verify that the stage1 (``compile_moe_gemm1``) and stage2
(``compile_moe_gemm2``) builders trace and lower without error for a small,
tile-valid fp8 shape. The package is fp8-only; full numerical / e2e coverage
lives in the routing-based MoE tests. This file just guards the builders
against API drift.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (os.path.join(_REPO_ROOT, "build", "python_packages"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.moe.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _fp8_supported() -> bool:
    arch = get_rocm_arch()
    return ("gfx95" in arch) or ("gfx94" in arch)


# Small, tile-valid shape (mirrors the "S" case of the former 2-stage suite):
#   model_dim=256, inter_dim=128, experts=4, topk=2
#   stage1 tile = (tile_m=16, tile_n1=64, tile_k1=128)
#   stage2 tile = (tile_m=16, tile_n2=64, tile_k2=128)
_SHAPE = dict(model_dim=256, inter_dim=128, experts=4, topk=2)


def test_moe_gemm1_builds():
    """Stage1 fp8 builder traces/lowers for a small tile-valid shape."""
    if not _fp8_supported():
        pytest.skip("fp8 stage1 requires gfx94*/gfx95*")

    exe = compile_moe_gemm1(
        **_SHAPE,
        tile_m=16,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        out_dtype="f16",
    )
    assert callable(exe)


def test_moe_gemm2_builds():
    """Stage2 fp8 builder traces/lowers for a small tile-valid shape."""
    if not _fp8_supported():
        pytest.skip("fp8 stage2 requires gfx94*/gfx95*")

    exe = compile_moe_gemm2(
        **_SHAPE,
        tile_m=16,
        tile_n=64,
        tile_k=128,
        doweight_stage2=False,
        out_dtype="f16",
    )
    assert callable(exe)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
