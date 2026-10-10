#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x MXFP4 block GEMM: unpack E2M1 to e4m3, then fp8 WMMA and E8M0 scales."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_mxfp4_block_gemm import mxfp4_block_gemm  # noqa: E402
from kernels.quant.rdna4_mxfp4_e2m1 import (  # noqa: E402
    dequantize_mxfp4_device,
    quantize_mxfp4_device,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"MXFP4 block GEMM requires gfx120x, got {ARCH}", allow_module_level=True)


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 64, 128)])
def test_mxfp4_block_gemm_matches_dequant_matmul(m, n, k):
    torch.manual_seed(0)
    a_f = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_f = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_q, a_s = quantize_mxfp4_device(a_f)
    b_q, b_s = quantize_mxfp4_device(b_f)
    out = mxfp4_block_gemm(a_q, b_q, a_s, b_s, out_dtype=torch.float32)
    a_dq = dequantize_mxfp4_device(a_q, a_s)
    b_dq = dequantize_mxfp4_device(b_q, b_s)
    ref = a_dq @ b_dq.T
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), f"max diff {(out - ref).abs().max().item()}"
