#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x rowwise INT8 quantize (rcp scale match)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_quantize_int8_rowwise import (  # noqa: E402
    quantize_int8_rowwise,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"int8 rowwise requires gfx120x, got {ARCH}", allow_module_level=True)


def _ref_rowwise(x: torch.Tensor) -> tuple[object, object]:
    x2 = x.reshape(-1, x.shape[-1]).float()
    amax = x2.abs().amax(dim=-1).clamp_min(1e-30)
    scale = (amax / 127.0).clamp_min(1e-30)
    # HIP uses rcp; IEEE ref may differ on ties — compare with slack.
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return q.reshape_as(x), scale.reshape(*x.shape[:-1], 1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,k", [(8, 64), (64, 256)])
def test_quantize_int8_rowwise(rows: int, k: int, dtype: torch.dtype) -> None:
    torch.manual_seed(20260930 + rows + k + int(dtype.is_floating_point))
    x = torch.randn(rows, k, device="cuda", dtype=dtype)
    q, s = quantize_int8_rowwise(x)
    torch.cuda.synchronize()
    qr, sr = _ref_rowwise(x)
    scale_maxdiff = (s.float() - sr.float()).abs().max().item()
    assert scale_maxdiff < 1e-4, f"scale_maxdiff={scale_maxdiff}"
    mism = (q != qr).sum().item()
    frac = mism / q.numel()
    assert frac < 0.02, f"q mismatch frac={frac} mism={mism}"
