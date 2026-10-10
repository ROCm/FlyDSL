#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x tensorwise INT8 quantize (single absmax scale)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_quantize_int8_tensorwise import (  # noqa: E402
    dequantize_int8_tensorwise,
    quantize_int8_tensorwise,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"int8 tensorwise requires gfx120x, got {ARCH}", allow_module_level=True)


def _torch_ref(x: torch.Tensor) -> tuple[object, object]:
    amax = x.detach().float().abs().amax()
    scale = torch.clamp(amax / 127.0, min=1e-30)
    q = torch.round(x.float() / scale).clamp(-128, 127).to(torch.int8)
    return q, scale.reshape(())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8, 64), (64, 256), (3, 7), (77, 128)])
def test_quantize_int8_tensorwise(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(20260930 + sum(shape) + (0 if dtype is torch.float32 else 1))
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    q, s = quantize_int8_tensorwise(x)
    torch.cuda.synchronize()
    assert q.shape == x.shape and q.dtype == torch.int8
    assert s.shape == () and s.dtype == torch.float32 and s.device.type == x.device.type
    qr, sr = _torch_ref(x)
    scale_maxdiff = (s.float() - sr.float()).abs().item()
    assert scale_maxdiff < 1e-5, f"scale_maxdiff={scale_maxdiff} got={s.item()} ref={sr.item()}"
    mism = (q != qr).sum().item()
    # Rounding-boundary noise is O(1). A flat 2% fraction lets large shapes
    # greenlight a systematic ~2% bug. Cap absolute mismatches size-aware.
    # A short tensor must be exact: one bad tail element used to hide in the floor.
    if q.numel() <= 128:
        assert mism == 0, f"q mismatch mism={mism}"
    else:
        max_mism = max(2, q.numel() // 5000)  # ~0.02% asymptotic, floor 2
        assert mism <= max_mism, f"q mismatch mism={mism} max_allowed={max_mism} frac={mism / q.numel()}"
    if shape == (3, 7):
        qg, sg = quantize_int8_tensorwise(x, scale=sr)
        torch.cuda.synchronize()
        assert torch.equal(qg, qr)
        assert (sg.float() - sr.float()).abs().item() < 1e-5


def test_dequantize_int8_tensorwise_matches_scale() -> None:
    torch.manual_seed(4)
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    q, scale = quantize_int8_tensorwise(x)
    out = dequantize_int8_tensorwise(q, scale, out_dtype=torch.float32)
    torch.cuda.synchronize()
    ref = q.float() * scale.float()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_supplied_negative_scale_keeps_sign() -> None:
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda")
    q, scale = quantize_int8_tensorwise(x, scale=-0.5)
    torch.cuda.synchronize()
    ref = torch.round(x / -0.5).to(torch.int8)
    assert torch.equal(q.view(-1), ref)
    assert float(scale) < 0
