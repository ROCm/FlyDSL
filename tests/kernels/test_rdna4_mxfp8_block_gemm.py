#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x MXFP8 block GEMM: software E8M0 + dense fp8 WMMA."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_mxfp8_block_gemm import mxfp8_block_gemm  # noqa: E402
from kernels.quant.rdna4_mxfp8_e8m0 import (  # noqa: E402
    dequantize_mxfp8_device,
    quantize_mxfp8_device,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"MXFP8 block GEMM requires gfx120x, got {ARCH}", allow_module_level=True)


def _e8m0_to_f32_cpu(scale_u8: torch.Tensor) -> torch.Tensor:
    exp = scale_u8.to(torch.int32)
    bits = exp << 23
    bits = torch.where(exp == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exp == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 64, 128)])
def test_mxfp8_block_gemm_matches_dequant_matmul(m, n, k):
    torch.manual_seed(0)
    a_f = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_f = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_q, a_s = quantize_mxfp8_device(a_f)
    b_q, b_s = quantize_mxfp8_device(b_f)
    out = mxfp8_block_gemm(a_q, b_q, a_s, b_s, out_dtype=torch.bfloat16)
    a_dq = dequantize_mxfp8_device(a_q, a_s)
    b_dq = dequantize_mxfp8_device(b_q, b_s)
    ref = (a_dq @ b_dq.T).to(torch.bfloat16)
    assert torch.allclose(
        out.float(), ref.float(), atol=2e-2, rtol=2e-2
    ), f"max diff {(out.float() - ref.float()).abs().max().item()}"


def test_mxfp8_block_gemm_odd_k_stays_in_kernel():
    """K not a multiple of 32 stays in the kernel when scale cols cover ceil(K/32)."""
    a = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(16, 16, device="cuda", dtype=torch.float8_e4m3fn)
    # ceil(16/32)=1 scale col covers pad to K=32
    sa = torch.zeros(16, 1, device="cuda", dtype=torch.uint8)
    sb = torch.zeros(16, 1, device="cuda", dtype=torch.uint8)
    out = mxfp8_block_gemm(a, b, sa, sb, out_dtype=torch.bfloat16)
    assert out.shape == (16, 16)
    assert a.shape[1] == 16
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("k", [24, 40, 48])
def test_mxfp8_block_gemm_odd_k_matches_dequant(k: int) -> None:
    """A partial MX group is zero-filled in the kernel. K=48 is a 16-byte load."""
    m, n = 32, 32
    torch.manual_seed(k)
    a_f = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_f = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_q, a_s = quantize_mxfp8_device(a_f)
    b_q, b_s = quantize_mxfp8_device(b_f)
    assert tuple(a_q.shape) == (m, k)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = mxfp8_block_gemm(a_q, b_q, a_s, b_s, out_dtype=torch.bfloat16, stream=stream)
        a_dq = dequantize_mxfp8_device(a_q, a_s)
        b_dq = dequantize_mxfp8_device(b_q, b_s)
        ref = (a_dq.float() @ b_dq.float().T).to(torch.bfloat16)
    stream.synchronize()
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_mxfp8_formats_odd_k_stays_unpadded() -> None:
    """Odd K stays on the caller's storage. Scales cover ceil(K/32)."""
    torch.manual_seed(9)
    for k in (24, 26, 240):
        x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
        x_ptr = x.data_ptr()
        q, s = quantize_mxfp8_device(x)
        y = dequantize_mxfp8_device(q, s)
        assert q.shape[-1] == k and y.shape[-1] == k
        assert int(s.shape[-1]) == (k + 31) // 32
        assert x.data_ptr() == x_ptr
        assert torch.isfinite(y).all()


def test_mxfp8_odd_k_matches_padded_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short last group matches quantizing an explicitly zero-padded row."""
    import torch.nn.functional as F

    import kernels.common.gfx120x_pad as padmod

    def _boom(*_args, **_kwargs):
        raise AssertionError("device_pad")

    monkeypatch.setattr(padmod, "device_pad", _boom)
    torch.manual_seed(13)
    stream = torch.cuda.Stream()
    cases = [(torch.bfloat16, k) for k in (1, 24, 26, 31, 40)]
    cases.append((torch.float32, 26))
    for dtype, k in cases:
        x = (torch.randn(3, k, device="cuda") * 0.5).to(dtype)
        with torch.cuda.stream(stream):
            q, s = quantize_mxfp8_device(x, stream=stream)
            y = dequantize_mxfp8_device(q, s, stream=stream)
            pad = (32 - (k % 32)) % 32
            x_pad = F.pad(x, (0, pad))
            q_pad, s_pad = quantize_mxfp8_device(x_pad, stream=stream)
        stream.synchronize()
        assert s.shape[-1] == (k + 31) // 32
        assert torch.equal(s.view(torch.uint8), s_pad.view(torch.uint8))
        assert torch.equal(q.view(torch.uint8), q_pad[..., :k].contiguous().view(torch.uint8))
        assert y.shape[-1] == k
        assert torch.isfinite(y).all()
