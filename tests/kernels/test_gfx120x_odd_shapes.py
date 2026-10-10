#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Odd-shape coverage across gfx120x product hosts.

Prefer shapes that are not tile multiples. The kernel zero-fills those tails.
Intentional hard raises stay documented here (paged float head dim, GQA).
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func  # noqa: E402
from kernels.common.gfx120x_arch import is_gfx120x  # noqa: E402
from kernels.gemm.rdna4_fused_mlp_nmajor import fused_swiglu_mlp_nmajor  # noqa: E402
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm  # noqa: E402
from kernels.gemm.rdna4_scaled_mm_fp8 import scaled_mm_fp8  # noqa: E402
from kernels.gemm.rdna4_w8a16_linear import w8a16_linear  # noqa: E402
from kernels.quant.rdna4_awq_w4a16 import gemv_awq_w4a16  # noqa: E402
from kernels.quant.rdna4_convrot_w4a4 import convrot_w4a4_linear, quantize_convrot_w4a4_weight  # noqa: E402
from kernels.quant.rdna4_int4_codec import pack_int4_row_major  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not is_gfx120x():
    pytest.skip(f"odd-shape suite requires gfx120x, got {ARCH}", allow_module_level=True)


def _pack_uint4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


@pytest.mark.parametrize("head_dim", [65, 80, 96, 127])
@pytest.mark.parametrize("seq", [33, 64, 77])
def test_fa_bf16_odd_head_dim_and_seq(head_dim: int, seq: int) -> None:
    """Dense FA soft-pads head_dim to next %32 tile in [64, 480]; odd seq is fine."""
    q = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()


def test_fa_head_dim_above_lds_still_raises() -> None:
    q = torch.randn(1, 32, 1, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func(q, q, q, causal=False)


@pytest.mark.parametrize("k", [17, 31, 48, 63])
def test_iu4_gemm_odd_k_pack_aware(k: int) -> None:
    """K not multiple of 16 stays in the kernel. Pack still needs an even K."""
    m, n = 16, 32
    a = torch.randint(-8, 7, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 7, (n, k), device="cuda", dtype=torch.int8)
    # Pack needs even K. The extra nibble is zero and is not a host GEMM pad.
    if k % 2 != 0:
        a = torch.nn.functional.pad(a, (0, 1))
        b = torch.nn.functional.pad(b, (0, 1))
    scale_a = torch.ones(m, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(n, device="cuda", dtype=torch.float32)
    y = iu4_gemm(
        pack_int4_row_major(a),
        pack_int4_row_major(b),
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert y.shape == (m, n)
    a0 = a[:, :k]
    b0 = b[:, :k]
    ref = (a0.float() @ b0.float().T) * scale_a.float().unsqueeze(1) * scale_b.float().unsqueeze(0)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("k", [33, 80, 100])
def test_int8_linear_odd_k(k: int) -> None:
    m, n = 16, 32
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32).abs() + 0.01
    y = w8a16_linear(x, weight, weight_scale)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [33, 96])
def test_scaled_mm_fp8_odd_k(k: int) -> None:
    m, n = 32, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    sa = float(a.abs().amax().clamp(min=1e-12) / 448.0)
    sb = float(b.abs().amax().clamp(min=1e-12) / 448.0)
    aq = (a.float() / sa).to(torch.float8_e4m3fn)
    bq = (b.float() / sb).to(torch.float8_e4m3fn)
    y = scaled_mm_fp8(
        aq,
        bq,
        torch.tensor([sa], device="cuda", dtype=torch.float32),
        torch.tensor([sb], device="cuda", dtype=torch.float32),
    )
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [48, 100])
def test_awq_odd_k_stays_in_kernel(k: int) -> None:
    """Scales cover ceil(K/G). X and packed W stay at the caller's K."""
    m, n, g = 2, 16, 64
    dtype = torch.bfloat16
    k_target = ((k + g - 1) // g) * g
    groups = k_target // g
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    # Pack logical K (pad one col if odd); host soft-pads to group when scales cover ceil(K/G).
    k_even = k if k % 2 == 0 else k + 1
    codes_e = torch.nn.functional.pad(codes, (0, k_even - k))
    qw = _pack_uint4_row_major(codes_e)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    wzeros = torch.randn((groups, n), device="cuda", dtype=dtype) * 0.01
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    y = gemv_awq_w4a16(x, qw, wscales, wzeros, bias=None, group_size=g)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k", [48, 80])
def test_convrot_odd_k_stays_in_kernel(k: int) -> None:
    m, n = 8, 32
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
    y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)
    assert y.shape == (m, n)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("k,ffn", [(8, 8), (12, 16), (16, 12)])
def test_fused_swiglu_odd_ffn_or_k(k: int, ffn: int) -> None:
    """Short K or FFN stays in the LDS kernel and matches the eager SwiGLU."""
    from tests.kernels.oracles.rdna4_fused_mlp_nmajor_oracle import reference_swiglu_mlp

    m = 16
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_gate = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(ffn, k, device="cuda", dtype=torch.bfloat16)
    w_down = torch.randn(k, ffn, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down, stream=stream)
    stream.synchronize()
    ref = reference_swiglu_mlp(x, w_gate, w_up, w_down)
    assert y.shape == (m, k)
    assert x.shape == (m, k)
    torch.testing.assert_close(y.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


def test_gqa_non_divisible_still_raises() -> None:
    q = torch.randn(1, 32, 3, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        flydsl_flash_attn_func(q, k, v, causal=False)


def test_empty_mnk_still_raises_on_iu4_native_path() -> None:
    """Empty M raises on private native path and public int4 host (no iu8 demotion)."""
    from kernels.quant.rdna4_convrot_w4a4 import _convrot_w4a4_native_iu4, convrot_w4a4_linear

    x = torch.randn(0, 64, device="cuda", dtype=torch.bfloat16)
    qw = torch.zeros(32, 32, device="cuda", dtype=torch.int8)
    ws = torch.ones(32, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="positive MNK"):
        _convrot_w4a4_native_iu4(x, qw, ws, convrot_groupsize=64)
    with pytest.raises(ValueError, match="positive MNK"):
        convrot_w4a4_linear(x, qw, ws, convrot_groupsize=64, linear_dtype="int4")
