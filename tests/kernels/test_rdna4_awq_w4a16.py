#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device tests for gfx120x AWQ W4A16 dequant + fused GEMV."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_awq_w4a16 import (  # noqa: E402
    dequant_awq_w4a16_weight,
    gemv_awq_w4a16,
    unpack_uint4_row_major,
)
from tests.kernels.oracles import reference_dequant_awq_w4a16  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"AWQ W4A16 requires gfx120x, got {ARCH}", allow_module_level=True)


def _pack_uint4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Inverse of unpack: (..., K) uint4 [0,15] → (..., K//2) int8."""
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _make_awq_weight(n: int, k: int, group_size: int, dtype: torch.dtype, seed: int) -> tuple[object, object, object]:
    torch.manual_seed(seed)
    assert k % group_size == 0
    groups = k // group_size
    # Synthetic uint4 codes + group scales/zeros
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    qweight = _pack_uint4_row_major(codes)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    wzeros = torch.randn((groups, n), device="cuda", dtype=dtype) * 0.01
    return qweight, wscales, wzeros


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [64])
def test_dequant_awq_w4a16_matches_ref(dtype: torch.dtype, group_size: int) -> None:
    n, k = 32, 256
    qweight, wscales, wzeros = _make_awq_weight(n, k, group_size, dtype, 20260930)
    torch.cuda.synchronize()
    out = dequant_awq_w4a16_weight(qweight, wscales, wzeros, group_size=group_size)
    torch.cuda.synchronize()
    ref = reference_dequant_awq_w4a16(qweight, wscales, wzeros, group_size=group_size)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_unpack_roundtrip() -> None:
    torch.manual_seed(7)
    vals = torch.randint(0, 16, (8, 64), device="cuda", dtype=torch.int32)
    packed = _pack_uint4_row_major(vals)
    got = unpack_uint4_row_major(packed).to(torch.int32)
    assert torch.equal(got, vals)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("shape", [(1, 32, 128), (4, 64, 256)], ids=["m1", "m4"])
def test_gemv_awq_w4a16_fused_matches_ref(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    """Fused GEMV: packed W stays packed; (q-8)*s+z in-reg; vs eager dequant+mm."""
    m, n, k = shape
    g = 64
    qweight, wscales, wzeros = _make_awq_weight(n, k, g, dtype, 11)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    bias = torch.randn((n,), device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    out = gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=bias, group_size=g)
    torch.cuda.synchronize()
    w = reference_dequant_awq_w4a16(qweight, wscales, wzeros, group_size=g)
    ref = x.to(dtype) @ w.t() + bias
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("k", [72, 100])
def test_awq_partial_group_matches_ref(k: int) -> None:
    """K not a multiple of the group stays on the real tensors."""
    n, g = 16, 64
    dtype = torch.bfloat16
    groups = (k + g - 1) // g
    torch.manual_seed(k)
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    qweight = _pack_uint4_row_major(codes)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    wzeros = torch.randn((groups, n), device="cuda", dtype=dtype) * 0.01
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        w = dequant_awq_w4a16_weight(qweight, wscales, wzeros, group_size=g, stream=stream)
        x = torch.randn((2, k), device="cuda", dtype=dtype)
        y = gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=None, group_size=g, stream=stream)
        ref_w = reference_dequant_awq_w4a16(qweight, wscales, wzeros, group_size=g)
        ref_y = x @ ref_w.t()
    stream.synchronize()
    assert tuple(qweight.shape) == (n, k // 2)
    assert tuple(w.shape) == (n, k)
    torch.testing.assert_close(w.float(), ref_w.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y.float(), ref_y.float(), rtol=2e-2, atol=2e-2)


def test_gemv_awq_w4a16_no_bias() -> None:
    dtype = torch.bfloat16
    m, n, k, g = 1, 16, 64, 64
    qweight, wscales, wzeros = _make_awq_weight(n, k, g, dtype, 3)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    out = gemv_awq_w4a16(x, qweight, wscales, wzeros, bias=None, group_size=g)
    w = reference_dequant_awq_w4a16(qweight, wscales, wzeros, group_size=g)
    ref = x @ w.t()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
