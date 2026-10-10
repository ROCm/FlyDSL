#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device tests for gfx120x SVDQuant W4A4 (+ int4 codec)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_int4_codec import (  # noqa: E402
    pack_int4_row_major,
    unpack_int4_row_major,
    unpack_uint4_row_major,
)
from kernels.quant.rdna4_svdquant_w4a4 import (  # noqa: E402
    dequant_svdquant_w4a4_weight,
    quantize_svdquant_w4a4,
    scaled_mm_svdquant_w4a4,
    svdquant_w4a4_linear,
)
from tests.kernels.oracles import (  # noqa: E402
    reference_dequant_svdquant_w4a4_weight,
    reference_scaled_mm_svdquant_w4a4,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"SVDQuant W4A4 requires gfx120x, got {ARCH}", allow_module_level=True)


def _make_svd_weight(n: int, k: int, group_size: int, dtype: torch.dtype, seed: int) -> tuple[object, object, object]:
    torch.manual_seed(seed)
    assert k % group_size == 0
    groups = k // group_size
    # Emission range [-7, 7] (nunchaku / layout contract)
    codes = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int32)
    qweight = pack_int4_row_major(codes)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    return qweight, wscales, codes


def test_pack_unpack_signed_roundtrip() -> None:
    torch.manual_seed(3)
    vals = torch.randint(-8, 8, (8, 64), device="cuda", dtype=torch.int32)
    packed = pack_int4_row_major(vals)
    got = unpack_int4_row_major(packed).to(torch.int32)
    assert torch.equal(got, vals)


def test_pack_unpack_unsigned_roundtrip() -> None:
    torch.manual_seed(5)
    vals = torch.randint(0, 16, (8, 64), device="cuda", dtype=torch.int32)
    packed = pack_int4_row_major(vals)
    got = unpack_uint4_row_major(packed).to(torch.int32)
    assert torch.equal(got, vals)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [64])
def test_dequant_svdquant_w4a4_matches_ref(dtype: torch.dtype, group_size: int) -> None:
    n, k = 32, 256
    qweight, wscales, _ = _make_svd_weight(n, k, group_size, dtype, 20260930)
    torch.cuda.synchronize()
    out = dequant_svdquant_w4a4_weight(qweight, wscales, group_size=group_size)
    torch.cuda.synchronize()
    ref = reference_dequant_svdquant_w4a4_weight(qweight, wscales, group_size=group_size)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("act_unsigned", [False, True])
def test_quantize_svdquant_w4a4_shapes_and_range(act_unsigned: bool) -> None:
    m, k, r, pad = 4, 256, 16, 16
    dtype = torch.bfloat16
    torch.manual_seed(9)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    if act_unsigned:
        x = x.abs()  # stay in positive-ish domain after shift convention
    smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.5) + 0.1
    proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.01
    q_x, ascales, lora_act = quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=pad, act_unsigned=act_unsigned)
    m_pad = ((m + pad - 1) // pad) * pad
    assert q_x.shape == (m_pad, k // 2)
    assert ascales.shape == (k // 64, m_pad)
    assert lora_act.shape == (m_pad, r)
    assert q_x.dtype == torch.int8
    assert ascales.dtype == dtype
    assert lora_act.dtype == torch.float32
    codes = unpack_uint4_row_major(q_x) if act_unsigned else unpack_int4_row_major(q_x)
    if act_unsigned:
        assert int(codes.min()) >= 0 and int(codes.max()) <= 15
    else:
        assert int(codes.min()) >= -7 and int(codes.max()) <= 7


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "host"])
@pytest.mark.parametrize("act_unsigned", [False, True], ids=["s4", "u4"])
def test_scaled_mm_svdquant_matches_ref(dtype: torch.dtype, fused: bool, act_unsigned: bool) -> None:
    m, n, k, r, g = 4, 64, 256, 16, 64
    pad = 16
    torch.manual_seed(11)
    qweight, wscales, _ = _make_svd_weight(n, k, g, dtype, 11)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    if act_unsigned:
        x = x.abs() + 0.2
    smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.4) + 0.15
    proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.02
    proj_up = torch.randn((n, r), device="cuda", dtype=dtype) * 0.02
    bias = torch.randn((n,), device="cuda", dtype=dtype)

    if act_unsigned:
        x_main = x + 0.171875
        lora_x = x
    else:
        x_main = x
        lora_x = None
    q_x, ascales, lora = quantize_svdquant_w4a4(
        x_main, smooth, proj_down, pad_size=pad, act_unsigned=act_unsigned, lora_x=lora_x
    )
    torch.cuda.synchronize()
    out = scaled_mm_svdquant_w4a4(
        q_x,
        qweight,
        ascales,
        wscales,
        lora,
        proj_up,
        bias=bias,
        act_unsigned=act_unsigned,
        group_size=g,
        fused=fused,
    )
    torch.cuda.synchronize()
    ref = reference_scaled_mm_svdquant_w4a4(
        q_x, qweight, ascales, wscales, lora, proj_up, bias=bias, act_unsigned=act_unsigned, group_size=g
    )
    # Fused path stores bf16 and uses host bf16 LoRA;
    # host/ref accumulate LoRA in fp32 — allow bf16-ish tol on fused.
    tol = 5e-2 if fused else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("act_unsigned", [False, True], ids=["s4", "u4"])
def test_scaled_mm_fused_vs_eager_ref(act_unsigned: bool) -> None:
    """Fused packed path vs in-tree torch reference (LoRA host bf16 residual)."""
    dtype = torch.bfloat16
    m, n, k, r, g = 4, 32, 128, 8, 64
    pad = 16
    torch.manual_seed(17)
    qweight, wscales, _ = _make_svd_weight(n, k, g, dtype, 17)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    if act_unsigned:
        x = x.abs() + 0.2
    smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.35) + 0.12
    proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.015
    proj_up = torch.randn((n, r), device="cuda", dtype=dtype) * 0.015
    bias = torch.randn((n,), device="cuda", dtype=dtype)

    if act_unsigned:
        x_main = x + 0.171875
        lora_x = x
    else:
        x_main = x
        lora_x = None

    q_x, ascales, lora = quantize_svdquant_w4a4(
        x_main, smooth, proj_down, pad_size=pad, act_unsigned=act_unsigned, lora_x=lora_x
    )
    torch.cuda.synchronize()
    got = scaled_mm_svdquant_w4a4(
        q_x,
        qweight,
        ascales,
        wscales,
        lora,
        proj_up,
        bias=bias,
        act_unsigned=act_unsigned,
        group_size=g,
        fused=True,
    )
    torch.cuda.synchronize()
    ref = reference_scaled_mm_svdquant_w4a4(
        q_x, qweight, ascales, wscales, lora, proj_up, bias=bias, act_unsigned=act_unsigned, group_size=g
    )
    # Reference LoRA accumulates fp32; host LoRA is bf16 — allow bf16-ish tol.
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_svdquant_w4a4_linear_host(dtype: torch.dtype) -> None:
    m, n, k, r, g = 3, 64, 256, 8, 64
    torch.manual_seed(13)
    qweight, wscales, _ = _make_svd_weight(n, k, g, dtype, 13)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.3) + 0.2
    proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.01
    proj_up = torch.randn((n, r), device="cuda", dtype=dtype) * 0.01
    bias = torch.randn((n,), device="cuda", dtype=dtype)

    torch.cuda.synchronize()
    out = svdquant_w4a4_linear(
        x,
        qweight,
        wscales,
        proj_down,
        proj_up,
        smooth,
        bias=bias,
        pad_size=16,
        group_size=g,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n)

    q_x, ascales, lora = quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=16)
    ref = reference_scaled_mm_svdquant_w4a4(q_x, qweight, ascales, wscales, lora, proj_up, bias=bias, group_size=g)[:m]
    torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("k", [80, 100])
def test_svdquant_partial_group_stays_unpadded(k: int) -> None:
    """K not a multiple of 64 is not cloned up to the group."""
    m, n, r, g = 4, 32, 8, 64
    dtype = torch.bfloat16
    groups = (k + g - 1) // g
    torch.manual_seed(k)
    codes = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int32)
    qweight = pack_int4_row_major(codes)
    wscales = (torch.randn((groups, n), device="cuda", dtype=dtype) * 0.05).abs() + 1e-3
    w = dequant_svdquant_w4a4_weight(qweight, wscales, group_size=g)
    ref_w = reference_dequant_svdquant_w4a4_weight(qweight, wscales, group_size=g)
    torch.cuda.synchronize()
    torch.testing.assert_close(w.float(), ref_w.float(), rtol=1e-2, atol=1e-2)

    x = torch.randn((m, k), device="cuda", dtype=dtype)
    smooth = (torch.randn(k, device="cuda", dtype=dtype).abs() * 0.3) + 0.2
    proj_down = torch.randn((k, r), device="cuda", dtype=dtype) * 0.01
    proj_up = torch.randn((n, r), device="cuda", dtype=dtype) * 0.01
    q_x, ascales, lora = quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=16)
    assert tuple(x.shape) == (m, k)
    assert q_x.shape[1] == k // 2
    assert ascales.shape[0] == groups
    got = scaled_mm_svdquant_w4a4(
        q_x[:m],
        qweight,
        ascales[:, :m],
        wscales,
        lora[:m],
        proj_up,
        group_size=g,
        fused=True,
    )
    ref = reference_scaled_mm_svdquant_w4a4(q_x[:m], qweight, ascales[:, :m], wscales, lora[:m], proj_up, group_size=g)
    torch.cuda.synchronize()
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=5e-2)
