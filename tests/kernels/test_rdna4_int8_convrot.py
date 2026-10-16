#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device tests for gfx120x INT8 ConvRot quantize (+ linear path)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_int8_convrot import (  # noqa: E402
    convrot_fwht,
    dequantize_int8_convrot_weight,
    int8_linear_convrot,
    quantize_int8_convrot_weight,
)
from tests.kernels.oracles import (  # noqa: E402
    reference_dequantize_int8_convrot_weight,
    reference_quantize_int8_convrot_weight,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"INT8 ConvRot requires gfx120x, got {ARCH}", allow_module_level=True)


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_int8_convrot_weight(group_size: int, dtype: torch.dtype) -> None:
    torch.manual_seed(20260929 + group_size)
    rows, k = 32, group_size * 2
    w = torch.randn((rows, k), device="cuda", dtype=dtype).clamp(-2, 2)
    q, scale = quantize_int8_convrot_weight(w, group_size=group_size)
    torch.cuda.synchronize()
    q_ref, scale_ref = reference_quantize_int8_convrot_weight(w, group_size=group_size)

    # Scales: rcp vs IEEE may differ slightly in bf16/fp16 paths; allow tiny slack.
    scale_maxdiff = (scale.float() - scale_ref.float()).abs().max().item()
    assert scale_maxdiff < 1e-4, f"scale_maxdiff={scale_maxdiff}"

    # INT8 codes: allow a few off-by-one from rcp vs div rounding.
    mism = (q.cpu() != q_ref.cpu()).sum().item()
    frac = mism / q.numel()
    assert frac < 0.02, f"q mismatch frac={frac} count={mism}/{q.numel()}"


def test_quantize_int8_convrot_weight_wan_like() -> None:
    """Wider K typical of Wan/Flux linear (group 256)."""
    torch.manual_seed(7)
    w = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16).clamp(-1.5, 1.5)
    q, scale = quantize_int8_convrot_weight(w, group_size=256)
    torch.cuda.synchronize()
    q_ref, scale_ref = reference_quantize_int8_convrot_weight(w, group_size=256)
    assert (scale.float() - scale_ref.float()).abs().max().item() < 1e-4
    frac = (q != q_ref).sum().item() / q.numel()
    assert frac < 0.02


def test_quantize_int8_convrot_weight_k16384() -> None:
    """K past the old full-row f32 LDS cap (8192). Group tile must still match."""
    torch.manual_seed(16384)
    w = torch.randn((2, 16384), device="cuda", dtype=torch.bfloat16).clamp(-1.5, 1.5)
    q, scale = quantize_int8_convrot_weight(w, group_size=256)
    torch.cuda.synchronize()
    q_ref, scale_ref = reference_quantize_int8_convrot_weight(w, group_size=256)
    assert (scale.float() - scale_ref.float()).abs().max().item() < 1e-4
    frac = (q != q_ref).sum().item() / q.numel()
    assert frac < 0.02, f"q mismatch frac={frac}"


def test_quantize_stochastic_seed_is_stable_and_differs() -> None:
    """seed > 0 uses floor(v/scale + U). Same seed repeats; round-even does not match it."""
    torch.manual_seed(3)
    w = torch.randn((4, 256), device="cuda", dtype=torch.float32).clamp(-1.5, 1.5)
    q_a, s_a = quantize_int8_convrot_weight(w, group_size=256, stochastic_rounding=11)
    q_b, s_b = quantize_int8_convrot_weight(w, group_size=256, stochastic_rounding=11)
    q0, s0 = quantize_int8_convrot_weight(w, group_size=256, stochastic_rounding=0)
    torch.cuda.synchronize()
    assert torch.equal(q_a, q_b)
    assert torch.allclose(s_a, s_b)
    assert torch.allclose(s_a, s0)
    assert not torch.equal(q_a, q0)


def test_dequant_kernel_matches_reference() -> None:
    torch.manual_seed(19)
    w = torch.randn((8, 256), device="cuda", dtype=torch.float32).clamp(-1, 1)
    q, scale = quantize_int8_convrot_weight(w, group_size=256)
    torch.cuda.synchronize()
    out = dequantize_int8_convrot_weight(q, scale, group_size=256, out_dtype=torch.float32)
    ref = reference_dequantize_int8_convrot_weight(q, scale, group_size=256, out_dtype=torch.float32)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_dequant_roundtrip_unrotates() -> None:
    torch.manual_seed(11)
    # Clamp like sibling quant tests so abs budget tracks INT8 LSB, not randn tails.
    w = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16).clamp(-1.5, 1.5)
    q, scale = quantize_int8_convrot_weight(w, group_size=256)
    torch.cuda.synchronize()
    recon = dequantize_int8_convrot_weight(q, scale, group_size=256, out_dtype=torch.float32)
    # Reconstruction error dominated by INT8 quantization, not rotation.
    err = (recon - w.float()).abs().max().item()
    # Half-LSB of per-row scale is the expected ceiling; keep a small slack for FWHT.
    bound = max(0.05, float(scale.float().abs().max().item()) * 0.75)
    assert err < bound, f"max_abs roundtrip={err} bound={bound} scale_max={float(scale.float().abs().max())}"


def test_int8_linear_convrot_matches_eager() -> None:
    """Act rotate+quant + GEMM vs torch reference with same ConvRot weights."""
    try:
        from kernels.gemm.rdna4_int8_linear import create_wmma_int8_linear_module  # noqa: F401
    except ImportError:
        pytest.skip("rdna4_int8_linear (iu8 atom / iu8) not on this branch")
    torch.manual_seed(13)
    m, n, k = 64, 64, 256
    group_size = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    wq, wscale = quantize_int8_convrot_weight(w, group_size=group_size)
    torch.cuda.synchronize()

    out = int8_linear_convrot(x, wq, wscale, group_size=group_size, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Reference: rotate+quant act (torch), int matmul, scale.
    aq, ascale = reference_quantize_int8_convrot_weight(x, group_size=group_size)
    acc = aq.float() @ wq.float().T
    ref = (acc * ascale.reshape(m, 1).float() * wscale.reshape(1, n).float()).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.3)


def test_int8_linear_convrot_odd_k_stays_in_kernel() -> None:
    """A K that is not a group multiple is zero-filled. The oracle matches."""
    try:
        from kernels.gemm.rdna4_int8_linear import create_wmma_int8_linear_module  # noqa: F401
    except ImportError:
        pytest.skip("rdna4_int8_linear (iu8) not on this branch")
    torch.manual_seed(17)
    m, n, k = 32, 32, 100  # 100 % 64 != 0
    group_size = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    wq, wscale = quantize_int8_convrot_weight(w, group_size=group_size)
    out = int8_linear_convrot(x, wq, wscale, group_size=group_size, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    # Full Hadamard groups. A prefix-only dot is not the orthogonal product.
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.5)
    assert int(wq.shape[1]) == 128


def test_int8_convrot_scale_trailing_singleton() -> None:
    """HIP-compatible scale ABI keeps trailing singleton dim (not flat rows)."""
    torch.manual_seed(3)
    w = torch.randn((8, 4, 128), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    q, scale = quantize_int8_convrot_weight(w, group_size=64)
    torch.cuda.synchronize()
    assert q.shape == w.shape
    assert scale.shape == (*w.shape[:-1], 1), scale.shape
    assert scale.dtype == torch.float32


def test_convrot_fwht_odd_k_stays_in_kernel() -> None:
    """FWHT zero-fills past K and stores the caller's columns."""
    torch.manual_seed(41)
    group_size = 64
    k = 100  # not divisible by 64
    w = torch.randn((8, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    out = convrot_fwht(w, group_size=group_size)
    w_pad = torch.nn.functional.pad(w, (0, group_size - (k % group_size)))
    out_pad = convrot_fwht(w_pad, group_size=group_size)
    torch.cuda.synchronize()
    assert out.shape == w.shape
    torch.testing.assert_close(out.float(), out_pad[:, :k].float())
    assert out.dtype == w.dtype
    assert torch.isfinite(out.float()).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
