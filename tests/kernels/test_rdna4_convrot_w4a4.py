#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device tests for gfx120x ConvRot W4A4 quantize (default linear_dtype='int4'; 'int8' unpacks to iu8)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_convrot_w4a4 import (  # noqa: E402
    convrot_w4a4_linear,
    dequantize_convrot_w4a4_weight,
    quantize_convrot_w4a4_weight,
)
from tests.kernels.oracles import reference_quantize_convrot_w4a4_weight  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"ConvRot W4A4 requires gfx120x, got {ARCH}", allow_module_level=True)


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_convrot_w4a4_weight(group_size: int, dtype: torch.dtype) -> None:
    torch.manual_seed(20260930 + group_size)
    rows, k = 32, max(group_size * 2, 128)
    # K must be divisible by quant_group_size=64
    k = ((k + 63) // 64) * 64
    w = torch.randn((rows, k), device="cuda", dtype=dtype).clamp(-2, 2)
    q, scale = quantize_convrot_w4a4_weight(w, convrot_groupsize=group_size)
    torch.cuda.synchronize()
    q_ref, scale_ref = reference_quantize_convrot_w4a4_weight(w, convrot_groupsize=group_size)

    scale_maxdiff = (scale.float() - scale_ref.float()).abs().max().item()
    assert scale_maxdiff < 1e-4, f"scale_maxdiff={scale_maxdiff}"

    mism = (q.cpu() != q_ref.cpu()).sum().item()
    frac = mism / q.numel()
    assert frac < 0.02, f"q mismatch frac={frac} count={mism}/{q.numel()}"


def test_quantize_convrot_w4a4_wan_like_k() -> None:
    """Wider K typical of Wan/Flux linear (hot model dims; comment-only)."""
    torch.manual_seed(7)
    # Wan-like K=1024 (group 256, quant groups of 64).
    w = torch.randn((64, 1024), device="cuda", dtype=torch.bfloat16).clamp(-1.5, 1.5)
    q, scale = quantize_convrot_w4a4_weight(w, convrot_groupsize=256)
    torch.cuda.synchronize()
    q_ref, scale_ref = reference_quantize_convrot_w4a4_weight(w, convrot_groupsize=256)
    assert (scale.float() - scale_ref.float()).abs().max().item() < 1e-4
    frac = (q != q_ref).sum().item() / q.numel()
    assert frac < 0.02


def test_dequant_roundtrip_unrotates() -> None:
    torch.manual_seed(11)
    w = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_convrot_w4a4_weight(w, convrot_groupsize=256)
    torch.cuda.synchronize()
    recon = dequantize_convrot_w4a4_weight(q, scale, convrot_groupsize=256, output_dtype=torch.float32)
    # INT4 quantization dominates reconstruction error.
    err = (recon - w.float()).abs().max().item()
    assert err < 1.0, f"max_abs roundtrip={err}"


def test_convrot_w4a4_linear_int8_fallback() -> None:
    """Unpack W4 → iu8 ConvRot linear vs torch reference."""
    try:
        from kernels.gemm.rdna4_int8_linear import create_wmma_int8_linear_module  # noqa: F401
    except ImportError:
        pytest.skip("rdna4_int8_linear (iu8 atom) not on this branch")
    torch.manual_seed(13)
    m, n, k = 64, 64, 256
    group_size = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    wq, wscale = quantize_convrot_w4a4_weight(w, convrot_groupsize=group_size)
    torch.cuda.synchronize()

    out = convrot_w4a4_linear(
        x,
        wq,
        wscale,
        convrot_groupsize=group_size,
        linear_dtype="int8",
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    from kernels.quant.rdna4_convrot_w4a4 import _unpack_int4_row_major
    from tests.kernels.oracles import reference_quantize_int8_convrot_weight

    w_int8 = _unpack_int4_row_major(wq)
    aq, ascale = reference_quantize_int8_convrot_weight(x, group_size=group_size)
    acc = aq.float() @ w_int8.float().T
    ref = (acc * ascale.reshape(m, 1).float() * wscale.reshape(1, n).float()).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.5)


def test_convrot_w4a4_linear_int4_optin_vs_torch_ref() -> None:
    """Default linear_dtype=int4 matches device act ConvRot-i4 + int4 mm.

    Covers the native iu4 path vs torch reference using
    ``reference_quantize_convrot_w4a4_weight`` on acts. Also smokes
    ``linear_dtype='int8'`` (unpack→iu8) on the same tensors.
    """
    try:
        from kernels.gemm.rdna4_iu4_gemm import iu4_gemm  # noqa: F401
    except ImportError:
        pytest.skip("iu4 GEMM atom not on this branch")
    from kernels.quant.rdna4_convrot_w4a4 import _unpack_int4_row_major
    from tests.kernels.oracles import reference_quantize_convrot_w4a4_weight

    torch.manual_seed(17)
    m, n, k = 64, 64, 256
    group_size = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    wq, wscale = quantize_convrot_w4a4_weight(w, convrot_groupsize=group_size)
    torch.cuda.synchronize()

    out4 = convrot_w4a4_linear(
        x,
        wq,
        wscale,
        convrot_groupsize=group_size,
        linear_dtype="int4",
        out_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    a_p, a_scale = reference_quantize_convrot_w4a4_weight(x, convrot_groupsize=group_size)
    a_i = _unpack_int4_row_major(a_p).to(torch.float32)
    b_i = _unpack_int4_row_major(wq).to(torch.float32)
    ref = (a_i @ b_i.T) * a_scale.reshape(m, 1) * wscale.reshape(1, n).float()
    torch.testing.assert_close(out4.float(), ref.float(), rtol=0.05, atol=0.5)

    # Default int8 still runs (different act quant) — smoke only.
    out8 = convrot_w4a4_linear(
        x,
        wq,
        wscale,
        convrot_groupsize=group_size,
        linear_dtype="int8",
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    assert out8.shape == out4.shape


def test_convrot_w4a4_linear_default_is_int4() -> None:
    """Production default is native int4 (linear_dtype default int4)."""
    import inspect

    from kernels.quant import rdna4_convrot_w4a4 as mod

    sig = inspect.signature(mod.convrot_w4a4_linear)
    assert sig.parameters["linear_dtype"].default == "int4"


def test_convrot_w4a4_int4_empty_mnk_raises() -> None:
    """linear_dtype=int4 raises on empty M; no silent unpack→iu8 demotion."""
    x = torch.randn(0, 64, device="cuda", dtype=torch.bfloat16)
    qw = torch.zeros(32, 32, device="cuda", dtype=torch.int8)
    ws = torch.ones(32, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="positive MNK"):
        convrot_w4a4_linear(x, qw, ws, convrot_groupsize=64, linear_dtype="int4")


def test_convrot_w4a4_int4_bad_wscales_raises() -> None:
    """Bad wscales on int4 path raise; do not demote to iu8."""
    torch.manual_seed(3)
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    qw, _ = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
    bad = torch.ones(16, device="cuda", dtype=torch.bfloat16)  # not float32
    with pytest.raises(ValueError, match="wscales must be float32"):
        convrot_w4a4_linear(x, qw, bad, convrot_groupsize=64, linear_dtype="int4")


def test_convrot_w4a4_odd_k_stays_in_kernel() -> None:
    """Odd K is zero-filled in the quant kernel. The output width is the Hadamard K."""
    torch.manual_seed(42)
    n, k = 32, 96  # 96 % 64 != 0
    group_size = 64
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    q, scale = quantize_convrot_w4a4_weight(w, convrot_groupsize=group_size)
    w_pad = torch.nn.functional.pad(w, (0, 32))
    q_pad, scale_pad = quantize_convrot_w4a4_weight(w_pad, convrot_groupsize=group_size)
    torch.cuda.synchronize()
    # Packed K is ceil(96, 64)=128 → k_half=64. The input row stays 96 wide.
    assert tuple(w.shape) == (n, k)
    assert q.shape == (n, 64), q.shape
    assert torch.equal(q, q_pad)
    torch.testing.assert_close(scale, scale_pad)
    recon = dequantize_convrot_w4a4_weight(
        q, scale, convrot_groupsize=group_size, output_dtype=torch.float32, logical_k=k
    )
    torch.cuda.synchronize()
    assert recon.shape == (n, k), recon.shape
    # Roundtrip is lossy (int4); just ensure finite and not all-zero.
    assert torch.isfinite(recon).all()
    assert recon.abs().mean() > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
