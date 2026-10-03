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
    dequantize_int8_convrot_weight,
    int8_linear_convrot,
    quantize_int8_convrot_weight,
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


def test_dequant_roundtrip_unrotates() -> None:
    torch.manual_seed(11)
    w = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_int8_convrot_weight(w, group_size=256)
    torch.cuda.synchronize()
    recon = dequantize_int8_convrot_weight(q, scale, group_size=256, out_dtype=torch.float32)
    # Reconstruction error dominated by INT8 quantization, not rotation.
    err = (recon - w.float()).abs().max().item()
    assert err < 0.5, f"max_abs roundtrip={err}"


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
