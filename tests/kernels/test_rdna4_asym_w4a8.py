#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device tests for gfx120x Asym W4A8 dequant (+ host quant wire check)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_asym_w4a8 import (  # noqa: E402
    dequant_int4_grouped_to_int8,
    quantize_w4a8_int8_weight,
    w4a8_int8_linear,
)
from tests.kernels.oracles import reference_dequant_int4_grouped_to_int8  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"Asym W4A8 requires gfx120x, got {ARCH}", allow_module_level=True)


@pytest.mark.parametrize("group_size", [16])
@pytest.mark.parametrize("codebook", [True, False])
def test_dequant_int4_grouped_to_int8(group_size: int, codebook: torch.Tensor) -> None:
    torch.manual_seed(20260930 + int(codebook))
    n, k = 32, 256
    # Host quant produces wire format; dequant kernel must match reference.
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1.5, 1.5)
    packed, s_rel, s_channel, correction, cb = quantize_w4a8_int8_weight(
        w,
        group_size=group_size,
        convrot_groupsize=256,
        codebook=codebook,
        scale_dtype=torch.float32,
    )
    assert correction is None
    torch.cuda.synchronize()

    out = dequant_int4_grouped_to_int8(packed, s_rel, cb if codebook else None, group_size=group_size)
    torch.cuda.synchronize()
    ref = reference_dequant_int4_grouped_to_int8(packed, s_rel, cb if codebook else None, group_size=group_size)
    mism = (out.cpu() != ref.cpu()).sum().item()
    frac = mism / out.numel()
    assert frac < 0.02, f"dequant mismatch frac={frac} count={mism}/{out.numel()}"


def test_quantize_w4a8_wire_shape_wan_like() -> None:
    """Pack shape for Wan-like K=1024 (hot model dim; comment-only)."""
    torch.manual_seed(3)
    w = torch.randn((48, 1024), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    packed, s_rel, s_channel, correction, cb = quantize_w4a8_int8_weight(
        w, group_size=16, convrot_groupsize=256, codebook=True
    )
    assert packed.shape == (48, 512)
    assert s_rel.shape == (48, 1024 // 16)
    assert s_channel.shape == (48,)
    assert correction is None
    assert cb is not None and tuple(cb.shape) == (16,)
    assert packed.dtype == torch.int8


def test_w4a8_int8_linear_matches_ref() -> None:
    try:
        from kernels.gemm.rdna4_int8_linear import create_wmma_int8_linear_module  # noqa: F401
    except ImportError:
        pytest.skip("rdna4_int8_linear (iu8 atom) not on this branch")
    torch.manual_seed(17)
    m, n, k = 64, 64, 256
    group_size = 16
    convrot_g = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    packed, s_rel, s_channel, correction, cb = quantize_w4a8_int8_weight(
        w,
        group_size=group_size,
        convrot_groupsize=convrot_g,
        codebook=True,
        scale_dtype=torch.float32,
    )
    assert correction is None
    torch.cuda.synchronize()

    out = w4a8_int8_linear(
        x,
        packed,
        s_rel,
        s_channel,
        codebook=cb,
        group_size=group_size,
        convrot_groupsize=convrot_g,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    from tests.kernels.oracles import reference_quantize_int8_convrot_weight

    w_int8 = reference_dequant_int4_grouped_to_int8(packed, s_rel, cb, group_size)
    aq, ascale = reference_quantize_int8_convrot_weight(x, group_size=convrot_g)
    acc = aq.float() @ w_int8.float().T
    ref = (acc * ascale.reshape(m, 1).float() * s_channel.reshape(1, n).float()).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.5)


def test_asymmetric_scale_uses_max_not_absmax() -> None:
    """Mode 2 scale is (max - min) / 15. abs-max collapses codes when the negative peak wins."""
    from kernels.quant.rdna4_asym_w4a8 import _quantize_rotated_w4a8

    vals = [-1.0, -0.2, 0.1] + [0.0] * 13
    w = torch.tensor([vals], device="cuda", dtype=torch.float32)
    packed, _s_rel, _sch, _corr, _cb = _quantize_rotated_w4a8(
        w, group_size=16, symmetric=False, scale_dtype=torch.float32, codebook=False
    )
    torch.cuda.synchronize()
    raw = packed.view(torch.uint8).cpu()
    codes = []
    for byte in raw.view(-1).tolist():
        codes.append(int(byte) & 15)
        codes.append((int(byte) >> 4) & 15)
    # abs-max scale would emit [0, 6, 8, ...]. (max-min)/15 emits [0, 11, 15, 14].
    assert codes[0] == 0
    assert codes[1] == 11
    assert codes[2] == 15


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
