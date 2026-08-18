#!/usr/bin/env python3
"""Correctness test for the layout-API flex/flash attention forward (gfx950).

Compares flydsl_flex_attention_layout against torch scaled_dot_product_attention
(non-causal). gfx950-only (uses cdna4-era MFMA + the layout API); skipped elsewhere.

Phase 0 kernel constraints (see the kernel's make_flex_attn_param): block_m=32,
block_n multiple of 32, head_dim multiple of 32, seqlen_kv multiple of block_n.
"""
import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

import pytest  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flex_attention_layout_gfx950 import (  # noqa: E402
    flydsl_flex_attention_layout,
    MASK_NONE,
    MASK_CAUSAL,
    MASK_SLIDING_WINDOW,
    SCORE_NONE,
    SCORE_ALIBI,
)

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="layout-API attention kernel targets gfx950",
)

_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}


def _sdpa_ref(q, k, v, scale, *, attn_mask=None, is_causal=False):
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    out = F.scaled_dot_product_attention(
        qh, kh, vh, scale=scale, attn_mask=attn_mask, is_causal=is_causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _run(B, Sq, Skv, H, D, dtype_str, *, num_groups=2, accurate_softmax=True):
    dtype = _DTYPES[dtype_str]
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)

    out = flydsl_flex_attention_layout(
        q, k, v, scale=scale, num_groups=num_groups,
        accurate_softmax=accurate_softmax,
    ).float()
    ref = _sdpa_ref(q, k, v, scale).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    return max_err, cos


_SHAPES = [
    (1, 64, 64, 4, 128),
    (1, 64, 128, 4, 128),
    (2, 128, 128, 8, 128),
    (1, 64, 64, 4, 64),
    (1, 128, 128, 4, 128),
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout(B, Sq, Skv, H, D, dtype_str):
    max_err, cos = _run(B, Sq, Skv, H, D, dtype_str)
    assert max_err < 8e-2 and cos > 0.98, (
        f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: max_err={max_err} cos={cos}"
    )


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout_approx_softmax(B, Sq, Skv, H, D, dtype_str):
    _, cos = _run(B, Sq, Skv, H, D, dtype_str, accurate_softmax=False)
    assert cos > 0.95, f"approx B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: cos={cos}"


_MOD_SHAPES = [
    (1, 64, 64, 4, 128),
    (2, 128, 128, 8, 128),
]


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_causal(B, Sq, Skv, H, D):
    dtype = torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)

    out = flydsl_flex_attention_layout(
        q, k, v, scale=scale, mask_type=MASK_CAUSAL,
    ).float()
    ref = _sdpa_ref(q, k, v, scale, is_causal=True).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    assert max_err < 8e-2 and cos > 0.98, (
        f"causal B{B} Sq{Sq} Skv{Skv} H{H} D{D}: max_err={max_err} cos={cos}"
    )


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_alibi(B, Sq, Skv, H, D):
    dtype = torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)
    slope = 0.125

    out = flydsl_flex_attention_layout(
        q, k, v, scale=scale, score_type=SCORE_ALIBI, score_alibi_slope=slope,
    ).float()
    # Reference: manually add alibi bias to SDPA attn_mask
    qi = torch.arange(Sq, device=dev).unsqueeze(1)
    ki = torch.arange(Skv, device=dev).unsqueeze(0)
    alibi_bias = (slope * (ki - qi)).float().unsqueeze(0).unsqueeze(0)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=alibi_bias).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    assert max_err < 8e-2 and cos > 0.98, (
        f"alibi B{B} Sq{Sq} Skv{Skv} H{H} D{D}: max_err={max_err} cos={cos}"
    )


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_sliding_window(B, Sq, Skv, H, D):
    dtype = torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)
    window = 16

    out = flydsl_flex_attention_layout(
        q, k, v, scale=scale, mask_type=MASK_SLIDING_WINDOW, mask_window=window,
    ).float()
    # Reference: causal + window distance mask
    qi = torch.arange(Sq, device=dev).unsqueeze(1)
    ki = torch.arange(Skv, device=dev).unsqueeze(0)
    causal = ki <= qi
    in_window = (qi - ki) <= window
    mask = (causal & in_window).float().unsqueeze(0).unsqueeze(0)
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=mask).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    assert max_err < 8e-2 and cos > 0.97, (
        f"sw B{B} Sq{Sq} Skv{Skv} H{H} D{D}: max_err={max_err} cos={cos}"
    )


def main():
    for (B, Sq, Skv, H, D) in _SHAPES:
        for dt in ["bf16", "f16"]:
            me, cos = _run(B, Sq, Skv, H, D, dt)
            ok = me < 3e-2 and cos > 0.99
            print(
                f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dt}: "
                f"max_err={me:.4g} cos={cos:.5f} -> {'PASS' if ok else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
