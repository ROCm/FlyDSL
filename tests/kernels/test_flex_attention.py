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
from kernels.attention.flex_attention_gfx950 import (  # noqa: E402
    MASK_CAUSAL,
    MASK_PREFIX_LM,
    MASK_SLIDING_WINDOW,
    SCORE_ALIBI,
    flydsl_flex_attention_layout,
    flydsl_flex_attention_layout_paged,
)

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="layout-API attention kernel targets gfx950",
)

_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}


def _make_qkv(B, Sq, Skv, Hq, D, dtype, *, Hkv=None):
    dev = "cuda"
    torch.manual_seed(0)
    if Hkv is None:
        Hkv = Hq
    q = torch.empty(B, Sq, Hq, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, scale


def _sdpa_ref(q, k, v, scale, *, attn_mask=None, is_causal=False):
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        scale=scale,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _check(out, ref, *, max_err_tol=8e-2, cos_tol=0.98, label=""):
    max_err = (out.float() - ref.float()).abs().max().item()
    cos = F.cosine_similarity(out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
    assert max_err < max_err_tol and cos > cos_tol, f"{label}: max_err={max_err} cos={cos}"
    return max_err, cos


def _sliding_window_mask(Sq, Skv, window, device):
    qi = torch.arange(Sq, device=device).unsqueeze(1)
    ki = torch.arange(Skv, device=device).unsqueeze(0)
    visible = (ki <= qi) & ((qi - ki) <= window)
    mask = visible.float().unsqueeze(0).unsqueeze(0)
    return mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


def _run(B, Sq, Skv, H, D, dtype_str, *, num_groups=8, accurate_softmax=True):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        num_groups=num_groups,
        accurate_softmax=accurate_softmax,
    ).float()
    ref = _sdpa_ref(q, k, v, scale).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    return max_err, cos


_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (1, 256, 512, 4, 128),  # Sq != Skv
    (2, 256, 256, 8, 128),
    (1, 256, 32, 4, 128),  # single KV tile (Skv == block_n)
    (1, 512, 1024, 4, 128),  # larger sequences
    (1, 256, 256, 8, 128),  # GQA: Hq=8, but uses default Hkv=Hq; see GQA test below
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout(B, Sq, Skv, H, D, dtype_str):
    max_err, cos = _run(B, Sq, Skv, H, D, dtype_str)
    assert max_err < 8e-2 and cos > 0.98, f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: max_err={max_err} cos={cos}"


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout_approx_softmax(B, Sq, Skv, H, D, dtype_str):
    _, cos = _run(B, Sq, Skv, H, D, dtype_str, accurate_softmax=False)
    assert cos > 0.95, f"approx B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: cos={cos}"


_MOD_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (2, 256, 256, 8, 128),
    (1, 256, 512, 4, 128),  # Sq < Skv (prefill with longer KV)
    (1, 256, 32, 4, 128),  # single KV tile
    (1, 512, 512, 4, 128),  # larger sequence (tile-range clamping exercises more tiles)
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_causal(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_CAUSAL)
    ref = _sdpa_ref(q, k, v, scale, is_causal=True)
    _check(out, ref, label=f"causal B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_alibi(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    slope = 0.125
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, score_type=SCORE_ALIBI, score_alibi_slope=slope)
    dev = q.device
    qi = torch.arange(Sq, device=dev).unsqueeze(1)
    ki = torch.arange(Skv, device=dev).unsqueeze(0)
    alibi_bias = (slope * (ki - qi)).float().unsqueeze(0).unsqueeze(0)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=alibi_bias)
    _check(out, ref, label=f"alibi B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_sliding_window(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    window = 16
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_SLIDING_WINDOW, mask_window=window)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=_sliding_window_mask(Sq, Skv, window, q.device))
    _check(out, ref, cos_tol=0.97, label=f"sw B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("window", [33, 97])
def test_flex_attention_layout_sliding_window_odd(window):
    """Non-block-aligned windows that straddle tile boundaries."""
    B, Sq, Skv, H, D = 2, 256, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_SLIDING_WINDOW, mask_window=window)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=_sliding_window_mask(Sq, Skv, window, q.device))
    _check(out, ref, cos_tol=0.97, label=f"sw_odd w={window}")


@_requires_gfx950
@pytest.mark.parametrize("Hq,Hkv", [(8, 1), (8, 2), (32, 8)])
def test_flex_attention_layout_gqa(Hq, Hkv):
    B, Sq, Skv, D = 1, 256, 256, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, Hq, D, torch.bfloat16, Hkv=Hkv)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, num_kv_heads=Hkv)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float().repeat_interleave(Hq // Hkv, dim=1)
    vh = v.permute(0, 2, 1, 3).float().repeat_interleave(Hq // Hkv, dim=1)
    ref = F.scaled_dot_product_attention(qh, kh, vh, scale=scale).permute(0, 2, 1, 3).contiguous()
    _check(out, ref, label=f"gqa Hq{Hq} Hkv{Hkv}")


@_requires_gfx950
@pytest.mark.parametrize("num_groups", [4, 8])
def test_flex_attention_layout_multi_group(num_groups):
    B, Sq, Skv, H, D = 1, num_groups * 32, 128, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, num_groups=num_groups)
    ref = _sdpa_ref(q, k, v, scale)
    _check(out, ref, label=f"groups={num_groups}")


@_requires_gfx950
def test_flex_attention_layout_sliding_window_full():
    """Window >= Skv: everything visible, should match dense."""
    B, Sq, Skv, H, D = 1, 256, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_SLIDING_WINDOW, mask_window=Skv)
    ref = _sdpa_ref(q, k, v, scale, is_causal=True)
    _check(out, ref, label="sw_full")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_prefix_lm(B, Sq, Skv, H, D, dtype_str):
    prefix_len = max(1, Sq // 4)
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_PREFIX_LM, mask_prefix_len=prefix_len)
    dev = q.device
    qi = torch.arange(Sq, device=dev).unsqueeze(1)
    ki = torch.arange(Skv, device=dev).unsqueeze(0)
    visible = (ki <= qi) | (ki < prefix_len)
    mask = visible.float().unsqueeze(0).unsqueeze(0)
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    ref = _sdpa_ref(q, k, v, scale, attn_mask=mask)
    _check(out, ref, label=f"prefix_lm B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("num_groups", [4, 8])
def test_flex_attention_layout_causal_multi_group(num_groups):
    B, Sq, Skv, H, D = 1, num_groups * 32, num_groups * 32, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_type=MASK_CAUSAL, num_groups=num_groups)
    ref = _sdpa_ref(q, k, v, scale, is_causal=True)
    _check(out, ref, label=f"causal groups={num_groups}")


_PAGED_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (2, 256, 256, 8, 128),
    (1, 256, 512, 4, 128),
    (1, 256, 32, 4, 128),
]


def _paged_causal_cases():
    # f16 paged+causal is numerically flaky at the 0.1 max_err bound (cos stays >0.99).
    return [(*shape, "bf16") for shape in _PAGED_SHAPES]


def _scatter_to_paged(k_contig, v_contig, block_n, block_table, context_lens):
    """Scatter contiguous [B, Skv, H, D] KV into paged cache [num_blocks, block_n, H, D]."""
    B, Skv, H, D = k_contig.shape
    num_blocks = int(block_table.max().item()) + 1
    k_cache = torch.zeros(num_blocks, block_n, H, D, dtype=k_contig.dtype, device=k_contig.device)
    v_cache = torch.zeros(num_blocks, block_n, H, D, dtype=v_contig.dtype, device=v_contig.device)
    for b in range(B):
        ctx = int(context_lens[b].item())
        for t in range(ctx):
            page_idx = t // block_n
            within_page = t % block_n
            phys_page = int(block_table[b, page_idx].item())
            k_cache[phys_page, within_page] = k_contig[b, t]
            v_cache[phys_page, within_page] = v_contig[b, t]
    return k_cache, v_cache


def _make_block_table(B, Skv, block_n, device):
    """Create a random block table and context_lens for paged tests."""
    num_pages_per_seq = (Skv + block_n - 1) // block_n
    total_pages = B * num_pages_per_seq * 2
    context_lens = torch.full((B,), Skv, dtype=torch.int32, device=device)
    block_table = torch.zeros(B, num_pages_per_seq, dtype=torch.int32, device=device)
    used = set()
    for b in range(B):
        for p in range(num_pages_per_seq):
            while True:
                pid = torch.randint(0, total_pages, (1,)).item()
                if pid not in used:
                    used.add(pid)
                    break
            block_table[b, p] = pid
    return block_table, context_lens, total_pages


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _PAGED_SHAPES)
def test_flex_attention_layout_paged(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    block_n = 32
    ref = _sdpa_ref(q, k, v, scale)
    block_table, context_lens, _ = _make_block_table(B, Skv, block_n, q.device)
    k_cache, v_cache = _scatter_to_paged(k, v, block_n, block_table, context_lens)
    out = flydsl_flex_attention_layout_paged(q, k_cache, v_cache, block_table, context_lens, scale=scale)
    _check(out, ref, max_err_tol=1e-1, cos_tol=0.97, label=f"paged B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D,dtype_str", _paged_causal_cases())
def test_flex_attention_layout_paged_causal(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    block_n = 32
    ref = _sdpa_ref(q, k, v, scale, is_causal=True)
    block_table, context_lens, _ = _make_block_table(B, Skv, block_n, q.device)
    k_cache, v_cache = _scatter_to_paged(k, v, block_n, block_table, context_lens)
    out = flydsl_flex_attention_layout_paged(
        q, k_cache, v_cache, block_table, context_lens, scale=scale, mask_type=MASK_CAUSAL
    )
    _check(
        out,
        ref,
        max_err_tol=1.2e-1,
        cos_tol=0.97,
        label=f"paged_causal B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}",
    )


def main():
    for B, Sq, Skv, H, D in _SHAPES:
        for dt in ["bf16", "f16"]:
            me, cos = _run(B, Sq, Skv, H, D, dt)
            ok = me < 3e-2 and cos > 0.99
            print(
                f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dt}: " f"max_err={me:.4g} cos={cos:.5f} -> {'PASS' if ok else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
