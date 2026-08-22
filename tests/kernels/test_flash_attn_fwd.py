#!/usr/bin/env python3
"""flash_attn_func kernel test and benchmark for FlyDSL.

Tests flash_attn_func against PyTorch SDPA.
"""

import argparse
import csv
import hashlib
import logging
import math
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import numpy as np
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
from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module  # noqa: E402
from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # noqa: E402
from kernels.attention.flash_attn_utils import (  # noqa: E402
    BIAS_MAX_DESCRIPTOR_BYTES,
    BIAS_MAX_OFFSET_ELEMS,
    bias_addressing_error,
)
from tests.test_common import run_perftest  # noqa: E402

UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123
PAGED_KV_MIN_CONTEXT_LENGTH = 16384
# Target share of the softmax mass placed on the attention sink (see calibrate_sink).
DEFAULT_SINK_SHARE = 0.5
# fp8 correctness gate (fixed; fp8 is lossy).
FP8_MAX_ERR = 5e-2
FP8_MIN_COS = 0.98
# OCP e4m3fn (NOT the fnuz variant) end-to-end on gfx950.
FP8_DTYPE = torch.float8_e4m3fn
# Defaults are used only if helpers run before main() populates CLI options.
FLASH_ATTN_FUNC_KERNEL_CONFIG: dict = {
    "waves_per_eu": 2,
    "daz": True,
    "dualwave_swp_lazy_rescale": True,
    "dualwave_swp_setprio": True,
    "dualwave_swp_debug_lazy_counts": False,
    "dualwave_swp_enable_stagger": True,
}

# (batch, seq_len, num_heads, num_kv_heads, num_kv_splits)
# num_kv_heads == num_heads -> MHA; num_kv_heads < num_heads -> GQA/MQA.
# num_kv_splits > 1 -> split-K path (gfx950 DUALWAVE_SWP only, seq_len >= 384, D=64/128).
DEFAULT_CONFIGS = [
    # set1
    (16, 8192, 64, 64, 1),
    (16, 8192, 64, 8, 1),
    (2, 1024, 64, 64, 1),
    # set2
    (8, 128, 64, 64, 1),
    (8, 256, 64, 64, 1),
    (8, 512, 64, 64, 1),
    (1, 128, 64, 64, 1),
    (1, 256, 64, 64, 1),
    (1, 384, 64, 64, 1),
    (1, 512, 64, 64, 1),
    (1, 1024, 64, 64, 1),
    (1, 2048, 64, 64, 1),
    (1, 4096, 64, 64, 1),
    (1, 8192, 64, 64, 1),
    (4, 8192, 64, 64, 1),
    (1, 2048, 32, 32, 1),
    (1, 4096, 32, 32, 1),
    (1, 8192, 32, 32, 1),
    (8, 8192, 32, 32, 1),
    (16, 8192, 16, 16, 1),
    (1, 8192, 16, 16, 1),
    (1, 2048, 16, 16, 1),
    (1, 4096, 16, 16, 1),
    (1, 2048, 8, 8, 1),
    (1, 4096, 8, 8, 1),
    (1, 8192, 8, 8, 1),
    (32, 8192, 8, 8, 1),
    # set3
    (1, 8192, 2, 2, 4),
    (1, 4096, 2, 2, 4),
    (1, 2048, 4, 4, 4),
    (1, 8192, 4, 4, 2),
    # set4
    (1, 98144, 3, 3, 5),
    (1, 147216, 3, 3, 5),
    (1, 196288, 3, 3, 5),
    (1, 245360, 3, 3, 5),
    (1, 294432, 3, 3, 5),
    (1, 12268, 24, 24, 1),
    (1, 18402, 24, 24, 1),
    (1, 24536, 24, 24, 1),
    (1, 30670, 24, 24, 2),
    (1, 36804, 24, 24, 2),
    (1, 32768, 24, 24, 1),
    (1, 32768, 32, 32, 1),
    # set5
    (3, 64, 4, 4, 1),
    (3, 1, 4, 4, 1),
    (3, 31, 4, 4, 1),
    (3, 33, 4, 4, 1),
    (2, 63, 4, 4, 1),
    (2, 65, 4, 4, 1),
    (2, 127, 4, 4, 1),
    (2, 129, 4, 4, 1),
    (1, 255, 4, 4, 1),
    (1, 257, 4, 4, 1),
    (1, 511, 4, 4, 1),
    (1, 513, 4, 4, 1),
]

# Additional dense/varlen/cross-length cases.
# Rows: [Sq, Skv, B, H, Hkv, kv_splits].
# Skv=None means packed varlen self-attn; B=None means packed varlen cross-attn.
# D is swept separately by head_dims_to_test.
EXTRA_CONFIGS = [
    # varlen
    [[1024, 1024], None, None, 64, 64, 1],
    [[1024, 8192], None, None, 64, 8, 1],
    [[1024, 8192, 3, 31, 65, 127], None, None, 64, 8, 1],
    [[1, 3, 31, 33, 63, 65], None, None, 64, 64, 1],
    # cross-length
    [31, 65, 1, 64, 8, 1],
    [63, 33, 1, 64, 8, 1],
    [129, 255, 1, 64, 8, 1],
    [257, 127, 1, 64, 8, 1],
    [1024, 8192, 1, 64, 8, 1],
    [8192, 1024, 1, 64, 8, 1],
    # varlen cross-length
    [[1024, 8192], [8192, 1024], None, 64, 8, 1],
    [[31, 33, 63, 65], [65, 63, 33, 31], None, 64, 64, 1],
    [[127, 129, 255, 257], [257, 255, 129, 127], None, 64, 8, 1],
]


def _short_label(value):
    label = str(value)
    return label if len(label) <= 24 else label[:21] + "..."


def _extra_case_from_config(row):
    seqlen_q, seqlen_kv, batch, nh, nh_kv, kv_splits = row
    if seqlen_kv is None:
        return {
            "sq_label": _short_label(seqlen_q),
            "skv_label": _short_label(seqlen_q),
            "nh": nh,
            "nh_kv": nh_kv,
            "kv_splits": kv_splits,
            "kwargs": {"varlen_seqlens_q": list(seqlen_q)},
        }
    if batch is not None:
        return {
            "sq_label": f"[{seqlen_q}]",
            "skv_label": f"[{seqlen_kv}]",
            "nh": nh,
            "nh_kv": nh_kv,
            "kv_splits": kv_splits,
            "kwargs": {"batch": batch, "seqlen_q": seqlen_q, "seqlen_kv": seqlen_kv},
        }
    return {
        "sq_label": _short_label(seqlen_q),
        "skv_label": _short_label(seqlen_kv),
        "nh": nh,
        "nh_kv": nh_kv,
        "kv_splits": kv_splits,
        "kwargs": {"varlen_seqlens_q": list(seqlen_q), "varlen_seqlens_kv": list(seqlen_kv)},
    }


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility across all RNG sources."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_alibi_slopes(batch, num_heads, two_d=False, device="cuda"):
    """Positive fp32 ALiBi slopes, [batch, num_heads] if two_d else [num_heads].

    The canonical geometric ladder 2**(-8*(h+1)/H) with a per-batch jitter on the
    2D form, so a bug that broadcast or swapped the head/batch index cannot alias
    into a pass. Deterministic: consumes no RNG, so Q/K/V stay bit-identical to a
    run without ALiBi.
    """
    base = torch.tensor(
        [2.0 ** (-((h + 1) * 8.0 / num_heads)) for h in range(num_heads)], dtype=torch.float32, device=device
    )
    if not two_d:
        return base.contiguous()
    jitter = 1.0 + 0.25 * torch.arange(batch, dtype=torch.float32, device=device)[:, None]
    return (base[None, :] * jitter).contiguous()


def _alibi_term(alibi_slopes, q_lo, q_hi, delta, key_idx):
    """-slope * |i + delta - j| for q rows [q_lo, q_hi) -> [B or 1, H, q_hi-q_lo, Skv].

    Built per Q chunk rather than materialized whole: a full [B, H, Sq, Skv] fp32
    ALiBi matrix is the same size as the score matrix the caller is already
    chunking to avoid.
    """
    i = torch.arange(q_lo, q_hi, device=alibi_slopes.device)[:, None]
    rel = (i + delta - key_idx.view(1, -1)).abs().to(torch.float32)
    s = alibi_slopes.float()
    if s.dim() == 1:
        s = s.unsqueeze(0)
    return -s[:, :, None, None] * rel


def _rows_logsumexp(q_t, k_t, causal, bias=None, alibi_slopes=None):
    """Per-head sum and count of row logsumexp(scores), chunked over Q.

    q_t: [B, Sq, H, D], k_t: [B, Skv, Hkv, D]. Returns (sum_per_head, count),
    both fp32 [H] / scalar, over finite rows only.
    """
    q_h = q_t.transpose(1, 2).float()
    k_h = k_t.transpose(1, 2).float()
    B, H, Sq, D = q_h.shape
    Skv = k_h.shape[2]
    if H != k_h.shape[1]:
        k_h = k_h.repeat_interleave(H // k_h.shape[1], dim=1)
    delta = Skv - Sq
    scale = 1.0 / math.sqrt(D)
    k_trans = k_h.transpose(-1, -2).contiguous()
    key_idx = torch.arange(Skv, device=q_t.device).view(1, 1, 1, Skv)
    chunk = max(1, min(Sq, (64 * 1024 * 1024) // max(B * H * Skv, 1)))
    total = torch.zeros(H, dtype=torch.float32, device=q_t.device)
    count = 0
    for s0 in range(0, Sq, chunk):
        s1 = min(s0 + chunk, Sq)
        sc = torch.matmul(q_h[:, :, s0:s1, :], k_trans) * scale
        if alibi_slopes is not None:
            sc = sc + _alibi_term(alibi_slopes, s0, s1, delta, key_idx)
        if bias is not None:
            sc = sc + bias[s0:s1].float()
        if causal:
            q_idx = torch.arange(s0, s1, device=q_t.device).view(1, 1, -1, 1)
            sc = sc.masked_fill(key_idx > q_idx + delta, float("-inf"))
        lse = torch.logsumexp(sc, dim=-1)  # [B, H, chunk]
        finite = torch.isfinite(lse)
        total += torch.where(finite, lse, torch.zeros_like(lse)).sum(dim=(0, 2))
        count += int(finite[:, 0, :].sum().item()) if finite.numel() else 0
    return total, max(count, 1)


def calibrate_sink(sum_lse, count, share):
    """Per-head sink logit placing `share` of the softmax mass on the sink.

    share = sigmoid(sink - logsumexp(scores)), so invert it per head. Calibration
    matters: logsumexp grows like ln(seqlen), so a sink drawn near 0 would own a
    fraction of a percent of the mass -- below the bf16 noise floor, and a row
    with a dropped sink would pass just as happily.
    """
    return (sum_lse / count + math.log(share / (1.0 - share))).float().contiguous()


def _sink_softmax(scores, sink):
    """softmax over [scores, sink], where the sink carries no value row.

    Returns probs summing to 1 - sink_share. A fully-masked row needs no
    special-casing: the max collapses to the sink, every score term is
    exp(-inf) = 0, and the row becomes all-sink -- probs 0, matching the kernel.
    """
    s = sink.view(1, -1, 1, 1)
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), s)
    e = torch.exp(scores - m)
    return e / (e.sum(dim=-1, keepdim=True) + torch.exp(s - m))


def pytorch_ref_attention(q, k, v, causal=True, bias=None, alibi_slopes=None, sink=None):
    if bias is not None or alibi_slopes is not None or sink is not None:
        # These must land after the sm_scale multiply and before the mask, and SDPA
        # rejects an additive attn_mask together with is_causal, so defer to the
        # explicit chunked path (identical result when Sq == Skv).
        return pytorch_ref_attention_qkv_diff(q, k, v, causal=causal, bias=bias, alibi_slopes=alibi_slopes, sink=sink)
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        assert nh_q % nh_kv == 0, f"num_heads ({nh_q}) must be divisible by num_kv_heads ({nh_kv})"
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    score_elems = q_t.shape[0] * q_t.shape[1] * q_t.shape[2] * k_t.shape[2]
    if score_elems > 128 * 1024 * 1024:
        return pytorch_ref_attention_chunked(q_t, k_t, v_t, causal=causal).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


@torch.no_grad()
def pytorch_ref_attention_chunked(q_t, k_t, v_t, causal=True):
    """Compute reference attention in Q chunks to avoid large SDPA workspaces."""
    B, H, S, D = q_t.shape
    max_score_elems = 1024 * 1024 * 1024  # 1 GiB → larger chunks, fewer kernel launches
    chunk_size = max(1, min(S, max_score_elems // max(B * H * S, 1)))
    out = torch.empty((B, H, S, D), device=q_t.device, dtype=torch.float32)
    k_trans = k_t.transpose(-1, -2).contiguous()
    scale = 1.0 / math.sqrt(D)
    key_idx = torch.arange(S, device=q_t.device).view(1, 1, 1, S)

    for q_start in range(0, S, chunk_size):
        q_end = min(q_start + chunk_size, S)
        q_chunk = q_t[:, :, q_start:q_end, :]
        scores = torch.matmul(q_chunk, k_trans) * scale
        if causal:
            q_idx = torch.arange(q_start, q_end, device=q_t.device).view(1, 1, -1, 1)
            scores = scores.masked_fill(key_idx > q_idx, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, q_start:q_end, :] = torch.matmul(probs, v_t)

    return out


@torch.no_grad()
def pytorch_ref_attention_qkv_diff(q, k, v, causal=True, bias=None, alibi_slopes=None, sink=None):
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        assert nh_q % nh_kv == 0, f"num_heads ({nh_q}) must be divisible by num_kv_heads ({nh_kv})"
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    B, H, Sq, D = q_t.shape
    Skv = k_t.shape[2]
    delta = Skv - Sq
    scale = 1.0 / math.sqrt(D)
    k_trans = k_t.transpose(-1, -2).contiguous()
    out = torch.empty((B, H, Sq, D), device=q_t.device, dtype=torch.float32)
    chunk = max(1, min(Sq, (64 * 1024 * 1024) // max(B * H * Skv, 1)))
    key_idx = torch.arange(Skv, device=q_t.device).view(1, 1, 1, Skv)
    for s0 in range(0, Sq, chunk):
        s1 = min(s0 + chunk, Sq)
        scores = torch.matmul(q_t[:, :, s0:s1, :], k_trans) * scale
        if alibi_slopes is not None:
            scores = scores + _alibi_term(alibi_slopes, s0, s1, delta, key_idx)
        if bias is not None:
            scores = scores + bias[s0:s1].float()
        if causal:
            q_idx = torch.arange(s0, s1, device=q_t.device).view(1, 1, -1, 1)
            scores = scores.masked_fill(key_idx > q_idx + delta, float("-inf"))
        if sink is not None:
            # The sink also fixes the all-masked row: it becomes all-sink, probs 0.
            probs = _sink_softmax(scores, sink)
        else:
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)  # all-masked row -> 0 output
        out[:, :, s0:s1, :] = torch.matmul(probs, v_t)
    return out.transpose(1, 2)


def _ceil_div(a, b):
    return (a + b - 1) // b


def bias_fits(rows, cols, elem_size=2):
    """Skip predicate mirroring the kernel's own bias addressing limits."""
    why = bias_addressing_error(rows * cols, elem_size)
    return why is None, why or ""


def _block_table_from_indices(kv_indptr_cpu, kv_indices_cpu, batch_size, max_num_pages_per_seq):
    block_table_cpu = torch.zeros((batch_size, max_num_pages_per_seq), dtype=torch.int32, device="cpu")
    for b in range(batch_size):
        start = kv_indptr_cpu[b].item()
        end = kv_indptr_cpu[b + 1].item()
        block_table_cpu[b, : end - start] = kv_indices_cpu[start:end]
    return block_table_cpu


def _kv_vector_size(dtype):
    return 16 // torch.empty((), dtype=dtype).element_size()


def _vectorize_paged_kv(k_cache, v_cache, num_kv_heads, head_dim, page_size, k_vector_size):
    k_cache = (
        k_cache.contiguous()
        .view(-1, page_size, num_kv_heads, head_dim // k_vector_size, k_vector_size)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.contiguous()
        .view(-1, page_size // k_vector_size, k_vector_size, num_kv_heads, head_dim)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


def _validate_kv_cache_layout(kv_cache_layout, page_size, head_dim, dtype):
    if kv_cache_layout not in ("linear", "vectorized"):
        return f"unsupported kv_cache_layout={kv_cache_layout}"
    if kv_cache_layout == "vectorized":
        k_vector_size = _kv_vector_size(dtype)
        if page_size % k_vector_size != 0 or head_dim % k_vector_size != 0:
            return (
                f"vectorized K/V cache layout requires page_size and head_dim divisible by "
                f"kVectorSize={k_vector_size}"
            )
    return None


def _build_paged_kv_for_test(
    batch_size,
    max_kv_len,
    page_size,
    num_kv_heads,
    head_dim,
    kv_lens,
    dtype,
    device,
    kv_cache_layout,
):
    """Build physical paged K/V cache plus both page-table forms for reference tests.

    Supported ``kv_cache_layout`` values:
    - ``linear``: 4D paged K/V, ``[NumBlocks, PageSize, NumKVHeads, HeadDim]``.
    - ``vectorized``: aiter-style 5D K/V, where
      ``K = [NumBlocks, NumKVHeads, HeadDim / kVectorSize, PageSize, kVectorSize]`` and
      ``V = [NumBlocks, NumKVHeads, PageSize / kVectorSize, HeadDim, kVectorSize]``.
      Here ``kVectorSize = 16 / element_size`` (bf16/fp16: 8, fp8: 16);
      page_size and head_dim must be divisible by it.
    """
    max_context_length = max(PAGED_KV_MIN_CONTEXT_LENGTH, max_kv_len)
    max_num_pages_per_seq = _ceil_div(max_context_length, page_size)
    total_num_pages = max_num_pages_per_seq * batch_size
    page_shape = (total_num_pages, page_size, num_kv_heads, head_dim)

    k_cache_4d = torch.empty(page_shape, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    v_cache_4d = torch.empty(page_shape, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    if kv_cache_layout == "linear":
        k_cache, v_cache = k_cache_4d, v_cache_4d
    elif kv_cache_layout == "vectorized":
        k_vector_size = _kv_vector_size(dtype)
        k_cache, v_cache = _vectorize_paged_kv(
            k_cache_4d,
            v_cache_4d,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
        )
    else:
        raise ValueError(f"unsupported kv_cache_layout={kv_cache_layout}")

    kv_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")
    kv_num_used_pages = torch.div(kv_lens_cpu + page_size - 1, page_size, rounding_mode="floor").int()
    kv_indptr_cpu = torch.cumsum(
        torch.cat((torch.tensor([0], dtype=torch.int32, device="cpu"), kv_num_used_pages)), dim=0
    ).int()
    kv_indices_cpu = torch.nn.functional.pad(torch.randperm(total_num_pages, device="cpu").int(), (0, 128), value=0)
    kv_last_page_len_cpu = ((kv_lens_cpu - 1) % page_size + 1).int()
    block_table_cpu = _block_table_from_indices(
        kv_indptr_cpu,
        kv_indices_cpu,
        batch_size,
        max_num_pages_per_seq,
    )

    return {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_lens_cpu": kv_lens_cpu,
        "kv_indptr_cpu": kv_indptr_cpu,
        "kv_indices_cpu": kv_indices_cpu,
        "kv_last_page_len_cpu": kv_last_page_len_cpu,
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "seqlen_k": kv_lens_cpu.to(device),
        "page_size": page_size,
        "max_context_length": max_context_length,
        "kv_cache_layout": kv_cache_layout,
    }


def _page_ids_for_batch(kv_cache, batch_idx):
    kv_len = kv_cache["kv_lens_cpu"][batch_idx].item()
    num_pages = _ceil_div(kv_len, kv_cache["page_size"])
    return kv_cache["block_table"][batch_idx, :num_pages].long()


def _logical_kv_from_pages(k_pages, v_pages, kv_cache_layout, kv_len):
    if kv_cache_layout == "linear":
        kb = k_pages.reshape(-1, k_pages.shape[-2], k_pages.shape[-1])
        vb = v_pages.reshape(-1, v_pages.shape[-2], v_pages.shape[-1])
    elif kv_cache_layout == "vectorized":
        kb = (
            k_pages.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(-1, k_pages.shape[1], k_pages.shape[2] * k_pages.shape[4])
        )
        vb = v_pages.permute(0, 2, 4, 1, 3).contiguous().reshape(-1, v_pages.shape[1], v_pages.shape[3])
    else:
        raise ValueError(f"unsupported kv_cache_layout={kv_cache_layout}")
    return kb[:kv_len], vb[:kv_len]


def _materialize_block_table_kv(kv_cache, dense):
    """Gather physical pages through the vLLM block table into logical K/V tensors."""
    k_cache = kv_cache["k_cache"]
    v_cache = kv_cache["v_cache"]
    kv_cache_layout = kv_cache["kv_cache_layout"]
    kv_lens = kv_cache["kv_lens_cpu"].tolist()

    k_batches = []
    v_batches = []
    for b, kv_len in enumerate(kv_lens):
        page_ids = _page_ids_for_batch(kv_cache, b)
        kb, vb = _logical_kv_from_pages(k_cache[page_ids], v_cache[page_ids], kv_cache_layout, kv_len)
        k_batches.append(kb)
        v_batches.append(vb)

    if dense:
        return torch.stack(k_batches, dim=0).contiguous(), torch.stack(v_batches, dim=0).contiguous()
    return torch.cat(k_batches, dim=0).contiguous(), torch.cat(v_batches, dim=0).contiguous()


def _build_paged_kv_from_logical_for_aiter(inputs, page_size=16):
    """Repack logical K/V into page_size=16 physical cache for aiter batch-prefill."""
    src_cache = inputs["kv_cache"]
    kv_cache_layout = src_cache["kv_cache_layout"]

    k_logical = inputs["k_t"]
    v_logical = inputs["v_t"]
    if inputs["varlen"]:
        kv_lens = list(inputs["vl_kv"])
        max_kv_len = max(kv_lens)
    else:
        kv_lens = [inputs["Skv"]] * inputs["B"]
        max_kv_len = inputs["Skv"]

    batch_size = inputs["B"]
    num_kv_heads = k_logical.shape[-2]
    head_dim = k_logical.shape[-1]
    dtype = k_logical.dtype
    device = k_logical.device
    max_context_length = max(PAGED_KV_MIN_CONTEXT_LENGTH, max_kv_len)
    max_num_pages_per_seq = _ceil_div(max_context_length, page_size)
    total_num_pages = max_num_pages_per_seq * batch_size

    k_cache_4d = torch.zeros(total_num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache_4d = torch.zeros_like(k_cache_4d)
    kv_num_used_pages = []
    kv_indices = []
    block_table_cpu = torch.zeros((batch_size, max_num_pages_per_seq), dtype=torch.int32, device="cpu")

    for b, kv_len in enumerate(kv_lens):
        num_pages = _ceil_div(kv_len, page_size)
        kv_num_used_pages.append(num_pages)
        page_ids = torch.arange(
            b * max_num_pages_per_seq,
            b * max_num_pages_per_seq + num_pages,
            dtype=torch.int32,
            device="cpu",
        )
        kv_indices.extend(page_ids.tolist())
        block_table_cpu[b, :num_pages] = page_ids
        if inputs["varlen"]:
            start, end = inputs["cukv"][b], inputs["cukv"][b + 1]
            kb = k_logical[start:end]
            vb = v_logical[start:end]
        else:
            kb = k_logical[b, :kv_len]
            vb = v_logical[b, :kv_len]
        padded_k = torch.zeros(num_pages * page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        padded_v = torch.zeros_like(padded_k)
        padded_k[:kv_len] = kb
        padded_v[:kv_len] = vb
        page_ids_gpu = page_ids.to(device=device, dtype=torch.long)
        k_cache_4d[page_ids_gpu] = padded_k.view(num_pages, page_size, num_kv_heads, head_dim)
        v_cache_4d[page_ids_gpu] = padded_v.view(num_pages, page_size, num_kv_heads, head_dim)

    if kv_cache_layout == "vectorized":
        k_vector_size = _kv_vector_size(dtype)
        k_cache, v_cache = _vectorize_paged_kv(
            k_cache_4d,
            v_cache_4d,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
        )
    else:
        k_cache, v_cache = k_cache_4d, v_cache_4d

    kv_num_used_pages_cpu = torch.tensor(kv_num_used_pages, dtype=torch.int32, device="cpu")
    kv_indptr_cpu = torch.cumsum(
        torch.cat((torch.tensor([0], dtype=torch.int32, device="cpu"), kv_num_used_pages_cpu)), dim=0
    )
    kv_indices_cpu = torch.nn.functional.pad(
        torch.tensor(kv_indices, dtype=torch.int32, device="cpu"), (0, 128), value=0
    )
    kv_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")
    kv_last_page_len_cpu = ((kv_lens_cpu - 1) % page_size + 1).int()
    return {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_lens_cpu": kv_lens_cpu,
        "kv_indptr_cpu": kv_indptr_cpu.int(),
        "kv_indices_cpu": kv_indices_cpu,
        "kv_last_page_len_cpu": kv_last_page_len_cpu,
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "seqlen_k": kv_lens_cpu.to(device),
        "page_size": page_size,
        "max_context_length": max_context_length,
        "kv_cache_layout": kv_cache_layout,
    }


def _build_attn_inputs_for_config(
    *,
    batch,
    seqlen_q,
    seqlen_kv,
    varlen_seqlens_q,
    varlen_seqlens_kv,
    num_heads,
    head_dim,
    num_kv_heads,
    dtype,
    use_block_table,
    page_size,
    kv_cache_layout,
    trigger_lazy_else,
    use_bias=False,
    use_alibi=False,
    alibi_two_d=False,
    use_sink=False,
    sink_share=DEFAULT_SINK_SHARE,
    causal=False,
):
    device = "cuda"
    H, D, H_KV = num_heads, head_dim, num_kv_heads
    varlen = varlen_seqlens_q is not None

    if varlen:
        vl_q = list(varlen_seqlens_q)
        vl_kv = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else vl_q
        B = len(vl_q)
        cuq = [0]
        [cuq.append(cuq[-1] + s) for s in vl_q]
        cukv = [0]
        [cukv.append(cukv[-1] + s) for s in vl_kv]
        total_q, total_kv = cuq[-1], cukv[-1]
        Sq = max(vl_q)
        cu_q_t = torch.tensor(cuq, dtype=torch.int32, device=device)
        cu_kv_t = torch.tensor(cukv, dtype=torch.int32, device=device)
        q_t = torch.empty(total_q, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        if use_block_table:
            kv_cache = _build_paged_kv_for_test(
                B,
                max(vl_kv),
                page_size,
                H_KV,
                D,
                vl_kv,
                dtype,
                device,
                kv_cache_layout,
            )
            k_t, v_t = _materialize_block_table_kv(kv_cache, dense=False)
        else:
            kv_cache = None
            k_t = torch.empty(total_kv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
            v_t = torch.empty(total_kv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        # Packed bias: row = global packed q token, column = per-batch-local key.
        # Drawn after Q/K/V so those stay bit-identical to a no-bias run.
        bias = (
            torch.empty(total_q, max(vl_kv), dtype=dtype, device=device).uniform_(*UNIFORM_RANGE) if use_bias else None
        )
        alibi_slopes = make_alibi_slopes(B, H, alibi_two_d, device) if use_alibi else None
        sink = None
        if use_sink:
            # One [H] table shared by every sequence, so calibrate over all their
            # rows together -- matching how the kernel consumes it.
            tot_lse = torch.zeros(H, dtype=torch.float32, device=device)
            tot_n = 0
            for b in range(B):
                sl, n = _rows_logsumexp(
                    q_t[cuq[b] : cuq[b + 1]].unsqueeze(0),
                    k_t[cukv[b] : cukv[b + 1]].unsqueeze(0),
                    causal,
                    bias=bias[cuq[b] : cuq[b + 1], : vl_kv[b]] if bias is not None else None,
                    alibi_slopes=(
                        (alibi_slopes[b] if alibi_slopes.dim() == 2 else alibi_slopes)
                        if alibi_slopes is not None
                        else None
                    ),
                )
                tot_lse += sl
                tot_n += n
            sink = calibrate_sink(tot_lse, tot_n, sink_share)
        return {
            "varlen": True,
            "sink": sink,
            "bias": bias,
            "alibi_slopes": alibi_slopes,
            "B": B,
            "Sq": Sq,
            "Skv": None,
            "vl_q": vl_q,
            "vl_kv": vl_kv,
            "cuq": cuq,
            "cukv": cukv,
            "total_q": total_q,
            "total_kv": total_kv,
            "cu_q_t": cu_q_t,
            "cu_kv_t": cu_kv_t,
            "q_t": q_t,
            "k_t": k_t,
            "v_t": v_t,
            "cross": any(vl_q[b] != vl_kv[b] for b in range(B)),
            "max_seqlen_kv": max(vl_kv),
            "kv_cache": kv_cache,
        }

    B, Sq = batch, seqlen_q
    Skv = seqlen_kv if seqlen_kv is not None else Sq
    q_t = torch.empty(B, Sq, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    if use_block_table:
        kv_cache = _build_paged_kv_for_test(
            B,
            Skv,
            page_size,
            H_KV,
            D,
            [Skv] * B,
            dtype,
            device,
            kv_cache_layout,
        )
        k_t, v_t = _materialize_block_table_kv(kv_cache, dense=True)
    else:
        kv_cache = None
        k_t = torch.empty(B, Skv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        v_t = torch.empty(B, Skv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)

    # Dense bias: (Sq, Skv), broadcast over batch and head. Drawn after Q/K/V so
    # those stay bit-identical to a no-bias run.
    bias = torch.empty(Sq, Skv, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE) if use_bias else None
    alibi_slopes = make_alibi_slopes(B, H, alibi_two_d, device) if use_alibi else None
    sink = (
        calibrate_sink(*_rows_logsumexp(q_t, k_t, causal, bias=bias, alibi_slopes=alibi_slopes), sink_share)
        if use_sink
        else None
    )

    if trigger_lazy_else:
        q_t.fill_(1.0)
        k_t.zero_()
        if Sq >= 128:
            k_t[:, 64:128, :, :].fill_(80.0)
        print(
            "[DUALWAVE_SWP_LAZY_ELSE_DEBUG] constructed Q=1, K tile0=0, " "K tile1=80 to force row_max - m_row > 8",
            flush=True,
        )

    return {
        "varlen": False,
        "sink": sink,
        "bias": bias,
        "alibi_slopes": alibi_slopes,
        "B": B,
        "Sq": Sq,
        "Skv": Skv,
        "vl_q": None,
        "vl_kv": None,
        "cuq": None,
        "cukv": None,
        "total_q": None,
        "total_kv": None,
        "cu_q_t": None,
        "cu_kv_t": None,
        "q_t": q_t,
        "k_t": k_t,
        "v_t": v_t,
        "cross": False,
        "max_seqlen_kv": None,
        "kv_cache": kv_cache,
    }


def _compute_reference_from_inputs(inputs, num_heads, head_dim, dtype, causal, seqlen_q, seqlen_kv):
    H, D = num_heads, head_dim
    q_t, k_t, v_t = inputs["q_t"], inputs["k_t"], inputs["v_t"]
    bias = inputs["bias"]
    alibi = inputs["alibi_slopes"]
    sink = inputs["sink"]

    if inputs["varlen"]:
        ref_t = torch.empty(inputs["total_q"], H, D, dtype=dtype, device=q_t.device)
        cuq, cukv = inputs["cuq"], inputs["cukv"]
        vl_q, vl_kv = inputs["vl_q"], inputs["vl_kv"]
        for b in range(inputs["B"]):
            qb = q_t[cuq[b] : cuq[b + 1]].unsqueeze(0).float()
            kb = k_t[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            vb = v_t[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            bias_b = bias[cuq[b] : cuq[b + 1], : vl_kv[b]] if bias is not None else None
            alibi_b = (alibi[b] if alibi.dim() == 2 else alibi) if alibi is not None else None
            ref_fn = pytorch_ref_attention if vl_q[b] == vl_kv[b] else pytorch_ref_attention_qkv_diff
            ref_t[cuq[b] : cuq[b + 1]] = (
                ref_fn(qb, kb, vb, causal=causal, bias=bias_b, alibi_slopes=alibi_b, sink=sink).to(dtype).squeeze(0)
            )
        return ref_t

    self_attn = seqlen_kv is None or seqlen_kv == seqlen_q
    ref_fn = pytorch_ref_attention if self_attn else pytorch_ref_attention_qkv_diff
    return ref_fn(q_t.float(), k_t.float(), v_t.float(), causal=causal, bias=bias, alibi_slopes=alibi, sink=sink).to(
        dtype
    )


def _build_inputs_and_reference_for_config(**kwargs):
    setup_seed(kwargs.pop("seed"))
    causal = kwargs.pop("causal")
    inputs = _build_attn_inputs_for_config(causal=causal, **kwargs)
    ref_t = _compute_reference_from_inputs(
        inputs,
        kwargs["num_heads"],
        kwargs["head_dim"],
        kwargs["dtype"],
        causal,
        kwargs["seqlen_q"],
        kwargs["seqlen_kv"],
    )
    return inputs, ref_t


def _precompute_paged_kv_inputs_and_ref(
    *,
    batch,
    seqlen_q,
    seqlen_kv,
    varlen_seqlens_q,
    varlen_seqlens_kv,
    num_heads,
    head_dim,
    num_kv_heads,
    dtype,
    causal,
    seed,
    page_size,
    kv_cache_layout,
    trigger_lazy_else=False,
    use_bias=False,
):
    invalid_layout = _validate_kv_cache_layout(kv_cache_layout, page_size, head_dim, dtype)
    if invalid_layout is not None:
        return None, None, {"skip": True, "skip_reason": invalid_layout}

    inputs, ref_t = _build_inputs_and_reference_for_config(
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        varlen_seqlens_q=varlen_seqlens_q,
        varlen_seqlens_kv=varlen_seqlens_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        seed=seed,
        use_block_table=True,
        page_size=page_size,
        kv_cache_layout=kv_cache_layout,
        trigger_lazy_else=trigger_lazy_else,
        use_bias=use_bias,
    )
    # ref_t folds in inputs["bias"]; a None here would compare the biased kernel
    # run against an unbiased reference and report a meaningless PASS.
    assert not use_bias or inputs["bias"] is not None, "use_bias=True produced no paged bias tensor"
    return inputs, ref_t, None


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 hash of a tensor's raw bytes."""
    return hashlib.md5(tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: list = None,
) -> dict:
    """Compare two numpy arrays and compute various difference metrics.

    Args:
        arr1: First input array (result), will be cast to float32.
        arr2: Second input array (reference), will be cast to float32.
        k: Number of top differences to report.
        thresholds: Difference magnitude buckets for histogram.

    Returns:
        Dictionary with top_k_diff, threshold_stats, nan_info, max_diff, max_diff_thr.
    """
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: arr1 {arr1.shape} vs arr2 {arr2.shape}")

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = int(np.sum(nan_mask1))
        print(f"  Warning: result contains {result['nan_info']['arr1_nan_count']} NaN values")
    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = int(np.sum(nan_mask2))
        print(f"  Warning: reference contains {result['nan_info']['arr2_nan_count']} NaN values")

    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = (diff / (1.0 + np.abs(arr2))).max()
    result["max_diff"] = float(diff.max())
    result["max_diff_thr"] = float(max_diff_thr)

    print(f"  diff.abs.max = {diff.max():.6f}")
    print(f"  diff.abs.mean = {diff.mean():.6f}")
    print(f"  max_diff_thr (rel) = {max_diff_thr:.6e}")

    flat_diff = diff.flatten()
    actual_k = min(k, len(flat_diff))
    top_k_indices = np.argpartition(flat_diff, -actual_k)[-actual_k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    print(f"  Top-{actual_k} differences:")
    for i in range(actual_k):
        idx = tuple(dim[i] for dim in orig_indices)
        entry = {
            "value": float(diff[idx]),
            "position": idx,
            "arr1_value": float(arr1[idx]),
            "arr2_value": float(arr2[idx]),
        }
        result["top_k_diff"].append(entry)
        print(f"    [{idx}] result={arr1[idx]:.6f}, ref={arr2[idx]:.6f}, diff={diff[idx]:.6f}")

    print(f"  Threshold distribution ({total_elements} elements):")
    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i], thresholds[i + 1]
        count = int(np.sum((diff >= lower) & (diff < upper)))
        pct = 100.0 * count / total_elements
        result["threshold_stats"].append({"range": f"[{lower:.0e}, {upper:.0e})", "count": count, "percentage": pct})
        print(f"    [{lower:.0e}, {upper:.0e}): {count:>8d} ({pct:6.2f}%)")

    count = int(np.sum(diff >= thresholds[-1]))
    pct = 100.0 * count / total_elements
    result["threshold_stats"].append({"range": f">={thresholds[-1]:.0e}", "count": count, "percentage": pct})
    print(f"    >={thresholds[-1]:.0e}       : {count:>8d} ({pct:6.2f}%)")

    return result


@pytest.mark.parametrize("query_length", range(1, 17))
@pytest.mark.parametrize("sliding_window", [0, 128])
def test_vectorized_paged_page16_short_query(query_length, sliding_window):
    batch_size = 2
    kv_length = 257
    num_heads = 8
    num_kv_heads = 1
    head_dim = 128
    inputs, _, skip = _precompute_paged_kv_inputs_and_ref(
        batch=batch_size,
        seqlen_q=query_length,
        seqlen_kv=kv_length,
        varlen_seqlens_q=None,
        varlen_seqlens_kv=None,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        dtype=torch.bfloat16,
        causal=True,
        seed=DEFAULT_SEED,
        page_size=16,
        kv_cache_layout="vectorized",
    )
    assert skip is None

    q = inputs["q_t"]
    k = inputs["k_t"].expand(-1, -1, num_heads, -1)
    v = inputs["v_t"].expand(-1, -1, num_heads, -1)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(head_dim)
    query_positions = torch.arange(query_length, device=q.device) + kv_length - query_length
    key_positions = torch.arange(kv_length, device=q.device)
    mask = key_positions[None, :] <= query_positions[:, None]
    if sliding_window:
        mask &= key_positions[None, :] >= query_positions[:, None] - sliding_window + 1
    scores.masked_fill_(~mask[None, None, :, :], float("-inf"))
    expected = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v.float()).to(q.dtype)

    kv_cache = inputs["kv_cache"]
    output = torch.empty_like(q)
    flydsl_flash_attn_func(
        q,
        kv_cache["k_cache"],
        kv_cache["v_cache"],
        causal=True,
        num_kv_heads=num_kv_heads,
        block_table=kv_cache["block_table"],
        seqlen_k=kv_cache["seqlen_k"],
        max_seqlen_kv=kv_length,
        kv_cache_layout="vectorized",
        sliding_window=sliding_window,
        out=output,
    )
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


def _cfg_kw():
    """Return flydsl_flash_attn_func kwargs from the global kernel config."""
    return dict(
        waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
        daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
        dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
        dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
        dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
    )


def _flops(Sq, Skv, H, D, B, causal):
    """Compute FLOPs for one config (bottom-right causal or non-causal)."""
    delta = Skv - Sq
    if causal:
        valid = sum(min(max(r + delta + 1, 0), Skv) for r in range(Sq))
    else:
        valid = Sq * Skv
    return 4.0 * valid * D * H * B


def _acc_metric(o_f32, ref_f32, D, compare_mode=False):
    """Return (max_err, min_cos, passed) with zero-row-safe cosine.

    compare_mode: skip cosine (expensive for large configs); min_cos returned
    as None and passed is based on max_err only.
    """
    max_err = (o_f32 - ref_f32).abs().max().item()
    if compare_mode:
        return max_err, None, bool(max_err < 1e-2)
    res_rows = o_f32.reshape(-1, D)
    ref_rows = ref_f32.reshape(-1, D)
    nz = ref_rows.norm(dim=1) > 1e-6
    if bool(nz.all()):
        # Avoid boolean-mask copies for large self-attn tensors.
        min_cos = F.cosine_similarity(res_rows, ref_rows, dim=1).min().item()
        zero_ok = True
    else:
        min_cos = F.cosine_similarity(res_rows[nz], ref_rows[nz], dim=1).min().item() if bool(nz.any()) else 1.0
        zero_ok = res_rows[~nz].abs().max().item() < 1e-2 if bool((~nz).any()) else True
    passed = bool(max_err < 1e-2 and min_cos > 0.99 and zero_ok)
    return max_err, min_cos, passed


def run_attn_config(
    num_heads,
    head_dim,
    dtype,
    causal,
    warmup,
    iters,
    *,
    batch=1,
    seqlen_q=None,
    seqlen_kv=None,
    varlen_seqlens_q=None,
    varlen_seqlens_kv=None,
    num_kv_heads=None,
    num_kv_splits=1,
    seed=DEFAULT_SEED,
    dtype_str="bf16",
    verbose=False,
    trigger_lazy_else=False,
    compare_mode=False,
    precomputed_ref=None,
    precomputed_inputs=None,
    use_block_table=False,
    page_size=64,
    kv_cache_layout="linear",
    use_bias=False,
    use_alibi=False,
    alibi_two_d=False,
    use_sink=False,
    sink_share=DEFAULT_SINK_SHARE,
):
    """Unified flash-attention test/bench function.

    Modes (mutually exclusive):
    - dense self-attn:       seqlen_q set, varlen_seqlens_q is None, seqlen_kv is None.
    - dense cross-attn:      seqlen_q set, seqlen_kv set (may differ), varlen_seqlens_q is None.
    - varlen self-attn:      varlen_seqlens_q set, varlen_seqlens_kv is None.
    - varlen cross-attn:     varlen_seqlens_q and varlen_seqlens_kv both set.
    - split-K:               seqlen_q set, num_kv_splits > 1 (dense only, gfx950).
    - paged-KV reference: if use_block_table=True, K/V are first created in
      a physical paged cache layout plus the selected lookup table, then
      materialized into the current dense/packed K/V ABI used by the kernel and reference.

    compare_mode: when True, skip cosine computation (expensive for large B*S*H) and
    use pytorch_ref_attention (fast path) for dense self-attn instead of the
    general cross-attn reference.

    Returns a result dict with keys: max_err, [min_cos], passed, [us, tflops], [all_below_true/false_count].
    On skippable shapes (split-K constraint violated): returns {'skip': True}.
    On build/exec error: returns {'err': <str>}.
    """
    results = {}
    device = "cuda"
    varlen = varlen_seqlens_q is not None
    splitk = num_kv_splits > 1

    if use_block_table and page_size < 1:
        return {"err": f"invalid page_size={page_size}"}

    if num_kv_heads is None:
        num_kv_heads = num_heads
    H, D, H_KV = num_heads, head_dim, num_kv_heads
    debug_lazy = FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_debug_lazy_counts"]

    if use_block_table:
        invalid_layout = _validate_kv_cache_layout(kv_cache_layout, page_size, D, dtype)
        if invalid_layout is not None:
            return {"skip": True, "skip_reason": invalid_layout}

    # ── split-K early-exit guard (mirrors run_splitk_config logic) ───────────
    if splitk:
        if D not in (64, 128) or dtype_str not in ("bf16", "f16") or (seqlen_q is not None and seqlen_q < 384):
            return {"skip": True}

    # ── bias addressing guard ────────────────────────────────────────────────
    if use_bias:
        if varlen:
            vl_q_cfg = list(varlen_seqlens_q)
            vl_kv_cfg = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else vl_q_cfg
            bias_rows, bias_cols = sum(vl_q_cfg), max(vl_kv_cfg)
            if max(vl_q_cfg) > vl_q_cfg[-1]:
                return {
                    "skip": True,
                    "skip_reason": (
                        f"varlen bias reads OOB when the last seqlen ({vl_q_cfg[-1]}) "
                        f"is below max_seqlen_q ({max(vl_q_cfg)})"
                    ),
                }
        else:
            bias_rows = seqlen_q
            bias_cols = seqlen_kv if seqlen_kv is not None else seqlen_q
        fits, why = bias_fits(bias_rows, bias_cols, torch.empty((), dtype=dtype).element_size())
        if not fits:
            return {"skip": True, "skip_reason": why}

    if use_block_table and (precomputed_inputs is None or precomputed_ref is None):
        return {"err": "block-table tests require precomputed_inputs and precomputed_ref"}

    if precomputed_inputs is None:
        setup_seed(seed)
        precomputed_inputs = _build_attn_inputs_for_config(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            varlen_seqlens_q=varlen_seqlens_q,
            varlen_seqlens_kv=varlen_seqlens_kv,
            num_heads=H,
            head_dim=D,
            num_kv_heads=H_KV,
            dtype=dtype,
            use_block_table=use_block_table,
            page_size=page_size,
            kv_cache_layout=kv_cache_layout,
            trigger_lazy_else=trigger_lazy_else,
            use_bias=use_bias,
            use_alibi=use_alibi,
            alibi_two_d=alibi_two_d,
            use_sink=use_sink,
            sink_share=sink_share,
            causal=causal,
        )

    varlen = precomputed_inputs["varlen"]
    B = precomputed_inputs["B"]
    Sq = precomputed_inputs["Sq"]
    Skv = precomputed_inputs["Skv"]
    vl_q = precomputed_inputs["vl_q"]
    vl_kv = precomputed_inputs["vl_kv"]
    cu_q_t = precomputed_inputs["cu_q_t"]
    cu_kv_t = precomputed_inputs["cu_kv_t"]
    q_t = precomputed_inputs["q_t"]
    k_t = precomputed_inputs["k_t"]
    v_t = precomputed_inputs["v_t"]
    cross = precomputed_inputs["cross"]
    max_seqlen_kv = precomputed_inputs["max_seqlen_kv"]
    kv_cache = precomputed_inputs["kv_cache"]
    bias_t = precomputed_inputs["bias"]
    alibi_t = precomputed_inputs["alibi_slopes"]
    sink_t = precomputed_inputs["sink"]
    # ref_t is built from these same inputs, so a missing bias makes both sides
    # unbiased and the comparison vacuous. Fail loudly instead.
    if use_bias and bias_t is None:
        return {"err": "use_bias=True but the precomputed inputs carry no bias"}

    debug_counts = torch.zeros(2, dtype=torch.float32, device=device) if debug_lazy else None
    o_t = torch.zeros_like(q_t)

    # ── kernel launch ────────────────────────────────────────────────────────
    try:
        if use_block_table and kv_cache is not None:
            # Native paged-KV uses physical K/V cache and block_table in the kernel.
            # Varlen Q passes cu_seqlens; dense Q passes none.
            _paged_varlen_kw = (
                dict(cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t, max_seqlen_q=Sq, cross_seqlen=cross) if varlen else {}
            )
            flydsl_flash_attn_func(
                q_t,
                kv_cache["k_cache"],
                kv_cache["v_cache"],
                causal=causal,
                num_kv_heads=H_KV,
                max_seqlen_kv=max_seqlen_kv if varlen else Skv,
                block_table=kv_cache["block_table"],
                seqlen_k=kv_cache["seqlen_k"],
                kv_cache_layout=kv_cache_layout,
                num_kv_splits=int(num_kv_splits),
                out=o_t,
                bias=bias_t,
                **_paged_varlen_kw,
                **_cfg_kw(),
            )
        else:
            flydsl_flash_attn_func(
                q_t,
                k_t,
                v_t,
                causal=causal,
                num_kv_heads=H_KV,
                cu_seqlens_q=cu_q_t,
                cu_seqlens_kv=cu_kv_t,
                max_seqlen_q=Sq if varlen else None,
                max_seqlen_kv=max_seqlen_kv if varlen else None,
                cross_seqlen=cross if varlen else None,
                num_kv_splits=int(num_kv_splits),
                bias=bias_t,
                alibi_slopes=alibi_t,
                sink=sink_t,
                out=o_t,
                debug_counts=debug_counts,
                **_cfg_kw(),
            )
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback

        traceback.print_exc()
        return results

    if debug_lazy and debug_counts is not None:
        counts = debug_counts.detach().cpu().tolist()
        results["all_below_true_count"] = int(counts[0])
        results["all_below_false_count"] = int(counts[1])
        print(
            f"[DUALWAVE_SWP_LAZY_COUNTS] all_below_true={int(counts[0])}, " f"all_below_false={int(counts[1])}",
            flush=True,
        )

    # ── reference ───────────────────────────────────────────────────────────
    # precomputed_ref makes FlyDSL/aiter_ck/aiter_asm share one reference tensor.
    # Otherwise compute the cheapest reference path for the active mode.
    # Delegated rather than inlined so the bias enters the reference in exactly one
    # place; two copies of this dispatch would be free to drift apart.
    if precomputed_ref is not None:
        ref_t = precomputed_ref
    else:
        ref_t = _compute_reference_from_inputs(precomputed_inputs, H, D, dtype, causal, seqlen_q, seqlen_kv)

    o_f32 = o_t.contiguous().reshape(-1).float()
    ref_f32 = ref_t.contiguous().reshape(-1).float()
    max_err, min_cos, passed = _acc_metric(o_f32, ref_f32, D, compare_mode=compare_mode)
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    if min_cos is not None:
        results["min_cos"] = min_cos
    results["passed"] = passed
    if use_block_table and kv_cache is not None:
        results["block_table_shape"] = tuple(kv_cache["block_table"].shape)
        results["kv_cache_layout"] = kv_cache_layout
        results["k_cache_shape"] = tuple(kv_cache["k_cache"].shape)
        results["v_cache_shape"] = tuple(kv_cache["v_cache"].shape)

    if verbose:
        o_flat = o_t.reshape(-1)
        ref_flat = ref_t.reshape(-1)
        _mask = "causal" if causal else "noncausal"
        _hkv = num_kv_heads if num_kv_heads is not None else H
        if varlen:
            tag = f"varlen Sq={list(vl_q)} Skv={list(vl_kv)} H={H} Hkv={_hkv} D={D} {_mask} splits={num_kv_splits}"
        else:
            tag = f"B={B} Sq={Sq} Skv={Skv} H={H} Hkv={_hkv} D={D} {_mask} splits={num_kv_splits}"
        if use_block_table:
            tag += f" BT page{page_size} {kv_cache_layout}"
        rm = compute_md5(o_flat)
        rm2 = compute_md5(ref_flat)
        print(f"  [{tag}] result_md5 = {rm}")
        print(f"  [{tag}] ref_md5    = {rm2}")
        if rm == rm2:
            print(f"  [{tag}] MD5 match: EXACT (bit-identical)")
        else:
            print(f"  [{tag}] MD5 match: DIFFER (not bit-identical)")
        print(f"  [{tag}] --- compare_arrays ---")
        compare_arrays(
            o_flat.to(torch.float32).detach().cpu().numpy(),
            ref_flat.to(torch.float32).detach().cpu().numpy(),
        )

    # ── benchmark ────────────────────────────────────────────────────────────
    try:
        if varlen:
            flops = sum(_flops(vl_q[b], vl_kv[b], H, D, 1, causal) for b in range(B))
        else:
            flops = _flops(Sq, Skv, H, D, B, causal)

        def kernel_fn():
            if use_block_table and kv_cache is not None:
                _paged_varlen_kw = (
                    dict(cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t, max_seqlen_q=Sq, cross_seqlen=cross)
                    if varlen
                    else {}
                )
                flydsl_flash_attn_func(
                    q_t,
                    kv_cache["k_cache"],
                    kv_cache["v_cache"],
                    causal=causal,
                    num_kv_heads=H_KV,
                    max_seqlen_kv=max_seqlen_kv if varlen else Skv,
                    block_table=kv_cache["block_table"],
                    seqlen_k=kv_cache["seqlen_k"],
                    kv_cache_layout=kv_cache_layout,
                    num_kv_splits=int(num_kv_splits),
                    out=o_t,
                    bias=bias_t,
                    **_paged_varlen_kw,
                    **_cfg_kw(),
                )
            else:
                flydsl_flash_attn_func(
                    q_t,
                    k_t,
                    v_t,
                    causal=causal,
                    num_kv_heads=H_KV,
                    cu_seqlens_q=cu_q_t,
                    cu_seqlens_kv=cu_kv_t,
                    max_seqlen_q=Sq if varlen else None,
                    max_seqlen_kv=max_seqlen_kv if varlen else None,
                    cross_seqlen=cross if varlen else None,
                    num_kv_splits=int(num_kv_splits),
                    bias=bias_t,
                    alibi_slopes=alibi_t,
                    sink=sink_t,
                    out=o_t,
                    debug_counts=debug_counts,
                    **_cfg_kw(),
                )

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                kernel_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(kernel_fn, num_iters=iters, num_warmup=warmup)
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_aiter_bench(
    batch,
    seq_len,
    nheads,
    head_dim,
    dtype,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    backend="ck",
    num_kv_heads=None,
    precomputed_ref=None,
    precomputed_inputs=None,
    seqlen_kv=None,
    varlen_seqlens_q=None,
    varlen_seqlens_kv=None,
    use_bias=False,
    use_alibi=False,
    use_sink=False,
):
    """Run true aiter_ck or true aiter_asm kernel via aiter and return {tflops, max_err, us}."""
    try:
        import aiter
    except Exception:
        return {"err": "aiter not installed"}

    varlen = varlen_seqlens_q is not None
    if backend == "asm" and dtype != torch.bfloat16:
        return {"skip": True}
    if backend == "asm" and head_dim != 128:
        return {"skip": True}
    if backend == "asm" and (varlen or (seqlen_kv is not None and seqlen_kv != seq_len)):
        return {"skip": True}
    bias = precomputed_inputs["bias"] if precomputed_inputs is not None else None
    if use_bias and (backend == "asm" or varlen or bias is None):
        return {"skip": True}
    alibi = precomputed_inputs["alibi_slopes"] if precomputed_inputs is not None else None
    if use_alibi and (backend == "asm" or causal or use_bias or alibi is None):
        return {"skip": True}
    sink = precomputed_inputs["sink"] if precomputed_inputs is not None else None
    if use_sink and (backend == "asm" or varlen or sink is None):
        return {"skip": True}

    results = {}
    torch.cuda.empty_cache()

    H, D = nheads, head_dim
    H_KV = num_kv_heads if num_kv_heads is not None else H
    if precomputed_inputs is not None:
        varlen = precomputed_inputs["varlen"]
        B = precomputed_inputs["B"]
        S = precomputed_inputs["Sq"]
        Skv = precomputed_inputs["Skv"]
        cu_q_t = precomputed_inputs["cu_q_t"]
        cu_kv_t = precomputed_inputs["cu_kv_t"]
        q = precomputed_inputs["q_t"]
        k = precomputed_inputs["k_t"]
        v = precomputed_inputs["v_t"]
        if varlen:
            vl_q = precomputed_inputs["vl_q"]
            vl_kv = precomputed_inputs["vl_kv"]
            cuq = precomputed_inputs["cuq"]
            cukv = precomputed_inputs["cukv"]
            total_q = precomputed_inputs["total_q"]
            total_kv = precomputed_inputs["total_kv"]
            q_pack, k_pack, v_pack = q, k, v
            S = max(vl_q)
            Skv = max(vl_kv)
            q = torch.zeros(B, S, H, D, dtype=dtype, device="cuda")
            k = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
            v = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
            for b in range(B):
                q[b, : vl_q[b]] = q_pack[cuq[b] : cuq[b + 1]]
                k[b, : vl_kv[b]] = k_pack[cukv[b] : cukv[b + 1]]
                v[b, : vl_kv[b]] = v_pack[cukv[b] : cukv[b + 1]]
    else:
        setup_seed(seed)
    if precomputed_inputs is None and varlen:
        vl_q = list(varlen_seqlens_q)
        vl_kv = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else vl_q
        B = len(vl_q)
        S = max(vl_q)
        Skv = max(vl_kv)
        cuq = [0]
        [cuq.append(cuq[-1] + s) for s in vl_q]
        cukv = [0]
        [cukv.append(cukv[-1] + s) for s in vl_kv]
        total_q, total_kv = cuq[-1], cukv[-1]
        cu_q_t = torch.tensor(cuq, dtype=torch.int32, device="cuda")
        cu_kv_t = torch.tensor(cukv, dtype=torch.int32, device="cuda")
        q_pack = torch.empty(total_q, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        k_pack = torch.empty(total_kv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        v_pack = torch.empty(total_kv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        q = torch.zeros(B, S, H, D, dtype=dtype, device="cuda")
        k = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
        v = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
        for b in range(B):
            q[b, : vl_q[b]] = q_pack[cuq[b] : cuq[b + 1]]
            k[b, : vl_kv[b]] = k_pack[cukv[b] : cukv[b + 1]]
            v[b, : vl_kv[b]] = v_pack[cukv[b] : cukv[b + 1]]
    elif precomputed_inputs is None:
        B, S, Skv = batch, seq_len, seqlen_kv if seqlen_kv is not None else seq_len
        cu_q_t = cu_kv_t = None
        q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        k = torch.empty(B, Skv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        v = torch.empty(B, Skv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "ck":

        def aiter_forward():
            return aiter.mha_fwd(
                q,  # q
                k,  # k
                v,  # v
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                1 if use_sink else 0,  # sink_size
                True,  # return_softmax_lse
                False,  # return_dropout_randval
                sink_ptr=sink if use_sink else None,
                cu_seqlens_q=cu_q_t,
                cu_seqlens_kv=cu_kv_t,
                out=None,
                bias=bias,
                alibi_slopes=alibi,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                gen=None,
            )

    elif backend == "asm":

        def aiter_forward():
            return aiter.fmha_v3_fwd(
                q,  # q
                k,  # k
                v,  # v
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                True,  # return_softmax_lse
                False,  # return_dropout_randval
                2,  # how_v3_bf16_cvt
                out=None,
                bias=None,
                alibi_slopes=None,
                gen=None,
            )

    else:
        return {"err": f"unsupported backend: {backend}"}

    try:
        out = aiter_forward()[0]
        torch.cuda.synchronize()
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    if precomputed_ref is not None:
        ref = precomputed_ref
    elif varlen:
        ref = torch.empty(total_q, H, D, dtype=dtype, device="cuda")
        for b in range(B):
            qb = q_pack[cuq[b] : cuq[b + 1]].unsqueeze(0).float()
            kb = k_pack[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            vb = v_pack[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            ref_fn = pytorch_ref_attention if vl_q[b] == vl_kv[b] else pytorch_ref_attention_qkv_diff
            ref[cuq[b] : cuq[b + 1]] = ref_fn(qb, kb, vb, causal=causal).to(dtype).squeeze(0)
    else:
        ref_fn = (
            pytorch_ref_attention if (seqlen_kv is None or seqlen_kv == seq_len) else pytorch_ref_attention_qkv_diff
        )
        ref = ref_fn(q.float(), k.float(), v.float(), causal=causal).to(dtype)
    if varlen:
        out_cmp = torch.empty(total_q, H, D, dtype=out.dtype, device="cuda")
        for b in range(B):
            out_cmp[cuq[b] : cuq[b + 1]] = out[b, : vl_q[b]]
    else:
        out_cmp = out
    max_err = (out_cmp.float() - ref.float()).abs().max().item()
    results["max_err"] = max_err

    try:

        def bench_fn():
            aiter_forward()

        # Warm up torch.profiler so run_perftest avoids first-session overhead.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        if varlen:
            flops = sum(_flops(vl_q[b], vl_kv[b], H, D, 1, causal) for b in range(B))
        else:
            flops = _flops(S, Skv, H, D, B, causal)
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


# ── fp8 (e4m3fn) support ─────────────────────────────────────────────────────
# Q/K/V are pre-quantized e4m3fn with per-tensor descales; output is bf16.
# The reference dequantizes the same fp8 inputs before SDPA.


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def quantize_per_tensor_fp8(x):
    """Per-tensor quantize a float tensor to e4m3fn + a shape-[1] fp32 descale.

    Mirrors aiter.ops.quant.per_tensor_quant: descale = amax / fp8_max, the
    stored fp8 value is round(x / descale), and dequant is fp8_value * descale.
    Uses aiter's helper when available (so the harness and the aiter comparator
    share identical quantization), with a numerically identical torch fallback.
    """
    try:
        from aiter import dtypes as _adtypes
        from aiter import per_tensor_quant as _ptq

        x_fp8, descale = _ptq(x, quant_dtype=_adtypes.fp8)
        # Enforce e4m3fn (not fnuz) and the expected per-tensor descale shape.
        if x_fp8.dtype != FP8_DTYPE:
            raise ValueError(f"aiter per_tensor_quant produced {x_fp8.dtype}, expected {FP8_DTYPE}")
        return x_fp8.contiguous(), descale.to(torch.float32).view(1).contiguous()
    except ImportError:
        fp8_max = torch.finfo(FP8_DTYPE).max
        amax = x.abs().max().to(torch.float32)
        descale = (amax / fp8_max).clamp(min=1e-12).view(1)
        x_fp8 = (x.to(torch.float32) / descale).to(FP8_DTYPE)
        return x_fp8.contiguous(), descale.to(torch.float32).contiguous()


def _dequant_fp8(x_fp8, descale):
    """Dequantize e4m3fn back to float32: fp8_value * descale."""
    return x_fp8.to(torch.float32) * descale.to(torch.float32)


def run_fp8_config(
    batch,
    seq_len,
    num_heads,
    head_dim,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    verbose=True,
    num_kv_heads=None,
    num_kv_splits=1,
):
    """Run the FlyDSL fp8 (e4m3fn) forward path and validate vs a dequantized-input
    SDPA reference at the fixed fp8 gate (max_err < 5e-2 and min_cos > 0.98).

    Unsupported fp8 configurations (non-gfx950, head_dim != 128, split-K) raise a
    clear error that is surfaced as an ERROR row (never a silent SKIP). Returns a
    run_config-compatible dict so it prints through the same summary table.
    """
    device = "cuda"
    results = {}

    if num_kv_heads is None:
        num_kv_heads = num_heads

    # fp8 split-K is not implemented. Reject it explicitly rather than silently
    # running a dense fp8 forward while the config row advertises kv_sp>1 (which
    # would validate the wrong path).
    if int(num_kv_splits) > 1:
        results["err"] = f"fp8 split-K (num_kv_splits={num_kv_splits}) is not implemented (dense fp8 only)"
        return results

    # fp8 forward is gfx950-only and head_dim==128 only. Reject anything else
    # up-front with a clear, specific error (surfaced as an ERROR row) rather
    # than a SKIP that would mask a real failure.
    try:
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        gpu_arch = ""
    if not gpu_arch.startswith("gfx950"):
        results["err"] = f"fp8 requires gfx950 (got '{gpu_arch or 'unknown'}')"
        return results
    if head_dim != 128:
        results["err"] = f"fp8 requires head_dim == 128 (got {head_dim})"
        return results
    if num_heads % num_kv_heads != 0:
        results["err"] = f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        return results
    if seq_len < 1:
        results["err"] = f"seq_len ({seq_len}) must be >= 1"
        return results

    B, S, H, D = batch, seq_len, num_heads, head_dim
    H_KV = num_kv_heads
    setup_seed(seed)

    # Host bf16 master tensors -> per-tensor e4m3fn + shape-[1] fp32 descales.
    q_bf16 = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    k_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    v_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    q_fp8, q_descale = quantize_per_tensor_fp8(q_bf16)
    k_fp8, k_descale = quantize_per_tensor_fp8(k_bf16)
    v_fp8, v_descale = quantize_per_tensor_fp8(v_bf16)

    o_bf16 = torch.zeros(B, S, H, D, dtype=torch.bfloat16, device=device)
    fp8_exec_kwargs = dict(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)

    try:
        flydsl_flash_attn_func(
            q_fp8,
            k_fp8,
            v_fp8,
            causal=causal,
            num_kv_heads=num_kv_heads,
            out=o_bf16,
            waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
            daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
            dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
            dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
            dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
            **fp8_exec_kwargs,
        )
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        return results

    o_flat = o_bf16.contiguous().view(-1)

    # Reference: dequantize the SAME e4m3fn Q/K/V (applying descales) and run the
    # high-precision SDPA reference the bf16 path uses.
    ref_4d = pytorch_ref_attention(
        _dequant_fp8(q_fp8, q_descale),
        _dequant_fp8(k_fp8, k_descale),
        _dequant_fp8(v_fp8, v_descale),
        causal=causal,
    )
    ref_flat = ref_4d.to(torch.float32).contiguous().view(-1)

    o_f32 = o_flat.float()
    ref_f32 = ref_flat.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1)
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS

    if verbose:
        tag = f"B={B} S={S} H={H} D={D} fp8"
        print(f"  [{tag}] --- compare_arrays ---")
        compare_arrays(
            o_f32.detach().cpu().numpy(),
            ref_f32.detach().cpu().numpy(),
        )

    try:

        def kernel_fn():
            # Time the same public-ABI call as correctness; descales must stay kwargs
            # so they do not bind to stride/head_dim positional slots.
            flydsl_flash_attn_func(
                q_fp8,
                k_fp8,
                v_fp8,
                causal=causal,
                num_kv_heads=num_kv_heads,
                out=o_bf16,
                waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
                daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
                dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
                dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
                dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
                **fp8_exec_kwargs,
            )

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                kernel_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(kernel_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        # A failed timing path must not be reportable as a clean PASS-with-N/A row.
        # Keep the correctness numbers visible but mark the row not-passed so the
        # summary surfaces the failure (see the status logic in main()).
        results["bench_err"] = str(e)
        results["passed"] = False

    return results


def aiter_asm_fp8_dispatch_ok(batch, seq_len, num_heads, num_kv_heads, head_dim):
    """Predicate: does this dense fp8 shape reach aiter's NATIVE gfx950 fp8 ASM
    kernel (fwd_hd128_fp8*.co, dtype fp8bf16, bf16_cvt=0)?

    Mirrors the aiter dispatch gate for native fp8 ASM: head_dim == 128, GQA
    ratio (num_heads / num_kv_heads) a power of two, and seqlen_q > 128. When
    this is False the aiter dispatcher falls back to CK, which must be labeled
    honestly per-shape (never reported as 'aiter asm fp8').
    """
    if head_dim != 128 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return False
    if not _is_pow2(num_heads // num_kv_heads):
        return False
    return seq_len > 128


def run_aiter_fp8_bench(
    batch,
    seq_len,
    nheads,
    head_dim,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    backend="asm",
    num_kv_heads=None,
):
    """Run aiter's fp8 forward and return {tflops, max_err, us, label}.

    backend="asm": drive the NATIVE gfx950 fp8 ASM kernel via
      aiter.ops.mha.fmha_v3_fwd(..., how_v3_bf16_cvt=0) directly. This is the
      genuine native-fp8 path (#2911), NOT the bf16-convert path (bf16_cvt!=0)
      and NOT a CK/triton fallback. Shapes that do not meet the native-asm gate
      are SKIPPED (so the asm column never silently substitutes CK for asm).
    backend="ck": secondary comparison via aiter.mha_fwd with descales.
    """
    try:
        import aiter
        from aiter.ops.mha import fmha_v3_fwd
    except Exception:
        return {"err": "aiter not installed"}

    if num_kv_heads is None:
        num_kv_heads = nheads
    asm_ok = aiter_asm_fp8_dispatch_ok(batch, seq_len, nheads, num_kv_heads, head_dim)
    if backend == "asm" and not asm_ok:
        # Native fp8 ASM kernel is not selected for this shape -> SKIP rather
        # than fall back to CK and mislabel it as asm.
        return {"skip": True}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    H_KV = num_kv_heads
    q_bf16 = torch.empty(B, S, H, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    k_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    v_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    q_fp8, q_descale = quantize_per_tensor_fp8(q_bf16)
    k_fp8, k_descale = quantize_per_tensor_fp8(k_bf16)
    v_fp8, v_descale = quantize_per_tensor_fp8(v_bf16)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "asm":
        results["label"] = "aiter_asm_fp8"

        def aiter_forward():
            out = torch.empty((B, S, H, D), device="cuda", dtype=torch.bfloat16)
            return fmha_v3_fwd(
                q_fp8,
                k_fp8,
                v_fp8,
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale (descales applied separately)
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                False,  # return_softmax_lse
                False,  # return_dropout_randval
                0,  # how_v3_bf16_cvt = 0 -> native fp8 (NOT bf16-convert)
                out,
                None,  # bias
                None,  # alibi_slopes
                q_descale,
                k_descale,
                v_descale,
                None,  # gen
            )

    elif backend == "ck":
        # CK fp8 is the labeled secondary comparison. Label honestly so a shape
        # that DID meet the asm gate is never silently reported as the headline.
        results["label"] = "aiter_ck_fp8" if asm_ok else "aiter_ck_fp8(fallback)"

        def aiter_forward():
            # CK fp8 needs an explicit bf16 output tensor; without it the op
            # infers an fp8 output and rejects ("invalid argument for fmha_fwd").
            out = torch.empty((B, S, H, D), device="cuda", dtype=torch.bfloat16)
            return aiter.mha_fwd(
                q_fp8,
                k_fp8,
                v_fp8,
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                0,  # sink_size
                False,  # return_softmax_lse
                False,  # return_dropout_randval
                cu_seqlens_q=None,
                cu_seqlens_kv=None,
                out=out,
                bias=None,
                alibi_slopes=None,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                gen=None,
            )

    else:
        return {"err": f"unsupported backend: {backend}"}

    try:
        res = aiter_forward()
        out = res[0] if isinstance(res, (tuple, list)) else res
        torch.cuda.synchronize()
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    # Compare against the dequantized-input SDPA reference (same e4m3fn inputs).
    # Compute the FULL fixed fp8 gate (max_err AND min_cos) so a claim that an
    # aiter fp8 row is within the gate is provable, not half-checked.
    ref = pytorch_ref_attention(
        _dequant_fp8(q_fp8, q_descale),
        _dequant_fp8(k_fp8, k_descale),
        _dequant_fp8(v_fp8, v_descale),
        causal=causal,
    )
    out_f32 = out.float()
    ref_f32 = ref.float()
    max_err = (out_f32 - ref_f32).abs().max().item()
    min_cos = F.cosine_similarity(out_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1).min().item()
    results["max_err"] = max_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS

    try:

        def bench_fn():
            aiter_forward()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_aiter_batch_prefill_bench(inputs, precomputed_ref, num_heads, head_dim, dtype, causal, warmup, iters):
    """Run aiter.mha_batch_prefill_func using the physical paged KV cache in inputs."""
    try:
        import aiter
    except Exception:
        return {"err": "aiter not installed"}

    results = {}
    q_t = inputs["q_t"]
    kv_cache = inputs["kv_cache"]
    if kv_cache is None:
        return {"err": "run_aiter_batch_prefill_bench requires inputs['kv_cache']"}
    if kv_cache["page_size"] >= 16:
        kv_cache = _build_paged_kv_from_logical_for_aiter(inputs, page_size=16)

    H, D = num_heads, head_dim
    B, Sq = inputs["B"], inputs["Sq"]
    q_indptr = inputs["cu_q_t"]
    if q_indptr is None:
        q_for_kernel = q_t.reshape(-1, H, D).contiguous()
        q_indptr = torch.arange(B + 1, dtype=torch.int32, device=q_t.device) * Sq
    else:
        q_for_kernel = q_t

    kv_indptr = kv_cache["kv_indptr_cpu"].to(q_t.device)
    kv_indices = kv_cache["kv_indices_cpu"].to(q_t.device)
    kv_last_page_lens = kv_cache["kv_last_page_len_cpu"].to(q_t.device)
    block_table = kv_cache["block_table"]
    seqlen_k = kv_cache["seqlen_k"]
    max_seqlen_k = inputs["max_seqlen_kv"] if inputs["varlen"] else inputs["Skv"]

    def aiter_forward():
        return aiter.mha_batch_prefill_func(
            q_for_kernel,
            kv_cache["k_cache"],
            kv_cache["v_cache"],
            q_indptr,
            kv_indptr,
            kv_indices,
            Sq,
            max_seqlen_k,
            causal=causal,
            kv_last_page_lens=kv_last_page_lens,
            block_table=block_table,
            seqlen_k=seqlen_k,
        )

    try:
        out = aiter_forward()
        torch.cuda.synchronize()
    except Exception as e:
        if "no matching kernel found" in str(e):
            return {"skip": True, "skip_reason": str(e)}
        import traceback

        traceback.print_exc()
        return {"err": f"aiter_batch_prefill: {e}"}

    ref = precomputed_ref.reshape(-1, H, D) if not inputs["varlen"] else precomputed_ref
    max_err = (out.float() - ref.float()).abs().max().item()
    results["max_err"] = max_err

    try:
        _, us = run_perftest(aiter_forward, num_iters=iters, num_warmup=warmup)
        if inputs["varlen"]:
            flops = sum(_flops(inputs["vl_q"][b], inputs["vl_kv"][b], H, D, 1, causal) for b in range(B))
        else:
            flops = _flops(Sq, inputs["Skv"], H, D, B, causal)
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def _fmt_result(r):
    """Format: 'Time(us) TFLOPS MaxErr MinCos St'.

    MinCos + a PASS/FAIL status are shown whenever the row carries the fixed-gate
    fields (fp8 comparator rows set min_cos/passed); for rows without them the
    extra columns render as '--' so the bf16/f16 layout is unchanged in width.
    """
    if r.get("skip"):
        return f"{'--':>10s} {'--':>8s} {'--':>8s} {'--':>7s} {'--':>4s}"
    if "err" in r:
        return f"{'--':>10s} {'ERR':>8s} {'--':>8s} {'--':>7s} {'--':>4s}"
    us = f"{r['us']:>10.1f}" if "us" in r else f"{'N/A':>10s}"
    tf = f"{r['tflops']:>8.1f}" if "tflops" in r else f"{'N/A':>8s}"
    err = f"{r['max_err']:>8.2e}" if "max_err" in r else f"{'N/A':>8s}"
    cos = f"{r['min_cos']:>7.4f}" if "min_cos" in r else f"{'--':>7s}"
    st = ("PASS" if r.get("passed") else "FAIL") if "passed" in r else "--"
    return f"{us} {tf} {err} {cos} {st:>4s}"


def _fmt_cmp(fly_r, other_r):
    """Format FlyDSL vs other: 'TFLOPS% MaxErr-ratio'."""
    return _fmt_cmp_values(_cmp_values(fly_r, other_r))


def _cmp_values(fly_r, other_r):
    """Return numeric comparison values for one valid FlyDSL/comparator row."""
    if other_r.get("skip") or "err" in other_r or "err" in fly_r:
        return {"skip": True}
    fly_tf = fly_r.get("tflops")
    oth_tf = other_r.get("tflops")
    fly_err = fly_r.get("max_err")
    oth_err = other_r.get("max_err")
    result = {}
    if fly_tf and oth_tf and oth_tf > 0:
        result["tflops_pct"] = fly_tf / oth_tf * 100
    if fly_err is not None and oth_err is not None and oth_err > 0:
        result["max_err_ratio"] = fly_err / oth_err
    return result


def _fmt_cmp_values(cmp_r):
    """Format numeric comparison values."""
    if cmp_r.get("skip"):
        return f"{'--':>7s} {'--':>6s}"
    if "tflops_pct" in cmp_r:
        pct = f"{cmp_r['tflops_pct']:>6.1f}%"
    else:
        pct = f"{'N/A':>7s}"
    if "max_err_ratio" in cmp_r:
        ratio = f"{cmp_r['max_err_ratio']:>5.2f}x"
    else:
        ratio = f"{'N/A':>6s}"
    return f"{pct} {ratio}"


def _gpu_short_name():
    """Extract short GPU name, e.g. 'AMD Instinct MI308X' -> 'MI308X'."""
    return torch.cuda.get_device_name(0).split()[-1]


def _csv_val(r, key):
    """Extract a value from result dict for CSV, formatted to match console."""
    if r.get("skip") or "err" in r:
        return ""
    v = r.get(key)
    if v is None:
        return ""
    if key in ("us", "tflops"):
        return f"{v:.1f}"
    if key == "max_err":
        return f"{v:.2e}"
    if key == "min_cos":
        return f"{v:.5f}"
    return v


def _csv_cmp(fly_r, other_r):
    """Compute (tflops_pct_str, maxerr_ratio_str) for CSV, formatted to match console."""
    return _csv_cmp_values(_cmp_values(fly_r, other_r))


def _csv_cmp_values(cmp_r):
    """Format numeric comparison values for CSV."""
    if cmp_r.get("skip"):
        return ("", "")
    pct = f"{cmp_r['tflops_pct']:.1f}%" if "tflops_pct" in cmp_r else ""
    rat = f"{cmp_r['max_err_ratio']:.2f}x" if "max_err_ratio" in cmp_r else ""
    return (pct, rat)


def _status_val(r):
    return ("PASS" if r.get("passed") else "FAIL") if "passed" in r else ""


def _write_cmp_csv(csv_path, data_rows, avg_rows):
    """Write compare-mode results to CSV."""
    header = [
        "B",
        "S",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "kv_sp",
        "KVLayout",
        "FlyDSL_Time(us)",
        "FlyDSL_TFLOPS",
        "FlyDSL_MaxErr",
        "FlyDSL_MinCos",
        "FlyDSL_Status",
        "aiter_ck_Time(us)",
        "aiter_ck_TFLOPS",
        "aiter_ck_MaxErr",
        "aiter_ck_MinCos",
        "aiter_ck_Status",
        "aiter_asm_Time(us)",
        "aiter_asm_TFLOPS",
        "aiter_asm_MaxErr",
        "aiter_asm_MinCos",
        "aiter_asm_Status",
        "Fly/aiter_ck_TFLOPS%",
        "Fly/aiter_ck_MaxErr_ratio",
        "Fly/aiter_asm_TFLOPS%",
        "Fly/aiter_asm_MaxErr_ratio",
    ]

    def _metrics(fr, cr, ar, cmp_overrides=None):
        if cmp_overrides is None:
            fck = _csv_cmp(fr, cr)
            fasm = _csv_cmp(fr, ar)
        else:
            fck, fasm = cmp_overrides
        return [
            _csv_val(fr, "us"),
            _csv_val(fr, "tflops"),
            _csv_val(fr, "max_err"),
            _csv_val(fr, "min_cos"),
            _status_val(fr),
            _csv_val(cr, "us"),
            _csv_val(cr, "tflops"),
            _csv_val(cr, "max_err"),
            _csv_val(cr, "min_cos"),
            _status_val(cr),
            _csv_val(ar, "us"),
            _csv_val(ar, "tflops"),
            _csv_val(ar, "max_err"),
            _csv_val(ar, "min_cos"),
            _status_val(ar),
            fck[0],
            fck[1],
            fasm[0],
            fasm[1],
        ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, fr, cr, ar in data_rows:
            w.writerow(list(cfg) + _metrics(fr, cr, ar))
        for avg_row in avg_rows:
            if len(avg_row) == 5:
                label, fa, ca, aa, cmp_overrides = avg_row
            else:
                label, fa, ca, aa = avg_row
                cmp_overrides = None
            # label + empty cfg columns
            w.writerow([label, "", "", "", "", "", "", "", ""] + _metrics(fa, ca, aa, cmp_overrides))


def _write_normal_csv(csv_path, data_rows, avg_rows):
    """Write normal-mode results to CSV."""
    header = [
        "B",
        "S",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "kv_sp",
        "KVLayout",
        "Path",
        "Status",
        "MaxErr",
        "MinCos",
        "Time(us)",
        "TFLOPS",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, path, status, r in data_rows:
            w.writerow(
                list(cfg)
                + [
                    path,
                    status,
                    _csv_val(r, "max_err"),
                    _csv_val(r, "min_cos"),
                    _csv_val(r, "us"),
                    _csv_val(r, "tflops"),
                ]
            )
        for label, avg in avg_rows:
            # label + empty cfg/path/status columns
            w.writerow(
                [
                    label,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "--",
                    _csv_val(avg, "max_err"),
                    _csv_val(avg, "min_cos"),
                    _csv_val(avg, "us"),
                    _csv_val(avg, "tflops"),
                ]
            )


def _write_varlen_cmp_csv(csv_path, data_rows, avg_rows=None):
    """Write compare-mode varlen / cross-length results to CSV."""
    header = [
        "Sq",
        "Skv",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "Path",
        "FlyDSL_Time(us)",
        "FlyDSL_TFLOPS",
        "FlyDSL_MaxErr",
        "FlyDSL_MinCos",
        "FlyDSL_Status",
        "aiter_ck_Time(us)",
        "aiter_ck_TFLOPS",
        "aiter_ck_MaxErr",
        "aiter_ck_MinCos",
        "aiter_ck_Status",
        "Fly/aiter_ck_TFLOPS%",
        "Fly/aiter_ck_MaxErr_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path, fly_r, ck_r in data_rows:
            fck = _csv_cmp(fly_r, ck_r)
            w.writerow(
                [
                    sq,
                    skv,
                    nh,
                    nh_kv,
                    hd,
                    dtype_key,
                    causal_tag,
                    path,
                    _csv_val(fly_r, "us"),
                    _csv_val(fly_r, "tflops"),
                    _csv_val(fly_r, "max_err"),
                    _csv_val(fly_r, "min_cos"),
                    _status_val(fly_r),
                    _csv_val(ck_r, "us"),
                    _csv_val(ck_r, "tflops"),
                    _csv_val(ck_r, "max_err"),
                    _csv_val(ck_r, "min_cos"),
                    _status_val(ck_r),
                    fck[0],
                    fck[1],
                ]
            )
        for label, fly_r, ck_r, fly_ck_cmp in avg_rows or []:
            w.writerow(
                [
                    label,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    _csv_val(fly_r, "us"),
                    _csv_val(fly_r, "tflops"),
                    _csv_val(fly_r, "max_err"),
                    _csv_val(fly_r, "min_cos"),
                    "--",
                    _csv_val(ck_r, "us"),
                    _csv_val(ck_r, "tflops"),
                    _csv_val(ck_r, "max_err"),
                    _csv_val(ck_r, "min_cos"),
                    "--",
                    _csv_cmp_values(fly_ck_cmp)[0],
                    _csv_cmp_values(fly_ck_cmp)[1],
                ]
            )


def _write_varlen_normal_csv(csv_path, data_rows, avg_rows=None):
    """Write normal-mode varlen / cross-length results to CSV."""
    header = [
        "Sq",
        "Skv",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "Path",
        "Status",
        "MaxErr",
        "MinCos",
        "Time(us)",
        "TFLOPS",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path, status, r in data_rows:
            w.writerow(
                [
                    sq,
                    skv,
                    nh,
                    nh_kv,
                    hd,
                    dtype_key,
                    causal_tag,
                    path,
                    status,
                    _csv_val(r, "max_err"),
                    _csv_val(r, "min_cos"),
                    _csv_val(r, "us"),
                    _csv_val(r, "tflops"),
                ]
            )
        for label, r in avg_rows or []:
            w.writerow(
                [
                    label,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "--",
                    _csv_val(r, "max_err"),
                    _csv_val(r, "min_cos"),
                    _csv_val(r, "us"),
                    _csv_val(r, "tflops"),
                ]
            )


def _valid_result(r):
    return not r.get("skip") and "err" not in r


def _avg_results(results_list, keys=("us", "tflops", "max_err")):
    """Average valid results over the specified keys."""
    valid = [r for r in results_list if _valid_result(r)]
    if not valid:
        return {"skip": True}
    avg = {}
    for key in keys:
        vals = [r[key] for r in valid if key in r]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def _avg_cmp_values(rows, fly_idx, other_idx):
    """Average per-row comparison values over rows where both sides are valid."""
    cmp_rows = [
        _cmp_values(row[fly_idx], row[other_idx])
        for row in rows
        if _valid_result(row[fly_idx]) and _valid_result(row[other_idx])
    ]
    if not cmp_rows:
        return {"skip": True}
    avg = {}
    for key in ("tflops_pct", "max_err_ratio"):
        vals = [r[key] for r in cmp_rows if key in r]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def _tag_group(cfg):
    """Extract (dtype_key, causal_tag) from config tuple (B, S, H, Hkv, D, dtype, causal, kv_sp)."""
    return cfg[5], cfg[6]


def _print_grouped_avgs(rows, tag_fn, print_avg_fn):
    """Print grouped averages: all, then dtype x causal, dtype-only, causal-only."""
    print_avg_fn("AVG (all)", rows)
    seen_dtypes, seen_causals = [], []
    for row in rows:
        dk, ct = tag_fn(row)
        if dk not in seen_dtypes:
            seen_dtypes.append(dk)
        if ct not in seen_causals:
            seen_causals.append(ct)
    if len(seen_dtypes) > 1 and len(seen_causals) > 1:
        for dk in seen_dtypes:
            for ct in seen_causals:
                subset = [r for r in rows if tag_fn(r) == (dk, ct)]
                if subset:
                    print_avg_fn(f"AVG ({dk} {ct})", subset)
    if len(seen_dtypes) > 1:
        for dk in seen_dtypes:
            subset = [r for r in rows if tag_fn(r)[0] == dk]
            if subset:
                print_avg_fn(f"AVG ({dk})", subset)
    if len(seen_causals) > 1:
        for ct in seen_causals:
            subset = [r for r in rows if tag_fn(r)[1] == ct]
            if subset:
                print_avg_fn(f"AVG ({ct})", subset)


_KV_LAYOUT_W = 24
_CFG_HDR = (
    f"{'B':>4s} {'S':>6s} {'H':>4s} {'Hkv':>4s} {'D':>4s} "
    f"{'dtype':>5s} {'causal':>8s} {'kv_sp':>5s} {'KVLayout':<{_KV_LAYOUT_W}s}"
)
_CFG_W = len(_CFG_HDR)
_PATH_W = 20
_KV_CACHE_LAYOUTS = ("linear", "vectorized")


def _selected_arg_values(value, all_values):
    return list(all_values) if value == "all" else [value]


def _kv_layout_label(path, page_size):
    return "dense" if not path else path


def _fmt_cfg(cfg):
    """Format config tuple (B, S, H, Hkv, D, dtype, causal, kv_sp, KVLayout)."""
    B, S, H, Hkv, D, dt, cs, ksp, kv_layout = cfg
    return f"{B:>4d} {S:>6d} {H:>4d} {Hkv:>4d} {D:>4d} " f"{dt:>5s} {cs:>8s} {ksp:>5d} {kv_layout:<{_KV_LAYOUT_W}s}"


def _fmt_normal_row(cfg, path, status, r):
    """Format one row for normal test mode."""
    cfg_s = _fmt_cfg(cfg) if isinstance(cfg, tuple) else f"{cfg:>{_CFG_W}s}"
    path_s = f"  {path:<{_PATH_W}s}" if path else f"  {'':<{_PATH_W}s}"
    prefix = f"{cfg_s}{path_s}"
    if "err" in r:
        return f"{prefix} | {'ERROR':>6s} | {r['err'][:60]}"
    if r.get("skip"):
        return f"{prefix} | {'SKIP':>6s} | n/a"
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    return f"{prefix} | {status:>6s} | " f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | " f"{us_s} {tf_s}"


_EXTRA_HDR = (
    f"  {'Sq':<24} {'Skv':<24} {'H':>4} {'Hkv':>4} {'D':>4} " f"{'dtype':>6} {'causal':>8} {'Path':<{_PATH_W}s}"
)
_EXTRA_W = len(_EXTRA_HDR)


def _fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path=""):
    return f"  {sq:<24} {skv:<24} {nh:>4} {nh_kv:>4} {hd:>4} " f"{dtype_key:>6} {causal_tag:>8} {path:<{_PATH_W}s}"


def _fmt_extra_cmp_row(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path, fly_r, ck_r):
    return (
        f"{_fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path=path)} | "
        f"{_fmt_result(fly_r)} | {_fmt_result(ck_r)} | {_fmt_cmp(fly_r, ck_r)}"
    )


def _fmt_extra_normal_row(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, status, r, path=""):
    prefix = _fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, path=path)
    if "err" in r:
        return f"{prefix} | {'ERROR':>6s} | {r['err'][:60]}"
    if r.get("skip"):
        return f"{prefix} | {'SKIP':>6s} | n/a"
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    min_cos = r.get("min_cos")
    min_cos_s = f"{min_cos:>8.5f}" if min_cos is not None else f"{'N/A':>8s}"
    return f"{prefix} | {status:>6s} | {r['max_err']:>8.2e} {min_cos_s} | {us_s} {tf_s}"


def _fmt_extra_cmp_avg_row(label, fly_r, ck_r, fly_ck_cmp):
    return f"{label:>{_EXTRA_W}s} | {_fmt_result(fly_r)} | {_fmt_result(ck_r)} | {_fmt_cmp_values(fly_ck_cmp)}"


def _fmt_extra_normal_avg_row(label, r):
    if r.get("skip"):
        return None
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    min_cos = r.get("min_cos")
    min_cos_s = f"{min_cos:>8.5f}" if min_cos is not None else f"{'N/A':>8s}"
    return f"{label:>{_EXTRA_W}s} | {'--':>6s} | {r['max_err']:>8.2e} {min_cos_s} | {us_s} {tf_s}"


def main():
    parser = argparse.ArgumentParser(description="flash_attn_func FlyDSL Test/Benchmark")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=None,
        help="KV head count for GQA/MQA. Default = num_heads (MHA). " "Requires num_heads %% num_kv_heads == 0.",
    )
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument(
        "--num_kv_splits",
        type=int,
        default=1,
        help="Split-K factor for the gfx950 DUALWAVE_SWP kernel. >1 runs the split-K "
        "path (+combine kernel) via run_splitk_config; D=64/128 bf16/f16, seq_len >= 384.",
    )
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument("--causal", action="store_true", dest="causal")
    causal_group.add_argument("--no-causal", action="store_false", dest="causal")
    parser.set_defaults(causal=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16", "fp8"],
        help="Data type: fp16, bf16, or fp8 (e4m3fn). Default: bf16+fp16; fp8 must be requested explicitly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FlyDSL vs aiter_ck vs aiter_asm performance (requires aiter)",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Run additional varlen/cross-length configs from EXTRA_CONFIGS",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Add an additive attention bias to the scores: softmax(q@k^T * sm_scale + bias). "
        "Dense bias is [Sq, Skv] broadcast over batch and head; varlen bias is packed "
        "[total_q, max_seqlen_kv] with global q rows and batch-local key columns. Combines "
        "with --block-table, where columns stay the logical (batch-local) key positions. "
        "gfx950 bf16/f16 D=64/128 only; incompatible with fp8. Rows whose "
        "bias exceeds the i32 offset / 4 GiB buffer limits are SKIPped.",
    )
    parser.add_argument(
        "--alibi",
        action="store_true",
        help="Add a per-head ALiBi positional bias: score += -slope * |i + seqlen_kv - seqlen_q - j|, "
        "applied after the 1/sqrt(D) scaling and bottom-right aligned like the causal mask. "
        "Slopes are the canonical 2**(-8*(h+1)/H) ladder. gfx950 bf16/f16 D=64/128 only; "
        "incompatible with --block-table and fp8. Combines with --bias.",
    )
    parser.add_argument(
        "--sink",
        action="store_true",
        help="Add a per-head attention sink: one extra softmax denominator logit with no matching V "
        "row, O = sum_j exp(s_j-m) v_j / (exp(sink-m) + sum_j exp(s_j-m)). The sink is calibrated per "
        "run to --sink-share of the softmax mass; an uncalibrated sink near 0 would sit below the bf16 "
        "noise floor and pass even if dropped. gfx950 bf16/f16 D=64/128 only; incompatible with "
        "--block-table and fp8. Combines with --bias and --alibi. Under --compare the aiter_ck "
        "column runs a real sink baseline (mha_fwd sink_size/sink_ptr); aiter_asm has no sink "
        "parameter and is SKIPped, as is the varlen path.",
    )
    parser.add_argument(
        "--sink-share",
        type=float,
        default=DEFAULT_SINK_SHARE,
        dest="sink_share",
        help=f"Fraction of the softmax mass the sink should take, in (0, 1). Default {DEFAULT_SINK_SHARE}. "
        "Values in 0.25-0.95 keep the sink well above bf16 noise. Requires --sink.",
    )
    parser.add_argument(
        "--alibi-2d",
        action="store_true",
        dest="alibi_two_d",
        help="Use a per-(batch, head) [B, H] slope table instead of the [H] form, exercising the "
        "kernel's alibi_stride_b path. Requires --alibi.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-config result_md5 / ref_md5 and bit-identical check (also enabled in --compare mode)",
    )
    parser.add_argument(
        "--block-table",
        action="store_true",
        help="Build K/V through a paged cache plus block_table before materializing the current dense/packed ABI",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1,
        help="Page size used by --block-table test data construction (default: 1)",
    )
    parser.add_argument(
        "--kv-cache-layout",
        type=str,
        default="linear",
        choices=["linear", "vectorized", "all"],
        help=(
            "Paged K/V cache memory layout used by --block-table. "
            "linear: 4D [NumBlocks, PageSize, NumKVHeads, HeadDim]; "
            "vectorized: aiter 5D layout, "
            "K=[NumBlocks, NumKVHeads, HeadDim/kVectorSize, PageSize, kVectorSize], "
            "V=[NumBlocks, NumKVHeads, PageSize/kVectorSize, HeadDim, kVectorSize]. "
            "kVectorSize = 16 / element_size (bf16/fp16: 8, fp8: 16). "
            "Use all to sweep all layouts. Default: linear."
        ),
    )
    # ── Kernel build options (override defaults without env vars) ──────────────
    parser.add_argument(
        "--waves-per-eu",
        type=int,
        default=2,
        dest="waves_per_eu",
        help="waves_per_eu occupancy hint passed to the FlyDSL kernel builder (default: 2)",
    )
    parser.add_argument(
        "--no-lazy-rescale",
        action="store_false",
        dest="dualwave_swp_lazy_rescale",
        help="Disable the DUALWAVE_SWP lazy online-softmax rescale (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_lazy_rescale=True)
    parser.add_argument(
        "--no-setprio",
        action="store_false",
        dest="dualwave_swp_setprio",
        help="Disable s_setprio scheduling hints in the DUALWAVE_SWP kernel (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_setprio=True)
    parser.add_argument(
        "--debug-lazy-counts",
        action="store_true",
        dest="dualwave_swp_debug_lazy_counts",
        help="Enable lazy-rescale branch counters (dualwave_swp_debug_lazy_counts=True, disabled by default)",
    )
    parser.add_argument(
        "--no-stagger",
        action="store_false",
        dest="dualwave_swp_enable_stagger",
        help="Disable wave-group phase stagger in the DUALWAVE_SWP kernel (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_enable_stagger=True)
    parser.add_argument(
        "--trigger-lazy-else",
        action="store_true",
        dest="trigger_lazy_else",
        help="Construct adversarial inputs (Q=1, K tile0=0, K tile1=80) to force the "
        "lazy-rescale else-branch (row_max - m_row > 8); dense mode only, for debugging",
    )
    args = parser.parse_args()
    if not args.block_table and args.kv_cache_layout != "linear":
        parser.error("--kv-cache-layout requires --block-table")
    # fp8 has no bias support in the kernel; reject rather than run a bias-free
    # kernel against a biased reference. Paged KV does support bias.
    if args.bias and args.dtype == "fp8":
        parser.error("--bias is not supported with --dtype fp8")
    if args.alibi and args.block_table:
        parser.error("--alibi is not supported with --block-table (paged KV)")
    if args.alibi and args.dtype == "fp8":
        parser.error("--alibi is not supported with --dtype fp8")
    if args.alibi_two_d and not args.alibi:
        parser.error("--alibi-2d requires --alibi")
    if args.sink and args.block_table:
        parser.error("--sink is not supported with --block-table (paged KV)")
    if args.sink and args.dtype == "fp8":
        parser.error("--sink is not supported with --dtype fp8")
    if args.sink_share != DEFAULT_SINK_SHARE and not args.sink:
        parser.error("--sink-share requires --sink")
    if not 0.0 < args.sink_share < 1.0:
        parser.error(f"--sink-share must be in (0, 1), got {args.sink_share}")

    # Build kernel config from parsed args (no env-var reads).
    FLASH_ATTN_FUNC_KERNEL_CONFIG.update(
        {
            "waves_per_eu": args.waves_per_eu,
            "dualwave_swp_lazy_rescale": args.dualwave_swp_lazy_rescale,
            "dualwave_swp_setprio": args.dualwave_swp_setprio,
            "dualwave_swp_debug_lazy_counts": args.dualwave_swp_debug_lazy_counts,
            "dualwave_swp_enable_stagger": args.dualwave_swp_enable_stagger,
        }
    )

    dtype_map = {
        "fp16": (torch.float16, "f16"),
        "bf16": (torch.bfloat16, "bf16"),
        "fp8": (torch.bfloat16, "fp8"),
    }
    dtypes_to_test = [args.dtype] if args.dtype else ["bf16", "fp16"]
    causals_to_test = [args.causal] if args.causal is not None else [True, False]

    head_dims_to_test = [args.head_dim] if args.head_dim is not None else [64, 128]

    if args.batch or args.seq_len or args.num_heads or args.head_dim or args.num_kv_heads:
        nh_single = args.num_heads or 8
        configs = [
            (
                args.batch or 1,
                args.seq_len or 128,
                nh_single,
                args.num_kv_heads if args.num_kv_heads is not None else nh_single,
                args.num_kv_splits,
            )
        ]
    else:
        configs = DEFAULT_CONFIGS

    causal_desc = {True: "causal", False: "non-causal", None: "causal+non-causal"}[args.causal]
    dtype_desc = args.dtype or "bf16+fp16"
    _terms = [t for t, on in (("bias", args.bias), ("alibi", args.alibi), ("sink", args.sink)) if on]
    bias_desc = ("; " + "+".join(_terms)) if _terms else ""
    # Keep biased and unbiased baselines in separate CSVs so they stay diffable.
    csv_tag = ("_" + "".join(_terms)) if _terms else ""
    extra_cases = (
        [_extra_case_from_config(row) for row in EXTRA_CONFIGS] if args.extra and configs is DEFAULT_CONFIGS else []
    )
    run_configs = [
        (batch, seq_len, nh, nh_kv_default, hd, cfg_kv_splits)
        for batch, seq_len, nh, nh_kv_default, cfg_kv_splits in configs
        for hd in head_dims_to_test
    ]
    extra_run_cases = [(case, hd) for case in extra_cases for hd in head_dims_to_test]
    paged_kv_paths = [(None, "")]
    if args.block_table:
        paged_kv_paths = [
            (kv_cache_layout, f"{kv_cache_layout}:p{args.page_size}")
            for kv_cache_layout in _selected_arg_values(args.kv_cache_layout, _KV_CACHE_LAYOUTS)
        ]

    if args.compare:
        # ---- Comparison mode: FlyDSL vs aiter_ck vs aiter_asm ----
        print("=" * 130)
        print(f"FlyDSL vs aiter_ck vs aiter_asm  ({causal_desc}, {dtype_desc}{bias_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if args.num_kv_splits > 1:
            print(
                f"  FlyDSL column: split-K path (num_kv_splits={args.num_kv_splits}); "
                f"D not in {{64,128}} / non-bf16,f16 / seq_len<384 / ws>4GiB configs SKIP"
            )
        print(f"  FlyDSL opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        if "fp8" in dtypes_to_test:
            print(
                "  fp8 mode: aiter_asm column = NATIVE gfx950 fp8 ASM (fmha_v3_fwd, "
                "how_v3_bf16_cvt=0); SKIP where the native-asm gate is not met. "
                "aiter_ck column = aiter_ck fp8 (mha_fwd with descales), secondary."
            )
        else:
            print("  aiter_ck: bf16+fp16, aiter_asm: bf16 only (how_v3_bf16_cvt=2, bf16-convert)")
        print("=" * 130)
        print("Running benchmarks ...")

        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, nh_kv_default, hd, cfg_kv_splits in run_configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads / --num_kv_splits (if set) override the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    kv_splits = args.num_kv_splits if args.num_kv_splits > 1 else cfg_kv_splits
                    for kv_cache_layout, path in paged_kv_paths:
                        cfg = (
                            batch,
                            seq_len,
                            nh,
                            nh_kv,
                            hd,
                            dtype_key,
                            causal_tag,
                            kv_splits,
                            _kv_layout_label(path, args.page_size),
                        )
                        print(f"  {_fmt_cfg(cfg)} ...", flush=True)

                        precomputed_inputs = None
                        if args.block_table:
                            precomputed_inputs, shared_ref, precompute_status = _precompute_paged_kv_inputs_and_ref(
                                batch=batch,
                                seqlen_q=seq_len,
                                seqlen_kv=None,
                                varlen_seqlens_q=None,
                                varlen_seqlens_kv=None,
                                num_heads=nh,
                                head_dim=hd,
                                num_kv_heads=nh_kv,
                                dtype=dtype,
                                causal=causal,
                                seed=args.seed,
                                page_size=args.page_size,
                                kv_cache_layout=kv_cache_layout or "linear",
                                trigger_lazy_else=args.trigger_lazy_else,
                                use_bias=args.bias,
                            )
                            if precompute_status is not None:
                                rows.append((cfg, precompute_status, precompute_status, {"skip": True}))
                                continue
                        else:
                            shared_ref = None
                            if dtype_str == "fp8":
                                pass
                            elif args.trigger_lazy_else or args.bias or args.alibi or args.sink:
                                precomputed_inputs, shared_ref = _build_inputs_and_reference_for_config(
                                    batch=batch,
                                    seqlen_q=seq_len,
                                    seqlen_kv=None,
                                    varlen_seqlens_q=None,
                                    varlen_seqlens_kv=None,
                                    num_heads=nh,
                                    head_dim=hd,
                                    num_kv_heads=nh_kv,
                                    dtype=dtype,
                                    causal=causal,
                                    seed=args.seed,
                                    use_block_table=False,
                                    page_size=args.page_size,
                                    kv_cache_layout=kv_cache_layout or "linear",
                                    trigger_lazy_else=args.trigger_lazy_else,
                                    use_bias=args.bias,
                                    use_alibi=args.alibi,
                                    use_sink=args.sink,
                                    sink_share=args.sink_share,
                                    alibi_two_d=args.alibi_two_d,
                                )
                            else:
                                # All three use the same seed -> same Q/K/V -> identical reference.
                                setup_seed(args.seed)
                                _q = torch.empty(batch, seq_len, nh, hd, dtype=dtype, device="cuda").uniform_(
                                    *UNIFORM_RANGE
                                )
                                _k = torch.empty(batch, seq_len, nh_kv, hd, dtype=dtype, device="cuda").uniform_(
                                    *UNIFORM_RANGE
                                )
                                _v = torch.empty(batch, seq_len, nh_kv, hd, dtype=dtype, device="cuda").uniform_(
                                    *UNIFORM_RANGE
                                )
                                shared_ref = pytorch_ref_attention(
                                    _q.float(), _k.float(), _v.float(), causal=causal
                                ).to(dtype)
                                del _q, _k, _v

                        try:
                            if dtype_str == "fp8":
                                if args.block_table:
                                    raise ValueError("fp8 flash_attn does not support --block-table")
                                fly_r = run_fp8_config(
                                    batch,
                                    seq_len,
                                    nh,
                                    hd,
                                    causal,
                                    warmup=args.warmup,
                                    iters=args.iters,
                                    seed=args.seed,
                                    verbose=False,
                                    num_kv_heads=nh_kv,
                                    num_kv_splits=kv_splits,
                                )
                            else:
                                fly_r = run_attn_config(
                                    nh,
                                    hd,
                                    dtype,
                                    causal,
                                    args.warmup,
                                    args.iters,
                                    batch=batch,
                                    seqlen_q=seq_len,
                                    num_kv_heads=nh_kv,
                                    num_kv_splits=kv_splits,
                                    seed=args.seed,
                                    dtype_str=dtype_str,
                                    verbose=args.verbose,
                                    trigger_lazy_else=args.trigger_lazy_else,
                                    compare_mode=True,
                                    precomputed_ref=shared_ref,
                                    precomputed_inputs=precomputed_inputs,
                                    use_block_table=args.block_table,
                                    page_size=args.page_size,
                                    kv_cache_layout=kv_cache_layout or "linear",
                                    use_bias=args.bias,
                                    use_alibi=args.alibi,
                                    use_sink=args.sink,
                                    sink_share=args.sink_share,
                                    alibi_two_d=args.alibi_two_d,
                                )
                        except Exception as _fly_err:
                            print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)}: {_fly_err}", flush=True)
                            fly_r = {"err": str(_fly_err)}

                        if dtype_str == "fp8":
                            ck_r = run_aiter_fp8_bench(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                backend="ck",
                                num_kv_heads=nh_kv,
                            )
                            asm_r = run_aiter_fp8_bench(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                backend="asm",
                                num_kv_heads=nh_kv,
                            )
                        elif args.block_table:
                            ck_r = run_aiter_batch_prefill_bench(
                                precomputed_inputs,
                                shared_ref,
                                nh,
                                hd,
                                dtype,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                            )
                            asm_r = {"skip": True}
                        else:
                            ck_r = run_aiter_bench(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                dtype,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                backend="ck",
                                num_kv_heads=nh_kv,
                                precomputed_ref=shared_ref,
                                precomputed_inputs=precomputed_inputs,
                                use_bias=args.bias,
                                use_alibi=args.alibi,
                                use_sink=args.sink,
                            )
                            asm_r = run_aiter_bench(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                dtype,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                backend="asm",
                                num_kv_heads=nh_kv,
                                precomputed_ref=shared_ref,
                                precomputed_inputs=precomputed_inputs,
                                use_bias=args.bias,
                                use_alibi=args.alibi,
                                use_sink=args.sink,
                            )
                        rows.append((cfg, fly_r, ck_r, asm_r))

        col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s} {'MinCos':>7s} {'St':>4s}"
        _col_w = len(col)
        cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
        hdr1 = (
            f"{_CFG_HDR} | {'FlyDSL':^{_col_w}s} | {'aiter_ck':^{_col_w}s} | {'aiter_asm':^{_col_w}s}"
            f" | {'Fly/aiter_ck':^14s} | {'Fly/aiter_asm':^14s}"
        )
        hdr2 = f"{'':>{_CFG_W}s} | {col} | {col} | {col}" f" | {cmp_col} | {cmp_col}"
        sep = "-" * len(hdr2)
        print(f"\n{hdr1}")
        print(hdr2)
        print(sep)
        for cfg, fly_r, ck_r, asm_r in rows:
            print(
                f"{_fmt_cfg(cfg)} | {_fmt_result(fly_r)} | "
                f"{_fmt_result(ck_r)} | {_fmt_result(asm_r)}"
                f" | {_fmt_cmp(fly_r, ck_r)}"
                f" | {_fmt_cmp(fly_r, asm_r)}"
            )

        cmp_avg_rows = []

        def _cmp_avg(label, subset):
            fa = _avg_results([f for _, f, _, _ in subset])
            ca = _avg_results([c for _, _, c, _ in subset])
            aa = _avg_results([a for _, _, _, a in subset])
            fck_cmp = _avg_cmp_values(subset, 1, 2)
            fasm_cmp = _avg_cmp_values(subset, 1, 3)
            print(
                f"{label:>{_CFG_W}s} | {_fmt_result(fa)} | "
                f"{_fmt_result(ca)} | {_fmt_result(aa)}"
                f" | {_fmt_cmp_values(fck_cmp)}"
                f" | {_fmt_cmp_values(fasm_cmp)}"
            )
            cmp_avg_rows.append(
                (
                    label,
                    fa,
                    ca,
                    aa,
                    (
                        _csv_cmp_values(fck_cmp),
                        _csv_cmp_values(fasm_cmp),
                    ),
                )
            )

        print(sep)
        _print_grouped_avgs(rows, lambda r: _tag_group(r[0]), _cmp_avg)
        print("=" * len(hdr2))

        csv_path = f"fmha_perf_compare{csv_tag}_{_gpu_short_name()}.csv"
        _write_cmp_csv(csv_path, rows, cmp_avg_rows)
        print(f"Results saved to: {csv_path}")

        if extra_cases:
            print("=" * 130)
            print("Additional dense/varlen/cross-length cases: FlyDSL vs aiter_ck")
            print("=" * 130)
            col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s}"
            cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
            xhdr1 = f"{_EXTRA_HDR} | " f"{'FlyDSL':^28} | {'aiter_ck':^28} | {'Fly/CK':^14}"
            xhdr2 = f"{'':>{_EXTRA_W}} | {col} | {col} | {cmp_col}"
            varlen_cmp_rows = []
            for dtype_key in dtypes_to_test:
                dtype, dtype_str = dtype_map[dtype_key]
                for causal in causals_to_test:
                    ctag = "causal" if causal else "nocausal"
                    for case, hd in extra_run_cases:
                        nh = case["nh"]
                        nh_kv_eff = args.num_kv_heads if args.num_kv_heads is not None else case["nh_kv"]
                        kv_splits = case.get("kv_splits", 1)
                        kwargs = dict(case["kwargs"])
                        for kv_cache_layout, path in paged_kv_paths:
                            pre = _fmt_extra_prefix(
                                case["sq_label"],
                                case["skv_label"],
                                nh,
                                nh_kv_eff,
                                hd,
                                dtype_key,
                                ctag,
                                path=path,
                            )
                            print(f"{pre} ...", flush=True)
                            precomputed_inputs = None
                            shared_ref = None
                            if args.block_table:
                                precomputed_inputs, shared_ref, precompute_status = _precompute_paged_kv_inputs_and_ref(
                                    batch=kwargs.get("batch", 1),
                                    seqlen_q=kwargs.get("seqlen_q"),
                                    seqlen_kv=kwargs.get("seqlen_kv"),
                                    varlen_seqlens_q=kwargs.get("varlen_seqlens_q"),
                                    varlen_seqlens_kv=kwargs.get("varlen_seqlens_kv"),
                                    num_heads=nh,
                                    head_dim=hd,
                                    num_kv_heads=nh_kv_eff,
                                    dtype=dtype,
                                    causal=causal,
                                    seed=args.seed,
                                    page_size=args.page_size,
                                    kv_cache_layout=kv_cache_layout or "linear",
                                    use_bias=args.bias,
                                )
                                if precompute_status is not None:
                                    varlen_cmp_rows.append(
                                        (
                                            case["sq_label"],
                                            case["skv_label"],
                                            nh,
                                            nh_kv_eff,
                                            hd,
                                            dtype_key,
                                            ctag,
                                            path,
                                            precompute_status,
                                            precompute_status,
                                        )
                                    )
                                    continue
                            try:
                                fly_r = run_attn_config(
                                    nh,
                                    hd,
                                    dtype,
                                    causal,
                                    args.warmup,
                                    args.iters,
                                    num_kv_heads=nh_kv_eff,
                                    num_kv_splits=kv_splits,
                                    seed=args.seed,
                                    dtype_str=dtype_str,
                                    verbose=args.verbose,
                                    compare_mode=True,
                                    precomputed_ref=shared_ref,
                                    precomputed_inputs=precomputed_inputs,
                                    use_block_table=args.block_table,
                                    page_size=args.page_size,
                                    kv_cache_layout=kv_cache_layout or "linear",
                                    use_bias=args.bias,
                                    use_alibi=args.alibi,
                                    use_sink=args.sink,
                                    sink_share=args.sink_share,
                                    alibi_two_d=args.alibi_two_d,
                                    **kwargs,
                                )
                            except Exception as _fly_err:
                                print(
                                    f"    [FlyDSL unsupported] Sq={case['sq_label']} "
                                    f"Skv={case['skv_label']} {path}: {_fly_err}",
                                    flush=True,
                                )
                                fly_r = {"err": str(_fly_err)}
                            if args.block_table:
                                ck_r = run_aiter_batch_prefill_bench(
                                    precomputed_inputs,
                                    shared_ref,
                                    nh,
                                    hd,
                                    dtype,
                                    causal,
                                    args.warmup,
                                    args.iters,
                                )
                            else:
                                ck_r = run_aiter_bench(
                                    kwargs.get("batch", 1),
                                    kwargs.get("seqlen_q", max(kwargs.get("varlen_seqlens_q", [1]))),
                                    nh,
                                    hd,
                                    dtype,
                                    causal,
                                    args.warmup,
                                    args.iters,
                                    seed=args.seed,
                                    backend="ck",
                                    num_kv_heads=nh_kv_eff,
                                    seqlen_kv=kwargs.get("seqlen_kv"),
                                    varlen_seqlens_q=kwargs.get("varlen_seqlens_q"),
                                    varlen_seqlens_kv=kwargs.get("varlen_seqlens_kv"),
                                    use_bias=args.bias,
                                    use_alibi=args.alibi,
                                    use_sink=args.sink,
                                )
                            varlen_cmp_rows.append(
                                (
                                    case["sq_label"],
                                    case["skv_label"],
                                    nh,
                                    nh_kv_eff,
                                    hd,
                                    dtype_key,
                                    ctag,
                                    path,
                                    fly_r,
                                    ck_r,
                                )
                            )
            print("\n" + xhdr1)
            print(xhdr2)
            print("  " + "-" * (len(xhdr2) - 2))
            for sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, path, fly_r, ck_r in varlen_cmp_rows:
                print(_fmt_extra_cmp_row(sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, path, fly_r, ck_r))
            print("  " + "-" * (len(xhdr2) - 2))

            varlen_cmp_avg_rows = []

            def _extra_cmp_avg(label, subset):
                fly_avg = _avg_results([row[8] for row in subset])
                ck_avg = _avg_results([row[9] for row in subset])
                fly_ck_cmp = _avg_cmp_values(subset, 8, 9)
                print(_fmt_extra_cmp_avg_row(label, fly_avg, ck_avg, fly_ck_cmp))
                varlen_cmp_avg_rows.append((label, fly_avg, ck_avg, fly_ck_cmp))

            _print_grouped_avgs(varlen_cmp_rows, lambda r: (r[5], r[6]), _extra_cmp_avg)
            print("=" * len(xhdr2))
            varlen_csv_path = f"fmha_varlen_perf_compare{csv_tag}_{_gpu_short_name()}.csv"
            _write_varlen_cmp_csv(varlen_csv_path, varlen_cmp_rows, varlen_cmp_avg_rows)
            print(f"Varlen results saved to: {varlen_csv_path}")

    else:
        # ---- Normal FlyDSL test mode ----
        print("=" * 130)
        print(f"FlyDSL flash_attn_func ({causal_desc}, {dtype_desc}{bias_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Kernel opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        if args.block_table:
            print(
                f"  Test data: paged KV reference paths (page_size={args.page_size}, "
                f"kv_cache_layout={args.kv_cache_layout})"
            )
        print("=" * 130)

        hdr = (
            f"{_CFG_HDR}  {'Path':<{_PATH_W}s} | {'Status':>6s} | {'MaxErr':>8s} "
            f"{'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
        )
        print(f"\n{hdr}")
        print("-" * len(hdr))

        all_passed = True
        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, nh_kv_default, hd, cfg_kv_splits in run_configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads / --num_kv_splits (if set) override the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    kv_splits = args.num_kv_splits if args.num_kv_splits > 1 else cfg_kv_splits
                    for kv_cache_layout, path in paged_kv_paths:
                        cfg = (
                            batch,
                            seq_len,
                            nh,
                            nh_kv,
                            hd,
                            dtype_key,
                            causal_tag,
                            kv_splits,
                            _kv_layout_label(path, args.page_size),
                        )
                        try:
                            precomputed_inputs = None
                            precomputed_ref = None
                            if args.block_table:
                                precomputed_inputs, precomputed_ref, precompute_status = (
                                    _precompute_paged_kv_inputs_and_ref(
                                        batch=batch,
                                        seqlen_q=seq_len,
                                        seqlen_kv=None,
                                        varlen_seqlens_q=None,
                                        varlen_seqlens_kv=None,
                                        num_heads=nh,
                                        head_dim=hd,
                                        num_kv_heads=nh_kv,
                                        dtype=dtype,
                                        causal=causal,
                                        seed=args.seed,
                                        page_size=args.page_size,
                                        kv_cache_layout=kv_cache_layout or "linear",
                                        trigger_lazy_else=args.trigger_lazy_else,
                                        use_bias=args.bias,
                                    )
                                )
                                if precompute_status is not None:
                                    status = "ERROR" if "err" in precompute_status else "SKIP"
                                    if status == "ERROR":
                                        all_passed = False
                                    print(_fmt_normal_row(cfg, path, status, precompute_status))
                                    rows.append((cfg, path, status, precompute_status))
                                    continue

                            r = run_attn_config(
                                nh,
                                hd,
                                dtype,
                                causal,
                                args.warmup,
                                args.iters,
                                batch=batch,
                                seqlen_q=seq_len,
                                num_kv_heads=nh_kv,
                                num_kv_splits=kv_splits,
                                seed=args.seed,
                                dtype_str=dtype_str,
                                verbose=True,
                                trigger_lazy_else=args.trigger_lazy_else,
                                use_block_table=args.block_table,
                                precomputed_ref=precomputed_ref,
                                precomputed_inputs=precomputed_inputs,
                                page_size=args.page_size,
                                kv_cache_layout=kv_cache_layout or "linear",
                                use_bias=args.bias,
                                use_alibi=args.alibi,
                                use_sink=args.sink,
                                sink_share=args.sink_share,
                                alibi_two_d=args.alibi_two_d,
                            )
                            if "err" in r:
                                print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)} {path}: {r['err']}", flush=True)
                                print(_fmt_normal_row(cfg, path, "ERROR", r))
                                all_passed = False
                                rows.append((cfg, path, "ERROR", r))
                                continue
                            if r.get("skip"):
                                print(_fmt_normal_row(cfg, path, "SKIP", r))
                                rows.append((cfg, path, "SKIP", r))
                                continue

                            if r.get("bench_err"):
                                print(f"    [FlyDSL bench failed] {_fmt_cfg(cfg)} {path}: {r['bench_err']}", flush=True)
                                status = "BENCHERR"
                                all_passed = False
                            else:
                                status = "PASS" if r["passed"] else "FAIL"
                                if not r["passed"]:
                                    all_passed = False
                            print(_fmt_normal_row(cfg, path, status, r))
                            rows.append((cfg, path, status, r))
                        except Exception as e:
                            print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)} {path}: {e}", flush=True)
                            print(_fmt_normal_row(cfg, path, "ERROR", {"err": str(e)}))
                            all_passed = False
                            rows.append((cfg, path, "ERROR", {"err": str(e)}))

        # ---- Summary table ----
        print(f"\n{hdr}")
        print("-" * len(hdr))
        for cfg, path, status, r in rows:
            print(_fmt_normal_row(cfg, path, status, r))

        normal_avg_rows = []

        def _normal_avg_fn(label, subset):
            avg = _avg_results(
                [r for _, _, _, r in subset],
                keys=("max_err", "min_cos", "us", "tflops"),
            )
            if not avg.get("skip"):
                print(_fmt_normal_row(label, "", "--", avg))
                normal_avg_rows.append((label, avg))

        print("-" * len(hdr))
        _print_grouped_avgs(rows, lambda r: _tag_group(r[0]), _normal_avg_fn)
        print("=" * len(hdr))

        csv_path = f"fmha_perf{csv_tag}_{_gpu_short_name()}.csv"
        _write_normal_csv(csv_path, rows, normal_avg_rows)
        print(f"Results saved to: {csv_path}")

        extra_ok = True
        if extra_cases:
            print("=" * 130)
            print("Additional dense/varlen/cross-length cases: FlyDSL vs reference")
            print("=" * 130)
            xhdr = (
                f"{_EXTRA_HDR} | " f"{'Status':>6s} | {'MaxErr':>8s} {'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
            )
            varlen_rows = []
            for dtype_key in dtypes_to_test:
                dtype, dtype_str = dtype_map[dtype_key]
                for causal in causals_to_test:
                    ctag = "causal" if causal else "nocausal"
                    for case, hd in extra_run_cases:
                        nh = case["nh"]
                        nh_kv_eff = args.num_kv_heads if args.num_kv_heads is not None else case["nh_kv"]
                        kv_splits = case.get("kv_splits", 1)
                        kwargs = dict(case["kwargs"])
                        for kv_cache_layout, path in paged_kv_paths:
                            pre = _fmt_extra_prefix(
                                case["sq_label"],
                                case["skv_label"],
                                nh,
                                nh_kv_eff,
                                hd,
                                dtype_key,
                                ctag,
                                path=path,
                            )
                            print(f"{pre} ...", flush=True)
                            try:
                                precomputed_inputs = None
                                precomputed_ref = None
                                if args.block_table:
                                    precomputed_inputs, precomputed_ref, precompute_status = (
                                        _precompute_paged_kv_inputs_and_ref(
                                            batch=kwargs.get("batch", 1),
                                            seqlen_q=kwargs.get("seqlen_q"),
                                            seqlen_kv=kwargs.get("seqlen_kv"),
                                            varlen_seqlens_q=kwargs.get("varlen_seqlens_q"),
                                            varlen_seqlens_kv=kwargs.get("varlen_seqlens_kv"),
                                            num_heads=nh,
                                            head_dim=hd,
                                            num_kv_heads=nh_kv_eff,
                                            dtype=dtype,
                                            causal=causal,
                                            seed=args.seed,
                                            page_size=args.page_size,
                                            kv_cache_layout=kv_cache_layout or "linear",
                                            use_bias=args.bias,
                                        )
                                    )
                                    if precompute_status is not None:
                                        print(f"{pre} {'ERR' if 'err' in precompute_status else 'SKIP'}")
                                        varlen_rows.append(
                                            (
                                                case["sq_label"],
                                                case["skv_label"],
                                                nh,
                                                nh_kv_eff,
                                                hd,
                                                dtype_key,
                                                ctag,
                                                path,
                                                "ERROR" if "err" in precompute_status else "SKIP",
                                                precompute_status,
                                            )
                                        )
                                        if "err" in precompute_status:
                                            extra_ok = False
                                        continue

                                if dtype_str == "fp8":
                                    if args.block_table:
                                        raise ValueError("fp8 flash_attn does not support --block-table")
                                    r = run_fp8_config(
                                        kwargs.get("batch", 1),
                                        kwargs.get("seqlen_q", max(kwargs.get("varlen_seqlens_q", [seq_len]))),
                                        nh,
                                        hd,
                                        causal,
                                        warmup=args.warmup,
                                        iters=args.iters,
                                        seed=args.seed,
                                        verbose=False,
                                        num_kv_heads=nh_kv_eff,
                                        num_kv_splits=kv_splits,
                                    )
                                else:
                                    r = run_attn_config(
                                        nh,
                                        hd,
                                        dtype,
                                        causal,
                                        args.warmup,
                                        args.iters,
                                        num_kv_heads=nh_kv_eff,
                                        num_kv_splits=kv_splits,
                                        seed=args.seed,
                                        dtype_str=dtype_str,
                                        verbose=True,
                                        use_block_table=args.block_table,
                                        precomputed_ref=precomputed_ref,
                                        precomputed_inputs=precomputed_inputs,
                                        page_size=args.page_size,
                                        kv_cache_layout=kv_cache_layout or "linear",
                                        use_bias=args.bias,
                                        use_alibi=args.alibi,
                                        use_sink=args.sink,
                                        sink_share=args.sink_share,
                                        alibi_two_d=args.alibi_two_d,
                                        **kwargs,
                                    )
                            except Exception as e:
                                print(f"{pre} RAISED: {e}")
                                varlen_rows.append(
                                    (
                                        case["sq_label"],
                                        case["skv_label"],
                                        nh,
                                        nh_kv_eff,
                                        hd,
                                        dtype_key,
                                        ctag,
                                        path,
                                        "ERROR",
                                        {"err": str(e)},
                                    )
                                )
                                extra_ok = False
                                continue
                            if "err" in r:
                                print(f"{pre} ERR: {r['err']}")
                                varlen_rows.append(
                                    (
                                        case["sq_label"],
                                        case["skv_label"],
                                        nh,
                                        nh_kv_eff,
                                        hd,
                                        dtype_key,
                                        ctag,
                                        path,
                                        "ERROR",
                                        r,
                                    )
                                )
                                extra_ok = False
                                continue
                            if r.get("skip"):
                                print(f"{pre} SKIP")
                                varlen_rows.append(
                                    (
                                        case["sq_label"],
                                        case["skv_label"],
                                        nh,
                                        nh_kv_eff,
                                        hd,
                                        dtype_key,
                                        ctag,
                                        path,
                                        "SKIP",
                                        r,
                                    )
                                )
                                continue
                            if r.get("bench_err"):
                                print(f"{pre} BENCHERR: {r['bench_err']}")
                                status = "BENCHERR"
                                extra_ok = False
                            else:
                                passed = bool(r.get("passed", False))
                                status = "PASS" if passed else "FAIL"
                                extra_ok = extra_ok and passed
                            varlen_rows.append(
                                (
                                    case["sq_label"],
                                    case["skv_label"],
                                    nh,
                                    nh_kv_eff,
                                    hd,
                                    dtype_key,
                                    ctag,
                                    path,
                                    status,
                                    r,
                                )
                            )
            print("\n" + xhdr)
            print("  " + "-" * (len(xhdr) - 2))
            for sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, path, status, r in varlen_rows:
                print(_fmt_extra_normal_row(sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, status, r, path=path))
            print("  " + "-" * (len(xhdr) - 2))

            varlen_avg_rows = []

            def _extra_normal_avg(label, subset):
                avg = _avg_results(
                    [row[9] for row in subset],
                    keys=("max_err", "min_cos", "us", "tflops"),
                )
                avg_row = _fmt_extra_normal_avg_row(label, avg)
                if avg_row is not None:
                    print(avg_row)
                    varlen_avg_rows.append((label, avg))

            _print_grouped_avgs(varlen_rows, lambda r: (r[5], r[6]), _extra_normal_avg)
            print("=" * len(xhdr))
            varlen_csv_path = f"fmha_varlen_perf{csv_tag}_{_gpu_short_name()}.csv"
            _write_varlen_normal_csv(varlen_csv_path, varlen_rows, varlen_avg_rows)
            print(f"Varlen results saved to: {varlen_csv_path}")

        if all_passed and extra_ok:
            print("All tests PASSED")
        else:
            print("Some tests FAILED")
            sys.exit(1)


# ============================================================================
# Forward LSE (log-sum-exp) correctness tests
# ----------------------------------------------------------------------------
# Validates return_lse=True output against a float32 PyTorch reference across the
# dense, split-K combine and varlen paths (plus GQA, cross-attn, fully-masked rows).
# ============================================================================

# bf16 inputs accumulate the dot product in the kernel with a slightly different
# ordering than the fp32 reference, so allow a small absolute tolerance.
_ATOL_BF16 = 8e-3
_ATOL_F16 = 4e-3

# Split-K and varlen LSE use the gfx950 DUALWAVE_SWP kernel; dense LSE runs on the
# gfx942-compatible generic path too. (RDNA is skipped file-wide via CDNA_ONLY_TESTS.)
_requires_gfx950 = pytest.mark.skipif(
    not str(get_rocm_arch()).startswith("gfx950"),
    reason=f"requires gfx950 DUALWAVE_SWP, got {get_rocm_arch()!s}",
)


def _lse_sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def _reference_lse(q, k, causal, num_kv_heads):
    """float32 reference LSE for dense inputs.

    q: [B, Sq, H, D], k: [B, Skv, Hkv, D] -> lse: [B, H, Sq] (natural log, scale
    folded). Causal is bottom-right aligned: key j is visible to query i iff
    ``j <= i + (Skv - Sq)`` (matches the kernel's bottom-right convention).
    """
    B, Sq, H, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    sm = _lse_sm_scale(D)
    qf = q.float().permute(0, 2, 1, 3)  # B, H, Sq, D
    kf = k.float().permute(0, 2, 1, 3)  # B, Hkv, Skv, D
    kf = kf.repeat_interleave(H // Hkv, dim=1)  # B, H, Skv, D
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm
    if causal:
        qi = torch.arange(Sq, device=q.device).view(Sq, 1)
        ki = torch.arange(Skv, device=q.device).view(1, Skv)
        mask = (ki > (qi + (Skv - Sq))).view(1, 1, Sq, Skv)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.logsumexp(scores, dim=-1)  # B, H, Sq


def _assert_lse_matches(lse, lse_ref, atol):
    """Finite rows match within atol; -inf (fully-masked) rows match exactly."""
    lse = lse.float()
    lse_ref = lse_ref.float()
    finite = torch.isfinite(lse_ref)
    # Fully-masked rows: reference is -inf, kernel must be non-finite (i.e. -inf).
    if (~finite).any():
        assert bool(((~torch.isfinite(lse)) == (~finite)).all()), "masked (-inf) rows mismatch"
    if finite.any():
        diff = (lse[finite] - lse_ref[finite]).abs().max().item()
        assert diff <= atol, f"LSE max abs diff {diff:.3e} exceeds atol {atol:.3e}"


def _rand_lse(*shape, dtype, device="cuda"):
    return torch.randn(*shape, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "B,S,H,Hkv,D",
    [
        (2, 256, 8, 8, 128),  # MHA
        (2, 256, 8, 2, 128),  # GQA
        (1, 256, 4, 1, 128),  # MQA
        (2, 192, 4, 4, 64),  # D=64
        (3, 128, 6, 3, 128),  # GQA, odd batch
    ],
)
def test_lse_dense(dtype, causal, B, S, H, Hkv, D):
    torch.manual_seed(S + H + D + int(causal))
    q = _rand_lse(B, S, H, D, dtype=dtype)
    k = _rand_lse(B, S, Hkv, D, dtype=dtype)
    v = _rand_lse(B, S, Hkv, D, dtype=dtype)
    _, lse = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv, return_lse=True)
    torch.cuda.synchronize()
    assert lse.shape == (B, H, S)
    assert lse.dtype == torch.float32
    atol = _ATOL_BF16 if dtype == torch.bfloat16 else _ATOL_F16
    _assert_lse_matches(lse, _reference_lse(q, k, causal, Hkv), atol)


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_kv_splits", [2, 3])
@pytest.mark.parametrize("D", [64, 128])
def test_lse_splitk(causal, num_kv_splits, D):
    # Split-K requires seq_len >= 384, D in {64,128}, bf16/f16.
    B, S, H = 1, 512, 8
    torch.manual_seed(S + D + num_kv_splits + int(causal))
    dtype = torch.bfloat16
    q = _rand_lse(B, S, H, D, dtype=dtype)
    k = _rand_lse(B, S, H, D, dtype=dtype)
    v = _rand_lse(B, S, H, D, dtype=dtype)
    _, lse = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_splits=num_kv_splits, return_lse=True)
    torch.cuda.synchronize()
    assert lse.shape == (B, H, S)
    _assert_lse_matches(lse, _reference_lse(q, k, causal, H), _ATOL_BF16)


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_lse_varlen(causal):
    dtype = torch.bfloat16
    D, H, Hkv = 128, 4, 4
    seqs = [100, 200, 60]
    torch.manual_seed(sum(seqs) + int(causal))
    cu = torch.tensor([0, 100, 300, 360], dtype=torch.int32, device="cuda")
    max_seqlen_q = max(seqs)
    total = int(cu[-1].item())
    q = _rand_lse(total, H, D, dtype=dtype)
    k = _rand_lse(total, Hkv, D, dtype=dtype)
    v = _rand_lse(total, Hkv, D, dtype=dtype)
    _, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_seqlen_q,
        cross_seqlen=False,
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert lse.shape == (len(seqs), H, max_seqlen_q)
    for b, (s0, s1) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        n = s1 - s0
        qb = q[s0:s1].unsqueeze(0)  # 1, n, H, D
        kb = k[s0:s1].unsqueeze(0)
        ref = _reference_lse(qb, kb, causal, Hkv)[0]  # H, n
        # Only the valid [:, :n] region is defined; padded rows are ignored.
        _assert_lse_matches(lse[b, :, :n], ref, _ATOL_BF16)


# ── attention bias ───────────────────────────────────────────────────────────


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,S,H,Hkv,D", [(1, 512, 8, 8, 128), (2, 384, 8, 4, 64)])
def test_bias_dense(causal, B, S, H, Hkv, D):
    """Dense bias is [Sq, Skv], broadcast over batch and head."""
    dtype = torch.bfloat16
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    bias = torch.empty(S, S, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    out = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv, bias=bias)
    torch.cuda.synchronize()
    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal, bias=bias)
    _, _, passed = _acc_metric(out.float().reshape(-1), ref.float().reshape(-1), D)
    assert passed, f"biased output does not match the biased reference (B={B} S={S} causal={causal})"

    # The bias must actually change the result: an unbiased run must NOT match the
    # biased reference, otherwise a silently-dropped bias would pass the check above.
    out_nb = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv)
    torch.cuda.synchronize()
    assert (out_nb.float() - ref.float()).abs().max().item() > 1e-2, "bias had no effect on the output"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_bias_varlen(causal):
    """Varlen bias is packed [total_q, max_seqlen_kv]: global q rows, batch-local key columns."""
    dtype = torch.bfloat16
    D, H, Hkv = 128, 8, 4
    seqs = [512, 256, 384]
    setup_seed(DEFAULT_SEED)
    cu_list = [0]
    for s in seqs:
        cu_list.append(cu_list[-1] + s)
    total, max_s = cu_list[-1], max(seqs)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    q = torch.empty(total, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    bias = torch.empty(total, max_s, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=Hkv,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_s,
        max_seqlen_kv=max_s,
        cross_seqlen=False,
        bias=bias,
    )
    torch.cuda.synchronize()
    for b, n in enumerate(seqs):
        s0, s1 = cu_list[b], cu_list[b + 1]
        ref = pytorch_ref_attention(
            q[s0:s1].unsqueeze(0).float(),
            k[s0:s1].unsqueeze(0).float(),
            v[s0:s1].unsqueeze(0).float(),
            causal=causal,
            bias=bias[s0:s1, :n],
        ).squeeze(0)
        _, _, passed = _acc_metric(out[s0:s1].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"varlen batch {b} (seqlen {n}, causal={causal}) does not match the biased reference"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_cache_layout", ["linear", "vectorized"])
def test_bias_paged(causal, kv_cache_layout):
    """Paged bias is [Sq, max_seqlen_kv]: q rows, batch-local logical key columns.

    The block table only redirects the K/V fetch, so the bias column is still the
    logical KV position -- the same index the causal mask already uses.
    """
    dtype = torch.bfloat16
    B, Sq, H, Hkv, D = 2, 512, 8, 4, 128
    # Uniform KV lengths: the dense paged path forwards only max_seqlen_kv, so ragged
    # lengths are rejected outright (see test_bias_paged_rejects_ragged_seqlen_k).
    kv_lens = [Sq, Sq]
    max_kv = max(kv_lens)
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    kv_cache = _build_paged_kv_for_test(B, max_kv, 64, Hkv, D, kv_lens, dtype, "cuda", kv_cache_layout)
    bias = torch.empty(Sq, max_kv, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    paged_kw = dict(
        causal=causal,
        num_kv_heads=Hkv,
        max_seqlen_kv=max_kv,
        block_table=kv_cache["block_table"],
        seqlen_k=kv_cache["seqlen_k"],
        kv_cache_layout=kv_cache_layout,
    )
    out = flydsl_flash_attn_func(q, kv_cache["k_cache"], kv_cache["v_cache"], bias=bias, **paged_kw)
    torch.cuda.synchronize()

    for b, n in enumerate(kv_lens):
        kb, vb = _logical_kv_from_pages(
            kv_cache["k_cache"][_page_ids_for_batch(kv_cache, b)],
            kv_cache["v_cache"][_page_ids_for_batch(kv_cache, b)],
            kv_cache_layout,
            n,
        )
        ref = pytorch_ref_attention(
            q[b].unsqueeze(0).float(),
            kb.unsqueeze(0).float(),
            vb.unsqueeze(0).float(),
            causal=causal,
            bias=bias[:, :n],
        ).squeeze(0)
        _, _, passed = _acc_metric(out[b].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"paged batch {b} ({kv_cache_layout}, causal={causal}) does not match the biased reference"

    # A bias-free paged run must NOT match, so a silently-dropped bias cannot pass.
    out_nb = flydsl_flash_attn_func(q, kv_cache["k_cache"], kv_cache["v_cache"], **paged_kw)
    torch.cuda.synchronize()
    assert (out_nb.float() - out.float()).abs().max().item() > 1e-2, "bias had no effect on the paged output"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_bias_paged_rejects_ragged_seqlen_k(causal):
    """Dense paged bias rejects ragged per-batch seqlen_k instead of answering wrongly.

    The dense paged launch reduces seqlen_k to a single max_seqlen_kv and never
    forwards the per-batch lengths, so a shorter batch would attend KV slots it
    does not own and mask against the wrong bottom-right offset.
    """
    dtype = torch.bfloat16
    B, Sq, H, Hkv, D = 2, 512, 8, 4, 128
    kv_lens = [256, 512]
    max_kv = max(kv_lens)
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    ragged = _build_paged_kv_for_test(B, max_kv, 64, Hkv, D, kv_lens, dtype, "cuda", "linear")
    bias = torch.empty(Sq, max_kv, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    paged_kw = dict(causal=causal, num_kv_heads=Hkv, max_seqlen_kv=max_kv, kv_cache_layout="linear")

    with pytest.raises(NotImplementedError, match="uniform seqlen_k"):
        flydsl_flash_attn_func(
            q,
            ragged["k_cache"],
            ragged["v_cache"],
            bias=bias,
            block_table=ragged["block_table"],
            seqlen_k=ragged["seqlen_k"],
            **paged_kw,
        )

    # The guard is about raggedness alone: identical shapes with uniform lengths run.
    uniform = _build_paged_kv_for_test(B, max_kv, 64, Hkv, D, [max_kv] * B, dtype, "cuda", "linear")
    flydsl_flash_attn_func(
        q,
        uniform["k_cache"],
        uniform["v_cache"],
        bias=bias,
        block_table=uniform["block_table"],
        seqlen_k=uniform["seqlen_k"],
        **paged_kw,
    )
    torch.cuda.synchronize()


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_bias_paged_varlen_ragged_seqlen_k(causal):
    """The varlen paged path -- what the dense rejection points callers at -- is correct.

    cu_seqlens_kv carries the per-batch KV lengths into the kernel, so ragged
    lengths mask and bottom-right-align per batch instead of against one global max.
    """
    dtype = torch.bfloat16
    H, Hkv, D = 8, 4, 128
    sq, skv = [512, 512], [256, 512]
    setup_seed(DEFAULT_SEED)
    cu_q = torch.tensor([0, sq[0], sum(sq)], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, skv[0], sum(skv)], dtype=torch.int32, device="cuda")
    total_q, max_q, max_kv = sum(sq), max(sq), max(skv)
    q = torch.empty(total_q, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    kv_cache = _build_paged_kv_for_test(len(sq), max_kv, 64, Hkv, D, skv, dtype, "cuda", "linear")
    bias = torch.empty(total_q, max_kv, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    out = flydsl_flash_attn_func(
        q,
        kv_cache["k_cache"],
        kv_cache["v_cache"],
        causal=causal,
        num_kv_heads=Hkv,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max_q,
        max_seqlen_kv=max_kv,
        cross_seqlen=True,
        block_table=kv_cache["block_table"],
        seqlen_k=kv_cache["seqlen_k"],
        kv_cache_layout="linear",
        bias=bias,
    )
    torch.cuda.synchronize()

    for b, n in enumerate(skv):
        s0, s1 = int(cu_q[b]), int(cu_q[b + 1])
        kb, vb = _logical_kv_from_pages(
            kv_cache["k_cache"][_page_ids_for_batch(kv_cache, b)],
            kv_cache["v_cache"][_page_ids_for_batch(kv_cache, b)],
            "linear",
            n,
        )
        ref = pytorch_ref_attention_qkv_diff(
            q[s0:s1].unsqueeze(0).float(),
            kb.unsqueeze(0).float(),
            vb.unsqueeze(0).float(),
            causal=causal,
            bias=bias[s0:s1, :n],
        ).squeeze(0)
        _, _, passed = _acc_metric(out[s0:s1].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"varlen paged batch {b} (Sq={sq[b]}, Skv={n}, causal={causal}) does not match"


# ── attention bias: addressing limits ────────────────────────────────────────
#
# The kernel computes bias element offsets as `row * stride + column` in signed
# i32 and describes the bias with a 32-bit-num_records buffer descriptor. A bias
# past either limit is unrepresentable, so it must be rejected up front instead
# of silently reading the wrong rows.

# 2^31 elements at row 32768, and 4,295,098,368 bytes: over both limits at once.
_OVERSIZED_BIAS_SHAPE = (32769, 65536)
_OVERSIZED_BIAS_MATCH = "i32 bias element offsets"


def _unbacked_bias(rows, cols, dtype=torch.bfloat16):
    """A [rows, cols] bias with zero-stride storage: shape without the allocation."""
    return torch.zeros(1, 1, dtype=dtype, device="cuda").expand(rows, cols)


@pytest.mark.parametrize(
    "rows,cols,elem_size,expect",
    [
        # The i32 element offset is the binding limit for the 2-byte bias dtypes.
        (*_OVERSIZED_BIAS_SHAPE, 2, "i32 bias element offsets"),
        (BIAS_MAX_OFFSET_ELEMS + 1, 1, 2, "i32 bias element offsets"),
        (BIAS_MAX_OFFSET_ELEMS, 1, 2, None),  # exactly at the limit still fits
        (46340, 46340, 2, None),  # ~4 GiB, the largest square bias that fits
        (65536, 32768, 2, "i32 bias element offsets"),  # exactly 2^31 elements: one over
        # A 4-byte element trips the descriptor limit while the offset still fits.
        (BIAS_MAX_OFFSET_ELEMS, 1, 4, "bias buffer descriptor"),
        (BIAS_MAX_DESCRIPTOR_BYTES // 4, 1, 4, None),
    ],
)
def test_bias_addressing_error_limits(rows, cols, elem_size, expect):
    why = bias_addressing_error(rows * cols, elem_size)
    if expect is None:
        assert why is None, f"bias {rows}x{cols} ({elem_size}B) should fit, got: {why}"
    else:
        assert why is not None, f"bias {rows}x{cols} ({elem_size}B) should be rejected"
        assert expect in why, f"unexpected reason for {rows}x{cols} ({elem_size}B): {why}"


def test_bias_dense_rejects_unaddressable():
    dtype = torch.bfloat16
    B, S, H, D = 1, 128, 4, 128
    q = torch.zeros(B, S, H, D, dtype=dtype, device="cuda")
    bias = _unbacked_bias(*_OVERSIZED_BIAS_SHAPE, dtype=dtype)
    with pytest.raises(ValueError, match=_OVERSIZED_BIAS_MATCH):
        flydsl_flash_attn_func(q, q.clone(), q.clone(), causal=False, num_kv_heads=H, bias=bias)


def test_bias_varlen_rejects_unaddressable():
    dtype = torch.bfloat16
    seqs = [128, 128]
    total, max_s = sum(seqs), max(seqs)
    H, Hkv, D = 4, 4, 128
    cu = torch.tensor([0, seqs[0], total], dtype=torch.int32, device="cuda")
    q = torch.zeros(total, H, D, dtype=dtype, device="cuda")
    kv = torch.zeros(total, Hkv, D, dtype=dtype, device="cuda")
    bias = _unbacked_bias(*_OVERSIZED_BIAS_SHAPE, dtype=dtype)
    with pytest.raises(ValueError, match=_OVERSIZED_BIAS_MATCH):
        flydsl_flash_attn_func(
            q,
            kv,
            kv.clone(),
            causal=False,
            num_kv_heads=Hkv,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_s,
            max_seqlen_kv=max_s,
            cross_seqlen=False,
            bias=bias,
        )


@_requires_gfx950
def test_bias_varlen_self_attn_rejects_narrow_bias():
    """Varlen self-attention bounds bias columns by max_seqlen_q, not max_seqlen_kv.

    max_seqlen_kv is legitimately None when cross_seqlen=False, so a too-narrow
    bias used to pass validation: the kernel then indexes key column j with
    bias_stride0 = bias.shape[1], reading the following bias rows instead of failing.
    """
    dtype = torch.bfloat16
    seqs = [512, 384]
    total, max_s = sum(seqs), max(seqs)
    H, Hkv, D = 8, 4, 128
    setup_seed(DEFAULT_SEED)
    cu = torch.tensor([0, seqs[0], total], dtype=torch.int32, device="cuda")
    q = torch.empty(total, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    self_attn_kw = dict(
        causal=False,
        num_kv_heads=Hkv,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_s,
        cross_seqlen=False,
    )

    for cols in (1, max_s - 1):
        narrow = torch.zeros(total, cols, dtype=dtype, device="cuda")
        with pytest.raises(ValueError, match="self-attention KV maximum"):
            flydsl_flash_attn_func(q, k, v, bias=narrow, **self_attn_kw)

    # A bias exactly at the bound still runs, and matches the per-batch reference.
    bias = torch.empty(total, max_s, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    out = flydsl_flash_attn_func(q, k, v, bias=bias, **self_attn_kw)
    torch.cuda.synchronize()
    for b, n in enumerate(seqs):
        s0, s1 = int(cu[b]), int(cu[b + 1])
        ref = pytorch_ref_attention(
            q[s0:s1].unsqueeze(0).float(),
            k[s0:s1].unsqueeze(0).float(),
            v[s0:s1].unsqueeze(0).float(),
            causal=False,
            bias=bias[s0:s1, :n],
        ).squeeze(0)
        _, _, passed = _acc_metric(out[s0:s1].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"varlen self-attention batch {b} (seqlen {n}) does not match the biased reference"


def test_bias_paged_rejects_unaddressable():
    dtype = torch.bfloat16
    B, Sq, H, Hkv, D = 1, 128, 4, 4, 128
    q = torch.zeros(B, Sq, H, D, dtype=dtype, device="cuda")
    kv_cache = _build_paged_kv_for_test(B, Sq, 64, Hkv, D, [Sq], dtype, "cuda", "linear")
    bias = _unbacked_bias(*_OVERSIZED_BIAS_SHAPE, dtype=dtype)
    with pytest.raises(ValueError, match=_OVERSIZED_BIAS_MATCH):
        flydsl_flash_attn_func(
            q,
            kv_cache["k_cache"],
            kv_cache["v_cache"],
            causal=True,
            num_kv_heads=Hkv,
            max_seqlen_kv=Sq,
            block_table=kv_cache["block_table"],
            seqlen_k=kv_cache["seqlen_k"],
            kv_cache_layout="linear",
            bias=bias,
        )


def test_precompute_paged_bias_reaches_inputs_and_reference():
    """`--block-table --bias` must generate a bias AND fold it into the reference.

    The helper used to ignore use_bias, so the paged benchmark ran unbiased and
    compared against an unbiased reference: a PASS that measured nothing.
    """
    kw = dict(
        batch=1,
        seqlen_q=256,
        seqlen_kv=None,
        varlen_seqlens_q=None,
        varlen_seqlens_kv=None,
        num_heads=4,
        head_dim=128,
        num_kv_heads=4,
        dtype=torch.bfloat16,
        causal=False,
        seed=DEFAULT_SEED,
        page_size=64,
        kv_cache_layout="linear",
    )
    biased_inputs, biased_ref, biased_status = _precompute_paged_kv_inputs_and_ref(**kw, use_bias=True)
    plain_inputs, plain_ref, plain_status = _precompute_paged_kv_inputs_and_ref(**kw)
    assert biased_status is None and plain_status is None
    assert biased_inputs["bias"] is not None, "use_bias=True must generate a paged bias"
    assert plain_inputs["bias"] is None, "use_bias defaults to no bias"

    # The bias is drawn after Q/K/V, so the same seed leaves the inputs identical
    # and any reference difference is the bias alone.
    for key in ("q_t", "k_t", "v_t"):
        assert torch.equal(biased_inputs[key], plain_inputs[key]), f"{key} must not depend on use_bias"
    assert (
        biased_ref.float() - plain_ref.float()
    ).abs().max().item() > 1e-2, "the paged reference must fold in the generated bias"


@_requires_gfx950
def test_bias_launcher_rejects_unaddressable():
    """The kernel launcher guards too, for callers that bypass flydsl_flash_attn_func.

    The guard also has to fire before the launcher's ``bias.contiguous()``, which
    would otherwise materialize gigabytes on the way to a guaranteed failure.
    """
    dtype = torch.bfloat16
    B, S, H, D = 1, 384, 4, 128
    launch = build_flash_attn_dualwave_swp_module(num_heads=H, head_dim=D, causal=False, has_bias=True)
    q, k, v, o = (torch.zeros(B, S, H, D, dtype=dtype, device="cuda") for _ in range(4))
    bias = _unbacked_bias(*_OVERSIZED_BIAS_SHAPE, dtype=dtype)
    free_before = torch.cuda.mem_get_info()[0]
    with pytest.raises(ValueError, match=_OVERSIZED_BIAS_MATCH):
        launch(q, k, v, o, B, S, bias=bias)
    assert torch.cuda.mem_get_info()[0] > free_before - 2**30, "rejected bias must not be materialized"


# ── ALiBi ────────────────────────────────────────────────────────────────────


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("two_d", [False, True])
@pytest.mark.parametrize("B,S,H,Hkv,D", [(2, 512, 8, 8, 128), (1, 384, 8, 4, 64)])
def test_alibi_dense(causal, two_d, B, S, H, Hkv, D):
    """score += -slope * |i + Skv - Sq - j|; slopes are [H] or [B, H] (alibi_stride_b)."""
    dtype = torch.bfloat16
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    slopes = make_alibi_slopes(B, H, two_d)
    assert slopes.shape == ((B, H) if two_d else (H,))

    out = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv, alibi_slopes=slopes)
    torch.cuda.synchronize()
    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal, alibi_slopes=slopes)
    _, _, passed = _acc_metric(out.float().reshape(-1), ref.float().reshape(-1), D)
    assert passed, f"ALiBi output does not match the reference (B={B} S={S} two_d={two_d} causal={causal})"

    # Without slopes the result must differ, else a dropped ALiBi term would pass above.
    out_nb = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv)
    torch.cuda.synchronize()
    assert (out_nb.float() - ref.float()).abs().max().item() > 1e-2, "ALiBi had no effect on the output"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_alibi_varlen(causal):
    """ALiBi positions are within-sequence: no packed-token base, per-batch lengths."""
    dtype = torch.bfloat16
    D, H, Hkv = 128, 8, 4
    seqs = [512, 256, 384]
    setup_seed(DEFAULT_SEED)
    cu_list = [0]
    for s in seqs:
        cu_list.append(cu_list[-1] + s)
    total, max_s = cu_list[-1], max(seqs)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    q = torch.empty(total, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    slopes = make_alibi_slopes(len(seqs), H, two_d=True)

    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=Hkv,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_s,
        max_seqlen_kv=max_s,
        cross_seqlen=False,
        alibi_slopes=slopes,
    )
    torch.cuda.synchronize()
    for b, n in enumerate(seqs):
        s0, s1 = cu_list[b], cu_list[b + 1]
        ref = pytorch_ref_attention(
            q[s0:s1].unsqueeze(0).float(),
            k[s0:s1].unsqueeze(0).float(),
            v[s0:s1].unsqueeze(0).float(),
            causal=causal,
            alibi_slopes=slopes[b],
        ).squeeze(0)
        _, _, passed = _acc_metric(out[s0:s1].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"varlen batch {b} (seqlen {n}, causal={causal}) does not match the ALiBi reference"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_alibi_and_bias_combined(causal):
    """ALiBi and bias are independent score terms and must both land."""
    dtype = torch.bfloat16
    B, S, H, D = 1, 512, 8, 128
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    bias = torch.empty(S, S, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    slopes = make_alibi_slopes(B, H)

    out = flydsl_flash_attn_func(q, k, v, causal=causal, bias=bias, alibi_slopes=slopes)
    torch.cuda.synchronize()
    qf, kf, vf = q.float(), k.float(), v.float()
    ref_both = pytorch_ref_attention(qf, kf, vf, causal=causal, bias=bias, alibi_slopes=slopes)
    _, _, passed = _acc_metric(out.float().reshape(-1), ref_both.float().reshape(-1), D)
    assert passed, "combined bias+ALiBi output does not match the combined reference"

    # Neither term alone explains the output.
    ref_alibi = pytorch_ref_attention(qf, kf, vf, causal=causal, alibi_slopes=slopes)
    ref_bias = pytorch_ref_attention(qf, kf, vf, causal=causal, bias=bias)
    assert (out.float() - ref_alibi.float()).abs().max().item() > 1e-2, "bias term missing"
    assert (out.float() - ref_bias.float()).abs().max().item() > 1e-2, "ALiBi term missing"


def test_lse_fully_masked_rows():
    """Cross-attention causal with Skv < Sq: leading query rows see no keys -> -inf."""
    dtype = torch.bfloat16
    B, Sq, Skv, H, D = 2, 128, 32, 4, 128
    torch.manual_seed(Sq + Skv)
    q = _rand_lse(B, Sq, H, D, dtype=dtype)
    k = _rand_lse(B, Skv, H, D, dtype=dtype)
    v = _rand_lse(B, Skv, H, D, dtype=dtype)
    _, lse = flydsl_flash_attn_func(q, k, v, causal=True, return_lse=True)
    torch.cuda.synchronize()
    ref = _reference_lse(q, k, True, H)
    assert (~torch.isfinite(ref)).any(), "test setup should produce fully-masked rows"
    _assert_lse_matches(lse, ref, _ATOL_BF16)


def test_return_lse_false_returns_only_out():
    """Backwards-compat: default return_lse=False returns a bare tensor."""
    dtype = torch.bfloat16
    q = _rand_lse(2, 128, 4, 128, dtype=dtype)
    k = _rand_lse(2, 128, 4, 128, dtype=dtype)
    v = _rand_lse(2, 128, 4, 128, dtype=dtype)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape


def test_return_lse_rejects_fp8():
    dtype = torch.bfloat16
    q = _rand_lse(2, 128, 4, 128, dtype=dtype)
    with pytest.raises(NotImplementedError):
        flydsl_flash_attn_func(
            q.to(torch.float8_e4m3fn),
            q.to(torch.float8_e4m3fn),
            q.to(torch.float8_e4m3fn),
            causal=False,
            q_descale=torch.ones(1, device="cuda"),
            k_descale=torch.ones(1, device="cuda"),
            v_descale=torch.ones(1, device="cuda"),
            return_lse=True,
        )


@_requires_gfx950
@pytest.mark.parametrize("H", [8, 16, 32, 64])
def test_xcd_swizzle_is_bit_identical(H):
    """The head-slow remap must not change a single bit of the output.

    It only re-derives (head, q_block) from the same linear workgroup id, so it
    is bijective by construction -- but a mistake in the derivation would show
    up as a permuted or partially-recomputed output rather than as an error, so
    this pins it. S clears the auto-dispatch threshold (num_q_blocks >= 64 at
    BLOCK_M=256) so both settings run on the shapes the remap targets.
    """
    S = 64 * 256
    dtype = torch.bfloat16
    torch.manual_seed(H)
    q = _rand_lse(1, S, H, 128, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)

    def run(flag):
        return flydsl_flash_attn_func(q, k, v, causal=False, dualwave_swp_xcd_swizzle=flag).clone()

    off, on = run(False), run(True)
    torch.cuda.synchronize()
    assert torch.equal(off, on)


@_requires_gfx950
@pytest.mark.parametrize("xcd_swizzle", [None, True])
def test_xcd_swizzle_heads_not_multiple_of_xcd(xcd_swizzle):
    """H % 8 != 0 must fall back rather than mis-map, however the flag is set.

    The remap divides the linear workgroup id by the q-block count to recover
    the head, which only lands each head on one XCD when the head count divides
    evenly into the 8 XCDs. Two guards enforce that: the dispatch condition
    below auto-selects against it, and _init_dualwave_thread_mapping re-checks
    NUM_HEADS_Q % NUM_XCD_GFX950 independently -- so forcing the flag on is safe
    and simply does not engage the remap. Both paths are checked here.
    """
    S, H = 64 * 256, 12
    dtype = torch.bfloat16
    torch.manual_seed(H)
    q = _rand_lse(1, S, H, 128, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)

    out = flydsl_flash_attn_func(q, k, v, causal=False, dualwave_swp_xcd_swizzle=xcd_swizzle)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), ref, atol=_ATOL_BF16, rtol=0)


if __name__ == "__main__":
    main()


# ── attention sink ───────────────────────────────────────────────────────────


def _sink_for(q, k, causal, share=DEFAULT_SINK_SHARE):
    return calibrate_sink(*_rows_logsumexp(q, k, causal), share)


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("share", [0.25, 0.9])
@pytest.mark.parametrize("B,S,H,Hkv,D", [(1, 512, 8, 8, 128), (2, 384, 8, 4, 64)])
def test_sink_dense(causal, share, B, S, H, Hkv, D):
    """One extra softmax denominator logit per head, with no matching V row."""
    dtype = torch.bfloat16
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    sink = _sink_for(q, k, causal, share)
    assert sink.shape == (H,) and sink.dtype == torch.float32

    out = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv, sink=sink)
    torch.cuda.synchronize()
    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal, sink=sink)
    _, _, passed = _acc_metric(out.float().reshape(-1), ref.float().reshape(-1), D)
    assert passed, f"sink output does not match the reference (B={B} S={S} share={share} causal={causal})"

    # Calibration is what makes this test meaningful: with the sink dropped the
    # result must visibly differ. An uncalibrated sink near 0 would not.
    out_ns = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=Hkv)
    torch.cuda.synchronize()
    assert (out_ns.float() - ref.float()).abs().max().item() > 1e-2, "sink had no effect on the output"


@_requires_gfx950
@pytest.mark.parametrize("causal", [False, True])
def test_sink_varlen(causal):
    dtype = torch.bfloat16
    D, H, Hkv = 128, 8, 4
    seqs = [512, 256, 384]
    setup_seed(DEFAULT_SEED)
    cu_list = [0]
    for s in seqs:
        cu_list.append(cu_list[-1] + s)
    total, max_s = cu_list[-1], max(seqs)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    q = torch.empty(total, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(total, Hkv, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    # One [H] table shared by every sequence, calibrated over all their rows.
    tot, cnt = torch.zeros(H, dtype=torch.float32, device="cuda"), 0
    for b in range(len(seqs)):
        s0, s1 = cu_list[b], cu_list[b + 1]
        sl, n = _rows_logsumexp(q[s0:s1].unsqueeze(0), k[s0:s1].unsqueeze(0), causal)
        tot += sl
        cnt += n
    sink = calibrate_sink(tot, cnt, DEFAULT_SINK_SHARE)

    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=Hkv,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_s,
        max_seqlen_kv=max_s,
        cross_seqlen=False,
        sink=sink,
    )
    torch.cuda.synchronize()
    for b, n in enumerate(seqs):
        s0, s1 = cu_list[b], cu_list[b + 1]
        ref = pytorch_ref_attention(
            q[s0:s1].unsqueeze(0).float(),
            k[s0:s1].unsqueeze(0).float(),
            v[s0:s1].unsqueeze(0).float(),
            causal=causal,
            sink=sink,
        ).squeeze(0)
        _, _, passed = _acc_metric(out[s0:s1].float().reshape(-1), ref.float().reshape(-1), D)
        assert passed, f"varlen batch {b} (seqlen {n}, causal={causal}) does not match the sink reference"


@_requires_gfx950
@pytest.mark.parametrize("num_kv_splits", [2, 3, 4])
def test_sink_splitk_counted_once(num_kv_splits):
    """Split-K writes sink-free partials and folds the sink in once, in the combine.

    LSE is the sharp signal: it is the log denominator, so a sink counted
    num_kv_splits times (or zero times) shows up directly instead of being
    normalized away as it is in O.
    """
    dtype = torch.bfloat16
    B, S, H, D = 1, 2048, 8, 128
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    sink = _sink_for(q, k, True)

    out1, lse1 = flydsl_flash_attn_func(q, k, v, causal=True, sink=sink, return_lse=True)
    outk, lsek = flydsl_flash_attn_func(q, k, v, causal=True, sink=sink, num_kv_splits=num_kv_splits, return_lse=True)
    torch.cuda.synchronize()
    # Split-K must agree with the single-split result it is meant to reproduce.
    assert (lsek - lse1).abs().max().item() < 2e-2, f"split-K LSE diverges at {num_kv_splits} splits"
    _, _, passed = _acc_metric(outk.float().reshape(-1), out1.float().reshape(-1), D)
    assert passed, f"split-K output diverges at {num_kv_splits} splits"

    # A sink counted once per split would shift LSE by ~ln(num_kv_splits); assert
    # we are nowhere near that, so the test cannot pass on a double-count.
    assert (lsek - lse1).abs().max().item() < 0.5 * math.log(num_kv_splits)


@_requires_gfx950
@pytest.mark.parametrize("Sq,Skv", [(512, 128), (512, 160)])
def test_sink_lse_cross_attn_skipped_blocks(Sq, Skv):
    """Causal cross-attention with Skv < Sq skips whole q blocks that see no key.

    A skipped block never reaches the main body's fold_sink, so the skip path has
    to write those rows' LSE itself: it is the per-head sink, not -inf and not
    whatever the caller's output buffer happened to hold. Skv=160 also puts some
    all-masked rows inside an active block, covering both paths at once.
    """
    dtype = torch.bfloat16
    B, H, D = 2, 8, 128
    setup_seed(DEFAULT_SEED)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    sink = _sink_for(q, k, True)

    out, lse = flydsl_flash_attn_func(q, k, v, causal=True, sink=sink, return_lse=True)
    torch.cuda.synchronize()

    # Sink-inclusive LSE = ln(exp(LSE_no_sink) + exp(sink)); -inf rows collapse to sink.
    lse_ref_ns = _reference_lse(q, k, True, H)  # B, H, Sq
    assert bool((~torch.isfinite(lse_ref_ns)).any()), "test setup should produce fully-masked q rows"
    lse_ref = torch.logaddexp(lse_ref_ns, sink.view(1, H, 1).expand_as(lse_ref_ns))
    diff = (lse.float() - lse_ref).abs().max().item()
    assert diff <= _ATOL_BF16, f"sink-inclusive LSE max abs diff {diff:.3e} exceeds atol {_ATOL_BF16:.3e}"

    # The all-masked rows are the regression: their whole denominator is the sink.
    n_masked = Sq - Skv
    masked_lse = lse[:, :, :n_masked].float()
    assert (masked_lse - sink.view(1, H, 1)).abs().max().item() <= 1e-4, "all-masked rows must carry the sink LSE"
    assert out[:, :n_masked].abs().max().item() == 0.0, "all-masked rows must have zero output"
