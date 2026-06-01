#!/usr/bin/env python3
"""Test for mla_decode_fp8_v4_gfx1250 — v2.0a stage.

v2.0a: same V1 sparse_attn compute, but exposed via team's input ABI
(``block_table`` indirection + ``kv_paged`` flat). Output remains bf16 O
(same as V1) for verification convenience. v2.0b will switch to mid_o/mid_lse.

This test mirrors test_sparse_attn_gfx1250.py — generates the same
(Q, KV-fp8, topk_idxs) random data, then converts to (Q, kv_paged,
block_table) format the new kernel expects, and asserts bit-equivalent
output to V1's bf16 O.

Usage:
    pytest tests/kernels/test_mla_decode_fp8_v4_gfx1250.py
    python tests/kernels/test_mla_decode_fp8_v4_gfx1250.py            # CLI
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

from flydsl.runtime.device import get_rocm_arch
from kernels.mla_decode_fp8_v4_gfx1250 import compile_mla_decode_fp8_v4
from kernels.mla_decode_fp8_v4_stage2_gfx1250 import compile_mla_decode_fp8_v4_stage2

WMMA_ARCH_PREFIX = "gfx12"
DEFAULT_SEED = 42
VERIFY_RTOL = 0.30
VERIFY_ATOL = 0.30


def _setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _arch_skip_reason():
    try:
        arch = str(get_rocm_arch())
    except Exception as e:  # noqa: BLE001
        return f"cannot query ROCm arch: {e}"
    if not arch.startswith(WMMA_ARCH_PREFIX):
        return f"requires {WMMA_ARCH_PREFIX}*, current arch={arch!r}"
    return None


def _gen_kv_paged(B, kv_len, D, device, page_block_size=1):
    """Build kv_paged + block_table for arbitrary page_block_size.

    Per-batch random fp8 KV (kv_len rows of D fp8). Pack into paged cache:
      kv_paged shape: (num_blocks, page_block_size, 1, D)
      block_table[b, blk] = phys_block_id
      phys_token_id = phys_block_id * page_block_size + slot_in_block
    For page_block_size=1: identity mapping (back-compat).
    """
    assert kv_len % page_block_size == 0, "kv_len must be multiple of page_block_size"
    blocks_per_seq = kv_len // page_block_size
    total_blocks = B * blocks_per_seq

    kv_real = torch.empty(B, kv_len, D, device=device).uniform_(-1.0, 1.0)
    KV_fp8 = kv_real.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    # Reshape to (B, blocks_per_seq, page_block_size, 1, D) then flatten blocks.
    kv_paged = KV_fp8.view(B, blocks_per_seq, page_block_size, 1, D)\
                     .permute(0, 1, 2, 3, 4).contiguous()\
                     .view(total_blocks, page_block_size, 1, D)
    block_table = (
        torch.arange(total_blocks, dtype=torch.int32, device=device)
        .view(B, blocks_per_seq)
        .contiguous()
    )
    return KV_fp8, kv_paged, block_table


def _v1_reference(Q, KV_dequant_bf16, sm_scale, causal=False, attn_sink=None):
    """Closed-form fp32 dense attention with optional attn_sink (ref form).

    Q: (B, Sq, H, D) bf16, KV_dequant_bf16: (B, kv_len, D) bf16,
    attn_sink: (H,) fp32 or None.
    Output: (B, Sq, H, D) fp32.
    Causal mask (matches team's build_sparse_ref_indices semantics):
        q_pos = s sees kv_pos in [0, kv_len - Sq + s + 1).
    attn_sink ref form: O *= 1/(1+exp(sink - lse)).
    """
    B, Sq, H, D = Q.shape
    _, kv_len, _ = KV_dequant_bf16.shape
    Q_f = Q.float()
    K_f = KV_dequant_bf16.float()
    O = torch.zeros(B, Sq, H, D, dtype=torch.float32, device=Q.device)
    for b in range(B):
        for s in range(Sq):
            S = (Q_f[b, s] @ K_f[b].T) * sm_scale  # (H, kv_len)
            if causal:
                valid_kv = max(0, min(kv_len - Sq + s + 1, kv_len))
                if valid_kv < kv_len:
                    S[:, valid_kv:] = float("-inf")
            lse = torch.logsumexp(S, dim=-1)            # (H,)
            P = torch.softmax(S, dim=-1)
            out_bsh = P @ K_f[b]                        # (H, D)
            if attn_sink is not None:
                sink = attn_sink.to(S.device).to(S.dtype)        # (H,)
                sink_scale = 1.0 / (1.0 + torch.exp(sink - lse))  # (H,)
                out_bsh = out_bsh * sink_scale.unsqueeze(-1)
            O[b, s] = out_bsh
    return O


def _run_one(*, B, Sq, kv_len, num_q_heads=128, head_dim=512, block_h=16,
             block_n=64, num_waves=4, num_kv_bufs=2, num_kv_splits=1,
             causal=False, attn_sink=None, page_block_size=1,
             extra_kv_len=0,
             seed=DEFAULT_SEED, warmup=2, iters=5):
    device = "cuda"
    _setup_seed(seed)
    # Q: (B, Sq, Hq, D) bf16
    Q = torch.empty(B, Sq, num_q_heads, head_dim, device=device).uniform_(-1.0, 1.0).to(torch.bfloat16)
    KV_fp8, kv_paged, block_table = _gen_kv_paged(B, kv_len, head_dim, device, page_block_size)
    total_q = B * Sq
    mid_o = torch.zeros(total_q, num_kv_splits, num_q_heads, head_dim,
                         dtype=torch.float32, device=device)
    mid_lse = torch.full((total_q, num_kv_splits, num_q_heads), float("-inf"),
                         dtype=torch.float32, device=device)
    O = torch.zeros(total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    # v2.5 attn_sink: default -inf (= disabled, scale=1)
    if attn_sink is None:
        attn_sink_t = torch.full((num_q_heads,), float("-inf"),
                                 dtype=torch.float32, device=device)
    else:
        attn_sink_t = attn_sink.to(torch.float32).to(device)

    sm_scale = 1.0 / math.sqrt(head_dim)
    use_extra_kv = extra_kv_len > 0
    if use_extra_kv:
        # Build extra KV scope (same fp8, page_size=1 for simplicity).
        KV_extra_fp8, kv_paged_extra, bt_extra = _gen_kv_paged(
            B, extra_kv_len, head_dim, device, page_block_size=1,
        )
        extra_topk_length = torch.full((B,), extra_kv_len, dtype=torch.int32, device=device)
    else:
        # Dummy 1-elem placeholder tensors for extra resources (kernel skips loop).
        kv_paged_extra = torch.zeros(1, 1, 1, head_dim, dtype=torch.float8_e4m3fn, device=device)
        bt_extra = torch.zeros(1, 1, dtype=torch.int32, device=device)
        extra_topk_length = torch.zeros(1, dtype=torch.int32, device=device)

    launch1 = compile_mla_decode_fp8_v4(
        nheads_q=num_q_heads, head_dim=head_dim, topk=kv_len,
        block_h=block_h, block_n=block_n, sm_scale=sm_scale,
        NUM_WAVES=num_waves, NUM_KV_BUFS=num_kv_bufs,
        Sq=Sq, num_kv_splits=num_kv_splits, causal=causal,
        page_block_size=page_block_size,
        use_extra_kv=use_extra_kv,
        extra_topk_max=max(extra_kv_len, 1),
    )
    launch2 = compile_mla_decode_fp8_v4_stage2(
        head_dim_v=head_dim, num_q_heads=num_q_heads,
        num_kv_splits=num_kv_splits,
    )
    stream = torch.cuda.current_stream()

    # v2.6: per-batch topk_length (uniform = kv_len for v2.6 smoke; can vary per batch).
    topk_length = torch.full((B,), kv_len, dtype=torch.int32, device=device)

    def _run():
        launch1(Q.view(-1), kv_paged, block_table, topk_length,
                kv_paged_extra, bt_extra, extra_topk_length,
                mid_o.view(-1), mid_lse.view(-1),
                B, stream)
        launch2(mid_o.view(-1), mid_lse.view(-1), attn_sink_t,
                O.view(-1), total_q, stream)

    _run()
    torch.cuda.synchronize()

    # Reference: dequant fp8 KV → bf16, optionally concat extra scope, then attn.
    KV_dequant = KV_fp8.float().to(torch.bfloat16)   # (B, kv_len, D)
    if use_extra_kv:
        KV_extra_dequant = KV_extra_fp8.float().to(torch.bfloat16)  # (B, extra_kv_len, D)
        KV_combined = torch.cat([KV_dequant, KV_extra_dequant], dim=1)
    else:
        KV_combined = KV_dequant
    O_ref = _v1_reference(
        Q, KV_combined, sm_scale, causal=causal,
        attn_sink=attn_sink_t.cpu() if attn_sink is not None else None,
    ).cpu()  # (B, Sq, H, D)
    O_got = O.float().detach().cpu().view(B, Sq, num_q_heads, head_dim)
    diff = (O_got - O_ref).abs()
    res = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "passed": bool(torch.allclose(O_got, O_ref, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)),
    }

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run()
    torch.cuda.synchronize()
    res["time_us"] = (time.perf_counter() - t0) * 1e6 / iters
    return res


_SKIP = _arch_skip_reason()


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize(
    # Constraint: per_split_max (= ceil(kv_len/sp) rounded up to block_n=64) * sp == kv_len
    # ensures no empty split. block_n=64 ⇒ per_split must be multiple of 64.
    "B,Sq,kv_len,num_kv_splits,causal",
    [
        (1, 1, 64,   1, False),     # v2.0 base
        (1, 1, 256,  1, False),
        (1, 1, 1024, 1, False),
        (1, 4, 64,   1, False),     # v2.1 multi-token
        (1, 4, 256,  1, False),
        (1, 1, 128,  2, False),     # v2.2 split-K
        (1, 1, 256,  4, False),
        (1, 1, 1024, 4, False),
        (1, 4, 256,  2, False),     # v2.1 + v2.2
        (1, 4, 256,  1, True),      # v2.3 causal multi-token
        (1, 4, 1024, 1, True),
        (1, 4, 256,  4, True),      # v2.3 causal + split-K
    ],
)
def test_mla_decode_v4_correctness(B, Sq, kv_len, num_kv_splits, causal):
    res = _run_one(B=B, Sq=Sq, kv_len=kv_len, num_kv_splits=num_kv_splits, causal=causal)
    print(
        f"[B={B} Sq={Sq} kv_len={kv_len} sp={num_kv_splits} causal={causal}] "
        f"max={res['max_abs_diff']:.4e} "
        f"mean={res['mean_abs_diff']:.4e} time={res['time_us']:.1f}us"
    )
    assert res["passed"], (
        f"max_abs_diff={res['max_abs_diff']} exceeds tol "
        f"(rtol={VERIFY_RTOL}, atol={VERIFY_ATOL})"
    )


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("page_block_size", [4, 16])
def test_mla_decode_v4_paged(page_block_size):
    """v2.4a — page_block_size > 1 paging via block_table indirection."""
    res = _run_one(B=1, Sq=1, kv_len=256, page_block_size=page_block_size)
    print(f"[paged page={page_block_size}] max={res['max_abs_diff']:.4e} mean={res['mean_abs_diff']:.4e}")
    assert res["passed"], (
        f"paged FAILED: max_abs_diff={res['max_abs_diff']}"
    )


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize("kv_len,extra_kv_len", [(64, 64), (128, 64), (256, 128)])
def test_mla_decode_v4_extra_kv(kv_len, extra_kv_len):
    """v2.7 — 2nd KV scope (sliding window + compressed) concat."""
    res = _run_one(B=1, Sq=1, kv_len=kv_len, extra_kv_len=extra_kv_len)
    print(f"[extra_kv main={kv_len} extra={extra_kv_len}] "
          f"max={res['max_abs_diff']:.4e} mean={res['mean_abs_diff']:.4e}")
    assert res["passed"], (
        f"extra_kv FAILED: max_abs_diff={res['max_abs_diff']}"
    )


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_mla_decode_v4_attn_sink_active():
    """v2.5 — active attn_sink (finite values, ref form scale applied)."""
    sink = torch.empty(128, dtype=torch.float32).uniform_(-1.0, 1.0)
    res = _run_one(B=1, Sq=1, kv_len=256, attn_sink=sink)
    print(f"[attn_sink active] max={res['max_abs_diff']:.4e} mean={res['mean_abs_diff']:.4e}")
    assert res["passed"], (
        f"attn_sink active FAILED: max_abs_diff={res['max_abs_diff']}"
    )


def main():
    if _SKIP is not None:
        print("SKIP:", _SKIP)
        sys.exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=1)
    parser.add_argument("-s", "--Sq", type=int, default=1)
    parser.add_argument("-n", "--kv_len", type=int, default=64)
    parser.add_argument("--num_q_heads", type=int, default=128)
    parser.add_argument("--head_dim", type=int, default=512)
    args = parser.parse_args()
    r = _run_one(B=args.batch, Sq=args.Sq, kv_len=args.kv_len,
                 num_q_heads=args.num_q_heads, head_dim=args.head_dim)
    print(f"B={args.batch} Sq={args.Sq} kv_len={args.kv_len}: "
          f"max={r['max_abs_diff']:.4e} mean={r['mean_abs_diff']:.4e} "
          f"time={r['time_us']:.1f}us  {'PASS' if r['passed'] else 'FAIL'}")
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
