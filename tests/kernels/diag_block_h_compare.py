#!/usr/bin/env python3
"""Diagnostic script: validate block_h=16/32/64/128 precision against fp32 reference.

Usage:
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python tests/kernels/diag_block_h_compare.py

Passes if max_diff < 0.05 between any block_h and the fp32 reference.

Test matrix (per task requirement):
  case1:  B=1 kv_len=320  splits=1  block_h=16   (regression check)
  case2:  B=1 kv_len=320  splits=1  block_h=32   (bug-affected)
  case3:  B=1 kv_len=320  splits=1  block_h=64   (primary fix target)
  case4:  B=2 kv_len=640  splits=5  block_h=64   (multi-split)
  case5:  B=1 kv_len=1024 splits=1  block_h=64   (longer kv)
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import numpy as np
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available")
    sys.exit(1)

from kernels.mla_decode_fp8_v4_gfx1250 import compile_mla_decode_fp8_v4
from kernels.mla_decode_fp8_v4_stage2_gfx1250 import compile_mla_decode_fp8_v4_stage2

SEED = 42
PASS_TOL = 0.05   # max_diff vs fp32 ref < this => PASS
DEVICE = "cuda"


def setup_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_kv_paged(B, kv_len, D, page_block_size=1):
    assert kv_len % page_block_size == 0
    blocks_per_seq = kv_len // page_block_size
    total_blocks = B * blocks_per_seq
    kv_real = torch.empty(B, kv_len, D, device=DEVICE).uniform_(-1.0, 1.0)
    KV_fp8 = kv_real.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    kv_paged = (
        KV_fp8.view(B, blocks_per_seq, page_block_size, 1, D)
        .contiguous()
        .view(total_blocks, page_block_size, 1, D)
    )
    block_table = (
        torch.arange(total_blocks, dtype=torch.int32, device=DEVICE)
        .view(B, blocks_per_seq)
        .contiguous()
    )
    return KV_fp8, kv_paged, block_table


def v1_reference(Q, KV_dequant_bf16, sm_scale, causal=False):
    """Dense fp32 attention reference (V==K, MLA style)."""
    B, Sq, H, D = Q.shape
    _, kv_len, _ = KV_dequant_bf16.shape
    Q_f = Q.float()
    K_f = KV_dequant_bf16.float()
    O = torch.zeros(B, Sq, H, D, dtype=torch.float32, device=Q.device)
    for b in range(B):
        for s in range(Sq):
            S = (Q_f[b, s] @ K_f[b].T) * sm_scale
            if causal:
                valid_kv = max(0, min(kv_len - Sq + s + 1, kv_len))
                if valid_kv < kv_len:
                    S[:, valid_kv:] = float("-inf")
            P = torch.softmax(S, dim=-1)
            O[b, s] = P @ K_f[b]
    return O


def run_kernel(block_h, B, Sq, kv_len, Hq, D, splits,
               Q, KV_fp8, kv_paged, block_table, sm_scale, causal=False,
               num_kv_bufs=2):
    """Run kernel with given block_h; return O as float32 cpu tensor (B,Sq,Hq,D)."""
    total_q = B * Sq
    mid_o = torch.zeros(total_q, splits, Hq, D, dtype=torch.float32, device=DEVICE)
    mid_lse = torch.full((total_q, splits, Hq), float("-inf"),
                         dtype=torch.float32, device=DEVICE)
    O = torch.zeros(total_q, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    attn_sink_t = torch.full((Hq,), float("-inf"), dtype=torch.float32, device=DEVICE)
    topk_length = torch.full((B,), kv_len, dtype=torch.int32, device=DEVICE)

    # Dummy extra_kv placeholders (use_extra_kv=False)
    kv_paged_extra = torch.zeros(1, 1, 1, D, dtype=torch.float8_e4m3fn, device=DEVICE)
    bt_extra = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    extra_topk_length = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    launch1 = compile_mla_decode_fp8_v4(
        nheads_q=Hq, head_dim=D, topk=kv_len,
        block_h=block_h, block_n=64, sm_scale=sm_scale,
        NUM_WAVES=4, NUM_KV_BUFS=num_kv_bufs,
        Sq=Sq, num_kv_splits=splits, causal=causal,
        page_block_size=1, use_extra_kv=False, extra_topk_max=1,
        cluster_n=1,
    )
    launch2 = compile_mla_decode_fp8_v4_stage2(
        head_dim_v=D, num_q_heads=Hq, num_kv_splits=splits,
    )
    stream = torch.cuda.current_stream()

    launch1(Q.view(-1), kv_paged, block_table, topk_length,
            kv_paged_extra, bt_extra, extra_topk_length,
            mid_o.view(-1), mid_lse.view(-1),
            B, stream)
    launch2(mid_o.view(-1), mid_lse.view(-1), attn_sink_t,
            O.view(-1), total_q, stream)
    torch.cuda.synchronize()

    return O.float().detach().cpu().view(B, Sq, Hq, D)


def run_case(label, B, Sq, kv_len, Hq, D, splits, block_h, causal=False,
             num_kv_bufs=2):
    """Run one (block_h, case) configuration and compare vs fp32 reference."""
    print(f"\n--- {label}: B={B} kv_len={kv_len} splits={splits} "
          f"block_h={block_h} num_kv_bufs={num_kv_bufs} ---")
    setup_seed(SEED)

    Q = torch.empty(B, Sq, Hq, D, device=DEVICE).uniform_(-1.0, 1.0).to(torch.bfloat16)
    KV_fp8, kv_paged, block_table = gen_kv_paged(B, kv_len, D)
    sm_scale = 1.0 / math.sqrt(D)

    print(f"  Running block_h={block_h} num_kv_bufs={num_kv_bufs} ...")
    O_kernel = run_kernel(block_h, B, Sq, kv_len, Hq, D, splits,
                          Q, KV_fp8, kv_paged, block_table, sm_scale, causal,
                          num_kv_bufs=num_kv_bufs)

    KV_dequant = KV_fp8.float().to(torch.bfloat16)
    O_ref = v1_reference(Q, KV_dequant, sm_scale, causal=causal).cpu()

    diff = (O_kernel - O_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # find worst position
    flat_idx = diff.reshape(-1).argmax().item()
    unraveled = np.unravel_index(flat_idx, O_kernel.shape)

    passed = max_diff < PASS_TOL
    print(f"  vs fp32ref: max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  worst position (B,Sq,H,D)={unraveled}")

    return {
        "label": label,
        "block_h": block_h,
        "num_kv_bufs": num_kv_bufs,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "worst_pos": unraveled,
        "passed": passed,
    }


def main():
    Hq = 128
    D = 512

    cases = [
        # (label, B, Sq, kv_len, splits, block_h[, num_kv_bufs])
        ("case1_regression",  1, 1, 320,   1, 16),
        ("case2_bh32",        1, 1, 320,   1, 32),
        ("case3_bh64_main",   1, 1, 320,   1, 64),
        ("case4_multisplit",  2, 1, 640,   5, 64),
        ("case5_longkv",      1, 1, 1024,  1, 64),
        # N-stage pipeline tests: NUM_KV_BUFS=3/4
        ("case6_nkv3_bh16",   1, 1, 320,   1, 16, 3),
        ("case7_nkv3_bh64",   1, 1, 320,   1, 64, 3),
        ("case8_nkv4_bh16",   1, 1, 320,   1, 16, 4),
        ("case9_nkv4_bh64",   1, 1, 320,   1, 64, 4),
        ("case10_nkv4_multi", 2, 1, 640,   5, 64, 4),
    ]

    print(f"\n{'='*60}")
    print(f"Block-H precision validation (PASS_TOL={PASS_TOL})")
    print(f"Hq={Hq} D={D} SEED={SEED}")
    print(f"{'='*60}")

    results = []
    for entry in cases:
        label, B, Sq, kv_len, splits, block_h = entry[:6]
        num_kv_bufs = entry[6] if len(entry) > 6 else 2
        try:
            r = run_case(label, B, Sq, kv_len, Hq, D, splits, block_h,
                         num_kv_bufs=num_kv_bufs)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            r = {"label": label, "block_h": block_h, "num_kv_bufs": num_kv_bufs,
                 "passed": False, "max_diff": float("nan"), "mean_diff": float("nan"),
                 "error": str(e)}
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    n_pass = sum(1 for r in results if r.get("passed", False))
    n_fail = len(results) - n_pass
    for r in results:
        status = "PASS" if r.get("passed", False) else "FAIL"
        nkv_tag = f" num_kv_bufs={r.get('num_kv_bufs', 2)}" if "num_kv_bufs" in r else ""
        if "error" in r:
            print(f"  [{status}] {r['label']} (block_h={r['block_h']}{nkv_tag}): ERROR — {r['error']}")
        else:
            print(f"  [{status}] {r['label']} (block_h={r['block_h']}{nkv_tag}): "
                  f"max_diff={r['max_diff']:.4e}, mean_diff={r['mean_diff']:.4e}")

    print(f"\nResult: {n_pass} pass, {n_fail} fail")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
