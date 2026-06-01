#!/usr/bin/env python3
"""Detailed diagnostic: locate which heads/positions fail for block_h=64."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import numpy as np
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available"); sys.exit(1)

from kernels.mla_decode_fp8_v4_gfx1250 import compile_mla_decode_fp8_v4
from kernels.mla_decode_fp8_v4_stage2_gfx1250 import compile_mla_decode_fp8_v4_stage2

SEED = 42
DEVICE = "cuda"

def setup_seed():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def gen_kv_paged(B, kv_len, D):
    total_blocks = B * kv_len
    kv_real = torch.empty(B, kv_len, D, device=DEVICE).uniform_(-1.0, 1.0)
    KV_fp8 = kv_real.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    kv_paged = KV_fp8.view(B, kv_len, 1, 1, D).contiguous().view(total_blocks, 1, 1, D)
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=DEVICE).view(B, kv_len).contiguous()
    return KV_fp8, kv_paged, block_table

def v1_ref(Q, KV_bf16, sm_scale):
    B, Sq, H, D = Q.shape
    _, kv_len, _ = KV_bf16.shape
    Q_f, K_f = Q.float(), KV_bf16.float()
    O = torch.zeros(B, Sq, H, D, dtype=torch.float32, device=Q.device)
    for b in range(B):
        for s in range(Sq):
            S = (Q_f[b, s] @ K_f[b].T) * sm_scale
            P = torch.softmax(S, dim=-1)
            O[b, s] = P @ K_f[b]
    return O

def run_bh(block_h, B, Sq, kv_len, Hq, D, splits, Q, kv_paged, block_table, sm_scale):
    total_q = B * Sq
    mid_o = torch.zeros(total_q, splits, Hq, D, dtype=torch.float32, device=DEVICE)
    mid_lse = torch.full((total_q, splits, Hq), float("-inf"), dtype=torch.float32, device=DEVICE)
    O = torch.zeros(total_q, Hq, D, dtype=torch.bfloat16, device=DEVICE)
    attn_sink_t = torch.full((Hq,), float("-inf"), dtype=torch.float32, device=DEVICE)
    topk_length = torch.full((B,), kv_len, dtype=torch.int32, device=DEVICE)
    kv_paged_extra = torch.zeros(1, 1, 1, D, dtype=torch.float8_e4m3fn, device=DEVICE)
    bt_extra = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    extra_topk_length = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    launch1 = compile_mla_decode_fp8_v4(
        nheads_q=Hq, head_dim=D, topk=kv_len,
        block_h=block_h, block_n=64, sm_scale=sm_scale,
        NUM_WAVES=4, NUM_KV_BUFS=2,
        Sq=Sq, num_kv_splits=splits, causal=False,
        page_block_size=1, use_extra_kv=False, extra_topk_max=1, cluster_n=1,
    )
    launch2 = compile_mla_decode_fp8_v4_stage2(
        head_dim_v=D, num_q_heads=Hq, num_kv_splits=splits,
    )
    stream = torch.cuda.current_stream()
    launch1(Q.view(-1), kv_paged, block_table, topk_length,
            kv_paged_extra, bt_extra, extra_topk_length,
            mid_o.view(-1), mid_lse.view(-1), B, stream)
    launch2(mid_o.view(-1), mid_lse.view(-1), attn_sink_t, O.view(-1), total_q, stream)
    torch.cuda.synchronize()
    return O.float().detach().cpu().view(B, Sq, Hq, D)

def main():
    B, Sq, kv_len, Hq, D, splits = 1, 1, 320, 128, 512, 1
    sm_scale = 1.0 / math.sqrt(D)

    setup_seed()
    Q = torch.empty(B, Sq, Hq, D, device=DEVICE).uniform_(-1.0, 1.0).to(torch.bfloat16)
    KV_fp8, kv_paged, block_table = gen_kv_paged(B, kv_len, D)

    print("Running block_h=16 ...")
    O_bh16 = run_bh(16, B, Sq, kv_len, Hq, D, splits, Q, kv_paged, block_table, sm_scale)
    print("Running block_h=64 ...")
    O_bh64 = run_bh(64, B, Sq, kv_len, Hq, D, splits, Q, kv_paged, block_table, sm_scale)

    KV_dequant = KV_fp8.float().to(torch.bfloat16)
    print("Computing reference ...")
    O_ref = v1_ref(Q, KV_dequant, sm_scale).cpu()  # (B, Sq, Hq, D)

    # Per-head analysis
    # shapes: (1, 1, 128, 512) => squeeze to (128, 512)
    ref_h = O_ref[0, 0]    # (Hq, D)
    bh16_h = O_bh16[0, 0]  # (Hq, D)
    bh64_h = O_bh64[0, 0]  # (Hq, D)

    diff16 = (bh16_h - ref_h).abs()   # (Hq, D)
    diff64 = (bh64_h - ref_h).abs()   # (Hq, D)
    cross  = (bh64_h - bh16_h).abs()  # (Hq, D)

    # Per-head max diff
    per_head_diff64 = diff64.max(dim=-1).values  # (Hq,)
    per_head_diff16 = diff16.max(dim=-1).values  # (Hq,)
    per_head_cross  = cross.max(dim=-1).values   # (Hq,)

    print("\n--- Per-head analysis (bh64 vs ref, top 20 worst heads) ---")
    sorted_heads = per_head_diff64.argsort(descending=True)
    for i in range(min(20, Hq)):
        h = sorted_heads[i].item()
        print(f"  head={h:3d}: diff64={per_head_diff64[h]:.4e}  diff16={per_head_diff16[h]:.4e}  cross={per_head_cross[h]:.4e}")

    # Which head_blocks (block_h=64 groups)? Head [0..63] = block 0, [64..127] = block 1
    print("\n--- Per-block_h=64 group analysis ---")
    for blk in range(Hq // 64):
        h_start = blk * 64
        h_end = h_start + 64
        blk_diff64 = diff64[h_start:h_end].max().item()
        blk_diff16 = diff16[h_start:h_end].max().item()
        blk_cross  = cross[h_start:h_end].max().item()
        print(f"  block[{blk}] heads[{h_start}:{h_end}]: "
              f"diff64={blk_diff64:.4e}  diff16={blk_diff16:.4e}  cross={blk_cross:.4e}")

    # Worst head: where in D is the error?
    worst_h = sorted_heads[0].item()
    print(f"\n--- Worst head={worst_h}: diff64 per dim (top 10 positions) ---")
    worst_diff = diff64[worst_h]  # (D,)
    top_dims = worst_diff.argsort(descending=True)[:10]
    for d in top_dims.tolist():
        print(f"  dim={d:4d}: ref={ref_h[worst_h, d]:.4f} bh16={bh16_h[worst_h, d]:.4f} bh64={bh64_h[worst_h, d]:.4f}  diff64={worst_diff[d]:.4e}")

    # Check if error clusters within bh=64 tile boundaries
    print("\n--- head-in-tile analysis (within each block_h=64 group) ---")
    # block 0: heads 0..63, block 1: heads 64..127
    for blk in range(Hq // 64):
        h_start = blk * 64
        h_end = h_start + 64
        per_head_in_blk = per_head_diff64[h_start:h_end]
        # which positions within the tile are worst
        local_worst = per_head_in_blk.argsort(descending=True)[:5]
        print(f"  block[{blk}] worst 5 local heads: {[h_start+x.item() for x in local_worst]}")
        print(f"    diffs: {[f'{per_head_in_blk[x]:.3e}' for x in local_worst]}")

    print("\nDone.")

if __name__ == "__main__":
    main()
