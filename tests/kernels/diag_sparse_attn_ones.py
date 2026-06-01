"""Diagnostic: Q=KV=ones smoke test for sparse_attn_gfx1250 v0 WMMA bug.

Reproduces the "97.66% correct, 12 cells per head wrong" pattern documented in
~/mla_notes/implementation/06_sparse_attn_wip.md §3.1. Used to A/B test fix
candidates without softmax overflow noise from random data.

Usage:
  PYTHONPATH=./ python tests/kernels/diag_sparse_attn_ones.py
"""

import sys
import math
import torch

sys.path.insert(0, ".")
from kernels.sparse_attn_gfx1250 import compile_sparse_attn_gfx1250


def main():
    import os
    B, n_kv, topk = 1, 64, 64
    H = 128
    D = int(os.environ.get("DIAG_HEAD_DIM", "512"))
    block_h, block_n = 16, 64
    num_waves, num_kv_bufs = 4, 2
    device = "cuda"

    # v1.1: KV is fp8_e4m3 in HBM, with per-block fp32 dequant scale shape
    # (n_kv, head_dim/128). For ones diag: all per-block scales = 1.0 + KV
    # stored as fp8(1.0) → dequant gives 1.0. Expected output = ones (uniform
    # attention over 64 keys with all-ones values).
    KV_SCALE_BLOCK = 128
    n_blocks = D // KV_SCALE_BLOCK
    Q = torch.ones(B, H, D, dtype=torch.bfloat16, device=device)
    KV = torch.ones(B, n_kv, D, dtype=torch.float8_e4m3fn, device=device)
    kv_scale = torch.ones(n_kv, n_blocks, dtype=torch.float32, device=device)
    topk_idxs = torch.arange(topk, dtype=torch.int32, device=device).view(1, topk).expand(B, topk).contiguous()
    O = torch.zeros(B, H, D, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(D)
    launch = compile_sparse_attn_gfx1250(
        nheads_q=H, head_dim=D, topk=topk,
        block_h=block_h, block_n=block_n, sm_scale=sm_scale,
        NUM_WAVES=num_waves, NUM_KV_BUFS=num_kv_bufs,
    )

    stream = torch.cuda.current_stream()
    launch(Q, KV, kv_scale, topk_idxs, O, B, n_kv, topk, stream)
    torch.cuda.synchronize()

    O_f = O.float().detach().cpu()
    total = O_f.numel()
    ones_cells = (O_f == 1.0).sum().item()
    pct = 100.0 * ones_cells / total
    bad_per_head = (O_f[0] != 1.0).sum(dim=-1)  # per head

    print(f"=== Q=KV=ones diagnostic ===")
    print(f"shape O: {tuple(O.shape)}  total: {total}")
    print(f"correct cells (== 1.0): {ones_cells} / {total} = {pct:.2f}%")
    print(f"bad cells per head (sample first 4): {bad_per_head[:4].tolist()}")
    print(f"bad cells per head (max/min/mean): max={bad_per_head.max().item()} "
          f"min={bad_per_head.min().item()} mean={bad_per_head.float().mean().item():.2f}")

    # Show the pattern in head 0
    head0 = O_f[0, 0]
    nonone_mask = (head0 != 1.0)
    bad_idxs = nonone_mask.nonzero(as_tuple=True)[0]
    print(f"\nhead 0 bad cell column indices: {bad_idxs.tolist()}")
    if len(bad_idxs) > 0:
        print(f"head 0 unique bad values: {sorted(set(head0[bad_idxs].tolist()))}")
        last16_start = D - 16
        print(f"head 0 cell-by-cell values (cols {last16_start}..{D-1}):")
        for c in range(last16_start, D):
            print(f"  col {c}: {head0[c].item():>20.4f}")
        # Also check head 1, head 64, head 127 for patterns
        for h in [1, 64, 127]:
            row = O_f[0, h]
            bad = (row != 1.0).nonzero(as_tuple=True)[0]
            if len(bad) > 0:
                vals = sorted(set(row[bad].tolist()))
                print(f"head {h} unique bad values: {vals}")

    # Detect NaN as separate failure mode
    nan_count = torch.isnan(O_f).sum().item()
    if nan_count > 0:
        print(f"\n!!! WARNING: {nan_count} NaN cells in output ({100*nan_count/total:.2f}%)")

    # Pass criteria for diagnostic: 100% ones means bug fixed
    if ones_cells == total:
        print("\n✅ PASS: 100% correct, bug appears FIXED")
        return 0
    else:
        print(f"\n❌ FAIL: {total - ones_cells} cells wrong ({100 - pct:.2f}%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
