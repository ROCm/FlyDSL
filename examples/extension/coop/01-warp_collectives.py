# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-scope ``warp_reduce`` / ``warp_exclusive_scan`` without any shared memory.

The kernel below computes, for every lane, where its element would go if the
warp compacted its positive elements, and how many the warp keeps in total.

  1. **The width is named rather than inherited.** Left alone, both collectives
     span a full warp, which is 64 lanes on CDNA and 32 on RDNA — so a kernel
     that says nothing computes a different thing on each. Asking for 32 pins
     the group to one size on either, which is what lets this one source run on
     both. On a wave64 target the block is then half a wave, and the named width
     is also what keeps the fold off the lanes that were never launched.
  2. **Every lane gets the answer.** Both collectives are all-to-all, so the
     result needs no follow-up broadcast.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

WIDTH = 32


@flyc.kernel
def warp_compaction_index(A: fx.Tensor, Rank: fx.Tensor, Total: fx.Tensor):
    tid = fx.thread_idx.x
    keep = (A[tid] > 0.0).select(1, 0)

    Rank[tid] = fx.coop.warp_exclusive_scan(keep, fx.ReductionOp.ADD, width=WIDTH)
    Total[tid] = fx.coop.warp_reduce(keep, fx.ReductionOp.ADD, width=WIDTH)


@flyc.jit
def warp_compaction(A: fx.Tensor, Rank: fx.Tensor, Total: fx.Tensor):
    warp_compaction_index(A, Rank, Total).launch(grid=(1, 1, 1), block=(WIDTH, 1, 1))


A = torch.randn(WIDTH, dtype=torch.float32, device="cuda")
Rank = torch.zeros(WIDTH, dtype=torch.int32, device="cuda")
Total = torch.zeros(WIDTH, dtype=torch.int32, device="cuda")

warp_compaction(A, Rank, Total)
torch.cuda.synchronize()

keep = (A.cpu() > 0).to(torch.int32).reshape(-1, WIDTH)
expected_rank = keep.cumsum(1, dtype=torch.int32) - keep
expected_total = keep.sum(1, dtype=torch.int32, keepdim=True).expand(-1, WIDTH)

if torch.equal(Rank.cpu().reshape(-1, WIDTH), expected_rank) and torch.equal(
    Total.cpu().reshape(-1, WIDTH), expected_total
):
    print(f"PASS ({WIDTH} lanes, {int(expected_total[0, 0])} kept)")
else:
    print("FAIL")
    raise SystemExit(1)
