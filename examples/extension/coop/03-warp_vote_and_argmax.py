# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register-scope stream compaction and extremum indexing with ``fx.coop``.

Every lane in a 32-lane logical warp inspects one ``Float32`` element without
shared memory:

  1. ``warp_any``, ``warp_count``, and ``warp_ballot_rank`` compact strictly
     positive lanes into dense output positions ``0 .. kept - 1`` in one
     hardware wave ballot + ``mbcnt`` pass.
  2. ``warp_argmax`` and ``warp_argmin`` locate the warp's maximum and minimum
     values along with their deterministic lowest-index winning lanes.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

WIDTH = 32


@flyc.kernel
def vote_and_argext_kernel(
    Src: fx.Tensor,
    Compacted: fx.Tensor,
    SummaryIdx: fx.Tensor,
    SummaryVal: fx.Tensor,
):
    tid = fx.thread_idx.x
    val = Src[tid]
    keep = val > 0.0

    has_pos = fx.coop.warp_any(keep, width=WIDTH)
    pos_count = fx.coop.warp_count(keep, width=WIDTH)
    rank, total = fx.coop.warp_ballot_rank(keep, width=WIDTH)
    max_val, max_lane = fx.coop.warp_argmax(val, width=WIDTH)
    min_val, min_lane = fx.coop.warp_argmin(val, width=WIDTH)

    if keep:
        Compacted[rank] = val

    if tid == 0:
        SummaryIdx[0] = has_pos.select(fx.Int32(1), fx.Int32(0))
        SummaryIdx[1] = pos_count
        SummaryIdx[2] = total
        SummaryIdx[3] = max_lane
        SummaryIdx[4] = min_lane
        SummaryVal[0] = max_val
        SummaryVal[1] = min_val


@flyc.jit
def run_vote_and_argext(
    Src: fx.Tensor,
    Compacted: fx.Tensor,
    SummaryIdx: fx.Tensor,
    SummaryVal: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    vote_and_argext_kernel(Src, Compacted, SummaryIdx, SummaryVal).launch(
        grid=(1, 1, 1),
        block=(WIDTH, 1, 1),
        stream=stream,
    )


torch.manual_seed(0)
src = torch.randn(WIDTH, dtype=torch.float32, device="cuda")
# Exercise deterministic lowest-lane tie-breaking on the maximum value.
src[5] = 12.5
src[19] = 12.5

compacted = torch.zeros(WIDTH, dtype=torch.float32, device="cuda")
summary_idx = torch.zeros(5, dtype=torch.int32, device="cuda")
summary_val = torch.zeros(2, dtype=torch.float32, device="cuda")

run_vote_and_argext(src, compacted, summary_idx, summary_val)
torch.cuda.synchronize()

src_cpu = src.cpu()
expected_pos = src_cpu[src_cpu > 0.0]
kept = int(expected_pos.numel())
expected_max_val = float(src_cpu.max().item())
expected_max_lane = int((src_cpu == expected_max_val).nonzero(as_tuple=False)[0].item())
expected_min_val = float(src_cpu.min().item())
expected_min_lane = int((src_cpu == expected_min_val).nonzero(as_tuple=False)[0].item())

idx_cpu = summary_idx.cpu().tolist()
val_cpu = summary_val.cpu().tolist()

ok = (
    idx_cpu == [int(kept > 0), kept, kept, expected_max_lane, expected_min_lane]
    and torch.equal(compacted.cpu()[:kept], expected_pos)
    and val_cpu[0] == expected_max_val
    and val_cpu[1] == expected_min_val
)

if ok:
    print(
        f"PASS ({kept}/{WIDTH} positive compacted, "
        f"argmax=({val_cpu[0]:.2f} @ lane {idx_cpu[3]}), "
        f"argmin=({val_cpu[1]:.2f} @ lane {idx_cpu[4]}))"
    )
else:
    print(f"FAIL (idx={idx_cpu}, val={val_cpu})")
    raise SystemExit(1)
