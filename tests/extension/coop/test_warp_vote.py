#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for warp predicate voting, ballot masks, compaction ranks, and WarpVote."""

from __future__ import annotations

import pytest
from coop_common import WARP_SIZE, WARP_WIDTHS
from coop_test_utils import matrix_cases, run_kernel
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal
ALL_WIDTHS = (1, *WARP_WIDTHS)


def width_id(width):
    return "full" if width is None else f"w{width}"


@pytest.mark.l1a_compile_no_target_dialect
def test_universal_vote_emits_no_target_dialect(ctx, insert_point):
    """Portable vote collectives must lower through generic GPU/arith ops only."""
    x = fx.arith.constant(1, type=fx.T.i32)
    pred = fx.Int32(x) > fx.Int32(0)
    any_v = portable.warp_any(pred, width=8)
    all_v = portable.warp_all(pred, width=8)
    mask = portable.warp_ballot(pred, width=8)
    cnt = portable.warp_count(pred, width=8)
    rank, total = portable.warp_ballot_rank(pred, width=8)
    V = portable.WarpVote[fx.Boolean, 8]
    v_mask = V.ballot(pred)
    _ = (any_v, all_v, mask, cnt, rank, total, v_mask)

    ir_str = str(ctx.module)
    assert "gpu.shuffle" in ir_str
    assert "rocdl." not in ir_str


@pytest.mark.l1a_compile_no_target_dialect
def test_vote_validation_errors(insert_point):
    with pytest.raises(TypeError, match="Boolean or integer"):
        fx.coop.WarpVote[fx.Float32, 8]
    with pytest.raises(TypeError, match="Boolean or integer"):
        fx.coop.warp_any(fx.Float32(1.0), width=4)
    with pytest.raises(TypeError, match="scalar predicate"):
        fx.coop.WarpVote[fx.Int32, 4].any(fx.Vector.filled(2, 1, fx.Int32))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "width,pattern",
    matrix_cases(ALL_WIDTHS, ("zeros", "ones", "alternating", "single_first", "single_last", "random")),
)
@pytest.mark.usefixtures("warp_default_device")
def test_warp_vote_functions_and_universal_parity(width, pattern):
    block = 2 * WARP_SIZE
    group = WARP_SIZE if width is None else width

    if pattern == "zeros":
        flags = torch.zeros(block, dtype=torch.int32, device="cuda")
    elif pattern == "ones":
        flags = torch.ones(block, dtype=torch.int32, device="cuda")
    elif pattern == "alternating":
        flags = torch.arange(block, dtype=torch.int32, device="cuda") % 2
    elif pattern == "single_first":
        flags = ((torch.arange(block, dtype=torch.int32, device="cuda") % group) == 0).to(torch.int32)
    elif pattern == "single_last":
        flags = ((torch.arange(block, dtype=torch.int32, device="cuda") % group) == (group - 1)).to(torch.int32)
    else:
        torch.manual_seed(42)
        flags = torch.randint(0, 2, (block,), dtype=torch.int32, device="cuda")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(Flags: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        tid = fx.thread_idx.x
        pred = Flags[tid] != fx.Int32(0)

        for idx, ns, dst in ((0, fx.coop, Fast), (1, portable, Ref)):
            _ = idx
            any_v = ns.warp_any(pred, width=width)
            all_v = ns.warp_all(pred, width=width)
            mask = ns.warp_ballot(pred, width=width)
            cnt = ns.warp_count(pred, width=width)
            rank, total = ns.warp_ballot_rank(pred, width=width)

            base = tid * 7
            dst[base + 0] = any_v.select(fx.Int32(1), fx.Int32(0))
            dst[base + 1] = all_v.select(fx.Int32(1), fx.Int32(0))
            dst[base + 2] = fx.Uint32(mask).bitcast(fx.Int32)
            dst[base + 3] = fx.Uint32(mask >> fx.Uint64(32)).bitcast(fx.Int32)
            dst[base + 4] = cnt
            dst[base + 5] = rank
            dst[base + 6] = total

    @flyc.jit
    def launch(Flags: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(Flags, Fast, Ref).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    fast = torch.zeros(block * 7, dtype=torch.int32, device="cuda")
    ref = torch.zeros(block * 7, dtype=torch.int32, device="cuda")
    launch(flags, fast, ref, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    fast_cpu = fast.cpu().reshape(block, 7)
    ref_cpu = ref.cpu().reshape(block, 7)
    assert torch.equal(fast_cpu, ref_cpu)

    # Host reference verification per logical group
    bits = (flags.cpu() != 0).reshape(-1, group).to(torch.int64)
    exp_any = bits.any(dim=1).to(torch.int32).repeat_interleave(group)
    exp_all = bits.all(dim=1).to(torch.int32).repeat_interleave(group)
    exp_cnt = bits.sum(dim=1).to(torch.int32).repeat_interleave(group)
    exp_rank = (bits.cumsum(dim=1) - bits).reshape(-1).to(torch.int32)

    assert torch.equal(fast_cpu[:, 0], exp_any)
    assert torch.equal(fast_cpu[:, 1], exp_all)
    assert torch.equal(fast_cpu[:, 4], exp_cnt)
    assert torch.equal(fast_cpu[:, 5], exp_rank)
    assert torch.equal(fast_cpu[:, 6], exp_cnt)

    for g in range(bits.shape[0]):
        mask_val = 0
        for k in range(group):
            if int(bits[g, k].item()):
                mask_val |= 1 << k
        lo_expected = mask_val & 0xFFFFFFFF
        hi_expected = (mask_val >> 32) & 0xFFFFFFFF
        for k in range(group):
            row = g * group + k
            got_lo = int(fast_cpu[row, 2].item()) & 0xFFFFFFFF
            got_hi = int(fast_cpu[row, 3].item()) & 0xFFFFFFFF
            assert got_lo == lo_expected and got_hi == hi_expected


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.usefixtures("warp_default_device")
def test_warp_vote_operator_and_stream_compaction(width):
    """Verify WarpVote integer-flag coercion and warp-scope stream compaction."""
    block = 2 * WARP_SIZE
    group = WARP_SIZE if width is None else width
    torch.manual_seed(7)
    values = torch.randint(-20, 21, (block,), dtype=torch.int32, device="cuda")

    @flyc.jit
    def apply(a, out):
        tid = fx.thread_idx.x
        v = a[tid]
        keep = v > fx.Int32(0)
        V = fx.coop.WarpVote[fx.Int32, width]
        flag_i32 = keep.select(fx.Int32(5), fx.Int32(0))
        rank, count = V.ballot_rank(flag_i32)
        group_base = (tid // group) * group
        out[tid] = fx.Int32(-1)
        fx.barrier()
        if keep:
            out[group_base + rank] = v
        out[block + tid] = count

    res = run_kernel(apply, values, 2 * block, block)
    compacted = res[:block].reshape(-1, group)
    counts = res[block:].reshape(-1, group)
    src = values.cpu().reshape(-1, group)
    for g in range(src.shape[0]):
        kept = src[g][src[g] > 0]
        assert torch.all(counts[g] == kept.numel())
        assert torch.equal(compacted[g, : kept.numel()], kept)
