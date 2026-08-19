#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm override for the warp collectives, and the portable form it displaces.

``coop/warp/rocdl.py`` replaces the shuffles behind ``warp_reduce`` and the
three scan entry points with a DPP sequence on gfx9. Two things have to hold
for that to be a safe swap, and they are what this file checks:

- **It computes the same thing.** Every case runs through both implementations
  and compares them against each other, not only against a host reference — a
  fold that is wrong in the same way in both would still pass a reference check
  on ones or on a symmetric distribution.
- **It is actually the code that runs.** A DPP sequence that quietly fell back
  would still be correct, so the shape of the fold is only observable in the
  generated ISA: on gfx9 every power-of-two width has to come out of the VALU,
  with the single ``ds_swizzle_b32`` a 32-lane reduce needs as the one
  exception. Everything the sequence does not cover — a non-gfx9 target, a
  value that is not a 32-bit ``Numeric`` — has to reach the portable
  implementation instead.

The functional coverage of the collectives themselves lives in
``test_warp_scan.py``; nothing here duplicates it.
"""

from __future__ import annotations

import pytest
from coop_common import WARP_WIDTHS

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None


# The portable forms under the same names the dispatched namespace exposes, so
# a test can name the collective once and run it through both. That is what
# ``fx.coop.universal`` is for, and using it here is also what checks it really
# is the displaced code rather than a second route to the override.
portable = fx.coop.universal

# ``WARP_WIDTHS`` is every width the DPP sequence takes a different route
# through: 2 and 4 are ``quad_perm`` alone, 8 brings in ``row_half_mirror``, 16
# ``row_ror:8``, 32 the one cross-row step, and the full wave the raking
# ``row_bcast`` ladder.
CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


def width_id(width):
    return "full" if width is None else f"w{width}"


requires_dpp = pytest.mark.skipif(not _is_gfx9(), reason="the DPP sequence is gfx9-only")


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_rocm_resolves_every_warp_collective_to_the_override():
    """On the ROCm backend the public names are the target's, not the portable ones."""
    from flydsl.extension.coop import warp
    from flydsl.extension.coop.warp import rocdl

    assert fx.coop.warp.rocdl is rocdl
    # The override covers the whole warp-scope surface, so nothing in it is
    # left resolving to a shuffle by accident.
    assert set(rocdl.__all__) == set(warp.__all__)
    # The portable implementations are still reachable, which is what lets the
    # override fall back to them and this file compare against them.
    assert portable.warp_reduce is not rocdl.warp_reduce
    assert portable.warp_inclusive_scan is not rocdl.warp_inclusive_scan


# ── the override and the portable form agree ───────────────────────────────


def _run_both(values, call, *, block):
    """Run *call* twice in one kernel: once dispatched, once portable.

    *call* takes the module to reach the collective through, so one line of the
    test names both the override and the form it displaces.
    """

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        tid = fx.thread_idx.x
        Fast[tid] = call(fx.coop, A[tid])
        Ref[tid] = call(portable, A[tid])

    @flyc.jit
    def launch(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Fast, Ref).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    fast = torch.zeros_like(values)
    ref = torch.zeros_like(values)
    launch(values, fast, ref, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return fast.cpu(), ref.cpu()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.parametrize(
    "form",
    ("reduce", "inclusive", "exclusive", "aggregate"),
)
def test_every_form_agrees_with_the_portable_one(form, width):
    """Integers wrap identically, so every one of these is exact on both sides."""
    warp_threads = fx.num_warp_threads()
    block = 2 * warp_threads  # two waves, so a fold that overran one would show
    group = warp_threads if width is None else width

    call = {
        "reduce": lambda m, v: m.warp_reduce(v, fx.ReductionOp.ADD, width=width),
        "inclusive": lambda m, v: m.warp_inclusive_scan(v, fx.ReductionOp.ADD, width=width),
        "exclusive": lambda m, v: m.warp_exclusive_scan(v, fx.ReductionOp.ADD, width=width),
        "aggregate": lambda m, v: m.warp_scan_with_aggregate(v, fx.ReductionOp.ADD, width=width)[2],
    }[form]

    torch.manual_seed(0)
    values = torch.randint(-(2**20), 2**20, (block,), dtype=torch.int32, device="cuda")
    fast, ref = _run_both(values, call, block=block)

    assert torch.equal(fast, ref)

    # Each group of *group* lanes folds on its own; nothing crosses the boundary.
    per_group = values.cpu().reshape(-1, group)
    inclusive = per_group.cumsum(1, dtype=torch.int32)
    expected = {
        "reduce": per_group.sum(1, dtype=torch.int32).repeat_interleave(group),
        "inclusive": inclusive.reshape(-1),
        "exclusive": (inclusive - per_group).reshape(-1),
        "aggregate": per_group.sum(1, dtype=torch.int32).repeat_interleave(group),
    }[form]
    assert torch.equal(fast, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("op", (fx.ReductionOp.ADD, fx.ReductionOp.MUL), ids=lambda o: o.name)
def test_float_folds_agree_with_the_portable_form(op):
    """Floats fold in a different order, so this is a tolerance check, not equality.

    The DPP sequence folds each 16-lane row in lane order and then carries the
    rows up; the butterfly pairs lanes by XOR distance. Both are valid, and
    neither is the other's rounding.
    """
    warp_threads = fx.num_warp_threads()
    block = 2 * warp_threads
    torch.manual_seed(0)
    if op is fx.ReductionOp.MUL:
        # Keep the product near 1 so the comparison is about the fold order and
        # not about a 64-way product overflowing.
        values = torch.rand(block, dtype=torch.float32, device="cuda") * 0.4 + 0.9
    else:
        values = torch.randn(block, dtype=torch.float32, device="cuda")

    fast, ref = _run_both(values, lambda m, v: m.warp_reduce(v, op), block=block)

    torch.testing.assert_close(fast, ref, rtol=1e-5, atol=1e-5)
    per_warp = values.cpu().reshape(-1, warp_threads).to(torch.float64)
    expected = (per_warp.sum(1) if op is fx.ReductionOp.ADD else per_warp.prod(1)).to(torch.float32)
    torch.testing.assert_close(fast, expected.repeat_interleave(warp_threads), rtol=1e-5, atol=1e-5)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.parametrize("form", ("reduce", "exclusive"))
def test_max_keeps_its_identity_inside_the_group(form, width):
    """MAX is the op whose identity is visible, so it is the one that pins the edges.

    ADD's identity is 0, which a lane that folded nothing also holds, so a
    boundary the DPP sequence got wrong would not necessarily show. MAX's is
    ``INT32_MIN``, every input below sits above it, and the exclusive form puts
    it in the output of each group's first lane.
    """
    warp_threads = fx.num_warp_threads()
    block = 2 * warp_threads
    group = warp_threads if width is None else width
    op = fx.ReductionOp.MAX

    call = {
        "reduce": lambda m, v: m.warp_reduce(v, op, width=width),
        "exclusive": lambda m, v: m.warp_exclusive_scan(v, op, width=width),
    }[form]

    torch.manual_seed(0)
    values = torch.randint(1, 2**20, (block,), dtype=torch.int32, device="cuda")
    fast, ref = _run_both(values, call, block=block)

    assert torch.equal(fast, ref)

    per_group = values.cpu().reshape(-1, group)
    if form == "reduce":
        expected = per_group.amax(1).repeat_interleave(group)
    else:
        identity = torch.full((per_group.shape[0], 1), -(2**31), dtype=torch.int32)
        expected = torch.cat([identity, per_group.cummax(1).values[:, :-1]], dim=1).reshape(-1)
    assert torch.equal(fast, expected)


# ── the override is actually the code that runs ────────────────────────────


def _isa(call, torch_dtype, *, tmp_path):
    """Compile a lone collective and return its final ISA as text."""
    import os
    from pathlib import Path

    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = call(A[tid])

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    values = torch.ones(warp_threads, dtype=torch_dtype, device="cuda")
    out = torch.zeros_like(values)

    previous = {k: os.environ.get(k) for k in ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        launch(values, out, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})

    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    return dumped[0].read_text()


def _count(isa, *needles):
    return sum(line.count(needle) for line in isa.splitlines() for needle in needles)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
def test_the_fold_stays_out_of_lds(width, tmp_path):
    """The whole point: the fold is VALU, so the LDS pipe stays free for the kernel.

    The one exception is a 32-lane reduce. Its last butterfly step is XOR by 16,
    which crosses the 16-lane rows DPP is built around, and gfx9 has no control
    for that — ``ds_swizzle_b32`` is the cheapest instruction that does it. The
    scans reach across the same boundary with ``row_bcast:15`` instead, because
    they only ever carry a value upwards.
    """

    def call(v):
        reduced = fx.coop.warp_reduce(v, fx.ReductionOp.ADD, width=width)
        inclusive, exclusive = fx.coop.warp_scan(v, fx.ReductionOp.ADD, width=width)
        return reduced + inclusive + exclusive

    isa = _isa(call, torch.float32, tmp_path=tmp_path)

    expected_lds = 1 if width == 32 else 0
    assert _count(isa, *CROSS_LANE_LDS) == expected_lds, "the DPP sequence did not take"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_a_full_wave_fold_rakes_and_lifts_the_total_into_an_sgpr(tmp_path):
    """At full width the fold is the ``row_shr`` ladder, not the butterfly.

    A whole wave has one lane holding the total once the ladder has run, so
    ``v_readlane_b32`` broadcasts it for free out of an SGPR; a narrower group
    has no such lane and has to butterfly instead.
    """
    isa = _isa(lambda v: fx.coop.warp_reduce(v, fx.ReductionOp.ADD), torch.float32, tmp_path=tmp_path)

    assert "row_shr:1" in isa and "row_bcast:15" in isa and "row_bcast:31" in isa
    assert "v_readlane_b32" in isa
    assert _count(isa, *CROSS_LANE_LDS) == 0


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_universal_is_still_the_shuffles(tmp_path):
    """``fx.coop.universal`` keeps the LDS fold, which only the ISA can show.

    Every comparison above holds the two implementations against each other by
    their results, and a ``universal`` that had quietly resolved to the
    override would agree with itself perfectly. A narrow group is where the
    difference is unambiguous: LLVM rewrites a whole-wave shuffle fold into DPP
    on its own, but not one confined to a group, so the shuffles survive into
    the ISA — where the same width through ``fx.coop`` reaches no LDS at all
    (``test_the_fold_stays_out_of_lds``).
    """
    isa = _isa(lambda v: portable.warp_reduce(v, fx.ReductionOp.ADD, width=8), torch.float32, tmp_path=tmp_path)

    assert _count(isa, *CROSS_LANE_LDS) > 0, "universal did not reach the portable implementation"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_a_dtype_the_sequence_cannot_move_falls_back(tmp_path):
    """DPP moves 32 bits at a time, so a 16-bit fold has to keep the shuffles."""
    isa = _isa(lambda v: fx.coop.warp_reduce(v, fx.ReductionOp.ADD), torch.int16, tmp_path=tmp_path)

    assert "row_bcast" not in isa, "the DPP sequence took a dtype it cannot move"
    assert _count(isa, *CROSS_LANE_LDS) > 0
