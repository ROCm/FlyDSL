#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm override for the warp collectives, and the portable form it displaces.

``coop/warp/rocdl.py`` replaces the shuffles behind ``warp_reduce``, the scan
entry points, ``warp_argmax``/``warp_argmin``, and the ``WarpVote`` family with
native VALU DPP + ``ballot``/``mbcnt`` sequences on both wave64 CDNA (``gfx9*``)
and wave32 GFX10+/RDNA/CDNA5 (``gfx10*``, ``gfx11*``, ``gfx12*``). Two things
have to hold for that to be a safe swap, and they are what this file checks:

- **It computes the same thing.** Every case runs through both implementations
  and compares them against each other, not only against a host reference — a
  fold that is wrong in the same way in both would still pass a reference check
  on ones or on a symmetric distribution.
- **It is actually the code that runs.** A DPP sequence that quietly fell back
  would still be correct, so the shape of the fold is only observable in the
  generated ISA: on wave32 GFX10+/RDNA/CDNA5 every power-of-two width comes out
  of VALU with zero cross-lane LDS instructions, and on wave64 CDNA every width
  comes out of VALU with the single ``ds_swizzle_b32`` a 32-lane reduce needs as
  the one exception.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from coop_common import WARP_WIDTHS
from coop_test_utils import matrix_cases
from coop_test_utils import warp_default_device as warp_default_device

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

CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")
CROSS_ARCH_TARGETS = ("gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250")


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


def _is_dpp_target():
    arch = current_target().arch
    return arch.startswith(("gfx9", "gfx10", "gfx11", "gfx12"))


def width_id(width):
    return "full" if width is None else f"w{width}"


requires_dpp = pytest.mark.skipif(not _is_dpp_target(), reason="requires CDNA or GFX10+/RDNA/CDNA5 DPP target")


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_rocm_declares_its_implemented_warp_overrides():
    """ROCm overrides are public names; newer families may use portable fallbacks."""
    from flydsl.extension.coop import warp
    from flydsl.extension.coop.warp import rocdl

    assert fx.coop.warp.rocdl is rocdl
    # The target only advertises algorithms it implements. Other public warp
    # families retain the portable implementation through Dispatcher.
    assert set(rocdl.__all__) <= set(warp.__all__)
    # The portable implementations are still reachable, which is what lets the
    # override fall back to them and this file compare against them.
    assert portable.warp_reduce is not rocdl.warp_reduce
    assert portable.warp_argmax is not rocdl.warp_argmax
    assert portable.warp_inclusive_scan is not rocdl.warp_inclusive_scan
    assert portable.warp_ballot is not rocdl.warp_ballot
    assert portable.warp_ballot_rank is not rocdl.warp_ballot_rank


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
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(values, fast, ref, stream=stream)
    torch.cuda.synchronize()
    return fast.cpu(), ref.cpu()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "width,form",
    matrix_cases(WARP_WIDTHS, ("reduce", "inclusive", "exclusive", "aggregate")),
)
@pytest.mark.usefixtures("warp_default_device")
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
@pytest.mark.usefixtures("warp_default_device")
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
@pytest.mark.parametrize("width,form", matrix_cases(WARP_WIDTHS, ("reduce", "exclusive")))
@pytest.mark.usefixtures("warp_default_device")
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

    per_group = values.cpu().reshape(-1, group)
    if form == "reduce":
        expected = per_group.amax(1).repeat_interleave(group)
    else:
        identity = torch.full((per_group.shape[0], 1), -(2**31), dtype=torch.int32, device="cpu")
        expected = torch.cat([identity, per_group.cummax(1).values[:, :-1]], dim=1).reshape(-1)
    assert torch.equal(fast, ref)
    assert torch.equal(fast, expected)


# ── the override is actually the code that runs ────────────────────────────


def _isa(call, torch_dtype, *, tmp_path, compile_only=False):
    """Compile a lone collective and return its final ISA as text."""
    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = call(A[tid])

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    dev = "cpu" if compile_only else "cuda"
    values = torch.ones(warp_threads, dtype=torch_dtype, device=dev)
    out = torch.zeros_like(values)

    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE", "COMPILE_ONLY")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    if compile_only:
        os.environ["COMPILE_ONLY"] = "1"
    try:
        if compile_only:
            launch(values, out)
        else:
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

    On wave32 GFX10+/RDNA/CDNA5, ``row_xmask``, ``row_share``, ``permlanex16``,
    and ``readlane`` cover every width with zero LDS ops. On wave64 CDNA, the
    one exception is a 32-lane sub-wave reduce (``ds_swizzle_b32`` XOR 16).
    """

    def call(v):
        reduced = fx.coop.warp_reduce(v, fx.ReductionOp.ADD, width=width)
        inclusive, exclusive, aggregate = fx.coop.warp_scan_with_aggregate(v, fx.ReductionOp.ADD, width=width)
        return reduced + inclusive + exclusive + aggregate if not _is_gfx9() else reduced + inclusive + exclusive

    isa = _isa(call, torch.float32, tmp_path=tmp_path)

    expected_lds = 1 if (_is_gfx9() and width == 32) else 0
    assert _count(isa, *CROSS_LANE_LDS) == expected_lds, "the DPP sequence did not take"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_a_full_wave_fold_rakes_and_lifts_the_total_into_an_sgpr(tmp_path):
    """At full width the fold is the ``row_shr`` ladder, not the butterfly.

    A whole wave has one lane holding the total once the ladder has run (lane 63
    on CDNA via ``row_bcast``, lane 31 on RDNA/GFX12 via ``v_permlanex16_b32``),
    so ``v_readlane_b32`` broadcasts it for free out of an SGPR.
    """
    isa = _isa(lambda v: fx.coop.warp_reduce(v, fx.ReductionOp.ADD), torch.float32, tmp_path=tmp_path)

    assert "row_shr:1" in isa
    if _is_gfx9():
        assert "row_bcast:15" in isa and "row_bcast:31" in isa
    else:
        assert "v_permlanex16_b32" in isa
    assert "v_readlane_b32" in isa
    assert _count(isa, *CROSS_LANE_LDS) == 0


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_universal_is_still_the_shuffles(tmp_path):
    """``fx.coop.universal`` keeps the LDS fold, which only the ISA can show."""
    isa = _isa(lambda v: portable.warp_reduce(v, fx.ReductionOp.ADD, width=8), torch.float32, tmp_path=tmp_path)

    assert _count(isa, *CROSS_LANE_LDS) > 0, "universal did not reach the portable implementation"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@requires_dpp
def test_a_dtype_the_sequence_cannot_move_falls_back(tmp_path):
    """DPP moves 32 bits at a time, so a 16-bit fold has to keep the shuffles."""
    isa = _isa(lambda v: fx.coop.warp_reduce(v, fx.ReductionOp.ADD, width=8), torch.int16, tmp_path=tmp_path)

    assert "row_bcast" not in isa and "row_xmask" not in isa, "the DPP sequence took a dtype it cannot move"
    assert _count(isa, *CROSS_LANE_LDS) > 0


# ── cross-architecture wave32 / wave64 ISA and True16 checks ───────────────


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", CROSS_ARCH_TARGETS)
def test_cross_arch_reduce_scan_vote_argmax_isa(monkeypatch, tmp_path, arch):
    """Verify native wave64 vs wave32 ISA selection and zero ``_fake16`` across architectures."""
    monkeypatch.setenv("ARCH", arch)
    is_wave64 = arch.startswith("gfx9")

    # 1) Full-wave reduce + scan_with_aggregate + vote + ballot_rank + argmax
    def full_wave_call(v):
        red = fx.coop.warp_reduce(v, fx.ReductionOp.ADD)
        inc, exc, agg = fx.coop.warp_scan_with_aggregate(v, fx.ReductionOp.ADD)
        rank, cnt = fx.coop.warp_ballot_rank(v > fx.Float32(0.0))
        max_v, max_i = fx.coop.warp_argmax(v)
        return red + inc + exc + agg + max_v + fx.Float32(rank + cnt + max_i)

    isa_full = _isa(full_wave_call, torch.float32, tmp_path=tmp_path / "full", compile_only=True)
    assert "_fake16" not in isa_full
    assert _count(isa_full, *CROSS_LANE_LDS) == 0
    assert "v_readlane_b32" in isa_full
    assert "v_mbcnt_lo_u32_b32" in isa_full
    if is_wave64:
        assert "row_bcast:15" in isa_full and "row_bcast:31" in isa_full
        assert "v_mbcnt_hi_u32_b32" in isa_full
        assert "v_permlanex16_b32" not in isa_full
    else:
        assert "v_permlanex16_b32" in isa_full
        assert "v_mbcnt_hi_u32_b32" not in isa_full
        assert "row_bcast" not in isa_full

    # 2) Narrow subgroup (width=8) reduce + scan_with_aggregate + ballot_rank + Float16 argmax (True16 check)
    def subwarp_f16_call(v):
        max_v, max_i = fx.coop.warp_argmax(v, width=8)
        min_v, min_i = fx.coop.warp_argmin(v, width=8)
        rank, cnt = fx.coop.warp_ballot_rank(fx.Float32(v) > fx.Float32(0.0), width=8)
        inc, exc, agg = fx.coop.warp_scan_with_aggregate(fx.Float32(v), fx.ReductionOp.ADD, width=8)
        return fx.Float16(
            fx.Float32(max_v) + fx.Float32(min_v) + inc + exc + agg + fx.Float32(max_i + min_i + rank + cnt)
        )

    isa_sub = _isa(subwarp_f16_call, torch.float16, tmp_path=tmp_path / "sub_f16", compile_only=True)
    assert "_fake16" not in isa_sub
    if is_wave64:
        # On wave64 CDNA, width=8 scan_with_aggregate uses one ds_swizzle_b32 for aggregate broadcast.
        assert _count(isa_sub, "ds_bpermute_b32", "ds_permute_b32") == 0
        assert "v_mbcnt_lo_u32_b32" in isa_sub and "v_mbcnt_hi_u32_b32" in isa_sub
    else:
        # On wave32 GFX10+/RDNA/CDNA5, width=8 stays 100% in VALU (row_xmask + row_share + mbcnt_lo).
        assert _count(isa_sub, *CROSS_LANE_LDS) == 0
        assert "row_xmask" in isa_sub
        assert "row_share" in isa_sub
        assert "v_mbcnt_lo_u32_b32" in isa_sub
        assert "v_mbcnt_hi_u32_b32" not in isa_sub
