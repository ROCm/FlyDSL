#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm override for the segmented warp reductions, and the portable form it displaces.

``coop/warp/rocdl/reduce.py`` replaces the portable suffix doubling -- a
``ds_bpermute_b32`` for the value and another for the segment-end flag at every
step -- with one ``ballot`` of the flags, SALU masks built from it, and DPP
moves of the values. As in ``test_warp_rocdl.py``, two things have to hold:

- **It computes the same thing.** Every case runs the dispatched
  ``fx.coop`` form and ``fx.coop.universal`` in one kernel over several flag
  patterns at once and checks both against a host fold at every segment's
  first lane (the only lane with a defined result). The patterns are the ones
  the sequence treats specially: a single flag on either side of each row
  (16-lane) and half-wave boundary, a segment running across three rows, every
  lane its own segment, and no flags at all. MAX runs on negative values and
  MIN on positive ones, so an identity leaking into a fold changes the result.
- **It is the code that runs.** Only the ISA tells a sequence that took from
  one that fell back: a ``ReductionOp`` reaches no cross-lane LDS instruction
  at any width, a callable only up to 16 lanes, and everything the sequence
  does not cover keeps the portable shuffles.

A callable keeps the portable form's contract: the op only ever sees two pieces
of one segment, and a one-lane segment never calls it. The rest of the contract
-- lane order for a non-commutative op, item ranges, the ``WarpReduce`` methods
-- already runs through the override in ``test_warp_reduce.py``,
``test_warp_item_containers.py`` and ``test_warp_operators.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from coop_test_utils import matrix_cases
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal
CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")
WIDTHS = (2, 4, 8, 16, 32, 64)
BLOCK = 256  # four waves per pattern, so a fold that leaked across a wave or a group shows
FORMS = ("tail", "head")


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


needs_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_dpp = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available() or not _is_gfx9(), reason="the DPP sequence is gfx9-only"
)


# -- flag patterns -----------------------------------------------------------------


def _at(*lanes):
    return [int(lane % 64 in lanes) for lane in range(BLOCK)]


# Each pattern runs as its own workgroup. Single flags sit on both sides of
# every row and half-wave boundary; 7 and 40 leave a segment spanning rows 0-2.
PATTERNS = {
    "none": [0] * BLOCK,
    "all": [1] * BLOCK,
    **{f"at{lane}": _at(lane) for lane in (15, 16, 31, 32, 47, 48)},
    "at7_40": _at(7, 40),
    "every5": [int(lane % 5 == 0) for lane in range(BLOCK)],
}


def _flags(names=None, dtype=torch.int32 if torch else None):
    names = list(PATTERNS) if names is None else names
    return names, torch.tensor([f for n in names for f in PATTERNS[n]], dtype=dtype, device="cpu")


def _segments(flags, width, head):
    """(first, end) lane ranges of every segment, in global lanes."""
    flags = flags.tolist()
    out = []
    for base in range(0, len(flags), width):
        f = flags[base : base + width]
        starts = [0] + [j for j in range(1, width) if f[j]] if head else [0] + [j + 1 for j in range(width - 1) if f[j]]
        out += [(base + s, base + e) for s, e in zip(starts, starts[1:] + [width])]
    return out


def _wrap32(x):
    return ((x + 2**31) % 2**32) - 2**31


HOST_FOLD = {
    "ADD": lambda a, b: _wrap32(a + b) if isinstance(a, int) else a + b,
    "MAX": max,
    "MIN": min,
}


def _reference(values, flags, width, head, fold):
    values = values.tolist()
    expected = {}
    for first, end in _segments(flags, width, head):
        acc = values[first]
        for x in values[first + 1 : end]:
            acc = fold(acc, x)
        expected[first] = acc
    return expected


def _values(opname, dtype, count, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = {"ADD": (-1000, 1000), "MAX": (-(2**20), -1), "MIN": (1, 2**20)}[opname]
    v = torch.randint(lo, hi, (count,), generator=gen, dtype=torch.int32, device="cpu")
    # Integer-valued floats keep every partial sum exact, whatever the fold order.
    return v.float() if dtype == "f32" else v


def _run_both(call, values, flags):
    """Run *call* through ``fx.coop`` and ``fx.coop.universal`` in one kernel; return CPU tensors."""
    grid = values.numel() // BLOCK

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, F: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        gid = fx.block_idx.x * BLOCK + fx.thread_idx.x
        Fast[gid] = call(fx.coop, A[gid], F[gid])
        Ref[gid] = call(portable, A[gid], F[gid])

    @flyc.jit
    def launch(A: fx.Tensor, F: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, F, Fast, Ref).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    a, f = values.to("cuda"), flags.to("cuda")
    fast, ref = torch.zeros_like(a), torch.zeros_like(a)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(a, f, fast, ref, stream=stream)
    stream.synchronize()
    return fast.cpu(), ref.cpu()


def _check_heads(expected, fast, ref, names):
    bad = [
        (names[lane // BLOCK], lane % BLOCK, fast[lane].item(), ref[lane].item(), want)
        for lane, want in expected.items()
        if fast[lane].item() != want or ref[lane].item() != want
    ]
    assert not bad, f"{len(bad)} wrong segment heads (pattern, lane, override, portable, expected): {bad[:6]}"


def _form(head, suffix="reduce"):
    return f"warp_{'head' if head else 'tail'}_segmented_{suffix}"


# -- the override agrees with the portable form ------------------------------------

OPS = (("ADD", "i32"), ("MAX", "i32"), ("ADD", "f32"), ("MIN", "i32"), ("MAX", "f32"))
# The diagonal visits every width, form and op; the cross-row widths get both
# forms and both dtypes explicitly.
AGREEMENT_CASES = tuple(
    dict.fromkeys(
        (
            *matrix_cases(WIDTHS, FORMS, OPS),
            (32, "head", ("ADD", "f32")),
            (64, "tail", ("ADD", "f32")),
            (64, "head", ("MAX", "i32")),
        )
    )
)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@needs_gpu
@pytest.mark.parametrize("width,form,op", AGREEMENT_CASES, ids=lambda v: "-".join(v) if isinstance(v, tuple) else None)
@pytest.mark.usefixtures("warp_default_device")
def test_reduction_ops_agree_with_the_portable_form(width, form, op):
    opname, dtype = op
    reduction = getattr(fx.ReductionOp, opname)
    head = form == "head"
    names, flags = _flags()
    values = _values(opname, dtype, flags.numel(), seed=width * 2 + head)

    fast, ref = _run_both(lambda m, x, f: getattr(m, _form(head))(x, f, reduction, width=width), values, flags)
    _check_heads(_reference(values, flags, width, head, HOST_FOLD[opname]), fast, ref, names)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@needs_gpu
@pytest.mark.parametrize("flag,form", matrix_cases((0, 1), FORMS))
@pytest.mark.usefixtures("warp_default_device")
def test_constant_flags(flag, form):
    """A Python-int flag: 0 is one segment per group, 1 makes every lane its own."""
    width, head = 32, form == "head"
    values = _values("ADD", "i32", BLOCK, seed=5)
    flags = torch.full((BLOCK,), flag, dtype=torch.int32, device="cpu")
    fast, ref = _run_both(
        lambda m, x, f: getattr(m, _form(head))(x, flag, fx.ReductionOp.ADD, width=width), values, flags
    )
    _check_heads(_reference(values, flags, width, head, HOST_FOLD["ADD"]), fast, ref, ["const"])


def _affine(lhs, rhs):
    """Composition over F_251 packed as (a << 16) | b: associative, not commutative."""
    a, b = (lhs >> 16 & 65535, lhs & 65535)
    c, d = (rhs >> 16 & 65535, rhs & 65535)
    return a * c % 251 << 16 | (a * d + b) % 251


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@needs_gpu
# The widths a callable takes the DPP sequence at; from 32 lanes it is portable.
@pytest.mark.parametrize("width,form,singletons", matrix_cases((16, 8), FORMS, (False, True), independent=True))
@pytest.mark.usefixtures("warp_default_device")
def test_a_callable_only_ever_sees_its_own_segment(width, form, singletons):
    """The portable contract (``test_warp_guards.py``), kept by the DPP sequence.

    The op counts its calls and its cross-segment calls through side effects,
    so evaluating it everywhere and selecting the result afterwards would be
    caught. Flags are ``1 << 32`` in an Int64, so truncating them would read
    them as zero.
    """
    head = form == "head"
    names = ["all"] if singletons else [n for n in PATTERNS if n != "all"]
    names, flags = _flags(names, dtype=torch.int64)
    values = torch.empty(flags.numel(), dtype=torch.int32, device="cpu")
    # Value = (segment id + 1) * 1000 + lane, so max is the segment's last lane.
    for seg, (first, end) in enumerate(_segments(flags, width, head)):
        values[first:end] = (seg + 1) * 1000 + torch.arange(first, end, dtype=torch.int32, device="cpu") % 64
    grid = flags.numel() // BLOCK

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, F: fx.Tensor, Out: fx.Tensor, Calls: fx.Tensor, Cross: fx.Tensor):
        gid = fx.block_idx.x * BLOCK + fx.thread_idx.x
        Calls[gid] = fx.Int32(0)
        Cross[gid] = fx.Int32(0)

        def maximum(lhs, rhs):
            Calls[gid] = Calls[gid] + 1
            Cross[gid] = Cross[gid] + fx.Int32(lhs // 1000 != rhs // 1000)
            return fx.max(lhs, rhs)

        Out[gid] = getattr(fx.coop, _form(head))(A[gid], F[gid] << 32, maximum, width=width)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        F: fx.Tensor,
        Out: fx.Tensor,
        Calls: fx.Tensor,
        Cross: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(A, F, Out, Calls, Cross).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    a, f = values.to("cuda"), flags.to("cuda")
    out, calls, cross = (torch.zeros_like(a) for _ in range(3))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(a, f, out, calls, cross, stream=stream)
    stream.synchronize()
    out, calls, cross = out.cpu(), calls.cpu(), cross.cpu()

    expected = _reference(values, flags, width, head, max)
    _check_heads(expected, out, out, names)
    assert torch.count_nonzero(cross).item() == 0, "the op saw two segments at once"
    if singletons:
        assert torch.count_nonzero(calls).item() == 0, "a one-lane segment called the op"
    else:
        assert calls.sum().item() > 0


# -- the override is the code that runs -----------------------------------------------


def _isa(call, torch_dtype, *, tmp_path):
    """Compile ``Out = call(A, F)`` alone on one wave and return its final ISA as text."""

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor, F: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = call(A[tid], F[tid])

    @flyc.jit
    def launch(A: fx.Tensor, F: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, F, Out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    values = torch.ones(64, dtype=torch_dtype, device="cuda")
    flags = torch.zeros(64, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(values)
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        stream = torch.cuda.Stream()
        launch(values, flags, out, stream=stream)
        stream.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    return dumped[0].read_text()


def _count(isa, *needles):
    return sum(line.count(needle) for line in isa.splitlines() for needle in needles)


def _portable_lds(width, head):
    """The portable form moves the value and the flag every step; its last flag move is dead."""
    steps = width.bit_length() - 1
    return 0 if steps == 0 else 2 * steps - 1 + int(head)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_dpp
@pytest.mark.parametrize("width,form,dtype", matrix_cases(WIDTHS, FORMS, ("i32", "f32")))
def test_a_reduction_op_stays_out_of_lds(width, form, dtype, tmp_path):
    """No cross-lane LDS at any width."""
    head = form == "head"
    op = fx.ReductionOp.ADD
    tdt = torch.int32 if dtype == "i32" else torch.float32
    fast = _isa(lambda x, f: getattr(fx.coop, _form(head))(x, f, op, width=width), tdt, tmp_path=tmp_path)

    assert _count(fast, *CROSS_LANE_LDS) == 0, "the override reached LDS"
    assert _count(fast, "row_shl:") == min(width.bit_length() - 1, 4)
    assert ("row_newbcast:15" in fast) == (width >= 32)
    assert _count(fast, "v_readlane_b32") == int(width == 64)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_dpp
@pytest.mark.parametrize("form", FORMS)
def test_the_class_api_takes_the_sequence(form, tmp_path):
    """The class methods reach the override too, not just agree with it."""
    P = fx.coop.WarpReduce[fx.Int32, 64]
    method = getattr(P, f"{form}_segmented_reduce")
    isa = _isa(lambda x, f: method(x, f, fx.ReductionOp.ADD), torch.int32, tmp_path=tmp_path)
    assert _count(isa, *CROSS_LANE_LDS) == 0
    assert "row_newbcast:15" in isa


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_dpp
@pytest.mark.parametrize("width", (16, 32))
def test_a_callable_takes_the_sequence_up_to_16_lanes(width, tmp_path):
    """From 32 lanes an expensive callable loses throughput to the portable form, so it keeps it."""
    isa = _isa(
        lambda x, f: fx.coop.warp_tail_segmented_reduce(x, f, _affine, width=width), torch.int32, tmp_path=tmp_path
    )
    if width <= 16:
        assert _count(isa, *CROSS_LANE_LDS) == 0
        assert _count(isa, "row_shl:") == width.bit_length() - 1
    else:
        assert "row_shl:" not in isa
        assert _count(isa, *CROSS_LANE_LDS) == _portable_lds(width, False)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_dpp
@pytest.mark.parametrize("dtype", ("int16", "float64"))
def test_a_dtype_the_sequence_cannot_move_falls_back(dtype, tmp_path):
    """DPP moves 32 bits; a narrower or wider value keeps the shuffles (a 64-bit one takes two each)."""
    op = fx.ReductionOp.ADD
    isa = _isa(
        lambda x, f: fx.coop.warp_tail_segmented_reduce(x, f, op, width=32), getattr(torch, dtype), tmp_path=tmp_path
    )
    assert "row_shl:" not in isa, "the sequence took a dtype it cannot move"
    assert _count(isa, *CROSS_LANE_LDS) >= _portable_lds(32, False)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_dpp
@pytest.mark.parametrize("width", (16, 32))
def test_before_gfx90a_rows_stay_but_crossing_them_is_portable(width, tmp_path, monkeypatch):
    """``row_newbcast`` is gfx90a+: an older gfx9 keeps the row steps but not the row crossing."""
    from flydsl.extension.coop.warp.rocdl import _common as rocdl_common

    class _Gfx908:
        arch = "gfx908"

    monkeypatch.setattr(rocdl_common, "current_target", lambda: _Gfx908)
    op = fx.ReductionOp.ADD
    isa = _isa(lambda x, f: fx.coop.warp_tail_segmented_reduce(x, f, op, width=width), torch.int32, tmp_path=tmp_path)
    assert "row_newbcast" not in isa
    if width < 32:
        assert _count(isa, *CROSS_LANE_LDS) == 0 and "row_shl:" in isa
    else:
        assert _count(isa, *CROSS_LANE_LDS) == _portable_lds(width, False)
