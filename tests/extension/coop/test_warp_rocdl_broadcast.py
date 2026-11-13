#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm ``warp_broadcast`` override, and the scan aggregate that shares its moves.

``coop/warp/rocdl/scan.py`` replaces the ``ds_bpermute_b32`` behind a narrow
broadcast on gfx90a+ (``row_newbcast`` is the control most of it rests on):

- a constant source lane is one DPP move (``quad_perm`` for 2 and 4 lanes,
  ``row_newbcast`` for 16), two (8 lanes; 32 lanes reading their first row;
  on gfx950 32 lanes reading their second row, ``row_newbcast`` and
  ``v_permlane16_swap_b32``) or one ``ds_swizzle_b32`` (32 lanes reading their
  second row elsewhere);
- a runtime source lane picks between itself and its neighbour by DPP for a
  pair, and otherwise keeps ``ds_bpermute_b32`` without the bounds check;
- a full wave stays portable, which LLVM already turns into ``v_readlane_b32``.

Every value moves word by word, so the dtype axis matters here in a way it does
not for the folds: 8/16-bit values ride in one 32-bit word, 64-bit values take
``v_mov_b64_dpp``, and records, pointers and Vectors move leaf by leaf.

As in ``test_warp_rocdl.py``, each case runs the dispatched ``fx.coop`` form
and ``fx.coop.universal`` in one kernel and checks them against each other and
against a host gather, on four waves so a move that leaked across a group or a
wave would show; the ISA checks show the sequences are the code that runs.
``warp_scan_with_aggregate`` reuses the single-move broadcasts for its
aggregate at 2, 4 and 16 lanes, and on gfx950 the ``row_newbcast`` plus
``v_permlane16_swap_b32`` pair at 32; it keeps ``ds_swizzle_b32`` at 8 lanes,
and at 32 elsewhere. Its values, and that no returned move folds into the
caller's subtraction, are checked in ``test_warp_rocdl.py``. A full wave is
the portable form and is only checked to stay that way.
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
from flydsl.expr.primitive import inttoptr, ptrtoint

torch = pytest.importorskip("torch")

portable = fx.coop.universal
BLOCK = 256  # four waves
CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


def _has_row_newbcast():
    # Mirrors ``warp/rocdl/_common._has_row_newbcast``, the override's own gate.
    return current_target().arch.startswith(("gfx90a", "gfx94", "gfx95"))


def width_id(width):
    return "full" if width is None else f"w{width}"


# The widths the override serves; the full wave is the portable form.
DPP_WIDTHS = tuple(width for width in WARP_WIDTHS if width is not None)


def _group(width):
    return fx.num_warp_threads() if width is None else width


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
]
requires_dpp = pytest.mark.skipif(not _is_gfx9(), reason="the DPP sequence is gfx9-only")
requires_permlane_swap = pytest.mark.skipif(
    not current_target().arch.startswith("gfx950"), reason="v_permlane16_swap is gfx950-only"
)
requires_newbcast = pytest.mark.skipif(not _has_row_newbcast(), reason="row_newbcast is gfx90a+")


# ── value kinds: how a lane builds its value, and how it is stored back ─────
#
# ``torch``/``n``: the input dtype and items per lane. ``store`` returns the
# values to write per result; ``rows`` > 1 widens each item on the host side,
# and ``host`` maps the gathered input to those rows.


def _scalar(fx_dtype, torch_dtype):
    return dict(torch=torch_dtype, n=1, build=lambda items: fx_dtype(items[0]), store=lambda v: [v])


def _vector(torch_dtype, n):
    return dict(
        torch=torch_dtype,
        n=n,
        build=lambda items: fx.Vector.from_elements(items),
        store=lambda v: [v[i] for i in range(n)],
    )


Record = fx.Struct["x" : fx.Int16, "y" : fx.Int64]

# One kind per way a value becomes words, ordered so the diagonal pairs the
# interesting ones: i8 padded into a quad move, i64 split across the banked
# 8-lane pair of moves, v4f32 as two ``v_mov_b64_dpp``, v3f16 padded to 64 bits
# through both 32-lane routes. f32 is the case every constant lane runs.
KINDS = {
    "i8": _scalar(fx.Int8, torch.int8),
    "f16": _scalar(fx.Float16, torch.float16),
    "i64": _scalar(fx.Int64, torch.int64),
    "v4f32": _vector(torch.float32, 4),
    "v3f16": _vector(torch.float16, 3),
    # i1: the low bit of an i32, stored back widened.
    "bool": dict(
        torch=torch.int32,
        n=1,
        build=lambda items: (items[0] & 1) == 1,
        store=lambda v: [v.select(fx.Int32(1), fx.Int32(0))],
        host=lambda x: x & 1,
    ),
    "f32": _scalar(fx.Float32, torch.float32),
    # The item-list form of the API: two i32 come back as a Vector.
    "list_i32": dict(torch=torch.int32, n=2, build=lambda items: list(items), store=lambda v: [v[0], v[1]]),
    "record": dict(
        torch=torch.int64,
        n=1,
        build=lambda items: Record(items[0].to(fx.Int16), items[0]),
        store=lambda v: [v.x.to(fx.Int64), v.y],
        rows=2,
        host=lambda x: torch.stack([x.to(torch.int16).to(torch.int64), x], dim=-1),
    ),
}


def _inputs(kind):
    count = BLOCK * kind["n"]
    if kind["torch"].is_floating_point:
        return torch.randn(count, device="cuda").to(kind["torch"])
    info = torch.iinfo(kind["torch"])
    low, high = max(info.min, -(2**40)), min(info.max, 2**40)
    return torch.randint(low, high, (count,), dtype=kind["torch"], device="cuda")


def _run(kind, width, sources, call, src_dtype=torch.int32):
    """Broadcast from every source in *sources* (ints, or "rt" for a per-lane one).

    *call(namespace, value, lane)* runs once through ``fx.coop`` and once
    through ``fx.coop.universal``. Returns host tensors (inputs, src, fast,
    ref); fast/ref are [len(sources), BLOCK, items * rows].
    """
    n, build, store = kind["n"], kind["build"], kind["store"]
    per = n * kind.get("rows", 1)
    count = len(sources)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, S: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        tid = fx.thread_idx.x
        value = build([A[tid * n + j] for j in range(n)])
        for i, source in enumerate(sources):
            lane = S[tid] if source == "rt" else source
            for out, namespace in ((Fast, fx.coop), (Ref, portable)):
                for j, item in enumerate(store(call(namespace, value, lane))):
                    out[(i * BLOCK + tid) * per + j] = item

    @flyc.jit
    def launch(A: fx.Tensor, S: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, S, Fast, Ref).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = _inputs(kind)
    src = torch.randint(0, _group(width), (BLOCK,), dtype=src_dtype, device="cuda")
    out_dtype = torch.int64 if "rows" in kind else kind["torch"]
    fast = torch.zeros(count * BLOCK * per, dtype=out_dtype, device="cuda")
    ref = torch.zeros_like(fast)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(values, src, fast, ref, stream=stream)
    stream.synchronize()
    # Compare on the host: device-side comparisons after a custom stream can hang.
    shape = (count, BLOCK, per)
    return values.cpu(), src.cpu(), fast.cpu().reshape(shape), ref.cpu().reshape(shape)


def _check(kind, width, sources, src_dtype=torch.int32):
    def call(namespace, value, lane):
        return namespace.warp_broadcast(value, lane, width=width)

    values, src, fast, ref = _run(kind, width, sources, call, src_dtype)
    group = _group(width)
    lane = torch.arange(BLOCK, device="cpu")
    per_lane = values.reshape(BLOCK, kind["n"])
    for i, source in enumerate(sources):
        lanes = (lane // group) * group + (src.to(torch.int64) if source == "rt" else source)
        gathered = per_lane[lanes]
        expected = kind["host"](gathered).reshape(BLOCK, -1) if "host" in kind else gathered
        assert torch.equal(fast[i], ref[i]), f"override != portable for source {source}"
        assert torch.equal(fast[i], expected.to(fast.dtype)), f"wrong lanes for source {source}"


def _spread(width):
    group = _group(width)
    return sorted({0, 1, group // 2 - 1, group // 2, group - 1}) + ["rt"]


# ── the override and the portable form agree ────────────────────────────────


@pytest.mark.parametrize("width", DPP_WIDTHS, ids=width_id)
@pytest.mark.usefixtures("warp_default_device")
def test_every_constant_lane(width):
    """Every source lane of every width: each one is its own DPP control or swizzle offset."""
    _check(KINDS["f32"], width, list(range(_group(width))) + ["rt"])


@pytest.mark.parametrize(
    "kind,width",
    matrix_cases(tuple(kind for kind in KINDS if kind != "f32"), DPP_WIDTHS),
    ids=lambda v: width_id(v) if isinstance(v, int) else v,
)
@pytest.mark.usefixtures("warp_default_device")
def test_every_dtype_moves_word_by_word(kind, width):
    """8/16-bit values pad into one word, wider ones split; records and lists move leaf by leaf."""
    _check(KINDS[kind], width, _spread(width))


@pytest.mark.parametrize("width", (2, 8), ids=width_id)
@pytest.mark.usefixtures("warp_default_device")
def test_a_64_bit_runtime_source(width):
    """A wide index is promoted as in the portable form, on both the DPP pair and the bpermute route."""
    _check(KINDS["f32"], width, ["rt"], src_dtype=torch.int64)


@pytest.mark.parametrize("width", (4, 32), ids=width_id)
@pytest.mark.usefixtures("warp_default_device")
def test_a_pointer_moves_as_its_address(width):
    """Pointers travel as their 64-bit address: one ``v_mov_b64_dpp`` route, one split swizzle."""
    elem = 4

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        tid = fx.thread_idx.x
        base = ptrtoint(A.iter).to(fx.Uint64)
        mine = inttoptr(A.iter.type, base + (tid * elem).to(fx.Uint64))
        for out, namespace in ((Fast, fx.coop), (Ref, portable)):
            moved = namespace.warp_broadcast(mine, width - 1, width=width)
            out[tid] = ((ptrtoint(moved) - base) // elem).to(fx.Int32)

    @flyc.jit
    def launch(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Fast, Ref).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = torch.zeros(BLOCK, dtype=torch.int32, device="cuda")
    fast, ref = torch.zeros_like(values), torch.zeros_like(values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(values, fast, ref, stream=stream)
    stream.synchronize()
    fast, ref = fast.cpu(), ref.cpu()
    lane = torch.arange(BLOCK, dtype=torch.int32, device="cpu")
    assert torch.equal(fast, ref)
    assert torch.equal(fast, (lane // width) * width + width - 1)


def _check_one_lane_aggregate():
    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor):
        tid = fx.thread_idx.x
        Fast[tid] = fx.coop.warp_scan_with_aggregate(A[tid], fx.ReductionOp.ADD, width=1)[2]
        Ref[tid] = portable.warp_scan_with_aggregate(A[tid], fx.ReductionOp.ADD, width=1)[2]

    @flyc.jit
    def launch(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Fast, Ref).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = torch.randint(-(2**20), 2**20, (BLOCK,), dtype=torch.int32, device="cuda")
    fast, ref = torch.zeros_like(values), torch.zeros_like(values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(values, fast, ref, stream=stream)
    stream.synchronize()
    assert torch.equal(fast.cpu(), ref.cpu())
    assert torch.equal(fast.cpu(), values.cpu())


@pytest.mark.parametrize("form", ("constant", "aggregate"))
@pytest.mark.usefixtures("warp_default_device")
def test_a_one_lane_group_reads_itself(form):
    """Regression: width 1 once took the pair's ``quad_perm[0,0,2,2]``, so lane 1 read lane 0.

    ``WARP_WIDTHS`` starts at 2, so this width needs its own case: a one-lane
    group is legal, and every lane must get its own value back.
    """
    if form == "aggregate":
        _check_one_lane_aggregate()
    else:
        _check(KINDS["f32"], 1, [0])


def test_a_constant_lane_outside_the_group_is_rejected():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(A: fx.Tensor):
        A[fx.thread_idx.x] = fx.coop.warp_broadcast(A[fx.thread_idx.x], 8, width=8)

    @flyc.jit
    def launch(A: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    with pytest.raises(ValueError, match="inside the logical warp"):
        launch(torch.zeros(64, dtype=torch.int32, device="cuda"), stream=torch.cuda.Stream())


# ── the override is actually the code that runs ────────────────────────────


def _isa(call, torch_dtype, tmp_path):
    """Compile *call(value, source)* in a lone wave and return its final ISA up to ``s_endpgm``."""
    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, S: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = call(A[tid], S[tid])

    @flyc.jit
    def launch(A: fx.Tensor, S: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, S, Out).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    values = torch.ones(warp_threads, dtype=torch_dtype, device="cuda")
    src = torch.zeros(warp_threads, dtype=torch.int32, device="cuda")
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        stream = torch.cuda.Stream()
        launch(values, src, torch.zeros_like(values), stream=stream)
        stream.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})

    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    return dumped[0].read_text().split("s_endpgm")[0]


def _count(isa, *needles):
    return sum(line.count(needle) for line in isa.splitlines() for needle in needles)


# (width, source lane): the DPP controls that must appear, and how many LDS
# cross-lane instructions the broadcast may use.
CONSTANT_ISA = {
    (2, 1): (["quad_perm:[1,1,3,3]"], 0),
    (4, 2): (["quad_perm:[2,2,2,2]"], 0),
    (8, 5): (["row_newbcast:5", "row_newbcast:13"], 0),
    (16, 9): (["row_newbcast:9"], 0),
    (32, 3): (["row_newbcast:3", "row_bcast:15"], 0),
    (32, 31): (["swizzle(BROADCAST,32,31)"], 1),
}
# gfx950 reads the second row of a 32-lane group without LDS.
if current_target().arch.startswith("gfx950"):
    CONSTANT_ISA[(32, 31)] = (["row_newbcast:15", "v_permlane16_swap_b32"], 0)


@requires_newbcast
@pytest.mark.parametrize("width,lane", list(CONSTANT_ISA), ids=lambda v: str(v))
def test_a_constant_lane_is_dpp_or_one_swizzle(width, lane, tmp_path):
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, lane, width=width), torch.float32, tmp_path)
    needles, lds = CONSTANT_ISA[(width, lane)]
    for needle in needles:
        assert needle in isa, f"missing {needle}"
    assert _count(isa, *CROSS_LANE_LDS) == lds
    assert "ds_bpermute_b32" not in isa


@requires_newbcast
def test_wide_and_narrow_values(tmp_path):
    """64-bit values take one ``v_mov_b64_dpp``; a 16-bit one rides one zero-extended 32-bit move."""
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, 3, width=16), torch.int64, tmp_path / "i64")
    assert _count(isa, "v_mov_b64_dpp") == 1 and _count(isa, *CROSS_LANE_LDS) == 0
    # ds_swizzle (gfx950: v_permlane16_swap) is 32-bit only, so the same value splits in two there.
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, 20, width=32), torch.int64, tmp_path / "swizzle")
    mover = "v_permlane16_swap_b32" if current_target().arch.startswith("gfx950") else "ds_swizzle_b32"
    assert _count(isa, mover) == 2 and "ds_bpermute_b32" not in isa
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, 3, width=16) + v, torch.float16, tmp_path / "f16")
    assert _count(isa, "row_newbcast:3") == 1 and _count(isa, *CROSS_LANE_LDS) == 0


@requires_dpp
def test_a_full_wave_stays_readlane(tmp_path):
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, 63), torch.float32, tmp_path)
    assert "v_readlane_b32" in isa and _count(isa, *CROSS_LANE_LDS) == 0


@requires_newbcast
def test_a_runtime_pair_picks_its_neighbour_by_dpp(tmp_path):
    isa = _isa(lambda v, s: fx.coop.warp_broadcast(v, s, width=2), torch.int32, tmp_path)
    assert "quad_perm:[1,0,3,2]" in isa and _count(isa, *CROSS_LANE_LDS) == 0


@requires_newbcast
@pytest.mark.parametrize("width", (4, 32), ids=width_id)
def test_a_runtime_source_drops_the_bounds_select(width, tmp_path):
    """One bpermute, without the generic lowering's compare + cndmask on the address."""
    fast = _isa(lambda v, s: fx.coop.warp_broadcast(v, s, width=width), torch.int32, tmp_path)
    assert _count(fast, "ds_bpermute_b32") == 1
    assert _count(fast, "v_cmp_") == 0 and _count(fast, "v_cndmask") == 0


@requires_newbcast
def test_the_warp_scan_member_reaches_the_override(tmp_path):
    isa = _isa(lambda v, s: fx.coop.WarpScan[fx.Float32, 16].broadcast(v, 3), torch.float32, tmp_path)
    assert "row_newbcast:3" in isa and _count(isa, *CROSS_LANE_LDS) == 0


# width: the instruction that names the group's last lane, and the LDS
# cross-lane instructions the whole scan may then use (the scan itself has none).
# One width per route: quad_perm (2 and 4), row_newbcast (16), ds_swizzle (8 and 32;
# gfx950 takes 32 with row_newbcast and v_permlane16_swap).
AGGREGATE_ISA = (
    pytest.param(2, "quad_perm:[1,1,3,3]", 0, marks=requires_dpp, id="w2"),
    pytest.param(8, "swizzle(BROADCAST,8,7)", 1, marks=requires_dpp, id="w8"),
    pytest.param(16, "row_newbcast:15", 0, marks=requires_newbcast, id="w16"),
    pytest.param(32, "v_permlane16_swap_b32", 0, marks=requires_permlane_swap, id="w32"),
)


@pytest.mark.parametrize("width,needle,lds", AGGREGATE_ISA)
def test_the_scan_aggregate_broadcast(width, needle, lds, tmp_path):
    """A pair, a quad and a row broadcast the aggregate by DPP; 8 and 32 lanes keep one swizzle."""

    def call(v, s):
        inclusive, exclusive, aggregate = fx.coop.warp_scan_with_aggregate(v, fx.ReductionOp.ADD, width=width)
        return inclusive + exclusive + aggregate

    isa = _isa(call, torch.float32, tmp_path)
    assert needle in isa
    assert _count(isa, *CROSS_LANE_LDS) == lds
