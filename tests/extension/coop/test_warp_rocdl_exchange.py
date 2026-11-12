#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm overrides of ``WarpExchange`` (SHUFFLE) and ``warp_merge_sort``.

``coop/warp/rocdl/exchange.py`` replaces the SHUFFLE gather -- ``width**2``
shuffles, because the register a lane needs depends on its lane id -- with a
``log2(width)``-stage butterfly transpose, DPP within a 16-lane row.
``coop/warp/rocdl/merge_sort.py`` sorts one item per lane by ranking every key
in the other run and pushing it to its slot. As in ``test_warp_rocdl.py``, two
things are checked:

- **The same result.** A transpose and a stable sort have exactly one correct
  output, so the dispatched ``fx.coop`` form is compared with a host reference,
  word for word. The portable forms, and the widths, policies and record items
  ``test_warp_exchange.py`` / ``test_warp_merge_sort.py`` already run through
  the override, are not repeated here.
- **The override is the code that runs.** The cross-lane instructions in the
  ISA are counted exactly; a sequence that quietly fell back would still pass
  the comparisons.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from coop_test_utils import matrix_cases
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.expr.primitive import range_constexpr
from flydsl.extension.coop.warp.exchange import WarpExchangeAlgorithm

try:
    import torch
except ImportError:
    torch = None

SHUFFLE = WarpExchangeAlgorithm.SHUFFLE
portable = fx.coop.universal

requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_gfx9 = pytest.mark.skipif(not current_target().arch.startswith("gfx9"), reason="the overrides are gfx9-only")


def _skip_wider_than_the_wave(width):
    if width > current_target().warp_size:
        pytest.skip("the logical width exceeds the target wave")


def _launch(kernel, *tensors, block):
    """Launch a five-tensor kernel on one block; kernels that need fewer repeat an input."""

    @flyc.jit
    def launch(
        a: fx.Tensor, b: fx.Tensor, c: fx.Tensor, d: fx.Tensor, e: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(a, b, c, d, e).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(*tensors, stream=stream)
    stream.synchronize()


# ── the class a ROCm specialization resolves to ─────────────────────────────


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_rocm_specializes_the_override_classes():
    from flydsl.extension.coop.warp import rocdl

    assert {"WarpExchange", "warp_merge_sort"} <= set(rocdl.__all__)
    fast = fx.coop.WarpExchange[fx.Int32, 8, 8, SHUFFLE]
    slow = portable.WarpExchange[fx.Int32, 8, 8, SHUFFLE]
    assert issubclass(fast, rocdl.WarpExchange)
    assert not issubclass(slow, rocdl.WarpExchange)
    assert fx.coop.WarpExchange[fx.Int32, 8, 8, SHUFFLE] is fast
    # WarpMergeSort.sort reaches warp_merge_sort through _invoke, so the class
    # only needs the dispatcher.
    assert fx.coop.WarpMergeSort._dispatcher is not None


# ── WarpExchange: the same result ───────────────────────────────────────────

# kind -> (dtype, make(Int32) -> item, observe(item) -> [Int32 words]). One kind
# per way an item becomes 32-bit words: a narrow scalar widens, a 64-bit one
# splits, a Vector moves component by component. Int32 and records already run
# through the override in ``test_warp_exchange.py``.
_HALVES = fx.Vector[fx.Int16, 3]


def _lo_hi(x):
    return [x.to(fx.Int32), (x >> 32).to(fx.Int32)]


KINDS = {
    "u8": (fx.Uint8, lambda x: (x * 7).to(fx.Uint8), lambda v: [v.to(fx.Int32)]),
    "i64": (fx.Int64, lambda x: (x.to(fx.Int64) << 37) + 17, _lo_hi),
    "vec_i16x3": (
        _HALVES,
        lambda x: _HALVES(fx.Vector.from_elements([x.to(fx.Int16), (x * 3 + 1).to(fx.Int16), (-x).to(fx.Int16)])),
        lambda v: [v[i].to(fx.Int32) for i in range(3)],
    ),
}
WORDS = {"u8": 1, "i64": 2, "vec_i16x3": 3}


def _observe(value, out, tid, count, words, observe):
    items = [value[i] for i in range(count)]
    for i in range(count):
        for j, word in enumerate(observe(items[i])):
            out[(tid * count + i) * words + j] = word


def _run_exchange(width, kind, block=128):
    """(input, blocked_to_striped, round trip) words, shaped (block, width, words)."""
    dtype, make, observe = KINDS[kind]
    count, words = width, WORDS[kind]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, orig: fx.Tensor, fast: fx.Tensor, back: fx.Tensor, unused: fx.Tensor):
        tid = fx.thread_idx.x
        items = [make(a[tid * count + i]) for i in range_constexpr(count)]
        value = fx.Vector.from_elements(items) if fx.const_expr(issubclass(dtype, fx.Numeric)) else tuple(items)
        _observe(value, orig, tid, count, words, observe)
        P = fx.coop.WarpExchange[dtype, width, count, SHUFFLE]
        result = P.blocked_to_striped(value)
        _observe(result, fast, tid, count, words, observe)
        _observe(P.striped_to_blocked(result), back, tid, count, words, observe)

    torch.manual_seed(width)
    a = torch.randint(-(2**20), 2**20, (block * count,), dtype=torch.int32, device="cuda")
    outputs = [torch.full((block * count * words,), -99, dtype=torch.int32, device="cuda") for _ in range(3)]
    _launch(kernel, a, *outputs, a, block=block)
    return [o.cpu().reshape(block, count, words) for o in outputs]


def _host_transpose(words, width):
    block, count, n = words.shape
    return words.reshape(block // width, width, count, n).transpose(1, 2).reshape(block, count, n)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
# Width 64 takes every butterfly stage (DPP in a row, ds_swizzle, ds_bpermute);
# width 8 puts several logical warps in each row.
@pytest.mark.parametrize("width,kind", [(64, "i64"), (8, "u8"), (8, "vec_i16x3")])
@pytest.mark.usefixtures("warp_default_device")
def test_exchange_matches_host(width, kind):
    """Both directions: the round trip checks ``striped_to_blocked`` against the input."""
    _skip_wider_than_the_wave(width)
    orig, fast, back = _run_exchange(width, kind)
    assert torch.equal(fast, _host_transpose(orig, width))
    assert torch.equal(back, orig), "the two directions are not each other's inverse"


# ── the override is the code that runs ──────────────────────────────────────

_DPP = re.compile(
    r"\b(quad_perm:|row_shl:|row_shr:|row_ror:|row_bcast:|row_newbcast:|wave_(shl|shr|rol|ror):|row_mirror\b|row_half_mirror\b)"
)


def _isa(call, count, *, tmp_path):
    """Final ISA of a one-wave kernel applying *call* to *count* Int32 values per lane."""
    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, b: fx.Tensor, c: fx.Tensor, d: fx.Tensor):
        tid = fx.thread_idx.x
        result = call([A[tid * count + i] for i in range_constexpr(count)])
        for i in range_constexpr(count):
            Out[tid * count + i] = result[i]

    values = torch.zeros(warp_threads * count, dtype=torch.int32, device="cuda")
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        _launch(kernel, values, torch.zeros_like(values), values, values, values, block=warp_threads)
    finally:
        for k, v in previous.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    return dumped[0].read_text()


def _cross_lane(isa):
    lines = [line.strip() for line in isa.split("s_endpgm")[0].splitlines()]
    lines = [line for line in lines if line and not line.startswith((";", "."))]
    count = {op: sum(line.startswith(op + " ") for line in lines) for op in ("ds_swizzle_b32", "ds_bpermute_b32")}
    count["ds_permute_b32"] = sum(line.startswith("ds_permute_b32 ") for line in lines)
    count["dpp"] = sum(bool(_DPP.search(line)) for line in lines)
    count["lds"] = count["ds_swizzle_b32"] + count["ds_bpermute_b32"] + count["ds_permute_b32"]
    return count


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_gfx9
def test_exchange_is_the_butterfly(tmp_path):
    """``width * min(log2 width, 4)`` DPP moves, and ``width / 2`` LDS-pipe moves per cross-row stage.

    Stages 1..8 stay inside a 16-lane row; 16 is ``ds_swizzle`` SWAP16 and 32
    the one ``ds_bpermute`` stage. A full wave takes every one of them.
    """
    width = 64
    P = fx.coop.WarpExchange[fx.Int32, width, width, SHUFFLE]
    count = _cross_lane(_isa(lambda v: P.blocked_to_striped(fx.Vector.from_elements(v)), width, tmp_path=tmp_path))
    assert count["dpp"] == width * 4
    assert count["ds_swizzle_b32"] == width // 2
    assert count["ds_bpermute_b32"] == width // 2
    assert count["ds_permute_b32"] == 0


# ── warp_merge_sort: the same result ────────────────────────────────────────
#
# Only one item per lane takes the rank path; ``test_warp_merge_sort.py`` already
# runs it at every width, with record keys and through ``WarpMergeSort``. What
# the ranks add -- ties under a strict weak order, validity, payloads -- is here.

COMPARE = {"desc": lambda a, b: a > b, "default": None, "mod5": lambda a, b: a % 5 < b % 5}
HOST_KEY = {"desc": lambda k: -k, "default": lambda k: k, "mod5": lambda k: k % 5}


def _run_sort(width, valid, compare, payload, block=128):
    runtime = valid == "runtime"
    static = None if runtime else valid
    nvalid = width - 3 if width > 3 else 1
    op = COMPARE[compare]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, v: fx.Tensor, out: fx.Tensor, c: fx.Tensor, d: fx.Tensor):
        tid = fx.thread_idx.x
        keys = fx.Vector.from_elements([a[tid]])
        ids = fx.Vector.from_elements([tid])
        valid_items = v[0] if fx.const_expr(runtime) else static
        result = fx.coop.warp_merge_sort(
            keys, ids if payload else None, width=width, compare_op=op, valid_items=valid_items
        )
        k, p = result if payload else (result, ids)
        out[tid] = k[0]
        out[block + tid] = p[0]

    torch.manual_seed(width)
    # Few distinct keys, so ties are everywhere; non-negative so % agrees with the host.
    a = torch.randint(0, 23, (block,), dtype=torch.int32, device="cuda")
    v = torch.tensor([nvalid], dtype=torch.int32, device="cuda")
    out = torch.full((2 * block,), -99, dtype=torch.int32, device="cuda")
    _launch(kernel, a, v, out, a, a, block=block)
    n = nvalid if runtime else (width if static is None else static)
    return *out.cpu().reshape(2, -1, width), a.cpu().reshape(-1, width), n


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize(
    "width,valid,compare,payload",
    # Width 64 takes every kind of pass: DPP pairs and quads, swizzled and bpermuted searches.
    matrix_cases([8, 64], ["runtime", None, 5], list(COMPARE), [True, False], independent=True),
)
@pytest.mark.usefixtures("warp_default_device")
def test_merge_sort_matches_host(width, valid, compare, payload):
    _skip_wider_than_the_wave(width)
    fk, fv, keys, n = _run_sort(width, valid, compare, payload)
    order = HOST_KEY[compare](keys[:, :n].long()).argsort(dim=1, stable=True)
    assert torch.equal(fk[:, :n], keys[:, :n].gather(1, order))
    if payload:
        base = torch.arange(keys.shape[0], device="cpu")[:, None] * width
        assert torch.equal(fv[:, :n].long(), order + base)


# ── merge sort: the rank path is the code that runs ─────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_gfx9
def test_one_item_merge_sort_ranks_and_scatters(tmp_path):
    """Pairs and quads are DPP; wider runs cost one search gather per step and one push per value.

    Each search starts at the middle of the other run, a lane pattern fixed at
    trace time, and LLVM turns that first gather into ``ds_swizzle`` whenever
    the pattern stays inside 32 lanes (runs 4..16). A full wave has every run.
    """
    width = 64
    runs = [1 << k for k in range(1, width.bit_length() - 1)]  # 2 .. width/2
    searched = [run for run in runs if run >= 4]
    swizzled = [run for run in searched if run <= 16]

    def call(v):
        k, p = fx.coop.warp_merge_sort(
            fx.Vector.from_elements(v[:1]), fx.Vector.from_elements(v[1:]), width=width, valid_items=width - 3
        )
        return [k[0], p[0]]

    count = _cross_lane(_isa(call, 2, tmp_path=tmp_path))
    assert count["dpp"] >= 1
    assert count["ds_permute_b32"] == 2 * len(runs)
    assert count["ds_swizzle_b32"] == len(swizzled)
    assert count["ds_bpermute_b32"] == sum(run.bit_length() for run in searched) - len(swizzled)
