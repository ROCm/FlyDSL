#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide prefix scans, partial groups and structured values.

Coverage includes built-in and callable operations, inclusive/exclusive
forms, aggregates, initial values, nested records.
Combined reduction/scan checks cover Boolean identities and 128-bit values.

The common width axis includes every supported power of two through the
target wave size. Each width test launches two physical warps so results
also expose scans that incorrectly cross logical group boundaries."""

from __future__ import annotations

import coop_warp_utils as checks
import pytest
from coop_common import WARP_WIDTHS, dtype_id, sample, wrap
from coop_test_utils import run_kernel

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


# int16 stands in for uint16: the runtime has no memref element type for
# torch.uint16.
SCAN_DTYPES = (
    (fx.Uint8, "torch.uint8"),
    (fx.Int16, "torch.int16"),
    (fx.Int32, "torch.int32"),
    (fx.Int64, "torch.int64"),
)


def width_id(width):
    return "full_warp" if width is None else f"w{width}"


def run_warp_scan(values, name, *, width, inclusive, block):
    """Scan *values* with one ``warp_inclusive/exclusive_scan`` per lane."""

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        form = fx.coop.warp_inclusive_scan if inclusive else fx.coop.warp_exclusive_scan
        Out[tid] = form(A[tid], fx.ReductionOp.ADD, width=width)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.zeros_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


def expected_scan(values, name, *, width, inclusive):
    """The per-group running fold, wrapped at the dtype's own width.

    Nothing crosses a group boundary, so the reference scans each row of a
    ``(-1, width)`` reshape on its own.
    """
    host = values.cpu().to(torch.int64).reshape(-1, width)
    widened = host.cumsum(1)
    if not inclusive:
        widened = widened - host
    return wrap(widened.reshape(-1), name)


# ── sum scan ──────────────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", SCAN_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_sum(entry, width, inclusive):
    """Each group of *width* lanes scans on its own."""
    _, name = entry
    warp_threads = fx.num_warp_threads()
    group = warp_threads if width is None else width
    block = 2 * warp_threads
    values = sample(name, block)

    out = run_warp_scan(values, name, width=width, inclusive=inclusive, block=block)
    assert torch.equal(out, expected_scan(values, name, width=group, inclusive=inclusive))


# ── combined inclusive/exclusive scan ─────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
def test_combination_scan(width):
    """Both forms come out of one scan and agree with the two separate ones."""
    name = "torch.int32"
    warp_threads = fx.num_warp_threads()
    group = warp_threads if width is None else width
    block = 2 * warp_threads

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Inc: fx.Tensor, Exc: fx.Tensor):
        tid = fx.thread_idx.x
        inclusive, exclusive = fx.coop.warp_scan(A[tid], fx.ReductionOp.ADD, width=width)
        Inc[tid] = inclusive
        Exc[tid] = exclusive

    @flyc.jit
    def launch(A: fx.Tensor, Inc: fx.Tensor, Exc: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Inc, Exc).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = sample(name, block)
    inc = torch.zeros_like(values)
    exc = torch.zeros_like(values)
    launch(values, inc, exc, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    assert torch.equal(inc.cpu(), expected_scan(values, name, width=group, inclusive=True))
    assert torch.equal(exc.cpu(), expected_scan(values, name, width=group, inclusive=False))


# ── warp aggregate ────────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", SCAN_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_warp_aggregate(entry, width, inclusive):
    """The scan still holds, and every lane reads its own group's total."""
    _, name = entry
    warp_threads = fx.num_warp_threads()
    group = warp_threads if width is None else width
    block = 2 * warp_threads

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        scanned, exclusive, aggregate = fx.coop.warp_scan_with_aggregate(A[tid], fx.ReductionOp.ADD, width=width)
        Out[tid] = scanned if inclusive else exclusive
        Agg[tid] = aggregate

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = sample(name, block)
    out = torch.zeros_like(values)
    agg = torch.zeros_like(values)
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), expected_scan(values, name, width=group, inclusive=inclusive))

    # The aggregate is the group's whole fold, and every lane of the group has it.
    totals = values.cpu().to(torch.int64).reshape(-1, group).sum(1)
    assert torch.equal(agg.cpu(), wrap(totals.repeat_interleave(group), name))


# ── array-based scan with an initial value ────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", WARP_WIDTHS, ids=width_id)
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_initial_value(width, inclusive):
    """*init* folds in ahead of the group, so lane 0 sees it and nothing else."""
    INIT = 3
    name = "torch.int32"
    warp_threads = fx.num_warp_threads()
    group = warp_threads if width is None else width
    block = 2 * warp_threads

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        form = fx.coop.warp_inclusive_scan if inclusive else fx.coop.warp_exclusive_scan
        Out[tid] = form(A[tid], fx.ReductionOp.ADD, width=width, init=fx.Int32(INIT))

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = sample(name, block)
    out = torch.zeros_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = expected_scan(values, name, width=group, inclusive=inclusive).to(torch.int64) + INIT
    assert torch.equal(out.cpu(), wrap(expected, name))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_initial_value_stays_out_of_the_aggregate():
    """*init* seeds the scan but is not part of the aggregate."""
    INIT = 3
    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        scanned, _, aggregate = fx.coop.warp_scan_with_aggregate(A[tid], fx.ReductionOp.ADD, init=fx.Int32(INIT))
        Out[tid] = scanned
        Agg[tid] = aggregate

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    values = torch.ones(warp_threads, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(values)
    agg = torch.zeros_like(values)
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # The scan carries init; the aggregate is the inputs alone.
    assert torch.equal(out.cpu(), torch.arange(1, warp_threads + 1, dtype=torch.int32, device="cpu") + INIT)
    assert torch.equal(agg.cpu(), torch.full((warp_threads,), warp_threads, dtype=torch.int32, device="cpu"))


# ── from test_coop.py ─────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_warp_scan_forms_agree():
    """The warp-scope API on its own, without a block wrapped around it."""
    WIDTH = 16  # narrower than a warp, so the lane mask has to respect the group
    warp_threads = fx.num_warp_threads()

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Inc: fx.Tensor, Exc: fx.Tensor):
        tid = fx.thread_idx.x
        inclusive, exclusive = fx.coop.warp_scan(A[tid], fx.ReductionOp.ADD, width=WIDTH)
        Inc[tid] = inclusive
        Exc[tid] = exclusive

    @flyc.jit
    def launch(A: fx.Tensor, Inc: fx.Tensor, Exc: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Inc, Exc).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    values = torch.arange(1, warp_threads + 1, dtype=torch.float32, device="cuda")
    inc = torch.zeros_like(values)
    exc = torch.zeros_like(values)
    launch(values, inc, exc, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Each group of WIDTH lanes scans on its own; nothing crosses the boundary.
    expected = values.cpu().reshape(-1, WIDTH).cumsum(1).reshape(-1)
    assert torch.equal(inc.cpu(), expected)
    assert torch.equal(exc.cpu(), expected - values.cpu())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_warp_width_defaults_to_the_whole_warp():
    """Leaving *width* out spans exactly one warp, whatever wave size the target has."""
    warp_threads = fx.num_warp_threads()
    BLOCK = 2 * warp_threads  # two warps, so a width that overran one would show

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, Total: fx.Tensor, Inc: fx.Tensor):
        tid = fx.thread_idx.x
        Total[tid] = fx.coop.warp_reduce(A[tid], fx.ReductionOp.ADD)
        Inc[tid] = fx.coop.warp_inclusive_scan(A[tid], fx.ReductionOp.ADD)

    @flyc.jit
    def launch(A: fx.Tensor, Total: fx.Tensor, Inc: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Total, Inc).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = torch.arange(1, BLOCK + 1, dtype=torch.int32, device="cuda")
    total = torch.zeros_like(values)
    inc = torch.zeros_like(values)
    launch(values, total, inc, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    per_warp = values.cpu().reshape(-1, warp_threads)
    assert torch.equal(inc.cpu(), per_warp.cumsum(1, dtype=torch.int32).reshape(-1))
    assert torch.equal(total.cpu()[::warp_threads], per_warp.sum(1, dtype=torch.int32))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("valid", [0, 1, 5, 8])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_warp_valid_counts_seed_broadcast(universal, valid, default_device):
    with torch.device(default_device):
        checks.check_warp_valid_counts_seed_broadcast(universal, valid)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("valid", [1, 5, 8])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_warp_partial_scan_semigroup_without_identity(valid, default_device):
    with torch.device(default_device):
        checks.check_warp_partial_scan_semigroup_without_identity(valid)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy", ["warp"])
@pytest.mark.parametrize("op", [fx.ReductionOp.MIN, fx.ReductionOp.MAX])
@pytest.mark.parametrize("valid", [0, 5])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_boolean_partial_identities(policy, op, valid, default_device):
    with torch.device(default_device):
        checks.check_boolean_partial_identities(policy, op, valid)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [fx.Int128, fx.Uint128])
@pytest.mark.parametrize("op", [fx.ReductionOp.ADD, fx.ReductionOp.MIN, fx.ReductionOp.MAX])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_warp_128_bit_reduce_scan(dtype, op, default_device):
    with torch.device(default_device):
        checks.check_warp_128_bit_reduce_scan(dtype, op)


@pytest.mark.rocm_lower
@pytest.mark.l2_device
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["scan"])
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_nested_record_scan(case, universal, default_device):
    with torch.device(default_device):
        checks.check_nested_record_warp_collectives(case, universal)


ScanPair = fx.Struct["first" : fx.Int32, "second" : fx.Int32]


def _counted_scan(universal, form, prefix, entry, valid):
    namespace = fx.coop.universal if universal else fx.coop
    count = 1 if form == "scalar" else 2
    stride = 3 * count + 1

    def apply(a, out):
        tid = fx.thread_idx.x
        out[tid * stride + 3 * count] = fx.Int32(0)

        def add(lhs, rhs):
            # Count calls even when their numerical results would be discarded.
            out[tid * stride + 3 * count] = out[tid * stride + 3 * count] + 1
            if form == "record":
                return ScanPair(lhs.first + rhs.first, lhs.second + rhs.second)
            return lhs + rhs

        items = tuple(a[16 + tid * count + i] for i in range(count))
        value = (
            items[0] if form == "scalar" else fx.Vector.from_elements(items) if form == "vector" else ScanPair(*items)
        )
        limit = a[tid // 4] if valid == "runtime" else valid
        initial = (
            (ScanPair(fx.Int32(10), fx.Int32(10)) if form == "record" else fx.Int32(10)) if prefix == "init" else None
        )
        result = getattr(namespace, entry)(
            value,
            add,
            width=4,
            valid_items=limit,
            init=initial,
        )
        results = (result,) if entry in ("warp_inclusive_scan", "warp_exclusive_scan") else result
        for slot, scanned in enumerate(results):
            if form == "record":
                scanned = (scanned.first, scanned.second)
            for i in range(count):
                out[tid * stride + slot * count + i] = scanned if form == "scalar" else scanned[i]

    return apply


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("form,prefix", [("scalar", "none"), ("vector", "init"), ("record", "init")])
@pytest.mark.parametrize(
    "entry", ["warp_inclusive_scan", "warp_exclusive_scan", "warp_scan", "warp_scan_with_aggregate"]
)
@pytest.mark.parametrize("valid", [0, 1, 3, 4, "runtime"])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_warp_scan_skips_invalid_operators(universal, form, prefix, entry, valid, default_device):
    with torch.device(default_device):
        block, width = 64, 4
        count = 1 if form == "scalar" else 2
        limits = [0, 1, 3, 4] * 4 if valid == "runtime" else [valid] * 16
        host = torch.arange(1, block * count + 1, dtype=torch.int32, device="cpu")
        values = torch.cat((torch.tensor(limits, dtype=torch.int32, device="cpu"), host)).cuda()
        actual = run_kernel(
            _counted_scan(universal, form, prefix, entry, valid), values, block * (3 * count + 1), block
        )
        actual = actual.reshape(-1, width, 3 * count + 1)
        grouped = host.reshape(-1, width, count)
        seed = 0 if prefix == "none" else 10
        for group, limit in enumerate(limits):
            assert torch.count_nonzero(actual[group, limit:, -1]).item() == 0
            if limit == 0:
                continue
            inclusive = grouped[group, :limit].cumsum(0, dtype=torch.int32)
            exclusive = inclusive - grouped[group, :limit]
            start = 1 if prefix == "none" else 0
            if entry != "warp_exclusive_scan":
                torch.testing.assert_close(actual[group, :limit, :count], inclusive + seed)
            if entry != "warp_inclusive_scan":
                slot = 0 if entry == "warp_exclusive_scan" else count
                torch.testing.assert_close(actual[group, start:limit, slot : slot + count], exclusive[start:] + seed)
            if entry == "warp_scan_with_aggregate":
                torch.testing.assert_close(actual[group, :, 2 * count : 3 * count], inclusive[-1].expand(width, -1))


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
@pytest.mark.parametrize("form,prefix", [("scalar", "none"), ("vector", "init"), ("record", "init")])
def test_warp_scan_valid_compile(monkeypatch, arch, form, prefix):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    apply = _counted_scan(False, form, prefix, "warp_scan_with_aggregate", "runtime")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(torch.empty(144, dtype=torch.int32, device="cpu"), torch.empty(448, dtype=torch.int32, device="cpu"))
