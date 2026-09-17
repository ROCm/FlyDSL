#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide prefix scan.

Covered below: the sum scan over the dtype list, the combined
``(inclusive, exclusive)`` pair, the warp aggregate, and the initial-value
form. Out of reach for lack of API surface: an arbitrary callable as the scan
op (*op* is a ``ReductionOp``, see ``coop/_common.py``), vector and
user-defined element types, and a partial scan — every lane of the logical
warp always participates.

The width axis is :data:`~coop_common.WARP_WIDTHS`: every width that
``resolve_warp_width`` accepts (``coop/_common.py``), which is the powers of
two up to the target's own wave. Each test launches two physical warps, so a
width that leaked past its group would show up as a scan crossing the
boundary.

The tests at the bottom came from ``test_coop.py`` when the algorithm tests
were split out by algorithm. ``warp_reduce`` has no file of its own: what
covers it is ``test_warp_rocdl.py``, which runs every fold through both the
dispatched override and the portable form and compares the two.
"""

from __future__ import annotations

import pytest
from coop_common import WARP_WIDTHS, dtype_id, sample, wrap

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
    assert torch.equal(out.cpu(), torch.arange(1, warp_threads + 1, dtype=torch.int32) + INIT)
    assert torch.equal(agg.cpu(), torch.full((warp_threads,), warp_threads, dtype=torch.int32))


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
    assert torch.equal(total.cpu(), per_warp.sum(1, dtype=torch.int32).repeat_interleave(warp_threads))
