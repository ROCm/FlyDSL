#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide prefix scan.

Covered below: the sum scan over the dtype list at one and at nine items per
thread, the block aggregate, and the initial-value form. Out of reach for lack
of API surface: an arbitrary callable as the scan op (*op* is a
``ReductionOp``, see ``coop/_common.py``), a callback carrying a running
prefix across calls, and vector or user-defined element types.

``BlockScan`` needs a power-of-two thread count (``coop/block/_spec.py``); a
block that does not fill a wave narrows the logical warp to itself rather than
being refused, so the widths below run from one thread up and
``items_per_thread`` is unconstrained. int16 stands in for uint16, which the
runtime cannot hand to a memref.

``RAKING`` and ``RAKING_MEMOIZE`` are named by the enum but not implemented, so
the algorithm axis collapses to ``WARP_SCANS``.

The tests at the bottom came from ``test_coop.py`` when the algorithm tests
were split out by algorithm.
"""

from __future__ import annotations

import pytest
from coop_common import (
    BLOCK_THREADS,
    SUB_WARP_BLOCK_THREADS,
    WARP_SIZE,
    dtype_id,
    is_float,
    linear_tid,
    sample,
    wrap,
)

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


# Integers only: a float scan's result depends on the order the partials are
# folded in, which is what the algorithm is free to choose.
SCAN_DTYPES = (
    (fx.Uint8, "torch.uint8"),
    (fx.Int16, "torch.int16"),
    (fx.Int32, "torch.int32"),
    (fx.Int64, "torch.int64"),
)

BLOCK_SHAPES = ((64, 1, 1), (128, 1, 1), (256, 1, 1), (64, 2, 2))


def run_block_scan(
    values, name, dtype, *, block_size, inclusive, items_per_thread=1, op=None, init=None, namespace=None
):
    """Scan *values* with one ``BlockScan`` call per thread; return the host result.

    *namespace* is which ``BlockScan`` to reach for — the dispatched one by
    default, or ``fx.coop.universal`` for the form that folds through the
    portable warp scan.
    """
    op = op if op is not None else fx.ReductionOp.ADD
    namespace = namespace if namespace is not None else fx.coop

    # Decide the per-thread read and write before tracing: one scalar, or a
    # Vector of several per-thread items.
    if items_per_thread == 1:

        def scan(block_scan, A, Out, tid, storage):
            form = block_scan.inclusive if inclusive else block_scan.exclusive
            Out[tid] = form(A[tid], op, storage=storage, init=init)

    else:

        def scan(block_scan, A, Out, tid, storage):
            base = tid * items_per_thread
            items = fx.Vector.from_elements([A[base + i] for i in range(items_per_thread)])
            form = block_scan.inclusive if inclusive else block_scan.exclusive
            out = form(items, op, storage=storage, init=init)
            for i in range(items_per_thread):
                Out[base + i] = out[i]

    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        block_scan = namespace.BlockScan[dtype, block_size, namespace.BlockScanAlgorithm.WARP_SCANS]
        storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
        scan(block_scan, A, Out, linear_tid(block_size), storage)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    out = torch.zeros_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


def check_scan(values, out, name, *, inclusive):
    """Compare against the running fold, wrapped at the dtype's own width."""
    host = values.cpu()
    if is_float(name):
        expected = host.cumsum(0)
        if not inclusive:
            expected = expected - host
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        return

    widened = host.to(torch.int64).cumsum(0)
    if not inclusive:
        widened = widened - host.to(torch.int64)
    assert torch.equal(out, wrap(widened, name))


# ── sum scan ──────────────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", SCAN_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("items_per_thread", (1, 9))
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_sum(entry, items_per_thread, inclusive):
    """Each thread's items fold in as if they sat consecutively in the block."""
    dtype, name = entry
    block_size = (128, 1, 1)
    values = sample(name, 128 * items_per_thread)

    out = run_block_scan(
        values, name, dtype, block_size=block_size, inclusive=inclusive, items_per_thread=items_per_thread
    )
    check_scan(values, out, name, inclusive=inclusive)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_size", BLOCK_SHAPES, ids=lambda b: "x".join(map(str, b)))
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_sum_across_block_shapes(block_size, inclusive):
    """The block shape axis: the same scan, however the block is shaped."""
    dtype, name = fx.Int32, "torch.int32"
    block_threads = block_size[0] * block_size[1] * block_size[2]
    values = sample(name, block_threads)

    out = run_block_scan(values, name, dtype, block_size=block_size, inclusive=inclusive)
    check_scan(values, out, name, inclusive=inclusive)


# ── block aggregate ───────────────────────────────────────────────────────

# 3 items per thread over a 64-wide block, in a 1-D and a 3-D block shape, plus
# a block that only fills half a wave. The sub-wave shape is here rather than
# only under the plain scan because the aggregate is broadcast from the last
# lane of the logical warp: read at the wave's width instead of the block's, it
# would come from a lane the launch never started.
AGGREGATE_SHAPES = ((64, 1, 1), (64, 2, 2), (WARP_SIZE // 2, 1, 1))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_size", AGGREGATE_SHAPES, ids=lambda b: "x".join(map(str, b)))
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_block_aggregate(block_size, inclusive):
    """The scan still holds, and every thread reads the whole block's total."""
    ITEMS = 3
    dtype, name = fx.Int32, "torch.int32"
    block_threads = block_size[0] * block_size[1] * block_size[2]

    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        block_scan = fx.coop.BlockScan[dtype, block_size]
        storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
        tid = linear_tid(block_size)
        base = tid * ITEMS
        items = fx.Vector.from_elements([A[base + i] for i in range(ITEMS)])
        form = block_scan.inclusive_with_aggregate if inclusive else block_scan.exclusive_with_aggregate
        out, aggregate = form(items, fx.ReductionOp.ADD, storage=storage)
        for i in range(ITEMS):
            Out[base + i] = out[i]
        Agg[tid] = aggregate

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    values = sample(name, block_threads * ITEMS)
    out = torch.zeros_like(values)
    agg = torch.zeros(block_threads, dtype=torch.int32, device="cuda")
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    check_scan(values, out.cpu(), name, inclusive=inclusive)

    total = values.cpu().to(torch.int64).sum()
    assert torch.equal(agg.cpu(), wrap(total.expand(block_threads), name))


# ── array-based scan with an initial value ────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("items_per_thread", (1, 3))
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_initial_value(items_per_thread, inclusive):
    """*init* folds in ahead of the block, so thread 0 sees it and nothing else."""
    INIT = 1
    name = "torch.int32"
    block_size = (64, 1, 1)
    values = sample(name, 64 * items_per_thread)

    out = run_block_scan(
        values,
        name,
        fx.Int32,
        block_size=block_size,
        inclusive=inclusive,
        items_per_thread=items_per_thread,
        init=fx.Int32(INIT),
    )

    widened = values.cpu().to(torch.int64).cumsum(0)
    if not inclusive:
        widened = widened - values.cpu().to(torch.int64)
    assert torch.equal(out, wrap(widened + INIT, name))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_initial_value_stays_out_of_the_aggregate():
    """*init* seeds the scan but is not part of the aggregate."""
    INIT = 1
    BLOCK = 128

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        block_scan = fx.coop.BlockScan[fx.Int32, BLOCK]
        storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        out, aggregate = block_scan.inclusive_with_aggregate(
            A[tid], fx.ReductionOp.ADD, storage=storage, init=fx.Int32(INIT)
        )
        Out[tid] = out
        Agg[tid] = aggregate

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = torch.ones(BLOCK, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(values)
    agg = torch.zeros_like(values)
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # The scan carries init; the aggregate is the inputs alone.
    assert torch.equal(out.cpu(), torch.arange(1, BLOCK + 1, dtype=torch.int32) + INIT)
    assert torch.equal(agg.cpu(), torch.full((BLOCK,), BLOCK, dtype=torch.int32))


# ── from test_coop.py ─────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_threads", BLOCK_THREADS, ids=lambda n: f"t{n}")
def test_scan_across_every_legal_block_width(block_threads):
    """One thread up to the 1024-thread launch limit — the whole legal range.

    What the widths below a wave add is the narrowed logical warp, which scans
    over the launched lanes alone; what the widths above 256 add is the
    per-warp prefix fold running over 8 and 16 slots rather than 2 or 4, and a
    launch at the limit the hardware imposes. Held to one dtype and the
    inclusive form on purpose: the axis under test is the width, and crossing
    it with the others would only re-run them.
    """
    dtype, name = fx.Int32, "torch.int32"
    values = sample(name, block_threads)

    out = run_block_scan(values, name, dtype, block_size=(block_threads, 1, 1), inclusive=True)
    check_scan(values, out, name, inclusive=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_universal_agrees_with_the_dispatched_scan(inclusive):
    """The portable warp scan underneath reaches the same prefixes as the target's.

    A block scan folds through warp scope, so ``fx.coop.universal`` is the only
    way to run one over the portable warp scan at all — and this is the only
    coverage that path has on a target that overrides it. Integers wrap the
    same either way, so the two results are equal outright.
    """
    dtype, name = fx.Int32, "torch.int32"
    values = sample(name, 256)
    shared = dict(block_size=(256, 1, 1), inclusive=inclusive)

    dispatched = run_block_scan(values, name, dtype, **shared)
    universal = run_block_scan(values, name, dtype, namespace=fx.coop.universal, **shared)

    assert torch.equal(universal, dispatched)
    check_scan(values, universal, name, inclusive=inclusive)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_block_scan_matches_torch(inclusive):
    """Each thread sees the fold of every thread before it, itself included or not."""
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32, device="cuda")
    out = run_block_scan(values, "torch.float32", fx.Float32, block_size=(256, 1, 1), inclusive=inclusive)

    expected = values.cpu().cumsum(0)
    if not inclusive:
        expected = expected - values.cpu()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ADD cannot stand in for MAX here: the point is an identity that is *visible*
# in the output, and ADD's is 0, which is indistinguishable from a thread that
# simply summed nothing. MAX's -inf is the only way to tell the two apart.


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_block_scan_max_starts_from_the_identity(inclusive):
    """MAX is where the exclusive form's identity shows: thread 0 gets -inf."""
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32, device="cuda")
    out = run_block_scan(
        values, "torch.float32", fx.Float32, block_size=(256, 1, 1), inclusive=inclusive, op=fx.ReductionOp.MAX
    )

    expected = values.cpu().cummax(0).values
    if not inclusive:
        expected = torch.cat([torch.full((1,), float("-inf")), expected[:-1]])
    assert torch.equal(out, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("inclusive", (True, False), ids=("inclusive", "exclusive"))
def test_block_scan_over_vector_items(inclusive):
    """A thread's Vector items scan as if they sat consecutively in the block."""
    ITEMS = 4
    BLOCK = 64
    values = torch.ones(BLOCK * ITEMS, dtype=torch.float32, device="cuda")
    out = run_block_scan(
        values,
        "torch.float32",
        fx.Float32,
        block_size=(BLOCK, 1, 1),
        inclusive=inclusive,
        items_per_thread=ITEMS,
    )

    expected = torch.arange(1, BLOCK * ITEMS + 1, dtype=torch.float32)
    if not inclusive:
        expected = expected - 1
    assert torch.equal(out, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_block_scan_single_warp_skips_shared_memory():
    """A one-warp block scans entirely in registers."""
    warp_threads = fx.num_warp_threads()
    values = torch.ones(warp_threads, dtype=torch.float32, device="cuda")
    out = run_block_scan(values, "torch.float32", fx.Float32, block_size=(warp_threads, 1, 1), inclusive=True)

    assert torch.equal(out, torch.arange(1, warp_threads + 1, dtype=torch.float32))


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("block_threads", SUB_WARP_BLOCK_THREADS, ids=lambda n: f"t{n}")
def test_a_sub_wave_block_narrows_its_logical_warp(block_threads):
    """The warp the scan folds over is the block, not the target's wave.

    Same contract as the reduction's, and for the same reason: a wave-wide
    scan in a block that only fills part of the wave would shuffle from lanes
    the launch never started.
    """
    block_scan = fx.coop.BlockScan[fx.Int32, block_threads]

    assert block_scan.warp_threads == block_threads
    assert block_scan.num_warps == 1


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_block_scan_integer_add():
    """Integers take the same path; the scan is exact rather than approximate."""
    values = torch.randint(-1000, 1000, (256,), dtype=torch.int32, device="cuda")
    out = run_block_scan(values, "torch.int32", fx.Int32, block_size=(256, 1, 1), inclusive=True)

    assert torch.equal(out, values.cpu().cumsum(0, dtype=torch.int32))
