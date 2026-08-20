#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide reduction.

Covered below: the sum reduction over the dtype list, in both of the
implemented algorithms. Out of reach for lack of API surface: a partial tile —
every thread of the block always contributes — an arbitrary callable as the
reduction op (*op* is a ``ReductionOp``, see ``coop/_common.py``), and vector
or user-defined element types.

``BlockReduce`` requires a power-of-two thread count (``coop/block/_spec.py``);
a block that does not fill a wave narrows the logical warp to itself rather
than being refused, so the widths below run from one thread up.
``RAKING_COMMUTATIVE_ONLY`` and ``WARP_REDUCTIONS_NONDETERMINISTIC`` are named
by the enum but not implemented, so the algorithm axis is the two policies that
are.

The tests at the bottom came from ``test_coop.py`` when the algorithm tests
were split out by algorithm.
"""

from __future__ import annotations

import pytest
from coop_common import (
    BLOCK_THREADS,
    DTYPES,
    SUB_WARP_BLOCK_THREADS,
    dtype_id,
    is_float,
    linear_tid,
    sample,
    torch_dtype,
    wrap,
)

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


ALGORITHMS = (
    fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS,
    fx.coop.BlockReduceAlgorithm.RAKING,
)

# The legal powers of two on a 64-lane wave, in both a 1-D and a 3-D block
# shape.
BLOCK_SHAPES = ((64, 1, 1), (128, 1, 1), (256, 1, 1), (64, 2, 2), (128, 2, 2))


def run_block_reduce(values, name, dtype, *, block_size, algorithm, items_per_thread=1, op=None, universal=False):
    """Reduce *values* with one ``BlockReduce`` call per thread.

    Returns the per-thread results on the host: the reduction is valid
    block-wide, so every entry should hold the same total.

    *universal* picks ``fx.coop.universal.BlockReduce``, the form that folds
    through the portable warp reduction, over the dispatched one.
    """
    op = op if op is not None else fx.ReductionOp.ADD
    block_threads = block_size[0] * block_size[1] * block_size[2]

    # Decide the per-thread read before tracing: one scalar, or a Vector of
    # several per-thread items.
    if items_per_thread == 1:

        def read(A, tid):
            return A[tid]

    else:

        def read(A, tid):
            return fx.Vector.from_elements([A[tid * items_per_thread + i] for i in range(items_per_thread)])

    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        namespace = fx.coop.universal if universal else fx.coop
        block_reduce = namespace.BlockReduce[dtype, block_size, algorithm]
        storage = fx.SharedAllocator().allocate(block_reduce.SharedStorage).peek()
        tid = linear_tid(block_size)
        Out[tid] = block_reduce(read(A, tid), op, storage=storage)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    out = torch.zeros(block_threads, dtype=torch_dtype(name), device="cuda")
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


def check_sum(values, out, name):
    """Every thread holds the whole-block sum, folded the way the dtype folds."""
    host = values.cpu()
    assert torch.equal(out, out[0].expand(out.numel())), "the result must be valid in every thread"

    if is_float(name):
        tol = dict(rel=1e-12, abs=1e-9) if torch_dtype(name) == torch.float64 else dict(rel=1e-5, abs=1e-3)
        assert float(out[0]) == pytest.approx(float(host.sum(dtype=torch.float64)), **tol)
    else:
        assert out[0] == wrap(host.to(torch.int64).sum(), name)


# ── sum reduction ─────────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_sum_over_a_full_tile(entry, items_per_thread, algorithm):
    """A full tile sums to the same value every thread can read."""
    dtype, name = entry
    block_size = (128, 1, 1)
    values = sample(name, 128 * items_per_thread)

    out = run_block_reduce(
        values, name, dtype, block_size=block_size, algorithm=algorithm, items_per_thread=items_per_thread
    )
    check_sum(values, out, name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_size", BLOCK_SHAPES, ids=lambda b: "x".join(map(str, b)))
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_sum_across_block_shapes(block_size, algorithm):
    """The block shape axis: the same sum, however the block is shaped."""
    dtype, name = fx.Int32, "torch.int32"
    block_threads = block_size[0] * block_size[1] * block_size[2]
    values = sample(name, block_threads)

    out = run_block_reduce(values, name, dtype, block_size=block_size, algorithm=algorithm)
    check_sum(values, out, name)


# ── from test_coop.py ─────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
@pytest.mark.parametrize(
    "op,reference",
    (
        (fx.ReductionOp.ADD, torch.sum if torch else None),
        (fx.ReductionOp.MAX, torch.max if torch else None),
        (fx.ReductionOp.MIN, torch.min if torch else None),
    ),
    ids=lambda v: getattr(v, "name", ""),
)
def test_block_reduce_matches_torch(algorithm, op, reference):
    """Every thread of the block sees the exact whole-block reduction."""
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32, device="cuda")
    out = run_block_reduce(values, "torch.float32", fx.Float32, block_size=(256, 1, 1), algorithm=algorithm, op=op)

    expected = reference(values.cpu())
    assert torch.equal(out, out[0].expand(256)), "result must be valid in every thread"
    assert float(out[0]) == pytest.approx(float(expected), rel=1e-5)


# What these two add over the sum cases above is the *compare/select* combining
# path — float and integer take different lowerings — so neither is covered by
# `test_sum_over_a_full_tile[items_per_thread=4]` (``Vector`` input under ADD)
# or `test_sum_across_block_shapes` (int32 under ADD).


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_block_reduce_over_vector_items(algorithm):
    """A per-thread ``Vector`` reduces exactly like the flat block would."""
    ITEMS = 4
    BLOCK = 256
    torch.manual_seed(0)
    values = torch.randn(BLOCK * ITEMS, dtype=torch.float32, device="cuda")
    out = run_block_reduce(
        values,
        "torch.float32",
        fx.Float32,
        block_size=(BLOCK, 1, 1),
        algorithm=algorithm,
        items_per_thread=ITEMS,
        op=fx.ReductionOp.MAX,
    )

    assert torch.equal(out, out[0].expand(BLOCK))
    assert float(out[0]) == pytest.approx(float(values.cpu().max()), rel=1e-6)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_block_reduce_integer_max(algorithm):
    """Integer max goes through the compare/select path, not the float one."""
    values = torch.randint(-1000, 1000, (256,), dtype=torch.int32, device="cuda")
    out = run_block_reduce(
        values, "torch.int32", fx.Int32, block_size=(256, 1, 1), algorithm=algorithm, op=fx.ReductionOp.MAX
    )

    assert torch.equal(out, values.cpu().max().expand(256))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_threads", BLOCK_THREADS, ids=lambda n: f"t{n}")
def test_sum_across_every_legal_block_width(block_threads):
    """One thread up to the 1024-thread launch limit — the whole legal range.

    What the widths below a wave add is the narrowed logical warp, which folds
    over the launched lanes alone; what the widths above 256 add is the
    per-warp fold running over 8 and 16 slots rather than 2 or 4, and a launch
    at the limit the hardware imposes. Held to one dtype and the default
    algorithm on purpose: the axis under test is the width, and crossing it
    with the others would only re-run them.
    """
    dtype, name = fx.Int32, "torch.int32"
    values = sample(name, block_threads)

    out = run_block_reduce(
        values,
        name,
        dtype,
        block_size=(block_threads, 1, 1),
        algorithm=fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS,
    )
    check_sum(values, out, name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_universal_agrees_with_the_dispatched_reduction(algorithm):
    """The portable warp fold underneath reaches the same total as the target's.

    A block reduction folds through warp scope, so ``fx.coop.universal`` is the
    only way to run one over the portable warp reduction at all — and this is
    the only coverage that path has on a target that overrides it. Integers
    wrap the same either way, so the two results are equal outright.
    """
    dtype, name = fx.Int32, "torch.int32"
    values = sample(name, 256)
    shared = dict(block_size=(256, 1, 1), algorithm=algorithm)

    dispatched = run_block_reduce(values, name, dtype, **shared)
    universal = run_block_reduce(values, name, dtype, universal=True, **shared)

    assert torch.equal(universal, dispatched)
    check_sum(values, universal, name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_single_warp_block_skips_shared_memory():
    """A one-warp block reduces entirely in registers."""
    warp_threads = fx.num_warp_threads()
    out = run_block_reduce(
        torch.ones(warp_threads, dtype=torch.float32, device="cuda"),
        "torch.float32",
        fx.Float32,
        block_size=(warp_threads, 1, 1),
        algorithm=fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS,
    )
    assert torch.equal(out, torch.full((warp_threads,), float(warp_threads)))


# ── sub-wave blocks ───────────────────────────────────────────────────────


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("block_threads", SUB_WARP_BLOCK_THREADS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_a_sub_wave_block_narrows_its_logical_warp(block_threads, algorithm):
    """The warp the collective folds over is the block, not the target's wave.

    This is what keeps every cross-lane read inside the lanes the launch
    actually started: a wave-wide fold in a block that only fills part of the
    wave would read lanes that were never launched. Whichever algorithm is
    named, the block is then a single warp, which is the condition each of
    them short-circuits on.
    """
    block_reduce = fx.coop.BlockReduce[fx.Int32, block_threads, algorithm]

    assert block_reduce.warp_threads == block_threads
    assert block_reduce.num_warps == 1


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_threads", SUB_WARP_BLOCK_THREADS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.name)
def test_sum_in_a_sub_wave_block(block_threads, algorithm):
    """Both algorithms still sum a block that occupies part of a wave.

    The width sweep above runs one algorithm; this one crosses the sub-wave
    widths with both, since the short-circuit each takes at a single warp is
    separate code. The narrowest case is a single thread, where the fold has
    no cross-lane step left at all.
    """
    dtype, name = fx.Int32, "torch.int32"
    values = sample(name, block_threads)

    out = run_block_reduce(values, name, dtype, block_size=(block_threads, 1, 1), algorithm=algorithm)
    check_sum(values, out, name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_multi_dimensional_block():
    """A 2-D block linearizes its thread id the way gpu.thread_id orders it."""
    DIM_X, DIM_Y = 64, 2
    BLOCK = DIM_X * DIM_Y
    values = torch.ones(BLOCK, dtype=torch.float32, device="cuda")
    out = run_block_reduce(
        values,
        "torch.float32",
        fx.Float32,
        block_size=(DIM_X, DIM_Y, 1),
        algorithm=fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS,
    )
    assert torch.equal(out, torch.full((BLOCK,), float(BLOCK)))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_block_size_can_come_from_the_traced_launch():
    """``known_block_size`` feeds the specialization, so the two cannot drift apart."""
    BLOCK = 128

    @flyc.kernel
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        block_reduce = fx.coop.BlockReduce[fx.Float32, fx.known_block_size()]
        assert block_reduce.block_threads == BLOCK
        storage = fx.SharedAllocator().allocate(block_reduce.SharedStorage).peek()
        tid = fx.thread_idx.x
        Out[tid] = block_reduce(A[tid], fx.ReductionOp.ADD, storage=storage)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    values = torch.ones(BLOCK, dtype=torch.float32, device="cuda")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="cuda")
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.full((BLOCK,), float(BLOCK)))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_dynamic_block_size_has_nothing_to_read():
    """A dynamic launch width leaves ``known_block_size`` with no answer to give."""

    @flyc.kernel
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        block_reduce = fx.coop.BlockReduce[fx.Float32, fx.known_block_size()]
        storage = fx.SharedAllocator().allocate(block_reduce.SharedStorage).peek()
        tid = fx.thread_idx.x
        Out[tid] = block_reduce(A[tid], fx.ReductionOp.ADD, storage=storage)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, nthreads: fx.Int32):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(nthreads, 1, 1))

    values = torch.ones(128, dtype=torch.float32, device="cuda")
    out = torch.zeros(128, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="no compile-time block size"):
        launch(values, out, 128)
