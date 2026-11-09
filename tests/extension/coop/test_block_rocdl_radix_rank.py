#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm override of ``BlockRadixRank`` (``coop/block/rocdl/radix_rank.py``).

On gfx9 the MATCH policies build each item's warp match mask from one
``ballot`` per digit bit instead of one shuffle per lane, rank inside it with
``v_mbcnt_lo/hi``, and MATCH_EARLY_COUNTS_ANY publishes its early counts from
each digit's first lane. Two things are checked:

- **The same result.** Every case ranks the same keys with the dispatched
  ``fx.coop`` class and ``fx.coop.universal`` in one kernel and compares both
  with a host reference. Ranks are a permutation fixed by the keys, so they
  must agree word for word. Each case also subtracts the returned ranks from
  the keys in user code: on gfx942 LLVM folds a DPP move into a following
  subtraction incorrectly (``v_subrev_u32_dpp``), so a returned value must
  never be such a move.
- **The override is the code that runs.** The gfx9 kernel has no
  ``v_readlane`` fan-out and one ``v_mbcnt`` pair per item; another target
  compiles the portable shuffles.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from coop_test_utils import matrix_cases

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.expr.primitive import range_constexpr

try:
    import torch
except ImportError:
    torch = None

A = fx.coop.BlockRadixRankAlgorithm
MATCHES = [A.MATCH, A.MATCH_EARLY_COUNTS_ANY, A.MATCH_EARLY_COUNTS_ATOMIC_OR]
portable = fx.coop.universal

requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_gfx9 = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available() or not current_target().arch.startswith("gfx9"),
    reason="the override is gfx9-only",
)


def _launch(kernel, *tensors, block):
    @flyc.jit
    def launch(
        a: fx.Tensor, b: fx.Tensor, c: fx.Tensor, d: fx.Tensor, e: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(a, b, c, d, e).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(*tensors, stream=stream)
    stream.synchronize()


def _rank_class(ns, capacity):
    return ns.BlockRadixRank if capacity is None else ns.BlockRadixRank.with_radix_bits(capacity)


# ── the class a ROCm specialization resolves to ─────────────────────────────


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_rocm_specializes_the_override_class():
    from flydsl.extension.coop.block import rocdl

    assert "BlockRadixRank" in rocdl.__all__
    fast = fx.coop.BlockRadixRank[fx.Int32, 128, 3, A.MATCH]
    nine = fx.coop.BlockRadixRank.with_radix_bits(9)[fx.Int32, 128, 3, A.MATCH]
    slow = portable.BlockRadixRank[fx.Int32, 128, 3, A.MATCH]
    assert issubclass(fast, rocdl.BlockRadixRank) and issubclass(nine, rocdl.BlockRadixRank)
    assert nine.configured_radix_bits == 9
    assert not issubclass(slow, rocdl.BlockRadixRank)
    # The override's internal scan is the public, dispatched BlockScan.
    assert rocdl.BlockRadixRank.scan_type is fx.coop.BlockScan


# ── the same result ─────────────────────────────────────────────────────────


def _keys(total, dist, bits):
    torch.manual_seed(total + bits)
    keys = torch.randint(-(2**20), 2**20, (total,), dtype=torch.int32)
    if dist == "same":  # one digit for the whole tile
        keys = keys & ~((1 << bits) - 1)
    elif dist == "skew":  # nine in ten digits are 0
        keys = torch.where(torch.rand(total) < 0.9, keys & ~((1 << bits) - 1), keys)
    return keys


def _host_ranks(digits, block, count, striped, descending):
    """Ranks per physical (thread * count + item) position; MATCH policies read warp-striped tiles."""
    mapping = torch.arange(block * count)
    if striped:
        width = min(block, fx.num_warp_threads())
        mapping = torch.cat(
            [mapping[s * count : (s + width) * count].reshape(-1, count).T.flatten() for s in range(0, block, width)]
        )
    order = digits[mapping].argsort(stable=True, descending=descending)
    ranks = torch.empty(block * count, dtype=torch.int32)
    ranks[mapping[order]] = torch.arange(block * count, dtype=torch.int32)
    return ranks


def _run(alg, block, count, bits, descending, keys, *, capacity=None, with_prefix=False):
    """(dispatched ranks, universal ranks, keys - dispatched ranks, dispatched prefix | dispatched counts)."""
    counts_out = alg in MATCHES

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, fast: fx.Tensor, ref: fx.Tensor, diff: fx.Tensor, extra: fx.Tensor):
        tid = fx.thread_idx.x
        values = fx.Vector.from_elements([a[tid * count + i] for i in range_constexpr(count)])
        allocator = fx.SharedAllocator()
        P = _rank_class(fx.coop, capacity)[fx.Int32, block, count, alg]
        Q = _rank_class(portable, capacity)[fx.Int32, block, count, alg]
        seen = []
        options = dict(radix_bits=bits, descending=descending, with_prefix=with_prefix)
        if fx.const_expr(with_prefix and counts_out):
            options["count_callback"] = lambda c: seen.append(c)
        result = P(values, storage=allocator.allocate(P.SharedStorage).peek(), **options)
        fx.barrier()
        other = Q(values, storage=allocator.allocate(Q.SharedStorage).peek(), **options)
        ranks, prefix = result if fx.const_expr(with_prefix) else (result, None)
        reference = other[0] if fx.const_expr(with_prefix) else other
        for i in range_constexpr(count):
            fast[tid * count + i] = ranks[i]
            ref[tid * count + i] = reference[i]
            diff[tid * count + i] = values[i] - ranks[i]
        if fx.const_expr(with_prefix):
            extra[tid * 2] = values[0] - prefix[0]
            extra[tid * 2 + 1] = seen[0][0] if fx.const_expr(counts_out) else fx.Int32(0)

    a = keys.cuda()
    outputs = [torch.full((block * max(count, 2),), -7, dtype=torch.int32, device="cuda") for _ in range(4)]
    _launch(kernel, a, *outputs, block=block)
    return [o.cpu() for o in outputs]


def _check(alg, block, count, bits, descending, dist, **options):
    keys = _keys(block * count, dist, bits)
    fast, ref, diff, extra = _run(alg, block, count, bits, descending, keys, **options)
    n = block * count
    digits = keys.long() & ((1 << bits) - 1)
    expected = _host_ranks(digits, block, count, alg in MATCHES, descending)
    assert torch.equal(fast[:n], expected)
    assert torch.equal(ref[:n], expected)
    # A returned rank must stay a correct subtrahend in user code.
    assert torch.equal(diff[:n], keys - expected)
    return keys, digits, extra


SHAPES = [
    # block, items, bits, descending, distribution
    (256, 4, 8, False, "uniform"),
    (32, 3, 6, True, "skew"),
    (192, 3, 5, True, "uniform"),
    (64, 9, 4, False, "same"),
    (128, 1, 8, False, "skew"),
]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize(
    "alg,shape",
    list(matrix_cases(MATCHES, SHAPES, independent=True)) + [(A.BASIC, SHAPES[0]), (A.MEMOIZE, SHAPES[1])],
)
def test_ranks_match_universal_and_host(alg, shape):
    _check(alg, *shape)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("alg", [A.MATCH_EARLY_COUNTS_ANY, A.BASIC])
def test_nine_bit_capacity(alg):
    """``with_radix_bits`` dispatches too; digits above 255 in one pass."""
    _check(alg, 64, 3, 9, False, "uniform", capacity=9)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("alg,descending", [(A.MATCH_EARLY_COUNTS_ANY, True), (A.MATCH, False), (A.BASIC, True)])
def test_digit_offsets_and_counts(alg, descending):
    block, count, bits = 256, 4, 8
    keys, digits, extra = _check(alg, block, count, bits, descending, "skew", with_prefix=True)
    histogram = torch.bincount(digits, minlength=1 << bits).int()
    starts = histogram.sum() - histogram.cumsum(0) if descending else histogram.cumsum(0) - histogram
    # Thread t reports digit t: its offset (subtracted from its first key) and, for MATCH policies, its count.
    assert torch.equal(extra[0 : 2 * block : 2], keys[0 : block * count : count] - starts.int())
    if alg in MATCHES:
        assert torch.equal(extra[1 : 2 * block : 2], histogram)


# ── the override is the code that runs ──────────────────────────────────────


def _isa(ns, alg, *, tmp_path, block=128, count=4, bits=8):
    """Final ISA lines of a kernel that only ranks with *ns*."""

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor, b: fx.Tensor, c: fx.Tensor, d: fx.Tensor):
        tid = fx.thread_idx.x
        R = ns.BlockRadixRank[fx.Int32, block, count, alg]
        storage = fx.SharedAllocator().allocate(R.SharedStorage).peek()
        ranks = R(
            fx.Vector.from_elements([a[tid * count + i] for i in range_constexpr(count)]),
            storage=storage,
            radix_bits=bits,
        )
        for i in range_constexpr(count):
            out[tid * count + i] = ranks[i]

    values = torch.zeros(block * count, dtype=torch.int32, device="cuda")
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        _launch(kernel, values, torch.zeros_like(values), values, values, values, block=block)
    finally:
        for k, v in previous.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    lines = [line.strip() for line in dumped[0].read_text().split("s_endpgm")[0].splitlines()]
    return [line.split()[0] for line in lines if line and not line.startswith((";", ".")) and not line.endswith(":")]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gfx9
@pytest.mark.parametrize("alg", [A.MATCH, A.MATCH_EARLY_COUNTS_ANY])
def test_gfx9_matches_with_ballots(alg, tmp_path):
    ops = _isa(fx.coop, alg, tmp_path=tmp_path / "fast")
    # Eight ballots (compares into SGPR pairs) per item instead of 64 readlanes, and one mbcnt pair per item.
    assert sum(op.startswith(("v_readlane", "v_writelane")) for op in ops) == 0
    assert ops.count("v_mbcnt_lo_u32_b32") >= 4 and ops.count("v_mbcnt_hi_u32_b32") >= 4
    slow = _isa(portable, alg, tmp_path=tmp_path / "slow")
    assert sum(op.startswith("v_readlane") for op in slow) >= 4 * 64
    assert 2 * len(ops) < len(slow)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
def test_other_targets_keep_the_portable_match(monkeypatch, tmp_path):
    """On wave32 RDNA4 the override hands the match back to the portable shuffles."""
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    ops = _isa(fx.coop, A.MATCH, tmp_path=tmp_path, block=64)
    assert sum(op.startswith("v_readlane") for op in ops) >= 4 * 32
