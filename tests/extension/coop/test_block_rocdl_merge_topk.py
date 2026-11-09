#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The gfx9 ``BlockMergeSort`` / ``BlockTopK`` overrides against the portable classes.

``coop/block/rocdl/merge_sort.py`` replaces the per-thread runs and the
shared-memory merge passes with a bitonic network (DPP inside a warp, one
shared-memory exchange per two cross-warp stages) for 32-bit integer keys in
their numeric order; ``coop/block/rocdl/topk.py`` sorts MERGE with the
dispatched merge sort and counts the AIR threshold with wave ballots. Both
must agree with ``fx.coop.universal`` bit for bit, not merely sort:

- **Results** of the dispatched ``fx.coop`` classes equal the portable ones
  over the whole tile -- keys, payloads, and the selected flags -- and a host
  reference over the specified region. A stable sort of integer keys, and
  AIR's selection, are unique, so any correct override reproduces them.
- **Values handed back are safe operands**: an Int32 ``x - result`` matches
  the portable one. (gfx942 LLVM folds a bare DPP move into a following
  subtraction as ``v_subrev_u32_dpp`` and gets it wrong.)
- **The override is the code that runs**, which only the ISA shows: the
  network's DPP moves and fewer barriers. Calls it does not cover, and every
  non-gfx9 target, reach the portable implementation.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path

import pytest
from coop_test_utils import full_matrix_enabled

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None


portable = fx.coop.universal
AIR, MERGE = fx.coop.BlockTopKAlgorithm.AIR, fx.coop.BlockTopKAlgorithm.MERGE
KEY = fx.Struct["major" : fx.Int32, "minor" : fx.Int64]

requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_gfx9 = pytest.mark.skipif(
    not current_target().arch.startswith("gfx9"), reason="the DPP network and the ballots are gfx9-only"
)


def _threads(block):
    return block if isinstance(block, int) else math.prod(block)


def _run(apply, inputs, block, width, rows):
    """``apply(a, b, tid, ns, allocator) -> [[Int64 x width] x rows]`` for ``fx.coop`` and ``fx.coop.universal``.

    Returns a CPU int64 tensor ``[2, rows, threads * width]``: dispatched first.
    """
    shape = (block, 1, 1) if isinstance(block, int) else tuple(block)
    threads = _threads(block)

    def body(a, b, out):
        tid = fx.thread_idx.x + fx.thread_idx.y * shape[0] + fx.thread_idx.z * shape[0] * shape[1]
        allocator = fx.SharedAllocator()  # one per kernel, shared by both forms
        for n, ns in enumerate((fx.coop, portable)):
            for j, row in enumerate(apply(a, b, tid, ns, allocator)):
                for i in range(width):
                    out[((n * rows + j) * threads + tid) * width + i] = row[i]

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor):
        body(a, b, out)

    @flyc.jit
    def launch(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, b, out).launch(grid=(1, 1, 1), block=shape, stream=stream)

    out = torch.zeros(2 * rows * threads * width, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch(inputs[0], inputs[1], out, stream=stream)
    stream.synchronize()
    return out.cpu().reshape(2, rows, threads * width)


def _inputs(n, dist, seed):
    g = torch.Generator().manual_seed(seed)
    keys = (
        torch.randint(0, 6, (n,), generator=g)
        if dist == "dup"
        else torch.randint(-(1 << 31), 1 << 31, (n,), generator=g)
    )
    return keys.cuda(), (torch.arange(n) * 3 + 1).cuda()


def _bits(x):
    if x.dtype.is_float:
        return x.bitcast(fx.Int32 if x.dtype.width == 32 else fx.Int64).to(fx.Int64)
    return x.to(fx.Int64)


# Inputs travel as int64; a kind turns one into its key and its key back into int64 bits.
KINDS = {
    "i32": (fx.Int32, lambda x: x.to(fx.Int32)),
    "u32": (fx.Uint32, lambda x: x.to(fx.Int32).bitcast(fx.Uint32)),
    "i64": (fx.Int64, lambda x: x * 1000003),
    "f32": (fx.Float32, lambda x: ((x & 0xFFFF) % 5 - 2).to(fx.Float32) * 0.5),
    "record": (KEY, lambda x: KEY(major=((x & 0xFFFF) % 5).to(fx.Int32) - 2, minor=(x & 0xFFFF) // 5)),
}


def _host_key(kind, x):
    m = x & 0xFFFF
    return {
        "i32": lambda: x.to(torch.int32).to(torch.int64),
        "u32": lambda: x.to(torch.int32).to(torch.int64) & 0xFFFFFFFF,
        "i64": lambda: x * 1000003,
        "f32": lambda: (m % 5 - 2).to(torch.float32) * 0.5,
        "record": lambda: (m % 5 - 2) * (1 << 40) + m // 5,
    }[kind]()


def _record_before(a, b):
    return (a.major < b.major) | ((a.major == b.major) & (a.minor < b.minor))


def _columns(kind, key):
    return [key.major.to(fx.Int64), key.minor] if kind == "record" else [_bits(key)]


def _decode(kind, rows):
    """Device key columns as the host keys' values."""
    if kind == "record":
        return rows[0] * (1 << 40) + rows[1]
    if kind == "f32":
        return rows[0].to(torch.int32).view(torch.float32)
    if kind == "u32":
        return rows[0] & 0xFFFFFFFF
    return rows[0]


# ── BlockMergeSort ─────────────────────────────────────────────────────────

# block, count, kind, pairs, descending, valid ("none" | "static" | "runtime")
_MERGE_DEFAULT = [
    (256, 4, "i32", False, False, "none"),  # block network, min/max
    (256, 1, "i32", True, True, "none"),  # block network, (key, position) ties
    (192, 4, "u32", True, False, "none"),  # three warps: warp networks, then merge passes
    (64, 2, "i32", False, True, "none"),  # one wave: no exchange at all
]
_MERGE_FULL = [
    (256, 4, "i32", True, False, "none"),
    (32, 4, "i32", True, True, "none"),  # sub-wave block
    ((8, 4, 4), 4, "u32", False, False, "none"),
    (64, 8, "i32", True, False, "none"),
    (256, 2, "i64", True, False, "none"),
    (256, 4, "f32", True, True, "none"),
    (64, 3, "record", True, False, "static"),
]
MERGE_CASES = _MERGE_DEFAULT + (_MERGE_FULL if full_matrix_enabled() else [])


def _merge_apply(block, count, kind, pairs, descending, valid):
    key_dtype, make_key = KINDS[kind]
    threads = _threads(block)
    dtype = (key_dtype, fx.Int32) if pairs else key_dtype
    compare = _record_before if kind == "record" else None
    static = threads * count - 37 if valid == "static" else None

    def apply(a, b, tid, ns, allocator):
        S = ns.BlockMergeSort[dtype, block, count]
        storage = allocator.allocate(S.SharedStorage).peek()
        keys = [make_key(a[tid * count + i]) for i in range(count)]
        keys = keys if kind == "record" else fx.Vector.from_elements(keys)
        n = (a[0] - a[0] + threads * count - 41).to(fx.Int32) if valid == "runtime" else static
        kwargs = dict(storage=storage, descending=descending, compare_op=compare, valid_items=n)
        if valid == "runtime":
            kwargs["oob_default"] = 77
        if pairs:
            payload = fx.Vector.from_elements([b[tid * count + i].to(fx.Int32) for i in range(count)])
            out_keys, out_values = S(keys, payload, **kwargs)
        else:
            out_keys, out_values = S(keys, **kwargs), None
        rows = [list(r) for r in zip(*[_columns(kind, out_keys[i]) for i in range(count)])]
        if pairs:
            rows.append([out_values[i].to(fx.Int64) for i in range(count)])
        return rows

    return apply


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("block,count,kind,pairs,descending,valid", MERGE_CASES)
def test_merge_sort_matches_portable(block, count, kind, pairs, descending, valid):
    threads = _threads(block)
    n = threads * count
    limit = {"none": n, "static": n - 37, "runtime": n - 41}[valid]
    rows = (2 if kind == "record" else 1) + pairs
    apply = _merge_apply(block, count, kind, pairs, descending, valid)
    for dist in ("rand", "dup"):
        keys, values = _inputs(n, dist, seed=7)
        mine, ref = _run(apply, (keys, values), block, count, rows)
        assert torch.equal(mine, ref), f"fx.coop != fx.coop.universal ({dist})"
        host = _host_key(kind, keys.cpu())[:limit]
        order = host.argsort(descending=descending, stable=True)
        assert torch.equal(_decode(kind, mine)[:limit], host[order])
        if pairs:
            assert torch.equal(mine[-1][:limit], values.cpu()[:limit][order])


# ── BlockTopK ──────────────────────────────────────────────────────────────

# algorithm, block, count, kind, pairs, k, num_valid, largest
_TOPK_DEFAULT = [
    (AIR, 256, 4, "i32", False, "runtime", None, True),
    (AIR, 192, 3, "u32", True, 41, "runtime", False),
    (AIR, 64, 1, "i64", False, 17, None, True),
]
_TOPK_FULL = [
    (MERGE, 256, 4, "i32", True, 100, None, True),  # the merge network, through the dispatched sort
    (MERGE, 192, 3, "i32", False, 17, "runtime", False),
    (AIR, 256, 1, "i32", True, 0, None, False),
    (AIR, 256, 4, "u32", True, 1024, None, True),
    (AIR, 256, 1, "f32", True, 128, None, False),
    (AIR, 32, 4, "i32", True, 9, None, True),
    (AIR, 2, 1, "i32", False, 1, None, True),  # too few scratch words for the ballot counts: portable
]
TOPK_CASES = _TOPK_DEFAULT + (_TOPK_FULL if full_matrix_enabled() else [])


def _topk_apply(algorithm, block, count, kind, pairs, k, num_valid):
    key_dtype, make_key = KINDS[kind]
    threads = _threads(block)
    dtype = (key_dtype, fx.Int64) if pairs else key_dtype

    def apply(a, b, tid, ns, allocator, largest):
        T = ns.BlockTopK[dtype, block, count, algorithm]
        storage = allocator.allocate(T.SharedStorage).peek()
        keys = fx.Vector.from_elements([make_key(a[tid * count + i]) for i in range(count)])
        kk = (a[0] - a[0] + 61).to(fx.Int32) if k == "runtime" else k
        nv = (a[0] - a[0] + threads * count - 23).to(fx.Int32) if num_valid == "runtime" else num_valid
        if pairs:
            payload = fx.Vector.from_elements([b[tid * count + i] for i in range(count)])
            y, p, flags = T(keys, kk, payload, storage=storage, largest=largest, num_valid=nv)
            rows = [[_bits(y[i]) for i in range(count)], [p[i] for i in range(count)]]
        else:
            y, flags = T(keys, kk, storage=storage, largest=largest, num_valid=nv)
            rows = [[_bits(y[i]) for i in range(count)]]
        return rows + [[flags[i].to(fx.Int64) for i in range(count)]]

    return apply


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("algorithm,block,count,kind,pairs,k,num_valid,largest", TOPK_CASES)
def test_topk_matches_portable(algorithm, block, count, kind, pairs, k, num_valid, largest):
    threads = _threads(block)
    n = threads * count
    k_value = 61 if k == "runtime" else k
    nv = n - 23 if num_valid == "runtime" else (n if num_valid is None else num_valid)
    base = _topk_apply(algorithm, block, count, kind, pairs, k, num_valid)

    def apply(a, b, tid, ns, allocator):
        return base(a, b, tid, ns, allocator, largest)

    for dist in ("rand", "dup"):
        keys, values = _inputs(n, dist, seed=11)
        mine, ref = _run(apply, (keys, values), block, count, 2 + pairs)
        assert torch.equal(mine, ref), f"fx.coop != fx.coop.universal ({dist})"
        assert torch.equal(mine[-1], (torch.arange(n) < k_value).long())
        if kind == "f32":
            continue  # zero signs are AIR's ordered-bit business; matching the portable output is the contract
        host = _host_key(kind, keys.cpu())[:nv]
        order = host.argsort(descending=largest, stable=True)[:k_value]
        if algorithm is AIR:
            order = order.sort().values  # AIR keeps the selected items in input order
        assert torch.equal(_decode(kind, mine)[:k_value], host[order])
        if pairs:
            assert torch.equal(mine[1][:k_value], values.cpu()[order])


# ── results as operands ───────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("block,count,pairs", [(64, 1, False), (256, 4, False), (256, 1, True)])
def test_results_subtract_like_portable(block, count, pairs):
    """``x - key`` and ``x - payload`` on the returned registers: no DPP move is folded into the subtraction."""

    def apply(a, b, tid, ns, allocator):
        dtype = (fx.Int32, fx.Int32) if pairs else fx.Int32
        S = ns.BlockMergeSort[dtype, block, count]
        storage = allocator.allocate(S.SharedStorage).peek()
        x = [a[tid * count + i].to(fx.Int32) for i in range(count)]
        if pairs:
            k, p = S(fx.Vector.from_elements(x), fx.Vector.from_elements([v + 5 for v in x]), storage=storage)
            return [
                [(x[i] - k[i]).to(fx.Int64) for i in range(count)],
                [(x[i] - p[i]).to(fx.Int64) for i in range(count)],
            ]
        k = S(fx.Vector.from_elements(x), storage=storage)
        return [[(x[i] - k[i]).to(fx.Int64) for i in range(count)]]

    keys, values = _inputs(block * count, "rand", seed=3)
    keys = keys & 0xFFFFF  # no Int32 overflow in the differences
    mine, ref = _run(apply, (keys, values), block, count, 1 + pairs)
    assert torch.equal(mine, ref)
    x = keys.cpu()
    assert torch.equal(mine[0], x - x.sort().values)


# ── the override is the code that runs ───────────────────────────────────


def _isa(call, block, count, tmp_path):
    """Compile one call over ``count`` Int32 items per thread; return the final ISA text."""

    def apply(A, Out):
        tid = fx.thread_idx.x
        values = fx.Vector.from_elements([A[tid * count + i] for i in range(count)])
        result = call(values)
        for i in range(count):
            Out[tid * count + i] = result[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        apply(A, Out)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = torch.ones(block * count, dtype=torch.int32, device="cuda")
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        out = torch.zeros_like(values)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        launch(values, out, stream=stream)
        stream.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert len(dumped) == 1, f"expected one ISA under {tmp_path}"
    return dumped[0].read_text()


def _count(isa, *needles):
    return sum(line.split(";")[0].count(n) for line in isa.splitlines() for n in needles)


def _merge_call(ns, block, count):
    def call(values):
        S = ns.BlockMergeSort[fx.Int32, block, count]
        return S(values, storage=fx.SharedAllocator().allocate(S.SharedStorage).peek())

    return call


def _air_call(ns, block, count):
    def call(values):
        T = ns.BlockTopK[fx.Int32, block, count, AIR]
        y, flags = T(values, 100, storage=fx.SharedAllocator().allocate(T.SharedStorage).peek())
        return fx.Vector.from_elements([y[i] + flags[i].to(fx.Int32) for i in range(count)])

    return call


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_gfx9
def test_merge_network_runs(tmp_path):
    """256 x 4 Int32 keys: DPP compare-exchanges, a tenth of the LDS reads, 3 barriers + the closing one."""
    mine = _isa(_merge_call(fx.coop, 256, 4), 256, 4, tmp_path / "coop")
    ref = _isa(_merge_call(portable, 256, 4), 256, 4, tmp_path / "universal")
    dpp = ("quad_perm", "row_ror", "row_half_mirror")
    assert _count(mine, *dpp) >= 100 and _count(ref, *dpp) == 0
    assert _count(mine, "s_barrier") <= 4 < _count(ref, "s_barrier")
    assert _count(mine, "ds_read") * 5 < _count(ref, "ds_read")


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_gfx9
def test_air_ballots_run(tmp_path):
    """AIR on 256 x 4 Int32 keys: one barrier per key bit instead of a scan and a barrier."""
    mine = _isa(_air_call(fx.coop, 256, 4), 256, 4, tmp_path / "coop")
    ref = _isa(_air_call(portable, 256, 4), 256, 4, tmp_path / "universal")
    assert _count(mine, "s_bcnt1_i32_b64") >= 32 and _count(mine, "v_mbcnt_hi") >= 2
    assert _count(mine, "s_barrier") <= 32 + 4 < _count(ref, "s_barrier")


@pytest.fixture
def overrides(monkeypatch):
    """Record which override sequences ran: the merge network and the AIR ballots."""
    calls = []
    for module_name, name in (("merge_sort", "_bitonic"), ("topk", "_air_ranks")):
        module = importlib.import_module(f"flydsl.extension.coop.block.rocdl.{module_name}")
        original = getattr(module, name)

        def spy(*args, original=original, name=name, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return calls


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ["gfx908", "gfx942", "gfx1201"])
def test_targets_compile(monkeypatch, overrides, arch):
    """gfx9 runs the merge network (TopK MERGE reaches it through the dispatched sort) and the AIR
    ballots, and hands back the sorts the network does not cover; other targets hand back every call."""
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    block, count = 128, 2

    def apply(K, Out):
        tid = fx.thread_idx.x
        values = fx.Vector.from_elements([K[tid * count + i] for i in range(count)])
        allocator = fx.SharedAllocator()

        def storage(P):
            return allocator.allocate(P.SharedStorage).peek()

        M = fx.coop.BlockTopK[fx.Int32, block, count, MERGE]
        T = fx.coop.BlockTopK[fx.Int32, block, count, AIR]
        keys, _ = M(values, 9, storage=storage(M))
        y, flags = T(keys, 7, storage=storage(T))
        # Not covered: three items, and a runtime valid count.
        S3 = fx.coop.BlockMergeSort[fx.Int32, block, 3]
        z = S3(fx.Vector.from_elements([y[0], y[1], flags[0].to(fx.Int32)]), storage=storage(S3))
        S1 = fx.coop.BlockMergeSort[fx.Int32, block, 1]
        w = S1(fx.Vector.from_elements([z[2]]), storage=storage(S1), valid_items=K[0] + 5)
        for i in range(count):
            Out[tid * count + i] = z[i] + w[0]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(K: fx.Tensor, Out: fx.Tensor):
        apply(K, Out)

    @flyc.jit
    def launch(K: fx.Tensor, Out: fx.Tensor):
        kernel(K, Out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)
    expected = ["_bitonic", "_air_ranks"] if arch.startswith("gfx9") else []
    assert overrides == expected
