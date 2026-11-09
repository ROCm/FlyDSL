#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm overrides of ``BlockExchange`` and the ``BlockShuffle`` family.

``coop/block/rocdl/exchange.py`` moves the blocked side with 64/128-bit LDS
accesses and, when a thread's tile spans several of them, XOR-swizzles the
tile so that consecutive lanes no longer collide in one LDS pass.
``coop/block/rocdl/_neighbors.py`` reads the neighbors of a single-wave block
with ``wave_shr:1`` / ``wave_shl:1`` DPP moves instead of shared memory. Two
things are checked:

- **The same result.** Every case runs the dispatched ``fx.coop`` class and
  ``fx.coop.universal`` in one kernel and compares both, bit for bit, with each
  other and with a host reference.
- **The override is the code that runs.** The ISA shows the swizzle's XOR
  address terms, the DPP moves, the absence of shared memory -- and, on a
  non-gfx9 target, exactly the portable code.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest
from coop_test_utils import matrix_cases, warp_indices

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal

requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_gfx9 = pytest.mark.skipif(
    torch is None or not current_target().arch.startswith("gfx9"), reason="the overrides are gfx9-only"
)

KV = fx.Struct["key" : fx.Int32, "value" : fx.Float32]
DTYPES = {
    "u8": (fx.Uint8, ("uint8",)),
    "i16": (fx.Int16, ("int16",)),
    "i32": (fx.Int32, ("int32",)),
    "f32": (fx.Float32, ("float32",)),
    "i64": (fx.Int64, ("int64",)),
    "f64": (fx.Float64, ("float64",)),
    "kv": (KV, ("int32", "float32")),
}
RECORD_FIELDS = {"kv": ("key", "value")}


# ── launching one block with the dispatched and the portable class side by side ──


def _threads(block):
    return block if isinstance(block, int) else math.prod(block)


def _shape(block):
    return (block, 1, 1) if isinstance(block, int) else tuple(block)


def _data(d, size, seed=0, modulo=None):
    generator = torch.Generator().manual_seed(seed)
    out = []
    for name in DTYPES[d][1]:
        dtype = getattr(torch, name)
        if dtype.is_floating_point:
            out.append((torch.randn(size, generator=generator) * 100).to(dtype))
        else:
            info = torch.iinfo(dtype)
            values = torch.randint(max(info.min, -(1 << 40)), min(info.max, 1 << 40), (size,), generator=generator)
            out.append((values % modulo if modulo else values).to(dtype))
    return out


def _launch(body, inputs, outputs, block):
    """Run ``body(tid, a, b, x, y)`` on one block; inputs/outputs are one or two tensors each."""
    shape = _shape(block)
    a, b = (inputs + inputs[:1])[:2]
    x, y = (outputs + outputs[:1])[:2]

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, b: fx.Tensor, x: fx.Tensor, y: fx.Tensor):
        tid = fx.thread_idx.x + fx.thread_idx.y * shape[0] + fx.thread_idx.z * (shape[0] * shape[1])
        body(tid, a, b, x, y)

    @flyc.jit
    def launch(a: fx.Tensor, b: fx.Tensor, x: fx.Tensor, y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, b, x, y).launch(grid=(1, 1, 1), block=shape, stream=stream)

    tensors = [t.cuda() for t in (a, b, x, y)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(*tensors, stream=stream)
    stream.synchronize()
    return [t.cpu() for t in tensors[2 : 2 + len(outputs)]]


def _load(d, a, b, index):
    if d in RECORD_FIELDS:
        return DTYPES[d][0](a[index], b[index])
    return a[index].to(DTYPES[d][0])


def _store(d, x, y, index, item):
    if d in RECORD_FIELDS:
        first, second = RECORD_FIELDS[d]
        x[index] = getattr(item, first)
        y[index] = getattr(item, second)
    else:
        x[index] = item


def _items(d, count, a, b, tid):
    items = [_load(d, a, b, tid * count + i) for i in range(count)]
    return tuple(items) if d in RECORD_FIELDS else fx.Vector.from_elements(items, dtype=DTYPES[d][0])


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def _assert_same(actual, expected):
    for got, want in zip(actual, expected):
        assert torch.equal(_bits(got), _bits(want))


def _halves(outputs, size):
    """Split side-by-side results into (dispatched, portable)."""
    return [o[:size] for o in outputs], [o[size : 2 * size] for o in outputs]


# ── BlockExchange ─────────────────────────────────────────────────────────────

LAYOUTS = ("blocked_to_striped", "striped_to_blocked", "blocked_to_warp_striped", "warp_striped_to_blocked")


def _ranks(tid, count, size):
    return [((tid * count + i) * 97 + 13) % size for i in range(count)]


def _exchange_reference(method, inputs, block, count):
    threads = _threads(block)
    if method == "scatter_to_blocked":
        ranks = (torch.arange(threads * count) * 97 + 13) % (threads * count)
        out = []
        for t in inputs:
            tile = torch.empty_like(t)
            tile[ranks] = t
            out.append(tile)
        return out
    width = min(64, threads) if "warp" in method else threads
    inverse = method in ("striped_to_blocked", "warp_striped_to_blocked")
    return [t[warp_indices(threads, count, width, inverse=inverse)] for t in inputs]


def _run_exchange(method, d, block, count):
    threads = _threads(block)
    size = threads * count
    inputs = _data(d, size)
    outputs = [torch.zeros(2 * size, dtype=t.dtype) for t in inputs]

    def body(tid, a, b, x, y):
        allocator = fx.SharedAllocator()
        for half, namespace in enumerate((fx.coop, portable)):
            E = namespace.BlockExchange[DTYPES[d][0], block, count]
            storage = allocator.allocate(E.SharedStorage).peek()
            items = _items(d, count, a, b, tid)
            if method == "scatter_to_blocked":
                result = E.scatter_to_blocked(items, _ranks(tid, count, size), storage=storage)
            else:
                result = getattr(E, method)(items, storage=storage)
            for i in range(count):
                _store(d, x, y, half * size + tid * count + i, result[i])

    dispatched, universal = _halves(_launch(body, inputs, outputs, block), size)
    expected = _exchange_reference(method, inputs, block, count)
    _assert_same(dispatched, expected)
    _assert_same(universal, expected)


@requires_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "method,count,block",
    [
        # One item per thread is the portable code.
        *[(method, count, 256) for method in (*LAYOUTS, "scatter_to_blocked") for count in (2, 3, 4, 8, 16)],
        *matrix_cases([*LAYOUTS, "scatter_to_blocked"], [16, 8, 4], [192, 64, (8, 4, 4), 32]),
    ],
    ids=str,
)
def test_exchange_int32(method, count, block):
    _run_exchange(method, "i32", block, count)


@requires_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "method,d,count",
    [
        (method, d, count)
        for (d, count), method in matrix_cases(
            [
                ("i64", 4),
                ("i64", 8),
                ("u8", 32),
                ("u8", 16),
                ("i16", 16),
                ("f64", 3),
                ("f32", 8),
            ],
            ["blocked_to_striped", "striped_to_blocked", "warp_striped_to_blocked", "scatter_to_blocked"],
        )
    ],
    ids=str,
)
def test_exchange_dtypes(method, d, count):
    _run_exchange(method, d, 128, count)


# ── neighbors ────────────────────────────────────────────────────────────────

NEIGHBOR_CALLS = {
    "up": ("BlockShuffle", lambda C, v, s: C.up(v, storage=s)),
    "up_suffix": ("BlockShuffle", lambda C, v, s: C.up(v, storage=s, predecessor=v[1], return_suffix=True)),
    "down": ("BlockShuffle", lambda C, v, s: C.down(v, storage=s)),
    "down_prefix": ("BlockShuffle", lambda C, v, s: C.down(v, storage=s, successor=v[0], return_prefix=True)),
    "offset+1": ("BlockShuffle", lambda C, v, s: C.offset(v[0], 1, storage=s)),
    "offset-1": ("BlockShuffle", lambda C, v, s: C.offset(v[0], -1, storage=s, fill=v[0])),
    "offset+3": ("BlockShuffle", lambda C, v, s: C.offset(v[0], 3, storage=s)),
    "rotate-1": ("BlockShuffle", lambda C, v, s: C.rotate(v[0], -1, storage=s)),
    "rotate+1": ("BlockShuffle", lambda C, v, s: C.rotate(v[0], 1, storage=s)),
    "subtract_left": ("BlockAdjacentDifference", lambda C, v, s: C.subtract_left(v, storage=s)),
    "subtract_left_valid": (
        "BlockAdjacentDifference",
        lambda C, v, s: C.subtract_left(v, storage=s, predecessor=v[0], valid_items=37),
    ),
    "subtract_right": ("BlockAdjacentDifference", lambda C, v, s: C.subtract_right(v, storage=s)),
    "subtract_right_valid": (
        "BlockAdjacentDifference",
        lambda C, v, s: C.subtract_right(v, storage=s, successor=v[0], valid_items=41),
    ),
    "heads_predecessors": (
        "BlockDiscontinuity",
        lambda C, v, s: C.flag_heads(v, storage=s, return_predecessors=True),
    ),
    "tails": ("BlockDiscontinuity", lambda C, v, s: C.flag_tails(v, storage=s)),
    "heads_tails_indexed": (
        "BlockDiscontinuity",
        lambda C, v, s: C.flag_heads_and_tails(
            v, lambda left, right, i: (left != right) | (i % 5 == 0), storage=s, predecessor=v[0], successor=v[1]
        ),
    ),
}


def _parts(result, count):
    """Flatten a call's result into tuples of ``count`` items (scalars repeat)."""
    parts = result if isinstance(result, tuple) and not isinstance(result[0], KV) else (result,)
    out = []
    for part in parts:
        if isinstance(part, (fx.Vector, tuple, list)):
            out.append(tuple(part[i] for i in range(count)))
        else:
            out.append((part,) * count)
    return out


def _run_neighbors(call, d, block, count, *, minus=False):
    """Dispatched vs portable, bit for bit; ``minus`` stores ``x - result`` for scalar results."""
    primitive, fn = NEIGHBOR_CALLS[call]
    threads = _threads(block)
    size = threads * count
    inputs = _data(d, size, seed=1, modulo=3 if d != "f32" else None)
    parts = 3  # at most (heads, tails, predecessors)
    flags = primitive == "BlockDiscontinuity"
    outputs = [torch.zeros(2 * parts * size, dtype=torch.int64 if flags else t.dtype) for t in inputs]

    def body(tid, a, b, x, y):
        allocator = fx.SharedAllocator()
        for half, namespace in enumerate((fx.coop, portable)):
            cls = getattr(namespace, primitive)[DTYPES[d][0], block]
            storage = allocator.allocate(cls.SharedStorage).peek()
            items = _items(d, count, a, b, tid)
            for j, part in enumerate(_parts(fn(cls, items, storage), count)):
                for i in range(count):
                    value = part[i]
                    if minus:
                        value = items[i] - value.to(items[i].dtype)
                    index = (half * parts + j) * size + tid * count + i
                    if flags and isinstance(value, fx.Numeric):
                        x[index] = value.to(fx.Int64)
                    else:
                        _store(d, x, y, index, value)

    dispatched, universal = _halves(_launch(body, inputs, outputs, block), parts * size)
    _assert_same(dispatched, universal)
    return inputs, dispatched


@requires_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "call,block,count",
    [
        *[
            (call, block, 1 if call.startswith(("offset", "rotate")) else 4)
            for call in NEIGHBOR_CALLS
            for block in (64,)
        ],
        # Single-wave blocks only: a multi-wave block is the portable code (see the ISA test below).
        *matrix_cases(list(NEIGHBOR_CALLS), [32, (4, 4, 4), 16], [3]),
    ],
    ids=str,
)
def test_neighbors_int32(call, block, count):
    if call.startswith(("offset", "rotate")):
        count = 1
    _run_neighbors(call, "i32", block, count)


@requires_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "call,block", matrix_cases(["up", "down", "offset+1", "rotate-1", "subtract_left", "subtract_right"], [64, 16])
)
def test_neighbors_minus_result(call, block):
    """``x - neighbor`` must not become a folded ``v_subrev_u32_dpp`` (wrong on gfx942)."""
    _run_neighbors(call, "i32", block, 1 if call.startswith(("offset", "rotate")) else 4, minus=True)


@requires_gpu
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "call,d", matrix_cases(["up_suffix", "down_prefix", "rotate+1", "subtract_left"], ["i64", "f32", "kv", "u8"])
)
def test_neighbors_dtypes(call, d):
    if d == "kv" and NEIGHBOR_CALLS[call][0] != "BlockShuffle":
        pytest.skip("the default difference is not defined for records")
    _run_neighbors(call, d, 64, 1 if call.startswith(("offset", "rotate")) else 3)


# ── the override is the code that runs ──────────────────────────────────────────


def _isa(body, block, count, tmp_path, dtype="int32"):
    """Final ISA of a one-block kernel; ``body(tid, a, out)`` with ``count`` items per thread."""
    threads = _threads(block)
    values = torch.zeros(threads * count, dtype=getattr(torch, dtype))

    @flyc.kernel(known_block_size=list(_shape(block)))
    def kernel(a: fx.Tensor, b: fx.Tensor, x: fx.Tensor, y: fx.Tensor):
        body(fx.thread_idx.x, a, x)

    @flyc.jit
    def launch(a: fx.Tensor, b: fx.Tensor, x: fx.Tensor, y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, b, x, y).launch(grid=(1, 1, 1), block=_shape(block), stream=stream)

    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        device = "cuda" if torch.cuda.is_available() and os.environ.get("COMPILE_ONLY") != "1" else "cpu"
        tensors = [values.to(device) for _ in range(4)]
        stream = fx.Stream(None)
        if device == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
        launch(*tensors, stream=stream)
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        for k, v in previous.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    dumped = sorted(Path(tmp_path).glob("**/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    lines = [line.split(";")[0].strip() for line in dumped[-1].read_text().split("s_endpgm")[0].splitlines()]
    return [line for line in lines if line and not line.startswith((".", "//")) and not line.endswith(":")]


def _ops(isa, prefix):
    return sum(line.split()[0].startswith(prefix) for line in isa)


def _exchange_body(namespace, method, count, block=256):
    def body(tid, a, out):
        E = namespace.BlockExchange[fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(E.SharedStorage).peek()
        items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        result = getattr(E, method)(items, storage=storage)
        for i in range(count):
            out[tid * count + i] = result[i]

    return body


@requires_gfx9
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("method", ["blocked_to_striped", "striped_to_blocked"])
def test_isa_swizzle_keeps_wide_conflict_free_accesses(method, tmp_path):
    """8 Int32 per thread: two b128 accesses per lane on the blocked side, XOR-swizzled."""
    dispatched = _isa(_exchange_body(fx.coop, method, 8), 256, 8, tmp_path / "d")
    universal = _isa(_exchange_body(portable, method, 8), 256, 8, tmp_path / "u")
    wide = "ds_write_b128" if method == "blocked_to_striped" else "ds_read_b128"
    striped = "ds_read2st64_b32" if method == "blocked_to_striped" else "ds_write2st64_b32"
    for isa in (dispatched, universal):
        assert _ops(isa, wide) == 2 and _ops(isa, striped) == 4
    # The swizzle's XOR address terms are the only difference.
    assert _ops(dispatched, "v_xor_b32") >= 2 and _ops(universal, "v_xor_b32") == 0


@requires_gfx9
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("call", ["subtract_left", "down", "rotate-1"])
def test_isa_single_wave_neighbors_use_dpp(call, tmp_path):
    primitive, fn = NEIGHBOR_CALLS[call]
    count = 1 if call == "rotate-1" else 4

    def make(namespace):
        def body(tid, a, out):
            cls = getattr(namespace, primitive)[fx.Int32, 64]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
            result = _parts(fn(cls, items, storage), count)[0]
            for i in range(count):
                out[tid * count + i] = items[i] - result[i]

        return body

    dispatched = _isa(make(fx.coop), 64, count, tmp_path / "d")
    universal = _isa(make(portable), 64, count, tmp_path / "u")
    assert _ops(dispatched, "ds_") == 0 and _ops(universal, "ds_") == 2
    dpp = [line for line in dispatched if re.search(r"wave_sh[lr]:1", line)]
    # One plain DPP move: never folded into the subtraction (v_subrev_u32_dpp is miscompiled).
    assert len(dpp) == 1 and dpp[0].startswith("v_mov_b32_dpp")


@requires_gfx9
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_isa_multi_wave_neighbors_stay_portable(tmp_path):
    def make(namespace):
        def body(tid, a, out):
            cls = namespace.BlockAdjacentDifference[fx.Int32, 256]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            items = fx.Vector.from_elements([a[tid * 4 + i] for i in range(4)])
            result = cls.subtract_left(items, storage=storage)
            for i in range(4):
                out[tid * 4 + i] = result[i]

        return body

    assert _isa(make(fx.coop), 256, 4, tmp_path / "d") == _isa(make(portable), 256, 4, tmp_path / "u")


@requires_gfx9
@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("io", ["BlockLoad", "BlockStore"])
def test_transpose_io_reaches_the_swizzle(io, tmp_path):
    """BlockLoad/BlockStore TRANSPOSE exchange through the dispatched BlockExchange."""
    count, block = 8, 256

    def make(namespace):
        def body(tid, a, out):
            if io == "BlockLoad":
                P = namespace.BlockLoad[fx.Int32, block, count, namespace.BlockLoadAlgorithm.TRANSPOSE]
                storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
                items = P(a, storage=storage)
                for i in range(count):
                    out[tid * count + i] = items[i]
            else:
                P = namespace.BlockStore[fx.Int32, block, count, namespace.BlockStoreAlgorithm.TRANSPOSE]
                storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
                P(out, fx.Vector.from_elements([a[tid * count + i] for i in range(count)]), storage=storage)

        return body

    dispatched = _isa(make(fx.coop), block, count, tmp_path / "d")
    universal = _isa(make(portable), block, count, tmp_path / "u")
    assert _ops(dispatched, "v_xor_b32") >= 2 and _ops(universal, "v_xor_b32") == 0

    # And the same data comes out.
    size = block * count
    values = torch.arange(size, dtype=torch.int32) * 7 + 3
    outputs = [torch.zeros(2 * size, dtype=torch.int32)]

    def run(tid, a, b, x, y):
        allocator = fx.SharedAllocator()
        for half, namespace in enumerate((fx.coop, portable)):
            if io == "BlockLoad":
                P = namespace.BlockLoad[fx.Int32, block, count, namespace.BlockLoadAlgorithm.TRANSPOSE]
                items = P(a, storage=allocator.allocate(P.SharedStorage).peek())
                for i in range(count):
                    x[half * size + tid * count + i] = items[i]
            else:
                P = namespace.BlockStore[fx.Int32, block, count, namespace.BlockStoreAlgorithm.TRANSPOSE]
                items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
                P(x, items, storage=allocator.allocate(P.SharedStorage).peek(), offset=half * size)

    dispatched_out, universal_out = _halves(_launch(run, [values], outputs, block), size)
    _assert_same(dispatched_out, universal_out)
    assert torch.equal(dispatched_out[0], values)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("case", ["blocked_to_striped", "scatter_to_blocked", "subtract_left"])
def test_non_gfx9_compiles_the_portable_code(case, tmp_path, monkeypatch):
    """On gfx1201 the dispatched classes compile to exactly the portable ISA."""
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = 64, 8

    def make(namespace):
        def body(tid, a, out):
            items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
            if case == "subtract_left":
                cls = namespace.BlockAdjacentDifference[fx.Int32, block]
                storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
                result = cls.subtract_left(items, storage=storage)
            else:
                E = namespace.BlockExchange[fx.Int32, block, count]
                storage = fx.SharedAllocator().allocate(E.SharedStorage).peek()
                if case == "scatter_to_blocked":
                    result = E.scatter_to_blocked(items, _ranks(tid, count, block * count), storage=storage)
                else:
                    result = getattr(E, case)(items, storage=storage)
            for i in range(count):
                out[tid * count + i] = result[i]

        return body

    dispatched = _isa(make(fx.coop), block, count, tmp_path / "d")
    universal = _isa(make(portable), block, count, tmp_path / "u")
    assert dispatched == universal
    assert not any(re.search(r"wave_sh[lr]|row_sh", line) for line in dispatched)
