# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block collective checks, kernels and host references; independent of warp test helpers."""

import math

import pytest
import torch
from coop_common import WARP_SIZE, sample
from coop_test_utils import (
    RECORD,
    Columns,
    as_items,
    full_matrix_enabled,
    make_items,
    run_kernel,
    run_tile,
    warp_indices,
)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

# Shared kernels and host references for primitive matrices and compilation.


def check_block_case(entry, block, count, case):
    dtype, name = entry
    primitive, method = case
    threads = block if isinstance(block, int) else block[0] * block[1] * block[2]
    values = sample(name, threads * count, seed=419)
    if primitive == "BlockDiscontinuity":
        values = (torch.arange(threads * count, device="cuda") // 7 % 3).to(values.dtype)

    def transform(value):
        op = getattr(fx.coop, primitive)
        tile = primitive in ("BlockExchange", "BlockMergeSort")
        op = op[dtype, block, count] if tile else op[dtype, block]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        arg = value[0] if count == 1 else value
        if method == "sort":
            out = op(arg, storage=storage)
        elif method.startswith("scatter"):
            shape = op.block_size
            tid = fx.thread_idx.x + fx.thread_idx.y * shape[0] + fx.thread_idx.z * shape[0] * shape[1]
            ranks = fx.Vector.from_elements([threads * count - 1 - (tid * count + i) for i in range(count)])
            out = getattr(op, method)(arg, ranks, storage=storage)
        else:
            out = getattr(op, method)(arg, storage=storage)
        return out if isinstance(out, fx.Vector) else fx.Vector.from_elements([out])

    output = run_tile(transform, values, block, count, dtype=dtype)
    host = values.cpu()
    if primitive == "BlockExchange":
        width = min(WARP_SIZE, threads) if "warp" in method else threads
        if method.startswith("scatter"):
            host = host.flip(0)
        if method.endswith("to_striped") or method == "blocked_to_warp_striped":
            expected = host.reshape(-1, count, width).transpose(1, 2).reshape(-1)
        elif method == "scatter_to_blocked":
            expected = host
        else:
            expected = host.reshape(-1, width, count).transpose(1, 2).reshape(-1)
    elif method == "sort":
        expected = host.sort().values
    else:
        left = method in ("up", "subtract_left", "flag_heads")
        endpoint = 0 if left else -1
        neighbor = host.roll(1 if left else -1)
        neighbor[endpoint] = host[endpoint]
        if method.startswith("flag"):
            expected = (neighbor != host).to(host.dtype)
            expected[endpoint] = 1
        elif method.startswith("subtract"):
            expected = host - neighbor
            expected[endpoint] = host[endpoint]
        else:
            expected = neighbor
    for actual in output:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=lambda msg: f"{msg} actual={actual.tolist()} expected={expected.tolist()}",
        )


def check_compile_family(monkeypatch, arch, case):

    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    block, count = 64, 4
    if case in ("BlockMergeSort", "BlockTopK") and not full_matrix_enabled():
        # Cross-target lowering needs the multi-item path, not a large tile.
        count = 2

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        root = getattr(fx.coop, case)
        if case in ("BlockShuffle", "BlockAdjacentDifference", "BlockDiscontinuity"):
            op = root[fx.Int32, block]
        elif case == "BlockLoad":
            op = root[fx.Int32, block, count, fx.coop.BlockLoadAlgorithm.TRANSPOSE]
        elif case == "BlockStore":
            op = root[fx.Int32, block, count, fx.coop.BlockStoreAlgorithm.TRANSPOSE]
        else:
            op = root[fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        if case == "BlockExchange":
            result = op.blocked_to_striped(value, storage=storage)
        elif case == "BlockShuffle":
            result = op.up(value, storage=storage)
        elif case == "BlockAdjacentDifference":
            result = op.subtract_left(value, storage=storage)
        elif case == "BlockDiscontinuity":
            result = op.flag_heads(value, storage=storage)
        elif case == "BlockLoad":
            result = op(a, storage=storage, valid_items=253)
        elif case == "BlockStore":
            op(out, value, storage=storage, valid_items=253)
            return
        elif case == "BlockHistogram":
            slots = op(value, storage=storage)
            result = fx.Vector.from_elements([slots[i] for i in range(count)])
        elif case == "BlockRunLengthDecode":
            result, _, _ = op(value, value & 3, storage=storage, decoded_items_per_thread=count)
        elif case == "BlockTopK":
            result, _ = op(value, 17, storage=storage)
        elif case in ("BlockRadixSort", "BlockRadixRank"):
            result = op(value, storage=storage, end_bit=3)
        else:
            result = op(value, storage=storage)
        for i in range(count):
            out[tid * count + i] = result[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)


# Shared load/store checks and kernels for movement policy compilation.


def check_guarded_io(
    policy, valid, block=64, count=3, entry=(fx.Int32, "int32"), universal=False, width=8, dynamic=False
):

    offset = 5
    valid_extent = block * count if valid is None else valid
    dtype, name = entry
    dt = getattr(torch, name)
    ns = fx.coop.universal if universal else fx.coop

    def apply(a, out, loaded):
        limit = fx.Int32(valid) if dynamic else valid
        loader = ns.BlockLoad[dtype, block, count, getattr(ns.BlockLoadAlgorithm, policy)]
        storer = ns.BlockStore[dtype, block, count, getattr(ns.BlockStoreAlgorithm, policy)]
        storage = fx.SharedAllocator().allocate(loader.SharedStorage).peek()
        items = loader(a, storage=storage, offset=offset, valid_items=limit, default=-7)
        fx.barrier()
        storer(out, items, storage=storage, offset=offset, valid_items=limit)
        for i in range(count):
            loaded[fx.thread_idx.x * count + i] = items[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor):
        apply(a, out, loaded)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out, loaded).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    # Deliberately allocate no invalid input elements: predication must guard
    # the load, not merely select its result after reading out of bounds.
    a = torch.arange(offset + valid_extent, dtype=dt, device="cuda")
    out = torch.full((offset + block * count + 4,), -99, dtype=dt, device="cuda")
    loaded = torch.empty(block * count, dtype=dt, device="cuda")
    launch(a, out, loaded, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    expected = torch.full_like(out.cpu(), -99)
    expected[offset : offset + valid_extent] = a.cpu()[offset:]
    torch.testing.assert_close(out.cpu(), expected)
    expected_items = torch.full((block * count,), -7, dtype=dt, device="cpu")
    expected_items[:valid_extent] = a.cpu()[offset:]
    if policy in ("STRIPED", "warp_striped"):
        width = (width or fx.num_warp_threads()) if policy == "warp_striped" else block
        expected_items = expected_items.reshape(-1, count, width).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(loaded.cpu(), expected_items)


def check_movement_io_policies(warp, policy, count, valid):

    block, width, offset = (32 if warp else 64), 8, 5
    size = block * count
    source = torch.arange(offset + valid, dtype=torch.int32, device="cuda")

    def apply(a, out):
        loader = fx.coop.BlockLoad[fx.Int32, block, count, getattr(fx.coop.BlockLoadAlgorithm, policy)]
        storer = fx.coop.BlockStore[fx.Int32, block, count, getattr(fx.coop.BlockStoreAlgorithm, policy)]
        storage = fx.SharedAllocator().allocate(loader.SharedStorage).peek()
        items = loader(a, storage=storage, offset=offset, valid_items=fx.Int32(valid), default=-7)
        fx.barrier()
        storer(out, items, storage=storage, offset=offset, valid_items=fx.Int32(valid))
        for i in range(count):
            out[offset + size + 4 + fx.thread_idx.x * count + i] = items[i]

    actual = run_kernel(apply, source, offset + size * 2 + 4, block)
    expected = torch.full_like(actual, -99)
    expected[offset : offset + valid] = source.cpu()[offset:]
    loaded = torch.full((size,), -7, dtype=torch.int32, device="cpu")
    loaded[:valid] = source.cpu()[offset:]
    if policy == "STRIPED":
        loaded = loaded[warp_indices(block, count, width if warp else block)]
    expected[offset + size + 4 :] = loaded
    torch.testing.assert_close(actual, expected)


def check_compile_movement_policies(monkeypatch, arch, case):

    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = (64 if case.startswith("warp") else 128), 4

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        pass
        if case.startswith("block_io"):
            policy = case.removeprefix("block_io_")
            loader = fx.coop.BlockLoad[fx.Int32, block, count, fx.coop.BlockLoadAlgorithm[policy]]
            storer = fx.coop.BlockStore[fx.Int32, block, count, fx.coop.BlockStoreAlgorithm[policy]]
            storage = fx.SharedAllocator().allocate(loader.SharedStorage).peek()
            result = loader(a, storage=storage, valid_items=block * count - 1)
            fx.barrier()
            storer(out, result, storage=storage, valid_items=block * count - 1)
            return
        elif case.startswith("block_exchange"):
            method = case.removeprefix("block_exchange_")
            exchange = fx.coop.BlockExchange[fx.Int32, block, count, fx.coop.BlockExchangeAlgorithm.WARP_TIME_SLICING]
            storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
            if method.startswith("scatter"):
                ranks = fx.Vector.from_elements([block * count - 1 - (tid * count + i) for i in range(count)])
                flags = ranks % 3 != 1
                if method.endswith("flagged"):
                    result = exchange.scatter_to_striped_flagged(value, ranks, flags, storage=storage, default=-7)
                elif method.endswith("guarded"):
                    ranks = flags.select(ranks, fx.Vector.filled(count, -1, fx.Int32))
                    result = exchange.scatter_to_striped_guarded(value, ranks, storage=storage, default=-7)
                else:
                    result = exchange.scatter_to_blocked(value, ranks, storage=storage)
            else:
                result = getattr(exchange, method)(value, storage=storage)
        else:
            raise AssertionError(f"unknown movement policy: {case}")
        for i in range(count):
            out[tid * count + i] = result[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)


def check_guarded_cross_dtype_io(
    warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial
):

    block, offset, width = (32 if warp else 64), 3, 8
    size = block * count
    valid = 1 if partial == "one" else size - 1
    loaded_dtype = target_dtype if convert_load else source_dtype
    loaded_name = target_name if convert_load else source_name

    def apply(a, out, loaded):
        loader = fx.coop.BlockLoad[loaded_dtype, block, count, fx.coop.BlockLoadAlgorithm[policy]]
        storer = fx.coop.BlockStore[loaded_dtype, block, count, fx.coop.BlockStoreAlgorithm[policy]]
        storage = fx.SharedAllocator().allocate(loader.SharedStorage).peek()
        items = loader(a, storage=storage, offset=offset, valid_items=fx.Int32(valid), default=-7)
        fx.barrier()
        storer(out, items, storage=storage, offset=offset, valid_items=fx.Int32(valid))
        assert items.dtype is loaded_dtype
        for i in range(count):
            loaded[fx.thread_idx.x * count + i] = items[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor):
        apply(a, out, loaded)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out, loaded).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = torch.arange(offset + valid, device="cuda", dtype=torch.int64)
    if source_dtype.is_float:
        values = values.to(getattr(torch, source_name)) * 1.25 - 60.375
    else:
        values = (values * 1777 - 91003).to(getattr(torch, source_name))
    output = torch.full((offset + size + 3,), -99, device="cuda", dtype=getattr(torch, target_name))
    loaded = torch.empty(size, device="cuda", dtype=getattr(torch, loaded_name))
    launch(values, output, loaded, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    expected = torch.full_like(output.cpu(), -99)
    expected[offset : offset + valid] = values.cpu()[offset:].to(getattr(torch, target_name))
    torch.testing.assert_close(output.cpu(), expected)
    expected_items = torch.full((size,), -7, dtype=getattr(torch, loaded_name), device="cpu")
    expected_items[:valid] = values.cpu()[offset:].to(getattr(torch, loaded_name))
    if policy == "STRIPED":
        expected_items = expected_items[warp_indices(block, count, width if warp else block)]
    torch.testing.assert_close(loaded.cpu(), expected_items)


# Shared record kernels and host references for exchange, load, and store checks.


def _make_record_kernel(case, policy, block, count):

    size = block * count

    def apply(a, b, c, x, y, z):
        source = Columns(a, b, c)
        destination = Columns(x, y, z)
        if case == "block_io":
            loader = fx.coop.BlockLoad[RECORD, block, count, fx.coop.BlockLoadAlgorithm[policy]]
            storer = fx.coop.BlockStore[RECORD, block, count, fx.coop.BlockStoreAlgorithm[policy]]
            storage = fx.SharedAllocator().allocate(loader.SharedStorage).peek()
            items = loader(source, storage=storage, valid_items=size - 3, default=-7)
            fx.barrier()
            storer(destination, items, storage=storage, valid_items=size - 3)
        else:
            items = make_items([source[fx.thread_idx.x * count + i] for i in range(count)])
            exchange = fx.coop.BlockExchange[RECORD, block, count, fx.coop.BlockExchangeAlgorithm[policy]]
            storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
            if case.startswith("scatter"):
                ranks = fx.Vector.from_elements([size - 1 - (fx.thread_idx.x * count + i) for i in range(count)])
                valid = ranks % 3 != 1
                if case.endswith("flagged"):
                    items = exchange.scatter_to_striped_flagged(items, ranks, valid, storage=storage, default=-7)
                else:
                    ranks = valid.select(ranks, fx.Vector.filled(count, -1, fx.Int32))
                    items = exchange.scatter_to_striped_guarded(items, ranks, storage=storage, default=-7)
            else:
                items = getattr(exchange, case)(items, storage=storage)
        for i, item in enumerate(items):
            destination[size + fx.thread_idx.x * count + i] = item

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor, x: fx.Tensor, y: fx.Tensor, z: fx.Tensor):
        apply(a, b, c, x, y, z)

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        c: fx.Tensor,
        x: fx.Tensor,
        y: fx.Tensor,
        z: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(a, b, c, x, y, z).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def check_record_movement(case, policy, count):

    warp_case = case.startswith("warp_") and policy not in ("SHARED_MEMORY", "WARP_TIME_SLICING")
    block = 64 if warp_case else 64
    size = block * count
    fields = (
        torch.arange(size, device="cuda", dtype=torch.int32),
        torch.arange(size, device="cuda", dtype=torch.float32) + 0.375,
        torch.arange(size, device="cuda", dtype=torch.int64) * (1 << 48) + 17,
    )
    outputs = [torch.full((size * 2,), -99, device="cuda", dtype=field.dtype) for field in fields]
    _make_record_kernel(case, policy, block, count)(*fields, *outputs, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    for field, output in zip(fields, outputs):
        host = field.cpu()
        expected = torch.full((size * 2,), -99, dtype=host.dtype, device="cpu")
        if case.endswith("io"):
            mask = (
                (torch.arange(size, device="cpu") % (8 * count) < 8 * count - 3)
                if case.startswith("warp")
                else torch.arange(size, device="cpu") < size - 3
            )
            expected[:size] = torch.where(mask, host, -99)
            items = torch.where(mask, host, -7)
            if policy == "STRIPED":
                items = items[warp_indices(block, count, 8 if case.startswith("warp") else block)]
        elif "scatter" in case:
            if warp_case:
                items = host.reshape(-1, 8 * count).flip(1).reshape(-1)[warp_indices(block, count, 8)]
            else:
                items = torch.where(torch.arange(size, device="cpu") % 3 != 1, host.flip(0), -7)[
                    warp_indices(block, count, block)
                ]
        else:
            if warp_case:
                width = 8
                inverse = case == "warp_striped_to_blocked"
            else:
                width = min(fx.num_warp_threads(), 1 << (block.bit_length() - 1)) if "warp" in case else block
                inverse = case in ("striped_to_blocked", "warp_striped_to_blocked")
            items = host[warp_indices(block, count, width, inverse=inverse)]
        expected[size:] = items
        torch.testing.assert_close(output.cpu(), expected)


def check_compile_record_movement(monkeypatch, arch, case, policy):

    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    warp_case = case.startswith("warp_") and policy not in ("SHARED_MEMORY", "WARP_TIME_SLICING")
    block, count = (64 if warp_case else 64), 3
    fields = [
        torch.empty(block * count, dtype=dtype, device="cpu") for dtype in (torch.int32, torch.float32, torch.int64)
    ]
    outputs = [torch.empty(block * count * 2, dtype=field.dtype, device="cpu") for field in fields]
    _make_record_kernel(case, policy, block, count)(*fields, *outputs)


# Shared kernels and reference checks for scalar reductions and scans.


def affine(lhs, rhs):
    """Composition over F_251, packed as (a << 16) | b; associative, noncommutative."""
    a, b = (lhs >> 16) & 65535, lhs & 65535
    c, d = (rhs >> 16) & 65535, rhs & 65535
    return (((a * c) % 251) << 16) | ((a * d + b) % 251)


affine.identity = 1 << 16


def host_fold(items, op=affine, init=1 << 16):
    result = init
    for item in items:
        result = op(result, int(item))
    return result


def inputs(count):
    ids = torch.arange(count, dtype=torch.int32, device="cpu")
    return ((((ids * 13 + 3) % 251) << 16) | ((ids * 7 + 11) % 251)).cuda()


def run_reduction(kernel, count, *extra):
    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(count, 1, 1), stream=stream)

    values = extra[0] if extra else inputs(count)
    out = torch.zeros_like(values)
    agg = torch.zeros(count, dtype=torch.int32, device="cuda")
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return values.cpu(), out.cpu(), agg.cpu()


def check_ordered_reduce_partial_blocks(block, algorithm, universal):
    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        namespace = fx.coop.universal if universal else fx.coop
        reduction = namespace.BlockReduce[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(reduction.SharedStorage).peek()
        tid = fx.thread_idx.x
        Out[tid] = reduction(A[tid], affine, storage=storage)

    values, out, _ = run_reduction(kernel, block)
    assert torch.equal(out, torch.full_like(out, host_fold(values)))


def check_reduce_policies_scalar_or_array(algorithm, valid):
    block, items = 128, 3 if valid is None else 1

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        reduction = fx.coop.universal.BlockReduce[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(reduction.SharedStorage).peek()
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        Agg[tid] = reduction(
            vector if valid is None else vector[0], fx.ReductionOp.ADD, storage=storage, valid_items=valid
        )

    values = torch.arange(block * items, dtype=torch.int32, device="cuda")
    host, _, agg = run_reduction(kernel, block, values)
    assert torch.equal(agg, torch.full_like(agg, host[:valid].sum()))


def check_ordered_scan_partial_blocks(block, algorithm, exclusive):
    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        scan = fx.coop.universal.BlockScan[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        form = scan.exclusive_with_aggregate if exclusive else scan.inclusive_with_aggregate
        result, aggregate = form(A[tid], affine, storage=storage)
        Out[tid] = result
        Agg[tid] = aggregate

    values, out, agg = run_reduction(kernel, block)
    expected = [host_fold(values[: i + (not exclusive)]) for i in range(block)]
    assert out.tolist() == expected
    assert torch.equal(agg, torch.full_like(agg, host_fold(values)))


def check_scan_callback_scalar_or_array(algorithm, valid):
    block, items = 128, 3 if valid is None else 1

    def callback(aggregate):
        return aggregate + fx.thread_idx.x + 7

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        scan = fx.coop.universal.BlockScan[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        result, aggregate = scan.exclusive_with_aggregate(
            vector if valid is None else vector[0],
            fx.ReductionOp.ADD,
            storage=storage,
            valid_items=valid,
            prefix_callback=callback,
        )
        for i in range(items):
            Out[tid * items + i] = result[i] if valid is None else result
        Agg[tid] = aggregate

    values = torch.arange(block * items, dtype=torch.int32, device="cuda")
    host, out, agg = run_reduction(kernel, block, values)
    total = host[:valid].sum()
    if valid is not None:
        host[valid:] = 0
    expected = host.cumsum(0) - host + total + 7
    assert out.tolist() == expected.tolist()
    assert torch.equal(agg, torch.full_like(agg, total))


def plain_affine(lhs, rhs):
    """The same semigroup operation, deliberately without identity metadata."""
    return affine(lhs, rhs)


def check_scan_semigroup_without_identity(algorithm, exclusive):
    block = 128
    initial = (7 << 16) | 19

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        scan = fx.coop.BlockScan[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        if fx.const_expr(exclusive):
            result, aggregate = scan.exclusive_with_aggregate(
                A[tid], plain_affine, storage=storage, init=fx.Int32(initial)
            )
        else:
            result, aggregate = scan.inclusive_with_aggregate(A[tid], plain_affine, storage=storage)
        Out[tid], Agg[tid] = result, aggregate

    values, out, agg = run_reduction(kernel, block)
    expected = [host_fold(values[: i + (not exclusive)], init=initial if exclusive else 1 << 16) for i in range(block)]
    assert out.tolist() == expected
    assert torch.equal(agg, torch.full_like(agg, host_fold(values)))


def check_commutative_policies_reject_undeclared_callable(algorithm):
    reduction = fx.coop.BlockReduce[fx.Int32, 64, algorithm]
    with pytest.raises(ValueError, match="commutative"):
        reduction(object(), affine, storage=None)


def plain_add(lhs, rhs):
    return lhs + rhs


plain_add.commutative = True


def check_reduce_semigroup_scalar_or_array(algorithm, valid):
    block, items = 128, 3 if valid is None else 1
    ordered = algorithm in (fx.coop.BlockReduceAlgorithm.RAKING, fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS)
    operation = plain_affine if ordered else plain_add

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        reduction = fx.coop.universal.BlockReduce[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(reduction.SharedStorage).peek()
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        Agg[tid] = reduction(
            vector if valid is None else vector[0],
            operation,
            storage=storage,
            valid_items=None if valid is None else fx.Int32(valid),
        )

    values = inputs(block * items) if ordered else torch.arange(block * items, dtype=torch.int32, device="cuda")
    host, _, agg = run_reduction(kernel, block, values)
    expected = host_fold(host[:valid]) if ordered else int(host[:valid].sum())
    assert torch.equal(agg, torch.full_like(agg, expected))


def check_scan_semigroup_unseeded_exclusive_vector(algorithm):
    block, items = 128, 3

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        scan = fx.coop.universal.BlockScan[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        result, aggregate = scan.exclusive_with_aggregate(vector, plain_affine, storage=storage)
        for i in range(items):
            Out[tid * items + i] = result[i]
        Agg[tid] = aggregate

    host, out, agg = run_reduction(kernel, block, inputs(block * items))
    assert out[1:].tolist() == [host_fold(host[:i]) for i in range(1, len(host))]
    assert torch.equal(agg, torch.full_like(agg, host_fold(host)))


def state_callback(state):
    def callback(total):
        tid = fx.thread_idx.x
        previous = state[tid]
        state[tid] = previous + total
        return previous

    return callback


def check_callback_first_physical_warp_and_state_across_tiles(block, algorithm):
    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        scan = fx.coop.universal.BlockScan[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        tid = fx.thread_idx.x
        Agg[tid] = fx.Int32(tid) + 7
        fx.barrier()
        callback = state_callback(Agg)
        scan.exclusive(A[tid], fx.ReductionOp.ADD, storage=storage, prefix_callback=callback)
        fx.barrier()
        second = scan.exclusive(A[tid], fx.ReductionOp.ADD, storage=storage, prefix_callback=callback)
        Out[tid] = second

    host, out, state = run_reduction(kernel, block, torch.ones(block, dtype=torch.int32, device="cuda"))
    assert out.tolist() == [7 + block + i for i in range(block)]
    active = min(block, fx.num_warp_threads())
    if algorithm is not fx.coop.BlockScanAlgorithm.WARP_SCANS:
        segment = (block + active - 1) // active
        active = (block + segment - 1) // segment
    assert state.tolist() == [i + 7 + (2 * block if i < active else 0) for i in range(block)]


def check_reduce_scan_policies_compile(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    block = 128

    def apply(a, out):
        tid = fx.thread_idx.x
        value = a[tid]
        if isinstance(case, fx.coop.BlockReduceAlgorithm):
            collective = fx.coop.BlockReduce[fx.Int32, block, case]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            commutative_only = case in (
                fx.coop.BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
                fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
            )
            op = fx.ReductionOp.ADD if commutative_only else affine
            result = collective(value, op, storage=storage, valid_items=67)
        elif isinstance(case, fx.coop.BlockScanAlgorithm):
            collective = fx.coop.BlockScan[fx.Int32, block, case]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            result = collective.exclusive(value, affine, storage=storage, init=fx.Int32(65539), valid_items=67)
        out[tid] = result

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)


def run_numeric(kernel, block, columns):
    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(Out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.zeros(block * columns, dtype=torch.int64, device="cuda")
    launch(out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu().reshape(block, columns)


def check_boolean_partial_identities(policy, op, valid):
    block = 64 if policy == "warp" else 128

    def apply(Out):
        tid = fx.thread_idx.x
        value = (tid % 7) != 0
        if isinstance(policy, fx.coop.BlockReduceAlgorithm):
            collective = fx.coop.BlockReduce[fx.Boolean, block, policy]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            result = collective(value, op, storage=storage, valid_items=valid)
            Out[tid * 2] = fx.Int64(result)
        else:
            collective = fx.coop.BlockScan[fx.Boolean, block, policy]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            result, aggregate = collective.exclusive_with_aggregate(value, op, storage=storage, valid_items=valid)
            Out[tid * 2], Out[tid * 2 + 1] = fx.Int64(result), fx.Int64(aggregate)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(Out: fx.Tensor):
        apply(Out)

    out = run_numeric(kernel, block, 2)
    fold = all if op is fx.ReductionOp.MIN else any
    for tid in range(block):
        base = (tid // 8) * 8 if policy == "warp" else 0
        total = int(fold(i % 7 != 0 for i in range(base, base + valid)))
        prefix = int(fold(i % 7 != 0 for i in range(base, base + min(tid - base, valid))))
        if isinstance(policy, fx.coop.BlockReduceAlgorithm):
            assert out[tid, 0] == total
        else:
            assert out[tid].tolist() == [prefix, total]


def pair(value):
    raw = value & ((1 << 128) - 1)
    words = [raw & ((1 << 64) - 1), raw >> 64]
    return [word if word < (1 << 63) else word - (1 << 64) for word in words]


def wide_value(dtype, tid):
    positive = (dtype(tid + 1) << 80) + (dtype(tid % 7) << 60) + dtype(tid)
    return (tid % 3 == 0).select(-positive, positive)


def store_wide(out, offset, value):
    bits = value.bitcast(fx.Uint128)
    out[offset] = bits.to(fx.Int64)
    out[offset + 1] = (bits >> 64).to(fx.Int64)


POLICY_DTYPES = (
    (fx.Uint8, torch.uint8),
    (fx.Int16, torch.int16),
    (fx.Int32, torch.int32),
    (fx.Int64, torch.int64),
    (fx.Float32, torch.float32),
    (fx.Float64, torch.float64),
)


POLICY_OPS = (fx.ReductionOp.ADD, fx.ReductionOp.MUL, fx.ReductionOp.MIN, fx.ReductionOp.MAX)


def policy_samples(dtype, op, count):
    index = torch.arange(count, dtype=torch.int64, device="cpu")
    if op is fx.ReductionOp.MUL:
        if dtype.is_floating_point:
            return (1 + ((index % 7) - 3) * 0.001).to(dtype)
        return torch.where(index % 3 == 0, -1, 1).to(dtype)
    values = (index * 29 % 251) - 125
    if dtype is torch.int64:
        values *= 1 << 35
    return values.to(dtype)


def policy_reference(values, op):
    dtype = values.dtype
    if op is fx.ReductionOp.ADD:
        inclusive, neutral = values.cumsum(0, dtype=dtype), 0
    elif op is fx.ReductionOp.MUL:
        inclusive, neutral = values.cumprod(0, dtype=dtype), 1
    elif op is fx.ReductionOp.MIN:
        inclusive = values.cummin(0).values
        neutral = float("inf") if dtype.is_floating_point else torch.iinfo(dtype).max
    else:
        inclusive = values.cummax(0).values
        neutral = float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min
    exclusive = torch.cat((torch.tensor([neutral], dtype=dtype, device="cpu"), inclusive[:-1]))
    return inclusive, exclusive


def check_policy_dtype_operation_matrix(policy, dtype, torch_dtype, op):
    block, items = 128, 9
    is_reduction = isinstance(policy, fx.coop.BlockReduceAlgorithm)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        root = fx.coop.universal.BlockReduce if is_reduction else fx.coop.universal.BlockScan
        collective = root[dtype, block, policy]
        storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        if fx.const_expr(is_reduction):
            Out[tid] = collective(vector, op, storage=storage)
        else:
            inclusive, aggregate = collective.inclusive_with_aggregate(vector, op, storage=storage)
            fx.barrier()
            exclusive = collective.exclusive(vector, op, storage=storage)
            for i in fx.range_constexpr(items):
                Out[tid * items + i] = inclusive[i]
                Out[block * items + tid * items + i] = exclusive[i]
            Agg[tid] = aggregate

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    host = policy_samples(torch_dtype, op, block * items)
    values = host.cuda()
    out = torch.empty(2 * block * items, dtype=torch_dtype, device="cuda")
    aggregate = torch.empty(block, dtype=torch_dtype, device="cuda")
    launch(values, out, aggregate, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    inclusive, exclusive = policy_reference(host, op)
    if is_reduction:
        actual = out[:block].cpu()
        expected = inclusive[-1].expand(block)
    else:
        actual = torch.cat((out.cpu(), aggregate.cpu()))
        expected = torch.cat((inclusive, exclusive, inclusive[-1].expand(block)))
    if torch_dtype.is_floating_point:
        tolerance = 1e-5 if torch_dtype is torch.float32 else 1e-12
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
    else:
        assert torch.equal(actual, expected)


# Nested record kernels and references shared by reduction and scan tests.


@fx.Struct
class Affine:
    a: fx.Int32
    b: fx.Int32


@fx.Struct
class Item:
    transform: Affine
    count: fx.Int64


def compose(lhs, rhs):
    return Item(
        transform=Affine(
            a=(lhs.transform.a * rhs.transform.a) % 251, b=(lhs.transform.a * rhs.transform.b + lhs.transform.b) % 251
        ),
        count=lhs.count + rhs.count,
    )


def compose_identity(dtype):
    return dtype(transform=Affine(a=fx.Int32(1), b=fx.Int32(0)), count=fx.Int64(0))


compose.identity = compose_identity


def record_add(lhs, rhs):
    return Item(
        transform=Affine(a=(lhs.transform.a + rhs.transform.a) % 251, b=(lhs.transform.b + rhs.transform.b) % 251),
        count=lhs.count + rhs.count,
    )


def additive_identity(dtype):
    return dtype(transform=Affine(a=fx.Int32(0), b=fx.Int32(0)), count=fx.Int64(0))


record_add.identity = additive_identity


record_add.commutative = True


def make_item(value, index):
    return Item(transform=Affine(a=(value >> 16) & 65535, b=value & 65535), count=(fx.Int64(index) + 1) * (1 << 33))


def store_item(out, index, value):
    out[index * 3] = fx.Int64(value.transform.a)
    out[index * 3 + 1] = fx.Int64(value.transform.b)
    out[index * 3 + 2] = value.count


def data(count):
    index = torch.arange(count, dtype=torch.int32, device="cpu")
    return (((index * 13 + 3) % 251) << 16) | ((index * 7 + 11) % 251)


def reference(host, indices, additive=False):
    a, b, total = (0 if additive else 1), 0, 0
    for index in indices:
        c, d = (int(host[index]) >> 16) & 65535, int(host[index]) & 65535
        a, b = ((a + c) % 251, (b + d) % 251) if additive else ((a * c) % 251, (a * d + b) % 251)
        total += (index + 1) * (1 << 33)
    return [a, b, total]


def launch_case(kernel, block, values, output_items, aggregate_items):
    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(output_items * 3, dtype=torch.int64, device="cuda")
    agg = torch.empty(aggregate_items * 3, dtype=torch.int64, device="cuda")
    launch(values.cuda(), out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu().reshape(-1, 3), agg.cpu().reshape(-1, 3)


def check_nested_record_reduce_policies(policy, valid):
    block, items = 128, 3 if valid is None else 1
    additive = policy in (
        fx.coop.BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
    )
    operation = record_add if additive else compose

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        collective = fx.coop.universal.BlockReduce[Item, block, policy]
        storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
        tid = fx.thread_idx.x
        values = tuple(make_item(A[tid * items + i], tid * items + i) for i in range(items))
        result = collective(
            values if valid is None else values[0],
            operation,
            storage=storage,
            valid_items=None if valid is None else fx.Int32(valid),
        )
        store_item(Out, tid, result)

    host = data(block * items)
    out, _ = launch_case(kernel, block, host, block, block)
    expected = reference(host, range(len(host) if valid is None else valid), additive=additive)
    assert out.tolist() == [expected] * block


def check_nested_record_scan_policies(policy, block):
    items, valid = 3, block * 3

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        collective = fx.coop.universal.BlockScan[Item, block, policy]
        storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
        tid = fx.thread_idx.x
        values = tuple(make_item(A[tid * items + i], tid * items + i) for i in range(items))
        result, aggregate = collective.exclusive_with_aggregate(values, compose, storage=storage)
        for i in fx.range_constexpr(items):
            store_item(Out, tid * items + i, result[i])
        store_item(Agg, tid, aggregate)

    host = data(block * items)
    out, agg = launch_case(kernel, block, host, block * items, block)
    assert out.tolist() == [reference(host, range(min(i, valid))) for i in range(len(host))]
    assert agg.tolist() == [reference(host, range(valid))] * block


def plain_compose(lhs, rhs):
    return compose(lhs, rhs)


def record_prefix(aggregate):
    return Item(
        transform=Affine(a=fx.Int32(2) + fx.thread_idx.x, b=fx.Int32(7)), count=fx.Int64(11) + fx.Int64(fx.thread_idx.x)
    )


def check_record_semigroup_exclusive_prefix_forms(policy, prefix):
    block, items = 128, 3

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        collective = fx.coop.BlockScan[Item, block, policy]
        storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
        tid = fx.thread_idx.x
        values = tuple(make_item(A[tid * items + i], tid * items + i) for i in range(items))
        initial = Item(transform=Affine(a=fx.Int32(2), b=fx.Int32(7)), count=fx.Int64(11)) if prefix == "init" else None
        callback = record_prefix if prefix == "callback" else None
        result, aggregate = collective.exclusive_with_aggregate(
            values, plain_compose, storage=storage, init=initial, prefix_callback=callback
        )
        for i in fx.range_constexpr(items):
            store_item(Out, tid * items + i, result[i])
        store_item(Agg, tid, aggregate)

    host = data(block * items)
    out, agg = launch_case(kernel, block, host, block * items, block)
    for i in range(1 if prefix == "unseeded" else 0, len(host)):
        expected = reference(host, range(i))
        if prefix != "unseeded":
            expected = [(2 * expected[0]) % 251, (2 * expected[1] + 7) % 251, expected[2] + 11]
        assert out[i].tolist() == expected
    assert agg.tolist() == [reference(host, range(len(host)))] * block


def check_record_collectives_compile(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block = 128

    def apply(A, Out):
        tid = fx.thread_idx.x
        value = make_item(A[tid], tid)
        if isinstance(case, fx.coop.BlockReduceAlgorithm):
            collective = fx.coop.BlockReduce[Item, block, case]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            commutative = case in (
                fx.coop.BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
                fx.coop.BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
            )
            result = collective(value, record_add if commutative else compose, storage=storage, valid_items=67)
        elif isinstance(case, fx.coop.BlockScanAlgorithm):
            collective = fx.coop.BlockScan[Item, block, case]
            storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
            result = collective.exclusive(value, compose, storage=storage, prefix_callback=record_prefix)
        store_item(Out, tid, result)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        apply(A, Out)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(data(block), torch.empty(block * 3, dtype=torch.int64, device="cpu"))


def check_record_reduce_semigroup_without_identity(policy, valid):
    block, items = 128, 3 if valid is None else 1

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        collective = fx.coop.BlockReduce[Item, block, policy]
        storage = fx.SharedAllocator().allocate(collective.SharedStorage).peek()
        tid = fx.thread_idx.x
        values = tuple(make_item(A[tid * items + i], tid * items + i) for i in range(items))
        result = collective(
            values if valid is None else values[0],
            plain_compose,
            storage=storage,
            valid_items=None if valid is None else fx.Int32(valid),
        )
        store_item(Out, tid, result)

    host = data(block * items)
    out, _ = launch_case(kernel, block, host, block, block)
    # Empty semigroup aggregates are unspecified, but the collective still
    # executes safely without requiring an invented identity.
    if valid is None or valid:
        assert out.tolist() == [reference(host, range(len(host) if valid is None else valid))] * block


# Shared host-reference checks for neighborhood, histogram, and run primitives.

Detail = fx.Struct["weight" : fx.Float64, "tag" : fx.Uint16]
Record = fx.Struct["key" : fx.Int32, "detail":Detail]


def check_neighbors(block, count, boundary, method):
    threads = block if isinstance(block, int) else math.prod(block)
    n = threads * count
    # Runs cross item, thread and warp boundaries.
    values = (torch.arange(n, device="cuda", dtype=torch.int32) // 7) % 3
    both = method == "flag_heads_and_tails"
    left = method in ("up", "subtract_left", "flag_heads")

    def transform(value):
        name = (
            "BlockDiscontinuity"
            if method.startswith("flag")
            else "BlockAdjacentDifference" if method.startswith("subtract") else "BlockShuffle"
        )
        primitive = getattr(fx.coop, name)[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        kwargs = (
            {"predecessor": boundary, "successor": boundary}
            if both
            else {"predecessor" if left else "successor": boundary}
        )
        return getattr(primitive, method)(value, storage=storage, **kwargs)

    result = run_tile(transform, values, block, count, outputs=2 if both else 1)
    host = values.cpu()
    for j, direction in enumerate([True, False] if both else [left]):
        neighbor = host.roll(1 if direction else -1)
        endpoint = 0 if direction else -1
        neighbor[endpoint] = host[endpoint] if boundary is None else boundary
        if method.startswith("flag"):
            expected = (neighbor != host).int()
            if boundary is None:
                expected[endpoint] = 1
        elif method.startswith("subtract"):
            expected = host - neighbor
            if boundary is None:
                expected[endpoint] = host[endpoint]
        else:
            expected = neighbor
        torch.testing.assert_close(result[j], expected)


def check_histogram(block, bins, count, composite, entry=(fx.Int32, "int32"), universal=False):
    dtype, name = entry
    ns = fx.coop.universal if universal else fx.coop
    values = (torch.arange(block * count, device="cuda", dtype=torch.int32) * 37) % (bins + 3) - 1

    values = values.to(getattr(torch, name))

    def apply(a, out):
        hist = ns.BlockHistogram[dtype, block, bins]
        storage = fx.SharedAllocator().allocate(hist.SharedStorage).peek()
        tid = fx.thread_idx.x
        items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)], dtype=dtype)
        counts = hist(items[0] if count == 1 else items, storage=storage)
        if composite:
            counts = hist.composite(items[0] if count == 1 else items, storage=storage)
        return counts

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        counts = apply(a, out)
        for i in fx.range_constexpr((bins + block - 1) // block):
            index = fx.thread_idx.x + i * block
            if index < bins:
                out[index] = counts[index]

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(bins, device="cuda", dtype=torch.int32)
    launch(values, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    host = values.cpu()
    expected = host[(host >= 0) & (host < bins)].long().bincount(minlength=bins).int() * (2 if composite else 1)
    torch.testing.assert_close(out.cpu(), expected)


def check_histogram_key_conversion(entry, algorithm):
    # Wide integer and floating-point keys are range-checked before the Int32
    # bin conversion, so truncation cannot move an out-of-range key into a bin.
    dtype, name = entry
    block, bins, count = 64, 7, 2
    torch_dtype = getattr(torch, name)
    if torch_dtype.is_floating_point:
        special = [float("nan"), -0.5, -1.0, 2.7, bins - 0.5, bins, float("inf"), -float("inf"), 6e4]
    else:
        special = [2**32 + 1, -(2**32) + 2, 2**40 + 3, -1, bins, 2**63 - 1]
    values = ((torch.arange(block * count, device="cpu") * 37) % (bins + 3) - 1).to(torch_dtype)
    values[: len(special)] = torch.tensor(special, dtype=torch_dtype, device="cpu")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        hist = fx.coop.BlockHistogram[dtype, block, bins, algorithm]
        storage = fx.SharedAllocator().allocate(hist.SharedStorage).peek()
        tid = fx.thread_idx.x
        counts = hist(fx.Vector.from_elements([a[tid * count + i] for i in range(count)], dtype=dtype), storage=storage)
        if tid < bins:
            out[tid] = counts[tid]

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(bins, device="cuda", dtype=torch.int32)
    launch(values.cuda(), out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    if torch_dtype.is_floating_point:
        host = values.double()
        indices = host[(host > -1) & (host < bins)].trunc().long()
    else:
        indices = values[(values >= 0) & (values < bins)].long()
    torch.testing.assert_close(out.cpu(), indices.bincount(minlength=bins).int())


def check_run_length_decode(block, runs, count, offset, empty, universal, entry=(fx.Int32, "int32")):
    dtype, name = entry
    values = torch.arange(block * runs, device="cuda", dtype=torch.int32) * 11
    lengths = torch.zeros_like(values) if empty else ((values + 1) % 5)
    values = values.to(getattr(torch, name))

    def apply(a, lengths, out, relative, totals):
        ns = fx.coop.universal if universal else fx.coop
        decode = ns.BlockRunLengthDecode[dtype, block, runs]
        storage = fx.SharedAllocator().allocate(decode.SharedStorage).peek()
        tid = fx.thread_idx.x
        keys = fx.Vector.from_elements([a[tid * runs + i] for i in range(runs)], dtype=dtype)
        lens = fx.Vector.from_elements([lengths[tid * runs + i] for i in range(runs)])
        decoded, offsets, total = decode(
            keys[0] if runs == 1 else keys,
            lens[0] if runs == 1 else lens,
            storage=storage,
            decoded_items_per_thread=count,
            offset=offset,
            default=-9,
        )
        for i in range(count):
            out[tid * count + i] = decoded[i]
            relative[tid * count + i] = offsets[i]
        totals[tid] = total

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, lengths: fx.Tensor, out: fx.Tensor, relative: fx.Tensor, totals: fx.Tensor):
        apply(a, lengths, out, relative, totals)

    @flyc.jit
    def launch(
        a: fx.Tensor,
        lengths: fx.Tensor,
        out: fx.Tensor,
        relative: fx.Tensor,
        totals: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(a, lengths, out, relative, totals).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(block * count, device="cuda", dtype=getattr(torch, name))
    relative = torch.empty(block * count, device="cuda", dtype=torch.int32)
    totals = torch.empty(block, device="cuda", dtype=torch.int32)
    launch(values, lengths, out, relative, totals, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    expanded = values.cpu().repeat_interleave(lengths.cpu().long())
    positions = torch.cat([torch.arange(n, device="cpu") for n in lengths.cpu().tolist()]).int()
    expected = torch.full((block * count,), -9, dtype=getattr(torch, name), device="cpu")
    expected_relative = torch.full((block * count,), -1, dtype=torch.int32, device="cpu")
    take = max(0, min(block * count, expanded.numel() - offset))
    expected[:take] = expanded[offset : offset + take]
    expected_relative[:take] = positions[offset : offset + take]
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(relative.cpu(), expected_relative)
    torch.testing.assert_close(totals.cpu(), torch.full((block,), expanded.numel(), dtype=torch.int32, device="cpu"))


def check_neighborhood_contract_dtypes(dtype, name, method):
    block, count, limit = 128, 9, 579
    host = ((torch.arange(block * count, dtype=torch.int64, device="cpu") * 19) % 43).to(getattr(torch, name))
    values = host.cuda()

    def transform(value):
        if method.startswith("partial"):
            cls = fx.coop.universal.BlockAdjacentDifference[dtype, block]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            fn = cls.subtract_left if method == "partial_left" else cls.subtract_right
            # Bounded positive output exercises unsigned restoration too.
            return fn(
                value,
                lambda a, b: (a + b) / 2 if dtype.is_float else (a + b) // 2,
                storage=storage,
                valid_items=fx.Int32(limit),
            )
        cls = fx.coop.universal.BlockDiscontinuity[dtype, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        fn = cls.flag_heads if method == "indexed_heads" else cls.flag_tails
        return fn(value, lambda a, b, i: (a < b) | ((i % 5) == 0), storage=storage).to(dtype)

    actual = run_tile(transform, values, block, count, dtype=dtype)[0]
    left = method in ("partial_left", "indexed_heads")
    neighbor = host.roll(1 if left else -1)
    index = torch.arange(host.numel(), device="cpu")
    if method.startswith("partial"):
        expected = (host.to(torch.float64) + neighbor.to(torch.float64)) / 2
        if not dtype.is_float:
            expected = expected.floor()
        expected = expected.to(host.dtype)
        active = (index > 0) & (index < limit) if left else index < limit - 1
        expected = torch.where(active, expected, host)
    else:
        expected = ((neighbor < host if left else host < neighbor) | (((index if left else index + 1) % 5) == 0)).to(
            host.dtype
        )
        expected[0 if left else -1] = 1
    torch.testing.assert_close(actual, expected)


def check_sequence_contracts_compile(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = 128, 3
    if case == "histogram" and not full_matrix_enabled():
        count = 2

    def indexed(a, b, index):
        return (a < b) | ((index % 5) == 0)

    def apply(a, out):
        ns = fx.coop.universal
        tid = fx.thread_idx.x
        items = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if case == "histogram":
            cls = ns.BlockHistogram[(fx.Int32, fx.Int64), block, 7, ns.BlockHistogramAlgorithm.SORT]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            result = cls(items % 7, storage=storage)
            out[tid] = result[tid % 7].to(fx.Int32)
        elif case.startswith("decode"):
            cls = ns.BlockRunLengthDecode[(fx.Int32, fx.Uint64), block, count]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            if case == "decode_lengths":
                state = cls.init_with_run_lengths(items, items & 3, storage=storage)
            else:
                offsets = fx.Vector.from_elements([(tid * count + i).to(fx.Uint64) * 10000000000 for i in range(count)])
                state = cls.init_with_run_offsets(items, offsets, storage=storage)
            first, _, _ = state.decode(offset=3)
            second, _, _ = state.decode(offset=10000000001)
            out[tid] = first[0] + second[0]
        else:
            name = {
                "difference": "BlockAdjacentDifference",
                "discontinuity": "BlockDiscontinuity",
                "shuffle": "BlockShuffle",
            }[case]
            cls = getattr(ns, name)[fx.Int32, block]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            if case == "difference":
                result = cls.subtract_right_partial_tile(items, fx.Int32(193), storage=storage)
            elif case == "discontinuity":
                result, _, _ = cls.flag_heads_and_tails(items, indexed, storage=storage, return_predecessors=True)
            else:
                result, suffix = cls.up(items, storage=storage, return_suffix=True)
                result = result + suffix
            out[tid] = result[0]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)


def check_sequence_records(block, count, method, universal, compile_only=False):
    ns = fx.coop.universal if universal else fx.coop
    size = block * count
    host = torch.arange(size, dtype=torch.int32, device="cpu")
    values = host if compile_only else host.cuda()
    lengths = (host % 4).long()
    limit = size - 1

    def difference(a, b):
        return Record(a.key * 3 - b.key, Detail(a.detail.weight - b.detail.weight * 2, a.detail.tag))

    def flag(a, b, right_index):
        return (a.key < b.key) & ((right_index % 3) != 0)

    def apply(a, out, weights, tags):
        tid = fx.thread_idx.x
        records = tuple(
            Record(
                a[tid * count + i],
                Detail((a[tid * count + i]).to(fx.Float64) + 0.5, (a[tid * count + i] + 60000).to(fx.Uint16)),
            )
            for i in range(count)
        )
        value = records[0] if count == 1 else records
        if method in ("up", "down"):
            cls = ns.BlockShuffle[Record, block]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            result = getattr(cls, method)(value, storage=storage)
        elif method in ("left", "right"):
            cls = ns.BlockAdjacentDifference[Record, block]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            result = getattr(cls, "subtract_" + method)(value, difference, storage=storage, valid_items=limit)
        elif method in ("heads", "tails"):
            cls = ns.BlockDiscontinuity[Record, block]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            flags = getattr(cls, "flag_" + method)(value, flag, storage=storage)
            result = tuple(Record(item, Detail(item.to(fx.Float64), item.to(fx.Uint16))) for item in as_items(flags))
        else:
            cls = ns.BlockRunLengthDecode[(Record, fx.Int64), block, count]
            storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
            lens = fx.Vector.from_elements([record.key.to(fx.Int64) % 4 for record in records])
            state = cls.init_with_run_lengths(value, lens[0] if count == 1 else lens, storage=storage)
            result, _, _ = state.decode(decoded_items_per_thread=count, offset=1)
        for i, record in enumerate(as_items(result)):
            index = tid * count + i
            out[index] = record.key
            weights[index] = record.detail.weight
            tags[index] = record.detail.tag.to(fx.Int32)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor, weights: fx.Tensor, tags: fx.Tensor):
        apply(a, out, weights, tags)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, weights: fx.Tensor, tags: fx.Tensor, stream: fx.Stream):
        kernel(a, out, weights, tags).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out, tags = torch.empty_like(values), torch.empty_like(values)
    weights = torch.empty(size, dtype=torch.float64, device=values.device)
    launch(values, out, weights, tags, fx.Stream(None) if compile_only else torch.cuda.current_stream())
    if compile_only:
        return
    torch.cuda.synchronize()
    expected_key, expected_weight, expected_tag = host.clone(), host.double() + 0.5, host + 60000
    if method in ("up", "down", "left", "right"):
        left = method in ("up", "left")
        neighbor = host.roll(1 if left else -1)
        neighbor[0 if left else -1] = host[0 if left else -1]
        if method in ("up", "down"):
            expected_key, expected_weight, expected_tag = neighbor, neighbor.double() + 0.5, neighbor + 60000
        else:
            for i in range(limit):
                if (left and i == 0) or (not left and i == limit - 1):
                    continue
                expected_key[i] = host[i] * 3 - neighbor[i]
                expected_weight[i] = host[i].double() + 0.5 - (neighbor[i].double() + 0.5) * 2
    elif method in ("heads", "tails"):
        indices = torch.arange(size, device="cpu")
        left = method == "heads"
        neighbor = host.roll(1 if left else -1)
        expected_key = (
            (neighbor < host if left else host < neighbor) & (((indices if left else indices + 1) % 3) != 0)
        ).int()
        expected_key[0 if left else -1] = 1
        expected_weight, expected_tag = expected_key.double(), expected_key
    else:
        expanded = host.repeat_interleave(lengths)[1 : 1 + size]
        expected_key = torch.zeros_like(host)
        expected_weight = torch.zeros(size, dtype=torch.float64, device="cpu")
        expected_tag = torch.zeros_like(host)
        expected_key[: len(expanded)] = expanded
        expected_weight[: len(expanded)] = expanded.double() + 0.5
        expected_tag[: len(expanded)] = expanded + 60000
    torch.testing.assert_close(out.cpu(), expected_key)
    torch.testing.assert_close(weights.cpu(), expected_weight)
    torch.testing.assert_close(tags.cpu(), expected_tag)


def check_composite_default_flags(kind):
    # The default predicate flags a boundary when any component or field differs.
    block, count = 64, 2
    size = block * count
    host = torch.arange(size, dtype=torch.int32, device="cpu")
    dtype = fx.Vector[fx.Int32, 2] if kind == "vector" else Record

    def item(index):
        if kind == "vector":
            return fx.Vector.from_elements([index // 3, index // 4], dtype=fx.Int32)
        return Record(index // 3, Detail((index // 4).to(fx.Float64), (index // 5).to(fx.Uint16)))

    def apply(a, out):
        tid = fx.thread_idx.x
        cls = fx.coop.BlockDiscontinuity[dtype, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        items = tuple(item(a[tid * count + i]) for i in range(count))
        heads, tails = cls.flag_heads_and_tails(items, storage=storage)
        for i, (head, tail) in enumerate(zip(as_items(heads), as_items(tails))):
            out[tid * count + i] = head
            out[size + tid * count + i] = tail

    result = run_kernel(apply, host.cuda(), 2 * size, block)
    fields = torch.stack([host // 3, host // 4] if kind == "vector" else [host // 3, host // 4, host // 5], dim=1)
    change = (fields[1:] != fields[:-1]).any(dim=1).int()
    one = torch.ones(1, dtype=torch.int32, device="cpu")
    torch.testing.assert_close(result, torch.cat((one, change, change, one)))


# Kernel runners and host references shared by sorting tests.


def _striped_to_linear(x, width, count):
    return x.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)


def _linear_to_striped(x, width, count):
    return x.reshape(-1, count, width).transpose(1, 2).reshape(-1)


_FLOATS = [
    (fx.Float16, torch.float16),
    (fx.BFloat16, torch.bfloat16),
    (fx.Float32, torch.float32),
    (fx.Float64, torch.float64),
]


_SORTS = ["bitonic", "warp_merge", "block_merge", "radix_scan", "radix_digit", "topk_merge", "air"]


_PATTERN = [-7.75, -0.0, 2.125, -3.5, 2.125, 0.0, 0.1875, -7.75, 4.25, -0.5, float("inf"), -float("inf")]


def _namespace(universal):
    return fx.coop.universal if universal else fx.coop


def _logical(physical, width, count, striped=False):
    if striped:
        return physical.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)
    return physical.reshape(-1, width * count)


def _striped(logical, width, count):
    return logical.reshape(-1, count, width).transpose(1, 2).reshape(-1)


def _assert_key_payload(keys, ids, source):
    expected = source[ids]
    torch.testing.assert_close(keys, expected.double(), rtol=0, atol=0)
    assert torch.equal(torch.signbit(keys), torch.signbit(expected))


def _ordered_float_bits(values, dtype):
    width = torch.finfo(dtype).bits
    integers = torch.int16 if width == 16 else torch.int32
    raw = values.to(dtype).view(integers).long() & ((1 << width) - 1)
    raw = torch.where(values == 0, 0, raw)
    sign = 1 << (width - 1)
    return torch.where((raw & sign) != 0, ~raw & ((1 << width) - 1), raw ^ sign)


def check_sort(primitive, block, count, descending):
    torch.manual_seed(14)
    values = torch.randint(-11, 12, (block * count,), dtype=torch.int32, device="cuda")
    width = min(8, block)

    def transform(value):
        sort = fx.coop.BlockMergeSort[fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        return sort(value, storage=storage, descending=descending)

    result = run_tile(transform, values, block, count)[0]
    group = width * count if primitive.startswith("warp") else block * count
    expected = values.cpu().reshape(-1, group).sort(dim=1, descending=descending).values.reshape(-1)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, width).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(result, expected)


def check_radix_sort(dtype, torch_dtype, descending, count):
    block = 128
    dt = getattr(torch, torch_dtype)
    if dt.is_floating_point:
        data = [float("-inf"), -0.0, 0.0, float("inf"), -1.25, 2.75] * 4
    elif torch_dtype == "uint8":
        data = [255, 0, 1, 128, 127, 1] * 4
    else:
        info = torch.iinfo(dt)
        data = [info.min, -1, 0, info.max, 1, -1] * 4
    data = (data * ((block * count + len(data) - 1) // len(data)))[: block * count]
    values = torch.tensor(data, dtype=dt, device="cuda")

    def transform(value):
        sort = fx.coop.BlockRadixSort[dtype, block, count]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        return sort(value, storage=storage, descending=descending)

    result = run_tile(transform, values, block, count, dtype=dtype)[0]
    expected = values.cpu().sort(descending=descending, stable=True).values
    torch.testing.assert_close(
        result, expected, rtol=0, atol=0, msg=lambda msg: f"{msg} actual={result.tolist()} expected={expected.tolist()}"
    )
    if dt.is_floating_point:
        assert torch.equal(torch.signbit(result), torch.signbit(expected))


def check_topk(k, largest):
    values = torch.arange(192, device="cuda", dtype=torch.int32) * 13 % 97

    def transform(value):
        topk = fx.coop.BlockTopK[fx.Int32, 64, 3]
        storage = fx.SharedAllocator().allocate(topk.SharedStorage).peek()
        keys, valid = topk(value, k, storage=storage, largest=largest)
        return keys, valid.to(fx.Int32)

    result = run_tile(transform, values, 64, 3, outputs=2)
    torch.testing.assert_close(result[0, :k], values.cpu().sort(descending=largest).values[:k])
    torch.testing.assert_close(result[1], (torch.arange(values.numel(), device="cpu") < k).int())


def check_merge_comparator_pairs_partial(scope, block, descending):
    count, width = 3, 8
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)
    total = width * count if scope == "warp" else block * count
    valid = total - 5

    def transform(value):
        keys = (value * 19) % 43
        op = lambda a, b: a % 5 < b % 5
        sort = fx.coop.BlockMergeSort[(fx.Int32, fx.Int32), block, count]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        return sort(
            keys,
            value,
            storage=storage,
            compare_op=op,
            descending=descending,
            valid_items=fx.Int32(valid),
            oob_default=999,
        )

    result = run_tile(transform, data, block, count, outputs=2)
    ids = data.cpu().reshape(-1, total)
    keys = (ids * 19) % 43
    permutation = (keys[:, :valid] % 5).argsort(dim=1, descending=descending, stable=True)
    expected_ids = torch.gather(ids[:, :valid], 1, permutation)
    torch.testing.assert_close(result[1].reshape(-1, total)[:, :valid], expected_ids)
    torch.testing.assert_close(result[0].reshape(-1, total)[:, :valid], (expected_ids * 19) % 43)
    assert (result[0].reshape(-1, total)[:, valid:] == 999).all()


def check_radix_pairs_decomposer(algorithm, striped):
    block, count = 64, 3
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = (value * 17) % 41
        sort = fx.coop.BlockRadixSort[(fx.Int32, fx.Int64), block, count, algorithm]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        return sort(
            keys,
            value.to(fx.Int64),
            storage=storage,
            blocked_to_striped=striped,
            decomposer=lambda key: ((key % 5).to(fx.Uint8), (key // 5).to(fx.Uint8)),
            end_bit=12,
        )

    result = run_tile(transform, data, block, count, outputs=2)
    keys = data.cpu() * 17 % 41
    decomposed = (keys % 5) * 256 + keys // 5
    order = decomposed.argsort(stable=True)
    expected_keys, expected_ids = keys[order], data.cpu()[order]
    if striped:
        expected_keys = _linear_to_striped(expected_keys, block, count)
        expected_ids = _linear_to_striped(expected_ids, block, count)
    torch.testing.assert_close(result[0], expected_keys)
    torch.testing.assert_close(result[1], expected_ids)


def check_topk_runtime_pairs_partial(algorithm, largest, k):
    block, count, valid = 64, 3, 41
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = value * 19 % 13
        topk = fx.coop.BlockTopK[(fx.Int32, fx.Int64), block, count, algorithm]
        storage = fx.SharedAllocator().allocate(topk.SharedStorage).peek()
        # Numeric SSA values exercise the runtime API, including k = 0.
        result, payload, selected = topk(
            keys,
            value[0] - value[0] + k,
            value.to(fx.Int64),
            storage=storage,
            num_valid=value[0] - value[0] + valid,
            largest=largest,
        )
        return result, payload.to(fx.Int32), selected.to(fx.Int32)

    result = run_tile(transform, data, block, count, outputs=3)
    keys = data.cpu() * 19 % 13
    order = keys[:valid].argsort(descending=largest, stable=True)[:k]
    torch.testing.assert_close(result[0, :k].sort().values, keys[order].sort().values)
    torch.testing.assert_close(result[1, :k].sort().values, order.int().sort().values)
    torch.testing.assert_close(result[0, :k], keys[result[1, :k].long()])
    torch.testing.assert_close(result[2], (torch.arange(block * count, device="cpu") < k).int())
    if algorithm is fx.coop.BlockTopKAlgorithm.MERGE:
        torch.testing.assert_close(result[0, :k], keys[order])


def check_large_tile_low_shared_memory(name):
    block, count = 1024, 9

    def operator(count):
        if name == "radix":
            op = fx.coop.BlockRadixSort[fx.Int32, block, count]
        elif name == "rank":
            op = fx.coop.BlockRadixRank[fx.Int32, block, count]
        elif name == "merge":
            op = fx.coop.BlockMergeSort[fx.Int32, block, count]
        else:
            op = fx.coop.BlockTopK[
                fx.Int32,
                block,
                count,
                fx.coop.BlockTopKAlgorithm.AIR if name == "air" else fx.coop.BlockTopKAlgorithm.MERGE,
            ]
        assert op.SharedStorage.__dsl_size_of__() <= 65536
        return op

    # Keep the original 1024 x 9 storage bound in every run. Executing that
    # combined stress shape is exhaustive-only; default GPU coverage still
    # allocates the large storage and exercises all 1024 threads.
    large_storage = operator(count).SharedStorage
    if name in ("merge", "topk") and not full_matrix_enabled():
        count = 1
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 37 % 251

    def transform(value):
        # Allocate both types in the same union so the default run retains the
        # original LDS footprint without compiling a nine-item merge network.
        op = operator(count)
        allocator = fx.SharedAllocator()
        scratch = fx.Union["large":large_storage, "active" : op.SharedStorage]
        storage = allocator.allocate(scratch).active.peek()
        if name in ("topk", "air"):
            return op(value, 17, storage=storage)[0]
        if name in ("rank", "radix"):
            return op(value, storage=storage, end_bit=8)
        return op(value, storage=storage)

    result = run_tile(transform, data, block, count)[0]
    expected = data.cpu().sort(descending=name in ("topk", "air"), stable=True).values
    if name == "rank":
        order = data.cpu().argsort(stable=True)
        expected = torch.empty_like(order)
        expected[order] = torch.arange(len(order), device="cpu")
        expected = expected.int()
    if name in ("topk", "air"):
        torch.testing.assert_close(result[:17].sort().values, expected[:17].sort().values)
    else:
        torch.testing.assert_close(result, expected)


def check_wide_numeric_policies(dtype, name, case):
    block, count = 64, 3
    dt = getattr(torch, name)
    if dt.is_floating_point:
        pattern = [-float("inf"), -0.0, 0.0, float("inf"), -3.5, 2.25, -0.0, 0.0]
    elif name == "uint8":
        pattern = [255, 0, 1, 128, 254, 1, 127, 0]
    else:
        pattern = [-(1 << 63), (1 << 63) - 1, -(1 << 50), (1 << 50), 0, 0, -1, 1]
    data = torch.tensor(
        (pattern * ((block * count + len(pattern) - 1) // len(pattern)))[: block * count], dtype=dt, device="cuda"
    )

    def transform(value):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)]).to(fx.Int64)
        if case == "air":
            op = fx.coop.BlockTopK[(dtype, fx.Int64), block, count, fx.coop.BlockTopKAlgorithm.AIR]
        else:
            op = fx.coop.BlockRadixSort[(dtype, fx.Int64), block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        if case == "air":
            keys, ids, valid = op(value, fx.Int32(23), ids, storage=storage, largest=True)
            return keys, ids
        return op(value, ids, storage=storage, descending=True)

    result = run_tile(transform, data, block, count, outputs=2, dtype=dtype)
    order = data.cpu().argsort(descending=True, stable=True)
    k = 23 if case == "air" else block * count
    torch.testing.assert_close(result[0, :k].sort().values, data.cpu()[order[:k]].sort().values, rtol=0, atol=0)
    if case == "air":
        torch.testing.assert_close(result[1, :k].long().sort().values, order[:k].sort().values)
    else:
        torch.testing.assert_close(result[1].long(), order)
    if dt.is_floating_point:
        assert torch.equal(torch.signbit(result[0, :k]), torch.signbit(data.cpu()[result[1, :k].long()]))


def check_compile_sorting_policies(monkeypatch, arch, case):

    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = 128, 3
    if case in ("record_merge", "air") and not full_matrix_enabled():
        count = 2

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if case in ("record_merge", "record_radix"):

            key_type = fx.Struct["major" : fx.Int32, "minor" : fx.Int64]
            keys = make_items([key_type(major=item % 13, minor=item.to(fx.Int64)) for item in value])
            if case == "record_merge":
                op = fx.coop.BlockMergeSort[(key_type, fx.Int64), block, count]
            else:
                op = fx.coop.BlockRadixSort[
                    (key_type, fx.Int64), block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT
                ]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            if case == "record_merge":
                keys, payload = op(keys, value.to(fx.Int64), storage=storage, compare_op=lambda a, b: a.major < b.major)
            else:
                keys, payload = op(
                    keys, value.to(fx.Int64), storage=storage, end_bit=13, decomposer=lambda key: (key.major, key.minor)
                )
            result = fx.Vector.from_elements([key.major for key in keys])
        elif case == "air":
            op = fx.coop.BlockTopK[(fx.Float64, fx.Int64), block, count, fx.coop.BlockTopKAlgorithm.AIR]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result = op(
                value.to(fx.Float64), fx.Int32(17), value.to(fx.Int64), storage=storage, num_valid=fx.Int32(193)
            )[0]
        elif case == "radix_digit":
            op = fx.coop.BlockRadixSort[(fx.Int64, fx.Uint8), block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result = op(value.to(fx.Int64), value.to(fx.Uint8), storage=storage, begin_bit=1, end_bit=10)[0]
        else:
            algorithm = {
                "rank_memoize": fx.coop.BlockRadixRankAlgorithm.MEMOIZE,
                "rank_match": fx.coop.BlockRadixRankAlgorithm.MATCH,
                "rank_early_any": fx.coop.BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ANY,
                "rank_early_atomic_or": fx.coop.BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ATOMIC_OR,
            }[case]
            op = fx.coop.BlockRadixRank[fx.Uint8, block, count, algorithm]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result = op(value.to(fx.Uint8), storage=storage, end_bit=3)
        for i in range(count):
            out[tid * count + i] = result[i].to(fx.Int32)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    tensor = torch.empty(block * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)


def check_record_keys_and_record_values(case, count):

    key_type = fx.Struct["major" : fx.Int32, "minor" : fx.Int64]
    value_type = fx.Struct["index" : fx.Int64, "tag" : fx.Float64]
    block, width = (8 if case in ("bitonic", "warp_merge") else 64), 8
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = make_items([key_type(major=(item * 17 % 11) - 5, minor=(item * 7 % 13).to(fx.Int64)) for item in value])
        payload = make_items([value_type(index=item.to(fx.Int64), tag=item.to(fx.Float64) * 2.0) for item in value])
        if count == 1:
            keys, payload = keys[0], payload[0]
        compare = lambda a, b: (a.major < b.major) | ((a.major == b.major) & (a.minor < b.minor))
        if case == "radix":
            op = fx.coop.BlockRadixSort[
                (key_type, value_type), block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT
            ]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result, carried = op(keys, payload, storage=storage, decomposer=lambda key: (key.major, key.minor))
        elif case == "block_merge":
            op = fx.coop.BlockMergeSort[(key_type, value_type), block, count]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result, carried = op(keys, payload, storage=storage, compare_op=compare)
        else:
            op = fx.coop.BlockTopK[(key_type, value_type), block, count]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            result, carried, valid = op(keys, 17, payload, storage=storage, largest=False, compare_op=compare)
        result, carried = as_items(result), as_items(carried)
        return (
            fx.Vector.from_elements([key.major for key in result]),
            fx.Vector.from_elements([key.minor.to(fx.Int32) for key in result]),
            fx.Vector.from_elements([item.index.to(fx.Int32) for item in carried]),
            fx.Vector.from_elements([item.tag.to(fx.Int32) for item in carried]),
        )

    result = run_tile(transform, data, block, count, outputs=4)
    ids = data.cpu()
    major, minor = ids * 17 % 11 - 5, ids * 7 % 13
    logical = _striped_to_linear(ids, width, count).flatten() if case == "bitonic" else ids
    composite = (major[logical] + 5) * 13 + minor[logical]
    order = logical[composite.argsort(stable=True)]
    k = 17 if case == "topk_merge" else len(ids)
    torch.testing.assert_close(result[0, :k], major[order[:k]])
    torch.testing.assert_close(result[1, :k], minor[order[:k]])
    torch.testing.assert_close(result[2, :k], order[:k])
    torch.testing.assert_close(result[3, :k], order[:k] * 2)


def check_air_record_payload_and_partial_record_sort():

    payload_type = fx.Struct["id" : fx.Int64, "weight" : fx.Float64]
    block, count, valid, k = 64, 3, 43, 19
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        payload = make_items([payload_type(id=item.to(fx.Int64), weight=item.to(fx.Float64) * 1.5) for item in value])
        op = fx.coop.BlockTopK[(fx.Int32, payload_type), block, count, fx.coop.BlockTopKAlgorithm.AIR]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(op.SharedStorage).peek()
        selected, carried, flags = op(value * 13 % 17, fx.Int32(k), payload, storage=storage, num_valid=fx.Int32(valid))
        ids = fx.Vector.from_elements([item.id.to(fx.Int32) for item in carried])
        weights = fx.Vector.from_elements([item.weight.to(fx.Int32) for item in carried])
        # Reuse another collective to exercise record oob defaults and scalar
        # comparator equivalence after selection has released its storage.
        sort = fx.coop.BlockMergeSort[payload_type, block, count]
        second = allocator.allocate(sort.SharedStorage).peek()
        ordered = sort(
            payload,
            storage=second,
            compare_op=lambda a, b: a.id > b.id,
            valid_items=valid,
            oob_default=payload_type(id=fx.Int64(-999), weight=fx.Float64(-999)),
        )
        sorted_ids = fx.Vector.from_elements([item.id.to(fx.Int32) for item in ordered])
        return ids, weights, sorted_ids

    result = run_tile(transform, data, block, count, outputs=3)
    keys = data.cpu() * 13 % 17
    expected = keys[:valid].argsort(descending=True, stable=True)[:k]
    torch.testing.assert_close(result[0, :k].sort().values, expected.int().sort().values)
    torch.testing.assert_close(result[1, :k], (result[0, :k].double() * 1.5).int())
    torch.testing.assert_close(result[2, :valid], torch.arange(valid - 1, -1, -1, device="cpu").int())
    assert (result[2, valid:] == -999).all()


def check_compact_record_merge():

    key_type = fx.Struct["major" : fx.Int64, "minor" : fx.Float64]
    value_type = fx.Struct["index" : fx.Int64, "tag" : fx.Float64]
    block, count = 1024, 3
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = make_items([key_type(major=(item * 37 % 251).to(fx.Int64), minor=item.to(fx.Float64)) for item in value])
        payload = make_items([value_type(index=item.to(fx.Int64), tag=item.to(fx.Float64)) for item in value])
        op = fx.coop.BlockMergeSort[(key_type, value_type), block, count]
        assert op.SharedStorage.__dsl_size_of__() <= 65536
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        keys, payload = op(keys, payload, storage=storage, compare_op=lambda a, b: a.major < b.major)
        return fx.Vector.from_elements([key.major.to(fx.Int32) for key in keys]), fx.Vector.from_elements(
            [item.index.to(fx.Int32) for item in payload]
        )

    result = run_tile(transform, data, block, count, outputs=2)
    keys = data.cpu() * 37 % 251
    order = keys.argsort(stable=True)
    torch.testing.assert_close(result[0], keys[order])
    torch.testing.assert_close(result[1], order.int())


def check_int128_register_keys(case):
    block, count = 64, 3
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = fx.Vector.from_elements(
            [((item * 17 % 31 - 15).to(fx.Int128) << 80) + (item % 7).to(fx.Int128) for item in value]
        )
        if case == "radix":
            op = fx.coop.BlockRadixSort[
                (fx.Int128, fx.Int32), block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT
            ]
        elif case == "rank":
            op = fx.coop.BlockRadixRank[fx.Int128, block, count]
        elif case == "merge":
            op = fx.coop.BlockMergeSort[(fx.Int128, fx.Int32), block, count]
        else:
            op = fx.coop.BlockTopK[(fx.Int128, fx.Int32), block, count, fx.coop.BlockTopKAlgorithm.AIR]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        if case == "rank":
            ranks = op(keys, storage=storage, begin_bit=79, end_bit=90)
            return ranks, ranks
        if case == "air":
            result, ids, valid = op(keys, 19, value, storage=storage)
        elif case == "radix":
            result, ids = op(keys, value, storage=storage, begin_bit=0, end_bit=128)
        else:
            result, ids = op(keys, value, storage=storage)
        return (result >> 80).to(fx.Int32), ids

    result = run_tile(transform, data, block, count, outputs=2)
    ids = data.cpu()
    major = ids * 17 % 31 - 15
    if case == "rank":
        digits = (major & 1023) * 2
        order = digits.argsort(stable=True)
        expected = torch.empty_like(order)
        expected[order] = torch.arange(len(order), device="cpu")
        torch.testing.assert_close(result[0], expected.int())
    else:
        order = (major * 8 + ids % 7).argsort(stable=True, descending=case == "air")
        k = 19 if case == "air" else len(ids)
        torch.testing.assert_close(result[1, :k].sort().values, order[:k].int().sort().values)
        torch.testing.assert_close(result[0, :k], major[result[1, :k].long()])


def check_boolean_radix_sort_rank(descending):
    block, count = 64, 3
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 13 % 3

    def transform(value):
        keys = value != 0
        allocator = fx.SharedAllocator()
        sort = fx.coop.BlockRadixSort[fx.Boolean, block, count, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT]
        sort_storage = allocator.allocate(sort.SharedStorage).peek()
        sorted_keys = sort(keys, storage=sort_storage, descending=descending)
        assert sorted_keys.dtype is fx.Boolean
        rank = fx.coop.BlockRadixRank[fx.Boolean, block, count]
        rank_storage = allocator.allocate(rank.SharedStorage).peek()
        ranks = rank(keys, storage=rank_storage, begin_bit=0, end_bit=1, descending=descending)
        topk = fx.coop.BlockTopK[fx.Boolean, block, count, fx.coop.BlockTopKAlgorithm.AIR]
        topk_storage = allocator.allocate(topk.SharedStorage).peek()
        selected, flags = topk(keys, 17, storage=topk_storage, largest=descending)
        assert selected.dtype is fx.Boolean
        return sorted_keys.to(fx.Int32), ranks, selected.to(fx.Int32)

    result = run_tile(transform, data, block, count, outputs=3)
    keys = (data.cpu() != 0).int()
    order = keys.argsort(descending=descending, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device="cpu")
    torch.testing.assert_close(result[0], keys[order])
    torch.testing.assert_close(result[1], ranks.int())

    torch.testing.assert_close(result[2, :17], keys[order[:17]])


def check_every_block_width(block, count, case, universal):
    n = block * count
    # Shape tests need cross-thread movement, not repeated radix passes. The
    # unsigned-digit regressions cover pass boundaries independently.
    bits = 8 if full_matrix_enabled() else 3
    modulus = 251 if full_matrix_enabled() or case == "BlockTopK" else 1 << bits
    values = (torch.arange(n, device="cuda", dtype=torch.int32) * 37) % modulus

    def transform(value):
        ns = _namespace(universal)
        op = getattr(ns, case)[fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        arg = value[0] if count == 1 else value
        if case == "BlockTopK":
            out, _ = op(arg, n // 2, storage=storage)
        else:
            out = op(arg, storage=storage, end_bit=bits)
        return out if isinstance(out, fx.Vector) else fx.Vector.from_elements([out])

    output = run_tile(transform, values, block, count)[0]
    if case == "BlockRadixRank":
        order = values.cpu().argsort(stable=True)
        expected = torch.empty(n, dtype=torch.int32, device="cpu")
        expected[order] = torch.arange(n, dtype=torch.int32, device="cpu")
    else:
        expected = values.cpu().sort(descending=case == "BlockTopK").values
    torch.testing.assert_close(output, expected)


def check_topk_dtypes(entry, count):
    dtype, name = entry
    block = 128 if full_matrix_enabled() or count == 1 else 2
    values = sample(name, block * count, seed=512)

    def transform(value):
        topk = fx.coop.BlockTopK[dtype, block, count]
        storage = fx.SharedAllocator().allocate(topk.SharedStorage).peek()
        keys, _ = topk(value, 17, storage=storage)
        return keys

    result = run_tile(transform, values, block, count, dtype=dtype)[0]
    torch.testing.assert_close(result[:17], values.cpu().sort(descending=True).values[:17], rtol=0, atol=0)


def check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    """Stable entry points preserve duplicate and signed-zero payload order."""
    ns = _namespace(universal)
    count, width = 3, 8
    # These checks target key/payload semantics. Thread scaling and the
    # compact multi-warp record path have separate regressions.
    block = 16 if case in ("bitonic", "warp_merge") else (64 if full_matrix_enabled() else 8)
    group = width * count if case in ("bitonic", "warp_merge") else block * count
    valid = group if case.startswith("radix") else group - 5
    k = 13
    data = torch.tensor(
        (_PATTERN * ((block * count + len(_PATTERN) - 1) // len(_PATTERN)))[: block * count],
        dtype=torch_dtype,
        device="cuda",
    )
    is_topk = case in ("topk_merge", "air")
    striped_output = case == "bitonic" or (case == "radix_digit" and descending)

    def transform(keys):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)]).to(fx.Int64)
        if case == "block_merge":
            op = ns.BlockMergeSort[(dtype, fx.Int64), block, count]
        elif case.startswith("radix"):
            algorithm = (
                ns.BlockRadixSortAlgorithm.RADIX_DIGIT
                if case == "radix_digit"
                else ns.BlockRadixSortAlgorithm.RADIX_SCAN
            )
            op = ns.BlockRadixSort[(dtype, fx.Int64), block, count, algorithm]
        else:
            algorithm = ns.BlockTopKAlgorithm.AIR if case == "air" else ns.BlockTopKAlgorithm.MERGE
            op = ns.BlockTopK[(dtype, fx.Int64), block, count, algorithm]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        if case == "block_merge":
            return op.stable_sort(keys, ids, storage=storage, descending=descending, valid_items=fx.Int32(valid))
        if case.startswith("radix"):
            return op(keys, ids, storage=storage, descending=descending, blocked_to_striped=striped_output)
        result, payload, selected = op(
            keys, fx.Int32(k), ids, storage=storage, largest=descending, num_valid=fx.Int32(valid)
        )
        return result, payload, selected.to(fx.Int32)

    result = run_tile(
        transform, data, block, count, outputs=3 if is_topk else 2, dtype=dtype, output_dtype=(fx.Float64, "float64")
    )
    group_width = width if case in ("bitonic", "warp_merge") else block
    source_ids = _logical(torch.arange(len(data), device="cpu"), group_width, count, case == "bitonic")
    logical_keys = data.cpu()[source_ids]
    order = logical_keys[:, :valid].argsort(dim=1, descending=descending, stable=True)
    expected_ids = source_ids[:, :valid].gather(1, order)
    got_keys = _logical(result[0], group_width, count, striped_output)
    got_ids = _logical(result[1].long(), group_width, count, striped_output)
    selected_count = k if is_topk else valid
    for group_id in range(len(source_ids)):
        ids = got_ids[group_id, :selected_count]
        keys = got_keys[group_id, :selected_count]
        expected = expected_ids[group_id, :selected_count]
        _assert_key_payload(keys, ids, data.cpu())
        if case == "air":
            torch.testing.assert_close(ids, expected.sort().values)
        else:
            torch.testing.assert_close(ids, expected)
    if is_topk:
        torch.testing.assert_close(result[2], (torch.arange(len(data), device="cpu") < k).double())


def check_nested_float_record_keys(case, projected, universal):
    """Nested float fields and untouched fields follow comparator/decomposer order."""
    ns = _namespace(universal)
    inner = fx.Struct["major" : fx.Float16, "minor" : fx.Float32]
    key_type = fx.Struct["parts":inner, "tag" : fx.Float64]
    info = fx.Struct["id" : fx.Int64, "weight" : fx.Float32]
    value_type = fx.Struct["info":info]
    width, count = 8, 3
    # These checks target key/payload semantics. Thread scaling and the
    # compact multi-warp record path have separate regressions.
    block = 16 if case in ("bitonic", "warp_merge") else (64 if full_matrix_enabled() else 8)
    group_width = width if case in ("bitonic", "warp_merge") else block
    n = block * count
    data = torch.tensor(([-3.5, 1.125, -0.0, 0.0, -3.5, 2.75, 1.125, -0.625] * ((n + 7) // 8))[:n], device="cuda")
    valid = group_width * count if case.startswith("radix") else group_width * count - 5
    k = 11

    def transform(value):
        ids = [fx.thread_idx.x * count + i for i in range(count)]
        keys = make_items(
            [
                key_type(
                    parts=inner(major=item.to(fx.Float16), minor=((idx % 5 - 2).to(fx.Float32) * 0.375)),
                    tag=item.to(fx.Float64) + idx.to(fx.Float64) * 0.125,
                )
                for item, idx in zip(value, ids)
            ]
        )
        payload = make_items(
            [value_type(info=info(id=idx.to(fx.Int64), weight=idx.to(fx.Float32) * -0.25)) for idx in ids]
        )
        if projected:
            compare = lambda a, b: a.parts.major < b.parts.major
            decompose = lambda key: (key.parts.major,)
        else:
            compare = lambda a, b: (
                (a.parts.major < b.parts.major) | ((a.parts.major == b.parts.major) & (a.parts.minor < b.parts.minor))
            )
            decompose = lambda key: (key.parts.major, key.parts.minor)
        if case == "block_merge":
            op = ns.BlockMergeSort[(key_type, value_type), block, count]
        elif case.startswith("radix"):
            algorithm = (
                ns.BlockRadixSortAlgorithm.RADIX_SCAN
                if case == "radix_scan"
                else ns.BlockRadixSortAlgorithm.RADIX_DIGIT
            )
            op = ns.BlockRadixSort[(key_type, value_type), block, count, algorithm]
        else:
            op = ns.BlockTopK[(key_type, value_type), block, count, ns.BlockTopKAlgorithm.MERGE]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        if case == "block_merge":
            ordered, carried = op.stable_sort(
                keys, payload, storage=storage, compare_op=compare, descending=True, valid_items=valid
            )
        elif case.startswith("radix"):
            ordered, carried = op(keys, payload, storage=storage, descending=True, decomposer=decompose)
        else:
            ordered, carried, _ = op(
                keys, k, payload, storage=storage, largest=True, compare_op=compare, num_valid=valid
            )
        return (
            fx.Vector.from_elements([key.parts.major for key in ordered]),
            fx.Vector.from_elements([key.parts.minor for key in ordered]),
            fx.Vector.from_elements([key.tag for key in ordered]),
            fx.Vector.from_elements([item.info.id for item in carried]),
            fx.Vector.from_elements([item.info.weight for item in carried]),
        )

    result = run_tile(transform, data, block, count, outputs=5, output_dtype=(fx.Float64, "float64"))
    source_ids = _logical(torch.arange(n, device="cpu"), group_width, count, case == "bitonic")
    host = data.cpu().half().double()
    minor = (torch.arange(n, device="cpu") % 5 - 2).double() * 0.375
    got = [_logical(row, group_width, count, case == "bitonic") for row in result]
    selected = k if case == "topk_merge" else valid
    for group_id, group_ids in enumerate(source_ids):
        expected_ids = torch.tensor(
            sorted(
                group_ids[:valid].tolist(),
                key=lambda i: (host[i].item(),) if projected else (host[i].item(), minor[i].item()),
                reverse=True,
            ),
            device="cpu",
        )[:selected]
        ids = got[3][group_id, :selected].long()
        _assert_key_payload(got[0][group_id, :selected], ids, host)
        torch.testing.assert_close(got[1][group_id, :selected], minor[ids], rtol=0, atol=0)
        torch.testing.assert_close(got[2][group_id, :selected], host[ids] + ids.double() * 0.125, rtol=0, atol=0)
        torch.testing.assert_close(got[4][group_id, :selected], ids.double() * -0.25, rtol=0, atol=0)
        torch.testing.assert_close(ids, expected_ids)


def check_float_decomposer_bit_range_crosses_fields(descending, universal):
    """Partial radix ranges address transformed bits across float-field boundaries."""
    ns = _namespace(universal)
    inner = fx.Struct["major" : fx.Float16, "minor" : fx.Float32]
    key_type = fx.Struct["parts":inner, "id" : fx.Int64]
    block, count = 64, 3
    data = torch.tensor(
        ([-5.5, -1.25, -0.0, 0.0, 0.625, 2.75, -5.5, 0.625] * ((block * count + 7) // 8))[: block * count],
        device="cuda",
    )

    def transform(value):
        ids = [fx.thread_idx.x * count + i for i in range(count)]
        keys = make_items(
            [
                key_type(
                    parts=inner(major=item.to(fx.Float16), minor=item * -0.5 + idx.to(fx.Float32) * 0.125),
                    id=idx.to(fx.Int64),
                )
                for item, idx in zip(value, ids)
            ]
        )
        op = ns.BlockRadixSort[key_type, block, count, ns.BlockRadixSortAlgorithm.RADIX_DIGIT]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        result = op(
            keys,
            storage=storage,
            descending=descending,
            begin_bit=28,
            end_bit=36,
            decomposer=lambda key: (key.parts.major, key.parts.minor),
        )
        return (
            fx.Vector.from_elements([key.id for key in result]),
            fx.Vector.from_elements([key.parts.major for key in result]),
            fx.Vector.from_elements([key.parts.minor for key in result]),
        )

    result = run_tile(transform, data, block, count, outputs=3, output_dtype=(fx.Float64, "float64"))
    ids = torch.arange(len(data), device="cpu")
    major = data.cpu()
    minor = major * -0.5 + ids.float() * 0.125
    raw = (_ordered_float_bits(major, torch.float16) << 32) | _ordered_float_bits(minor, torch.float32)
    digit = (raw >> 28) & 255
    expected = digit.argsort(descending=descending, stable=True)
    torch.testing.assert_close(result[0].long(), expected)
    _assert_key_payload(result[1], result[0].long(), major)
    torch.testing.assert_close(result[2], minor[expected].double(), rtol=0, atol=0)


def check_float_radix_exact_bits(dtype, unsigned, mantissa, algorithm, descending, universal):
    block, count = 64, 3
    width = dtype.width
    sign, mask = 1 << (width - 1), (1 << width) - 1
    fraction = (1 << mantissa) - 1
    infinity = (sign - 1) ^ fraction
    # Both signs of zero, infinities, minimum/maximum subnormals, minimum
    # normals, maximum finite values and signaling/quiet NaNs with payloads.
    positive = [
        0,
        infinity,
        1,
        fraction,
        1 << mantissa,
        infinity - 1,
        infinity | 1,
        infinity | (1 << (mantissa - 1)) | 5,
    ]
    pattern = [item for raw in positive for item in (raw, raw | sign)]
    raw = [pattern[(i * 7) % len(pattern)] for i in range(block * count)]
    signed = [item - (1 << 64) if item >= (1 << 63) else item for item in raw]
    data = torch.tensor(signed, dtype=torch.int64, device="cuda")
    ns = fx.coop.universal if universal else fx.coop

    def transform(value):
        keys = fx.Vector.from_elements([item.to(unsigned).bitcast(dtype) for item in value])
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)])
        sort = (
            ns.BlockTopK[(dtype, fx.Int32), block, count, ns.BlockTopKAlgorithm.AIR]
            if algorithm == "air"
            else ns.BlockRadixSort[(dtype, fx.Int32), block, count, algorithm]
        )
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        if algorithm == "air":
            keys, ids, _ = sort(keys, 13, ids, storage=storage, largest=descending, num_valid=block * count - 5)
        else:
            keys, ids = sort(keys, ids, storage=storage, descending=descending)
        return fx.Vector.from_elements([item.bitcast(unsigned).to(fx.Int64) for item in keys]), ids

    result = run_tile(transform, data, block, count, outputs=2)

    def ordered_bits(bits):
        if bits & (sign - 1) == 0:
            return sign
        return (~bits & mask) if bits & sign else bits ^ sign

    valid = len(raw) - 5 if algorithm == "air" else len(raw)
    expected_ids = torch.tensor(
        sorted(range(valid), key=lambda i: ordered_bits(raw[i]), reverse=descending), device="cpu"
    )
    if algorithm == "air":
        expected_ids = expected_ids[:13].sort().values
        result = result[:, :13]
    torch.testing.assert_close(result[1], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(result[0], data.cpu()[expected_ids], rtol=0, atol=0)


# Kernel runners and host references shared by radix-rank tests.

FLOATS = {"float16": fx.Float16, "bfloat16": fx.BFloat16, "float32": fx.Float32, "float64": fx.Float64}


INNER = fx.Struct["fraction" : fx.Float64]


KEY = fx.Struct["nested":INNER, "tag" : fx.Uint8]


def _keys(values, kind, form="vector"):
    if kind == "record":
        keys = [KEY(INNER(value.to(fx.Float64)), fx.Uint8(1)) for value in values]
        dtype = KEY
        extract = lambda key: fx.Int32((key.nested.fraction + 2.0) * 4.0) ^ fx.Int32(key.tag)
    else:
        dtype = FLOATS[kind]
        keys = [value.to(dtype) for value in values]
        extract = lambda key: fx.Int32((key + dtype(2.0)) * dtype(4.0))
    if form == "scalar":
        keys = keys[0]
    elif form == "tuple":
        keys = tuple(keys)
    elif form != "list":
        keys = make_items(keys)
    return keys, dtype, extract


def _expected(digits, algorithm, block, count):
    # BASIC/MEMOIZE enumerate thread-major items; MATCH enumerates each
    # logical warp's items lane-major, including a partially populated warp.
    mapping = torch.arange(block * count, device="cpu")
    if algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE):
        width = min(fx.num_warp_threads(), 1 << (block.bit_length() - 1))
        mapping = torch.cat(
            [
                mapping[start * count : min(start + width, block) * count].reshape(-1, count).T.flatten()
                for start in range(0, block, width)
            ]
        )
    return mapping, digits[mapping]


def _assert_ranks(actual, digits, algorithm, block, count, descending):
    mapping, logical = _expected(digits, algorithm, block, count)
    order = logical.argsort(stable=True, descending=descending)
    expected = torch.empty(block * count, dtype=torch.int32, device="cpu")
    expected[mapping[order]] = torch.arange(block * count, dtype=torch.int32, device="cpu")
    for ranks in actual:
        torch.testing.assert_close(ranks, expected, rtol=0, atol=0)


def _transform(kind, algorithm, block, count, descending, form="vector", namespaces=None):
    namespaces = (fx.coop, fx.coop.universal) if namespaces is None else namespaces

    def transform(values):
        keys, dtype, extractor = _keys(values, kind, form)
        result = []
        allocator = fx.SharedAllocator()
        for ns in namespaces:
            rank = ns.BlockRadixRank[dtype, block, count, algorithm]
            storage = allocator.allocate(rank.SharedStorage).peek()
            ranks = rank(keys, storage=storage, digit_extractor=extractor, radix_bits=4, descending=descending)
            assert isinstance(ranks, fx.Vector) == (form != "scalar")
            result.append(fx.Vector.from_elements(as_items(ranks)))
        return tuple(result)

    return transform


_RANK_RECORD = fx.Struct["key" : fx.Float32]


def check_radix_rank(begin, end, descending, universal):
    block, count = 64, 3
    values = torch.arange(block * count, device="cuda", dtype=torch.int32) * 71 - 37

    def transform(value):
        ns = fx.coop.universal if universal else fx.coop
        rank = ns.BlockRadixRank[fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        return rank(value, storage=storage, begin_bit=begin, end_bit=end, descending=descending)

    result = run_tile(transform, values, block, count)[0]
    digits = (values.cpu() >> begin) & ((1 << (end - begin)) - 1)
    order = digits.argsort(descending=descending, stable=True)
    expected = torch.empty_like(order)
    expected[order] = torch.arange(values.numel(), device="cpu")
    torch.testing.assert_close(result, expected.int())


def check_rank_algorithms_layout(algorithm, block):
    count = 3
    data = (torch.arange(block * count, device="cuda", dtype=torch.int32) * 17) % 31
    width = min(fx.num_warp_threads(), 1 << (block.bit_length() - 1))
    match = algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE)

    def transform(value):
        rank = fx.coop.BlockRadixRank[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        return rank(value, storage=storage, begin_bit=1, end_bit=4, descending=True)

    result = run_tile(transform, data, block, count)[0]
    physical = data.cpu()
    mapping = torch.arange(block * count, device="cpu")
    if match:
        parts, maps = [], []
        for start in range(0, block, width):
            active = min(width, block - start)
            part = physical[start * count : (start + active) * count]
            idx = mapping[start * count : (start + active) * count]
            parts.append(_striped_to_linear(part, active, count).flatten())
            maps.append(_striped_to_linear(idx, active, count).flatten())
        logical = torch.cat(parts)
        mapping = torch.cat(maps)
    else:
        logical = physical
    digits = (logical >> 1) & 7
    order = digits.argsort(descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device="cpu")
    expected = torch.empty_like(ranks)
    expected[mapping] = ranks
    torch.testing.assert_close(result, expected.int())


def check_rank_counts_callback(algorithm, descending):
    block, count = 64, 3
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 13 % 8

    def transform(value):
        rank = fx.coop.BlockRadixRank[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        seen = []
        ranks, prefix = rank(
            value,
            storage=storage,
            radix_bits=3,
            descending=descending,
            digit_extractor=lambda key: key,
            count_callback=lambda counts: seen.append(counts),
            with_prefix=True,
        )
        return ranks, fx.Vector.from_elements([seen[0][0]] * count), fx.Vector.from_elements([prefix[0]] * count)

    result = run_tile(transform, data, block, count, outputs=3)
    counts = torch.bincount(data.cpu().long(), minlength=8).int()
    torch.testing.assert_close(result[1].reshape(block, count)[:8, 0], counts)
    expected_prefix = counts.sum().int() - counts.cumsum(0).int() if descending else counts.cumsum(0).int() - counts
    torch.testing.assert_close(result[2].reshape(block, count)[:8, 0], expected_prefix)


def check_rank_nine_bit_configuration(algorithm):
    # This regression checks nine-bit capacity and digits above 255. A small
    # tile reaches those bins too; broad shape and item coverage lives in the
    # rank matrices. Exhaustive runs retain the original stress size.
    block, count = (64, 3) if full_matrix_enabled() else (8, 1)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 149 % 512
    data[0], data[-1] = 0, 511

    def transform(value):
        rank = fx.coop.BlockRadixRank.with_radix_bits(9)[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        return rank(value, storage=storage, radix_bits=9)

    result = run_tile(transform, data, block, count)[0]
    match = algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE)
    width = min(WARP_SIZE, block)
    logical = _striped_to_linear(data.cpu(), width, count).flatten() if match else data.cpu()
    order = logical.argsort(stable=True)
    expected = torch.empty_like(order)
    expected[order] = torch.arange(len(order), device="cpu")
    if match:
        expected = _linear_to_striped(expected, width, count)
    torch.testing.assert_close(result, expected.int())


def check_radix_rank_dtypes(entry, count):
    dtype, name = entry
    end_bit = min(7 if full_matrix_enabled() else 4, dtype.width)
    values = sample(name, 128 * count, seed=443)

    def transform(value):
        rank = fx.coop.BlockRadixRank[dtype, 128, count]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        return rank(value, storage=storage, begin_bit=1, end_bit=end_bit)

    result = run_tile(transform, values, 128, count, dtype=dtype, output_dtype=(fx.Int32, "int32"))[0]
    digits = (values.cpu().long() >> 1) & ((1 << (end_bit - 1)) - 1)
    order = digits.argsort(stable=True)
    expected = torch.empty(values.numel(), dtype=torch.int32, device="cpu")
    expected[order] = torch.arange(values.numel(), dtype=torch.int32, device="cpu")
    torch.testing.assert_close(result, expected)


def check_noninteger_rank(kind, algorithm, block, count, form, descending):
    digits = (torch.arange(block * count, dtype=torch.int32, device="cpu") * 7) % 16
    data = (digits.float() / 4.0 - 2.0).cuda()
    result = run_tile(
        _transform(kind, algorithm, block, count, descending, form),
        data,
        block,
        count,
        outputs=2,
        output_dtype=(fx.Int32, "int32"),
    )
    _assert_ranks(result, digits ^ 1 if kind == "record" else digits, algorithm, block, count, descending)


def check_record_rank_range_input(form, algorithm):
    block, count = 128, 3
    digits = torch.arange(block * count, dtype=torch.int32, device="cpu") * 11 % 16
    result = run_tile(
        _transform("record", algorithm, block, count, False, form),
        (digits.float() / 4 - 2).cuda(),
        block,
        count,
        outputs=2,
        output_dtype=(fx.Int32, "int32"),
    )
    _assert_ranks(result, digits ^ 1, algorithm, block, count, False)


def check_noninteger_counts_and_prefixes(kind, algorithm, descending):
    block, count, bins = 64, 3, 8
    digits = torch.arange(block * count, dtype=torch.int32, device="cpu") * 5 % bins
    data = (digits.float() / 4 - 2).cuda()

    def transform(values):
        keys, dtype, extractor = _keys(values, kind)
        outputs = []
        allocator = fx.SharedAllocator()
        for ns in (fx.coop, fx.coop.universal):
            rank = ns.BlockRadixRank[dtype, block, count, algorithm]
            storage = allocator.allocate(rank.SharedStorage).peek()
            seen = []
            ranks, prefix = rank(
                keys,
                storage=storage,
                radix_bits=3,
                digit_extractor=extractor,
                descending=descending,
                count_callback=(
                    (lambda counts: seen.append(counts))
                    if algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE)
                    else None
                ),
                with_prefix=True,
            )
            outputs.extend(
                (
                    ranks,
                    fx.Vector.filled(count, prefix[0], fx.Int32),
                    fx.Vector.filled(count, seen[0][0] if seen else -1, fx.Int32),
                )
            )
        return tuple(outputs)

    result = run_tile(transform, data, block, count, outputs=6, output_dtype=(fx.Int32, "int32"))
    digits = digits ^ 1 if kind == "record" else digits
    _assert_ranks(result[[0, 3]], digits, algorithm, block, count, descending)
    counts = torch.bincount(digits.long(), minlength=bins).int()
    prefix = counts.sum().int() - counts.cumsum(0).int() if descending else counts.cumsum(0).int() - counts
    for offset in (0, 3):
        torch.testing.assert_close(result[offset + 1].reshape(block, count)[:bins, 0], prefix)
        if algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE):
            torch.testing.assert_close(result[offset + 2].reshape(block, count)[:bins, 0], counts)
            assert result[offset + 2].reshape(block, count)[bins, 0].item() == 0


def check_extracted_digit_wider_than_key(algorithm):
    block, count = 64, 3
    data = (torch.arange(block * count, dtype=torch.int32, device="cpu") % 3 == 0).int().cuda()

    def transform(values):
        keys = values.to(fx.Boolean)
        result = []
        allocator = fx.SharedAllocator()
        for ns in (fx.coop, fx.coop.universal):
            rank = ns.BlockRadixRank[fx.Boolean, block, count, algorithm]
            storage = allocator.allocate(rank.SharedStorage).peek()
            result.append(
                rank(
                    keys,
                    storage=storage,
                    radix_bits=2,
                    digit_extractor=lambda key: key.select(fx.Int32(3), fx.Int32(0)),
                )
            )
        return tuple(result)

    result = run_tile(transform, data, block, count, outputs=2)
    _assert_ranks(result, data.cpu() * 3, algorithm, block, count, False)


def check_compile_noninteger_rank(monkeypatch, arch, kind, algorithm):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = 128, 3
    transform = _transform(kind, algorithm, block, count, True, namespaces=(fx.coop.universal,))

    def apply(a, out):
        values = fx.Vector.from_elements([a[fx.thread_idx.x * count + i] for i in range(count)])
        ranks = transform(values)[0]
        for i in range(count):
            out[fx.thread_idx.x * count + i] = ranks[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(
        torch.empty(block * count, dtype=torch.float32, device="cpu"),
        torch.empty(block * count, dtype=torch.int32, device="cpu"),
    )


def check_noninteger_rank_configuration(ns, record, case):
    dtype = _RANK_RECORD if record else fx.Float32
    key = _RANK_RECORD(fx.Float32(1.25)) if record else fx.Float32(1.25)
    op = ns.BlockRadixRank[dtype, 64, 1]
    if case == "missing_extractor":
        with pytest.raises(TypeError, match="require a digit_extractor"):
            op(key, storage=None)
    elif case == "missing_bits":
        with pytest.raises(ValueError, match="requires radix_bits"):
            op(key, storage=None, digit_extractor=lambda key: fx.Int32(0))
    else:
        with pytest.raises(ValueError, match="requires one digit pass"):
            op(key, storage=None, digit_extractor=lambda key: fx.Int32(0), radix_bits=4, end_bit=8)


def check_bad_extracted_digit(monkeypatch, case):
    monkeypatch.setenv("COMPILE_ONLY", "1")
    extract = {
        "float": lambda key: key,
        "record": lambda key: _RANK_RECORD(key),
        "negative": lambda key: -1,
        "overflow": lambda key: 16,
    }[case]

    def apply(a, out):
        op = fx.coop.universal.BlockRadixRank[fx.Float32, 64, 1]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        out[fx.thread_idx.x] = op(a[fx.thread_idx.x], storage=storage, radix_bits=4, digit_extractor=extract)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    error, message = (TypeError, "integer digit") if case in ("float", "record") else (ValueError, "in \\[0,")
    with pytest.raises(error, match=message):
        launch(torch.empty(64, dtype=torch.float32, device="cpu"), torch.empty(64, dtype=torch.int32, device="cpu"))


def check_vector_element_collective(block, algorithm, universal, custom, scan, operation=fx.ReductionOp.ADD):
    """An array of vector elements stays one task, with one vector aggregate."""
    ns = fx.coop.universal if universal else fx.coop
    host = torch.arange(1, block * 4 + 1, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [fx.Vector.from_elements([a[tid * 4 + 2 * i + j] for j in range(2)]) for i in range(2)]
        root = ns.BlockScan if scan else ns.BlockReduce
        primitive = root[fx.Vector[fx.Int32, 2], block, algorithm]
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()

        def add(left, right):
            return left + right

        add.commutative = True
        op = add if custom else operation
        if scan:
            seed = fx.Vector.from_elements([fx.Int32(7), fx.Int32(11)])
            inc, total = primitive.inclusive_with_aggregate(items, op, storage=storage, init=seed)
            fx.barrier()
            exc = primitive.exclusive(items, op, storage=storage, init=seed)
            assert isinstance(total, fx.Vector) and len(total) == 2
            for i in range(2):
                for j in range(2):
                    out[tid * 4 + 2 * i + j] = inc[i][j]
                    out[block * 4 + tid * 4 + 2 * i + j] = exc[i][j]
                    out[block * 8 + tid * 2 + j] = total[j]
        else:
            total = primitive(items, op, storage=storage)
            assert isinstance(total, fx.Vector) and len(total) == 2
            for j in range(2):
                out[tid * 2 + j] = total[j]

    result = run_kernel(apply, host.to(device="cuda"), block * (10 if scan else 2), block)
    vectors = host.reshape(-1, 2)
    if operation is fx.ReductionOp.MAX:
        total = vectors.max(0).values
        prefixes = vectors.cummax(0).values
    elif operation is fx.ReductionOp.MIN:
        total = vectors.min(0).values
        prefixes = vectors.cummin(0).values
    else:
        total = vectors.sum(0, dtype=torch.int32)
        prefixes = vectors.cumsum(0, dtype=torch.int32)
    total = total.repeat(block)
    if scan:
        seed = torch.tensor([7, 11], dtype=torch.int32, device="cpu")
        if operation is fx.ReductionOp.MAX:
            prefixes = torch.maximum(prefixes, seed)
        elif operation is fx.ReductionOp.MIN:
            prefixes = torch.minimum(prefixes, seed)
        else:
            prefixes = prefixes + seed
        exclusive = torch.cat((seed.reshape(1, 2), prefixes[:-1]))
        torch.testing.assert_close(result[: block * 4], prefixes.flatten())
        torch.testing.assert_close(result[block * 4 : block * 8], exclusive.flatten())
        torch.testing.assert_close(result[block * 8 :], total)
    else:
        torch.testing.assert_close(result, total)


def check_neighbor_operator_calls(method, valid, boundary):
    """Missing neighbors and invalid inputs must not execute the user callable."""
    block, count = WARP_SIZE, 2
    size = block * count
    host = torch.arange(size, device="cpu", dtype=torch.int32)
    adjacent = method.startswith("subtract")
    left = method in ("subtract_left", "flag_heads")
    limit = size if valid is None else valid

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [a[tid * count + i] for i in range(count)]
        out[size + tid] = fx.Int32(0)

        def op(lhs, rhs):
            out[size + tid] = out[size + tid] + 1
            return lhs - rhs if adjacent else lhs != rhs

        root = fx.coop.BlockAdjacentDifference if adjacent else fx.coop.BlockDiscontinuity
        primitive = root[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        kwargs = {"predecessor" if left else "successor": -7} if boundary else {}
        if adjacent:
            kwargs["valid_items"] = valid
        result = getattr(primitive, method)(items, op, storage=storage, **kwargs)
        for i in range(count):
            out[tid * count + i] = result[i]

    result = run_kernel(apply, host.to(device="cuda"), size + block, block)
    calls = limit if boundary else max(limit - 1, 0)
    assert result[size:].sum().item() == calls
    expected = host.clone() if adjacent else torch.ones(size, dtype=torch.int32, device="cpu")
    if adjacent and limit:
        neighbor = host.roll(1 if left else -1)
        neighbor[0 if left else limit - 1] = -7
        begin = 0 if boundary or not left else 1
        end = limit if boundary or left else limit - 1
        expected[begin:end] = host[begin:end] - neighbor[begin:end]
    torch.testing.assert_close(result[:size], expected)


def check_rank_unsigned_digits(algorithm, descending):
    """Later radix passes must retain the high bit of compressed digits."""
    block, count = 2 * WARP_SIZE, 2
    host = ((torch.arange(block * count, dtype=torch.int32, device="cpu") * 977) % 65536) | 0x8000
    match = algorithm not in (fx.coop.BlockRadixRankAlgorithm.BASIC, fx.coop.BlockRadixRankAlgorithm.MEMOIZE)

    def transform(value):
        rank = fx.coop.BlockRadixRank[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(rank.SharedStorage).peek()
        return rank(value, storage=storage, begin_bit=0, end_bit=16, descending=descending)

    result = run_tile(transform, host.to(device="cuda"), block, count)[0]
    mapping = torch.arange(block * count, device="cpu")
    if match:
        mapping = mapping.reshape(-1, WARP_SIZE, count).transpose(1, 2).flatten()
    logical = host[mapping]
    order = logical.argsort(descending=descending, stable=True)
    expected = torch.empty(block * count, dtype=torch.int32, device="cpu")
    expected[mapping[order]] = torch.arange(block * count, dtype=torch.int32, device="cpu")
    torch.testing.assert_close(result, expected)


def check_unsigned_shared_values():
    """Typed scratch loads preserve unsigned comparisons and widening."""
    from flydsl.extension.coop._values import _shared_load

    block = WARP_SIZE
    host = torch.tensor([127, 128, 255, 32767, 32768, 65535], dtype=torch.int64, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        allocator = fx.SharedAllocator()
        bytes_ = allocator.allocate(fx.Array[fx.Uint8, block]).peek()
        words = allocator.allocate(fx.Array[fx.Uint16, block]).peek()
        masks = allocator.allocate(fx.Array[fx.Uint64, block]).peek()
        bytes_[tid] = a[tid % 3].to(fx.Uint8)
        words[tid] = a[3 + tid % 3].to(fx.Uint16)
        masks[tid] = fx.Uint64(1) << 63
        fx.barrier()
        index = (tid + 1) % block
        byte = _shared_load(bytes_, index)
        word = _shared_load(words, index)
        mask = _shared_load(masks, index)
        out[tid * 4] = byte.to(fx.Int64)
        out[tid * 4 + 1] = word.to(fx.Int64)
        out[tid * 4 + 2] = fx.Int64(word > fx.Uint16(32767))
        out[tid * 4 + 3] = (mask >> 63).to(fx.Int64)

    result = run_kernel(apply, host.to(device="cuda"), block * 4, block).reshape(block, 4)
    index = ((torch.arange(block, device="cpu") + 1) % block) % 3
    expected = torch.stack(
        (
            host[index],
            host[3 + index],
            (host[3 + index] > 32767).long(),
            torch.ones(block, dtype=torch.int64, device="cpu"),
        ),
        dim=1,
    )
    torch.testing.assert_close(result, expected)


def check_secondary_item_containers(form, case, algorithm=None):
    block, count, bins = WARP_SIZE // 2, 2, 7
    size = block * count
    host = torch.arange(size, dtype=torch.int32, device="cpu")

    def container(items):
        if form == "list":
            return list(items)
        if form == "tuple":
            return tuple(items)
        return fx.Vector.from_elements(items)

    def apply(a, out):
        tid = fx.thread_idx.x
        items = container([a[tid * count + i] for i in range(count)])
        if case == "histogram":
            primitive = fx.coop.BlockHistogram[fx.Int32, block, bins, algorithm]
            storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
            keys = container([(item % (bins + 2)) - 1 for item in as_items(items)])
            primitive(keys, storage=storage)
            counts = primitive.composite(keys, storage=storage)
            out[tid] = counts[tid % bins]
        elif case.startswith("decode"):
            primitive = fx.coop.BlockRunLengthDecode[fx.Int32, block, count]
            storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
            if case == "decode_lengths":
                lengths = container([fx.Int32(2), fx.Int32(2)])
                state = primitive.init_with_run_lengths(items, lengths, storage=storage)
            else:
                starts = container([2 * (tid * count + i) for i in range(count)])
                state = primitive.init_with_run_offsets(items, starts, storage=storage, total=size * 2)
            decoded, relative, total = state.decode(decoded_items_per_thread=count, offset=3, default=-1)
            for i in range(count):
                out[tid * count + i] = decoded[i]
                out[size + tid * count + i] = relative[i]
            out[2 * size + tid] = total
        else:
            primitive = fx.coop.BlockExchange[fx.Int32, block, count, algorithm]
            storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
            ranks = container([size - 1 - (tid * count + i) for i in range(count)])
            if case == "scatter_flagged":
                flags = container([fx.Int32(i == 0) for i in range(count)])
                result = primitive.scatter_to_striped_flagged(items, ranks, flags, storage=storage, default=-1)
            elif case == "scatter_guarded":
                ranks = container([ranks[0], fx.Int32(-1)])
                result = primitive.scatter_to_striped_guarded(items, ranks, storage=storage, default=-1)
            else:
                result = primitive.scatter_to_blocked(items, ranks, storage=storage)
            for i in range(count):
                out[tid * count + i] = result[i]

    out_size = block if case == "histogram" else 2 * size + block if case.startswith("decode") else size
    result = run_kernel(apply, host.to(device="cuda"), out_size, block)
    if case == "histogram":
        result = result[:bins]
        keys = host % (bins + 2) - 1
        expected = keys[(keys >= 0) & (keys < bins)].long().bincount(minlength=bins).int() * 2
    elif case.startswith("decode"):
        expected = torch.cat(
            (
                host.repeat_interleave(2)[3 : 3 + size],
                torch.arange(3, 3 + size, dtype=torch.int32, device="cpu") % 2,
                torch.full((block,), size * 2, dtype=torch.int32, device="cpu"),
            )
        )
    else:
        expected = host.flip(0)
        if case in ("scatter_flagged", "scatter_guarded"):
            expected = torch.where(expected % 2 == 0, expected, -1)
            expected = expected.reshape(count, block).T.flatten()
    torch.testing.assert_close(result, expected)
