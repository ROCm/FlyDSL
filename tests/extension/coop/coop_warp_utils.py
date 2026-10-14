# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp collective checks, kernels and host references; independent of block test helpers."""

import torch
from coop_common import WARP_SIZE, sample
from coop_test_utils import (
    as_items,
    batched_columns,
    items_dtype,
    make_items,
    run_kernel,
    run_tile,
    warp_indices,
    warp_storage,
    warp_valid_items,
)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

# Shared kernels and host references for primitive matrices and compilation.


def check_warp_dtypes_and_widths(entry, width, count, primitive):
    dtype, name = entry
    values = sample(name, 128 * count, seed=811)
    group = width or WARP_SIZE

    def transform(value):
        results = []
        allocator = fx.SharedAllocator()
        for ns in (fx.coop, fx.coop.universal):
            arg = value if primitive.endswith("sort") or count != 1 else value[0]
            if primitive.endswith("sort"):
                out = getattr(ns, primitive)(arg, width=width, compare_op=lambda a, b: a < b)
            else:
                P = ns.WarpExchange[dtype, width, count]
                storage = warp_storage(P, allocator)
                out = getattr(P, primitive.removeprefix("warp_"))(arg, storage=storage)
                fx.barrier()
            results.append(out if isinstance(out, fx.Vector) else fx.Vector.from_elements([out]))
        return tuple(results)

    output = run_tile(transform, values, 128, count, outputs=2, dtype=dtype)
    host = values.cpu()
    if primitive == "warp_blocked_to_striped":
        expected = host.reshape(-1, count, group).transpose(1, 2).reshape(-1)
    elif primitive == "warp_striped_to_blocked":
        expected = host.reshape(-1, group, count).transpose(1, 2).reshape(-1)
    else:
        expected = host.reshape(-1, group * count).sort(dim=1).values.reshape(-1)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, group).transpose(1, 2).reshape(-1)
    for actual in output:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=lambda msg: f"{msg} actual={actual.tolist()} expected={expected.tolist()}",
        )


def check_batched_reduction_dtypes_and_operations(entry, width, op):
    dtype, name = entry
    count, block = (9, 128)
    values = sample(name, block * count, seed=867)
    if values.dtype.is_floating_point and op is fx.ReductionOp.MUL:
        values = values * 0.01 + 1

    def transform(value):
        return (
            batched_columns(fx.coop, value, op, width=width),
            batched_columns(fx.coop.universal, value, op, width=width),
        )

    result = run_tile(transform, values, block, count, outputs=2, dtype=dtype)
    group = width or WARP_SIZE
    host = values.cpu().reshape(-1, group, count)
    if op is fx.ReductionOp.ADD:
        expected = host.sum(dim=1, keepdim=True)
    elif op is fx.ReductionOp.MUL:
        expected = host.prod(dim=1, keepdim=True)
    elif op is fx.ReductionOp.MIN:
        expected = host.amin(dim=1, keepdim=True)
    else:
        expected = host.amax(dim=1, keepdim=True)
    expected = expected.expand(-1, group, -1).reshape(-1).to(values.dtype)
    for actual in result:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-05 if values.dtype.is_floating_point else 0,
            atol=1e-05 if values.dtype.is_floating_point else 0,
        )


def check_compile_family(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    block, count = (64, 4)

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if case == "warp_load":
            result = fx.coop.WarpLoad[fx.Int32, 8, count].load(a, offset=tid // 8 * 8 * count, valid_items=31)
        elif case == "warp_store":
            fx.coop.WarpStore[fx.Int32, 8, count].store(out, value, offset=tid // 8 * 8 * count, valid_items=31)
            return
        elif case == "warp_reduce_batched":
            result = batched_columns(fx.coop, value, fx.ReductionOp.ADD, width=8)
        else:
            if case.endswith("sort"):
                result = getattr(fx.coop, case)(value, width=8, compare_op=lambda a, b: a < b)
            else:
                P = fx.coop.WarpExchange[fx.Int32, 8, count]
                result = getattr(P, case.removeprefix("warp_"))(value, storage=warp_storage(P))
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
        group = width or fx.num_warp_threads()
        warp_offset = offset + fx.thread_idx.x // group * group * count
        items = ns.WarpLoad[
            a.dtype,
            width,
            count,
            fx.coop.WarpLoadAlgorithm.STRIPED if policy == "warp_striped" else fx.coop.WarpLoadAlgorithm.DIRECT,
        ].load(
            a,
            offset=warp_offset,
            valid_items=warp_valid_items(limit, width, count),
            default=-7 if valid is not None else None,
        )
        ns.WarpStore[
            items_dtype(items),
            width,
            len(as_items(items)),
            fx.coop.WarpStoreAlgorithm.STRIPED if policy == "warp_striped" else fx.coop.WarpStoreAlgorithm.DIRECT,
        ].store(out, items, offset=warp_offset, valid_items=warp_valid_items(limit, width, count))
        for i in range(count):
            loaded[fx.thread_idx.x * count + i] = items[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor):
        apply(a, out, loaded)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, loaded: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out, loaded).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

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
        width = width or fx.num_warp_threads() if policy == "warp_striped" else block
        expected_items = expected_items.reshape(-1, count, width).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(loaded.cpu(), expected_items)


def check_movement_io_policies(warp, policy, count, valid):
    block, width, offset = (32, 8, 5)
    size = block * count
    source = torch.arange(offset + valid, dtype=torch.int32, device="cuda")

    def apply(a, out):
        warp_offset = offset + fx.thread_idx.x // width * width * count
        items = fx.coop.WarpLoad[a.dtype, width, count, getattr(fx.coop.WarpLoadAlgorithm, policy)].load(
            a, offset=warp_offset, valid_items=warp_valid_items(fx.Int32(valid), width, count), default=-7
        )
        fx.coop.WarpStore[
            items_dtype(items), width, len(as_items(items)), getattr(fx.coop.WarpStoreAlgorithm, policy)
        ].store(out, items, offset=warp_offset, valid_items=warp_valid_items(fx.Int32(valid), width, count))
        for i in range(count):
            out[offset + size + 4 + fx.thread_idx.x * count + i] = items[i]

    actual = run_kernel(apply, source, offset + size * 2 + 4, block)
    expected = torch.full_like(actual, -99)
    expected[offset : offset + valid] = source.cpu()[offset:]
    loaded = torch.full((size,), -7, dtype=torch.int32, device="cpu")
    loaded[:valid] = source.cpu()[offset:]
    if policy == "STRIPED":
        loaded = loaded[warp_indices(block, count, width)]
    expected[offset + size + 4 :] = loaded
    torch.testing.assert_close(actual, expected)


def check_compile_movement_policies(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = (64, 4)

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if case.startswith("warp_smem"):
            P = fx.coop.WarpExchange[fx.Int32, 8, count, fx.coop.WarpExchangeAlgorithm.SHARED]
            kwargs = dict(storage=warp_storage(P))
            if case.endswith("blocked"):
                result = P.blocked_to_striped(value, **kwargs)
            elif case.endswith("striped"):
                result = P.striped_to_blocked(value, **kwargs)
            else:
                ranks = fx.Vector.from_elements([31 - (tid % 8 * count + i) for i in range(count)])
                result = P.scatter_to_striped(value, ranks, **kwargs)
        elif case.startswith("warp_io"):
            policy = case.removeprefix("warp_io_")
            warp_offset = tid // 8 * 8 * count
            result = fx.coop.WarpLoad[a.dtype, 8, count, fx.coop.WarpLoadAlgorithm[policy]].load(
                a, offset=warp_offset, valid_items=31, default=-7
            )
            fx.coop.WarpStore[items_dtype(result), 8, len(as_items(result)), fx.coop.WarpStoreAlgorithm[policy]].store(
                out, result, offset=warp_offset, valid_items=31
            )
            return
        else:
            raise ValueError(f"unsupported warp compile case: {case}")
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
    block, offset, width = (32, 3, 8)
    size = block * count
    valid = 1 if partial == "one" else size - 1
    loaded_dtype = target_dtype if convert_load else source_dtype
    loaded_name = target_name if convert_load else source_name

    def apply(a, out, loaded):
        warp_offset = offset + fx.thread_idx.x // width * width * count
        items = fx.coop.WarpLoad[loaded_dtype, width, count, fx.coop.WarpLoadAlgorithm[policy]].load(
            a, offset=warp_offset, valid_items=warp_valid_items(fx.Int32(valid), width, count), default=-7
        )
        fx.coop.WarpStore[items_dtype(items), width, len(as_items(items)), fx.coop.WarpStoreAlgorithm[policy]].store(
            out, items, offset=warp_offset, valid_items=warp_valid_items(fx.Int32(valid), width, count)
        )
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
        expected_items = expected_items[warp_indices(block, count, width)]
    torch.testing.assert_close(loaded.cpu(), expected_items)


# Shared record kernels and host references for exchange, load, and store checks.

PAYLOAD = fx.Struct["fraction" : fx.Float32, "tag" : fx.Int64]
RECORD = fx.Struct["key" : fx.Int32, "payload":PAYLOAD]


def _make_record_kernel(case, policy, block, count):
    size = block * count

    def apply(a, b, c, x, y, z):
        allocator = fx.SharedAllocator()
        source = allocator.allocate(fx.Array[RECORD, size]).peek()
        destination = allocator.allocate(fx.Array[RECORD, size * 2]).peek()
        tid = fx.thread_idx.x
        for i in range(count):
            index = tid * count + i
            source[index] = RECORD(a[index], PAYLOAD(b[index], c[index]))
            destination[index] = RECORD(fx.Int32(-99), PAYLOAD(fx.Float32(-99), fx.Int64(-99)))
            destination[size + index] = RECORD(fx.Int32(-99), PAYLOAD(fx.Float32(-99), fx.Int64(-99)))
        fx.barrier()
        if case == "warp_io":
            warp_offset = fx.thread_idx.x // 8 * 8 * count
            items = fx.coop.WarpLoad[source.dtype, 8, count, fx.coop.WarpLoadAlgorithm[policy]].load(
                source, offset=warp_offset, valid_items=8 * count - 3, default=-7
            )
            fx.coop.WarpStore[items_dtype(items), 8, len(as_items(items)), fx.coop.WarpStoreAlgorithm[policy]].store(
                destination, items, offset=warp_offset, valid_items=8 * count - 3
            )
        else:
            items = make_items([source[fx.thread_idx.x * count + i] for i in range(count)])
            P = fx.coop.WarpExchange[RECORD, 8, count, fx.coop.WarpExchangeAlgorithm[policy]]
            kwargs = dict(storage=warp_storage(P, allocator))
            if case == "warp_blocked_to_striped":
                items = P.blocked_to_striped(items, **kwargs)
            elif case == "warp_striped_to_blocked":
                items = P.striped_to_blocked(items, **kwargs)
            else:
                ranks = fx.Vector.from_elements(
                    [8 * count - 1 - (fx.thread_idx.x % 8 * count + i) for i in range(count)]
                )
                items = P.scatter_to_striped(items, ranks, **kwargs)
        for i, item in enumerate(items):
            destination[size + fx.thread_idx.x * count + i] = item
        fx.barrier()
        for i in range(count * 2):
            index = tid * count * 2 + i
            item = destination[index]
            x[index] = item.key
            y[index] = item.payload.fraction
            z[index] = item.payload.tag

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
    case.startswith("warp_") and policy not in ("SHARED_MEMORY", "WARP_TIME_SLICING")
    block = 64
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
            mask = torch.arange(size, device="cpu") % (8 * count) < 8 * count - 3
            expected[:size] = torch.where(mask, host, -99)
            items = torch.where(mask, host, -7)
            if policy == "STRIPED":
                items = items[warp_indices(block, count, 8)]
        elif "scatter" in case:
            items = host.reshape(-1, 8 * count).flip(1).reshape(-1)[warp_indices(block, count, 8)]
        else:
            width = 8
            inverse = case == "warp_striped_to_blocked"
            items = host[warp_indices(block, count, width, inverse=inverse)]
        expected[size:] = items
        torch.testing.assert_close(output.cpu(), expected)


def check_compile_record_movement(monkeypatch, arch, case, policy):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    case.startswith("warp_") and policy not in ("SHARED_MEMORY", "WARP_TIME_SLICING")
    block, count = 64, 8 if policy == "SHUFFLE" else 3
    fields = [
        torch.empty(block * count, dtype=dtype, device="cpu") for dtype in (torch.int32, torch.float32, torch.int64)
    ]
    outputs = [torch.empty(block * count * 2, dtype=field.dtype, device="cpu") for field in fields]
    _make_record_kernel(case, policy, block, count)(*fields, *outputs)


# Shared kernels and reference checks for scalar reductions and scans.


def affine(lhs, rhs):
    """Composition over F_251, packed as (a << 16) | b; associative, noncommutative."""
    a, b = (lhs >> 16 & 65535, lhs & 65535)
    c, d = (rhs >> 16 & 65535, rhs & 65535)
    return a * c % 251 << 16 | (a * d + b) % 251


affine.identity = 1 << 16


def host_fold(items, op=affine, init=1 << 16):
    result = init
    for item in items:
        result = op(result, int(item))
    return result


def inputs(count):
    ids = torch.arange(count, dtype=torch.int32, device="cpu")
    return ((ids * 13 + 3) % 251 << 16 | (ids * 7 + 11) % 251).cuda()


def run_reduction(kernel, count, *extra):

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out, Agg).launch(grid=(1, 1, 1), block=(count, 1, 1), stream=stream)

    values = extra[0] if extra else inputs(count)
    out = torch.zeros_like(values)
    agg = torch.zeros(count, dtype=torch.int32, device="cuda")
    launch(values, out, agg, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return (values.cpu(), out.cpu(), agg.cpu())


def check_segmented_ordered_reduction(width, head):
    block = 128

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        lane = tid % width
        flag = lane % 5 == (0 if head else 4)
        form = fx.coop.warp_head_segmented_reduce if head else fx.coop.warp_tail_segmented_reduce
        Out[tid] = form(A[tid], flag, affine, width=width)

    values, out, _ = run_reduction(kernel, block)
    for base in range(0, block, width):
        for first in range(0, width, 5):
            assert out[base + first] == host_fold(values[base + first : base + min(first + 5, width)])


def check_batched_result_ownership(batches, layout):
    block, width = (64, 8)
    count = (batches + width - 1) // width

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        vector = tuple((A[tid * batches + i] for i in range(batches)))
        result = getattr(fx.coop, "warp_reduce_batched_to_" + layout)(vector, affine, width=width)
        for i in fx.range_constexpr(count):
            Out[tid * count + i] = result[i]

    values = inputs(max(1, block * batches))
    host, out, _ = run_reduction(kernel, block, values)
    for tid in range(block):
        base, lane = (tid // width * width, tid % width)
        for slot in range(count):
            batch = lane * count + slot if layout == "blocked" else lane + slot * width
            if batch < batches:
                expected = host_fold([host[(base + j) * batches + batch] for j in range(width)])
                assert out[tid * count + slot] == expected


def plain_affine(lhs, rhs):
    """The same semigroup operation, deliberately without identity metadata."""
    return affine(lhs, rhs)


def check_warp_valid_counts_seed_broadcast(universal, valid):
    block, width = (64, 8)
    initial = 7 << 16 | 19

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        namespace = fx.coop.universal if universal else fx.coop
        tid = fx.thread_idx.x
        inclusive, exclusive, aggregate = namespace.warp_scan_with_aggregate(
            A[tid], affine, width=width, valid_items=fx.Int32(valid), init=fx.Int32(initial)
        )
        Out[tid] = exclusive
        Agg[tid] = namespace.warp_reduce(A[tid], affine, width=width, valid_items=fx.Int32(valid))

    values, out, agg = run_reduction(kernel, block)
    for base in range(0, block, width):
        total = host_fold(values[base : base + valid])
        for lane in range(width):
            if lane < valid:
                expected = host_fold(values[base : base + lane], init=initial)
                assert out[base + lane] == expected
            if valid and lane == 0:
                assert agg[base + lane] == total


def check_batched_scalar_and_warp_broadcast(universal):
    block, width, batches = (64, 8, 5)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        namespace = fx.coop.universal if universal else fx.coop
        tid = fx.thread_idx.x
        vector = fx.Vector.from_elements([A[tid * batches + i] for i in range(batches)])
        Out[tid] = namespace.warp_reduce_batched(vector, affine, width=width)
        Agg[tid] = namespace.warp_broadcast(A[tid * batches], width - 1, width=width)

    host, out, agg = run_reduction(kernel, block, inputs(block * batches))
    for tid in range(block):
        base, lane = (tid // width * width, tid % width)
        if lane < batches:
            assert out[tid] == host_fold([host[(base + i) * batches + lane] for i in range(width)])
        assert agg[tid] == host[(base + width - 1) * batches]


def check_warp_partial_scan_semigroup_without_identity(valid):
    block, width = (64, 8)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        inclusive, exclusive, aggregate = fx.coop.warp_scan_with_aggregate(
            A[tid], plain_affine, width=width, valid_items=fx.Int32(valid)
        )
        Out[tid], Agg[tid] = (exclusive, aggregate)

    host, out, agg = run_reduction(kernel, block)
    for base in range(0, block, width):
        for lane in range(1, valid):
            assert out[base + lane] == host_fold(host[base : base + lane])
        assert torch.equal(
            agg[base : base + width],
            torch.full((width,), host_fold(host[base : base + valid]), dtype=agg.dtype, device="cpu"),
        )


def check_batched_wspro_synchronization_scope(sync_physical_warp, layout):
    block, width, batches = (128, 8, 19)
    count = (batches + width - 1) // width

    def apply(A, Out, tid):
        vector = fx.Vector.from_elements([A[tid * batches + i] for i in range(batches)])
        result = getattr(fx.coop, "warp_reduce_batched_to_" + layout)(
            vector, affine, width=width, sync_physical_warp=sync_physical_warp
        )
        for i in range(count):
            Out[tid * count + i] = result[i]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        if fx.const_expr(sync_physical_warp):
            apply(A, Out, tid)
        elif tid // width % 2 == 0:
            apply(A, Out, tid)

    host, out, _ = run_reduction(kernel, block, inputs(block * batches))
    for tid in range(block):
        base, lane = (tid // width * width, tid % width)
        if not sync_physical_warp and tid // width % 2 != 0:
            continue
        for slot in range(count):
            batch = lane * count + slot if layout == "blocked" else lane + slot * width
            if batch < batches:
                assert out[tid * count + slot] == host_fold([host[(base + j) * batches + batch] for j in range(width)])


def check_warp_reduce_array_blocked_order(universal):

    block, width, items = (64, 8, 3)
    operation = fx.coop.universal.warp_reduce if universal else fx.coop.warp_reduce

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        values = fx.Vector.from_elements([A[tid * items + i] for i in range(items)])
        Agg[tid] = operation(values, plain_affine, width=width)

    host, _, aggregate = run_reduction(kernel, block, inputs(block * items))
    for tid in range(block):
        base = tid // width * width * items
        assert aggregate[tid // width * width] == host_fold(host[base : base + width * items])


def check_reduce_scan_policies_compile(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    block = 128 if case == "warp_batched_wspro" else 128

    def apply(a, out):
        tid = fx.thread_idx.x
        value = a[tid]
        if case == "warp_batched_wspro":
            values = fx.Vector.from_elements([value + i for i in range(5)])
            blocked = fx.coop.warp_reduce_batched_to_blocked(values, affine, width=4, sync_physical_warp=True)
            striped = fx.coop.warp_reduce_batched_to_striped(values, affine, width=4, sync_physical_warp=False)
            result = blocked[0] ^ striped[1]
        else:
            items = fx.Vector.from_elements([value, value + 1, value + 2])
            reduced = fx.coop.warp_reduce(items, affine, width=4)
            scanned = fx.coop.warp_exclusive_scan(value, affine, width=4, init=fx.Int32(65539))
            segmented = fx.coop.warp_head_segmented_reduce(value, tid % 3 == 0, affine, width=4)
            result = reduced ^ scanned ^ segmented
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
    block = 64

    def apply(Out):
        tid = fx.thread_idx.x
        value = tid % 7 != 0
        reduction = fx.coop.warp_reduce(value, op, width=8, valid_items=valid)
        scan = fx.coop.warp_exclusive_scan(
            value, op, width=8, init=fx.Boolean(op is fx.ReductionOp.MIN), valid_items=valid
        )
        Out[tid * 2], Out[tid * 2 + 1] = (fx.Int64(reduction), fx.Int64(scan))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(Out: fx.Tensor):
        apply(Out)

    out = run_numeric(kernel, block, 2)
    fold = all if op is fx.ReductionOp.MIN else any
    for tid in range(block):
        base = tid // 8 * 8
        total = int(fold((i % 7 != 0 for i in range(base, base + valid))))
        prefix = int(fold((i % 7 != 0 for i in range(base, base + min(tid - base, valid)))))
        if valid and tid == base:
            assert out[tid, 0] == total
        if tid - base < valid:
            assert out[tid, 1] == prefix


def pair(value):
    raw = value & (1 << 128) - 1
    words = [raw & (1 << 64) - 1, raw >> 64]
    return [word if word < 1 << 63 else word - (1 << 64) for word in words]


def wide_value(dtype, tid):
    positive = (dtype(tid + 1) << 80) + (dtype(tid % 7) << 60) + dtype(tid)
    return (tid % 3 == 0).select(-positive, positive)


def store_wide(out, offset, value):
    bits = value.bitcast(fx.Uint128)
    out[offset] = bits.to(fx.Int64)
    out[offset + 1] = (bits >> 64).to(fx.Int64)


def check_warp_128_bit_reduce_scan(dtype, op):
    block, width, valid = (64, 8, 5)
    lowest = -(1 << 127) if dtype is fx.Int128 else 0
    highest = (1 << 127) - 1 if dtype is fx.Int128 else (1 << 128) - 1
    neutral = 0 if op is fx.ReductionOp.ADD else highest if op is fx.ReductionOp.MIN else lowest

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        value = wide_value(dtype, tid)
        reduction = fx.coop.warp_reduce(value, op, width=width, valid_items=valid)
        scan = fx.coop.warp_exclusive_scan(value, op, width=width, init=dtype(neutral), valid_items=valid)
        store_wide(Out, tid * 4, reduction)
        store_wide(Out, tid * 4 + 2, scan)

    out = run_numeric(kernel, block, 4)
    host = []
    for i in range(block):
        value = (i + 1 << 80) + (i % 7 << 60) + i
        value = -value if i % 3 == 0 else value
        host.append(value if dtype is fx.Int128 else value % (1 << 128))
    operation = sum if op is fx.ReductionOp.ADD else min if op is fx.ReductionOp.MIN else max
    for tid in range(block):
        base, lane = (tid // width * width, tid % width)
        total = operation(host[base : base + valid])
        values = host[base : base + min(lane, valid)]
        prefix = operation(values) if values else neutral
        assert out[base, :2].tolist() == pair(total)
        if lane < valid:
            assert out[tid, 2:].tolist() == pair(prefix)


def check_batched_reduce(width, universal):
    values = torch.arange(128 * 3, device="cuda", dtype=torch.int32) % 17

    def transform(value):
        ns = fx.coop.universal if universal else fx.coop
        return batched_columns(ns, value, fx.ReductionOp.ADD, width=width)

    result = run_tile(transform, values, 128, 3)[0]
    group = width or fx.num_warp_threads()
    expected = values.cpu().reshape(-1, group, 3).sum(1, keepdim=True).expand(-1, group, -1).reshape(-1).int()
    torch.testing.assert_close(result, expected)


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
            a=lhs.transform.a * rhs.transform.a % 251, b=(lhs.transform.a * rhs.transform.b + lhs.transform.b) % 251
        ),
        count=lhs.count + rhs.count,
    )


def compose_identity(dtype):
    return dtype(transform=Affine(a=fx.Int32(1), b=fx.Int32(0)), count=fx.Int64(0))


compose.identity = compose_identity


def make_item(value, index):
    return Item(transform=Affine(a=value >> 16 & 65535, b=value & 65535), count=(fx.Int64(index) + 1) * (1 << 33))


def store_item(out, index, value):
    out[index * 3] = fx.Int64(value.transform.a)
    out[index * 3 + 1] = fx.Int64(value.transform.b)
    out[index * 3 + 2] = value.count


def data(count):
    index = torch.arange(count, dtype=torch.int32, device="cpu")
    return (index * 13 + 3) % 251 << 16 | (index * 7 + 11) % 251


def reference(host, indices, additive=False):
    a, b, total = (0 if additive else 1, 0, 0)
    for index in indices:
        c, d = (int(host[index]) >> 16 & 65535, int(host[index]) & 65535)
        a, b = ((a + c) % 251, (b + d) % 251) if additive else (a * c % 251, (a * d + b) % 251)
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
    return (out.cpu().reshape(-1, 3), agg.cpu().reshape(-1, 3))


def check_nested_record_warp_collectives(case, universal):
    block, width = (64, 8)
    batches = 11 if case == "batched" else 1
    outputs = 2 if case == "batched" else 1

    def apply(A, Out, Agg):
        namespace = fx.coop.universal if universal else fx.coop
        tid = fx.thread_idx.x
        value = make_item(A[tid], tid)
        if case == "reduce":
            result = namespace.warp_reduce(value, compose, width=width, valid_items=5)
        elif case == "scan":
            result = namespace.warp_exclusive_scan(
                value, compose, width=width, init=compose_identity(Item), valid_items=5
            )
        elif case == "segmented":
            result = namespace.warp_head_segmented_reduce(value, tid % width % 3 == 0, compose, width=width)
        else:
            values = tuple((make_item(A[tid * batches + i], tid * batches + i) for i in range(batches)))
            result = namespace.warp_reduce_batched_to_blocked(values, compose, width=width)
        for i, item in enumerate(as_items(result)):
            store_item(Out, tid * outputs + i, item)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        apply(A, Out, Agg)

    host = data(block * batches)
    out, _ = launch_case(kernel, block, host, block * outputs, block)
    for tid in range(block):
        base, lane = (tid // width * width, tid % width)
        if case == "reduce" and lane == 0:
            assert out[tid].tolist() == reference(host, range(base, base + 5))
        elif case == "scan" and lane < 5:
            assert out[tid].tolist() == reference(host, range(base, base + min(lane, 5)))
        elif case == "segmented" and lane % 3 == 0:
            assert out[tid].tolist() == reference(host, range(tid, base + min(lane + 3, width)))
        elif case == "batched":
            for slot in range(outputs):
                batch = lane * outputs + slot
                if batch < batches:
                    assert out[tid * outputs + slot].tolist() == reference(
                        host, [(base + i) * batches + batch for i in range(width)]
                    )


def plain_compose(lhs, rhs):
    return compose(lhs, rhs)


def check_record_collectives_compile(monkeypatch, arch, case):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block = 128

    def apply(A, Out):
        tid = fx.thread_idx.x
        value = make_item(A[tid], tid)
        reduced = fx.coop.warp_reduce(value, compose, width=4)
        exclusive = fx.coop.warp_exclusive_scan(value, compose, width=4)
        batched = fx.coop.warp_reduce_batched((value, value, value), compose, width=4)
        result = compose(compose(reduced, exclusive), batched)
        store_item(Out, tid, result)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        apply(A, Out)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(data(block), torch.empty(block * 3, dtype=torch.int64, device="cpu"))


def check_warp_reduce_record_items_blocked_order(universal):

    block, width, items = (64, 8, 3)
    operation = fx.coop.universal.warp_reduce if universal else fx.coop.warp_reduce

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        values = make_items([make_item(A[tid * items + i], tid * items + i) for i in range(items)])
        result = operation(values, plain_compose, width=width)
        store_item(Out, tid, result)

    host = data(block * items)
    out, _ = launch_case(kernel, block, host, block, block)
    for tid in range(block):
        base = tid // width * width * items
        assert out[tid // width * width].tolist() == reference(host, range(base, base + width * items))


# Kernel runners and host references shared by sorting tests.


def _striped_to_linear(x, width, count):
    return x.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)


_FLOATS = [
    (fx.Float16, torch.float16),
    (fx.BFloat16, torch.bfloat16),
    (fx.Float32, torch.float32),
    (fx.Float64, torch.float64),
]
_PATTERN = [-7.75, -0.0, 2.125, -3.5, 2.125, 0.0, 0.1875, -7.75, 4.25, -0.5, float("inf"), -float("inf")]


def _namespace(universal):
    return fx.coop.universal if universal else fx.coop


def _logical(physical, width, count, striped=False):
    if striped:
        return physical.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)
    return physical.reshape(-1, width * count)


def _assert_key_payload(keys, ids, source):
    expected = source[ids]
    torch.testing.assert_close(keys, expected.double(), rtol=0, atol=0)
    assert torch.equal(torch.signbit(keys), torch.signbit(expected))


def check_sort(primitive, block, count, descending):
    torch.manual_seed(14)
    values = torch.randint(-11, 12, (block * count,), dtype=torch.int32, device="cuda")
    width = min(8, block)

    def transform(value):
        return getattr(fx.coop, primitive)(value, width=width, compare_op=lambda a, b: a > b if descending else a < b)

    result = run_tile(transform, values, block, count)[0]
    group = width * count
    expected = values.cpu().reshape(-1, group).sort(dim=1, descending=descending).values.reshape(-1)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, width).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(result, expected)


def check_sort_pairs(primitive, count, width, descending):
    block = 128
    group = (width or fx.num_warp_threads()) * count
    indices = torch.arange(block * count, dtype=torch.int32, device="cuda")

    def transform(value):
        keys = value * 17 % 11 - 5
        return getattr(fx.coop, primitive)(
            keys, value, width=width, compare_op=lambda a, b: a > b if descending else a < b
        )

    result = run_tile(transform, indices, block, count, outputs=2)
    keys = (indices.cpu() * 17 % 11 - 5).reshape(-1, group)
    expected, permutation = keys.sort(dim=1, descending=descending, stable=True)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, width or fx.num_warp_threads()).transpose(1, 2)
    torch.testing.assert_close(result[0], expected.reshape(-1))
    payload = result[1].long()
    torch.testing.assert_close((payload * 17 % 11 - 5).int(), result[0])
    torch.testing.assert_close(payload.sort().values, indices.cpu().long())
    if primitive == "warp_merge_sort":
        expected_payload = (
            permutation + torch.arange(block // (width or fx.num_warp_threads()), device="cpu")[:, None] * group
        )
        torch.testing.assert_close(payload, expected_payload.reshape(-1))


def check_bitonic_striped_pairs(count, partial):
    width, block = (8, 64)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 17 % 29
    valid = width * count - 3 if partial and count > 1 else width * count

    def transform(value):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)])
        return fx.coop.warp_bitonic_sort(
            value, ids.to(fx.Int64), width=width, compare_op=lambda a, b: a > b, valid_items=fx.Int32(valid)
        )

    result = run_tile(transform, data, block, count, outputs=2)
    logical = _striped_to_linear(data.cpu(), width, count)
    expected = logical[:, :valid].sort(descending=True).values
    actual = _striped_to_linear(result[0], width, count)
    torch.testing.assert_close(actual[:, :valid], expected)
    ids = _striped_to_linear(result[1], width, count)[:, :valid].long()
    torch.testing.assert_close(data.cpu()[ids], expected)


def check_merge_comparator_pairs_partial(scope, block, descending):
    count, width = (3, 8)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)
    total = width * count
    valid = total - 5

    def transform(value):
        keys = value * 19 % 43
        op = lambda a, b: a % 5 < b % 5
        return fx.coop.warp_merge_sort(
            keys,
            value,
            width=width,
            compare_op=lambda a, b: op(b, a) if descending else op(a, b),
            valid_items=fx.Int32(valid),
        )

    result = run_tile(transform, data, block, count, outputs=2)
    ids = data.cpu().reshape(-1, total)
    keys = ids * 19 % 43
    permutation = (keys[:, :valid] % 5).argsort(dim=1, descending=descending, stable=True)
    expected_ids = torch.gather(ids[:, :valid], 1, permutation)
    torch.testing.assert_close(result[1].reshape(-1, total)[:, :valid], expected_ids)
    torch.testing.assert_close(result[0].reshape(-1, total)[:, :valid], expected_ids * 19 % 43)


def check_record_keys_and_record_values(case, count):

    key_type = fx.Struct["major" : fx.Int32, "minor" : fx.Int64]
    value_type = fx.Struct["index" : fx.Int64, "tag" : fx.Float64]
    block, width = (8, 8)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = make_items([key_type(major=item * 17 % 11 - 5, minor=(item * 7 % 13).to(fx.Int64)) for item in value])
        payload = make_items([value_type(index=item.to(fx.Int64), tag=item.to(fx.Float64) * 2.0) for item in value])
        compare = lambda a, b: (a.major < b.major) | (a.major == b.major) & (a.minor < b.minor)
        if case == "bitonic":
            result, carried = fx.coop.warp_bitonic_sort(keys, payload, width=width, compare_op=compare)
        elif case == "warp_merge":
            result, carried = fx.coop.warp_merge_sort(keys, payload, width=width, compare_op=compare)
        result, carried = (as_items(result), as_items(carried))
        return (
            fx.Vector.from_elements([key.major for key in result]),
            fx.Vector.from_elements([key.minor.to(fx.Int32) for key in result]),
            fx.Vector.from_elements([item.index.to(fx.Int32) for item in carried]),
            fx.Vector.from_elements([item.tag.to(fx.Int32) for item in carried]),
        )

    result = run_tile(transform, data, block, count, outputs=4)
    ids = data.cpu()
    major, minor = (ids * 17 % 11 - 5, ids * 7 % 13)
    logical = _striped_to_linear(ids, width, count).flatten() if case == "bitonic" else ids
    composite = (major[logical] + 5) * 13 + minor[logical]
    order = logical[composite.argsort(stable=True)]
    if case == "bitonic":
        for i in range(4):
            result[i] = _striped_to_linear(result[i], width, count).flatten()
    k = len(ids)
    torch.testing.assert_close(result[0, :k], major[order[:k]])
    torch.testing.assert_close(result[1, :k], minor[order[:k]])
    torch.testing.assert_close(result[2, :k], order[:k])
    torch.testing.assert_close(result[3, :k], order[:k] * 2)


def check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    """Stable entry points preserve duplicate and signed-zero payload order."""
    ns = _namespace(universal)
    count, width = (3, 8)
    block = 16
    group = width * count
    valid = group - 5
    data = torch.tensor(
        (_PATTERN * ((block * count + len(_PATTERN) - 1) // len(_PATTERN)))[: block * count],
        dtype=torch_dtype,
        device="cuda",
    )
    striped_output = case == "bitonic" or (case == "radix_digit" and descending)

    def transform(keys):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)]).to(fx.Int64)
        if case == "bitonic":
            return ns.warp_bitonic_sort(
                keys,
                ids,
                width=width,
                valid_items=fx.Int32(valid),
                compare_op=lambda a, b: a > b if descending else a < b,
            )
        if case == "warp_merge":
            return ns.warp_merge_sort(
                keys,
                ids,
                width=width,
                valid_items=fx.Int32(valid),
                compare_op=lambda a, b: a > b if descending else a < b,
            )
        raise ValueError(f"unsupported warp sorting case: {case}")

    result = run_tile(transform, data, block, count, outputs=2, dtype=dtype, output_dtype=(fx.Float64, "float64"))
    group_width = width
    source_ids = _logical(torch.arange(len(data), device="cpu"), group_width, count, case == "bitonic")
    logical_keys = data.cpu()[source_ids]
    order = logical_keys[:, :valid].argsort(dim=1, descending=descending, stable=True)
    expected_ids = source_ids[:, :valid].gather(1, order)
    got_keys = _logical(result[0], group_width, count, striped_output)
    got_ids = _logical(result[1].long(), group_width, count, striped_output)
    selected_count = valid
    for group_id in range(len(source_ids)):
        ids = got_ids[group_id, :selected_count]
        keys = got_keys[group_id, :selected_count]
        expected = expected_ids[group_id, :selected_count]
        _assert_key_payload(keys, ids, data.cpu())
        if case == "bitonic":
            torch.testing.assert_close(keys, data.cpu()[expected].double(), rtol=0, atol=0)
            torch.testing.assert_close(ids.sort().values, source_ids[group_id, :valid].sort().values)
        else:
            torch.testing.assert_close(ids, expected)


def check_nested_float_record_keys(case, projected, universal):
    """Nested float fields and untouched fields follow comparator/decomposer order."""
    ns = _namespace(universal)
    inner = fx.Struct["major" : fx.Float16, "minor" : fx.Float32]
    key_type = fx.Struct["parts":inner, "tag" : fx.Float64]
    info = fx.Struct["id" : fx.Int64, "weight" : fx.Float32]
    value_type = fx.Struct["info":info]
    width, count = (8, 3)
    block = 16
    group_width = width
    n = block * count
    data = torch.tensor(([-3.5, 1.125, -0.0, 0.0, -3.5, 2.75, 1.125, -0.625] * ((n + 7) // 8))[:n], device="cuda")
    valid = group_width * count - 5

    def transform(value):
        ids = [fx.thread_idx.x * count + i for i in range(count)]
        keys = make_items(
            [
                key_type(
                    parts=inner(major=item.to(fx.Float16), minor=(idx % 5 - 2).to(fx.Float32) * 0.375),
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
        else:
            compare = lambda a, b: (a.parts.major < b.parts.major) | (a.parts.major == b.parts.major) & (
                a.parts.minor < b.parts.minor
            )
        if case == "bitonic":
            ordered, carried = ns.warp_bitonic_sort(
                keys,
                payload,
                width=width,
                compare_op=lambda a, b: compare(b, a),
                valid_items=valid,
            )
        elif case == "warp_merge":
            ordered, carried = ns.warp_merge_sort(
                keys,
                payload,
                width=width,
                compare_op=lambda a, b: compare(b, a),
                valid_items=valid,
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
    selected = valid
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
        if case == "bitonic":
            torch.testing.assert_close(got[0][group_id, :selected], host[expected_ids], rtol=0, atol=0)
            if not projected:
                torch.testing.assert_close(got[1][group_id, :selected], minor[expected_ids], rtol=0, atol=0)
            torch.testing.assert_close(ids.sort().values, group_ids[:valid].sort().values)
        else:
            torch.testing.assert_close(ids, expected_ids)
