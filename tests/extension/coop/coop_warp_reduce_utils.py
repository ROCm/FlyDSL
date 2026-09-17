# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared kernels and reference checks for scalar reductions and scans."""

import torch
from coop_test_utils import batched_columns, run_tile

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.extension.coop.warp.reduce import warp_head_segmented_reduce, warp_tail_segmented_reduce


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
        form = warp_head_segmented_reduce if head else warp_tail_segmented_reduce
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
    from flydsl.extension.coop.warp.reduce import warp_reduce as portable
    from flydsl.extension.coop.warp.rocdl import warp_reduce as dispatched

    block, width, items = (64, 8, 3)
    operation = portable if universal else dispatched

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
