# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Nested record kernels and references shared by reduction and scan tests."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop._values import _as_items


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
        for i, item in enumerate(_as_items(result)):
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
    from flydsl.extension.coop._values import _from_items
    from flydsl.extension.coop.warp.reduce import warp_reduce as portable
    from flydsl.extension.coop.warp.rocdl import warp_reduce as dispatched

    block, width, items = (64, 8, 3)
    operation = portable if universal else dispatched

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor, Agg: fx.Tensor):
        tid = fx.thread_idx.x
        values = _from_items([make_item(A[tid * items + i], tid * items + i) for i in range(items)])
        result = operation(values, plain_compose, width=width)
        store_item(Out, tid, result)

    host = data(block * items)
    out, _ = launch_case(kernel, block, host, block, block)
    for tid in range(block):
        base = tid // width * width * items
        assert out[tid // width * width].tolist() == reference(host, range(base, base + width * items))
