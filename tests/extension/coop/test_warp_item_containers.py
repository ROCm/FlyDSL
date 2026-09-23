# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordinary item ranges form one blocked task in both class and function APIs."""

import pytest
import torch
from coop_common import WARP_SIZE
from coop_test_utils import run_kernel
from coop_warp_utils import host_fold, plain_affine

import flydsl.expr as fx

Record = fx.Struct["value" : fx.Int32, "tag" : fx.Int32]


def add_record(lhs, rhs):
    return Record(lhs.value + rhs.value, lhs.tag + rhs.tag)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("form", ["vector", "tuple", "list", "record_tuple", "record_list"])
@pytest.mark.parametrize("seed", ["none", "scalar"])
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [1, 4])
def test_warp_item_containers(form, seed, universal, default_device, width):
    with torch.device(default_device):
        _check_warp_item_containers(form, seed, universal, width)


def _check_warp_item_containers(form, seed, universal, width):
    block, count = 64, 2
    record = form.startswith("record")
    dtype = Record if record else fx.Int32
    op = add_record if record else fx.ReductionOp.ADD
    namespace = fx.coop.universal if universal else fx.coop
    fields = 2 if record else 1
    source_lane = min(2, width - 1)
    # inclusive, exclusive, both forms of scan, three aggregate results,
    # two segmented reductions and broadcast, for each of the two APIs.
    results_per_api = 10
    host = torch.arange(1, block * count + 1, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [a[tid * count + i] for i in range(count)]
        if record:
            items = [Record(item, item * 3) for item in items]
        value = (
            fx.Vector.from_elements(items) if form == "vector" else (items if form.endswith("list") else tuple(items))
        )
        if seed == "none":
            init = None
        elif seed == "scalar":
            init = Record(fx.Int32(10), fx.Int32(100)) if record else fx.Int32(10)
        scan = namespace.WarpScan[dtype, width]
        reduce = namespace.WarpReduce[dtype, width]
        head = tid % width % 2 == 0
        tail = tid % width % 2 == 1
        direct = (
            namespace.warp_inclusive_scan(value, op, width=width, init=init),
            namespace.warp_exclusive_scan(value, op, width=width, init=init),
            *namespace.warp_scan(value, op, width=width, init=init),
            *namespace.warp_scan_with_aggregate(value, op, width=width, init=init),
            namespace.warp_head_segmented_reduce(value, head, op, width=width),
            namespace.warp_tail_segmented_reduce(value, tail, op, width=width),
            namespace.warp_broadcast(value, source_lane, width=width),
        )
        member = (
            scan.inclusive_scan(value, op, init=init),
            scan.exclusive_scan(value, op, init=init),
            *scan.scan(value, op, init=init),
            *scan.scan_with_aggregate(value, op, init=init),
            reduce.head_segmented_reduce(value, head, op),
            reduce.tail_segmented_reduce(value, tail, op),
            scan.broadcast(value, source_lane),
        )
        for result_index, result in enumerate(direct + member):
            single = result_index % results_per_api in (6, 7, 8)
            if single:
                assert isinstance(result, Record if record else fx.Numeric)
            else:
                assert isinstance(result, tuple if record else fx.Vector)
                assert len(result) == count
            for i in range(1 if single else count):
                item = result if single else result[i]
                base = ((result_index * block + tid) * count + i) * fields
                out[base] = item.value if record else item
                if record:
                    out[base + 1] = item.tag

    actual = run_kernel(apply, host.cuda(), 2 * results_per_api * block * count * fields, block)
    actual = actual.reshape(2, results_per_api, block // width, width, count, fields)
    inputs = host.reshape(block // width, width, count, 1)
    if record:
        inputs = torch.cat((inputs, inputs * 3), dim=-1)
    flat = inputs.reshape(block // width, width * count, fields)
    prefix = flat.cumsum(1).to(torch.int32)
    initial = torch.zeros((fields,), dtype=torch.int32, device="cpu")
    if seed != "none":
        initial[0] = 10
        if record:
            initial[1] = 100
    inclusive = prefix + initial
    exclusive = prefix - flat + initial
    aggregate = flat.sum(1).to(torch.int32)[:, None, :].expand(-1, width, -1)
    segment = min(2, width)
    segmented = flat.reshape(block // width, width // segment, segment * count, fields).sum(2).to(torch.int32)
    broadcast = inputs[:, source_lane : source_lane + 1].expand_as(inputs)
    for api in range(2):
        for index in (0, 2, 4):
            torch.testing.assert_close(actual[api, index].reshape_as(flat), inclusive)
        for index in (1, 3, 5):
            # Only the first item of an unseeded custom exclusive scan is unspecified.
            first = 1 if record and seed == "none" else 0
            torch.testing.assert_close(actual[api, index].reshape_as(flat)[:, first:], exclusive[:, first:])
        torch.testing.assert_close(actual[api, 6, :, :, 0], aggregate)
        for index in (7, 8):
            torch.testing.assert_close(actual[api, index, :, ::segment, 0], segmented)
        torch.testing.assert_close(actual[api, 9], broadcast)


def _ordered_arrays(universal, count, width, initial):
    namespace = fx.coop.universal if universal else fx.coop

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [a[tid * count + i] for i in range(count)]
        inc, exc, aggregate = namespace.warp_scan_with_aggregate(
            items, plain_affine, width=width, init=None if initial is None else fx.Int32(initial)
        )
        head = namespace.warp_head_segmented_reduce(items, tid % 2 == 0, plain_affine, width=width)
        tail = namespace.warp_tail_segmented_reduce(items, tid % 2 == 1, plain_affine, width=width)
        for i in range(count):
            out[tid * (2 * count + 3) + i] = inc[i]
            out[tid * (2 * count + 3) + count + i] = exc[i]
        for i, item in enumerate((aggregate, head, tail)):
            out[tid * (2 * count + 3) + 2 * count + i] = item

    return apply


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("width", [1, 4, WARP_SIZE])
@pytest.mark.parametrize("initial", [None, (7 << 16) | 19])
def test_warp_arrays_preserve_noncommutative_order(universal, default_device, count, width, initial):
    with torch.device(default_device):
        block = 64
        ids = torch.arange(block * count, dtype=torch.int32, device="cpu")
        host = ((ids * 13 + 3) % 251 << 16) | (ids * 7 + 11) % 251
        actual = run_kernel(
            _ordered_arrays(universal, count, width, initial), host.to(device="cuda"), block * (2 * count + 3), block
        ).reshape(block // width, width, 2 * count + 3)
        for group in range(block // width):
            values = host[group * width * count : (group + 1) * width * count]
            inclusive = actual[group, :, :count].flatten().tolist()
            exclusive = actual[group, :, count : 2 * count].flatten().tolist()
            seed = 1 << 16 if initial is None else initial
            assert inclusive == [host_fold(values[: i + 1], init=seed) for i in range(width * count)]
            start = 1 if initial is None else 0
            assert exclusive[start:] == [host_fold(values[:i], init=seed) for i in range(start, width * count)]
            assert actual[group, :, 2 * count].tolist() == [host_fold(values)] * width
            segment = min(2, width)
            for lane in range(0, width, segment):
                expected = host_fold(values[lane * count : (lane + segment) * count])
                assert actual[group, lane, 2 * count + 1 :].tolist() == [expected, expected]


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("width", [4, None], ids=["subwarp", "fullwarp"])
def test_warp_arrays_compile(monkeypatch, arch, universal, width):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    values = torch.empty(64 * 3, dtype=torch.int32, device="cpu")
    run_kernel(_ordered_arrays(universal, 3, width, None), values, 64 * 9, 64, compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("form", ["list", "tuple"])
@pytest.mark.parametrize("member", [False, True])
@pytest.mark.parametrize("builtin", [False, True])
@pytest.mark.parametrize("width", [1, 4, WARP_SIZE])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_reduce_fallback_preserves_vector_items(form, member, builtin, width, default_device):
    """A Vector inside an outer array is one T; backend fallback must not fold it again."""
    with torch.device(default_device):
        block, count = 64, 2
        host = torch.arange(1, block * count + 1, dtype=torch.int32, device="cpu")

        def apply(a, out):
            tid = fx.thread_idx.x
            items = [fx.Vector.from_elements([a[tid * count + i], a[tid * count + i] * 10]) for i in range(count)]
            value = tuple(items) if form == "tuple" else items
            op = fx.ReductionOp.ADD if builtin else lambda lhs, rhs: lhs + rhs
            for api, namespace in enumerate((fx.coop, fx.coop.universal)):
                if member:
                    result = namespace.WarpReduce[fx.Vector[fx.Int32, 2], width].reduce(value, op)
                else:
                    result = namespace.warp_reduce(value, op, width=width)
                assert isinstance(result, fx.Vector) and len(result) == 2
                for i in range(2):
                    out[api * block * 2 + tid * 2 + i] = result[i]

        actual = run_kernel(apply, host.to(device="cuda"), 2 * block * 2, block).reshape(2, block, 2)
        totals = host.reshape(-1, width * count).sum(1).to(torch.int32).repeat_interleave(width)
        expected = torch.stack((totals, totals * 10), dim=1)
        for result in actual:
            torch.testing.assert_close(result, expected)
