# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Class and function APIs share per-column values, seeds and result types."""

import pytest
import torch
from coop_test_utils import run_kernel

import flydsl.expr as fx

Record = fx.Struct["value" : fx.Int32, "tag" : fx.Int32]


def add_record(lhs, rhs):
    return Record(lhs.value + rhs.value, lhs.tag + rhs.tag)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("form", ["vector", "tuple", "list", "record_tuple", "record_list"])
@pytest.mark.parametrize("seed", ["none", "scalar", "items"])
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("width,valid", [(1, None), (4, None), (4, 3)])
def test_warp_column_containers(form, seed, universal, default_device, width, valid):
    with torch.device(default_device):
        _check_warp_column_containers(form, seed, universal, width, valid)


def _check_warp_column_containers(form, seed, universal, width, valid):
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
        else:
            seeds = [
                Record(fx.Int32(10 * (i + 1)), fx.Int32(100 * (i + 1))) if record else fx.Int32(10 * (i + 1))
                for i in range(count)
            ]
            init = seeds if form.endswith("list") else tuple(seeds)
        scan = namespace.WarpScan[dtype, width]
        reduce = namespace.WarpReduce[dtype, width]
        head = tid % width % 2 == 0
        tail = tid % width % 2 == 1
        direct = (
            namespace.warp_inclusive_scan(value, op, width=width, init=init, valid_items=valid),
            namespace.warp_exclusive_scan(value, op, width=width, init=init, valid_items=valid),
            *namespace.warp_scan(value, op, width=width, init=init, valid_items=valid),
            *namespace.warp_scan_with_aggregate(value, op, width=width, init=init, valid_items=valid),
            namespace.warp_head_segmented_reduce(value, head, op, width=width),
            namespace.warp_tail_segmented_reduce(value, tail, op, width=width),
            namespace.warp_broadcast(value, source_lane, width=width),
        )
        member = (
            scan.inclusive_scan(value, op, init=init, valid_items=valid),
            scan.exclusive_scan(value, op, init=init, valid_items=valid),
            *scan.scan(value, op, init=init, valid_items=valid),
            *scan.scan_with_aggregate(value, op, init=init, valid_items=valid),
            reduce.head_segmented_reduce(value, head, op),
            reduce.tail_segmented_reduce(value, tail, op),
            scan.broadcast(value, source_lane),
        )
        for result_index, result in enumerate(direct + member):
            assert isinstance(result, tuple if record else fx.Vector)
            assert len(result) == count
            for i in range(count):
                base = ((result_index * block + tid) * count + i) * fields
                out[base] = result[i].value if record else result[i]
                if record:
                    out[base + 1] = result[i].tag

    actual = run_kernel(apply, host.cuda(), 2 * results_per_api * block * count * fields, block)
    actual = actual.reshape(2, results_per_api, block // width, width, count, fields)
    inputs = host.reshape(block // width, width, count, 1)
    if record:
        inputs = torch.cat((inputs, inputs * 3), dim=-1)
    prefix = inputs.cumsum(1).to(torch.int32)
    initial = torch.zeros((count, fields), dtype=torch.int32, device="cpu")
    if seed != "none":
        for i in range(count):
            initial[i, 0] = 10 * (i + 1 if seed == "items" else 1)
            if record:
                initial[i, 1] = 100 * (i + 1 if seed == "items" else 1)
    inclusive = prefix + initial
    exclusive = prefix - inputs + initial
    limit = width if valid is None else valid
    aggregate = inputs[:, :limit].sum(1, keepdim=True).to(torch.int32).expand_as(inputs)
    segment = min(2, width)
    segmented = inputs.reshape(block // width, width // segment, segment, count, fields).sum(2).to(torch.int32)
    broadcast = inputs[:, source_lane : source_lane + 1].expand_as(inputs)
    for api in range(2):
        for index in (0, 2, 4):
            torch.testing.assert_close(actual[api, index, :, :limit], inclusive[:, :limit])
        for index in (1, 3, 5):
            # A custom unseeded exclusive scan leaves the first lane unspecified.
            first = 1 if record and seed == "none" else 0
            torch.testing.assert_close(actual[api, index, :, first:limit], exclusive[:, first:limit])
        torch.testing.assert_close(actual[api, 6], aggregate)
        for index in (7, 8):
            torch.testing.assert_close(actual[api, index, :, ::segment], segmented)
        torch.testing.assert_close(actual[api, 9], broadcast)
