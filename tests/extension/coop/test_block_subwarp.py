# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Subwarp block collectives preserve full-tile and partial-tile semantics."""

import math

import pytest
import torch
from coop_common import SUB_WARP_BLOCK_THREADS, linear_tid
from coop_test_utils import as_items, run_kernel
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.expr as fx

POLICIES = [("reduce", p) for p in fx.coop.BlockReduceAlgorithm] + [("scan", p) for p in fx.coop.BlockScanAlgorithm]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("coop_default_device")
@pytest.mark.parametrize("shape", [*SUB_WARP_BLOCK_THREADS, (2, 2, 2)])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("family,policy", POLICIES, ids=[f"{family}-{p.name}" for family, p in POLICIES])
def test_subwarp_block_collectives(shape, count, universal, family, policy):
    shape = (shape, 1, 1) if isinstance(shape, int) else shape
    threads = math.prod(shape)
    size = threads * count
    namespace = fx.coop.universal if universal else fx.coop
    primitive = getattr(namespace, "BlockReduce" if family == "reduce" else "BlockScan")[fx.Int32, shape, policy]
    assert primitive.warp_threads == threads
    assert primitive.num_warps == 1
    result_size = threads if family == "reduce" else 2 * size + threads

    def apply(a, out):
        tid = linear_tid(shape)
        items = [a[tid * count + i] for i in range(count)]
        value = items[0] if count == 1 else fx.Vector.from_elements(items)
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        for partial in (False, True):
            limit = size - 1 if partial else None
            offset = result_size if partial else 0
            if family == "reduce":
                out[offset + tid] = primitive(value, fx.ReductionOp.ADD, storage=storage, valid_items=limit)
            else:
                inclusive, aggregate = primitive.inclusive_with_aggregate(
                    value, fx.ReductionOp.ADD, storage=storage, init=7, valid_items=limit
                )
                for i, item in enumerate(as_items(inclusive)):
                    out[offset + tid * count + i] = item
                out[offset + 2 * size + tid] = aggregate
                fx.barrier()
                exclusive = primitive.exclusive(value, fx.ReductionOp.ADD, storage=storage, init=7, valid_items=limit)
                for i, item in enumerate(as_items(exclusive)):
                    out[offset + size + tid * count + i] = item
            fx.barrier()

    host = torch.arange(1, size + 1, device="cpu", dtype=torch.int32)
    actual = run_kernel(apply, host.to(device="cuda"), result_size * 2, shape).reshape(2, result_size)
    for partial in (False, True):
        expected_items = host.clone()
        if partial:
            expected_items[-1] = 0
        total = expected_items.sum(dtype=torch.int32).expand(threads)
        if family == "reduce":
            expected = total
        else:
            prefix = expected_items.cumsum(0, dtype=torch.int32)
            expected = torch.cat((prefix + 7, prefix - expected_items + 7, total))
        torch.testing.assert_close(actual[int(partial)], expected)
