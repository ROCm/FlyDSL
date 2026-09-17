# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared record kernels and host references for exchange, load, and store checks."""

from coop_test_utils import warp_indices

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop._values import _from_items
from flydsl.extension.coop.warp.exchange import (
    WarpExchangeAlgorithm,
    warp_blocked_to_striped,
    warp_exchange_storage,
    warp_scatter_to_striped,
    warp_striped_to_blocked,
)
from flydsl.extension.coop.warp.load import WarpLoadAlgorithm, warp_load
from flydsl.extension.coop.warp.store import WarpStoreAlgorithm, warp_store

try:
    import torch
except ImportError:
    torch = None

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
            items = warp_load(
                source,
                count,
                width=8,
                offset=warp_offset,
                algorithm=WarpLoadAlgorithm[policy],
                valid_items=8 * count - 3,
                default=-7,
            )
            warp_store(
                destination,
                items,
                width=8,
                offset=warp_offset,
                algorithm=WarpStoreAlgorithm[policy],
                valid_items=8 * count - 3,
            )
        else:
            items = _from_items([source[fx.thread_idx.x * count + i] for i in range(count)])
            storage = allocator.allocate(warp_exchange_storage(RECORD, count)).peek()
            kwargs = dict(width=8, algorithm=WarpExchangeAlgorithm[policy], storage=storage)
            if case == "warp_blocked_to_striped":
                items = warp_blocked_to_striped(items, **kwargs)
            elif case == "warp_striped_to_blocked":
                items = warp_striped_to_blocked(items, **kwargs)
            else:
                ranks = fx.Vector.from_elements(
                    [8 * count - 1 - (fx.thread_idx.x % 8 * count + i) for i in range(count)]
                )
                items = warp_scatter_to_striped(items, ranks, **kwargs)
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
