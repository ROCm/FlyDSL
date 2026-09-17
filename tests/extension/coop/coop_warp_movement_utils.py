# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared load/store checks and kernels for movement policy compilation."""

from coop_test_utils import run_kernel, warp_indices, warp_valid_items

import flydsl.compiler as flyc
import flydsl.expr as fx
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
        items = ns.warp_load(
            a,
            count,
            width=width,
            offset=warp_offset,
            valid_items=warp_valid_items(limit, width, count),
            default=-7 if valid is not None else None,
            algorithm=(
                fx.coop.WarpLoadAlgorithm.STRIPED if policy == "warp_striped" else fx.coop.WarpLoadAlgorithm.DIRECT
            ),
        )
        ns.warp_store(
            out,
            items,
            width=width,
            offset=warp_offset,
            valid_items=warp_valid_items(limit, width, count),
            algorithm=(
                fx.coop.WarpStoreAlgorithm.STRIPED if policy == "warp_striped" else fx.coop.WarpStoreAlgorithm.DIRECT
            ),
        )
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
        items = warp_load(
            a,
            count,
            width=width,
            algorithm=getattr(WarpLoadAlgorithm, policy),
            offset=warp_offset,
            valid_items=warp_valid_items(fx.Int32(valid), width, count),
            default=-7,
        )
        warp_store(
            out,
            items,
            width=width,
            algorithm=getattr(WarpStoreAlgorithm, policy),
            offset=warp_offset,
            valid_items=warp_valid_items(fx.Int32(valid), width, count),
        )
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
            storage = fx.SharedAllocator().allocate(warp_exchange_storage(fx.Int32, count)).peek()
            policy = WarpExchangeAlgorithm.SMEM
            kwargs = dict(width=8, algorithm=policy, storage=storage)
            if case.endswith("blocked"):
                result = warp_blocked_to_striped(value, **kwargs)
            elif case.endswith("striped"):
                result = warp_striped_to_blocked(value, **kwargs)
            else:
                ranks = fx.Vector.from_elements([31 - (tid % 8 * count + i) for i in range(count)])
                result = warp_scatter_to_striped(value, ranks, **kwargs)
        elif case.startswith("warp_io"):
            policy = case.removeprefix("warp_io_")
            warp_offset = tid // 8 * 8 * count
            result = warp_load(
                a, count, width=8, offset=warp_offset, algorithm=WarpLoadAlgorithm[policy], valid_items=31, default=-7
            )
            warp_store(out, result, width=8, offset=warp_offset, algorithm=WarpStoreAlgorithm[policy], valid_items=31)
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
        items = warp_load(
            a,
            count,
            width=width,
            algorithm=WarpLoadAlgorithm[policy],
            offset=warp_offset,
            valid_items=warp_valid_items(fx.Int32(valid), width, count),
            default=-7,
        )
        warp_store(
            out,
            items,
            width=width,
            algorithm=WarpStoreAlgorithm[policy],
            offset=warp_offset,
            valid_items=warp_valid_items(fx.Int32(valid), width, count),
        )
        if convert_load:
            items = items.to(loaded_dtype)
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
