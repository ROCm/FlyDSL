# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared launch helpers, layout references and compilation targets for coop tests."""

import math

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

ARCHES = ("gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1151", "gfx1201")


def run_tile(transform, values, block, count, outputs=1, dtype=None, output_dtype=None):
    shape = (block, 1, 1) if isinstance(block, int) else block
    threads = math.prod(shape)

    def apply(a, out):
        tid = fx.thread_idx.x + fx.thread_idx.y * shape[0] + fx.thread_idx.z * shape[0] * shape[1]
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if dtype is not None:
            value = value.to(dtype)
        result = transform(value)
        result = (result,) if outputs == 1 else result
        for j, vector in enumerate(result):
            for i in range(count):
                out[j * threads * count + tid * count + i] = vector[i].to(
                    value.dtype if output_dtype is None else output_dtype[0]
                )

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=shape, stream=stream)

    out = torch.empty(
        values.numel() * outputs,
        dtype=values.dtype if output_dtype is None else getattr(torch, output_dtype[1]),
        device="cuda",
    )
    launch(values, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu().reshape(outputs, -1)


def run_kernel(apply, values, out_size, block):
    shape = (block, 1, 1) if isinstance(block, int) else block

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=shape, stream=stream)

    out = torch.full((out_size,), -99, device="cuda", dtype=values.dtype)
    argument = values if values.stride() == (1,) else flyc.from_dlpack(values)
    launch(argument, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def warp_indices(threads, count, width, inverse=False):
    out = []
    for first in range(0, threads, width):
        extent = min(width, threads - first)
        for lane in range(extent):
            for item in range(count):
                position = lane * count + item
                relative = (position % extent) * count + position // extent if inverse else lane + item * extent
                out.append(first * count + relative)
    return torch.tensor(out, device="cpu")


def dtype_entry(entry):
    return entry[0], entry[1].removeprefix("torch.")


def batched_columns(namespace, value, op, *, width=None, sync_physical_warp=False):
    """Collect distributed batch results with the public broadcast primitive."""
    from flydsl.extension.coop._values import _from_items

    group = width or fx.num_warp_threads()
    distributed = namespace.warp_reduce_batched_to_striped(
        value, op, width=width, sync_physical_warp=sync_physical_warp
    )
    return _from_items(
        [namespace.warp_broadcast(distributed[i // group], i % group, width=width) for i in range(len(value))]
    )


def warp_valid_items(valid, width, count):
    """Convert a caller's block prefix to this logical warp's element count."""
    if valid is None:
        return None
    group = width or fx.num_warp_threads()
    remaining = fx.Int32(valid) - (fx.thread_idx.x // group) * group * count
    return fx.min(fx.max(remaining, 0), group * count)


@pytest.fixture(params=["cpu", "cuda"], ids=["default-cpu", "default-cuda"])
def warp_default_device(request):
    """Exercise host references under both defaults without leaking state."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    with torch.device(request.param):
        yield
