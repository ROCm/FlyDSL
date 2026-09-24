# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Guarded warp callbacks and launches with runtime block dimensions."""

import math

import pytest
import torch
from coop_test_utils import run_kernel

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop.warp.load import WarpLoadAlgorithm
from flydsl.extension.coop.warp.store import WarpStoreAlgorithm

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU"),
]


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.parametrize("member", [False, True], ids=["function", "class"])
@pytest.mark.parametrize("head", [False, True], ids=["tail", "head"])
@pytest.mark.parametrize("flag_value", [1, 1 << 32, -(1 << 32)])
@pytest.mark.parametrize("singletons", [False, True], ids=["mixed", "singletons"])
def test_segmented_reduce_flag_truth_and_operator_domain(
    default_device, universal, member, head, flag_value, singletons
):
    with torch.device(default_device):
        block, width = 64, 8
        ns = fx.coop.universal if universal else fx.coop
        segments = list(range(width)) if singletons else [0, 1, 1, 1, 2, 2, 3, 3]
        heads = [i for i in range(width) if i == 0 or segments[i] != segments[i - 1]]
        tails = [i for i in range(width) if i == width - 1 or segments[i] != segments[i + 1]]
        flags = torch.zeros(block, dtype=torch.int64, device="cpu")
        values = torch.empty(block, dtype=torch.int64, device="cpu")
        expected = torch.empty(block, dtype=torch.int64, device="cpu")
        for base in range(0, block, width):
            for lane in range(width):
                # The high digits identify the operator's domain; max preserves it.
                values[base + lane] = (base + segments[lane] + 1) * 1000 + lane
            for first, last in zip(heads, tails):
                expected[base + first] = values[base + last]
            for lane in heads if head else tails:
                flags[base + lane] = flag_value
        inputs = torch.cat([values, flags]).to(device="cuda")

        def apply(a, out):
            tid = fx.thread_idx.x
            out[tid * 3 + 1] = fx.Int64(0)
            out[tid * 3 + 2] = fx.Int64(0)

            def maximum(lhs, rhs):
                out[tid * 3 + 1] = out[tid * 3 + 1] + 1
                out[tid * 3 + 2] = out[tid * 3 + 2] + fx.Int64(lhs // 1000 != rhs // 1000)
                return fx.max(lhs, rhs)

            value, flag = a[tid].to(fx.Int32), a[block + tid].to(fx.Int64)
            if member:
                primitive = ns.WarpReduce[fx.Int32, width]
                method = primitive.head_segmented_reduce if head else primitive.tail_segmented_reduce
                result = method(value, flag, maximum)
            else:
                method = ns.warp_head_segmented_reduce if head else ns.warp_tail_segmented_reduce
                result = method(value, flag, maximum, width=width)
            out[tid * 3] = fx.Int64(result)

        actual = run_kernel(apply, inputs, block * 3, block).reshape(block, 3)
        owners = torch.tensor([base + lane for base in range(0, block, width) for lane in heads], device="cpu")
        torch.testing.assert_close(actual[owners, 0], expected[owners])
        assert torch.count_nonzero(actual[:, 2]).item() == 0
        if singletons:
            assert torch.count_nonzero(actual[:, 1]).item() == 0
        else:
            assert actual[:, 1].sum().item() > 0


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.parametrize("member", [False, True], ids=["function", "class"])
@pytest.mark.parametrize("bitonic", [False, True], ids=["merge", "bitonic"])
@pytest.mark.parametrize("descending", [False, True])
def test_partial_sort_skips_invalid_comparisons(default_device, universal, member, bitonic, descending):
    with torch.device(default_device):
        block, width, count = 64, 8, 3
        tile, size = width * count, block * count
        ns = fx.coop.universal if universal else fx.coop
        limits = [0, 1, 5, tile] * (block // width // 4)
        # A non-power-of-two item count also exercises the bitonic padding.
        logical = torch.full((block // width, tile), -99, dtype=torch.int32, device="cpu")
        for group, valid in enumerate(limits):
            logical[group, :valid] = (torch.arange(valid, device="cpu") * 7 + group) % 13 + 1
        registers = logical.reshape(-1, count, width).transpose(1, 2).reshape(-1) if bitonic else logical.flatten()
        inputs = torch.cat([registers, torch.tensor(limits, dtype=torch.int32, device="cpu")]).to(device="cuda")

        def apply(a, out):
            tid = fx.thread_idx.x
            out[size + tid] = fx.Int32(0)

            def compare(lhs, rhs):
                out[size + tid] = out[size + tid] + fx.Int32((lhs == -99) | (rhs == -99))
                return lhs > rhs if descending else lhs < rhs

            keys = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
            valid = a[size + tid // width]
            if member:
                primitive = ns.WarpBitonicSort if bitonic else ns.WarpMergeSort
                result = primitive[fx.Int32, width, count].sort(keys, compare_op=compare, valid_items=valid)
            else:
                method = ns.warp_bitonic_sort if bitonic else ns.warp_merge_sort
                result = method(keys, width=width, compare_op=compare, valid_items=valid)
            for i in range(count):
                out[tid * count + i] = result[i]

        actual = run_kernel(apply, inputs, size + block, block)
        assert torch.count_nonzero(actual[size:]).item() == 0
        ordered = actual[:size].reshape(-1, width, count)
        if bitonic:
            ordered = ordered.transpose(1, 2)
        ordered = ordered.reshape(-1, tile)
        for group, valid in enumerate(limits):
            expected = torch.sort(logical[group, :valid], descending=descending).values
            torch.testing.assert_close(ordered[group, :valid], expected)


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("universal", [False, True], ids=["public", "universal"])
@pytest.mark.parametrize("policy", list(WarpLoadAlgorithm))
@pytest.mark.parametrize("shape", [(128, 1, 1), (4, 4, 8)])
def test_warp_io_runtime_block_size(default_device, universal, policy, shape):
    with torch.device(default_device):
        block, width, count = math.prod(shape), 8, 4
        tile, size = width * count, block * count
        ns = fx.coop.universal if universal else fx.coop
        source = torch.arange(size, dtype=torch.int32, device="cuda")
        out = torch.full((size * 2,), -99, dtype=torch.int32, device="cuda")

        @flyc.kernel
        def kernel(a: fx.Tensor, output: fx.Tensor):
            tid = fx.thread_idx.x + fx.block_dim.x * (fx.thread_idx.y + fx.block_dim.y * fx.thread_idx.z)
            group = tid // width
            offset = group * tile
            valid = (group % 2 == 0).select(fx.Int32(tile), fx.Int32(tile - 3))
            items = ns.WarpLoad[fx.Int32, width, count, policy].load(a, offset=offset, valid_items=valid, default=-7)
            ns.WarpStore[fx.Int32, width, count, WarpStoreAlgorithm[policy.name]].store(
                output, items, offset=offset, valid_items=valid
            )
            for i in fx.range_constexpr(count):
                output[size + tid * count + i] = items[i]

        @flyc.jit
        def launch(
            a: fx.Tensor,
            output: fx.Tensor,
            nx: fx.Int32,
            ny: fx.Int32,
            nz: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            kernel(a, output).launch(grid=(1, 1, 1), block=(nx, ny, nz), stream=stream)

        launch(source, out, *shape, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        indices = torch.arange(size, device="cpu")
        limits = torch.where(indices // tile % 2 == 0, tile, tile - 3)
        valid_mask = indices % tile < limits
        host = source.cpu()
        stored = torch.where(valid_mask, host, -99)
        loaded = torch.where(valid_mask, host, -7)
        if policy is WarpLoadAlgorithm.STRIPED:
            loaded = loaded.reshape(-1, count, width).transpose(1, 2).reshape(-1)
        torch.testing.assert_close(out.cpu(), torch.cat([stored, loaded]))
