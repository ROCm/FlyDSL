# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared kernels and host references for primitive matrices and compilation."""

from coop_common import WARP_SIZE, sample
from coop_test_utils import batched_columns, run_tile

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None


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
                storage = allocator.allocate(ns.warp_exchange_storage(dtype, count)).peek()
                out = getattr(ns, primitive)(arg, width=width, storage=storage)
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
        op = getattr(fx.coop, case)
        if case == "warp_load":
            result = op(a, count, width=8, offset=tid // 8 * 8 * count, valid_items=31)
        elif case == "warp_store":
            op(out, value, width=8, offset=tid // 8 * 8 * count, valid_items=31)
            return
        elif case == "warp_reduce_batched":
            result = batched_columns(fx.coop, value, fx.ReductionOp.ADD, width=8)
        else:
            if case.endswith("sort"):
                result = op(value, width=8, compare_op=lambda a, b: a < b)
            else:
                storage = fx.SharedAllocator().allocate(fx.coop.warp_exchange_storage(fx.Int32, count)).peek()
                result = op(value, width=8, storage=storage)
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
