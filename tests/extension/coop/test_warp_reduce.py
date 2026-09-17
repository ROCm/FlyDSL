# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp reductions, segmented folds, item order and nested records."""

import coop_warp_reduce_record_utils as record_checks
import coop_warp_reduce_utils as reduce_checks
import pytest
import torch
from coop_common import WARP_SIZE
from coop_test_utils import ARCHES, batched_columns, run_kernel

import flydsl.compiler as flyc
import flydsl.expr as fx


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [1, 2, 8, 32, 64])
@pytest.mark.parametrize("head", [False, True])
def test_segmented_ordered_reduction(width, head):
    reduce_checks.check_segmented_ordered_reduction(width, head)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True])
def test_warp_reduce_array_blocked_order(universal):
    reduce_checks.check_warp_reduce_array_blocked_order(universal)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_noncommutative"])
def test_noncommutative_collectives_compile(monkeypatch, arch, case):
    reduce_checks.check_reduce_scan_policies_compile(monkeypatch, arch, case)


@pytest.mark.rocm_lower
@pytest.mark.l2_device
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True])
def test_warp_reduce_record_items_blocked_order(universal):
    record_checks.check_warp_reduce_record_items_blocked_order(universal)


@pytest.mark.rocm_lower
@pytest.mark.l2_device
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", ["reduce", "segmented"])
@pytest.mark.parametrize("universal", [False, True])
def test_nested_record_reduce_forms(case, universal):
    record_checks.check_nested_record_warp_collectives(case, universal)


@pytest.mark.rocm_lower
@pytest.mark.l1b_target_dialect
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp"])
def test_record_collectives_compile(monkeypatch, arch, case):
    record_checks.check_record_collectives_compile(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [8, None])
@pytest.mark.parametrize("form", ["vector", "tuple", "list"])
@pytest.mark.parametrize("valid", [None, 0, 3])
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_array_reduce_and_batched_columns(width, form, valid, universal, default_device):
    with torch.device(default_device):
        _check_array_reduce_and_batched_columns(width, form, valid, universal)


def _check_array_reduce_and_batched_columns(width, form, valid, universal):
    """Array reduction folds the tile; batched reduction reduces each column independently."""
    block, count = 128, 3
    group_width = width or WARP_SIZE
    values = torch.arange(block * count, dtype=torch.int32, device="cuda") % 29
    ns = fx.coop.universal if universal else fx.coop

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [a[tid * count + i] for i in range(count)]
        value = fx.Vector.from_elements(items) if form == "vector" else tuple(items) if form == "tuple" else items
        limit = None if valid is None else fx.Int32(valid)
        local_total = items[0] + items[1] + items[2]
        total = (
            ns.warp_reduce(value, fx.ReductionOp.ADD, width=width)
            if valid is None
            else ns.warp_reduce(local_total, fx.ReductionOp.ADD, width=width, valid_items=limit)
        )
        columns = batched_columns(ns, value, fx.ReductionOp.ADD, width=width)
        out[tid * (count + 1)] = ns.warp_broadcast(total, 0, width=width)
        for i in range(count):
            out[tid * (count + 1) + i + 1] = columns[i]

    actual = run_kernel(apply, values, block * (count + 1), block).reshape(block, count + 1)
    grouped = values.cpu().reshape(-1, group_width, count)
    total = grouped[:, : group_width if valid is None else valid].sum(dim=(1, 2))
    columns = grouped.sum(dim=1)
    expected = torch.cat([total[:, None], columns], dim=1).repeat_interleave(group_width, dim=0).int()
    if valid == 0:
        # Empty reductions are unspecified; batched column results remain defined.
        torch.testing.assert_close(actual[:, 1:], expected[:, 1:])
    else:
        torch.testing.assert_close(actual, expected)


@flyc.jit
def _fold_valid_items(items, op, limit):
    """Fold local ranges only in lanes contributing to the guarded scalar reduction."""
    partial = items[0]
    if fx.thread_idx.x % 4 < limit:
        for i in fx.range_constexpr(1, len(items)):
            partial = op(partial, items[i])
    return partial


def _counted_reduce(universal, form, valid, runtime):
    ns = fx.coop.universal if universal else fx.coop
    count = 1 if form == "scalar" else 3

    def apply(a, out):
        tid = fx.thread_idx.x
        out[tid * 2 + 1] = fx.Int32(0)

        def add(lhs, rhs):
            # Observe operator evaluation, including calls whose results are discarded.
            out[tid * 2 + 1] = out[tid * 2 + 1] + 1
            return lhs + rhs

        items = tuple(a[1 + tid * count + i] for i in range(count))
        items[0] if form == "scalar" else fx.Vector.from_elements(items) if form == "vector" else items
        limit = a[0] if runtime else valid
        value = fx.Vector.from_elements(items) if form == "vector" else items
        partial = items[0] if form == "scalar" else _fold_valid_items(value, add, limit)
        out[tid * 2] = ns.warp_reduce(partial, add, width=4, valid_items=limit)

    return apply


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("form", ["scalar", "vector", "tuple"])
@pytest.mark.parametrize("valid", [0, 1, 3, 4])
@pytest.mark.parametrize("runtime", [False, True], ids=["static", "runtime"])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_warp_reduce_skips_invalid_operators(universal, form, valid, runtime, default_device):
    with torch.device(default_device):
        block = 64
        count = 1 if form == "scalar" else 3
        values = torch.arange(1 + block * count, dtype=torch.int32, device="cuda")
        values[0] = valid
        actual = run_kernel(_counted_reduce(universal, form, valid, runtime), values, block * 2, block)
        actual = actual.reshape(-1, 4, 2)
        # Invalid lanes must never call the operator, even for their local items.
        assert torch.count_nonzero(actual[:, valid:, 1]).item() == 0
        if valid:
            expected = values[1:].cpu().reshape(-1, 4, count)[:, :valid].sum(dim=(1, 2)).int()
            torch.testing.assert_close(actual[:, 0, 0], expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("form", ["scalar", "vector", "tuple"])
def test_warp_reduce_valid_compile(monkeypatch, arch, universal, form):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    apply = _counted_reduce(universal, form, 3, True)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    values = torch.empty(193, dtype=torch.int32, device="cpu")
    out = torch.empty(128, dtype=torch.int32, device="cpu")
    launch(values, out)
