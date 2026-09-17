# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp exchange layouts, independent groups, policies, and record fields."""

import pytest
from coop_test_utils import ARCHES, run_tile, warp_indices
from coop_test_utils import warp_default_device as warp_default_device
from coop_warp_movement_utils import check_compile_movement_policies
from coop_warp_record_movement_utils import check_compile_record_movement, check_record_movement

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop.warp.exchange import (
    WarpExchangeAlgorithm,
    warp_blocked_to_striped,
    warp_exchange_storage,
    warp_scatter_to_striped,
    warp_striped_to_blocked,
)

try:
    import torch
except ImportError:
    torch = None
import coop_warp_primitive_utils as primitive_checks
from coop_common import DTYPES, WARP_WIDTHS, dtype_id


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [1, 2, 8, 32, None])
@pytest.mark.parametrize("count", [1, 3, 8])
@pytest.mark.parametrize("inverse", [False, True])
def test_warp_exchange(width, count, inverse):
    threads = 128
    group = width or fx.num_warp_threads()
    values = torch.arange(threads * count, dtype=torch.int32, device="cuda")

    def transform(value):
        fn = fx.coop.warp_striped_to_blocked if inverse else fx.coop.warp_blocked_to_striped
        storage = fx.SharedAllocator().allocate(warp_exchange_storage(value.dtype, count)).peek()
        return fn(value, width=width, storage=storage)

    result = run_tile(transform, values, threads, count)[0]
    shape = (-1, group, count) if inverse else (-1, count, group)
    expected = values.cpu().reshape(shape).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", list(WarpExchangeAlgorithm))
@pytest.mark.parametrize("width", [1, 8, 32, 64])
@pytest.mark.parametrize("method", ["blocked_to_striped", "striped_to_blocked", "scatter_to_striped"])
@pytest.mark.usefixtures("warp_default_device")
def test_warp_exchange_policies(algorithm, width, method):
    if algorithm is WarpExchangeAlgorithm.SHUFFLE and method == "scatter_to_striped":
        with pytest.raises(ValueError, match="SMEM"):
            warp_scatter_to_striped(None, None, width=width, algorithm=algorithm)
        return
    block, count = 128, width if algorithm is WarpExchangeAlgorithm.SHUFFLE else 3
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        storage = fx.SharedAllocator().allocate(warp_exchange_storage(fx.Int32, count)).peek()
        kwargs = dict(width=width, algorithm=algorithm, storage=storage)
        if method == "scatter_to_striped":
            ranks = width * count - 1 - (value % (width * count))
            return warp_scatter_to_striped(value, ranks, **kwargs)
        fn = warp_blocked_to_striped if method == "blocked_to_striped" else warp_striped_to_blocked
        return fn(value, **kwargs)

    actual = run_tile(transform, values, block, count)[0]
    host = values.cpu()
    if method == "scatter_to_striped":
        host = host.reshape(-1, width * count).flip(1).reshape(-1)
    indices = warp_indices(block, count, width, inverse=method == "striped_to_blocked")
    torch.testing.assert_close(actual, host[indices])


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [8, 64])
def test_warp_smem_independent_branches(width):
    block, count = 128, 3
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    @flyc.jit
    def exchange_branch(value, storage):
        result = value
        if (fx.thread_idx.x // width) % 2 == 0:
            result = warp_blocked_to_striped(value, width=width, algorithm=WarpExchangeAlgorithm.SMEM, storage=storage)
        else:
            result = warp_striped_to_blocked(value, width=width, algorithm=WarpExchangeAlgorithm.SMEM, storage=storage)
        return result

    def transform(value):
        storage = fx.SharedAllocator().allocate(warp_exchange_storage(fx.Int32, count)).peek()
        return exchange_branch(value, storage)

    actual = run_tile(transform, values, block, count)[0]
    blocked = warp_indices(block, count, width)
    striped = warp_indices(block, count, width, inverse=True)
    even = (torch.arange(block * count, device="cpu") // (width * count)) % 2 == 0
    torch.testing.assert_close(actual, values.cpu()[torch.where(even, blocked, striped)])


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,policy,count",
    [
        (case, policy.name, count)
        for policy in WarpExchangeAlgorithm
        for case in ["warp_blocked_to_striped", "warp_striped_to_blocked", "warp_scatter_to_striped"]
        for count in ([8] if policy is WarpExchangeAlgorithm.SHUFFLE else [3, 4])
        if policy is WarpExchangeAlgorithm.SMEM or case != "warp_scatter_to_striped"
    ],
)
@pytest.mark.usefixtures("warp_default_device")
def test_record_movement(case, policy, count):
    check_record_movement(case, policy, count)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "case,policy",
    [
        (case, policy.name)
        for policy in WarpExchangeAlgorithm
        for case in ["warp_blocked_to_striped", "warp_striped_to_blocked", "warp_scatter_to_striped"]
        if policy is WarpExchangeAlgorithm.SMEM or case != "warp_scatter_to_striped"
    ],
)
def test_compile_record_movement(monkeypatch, arch, case, policy):
    check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_smem_blocked", "warp_smem_striped", "warp_smem_scatter"])
def test_compile_movement_policies(monkeypatch, arch, case):
    check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", (1, *WARP_WIDTHS))
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("primitive", ["warp_blocked_to_striped", "warp_striped_to_blocked"])
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    primitive_checks.check_warp_dtypes_and_widths(entry, width, count, primitive)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_blocked_to_striped", "warp_striped_to_blocked"])
def test_compile_family(monkeypatch, arch, case):
    primitive_checks.check_compile_family(monkeypatch, arch, case)
