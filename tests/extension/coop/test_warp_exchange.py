# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp exchange layouts, independent groups, policies, and record fields."""

import pytest
from coop_test_utils import ARCHES, run_tile, warp_indices, warp_storage
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop.warp.exchange import WarpExchangeAlgorithm

try:
    import torch
except ImportError:
    torch = None
import coop_warp_utils as checks
from coop_common import DTYPES, WARP_SIZE, WARP_WIDTHS, dtype_id


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
        P = fx.coop.WarpExchange[value.dtype, width, count]
        storage = warp_storage(P)
        fn = P.striped_to_blocked if inverse else P.blocked_to_striped
        return fn(value, storage=storage)

    result = run_tile(transform, values, threads, count)[0]
    shape = (-1, group, count) if inverse else (-1, count, group)
    expected = values.cpu().reshape(shape).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm", list(WarpExchangeAlgorithm))
@pytest.mark.parametrize("width", [width for width in (1, 8, 32, 64) if width <= WARP_SIZE])
@pytest.mark.parametrize("method", ["blocked_to_striped", "striped_to_blocked", "scatter_to_striped"])
@pytest.mark.usefixtures("warp_default_device")
def test_warp_exchange_policies(algorithm, width, method):
    if algorithm is WarpExchangeAlgorithm.SHUFFLE and method == "scatter_to_striped":
        with pytest.raises(ValueError, match="Shared"):
            fx.coop.WarpExchange[fx.Int32, width, width, algorithm].scatter_to_striped(None, None)
        return
    block, count = 128, width if algorithm is WarpExchangeAlgorithm.SHUFFLE else 3
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        P = fx.coop.WarpExchange[fx.Int32, width, count, algorithm]
        kwargs = dict(storage=warp_storage(P))
        if method == "scatter_to_striped":
            ranks = width * count - 1 - (value % (width * count))
            return P.scatter_to_striped(value, ranks, **kwargs)
        fn = P.blocked_to_striped if method == "blocked_to_striped" else P.striped_to_blocked
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
@pytest.mark.parametrize("width", [width for width in (8, 64) if width <= WARP_SIZE])
def test_warp_smem_independent_branches(width):
    block, count = 128, 3
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    @flyc.jit
    def exchange_branch(value, storage):
        result = value
        if (fx.thread_idx.x // width) % 2 == 0:
            result = fx.coop.WarpExchange[fx.Int32, width, count].blocked_to_striped(value, storage=storage)
        else:
            result = fx.coop.WarpExchange[fx.Int32, width, count].striped_to_blocked(value, storage=storage)
        return result

    def transform(value):
        storage = warp_storage(fx.coop.WarpExchange[fx.Int32, width, count])
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
        if policy is WarpExchangeAlgorithm.SHARED or case != "warp_scatter_to_striped"
    ],
)
@pytest.mark.usefixtures("warp_default_device")
def test_record_movement(case, policy, count):
    checks.check_record_movement(case, policy, count)


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
        if policy is WarpExchangeAlgorithm.SHARED or case != "warp_scatter_to_striped"
    ],
)
def test_compile_record_movement(monkeypatch, arch, case, policy):
    checks.check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_smem_blocked", "warp_smem_striped", "warp_smem_scatter"])
def test_compile_movement_policies(monkeypatch, arch, case):
    checks.check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", (1, *WARP_WIDTHS))
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("primitive", ["warp_blocked_to_striped", "warp_striped_to_blocked"])
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    checks.check_warp_dtypes_and_widths(entry, width, count, primitive)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_blocked_to_striped", "warp_striped_to_blocked"])
def test_compile_family(monkeypatch, arch, case):
    checks.check_compile_family(monkeypatch, arch, case)
