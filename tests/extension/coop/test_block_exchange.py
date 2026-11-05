# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block exchange layouts, guarded scatter, records, and target compilation."""

import math

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, matrix_cases, run_tile, warp_indices
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.expr as fx
from flydsl.extension.coop.block.exchange import BlockExchange, BlockExchangeAlgorithm

try:
    import torch
except ImportError:
    torch = None
import coop_block_utils as checks


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,algorithm,method",
    matrix_cases(
        [64, 128, (4, 4, 12)],
        [1, 3, 4],
        list(BlockExchangeAlgorithm),
        ["blocked_to_striped", "striped_to_blocked", "blocked_to_warp_striped", "warp_striped_to_blocked"],
    ),
)
def test_block_exchange_tail(block, count, algorithm, method):
    threads = block if isinstance(block, int) else math.prod(block)
    values = torch.arange(threads * count, dtype=torch.int32, device="cuda")

    def transform(value):
        exchange = BlockExchange[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        return getattr(exchange, method)(value, storage=storage)

    actual = run_tile(transform, values, block, count)[0]
    width = min(fx.num_warp_threads(), 1 << (threads.bit_length() - 1)) if "warp" in method else threads
    indices = warp_indices(threads, count, width, inverse=method in ("striped_to_blocked", "warp_striped_to_blocked"))
    torch.testing.assert_close(actual, values.cpu()[indices])


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("algorithm,flagged", matrix_cases(list(BlockExchangeAlgorithm), [False, True]))
def test_guarded_scatter(algorithm, flagged):
    block, count = 128, 3
    size = block * count
    values = torch.arange(size, device="cuda", dtype=torch.int32)

    def transform(value):
        exchange = BlockExchange[fx.Int32, block, count, algorithm]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        ranks = size - 1 - value
        valid = value % 3 != 1
        if flagged:
            return exchange.scatter_to_striped_flagged(value, ranks, valid, storage=storage, default=-7)
        ranks = valid.select(ranks, fx.Vector.filled(count, -1, fx.Int32))
        return exchange.scatter_to_striped_guarded(value, ranks, storage=storage, default=-7)

    actual = run_tile(transform, values, block, count)[0]
    host = values.cpu()
    expected = torch.where(host % 3 != 1, host, -7).flip(0).reshape(count, block).T.reshape(-1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,policy,count",
    [
        (case, policy, count)
        for (case, policy), count in matrix_cases(
            [
                (case, policy.name)
                for policy in BlockExchangeAlgorithm
                for case in [
                    "blocked_to_striped",
                    "striped_to_blocked",
                    "blocked_to_warp_striped",
                    "warp_striped_to_blocked",
                    "scatter_to_striped_guarded",
                    "scatter_to_striped_flagged",
                ]
            ],
            [3, 4],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_record_movement(case, policy, count):
    checks.check_record_movement(case, policy, count)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case,policy",
    [
        (arch, case, policy)
        for arch, (case, policy) in matrix_cases(
            ARCHES,
            [
                (case, policy.name)
                for policy in BlockExchangeAlgorithm
                for case in [
                    "blocked_to_striped",
                    "striped_to_blocked",
                    "blocked_to_warp_striped",
                    "warp_striped_to_blocked",
                    "scatter_to_striped_guarded",
                    "scatter_to_striped_flagged",
                ]
            ],
        )
    ],
)
def test_compile_record_movement(monkeypatch, arch, case, policy):
    checks.check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(
        ARCHES,
        [
            "block_exchange_blocked_to_striped",
            "block_exchange_striped_to_blocked",
            "block_exchange_blocked_to_warp_striped",
            "block_exchange_warp_striped_to_blocked",
            "block_exchange_scatter_to_blocked",
            "block_exchange_scatter_to_striped_guarded",
            "block_exchange_scatter_to_striped_flagged",
        ],
    ),
)
def test_compile_movement_policies(monkeypatch, arch, case):
    checks.check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,case",
    matrix_cases(
        DTYPES,
        [1, 9],
        [
            ("BlockExchange", "blocked_to_striped"),
            ("BlockExchange", "striped_to_blocked"),
            ("BlockExchange", "blocked_to_warp_striped"),
            ("BlockExchange", "warp_striped_to_blocked"),
            ("BlockExchange", "scatter_to_blocked"),
            ("BlockExchange", "scatter_to_striped"),
        ],
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_dtypes(entry, count, case):
    checks.check_block_case(entry, 128, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,case",
    matrix_cases(
        BLOCK_THREADS,
        [1, 9],
        [
            ("BlockExchange", "blocked_to_striped"),
            ("BlockExchange", "striped_to_blocked"),
            ("BlockExchange", "blocked_to_warp_striped"),
            ("BlockExchange", "warp_striped_to_blocked"),
            ("BlockExchange", "scatter_to_blocked"),
            ("BlockExchange", "scatter_to_striped"),
        ],
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_shapes(block, count, case):
    checks.check_block_case((fx.Int32, "torch.int32"), block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case",
    [
        ("BlockExchange", "blocked_to_striped"),
        ("BlockExchange", "striped_to_blocked"),
        ("BlockExchange", "blocked_to_warp_striped"),
        ("BlockExchange", "warp_striped_to_blocked"),
        ("BlockExchange", "scatter_to_blocked"),
        ("BlockExchange", "scatter_to_striped"),
    ],
    ids=lambda c: c[1],
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_three_dimensional_block(case):
    checks.check_block_case((fx.Float64, "torch.float64"), (8, 4, 4), 3, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "form,case,algorithm",
    matrix_cases(
        ["list", "tuple", "vector"],
        ["scatter_blocked", "scatter_guarded", "scatter_flagged"],
        list(fx.coop.BlockExchangeAlgorithm),
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_secondary_item_containers(form, case, algorithm):
    checks.check_secondary_item_containers(form, case, algorithm)
