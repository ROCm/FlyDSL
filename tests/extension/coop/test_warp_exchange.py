# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Warp exchange layouts, independent groups, policies, and record fields."""

import pytest
from coop_test_utils import ARCHES, full_matrix_enabled, matrix_cases, run_tile, warp_indices, warp_storage
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
from coop_common import DTYPES, WARP_SIZE, WARP_WIDTHS

ALL_RECORD_MOVEMENT_CASES = [
    (case, policy.name, count)
    for policy in WarpExchangeAlgorithm
    for case in ["warp_blocked_to_striped", "warp_striped_to_blocked", "warp_scatter_to_striped"]
    for count in ([8] if policy is WarpExchangeAlgorithm.SHUFFLE else [3, 4])
    if policy is WarpExchangeAlgorithm.SHARED or case != "warp_scatter_to_striped"
]
MINIMAL_RECORD_MOVEMENT_CASES = [
    ("warp_blocked_to_striped", WarpExchangeAlgorithm.SHARED.name, 3),
    ("warp_striped_to_blocked", WarpExchangeAlgorithm.SHARED.name, 4),
    ("warp_scatter_to_striped", WarpExchangeAlgorithm.SHARED.name, 3),
    ("warp_blocked_to_striped", WarpExchangeAlgorithm.SHUFFLE.name, 8),
    ("warp_striped_to_blocked", WarpExchangeAlgorithm.SHUFFLE.name, 8),
]
RECORD_MOVEMENT_CASES = ALL_RECORD_MOVEMENT_CASES if full_matrix_enabled() else MINIMAL_RECORD_MOVEMENT_CASES
RECORD_COMPILE_CASES = tuple(dict.fromkeys((case, policy) for case, policy, _ in RECORD_MOVEMENT_CASES))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "width,count,inverse",
    matrix_cases([1, 2, 8, 32, None], [1, 3, 8], [False, True]),
)
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
@pytest.mark.parametrize(
    "algorithm,width,method",
    matrix_cases(
        list(WarpExchangeAlgorithm),
        [width for width in (1, 8, 32, 64) if width <= WARP_SIZE],
        ["blocked_to_striped", "striped_to_blocked", "scatter_to_striped"],
    ),
)
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
@pytest.mark.parametrize("width,count,universal", matrix_cases([1, 8, WARP_SIZE], [1, 3, 4], [False, True]))
def test_scatter_layout_matches_explicit_ranks(width, count, universal):
    block = 128
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        namespace = fx.coop.universal if universal else fx.coop
        P = namespace.WarpExchange[fx.Int32, width, count]
        storage = warp_storage(P)
        # Nested item mode also accepts a flattened item coordinate.
        src = fx.make_layout((width, (1, count)), (count, (0, 1)))
        dst = fx.make_layout((width, count), (1, width))
        from_layout = P.scatter_to_striped(value, src_layout=src, dst_layout=dst, storage=storage)
        fx.barrier()
        ranks = value % (width * count)
        from_ranks = P.scatter_to_striped(value, ranks=ranks, storage=storage)
        return from_layout, from_ranks

    actual, explicit = run_tile(transform, values, block, count, outputs=2)
    expected = values.cpu()[warp_indices(block, count, width)]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, explicit)


def mfma_scatter(value, width, round_trip=False):
    P = fx.coop.WarpExchange[value.dtype, width, 4]
    # Both layouts map (lane, item) to row-major logical matrix indices.
    src = fx.make_layout(((16, width // 16), 4), ((1, 64), 16))
    dst = fx.make_layout((width, 4), (4, 1))
    storage = warp_storage(P)
    output = P.scatter_to_striped(value, src_layout=src, dst_layout=dst, storage=storage)
    if round_trip:
        fx.barrier()
        restored = P.scatter_to_striped(output, src_layout=dst, dst_layout=src, storage=storage)
        return output, restored
    return output


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", [width for width in (16, 32, 64) if width <= WARP_SIZE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16] if torch is not None else [])
def test_scatter_mfma_layout(width, dtype):
    block, count = 128, 4
    # Independent logical groups own matrices with distinct values. Each lane
    # initially holds four rows of one column; output must be row-major chunks.
    lane = torch.arange(block, device="cpu") % width
    row = 4 * (lane // 16)
    group = torch.arange(block, device="cpu") // width
    logical = group[:, None] * width * count + (row[:, None] + torch.arange(count)) * 16 + lane[:, None] % 16
    values = logical.flatten().to(device="cuda", dtype=dtype)
    actual, restored = run_tile(
        lambda value: mfma_scatter(value, width, round_trip=True), values, block, count, outputs=2
    )
    torch.testing.assert_close(actual, torch.arange(block * count, dtype=torch.int32).to(dtype))
    torch.testing.assert_close(restored, values.cpu())


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("case", ["source_only", "destination_only", "ranks_and_layouts"])
def test_scatter_layout_argument_conflicts(ctx, insert_point, case):
    P = fx.coop.WarpExchange[fx.Int32, 8, 4]
    value = fx.Vector.filled(4, 0, fx.Int32)
    layout = fx.make_layout((8, 4), (4, 1))
    kwargs = dict(src_layout=layout, dst_layout=layout)
    if case == "source_only":
        del kwargs["dst_layout"]
    elif case == "destination_only":
        del kwargs["src_layout"]
    else:
        kwargs["ranks"] = value
    with pytest.raises(ValueError, match="not both" if case == "ranks_and_layouts" else "provided together"):
        P.scatter_to_striped(value, storage=object(), **kwargs)


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
@pytest.mark.parametrize("case,policy,count", RECORD_MOVEMENT_CASES)
@pytest.mark.usefixtures("warp_default_device")
def test_record_movement(case, policy, count):
    checks.check_record_movement(case, policy, count)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case,policy",
    [(arch, case, policy) for arch, (case, policy) in matrix_cases(ARCHES, RECORD_COMPILE_CASES)],
)
def test_compile_record_movement(monkeypatch, arch, case, policy):
    checks.check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(ARCHES, ["warp_smem_blocked", "warp_smem_striped", "warp_smem_scatter"]),
)
def test_compile_movement_policies(monkeypatch, arch, case):
    checks.check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,width,count,primitive",
    matrix_cases(
        DTYPES,
        (1, *WARP_WIDTHS),
        [1, 9],
        ["warp_blocked_to_striped", "warp_striped_to_blocked"],
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_warp_dtypes_and_widths(entry, width, count, primitive):
    checks.check_warp_dtypes_and_widths(entry, width, count, primitive)
