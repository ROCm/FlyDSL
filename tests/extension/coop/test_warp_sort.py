# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""``warp_sort`` results: a blocked sort, stable on request.

The sorted tile and, with ``stable=True``, the payload order are compared with
a host sort, through ``fx.coop`` and ``fx.coop.universal``. The shapes, types
and comparators that ``test_warp_bitonic_sort.py`` and
``test_warp_merge_sort.py`` run through the two sorts are not repeated here.

Which sort a call runs is checked per target in ``test_target_compilation.py``
(``test_warp_sort_choice``): a wrong choice still sorts correctly.
"""

from __future__ import annotations

import coop_warp_utils as checks
import pytest
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import matrix_cases, run_tile

import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal
requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize(
    "block,count,descending",
    [
        (block, count, descending)
        for (block, count), descending in matrix_cases([(1, 1), (8, 1), (8, 3), (64, 4), (128, 1)], [False, True])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sort(block, count, descending):
    checks.check_sort("warp_sort", block, count, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("count,width,descending", matrix_cases([1, 3], [1, 2, 32, None], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_sort_pairs(count, width, descending):
    checks.check_sort_pairs("warp_sort", count, width, descending)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("universal,count,width", matrix_cases([False, True], [1, 3], [16, 2, None], independent=True))
@pytest.mark.usefixtures("coop_default_device")
def test_stable_pairs_keep_the_order_of_equal_keys(universal, count, width):
    ns = portable if universal else fx.coop
    block = 128
    lanes = width or fx.num_warp_threads()
    group = lanes * count
    indices = torch.arange(block * count, dtype=torch.int32, device="cuda")

    def transform(value):
        return ns.warp_sort(value * 7 % 5, value, width=width, stable=True)

    keys, payload = run_tile(transform, indices, block, count, outputs=2)
    expected, permutation = (indices.cpu() * 7 % 5).reshape(-1, group).sort(dim=1, stable=True)
    torch.testing.assert_close(keys, expected.reshape(-1))
    offsets = torch.arange(block // lanes, device="cpu")[:, None] * group
    torch.testing.assert_close(payload.long(), (permutation + offsets).reshape(-1))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize(
    "universal,stable,count,width", matrix_cases([False, True], [False, True], [1, 2], [4, None], independent=True)
)
@pytest.mark.usefixtures("coop_default_device")
def test_integer_keys_alone_sort_the_same_with_and_without_stable(universal, stable, count, width):
    ns = portable if universal else fx.coop
    block = 128
    group = (width or fx.num_warp_threads()) * count
    torch.manual_seed(5)
    values = torch.randint(-6, 7, (block * count,), dtype=torch.int32, device="cuda")
    values[::7], values[3::11] = torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max

    result = run_tile(lambda value: ns.warp_sort(value, width=width, stable=stable), values, block, count)[0]
    torch.testing.assert_close(result, values.cpu().reshape(-1, group).sort(dim=1).values.reshape(-1))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("universal,runtime,pairs", matrix_cases([False, True], [False, True], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_valid_prefix(universal, runtime, pairs):
    """The first ``valid_items`` slots hold the sorted valid keys, each with its own payload."""
    ns = portable if universal else fx.coop
    block, width, valid = 128, 16, 5
    torch.manual_seed(7)
    values = torch.randint(-6, 7, (block,), dtype=torch.int32, device="cuda")

    def transform(value):
        limit = fx.Int32(valid) if runtime else valid
        return ns.warp_sort(value, value * 3 if pairs else None, width=width, valid_items=limit)

    result = run_tile(transform, values, block, 1, outputs=2 if pairs else 1)
    expected = values.cpu().reshape(-1, width)[:, :valid].sort(dim=1).values
    torch.testing.assert_close(result[0].reshape(-1, width)[:, :valid], expected)
    if pairs:
        torch.testing.assert_close(result[1].reshape(-1, width)[:, :valid], expected * 3)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("universal", [False, True])
def test_argument_errors(universal):
    ns = portable if universal else fx.coop
    with pytest.raises(TypeError, match="stable must be a compile-time bool"):
        ns.warp_sort([0], stable=1)
    with pytest.raises(TypeError, match="keys must be a nonempty outer item sequence"):
        ns.warp_sort(0)
    with pytest.raises(ValueError, match="warp_sort width"):
        ns.warp_sort([0], width=3)
