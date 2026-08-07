#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU-free tests for the RDNA3 GEMM workgroup swizzle.

``_group_width`` / ``_swizzle_tile_id`` map a linear workgroup id onto the tile
grid, walking down M before stepping in N so concurrent workgroups share B tiles
in L2. The mapping must be a bijection onto the grid. It was not, for any grid_m
that the grouping cap did not divide: the last group ran off the end, bid_m
exceeded grid_m, and the kernel wrote past C.

All plain integer arithmetic that runs before a kernel is built, so no GPU is
needed. The device-side counterpart is
``tests/kernels/test_rdna_gemm.py::test_f16_gemm_grid_m_not_a_multiple_of_the_group_width``.
"""

import pytest

from kernels.gemm.rdna3_f16_gemm import _group_width, _swizzle_tile_id

pytestmark = pytest.mark.l0_backend_agnostic

DEFAULT_GROUP_M = 8  # create_wmma_gemm_module's default grouping cap


def _swizzled_tiles(grid_m, grid_n, group_m=DEFAULT_GROUP_M):
    width = _group_width(grid_m, group_m)
    return [_swizzle_tile_id(pid, grid_n, width) for pid in range(grid_m * grid_n)]


@pytest.mark.parametrize(
    "grid_m, expected",
    [
        (1, 1),
        (3, 3),
        (5, 5),
        (8, 8),
        (12, 6),   # 1536 / 128 — used to take 8 and address tiles 12..15
        (16, 8),
        (20, 5),   # 2560 / 128 — used to take 8 and address tiles 20..23
        (32, 8),
    ],
)
def test_group_width_is_the_largest_divisor_within_the_cap(grid_m, expected):
    assert _group_width(grid_m, DEFAULT_GROUP_M) == expected


@pytest.mark.parametrize("group_m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("grid_m", range(1, 65))
def test_group_width_always_divides_the_grid(grid_m, group_m):
    width = _group_width(grid_m, group_m)
    assert grid_m % width == 0
    assert 1 <= width <= min(group_m, grid_m)


@pytest.mark.parametrize(
    "grid_m, grid_n",
    [
        (1, 1), (1, 8), (8, 1),
        (12, 12),   # 1536x1536 at 128x128: the grid that faulted
        (20, 20),   # 2560x2560 at 128x128: likewise
        (3, 7), (5, 4), (7, 3), (16, 16), (12, 5), (20, 3),
    ],
)
def test_swizzle_covers_every_tile_exactly_once(grid_m, grid_n):
    """The swizzle must be a bijection onto the grid.

    With a grouping width that does not divide grid_m the last group runs off the
    end: bid_m exceeds grid_m, the kernel writes past C, and the launch faults.
    """
    mapped = _swizzled_tiles(grid_m, grid_n)

    assert all(0 <= m < grid_m and 0 <= n < grid_n for m, n in mapped)
    assert set(mapped) == {(m, n) for m in range(grid_m) for n in range(grid_n)}
