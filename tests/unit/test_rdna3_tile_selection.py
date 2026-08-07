#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU-free tests for the RDNA3 GEMM host-side tile selection.

``pick_tile`` decides which tile a shape gets. Beyond the measured expectations,
two properties make it safe to enable by default: it never returns a tile the
shape cannot build, and it never moves a shape off 128x128x32 while that tile
still has a deep enough grid to win.

All plain integer arithmetic that runs before a kernel is built, so no GPU is
needed. The swizzle the chosen tile feeds into is covered by
``test_rdna3_grid_swizzle.py``, and the device-side counterpart is
``tests/kernels/test_rdna_gemm.py``.
"""

import pytest

from kernels.gemm.rdna3_f16_gemm import _group_width, _swizzle_tile_id
from kernels.gemm.rdna3_f16_gemm_autotune import (
    NUM_CU,
    TILE_32x32x64,
    TILE_32x64x64,
    TILE_64x64x64,
    TILE_128x64x32,
    TILE_128x128x32,
    _ladder_for,
    _tile_workgroups,
    feasible_tiles,
    pick_tile,
)

pytestmark = pytest.mark.l0_backend_agnostic

DEFAULT_TILE = TILE_128x128x32  # the tile used before selection existed
DEFAULT_GROUP_M = 8  # create_wmma_gemm_module's default grouping cap

# Every shape the tile heuristic was measured on, with the tile it is expected to
# pick, so that a heuristic edit has to state which shapes it moves. Each of
# these was timed against the whole ladder; the expectation is the fastest tile
# except where noted.
MEASURED_SHAPES = [
    pytest.param((256, 256, 4096), TILE_32x32x64, id="256x256x4096-32x32x64"),
    pytest.param((256, 256, 1024), TILE_64x64x64, id="256x256x1024-64x64x64"),
    pytest.param((384, 384, 2048), TILE_64x64x64, id="384x384x2048-64x64x64"),
    pytest.param((512, 512, 1024), TILE_64x64x64, id="512x512x1024-64x64x64"),
    pytest.param((512, 512, 4096), TILE_64x64x64, id="512x512x4096-64x64x64"),
    pytest.param((256, 1024, 4096), TILE_64x64x64, id="256x1024x4096-64x64x64"),
    pytest.param((768, 768, 2048), TILE_64x64x64, id="768x768x2048-64x64x64"),
    pytest.param((1024, 1024, 512), TILE_128x64x32, id="1024x1024x512-128x64x32"),
    pytest.param((1024, 1024, 1024), TILE_128x64x32, id="1024x1024x1024-128x64x32"),
    pytest.param((1024, 1024, 4096), TILE_64x64x64, id="1024x1024x4096-64x64x64"),
    # 128x128x32 is 8.1% faster here, the worst case the heuristic accepts.
    pytest.param((1152, 1152, 1024), TILE_64x64x64, id="1152x1152x1024-64x64x64"),
    pytest.param((1536, 1536, 1024), TILE_64x64x64, id="1536x1536x1024-64x64x64"),
    pytest.param((1792, 1792, 1024), TILE_64x64x64, id="1792x1792x1024-64x64x64"),
    pytest.param((2048, 2048, 512), DEFAULT_TILE, id="2048x2048x512-128x128x32"),
    pytest.param((2048, 2048, 2048), DEFAULT_TILE, id="2048x2048x2048-128x128x32"),
    pytest.param((3072, 3072, 1024), DEFAULT_TILE, id="3072x3072x1024-128x128x32"),
    pytest.param((4096, 4096, 4096), DEFAULT_TILE, id="4096x4096x4096-128x128x32"),
]

MEASURED_ONLY_SHAPES = [p.values[0] for p in MEASURED_SHAPES]

# A spread for the property tests: square and skewed, K on both sides of the
# ladder split, sizes from "cannot fill the machine" up to "fills it easily".
PROPERTY_SHAPES = [
    (M, N, K)
    for M in (256, 512, 768, 1024, 1536, 2048, 4096)
    for N in (256, 512, 1024, 2048)
    for K in (512, 1024, 2048, 4096)
]

_shape_id = lambda shape: "x".join(map(str, shape))  # noqa: E731


def _block_shape(cfg):
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg
    return 16 * reg_m * waves_m, 16 * reg_n * waves_n, 16 * reg_k


def _tile_grid(shape):
    """(grid_m, grid_n) the kernel would launch for this shape."""
    M, N, _ = shape
    block_m, block_n, _ = _block_shape(pick_tile(*shape))
    return M // block_m, N // block_n


@pytest.mark.parametrize("shape, expected", MEASURED_SHAPES)
def test_pick_tile_matches_measured_shapes(shape, expected):
    """The tile picked for each benchmarked shape is the one that was measured."""
    assert pick_tile(*shape) == expected


@pytest.mark.parametrize("shape", PROPERTY_SHAPES, ids=_shape_id)
def test_picked_tile_is_usable_by_the_shape(shape):
    """Whenever the ladder has a usable tile, the picked one is usable.

    ``create_wmma_gemm_module`` asserts the divisibility and vectorization rules
    that ``_tile_workgroups`` screens for, so returning an unusable tile turns
    into an assertion failure at build time.
    """
    if all(_tile_workgroups(*shape, cfg) is None for cfg in _ladder_for(shape[2])):
        pytest.skip("no tile in the ladder fits this shape")
    assert _tile_workgroups(*shape, pick_tile(*shape)) is not None


@pytest.mark.parametrize("shape", PROPERTY_SHAPES, ids=_shape_id)
def test_deep_grids_keep_the_original_tile(shape):
    """A shape whose 128x128x32 grid is deep enough must not be moved off it.

    This is what makes the heuristic safe to turn on by default: the large
    shapes take the same path as before selection existed, so they cannot
    regress. Merely covering the machine is not enough — 128x128x32 lost by 19%
    at 1792x1792x1024 and 37% at 1664x1664x1024, both well past one workgroup
    per CU — so the bar is the measured one.
    """
    wgs = _tile_workgroups(*shape, DEFAULT_TILE)
    if wgs is None or wgs < 2.5 * NUM_CU:
        pytest.skip("128x128x32's grid is not deep enough for this shape")
    assert pick_tile(*shape) == DEFAULT_TILE


@pytest.mark.parametrize("shape", PROPERTY_SHAPES, ids=_shape_id)
def test_the_narrowest_tile_needs_a_starved_grid(shape):
    """32x32x64 trades compute intensity for parallelism, and loses when it does not need to.

    It gave up 12-23% against 64x64x64 wherever 64x64x64 had the workgroups to
    keep the machine busy, so it must be unreachable unless 64x64x64 either does
    not fit or cannot fill the machine.
    """
    if pick_tile(*shape) != TILE_32x32x64:
        pytest.skip("this shape does not take the narrowest tile")
    feasible = dict(feasible_tiles(*shape))
    assert TILE_64x64x64 not in feasible or feasible[TILE_64x64x64] < NUM_CU / 4


@pytest.mark.parametrize("shape", PROPERTY_SHAPES, ids=_shape_id)
def test_32x64x64_is_only_ever_a_fallback(shape):
    """32x64x64 was not the fastest tile on any of the 27 shapes swept.

    The old heuristic chose it for five of them, each time losing to 64x64x64.
    It stays on the ladder only to cover shapes that nothing else divides.
    """
    if pick_tile(*shape) != TILE_32x64x64:
        pytest.skip("this shape does not take 32x64x64")
    assert dict(feasible_tiles(*shape)).keys() == {TILE_32x64x64}


@pytest.mark.parametrize("shape", MEASURED_ONLY_SHAPES, ids=_shape_id)
def test_picked_tile_and_default_grouping_stay_inside_the_grid(shape):
    """Selection and swizzle have to agree, not just work in isolation.

    The fault needs both halves to line up: the tile decides grid_m, and only then
    does the grouping width become a divisor question.
    """
    grid_m, grid_n = _tile_grid(shape)
    width = _group_width(grid_m, DEFAULT_GROUP_M)
    mapped = [_swizzle_tile_id(pid, grid_n, width) for pid in range(grid_m * grid_n)]

    assert set(mapped) == {(m, n) for m in range(grid_m) for n in range(grid_n)}
