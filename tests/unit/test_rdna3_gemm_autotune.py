#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU-free tests for the RDNA3 GEMM autotune wrapper.

The wrapper's job is to expose tile selection through the shared autotuner
without a call having to opt into benchmarking, so the properties worth pinning
are about agreement rather than performance:

  * with nothing configured, the tuner's ``default`` is exactly ``pick_tile`` —
    a call that never benchmarks must land on the heuristic's tile;
  * the search space only contains tiles the shape can actually build, and it
    always contains the default, so a search can confirm the heuristic and can
    never be forced to pick something that fails at build time;
  * ``feasible_tiles`` reports the same tiles, in the same order, that
    ``pick_tile`` chooses from.

Device-side coverage is ``tests/kernels/test_rdna_gemm.py``.
"""

import pytest

from kernels.gemm.rdna3_f16_gemm_autotune import (
    _TILE_FIELDS,
    _default_config,
    _ladder_for,
    _search_configs,
    _tile_workgroups,
    feasible_tiles,
    pick_tile,
)

pytestmark = pytest.mark.l0_backend_agnostic

SHAPES = [
    (M, N, K)
    for M in (256, 512, 768, 1024, 1536, 2048, 4096)
    for N in (256, 512, 1024, 2048)
    for K in (512, 1024, 2048, 4096)
]

_shape_id = lambda shape: "x".join(map(str, shape))  # noqa: E731


def _tile_of(config):
    return tuple(config.kwargs[field] for field in _TILE_FIELDS)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_default_config_is_the_heuristic_tile(shape):
    """An untuned call must be indistinguishable from calling the kernel directly."""
    M, N, K = shape
    assert _tile_of(_default_config(M=M, N=N, K=K)) == pick_tile(*shape)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_search_space_only_holds_buildable_tiles(shape):
    """Every candidate must survive ``_tile_workgroups``.

    A candidate the shape cannot use does not merely lose the benchmark: it
    raises out of ``create_wmma_gemm_module``, so the search would spend its
    time measuring build failures.
    """
    M, N, K = shape
    if not feasible_tiles(*shape):
        pytest.skip("no tile in the ladder fits this shape")
    for config in _search_configs(M=M, N=N, K=K):
        assert _tile_workgroups(*shape, _tile_of(config)) is not None


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_default_is_reachable_by_the_search(shape):
    """The search must be able to return the heuristic's own answer."""
    M, N, K = shape
    if not feasible_tiles(*shape):
        pytest.skip("no tile in the ladder fits this shape")
    tiles = [_tile_of(config) for config in _search_configs(M=M, N=N, K=K)]
    assert pick_tile(*shape) in tiles


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_feasible_tiles_is_the_ladder_filtered_in_order(shape):
    """``pick_tile`` walks this list, so it has to stay a subsequence of the ladder."""
    ladder = _ladder_for(shape[2])
    reported = feasible_tiles(*shape)

    assert [tile for tile, _ in reported] == [cfg for cfg in ladder if _tile_workgroups(*shape, cfg) is not None]
    for tile, workgroups in reported:
        assert workgroups == _tile_workgroups(*shape, tile)
