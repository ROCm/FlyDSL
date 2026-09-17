# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Tile and persistent-grid selection for the RDNA3 INT8 WMMA GEMM."""

import functools
import math

import torch

from flydsl.autotune import Config, autotune, do_bench
from kernels.gemm.rdna3_int8_gemm import (
    LDS_BYTES,
    LOAD_VEC,
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    _k_pad,
    create_wmma_int8_gemm_module,
)

# (reg_m, reg_n, reg_k, waves_m, waves_n), named by the resulting block tile.
TILE_256x256x32 = (4, 4, 2, 4, 4)
TILE_128x128x64 = (4, 4, 4, 2, 2)
TILE_128x64x64 = (4, 2, 4, 2, 2)
TILE_64x128x64 = (2, 4, 4, 2, 2)
TILE_64x64x64 = (2, 2, 4, 2, 2)
TILE_32x64x64 = (2, 2, 4, 1, 2)
TILE_32x32x64 = (2, 2, 4, 1, 1)

_DEFAULT_OPTS = {"lds_layout": "pad", "sched_hint": True, "group_m": 8, "stagger": 1}
_TILE_OPTS = {
    TILE_256x256x32: {"lds_layout": "kblock", "sched_hint": True, "group_m": 16, "stagger": 1},
}
_LADDER = [
    TILE_256x256x32,
    TILE_128x128x64,
    TILE_128x64x64,
    TILE_64x128x64,
    TILE_64x64x64,
    TILE_32x64x64,
    TILE_32x32x64,
]
_CONFIG_FIELDS = ("reg_m", "reg_n", "reg_k", "waves_m", "waves_n", "use_persistent", "split_k")
# How far past the heuristic split the search looks; the heuristic measured
# within a few percent of the best split on every shape tried.
_SPLIT_SEARCH_REACH = 4


def tile_opts(tile):
    return _TILE_OPTS.get(tuple(tile), _DEFAULT_OPTS)


@functools.lru_cache(maxsize=None)
def device_cu_count(device=None):
    """Return the active device CU count instead of assuming a gfx1100."""
    if device is None:
        device = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _tile_geometry(tile):
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    threads = waves_m * waves_n * WAVE_SIZE
    return block_m, block_n, block_k, threads


def _lds_bytes(tile):
    block_m, block_n, block_k, _threads = _tile_geometry(tile)
    pad = 0 if tile_opts(tile)["lds_layout"] == "kblock" else _k_pad(block_k)
    return 2 * (block_m + block_n) * (block_k + pad)


def _tile_workgroups(M, N, K, tile):
    """Return the workgroup count, or None when the tile cannot run."""
    block_m, block_n, block_k, threads = _tile_geometry(tile)
    if N % block_n or K % block_k or K // block_k < 2:
        return None
    thrs_k = block_k // LOAD_VEC
    if not thrs_k or threads % thrs_k:
        return None
    thrs_m = threads // thrs_k
    if block_m % thrs_m or block_n % thrs_m:
        return None
    if _lds_bytes(tile) > LDS_BYTES:
        return None
    return -(-M // block_m) * (N // block_n)


def feasible_tiles(M, N, K):
    return [(tile, wgs) for tile in _LADDER if (wgs := _tile_workgroups(M, N, K, tile)) is not None]


def _padded_rows(M, tile):
    """Rows the grid multiplies, counting the padding past M."""
    block_m = _tile_geometry(tile)[0]
    return -(-M // block_m) * block_m


# Ordered large to small: a wider tile reloads less of A and B, a narrower one
# spreads the same output over more workgroups.
_PREFERENCE = (TILE_64x128x64, TILE_64x64x64, TILE_32x64x64, TILE_32x32x64)
# Workgroups per processor before the grid counts as covering the device.
_GRID_FILL = 2


def pick_tile(M, N, K, *, num_cu=None, splittable=True):
    """Choose a no-search default while preserving the tuned kernel defaults."""
    num_cu = device_cu_count() if num_cu is None else int(num_cu)
    feasible = dict(feasible_tiles(M, N, K))
    if not feasible:
        return TILE_64x128x64

    wide_wgs = feasible.get(TILE_128x128x64, 0)
    if wide_wgs >= 40 * num_cu:
        return TILE_128x128x64

    # A tile taller than M pads every workgroup with rows the WMMA multiplies
    # anyway, so a 64-row tile on a 32-row problem throws away half the work.
    fewest_rows = min(_padded_rows(M, tile) for tile in feasible)
    candidates = [tile for tile in _PREFERENCE if tile in feasible and _padded_rows(M, tile) == fewest_rows]
    candidates = candidates or [tile for tile in _PREFERENCE if tile in feasible]
    if not candidates:
        return next(reversed(feasible))

    def grid(tile):
        splits = pick_split_k(M, N, K, tile, num_cu=num_cu) if splittable else 1
        return feasible[tile] * splits

    # Split-K counts towards the fill because it multiplies the same grid.
    for tile in candidates:
        if grid(tile) >= _GRID_FILL * num_cu:
            return tile
    return max(candidates, key=grid)


# ``multi_processor_count`` reports work-group processors on RDNA, and each owns
# two CUs, so a processor hands out twice one workgroup's LDS budget.
_LDS_BYTES_PER_PROCESSOR = 2 * LDS_BYTES


def _residents_per_processor(tile):
    return max(1, min(8, _LDS_BYTES_PER_PROCESSOR // _lds_bytes(tile)))


def _powers_of_two(limit):
    splits = []
    split = 1
    while split <= limit:
        splits.append(split)
        split *= 2
    return splits


def feasible_splits(M, N, K, tile):
    """The ``split_k`` values the kernel accepts for this tile.

    Powers of two that divide the K-tile count and leave each slice the two
    K-tiles its prefetch pipeline needs.
    """
    _bm, _bn, block_k, _threads = _tile_geometry(tile)
    k_tiles = K // block_k
    return [split for split in _powers_of_two(k_tiles // 2) if k_tiles % split == 0]


def pick_split_k(M, N, K, tile, *, num_cu=None):
    """Split K only when the output tiles cannot fill the device.

    Each extra slice re-reads nothing but writes its partial sum into C again,
    so the split has to buy more occupancy than the added output traffic costs.
    """
    num_tiles = _tile_workgroups(M, N, K, tile)
    if not num_tiles:
        return 1
    num_cu = device_cu_count() if num_cu is None else int(num_cu)
    reach = num_cu * _residents_per_processor(tile) // num_tiles
    if reach < 2:
        return 1
    affordable = [
        split for split in feasible_splits(M, N, K, tile) if split <= reach and split * M * N * 4 <= (M + N) * K
    ]
    return max(affordable, default=1)


def persistent_workgroups(M, N, K, tile, *, num_cu=None):
    """Size a persistent grid from the device's per-workgroup LDS residency."""
    num_tiles = _tile_workgroups(M, N, K, tile)
    if not num_tiles:
        return 0
    num_cu = device_cu_count() if num_cu is None else int(num_cu)
    active_wgs = min(num_tiles, num_cu * _residents_per_processor(tile))
    # Below two tiles per workgroup there is no tail left to balance.
    return active_wgs if num_tiles >= 2 * active_wgs else 0


def _config(tile, use_persistent=False, split_k=1):
    return Config(**dict(zip(_CONFIG_FIELDS, (*tile, use_persistent, split_k))))


def _signature(M, N, K, in_dtype, out_dtype, scale_mode, lda, ldb, ldc):
    return (M, N, K, in_dtype, out_dtype, scale_mode, lda, ldb, ldc)


_resolved = {}


@functools.lru_cache(maxsize=None)
def _build(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    scale_mode,
    reg_m,
    reg_n,
    reg_k,
    waves_m,
    waves_n,
    use_persistent,
    split_k,
    lda,
    ldb,
    ldc,
):
    tile = (reg_m, reg_n, reg_k, waves_m, waves_n)
    opts = dict(tile_opts(tile))
    if lda > K or ldb > K:
        opts["stagger"] = 0
    persistent_wgs = persistent_workgroups(M, N, K, tile) if use_persistent else 0
    launch, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_mode=scale_mode,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        persistent_wgs=persistent_wgs,
        split_k=split_k,
        lda=lda,
        ldb=ldb,
        ldc=ldc,
        **opts,
    )
    return launch


def rdna3_int8_gemm_dispatch(
    C,
    A,
    B_T,
    scale_a=None,
    scale_b=None,
    M=None,
    N=None,
    K=None,
    in_dtype="int8",
    out_dtype="i32",
    scale_mode="none",
    reg_m=None,
    reg_n=None,
    reg_k=None,
    waves_m=None,
    waves_n=None,
    use_persistent=False,
    split_k=None,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    splittable = _splittable(out_dtype, scale_mode)
    tile = tuple(
        auto if given is None else given
        for auto, given in zip(pick_tile(M, N, K, splittable=splittable), (reg_m, reg_n, reg_k, waves_m, waves_n))
    )
    if split_k is None:
        split_k = pick_split_k(M, N, K, tile) if splittable else 1
    strides = (A.stride(0), B_T.stride(0), C.stride(0))
    launch = _build(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        scale_mode,
        *tile,
        use_persistent,
        split_k,
        *strides,
    )
    _resolved[_signature(M, N, K, in_dtype, out_dtype, scale_mode, *strides)] = launch
    return launch(C, A, B_T, stream, scale_a, scale_b)


def _splittable(out_dtype, scale_mode):
    """The atomic split-K epilogue only accumulates the exact i32 output."""
    return out_dtype == "i32" and scale_mode == "none"


def _default_config(
    C=None,
    A=None,
    B_T=None,
    scale_a=None,
    scale_b=None,
    M=None,
    N=None,
    K=None,
    out_dtype="i32",
    scale_mode="none",
    **_kwargs,
):
    splittable = _splittable(out_dtype, scale_mode)
    tile = pick_tile(M, N, K, splittable=splittable)
    split_k = pick_split_k(M, N, K, tile) if splittable else 1
    return _config(tile, split_k=split_k)


def _search_configs(
    C=None,
    A=None,
    B_T=None,
    scale_a=None,
    scale_b=None,
    M=None,
    N=None,
    K=None,
    out_dtype="i32",
    scale_mode="none",
    **_kwargs,
):
    """Order candidates so the heuristic default wins measurement ties.

    ``Autotuner`` keeps the first config that reaches the minimum time, and
    ``_graph_bench`` rounds to the noise floor, so listing the default first
    (and every plain grid before any persistent or split one) keeps the search
    from trading the known-good pick for an unmeasurable difference.
    """
    default_tile = pick_tile(M, N, K, splittable=_splittable(out_dtype, scale_mode))
    tiles = [tile for tile, _wgs in feasible_tiles(M, N, K)]
    tiles.sort(key=lambda tile: tile != default_tile)
    configs = [_config(tile) for tile in tiles]
    configs += [_config(tile, use_persistent=True) for tile in tiles if persistent_workgroups(M, N, K, tile)]

    if tiles and _splittable(out_dtype, scale_mode):
        heuristic = pick_split_k(M, N, K, default_tile)
        if heuristic > 1:
            allowed = set(feasible_splits(M, N, K, default_tile))
            reach = heuristic * _SPLIT_SEARCH_REACH
            configs += [
                _config(default_tile, split_k=split)
                for split in _powers_of_two(reach)
                if split > 1 and split in allowed
            ]
    return configs or [_default_config(M=M, N=N, K=K, out_dtype=out_dtype, scale_mode=scale_mode)]


# Run-to-run spread on a warm GPU is around half a percent, so measurements
# closer together than this are treated as equal.
_NOISE_FLOOR = 0.01
# Bound the search cost of a large shape while still giving a small one enough
# launches to time accurately.
_TARGET_ROUND_MS = 2.0
_MAX_LAUNCHES_PER_ROUND = 160
_MAX_GRAPH_LAUNCHES = 20


def _quantize(ms):
    """Round to the noise floor, on a relative grid so it works at any scale."""
    if not ms > 0 or not math.isfinite(ms):
        return ms
    step = math.log1p(_NOISE_FLOOR)
    return math.exp(round(math.log(ms) / step) * step)


def _time_ms(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _round_shape(fn):
    """Split a ~``_TARGET_ROUND_MS`` round into graph launches and replays."""
    per_call = max(_time_ms(fn, 5), 1e-4)
    per_round = max(1, min(_MAX_LAUNCHES_PER_ROUND, round(_TARGET_ROUND_MS / per_call)))
    launches = min(_MAX_GRAPH_LAUNCHES, per_round)
    return launches, max(1, -(-per_round // launches))


def _graph_bench(fn, warmup=5, rep=25):
    """Median time of one launch, measured through a captured graph.

    Replaying a graph keeps the dispatch gap out of the numbers, which matters
    because the small tiles finish faster than the host can queue them. The
    median (rather than the minimum) keeps one lucky round from deciding the
    search.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    launches, replays = _round_shape(fn)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(launches):
                fn()
    except Exception:
        return _quantize(do_bench(fn, warmup=warmup, rep=rep))

    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()

    rounds = max(3, rep // 5)
    times = sorted(_time_ms(graph.replay, replays) / launches for _ in range(rounds))
    return _quantize(times[len(times) // 2])


_gemm_tuner = autotune(
    configs=_search_configs,
    key=["M", "N", "K", "in_dtype", "out_dtype", "scale_mode"],
    default=_default_config,
    do_bench=_graph_bench,
    artifact_name="rdna3_int8_gemm",
)(rdna3_int8_gemm_dispatch)


def rdna3_int8_gemm_autotuned(
    C,
    A,
    B_T,
    scale_a=None,
    scale_b=None,
    in_dtype="int8",
    out_dtype="i32",
    scale_mode="none",
    stream=None,
):
    """Run ``C = A @ B_T.T`` with a shape-selected or measured tile."""
    M, K = map(int, A.shape)
    N = int(B_T.shape[0])
    signature = _signature(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        scale_mode,
        A.stride(0),
        B_T.stride(0),
        C.stride(0),
    )
    launch = _resolved.get(signature)
    if launch is not None:
        return launch(
            C,
            A,
            B_T,
            torch.cuda.current_stream() if stream is None else stream,
            scale_a,
            scale_b,
        )

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    with torch.cuda.device(A.device), torch.cuda.stream(launch_stream):
        return _gemm_tuner(
            C,
            A,
            B_T,
            scale_a,
            scale_b,
            M=M,
            N=N,
            K=K,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            scale_mode=scale_mode,
            stream=None,
        )
