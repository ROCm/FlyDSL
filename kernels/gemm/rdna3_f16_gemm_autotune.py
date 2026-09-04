# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tile selection for the RDNA3 WMMA GEMM.

``rdna3_f16_gemm`` builds whatever tile it is handed and defaults to 128x128x32.
This module picks a tile from the problem shape:

  * ``pick_tile`` — a compile-time heuristic; no GPU measurement by default.
  * the shared autotuner — ``FLYDSL_AUTOTUNE=1`` sweeps ``feasible_tiles`` on
    the current device and can freeze the result into an offline artifact.

Each tile is a separate compiled module, so the dispatcher caches by tile and
shape, like ``conv3d_implicit_autotune``.
"""

import functools

import torch

from flydsl.autotune import Config, autotune, do_bench
from kernels.gemm.rdna3_f16_gemm import K_PAD, WAVE_SIZE, WMMA_K, WMMA_M, WMMA_N, create_wmma_gemm_module

# gfx1100 CU count; thresholds shift one ladder step if this is off.
NUM_CU = 96

# gfx1151 has 40 physical CUs.
GFX1151_NUM_CU = 40

# Per-workgroup LDS budget (K_PAD comes from the kernel).
LDS_BYTES = 64 * 1024

# Named by the block tile they produce: (reg_m, reg_n, reg_k, waves_m, waves_n).
TILE_256x256x32 = (4, 4, 2, 4, 4)
TILE_128x128x32 = (4, 4, 2, 2, 2)
TILE_128x64x32 = (4, 2, 2, 2, 2)
TILE_64x64x64 = (2, 2, 4, 2, 2)
TILE_32x64x64 = (2, 2, 4, 1, 2)
TILE_32x32x64 = (2, 2, 4, 1, 1)

# Per-tile options kept separate so Config stays the five tile fields.
# 256x256x32 uses kblock layout (unpadded) to fit in 64 KB LDS.
_DEFAULT_OPTS = {"lds_layout": "pad", "sched_hint": False, "group_m": 8, "stagger": 1}
_TILE_OPTS = {
    TILE_256x256x32: {"lds_layout": "kblock", "sched_hint": False, "group_m": 16, "stagger": 1},
    TILE_128x128x32: {"lds_layout": "pad", "sched_hint": True, "group_m": 8, "stagger": 1},
}


def tile_opts(tile):
    return _TILE_OPTS.get(tuple(tile), _DEFAULT_OPTS)


# Tile ladder, widest first. Small-K ladder adds 128x64x32 for short problems.
_LADDER_LARGE_K = [TILE_256x256x32, TILE_128x128x32, TILE_64x64x64, TILE_32x64x64, TILE_32x32x64]
_LADDER_SMALL_K = [TILE_128x128x32, TILE_128x64x32, TILE_64x64x64, TILE_32x64x64, TILE_32x32x64]


def _tile_workgroups(M, N, K, cfg):
    """Workgroup count for this tile, or None if the shape cannot use it."""
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg
    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    threads = waves_m * waves_n * WAVE_SIZE
    if M % block_m or N % block_n or K % block_k:
        return None
    if K // block_k < 2:  # the prefetch pipeline needs at least two k-tiles
        return None
    # Every thread must carry a whole 8-element vector of both tiles.
    if (block_m * block_k) % (threads * 8) or (block_n * block_k) % (threads * 8):
        return None
    pad = 0 if tile_opts(cfg)["lds_layout"] == "kblock" else K_PAD
    if 2 * (block_m + block_n) * (block_k + pad) * 2 > LDS_BYTES:  # 2 buffers, 2 bytes/elem
        return None
    return (M // block_m) * (N // block_n)


def _ladder_for(K):
    return _LADDER_SMALL_K if K <= 1024 else _LADDER_LARGE_K


def feasible_tiles(M, N, K):
    """``(tile, workgroup count)`` for every ladder tile this shape can run, widest first.

    Also the search space the autotuner sweeps: anything excluded here does not
    divide the shape, cannot fill the prefetch pipeline, or does not fit in LDS,
    so benchmarking it would only measure a build failure.
    """
    return [(cfg, wgs) for cfg in _ladder_for(K) if (wgs := _tile_workgroups(M, N, K, cfg)) is not None]


def _pick_tile_gfx1100(M, N, K):
    """Return a tile for this shape.

    Prefer 64x64x64 when no wide tile is justified; escalate to 128x128x32,
    256x256x32, 128x64x32 (small K), or 32x32x64 (CU-starved long-K) based on
    workgroup count and ``NUM_CU``.
    """
    feasible = dict(feasible_tiles(M, N, K))
    if not feasible:
        return _ladder_for(K)[0]

    if feasible.get(TILE_256x256x32, 0) >= 5 * NUM_CU:
        return TILE_256x256x32
    if feasible.get(TILE_128x128x32, 0) >= 2.5 * NUM_CU:
        return TILE_128x128x32
    if NUM_CU <= feasible.get(TILE_128x64x32, 0) <= 1.5 * NUM_CU:
        return TILE_128x64x32
    if TILE_64x64x64 in feasible:
        starved = feasible[TILE_64x64x64] < NUM_CU / 4 and K >= 2048
        if starved and TILE_32x32x64 in feasible:
            return TILE_32x32x64
        return TILE_64x64x64
    return list(feasible)[-1]


def _pick_tile_gfx1151(M, N, K):
    """Return a tile for the measured gfx1151 shape families.

    The 256x256 occupancy band extends to near-square shapes; other rectangular
    shapes retain the gfx1100 fallback.
    """
    feasible = dict(feasible_tiles(M, N, K))
    if not feasible:
        return _ladder_for(K)[0]

    if M == N == 256 and TILE_32x64x64 in feasible:
        return TILE_32x64x64

    wgs_128 = feasible.get(TILE_128x128x32, 0)
    one_wave = 0.9 * GFX1151_NUM_CU <= wgs_128 <= 1.6 * GFX1151_NUM_CU
    deep_grid = wgs_128 >= 2.5 * GFX1151_NUM_CU
    if one_wave or deep_grid:
        wgs_256 = feasible.get(TILE_256x256x32, 0)
        near_square = max(M, N) <= 2 * min(M, N)
        if near_square and 32 <= wgs_256 <= GFX1151_NUM_CU:
            return TILE_256x256x32
        if M == N:
            if M == 1024 and K >= 1024 and TILE_64x64x64 in feasible:
                return TILE_64x64x64
            return TILE_128x128x32

    return _pick_tile_gfx1100(M, N, K)


def pick_tile(M, N, K, arch=None):
    """Return a tile for this shape."""
    arch = str(arch or "")
    if arch.startswith("gfx1151"):
        return _pick_tile_gfx1151(M, N, K)
    return _pick_tile_gfx1100(M, N, K)


@functools.lru_cache(maxsize=None)
def _device_arch(device_index):
    return str(torch.cuda.get_device_properties(device_index).gcnArchName).split(":", 1)[0]


_GRAPH_LAUNCHES = 20
_REPLAYS_PER_ROUND = 8

_TILE_FIELDS = ("reg_m", "reg_n", "reg_k", "waves_m", "waves_n")

# Cached launcher after the autotuner resolves a tile for a signature.
_resolved = {}


# The row strides belong in the key because they are compile-time arguments of
# the built module: a padded slice and a tight one of the same M, N, K resolve to
# different kernels, and serving one from the other's entry reads the operands at
# the wrong pitch.
def _signature(M, N, K, arch, in_dtype, out_dtype, rounding, lda, ldb, ldc):
    return (M, N, K, arch, in_dtype, out_dtype, rounding, lda, ldb, ldc)


def _tile_config(tile):
    return Config(**dict(zip(_TILE_FIELDS, tile)))


def persistent_wgs(M, N, K, tile):
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    block_m, block_n = 16 * reg_m * waves_m, 16 * reg_n * waves_n
    if (block_m, block_n) != (128, 128) or K // (16 * reg_k) < 2:
        return 0
    num_tiles = -(-M // block_m) * -(-N // block_n)
    return num_tiles if num_tiles >= 256 else 0


@functools.lru_cache(maxsize=None)
def _build(M, N, K, arch, in_dtype, out_dtype, rounding, reg_m, reg_n, reg_k, waves_m, waves_n, lda, ldb, ldc):
    opts = dict(tile_opts((reg_m, reg_n, reg_k, waves_m, waves_n)))
    # Padded strides and stagger both break L2 set camping; do not combine them.
    if lda > K or ldb > K:
        opts["stagger"] = 0
    launch_fn, _, _, _ = create_wmma_gemm_module(
        M,
        N,
        K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        rounding=rounding,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        lda=lda,
        ldb=ldb,
        ldc=ldc,
        persistent_wgs=persistent_wgs(M, N, K, (reg_m, reg_n, reg_k, waves_m, waves_n)),
        arch=arch,
        **opts,
    )
    return launch_fn


def rdna3_gemm_dispatch(
    C,
    A,
    B_T,
    M,
    N,
    K,
    in_dtype="bf16",
    out_dtype="bf16",
    rounding="rn",
    reg_m=None,
    reg_n=None,
    reg_k=None,
    waves_m=None,
    waves_n=None,
    stream=None,
    sr_seed=0,
    arch=None,
):
    """Run the GEMM on one tile. Unset tile fields fall back to ``pick_tile``.

    The stream is resolved here rather than by the caller so that this stays
    capturable: under ``torch.cuda.graph`` the current stream is the capture
    stream, and enqueueing onto a stream captured before then aborts the capture.
    """
    if stream is None:
        stream = torch.cuda.current_stream(A.device)
    arch = _device_arch(A.device.index) if arch is None else str(arch)
    # Resolve before the cache key so a partially specified tile and the fully
    # spelled-out one it means share a single built module.
    tile = tuple(
        auto if given is None else given
        for auto, given in zip(pick_tile(M, N, K, arch=arch), (reg_m, reg_n, reg_k, waves_m, waves_n))
    )
    # Taken from the tensors rather than asked of the caller, so that handing in
    # a row-padded slice is all it takes to get the padded kernel.
    strides = (A.stride(0), B_T.stride(0), C.stride(0))
    launch_fn = _build(M, N, K, arch, in_dtype, out_dtype, rounding, *tile, *strides)
    # A search calls this once per candidate and then once more on the winner,
    # so the last write is the config the tuner settled on.
    _resolved[_signature(M, N, K, arch, in_dtype, out_dtype, rounding, *strides)] = launch_fn
    return launch_fn(C, A, B_T, stream, sr_seed)


def _default_config(
    C=None,
    A=None,
    B_T=None,
    M=None,
    N=None,
    K=None,
    in_dtype="bf16",
    out_dtype="bf16",
    rounding="rn",
    arch=None,
    **_kwargs,
):
    arch = _device_arch(A.device.index) if arch is None and A is not None else arch
    return _tile_config(pick_tile(M, N, K, arch=arch))


def _search_configs(
    C=None,
    A=None,
    B_T=None,
    M=None,
    N=None,
    K=None,
    in_dtype="bf16",
    out_dtype="bf16",
    rounding="rn",
    arch=None,
    **_kwargs,
):
    candidates = [_tile_config(tile) for tile, _wgs in feasible_tiles(M, N, K)]
    return candidates or [_default_config(A=A, M=M, N=N, K=K, arch=arch)]


def _graph_bench(fn, warmup=5, rep=25):
    """Fastest ms per launch via captured-graph replay.

    Amortises host launch overhead that would otherwise dominate small kernels.
    Falls back to ``do_bench`` if capture fails.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(_GRAPH_LAUNCHES):
                fn()
    except Exception:
        return do_bench(fn, warmup=warmup, rep=rep)

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    # Time a batch of replays per round and keep the fastest round.
    rounds = max(3, rep // _REPLAYS_PER_ROUND)
    best = float("inf")
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_REPLAYS_PER_ROUND):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / (_REPLAYS_PER_ROUND * _GRAPH_LAUNCHES))
    return best


_gemm_tuner = autotune(
    configs=_search_configs,
    key=["M", "N", "K", "in_dtype", "out_dtype", "rounding", "arch"],
    default=_default_config,
    do_bench=_graph_bench,
    artifact_name="rdna3_f16_gemm",
)(rdna3_gemm_dispatch)


def rdna3_gemm_autotuned(
    C,
    A,
    B_T,
    in_dtype="bf16",
    out_dtype="bf16",
    rounding="rn",
    stream=None,
    sr_seed=0,
):
    """``C = A @ B_T.T`` on the tile chosen for this shape."""
    M, K = A.shape
    N = B_T.shape[0]
    M, N, K = int(M), int(N), int(K)
    arch = _device_arch(A.device.index)

    launch_fn = _resolved.get(
        _signature(M, N, K, arch, in_dtype, out_dtype, rounding, A.stride(0), B_T.stride(0), C.stride(0))
    )
    if launch_fn is not None:
        return launch_fn(C, A, B_T, torch.cuda.current_stream(A.device) if stream is None else stream, sr_seed)

    # Make the caller's stream current instead of passing it down, so the
    # dispatcher picks it up while a benchmark capture still gets its own.
    launch_stream = torch.cuda.current_stream(A.device) if stream is None else stream
    with torch.cuda.device(A.device), torch.cuda.stream(launch_stream):
        return _gemm_tuner(
            C,
            A,
            B_T,
            M=M,
            N=N,
            K=K,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            rounding=rounding,
            stream=None,
            sr_seed=sr_seed,
            arch=arch,
        )
