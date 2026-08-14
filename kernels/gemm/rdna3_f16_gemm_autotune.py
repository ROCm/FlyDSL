# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tile selection for the RDNA3 WMMA GEMM.

``rdna3_f16_gemm`` builds whatever tile it is handed and defaults to 128x128x32.
That tile is right once the problem fills the grid, but it cuts only 4
workgroups at 256x256 and 16 at 512x512, so on a 96-CU part most CUs idle no
matter how good the inner loop is. Choosing the tile from the shape is worth up
to 3.0x there, and this module owns that decision in two layers:

  * ``pick_tile`` — a heuristic fitted to a sweep of every feasible tile on 27
    shapes. It needs no GPU and no measurement, and it is what a call resolves
    to with nothing configured, so the wrapper benchmarks nothing by default.
  * the shared autotuner — ``FLYDSL_AUTOTUNE=1`` sweeps ``feasible_tiles`` for
    real on the GPU in hand, and the result can be frozen into an offline
    artifact so other machines with the same device fingerprint skip the search.

The second layer earns its keep mainly where the first cannot reach: ``NUM_CU``
is hard-coded for gfx1100, so the thresholds do not transfer to a gfx11 part
with a different CU count, and shapes outside the fitted set are extrapolation.
It is also the least settled part of this — see ``_graph_bench`` for why its
verdict on the shortest kernels should not be taken at face value. The default
path does not benchmark and is unaffected.

The tile is not a Constexpr argument of one compiled entry point: it decides the
block shape, the wave grid and the LDS budget, so each candidate is a separate
module. The dispatcher below therefore takes the tile as ordinary keyword
arguments and looks the built module up in a cache, the same shape of
indirection ``conv3d_implicit_autotune`` uses.
"""

import functools

import torch

from flydsl.autotune import Config, autotune, do_bench
from kernels.gemm.rdna3_f16_gemm import K_PAD, WAVE_SIZE, WMMA_K, WMMA_M, WMMA_N, create_wmma_gemm_module

# gfx1100 (W7900) exposes 96 CUs. Used only to decide when a tile is too coarse
# to fill the machine; being off by a little just shifts one ladder step.
NUM_CU = 96

# LDS budget per workgroup. K_PAD comes from the kernel so the feasibility check
# below cannot drift from the allocation it is predicting.
LDS_BYTES = 64 * 1024

# Named by the block tile they produce: (reg_m, reg_n, reg_k, waves_m, waves_n).
TILE_256x256x32 = (4, 4, 2, 4, 4)
TILE_128x256x32 = (4, 4, 2, 2, 4)
TILE_128x128x32 = (4, 4, 2, 2, 2)
TILE_128x64x32 = (4, 2, 2, 2, 2)
TILE_64x64x64 = (2, 2, 4, 2, 2)
TILE_32x64x64 = (2, 2, 4, 1, 2)
TILE_32x32x64 = (2, 2, 4, 1, 1)

# Options a tile needs beyond its shape. Kept next to the ladder rather than in
# the ladder tuples so a Config stays the five tile fields the autotuner sweeps.
#
# 256x256x32 is only reachable unpadded: with K_PAD it wants 80 KB and the pad is
# what the LDS budget cannot afford at that width. Unpadded it is exactly 64 KB.
# The hints are not a tuning knob there but a requirement — without the immediate
# offsets the pad used to give it, LLVM assigns every A fragment the same register
# quad, so each ds_load waits on a full lgkmcnt(0) drain instead of overlapping
# the WMMA stream. Measured at 128x256x32: 53 TFLOP/s without the hints, 84 with.
#
# group_m is the width of the L2 grouping. The wide tiles read a whole 256-row
# band of A per workgroup, so the default 8 walks off the reuse the swizzle is
# there to get; 16 was the best of 1/4/8/16/32 for both, worth 1-2%.
_DEFAULT_OPTS = {"lds_layout": "pad", "sched_hint": False, "group_m": 8}
_TILE_OPTS = {
    TILE_256x256x32: {"lds_layout": "kblock", "sched_hint": True, "group_m": 16},
    TILE_128x256x32: {"lds_layout": "pad", "sched_hint": False, "group_m": 16},
}


def tile_opts(tile):
    return _TILE_OPTS.get(tuple(tile), _DEFAULT_OPTS)


# Tile ladder, widest first.
#
# 128x64x32 exists only on the small-K ladder: with large K a workgroup runs long
# enough that the deeper k-tile (fewer barriers, twice as long for the gmem
# prefetch to land) pays off, while with small K the per-workgroup prologue and
# epilogue dominate and the wider tile amortizes them.
#
# The ladder is the feasibility-ordered search space; pick_tile does not walk it
# in order. 32x64x64 is here only as a fallback for shapes the others cannot
# divide -- it was not the fastest tile on any of the 27 shapes swept.
_LADDER_LARGE_K = [TILE_256x256x32, TILE_128x256x32, TILE_128x128x32, TILE_64x64x64,
                   TILE_32x64x64, TILE_32x32x64]
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


def pick_tile(M, N, K):
    """Tile for this shape, fitted to a sweep of every ladder tile on 27 shapes.

    64x64x64 is the default rather than the widest tile that covers the machine.
    Measured on gfx1100 it is the fastest tile on 16 of the 27 shapes and holds
    50-59 TFLOP/s across the whole range, where 128x128x32 swings between 40 and
    72 depending on how its much coarser grid happens to land. Taking the widest
    covering tile cost up to 37% (1664x1664x1024) and averaged 6.5%.

    Five exceptions, in order. The first two are the wide tiles, and both are
    about operand traffic rather than about filling the machine: a tile reads
    ``(BM + BN) / (BM * BN)`` of A and B per output, so 128x256 moves a quarter
    less than 128x128 and 256x256 half as much again. Measured against rocBLAS TN
    in one thermal window, as a fraction of its time:

        square      4096   8192   12288  16384  20480
        128x128x32  0.805  0.903  0.919  0.875  0.829
        128x256x32  0.826  0.933  0.990  0.988  0.928
        256x256x32  0.797  0.913  0.979  1.012  0.999

    So the widest tile is not simply best: 256x256 needs roughly 32 workgroups per
    CU before its traffic saving outweighs how coarsely its grid lands, and below
    that 128x256 leads. Above it the order reverses and stays reversed, because
    128x256 is the one that starts falling off as the footprint grows.

      * 256x256x32 once the grid is worth ~32 workgroups per CU. Only reachable
        unpadded; see _TILE_OPTS for why that drags the scheduling hints with it.
      * 128x256x32 from ~4 workgroups per CU up. Never behind 128x128x32 anywhere
        it applies, by 2-11%.
      * 128x128x32 once the grid is worth at least ~2.5 workgroups per CU. Its
        compute intensity wins outright there, by 5-11% over 64x64x64.
      * 128x64x32 (small-K ladder only) when it lands near one workgroup per CU.
        That is the narrow band around 1024x1024, where it leads by 13-28%.
      * 32x32x64 when 64x64x64 cannot fill a quarter of the machine and K is
        long enough for the idle CUs to dominate: 256x256x4096, worth 23%.

    Against the per-shape fastest tile this averages 0.6%, worst case 8.1% at
    1152x1152x1024, where 128x128x32's grid happens to land well.
    """
    feasible = dict(feasible_tiles(M, N, K))
    if not feasible:
        return _ladder_for(K)[0]

    if feasible.get(TILE_256x256x32, 0) >= 32 * NUM_CU:
        return TILE_256x256x32
    if feasible.get(TILE_128x256x32, 0) >= 4 * NUM_CU:
        return TILE_128x256x32
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


# Launches to capture per graph. Enough that replay is dominated by the kernel
# rather than by graph launch, small enough to keep capture cheap.
_GRAPH_LAUNCHES = 20
_REPLAYS_PER_ROUND = 8

_TILE_FIELDS = ("reg_m", "reg_n", "reg_k", "waves_m", "waves_n")

# Launcher for a call signature whose tile the autotuner has already resolved.
# The tuner re-derives its cache key from scratch on every call — fingerprinting
# the environment, toolchain and device — which costs more host time than a
# small GEMM takes on the GPU, and under FLYDSL_AUTOTUNE=1 it re-runs the whole
# search per call. Consulting it once per signature keeps steady-state dispatch
# as cheap as calling the built module directly.
_resolved = {}


def _tile_config(tile):
    return Config(**dict(zip(_TILE_FIELDS, tile)))


@functools.lru_cache(maxsize=None)
def _build(M, N, K, in_dtype, out_dtype, rounding, reg_m, reg_n, reg_k, waves_m, waves_n):
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
        **tile_opts((reg_m, reg_n, reg_k, waves_m, waves_n)),
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
):
    """Run the GEMM on one tile. Unset tile fields fall back to ``pick_tile``.

    The stream is resolved here rather than by the caller so that this stays
    capturable: under ``torch.cuda.graph`` the current stream is the capture
    stream, and enqueueing onto a stream captured before then aborts the capture.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    # Resolve before the cache key so a partially specified tile and the fully
    # spelled-out one it means share a single built module.
    tile = tuple(
        auto if given is None else given
        for auto, given in zip(pick_tile(M, N, K), (reg_m, reg_n, reg_k, waves_m, waves_n))
    )
    launch_fn = _build(M, N, K, in_dtype, out_dtype, rounding, *tile)
    # A search calls this once per candidate and then once more on the winner,
    # so the last write is the config the tuner settled on.
    _resolved[(M, N, K, in_dtype, out_dtype, rounding)] = launch_fn
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
    **_kwargs,
):
    return _tile_config(pick_tile(M, N, K))


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
    **_kwargs,
):
    candidates = [_tile_config(tile) for tile, _wgs in feasible_tiles(M, N, K)]
    return candidates or [_default_config(M=M, N=N, K=K)]


def _graph_bench(fn, warmup=5, rep=25):
    """Fastest observed ms per launch, timed by replaying a captured graph.

    The stock ``do_bench`` pays one launch plus one full sync per measurement,
    about 90us of host time on this kernel. That is longer than the kernel runs
    on any shape small enough for the tile to be worth choosing, so all the
    candidates measure alike and the search ends up ranking dispatch noise.
    Capturing the launches amortises that overhead away and makes a sweep
    reproducible to within a percent.

    It is still not trustworthy on the shortest kernels. Measured in isolation,
    512x512x2048 runs at 28.6us on 64x64x64 and 31.0us on 32x32x64; measured
    here the multi-wave tiles read about 5us high and the ranking inverts, so a
    forced sweep of that shape emits an artifact for the slower tile. Treat a
    tuned result for a sub-50us kernel as a hypothesis to confirm, not a fact.

    Falls back to the stock timer if the kernel turns out not to be capturable.
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

    # Time a batch of replays per round so the graph launch is amortised too,
    # and keep the fastest round. This runs on shared nodes where a neighbour
    # can inflate a whole round two- or threefold; taking the median carries
    # those rounds into the result, taking the minimum drops them.
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
    key=["M", "N", "K", "in_dtype", "out_dtype", "rounding"],
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

    launch_fn = _resolved.get((M, N, K, in_dtype, out_dtype, rounding))
    if launch_fn is not None:
        return launch_fn(C, A, B_T, torch.cuda.current_stream() if stream is None else stream, sr_seed)

    # Make the caller's stream current instead of passing it down, so the
    # dispatcher picks it up while a benchmark capture still gets its own.
    launch_stream = torch.cuda.current_stream() if stream is None else stream
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
        )
