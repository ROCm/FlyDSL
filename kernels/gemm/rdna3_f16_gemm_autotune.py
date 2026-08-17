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
#
# group_m is the width of the L2 grouping. The wide tiles read a whole 256-row
# band of A per workgroup, so the default 8 walks off the reuse the swizzle is
# there to get; 16 was the best of 1/4/8/16/32 for both, worth 1-2%.
#
# stagger is on everywhere. The kernel turns it into a no-op unless the k-tile
# count is a power of two, which is exactly the case where the row stride is one
# too and workgroups reading the same k-slice in lockstep pile onto the same
# memory channels. Where it engages it is worth 2.5% (512x512x4096) to 12%
# (2048x2048x1024) and it was not behind anywhere.
#
# sched_hint splits by tile, and the split moved. It was once required at the
# wide tiles -- without the immediate offsets the pad used to give them, LLVM
# assigned every A fragment the same register quad and each ds_load waited on a
# full lgkmcnt(0) drain. Against flydsl 0.3.1 that is no longer true at
# 256x256x32, where the hint now costs 6% at 8192, 12288 and 16384 square
# (0.1-0.4% spread), and the narrower tiles are the ones that want it: 128x128x32
# gains 2-3% alone and 9% together with stagger, 128x256x32 gains 5.6-8.5%.
# 64x64x64 is left without it -- it gained 1.7% at 1024 square and lost 9.4% at
# 512x512x4096.
_DEFAULT_OPTS = {"lds_layout": "pad", "sched_hint": False, "group_m": 8, "stagger": 1}
_TILE_OPTS = {
    TILE_256x256x32: {"lds_layout": "kblock", "sched_hint": False, "group_m": 16, "stagger": 1},
    TILE_128x128x32: {"lds_layout": "pad", "sched_hint": True, "group_m": 8, "stagger": 1},
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


def pick_tile(M, N, K):
    """Tile for this shape, fitted to a sweep of every ladder tile on 27 shapes.

    64x64x64 is the default rather than the widest tile that covers the machine.
    Measured on gfx1100 it is the fastest tile on 16 of the 27 shapes and holds
    50-59 TFLOP/s across the whole range, where 128x128x32 swings between 40 and
    72 depending on how its much coarser grid happens to land. Taking the widest
    covering tile cost up to 37% (1664x1664x1024) and averaged 6.5%.

    Four exceptions, in order. The first is the widest tile, and it is about
    operand traffic rather than about filling the machine: a tile reads
    ``(BM + BN) / (BM * BN)`` of A and B per output, so 256x256 moves half what
    128x128 does. All four wide tiles, one thermal window, microseconds:

        shape             128x128  128x256  256x128  256x256
        2048x2048x2048      223.0    237.0    257.3    294.5
        4096x2048x8192     1598.5   1760.2   1866.7   1738.7
        4096x4096x4096     1572.7   1627.9   1748.4   1704.0
        6144x6144x4096     3469.3   3550.1   3669.4   3479.1
        4096x11008x4096    4225.1   4288.8   4415.4   4225.1
        8192x8192x8192    12995.6  12636.0  13490.8  12486.8
        12288x12288x4096  14050.4  14191.3  14668.9  13907.5
        16384x16384x2048  12672.9  12723.2  13048.1  12508.9

    The mixed tiles are never the answer. 128x256 was fastest on none of the 12
    shapes swept and 256x128 was last or next to last on all of them, so the
    choice is only ever 128x128 against 256x256, and neither mixed tile is
    reachable from here any more.

    What decides between the two is not the traffic ratio, which never moves:
    holding 88 TFLOP/s needs 1.375 TB/s at 128x128 and 688 GB/s at 256x256
    whatever the shape. What moves is how much of that the 96 MB of Infinity
    Cache absorbs. While the footprint fits, 128x128 sustains 1.37 TB/s of
    operand traffic and its finer grid wins outright -- 32% at 2048 square, still
    8% at 4096. Once it does not fit, 128x128 saturates at that same 1.32-1.37
    TB/s and 256x256 takes over, by 4% at 8192 square. Between roughly 2000 and
    4000 workgroups' worth of 128x128 tiles the two land within 0.3% of each
    other and the threshold below is arbitrary.

      * 256x256x32 once the grid is worth ~5 workgroups per CU. Only reachable
        unpadded; see _TILE_OPTS for why that drags the scheduling hints with it.
      * 128x128x32 once the grid is worth at least ~2.5 workgroups per CU. Its
        compute intensity wins outright there, by 5-11% over 64x64x64. Whatever
        can run 128x256 can run this, so dropping that tile strands nothing.
      * 128x64x32 (small-K ladder only) when it lands near one workgroup per CU.
        That is the narrow band around 1024x1024, where it leads by 13-28%.
      * 32x32x64 when 64x64x64 cannot fill a quarter of the machine and K is
        long enough for the idle CUs to dominate: 256x256x4096, worth 23%.

    Against the per-shape fastest tile the small-tile half of this averages 0.6%,
    worst case 8.1% at 1152x1152x1024, where 128x128x32's grid happens to land
    well. The wide-tile half is within 0.3% on all 12 shapes it was fitted to,
    and dropping 128x256 moved 15 shapes by a mean of 1.020x and a worst of
    1.009x with nothing behind, 8 of them outside the fitted set.
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


# The row strides belong in the key because they are compile-time arguments of
# the built module: a padded slice and a tight one of the same M, N, K resolve to
# different kernels, and serving one from the other's entry reads the operands at
# the wrong pitch.
def _signature(M, N, K, in_dtype, out_dtype, rounding, lda, ldb, ldc):
    return (M, N, K, in_dtype, out_dtype, rounding, lda, ldb, ldc)


def _tile_config(tile):
    return Config(**dict(zip(_TILE_FIELDS, tile)))


# One whole tile per persistent workgroup. ``persistent_wgs=num_tiles`` gives every
# workgroup a band of whole tiles; only 0 or num_tiles is accepted by the kernel.
#
# Not knowing the mechanism is why this is allowed at one tile rather than by a
# size threshold: the effect does not generalise across tiles and going wider is
# where it turns. 256x256x32 is a rout, losing 14% at both 16384x16384x2048 and
# x4096. The small
# tiles lose 1-2.5% (512 square, 512x512x2048, 256x256x4096), where the whole
# kernel is tens of microseconds and one more loop around the body never
# amortizes.
def persistent_wgs(M, N, K, tile):
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    block_m, block_n = 16 * reg_m * waves_m, 16 * reg_n * waves_n
    if (block_m, block_n) != (128, 128) or K // (16 * reg_k) < 2:
        return 0
    num_tiles = -(-M // block_m) * -(-N // block_n)
    return num_tiles if num_tiles >= 256 else 0


@functools.lru_cache(maxsize=None)
def _build(M, N, K, in_dtype, out_dtype, rounding, reg_m, reg_n, reg_k, waves_m, waves_n, lda, ldb, ldc):
    opts = dict(tile_opts((reg_m, reg_n, reg_k, waves_m, waves_n)))
    # A padded row stride and stagger are two ways to break up the same L2 set
    # camping, so a caller who has already padded should not also pay for the
    # stagger: at 2048 cubed, tight with stagger is 224.22 us, ld+64 without it
    # 221.70, and the two together 227.70 -- worse than either alone.
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
    # Taken from the tensors rather than asked of the caller, so that handing in
    # a row-padded slice is all it takes to get the padded kernel.
    strides = (A.stride(0), B_T.stride(0), C.stride(0))
    launch_fn = _build(M, N, K, in_dtype, out_dtype, rounding, *tile, *strides)
    # A search calls this once per candidate and then once more on the winner,
    # so the last write is the config the tuner settled on.
    _resolved[_signature(M, N, K, in_dtype, out_dtype, rounding, *strides)] = launch_fn
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

    launch_fn = _resolved.get(
        _signature(M, N, K, in_dtype, out_dtype, rounding, A.stride(0), B_T.stride(0), C.stride(0))
    )
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
