"""Auto-generated tune config for standalone grouped-MoE GEMM1 (a8w4), gfx950 MI355X.
Shape: K=7168 N=3072 experts=48 topk=6 EP8. Timing: run_perftest testGraph=False, out fp8.
bs16-32768 full-swept over ALL dims (tile/wave/wgm/gm/nmajor/pipe_weights/mfma_amajor/swizzle_a).
"""

TUNE_CONFIG = {
    8: dict(tile_m=32, tile_n=256, tile_k=256, num_waves=4, wgm=4, grid_mult=4, sched_nmajor=False, pipe_weights=True),  # 111.2us (not re-swept)
    16: dict(tile_m=32, tile_n=256, tile_k=256, num_waves=4, wgm=1, grid_mult=8, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 184.0us
    32: dict(tile_m=32, tile_n=256, tile_k=256, num_waves=4, wgm=2, grid_mult=8, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 191.7us
    64: dict(tile_m=32, tile_n=256, tile_k=256, num_waves=4, wgm=2, grid_mult=8, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 195.5us
    128: dict(tile_m=32, tile_n=256, tile_k=256, num_waves=4, wgm=2, grid_mult=8, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 195.5us
    256: dict(tile_m=64, tile_n=128, tile_k=256, num_waves=4, wgm=2, grid_mult=8, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 226.7us
    512: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=1, grid_mult=2, sched_nmajor=False, pipe_weights=False, mfma_amajor=False, swizzle_a=True),  # 286.5us
    1024: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=1, grid_mult=4, sched_nmajor=True, pipe_weights=False, mfma_amajor=False, swizzle_a=True),  # 410.5us
    2048: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=1, grid_mult=2, sched_nmajor=True, pipe_weights=False, mfma_amajor=False, swizzle_a=True),  # 635.6us
    4096: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=4, grid_mult=2, sched_nmajor=True, pipe_weights=False, mfma_amajor=False, swizzle_a=True),  # 1099.5us
    8192: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=4, grid_mult=2, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 2127.8us
    16384: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=4, grid_mult=2, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 4124.4us
    32768: dict(tile_m=64, tile_n=256, tile_k=256, num_waves=4, wgm=4, grid_mult=2, sched_nmajor=False, pipe_weights=True, mfma_amajor=True, swizzle_a=True),  # 8118.8us (sweep pending; current best)
}


def get_config(tokens):
    for bp in sorted(TUNE_CONFIG):
        if tokens <= bp:
            return TUNE_CONFIG[bp]
    return TUNE_CONFIG[max(TUNE_CONFIG)]
