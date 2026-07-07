# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Two-track tuning configs for RMSNorm (quack-style get_default + exhaustive):

  - ``get_default`` — analytic BLOCK_THREADS, zero search (normal runs).
  - ``get_all_configs`` — BLOCK_THREADS x waves_per_eu, swept under FLYDSL_AUTOTUNE=1.

VEC_WIDTH is not tuned (pinned to 128//elem_bits by the 128-bit buffer copy).
"""

from flydsl.autotune import Config
from kernels.rmsnorm_kernel import SMALL_N_THRESHOLD

# Candidate block sizes. All are multiples of the warp size (64 on CDNA) and
# span both the <=256 (no known_block_size) and >256 (needs known_block_size)
# regimes so the tuner can trade occupancy against per-thread work.
_BLOCK_THREADS_CHOICES = (128, 256, 512, 1024)
_WAVES_PER_EU_CHOICES = (0, 1, 2)  # 0 == leave to the compiler; nonzero lowers to the rocdl.waves_per_eu func attr


def _elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def get_default(N: int, dtype_str: str, arch: str = None) -> Config:
    """Analytic default — a solid BLOCK_THREADS without searching.

    Heuristic: pick the smallest block whose vectorized tiles cover the row in a
    handful of iterations, clamped to [128, 1024]. Wider rows want more threads;
    narrow rows keep the block small to preserve occupancy.
    """
    vec_width = 128 // _elem_bits(dtype_str)
    # Aim for ~2 vectorized tiles per row: block ≈ N / (2 * vec_width).
    target = N // max(1, (2 * vec_width))
    block = 128
    for choice in _BLOCK_THREADS_CHOICES:
        if choice <= max(128, target):
            block = choice
    return Config(BLOCK_THREADS=block)


def get_all_configs(N: int, dtype_str: str, arch: str = None):
    """Exhaustive search space: BLOCK_THREADS x waves_per_eu.

    bf16/f16 take the vectorized fast path (gated on ``elem_bits <= 16`` in the
    kernel), so only BLOCK_THREADS whose tile ``BLOCK_THREADS * VEC_WIDTH``
    evenly divides the row are kept. f32 never takes that path -- it uses the
    scalar loop, which strides by BLOCK_THREADS and handles any N -- so every
    BLOCK_THREADS is a valid, distinct f32 candidate (no tile filter). Previously
    f32 was dropped entirely and silently collapsed to the single default."""
    # Small-N kernel ignores BLOCK_THREADS, so there's nothing to sweep.
    if N <= SMALL_N_THRESHOLD:
        return [get_default(N, dtype_str, arch)]

    vectorized = _elem_bits(dtype_str) <= 16
    vec_width = 128 // _elem_bits(dtype_str)
    configs = []
    for block in _BLOCK_THREADS_CHOICES:
        if vectorized:
            tile_cols = block * vec_width
            # bf16/f16: keep only blocks that hit the vectorized fast path for N.
            if N < tile_cols or N % tile_cols != 0:
                continue
        for wpe in _WAVES_PER_EU_CHOICES:
            waves = None if wpe == 0 else wpe
            configs.append(Config(waves_per_eu=waves, BLOCK_THREADS=block))
    # Fall back to the heuristic default if nothing fit (e.g. an odd bf16 N).
    if not configs:
        configs.append(get_default(N, dtype_str, arch))
    return configs
