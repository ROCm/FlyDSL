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
_WAVES_PER_EU_CHOICES = (0, 1, 2)  # 0 == leave to the compiler


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
    """Exhaustive search space: BLOCK_THREADS x waves_per_eu. Configs whose
    vectorized tile doesn't evenly divide the row are dropped (they'd fall to
    the untuned scalar path)."""
    # Small-N kernel ignores BLOCK_THREADS, so there's nothing to sweep.
    if N <= SMALL_N_THRESHOLD:
        return [get_default(N, dtype_str, arch)]

    vec_width = 128 // _elem_bits(dtype_str)
    configs = []
    for block in _BLOCK_THREADS_CHOICES:
        tile_cols = block * vec_width
        # Keep only configs that hit the vectorized fast path for this N.
        if N < tile_cols or N % tile_cols != 0 or _elem_bits(dtype_str) > 16:
            continue
        for wpe in _WAVES_PER_EU_CHOICES:
            kw = {"BLOCK_THREADS": block}
            waves = None if wpe == 0 else wpe
            configs.append(Config(waves_per_eu=waves, **kw))
    # Always include the heuristic default so the search never regresses below it.
    if not configs:
        configs.append(get_default(N, dtype_str, arch))
    return configs
