# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Two-track tuning configs for RMSNorm (quack-style get_default + exhaustive):

  - ``get_default`` — analytic BLOCK_THREADS, zero search (normal runs).
  - ``get_all_configs`` — BLOCK_THREADS x waves_per_eu, swept under FLYDSL_AUTOTUNE=1.

VEC_WIDTH is not tuned (pinned to 128//elem_bits by the 128-bit buffer copy).
"""

from flydsl.autotune import Config
from kernels.common.kernels_common import get_warp_size
from kernels.norm.rmsnorm_kernel import SMALL_N_THRESHOLD

# Candidate block sizes. All are multiples of the warp size (64 on CDNA) and
# span both the <=256 (no known_block_size) and >256 (needs known_block_size)
# regimes so the tuner can trade occupancy against per-thread work.
_BLOCK_THREADS_CHOICES = (128, 256, 512, 1024)
_WAVES_PER_EU_CHOICES = (0, 1, 2)  # Triton-style exact WPE; 0 means compiler default.
_CDNA_EUS_PER_CU = 4


def _elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def get_default(N: int, dtype_str: str, arch: str = None) -> Config:
    """Analytic default — a solid BLOCK_THREADS without searching.

    Heuristic: pick the smallest block whose vectorized tiles cover the row in a
    handful of iterations, clamped to [128, 1024]. Wider rows want more threads;
    narrow rows keep the block small to preserve occupancy.

    For bf16/f16 the pick is then shrunk to the largest candidate whose tile
    (``BLOCK_THREADS * VEC_WIDTH``) evenly divides N, so the zero-search default
    actually hits the vectorized fast path (the kernel gates it on
    ``N % tile_cols == 0``) -- matching the divisibility filter get_all_configs
    applies. Without this, a "tile-aware" block can still miss the fast path and
    silently fall to the slow scalar loop for common N (e.g. bf16 N=5120 picks
    256 whose tile 2048 does not divide 5120 -> scalar, while 128 would
    vectorize). If nothing divides N, every block runs scalar anyway, so keep
    the heuristic pick. f32 always uses the scalar loop, so no filter applies.
    """
    vec_width = 128 // _elem_bits(dtype_str)
    # Aim for ~2 vectorized tiles per row: block ≈ N / (2 * vec_width).
    target = N // max(1, (2 * vec_width))
    block = 128
    for choice in _BLOCK_THREADS_CHOICES:
        if choice <= max(128, target):
            block = choice

    if _elem_bits(dtype_str) <= 16:
        dividing = [b for b in _BLOCK_THREADS_CHOICES if b <= block and N >= b * vec_width and N % (b * vec_width) == 0]
        if dividing:
            block = max(dividing)

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
    warp_size = get_warp_size(arch)
    configs = []
    for block in _BLOCK_THREADS_CHOICES:
        if vectorized:
            tile_cols = block * vec_width
            # bf16/f16: keep only blocks that hit the vectorized fast path for N.
            if N < tile_cols or N % tile_cols != 0:
                continue
        for wpe in _WAVES_PER_EU_CHOICES:
            # A workgroup imposes its own occupancy floor. On CDNA, block / 64
            # waves are distributed over four EUs; exact WPE below that floor is
            # impossible, so LLVM falls back to its default occupancy decision.
            if wpe and warp_size == 64 and block > wpe * warp_size * _CDNA_EUS_PER_CU:
                continue
            configs.append(Config(waves_per_eu=wpe, BLOCK_THREADS=block))
    # Fall back to the heuristic default if nothing fit (e.g. an odd bf16 N).
    if not configs:
        configs.append(get_default(N, dtype_str, arch))
    return configs


def _get_default_for_autotune(input_t, gamma, output, m_in, N, dtype_str="bf16", stream=None):
    """Adapt the direct launcher's call signature to the default heuristic."""
    return get_default(N, dtype_str)


def _get_all_configs_for_autotune(input_t, gamma, output, m_in, N, dtype_str="bf16", stream=None):
    """Adapt the direct launcher's call signature to the search space."""
    return get_all_configs(N, dtype_str)
