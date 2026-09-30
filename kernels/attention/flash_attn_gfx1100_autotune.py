# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tile selection for gfx1100 dense Flash Attention.

``flash_attn_gfx1100`` builds whatever tile it is handed. This module picks one:

  * ``pick_tile`` — compile-time heuristic from measured shapes. No GPU run.
  * ``feasible_tiles`` — tiles that divide the problem and fit in LDS. A search
    should stay inside this list; anything else fails to build.
"""

WMMA_M = WMMA_K = 16
WAVE_SIZE = 32
LOAD_VEC = 8
LDS_PAD = 8
LDS_CAPACITY = 65536
NUM_CU = 96

# (num_waves, q_tiles, block_n, vt_rows, k_from_gmem)
_WAVES = (1, 2, 4, 8)
_Q_TILES = (1, 2)
_BLOCK_N = (32, 64)
_VT_ROWS = (4, 8)


def tile_fits(head_dim, tile):
    """True when this tile can be built for ``head_dim``."""
    num_waves, q_tiles, block_n, vt_rows, k_from_gmem = tile
    if num_waves < 1 or q_tiles < 1:
        return False
    if head_dim % WMMA_K or head_dim % LOAD_VEC:
        return False
    if block_n % WMMA_K or vt_rows not in _VT_ROWS or block_n % vt_rows:
        return False
    threads = num_waves * WAVE_SIZE
    k_elems = 0 if k_from_gmem else block_n * (head_dim + LDS_PAD)
    vt_elems = head_dim * (block_n + LDS_PAD)
    if (k_elems + vt_elems) * 2 > LDS_CAPACITY:
        return False
    chunks_per_row = head_dim // LOAD_VEC
    if not k_from_gmem and (block_n * chunks_per_row) % threads:
        return False
    v_total_chunks = (block_n // vt_rows) * chunks_per_row
    if v_total_chunks < 1:
        return False
    if v_total_chunks < threads:
        return threads % v_total_chunks == 0
    return v_total_chunks % threads == 0


def feasible_tiles(head_dim):
    """Tiles this head dim can build, wider ``block_m`` first."""
    tiles = []
    for num_waves in _WAVES:
        for q_tiles in _Q_TILES:
            for block_n in _BLOCK_N:
                for vt_rows in _VT_ROWS:
                    for k_from_gmem in (False, True):
                        tile = (num_waves, q_tiles, block_n, vt_rows, k_from_gmem)
                        if tile_fits(head_dim, tile):
                            tiles.append(tile)
    tiles.sort(key=lambda t: (WMMA_M * t[0] * t[1], t[2]), reverse=True)
    return tiles


def pick_tile(head_dim, seq_q, causal, bh):
    """Return ``(num_waves, q_tiles, block_n, vt_rows, k_from_gmem)``."""
    if seq_q <= WMMA_M:
        if head_dim == 64:
            block_n = 32 if bh >= 4 * NUM_CU else 64
            return 1, 1, block_n, 8, True
        return 4, 1, 32, 8, False

    long_prefill = bh * ((seq_q + 127) // 128) >= 4 * NUM_CU
    if head_dim == 256:
        return 8, 1, 32, 8, False
    if not long_prefill:
        if head_dim == 64:
            return (4, 1, 64, 4, True) if causal else (4, 2, 64, 4, True)
        return 4, 1, 64, 8, True
    if head_dim == 64:
        if causal and seq_q < 32768:
            return 4, 2, 64, 4, False
        if not causal:
            # An unaligned KV length makes every iteration use bounds-checked
            # global fetches, not just the tail, so halve the iteration count.
            if seq_q % 64:
                return (4, 2, 64, 8, False) if bh <= 16 else (8, 2, 64, 8, False)
            if bh <= 16:
                return (4, 2, 32, 4, False) if seq_q <= 32768 else (4, 2, 64, 8, False)
            if seq_q <= 16384:
                return 4, 2, 32, 4, False
        return 8, 2, 32, 4, False
    if not causal and bh <= 16:
        return 8, 1, 64, 8, False
    if not causal and seq_q >= 24576:
        return 8, 2, 32, 8, False
    if causal and seq_q >= 65536:
        return 8, 2, 32, 8, False
    return 8, 1, 64, 8, False
