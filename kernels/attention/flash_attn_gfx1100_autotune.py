# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tile selection for gfx1100 dense Flash Attention.

``pick_tile`` is a compile-time heuristic; with ``FLYDSL_AUTOTUNE`` set,
``select_tile`` benchmarks ``feasible_tiles`` instead.
"""

import os

try:  # pragma: no cover - exercised in GPU tests
    import torch
except ImportError:  # pragma: no cover - CPU-only environment
    torch = None

from flydsl.autotune import do_bench

WMMA_M = WMMA_K = 16
WAVE_SIZE = 32
LOAD_VEC = 8
LDS_PAD = 8
LDS_CAPACITY = 65536

# (num_waves, q_tiles, block_n, vt_rows, k_from_gmem)
_WAVES = (1, 2, 4, 8)
_Q_TILES = (1, 2)
_BLOCK_N = (32, 64)
_VT_ROWS = (4, 8)

_AUTOTUNE_SENTINEL = {"1", "true", "yes", "on"}
_TUNED_TILES: dict[tuple, tuple] = {}
_MAX_TUNE_ELEMENTS = 32_000_000


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


def _long_prefill(seq_q, bh, num_cu):
    return bh * ((seq_q + 127) // 128) >= 4 * num_cu


def _few_q_blocks(seq_q, bh, num_cu):
    """True when 64-row Q blocks give at most two workgroups per CU."""
    return bh * ((seq_q + 63) // 64) <= 2 * num_cu


def prefill_schedule(head_dim, seq_q, causal, bh, *, num_cu):
    """Return ``(prefetch, reverse_q)`` for a ``seq_q > 16`` prefill kernel.

    ``reverse_q`` launches the longest causal Q blocks first. At D 64/128 with
    ``seq_q > 8192`` and ``bh > 16`` that order is unstable from run to run, so
    those shapes keep the forward order.
    """
    if not causal:
        return False, False
    prefetch = head_dim != 256 and not _long_prefill(seq_q, bh, num_cu) and _few_q_blocks(seq_q, bh, num_cu)
    reverse_q = head_dim == 256 or seq_q <= 8192 or bh <= 16
    return prefetch, reverse_q


def pick_tile(head_dim, seq_q, causal, bh, *, num_cu, dtype_str=None):
    """Return ``(num_waves, q_tiles, block_n, vt_rows, k_from_gmem)``."""
    if seq_q <= WMMA_M:
        if head_dim == 64:
            if bh >= 4 * num_cu:
                return 1, 1, 32, 8, True
            if bh == 1 and not causal and dtype_str == "f16":
                # Elsewhere the single-wave decode kernel spills and is slower.
                return 1, 1, 64, 8, True
            return 4, 1, 64, 8, False
        return 4, 1, 32, 8, False

    if head_dim == 256:
        return 8, 1, 32, 8, False
    if not _long_prefill(seq_q, bh, num_cu):
        if _few_q_blocks(seq_q, bh, num_cu):
            return (4, 1, 32, 8, False) if head_dim == 128 else (4, 1, 64, 8, False)
        if head_dim == 64:
            return (4, 1, 64, 8, False) if causal else (4, 2, 64, 4, True)
        return (8, 1, 64, 8, False) if causal else (4, 1, 64, 8, True)
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
        return 8, 1, 64, 8, False
    if causal and seq_q >= 65536:
        return 8, 1, 64, 8, False
    return 8, 1, 64, 8, False


def _autotune_enabled() -> bool:
    return os.environ.get("FLYDSL_AUTOTUNE", "").strip().lower() in _AUTOTUNE_SENTINEL


def _dtype_from_str(dtype_str: str):
    if torch is None:
        return None
    if dtype_str == "f16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return None


def select_tile(
    *,
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    dtype_str: str,
    causal: bool,
    num_cu: int,
    build_kernel=None,
):
    """Pick a tile, optionally benchmarking the feasible set when tuning is on."""

    bh = batch * num_heads
    default = pick_tile(head_dim, seq_q, causal, bh, num_cu=num_cu, dtype_str=dtype_str)
    if not _autotune_enabled() or build_kernel is None or torch is None or not torch.cuda.is_available():
        return default

    total_elems = batch * num_heads * head_dim * (seq_q + seq_kv)
    if total_elems > _MAX_TUNE_ELEMENTS:
        return default

    key = (head_dim, seq_q, seq_kv, causal, bh, dtype_str, int(num_cu))
    cached = _TUNED_TILES.get(key)
    if cached is not None:
        return cached

    candidates = feasible_tiles(head_dim)
    if not candidates:
        tile = default
    else:
        tile = _tune_tile(
            batch=batch,
            num_heads=num_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            num_cu=num_cu,
            candidates=candidates,
            build_kernel=build_kernel,
        )
    _TUNED_TILES[key] = tile
    return tile


def _tune_tile(
    *,
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    num_cu: int,
    candidates: list[tuple[int, int, int, int, bool]],
    build_kernel,
):
    dtype = _dtype_from_str(dtype_str)
    if dtype is None:
        return pick_tile(head_dim, seq_q, causal, batch * num_heads, num_cu=num_cu, dtype_str=dtype_str)

    device = torch.device("cuda")
    q = torch.randn(batch, seq_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_kv, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq_kv, num_heads, head_dim, dtype=dtype, device=device)
    out = torch.empty_like(q)

    best_tile = None
    best_time = None
    for tile in candidates:
        try:
            launcher = build_kernel(tile)
        except Exception:
            continue

        def _call():
            launcher(q, k, v, out)

        try:
            timing = do_bench(_call, warmup=2, rep=6)  # pragma: no cover - GPU timing dependent
        except Exception:
            continue

        if best_time is None or timing < best_time:
            best_time = timing
            best_tile = tile

    torch.cuda.synchronize()
    if best_tile is None:
        return pick_tile(head_dim, seq_q, causal, batch * num_heads, num_cu=num_cu, dtype_str=dtype_str)
    return best_tile
