# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Manual tile autotuner for the implicit-GEMM conv3d kernels.

The tile config (TILE_M/TILE_N/WAVE_M/WAVE_N) is baked into the lru_cache'd
``compile_conv3d_*`` factory as a compile-time constant, so the ``@autotune``
decorator (which injects config as ``@flyc.jit`` kwargs) does not fit. Instead we
benchmark a small candidate list per problem shape with ``do_bench`` and cache
the winner (in-memory + JSON on disk), reusing the fingerprint helpers from
``flydsl.autotune``.
"""

import json
import os
from pathlib import Path

from flydsl.autotune import do_bench


def _device_fingerprint():
    """GPU arch string (e.g. 'gfx950'), or '' if unavailable."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        return str(get_rocm_arch())
    except Exception:
        return ""


def _toolchain_fingerprint():
    """FlyDSL version, so a rebuild invalidates stale cached configs."""
    try:
        import flydsl

        return str(getattr(flydsl, "__version__", ""))
    except Exception:
        return ""


# Curated legal candidates (TILE_M, TILE_N, WAVE_M, WAVE_N) from the enumerated
# constraint-satisfying space. Kept small to bound tuning time; illegal configs
# for a given shape are skipped at compile time (try/except in the sweep).
BF16_CANDIDATES = [
    (128, 128, 2, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 2, 4),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
    (64, 64, 2, 2),
]

FP8_CANDIDATES = [
    (128, 128, 2, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 2, 4),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
]

_MEM_CACHE = {}


def _cache_dir():
    return Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", os.path.expanduser("~/.flydsl/autotune")))


def _cache_file(kind):
    return _cache_dir() / f"conv3d_{kind}.json"


def _make_key(kind, shape, dtype_str):
    return (
        kind,
        tuple(shape),
        dtype_str,
        _device_fingerprint(),
        _toolchain_fingerprint(),
    )


def _load_disk(kind, key):
    f = _cache_file(kind)
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except Exception:
        return None
    ent = data.get(json.dumps(list(key)))
    return tuple(ent) if ent is not None else None


def _save_disk(kind, key, tile):
    f = _cache_file(kind)
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if f.exists():
        try:
            data = json.loads(f.read_text())
        except Exception:
            data = {}
    data[json.dumps(list(key))] = list(tile)
    f.write_text(json.dumps(data, indent=2))


def autotune_conv3d(kind, shape, dtype_str, candidates, device, run_tile, warmup=5, rep=20):
    """Return the best (TILE_M, TILE_N, WAVE_M, WAVE_N) for this problem shape.

    ``run_tile(tile)`` must launch one full conv for the given tile (used both to
    benchmark and, by the caller, for the final real run). Split-K is re-derived
    deterministically from the chosen tile at call time, so only the tile is
    cached.
    """
    key = _make_key(kind, shape, dtype_str)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    disk = _load_disk(kind, key)
    if disk is not None:
        _MEM_CACHE[key] = disk
        return disk

    results = []
    for tile in candidates:
        try:
            t = do_bench(lambda: run_tile(tile), warmup=warmup, rep=rep)
            results.append((tile, t))
        except Exception:
            pass  # skip illegal / register-spilling / OOM configs
    if not results:
        raise RuntimeError(f"all conv3d {kind} autotune configs failed for shape {shape}")

    best = min(results, key=lambda x: x[1])[0]
    _MEM_CACHE[key] = best
    _save_disk(kind, key, best)
    return best
