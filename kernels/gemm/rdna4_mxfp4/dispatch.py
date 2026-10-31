# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Adaptive RDNA4 register/hybrid selection from measured configuration regions.

The exported decision tree uses dimensions, alignment, K factors and a cache
hint. Its leaves rank successful configurations. Selection is CPU-only and
cached; K tiles/unroll and masked copies adapt to the actual unseen shape.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from .dispatch_features import (
    CONTEXT_FEATURE_NAMES,
    HYBRID_DEFAULTS,
    REGISTER_DEFAULTS,
    context_features,
    geometry,
    legal_config,
    normalize_config,
)

CACHE_POLICIES = ("auto", "hot", "rotating")


@lru_cache(maxsize=1)
def _model():
    model = json.loads(Path(__file__).with_name("dispatch_model.json").read_text())
    if model["version"] != 2 or tuple(model["feature_names"]) != CONTEXT_FEATURE_NAMES:
        raise ValueError("dispatch model and feature definitions do not match")
    return model


def _check(m, n, k, cache_policy):
    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("M/N must be positive and K a positive multiple of 32")
    if cache_policy not in CACHE_POLICIES:
        raise ValueError("cache_policy must be auto, hot or rotating")


def _region(m, n, k, cache_policy):
    hint = 0.5 if cache_policy == "auto" else float(cache_policy == "hot")
    values = context_features(m, n, k, hint)
    tree, index = _model()["tree"], 0
    while tree[index][0] >= 0:
        feature, threshold, left, right = tree[index]
        index = left if values[feature] <= threshold else right
    return index


def _adapt(config, m, n, k):
    config = dict(config)
    # Partial tiny-M waves exhibit a large latency cliff at long K unrolls.
    # Retain divisor-five schedules, but avoid the >=6 region for these tails.
    if config["engine"] == "register" and m < 16 and m & (m - 1):
        config["k_unroll"] = min(config.get("k_unroll", 1), 5)
    cfg = normalize_config(config, m, k)
    if cfg["engine"] == "register":
        bm = 16 * cfg["waves_m"] * cfg["repeat_m"]
        bn = 16 * cfg["waves_n"] * cfg["repeat_n"]
        if m % bm or n % bn:
            cfg["buffer_io"] = True
    else:
        cfg["tile_k"] = max(b for b in (32, 64, 128, 256) if b <= cfg["tile_k"] and k % b == 0)
        cfg["operand_prefetch"] = min(cfg["operand_prefetch"], cfg["tile_k"] // 16 - 1)
    return cfg


def _candidates(m, n, k, cache_policy="auto"):
    model = _model()
    seen = set()
    for index in model["leaves"][str(_region(m, n, k, cache_policy))]:
        cfg = _adapt(model["candidates"][index], m, n, k)
        ident = tuple(sorted(cfg.items()))
        if ident not in seen and legal_config(cfg, m, n, k):
            seen.add(ident)
            yield cfg
    if seen:
        return
    # A short K can invalidate a learned B-copy distribution. Use a complete
    # K32-capable tile, with masked copies, if every region candidate is illegal.
    cfg = (
        dict(
            REGISTER_DEFAULTS,
            waves_n=4,
            repeat_m=1 if m <= 16 else 2 if m <= 32 else 4,
            repeat_n=1,
            k_unroll=4,
        )
        if m <= 64 or n * k < 2**20
        else dict(
            HYBRID_DEFAULTS,
            tile_m=128 if m < 256 else 256,
            tile_n=64,
            tile_k=32,
            waves_m=4,
            waves_n=2,
            operand_prefetch=1,
        )
    )
    cfg = _adapt(cfg, m, n, k)
    if not legal_config(cfg, m, n, k):
        raise RuntimeError("no legal register/hybrid configuration")
    yield cfg


def rank_configurations(m, n, k, *, cache_policy="auto", limit=None):
    """Explain measured configuration priorities without running a GPU.

    Auto learns the geometric-mean objective over reused and rotating weights.
    Priority is an ordering within the selected region, not a latency estimate.
    """
    _check(m, n, k, cache_policy)
    region = _region(m, n, k, cache_policy)
    rows = [dict(config=cfg, priority=j, region=region) for j, cfg in enumerate(_candidates(m, n, k, cache_policy))]
    return rows if limit is None else rows[:limit]


@lru_cache(maxsize=4096)
def _selected(m, n, k, mode, cache_policy):
    _check(m, n, k, cache_policy)
    if mode == "grouped":
        cfg = dict(
            REGISTER_DEFAULTS,
            waves_n=4,
            repeat_m=1 if m <= 16 else 2 if m <= 32 else 4,
            repeat_n=2 if m >= 256 else 1,
            k_unroll=1,
        )
    elif mode in ("e4m3", "e5m2"):
        cfg = next(_candidates(m, n, k, cache_policy))
    else:
        raise ValueError("unknown weight preparation mode")
    return tuple(sorted(cfg.items()))


def choose_config(m, n, k, mode="e4m3", *, cache_policy="auto"):
    """Return an independent configuration dictionary for this shape/hint."""
    return dict(_selected(m, n, k, mode, cache_policy))


def tuning_candidates(m, n, k, *, cache_policy="auto", limit=64):
    """Bound a native FlyDSL calibration using nearby measured regions.

    Static proposals seed the search. Nearby training regions contribute
    alternative geometries; register divisors cover K factors and tail cliffs.
    No measured latency for the query shape is used to build this shortlist.
    """
    _check(m, n, k, cache_policy)
    selected, seen = [], set()

    def add(raw):
        cfg = dict(raw)
        if cfg["engine"] == "hybrid":
            cfg["tile_k"] = max(b for b in (32, 64, 128, 256) if b <= cfg["tile_k"] and k % b == 0)
            cfg["operand_prefetch"] = min(cfg["operand_prefetch"], cfg["tile_k"] // 16 - 1)
        cfg = normalize_config(cfg, m, k)
        if cfg["engine"] == "register":
            bm = 16 * cfg["waves_m"] * cfg["repeat_m"]
            bn = 16 * cfg["waves_n"] * cfg["repeat_n"]
            if m % bm or n % bn:
                cfg["buffer_io"] = True
        ident = tuple(sorted(cfg.items()))
        if ident not in seen and legal_config(cfg, m, n, k):
            selected.append(cfg)
            seen.add(ident)

    hints = [cache_policy] + [p for p in CACHE_POLICIES if p != cache_policy]
    for hint in hints:
        add(choose_config(m, n, k, cache_policy=hint))
    bk = max(b for b in (32, 64, 128, 256) if k % b == 0)
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128 if m < 256 else 256
    for wn in (1, 2):
        add(
            dict(
                HYBRID_DEFAULTS,
                tile_m=bm,
                tile_n=32 if wn == 1 else 64,
                tile_k=bk,
                waves_m=min(4, bm // 16),
                waves_n=wn,
                operand_prefetch=min(2, bk // 16 - 1),
            )
        )
    groups = k // 32
    divisors = [d for d in range(1, 17) if groups % d == 0]
    for hint in hints:
        base = choose_config(m, n, k, cache_policy=hint)
        if base["engine"] == "register":
            for unroll in divisors:
                add(dict(base, k_unroll=unroll))
    for waves_n in (2, 4, 1):
        for repeat_m in sorted({1, 2, min(4, (m + 15) // 16)}):
            for unroll in divisors:
                add(
                    dict(
                        REGISTER_DEFAULTS,
                        waves_n=waves_n,
                        repeat_m=repeat_m,
                        repeat_n=1,
                        k_unroll=unroll,
                    )
                )
                if waves_n == 2 and unroll in (divisors[0], divisors[-1]):
                    add(
                        dict(
                            REGISTER_DEFAULTS,
                            waves_n=waves_n,
                            repeat_m=repeat_m,
                            repeat_n=1,
                            k_unroll=unroll,
                            table_mode="lut",
                        )
                    )
    # Reserve at least half the budget for geometries selected in nearby
    # measured regions, rather than spending it entirely on register unrolls.
    selected = selected[: max(12, limit // 2)]
    seen = {tuple(sorted(c.items())) for c in selected}
    model = _model()
    neighbors = sorted(
        model["training"]["shapes"],
        key=lambda s: (
            2 * abs(math.log2(m / s[0]))
            + abs(math.log2(n * k / (s[1] * s[2])))
            + 0.5 * abs(math.log2(n / s[1]))
            + 0.5 * abs(math.log2(k / s[2]))
        ),
    )
    geometry_seen = set()
    deferred = []
    for shape in neighbors[:24]:
        for hint in hints:
            for index in model["leaves"][str(_region(*shape, hint))][:2]:
                cfg = model["candidates"][index]
                signature = (cfg["engine"], *geometry(cfg)[:5])
                if signature in geometry_seen:
                    deferred.append(cfg)
                else:
                    geometry_seen.add(signature)
                    add(cfg)
                if len(selected) >= limit:
                    return selected[:limit]
    for cfg in deferred:
        add(cfg)
        if len(selected) >= limit:
            break
    return selected[:limit]
