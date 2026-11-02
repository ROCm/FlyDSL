# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Select measured configurations without crossing compiler specializations."""

import json
import math
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

STATIC_M = (1, 2, 4, 8, 16)
POLICIES = ("auto", "hot", "rotating")


def specialization(m, config):
    """Mirror the M-dependent portion of api.specialization_key."""
    if m in STATIC_M:
        return m, False
    bm = (
        config["tile_m"]
        if config["engine"] == "hybrid"
        else 16 * config["waves_m"] * config["repeat_m"]
    )
    return 0, m % bm == 0


def profile_choice(model, m, n, k, policy, adapt):
    """Return (source M, config), or None when no measured executable matches."""
    points = model["families"].get(f"{n}:{k}")
    if not points:
        return None
    index = POLICIES.index(policy) + 1
    ms = [point[0] for point in points]
    position = bisect_left(ms, m)
    if position < len(points) and ms[position] == m:
        return m, dict(model["candidates"][points[position][index]])
    left, right, choices = position - 1, position, []
    while (left >= 0 or right < len(points)) and len(choices) < model["neighbors"]:
        if right == len(points) or left >= 0 and m * m <= ms[left] * ms[right]:
            point = points[left]
            left -= 1
        else:
            point = points[right]
            right += 1
        source_m = point[0]
        if n <= 128 and m % 16 == 0 and source_m % 16:
            continue
        raw = model["candidates"][point[index]]
        before, after = adapt(raw, source_m, n, k), adapt(raw, m, n, k)
        if before != after or specialization(source_m, before) != specialization(
            m, after
        ):
            continue
        bm = (
            after["tile_m"]
            if after["engine"] == "hybrid"
            else 16 * after["waves_m"] * after["repeat_m"]
        )
        source_grid, query_grid = (source_m + bm - 1) // bm, (m + bm - 1) // bm
        if max(source_grid, query_grid) > 2 * min(source_grid, query_grid):
            continue
        distance = abs(math.log2(m / source_m)) + model["grid_weight"] * abs(
            math.log2(query_grid / source_grid)
        )
        choices.append((distance, source_m, dict(raw)))
    return min(choices, key=lambda row: (row[0], row[1]))[1:] if choices else None


@lru_cache(maxsize=1)
def profile_model():
    model = json.loads(Path(__file__).with_name("profile_model.json").read_text())
    if model["version"] != 1:
        raise ValueError("unsupported configuration-profile schema")
    for config in model["candidates"]:
        if config.get("b_swizzle") is not None:
            config["b_swizzle"] = tuple(config["b_swizzle"])
    return model


def measured_choice(m, n, k, policy, adapt):
    return profile_choice(profile_model(), m, n, k, policy, adapt)
