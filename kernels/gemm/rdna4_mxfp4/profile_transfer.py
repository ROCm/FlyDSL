# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Propose nearby measured strategies while preserving their K-loop geometry."""

import math
from functools import lru_cache

from .profile_policy import profile_choice


def footprint(m, n, k):
    padded_n = (n + 15) // 16 * 16
    return padded_n * (17 * (k // 32) + 4) + m * k + 2 * m * n


@lru_cache(maxsize=4096)
def _unroll_divisors(k):
    return tuple(d for d in range(1, 17) if k // 32 % d == 0)


def transferred_choice(model, m, n, k, policy, adapt, legal):
    """Return source M/N/K and a configuration; new N/K still requires warmup.

    These are parameter proposals, not aliases for a source executable.
    Exact measured families are handled by the caller before this fallback.
    """
    options = []
    for family in model["families"]:
        source_n, source_k = map(int, family.split(":"))
        if (source_n, source_k) == (n, k) or source_n % 2 != n % 2:
            continue
        nd, kd = abs(math.log2(n / source_n)), abs(math.log2(k / source_k))
        if nd > 1 or kd > math.log2(1.5) or nd + 2 * kd > 1.25:
            continue
        match = profile_choice(model, m, source_n, source_k, policy, adapt)
        if match is None:
            continue
        source_m, raw = match
        before, after = adapt(raw, m, source_n, source_k), adapt(raw, m, n, k)
        # A different divisor or BK changes the algorithm's instruction
        # sequence; the general cost fallback handles that case instead.
        names = (
            ("tile_k", "operand_prefetch")
            if raw["engine"] == "hybrid"
            else ("k_unroll",)
        )
        if any(before.get(name) != after.get(name) for name in names):
            continue
        if raw["engine"] == "register":
            if _unroll_divisors(source_k) != _unroll_divisors(k):
                continue
            bn = 16 * before["waves_n"] * before["repeat_n"]
            if (source_n % bn == 0) != (n % bn == 0):
                continue
        if not legal(after, m, n, k):
            continue
        if (footprint(source_m, source_n, source_k) <= 64 * 2**20) != (
            footprint(m, n, k) <= 64 * 2**20
        ):
            continue
        same_storage = source_k == k and (source_n + 15) // 16 == (n + 15) // 16
        distance = nd + 2 * kd - (0.25 if same_storage else 0)
        options.append((distance, source_n, source_k, source_m, after))
    if not options:
        return None
    _, source_n, source_k, source_m, config = min(options, key=lambda row: row[:4])
    return source_m, source_n, source_k, config
