# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only configuration selection for the register and hybrid strategies."""

import json
import math
import struct
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

__all__ = ["choose_config", "rank_configurations"]


@lru_cache(maxsize=1)
def _data():
    data = json.loads(Path(__file__).with_name("heuristic.json").read_text())
    if data["format_version"] != 1:
        raise ValueError("unsupported MXFP4 heuristic data format")
    return data


# Shape features and legality

REGISTER_MAX_ACCUMULATORS = 128


REGISTER_DEFAULTS = dict(
    engine="register",
    waves_m=1,
    waves_n=2,
    repeat_m=1,
    repeat_n=2,
    k_unroll=1,
    table_mode="alu",
    buffer_io=True,
    fast_epilogue=True,
)


HYBRID_DEFAULTS = dict(
    engine="hybrid",
    tile_m=256,
    tile_n=128,
    tile_k=64,
    waves_m=4,
    waves_n=2,
    stages=2,
    operand_prefetch=2,
    cache_group_m=4,
    fragment_lds=True,
    pad=8,
    b_swizzle=None,
    barrier_mode="lds",
    schedule="barrier",
    operand_schedule="none",
    epilogue_prefetch=False,
    table_mode="alu",
    fast_epilogue=True,
)


def _normalize_config(config, m, k):
    """Resolve defaults, legal K divisors and inactive layout options."""
    engine = config["engine"]
    defaults = REGISTER_DEFAULTS if engine == "register" else HYBRID_DEFAULTS
    result = dict(defaults, **config)
    result.pop("weight_bits", None)
    if engine == "register":
        request = result["k_unroll"]
        result["k_unroll"] = max(d for d in range(1, request + 1) if k // 32 % d == 0)
    elif engine == "hybrid":
        result["cache_group_m"] = min(result["cache_group_m"], (m + result["tile_m"] - 1) // result["tile_m"])
        if result["b_swizzle"] is not None:
            result["b_swizzle"] = tuple(result["b_swizzle"])
            result["fragment_lds"] = True
        if result["fragment_lds"] or result["b_swizzle"] is not None:
            result["pad"] = 8
    else:
        raise ValueError("unknown kernel engine")
    return result


def _geometry(config):
    wm, wn = config["waves_m"], config["waves_n"]
    if config["engine"] == "register":
        rm, rn = config["repeat_m"], config["repeat_n"]
        return wm * rm * 16, wn * rn * 16, 32 * config["k_unroll"], wm, wn, rm, rn, 0
    bm, bn, bk = (config[name] for name in ("tile_m", "tile_n", "tile_k"))
    bb = (
        bn * bk
        if config["b_swizzle"]
        else bn // 16 * (bk // 16) * 34 * 8
        if config["fragment_lds"]
        else bn * (bk + config["pad"])
    )
    return bm, bn, bk, wm, wn, bm // (16 * wm), bn // (16 * wn), bb * config["stages"]


def _legal_config(config, m, n, k):
    """Enforce the retained factories' tile, buffer and LDS requirements."""
    if min(m, n, k) <= 0 or k % 32:
        return False
    try:
        bm, bn, bk, wm, wn, rm, rn, lds = _geometry(config)
        threads = wm * wn * 32
        if not 32 <= threads <= 1024 or rm * rn * 8 > 192:
            return False
        if config["engine"] == "register":
            if rm * rn * 8 > REGISTER_MAX_ACCUMULATORS:
                return False
            return 1 <= config["k_unroll"] <= 16 and k % bk == 0 and (config["buffer_io"] or m % bm == n % bn == 0)
        if config["engine"] != "hybrid":
            return False
        if (
            bm % (wm * 16)
            or bn % (wn * 16)
            or k % bk
            or bk % 32
            or config["stages"] not in (1, 2)
            or not 0 < lds <= 65536
            or not 0 <= config["operand_prefetch"] < bk // 16
            or (bn // 16 * (bk // 16) * 32) % threads
        ):
            return False
        sw = config["b_swizzle"]
        return not sw or (sw[1] >= 3 and bn * bk % (1 << (sw[0] + sw[1])) == 0)
    except (KeyError, ZeroDivisionError, ValueError):
        return False


CONTEXT_FEATURE_NAMES = (
    "log_m",
    "log_n",
    "log_k",
    "log_weight_bytes",
    "log_n_over_k",
    "log_m_over_k",
    "m_power_of_two",
    "log_m_alignment",
    "log_n_alignment",
    "log_k_alignment",
    "hot",
    "log_output_bytes",
    "log_operations",
    "log_mn_tiles16",
    *(f"m_utilization_{tile}" for tile in (16, 32, 64, 128, 256, 512, 1024)),
    *(f"n_utilization_{tile}" for tile in (16, 32, 64, 128, 256, 512)),
    *(f"k_groups_divisible_{factor}" for factor in (3, 5, 7, 9, 11, 13, 15)),
    "log_max_k_divisor4",
    "log_max_k_divisor8",
    "log_max_k_divisor16",
    "k_divisor_count16",
    *(
        f"log_workgroups_{bm}x{bn}"
        for bm, bn in (
            (16, 32),
            (16, 64),
            (64, 32),
            (64, 64),
            (128, 64),
            (128, 128),
            (256, 64),
            (256, 128),
            (512, 128),
            (1024, 64),
        )
    ),
)

_CONTEXT_FLOATS = struct.Struct(f"<{len(CONTEXT_FEATURE_NAMES)}f")


def _context_features(m, n, k, hot):
    weight_bytes = ((n + 15) // 16 * 16) * (k * 17 / 32 + 4)
    divisors = [d for d in range(1, 17) if k // 32 % d == 0]
    values = [
        math.log2(m),
        math.log2(n),
        math.log2(k),
        math.log2(weight_bytes),
        math.log2(n / k),
        math.log2(m / k),
        float(m & (m - 1) == 0),
        math.log2(m & -m),
        math.log2(n & -n),
        math.log2(k & -k),
        float(hot),
        math.log2(2 * m * n),
        math.log2(2 * m * n * k),
        math.log2(((m + 15) // 16) * ((n + 15) // 16)),
        *(m / (((m + tile - 1) // tile) * tile) for tile in (16, 32, 64, 128, 256, 512, 1024)),
        *(n / (((n + tile - 1) // tile) * tile) for tile in (16, 32, 64, 128, 256, 512)),
        *(float(k // 32 % factor == 0) for factor in (3, 5, 7, 9, 11, 13, 15)),
        *(math.log2(max(d for d in divisors if d <= bound)) for bound in (4, 8, 16)),
        len(divisors) / 16,
        *(
            math.log2(((m + bm - 1) // bm) * ((n + bn - 1) // bn))
            for bm, bn in (
                (16, 32),
                (16, 64),
                (64, 32),
                (64, 64),
                (128, 64),
                (128, 128),
                (256, 64),
                (256, 128),
                (512, 128),
                (1024, 64),
            )
        ),
    ]
    # sklearn evaluates tree inputs as float32. Preserve those exact branch
    # decisions at deployment without a NumPy dependency in the CPU selector.
    return _CONTEXT_FLOATS.unpack(_CONTEXT_FLOATS.pack(*values))


# Configuration features and prediction

COST_FEATURE_NAMES = (
    "hybrid",
    "log_bm",
    "log_bn",
    "log_bk",
    "log_waves_m",
    "log_waves_n",
    "log_repeat_m",
    "log_repeat_n",
    "log_accumulators",
    "unroll",
    "log_iterations",
    "m_tile_utilization",
    "n_tile_utilization",
    "m_aligned",
    "padded_n_aligned",
    "first_m_tile_utilization",
    "first_n_tile_utilization",
    "log_grid",
    "log_grid_m",
    "log_grid_n",
    "log_waves",
    "static_m",
    "one_m_tile",
    "lut",
    "fast_epilogue",
    "buffer_io",
    "stages",
    "prefetch",
    "prefetch_fraction",
    "log_cache_group",
    "fragment_lds",
    "swizzle",
    "schedule_groups",
    "schedule_barrier",
    "operand_barrier",
    "epilogue_prefetch",
    "pad",
    "log_lds",
    "broadcast_x",
    "unroll_power_two",
    "log_wmma_per_loop",
    "log_a_bytes_per_tile",
    "log_b_bytes_per_tile",
    "n_aligned",
    "m_remainder",
    "n_remainder",
    "dynamic_both_tails",
    "epilogue_full",
    "epilogue_row_mask",
    "epilogue_column_mask",
    "epilogue_both_masks",
    "log_launch_waves",
    "log_nominal_operations",
    "log_estimated_a_read_bytes",
    "log_estimated_b_read_bytes",
    "log_a_read_amplification",
    "log_b_read_amplification",
    "log_lds_per_wave",
    "log_accumulator_registers_per_block",
    "log_b_copy_words_per_thread",
    "log_a_prefetch_register_proxy",
    "tile_work_utilization",
    "log_wave_batches256",
)

_FLOATS = struct.Struct(f"<{len(COST_FEATURE_NAMES)}f")


def _configuration_features(m, n, k, config):
    """Describe the actual configuration, without GPU work or ML dependencies."""
    hybrid = config["engine"] == "hybrid"
    wm, wn = config["waves_m"], config["waves_n"]
    if hybrid:
        bm, bn, bk = (config[key] for key in ("tile_m", "tile_n", "tile_k"))
        rm, rn = bm // (16 * wm), bn // (16 * wn)
        unroll = 1
        bbytes = (
            bn * bk
            if config["b_swizzle"]
            else bn // 16 * (bk // 16) * 34 * 8
            if config["fragment_lds"]
            else bn * (bk + config["pad"])
        )
        lds = bbytes * config["stages"]
    else:
        rm, rn = config["repeat_m"], config["repeat_n"]
        bm, bn = wm * rm * 16, wn * rn * 16
        unroll = max(d for d in range(1, config["k_unroll"] + 1) if k // 32 % d == 0)
        bk, lds = 32 * unroll, 0
    gm, gn = (m + bm - 1) // bm, (n + bn - 1) // bn
    npad = (n + 15) // 16 * 16
    prefetch = config.get("operand_prefetch", 0)
    static_m = m in (1, 2, 4, 8, 16)
    m_full, n_full = m % bm == 0, n % bn == 0
    fast = config["fast_epilogue"]
    threads = wm * wn * 32
    a_reads = gm * gn * bm * k * wn
    b_reads = gm * gn * bn * (k * 17 / 32 + 4) * (1 if hybrid else wm)
    values = (
        float(hybrid),
        *map(math.log2, (bm, bn, bk, wm, wn, rm, rn, 8 * rm * rn)),
        unroll,
        math.log2(k / bk),
        m / (gm * bm),
        n / (gn * bn),
        float(m % bm == 0),
        float(npad % bn == 0),
        min(m, bm) / bm,
        min(npad, bn) / bn,
        math.log2(gm * gn),
        math.log2(gm),
        math.log2(gn),
        math.log2(wm * wn),
        float(static_m),
        float(m <= bm),
        float(config["table_mode"] == "lut"),
        float(config["fast_epilogue"]),
        float(config.get("buffer_io", False)),
        config.get("stages", 0),
        prefetch,
        prefetch / (bk / 16),
        math.log2(max(1, config.get("cache_group_m", 1))),
        float(config.get("fragment_lds", False)),
        float(config.get("b_swizzle") is not None),
        float(config.get("schedule") == "groups"),
        float(config.get("schedule") == "barrier"),
        float(config.get("operand_schedule") == "barrier"),
        float(config.get("epilogue_prefetch", False)),
        config.get("pad", 0),
        math.log2(1 + lds),
        float(m == 1),
        float(unroll & (unroll - 1) == 0),
        math.log2(2 * rm * rn * (bk // 32)),
        math.log2(bm * bk),
        math.log2(bn * bk / 2),
        float(n_full),
        (m % bm) / bm,
        (n % bn) / bn,
        float(not static_m and not m_full and not n_full),
        float(fast and m_full and n_full),
        float(fast and not m_full and n_full),
        float(fast and m_full and not n_full),
        float(not fast or not m_full and not n_full),
        math.log2(gm * gn * wm * wn),
        math.log2(2 * gm * gn * bm * bn * k),
        math.log2(a_reads),
        math.log2(b_reads),
        math.log2(a_reads / (m * k)),
        math.log2(b_reads / (npad * (k * 17 / 32 + 4))),
        math.log2(1 + lds / (wm * wn)),
        math.log2(8 * rm * rn * threads),
        math.log2(1 + bn * bk / (8 * threads)),
        math.log2(1 + 4 * rm * (prefetch + 1)),
        m * n / (gm * gn * bm * bn),
        math.log2(max(1, math.ceil(gm * gn * wm * wn / 256))),
    )
    return _FLOATS.unpack(_FLOATS.pack(*values))


def _predict_cost(model, values):
    """Evaluate the exported histogram trees; output is a score, not a latency."""
    result = model["baseline"]
    for tree in model["trees"]:
        index = 0
        while tree[index][0] >= 0:
            feature, threshold, left, right = tree[index]
            # NumPy weak scalar promotion can round a float64 tree threshold
            # to float32. Compare the quantized feature as a Python float.
            index = left if float(values[feature]) <= threshold else right
        result += tree[index][1]
    return result


# Bounded configuration proposals


def _bounded_candidates(m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults):
    """Combine regional priorities with a small, fixed set of legal layouts."""
    result, seen = [], set()

    def add(raw):
        config = adapt(raw, m, n, k)
        key = tuple(sorted(config.items()))
        if key in seen or not legal(config, m, n, k):
            return False
        seen.add(key)
        result.append(config)
        return True

    for hint in ("auto", "hot", "rotating"):
        for config in ranked(hint):
            add(config)
    small_unroll = max(d for d in range(1, 5) if k // 32 % d == 0)
    for wm, wn, rm in (
        (1, 1, 1),
        (1, 2, 1),
        (1, 4, 1),
        (2, 1, 1),
        (4, 1, 1),
        (1, 1, 2),
        (1, 1, 4),
    ):
        for unroll in sorted({1, small_unroll}):
            add(
                dict(
                    register_defaults,
                    waves_m=wm,
                    waves_n=wn,
                    repeat_m=rm,
                    repeat_n=1,
                    k_unroll=unroll,
                    buffer_io=True,
                    fast_epilogue=True,
                )
            )
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128 if m < 256 else 256
    for tile_m in sorted({bm, max(16, bm // 2)}):
        for tile_n in (32, 64):
            for wn in (1, 2):
                for bk in (256, 128, 64, 32):
                    if k % bk:
                        continue
                    config = dict(
                        hybrid_defaults,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=bk,
                        waves_m=min(4, tile_m // 16),
                        waves_n=wn,
                        stages=2,
                        operand_prefetch=min(2, bk // 16 - 1),
                        cache_group_m=1,
                        fragment_lds=True,
                        b_swizzle=None,
                        pad=8,
                        schedule="none",
                        operand_schedule="none",
                        epilogue_prefetch=True,
                        table_mode="lut",
                        fast_epilogue=True,
                    )
                    resolved = adapt(config, m, n, k)
                    if legal(resolved, m, n, k):
                        add(resolved)
                        break
    assert result and len(result) <= 64
    return result


def _expanded_candidates(m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults):
    """Retain the old bank and add narrow-M and smaller-LDS alternatives."""
    result = _bounded_candidates(m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults)
    seen = {tuple(sorted(config.items())) for config in result}

    def add(raw):
        if len(result) >= 96:
            return
        config = adapt(raw, m, n, k)
        key = tuple(sorted(config.items()))
        if key not in seen and legal(config, m, n, k):
            seen.add(key)
            result.append(config)

    if m > 64:
        bk = max(value for value in (32, 64, 128, 256) if k % value == 0)
        for bm in (32, 64):
            for bn in (32, 64):
                for wn in (1, 2):
                    add(
                        dict(
                            hybrid_defaults,
                            tile_m=bm,
                            tile_n=bn,
                            tile_k=bk,
                            waves_m=min(4, bm // 16),
                            waves_n=wn,
                            stages=2,
                            operand_prefetch=min(2, bk // 16 - 1),
                            cache_group_m=1,
                            fragment_lds=True,
                            b_swizzle=None,
                            pad=8,
                            schedule="none",
                            operand_schedule="none",
                            epilogue_prefetch=True,
                            table_mode="lut",
                            fast_epilogue=True,
                        )
                    )
        bk = max(value for value in (32, 64, 128) if k % value == 0)
        for bm, bn, wm, wn, group in (
            (64, 32, 2, 1, 4),
            (128, 32, 8, 2, 8),
            (64, 64, 2, 2, 4),
            (128, 64, 4, 4, 4),
        ):
            add(
                dict(
                    hybrid_defaults,
                    tile_m=bm,
                    tile_n=bn,
                    tile_k=bk,
                    waves_m=wm,
                    waves_n=wn,
                    stages=2,
                    operand_prefetch=min(2, bk // 16 - 1),
                    cache_group_m=group,
                    fragment_lds=True,
                    b_swizzle=None,
                    pad=8,
                    schedule="barrier",
                    operand_schedule="none",
                    epilogue_prefetch=False,
                    table_mode="alu",
                    fast_epilogue=True,
                )
            )
    if m <= 64 or n <= 128:
        divisors = [d for d in range(5, 17) if k // 32 % d == 0]
        if len(divisors) > 3:
            divisors = [divisors[0], *divisors[-2:]]
        repeats = 1 if m > 64 else min(4, (m + 15) // 16)
        for wn, rn, table in (
            (1, 1, "alu"),
            (2, 1, "alu"),
            (4, 1, "alu"),
            (1, 2, "alu"),
            (2, 2, "alu"),
            (4, 2, "alu"),
            (1, 1, "lut"),
            (2, 1, "lut"),
            (4, 1, "lut"),
            (1, 2, "lut"),
            (2, 2, "lut"),
            (4, 2, "lut"),
        ):
            for unroll in divisors:
                add(
                    dict(
                        register_defaults,
                        waves_m=1,
                        waves_n=wn,
                        repeat_m=repeats,
                        repeat_n=rn,
                        k_unroll=unroll,
                        table_mode=table,
                        buffer_io=True,
                        fast_epilogue=True,
                    )
                )
        if m <= 16 or n <= 128 and m > 64:
            for table in ("alu", "lut"):
                for unroll in divisors:
                    add(
                        dict(
                            register_defaults,
                            waves_m=1,
                            waves_n=2,
                            repeat_m=2,
                            repeat_n=1,
                            k_unroll=unroll,
                            table_mode=table,
                            buffer_io=True,
                            fast_epilogue=True,
                        )
                    )
    # Keep the smaller M tile available across partial two/four-tile batches.
    # Rounding M up to a repeat count omitted measured KU5 winners at M17/31.
    if 16 < m <= 64:
        for wn in (4, 2, 1):
            for table in ("alu", "lut"):
                for unroll in range(5, 17):
                    if k // 32 % unroll == 0:
                        add(
                            dict(
                                register_defaults,
                                waves_m=1,
                                waves_n=wn,
                                repeat_m=1,
                                repeat_n=1,
                                k_unroll=unroll,
                                table_mode=table,
                                buffer_io=True,
                                fast_epilogue=True,
                            )
                        )
    assert len(result) <= 96
    return result


# Measured profiles

STATIC_M = (1, 2, 4, 8, 16)
POLICIES = ("auto", "hot", "rotating")


def _specialization(m, config):
    """Mirror the M-dependent portion of entrypoint.specialization_key."""
    if m in STATIC_M:
        return m, False
    bm = config["tile_m"] if config["engine"] == "hybrid" else 16 * config["waves_m"] * config["repeat_m"]
    return 0, m % bm == 0


def _profile_choice(model, m, n, k, policy, adapt):
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
        if before != after or _specialization(source_m, before) != _specialization(m, after):
            continue
        bm = after["tile_m"] if after["engine"] == "hybrid" else 16 * after["waves_m"] * after["repeat_m"]
        source_grid, query_grid = (source_m + bm - 1) // bm, (m + bm - 1) // bm
        if max(source_grid, query_grid) > 2 * min(source_grid, query_grid):
            continue
        distance = abs(math.log2(m / source_m)) + model["grid_weight"] * abs(math.log2(query_grid / source_grid))
        choices.append((distance, source_m, dict(raw)))
    return min(choices, key=lambda row: (row[0], row[1]))[1:] if choices else None


@lru_cache(maxsize=1)
def _profile_model():
    model = _data()["profiles"]
    if model["version"] != 1:
        raise ValueError("unsupported configuration-profile schema")
    for config in model["candidates"]:
        if config.get("b_swizzle") is not None:
            config["b_swizzle"] = tuple(config["b_swizzle"])
    return model


def _measured_choice(m, n, k, policy, adapt):
    return _profile_choice(_profile_model(), m, n, k, policy, adapt)


# Compatible profile transfers


def _footprint(m, n, k):
    padded_n = (n + 15) // 16 * 16
    return padded_n * (17 * (k // 32) + 4) + m * k + 2 * m * n


@lru_cache(maxsize=4096)
def _unroll_divisors(k):
    return tuple(d for d in range(1, 17) if k // 32 % d == 0)


def _transferred_choice(model, m, n, k, policy, adapt, legal):
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
        match = _profile_choice(model, m, source_n, source_k, policy, adapt)
        if match is None:
            continue
        source_m, raw = match
        before, after = adapt(raw, m, source_n, source_k), adapt(raw, m, n, k)
        # A different divisor or BK changes the algorithm's instruction
        # sequence; the general cost fallback handles that case instead.
        names = ("tile_k", "operand_prefetch") if raw["engine"] == "hybrid" else ("k_unroll",)
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
        if (_footprint(source_m, source_n, source_k) <= 64 * 2**20) != (_footprint(m, n, k) <= 64 * 2**20):
            continue
        same_storage = source_k == k and (source_n + 15) // 16 == (n + 15) // 16
        distance = nd + 2 * kd - (0.25 if same_storage else 0)
        options.append((distance, source_n, source_k, source_m, after))
    if not options:
        return None
    _, source_n, source_k, source_m, config = min(options, key=lambda row: row[:4])
    return source_m, source_n, source_k, config


# Cost ranking


@lru_cache(maxsize=1)
def _cost_model():
    model = _data()["cost"]
    if tuple(model["feature_names"]) != (*CONTEXT_FEATURE_NAMES, *COST_FEATURE_NAMES):
        raise ValueError("cost model and CPU feature definitions do not match")
    return model


def _rerank_configs(
    m,
    n,
    k,
    cache_policy,
    regional,
    adapt,
    legal,
    register_defaults,
    hybrid_defaults,
    *,
    improvement_margin=0.0,
):
    """Keep the regional first choice unless the cost model finds a clear loss."""
    originals = list(regional(cache_policy))
    default = originals[0]
    model = _cost_model()
    hints = (0, 1) if cache_policy == "auto" else (int(cache_policy == "hot"),)
    contexts = [_context_features(m, n, k, hint) for hint in hints]
    scores = {}

    def score(config):
        key = tuple(sorted(config.items()))
        if key not in scores:
            extra = _configuration_features(m, n, k, config)
            scores[key] = sum(_predict_cost(model, (*values, *extra)) for values in contexts) / len(contexts)
        return scores[key]

    default_score = score(default)
    if default_score <= model["gate"]["floor"]:
        return originals
    bank = model.get("candidate_bank", "bounded")
    if bank not in ("bounded", "expanded"):
        raise ValueError("unsupported cost-model candidate bank")
    factory = _expanded_candidates if bank == "expanded" else _bounded_candidates
    candidates = factory(m, n, k, regional, adapt, legal, register_defaults, hybrid_defaults)
    winner = min(candidates, key=score)
    if default_score - score(winner) <= max(model["gate"]["margin"], improvement_margin):
        return originals
    return [winner, *(config for config in originals if config != winner)]


# Public configuration selection

CACHE_POLICIES = ("auto", "hot", "rotating")


@lru_cache(maxsize=1)
def _model():
    model = _data()["dispatch"]
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
    values = _context_features(m, n, k, hint)
    tree, index = _model()["tree"], 0
    while tree[index][0] >= 0:
        feature, threshold, left, right = tree[index]
        index = left if values[feature] <= threshold else right
    return index


def _adapt(config, m, n, k):
    config = dict(config)
    cfg = _normalize_config(config, m, k)
    if cfg["engine"] == "register":
        bm = 16 * cfg["waves_m"] * cfg["repeat_m"]
        bn = 16 * cfg["waves_n"] * cfg["repeat_n"]
        if m % bm or n % bn:
            cfg["buffer_io"] = True
    else:
        cfg["tile_k"] = max(b for b in (32, 64, 128, 256) if b <= cfg["tile_k"] and k % b == 0)
        cfg["operand_prefetch"] = min(cfg["operand_prefetch"], cfg["tile_k"] // 16 - 1)
    return cfg


def _regional_candidates(m, n, k, cache_policy="auto"):
    model = _model()
    seen = set()
    for index in model["leaves"][str(_region(m, n, k, cache_policy))]:
        cfg = _adapt(model["candidates"][index], m, n, k)
        ident = tuple(sorted(cfg.items()))
        if ident not in seen and _legal_config(cfg, m, n, k):
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
    if not _legal_config(cfg, m, n, k):
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


def _tuning_candidates(m, n, k, *, cache_policy="auto", limit=64):
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
        cfg = _normalize_config(cfg, m, k)
        if cfg["engine"] == "register":
            bm = 16 * cfg["waves_m"] * cfg["repeat_m"]
            bn = 16 * cfg["waves_n"] * cfg["repeat_n"]
            if m % bm or n % bn:
                cfg["buffer_io"] = True
        ident = tuple(sorted(cfg.items()))
        if ident not in seen and _legal_config(cfg, m, n, k):
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
                signature = (cfg["engine"], *_geometry(cfg)[:5])
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


@lru_cache(maxsize=4096)
def _profile_proposal(m, n, k, cache_policy):

    exact = _measured_choice(m, n, k, cache_policy, _adapt)
    if exact is not None:
        return exact[0], tuple(sorted(exact[1].items()))

    nearby = _transferred_choice(_profile_model(), m, n, k, cache_policy, _adapt, _legal_config)
    if nearby is not None:
        return None, tuple(sorted(nearby[3].items()))
    return None


def _proposal_regional(m, n, k, cache_policy):
    proposal = _profile_proposal(m, n, k, cache_policy)
    first = dict(proposal[1]) if proposal is not None else None
    if first is not None:
        yield first
    for config in _regional_candidates(m, n, k, cache_policy):
        if config != first:
            yield config


def _candidates(m, n, k, cache_policy="auto"):
    proposal = _profile_proposal(m, n, k, cache_policy)
    if proposal is not None:
        source_m, config = proposal
        if source_m is not None and max(m, source_m) <= 1.25 * min(m, source_m):
            yield dict(config)
            return

    yield from _rerank_configs(
        m,
        n,
        k,
        cache_policy,
        lambda hint: _proposal_regional(m, n, k, hint),
        _adapt,
        _legal_config,
        REGISTER_DEFAULTS,
        HYBRID_DEFAULTS,
        improvement_margin=math.log(1.10) if proposal is not None else 0.0,
    )
