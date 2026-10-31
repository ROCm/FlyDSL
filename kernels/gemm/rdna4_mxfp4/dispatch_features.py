# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU shape, resource and legality helpers for the MXFP4 selector."""

import math

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


def normalize_config(config, m, k):
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


def geometry(config):
    wm, wn = config["waves_m"], config["waves_n"]
    if config["engine"] == "register":
        rm, rn = config["repeat_m"], config["repeat_n"]
        return wm * rm * 16, wn * rn * 16, 32 * config["k_unroll"], wm, wn, rm, rn, 0
    bm, bn, bk = (config[name] for name in ("tile_m", "tile_n", "tile_k"))
    bb = (
        bn * bk
        if config["b_swizzle"]
        else bn // 16 * (bk // 16) * 34 * 8 if config["fragment_lds"] else bn * (bk + config["pad"])
    )
    return bm, bn, bk, wm, wn, bm // (16 * wm), bn // (16 * wn), bb * config["stages"]


def legal_config(config, m, n, k):
    """Enforce the retained factories' tile, buffer and LDS requirements."""
    if min(m, n, k) <= 0 or k % 32:
        return False
    try:
        bm, bn, bk, wm, wn, rm, rn, lds = geometry(config)
        threads = wm * wn * 32
        if not 32 <= threads <= 1024 or rm * rn * 8 > 192:
            return False
        if config["engine"] == "register":
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
)


def context_features(m, n, k, hot):
    weight_bytes = ((n + 15) // 16 * 16) * (k * 17 / 32 + 4)
    return [
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
    ]
