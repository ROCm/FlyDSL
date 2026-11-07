# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Analytical RDNA4 MXFP4 configuration selection using shape and GPU capacities."""

from functools import lru_cache

__all__ = ["choose_config", "rank_configurations"]

REGISTER_MAX_ACCUMULATORS = 128

REGISTER_DEFAULTS = dict(
    engine="register",
    waves_m=1,
    waves_n=2,
    repeat_m=1,
    repeat_n=2,
    k_unroll=1,
    table_mode="alu",
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
    schedule="barrier",
    operand_schedule="none",
    epilogue_prefetch=False,
    table_mode="alu",
)

# Accept fixed defaults from previously saved configurations.
_RETIRED_DEFAULTS = {
    "register": dict(buffer_io=True, weight_bits=4),
    "hybrid": dict(pad=8, b_swizzle=None, barrier_mode="lds", fast_epilogue=True, weight_bits=4),
}


def _normalize_config(config, m, k):
    """Resolve factory defaults and accept retired options at their fixed values."""
    engine = config.get("engine", "register")
    if engine not in _RETIRED_DEFAULTS:
        raise ValueError("engine must be register or hybrid")
    defaults = REGISTER_DEFAULTS if engine == "register" else HYBRID_DEFAULTS
    result = dict(defaults, **config)
    for name, value in _RETIRED_DEFAULTS[engine].items():
        if result.pop(name, value) != value:
            raise ValueError(f"{engine} fixes {name} to {value!r}")
    if engine == "register":
        request = result["k_unroll"]
        if not isinstance(request, int) or not 1 <= request <= 16:
            raise ValueError("k_unroll must be an integer from 1 to 16")
        if k // 32 % request:
            result["k_unroll"] = max(d for d in range(1, request + 1) if k // 32 % d == 0)
    elif m:
        # M=0 identifies a dynamic-row warmup key with an already chosen group.
        result["cache_group_m"] = min(result["cache_group_m"], (m + result["tile_m"] - 1) // result["tile_m"])
    return result


def _geometry(config):
    wm, wn = config["waves_m"], config["waves_n"]
    if config["engine"] == "register":
        rm, rn = config["repeat_m"], config["repeat_n"]
        return wm * rm * 16, wn * rn * 16, 32 * config["k_unroll"], wm, wn, rm, rn, 0
    bm, bn, bk = (config[name] for name in ("tile_m", "tile_n", "tile_k"))
    bb = bn // 16 * (bk // 16) * 34 * 8 if config["fragment_lds"] else bn * (bk + 8)
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
            return 1 <= config["k_unroll"] <= 16 and k % bk == 0
        if config["engine"] != "hybrid":
            return False
        if (
            bm % (wm * 16)
            or bn % (wn * 16)
            or k % bk
            or bk not in (32, 64, 128, 256)
            or config["stages"] not in (1, 2)
            or not 0 < lds <= 65536
            or not 0 <= config["operand_prefetch"] < bk // 16
            or (bn // 16 * (bk // 16) * 32) % threads
        ):
            return False
        return True
    except (KeyError, ZeroDivisionError, ValueError):
        return False


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _divisor(groups: int, bound: int) -> int:
    return max(d for d in range(1, min(groups, bound) + 1) if groups % d == 0)


def _partial_unroll(groups):
    bounded = _divisor(groups, 4)
    if bounded > 1:
        return bounded
    return next((d for d in range(5, min(groups, 16) + 1) if groups % d == 0), 1)


def _narrow(n, k, processors):
    return n <= 128 or n <= 16 * processors and 8 * n <= k


def _weight_bytes(n, k):
    return _ceil_div(n, 16) * 16 * (17 * (k // 32) + 4)


def _cached_reuse(m, n, k, processors, cache_bytes):
    return (
        64 <= m <= 128
        and k % 64
        and cache_bytes
        and _weight_bytes(n, k) <= cache_bytes // 4
        and m * n <= processors * 16384
    )


def _three_group_reuse(m, n, k, processors, cache_bytes):
    return (
        16 < m <= 128
        and n >= processors * 4
        and k % 128
        and _divisor(k // 32, 4) == 3
        and (
            m <= 32
            or (k % 64 or n <= processors * 64 and _weight_bytes(n, k) <= cache_bytes // 4)
            and cache_bytes
            and _weight_bytes(n, k) <= cache_bytes // (1 if m <= 64 else 4)
            and m * n <= processors * 16384
        )
    )


def _row_reuse(m, n, k, processors, cache_bytes):
    # Odd K32 factors limit LDS tiles; register row reuse can amortize decode.
    return (
        32 < m <= 128
        and n > 128
        and k % 64
        and (m <= 64 or not cache_bytes or _weight_bytes(n, k) >= cache_bytes // 8)
        and (
            _three_group_reuse(m, n, k, processors, cache_bytes) or not _cached_reuse(m, n, k, processors, cache_bytes)
        )
    )


def _narrow_tail_lookahead(n, k):
    # Keep the wider N-wave tail layout within a five-group K lookahead.
    return 48 < n < 64 and k >= 8192 and _divisor(k // 32, 4) == 1 and _partial_unroll(k // 32) == 5


def _long_skinny_register(m, n, k, processors):
    # Small grids need more independent K work per wave, not more N padding.
    return (
        m > 512
        and n <= 64
        and (n % 16 == 0 or _narrow_tail_lookahead(n, k))
        and k % 128 != 0
        and _divisor(k // 32, 12) >= 5
        and _ceil_div(m, 64) * _ceil_div(n, 32) <= 4 * processors
    )


def _row_register(m, n, k, processors, cache_bytes):
    # Keep multi-row register copies bounded by weight traffic and K lookahead.
    weight_bytes = _weight_bytes(n, k)
    return (
        m <= 64
        and _row_reuse(m, n, k, processors, cache_bytes)
        and cache_bytes > 0
        and (_divisor(k // 32, 8) >= 3 or _ceil_div(n, 16) >= 12 * processors)
        and (
            weight_bytes <= cache_bytes // 2 or weight_bytes <= cache_bytes and 2 * n <= k and _divisor(k // 32, 8) >= 5
        )
    )


def _long_thin_register(m, n, k, cache_bytes):
    # Avoid a padded M64 LDS tile when long, cacheable weights have few columns.
    return (
        32 < m <= 64
        and _ceil_div(m, 16) * 16 * 1.25 < 64
        and n > 128
        and k >= 8192
        and k % 128 == 0
        and 8 * n <= k
        and cache_bytes > 0
        and _weight_bytes(n, k) <= cache_bytes // 2
    )


def _register(m, n, k, processors, cache_bytes, policy):
    # Narrow outputs need small independent CTAs; a longer unroll amortizes
    # loop control when a single wave owns little packed weight data.
    if _narrow(n, k, processors):
        wm = 1 if m <= 256 else 2 if m <= 512 else 4
        wn = 1
        unroll = _divisor(k // 32, min(16, max(4, k // (32 * 16))))
        if unroll == 1:
            unroll = _divisor(k // 32, 16)
        if m in (1, 2, 4, 8, 16) and unroll < 4:
            unroll = _divisor(k // 32, 16)
        if n % 16 and m not in (1, 2, 4, 8, 16) and m % (16 * wm):
            unroll = _divisor(k // 32, 5) if n > 128 else _partial_unroll(k // 32)
        if m % (16 * wm) == 0:
            unroll = _divisor(k // 32, 16)
        if _long_skinny_register(m, n, k, processors):
            return dict(
                REGISTER_DEFAULTS,
                waves_m=1 if n % 16 else wm,
                waves_n=1,
                repeat_m=1,
                repeat_n=1,
                k_unroll=_partial_unroll(k // 32) if n % 16 else _divisor(k // 32, 12),
                table_mode="lut",
            )
        if 16 < m <= 512 and _narrow_tail_lookahead(n, k):
            return dict(
                REGISTER_DEFAULTS,
                waves_m=1,
                waves_n=4,
                repeat_m=1,
                repeat_n=1,
                k_unroll=_partial_unroll(k // 32),
                table_mode="lut",
            )
        if 16 < n <= 128 and n % 16 and m > 16 and _divisor(k // 32, 4) == 1 and unroll > 1:
            if m % 16 == 0:
                return dict(
                    REGISTER_DEFAULTS,
                    waves_m=1,
                    waves_n=1,
                    repeat_m=1,
                    repeat_n=1,
                    k_unroll=unroll,
                    table_mode="lut" if k >= 8192 else "alu",
                )
            return dict(
                REGISTER_DEFAULTS,
                waves_m=1,
                waves_n=1,
                repeat_m=1,
                repeat_n=2,
                k_unroll=unroll,
            )
        repeat_m = 1
        if n > 128 and m >= 128 and k % 128 and _ceil_div(m, 32 * wm) * _ceil_div(n, 16) >= processors:
            bounded = _divisor(k // 32, 4)
            if bounded == 1 and unroll > 1:
                pass
            elif m % (16 * wm) == 0 and bounded == 2 and _divisor(k // 32, 16) > 8:
                unroll = _divisor(k // 32, 16)
            else:
                repeat_m, unroll = 2, bounded
        return dict(
            REGISTER_DEFAULTS,
            waves_m=wm,
            waves_n=wn,
            repeat_m=repeat_m,
            repeat_n=1,
            k_unroll=unroll,
        )
    if _long_thin_register(m, n, k, cache_bytes):
        return dict(
            REGISTER_DEFAULTS,
            waves_m=1,
            waves_n=1,
            repeat_m=1,
            repeat_n=1,
            k_unroll=_divisor(k // 32, 4),
            table_mode="alu",
        )
    if (
        _row_reuse(m, n, k, processors, cache_bytes)
        and (_divisor(k // 32, 8) > 1 or n >= processors * 256)
        and not (
            policy == "hot" and cache_bytes and _weight_bytes(n, k) >= cache_bytes // 2 and _divisor(k // 32, 8) == 1
        )
    ):
        padded = _ceil_div(m, 16) * 16
        three_group = _three_group_reuse(m, n, k, processors, cache_bytes)
        streamed = cache_bytes and _weight_bytes(n, k) >= cache_bytes // 2
        wn = 1 if three_group else 4 if n >= processors * 256 or streamed else 2
        repeats = [
            r
            for r in (1, 2, 3, 4)
            if _ceil_div(m, 16 * r) * 16 * r <= padded * 1.125
            and _ceil_div(m, 16 * r) * _ceil_div(n, 16 * wn) * wn * min(4, _divisor(k // 32, max(1, 8 // r)))
            >= (16 if r == 3 else 12) * processors
        ]
        if not repeats and three_group:
            compact = _ceil_div(m, 32) * 32 <= padded * 1.125
            return dict(
                REGISTER_DEFAULTS,
                waves_m=1,
                waves_n=1 if compact else 2,
                repeat_m=2 if compact else 1,
                repeat_n=1,
                k_unroll=3,
            )
        repeat = max(repeats, default=1)
        if three_group and repeat >= 3:
            compact = _ceil_div(m, 32) * 32 <= padded * 1.125
            previous_repeat = 2 if compact else 1
            if _ceil_div(m, 16 * previous_repeat) < 2 * _ceil_div(m, 16 * repeat):
                return dict(
                    REGISTER_DEFAULTS,
                    waves_m=1,
                    waves_n=1 if compact else 2,
                    repeat_m=previous_repeat,
                    repeat_n=1,
                    k_unroll=3,
                )
        if streamed and k >= 8192:
            repeat = 1
        if repeat == 1 and three_group:
            wn = 2
        while wn > 1 and _ceil_div(m, 16 * repeat) * _ceil_div(n, 16 * wn) < processors:
            wn //= 2
        return dict(
            REGISTER_DEFAULTS,
            waves_m=1,
            waves_n=wn,
            repeat_m=repeat,
            repeat_n=1,
            k_unroll=3 if three_group and repeat == 1 else _divisor(k // 32, max(1, 8 // repeat)),
            table_mode="lut" if not three_group and not streamed and k >= 4096 and n < processors * 256 else "alu",
        )
    weight_bytes = _weight_bytes(n, k)
    if _three_group_reuse(m, n, k, processors, cache_bytes):
        lookahead = _divisor(k // 32, 8)
        if (
            m <= 32
            and k >= 4096
            and k % 64
            and lookahead >= 5
            and cache_bytes
            and _weight_bytes(n, k) <= cache_bytes // 4
            and _ceil_div(m, 16) * _ceil_div(n, 16) >= 4 * processors
        ):
            return dict(
                REGISTER_DEFAULTS, waves_m=1, waves_n=1, repeat_m=1, repeat_n=1, k_unroll=lookahead, table_mode="lut"
            )
        compact = _ceil_div(m, 32) * 32 <= _ceil_div(m, 16) * 16 * 1.125
        return dict(
            REGISTER_DEFAULTS,
            waves_m=1,
            waves_n=1 if compact else 2,
            repeat_m=2 if compact else 1,
            repeat_n=1,
            k_unroll=3,
            table_mode="alu",
        )
    if m not in (1, 2, 4, 8, 16) and 16 < m <= 32 and n >= processors * 256 and k % 128:
        # Repeats supply independent WMMA work while bounding the K body.
        return dict(
            REGISTER_DEFAULTS,
            waves_m=1,
            waves_n=4,
            repeat_m=2,
            repeat_n=1 if k % 64 == 0 else 2,
            k_unroll=2 if k % 64 == 0 else 1,
            table_mode="alu",
            fast_epilogue=k % 64 == 0,
        )
    if _cached_reuse(m, n, k, processors, cache_bytes):
        return dict(
            REGISTER_DEFAULTS,
            waves_m=1,
            waves_n=1,
            repeat_m=2,
            repeat_n=2,
            k_unroll=1,
            table_mode="alu",
        )
    # Wider N waves amortize address/decode work for streaming weight tiles.
    wn = 4 if n >= processors * 64 or not cache_bytes or weight_bytes >= cache_bytes // 4 or m > 16 else 2
    while wn > 1 and _ceil_div(m, 16) * _ceil_div(n, 16 * wn) < processors // 2:
        wn //= 2
    unroll = _divisor(k // 32, 8)
    if unroll == 1 and k >= 1024 and (m in (1, 2, 4, 8, 16) or m % 16 == 0):
        unroll = _divisor(k // 32, 16)
    if m not in (1, 2, 4, 8, 16) and m % 16:
        bounded = _divisor(k // 32, 5 if m <= 16 else 4)
        if bounded > 1:
            unroll = bounded
    if m in (8, 16) and unroll != 8:
        bounded = _divisor(k // 32, 4)
        if bounded > 1:
            unroll = bounded
    table = "lut" if m in (8, 16) and k >= 8192 and n <= k // 2 else "alu"
    fast = True
    if m in (1, 2, 4, 8, 16) and _divisor(k // 32, 8) != 8:
        extended = _divisor(k // 32, 16)
        if extended > _divisor(k // 32, 8):
            wn, unroll = 1 if extended > 8 else 2, extended
            if m in (8, 16) and extended > 8:
                table = "alu" if _divisor(k // 32, 8) == 1 else "lut"
    if (
        4 < m <= 16
        and k >= 16384
        and n <= k // 2
        and cache_bytes
        and weight_bytes >= cache_bytes // 2
        and _divisor(k // 32, 8) > 1
    ):
        wn, unroll, table, fast = 1, _divisor(k // 32, 8), "lut", False
    if m in (8, 16) and cache_bytes and weight_bytes >= cache_bytes and k >= 8192:
        wn, unroll, table, fast = 1, _divisor(k // 32, 4), "alu", True
    if m in (8, 16) and unroll == 1 and k >= 8192:
        wn, table, fast = 2, "alu", False
    if m in (8, 16) and wn == 4 and unroll == 8 and n % 64:
        wn, unroll, table = 1, _divisor(k // 32, 4), "alu"
    if (
        m in (8, 16)
        and k % 128 == 64
        and unroll == 3
        and n <= processors * 64
        and cache_bytes
        and weight_bytes <= cache_bytes // 4
    ):
        wn, unroll, table = 1, _divisor(k // 32, 8), "alu"
    repeat_m = 1
    if m >= 32 and unroll <= 2:
        # When K has little legal unrolling, reuse each decoded B fragment
        # over more rows instead of replicating loop/decode work across CTAs.
        repeat_m = 4 if n >= processors * 64 and m > 32 else 2
        if n < processors * 128:
            wn = 2
    return dict(
        REGISTER_DEFAULTS,
        waves_m=1,
        waves_n=wn,
        repeat_m=repeat_m,
        repeat_n=1,
        k_unroll=unroll,
        table_mode=table,
        fast_epilogue=fast,
    )


def _m_tile(m, n, processors, bn, wide_tile=False, reuse_groups=2):
    # Minimize padding first for partial prefill tiles, then favor B reuse.
    candidates = (64, 96, 128, 192, 256)
    min_rows = min(_ceil_div(m, b) * b for b in candidates)
    viable = [
        b
        for b in candidates
        if _ceil_div(m, b) * b <= min_rows * 1.125
        and _ceil_div(m, b) * _ceil_div(n, bn * (2 if b == 256 and wide_tile else 1))
        >= processors * (reuse_groups if b >= 192 else 1)
    ]
    return max(viable) if viable else min(candidates)


# Launch configurations of the hybrid strategy, not additional algorithms.
# M/N/K tile, M/N waves, stages, lookahead, M grouping, fragment LDS,
# scheduling, operand scheduling, epilogue prefetch and nibble conversion.
_HYBRID_PARAMETERS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "waves_m",
    "waves_n",
    "stages",
    "operand_prefetch",
    "cache_group_m",
    "fragment_lds",
    "schedule",
    "operand_schedule",
    "epilogue_prefetch",
    "table_mode",
)
_HYBRID_PRESETS = {
    "ragged_n": (32, 32, 256, 2, 2, 2, 15, 1, False, "groups", "none", False, "alu"),
    "decode": (16, 64, 256, 1, 4, 2, 2, 1, True, "none", "none", False, "lut"),
    "microbatch": (32, 32, 256, 2, 2, 2, 2, 1, True, "none", "none", False, "lut"),
    "skinny": (64, 32, 256, 4, 1, 2, 2, 4, True, "groups", "none", False, "alu"),
    "skinny_reuse": (128, 32, 256, 4, 1, 2, 2, 4, True, "groups", "none", False, "alu"),
    "narrow_prefill": (128, 32, 256, 4, 1, 2, 6, 1, True, "groups", "none", True, "lut"),
    "small_prefill": (64, 32, 256, 4, 1, 2, 2, 1, True, "none", "none", True, "lut"),
    "small_wide": (64, 64, 128, 4, 1, 2, 2, 1, True, "groups", "none", False, "lut"),
    "short_k": (64, 32, 32, 2, 2, 2, 1, 1, True, "barrier", "none", False, "alu"),
    "partial96": (96, 64, 256, 2, 2, 1, 13, 1, True, "groups", "none", False, "alu"),
    "pipelined_partial": (128, 64, 128, 4, 1, 2, 4, 1, True, "groups", "none", True, "lut"),
    "medium_prefill": (128, 64, 128, 4, 1, 2, 2, 1, True, "groups", "none", False, "lut"),
    "partial192": (192, 64, 64, 4, 1, 2, 2, 1, False, "groups", "none", True, "alu"),
    "long_k": (256, 64, 128, 8, 1, 2, 7, 2, False, "groups", "none", True, "alu"),
    "large_reuse": (256, 128, 128, 4, 2, 2, 2, 2, True, "barrier", "barrier", False, "lut"),
}


def _hybrid(m, n, k, processors, cache_bytes, policy="auto"):
    narrow_tail = n <= 32 and n % 16 and m not in (1, 2, 4, 8, 16) and m % 16
    general_rows = 128 if k % 128 == 0 and _ceil_div(m, 128) >= processors // 2 else 256
    wide_skinny = (
        n > 64
        and k >= 8192
        and k % 256 == 128
        and _ceil_div(n, 64) * 64 <= _ceil_div(n, 32) * 32 * 1.125
        and _ceil_div(m, 256) * _ceil_div(n, 64) >= 2 * processors
    )
    if narrow_tail and m <= 64 and k >= 4096:
        preset = "ragged_n"
    elif m <= 32:
        preset = "decode" if m <= 16 and n >= 128 else "microbatch"
    elif n > 128 and _narrow(n, k, processors) and m <= 256 and (k <= 8192 or m > 128) and k % 256 == 0:
        preset = "microbatch" if m <= 128 else "small_prefill"
    elif (
        n <= 128
        and not wide_skinny
        or _narrow(n, k, processors)
        and (
            _ceil_div(m, general_rows) * _ceil_div(n, 64) < processors
            or _ceil_div(m, 256) * _ceil_div(n, 64) == processors
            and k % 128 != 0
        )
    ):
        grid = _ceil_div(m, 128) * _ceil_div(n, 32)
        preset = "skinny_reuse" if grid >= 2 * processors else "skinny"
    elif processors <= _ceil_div(n, 32) <= processors * 2 and 64 < m <= 128:
        preset = "narrow_prefill"
    elif k % 64 and m <= 128 and _ceil_div(m, 64) * _ceil_div(n, 32) <= 8 * processors:
        preset = "short_k"
    else:
        # An eight-wave tile can retain enough active waves with fewer CTAs
        # when full K128 iterations amortize the longer cooperative pipeline.
        reuse_groups = 0.75 if k >= 8192 and k % 128 == 0 and n <= processors * 8 else 1 if k >= 8192 else 2
        bm = _m_tile(m, n, processors, 64, n >= k, reuse_groups)
        if bm <= 64:
            preset = "small_wide" if _ceil_div(n, 64) >= 2 * processors else "small_prefill"
        elif bm == 96:
            hot_fit = policy == "hot" and cache_bytes and _weight_bytes(n, k) <= cache_bytes // 2
            preset = "pipelined_partial" if k >= 8192 and k & (k - 1) == 0 and not hot_fit else "partial96"
        elif bm == 128:
            preset = "medium_prefill"
        elif bm == 192:
            preset = "partial192"
        else:
            preset = "long_k" if n < k or m <= 256 and k % 128 == 64 and n >= 2 * k else "large_reuse"
    cfg = dict(HYBRID_DEFAULTS, **dict(zip(_HYBRID_PARAMETERS, _HYBRID_PRESETS[preset])))
    if preset == "microbatch" and m > 32:
        cfg["epilogue_prefetch"] = True
    if m <= 16 and n < 128 and preset == "microbatch":
        cfg["tile_m"], cfg["waves_m"] = 16, 1
    # The cooperative copy layout requires power-of-two K tiles. Register
    # unrolling can still use odd divisors of K32 groups.
    cfg["tile_k"] = max(b for b in (32, 64, 128, 256) if b <= cfg["tile_k"] and k % b == 0)
    cfg["operand_prefetch"] = min(cfg["operand_prefetch"], cfg["tile_k"] // 16 - 1)
    if preset == "medium_prefill" and cfg["tile_k"] <= 64:
        cfg.update(
            table_mode="alu",
            operand_prefetch=cfg["tile_k"] // 16 - 1,
            schedule="barrier" if cfg["tile_k"] == 32 else "groups",
        )
    if preset == "small_wide" and cfg["tile_k"] == 32:
        cfg.update(schedule="none", epilogue_prefetch=True)
    if preset == "partial96" and cfg["tile_k"] <= 64:
        cfg.update(stages=2, operand_prefetch=min(2, cfg["tile_k"] // 16 - 1))
    if preset == "long_k" and m <= 256 and cfg["tile_k"] == 64:
        cfg.update(schedule="barrier")
    if (
        cfg["tile_k"] == 32
        and 32 < n <= 128
        and _ceil_div(n, 128) * 128 <= _ceil_div(n, 32) * 32 * 1.125
        and _ceil_div(m, 128) * _ceil_div(n, 128) >= 2 * processors
    ):
        cfg.update(
            tile_m=128,
            tile_n=128,
            waves_m=4,
            waves_n=1,
            cache_group_m=4,
            fragment_lds=True,
            table_mode="alu",
            schedule="barrier",
            epilogue_prefetch=False,
        )
    if (
        cfg["tile_k"] == 32
        and m >= 256
        and k >= 2048
        and n >= processors * 8
        and _ceil_div(n, 128) * 128 <= _ceil_div(n, 64) * 64 * 1.125
        and _ceil_div(m, 256) * _ceil_div(n, 128) >= processors * (2 if n < processors * 32 else 1.5)
    ):
        cfg.update(
            tile_m=256,
            tile_n=128,
            waves_m=4,
            waves_n=2,
            stages=2,
            operand_prefetch=1,
            cache_group_m=2 if n < processors * 32 else 4,
            fragment_lds=True,
            table_mode="lut" if n < processors * 32 else "alu",
            schedule="barrier",
            operand_schedule="barrier" if n < processors * 32 else "none",
            epilogue_prefetch=False,
        )
    return cfg


@lru_cache(maxsize=4096)
def _selected(m, n, k, mode, policy, processors, cache_bytes, lds_bytes):
    if min(m, n, k, processors, lds_bytes) <= 0 or k % 32:
        raise ValueError("M/N and hardware capacities must be positive; K must be a multiple of32")
    if policy not in ("auto", "hot", "rotating"):
        raise ValueError("unsupported cache_policy")
    if mode not in ("e4m3", "e5m2", "grouped"):
        raise ValueError("unsupported preparation mode")
    if cache_bytes < 0:
        raise ValueError("cache capacity must be nonnegative")
    if mode == "grouped":
        cfg = dict(
            REGISTER_DEFAULTS,
            waves_n=4,
            repeat_m=1 if m <= 16 else 2 if m <= 32 else 4,
            repeat_n=2 if m >= 256 else 1,
            k_unroll=1,
        )
    else:
        weight_bytes = _weight_bytes(n, k)
        skinny_groups = _ceil_div(m, 64) * _ceil_div(n, 32)
        small_reused = m <= 64 and (
            policy == "hot" and cache_bytes and weight_bytes < cache_bytes // 64 and m * n <= processors * 2048
        )
        static_decode = m in (1, 2, 4, 8, 16)
        narrow_tail = (
            n <= 32 and n % 16 and m not in (1, 2, 4, 8, 16) and m % 16 and k >= 4096 and (k % 64 or k % 128 == 0)
        )
        short_cached = _cached_reuse(m, n, k, processors, cache_bytes)
        odd_loop = k % 128 != 0 and (
            m <= 32 or k <= 4096 and (m <= 64 and k % 64 != 0 or k % 64 != 0 and m * n < processors * 8192)
        )
        if (
            static_decode
            or _long_skinny_register(m, n, k, processors)
            or _long_thin_register(m, n, k, cache_bytes)
            or _row_register(m, n, k, processors, cache_bytes)
            or _three_group_reuse(m, n, k, processors, cache_bytes)
            or short_cached
            or odd_loop
            or _narrow(n, k, processors)
            and skinny_groups < (4 if n > 128 and k % 64 else 2 if n > 128 else 1.25) * processors + bool(k % 128)
            and not (
                n > 128
                and m > 64
                and (m % 16 or m >= 256)
                and (k <= 8192 or 128 < m <= 256)
                and k % 256 == 0
                and skinny_groups >= processors // 2
            )
            or small_reused
        ) and not narrow_tail:
            cfg = _register(m, n, k, processors, cache_bytes, policy)
        else:
            cfg = _hybrid(m, n, k, processors, cache_bytes, policy)
    cfg = _normalize_config(cfg, m, k)
    while cfg["engine"] == "hybrid" and _geometry(cfg)[-1] > lds_bytes and cfg["stages"] > 1:
        cfg["stages"] -= 1
    if not _legal_config(cfg, m, n, k) or _geometry(cfg)[-1] > lds_bytes:
        cfg = dict(REGISTER_DEFAULTS, waves_m=1, waves_n=1, repeat_m=1, repeat_n=1, k_unroll=1)
    return tuple(sorted(cfg.items()))


def choose_config(
    m: int,
    n: int,
    k: int,
    mode: str = "e4m3",
    *,
    cache_policy: str = "auto",
    num_processors: int = 32,
    cache_size_bytes: int = 64 * 1024**2,
    lds_size_bytes: int = 65536,
) -> dict:
    """Select one configuration using shape and explicit GPU capacities.

    Capacity defaults describe R9700. Callers discover their device once at
    initialization and pass identical capacities to execution and warmup.
    No GPU query, compilation, file lookup or timing runs in this function.
    """
    return dict(
        _selected(
            m,
            n,
            k,
            mode,
            cache_policy,
            num_processors,
            cache_size_bytes,
            lds_size_bytes,
        )
    )


def rank_configurations(m, n, k, *, cache_policy="auto", limit=None, **hardware):
    """Explain the single analytical choice without device work or timing."""
    rows = [
        dict(
            config=choose_config(m, n, k, cache_policy=cache_policy, **hardware),
            priority=0,
        )
    ]
    return rows if limit is None else rows[:limit]
