# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Bounded register/hybrid alternatives for CPU cost checks."""


def bounded_candidates(
    m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults
):
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
    bm = (
        16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128 if m < 256 else 256
    )
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


def expanded_candidates(
    m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults
):
    """Retain the old bank and add narrow-M and smaller-LDS alternatives."""
    result = bounded_candidates(
        m, n, k, ranked, adapt, legal, register_defaults, hybrid_defaults
    )
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
