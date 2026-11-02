# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU tile, compiler-mode and layout features for configuration scoring."""

import math
import struct

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


def configuration_features(m, n, k, config):
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


def predict_cost(model, values):
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
