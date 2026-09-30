#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Offline tile tuner for FlyDSL conv3d (lightweight native port of aiter's).

Reads an untuned CSV, sweeps ``conv3d_policy`` configs, writes winners to a
tuned CSV. Same candidate set, CSV columns, err=0 bar, splitK pinning, and
NDHWC timing as aiter ``csrc/flydsl_conv3d/conv3d_tune.py``.

    python -m kernels.conv.conv3d_tune \\
        -i kernels/conv/configs/qwenimage_vae_bf16_untuned_conv3d.csv \\
        -o kernels/conv/configs/qwenimage_vae_bf16_tuned_conv3d.csv
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from flydsl.autotune import do_bench

from .conv3d_gfx950_utils import out_extent
from .conv3d_implicit import (
    LIBTYPE_FLYDSL,
    TUNED_DEVICE_COLUMNS,
    TUNED_KEY_COLUMNS,
    TUNED_LIBTYPE_COLUMN,
    TUNED_RESULT_COLUMNS,
    _borrow_tuned_tile,
    _check_supported_arch,
    _get_cu_num,
    _get_gfx,
    _is_matmul_fast_path,
    _load_tuned_table,
    _pad_channels,
    _parse_tuned_bool,
    _resolve_splitk,
    _tuned_rows_by_layer,
    flydsl_conv_implicit,
)
from .conv3d_policy import get_flydsl_conv3d_configs, tile_kernel_name

log = logging.getLogger("conv3d_tune")

SHAPE_KEYS = list(TUNED_KEY_COLUMNS)
KEYS = [*TUNED_DEVICE_COLUMNS, *SHAPE_KEYS]
RESULT_COLS = [
    TUNED_LIBTYPE_COLUMN,
    *TUNED_RESULT_COLUMNS,
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]
COLUMNS = [*KEYS, *RESULT_COLS]

RUN_CONFIG_REPS = 3
RTOL = ATOL = 2e-2
LAYOUT = "NDHWC"
INVALID_TIME = -1.0


def _row_params(row):
    return {
        "stride": (int(row["stride_d"]), int(row["stride_h"]), int(row["stride_w"])),
        "padding": (int(row["pad_d"]), int(row["pad_h"]), int(row["pad_w"])),
        "dilation": (int(row["dil_d"]), int(row["dil_h"]), int(row["dil_w"])),
        "groups": int(row["groups"]),
    }


def _shape_key(row):
    return tuple(_parse_tuned_bool(v) if c == "bias" else int(v) for c, v in zip(SHAPE_KEYS, row))


def _gemm_dims(kv):
    do = out_extent(int(kv["D"]), int(kv["pad_d"]), int(kv["dil_d"]), int(kv["kT"]), int(kv["stride_d"]))
    ho = out_extent(int(kv["H"]), int(kv["pad_h"]), int(kv["dil_h"]), int(kv["kH"]), int(kv["stride_h"]))
    wo = out_extent(int(kv["W"]), int(kv["pad_w"]), int(kv["dil_w"]), int(kv["kW"]), int(kv["stride_w"]))
    groups = int(kv["groups"])
    m = int(kv["N"]) * do * ho * wo
    n = int(kv["K"]) // groups
    k = (int(kv["C"]) // groups) * int(kv["kT"]) * int(kv["kH"]) * int(kv["kW"])
    return m, n, k, (do, ho, wo)


def generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias, seed=0, device=None):
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    x = torch.randn((n, d, h, w, c), device=device, dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, kt, kh, kw), device=device, dtype=torch.bfloat16)
    bias = torch.randn((k,), device=device, dtype=torch.float32) if has_bias else None
    return {"x": x, "weight": weight, "bias": bias}


def run_flydsl_conv3d(x, weight, bias, params, tile, wgm, splitk):
    return flydsl_conv_implicit(
        x,
        weight,
        bias=bias,
        tile=tile,
        wgm=wgm,
        splitk=splitk,
        input_layout=LAYOUT,
        output_layout=LAYOUT,
        **params,
    )


def conv3d_ref(x, weight, bias, params):
    ref_bias = bias.to(x.dtype) if bias is not None else None
    y = F.conv3d(x.permute(0, 4, 1, 2, 3).contiguous(), weight, bias=ref_bias, **params)
    return y.permute(0, 2, 3, 4, 1).contiguous()


def _err_ratio(out, ref):
    if out.shape != ref.shape:
        return 1.0
    close = torch.isclose(out, ref, rtol=RTOL, atol=ATOL)
    return float((~close).sum().item()) / max(out.numel(), 1)


def _calculate(kv, us):
    if us <= 0:
        return 0.0, 0.0
    m, n, k, (do, ho, wo) = _gemm_dims(kv)
    tflops = round(m * n * k * 2 / (us * 1e6), 2)
    x_elems = int(kv["N"]) * int(kv["C"]) * int(kv["D"]) * int(kv["H"]) * int(kv["W"])
    w_elems = int(kv["K"]) * (int(kv["C"]) // int(kv["groups"])) * int(kv["kT"]) * int(kv["kH"]) * int(kv["kW"])
    y_elems = int(kv["N"]) * int(kv["K"]) * do * ho * wo
    bw = round((x_elems + w_elems + y_elems) * 2 / (us * 1e-6) / 1e9, 2)
    return tflops, bw


def _ramp_clocks(seconds=2.0):
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(20):
            a = torch.mm(a, a).clamp_(-1.0, 1.0)
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def _clear_caches():
    _load_tuned_table.cache_clear()
    _tuned_rows_by_layer.cache_clear()
    from . import conv3d_implicit as mod

    mod._TUNED_LOOKUP_LOGGED.clear()


def _shape_configs(kv, max_configs, gfx, cu_num):
    m_gemm, n_gemm, _, _ = _gemm_dims(kv)
    groups = int(kv["groups"])
    configs = get_flydsl_conv3d_configs(m_gemm, n_gemm, groups, cu_num, max_configs=max_configs)
    shape_key = tuple(_parse_tuned_bool(kv[c]) if c == "bias" else int(kv[c]) for c in SHAPE_KEYS)
    borrowed = _borrow_tuned_tile((gfx, cu_num), shape_key)
    if borrowed is not None and (*borrowed[0], borrowed[1]) not in configs:
        configs.append((*borrowed[0], borrowed[1]))
    return configs, m_gemm


def _bench_one(kv, tile_m, tile_n, wave_m, wave_n, wgm, splitk, err_ratio_bar):
    n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
    k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
    groups = int(kv["groups"])
    has_bias = _parse_tuned_bool(kv["bias"])
    params = _row_params(kv)
    tile = (tile_m, tile_n, wave_m, wave_n)
    data = generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias)
    try:
        out = run_flydsl_conv3d(data["x"], data["weight"], data["bias"], params, tile, wgm, splitk)
        ref = conv3d_ref(data["x"], data["weight"], data["bias"], params)
        err = _err_ratio(out, ref)
        if err > err_ratio_bar:
            return INVALID_TIME, err
        us = do_bench(
            lambda: run_flydsl_conv3d(data["x"], data["weight"], data["bias"], params, tile, wgm, splitk),
            warmup=5,
            rep=20,
        )
        return float(us), err
    except Exception as exc:  # noqa: BLE001
        log.debug("candidate failed: %s", exc)
        return INVALID_TIME, 1.0


def _tune_shape(kv, max_configs, err_ratio_bar, gfx, cu_num):
    configs, m_gemm = _shape_configs(kv, max_configs, gfx, cu_num)
    groups = int(kv["groups"])
    c = int(kv["C"])
    k = int(kv["K"])
    kt, kh, kw = int(kv["kT"]), int(kv["kH"]), int(kv["kW"])
    cgp = _pad_channels(c // groups)
    crs = cgp * kt * kh * kw

    best = None
    profile_rows = []
    for tile_m, tile_n, wave_m, wave_n, wgm in configs:
        tile = (tile_m, tile_n, wave_m, wave_n)
        sk = _resolve_splitk(None, m_gemm, crs, k, None, tile, groups, num_cu=cu_num)
        us, err = _bench_one(kv, tile_m, tile_n, wave_m, wave_n, wgm, sk, err_ratio_bar)
        name = tile_kernel_name(tile_m, tile_n, wave_m, wave_n, wgm)
        tflops, bw = _calculate(kv, us)
        row = {
            **{c: kv[c] for c in SHAPE_KEYS},
            "gfx": gfx,
            "cu_num": cu_num,
            TUNED_LIBTYPE_COLUMN: LIBTYPE_FLYDSL,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "wave_m": wave_m,
            "wave_n": wave_n,
            "wgm": wgm,
            "splitK": sk,
            "us": us,
            "kernelName": name,
            "err_ratio": err,
            "tflops": tflops,
            "bw": bw,
        }
        profile_rows.append(row)
        if us == INVALID_TIME:
            continue
        if best is None or us < best["us"]:
            best = row
    return best, profile_rows


def _load_untuned(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in SHAPE_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"untuned CSV missing columns {missing}")
    return df


def _upsert_tuned(path, winners, untuned_order):
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        old.columns = old.columns.str.strip()
    else:
        old = pd.DataFrame(columns=COLUMNS)

    by_key = {}
    if not old.empty:
        for _, row in old.iterrows():
            key = (str(row["gfx"]), int(row["cu_num"])) + _shape_key(tuple(row[c] for c in SHAPE_KEYS))
            by_key[key] = row.to_dict()

    for row in winners:
        key = (str(row["gfx"]), int(row["cu_num"])) + _shape_key(tuple(row[c] for c in SHAPE_KEYS))
        by_key[key] = row

    rows = list(by_key.values())
    order = {_shape_key(tuple(r[c] for c in SHAPE_KEYS)): i for i, r in enumerate(untuned_order)}
    rows.sort(key=lambda r: order.get(_shape_key(tuple(r[c] for c in SHAPE_KEYS)), len(order)))
    out = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _worker(gpu, shapes, max_configs, err_ratio_bar, result_queue):
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)
    _check_supported_arch()
    _ramp_clocks()
    gfx, cu_num = _get_gfx(), _get_cu_num()
    winners, profiles = [], []
    for kv in shapes:
        best, prof = _tune_shape(kv, max_configs, err_ratio_bar, gfx, cu_num)
        if best is not None:
            winners.append(best)
            log.info(
                "winner %sx%sx%sx%sx%s->%s tile=(%s,%s,%s,%s) wgm=%s us=%s",
                kv["N"],
                kv["C"],
                kv["D"],
                kv["H"],
                kv["W"],
                kv["K"],
                best["tile_m"],
                best["tile_n"],
                best["wave_m"],
                best["wave_n"],
                best["wgm"],
                best["us"],
            )
        else:
            log.warning("no legal candidate for %s", {c: kv[c] for c in SHAPE_KEYS})
        profiles.extend(prof)
    result_queue.put((winners, profiles))


def _run_config(untunedf, err_ratio_bar):
    _clear_caches()
    _ramp_clocks()
    results = []
    for _, row in untunedf.iterrows():
        kv = {c: row[c] for c in SHAPE_KEYS}
        n, c, d, h, w = (int(kv[x]) for x in ("N", "C", "D", "H", "W"))
        k, kt, kh, kw = (int(kv[x]) for x in ("K", "kT", "kH", "kW"))
        groups = int(kv["groups"])
        has_bias = _parse_tuned_bool(kv["bias"])
        params = _row_params(kv)
        shape = f"{n}x{c}x{d}x{h}x{w}->{k} {kt}x{kh}x{kw}"
        try:
            data = generate_data(n, c, d, h, w, k, kt, kh, kw, groups, has_bias)
            out, us = None, float("inf")
            for _ in range(RUN_CONFIG_REPS):
                t = do_bench(
                    lambda: flydsl_conv_implicit(
                        data["x"],
                        data["weight"],
                        bias=data["bias"],
                        input_layout=LAYOUT,
                        output_layout=LAYOUT,
                        **params,
                    ),
                    warmup=5,
                    rep=20,
                )
                if t < us:
                    us = t
                    out = flydsl_conv_implicit(
                        data["x"],
                        data["weight"],
                        bias=data["bias"],
                        input_layout=LAYOUT,
                        output_layout=LAYOUT,
                        **params,
                    )
            ref = conv3d_ref(data["x"], data["weight"], data["bias"], params)
            ok = _err_ratio(out, ref) <= err_ratio_bar
            results.append({"shape": shape, "e2e_us": round(us, 4), "status": "ok" if ok else "mismatch"})
        except Exception as exc:  # noqa: BLE001
            results.append({"shape": shape, "e2e_us": -1, "status": f"error: {exc}"})
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FlyDSL conv3d bf16 tile tuner")
    p.add_argument("-i", "--untune_file", required=True, help="untuned CSV path")
    p.add_argument("-o", "--tune_file", default=None, help="tuned CSV output path")
    p.add_argument("--max_configs", type=int, default=96)
    p.add_argument("--errRatio", type=float, default=0.0)
    p.add_argument("--run_config", nargs="?", const="", default=None, help="benchmark production entry only")
    p.add_argument("-o2", "--profile_file", default=None, help="save every candidate")
    p.add_argument("--gpus", default=None, help="comma-separated GPU ids for parallel tuning")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    _check_supported_arch()
    untunedf = _load_untuned(args.untune_file)
    skip = untunedf.apply(_is_matmul_fast_path, axis=1)
    if skip.any():
        log.info("skipping %d 1x1 matmul fast-path shapes", int(skip.sum()))
        untunedf = untunedf[~skip].reset_index(drop=True)

    if args.run_config is not None:
        if args.run_config:
            os.environ["FLYDSL_CONV3D_BF16_CONFIG"] = args.run_config
            _clear_caches()
        results = _run_config(untunedf, args.errRatio)
        for r in results:
            print(f"{r['shape']}: {r['e2e_us']} us  {r['status']}")
        return 0 if all(r["status"] == "ok" for r in results) else 1

    if not args.tune_file:
        raise SystemExit("-o/--tune_file is required unless --run_config is set")

    gfx, cu_num = _get_gfx(), _get_cu_num()
    untunedf = untunedf.copy()
    untunedf["gfx"] = gfx
    untunedf["cu_num"] = cu_num

    shapes = [{c: row[c] for c in SHAPE_KEYS} for _, row in untunedf.iterrows()]
    if not shapes:
        log.info("no shapes to tune")
        return 0

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else [torch.cuda.current_device()]
    if len(gpus) == 1:
        _ramp_clocks()
        winners, profiles = [], []
        for kv in shapes:
            best, prof = _tune_shape(kv, args.max_configs, args.errRatio, gfx, cu_num)
            if best is not None:
                winners.append(best)
            profiles.extend(prof)
    else:
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        shards = [shapes[i :: len(gpus)] for i in range(len(gpus))]
        procs = []
        for gpu, shard in zip(gpus, shards):
            if not shard:
                continue
            p = ctx.Process(
                target=_worker,
                args=(gpu, shard, args.max_configs, args.errRatio, q),
            )
            p.start()
            procs.append(p)
        winners, profiles = [], []
        for _ in procs:
            w, p = q.get()
            winners.extend(w)
            profiles.extend(p)
        for p in procs:
            p.join()
            if p.exitcode:
                raise RuntimeError(f"tuner worker exited with {p.exitcode}")

    untuned_order = [{c: row[c] for c in SHAPE_KEYS} for _, row in untunedf.iterrows()]
    _upsert_tuned(args.tune_file, winners, untuned_order)
    if args.profile_file:
        pd.DataFrame(profiles).to_csv(args.profile_file, index=False)
    log.info("wrote %d winners to %s", len(winners), args.tune_file)
    return 0 if len(winners) == len(shapes) else 1


if __name__ == "__main__":
    # Allow `python -m kernels.conv.conv3d_tune` from the FlyDSL root.
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    raise SystemExit(main())
