#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Qwen-Image VAE conv: three-tier on-device comparison vs hipBLASLt / MIOpen.

Each of the 18 shapes runs in a fresh process so JIT / autotune state cannot
leak. Writes ``fair_baseline.json`` next to this script.

Usage::

    python tests/perf/qwenimage_vae_conv/bench_fair.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "fair_baseline.json"

# sid, cin, cout, hin, stride, padding, freq, path
SHAPES = [
    ("enc_conv_in", 3, 96, 1024, 1, 1, 1, "1024"),
    ("enc_e0_res__dec_d3_res", 96, 96, 1024, 1, 1, 10, "1024"),
    ("enc_e1_res1", 96, 192, 512, 1, 1, 1, "1024"),
    ("enc_e1_res2__dec_d2_res", 192, 192, 512, 1, 1, 9, "1024"),
    ("enc_e2_res1__dec_d1_res1", 192, 384, 256, 1, 1, 2, "1024"),
    ("enc_e2_res2__dec_d1_res", 384, 384, 256, 1, 1, 8, "1024"),
    ("enc_e3_mid__dec_mid_d0", 384, 384, 128, 1, 1, 18, "1024"),
    ("enc_conv_out", 384, 32, 128, 1, 1, 1, "1024"),
    ("dec_conv_in", 16, 384, 128, 1, 1, 1, "1024"),
    ("dec_conv_out", 96, 3, 1024, 1, 1, 1, "1024"),
    ("enc_e0_downsample", 96, 96, 1025, 2, 0, 1, "1024"),
    ("enc_e1_downsample_spatial", 192, 192, 513, 2, 0, 1, "1024"),
    ("enc_e2_downsample_spatial", 384, 384, 257, 2, 0, 1, "1024"),
    ("dec_d0_upsample", 384, 192, 256, 1, 1, 1, "1024"),
    ("dec_d1_upsample", 384, 192, 512, 1, 1, 1, "1024"),
    ("dec_d2_upsample", 192, 96, 1024, 1, 1, 1, "1024"),
    ("dec_bottleneck_1328", 384, 384, 166, 1, 1, 18, "1328"),
    ("dec_d3_res_hot_1328", 96, 96, 1328, 1, 1, 10, "1328"),
]

FIELDS = [
    ("sid", str),
    ("cin", int),
    ("cout", int),
    ("hin", int),
    ("stride", int),
    ("freq", int),
    ("path", str),
    ("M", int),
    ("N", int),
    ("K", int),
    ("flops", float),
    ("fly_def", float),
    ("gemm_def", float),
    ("tr_def", float),
    ("fly_best", float),
    ("gemm_best", float),
    ("tr_best", float),
    ("tile", str),
    ("mm", float),
    ("im2col", float),
    ("unfold_mm", float),
    ("mio", float),
]

R = S = 3
ITERS, TRIALS = 10, 2
NAN = float("nan")


def flydsl_root() -> Path:
    env = os.environ.get("FLYDSL_ROOT")
    if env:
        return Path(env).resolve()
    return HERE.parents[2]


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(flydsl_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["FLYDSL_CONV3D_AUTOTUNE"] = "0"
    return env


def gpu(torch, profile, ProfilerActivity, call):
    best, best_per = None, None
    for _ in range(TRIALS):
        for _ in range(4):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                call()
            torch.cuda.synchronize()
        per = {}
        for e in prof.key_averages():
            if (
                e.device_type != torch.autograd.DeviceType.CUDA
                or e.self_device_time_total <= 0
            ):
                continue
            k = e.key.split("(")[0][:40]
            per[k] = per.get(k, 0.0) + e.self_device_time_total / ITERS
        t = sum(per.values())
        if best is None or t < best:
            best, best_per = t, per
    return best, best_per


def split_fly(per):
    g = sum(v for k, v in per.items() if "conv3d_implicit_kernel" in k)
    t = sum(v for k, v in per.items() if "transpose_kernel" in k)
    return g, t


def run_one(sid, cin, cout, hin, stride, padding, freq, path) -> None:
    sys.path.insert(0, str(flydsl_root()))
    os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from kernels.conv.conv3d_implicit import DEFAULT_TILE, conv3d_implicit

    cin, cout, hin, stride, padding, freq = map(
        int, (cin, cout, hin, stride, padding, freq)
    )
    candidates = [
        tuple(DEFAULT_TILE),
        (128, 256, 2, 4),
        (256, 128, 2, 4),
        (256, 256, 4, 4),
        (128, 128, 4, 2),
    ]

    torch.manual_seed(0)
    x4 = torch.randn((1, cin, hin, hin), device="cuda", dtype=torch.bfloat16)
    w4 = torch.randn((cout, cin, R, S), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((cout,), device="cuda", dtype=torch.float32)
    bbf = b.to(torch.bfloat16)
    ref = F.conv2d(x4, w4, bias=bbf, stride=stride, padding=padding)
    p = ref.shape[2]
    M, N, K = p * p, cout, cin * R * S
    flops = 2.0 * M * N * K

    def check(y):
        assert y.shape == ref.shape, f"{sid}: {tuple(y.shape)} vs {tuple(ref.shape)}"
        e = ((y.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
        assert e < 2e-2, f"{sid}: rel err {e:.2e}"

    def fly(tile):
        return conv3d_implicit(x4, w4, bias=b, stride=stride, padding=padding, tile=tile)

    def time(call):
        return gpu(torch, profile, ProfilerActivity, call)

    check(fly(tuple(DEFAULT_TILE)))
    t_def, per_def = time(lambda: fly(tuple(DEFAULT_TILE)))
    g_def, tr_def = split_fly(per_def)
    best = (g_def, t_def, tr_def, tuple(DEFAULT_TILE))
    for tile in candidates[1:]:
        try:
            check(fly(tile))
            t, per = time(lambda: fly(tile))
            g, tr = split_fly(per)
            if g < best[0]:
                best = (g, t, tr, tile)
        except Exception:
            continue
    g_best, t_best, tr_best, tile_best = best

    t_mm = t_im2col = t_unfold_mm = NAN
    try:
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        Bm = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        Cm = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        t_mm, _ = time(lambda: torch.mm(A, Bm, out=Cm))
        del A, Bm, Cm
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    try:
        t_im2col, _ = time(lambda: F.unfold(x4, (R, S), padding=padding, stride=stride))
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    try:
        w2 = w4.reshape(cout, K).t().contiguous()

        def unfold_mm():
            a = F.unfold(x4, (R, S), padding=padding, stride=stride).squeeze(0).t()
            return (a @ w2 + bbf).t().reshape(1, cout, p, p)

        check(unfold_mm())
        t_unfold_mm, _ = time(unfold_mm)
        del w2
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    t_mio, _ = time(lambda: F.conv2d(x4, w4, bias=bbf, stride=stride, padding=padding))
    print(
        f"RESULT\t{sid}\t{cin}\t{cout}\t{hin}\t{stride}\t{freq}\t{path}\t{M}\t{N}\t{K}\t{flops:.6e}\t"
        f"{t_def:.2f}\t{g_def:.2f}\t{tr_def:.2f}\t"
        f"{t_best:.2f}\t{g_best:.2f}\t{tr_best:.2f}\t{'x'.join(str(v) for v in tile_best)}\t"
        f"{t_mm:.2f}\t{t_im2col:.2f}\t{t_unfold_mm:.2f}\t{t_mio:.2f}"
    )


def drive() -> None:
    env = child_env()
    cwd = str(flydsl_root())
    rows = []
    print(
        f"{'shape':18s} {'x':>3s} {'M':>8s} {'K':>5s} | {'hipBLASLt':>9s} {'FlyGEMM':>8s} {'比':>5s} "
        f"{'BLASt T/s':>9s} {'Fly T/s':>8s} | {'im2col':>7s} {'unfold+mm':>10s} {'Fly 全':>7s} {'MIOpen':>8s} {'最优tile':>13s}"
    )
    for sp in SHAPES:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", *[str(v) for v in sp]],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        if line is None:
            print(f"FAILED {sp[0]}\n{proc.stderr[-800:]}")
            continue
        r = {name: cast(v) for (name, cast), v in zip(FIELDS, line.split("\t")[1:])}
        rows.append(r)
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}" + ("" if r["stride"] == 1 else " s2")
        print(
            f"{shp:18s} {r['freq']:3d} {r['M']:8d} {r['K']:5d} | {r['mm']:9.1f} {r['gemm_best']:8.1f} "
            f"{r['mm'] / r['gemm_best']:4.2f}x {r['flops'] / r['mm'] / 1e6:9.0f} "
            f"{r['flops'] / r['gemm_best'] / 1e6:8.0f} | {r['im2col']:7.1f} {r['unfold_mm']:10.1f} "
            f"{r['fly_best']:7.1f} {r['mio']:8.1f} {r['tile']:>13s}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")
    for path in ("1024", "1328"):
        sel = [r for r in rows if r["path"] == path]
        if not sel:
            continue
        W = lambda k: sum(r[k] * r["freq"] for r in sel) / 1e3  # noqa: E731
        fl = sum(r["flops"] * r["freq"] for r in sel)
        print(
            f"\n{path} 路径 · {len(sel)} shapes / {sum(r['freq'] for r in sel)} 次调用\n"
            f"  A  unfold+mm {W('unfold_mm'):.2f} ms   MIOpen {W('mio'):.2f}   "
            f"Fly 默认 {W('fly_def'):.2f}   Fly 最优 {W('fly_best'):.2f}\n"
            f"  B  hipBLASLt mm {W('mm'):.2f} ms ({fl / W('mm') / 1e9:.0f} TF/s)   "
            f"Fly GEMM 最优 {W('gemm_best'):.2f} ({fl / W('gemm_best') / 1e9:.0f} TF/s)  "
            f"{W('mm') / W('gemm_best'):.3f}x   im2col {W('im2col'):.2f} ms\n"
            f"  C  转置 {W('tr_best'):.2f} ms ({100 * W('tr_best') / W('fly_best'):.0f}%)   "
            f"GEMM 赢 {sum(1 for r in sel if r['mm'] >= r['gemm_best'])}/{len(sel)}"
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        run_one(*sys.argv[2:])
    else:
        drive()
