#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Qwen-Image VAE conv: FlyDSL GEMM kernel time vs hipBLASLt, per shape.

Two arms only -- the shipped default path (no ``tile`` argument, so
``_pick_tile`` / ``_pick_wgm`` decide) and ``torch.mm`` on the same implicit-GEMM
``M/N/K``. That makes it a few minutes rather than bench_fair.py's full
three-tier sweep, so it is the one to re-run after a picker change.

Each of the 18 shapes runs in a fresh process so JIT state cannot leak. Writes
``gemm_time.json`` next to this script; plot it with ``plot_gemm_time.py``.

Usage::

    python tests/perf/qwenimage_vae_conv/bench_gemm_time.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "gemm_time.json"

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
    ("gemm", float),
    ("tile", str),
    ("wgm", int),
    ("mm", float),
    ("act_mb", float),
]

R = S = 3
ITERS, TRIALS = 10, 3
NAN = float("nan")


def flydsl_root() -> Path:
    env = os.environ.get("FLYDSL_ROOT")
    if env:
        return Path(env).resolve()
    return HERE.parents[2]


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(flydsl_root()) + os.pathsep + env.get("PYTHONPATH", "")
    env["FLYDSL_CONV3D_AUTOTUNE"] = "0"
    return env


def gpu(torch, profile, ProfilerActivity, call, name_filter=None):
    """Best-of-TRIALS on-device time, optionally restricted to one kernel name."""
    best = None
    for _ in range(TRIALS):
        for _ in range(4):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                call()
            torch.cuda.synchronize()
        t = 0.0
        for e in prof.key_averages():
            if e.device_type != torch.autograd.DeviceType.CUDA or e.self_device_time_total <= 0:
                continue
            if name_filter and name_filter not in e.key:
                continue
            t += e.self_device_time_total / ITERS
        if best is None or t < best:
            best = t
    return best


def run_one(sid, cin, cout, hin, stride, padding, freq, path) -> None:
    sys.path.insert(0, str(flydsl_root()))
    os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")

    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from kernels.conv.conv3d_implicit import _pick_tile, _pick_wgm, conv3d_implicit

    cin, cout, hin, stride, padding, freq = map(int, (cin, cout, hin, stride, padding, freq))
    torch.manual_seed(0)
    x4 = torch.randn((1, cin, hin, hin), device="cuda", dtype=torch.bfloat16)
    w4 = torch.randn((cout, cin, R, S), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((cout,), device="cuda", dtype=torch.float32)
    ref = F.conv2d(x4, w4, bias=b.to(torch.bfloat16), stride=stride, padding=padding)
    p = ref.shape[2]
    M, N, K = p * p, cout, cin * R * S
    flops = 2.0 * M * N * K

    y = conv3d_implicit(x4, w4, bias=b, stride=stride, padding=padding)
    err = ((y.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
    assert err < 2e-2, f"{sid}: rel err {err:.2e}"

    t_gemm = gpu(
        torch,
        profile,
        ProfilerActivity,
        lambda: conv3d_implicit(x4, w4, bias=b, stride=stride, padding=padding),
        name_filter="conv3d_implicit_kernel",
    )
    tile = _pick_tile(M, N, 1, x4.device)
    wgm = _pick_wgm(M, N, 1, tile, x4.device)

    # Source activation + output: the traffic conv actually touches, versus the
    # M x K im2col matrix hipBLASLt needs materialized.
    act_mb = (x4.numel() + ref.numel()) * 2 / 1e6

    t_mm = NAN
    try:
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        Bm = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        Cm = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        t_mm = gpu(torch, profile, ProfilerActivity, lambda a=A, b=Bm, c=Cm: torch.mm(a, b, out=c))
        del A, Bm, Cm
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    print(
        f"RESULT\t{sid}\t{cin}\t{cout}\t{hin}\t{stride}\t{freq}\t{path}\t{M}\t{N}\t{K}\t"
        f"{flops:.6e}\t{t_gemm:.2f}\t{'x'.join(str(v) for v in tile)}\t{wgm}\t{t_mm:.2f}\t{act_mb:.1f}"
    )


def drive() -> None:
    env = child_env()
    rows = []
    hdr = (
        f"{'shape':18s} {'x':>3s} {'M':>8s} {'N':>4s} {'K':>5s} | "
        f"{'Fly µs':>8s} {'mm µs':>8s} {'比':>5s} | {'Fly TF/s':>8s} {'mm TF/s':>8s} | {'tile':>14s} {'wgm':>3s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sp in SHAPES:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", *[str(v) for v in sp]],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(flydsl_root()),
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")), None)
        if line is None:
            print(f"FAILED {sp[0]}\n{proc.stderr[-600:]}")
            continue
        r = {name: cast(v) for (name, cast), v in zip(FIELDS, line.split("\t")[1:])}
        rows.append(r)
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}" + ("" if r["stride"] == 1 else " s2")
        print(
            f"{shp:18s} {r['freq']:3d} {r['M']:8d} {r['N']:4d} {r['K']:5d} | "
            f"{r['gemm']:8.1f} {r['mm']:8.1f} {r['mm'] / r['gemm']:4.2f}x | "
            f"{r['flops'] / r['gemm'] / 1e6:8.0f} {r['flops'] / r['mm'] / 1e6:8.0f} | "
            f"{r['tile']:>14s} {r['wgm']:3d}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")
    for path in ("1024", "1328"):
        sel = [r for r in rows if r["path"] == path]
        if not sel:
            continue
        w = lambda k: sum(r[k] * r["freq"] for r in sel) / 1e3  # noqa: E731
        print(
            f"{path} 路径 · {len(sel)} shapes / {sum(r['freq'] for r in sel)} 次调用: "
            f"FlyDSL GEMM {w('gemm'):.2f} ms vs hipBLASLt {w('mm'):.2f} ms  {w('mm') / w('gemm'):.3f}x"
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        run_one(*sys.argv[2:])
    else:
        drive()
