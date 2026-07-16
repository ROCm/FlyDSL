#!/usr/bin/env python3

"""Kernel-only benchmark: fp8 conv3d gemm4w vs gemm8w vs the general fp8 conv.

Times the compiled *core* kernel with pre-transposed NDHWC activation and
pre-packed KTRSC weight (no per-call transpose / output alloc), matching the PR's
"kernel-only, pre-packed fp8" methodology. TFLOPS = 2*N*Do*Ho*Wo*K*C*Kt*Kh*Kw / t.

Run from a configured FlyDSL environment (see amd-inference container notes).
"""

from __future__ import annotations

import argparse

import torch

import flydsl.compiler as flyc
from kernels.conv.conv3d_implicit_fp8 import (
    DEFAULT_TILE,
    _prep_weight_fp8,
    _resolve_splitk,
    compile_conv3d_implicit_fp8,
)
from kernels.conv.conv3d_implicit_fp8_gemm4w import compile_conv3d_implicit_fp8_gemm4w
from kernels.conv.conv3d_implicit_fp8_gemm8w import (
    _transpose_activation_fp8,
    compile_conv3d_implicit_fp8_gemm8w,
)

# (name, n, c, d, h, w, k, kt, kh, kw) with symmetric spatial padding kt//2 etc.
SHAPES = [
    ("C1024 D120 1x3x3", 1, 1024, 120, 160, 90, 1024, 1, 3, 3),
    ("C1024 D120 3x1x1", 1, 1024, 120, 160, 90, 1024, 3, 1, 1),
    ("C1024 D120 3x3x3", 1, 1024, 120, 160, 90, 1024, 3, 3, 3),
    ("C512 D240 1x3x3", 1, 512, 240, 320, 180, 512, 1, 3, 3),
    ("C2048 D60 1x3x3", 1, 2048, 60, 80, 45, 2048, 1, 3, 3),
]


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        b = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        b.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(b.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _fp8(t):
    return t.to(torch.float8_e4m3fn)


def bench_shape(name, n, c, d, h, w, k, kt, kh, kw, warmup, rep):
    pt, ph, pw = kt // 2, kh // 2, kw // 2
    st = sh = sw = 1
    do = d + 2 * pt - kt + 1
    ho = h + 2 * ph - kh + 1
    wo = w + 2 * pw - kw + 1
    npq = n * do * ho * wo
    crs = c * kt * kh * kw
    flops = 2 * npq * k * crs
    stream = torch.cuda.current_stream()

    x = _fp8(torch.randn((n, c, d, h, w), device="cuda", dtype=torch.bfloat16))
    weight = _fp8(torch.randn((k, c, kt, kh, kw), device="cuda", dtype=torch.bfloat16))

    # Shared pre-packed inputs.
    x_arg = flyc.from_torch_tensor(_transpose_activation_fp8(x))
    w_arg = flyc.from_torch_tensor(_prep_weight_fp8(weight))
    bias = flyc.from_torch_tensor(torch.empty(1, device="cuda", dtype=torch.float32))

    results = {}

    def _run_exe(exe, y_arg):
        exe(y_arg, x_arg, w_arg, bias, stream)

    # gemm4w / gemm8w cores: (y bf16, x_ndhwc, w_ktrsc, bias).
    for label, compile_fn in (
        ("gemm4w", compile_conv3d_implicit_fp8_gemm4w),
        ("gemm8w", compile_conv3d_implicit_fp8_gemm8w),
    ):
        try:
            y = torch.empty((n, k, do, ho, wo), device="cuda", dtype=torch.bfloat16)
            y_arg = flyc.from_torch_tensor(y.view(-1))
            exe = compile_fn(n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, False)
            ms = _bench(lambda exe=exe, y_arg=y_arg: _run_exe(exe, y_arg), warmup, rep)
            results[label] = (ms, flops / (ms * 1e-3) / 1e12)
        except Exception as exc:
            results[label] = (None, f"{type(exc).__name__}: {exc}")

    # General fp8 conv core: resolve split-K + tile like the public entry.
    try:
        tile = DEFAULT_TILE
        sk = _resolve_splitk(None, npq, crs, k, torch.device("cuda"), tile)
        if sk > 1:
            y = torch.zeros((npq, k), device="cuda", dtype=torch.float32)
        else:
            y = torch.empty((n, k, do, ho, wo), device="cuda", dtype=torch.bfloat16)
        y_arg = flyc.from_torch_tensor(y.view(-1))
        exe = compile_conv3d_implicit_fp8(n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, False, sk, tile)
        ms = _bench(lambda exe=exe, y_arg=y_arg: _run_exe(exe, y_arg), warmup, rep)
        results["fp8_conv"] = (ms, flops / (ms * 1e-3) / 1e12, sk)
    except Exception as exc:
        results["fp8_conv"] = (None, f"{type(exc).__name__}: {exc}", None)

    return results, flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rep", type=int, default=10)
    args = ap.parse_args()

    print(f"{'Shape':22} {'gemm4w':>18} {'gemm8w':>18} {'fp8_conv':>22}  {'4w/8w':>7}")
    print("-" * 96)
    for name, *dims in SHAPES:
        res, _ = bench_shape(name, *dims, args.warmup, args.rep)

        def fmt(label):
            v = res.get(label)
            if v is None or v[0] is None:
                return "FAIL"
            ms, tf = v[0], v[1]
            extra = f" sk{v[2]}" if len(v) > 2 and v[2] and v[2] > 1 else ""
            return f"{ms:.3f}ms {tf:.0f}TF{extra}"

        r4 = res.get("gemm4w")
        r8 = res.get("gemm8w")
        ratio = "-"
        if r4 and r8 and r4[0] and r8[0]:
            ratio = f"{r8[0] / r4[0]:.2f}x"
        print(f"{name:22} {fmt('gemm4w'):>18} {fmt('gemm8w'):>18} {fmt('fp8_conv'):>22}  {ratio:>7}")


if __name__ == "__main__":
    main()
