#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compare plain RMSNorm forward performance against AITER HIP and Triton.

The dedicated BF16 matrix covers decode-sized row counts, the runtime
one-row/multi-row dispatch boundary, and representative model hidden sizes.
Compilation and correctness checks happen before timing; reported values
contain GPU execution time only.

Example:
    AITER_REPO=/path/to/aiter python3 tests/kernels/benchmark_rmsnorm_backends.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EPS = 1e-5

# Keep performance coverage independent from the correctness matrix.  The two
# small-N widths exercise the runtime M crossover; 4096 and 8192 represent
# common model hidden sizes that stay on the one-row implementation.
PLAIN_RMSNORM_PERF_M = (1, 16, 128, 1024, 8192, 8193, 32768)
PLAIN_RMSNORM_PERF_N = (512, 1024, 2048, 4096, 8192)
PLAIN_RMSNORM_PERF_CONFIGS = tuple((m, n, "bf16") for n in PLAIN_RMSNORM_PERF_N for m in PLAIN_RMSNORM_PERF_M)


@dataclass(frozen=True)
class BenchResult:
    shape: str
    dtype: str
    flydsl_us: float
    hip_us: float | None
    triton_us: float | None
    hip_status: str
    triton_status: str


def _torch_dtype(dtype: str) -> torch.dtype:
    return {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float()
    weight_f32 = weight.float()
    rrms = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + EPS)
    return (x_f32 * rrms * weight_f32).to(x.dtype)


def _check_output(backend: str, output: torch.Tensor, reference: torch.Tensor, dtype: str) -> None:
    tolerance = {"f32": 1e-4, "f16": 1e-2, "bf16": 2e-2}[dtype]
    try:
        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=tolerance,
            atol=tolerance,
        )
    except AssertionError as exc:
        max_error = (output.float() - reference.float()).abs().max().item()
        raise AssertionError(
            f"{backend} correctness failed for shape={tuple(output.shape)}, "
            f"dtype={dtype}, max_abs_error={max_error:.3e}"
        ) from exc


def _bench_kernel_us(run: Callable[[], None], *, warmup: int, iters: int) -> float:
    """Return mean GPU kernel time, excluding Python/backend wrapper time."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    import torch.profiler as tpf

    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()

    device_us = sum(
        event.self_device_time_total for event in prof.events() if str(event.device_type).split(".")[-1] == "CUDA"
    )
    if device_us <= 0:
        raise RuntimeError("torch profiler did not record any GPU kernel time")
    return device_us / iters


def _load_aiter(aiter_repo: str | None):
    if aiter_repo:
        repo_path = Path(aiter_repo).expanduser().resolve()
        if not (repo_path / "aiter").is_dir():
            raise ValueError(f"--aiter-repo does not contain an aiter package: {repo_path}")
        sys.path.insert(0, str(repo_path))

    original_find_spec = importlib.util.find_spec

    def find_baseline_dependency(name, *args, **kwargs):
        # Do not let AITER recursively load its optional FlyDSL kernels. This
        # process uses the current checkout as the candidate implementation.
        if name == "flydsl" or name.startswith("flydsl."):
            return None
        return original_find_spec(name, *args, **kwargs)

    try:
        with patch("importlib.util.find_spec", side_effect=find_baseline_dependency):
            import aiter
            from aiter.ops.rmsnorm import rmsnorm as hip_rmsnorm
            from aiter.ops.triton.normalization.rmsnorm import rms_norm as triton_rmsnorm
    except Exception as exc:
        raise RuntimeError(
            "AITER HIP/Triton RMSNorm is unavailable. Install AITER or set AITER_REPO=/path/to/aiter."
        ) from exc

    return aiter, hip_rmsnorm, triton_rmsnorm


def _compile_flydsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    dtype: str,
) -> Callable[[], None]:
    import flydsl.compiler as flyc
    from kernels.norm.rmsnorm_kernel import build_rmsnorm_module

    m, n = x.shape
    launch_fn = build_rmsnorm_module(n, dtype)
    stream = torch.cuda.current_stream()
    compiled = flyc.compile(launch_fn, x, weight, output, m, stream)

    def run():
        compiled(x, weight, output, m, stream)

    return run


def _run_case(
    m: int,
    n: int,
    dtype: str,
    hip_rmsnorm,
    triton_rmsnorm,
    warmup: int,
    iters: int,
) -> BenchResult:
    torch.manual_seed(42)
    torch_dtype = _torch_dtype(dtype)
    x = torch.randn((m, n), device="cuda", dtype=torch_dtype).contiguous()
    weight = torch.rand((n,), device="cuda", dtype=torch_dtype).contiguous()
    reference = _reference(x, weight)

    flydsl_output = torch.empty_like(x)
    run_flydsl = _compile_flydsl(x, weight, flydsl_output, dtype)
    run_flydsl()
    torch.cuda.synchronize()
    _check_output("FlyDSL", flydsl_output, reference, dtype)

    flydsl_us = _bench_kernel_us(run_flydsl, warmup=warmup, iters=iters)
    hip_us = None
    hip_status = "SKIP: f32 unsupported"
    if dtype in ("f16", "bf16"):
        try:
            hip_output = torch.empty_like(x)

            def run_hip():
                hip_rmsnorm(hip_output, x, weight, EPS)

            run_hip()
            torch.cuda.synchronize()
            _check_output("HIP", hip_output, reference, dtype)
            hip_us = _bench_kernel_us(run_hip, warmup=warmup, iters=iters)
            hip_status = "ok"
        except Exception as exc:
            hip_status = f"ERR: {type(exc).__name__}: {exc}"

    triton_us = None
    triton_status = "ok"
    try:
        triton_output = triton_rmsnorm(x, weight, EPS)
        torch.cuda.synchronize()
        _check_output("Triton", triton_output, reference, dtype)
        triton_us = _bench_kernel_us(lambda: triton_rmsnorm(x, weight, EPS), warmup=warmup, iters=iters)
    except Exception as exc:
        triton_status = f"ERR: {type(exc).__name__}: {exc}"

    return BenchResult(
        shape=f"{m}x{n}",
        dtype=dtype,
        flydsl_us=flydsl_us,
        hip_us=hip_us,
        triton_us=triton_us,
        hip_status=hip_status,
        triton_status=triton_status,
    )


def _fmt_us(value: float | None, status: str) -> str:
    if value is not None:
        return f"{value:.2f}"
    return "SKIP" if status.startswith("SKIP") else "ERR"


def _fmt_speedup(flydsl_us: float, baseline_us: float | None) -> str:
    return "-" if baseline_us is None else f"{baseline_us / flydsl_us:.2f}x"


def _print_results(results: list[BenchResult]) -> None:
    print()
    print(
        f"{'shape':>12} {'dtype':>6} {'FlyDSL(us)':>12} {'HIP(us)':>10} "
        f"{'Triton(us)':>12} {'FlyDSL/HIP':>12} {'FlyDSL/Triton':>16}"
    )
    for result in results:
        print(
            f"{result.shape:>12} {result.dtype:>6} {result.flydsl_us:>12.2f} "
            f"{_fmt_us(result.hip_us, result.hip_status):>10} "
            f"{_fmt_us(result.triton_us, result.triton_status):>12} "
            f"{_fmt_speedup(result.flydsl_us, result.hip_us):>12} "
            f"{_fmt_speedup(result.flydsl_us, result.triton_us):>16}"
        )
    print("\nSpeedups greater than 1.0x mean FlyDSL is faster than that baseline.")
    for result in results:
        if result.hip_status != "ok":
            print(f"{result.shape} {result.dtype} HIP: {result.hip_status}")
        if result.triton_status != "ok":
            print(f"{result.shape} {result.dtype} Triton: {result.triton_status}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aiter-repo",
        default=os.environ.get("AITER_REPO"),
        help="AITER checkout to import when it is not installed (default: AITER_REPO).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")

    aiter, hip_rmsnorm, triton_rmsnorm = _load_aiter(args.aiter_repo)
    configs = PLAIN_RMSNORM_PERF_CONFIGS
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    print(f"Device: {torch.cuda.get_device_name(device)} ({getattr(properties, 'gcnArchName', 'unknown')})")
    print(f"AITER: {aiter.__file__}")
    print(f"Cases: {configs}")

    results = [_run_case(m, n, dtype, hip_rmsnorm, triton_rmsnorm, args.warmup, args.iters) for m, n, dtype in configs]
    _print_results(results)
    if any(result.hip_status.startswith("ERR") or result.triton_status.startswith("ERR") for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
