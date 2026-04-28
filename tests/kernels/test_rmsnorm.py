#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
RMSNorm Operator Test
Implementation of a Block-wise RMSNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (sum of squares)

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
"""

import os

from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

EPS: float = 1e-5
from kernels.rmsnorm_kernel import (
    build_rmsnorm_module,
    build_rmsnorm_dynamicquant_module,
    build_rmsnorm_smoothquant_module,
    KERNEL_NAME as RMSNORM_KERNEL_NAME,
    BLOCK_THREADS,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100

def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_rmsnorm_module(M, N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # PyTorch CPU Reference:
    # RMS(x) = sqrt(mean(x^2) + eps) ; RMSNorm(x) = x / RMS(x) * gamma
    x = input_ref
    gamma = gamma_ref
    sq_mean = (x * x).mean(dim=1, keepdim=True)
    rms = torch.sqrt(sq_mean + EPS)
    expected = (x / rms) * gamma
    expected = expected.to(DTYPE_FP32)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, output_dev, M, stream=stream)

    # run_perftest returns (data, avg_us)
    _, avg_us = run_perftest(lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    # Bandwidth estimate: read input + read gamma + write output
    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = 2 * M * N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm gpu: {flydsl_gpu_us:.1f} us")

    # Verification (pure torch style; compute max error in torch)
    output_ref = output_dev.to(DTYPE_FP32)

    error = (output_ref - expected).abs().max().item()
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        ok = False
    return ok, flydsl_gpu_us

def test_all():
    print("="*80)
    print("Running RMSNorm Tests")
    print("="*80)

    shapes_env = os.environ.get("ROCDSL_RMSNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    else:
        # Prefer N multiples of BLOCK_THREADS*VEC_WIDTH (=2048) to exercise the fast path.
        configs = [
            # (64, 256, "f32"),     # Aligned
            # (128, 1024, "f32"),   # Aligned
            # (32, 128, "f16"),     # Aligned
            # (64, 2000, "f32"),    # Unaligned (tail handling)
            # (16, 512, "bf16"),    # BF16
            # (1024, 8192, "bf16"), # BF16
            (32768, 8192, "bf16"),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch
            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm
                    x = torch.randn((M, N), device="cuda", dtype=DTYPE_BF16 if dtype == "bf16" else (DTYPE_FP16 if dtype == "f16" else DTYPE_FP32))
                    w = torch.rand((N,), device="cuda", dtype=x.dtype)

                    def run_aiter():
                        aiter_rms_norm(x, w, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter rmsnorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter rmsnorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(PerfRow(op="rmsnorm", shape=f"{M}x{N}", dtype=dtype, flydsl_gpu_us=flydsl_gpu_us, aiter_gpu_us=aiter_us))

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def _torch_dtype(dtype: str):
    if dtype == "f32":
        return DTYPE_FP32
    if dtype == "f16":
        return DTYPE_FP16
    if dtype == "bf16":
        return DTYPE_BF16
    raise ValueError(f"unsupported dtype: {dtype}")


def _get_rmsnorm_configs():
    shapes_env = os.environ.get("ROCDSL_RMSNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
        return configs

    # Prefer N multiples of BLOCK_THREADS*VEC_WIDTH (=2048) to exercise the fast path.
    return [
        # (64, 256, "f32"),     # Aligned
        # (128, 1024, "f32"),   # Aligned
        # (32, 128, "f16"),     # Aligned
        # (64, 2000, "f32"),    # Unaligned (tail handling)
        # (16, 512, "bf16"),    # BF16
        # (1024, 8192, "bf16"), # BF16
        (32768, 8192, "bf16"),
    ]


def _reference_rmsnorm_quant(input_dev, gamma_dev, *, xscale_dev=None):
    x = input_dev.to(DTYPE_FP32)
    gamma = gamma_dev.to(DTYPE_FP32)
    expected = (x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + EPS)) * gamma
    if xscale_dev is not None:
        expected = expected * xscale_dev.to(DTYPE_FP32)

    yscale = expected.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(expected / yscale.unsqueeze(1)), -127, 127).to(torch.int8)
    return expected, q, yscale


def _bench_aiter_rmsnorm_quant(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    torch_dtype = _torch_dtype(dtype)

    try:
        if is_smooth:
            from aiter.ops.triton.normalization.rmsnorm import (
                rmsnorm2d_fwd_with_smoothquant as aiter_rmsnorm_quant,
            )
        else:
            from aiter.ops.triton.normalization.rmsnorm import (
                rmsnorm2d_fwd_with_dynamicquant as aiter_rmsnorm_quant,
            )
    except Exception as e:
        print(f"[Perf] AIter rmsnorm {mode} skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    y = torch.empty((M, N), dtype=torch.int8, device="cuda")
    yscale = torch.empty((M, 1), dtype=torch.float32, device="cuda")

    if is_smooth:
        xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()

        def run_aiter():
            aiter_rmsnorm_quant(y, x, xscale, yscale, w, EPS)
    else:
        def run_aiter():
            aiter_rmsnorm_quant(y, x, yscale, w, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter rmsnorm {mode} gpu: {aiter_us:.1f} us")
    return aiter_us


def run_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting RMSNorm {mode} (M={M}, N={N}, dtype={dtype})")

    torch_dtype = _torch_dtype(dtype)
    try:
        if is_smooth:
            launch_fn = build_rmsnorm_smoothquant_module(M, N, dtype)
        else:
            launch_fn = build_rmsnorm_dynamicquant_module(M, N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for {mode} (M={M}, N={N}, dtype={dtype}): "
            f"{type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=torch.int8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=torch.float32)

    xscale_dev = None
    if is_smooth:
        xscale_dev = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous() + 0.5

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        if is_smooth:
            launch_fn(
                input_dev,
                gamma_dev,
                xscale_dev,
                output_dev,
                yscale_dev,
                M,
                stream=stream,
            )
        else:
            launch_fn(input_dev, gamma_dev, output_dev, yscale_dev, M, stream=stream)

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()),
        num_iters=BENCH_ITERS,
        num_warmup=WARMUP_ITERS,
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = M * N * elem_bytes + N * elem_bytes + M * N + M * 4
    if is_smooth:
        total_bytes += N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(
        f"Kernel avg time: {avg_ms:.4f} ms via run_perftest "
        f"(warmup={WARMUP_ITERS}, iters={BENCH_ITERS})"
    )
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm {mode} gpu: {flydsl_gpu_us:.1f} us")

    expected, q_ref, yscale_ref = _reference_rmsnorm_quant(
        input_dev,
        gamma_dev,
        xscale_dev=xscale_dev,
    )
    q_out = output_dev.to(torch.int16)
    q_expected = q_ref.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_expected = yscale_ref.cpu()

    q_diff = (q_out - q_expected).abs().max().item()
    scale_diff = (yscale_out - yscale_expected).abs().max().item()
    recon = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    recon_err = (recon - expected).abs().max().item()

    scale_tol = 1e-2 if is_smooth else 5e-3
    recon_tol = 0.25 if is_smooth else 0.2

    print(f"Max quant diff: {q_diff}")
    print(f"Max scale diff: {scale_diff:.2e} (tol={scale_tol})")
    print(f"Max recon error: {recon_err:.2e} (tol={recon_tol})")

    ok = q_diff <= 1 and scale_diff < scale_tol and recon_err < recon_tol
    if ok:
        print("PASSED")
    else:
        print("FAILED")
    return ok, flydsl_gpu_us


def test_rmsnorm_dynamicquant():
    print("="*80)
    print("Running RMSNorm DynamicQuant Tests")
    print("="*80)

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in _get_rmsnorm_configs():
        ok, flydsl_gpu_us = run_quant_test(M, N, dtype, is_smooth=False)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_rmsnorm_quant(M, N, dtype, is_smooth=False)
            perf_rows.append(
                PerfRow(
                    op="rmsnorm_dq",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def test_rmsnorm_smoothquant():
    print("="*80)
    print("Running RMSNorm SmoothQuant Tests")
    print("="*80)

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in _get_rmsnorm_configs():
        ok, flydsl_gpu_us = run_quant_test(M, N, dtype, is_smooth=True)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_rmsnorm_quant(M, N, dtype, is_smooth=True)
            perf_rows.append(
                PerfRow(
                    op="rmsnorm_sq",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_all()
