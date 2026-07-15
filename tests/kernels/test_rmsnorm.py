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

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

# Imported after the torch guard: rmsnorm() is only defined when torch is present,
# so importing it earlier makes a torch-less collection fail (ImportError) instead of skip.
import flydsl.compiler as flyc  # noqa: E402
from kernels.norm.rmsnorm_kernel import (  # noqa: E402
    build_fused_add_rmsnorm_bwd_module,
    build_fused_add_rmsnorm_dynamicquant_module,
    build_fused_add_rmsnorm_module,
    build_fused_add_rmsnorm_smoothquant_module,
    build_rmsnorm_bwd_module,
    build_rmsnorm_dynamicquant_module,
    build_rmsnorm_module,
    build_rmsnorm_smoothquant_module,
    fused_add_rmsnorm,
    rmsnorm,
)
from tests.kernels.benchmark_common import (  # noqa: E402
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
from tests.test_common import run_perftest  # noqa: E402

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16
DTYPE_INT8 = torch.int8

EPS: float = 1e-5

WARMUP_ITERS = 10
BENCH_ITERS = 100


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
    else:
        configs = [
            (64, 256, "f32"),  # f32 aligned
            (32, 128, "f16"),  # f16 aligned
            (64, 2000, "f32"),  # unaligned tail handling
            (16, 512, "bf16"),  # bf16 small shape
            (64, 8192, "bf16"),  # bf16 fast-path N with small M
        ]
    return configs


def _get_rmsnorm_large_configs():
    return [
        (32768, 8192, "bf16"),
    ]


def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_rmsnorm_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    output_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    input_ref = input_dev.to(DTYPE_FP32)
    gamma_ref = gamma_dev.to(DTYPE_FP32)
    if dtype == "f32":
        atol = 1e-4
    elif dtype == "f16":
        atol = 1e-2
    elif dtype == "bf16":
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    expected = _reference_rmsnorm(input_ref, gamma_ref)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()
    compiled_fn = flyc.compile(launch_fn, input_dev, gamma_dev, output_dev, M, stream)

    def kernel_launch():
        compiled_fn(input_dev, gamma_dev, output_dev, M, stream)

    # run_perftest returns (data, avg_us)
    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
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


def run_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting RMSNorm {mode} (M={M}, N={N}, dtype={dtype})")

    try:
        if is_smooth:
            launch_fn = build_rmsnorm_smoothquant_module(N, dtype)
        else:
            launch_fn = build_rmsnorm_dynamicquant_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for {mode} (M={M}, N={N}, dtype={dtype}): " f"{type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    xscale_dev = None
    if is_smooth:
        xscale_dev = (torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5).to(torch_dtype).contiguous()
    scale_tol = 1e-3

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    if is_smooth:
        compiled_fn = flyc.compile(launch_fn, input_dev, gamma_dev, xscale_dev, output_dev, yscale_dev, M, stream)

        def kernel_launch():
            compiled_fn(input_dev, gamma_dev, xscale_dev, output_dev, yscale_dev, M, stream)

    else:
        compiled_fn = flyc.compile(launch_fn, input_dev, gamma_dev, output_dev, yscale_dev, M, stream)

        def kernel_launch():
            compiled_fn(input_dev, gamma_dev, output_dev, yscale_dev, M, stream)

    # run_perftest returns (data, avg_us)
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

    # Bandwidth estimate: read input + read gamma + write output
    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = M * N * elem_bytes + N * elem_bytes + M * N + M * 4
    if is_smooth:
        total_bytes += N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm {mode} gpu: {flydsl_gpu_us:.1f} us")

    # PyTorch Reference:
    # RMS(x) = sqrt(mean(x^2) + eps) ; RMSNorm(x) = x / RMS(x) * gamma
    # Quant path additionally computes per-row yscale and int8 output from the fp32 reference.
    q_ref, yscale_ref = _reference_rmsnorm_quant(
        input_dev,
        gamma_dev,
        xscale_dev=xscale_dev,
    )
    q_out = output_dev.to(torch.int16)
    q_expected = q_ref.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_expected = yscale_ref.cpu()

    quant_error = (q_out - q_expected).abs().max().item()
    scale_error = (yscale_out - yscale_expected).abs().max().item()

    print(f"Max quant diff: {quant_error}")
    print(f"Max scale diff: {scale_error:.2e} (tol={scale_tol})")

    ok = quant_error <= 1 and scale_error < scale_tol
    if ok:
        print("PASSED")
    else:
        print("FAILED")
        print("First row Quant Expected:")
        print(q_expected[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_expected[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
    return ok, flydsl_gpu_us


def run_fused_add_test(M: int, N: int, dtype: str):
    print(f"\nTesting FusedAdd RMSNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_fused_add_rmsnorm_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for fused_add rmsnorm (M={M}, N={N}, dtype={dtype}): " f"{type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    residual_in_dev = residual_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    output_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    residual_out_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    if dtype == "f32":
        atol = 1e-4
    elif dtype == "f16":
        atol = 1e-2
    elif dtype == "bf16":
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    print("Launching kernel...")
    stream = torch.cuda.current_stream()
    compiled_fn = flyc.compile(
        launch_fn,
        input_dev,
        residual_in_dev,
        gamma_dev,
        output_dev,
        residual_out_dev,
        M,
        stream,
    )

    def kernel_launch():
        compiled_fn(input_dev, residual_in_dev, gamma_dev, output_dev, residual_out_dev, M, stream)

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
    total_bytes = (4 * M * N + N) * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest " f"(warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add rmsnorm gpu: {flydsl_gpu_us:.1f} us")

    # PyTorch Reference:
    # RMS(x) = sqrt(mean(x^2) + eps) ; RMSNorm(x) = x / RMS(x) * gamma
    residual_expected, output_expected = _reference_fused_add_rmsnorm(
        input_dev,
        residual_in_dev,
        gamma_dev,
    )
    residual_out_ref = residual_out_dev.to(DTYPE_FP32)
    output_ref = output_dev.to(DTYPE_FP32)

    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    output_error = (output_ref - output_expected).abs().max().item()

    print(f"Max residual error: {residual_error:.2e} (atol={atol})")
    print(f"Max output error: {output_error:.2e} (atol={atol})")

    ok = residual_error < atol and output_error < atol
    if ok:
        print("PASSED")
    else:
        print("FAILED")
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
        print("First row Output Expected:")
        print(output_expected[0, :5])
        print("First row Output Actual:")
        print(output_ref[0, :5])
    return ok, flydsl_gpu_us


def run_fused_add_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting FusedAdd RMSNorm {mode} (M={M}, N={N}, dtype={dtype})")

    try:
        if is_smooth:
            launch_fn = build_fused_add_rmsnorm_smoothquant_module(N, dtype)
        else:
            launch_fn = build_fused_add_rmsnorm_dynamicquant_module(N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for fused_add rmsnorm {mode} "
            f"(M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    residual_in_dev = residual_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    residual_out_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    if dtype == "f32":
        residual_atol = 1e-4
    elif dtype == "f16":
        residual_atol = 1e-2
    elif dtype == "bf16":
        residual_atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    xscale_dev = None
    if is_smooth:
        xscale_dev = (torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5).to(torch_dtype).contiguous()
    scale_tol = 1e-3

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    if is_smooth:
        compiled_fn = flyc.compile(
            launch_fn,
            input_dev,
            residual_in_dev,
            gamma_dev,
            xscale_dev,
            output_dev,
            residual_out_dev,
            yscale_dev,
            M,
            stream,
        )

        def kernel_launch():
            compiled_fn(
                input_dev,
                residual_in_dev,
                gamma_dev,
                xscale_dev,
                output_dev,
                residual_out_dev,
                yscale_dev,
                M,
                stream,
            )

    else:
        compiled_fn = flyc.compile(
            launch_fn,
            input_dev,
            residual_in_dev,
            gamma_dev,
            output_dev,
            residual_out_dev,
            yscale_dev,
            M,
            stream,
        )

        def kernel_launch():
            compiled_fn(
                input_dev,
                residual_in_dev,
                gamma_dev,
                output_dev,
                residual_out_dev,
                yscale_dev,
                M,
                stream,
            )

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
    total_bytes = 3 * M * N * elem_bytes + N * elem_bytes + M * N + M * 4
    if is_smooth:
        total_bytes += N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add rmsnorm {mode} gpu: {flydsl_gpu_us:.1f} us")

    # PyTorch Reference:
    # RMS(x) = sqrt(mean(x^2) + eps) ; RMSNorm(x) = x / RMS(x) * gamma
    residual_expected, q_ref, yscale_ref = _reference_fused_add_rmsnorm_quant(
        input_dev,
        residual_in_dev,
        gamma_dev,
        xscale_dev=xscale_dev,
    )
    residual_out_ref = residual_out_dev.to(DTYPE_FP32)
    q_out = output_dev.to(torch.int16)
    q_expected = q_ref.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_expected = yscale_ref.cpu()

    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    scale_error = (yscale_out - yscale_expected).abs().max().item()
    quant_error = (q_out - q_expected).abs().max().item()

    print(f"Max residual error: {residual_error:.2e} (tol={residual_atol})")
    print(f"Max scale error: {scale_error:.2e} (tol={scale_tol})")
    print(f"Max quant error: {quant_error}")

    ok = residual_error < residual_atol and scale_error < scale_tol and quant_error <= 1
    if ok:
        print("PASSED")
    else:
        print("FAILED")
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
        print("First row Quant Expected:")
        print(q_expected[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_expected[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
    return ok, flydsl_gpu_us


def _reference_rmsnorm(input_dev, gamma_dev):
    x = input_dev.to(DTYPE_FP32)
    gamma = gamma_dev.to(DTYPE_FP32)
    return ((x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + EPS)) * gamma).to(DTYPE_FP32)


def _reference_rmsnorm_quant(input_dev, gamma_dev, *, xscale_dev=None):
    normalized = _reference_rmsnorm(input_dev, gamma_dev)
    if xscale_dev is not None:
        normalized = normalized * xscale_dev.to(DTYPE_FP32)

    yscale = normalized.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(normalized / yscale.unsqueeze(1)), -127, 127).to(torch.int8)
    return q, yscale


def _reference_fused_add_rmsnorm(input_dev, residual_in_dev, gamma_dev):
    added = input_dev + residual_in_dev
    added_fp32 = added.to(DTYPE_FP32)
    gamma = gamma_dev.to(DTYPE_FP32)
    expected = (added_fp32 / torch.sqrt((added_fp32 * added_fp32).mean(dim=1, keepdim=True) + EPS)) * gamma
    return added_fp32, expected


def _reference_fused_add_rmsnorm_quant(
    input_dev,
    residual_in_dev,
    gamma_dev,
    *,
    xscale_dev=None,
):
    added = input_dev + residual_in_dev
    residual_expected = added.to(DTYPE_FP32)
    q, yscale = _reference_rmsnorm_quant(
        added,
        gamma_dev,
        xscale_dev=xscale_dev,
    )
    return residual_expected, q, yscale


def _reference_rmsnorm_bwd(x_dev, weight_dev, dy_dev):
    """Eager rmsnorm backward via autograd. Returns dx, dw, rstd (all fp32)."""
    x = x_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    w = weight_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    rstd = torch.rsqrt((x * x).mean(dim=1, keepdim=True) + EPS)
    y = x * rstd * w
    dx, dw = torch.autograd.grad(y, [x, w], grad_outputs=dy_dev.to(DTYPE_FP32))
    return dx.detach(), dw.detach(), rstd.detach().squeeze(1).contiguous()


def run_bwd_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm backward (M={M}, N={N}, dtype={dtype})")

    torch_dtype = _torch_dtype(dtype)
    try:
        fwd_fn = build_rmsnorm_module(N, dtype, store_rstd=True)
        bwd_fn = build_rmsnorm_bwd_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for bwd (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False

    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    dy = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()

    dx_ref, dw_ref, rstd_ref = _reference_rmsnorm_bwd(x, weight, dy)

    stream = torch.cuda.current_stream()

    # --- forward with store_rstd: validates rstd from the kernel ---
    out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    rstd = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    fwd_c = flyc.compile(fwd_fn, x, weight, out, rstd, M, stream)
    fwd_c(x, weight, out, rstd, M, stream)
    torch.cuda.synchronize()
    rstd_err = (rstd - rstd_ref).abs().max().item()

    # --- backward: dx + dweight ---
    dx = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    dweight = torch.zeros((N,), device="cuda", dtype=DTYPE_FP32)
    bwd_c = flyc.compile(bwd_fn, x, weight, dy, rstd, dx, dweight, M, stream)
    dweight.zero_()
    bwd_c(x, weight, dy, rstd, dx, dweight, M, stream)
    torch.cuda.synchronize()

    dx_err = (dx.to(DTYPE_FP32) - dx_ref).abs().max().item()
    dw_mag = dw_ref.abs().max().item()

    # Tolerances (calibrated). dweight is summed over M -> larger magnitude -> relative.
    rstd_atol = 1e-3
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dw_rtol = {"f32": 1e-4, "f16": 3e-2, "bf16": 1e-1}[dtype]
    dw_atol = {"f32": 1e-2, "f16": 1e-1, "bf16": 5e-1}[dtype]

    print(f"  rstd max abs err    = {rstd_err:.3e} (atol={rstd_atol})")
    print(f"  dx max abs err      = {dx_err:.3e} (atol={dx_atol})")
    print(f"  dweight |max|       = {dw_mag:.3e}")

    dw_ok = True
    try:
        torch.testing.assert_close(dweight, dw_ref, rtol=dw_rtol, atol=dw_atol)
    except AssertionError as e:
        dw_ok = False
        dw_err = (dweight - dw_ref).abs().max().item()
        print(f"  dweight max abs err = {dw_err:.3e} (rtol={dw_rtol}, atol={dw_atol})")
        print(f"  [dweight mismatch] {e}")
    else:
        dw_err = (dweight - dw_ref).abs().max().item()
        print(f"  dweight max abs err = {dw_err:.3e} (rtol={dw_rtol}, atol={dw_atol})")

    ok = rstd_err < rstd_atol and dx_err < dx_atol and dw_ok
    print(f"  -> {'PASSED' if ok else 'FAILED'}")
    return ok


def _reference_fused_add_rmsnorm_bwd(x_dev, residual_dev, weight_dev, dy_dev, dres_out_dev=None):
    """Eager fused-add rmsnorm backward via autograd.

    added = x + residual ; rstd = rsqrt(mean(added^2)+eps) ; out = added*rstd*weight.
    In the prenorm case a downstream grad also flows into residual_out (== added),
    so the graph output includes residual_out and dres_out feeds its grad.
    Returns dx, dresidual, dw, rstd (all fp32).
    """
    x = x_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    res = residual_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    w = weight_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    added = x + res
    rstd = torch.rsqrt((added * added).mean(dim=1, keepdim=True) + EPS)
    out = added * rstd * w
    if dres_out_dev is None:
        outputs = [out]
        grads = [dy_dev.to(DTYPE_FP32)]
    else:
        outputs = [out, added]
        grads = [dy_dev.to(DTYPE_FP32), dres_out_dev.to(DTYPE_FP32)]
    dx, dres, dw = torch.autograd.grad(outputs, [x, res, w], grad_outputs=grads)
    return (
        dx.detach(),
        dres.detach(),
        dw.detach(),
        rstd.detach().squeeze(1).contiguous(),
    )


def run_fused_add_bwd_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting FusedAdd RMSNorm backward (M={M}, N={N}, dtype={dtype})")

    torch_dtype = _torch_dtype(dtype)
    try:
        fwd_fn = build_fused_add_rmsnorm_module(N, dtype, store_rstd=True)
        bwd_fn = build_fused_add_rmsnorm_bwd_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for fused_add bwd (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False

    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    residual = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    dy = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()

    stream = torch.cuda.current_stream()

    # --- forward with store_rstd: residual_out + rstd from the kernel ---
    out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    residual_out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    rstd = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    fwd_c = flyc.compile(fwd_fn, x, residual, weight, out, residual_out, rstd, M, stream)
    fwd_c(x, residual, weight, out, residual_out, rstd, M, stream)
    torch.cuda.synchronize()

    rstd_atol = 1e-3
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dw_rtol = {"f32": 1e-4, "f16": 3e-2, "bf16": 1e-1}[dtype]
    dw_atol = {"f32": 1e-2, "f16": 1e-1, "bf16": 5e-1}[dtype]

    all_ok = True
    for case_name, dres_out in (("dres_out=None", None), ("dres_out=rand", None)):
        if case_name == "dres_out=rand":
            dres_out = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()

        dx_ref, dres_ref, dw_ref, rstd_ref = _reference_fused_add_rmsnorm_bwd(
            x, residual, weight, dy, dres_out_dev=dres_out
        )
        rstd_err = (rstd - rstd_ref).abs().max().item()

        # --- backward: dx (== dresidual by construction), dweight ---
        # The kernel writes dx only; dresidual is dx aliased in the wrapper.
        dx = torch.empty((M, N), device="cuda", dtype=torch_dtype)
        dweight = torch.zeros((N,), device="cuda", dtype=DTYPE_FP32)
        dres_out_arg = dres_out if dres_out is not None else torch.zeros((M, N), device="cuda", dtype=torch_dtype)
        bwd_c = flyc.compile(bwd_fn, residual_out, weight, dy, dres_out_arg, rstd, dx, dweight, M, stream)
        dweight.zero_()
        bwd_c(residual_out, weight, dy, dres_out_arg, rstd, dx, dweight, M, stream)
        torch.cuda.synchronize()

        dx_err = (dx.to(DTYPE_FP32) - dx_ref).abs().max().item()
        # dresidual == dx (kernel computes once); verify against the residual ref too.
        dres_err = (dx.to(DTYPE_FP32) - dres_ref).abs().max().item()

        print(f"  [{case_name}]")
        print(f"    rstd max abs err     = {rstd_err:.3e} (atol={rstd_atol})")
        print(f"    dx max abs err       = {dx_err:.3e} (atol={dx_atol})")
        print(f"    dresidual max abs err= {dres_err:.3e} (atol={dx_atol})")

        dw_ok = True
        try:
            torch.testing.assert_close(dweight, dw_ref, rtol=dw_rtol, atol=dw_atol)
        except AssertionError as e:
            dw_ok = False
            dw_err = (dweight - dw_ref).abs().max().item()
            print(f"    dweight max abs err  = {dw_err:.3e} (rtol={dw_rtol}, atol={dw_atol})")
            print(f"    [dweight mismatch] {e}")
        else:
            dw_err = (dweight - dw_ref).abs().max().item()
            print(f"    dweight max abs err  = {dw_err:.3e} (rtol={dw_rtol}, atol={dw_atol})")

        case_ok = rstd_err < rstd_atol and dx_err < dx_atol and dres_err < dx_atol and dw_ok
        print(f"    -> {'PASSED' if case_ok else 'FAILED'}")
        all_ok = all_ok and case_ok

    print(f"  -> {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def run_fused_add_autograd_test(M: int, N: int, dtype: str = "f32"):
    """End-to-end: public fused_add_rmsnorm() prenorm path with grads on
    x + residual + weight, including batched (3D) reshape."""
    print(f"\nTesting fused_add_rmsnorm() autograd (M={M}, N={N}, dtype={dtype})")
    torch_dtype = _torch_dtype(dtype)
    torch.manual_seed(42)

    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    residual = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    dy = torch.randn((M, N), device="cuda", dtype=torch_dtype)
    dres = torch.randn((M, N), device="cuda", dtype=torch_dtype)

    out, residual_out = fused_add_rmsnorm(x, residual, weight, prenorm=True)
    torch.autograd.backward([out, residual_out], [dy, dres])
    dx_out, dres_out, dw_out = x.grad.detach(), residual.grad.detach(), weight.grad.detach()

    # fp32 autograd reference
    xf = x.detach().to(DTYPE_FP32).requires_grad_(True)
    resf = residual.detach().to(DTYPE_FP32).requires_grad_(True)
    wf = weight.detach().to(DTYPE_FP32).requires_grad_(True)
    added = xf + resf
    rstd = torch.rsqrt((added * added).mean(dim=1, keepdim=True) + EPS)
    yr = added * rstd * wf
    dxr, dresr, dwr = torch.autograd.grad([yr, added], [xf, resf, wf], [dy.to(DTYPE_FP32), dres.to(DTYPE_FP32)])

    out_err = (out.detach().to(DTYPE_FP32) - yr.detach()).abs().max().item()
    dx_err = (dx_out.to(DTYPE_FP32) - dxr).abs().max().item()
    dres_err = (dres_out.to(DTYPE_FP32) - dresr).abs().max().item()
    dw_err = (dw_out.to(DTYPE_FP32) - dwr).abs().max().item()

    out_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dw_atol = {"f32": 1e-2, "f16": 2e-1, "bf16": 1.0}[dtype]

    print(f"  out max abs err = {out_err:.3e} (atol={out_atol})")
    print(f"  dx  max abs err = {dx_err:.3e} (atol={dx_atol})")
    print(f"  dres max abs err= {dres_err:.3e} (atol={dx_atol})")
    print(f"  dw  max abs err = {dw_err:.3e} (atol={dw_atol})")

    # Batched (3D) input must reshape correctly through the public entry.
    x3 = torch.randn((4, M // 4 if M >= 4 else 1, N), device="cuda", dtype=torch_dtype, requires_grad=True)
    r3 = torch.randn_like(x3, requires_grad=True)
    y3, r3_out = fused_add_rmsnorm(x3, r3, weight, prenorm=True)
    shape_ok = tuple(y3.shape) == tuple(x3.shape) and tuple(r3_out.shape) == tuple(x3.shape)
    (y3.sum() + r3_out.sum()).backward()
    grad_ok = x3.grad is not None and tuple(x3.grad.shape) == tuple(x3.shape)
    print(f"  3D reshape: out_shape_ok={shape_ok} grad_shape_ok={grad_ok}")

    ok = out_err < out_atol and dx_err < dx_atol and dres_err < dx_atol and dw_err < dw_atol and shape_ok and grad_ok
    print(f"  -> {'PASSED' if ok else 'FAILED'}")
    return ok


def test_fused_add_rmsnorm_backward():
    print("=" * 80)
    print("Running FusedAdd RMSNorm Backward Tests")
    print("=" * 80)

    configs = [
        (64, 256, "f32"),  # generic path, f32 aligned
        (16, 512, "bf16"),  # generic path, bf16
        (4096, 4096, "bf16"),  # generic path, large
        (64, 2000, "f32"),  # generic path, unaligned
        (128, 4096, "f16"),  # generic path, f16
    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_fused_add_bwd_test(M, N, dtype):
            failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def test_fused_add_rmsnorm_autograd():
    print("=" * 80)
    print("Running fused_add_rmsnorm() Autograd (end-to-end) Tests")
    print("=" * 80)

    configs = [
        (64, 256, "f32"),  # generic path
        (128, 4096, "bf16"),  # generic path
        (128, 4096, "f16"),  # generic path, f16
    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_fused_add_autograd_test(M, N, dtype):
            failures += 1

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def test_fused_add_rmsnorm_dtype_mismatch():
    """Mixed x/residual/weight dtypes must be rejected (kernel uses one elem dtype)."""
    print("=" * 80)
    print("Running fused_add_rmsnorm dtype-mismatch guard Test")
    print("=" * 80)
    N = 256
    x = torch.randn((4, N), device="cuda", dtype=DTYPE_BF16)
    res_bad = torch.randn((4, N), device="cuda", dtype=DTYPE_FP16)
    w = torch.rand((N,), device="cuda", dtype=DTYPE_BF16)
    for bad in (res_bad, w.to(DTYPE_FP32)):
        try:
            fused_add_rmsnorm(x, res_bad if bad is res_bad else torch.randn_like(x), bad if bad is not res_bad else w)
            raise SystemExit("dtype mismatch was NOT rejected")
        except AssertionError:
            pass
    print("  -> PASSED")


@pytest.mark.multi_gpu
def test_fused_add_rmsnorm_device_mismatch():
    """Operands on different devices must be rejected (kernel binds to x.device)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 GPUs")
    print("=" * 80)
    print("Running fused_add_rmsnorm device-mismatch guard Test")
    print("=" * 80)
    N = 256
    x = torch.randn((4, N), device="cuda:0", dtype=DTYPE_BF16)
    residual = torch.randn((4, N), device="cuda:0", dtype=DTYPE_BF16)
    weight = torch.rand((N,), device="cuda:1", dtype=DTYPE_BF16)  # wrong device
    try:
        fused_add_rmsnorm(x, residual, weight)
        raise SystemExit("device mismatch was NOT rejected")
    except AssertionError:
        pass
    print("  -> PASSED")


def _bench_aiter_rmsnorm(M: int, N: int, dtype: str):
    torch_dtype = _torch_dtype(dtype)

    try:
        from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm
    except Exception as e:
        print(f"[Perf] AIter rmsnorm skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
    w = torch.rand((N,), device="cuda", dtype=torch_dtype)

    def run_aiter():
        aiter_rms_norm(x, w, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter rmsnorm gpu: {aiter_us:.1f} us")
    return aiter_us


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


def _bench_aiter_fused_add_rmsnorm(M: int, N: int, dtype: str):
    torch_dtype = _torch_dtype(dtype)

    try:
        from aiter.ops.triton.normalization.rmsnorm import (
            rmsnorm2d_fwd_with_add as aiter_fused_add_rmsnorm,
        )
    except Exception as e:
        print(f"[Perf] AIter fused_add rmsnorm skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    residual_in = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    residual_out = torch.empty((M, N), device="cuda", dtype=torch_dtype)

    def run_aiter():
        aiter_fused_add_rmsnorm(out, x, residual_in, residual_out, w, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter fused_add rmsnorm gpu: {aiter_us:.1f} us")
    return aiter_us


def _bench_aiter_fused_add_rmsnorm_quant(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    torch_dtype = _torch_dtype(dtype)

    try:
        if is_smooth:
            from aiter.ops.triton.normalization.rmsnorm import (
                rmsnorm2d_fwd_with_add_smoothquant as aiter_fused_add_rmsnorm_quant,
            )
        else:
            from aiter.ops.triton.normalization.rmsnorm import (
                rmsnorm2d_fwd_with_add_dynamicquant as aiter_fused_add_rmsnorm_quant,
            )
    except Exception as e:
        print(f"[Perf] AIter fused_add rmsnorm {mode} skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    residual_in = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    y = torch.empty((M, N), dtype=torch.int8, device="cuda")
    residual_out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    yscale = torch.empty((M, 1), dtype=torch.float32, device="cuda")

    if is_smooth:
        xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()

        def run_aiter():
            aiter_fused_add_rmsnorm_quant(y, x, residual_in, residual_out, xscale, yscale, w, EPS)

    else:

        def run_aiter():
            aiter_fused_add_rmsnorm_quant(y, x, residual_in, residual_out, yscale, w, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter fused_add rmsnorm {mode} gpu: {aiter_us:.1f} us")
    return aiter_us


def test_rmsnorm():
    print("=" * 80)
    print("Running RMSNorm Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_rmsnorm(M, N, dtype)

            perf_rows.append(
                PerfRow(op="rmsnorm", shape=f"{M}x{N}", dtype=dtype, flydsl_gpu_us=flydsl_gpu_us, aiter_gpu_us=aiter_us)
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def test_rmsnorm_backward():
    print("=" * 80)
    print("Running RMSNorm Backward Tests")
    print("=" * 80)

    configs = [
        (64, 256, "f32"),  # small-N path, f32
        (16, 512, "bf16"),  # small-N path, bf16
        (4096, 4096, "bf16"),  # fast vectorized path (N % 2048 == 0, 16-bit)
        (64, 2000, "f32"),  # small-N path, unaligned f32
        (128, 4096, "f16"),  # fast vectorized path, f16
        (64, 3000, "f32"),  # generic scalar path (N > 2048, f32)
    ]

    failures = 0
    for M, N, dtype in configs:
        ok = run_bwd_test(M, N, dtype)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def run_autograd_test(M: int, N: int, dtype: str = "f32"):
    """End-to-end: the public rmsnorm() autograd path (what quack calls),
    including batched (>2D) input reshape and grads on x + weight."""
    print(f"\nTesting rmsnorm() autograd (M={M}, N={N}, dtype={dtype})")
    torch_dtype = _torch_dtype(dtype)
    torch.manual_seed(42)

    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    dy = torch.randn((M, N), device="cuda", dtype=torch_dtype)

    out = rmsnorm(x, weight)
    out.backward(dy)
    dx_out, dw_out = x.grad.detach(), weight.grad.detach()

    # fp32 autograd reference
    xf = x.detach().to(DTYPE_FP32).requires_grad_(True)
    wf = weight.detach().to(DTYPE_FP32).requires_grad_(True)
    rstd = torch.rsqrt((xf * xf).mean(dim=1, keepdim=True) + EPS)
    yr = xf * rstd * wf
    dxr, dwr = torch.autograd.grad(yr, [xf, wf], dy.to(DTYPE_FP32))

    out_err = (out.detach().to(DTYPE_FP32) - yr.detach()).abs().max().item()
    dx_err = (dx_out.to(DTYPE_FP32) - dxr).abs().max().item()
    dw_err = (dw_out.to(DTYPE_FP32) - dwr).abs().max().item()

    out_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dw_atol = {"f32": 1e-2, "f16": 2e-1, "bf16": 1.0}[dtype]

    print(f"  out max abs err = {out_err:.3e} (atol={out_atol})")
    print(f"  dx  max abs err = {dx_err:.3e} (atol={dx_atol})")
    print(f"  dw  max abs err = {dw_err:.3e} (atol={dw_atol})")

    # Batched (3D) input must reshape correctly through the public entry.
    x3 = torch.randn((4, M // 4 if M >= 4 else 1, N), device="cuda", dtype=torch_dtype, requires_grad=True)
    y3 = rmsnorm(x3, weight)
    shape_ok = tuple(y3.shape) == tuple(x3.shape)
    y3.sum().backward()
    grad_ok = x3.grad is not None and tuple(x3.grad.shape) == tuple(x3.shape)
    print(f"  3D reshape: out_shape_ok={shape_ok} grad_shape_ok={grad_ok}")

    ok = out_err < out_atol and dx_err < dx_atol and dw_err < dw_atol and shape_ok and grad_ok
    print(f"  -> {'PASSED' if ok else 'FAILED'}")
    return ok


def test_rmsnorm_autograd():
    print("=" * 80)
    print("Running rmsnorm() Autograd (end-to-end) Tests")
    print("=" * 80)

    configs = [
        (64, 256, "f32"),  # small-N path
        (128, 4096, "bf16"),  # fast vectorized path
        (128, 4096, "f16"),  # fast vectorized path, f16
        (128, 3000, "f32"),  # generic scalar path (N > 2048, f32)
    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_autograd_test(M, N, dtype):
            failures += 1

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def test_rmsnorm_eps_honored():
    """eps must be baked into the kernel, not silently replaced by the module EPS."""
    print("=" * 80)
    print("Running RMSNorm eps-honored Test")
    print("=" * 80)
    torch.manual_seed(0)
    # Cover both forward builders: small-N (N <= 2048) and the generic scalar path
    # (N > 2048; f32 avoids the 16-bit fast path), so eps is verified on each.
    for M, N in ((32, 256), (32, 3000)):
        x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
        w = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

        for eps in (1e-5, 1e-6, 1e-2):
            y = rmsnorm(x, w, eps=eps)
            ref = x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + eps) * w
            err = (y - ref).abs().max().item()
            print(f"  N={N} eps={eps:g}: max err vs torch ref = {err:.3e}")
            assert err < 1e-4, f"N={N} eps={eps} not honored (err={err})"

        # A non-default eps must actually change the output (guards silent-ignore regressions).
        diff = (rmsnorm(x, w, eps=1e-2) - rmsnorm(x, w, eps=1e-6)).abs().max().item()
        print(f"  N={N} eps 1e-2 vs 1e-6 output diff = {diff:.3e} (must be > 0)")
        assert diff > 0, f"N={N}: eps appears to be ignored"
    print("  -> PASSED")


@pytest.mark.multi_gpu
def test_rmsnorm_multi_gpu():
    """Compiled-fn cache must not reuse a device-0 kernel on device 1 (would fault)."""
    print("=" * 80)
    print("Running RMSNorm multi-GPU Test")
    print("=" * 80)
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    torch.manual_seed(0)
    N = 256
    for dev in ("cuda:0", "cuda:1"):
        x = torch.randn((16, N), device=dev, dtype=DTYPE_FP32, requires_grad=True)
        w = torch.rand((N,), device=dev, dtype=DTYPE_FP32, requires_grad=True)
        dy = torch.randn((16, N), device=dev, dtype=DTYPE_FP32)
        y = rmsnorm(x, w)
        y.backward(dy)
        torch.cuda.synchronize(dev)
        ref = x.detach() / torch.sqrt((x.detach() ** 2).mean(1, keepdim=True) + EPS) * w.detach()
        err = (y.detach() - ref).abs().max().item()
        print(f"  {dev}: out err={err:.3e}, dx finite={torch.isfinite(x.grad).all().item()}")
        assert err < 1e-4 and torch.isfinite(x.grad).all()
    print("  -> PASSED")


@pytest.mark.large_shape
def test_rmsnorm_large_shape():
    print("=" * 80)
    print("Running RMSNorm Large Shape Tests")
    print("=" * 80)

    for M, N, dtype in _get_rmsnorm_large_configs():
        ok, _ = run_test(M, N, dtype)
        assert ok


def test_rmsnorm_dynamicquant():
    print("=" * 80)
    print("Running RMSNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_quant_test(M, N, dtype, is_smooth=False)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_rmsnorm_quant(M, N, dtype, is_smooth=False)

            perf_rows.append(
                PerfRow(
                    op="rmsnorm_dynamicquant",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def test_rmsnorm_smoothquant():
    print("=" * 80)
    print("Running RMSNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_quant_test(M, N, dtype, is_smooth=True)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_rmsnorm_quant(M, N, dtype, is_smooth=True)

            perf_rows.append(
                PerfRow(
                    op="rmsnorm_smoothquant",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def test_fused_add_rmsnorm():
    print("=" * 80)
    print("Running FusedAdd RMSNorm Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_fused_add_rmsnorm(M, N, dtype)
            perf_rows.append(
                PerfRow(
                    op="rmsnorm_add",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


def test_fused_add_rmsnorm_dynamicquant():
    print("=" * 80)
    print("Running FusedAdd RMSNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_quant_test(M, N, dtype, is_smooth=False)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_fused_add_rmsnorm_quant(M, N, dtype, is_smooth=False)
            perf_rows.append(
                PerfRow(
                    op="rmsnorm_add_dynamicquant",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(1)


def test_fused_add_rmsnorm_smoothquant():
    print("=" * 80)
    print("Running FusedAdd RMSNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_rmsnorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_quant_test(M, N, dtype, is_smooth=True)
        if not ok:
            failures += 1

        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_fused_add_rmsnorm_quant(M, N, dtype, is_smooth=True)
            perf_rows.append(
                PerfRow(
                    op="rmsnorm_add_smoothquant",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_rmsnorm()
    test_rmsnorm_backward()
    test_rmsnorm_autograd()
    test_rmsnorm_eps_honored()
    if torch.cuda.device_count() >= 2:
        test_rmsnorm_multi_gpu()
    test_rmsnorm_dynamicquant()
    test_rmsnorm_smoothquant()
    test_fused_add_rmsnorm()
    test_fused_add_rmsnorm_backward()
    test_fused_add_rmsnorm_autograd()
    test_fused_add_rmsnorm_dtype_mismatch()
    test_fused_add_rmsnorm_dynamicquant()
    test_fused_add_rmsnorm_smoothquant()
