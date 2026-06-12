#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
LayerNorm Operator Test
Implementation of a Block-wise LayerNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (mean and variance)

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
"""

import os

import pytest

from kernels.layernorm_kernel import (
    build_fused_add_layernorm_dynamicquant_module,
    build_fused_add_layernorm_module,
    build_fused_add_layernorm_smoothquant_module,
    build_layernorm_dynamicquant_module,
    build_layernorm_module,
    build_layernorm_smoothquant_module,
)
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)
from tests.test_common import run_perftest

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


def _get_layernorm_configs():
    shapes_env = os.environ.get("ROCDSL_LAYERNORM_SHAPES", "").strip()
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
            (64, 256, "f32"),     # Aligned
            (128, 1024, "f32"),   # Aligned
            (32, 128, "f16"),     # Aligned
            (64, 2000, "f32"),    # Unaligned (tail handling)
            (16, 512, "bf16"),    # BF16
            (1024, 8192, "bf16"), # BF16
            (32768, 8192, "bf16"),
        ]
    return configs


def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_layernorm_module(M, N, dtype)
    except ValueError as e:
        print(f"[FAIL] Compile failed: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # PyTorch CPU Reference (variance uses unbiased=False)
    x = input_ref
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    expected = expected.to(DTYPE_FP32)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, beta_dev, output_dev, M, stream=stream)

    # One run for correctness visibility, then benchmark via shared harness.
    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (2 * M * N + 2 * N) * elem_bytes  # read input + write output + (gamma+beta)
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL layernorm gpu: {flydsl_gpu_us:.1f} us")

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


def test_layernorm():
    print("=" * 80)
    print("Running LayerNorm Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

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
                    from aiter.ops.triton.norm import layer_norm as aiter_layer_norm

                    x = torch.randn(
                        (M, N),
                        device="cuda",
                        dtype=_torch_dtype(dtype),
                    )
                    w = torch.rand((N,), device="cuda", dtype=x.dtype)
                    b = torch.rand((N,), device="cuda", dtype=x.dtype)

                    def run_aiter():
                        aiter_layer_norm(x, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter layernorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter layernorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm", shape=f"{M}x{N}", dtype=dtype, flydsl_gpu_us=flydsl_gpu_us, aiter_gpu_us=aiter_us
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


def run_fused_add_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting FusedAdd LayerNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_fused_add_layernorm_module(M, N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for fused_add layernorm (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        residual_dev = residual_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        residual_dev = residual_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        residual_dev = residual_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    residual_expected = (input_dev + residual_dev).to(DTYPE_FP32)
    x = residual_expected
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    expected = expected.to(DTYPE_FP32)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, residual_dev, gamma_dev, beta_dev, output_dev, residual_out_dev, M, stream=stream)

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (4 * M * N + 2 * N) * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add layernorm gpu: {flydsl_gpu_us:.1f} us")

    output_ref = output_dev.to(DTYPE_FP32)
    residual_out_ref = residual_out_dev.to(DTYPE_FP32)

    output_error = (output_ref - expected).abs().max().item()
    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    print(f"Max output error: {output_error:.2e} (atol={atol})")
    print(f"Max residual error: {residual_error:.2e} (atol={atol})")

    if output_error < atol and residual_error < atol:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
        ok = False

    return ok, flydsl_gpu_us


def test_fused_add_layernorm():
    print("=" * 80)
    print("Running FusedAdd LayerNorm Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch

            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_add

                    torch_dtype = _torch_dtype(dtype)
                    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual_out = torch.empty_like(x)
                    out = torch.empty_like(x)
                    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()

                    def run_aiter():
                        layernorm2d_fwd_with_add(out, x, residual, residual_out, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter fused_add layernorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter fused_add layernorm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm_fused_add",
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


def run_dynamicquant_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm DynamicQuant (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_layernorm_dynamicquant_module(M, N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for dynamicquant layernorm (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)

    x = input_ref
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    yscale_expected = expected.abs().amax(dim=1) / 127.0
    yscale_expected = torch.where(yscale_expected == 0, torch.ones_like(yscale_expected), yscale_expected)
    q_expected = torch.clamp(torch.trunc(expected / yscale_expected.unsqueeze(1)), -127, 127).to(DTYPE_INT8)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, beta_dev, output_dev, yscale_dev, M, stream=stream)

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (M * N + 2 * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL layernorm dynamicquant gpu: {flydsl_gpu_us:.1f} us")

    output_ref = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    recon_error = (output_ref - expected).abs().max().item()
    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max recon error: {recon_error:.2e} (tol=0.3)")
    print(f"Max scale diff: {scale_diff:.2e} (tol=1e-2)")
    print(f"Max quant diff: {quant_diff}")

    if recon_error < 0.3 and scale_diff < 1e-2 and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        print("First row Quant Expected:")
        print(q_ref[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_ref[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
        ok = False

    return ok, flydsl_gpu_us


def test_layernorm_dynamicquant():
    print("=" * 80)
    print("Running LayerNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_dynamicquant_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch

            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_dynamicquant

                    torch_dtype = _torch_dtype(dtype)
                    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
                    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

                    def run_aiter():
                        layernorm2d_fwd_with_dynamicquant(q_out, x, yscale, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter layernorm dynamicquant gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter layernorm dynamicquant skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm_dynamicquant",
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


def run_smoothquant_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm SmoothQuant (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_layernorm_smoothquant_module(M, N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for smoothquant layernorm (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    xscale_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        xscale_dev = xscale_t.contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        xscale_dev = xscale_t.to(DTYPE_FP16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        xscale_dev = xscale_t.to(DTYPE_BF16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)

    x = input_ref
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    expected = expected * xscale_ref
    yscale_expected = expected.abs().amax(dim=1) / 127.0
    yscale_expected = torch.where(yscale_expected == 0, torch.ones_like(yscale_expected), yscale_expected)
    q_expected = torch.clamp(torch.trunc(expected / yscale_expected.unsqueeze(1)), -127, 127).to(DTYPE_INT8)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, beta_dev, xscale_dev, output_dev, yscale_dev, M, stream=stream)

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (M * N + 3 * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL layernorm smoothquant gpu: {flydsl_gpu_us:.1f} us")

    output_ref = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    recon_error = (output_ref - expected).abs().max().item()
    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max recon error: {recon_error:.2e} (tol=0.3)")
    print(f"Max scale diff: {scale_diff:.2e} (tol=1e-2)")
    print(f"Max quant diff: {quant_diff}")

    if recon_error < 0.3 and scale_diff < 1e-2 and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        print("First row Quant Expected:")
        print(q_ref[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_ref[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
        ok = False

    return ok, flydsl_gpu_us


def test_layernorm_smoothquant():
    print("=" * 80)
    print("Running LayerNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_smoothquant_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch

            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_smoothquant

                    torch_dtype = _torch_dtype(dtype)
                    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()
                    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
                    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

                    def run_aiter():
                        layernorm2d_fwd_with_smoothquant(q_out, x, xscale, yscale, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter layernorm smoothquant gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter layernorm smoothquant skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm_smoothquant",
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


def run_fused_add_dynamicquant_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting FusedAdd LayerNorm DynamicQuant (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_fused_add_layernorm_dynamicquant_module(M, N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for fused_add dynamicquant layernorm "
            f"(M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        residual_dev = residual_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        residual_atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        residual_dev = residual_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        residual_atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        residual_dev = residual_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        residual_atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)

    residual_expected = (input_dev + residual_dev).to(DTYPE_FP32)
    x = residual_expected
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    yscale_expected = expected.abs().amax(dim=1) / 127.0
    yscale_expected = torch.where(yscale_expected == 0, torch.ones_like(yscale_expected), yscale_expected)
    q_expected = torch.clamp(torch.trunc(expected / yscale_expected.unsqueeze(1)), -127, 127).to(DTYPE_INT8)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(
            input_dev, residual_dev, gamma_dev, beta_dev, output_dev, residual_out_dev, yscale_dev, M, stream=stream
        )

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (3 * M * N + 2 * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add layernorm dynamicquant gpu: {flydsl_gpu_us:.1f} us")

    residual_out_ref = residual_out_dev.to(DTYPE_FP32)
    output_ref = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    recon_error = (output_ref - expected).abs().max().item()
    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max residual error: {residual_error:.2e} (atol={residual_atol})")
    print(f"Max recon error: {recon_error:.2e} (tol=0.3)")
    print(f"Max scale diff: {scale_diff:.2e} (tol=1e-2)")
    print(f"Max quant diff: {quant_diff}")

    if residual_error < residual_atol and recon_error < 0.3 and scale_diff < 1e-2 and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        print("First row Quant Expected:")
        print(q_ref[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_ref[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
        ok = False

    return ok, flydsl_gpu_us


def test_fused_add_layernorm_dynamicquant():
    print("=" * 80)
    print("Running FusedAdd LayerNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_dynamicquant_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch

            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_add_dynamicquant

                    torch_dtype = _torch_dtype(dtype)
                    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual_out = torch.empty_like(x)
                    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
                    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

                    def run_aiter():
                        layernorm2d_fwd_with_add_dynamicquant(q_out, x, residual, residual_out, yscale, w, b, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter fused_add layernorm dynamicquant gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter fused_add layernorm dynamicquant skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm_fused_add_dynamicquant",
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


def run_fused_add_smoothquant_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting FusedAdd LayerNorm SmoothQuant (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_fused_add_layernorm_smoothquant_module(M, N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for fused_add smoothquant layernorm "
            f"(M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    xscale_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5

    if dtype == "f32":
        input_dev = input_t.contiguous()
        residual_dev = residual_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        beta_dev = beta_t.contiguous()
        xscale_dev = xscale_t.contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
        residual_atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        residual_dev = residual_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        beta_dev = beta_t.to(DTYPE_FP16).contiguous()
        xscale_dev = xscale_t.to(DTYPE_FP16).contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
        residual_atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        residual_dev = residual_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        beta_dev = beta_t.to(DTYPE_BF16).contiguous()
        xscale_dev = xscale_t.to(DTYPE_BF16).contiguous()
        residual_out_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        beta_ref = beta_dev.to(DTYPE_FP32)
        xscale_ref = xscale_dev.to(DTYPE_FP32)
        residual_atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)

    residual_expected = (input_dev + residual_dev).to(DTYPE_FP32)
    x = residual_expected
    gamma = gamma_ref
    beta = beta_ref
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    expected = expected * xscale_ref
    yscale_expected = expected.abs().amax(dim=1) / 127.0
    yscale_expected = torch.where(yscale_expected == 0, torch.ones_like(yscale_expected), yscale_expected)
    q_expected = torch.clamp(torch.trunc(expected / yscale_expected.unsqueeze(1)), -127, 127).to(DTYPE_INT8)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(
            input_dev,
            residual_dev,
            gamma_dev,
            beta_dev,
            xscale_dev,
            output_dev,
            residual_out_dev,
            yscale_dev,
            M,
            stream=stream,
        )

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = (3 * M * N + 3 * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add layernorm smoothquant gpu: {flydsl_gpu_us:.1f} us")

    residual_out_ref = residual_out_dev.to(DTYPE_FP32)
    output_ref = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    recon_error = (output_ref - expected).abs().max().item()
    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max residual error: {residual_error:.2e} (atol={residual_atol})")
    print(f"Max recon error: {recon_error:.2e} (tol=0.3)")
    print(f"Max scale diff: {scale_diff:.2e} (tol=1e-2)")
    print(f"Max quant diff: {quant_diff}")

    if residual_error < residual_atol and recon_error < 0.3 and scale_diff < 1e-2 and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        print("First row Quant Expected:")
        print(q_ref[0, :8])
        print("First row Quant Actual:")
        print(q_out[0, :8])
        print("First few YScale Expected:")
        print(yscale_ref[:5])
        print("First few YScale Actual:")
        print(yscale_out[:5])
        ok = False

    return ok, flydsl_gpu_us


def test_fused_add_layernorm_smoothquant():
    print("=" * 80)
    print("Running FusedAdd LayerNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0

    for M, N, dtype in configs:
        ok, flydsl_gpu_us = run_fused_add_smoothquant_test(M, N, dtype)
        if not ok:
            failures += 1

        if do_compare:
            import torch

            aiter_us = None
            if maybe_enable_aiter():
                try:
                    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_add_smoothquant

                    torch_dtype = _torch_dtype(dtype)
                    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
                    residual_out = torch.empty_like(x)
                    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
                    xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()
                    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
                    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

                    def run_aiter():
                        layernorm2d_fwd_with_add_smoothquant(
                            q_out, x, residual, residual_out, xscale, yscale, w, b, EPS
                        )

                    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter fused_add layernorm smoothquant gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter fused_add layernorm smoothquant skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="layernorm_fused_add_smoothquant",
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
    test_layernorm()
    test_fused_add_layernorm()
    test_layernorm_dynamicquant()
    test_layernorm_smoothquant()
    test_fused_add_layernorm_dynamicquant()
    test_fused_add_layernorm_smoothquant()
