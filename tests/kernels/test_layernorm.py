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

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.norm.layernorm_kernel import (
    build_fused_add_layernorm_dynamicquant_module,
    build_fused_add_layernorm_module,
    build_fused_add_layernorm_smoothquant_module,
    build_layernorm_bwd_module,
    build_layernorm_dynamicquant_module,
    build_layernorm_module,
    build_layernorm_smoothquant_module,
    layernorm,
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


def _flydsl_elem_dtype(dtype: str):
    if dtype == "f32":
        return fx.Float32
    if dtype == "f16":
        return fx.Float16
    if dtype == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _as_pointer(tensor, elem_dtype):
    return flyc.from_c_void_p(elem_dtype, tensor.data_ptr())


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
            (64, 256, "f32"),  # f32 aligned
            (32, 128, "f16"),  # f16 aligned
            (64, 2000, "f32"),  # unaligned tail handling
            (16, 512, "bf16"),  # bf16 small shape
            (64, 8192, "bf16"),  # bf16 fast-path N with small M
        ]
    return configs


def _get_layernorm_large_configs():
    return [
        (32768, 8192, "bf16"),
    ]


def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_layernorm_module(N, dtype)
    except ValueError as e:
        print(f"[FAIL] Compile failed: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    beta_dev = beta_t.to(torch_dtype).contiguous()
    output_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    input_ref = input_dev.to(DTYPE_FP32)
    gamma_ref = gamma_dev.to(DTYPE_FP32)
    beta_ref = beta_dev.to(DTYPE_FP32)
    if dtype == "f32":
        atol = 1e-4
    elif dtype == "f16":
        atol = 1e-2
    elif dtype == "bf16":
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    expected = _reference_layernorm(input_ref, gamma_ref, beta_ref)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()
    elem_dtype = _flydsl_elem_dtype(dtype)
    compiled_fn = flyc.compile(
        launch_fn,
        _as_pointer(input_dev, elem_dtype),
        _as_pointer(gamma_dev, elem_dtype),
        _as_pointer(beta_dev, elem_dtype),
        _as_pointer(output_dev, elem_dtype),
        M,
        stream,
    )

    def kernel_launch():
        compiled_fn(
            input_dev.data_ptr(),
            gamma_dev.data_ptr(),
            beta_dev.data_ptr(),
            output_dev.data_ptr(),
            M,
            stream,
        )

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


def run_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting LayerNorm {mode} (M={M}, N={N}, dtype={dtype})")

    try:
        if is_smooth:
            launch_fn = build_layernorm_smoothquant_module(N, dtype)
        else:
            launch_fn = build_layernorm_dynamicquant_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for {mode} layernorm (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    xscale_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5 if is_smooth else None

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    beta_dev = beta_t.to(torch_dtype).contiguous()
    input_ref = input_dev.to(DTYPE_FP32)
    gamma_ref = gamma_dev.to(DTYPE_FP32)
    beta_ref = beta_dev.to(DTYPE_FP32)
    if is_smooth:
        xscale_dev = xscale_t.to(torch_dtype).contiguous()

    output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    scale_tol = 1e-3

    q_expected, yscale_expected = _reference_layernorm_quant(
        input_ref,
        gamma_ref,
        beta_ref,
        xscale_dev=xscale_dev if is_smooth else None,
    )

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    if is_smooth:
        compiled_fn = flyc.compile(
            launch_fn, input_dev, gamma_dev, beta_dev, xscale_dev, output_dev, yscale_dev, M, stream
        )

        def kernel_launch():
            compiled_fn(input_dev, gamma_dev, beta_dev, xscale_dev, output_dev, yscale_dev, M, stream)

    else:
        compiled_fn = flyc.compile(launch_fn, input_dev, gamma_dev, beta_dev, output_dev, yscale_dev, M, stream)

        def kernel_launch():
            compiled_fn(input_dev, gamma_dev, beta_dev, output_dev, yscale_dev, M, stream)

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
    total_bytes = (M * N + (3 if is_smooth else 2) * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL layernorm {mode} gpu: {flydsl_gpu_us:.1f} us")

    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max quant diff: {quant_diff}")
    print(f"Max scale diff: {scale_diff:.2e} (tol={scale_tol})")

    if scale_diff < scale_tol and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
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


def run_fused_add_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting FusedAdd LayerNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_fused_add_layernorm_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for fused_add layernorm (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    residual_dev = residual_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    beta_dev = beta_t.to(torch_dtype).contiguous()
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

    residual_expected, expected = _reference_fused_add_layernorm(input_dev, residual_dev, gamma_dev, beta_dev)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()
    compiled_fn = flyc.compile(
        launch_fn,
        input_dev,
        residual_dev,
        gamma_dev,
        beta_dev,
        output_dev,
        residual_out_dev,
        M,
        stream,
    )

    def kernel_launch():
        compiled_fn(input_dev, residual_dev, gamma_dev, beta_dev, output_dev, residual_out_dev, M, stream)

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


def run_fused_add_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting FusedAdd LayerNorm {mode} (M={M}, N={N}, dtype={dtype})")

    try:
        if is_smooth:
            launch_fn = build_fused_add_layernorm_smoothquant_module(N, dtype)
        else:
            launch_fn = build_fused_add_layernorm_dynamicquant_module(N, dtype)
    except Exception as e:
        print(
            f"[FAIL] Compile failed for fused_add {mode} layernorm "
            f"(M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}"
        )
        return False, None

    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    xscale_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5 if is_smooth else None

    torch_dtype = _torch_dtype(dtype)
    input_dev = input_t.to(torch_dtype).contiguous()
    residual_dev = residual_t.to(torch_dtype).contiguous()
    gamma_dev = gamma_t.to(torch_dtype).contiguous()
    beta_dev = beta_t.to(torch_dtype).contiguous()
    residual_out_dev = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    if is_smooth:
        xscale_dev = xscale_t.to(torch_dtype).contiguous()
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
    scale_tol = 1e-3

    residual_expected, q_expected, yscale_expected = _reference_fused_add_layernorm_quant(
        input_dev,
        residual_dev,
        gamma_dev,
        beta_dev,
        xscale_dev=xscale_dev if is_smooth else None,
    )

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    if is_smooth:
        compiled_fn = flyc.compile(
            launch_fn,
            input_dev,
            residual_dev,
            gamma_dev,
            beta_dev,
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
                residual_dev,
                gamma_dev,
                beta_dev,
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
            residual_dev,
            gamma_dev,
            beta_dev,
            output_dev,
            residual_out_dev,
            yscale_dev,
            M,
            stream,
        )

        def kernel_launch():
            compiled_fn(
                input_dev,
                residual_dev,
                gamma_dev,
                beta_dev,
                output_dev,
                residual_out_dev,
                yscale_dev,
                M,
                stream,
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
    total_bytes = (3 * M * N + (3 if is_smooth else 2) * N) * elem_bytes + M * N + M * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms via run_perftest (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL fused_add layernorm {mode} gpu: {flydsl_gpu_us:.1f} us")

    residual_out_ref = residual_out_dev.to(DTYPE_FP32)
    q_out = output_dev.to(torch.int16)
    q_ref = q_expected.to(torch.int16)
    yscale_out = yscale_dev.cpu()
    yscale_ref = yscale_expected.cpu()

    residual_error = (residual_out_ref - residual_expected).abs().max().item()
    scale_diff = (yscale_out - yscale_ref).abs().max().item()
    quant_diff = (q_out - q_ref).abs().max().item()

    print(f"Max residual error: {residual_error:.2e} (atol={residual_atol})")
    print(f"Max quant diff: {quant_diff}")
    print(f"Max scale diff: {scale_diff:.2e} (tol={scale_tol})")

    if residual_error < residual_atol and scale_diff < scale_tol and quant_diff <= 1:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Residual Expected:")
        print(residual_expected[0, :5])
        print("First row Residual Actual:")
        print(residual_out_ref[0, :5])
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


def _reference_layernorm(input_dev, gamma_dev, beta_dev):
    x = input_dev.to(DTYPE_FP32)
    gamma = gamma_dev.to(DTYPE_FP32)
    beta = beta_dev.to(DTYPE_FP32)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    return ((x - mean) / torch.sqrt(var + EPS) * gamma + beta).to(DTYPE_FP32)


def _reference_layernorm_quant(input_dev, gamma_dev, beta_dev, *, xscale_dev=None):
    normalized = _reference_layernorm(input_dev, gamma_dev, beta_dev)
    if xscale_dev is not None:
        normalized = normalized * xscale_dev.to(DTYPE_FP32)

    yscale = normalized.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(normalized / yscale.unsqueeze(1)), -127, 127).to(DTYPE_INT8)
    return q, yscale


def _reference_fused_add_layernorm(input_dev, residual_dev, gamma_dev, beta_dev):
    added = input_dev + residual_dev
    residual_expected = added.to(DTYPE_FP32)
    expected = _reference_layernorm(added, gamma_dev, beta_dev)
    return residual_expected, expected


def _reference_fused_add_layernorm_quant(input_dev, residual_dev, gamma_dev, beta_dev, *, xscale_dev=None):
    added = input_dev + residual_dev
    residual_expected = added.to(DTYPE_FP32)
    q, yscale = _reference_layernorm_quant(
        added,
        gamma_dev,
        beta_dev,
        xscale_dev=xscale_dev,
    )
    return residual_expected, q, yscale


def _bench_aiter_layernorm(M: int, N: int, dtype: str):
    torch_dtype = _torch_dtype(dtype)

    try:
        from aiter.ops.triton.norm import layer_norm as aiter_layer_norm
    except Exception as e:
        print(f"[Perf] AIter layernorm skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
    w = torch.rand((N,), device="cuda", dtype=torch_dtype)
    b = torch.rand((N,), device="cuda", dtype=torch_dtype)

    def run_aiter():
        aiter_layer_norm(x, w, b, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter layernorm gpu: {aiter_us:.1f} us")
    return aiter_us


def _bench_aiter_fused_add_layernorm(M: int, N: int, dtype: str):
    torch_dtype = _torch_dtype(dtype)

    try:
        from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_add
    except Exception as e:
        print(f"[Perf] AIter fused_add layernorm skipped: {type(e).__name__}: {e!r}")
        return None

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
    return aiter_us


def _bench_aiter_layernorm_quant(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    torch_dtype = _torch_dtype(dtype)

    try:
        if is_smooth:
            from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_smoothquant as aiter_layernorm_quant
        else:
            from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_dynamicquant as aiter_layernorm_quant
    except Exception as e:
        print(f"[Perf] AIter layernorm {mode} skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

    if is_smooth:
        xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()

        def run_aiter():
            aiter_layernorm_quant(q_out, x, xscale, yscale, w, b, EPS)

    else:

        def run_aiter():
            aiter_layernorm_quant(q_out, x, yscale, w, b, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter layernorm {mode} gpu: {aiter_us:.1f} us")
    return aiter_us


def _bench_aiter_fused_add_layernorm_quant(M: int, N: int, dtype: str, *, is_smooth: bool):
    mode = "smoothquant" if is_smooth else "dynamicquant"
    torch_dtype = _torch_dtype(dtype)

    try:
        if is_smooth:
            from aiter.ops.triton.normalization.norm import (
                layernorm2d_fwd_with_add_smoothquant as aiter_fused_add_layernorm_quant,
            )
        else:
            from aiter.ops.triton.normalization.norm import (
                layernorm2d_fwd_with_add_dynamicquant as aiter_fused_add_layernorm_quant,
            )
    except Exception as e:
        print(f"[Perf] AIter fused_add layernorm {mode} skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    residual = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    residual_out = torch.empty_like(x)
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    q_out = torch.empty((M, N), device="cuda", dtype=DTYPE_INT8)
    yscale = torch.empty((M, 1), device="cuda", dtype=DTYPE_FP32)

    if is_smooth:
        xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()

        def run_aiter():
            aiter_fused_add_layernorm_quant(q_out, x, residual, residual_out, xscale, yscale, w, b, EPS)

    else:

        def run_aiter():
            aiter_fused_add_layernorm_quant(q_out, x, residual, residual_out, yscale, w, b, EPS)

    aiter_us = bench_gpu_us_torch(run_aiter, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter fused_add layernorm {mode} gpu: {aiter_us:.1f} us")
    return aiter_us


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
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_layernorm(M, N, dtype)

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


@pytest.mark.large_shape
def test_layernorm_large_shape():
    print("=" * 80)
    print("Running LayerNorm Large Shape Tests")
    print("=" * 80)

    for M, N, dtype in _get_layernorm_large_configs():
        ok, _ = run_test(M, N, dtype)
        assert ok


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
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter_fused_add_layernorm(M, N, dtype)

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


def test_layernorm_dynamicquant():
    print("=" * 80)
    print("Running LayerNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

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
                aiter_us = _bench_aiter_layernorm_quant(M, N, dtype, is_smooth=False)

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


def test_layernorm_smoothquant():
    print("=" * 80)
    print("Running LayerNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

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
                aiter_us = _bench_aiter_layernorm_quant(M, N, dtype, is_smooth=True)

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


def test_fused_add_layernorm_dynamicquant():
    print("=" * 80)
    print("Running FusedAdd LayerNorm DynamicQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

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
                aiter_us = _bench_aiter_fused_add_layernorm_quant(M, N, dtype, is_smooth=False)

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


def test_fused_add_layernorm_smoothquant():
    print("=" * 80)
    print("Running FusedAdd LayerNorm SmoothQuant Tests")
    print("=" * 80)

    configs = _get_layernorm_configs()

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
                aiter_us = _bench_aiter_fused_add_layernorm_quant(M, N, dtype, is_smooth=True)

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


def _reference_layernorm_bwd(x_dev, weight_dev, bias_dev, dy_dev):
    """Eager layernorm backward via autograd. Returns dx, dgamma, dbias, mean, rstd (all fp32)."""
    x = x_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    w = weight_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    b = bias_dev.detach().to(DTYPE_FP32).requires_grad_(True)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + EPS)
    y = (x - mean) * rstd * w + b
    dx, dgamma, dbias = torch.autograd.grad(y, [x, w, b], grad_outputs=dy_dev.to(DTYPE_FP32))
    return (
        dx.detach(),
        dgamma.detach(),
        dbias.detach(),
        mean.detach().squeeze(1).contiguous(),
        rstd.detach().squeeze(1).contiguous(),
    )


def run_layernorm_bwd_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm backward (M={M}, N={N}, dtype={dtype})")

    torch_dtype = _torch_dtype(dtype)
    try:
        fwd_fn = build_layernorm_module(N, dtype, store_stats=True)
        bwd_fn = build_layernorm_bwd_module(N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for bwd (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False

    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    bias = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    dy = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()

    dx_ref, dgamma_ref, dbias_ref, mean_ref, rstd_ref = _reference_layernorm_bwd(x, weight, bias, dy)

    stream = torch.cuda.current_stream()

    # --- forward with store_stats: validates mean + rstd from the kernel ---
    out = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    mean = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    rstd = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    elem_dtype = _flydsl_elem_dtype(dtype)
    fwd_c = flyc.compile(
        fwd_fn,
        _as_pointer(x, elem_dtype),
        _as_pointer(weight, elem_dtype),
        _as_pointer(bias, elem_dtype),
        _as_pointer(out, elem_dtype),
        _as_pointer(mean, fx.Float32),
        _as_pointer(rstd, fx.Float32),
        M,
        stream,
    )
    fwd_c(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        mean.data_ptr(),
        rstd.data_ptr(),
        M,
        stream,
    )
    torch.cuda.synchronize()
    mean_err = (mean - mean_ref).abs().max().item()
    rstd_err = (rstd - rstd_ref).abs().max().item()

    # --- backward: dx + dgamma + dbias ---
    dx = torch.empty((M, N), device="cuda", dtype=torch_dtype)
    dgamma = torch.zeros((N,), device="cuda", dtype=DTYPE_FP32)
    dbias = torch.zeros((N,), device="cuda", dtype=DTYPE_FP32)
    x_ptr = _as_pointer(x, elem_dtype)
    weight_ptr = _as_pointer(weight, elem_dtype)
    dy_ptr = _as_pointer(dy, elem_dtype)
    mean_ptr = _as_pointer(mean, fx.Float32)
    rstd_ptr = _as_pointer(rstd, fx.Float32)
    dx_ptr = _as_pointer(dx, elem_dtype)
    dgamma_ptr = _as_pointer(dgamma, fx.Float32)
    dbias_ptr = _as_pointer(dbias, fx.Float32)
    bwd_c = flyc.compile(
        bwd_fn,
        x_ptr,
        weight_ptr,
        dy_ptr,
        mean_ptr,
        rstd_ptr,
        dx_ptr,
        dgamma_ptr,
        dbias_ptr,
        M,
        stream,
    )
    dgamma.zero_()
    dbias.zero_()
    bwd_c(
        x.data_ptr(),
        weight.data_ptr(),
        dy.data_ptr(),
        mean.data_ptr(),
        rstd.data_ptr(),
        dx.data_ptr(),
        dgamma.data_ptr(),
        dbias.data_ptr(),
        M,
        stream,
    )
    torch.cuda.synchronize()

    dx_err = (dx.to(DTYPE_FP32) - dx_ref).abs().max().item()

    # Tolerances (calibrated). dgamma/dbias summed over M -> larger magnitude -> relative.
    stat_atol = 1e-3
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dg_rtol = {"f32": 1e-4, "f16": 3e-2, "bf16": 1e-1}[dtype]
    dg_atol = {"f32": 1e-2, "f16": 1e-1, "bf16": 5e-1}[dtype]

    print(f"  mean max abs err    = {mean_err:.3e} (atol={stat_atol})")
    print(f"  rstd max abs err    = {rstd_err:.3e} (atol={stat_atol})")
    print(f"  dx max abs err      = {dx_err:.3e} (atol={dx_atol})")

    dg_ok = True
    try:
        torch.testing.assert_close(dgamma, dgamma_ref, rtol=dg_rtol, atol=dg_atol)
    except AssertionError as e:
        dg_ok = False
        dg_err = (dgamma - dgamma_ref).abs().max().item()
        print(f"  dgamma max abs err  = {dg_err:.3e} (rtol={dg_rtol}, atol={dg_atol})")
        print(f"  [dgamma mismatch] {e}")
    else:
        dg_err = (dgamma - dgamma_ref).abs().max().item()
        print(f"  dgamma max abs err  = {dg_err:.3e} (rtol={dg_rtol}, atol={dg_atol})")

    db_ok = True
    try:
        torch.testing.assert_close(dbias, dbias_ref, rtol=dg_rtol, atol=dg_atol)
    except AssertionError as e:
        db_ok = False
        db_err = (dbias - dbias_ref).abs().max().item()
        print(f"  dbias max abs err   = {db_err:.3e} (rtol={dg_rtol}, atol={dg_atol})")
        print(f"  [dbias mismatch] {e}")
    else:
        db_err = (dbias - dbias_ref).abs().max().item()
        print(f"  dbias max abs err   = {db_err:.3e} (rtol={dg_rtol}, atol={dg_atol})")

    ok = mean_err < stat_atol and rstd_err < stat_atol and dx_err < dx_atol and dg_ok and db_ok
    print(f"  -> {'PASSED' if ok else 'FAILED'}")
    return ok


def run_layernorm_autograd_test(M: int, N: int, dtype: str = "f32"):
    """End-to-end: public layernorm() grads on x + weight + bias, incl. 3D reshape."""
    print(f"\nTesting layernorm() autograd (M={M}, N={N}, dtype={dtype})")
    torch_dtype = _torch_dtype(dtype)
    torch.manual_seed(42)

    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    bias = torch.rand((N,), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).requires_grad_(True)
    dy = torch.randn((M, N), device="cuda", dtype=torch_dtype)

    out = layernorm(x, weight, bias)
    out.backward(dy)
    dx_out, dw_out, db_out = x.grad.detach(), weight.grad.detach(), bias.grad.detach()

    # fp32 autograd reference
    xf = x.detach().to(DTYPE_FP32).requires_grad_(True)
    wf = weight.detach().to(DTYPE_FP32).requires_grad_(True)
    bf = bias.detach().to(DTYPE_FP32).requires_grad_(True)
    mean = xf.mean(dim=1, keepdim=True)
    var = xf.var(dim=1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + EPS)
    yr = (xf - mean) * rstd * wf + bf
    dxr, dwr, dbr = torch.autograd.grad(yr, [xf, wf, bf], dy.to(DTYPE_FP32))

    out_err = (out.detach().to(DTYPE_FP32) - yr.detach()).abs().max().item()
    dx_err = (dx_out.to(DTYPE_FP32) - dxr).abs().max().item()
    dw_err = (dw_out.to(DTYPE_FP32) - dwr).abs().max().item()
    db_err = (db_out.to(DTYPE_FP32) - dbr).abs().max().item()

    out_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dx_atol = {"f32": 1e-3, "f16": 3e-2, "bf16": 2e-1}[dtype]
    dw_atol = {"f32": 1e-2, "f16": 2e-1, "bf16": 1.0}[dtype]

    print(f"  out max abs err = {out_err:.3e} (atol={out_atol})")
    print(f"  dx  max abs err = {dx_err:.3e} (atol={dx_atol})")
    print(f"  dw  max abs err = {dw_err:.3e} (atol={dw_atol})")
    print(f"  db  max abs err = {db_err:.3e} (atol={dw_atol})")

    # Batched (3D) input must reshape correctly through the public entry.
    x3 = torch.randn((4, M // 4 if M >= 4 else 1, N), device="cuda", dtype=torch_dtype, requires_grad=True)
    y3 = layernorm(x3, weight, bias)
    shape_ok = tuple(y3.shape) == tuple(x3.shape)
    y3.sum().backward()
    grad_ok = x3.grad is not None and tuple(x3.grad.shape) == tuple(x3.shape)
    print(f"  3D reshape: out_shape_ok={shape_ok} grad_shape_ok={grad_ok}")

    ok = out_err < out_atol and dx_err < dx_atol and dw_err < dw_atol and db_err < dw_atol and shape_ok and grad_ok
    print(f"  -> {'PASSED' if ok else 'FAILED'}")
    return ok


def test_layernorm_backward():
    print("=" * 80)
    print("Running LayerNorm Backward Tests")
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
        if not run_layernorm_bwd_test(M, N, dtype):
            failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def test_layernorm_autograd():
    print("=" * 80)
    print("Running layernorm() Autograd (end-to-end) Tests")
    print("=" * 80)

    configs = [
        (64, 256, "f32"),  # generic path
        (128, 4096, "bf16"),  # generic path
        (128, 4096, "f16"),  # generic path, f16
    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_layernorm_autograd_test(M, N, dtype):
            failures += 1

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TESTS FAILED")
    print("=" * 80)
    if failures != 0:
        raise SystemExit(1)


def test_layernorm_eps_honored():
    """eps must be baked into the kernel, not silently replaced by the module EPS."""
    print("=" * 80)
    print("Running LayerNorm eps-honored Test")
    print("=" * 80)
    torch.manual_seed(0)
    M, N = 32, 256
    x = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    w = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    b = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    for eps in (1e-5, 1e-6, 1e-2):
        y = layernorm(x, w, b, eps=eps)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        ref = (x - mean) * torch.rsqrt(var + eps) * w + b
        err = (y - ref).abs().max().item()
        print(f"  eps={eps:g}: max err vs torch ref = {err:.3e}")
        assert err < 1e-4, f"eps={eps} not honored (err={err})"

    # A non-default eps must actually change the output (guards silent-ignore regressions).
    diff = (layernorm(x, w, b, eps=1e-2) - layernorm(x, w, b, eps=1e-6)).abs().max().item()
    print(f"  eps 1e-2 vs 1e-6 output diff = {diff:.3e} (must be > 0)")
    assert diff > 0, "eps appears to be ignored"
    print("  -> PASSED")


@pytest.mark.parametrize("M,N,dtype", [(3, 257, "f32"), (2, 8192, "bf16")])
def test_layernorm_pointer_storage_offset(M, N, dtype):
    """Pointer row math must be relative to each contiguous view's data_ptr."""
    torch_dtype = _torch_dtype(dtype)

    def offset_rand(shape):
        numel = 1
        for dim in shape:
            numel *= dim
        tensor = torch.randn((numel + 1,), device="cuda", dtype=torch_dtype)[1:].view(shape)
        assert tensor.is_contiguous() and tensor.storage_offset() == 1
        return tensor

    x = offset_rand((M, N)).detach().requires_grad_(True)
    weight = offset_rand((N,)).detach().requires_grad_(True)
    bias = offset_rand((N,)).detach().requires_grad_(True)
    dout = offset_rand((M, N))

    out = layernorm(x, weight, bias)
    out.backward(dout)

    out_ref = _reference_layernorm(x.detach(), weight.detach(), bias.detach())
    dx_ref, dweight_ref, dbias_ref, _, _ = _reference_layernorm_bwd(x.detach(), weight.detach(), bias.detach(), dout)
    out_atol = {"f32": 1e-4, "bf16": 2e-2}[dtype]
    dx_atol = {"f32": 1e-3, "bf16": 2e-1}[dtype]
    grad_rtol = {"f32": 1e-4, "bf16": 1e-1}[dtype]
    grad_atol = {"f32": 1e-2, "bf16": 1.0}[dtype]
    torch.testing.assert_close(out.to(DTYPE_FP32), out_ref, rtol=grad_rtol, atol=out_atol)
    torch.testing.assert_close(x.grad.to(DTYPE_FP32), dx_ref, rtol=grad_rtol, atol=dx_atol)
    torch.testing.assert_close(weight.grad.to(DTYPE_FP32), dweight_ref, rtol=grad_rtol, atol=grad_atol)
    torch.testing.assert_close(bias.grad.to(DTYPE_FP32), dbias_ref, rtol=grad_rtol, atol=grad_atol)


def test_layernorm_strided_input_and_affine_params():
    """Public layernorm materializes layouts required by its raw-pointer kernels."""
    torch.manual_seed(0)
    M, N = 3, 257

    x_storage = torch.randn((2 * M, N), device="cuda", dtype=DTYPE_FP32)
    weight_storage = torch.randn((2 * N,), device="cuda", dtype=DTYPE_FP32)
    # Keep N elements allocated so a layout regression reads in-bounds data
    # instead of risking an out-of-bounds access in optimized Python mode.
    bias_storage = torch.randn((N,), device="cuda", dtype=DTYPE_FP32)

    x = x_storage[::2].detach().requires_grad_(True)
    weight = weight_storage[::2].detach().requires_grad_(True)
    bias = bias_storage[:1].expand(N).detach().requires_grad_(True)
    dout = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)

    assert x.stride() == (2 * N, 1)
    assert weight.stride() == (2,)
    assert bias.stride() == (0,)
    assert not x.is_contiguous()
    assert not weight.is_contiguous()
    assert not bias.is_contiguous()

    out = layernorm(x, weight, bias)
    out.backward(dout)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    out_ref = torch.nn.functional.layer_norm(x_ref, (N,), weight_ref, bias_ref, EPS)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-2)
    torch.testing.assert_close(bias.grad, bias_ref.grad, rtol=1e-4, atol=1e-2)


@pytest.mark.multi_gpu
def test_layernorm_multi_gpu():
    """Compiled-fn cache must not reuse a device-0 kernel on device 1 (would fault)."""
    print("=" * 80)
    print("Running LayerNorm multi-GPU Test")
    print("=" * 80)
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    torch.manual_seed(0)
    N = 256
    for dev in ("cuda:0", "cuda:1"):
        x = torch.randn((16, N), device=dev, dtype=DTYPE_FP32, requires_grad=True)
        w = torch.rand((N,), device=dev, dtype=DTYPE_FP32, requires_grad=True)
        b = torch.rand((N,), device=dev, dtype=DTYPE_FP32, requires_grad=True)
        dy = torch.randn((16, N), device=dev, dtype=DTYPE_FP32)
        y = layernorm(x, w, b)
        y.backward(dy)
        torch.cuda.synchronize(dev)
        mean = x.detach().mean(1, keepdim=True)
        var = x.detach().var(1, keepdim=True, unbiased=False)
        ref = (x.detach() - mean) * torch.rsqrt(var + EPS) * w.detach() + b.detach()
        err = (y.detach() - ref).abs().max().item()
        print(f"  {dev}: out err={err:.3e}, dx finite={torch.isfinite(x.grad).all().item()}")
        assert err < 1e-4 and torch.isfinite(x.grad).all()
    print("  -> PASSED")


@pytest.mark.multi_gpu
def test_layernorm_device_mismatch():
    """Raw pointer launches must reject cross-device affine parameters."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    x = torch.randn((4, 256), device="cuda:0", dtype=DTYPE_FP32)
    w0 = torch.rand((256,), device="cuda:0", dtype=DTYPE_FP32)
    b0 = torch.rand((256,), device="cuda:0", dtype=DTYPE_FP32)
    w1 = w0.to("cuda:1")
    b1 = b0.to("cuda:1")

    with pytest.raises(AssertionError, match="same device"):
        layernorm(x, w1, b0)
    with pytest.raises(AssertionError, match="same device"):
        layernorm(x, w0, b1)


def test_layernorm_dtype_mismatch():
    """Mixed x/weight/bias dtypes must be rejected (kernel uses one elem dtype)."""
    print("=" * 80)
    print("Running layernorm dtype-mismatch guard Test")
    print("=" * 80)
    N = 256
    x = torch.randn((4, N), device="cuda", dtype=DTYPE_BF16)
    w = torch.rand((N,), device="cuda", dtype=DTYPE_BF16)
    b = torch.rand((N,), device="cuda", dtype=DTYPE_BF16)
    for bad_w, bad_b in ((w.to(DTYPE_FP16), b), (w, b.to(DTYPE_FP32))):
        try:
            layernorm(x, bad_w, bad_b)
            raise SystemExit("dtype mismatch was NOT rejected")
        except AssertionError:
            pass
    print("  -> PASSED")


def test_layernorm_affine_shape_mismatch():
    """Pointer views require exactly N affine elements, not merely a trailing N."""
    N = 256
    x = torch.randn((4, N), device="cuda", dtype=DTYPE_FP32)
    weight = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    bias = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    with pytest.raises(AssertionError, match="1D"):
        layernorm(x, weight.view(1, N), bias)
    with pytest.raises(AssertionError, match="1D"):
        layernorm(x, weight, bias.view(1, N))
    with pytest.raises(AssertionError, match="1D"):
        layernorm(x, weight.view(1, N).expand(2, N).contiguous(), bias)
    with pytest.raises(AssertionError, match="1D"):
        layernorm(x, weight, bias.view(1, N).expand(2, N).contiguous())


if __name__ == "__main__":
    test_layernorm()
    test_layernorm_backward()
    test_layernorm_autograd()
    test_layernorm_eps_honored()
    test_layernorm_multi_gpu()
    test_layernorm_dtype_mismatch()
    test_fused_add_layernorm()
    test_layernorm_dynamicquant()
    test_layernorm_smoothquant()
    test_fused_add_layernorm_dynamicquant()
    test_fused_add_layernorm_smoothquant()
