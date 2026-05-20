#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm operator tests, including AIter/Triton-aligned variants."""

import os

import pytest

from kernels.layernorm_kernel import (
    BLOCK_THREADS,
    KERNEL_NAME as LAYERNORM_KERNEL_NAME,
    build_fused_add_layernorm_dynamicquant_module,
    build_fused_add_layernorm_module,
    build_fused_add_layernorm_smoothquant_module,
    build_layernorm_dynamicquant_module,
    build_layernorm_module,
    build_layernorm_smoothquant_module,
)
from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    maybe_enable_aiter,
    print_perf_table,
)

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


def _atol(dtype: str) -> float:
    if dtype == "f32":
        return 1e-4
    if dtype == "f16":
        return 1e-2
    if dtype == "bf16":
        return 2e-2
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
        return configs

    # Prefer N multiples of 2048 to exercise the fast path.
    return [
        # (64, 256, "f32"),     # Aligned
        # (128, 1024, "f32"),   # Aligned
        # (32, 128, "f16"),     # Aligned
        # (64, 2000, "f32"),    # Unaligned (tail handling)
        # (16, 512, "bf16"),    # BF16
        # (1024, 8192, "bf16"), # BF16
        (32768, 8192, "bf16"),
    ]


def _make_inputs(M: int, N: int, dtype: str):
    torch_dtype = _torch_dtype(dtype)
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    beta_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    xscale_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32) + 0.5

    return (
        input_t.to(torch_dtype).contiguous(),
        gamma_t.to(torch_dtype).contiguous(),
        beta_t.to(torch_dtype).contiguous(),
        residual_t.to(torch_dtype).contiguous(),
        xscale_t.to(torch_dtype).contiguous(),
    )


def _reference_layernorm(input_dev, gamma_dev, beta_dev, *, residual_dev=None, xscale_dev=None):
    x = input_dev.to(DTYPE_FP32)
    residual_out = None
    if residual_dev is not None:
        residual_out = x + residual_dev.to(DTYPE_FP32)
        x = residual_out
    gamma = gamma_dev.to(DTYPE_FP32)
    beta = beta_dev.to(DTYPE_FP32)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + EPS) * gamma + beta
    if xscale_dev is not None:
        expected = expected * xscale_dev.to(DTYPE_FP32)
    return expected, residual_out


def _reference_quant(input_dev, gamma_dev, beta_dev, *, residual_dev=None, xscale_dev=None):
    expected, residual_out = _reference_layernorm(
        input_dev,
        gamma_dev,
        beta_dev,
        residual_dev=residual_dev,
        xscale_dev=xscale_dev,
    )
    yscale = expected.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(expected / yscale.unsqueeze(1)), -127, 127).to(torch.int8)
    return expected, residual_out, q, yscale


def _bench_aiter(M: int, N: int, dtype: str, mode: str):
    torch_dtype = _torch_dtype(dtype)
    try:
        from aiter.ops.triton.normalization.norm import (
            layer_norm,
            layernorm2d_fwd_with_add,
            layernorm2d_fwd_with_add_dynamicquant,
            layernorm2d_fwd_with_add_smoothquant,
            layernorm2d_fwd_with_dynamicquant,
            layernorm2d_fwd_with_smoothquant,
        )
    except Exception as e:
        print(f"[Perf] AIter layernorm {mode} skipped: {type(e).__name__}: {e!r}")
        return None

    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    w = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    b = torch.rand((N,), device="cuda", dtype=torch_dtype).contiguous()
    residual = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    residual_out = torch.empty_like(x)
    xscale = (torch.rand((N,), device="cuda", dtype=torch_dtype) + 0.5).contiguous()
    q_out = torch.empty((M, N), device="cuda", dtype=torch.int8)
    yscale = torch.empty((M, 1), device="cuda", dtype=torch.float32)

    if mode == "base":
        run = lambda: layer_norm(x, w, b, EPS)
    elif mode == "fused_add":
        out = torch.empty_like(x)
        run = lambda: layernorm2d_fwd_with_add(out, x, residual, residual_out, w, b, EPS)
    elif mode == "dynamicquant":
        run = lambda: layernorm2d_fwd_with_dynamicquant(q_out, x, yscale, w, b, EPS)
    elif mode == "smoothquant":
        run = lambda: layernorm2d_fwd_with_smoothquant(q_out, x, xscale, yscale, w, b, EPS)
    elif mode == "fused_add_dynamicquant":
        run = lambda: layernorm2d_fwd_with_add_dynamicquant(q_out, x, residual, residual_out, yscale, w, b, EPS)
    elif mode == "fused_add_smoothquant":
        run = lambda: layernorm2d_fwd_with_add_smoothquant(q_out, x, residual, residual_out, xscale, yscale, w, b, EPS)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    aiter_us = bench_gpu_us_torch(run, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    print(f"[Perf] AIter layernorm {mode} gpu: {aiter_us:.1f} us")
    return aiter_us


def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")
    launch_fn = build_layernorm_module(M, N, dtype)
    input_dev, gamma_dev, beta_dev, _, _ = _make_inputs(M, N, dtype)
    output_dev = torch.empty((M, N), device="cuda", dtype=_torch_dtype(dtype))
    expected, _ = _reference_layernorm(input_dev, gamma_dev, beta_dev)
    atol = _atol(dtype)
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, beta_dev, output_dev, M, stream=stream)

    kernel_launch()
    torch.cuda.synchronize()
    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    error = (output_dev.to(DTYPE_FP32) - expected).abs().max().item()
    print(f"Kernel avg time: {avg_us / 1000.0:.4f} ms")
    print(f"Max absolute error: {error:.2e} (atol={atol})")
    return error < atol, flydsl_gpu_us


def run_fused_add_test(M: int, N: int, dtype: str):
    print(f"\nTesting LayerNorm fused_add (M={M}, N={N}, dtype={dtype})")
    launch_fn = build_fused_add_layernorm_module(M, N, dtype)
    input_dev, gamma_dev, beta_dev, residual_dev, _ = _make_inputs(M, N, dtype)
    output_dev = torch.empty((M, N), device="cuda", dtype=_torch_dtype(dtype))
    residual_out_dev = torch.empty_like(output_dev)
    expected, residual_expected = _reference_layernorm(input_dev, gamma_dev, beta_dev, residual_dev=residual_dev)
    atol = _atol(dtype)
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, residual_dev, gamma_dev, beta_dev, output_dev, residual_out_dev, M, stream=stream)

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    out_err = (output_dev.to(DTYPE_FP32) - expected).abs().max().item()
    residual_err = (residual_out_dev.to(DTYPE_FP32) - residual_expected).abs().max().item()
    print(f"Kernel avg time: {avg_us / 1000.0:.4f} ms")
    print(f"Max output error: {out_err:.2e} (atol={atol})")
    print(f"Max residual error: {residual_err:.2e} (atol={atol})")
    return out_err < atol and residual_err < atol, flydsl_gpu_us


def run_quant_test(M: int, N: int, dtype: str, *, is_smooth: bool, is_fused_add: bool):
    mode = ""
    if is_fused_add:
        mode += "fused_add_"
    mode += "smoothquant" if is_smooth else "dynamicquant"
    print(f"\nTesting LayerNorm {mode} (M={M}, N={N}, dtype={dtype})")

    if is_fused_add and is_smooth:
        launch_fn = build_fused_add_layernorm_smoothquant_module(M, N, dtype)
    elif is_fused_add:
        launch_fn = build_fused_add_layernorm_dynamicquant_module(M, N, dtype)
    elif is_smooth:
        launch_fn = build_layernorm_smoothquant_module(M, N, dtype)
    else:
        launch_fn = build_layernorm_dynamicquant_module(M, N, dtype)

    input_dev, gamma_dev, beta_dev, residual_dev, xscale_dev = _make_inputs(M, N, dtype)
    output_dev = torch.empty((M, N), device="cuda", dtype=torch.int8)
    yscale_dev = torch.empty((M,), device="cuda", dtype=DTYPE_FP32)
    residual_out_dev = torch.empty((M, N), device="cuda", dtype=_torch_dtype(dtype))

    expected, residual_expected, q_ref, yscale_ref = _reference_quant(
        input_dev,
        gamma_dev,
        beta_dev,
        residual_dev=residual_dev if is_fused_add else None,
        xscale_dev=xscale_dev if is_smooth else None,
    )

    stream = torch.cuda.current_stream()

    def kernel_launch():
        if is_fused_add and is_smooth:
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
        elif is_fused_add:
            launch_fn(
                input_dev, residual_dev, gamma_dev, beta_dev, output_dev, residual_out_dev, yscale_dev, M, stream=stream
            )
        elif is_smooth:
            launch_fn(input_dev, gamma_dev, beta_dev, xscale_dev, output_dev, yscale_dev, M, stream=stream)
        else:
            launch_fn(input_dev, gamma_dev, beta_dev, output_dev, yscale_dev, M, stream=stream)

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()), num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS
    )
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)

    q_diff = (output_dev.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item()
    scale_diff = (yscale_dev.cpu() - yscale_ref.cpu()).abs().max().item()
    recon = output_dev.to(DTYPE_FP32) * yscale_dev.unsqueeze(1)
    recon_err = (recon - expected).abs().max().item()
    ok = q_diff <= 1 and scale_diff < 1e-2 and recon_err < 0.3

    if is_fused_add:
        residual_err = (residual_out_dev.to(DTYPE_FP32) - residual_expected).abs().max().item()
        ok = ok and residual_err < _atol(dtype)
        print(f"Max residual error: {residual_err:.2e} (atol={_atol(dtype)})")

    print(f"Kernel avg time: {avg_us / 1000.0:.4f} ms")
    print(f"Max quant diff: {q_diff}")
    print(f"Max scale diff: {scale_diff:.2e}")
    print(f"Max recon error: {recon_err:.2e}")
    return ok, flydsl_gpu_us


def _run_configs(op: str, runner):
    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []
    failures = 0
    for M, N, dtype in _get_layernorm_configs():
        ok, flydsl_gpu_us = runner(M, N, dtype)
        if not ok:
            failures += 1
        if do_compare:
            aiter_us = None
            if maybe_enable_aiter():
                aiter_us = _bench_aiter(M, N, dtype, op)
            perf_rows.append(
                PerfRow(
                    op=f"layernorm_{op}",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                )
            )
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(f"{failures} {op} tests failed")


def test_layernorm_base():
    print("=" * 80)
    print("Running LayerNorm Tests")
    print("=" * 80)
    _run_configs("base", run_test)


def test_fused_add_layernorm():
    _run_configs("fused_add", run_fused_add_test)


def test_layernorm_dynamicquant():
    _run_configs("dynamicquant", lambda M, N, dtype: run_quant_test(M, N, dtype, is_smooth=False, is_fused_add=False))


def test_layernorm_smoothquant():
    _run_configs("smoothquant", lambda M, N, dtype: run_quant_test(M, N, dtype, is_smooth=True, is_fused_add=False))


def test_fused_add_layernorm_dynamicquant():
    _run_configs(
        "fused_add_dynamicquant", lambda M, N, dtype: run_quant_test(M, N, dtype, is_smooth=False, is_fused_add=True)
    )


def test_fused_add_layernorm_smoothquant():
    _run_configs(
        "fused_add_smoothquant", lambda M, N, dtype: run_quant_test(M, N, dtype, is_smooth=True, is_fused_add=True)
    )


if __name__ == "__main__":
    test_layernorm_base()
    test_fused_add_layernorm()
    test_layernorm_dynamicquant()
    test_layernorm_smoothquant()
    test_fused_add_layernorm_dynamicquant()
    test_fused_add_layernorm_smoothquant()
