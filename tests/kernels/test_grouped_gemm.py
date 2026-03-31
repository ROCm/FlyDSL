#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for Grouped FP8 GEMM kernel.

Tests the grouped FP8 GEMM with block scaling, matching DeepGEMM's
m_grouped_fp8_gemm_nt_contiguous API.
"""

import os
import sys
import logging

import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.grouped_gemm import compile_grouped_fp8_gemm
from flydsl.runtime.device import get_rocm_arch
from tests.test_common import run_perftest, verify_output

logging.basicConfig(level=logging.INFO)

ARCH = get_rocm_arch()
# Use appropriate FP8 dtype for the architecture
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def quantize_to_fp8(x: torch.Tensor, scale_block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 with per-row, per-block scaling.

    Args:
        x: Input tensor [M, K]
        scale_block_k: K-dimension block size for scaling

    Returns:
        (x_fp8, scale): FP8 tensor and scale factors [scale_k, M]
    """
    M, K = x.shape
    nblk_k = K // scale_block_k

    # Reshape to [M, nblk_k, scale_block_k]
    x_blocks = x.view(M, nblk_k, scale_block_k)

    # Compute per-block max (for scale)
    x_amax = x_blocks.abs().amax(dim=2).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = x_amax / fp8_max

    # Quantize
    x_scaled = x_blocks / scale.unsqueeze(2)
    x_fp8 = x_scaled.to(DTYPE_FP8).view(M, K)

    # Transpose scale to [scale_k, M] to match DeepGEMM layout
    scale = scale.T.contiguous()

    return x_fp8, scale


def quantize_b_to_fp8(
    b: torch.Tensor, scale_block_n: int = 128, scale_block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize B tensor to FP8 with per-block scaling.

    Args:
        b: Input tensor [num_groups, N, K]
        scale_block_n: N-dimension block size
        scale_block_k: K-dimension block size

    Returns:
        (b_fp8, scale_b): FP8 tensor and scale factors [num_groups, scale_n, scale_k]
    """
    num_groups, N, K = b.shape
    nblk_n = N // scale_block_n
    nblk_k = K // scale_block_k

    # Reshape to [num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k]
    b_blocks = b.view(num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k)

    # Compute per-block max
    b_amax = b_blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = b_amax / fp8_max

    # Quantize
    b_scaled = b_blocks / scale.view(num_groups, nblk_n, 1, nblk_k, 1)
    b_fp8 = b_scaled.to(DTYPE_FP8).view(num_groups, N, K)

    return b_fp8, scale


def torch_grouped_gemm_ref(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    grouped_layout: torch.Tensor,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
) -> torch.Tensor:
    """PyTorch reference implementation for grouped FP8 GEMM with block scaling.

    Args:
        a: [M, K] FP8 tensor
        scale_a: [scale_k, M] FP32 scale factors (transposed layout)
        b: [num_groups, N, K] FP8 tensor
        scale_b: [num_groups, scale_n, scale_k] FP32 scale factors
        grouped_layout: [M] INT32 mapping rows to groups (-1 for padding)
        scale_block_k: K-dimension scale block size
        scale_block_n: N-dimension scale block size

    Returns:
        d: [M, N] BF16 output tensor
    """
    M, K = a.shape
    num_groups, N, _ = b.shape
    nblk_k = K // scale_block_k
    nblk_n = N // scale_block_n

    # Dequantize A
    a_f32 = a.to(torch.float32)
    # scale_a is [scale_k, M], transpose to [M, scale_k]
    scale_a_t = scale_a.T  # [M, scale_k]
    # Expand to element-wise: [M, nblk_k, scale_block_k]
    a_scaled = a_f32.view(M, nblk_k, scale_block_k) * scale_a_t.view(M, nblk_k, 1)
    a_scaled = a_scaled.view(M, K)

    # Dequantize B per group
    # scale_b: [num_groups, scale_n, scale_k]
    # Expand to [num_groups, N, K]
    b_f32 = b.to(torch.float32)
    b_scaled = b_f32.view(num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k)
    b_scaled = b_scaled * scale_b.view(num_groups, nblk_n, 1, nblk_k, 1)
    b_scaled = b_scaled.view(num_groups, N, K)

    # Compute grouped GEMM (on CPU to avoid hipBLAS issues with small/irregular shapes)
    a_scaled_cpu = a_scaled.cpu()
    b_scaled_cpu = b_scaled.cpu()
    grouped_layout_cpu = grouped_layout.cpu()
    d = torch.zeros(M, N, dtype=torch.float32)
    for g in range(num_groups):
        mask = grouped_layout_cpu == g
        if mask.any():
            d[mask] = a_scaled_cpu[mask] @ b_scaled_cpu[g].T

    return d.to(torch.bfloat16).to(a.device)


def generate_grouped_gemm_inputs(
    num_groups: int,
    m_per_group: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    device: str = "cuda",
):
    """Generate test inputs for grouped GEMM.

    Args:
        num_groups: Number of groups
        m_per_group: Approximate M rows per group
        n: N dimension
        k: K dimension
        scale_block_k: K-dimension scale block size
        scale_block_n: N-dimension scale block size
        device: Device to create tensors on

    Returns:
        Tuple of (a_fp8, scale_a, b_fp8, scale_b, grouped_layout, d, ref_d)
    """
    # Generate variable group sizes (aligned to tile_m=128)
    tile_m = 128
    ms = []
    for _ in range(num_groups):
        m = int(m_per_group * (0.8 + 0.4 * torch.rand(1).item()))
        m = align(m, tile_m)
        ms.append(m)
    M = sum(ms)

    # Create grouped_layout
    grouped_layout = torch.empty(M, dtype=torch.int32, device=device)
    start = 0
    for g, m in enumerate(ms):
        grouped_layout[start : start + m] = g
        start += m

    # Generate random data
    a_f32 = torch.randn(M, k, device=device, dtype=torch.float32)
    b_f32 = torch.randn(num_groups, n, k, device=device, dtype=torch.float32)

    # Quantize to FP8
    a_fp8, scale_a = quantize_to_fp8(a_f32, scale_block_k)
    b_fp8, scale_b = quantize_b_to_fp8(b_f32, scale_block_n, scale_block_k)

    # Output buffer
    d = torch.zeros(M, n, dtype=torch.bfloat16, device=device)

    # Reference output
    ref_d = torch_grouped_gemm_ref(
        a_fp8, scale_a, b_fp8, scale_b, grouped_layout, scale_block_k, scale_block_n
    )

    return a_fp8, scale_a, b_fp8, scale_b, grouped_layout, d, ref_d, M


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """View FP8 tensor as int8 for kernel interface."""
    return t.view(torch.int8)


@pytest.mark.parametrize(
    "num_groups,m_per_group,n,k",
    [
        pytest.param(1, 128, 128, 128, id="single-group-small"),
        pytest.param(2, 128, 128, 128, id="two-groups-small"),
        pytest.param(4, 128, 256, 256, id="four-groups-medium"),
        pytest.param(8, 256, 512, 512, id="eight-groups-larger"),
    ],
)
def test_grouped_fp8_gemm_correctness(num_groups, m_per_group, n, k):
    """Test grouped FP8 GEMM correctness against PyTorch reference."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs
    a_fp8, scale_a, b_fp8, scale_b, grouped_layout, d, ref_d, M = generate_grouped_gemm_inputs(
        num_groups, m_per_group, n, k, scale_block_k, scale_block_n
    )

    # Compile kernel
    launch_fn = compile_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=128,
        tile_n=128,
        tile_k=128,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype="bf16",
    )

    # Launch wrapper
    stream = torch.cuda.current_stream()

    def launch_kernel(d, a, b, sa, sb, gl):
        launch_fn(d, a, b, sa, sb, gl, M, n, k, num_groups, stream)

    launch_kernel(
        d.contiguous().view(-1),
        _as_i8(a_fp8.contiguous().view(-1)),
        _as_i8(b_fp8.contiguous().view(-1)),
        scale_a.contiguous().view(-1),
        scale_b.contiguous().view(-1),
        grouped_layout.contiguous(),
    )
    torch.cuda.synchronize()

    # Verify correctness
    c_out_f32 = d.to(torch.float32)
    c_ref = ref_d.to(torch.float32)
    msg = f"num_groups={num_groups}, M={M}, N={n}, K={k}"
    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=1e-2, msg=msg)
    assert passed, f"Correctness check failed for {msg}"


@pytest.mark.parametrize(
    "num_groups,m_per_group,n,k",
    [
        pytest.param(8, 512, 1024, 1024, id="perf-8g-512m"),
    ],
)
def test_grouped_fp8_gemm_performance(num_groups, m_per_group, n, k):
    """Benchmark grouped FP8 GEMM performance."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs
    a_fp8, scale_a, b_fp8, scale_b, grouped_layout, d, ref_d, M = generate_grouped_gemm_inputs(
        num_groups, m_per_group, n, k, scale_block_k, scale_block_n
    )

    # Compile kernel
    launch_fn = compile_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=128,
        tile_n=128,
        tile_k=128,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype="bf16",
    )

    stream = torch.cuda.current_stream()

    def launch_kernel(d, a, b, sa, sb, gl):
        launch_fn(d, a, b, sa, sb, gl, M, n, k, num_groups, stream)

    _, us = run_perftest(
        launch_kernel,
        d.contiguous().view(-1),
        _as_i8(a_fp8.contiguous().view(-1)),
        _as_i8(b_fp8.contiguous().view(-1)),
        scale_a.contiguous().view(-1),
        scale_b.contiguous().view(-1),
        grouped_layout.contiguous(),
        num_iters=20,
        num_warmup=3,
    )

    flops = 2 * M * n * k
    tflops = flops / (us / 1e6) / 1e12
    bytes_a = M * k  # FP8
    bytes_b = num_groups * n * k  # FP8
    bytes_d = M * n * 2  # BF16
    bytes_scales = (k // scale_block_k) * M * 4 + num_groups * (n // scale_block_n) * (k // scale_block_k) * 4
    total_bytes = bytes_a + bytes_b + bytes_d + bytes_scales
    bandwidth_tbs = total_bytes / (us / 1e6) / 1e12

    print(f"\nPerformance: num_groups={num_groups}, M={M}, N={n}, K={k}")
    print(f"  Time: {us:.2f} us")
    print(f"  TFLOPS: {tflops:.2f}")
    print(f"  Bandwidth: {bandwidth_tbs:.2f} TB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grouped FP8 GEMM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--m_per_group", type=int, default=256)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["bf16", "f16"])
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=5)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    test_grouped_fp8_gemm_correctness(args.num_groups, args.m_per_group, args.N, args.K)
    test_grouped_fp8_gemm_performance(args.num_groups, args.m_per_group, args.N, args.K)
