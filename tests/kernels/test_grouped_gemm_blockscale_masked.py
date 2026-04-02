#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for Masked Grouped FP8 GEMM kernel.

Tests the masked grouped FP8 GEMM with block scaling, matching DeepGEMM's
m_grouped_fp8_gemm_nt_masked API.
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

# Assuming the previous kernel code was saved here
from kernels.masked_grouped_gemm import compile_masked_grouped_fp8_gemm
from flydsl.runtime.device import get_rocm_arch
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight

logging.basicConfig(level=logging.INFO)

ARCH = get_rocm_arch()
# Use appropriate FP8 dtype for the architecture
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def quantize_a_masked_to_fp8(x: torch.Tensor, scale_block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize padded 3D A tensor to FP8 with per-row, per-block scaling.

    Args:
        x: Input tensor [G, max_m, K]
        scale_block_k: K-dimension block size for scaling

    Returns:
        (x_fp8, scale): FP8 tensor and scale factors [G, scale_k, max_m]
    """
    G, max_m, K = x.shape
    nblk_k = K // scale_block_k

    # Reshape to [G, max_m, nblk_k, scale_block_k]
    x_blocks = x.view(G, max_m, nblk_k, scale_block_k)

    # Compute per-block max (for scale)
    x_amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = x_amax / fp8_max

    # Quantize
    x_scaled = x_blocks / scale.unsqueeze(-1)
    x_fp8 = x_scaled.to(DTYPE_FP8).view(G, max_m, K)

    # Transpose scale to [G, scale_k, max_m] to match kernel layout
    scale = scale.transpose(1, 2).contiguous()

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


def torch_masked_grouped_gemm_ref(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    masked_m: torch.Tensor,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
) -> torch.Tensor:
    """PyTorch reference implementation for masked grouped FP8 GEMM.

    Args:
        a: [G, max_m, K] FP8 tensor
        scale_a: [G, scale_k, max_m] FP32 scale factors (transposed layout)
        b: [G, N, K] FP8 tensor
        scale_b: [G, scale_n, scale_k] FP32 scale factors
        masked_m: [G] INT32 true sequence length per group
        scale_block_k: K-dimension scale block size
        scale_block_n: N-dimension scale block size

    Returns:
        d: [G, max_m, N] BF16 output tensor
    """
    G, max_m, K = a.shape
    _, N, _ = b.shape
    nblk_k = K // scale_block_k
    nblk_n = N // scale_block_n

    # Dequantize A
    a_f32 = a.to(torch.float32)
    # scale_a is [G, scale_k, max_m], transpose to [G, max_m, scale_k]
    scale_a_t = scale_a.transpose(1, 2)  
    a_scaled = a_f32.view(G, max_m, nblk_k, scale_block_k) * scale_a_t.unsqueeze(-1)
    a_scaled = a_scaled.view(G, max_m, K)

    # Dequantize B
    b_f32 = b.to(torch.float32)
    b_scaled = b_f32.view(G, nblk_n, scale_block_n, nblk_k, scale_block_k)
    b_scaled = b_scaled * scale_b.view(G, nblk_n, 1, nblk_k, 1)
    b_scaled = b_scaled.view(G, N, K)

    # Compute masked grouped GEMM on CPU
    a_scaled_cpu = a_scaled.cpu()
    b_scaled_cpu = b_scaled.cpu()
    m_cpu = masked_m.cpu()
    
    d = torch.zeros(G, max_m, N, dtype=torch.float32, device="cpu")
    for g in range(G):
        m_actual = m_cpu[g].item()
        if m_actual > 0:
            d[g, :m_actual, :] = a_scaled_cpu[g, :m_actual, :] @ b_scaled_cpu[g].T

    return d.to(torch.bfloat16).to(a.device)


def generate_masked_grouped_gemm_inputs(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    device: str = "cuda",
):
    """Generate test inputs for masked grouped GEMM.

    Args:
        num_groups: Number of groups
        max_m: Capacity padding dimension for M
        expected_m_per_group: Average actual M rows per group
        n: N dimension
        k: K dimension

    Returns:
        Tuple of (a_fp8, scale_a, b_shuffled, scale_b, masked_m, d, ref_d)
    """
    # Generate valid length array
    masked_m = torch.empty(num_groups, dtype=torch.int32, device=device)
    for g in range(num_groups):
        m_val = int(expected_m_per_group * (0.8 + 0.4 * torch.rand(1).item()))
        m_val = min(m_val, max_m)  # cap at max_m
        masked_m[g] = m_val

    # Generate random padded data
    a_f32 = torch.randn(num_groups, max_m, k, device=device, dtype=torch.float32)
    b_f32 = torch.randn(num_groups, n, k, device=device, dtype=torch.float32)

    # Quantize to FP8
    a_fp8, scale_a = quantize_a_masked_to_fp8(a_f32, scale_block_k)
    b_fp8, scale_b = quantize_b_to_fp8(b_f32, scale_block_n, scale_block_k)

    # Output buffer
    d = torch.zeros(num_groups, max_m, n, dtype=torch.bfloat16, device=device)

    # Reference output
    ref_d = torch_masked_grouped_gemm_ref(
        a_fp8, scale_a, b_fp8, scale_b, masked_m, scale_block_k, scale_block_n
    )

    # Preshuffle B for kernel (applied per-group, batch dim folded automatically)
    b_shuffled = shuffle_weight(b_fp8, layout=(16, 16))

    return a_fp8, scale_a, b_shuffled, scale_b, masked_m, d, ref_d


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """View FP8 tensor as int8 for kernel interface."""
    return t.view(torch.int8)


@pytest.mark.parametrize(
    "num_groups,max_m,expected_m,n,k",
    [
        pytest.param(1, 256, 100, 128, 128, id="single-group-small"),
        pytest.param(4, 256, 150, 128, 128, id="four-groups-small"),
        pytest.param(8, 512, 300, 256, 256, id="eight-groups-medium"),
        pytest.param(8, 1024, 800, 512, 512, id="eight-groups-larger", marks=pytest.mark.large_shape),
    ],
)
def test_masked_grouped_fp8_gemm_correctness(num_groups, max_m, expected_m, n, k,
                                             tile_m=128, tile_n=128, tile_k=128,
                                             out_dtype="bf16"):
    """Test masked grouped FP8 GEMM correctness against PyTorch reference."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs
    a_fp8, scale_a, b_fp8, scale_b, masked_m, d, ref_d = generate_masked_grouped_gemm_inputs(
        num_groups, max_m, expected_m, n, k, scale_block_k, scale_block_n
    )

    # Compile kernel
    launch_fn = compile_masked_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )

    # Launch wrapper
    stream = torch.cuda.current_stream()

    def launch_kernel(d, a, b, sa, sb, mask):
        launch_fn(d, a, b, sa, sb, mask, max_m, n, k, num_groups, stream)

    launch_kernel(
        d.contiguous().view(-1),
        _as_i8(a_fp8.contiguous().view(-1)),
        _as_i8(b_fp8.contiguous().view(-1)),
        scale_a.contiguous().view(-1),
        scale_b.contiguous().view(-1),
        masked_m.contiguous(),
    )
    torch.cuda.synchronize()

    # Verify correctness
    c_out_f32 = d.to(torch.float32)
    c_ref = ref_d.to(torch.float32)

    # Note: the kernel computes in block sizes of `tile_m` and does NOT mask out D 
    # at the granularity of individual elements in the store epilogue.
    # Therefore, padded rows (m_actual to max_m) in the computed tiles contain garbage.
    # We must explicitly zero out the unused rows in the output before comparing.
    for g in range(num_groups):
        m_val = masked_m[g].item()
        c_out_f32[g, m_val:, :] = 0.0
        c_ref[g, m_val:, :] = 0.0

    msg = f"num_groups={num_groups}, max_m={max_m}, N={n}, K={k}"
    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=1e-2, msg=msg)
    assert passed, f"Correctness check failed for {msg}"


@pytest.mark.parametrize(
    "num_groups,max_m,expected_m,n,k",
    [
        pytest.param(8, 1024, 800, 1024, 1024, id="perf-8g-800m", marks=pytest.mark.large_shape),
    ],
)
def test_masked_grouped_fp8_gemm_performance(num_groups, max_m, expected_m, n, k,
                                             tile_m=128, tile_n=128, tile_k=128,
                                             out_dtype="bf16",
                                             num_iters=20, num_warmup=3):
    """Benchmark masked grouped FP8 GEMM performance."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs
    a_fp8, scale_a, b_fp8, scale_b, masked_m, d, ref_d = generate_masked_grouped_gemm_inputs(
        num_groups, max_m, expected_m, n, k, scale_block_k, scale_block_n
    )

    # Compile kernel
    launch_fn = compile_masked_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )

    stream = torch.cuda.current_stream()

    def launch_kernel(d, a, b, sa, sb, mask):
        launch_fn(d, a, b, sa, sb, mask, max_m, n, k, num_groups, stream)

    _, us = run_perftest(
        launch_kernel,
        d.contiguous().view(-1),
        _as_i8(a_fp8.contiguous().view(-1)),
        _as_i8(b_fp8.contiguous().view(-1)),
        scale_a.contiguous().view(-1),
        scale_b.contiguous().view(-1),
        masked_m.contiguous(),
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    # Compute effective FLOPs/BW based on ACTUAL valid tokens (as padding is mostly skipped)
    valid_m_sum = masked_m.sum().item()
    flops = 2 * valid_m_sum * n * k
    tflops = flops / (us / 1e6) / 1e12
    
    bytes_a = valid_m_sum * k  # FP8
    bytes_b = num_groups * n * k  # FP8
    bytes_d = valid_m_sum * n * 2  # BF16
    bytes_scales = (k // scale_block_k) * valid_m_sum * 4 + num_groups * (n // scale_block_n) * (k // scale_block_k) * 4
    total_bytes = bytes_a + bytes_b + bytes_d + bytes_scales
    bandwidth_tbs = total_bytes / (us / 1e6) / 1e12

    print(f"\n  [{num_groups} groups, max_m={max_m}, expected_m={expected_m}, N={n}, K={k}]")
    print(f"  Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bandwidth_tbs:.3f} TB/s (Effective)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Masked Grouped FP8 GEMM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--max_m", type=int, default=512)
    parser.add_argument("--expected_m", type=int, default=0,
                        help="Approx valid M rows per group (0 = sweep [128, 256, 384])")
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

    m_list = [args.expected_m] if args.expected_m > 0 else [128, 256, 384]

    for expected_m in m_list:
        test_masked_grouped_fp8_gemm_correctness(args.num_groups, args.max_m, expected_m, args.N, args.K,
                                                 tile_m=args.tile_m, tile_n=args.tile_n,
                                                 tile_k=args.tile_k, out_dtype=args.out_dtype)
        test_masked_grouped_fp8_gemm_performance(args.num_groups, args.max_m, expected_m, args.N, args.K,
                                                 tile_m=args.tile_m, tile_n=args.tile_n,
                                                 tile_k=args.tile_k, out_dtype=args.out_dtype,
                                                 num_iters=args.num_iters, num_warmup=args.num_warmup)