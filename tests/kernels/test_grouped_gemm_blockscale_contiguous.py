#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for Contiguous Grouped FP8 GEMM kernel (blockscale).

Tests the contiguous grouped FP8 GEMM with block scaling, matching DeepGEMM's
m_grouped_fp8_gemm_nt_contiguous API.
"""

import os
import sys
import logging

import torch
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.grouped_gemm_blockscale_contiguous import compile_grouped_gemm_blockscale_contiguous
from tests.kernels.blockscale_gemm_test_utils import (
    ARCH,
    DTYPE_FP8,
    USE_UE8M0,
    _as_i8,
    align,
    ceil_div,
    fp32_e8m0_to_byte,
    fp32_to_e8m0,
    quantize_b_to_fp8,
)
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


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

    # Round to E8M0 when hardware scaling is used (gfx950); keep as FP32 for the
    # quantization divide, then pre-pack to uint8 below for kernel consumption.
    if USE_UE8M0:
        scale = fp32_to_e8m0(scale)

    # Quantize
    x_scaled = x_blocks / scale.unsqueeze(2)
    x_fp8 = x_scaled.to(DTYPE_FP8).view(M, K)

    # Pre-pack scale as uint8 (one E8M0 byte per scale) so the kernel can load
    # 1 byte/scale and skip the in-kernel bitwise extract.
    if USE_UE8M0:
        scale = fp32_e8m0_to_byte(scale)

    # Transpose scale to [scale_k, M] to match the kernel's expected layout
    scale = scale.T.contiguous()

    return x_fp8, scale


def generate_grouped_gemm_inputs(
    num_groups: int,
    m_per_group: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
    device: str = "cuda",
):
    """Generate test inputs for grouped GEMM.

    Generates variable actual group sizes (unaligned), pads each group to
    128-row alignment, and marks padding rows with -1 in grouped_layout.

    Args:
        num_groups: Number of groups
        m_per_group: Approximate actual M rows per group (before alignment)
        n: N dimension
        k: K dimension
        scale_block_k: K-dimension scale block size
        scale_block_n: N-dimension scale block size
        out_dtype: Output data type ("bf16" or "f16")
        device: Device to create tensors on

    Returns:
        Tuple of (a_fp8, scale_a, b_shuffled, scale_b, grouped_layout, d, ref_d, M)
    """
    alignment = 128  # M-row alignment for contiguous layout
    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    # Generate variable actual group sizes, then align
    actual_ms = []
    aligned_ms = []
    for _ in range(num_groups):
        m_actual = int(m_per_group * (0.8 + 0.4 * torch.rand(1).item()))
        m_actual = max(1, m_actual)  # at least 1 row
        m_aligned = align(m_actual, alignment)
        actual_ms.append(m_actual)
        aligned_ms.append(m_aligned)
    M = sum(aligned_ms)

    # Create grouped_layout with -1 padding
    grouped_layout = torch.empty(M, dtype=torch.int32, device=device)
    start = 0
    for g, (m_actual, m_aligned) in enumerate(zip(actual_ms, aligned_ms)):
        grouped_layout[start : start + m_actual] = g
        grouped_layout[start + m_actual : start + m_aligned] = -1
        start += m_aligned

    # Generate random data
    a_f32 = torch.randn(M, k, device=device, dtype=torch.float32)
    b_f32 = torch.randn(num_groups, n, k, device=device, dtype=torch.float32)

    # Zero out padding rows in A
    start = 0
    for m_actual, m_aligned in zip(actual_ms, aligned_ms):
        a_f32[start + m_actual : start + m_aligned] = 0
        start += m_aligned

    # Reference output from original FP32 data BEFORE quantization
    # (ref absorbs all quantization + scale errors).
    # Per-group matmul.
    ref_d = torch.zeros(M, n, dtype=torch.float32, device=device)
    for g in range(num_groups):
        mask = grouped_layout == g
        if mask.any():
            ref_d[mask] = a_f32[mask] @ b_f32[g].T
    ref_d = ref_d.to(torch_out_dtype)

    # Quantize to FP8
    a_fp8, scale_a = quantize_to_fp8(a_f32, scale_block_k)
    b_fp8, scale_b = quantize_b_to_fp8(b_f32, scale_block_n, scale_block_k)

    # Output buffer
    d = torch.zeros(M, n, dtype=torch_out_dtype, device=device)

    # Preshuffle B for kernel (applied per-group, batch dim folded automatically)
    b_shuffled = shuffle_weight(b_fp8, layout=(16, 16))

    return a_fp8, scale_a, b_shuffled, scale_b, grouped_layout, d, ref_d, M


@pytest.mark.parametrize(
    "num_groups,m_per_group,n,k",
    [
        # Basic shapes
        pytest.param(1, 128, 128, 128, id="1g-128m-128n-128k"),
        pytest.param(2, 128, 128, 128, id="2g-128m-128n-128k"),
        pytest.param(4, 128, 256, 256, id="4g-128m-256n-256k"),
        # Unaligned M (produces -1 padding rows in grouped_layout)
        pytest.param(2, 100, 128, 128, id="2g-100m-unaligned"),
        pytest.param(4, 200, 256, 256, id="4g-200m-unaligned"),
        # Larger shapes
        pytest.param(8, 256, 512, 512, id="8g-256m-512n-512k", marks=pytest.mark.large_shape),
        pytest.param(8, 256, 2048, 7168, id="8g-2048x7168-large", marks=pytest.mark.large_shape),
        pytest.param(8, 256, 7168, 2304, id="8g-7168x2304-large", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("out_dtype", [
    pytest.param("bf16", id="bf16"),
    pytest.param("f16", id="f16"),
])
def test_grouped_fp8_gemm(num_groups, m_per_group, n, k, out_dtype,
                          *, tile_m=128, tile_n=128, tile_k=128,
                          bench_iters=20, bench_warmup=3):
    """Verify grouped FP8 GEMM correctness against a PyTorch reference and
    report throughput. Single function combining correctness + bench, matching
    the convention of test_blockscale_preshuffle_gemm.py."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs
    a_fp8, scale_a, b_fp8, scale_b, grouped_layout, d, ref_d, M = generate_grouped_gemm_inputs(
        num_groups, m_per_group, n, k, scale_block_k, scale_block_n, out_dtype=out_dtype,
    )

    # Compile kernel
    launch_fn = compile_grouped_gemm_blockscale_contiguous(
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

    def launch_kernel(d, a, b, sa, sb, gl):
        launch_fn(d, a, b, sa, sb, gl, M, n, k, num_groups, stream)

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        d.contiguous().view(-1),
        _as_i8(a_fp8.contiguous().view(-1)),
        _as_i8(b_fp8.contiguous().view(-1)),
        scale_a.contiguous().view(-1),
        scale_b.contiguous().view(-1),
        grouped_layout.contiguous(),
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()

    # Zero out padding rows (group_id == -1) before comparison
    padding_mask = grouped_layout.cpu() == -1
    c_out_f32 = d.to(torch.float32)
    c_ref = ref_d.to(torch.float32)
    c_out_f32[padding_mask] = 0.0
    c_ref[padding_mask] = 0.0

    msg = f"num_groups={num_groups}, M={M}, N={n}, K={k}, out={out_dtype}"
    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=1e-2, msg=msg,
                           logits_diff_threshold=1e-3)
    assert passed, f"Correctness check failed for {msg}"

    flops = 2 * M * n * k
    tflops = flops / (us / 1e6) / 1e12
    bytes_a = M * k  # FP8
    bytes_b = num_groups * n * k  # FP8
    bytes_d = M * n * 2  # BF16
    bytes_scales = (k // scale_block_k) * M * 4 + num_groups * (n // scale_block_n) * (k // scale_block_k) * 4
    total_bytes = bytes_a + bytes_b + bytes_d + bytes_scales
    bandwidth_tbs = total_bytes / (us / 1e6) / 1e12

    print(f"\n  [{num_groups} groups, M={M}, N={n}, K={k}]")
    print(f"  Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bandwidth_tbs:.3f} TB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grouped FP8 GEMM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--m_per_group", type=int, default=0,
                        help="Approx M rows per group (0 = sweep [128, 256, 512, 1024])")
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

    m_list = [args.m_per_group] if args.m_per_group > 0 else [128, 256, 512, 1024]

    for m_per_group in m_list:
        test_grouped_fp8_gemm(args.num_groups, m_per_group, args.N, args.K,
                              out_dtype=args.out_dtype,
                              tile_m=args.tile_m, tile_n=args.tile_n,
                              tile_k=args.tile_k,
                              bench_iters=args.num_iters, bench_warmup=args.num_warmup)
