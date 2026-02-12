#!/usr/bin/env python3
"""Blockscale Preshuffle GEMM Test (FP8 A8W8 with per-block scales).

Reference: aiter/op_tests/test_gemm_a8w8_blockscale.py

This test verifies the blockscale preshuffle GEMM kernel against a torch
reference that applies per-block scales to FP8 A and B matrices.

Scale tensors:
- scale_a: [M, scale_k], scale_k = ceil(K / 128)
- scale_b: [scale_n, scale_k], scale_n = ceil(N / 128)

The kernel expects scale_a transposed to [scale_k, M] for vectorized loading.
"""

import os
import sys
import logging

import torch
import torch.nn.functional as F
import pytest

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

# Ensure repo-local flydsl is on the path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from kernels.blockscale_preshuffle_gemm import compile_blockscale_preshuffle_gemm, select_tile_config
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight
from flydsl.runtime.device import get_rocm_arch


logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

BLOCK_SHAPE = (128, 128)  # (block_n, block_k)

# Aiter imports (optional)
try:
    import aiter
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight

    HAS_AITER = True
except ImportError:
    HAS_AITER = False


def run_torch_blockscale(x, weight, x_scale, w_scale, block_shape=BLOCK_SHAPE,
                         dtype=torch.bfloat16):
    """Torch reference for blockscale GEMM (from aiter test_gemm_a8w8_blockscale.py)."""
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    # Dequantize A: apply per-row per-K-block scale
    x_f32 = x.to(x_scale.dtype).view(m, k // block_shape_k, block_shape_k) * x_scale.unsqueeze(-1)
    x_f32 = x_f32.view(m, k)

    # Dequantize B: apply per-N-block per-K-block scale
    # Expand block scales to full weight shape: [sn, sk, bn, bk] -> [sn*bn, sk*bk]
    w_scale_expanded = (
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    w_scale_expanded = w_scale_expanded[:n, :k]
    weight_f32 = weight.to(w_scale_expanded.dtype) * w_scale_expanded

    out = F.linear(x_f32.to(torch.float32), weight_f32.to(torch.float32))
    return out.to(dtype)


@pytest.mark.parametrize(
    "M, N, K",
    [
        (128, 1536, 7168),
        (256, 3072, 1536),
        (512, 4608, 7168),
        (1024, 7168, 2048),
        (2048, 7168, 2304),
        (4096, 1536, 7168),
    ],
)
@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
def test_blockscale_preshuffle_gemm(
    M, N, K, out_dtype,
    *,
    bench_iters=20,
    bench_warmup=3,
):
    """Test blockscale preshuffle GEMM kernel against torch reference."""
    block_shape_n, block_shape_k = BLOCK_SHAPE
    scale_k = (K + block_shape_k - 1) // block_shape_k
    scale_n = (N + block_shape_n - 1) // block_shape_n

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    # Auto-select tile config
    tile_m, tile_n, tile_k = select_tile_config(M, N, K, scale_block_k=block_shape_k)

    print("=" * 80)
    print(
        f"Blockscale GEMM Test: M={M}, N={N}, K={K}, "
        f"tile=({tile_m}x{tile_n}x{tile_k}), out={out_dtype}"
    )
    print("=" * 80)

    # Compile kernel
    exe = compile_blockscale_preshuffle_gemm(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=block_shape_k,
        out_dtype=out_dtype,
    )
    print(f"  Compiled OK")

    device = torch.device("cuda")

    # Generate FP8 inputs
    x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)
    weight = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)

    # Generate random block scales
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)

    # Torch reference
    c_ref = run_torch_blockscale(x, weight, x_scale, w_scale, dtype=torch.float32)

    # Prepare kernel inputs:
    # 1. Preshuffle B
    b_shuffled = shuffle_weight(weight, layout=(16, 16))

    # 2. Transpose scale_a to [scale_k, M] for vectorized M-axis loading
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(-1)

    # 3. Flatten scale_b
    w_scale_flat = w_scale.contiguous().view(-1)

    # Output buffer
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)

    def launch_kernel(c, a, b, sa, sb):
        stream_ptr = torch.cuda.current_stream().cuda_stream
        exe(c, a, b, sa, sb, M, N, K, stream_ptr)

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        c_out, x, b_shuffled, x_scale_t, w_scale_flat,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
    c_out_f32 = c_out.to(torch.float32)

    # Verify
    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=0.01)

    # Performance stats
    flops = 2 * M * N * K
    elem_bytes = 1  # fp8
    bytes_moved = (M * K * elem_bytes) + (N * K * elem_bytes) + (M * N * 2) + (M * scale_k + scale_n * scale_k) * 4
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"  Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s"
    )

    # Optional: compare with aiter
    if HAS_AITER:
        try:
            def launch_aiter(a, b, sa, sb):
                return aiter.gemm_a8w8_blockscale_bpreshuffle(
                    a, b, sa, sb, torch_out_dtype
                )

            x_scale_t_2d = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
            b_aiter = aiter_shuffle_weight(weight, layout=(16, 16))
            c_aiter, us_aiter = run_perftest(
                launch_aiter, x, b_aiter, x_scale_t_2d, w_scale,
            )
            c_aiter_f32 = c_aiter.to(torch.float32)
            verify_output(c_aiter_f32, c_ref, rtol=1e-2, atol=0.01)
            tflops_aiter = flops / (us_aiter / 1e6) / 1e12
            print(
                f"  Aiter: {us_aiter:.1f} us, {tflops_aiter:.2f} TFLOPS, "
                f"speedup: {us_aiter/us:.2f}x"
            )
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            print(f"  Aiter comparison skipped: {msg}")

    assert passed, "Kernel output verification failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blockscale Preshuffle GEMM benchmark")
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=1536)
    parser.add_argument("-K", type=int, default=7168)
    parser.add_argument("--tile_m", type=int, default=None)
    parser.add_argument("--tile_n", type=int, default=None)
    parser.add_argument("--tile_k", type=int, default=None)
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=3)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    # Auto-select tiles if not specified
    if args.tile_m is None or args.tile_n is None or args.tile_k is None:
        auto_tm, auto_tn, auto_tk = select_tile_config(args.M, args.N, args.K)
        tile_m = args.tile_m or auto_tm
        tile_n = args.tile_n or auto_tn
        tile_k = args.tile_k or auto_tk
    else:
        tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k
    print(f"Using tile config: {tile_m}x{tile_n}x{tile_k}")
    test_blockscale_preshuffle_gemm(
        M=args.M, N=args.N, K=args.K,
        out_dtype=args.out_dtype,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
    )
