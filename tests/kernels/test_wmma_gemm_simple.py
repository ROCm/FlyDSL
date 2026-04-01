#!/usr/bin/env python3
"""WMMA GEMM tests for gfx1250 — @flyc.kernel API.

Kernel implementation lives in `kernels/wmma_gemm_simple.py`.
This file is the correctness + perf harness.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from flydsl.runtime.device import get_rocm_arch
from kernels.wmma_gemm_simple import compile_wmma_gemm
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, block_threads",
    [
        (32, 32, 32, 32, 32, 32, 32),
        (64, 64, 32, 64, 64, 32, 128),
        (128, 128, 32, 64, 128, 32, 256),
        (128, 128, 64, 64, 128, 32, 256),
        (256, 256, 32, 64, 64, 32, 128),
        (200, 180, 64, 64, 64, 32, 128),
        (128, 128, 128, 64, 128, 64, 256),
    ],
)
def test_wmma_gemm(in_dtype, M, N, K, tile_m, tile_n, tile_k, block_threads):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")
    print(f"Running WMMA GEMM test with: M={M}, N={N}, K={K}, "
          f"tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}, "
          f"block_threads={block_threads}, dtype={in_dtype}, arch={arch}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Pad M/N to tile boundaries
    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    # torch gpu randn has some issues on gfx1250 AM simulator
    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device=device)
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device=device)
    a_pad[:M, :] = a
    b_pad[:, :N] = b
    c_pad = torch.zeros((mpad, npad), dtype=torch.float32, device=device)

    launch_fn = compile_wmma_gemm(
        M=mpad,
        N=npad,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        block_threads=block_threads,
    )
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_pad.contiguous().view(-1),
        mpad,
        npad,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.matmul(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    assert verify_output(c_pad[:M, :N].cpu(), ref, rtol=3e-2, atol=3e-2)
    print("✓ PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256, help='problem M size')
    parser.add_argument("-N", type=int, default=256, help='problem N size')
    parser.add_argument("-K", type=int, default=1024, help='problem K size')
    parser.add_argument("--tile_m", type=int, default=256)
    parser.add_argument("--tile_n", type=int, default=256)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                        help="Input data type")
    args = parser.parse_args()

    WARP_SIZE = 32
    BLOCK_THREADS = min(args.tile_n, 8 * WARP_SIZE)

    test_wmma_gemm(
        args.dtype,
        args.M,
        args.N,
        args.K,
        args.tile_m,
        args.tile_n,
        args.tile_k,
        BLOCK_THREADS,
    )
