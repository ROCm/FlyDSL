#!/usr/bin/env python3
"""WMMA GEMM using TDM tests for gfx1250.

Kernel implementation lives in `kernels/wmma_gemm_tdm_flyc.py`.
This file is the correctness harness.
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from flydsl.runtime.device import get_rocm_arch
from kernels.wmma_gemm_tdm_flyc import compile_wmma_gemm_tdm, preshuffle_b_weight
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (128, 128, 64, 64, 128, 32),
        (128, 128, 256, 64, 128, 128),
        (256, 256, 256, 64, 256, 128),
        (256, 256, 192, 64, 256, 64),
        (256, 512, 256, 64, 256, 128),
        (512, 512, 512, 64, 256, 128),
        (201, 179, 128, 64, 128, 64),
        (300, 399, 256, 64, 256, 128),
        (256, 256, 256, 256, 256, 128),
        (1024, 1024, 1024, 256, 256, 128),
        (512, 512, 512, 256, 256, 128),
    ],
)
@pytest.mark.parametrize("use_cshuffle", [False, True])
@pytest.mark.parametrize("use_preshuffle", [False, True])
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_wmma_gemm_tdm(in_dtype, M, N, K, tile_m, tile_n, tile_k,
                        use_cshuffle, use_preshuffle, num_buffers,
                        m_warp=2, n_warp=4, l2_prefetch_distance=0):
    arch = str(get_rocm_arch(timeout_s=300))
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    # Triple buffer requires num_k_tiles >= 3
    num_k_tiles = K // tile_k
    if num_buffers == 3 and num_k_tiles < 3:
        pytest.skip(f"Triple buffer requires num_k_tiles >= 3, got {num_k_tiles}")

    # Check LDS budget: A+B per buffer, times num_buffers, must fit 384KB
    lds_pad = 8
    elem_bytes = 2
    a_buf = tile_m * (tile_k + lds_pad) * elem_bytes
    b_buf = 0 if use_preshuffle else tile_n * (tile_k + lds_pad) * elem_bytes
    total_lds = (a_buf + b_buf) * num_buffers
    if total_lds > 327680:
        pytest.skip(f"LDS budget exceeded: {total_lds} > 327680")

    print(f"Running WMMA GEMM TDM: M={M}, N={N}, K={K}, "
          f"dtype={in_dtype}, cs={use_cshuffle}, ps={use_preshuffle}, "
          f"bufs={num_buffers}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)

    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((N, K), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device=device)
    b_pad = torch.zeros((npad, K), dtype=torch_dtype, device=device)
    a_pad[:M, :] = a
    b_pad[:N, :] = b

    if use_preshuffle:
        b_input = preshuffle_b_weight(b_pad.cpu(), tile_k=tile_k).cuda()
    else:
        b_input = b_pad

    c_pad = torch.zeros((mpad, npad), dtype=torch.float32, device=device)

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        use_cshuffle=use_cshuffle,
        use_preshuffle=use_preshuffle,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
    )
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_input.contiguous().view(-1),
        mpad, npad, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.mm(a.cpu().to(torch.float32), b.cpu().to(torch.float32).T)
    rtol = 3e-2
    atol = 3e-2
    assert verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=rtol, atol=atol)
    print("PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--no-cshuffle", action="store_true")
    parser.add_argument("--no-preshuffle", action="store_true")
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    args = parser.parse_args()

    test_wmma_gemm_tdm(
        args.dtype, args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        use_cshuffle=not args.no_cshuffle,
        use_preshuffle=not args.no_preshuffle,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
    )
