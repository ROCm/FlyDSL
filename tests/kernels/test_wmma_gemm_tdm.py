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
    "M, N, K",
    [
        (128, 128, 64),
        (128, 128, 128),
        (256, 256, 128),
        (256, 512, 256),
        (512, 512, 512),
        (200, 180, 64),
        (300, 400, 128),
    ],
)
@pytest.mark.parametrize("use_double_buffer", [False, True])
@pytest.mark.parametrize("use_cshuffle", [False, True])
@pytest.mark.parametrize("use_async_copy", [False, True])
@pytest.mark.parametrize("use_preshuffle", [False, True])
def test_wmma_gemm_tdm(in_dtype, M, N, K,
                        use_double_buffer, use_cshuffle,
                        use_async_copy, use_preshuffle):
    arch = str(get_rocm_arch(timeout_s=300))
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    tile_m, tile_n, tile_k = 128, 128, 64
    m_warp, n_warp = 2, 4

    print(f"Running WMMA GEMM TDM: M={M}, N={N}, K={K}, "
          f"dtype={in_dtype}, db={use_double_buffer}, cs={use_cshuffle}, "
          f"async={use_async_copy}, ps={use_preshuffle}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)

    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device=device)
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device=device)
    a_pad[:M, :] = a
    b_pad[:, :N] = b

    if use_preshuffle:
        b_input = preshuffle_b_weight(b_pad.cpu(), tile_k=tile_k).cuda()
    else:
        b_input = b_pad

    c_pad = torch.zeros((mpad, npad), dtype=torch.float32, device=device)

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        use_double_buffer=use_double_buffer,
        use_cshuffle=use_cshuffle,
        use_async_copy=use_async_copy,
        use_preshuffle=use_preshuffle,
    )
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_input.contiguous().view(-1),
        mpad, npad, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.matmul(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    rtol = 5e-2 if use_cshuffle else 3e-2
    atol = 5e-2 if use_cshuffle else 3e-2
    assert verify_output(c_pad[:M, :N].cpu(), ref, rtol=rtol, atol=atol)
    print("PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=256)
    parser.add_argument("-K", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--no-double-buffer", action="store_true")
    parser.add_argument("--no-cshuffle", action="store_true")
    parser.add_argument("--async-copy", action="store_true")
    parser.add_argument("--preshuffle", action="store_true")
    args = parser.parse_args()

    test_wmma_gemm_tdm(
        args.dtype, args.M, args.N, args.K,
        use_double_buffer=not args.no_double_buffer,
        use_cshuffle=not args.no_cshuffle,
        use_async_copy=args.async_copy,
        use_preshuffle=args.preshuffle,
    )
