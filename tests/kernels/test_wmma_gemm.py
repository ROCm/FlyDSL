#!/usr/bin/env python3
"""WMMA GEMM tests for gfx1250 — @flyc.kernel API.

Kernel implementation lives in `kernels/wmma_gemm_flyc.py`.
This file is the correctness + perf harness.
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
from kernels.wmma_gemm_flyc import compile_wmma_gemm


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (16, 16, 32),
        (33, 33, 32),
        (127, 65, 64),
    ],
)
def test_wmma_gemm(in_dtype, M, N, K):
    if str(get_rocm_arch()) != "gfx1250":
        pytest.skip(f"WMMA baseline requires gfx1250, got {get_rocm_arch()}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)

    mpad = (M + 15) // 16 * 16
    npad = (N + 15) // 16 * 16

    a = torch.randn((M, K), dtype=torch_dtype, device=device)
    b = torch.randn((K, N), dtype=torch_dtype, device=device)
    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device=device)
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device=device)
    a_pad[:M, :] = a
    b_pad[:, :N] = b
    c_pad = torch.zeros((mpad, npad), dtype=torch.float32, device=device)

    launch_fn = compile_wmma_gemm(
        M=mpad,
        N=npad,
        K=K,
        tile_m=16,
        tile_n=16,
        tile_k=32,
        in_dtype=in_dtype,
        block_threads=32,
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

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    assert torch.allclose(c_pad[:M, :N], ref, rtol=3e-2, atol=3e-2)
