#!/usr/bin/env python3
"""WMMA GEMM using TDM tests for gfx1250.

Kernel implementation lives in `kernels/wmma_gemm_gfx1250.py`.
This file is the correctness harness.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

# workaround for simulator
import flydsl  # noqa: E402,F401 -- preload system comgr before torch/HIP loads LLVM

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.wmma_gemm_gfx1250 import compile_wmma_gemm_tdm
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
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_wmma_gemm_tdm(in_dtype, M, N, K, tile_m, tile_n, tile_k,
                        num_buffers,
                        m_warp=2, n_warp=4, l2_prefetch_distance=2,
                        out_dtype=None, use_tdm_store=True,
                        cluster_m=1, cluster_n=1):
    """Non-cluster GEMM correctness test."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    num_k_tiles = K // tile_k
    if num_buffers == 3 and num_k_tiles < 3:
        pytest.skip(f"Triple buffer requires num_k_tiles >= 3, got {num_k_tiles}")

    lds_pad = 8
    elem_bytes = 2
    a_buf = tile_m * (tile_k + lds_pad) * elem_bytes
    b_buf = tile_k * (tile_n + lds_pad) * elem_bytes
    total_lds = (a_buf + b_buf) * num_buffers
    if total_lds > 327680:
        pytest.skip(f"LDS budget exceeded: {total_lds} > 327680")

    _eff_out = out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[_eff_out]
    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n
    wg_m = mpad // tile_m
    wg_n = npad // tile_n

    if cluster_m < 1 or cluster_n < 1:
        pytest.skip(f"Invalid cluster dims: ({cluster_m}, {cluster_n}), both must be >= 1")
    if cluster_m > 1 or cluster_n > 1:
        if wg_m < cluster_m or wg_n < cluster_n:
            pytest.skip(
                "Cluster dims exceed launch grid: "
                f"wg_grid=({wg_m},{wg_n}), cluster=({cluster_m},{cluster_n})"
            )
        if (wg_m % cluster_m) != 0 or (wg_n % cluster_n) != 0:
            pytest.skip(
                "WG grid must be divisible by cluster dims: "
                f"wg_grid=({wg_m},{wg_n}), cluster=({cluster_m},{cluster_n})"
            )

    print(
        f"Running WMMA GEMM TDM: M={M}, N={N}, K={K}, "
        f"dtype={in_dtype}, out={_eff_out}, bufs={num_buffers}, "
        f"tdm_store={use_tdm_store}, cluster=({cluster_m},{cluster_n})"
    )

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device='cpu').cuda()
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device='cpu').cuda()
    a_pad[:M, :] = a
    b_pad[:, :N] = b

    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cpu').cuda()

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad, N=npad, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_pad.contiguous().view(-1),
        mpad, npad, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.mm(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    rtol = 3e-2
    atol = 3e-2
    assert verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=rtol, atol=atol)
    print("PASSED")


@pytest.mark.parametrize("in_dtype", ["fp16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (1024, 1024, 1024, 128, 256, 128),
        (2048, 2048, 1024, 128, 256, 128),
        (2048, 2048, 2048, 128, 256, 128),
        (4096, 4096, 1024, 128, 256, 128),
    ],
)
@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 2), (4, 4)])
def test_wmma_gemm_tdm_mcast(in_dtype, M, N, K, tile_m, tile_n, tile_k,
                              cluster_m, cluster_n):
    """Cluster multicast GEMM correctness test (large shapes only)."""
    test_wmma_gemm_tdm(
        in_dtype, M, N, K, tile_m, tile_n, tile_k,
        num_buffers=2, m_warp=2, n_warp=4,
        l2_prefetch_distance=2,
        cluster_m=cluster_m, cluster_n=cluster_n,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    args = parser.parse_args()

    test_wmma_gemm_tdm(
        args.dtype, args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
        use_tdm_store=not args.no_tdm_store,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
    )
