#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cluster multicast GEMM tests for gfx1250.

Validates WMMA GEMM with TDM multicast via compile_wmma_gemm_tdm.
These tests exercise the full cluster launch + data sharing pipeline.

Status: DEFERRED — currently hangs during JIT compilation of
compile_wmma_gemm_tdm with cluster parameters. See
temp-doc/cluster_mcast_gemm_known_issue.md for details.
"""

import pytest

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

_arch = str(get_rocm_arch())
if _arch != "gfx1250":
    pytest.skip(f"Cluster mcast GEMM requires gfx1250, got {_arch}", allow_module_level=True)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _pad_2d_tensor(t, rows, cols):
    r, c = t.shape
    if r == rows and c == cols:
        return t
    out = torch.zeros((rows, cols), dtype=t.dtype, device=t.device)
    out[:r, :c] = t
    return out


@pytest.mark.skip(reason="Hangs during JIT compilation with cluster params — deferred to another PR")
@pytest.mark.parametrize(
    "M, N, K, cluster_m, cluster_n",
    [
        (512, 512, 512, 2, 2),
        (1024, 1024, 512, 2, 2),
    ],
)
def test_cluster_mcast_gemm(M, N, K, cluster_m, cluster_n):
    """WMMA GEMM with TDM multicast: end-to-end cluster launch + data sharing."""
    from kernels.wmma_gemm_gfx1250 import compile_wmma_gemm_tdm
    from tests.test_common import verify_output

    tile_m, tile_n, tile_k = 128, 256, 128
    num_buffers = 2
    in_dtype = "bf16"

    mpad = _align_up(M, tile_m)
    npad = _align_up(N, tile_n)
    kpad = _align_up(K, tile_k)

    num_k_tiles = kpad // tile_k
    if num_k_tiles < num_buffers:
        pytest.skip(f"K too small for {num_buffers}-buffer pipeline")

    wg_m = mpad // tile_m
    wg_n = npad // tile_n
    if wg_m < cluster_m or wg_n < cluster_n:
        pytest.skip(f"WG grid ({wg_m},{wg_n}) too small for cluster ({cluster_m},{cluster_n})")
    if (wg_m % cluster_m) != 0 or (wg_n % cluster_n) != 0:
        pytest.skip(f"WG grid ({wg_m},{wg_n}) not divisible by cluster ({cluster_m},{cluster_n})")

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    a = torch.randn((M, K), dtype=torch_dtype, device="cpu")
    b = torch.randn((K, N), dtype=torch_dtype, device="cpu")
    ref = torch.mm(a.to(torch.float32), b.to(torch.float32))

    a_pad = _pad_2d_tensor(a, mpad, kpad).cuda()
    b_pad = _pad_2d_tensor(b, kpad, npad).cuda()
    c_pad = torch.zeros((mpad, npad), dtype=torch_dtype, device="cuda")

    launch_fn = compile_wmma_gemm_tdm(
        M=mpad,
        N=npad,
        K=kpad,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=2,
        n_warp=4,
        in_dtype=in_dtype,
        num_buffers=num_buffers,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
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

    assert verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=3e-2, atol=3e-2)
