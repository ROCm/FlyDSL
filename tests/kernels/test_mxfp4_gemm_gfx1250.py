#!/usr/bin/env python3
"""MXFP4 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/mxfp4_gemm_gfx1250.py
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

from flydsl.runtime.device import get_rocm_arch
from kernels.mxfp4_gemm_gfx1250 import compile_mxfp4_gemm
from tests.kernels.utils import fp4_utils


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SCALE_BLOCK = 32


def preshuffle_e8m0_scale(scale: torch.Tensor, warp_tile: int,
                          scale_k_per_tile: int = 4,
                          WMMA_DIM: int = 16) -> torch.Tensor:
    """Preshuffle E8M0 scale for WMMA_SCALE: byte swap + interleave for ds_load_b128. """
    _, K_scale = scale.shape
    assert K_scale % 4 == 0, f"K_scale must be divisible by 4, got {K_scale}"

    grouped = scale.view(-1, K_scale // 4, 4)
    shuffled = grouped[:, :, [0, 2, 1, 3]].contiguous()
    scale = shuffled.view(-1, K_scale)

    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def random_mxfp4_packed(rows: int, cols: int, *, device="cpu") -> torch.Tensor:
    """Generate random packed MXFP4 data [rows, cols//2] uint8. """
    assert cols % 2 == 0
    unpacked = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device=device)
    return fp4_utils.pack_uint4(unpacked)


def random_e8m0(rows: int, cols: int, *, low_exp=127, high_exp=132,
                device="cpu") -> torch.Tensor:
    """Generate random E8M0 scale bytes [rows, cols] uint8. """
    return torch.randint(low_exp, high_exp + 1, (rows, cols),
                         dtype=torch.uint8, device=device)


def reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K):
    """Reference MXFP4 GEMM: D = (A * A_scale) @ (B * B_scale)^T.

    Args:
        a_packed: [M, K//2] uint8 packed FP4
        b_packed: [N, K//2] uint8 packed FP4
        a_scale:  [M, K//SCALE_BLOCK] uint8 E8M0
        b_scale:  [N, K//SCALE_BLOCK] uint8 E8M0

    Returns:
        [M, N] float32 result.
    """
    a_f32 = fp4_utils.mxfp4_to_f32(a_packed.view(torch.uint8))[:M, :K]
    b_f32 = fp4_utils.mxfp4_to_f32(b_packed.view(torch.uint8))[:N, :K]

    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))

    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]

    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 128, 256, 128, 128, 128, 2, 2),
        (128, 128, 512, 128, 128, 128, 2, 2),
        (128, 128, 1024, 128, 128, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2])
def test_mxfp4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                     num_buffers, l2_prefetch_distance=0,
                     cluster_m=1, cluster_n=1, scale_preshuffle=True):
    """MXFP4 GEMM correctness unit test."""
    arch = str(get_rocm_arch(timeout_s=300))
    if arch != "gfx1250":
        pytest.skip(f"WMMA_SCALE requires gfx1250, got {arch}")

    num_k_tiles = K // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    torch.manual_seed(0)

    mcast_str = f", cluster=({cluster_m},{cluster_n})" if cluster_m > 1 or cluster_n > 1 else ""
    print(f"\nRunning MXFP4 GEMM: M={M}, N={N}, K={K}, "
          f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}{mcast_str}")

    a_packed = random_mxfp4_packed(M, K)
    b_packed = random_mxfp4_packed(N, K)
    a_scale = random_e8m0(M, K // SCALE_BLOCK)
    b_scale = random_e8m0(N, K // SCALE_BLOCK)

    ref = reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K)
    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, "
          f"mean={ref.mean():.2f}, std={ref.std():.2f}")

    if scale_preshuffle:
        skt = tile_k // SCALE_BLOCK
        a_scale = preshuffle_e8m0_scale(a_scale, tile_m // m_warp, scale_k_per_tile=skt)
        b_scale = preshuffle_e8m0_scale(b_scale, tile_n // n_warp, scale_k_per_tile=skt)

    a_gpu = a_packed.cuda()
    b_gpu = b_packed.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(M, N, dtype=torch.float32, device="cpu").cuda()

    launch_fn = compile_mxfp4_gemm(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m, cluster_n=cluster_n,
        scale_preshuffle=scale_preshuffle,
    )
    launch_fn(
        c_gpu.contiguous().view(-1),
        a_gpu.contiguous().view(-1),
        b_gpu.contiguous().view(-1),
        as_gpu.contiguous().view(-1),
        bs_gpu.contiguous().view(-1),
        M, N, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu.cpu()

    print(f"Out stats: min={c_out.min():.2f}, max={c_out.max():.2f}, "
          f"mean={c_out.mean():.2f}, std={c_out.std():.2f}")

    if c_out.abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    diff = (c_out - ref).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    cos_sim = torch.nn.functional.cosine_similarity(
        c_out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    torch.testing.assert_close(c_out, ref, rtol=1e-5, atol=1e-8)
    print("PASSED")


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n",
    [
        # 2x2 cluster: needs >= 2 tile-rows and 2 tile-cols
        (256, 256, 256, 128, 128, 128, 2, 2, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4, 2, 2),
        # 1x2 cluster: B shared along N
        (128, 256, 256, 128, 128, 128, 2, 2, 1, 2),
        # 2x1 cluster: A shared along M
        (256, 128, 256, 128, 128, 128, 2, 2, 2, 1),
    ],
)
@pytest.mark.parametrize("num_buffers", [2])
def test_mxfp4_gemm_mcast(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                            cluster_m, cluster_n, num_buffers):
    """MXFP4 GEMM correctness test with cluster MCAST."""
    test_mxfp4_gemm(
        M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers=num_buffers,
        l2_prefetch_distance=2,
        cluster_m=cluster_m, cluster_n=cluster_n,
        scale_preshuffle=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-N", type=int, default=128)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--l2-prefetch-distance", type=int, default=0)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-scale-preshuffle", action="store_true", default=False)
    args = parser.parse_args()

    test_mxfp4_gemm(
        args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        num_buffers=args.num_buffers,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        scale_preshuffle=not args.no_scale_preshuffle,
    )
