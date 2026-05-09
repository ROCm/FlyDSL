#!/usr/bin/env python3
"""DeepSeek blockscale FP8 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/blockscale_gemm_gfx1250.py

Data layout (test-side):
  x:        [M, K] fp8 e4m3
  weight:   [N, K] fp8 e4m3
  x_scale:  [M, K/128] fp32  (preshuffled before passing to kernel)
  w_scale:  [N/128, K/128] fp32  (no preshuffle)

Reference:
  Reuses ``run_torch_blockscale`` from
  ``tests/kernels/test_blockscale_preshuffle_gemm.py``.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch
import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.blockscale_gemm_gfx1250 import compile_blockscale_gemm
from tests.kernels.utils.fp4_utils import (
    preshuffle_a_scale_for_wmma,
    preshuffle_b_16x16,
)
from tests.kernels.test_blockscale_preshuffle_gemm import run_torch_blockscale
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.",
                allow_module_level=True)


BLOCK_SHAPE = (128, 128)  # (block_n, block_k) — DeepSeek V3.1 default
DTYPE_FP8 = torch.float8_e4m3fn  # gfx1250 uses fnuz=False


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _run_blockscale_gemm_test(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
    num_buffers, use_tdm_store, out_dtype,
    *,
    data_format: str = "fp8",
    scale_block_n: int = 128,
    scale_block_k: int = 128,
    wave_specialized_tdm: bool = False,
    l2_prefetch_distance: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    inst_prefetch: bool = False,
    waves_per_eu=None,
    expert_sched_mode: bool = True,
    split_k: int = 1,
    atomic_barrier_enable: bool = False,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"blockscale gfx1250 GEMM requires gfx1250, got {arch}")

    if K % scale_block_k != 0:
        pytest.skip(f"K={K} must be divisible by scale_block_k={scale_block_k}")

    # Pad runtime problem to tile-aligned kernel dimensions.
    padded_m = _align_up(M, tile_m)
    padded_n = _align_up(N, tile_n)
    padded_k = _align_up(K, tile_k * split_k)

    num_k_tiles = (padded_k // split_k) // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")
    if not (padded_n % scale_block_n == 0 and padded_k % scale_block_k == 0):
        pytest.skip("padded shape not divisible by scale block")

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16,
                  "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]

    torch.manual_seed(0)

    mcast_str = f", cluster=({cluster_m},{cluster_n})" \
        if cluster_m > 1 or cluster_n > 1 else ""
    tdm_str = ", tdm_store" if use_tdm_store else ", buffer_store"
    pad_str = ""
    if (padded_m, padded_n, padded_k) != (M, N, K):
        pad_str = f", kernel_pad=({padded_m},{padded_n},{padded_k})"
    print(f"\nRunning Blockscale FP8 GEMM: M={M}, N={N}, K={K}{pad_str}, "
          f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}"
          f"{mcast_str}{tdm_str}, out={out_dtype}")

    device = torch.device("cuda")

    # Generate input data (small magnitude to avoid FP8 overflow)
    x_fp16 = (torch.rand((M, K), dtype=torch.float16, device=device) / 10)
    weight_fp16 = (torch.rand((N, K), dtype=torch.float16, device=device) / 10)
    x = x_fp16.to(DTYPE_FP8)
    weight = weight_fp16.to(DTYPE_FP8)

    scale_k_dim = K // scale_block_k
    scale_n_dim = (N + scale_block_n - 1) // scale_block_n
    x_scale = torch.rand([M, scale_k_dim], dtype=torch.float32, device=device)
    w_scale = torch.rand([scale_n_dim, scale_k_dim], dtype=torch.float32,
                         device=device)

    # Reference (PyTorch fp32 dequant + matmul)
    c_ref = run_torch_blockscale(
        x, weight, x_scale, w_scale,
        block_shape=(scale_block_n, scale_block_k),
        dtype=torch.float32,
    )

    # Pad to kernel's expected tile-aligned dims
    if (padded_m, padded_n, padded_k) != (M, N, K):
        x_pad = torch.zeros((padded_m, padded_k), dtype=DTYPE_FP8, device=device)
        x_pad[:M, :K] = x
        x = x_pad
        w_pad = torch.zeros((padded_n, padded_k), dtype=DTYPE_FP8, device=device)
        w_pad[:N, :K] = weight
        weight = w_pad
        xs_pad = torch.zeros((padded_m, padded_k // scale_block_k),
                             dtype=torch.float32, device=device)
        xs_pad[:M, :scale_k_dim] = x_scale
        x_scale = xs_pad
        ws_pad = torch.zeros((padded_n // scale_block_n, padded_k // scale_block_k),
                             dtype=torch.float32, device=device)
        ws_pad[:scale_n_dim, :scale_k_dim] = w_scale
        w_scale = ws_pad

    # B 16x16 byte preshuffle (same as mxscale)
    b_shuffled = preshuffle_b_16x16(weight, padded_n, padded_k)

    # A scale wave32-friendly preshuffle
    scale_k_per_tile = tile_k // scale_block_k
    a_scale_pre = preshuffle_a_scale_for_wmma(
        x_scale, padded_m, padded_k, tile_m, m_warp,
        scale_block_k=scale_block_k,
        scale_k_per_tile=scale_k_per_tile,
    )
    # B scale: no preshuffle, just flatten
    b_scale_flat = w_scale.contiguous().view(-1)

    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device=device)

    launch_fn = compile_blockscale_gemm(
        data_format=data_format,
        M=padded_m, N=padded_n, K=padded_k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m, cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        split_k=split_k,
        expert_sched_mode=expert_sched_mode,
        atomic_barrier_enable=atomic_barrier_enable,
    )

    launch_fn(
        c_gpu.contiguous().view(-1),
        x.contiguous().view(-1),
        b_shuffled.contiguous().view(-1),
        a_scale_pre,
        b_scale_flat,
        padded_m, padded_n, torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].cpu().float()
    c_ref_f = c_ref.cpu().float()

    print(f"Out  stats: min={c_out.min():.2f}, max={c_out.max():.2f}, "
          f"mean={c_out.mean():.2f}, std={c_out.std():.2f}")
    print(f"Ref  stats: min={c_ref_f.min():.2f}, max={c_ref_f.max():.2f}, "
          f"mean={c_ref_f.mean():.2f}, std={c_ref_f.std():.2f}")

    if c_out.abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    diff = (c_out - c_ref_f).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    # Cosine similarity is for diagnostic visibility only; not used as the
    # pass/fail gate. The actual gate is verify_output below.
    cos_sim = torch.nn.functional.cosine_similarity(
        c_out.flatten().unsqueeze(0), c_ref_f.flatten().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    passed = verify_output(c_out, c_ref_f, rtol=1e-2, atol=1e-2)
    assert passed, "verify_output failed"
    print("PASSED")


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 512, 512, 128, 128, 128, 2, 2),
        (128, 512, 1024, 128, 128, 128, 2, 2),
        (256, 256, 512, 128, 128, 128, 2, 2),
        (128, 128, 1024, 128, 128, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("out_dtype", ["bf16", "f16"])
def test_blockscale_gemm_basic(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                                num_buffers, use_tdm_store, out_dtype):
    _run_blockscale_gemm_test(
        M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype)


def test_blockscale_gemm_b_scale_crosses_n_boundary():
    # tile_n=192 with n_warp=2 gives warp_tile_n=96. The second N warp covers
    # columns 96..191 and must use both b_scale rows 0 and 1.
    _run_blockscale_gemm_test(
        128, 384, 512,
        128, 192, 128,
        2, 2,
        num_buffers=2,
        use_tdm_store=False,
        out_dtype="bf16",
    )


# DeepSeek V3 production shapes
@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 7168, 2304, id="DS-7168x2304-M128"),
        pytest.param(256, 7168, 2304, id="DS-7168x2304-M256",
                     marks=pytest.mark.large_shape),
        pytest.param(128, 2112, 7168, id="DS-2112x7168-M128"),
        pytest.param(256, 2112, 7168, id="DS-2112x7168-M256",
                     marks=pytest.mark.large_shape),
        pytest.param(128, 3072, 1536, id="DS-3072x1536-M128"),
    ],
)
@pytest.mark.parametrize("out_dtype", ["bf16", "f16"])
def test_blockscale_gemm_deepseek_shapes(M, N, K, out_dtype):
    # Auto-pick a reasonable tile config.
    tile_m, tile_n, tile_k = 128, 256, 128
    m_warp, n_warp = 2, 4
    num_buffers = 3
    _run_blockscale_gemm_test(
        M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store=False, out_dtype=out_dtype,
        l2_prefetch_distance=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blockscale FP8 GEMM (gfx1250)")
    parser.add_argument("--data-format", type=str, default="fp8",
                        choices=["fp8"],
                        help="Element format (only 'fp8' supported in v1)")
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--scale-block-k", type=int, default=128,
                        help="K-dim scale block size (must equal WMMA_K=128 in v1)")
    parser.add_argument("--scale-block-n", type=int, default=128,
                        help="N-dim scale block size (must be multiple of WMMA_N=16)")
    parser.add_argument("--l2-prefetch-distance", type=int, default=2)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--no-tdm-store", action="store_true", default=False,
                        help="Disable TDM store epilogue (use buffer_store)")
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--wave-spec-tdm", action="store_true", default=False,
                        help="Wave-specialized TDM (1 loader wave per A/B stream)")
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode",
                        action="store_false", default=True)
    parser.add_argument("--atomic-barrier-enable", action="store_true", default=False,
                        help="Enable TDM atomic_barrier_enable (hardware auto-barrier)")
    parser.add_argument("--out-dtype", type=str, default="bf16",
                        choices=["f32", "bf16", "f16"])
    args = parser.parse_args()

    use_tdm_store = (not args.no_tdm_store) and args.split_k == 1
    _run_blockscale_gemm_test(
        args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        args.m_warp, args.n_warp,
        num_buffers=args.num_buffers,
        use_tdm_store=use_tdm_store,
        out_dtype=args.out_dtype,
        data_format=args.data_format,
        scale_block_k=args.scale_block_k,
        scale_block_n=args.scale_block_n,
        wave_specialized_tdm=args.wave_spec_tdm,
        split_k=args.split_k,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m, cluster_n=args.cluster_n,
        inst_prefetch=args.inst_prefetch,
        waves_per_eu=args.waves_per_eu,
        expert_sched_mode=args.expert_sched_mode,
        atomic_barrier_enable=args.atomic_barrier_enable,
    )
