#!/usr/bin/env python3
"""Precision tests for a16w8 batched GEMM (fp8 weight + per-block scale) on gfx1250.

Operation: C[b,m,n] = sum_k A[b,m,k] * (fp8_B[b,n,k] * scale[b,k//128,n//128])
  where B[b,n,k] = wo_a layout (N-outer, K-inner, K-major)
Reference: torch.bmm(A.float(), B_dequant.transpose(-1,-2).float()).bfloat16()
  where B_dequant[b,n,k] = fp8_B[b,n,k].float() * scale_expanded[b,n,k]
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

import flydsl  # noqa: E402,F401

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.bmm_a16w8_gfx1250 import compile_bmm_a16w8_gfx1250
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _make_fp8_weight(shape, device="cuda"):
    """Create a random fp8_e4m3fn tensor (gfx1250 hardware format, bias=7)."""
    raw_f32 = torch.randn(shape, dtype=torch.float32, device=device) * 0.1
    raw_f32 = raw_f32.clamp(-1.0, 1.0)
    try:
        return raw_f32.to(torch.float8_e4m3fn)
    except (AttributeError, RuntimeError):
        return raw_f32.to(torch.float16).view(torch.uint8)


def _dequant_b(b_fp8, scale, group_k, group_n):
    """Dequant B from fp8 + per-block scale to f32.

    Args:
        b_fp8: [B, N, K] fp8 (wo_a layout, K-inner)
        scale: [B, K//gk, N//gn] fp32
        group_k, group_n: scale block sizes
    Returns:
        [B, N, K] float32
    """
    B, N, K = b_fp8.shape
    # Convert fp8 to f32
    try:
        b_f32 = b_fp8.float()
    except RuntimeError:
        b_f32 = b_fp8.view(torch.int8).float() / 127.0  # rough approximation

    # Expand scale: [B, K//gk, N//gn] → [B, N, K]
    # scale_exp[b, n, k] = scale[b, k//gk, n//gn]
    scale_exp = scale.view(B, K // group_k, N // group_n, 1, 1)
    scale_exp = scale_exp.expand(-1, -1, -1, group_n, group_k)  # [B, K//gk, N//gn, gn, gk]
    scale_exp = scale_exp.permute(0, 2, 3, 1, 4)                # [B, N//gn, gn, K//gk, gk]
    scale_exp = scale_exp.reshape(B, N, K)

    return b_f32 * scale_exp


def run_a16w8_test(
    B, M, N, K,
    group_k=128, group_n=128,
    tile_m=128, tile_n=128, tile_k=128,
    num_buffers=3,
    m_warp=2, n_warp=4,
    l2_prefetch_distance=2,
    use_tdm_store=True,
    use_e8m0_scale=False,
    no_scale=False,
    cluster_m=1, cluster_n=1,
    waves_per_eu=None,
    atol=5e-2, rtol=5e-2,
    bench=False, bench_iters=100,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"a16w8 BMM requires gfx1250, got {arch}")

    mpad = _align_up(M, tile_m)
    if N % tile_n != 0:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n}")
    if K % tile_k != 0:
        pytest.skip(f"K={K} must be divisible by tile_k={tile_k}")
    if tile_k != group_k:
        pytest.skip(f"tile_k ({tile_k}) must equal group_k ({group_k})")
    if tile_n != group_n:
        pytest.skip(f"tile_n ({tile_n}) must equal group_n ({group_n})")

    torch.manual_seed(0)

    # A: [B, M, K] bf16
    a = torch.randn((B, M, K), dtype=torch.bfloat16, device="cpu") * 0.1
    # scale: [B, K//gk, N//gn] fp32 (small positive values)
    scale_fp32 = (torch.rand((B, K // group_k, N // group_n), dtype=torch.float32) * 0.1 + 0.01)

    if no_scale:
        # no_scale=True: kernel uses E8M0=127 (scale=1.0 constant), no HBM scale load.
        # Reference: plain fp8→f32 type conversion, no scale multiply.
        scale = torch.zeros(1, dtype=torch.uint8)  # dummy, unused
        scale_ref = None
    elif use_e8m0_scale:
        # Round fp32 scale to nearest power-of-2 (E8M0 format).
        # E8M0 byte = biased exponent: 2^(e8m0 - 127). round(log2(s)) → nearest 2^n.
        log2_s = torch.log2(scale_fp32.clamp(min=1e-38))
        exp_rounded = log2_s.round().to(torch.int32)
        e8m0_bytes = (exp_rounded + 127).clamp(0, 255).to(torch.uint8)
        # Reference uses rounded scales: 2^(e8m0-127) exactly
        scale_ref = (2.0 ** exp_rounded.float())
        scale = e8m0_bytes  # uint8 for kernel
    else:
        scale_ref = scale_fp32
        scale = scale_fp32

    # B in fp8_e4m3fn [B, N, K] (wo_a layout, K-inner/K-major).
    # Note: float8_e4m3fnuz (bias=8) causes 2× decode error on gfx1250.
    try:
        b_f32_raw = torch.randn((B, N, K), dtype=torch.float32) * 0.1
        b_fp8 = b_f32_raw.clamp(-1.0, 1.0).to(torch.float8_e4m3fn)
        b_for_ref = b_fp8.float()
    except (AttributeError, RuntimeError):
        b_for_ref = torch.randn((B, N, K), dtype=torch.float32) * 0.1
        b_fp8 = (b_for_ref.clamp(-1.0, 1.0) * 127).to(torch.int8)

    # Reference: dequant B then batched GEMM in f32.
    # b_for_ref is [B, N, K] (wo_a layout); transpose to [B, K, N] for bmm.
    if no_scale:
        ref = torch.bmm(a.float(), b_for_ref.transpose(-1, -2)).bfloat16()
    else:
        # Expand scale [B, K//gk, N//gn] → [B, N, K]: scale_expanded[b,n,k]=scale[b,k//gk,n//gn]
        scale_expanded = (scale_ref
                          .view(B, K // group_k, N // group_n, 1, 1)
                          .expand(-1, -1, -1, group_n, group_k)   # [B, K//gk, N//gn, gn, gk]
                          .permute(0, 2, 3, 1, 4)                 # [B, N//gn, gn, K//gk, gk]
                          .reshape(B, N, K))
        b_dequant_f32 = b_for_ref * scale_expanded          # [B, N, K]
        ref = torch.bmm(a.float(), b_dequant_f32.transpose(-1, -2)).bfloat16()

    # Pad M if needed
    if mpad > M:
        a_pad = torch.zeros((B, mpad, K), dtype=torch.bfloat16)
        a_pad[:, :M, :] = a
    else:
        a_pad = a

    # Move to GPU
    a_gpu = a_pad.cuda().contiguous()
    b_gpu = b_fp8.cuda().contiguous()
    scale_gpu = scale.cuda().contiguous()
    c_gpu = torch.zeros((B, mpad, N), dtype=torch.bfloat16, device="cuda")

    print(
        f"a16w8 BMM B={B} M={M}(pad={mpad}) K={K} N={N} "
        f"gk={group_k} gn={group_n} tile={tile_m}x{tile_n}x{tile_k} "
        f"bufs={num_buffers} tdm_store={use_tdm_store} e8m0={use_e8m0_scale} "
        f"no_scale={no_scale} cluster=({cluster_m},{cluster_n}) wpe={waves_per_eu}"
    )

    launch_fn = compile_bmm_a16w8_gfx1250(
        B=B, M=mpad, N=N, K=K,
        group_k=group_k, group_n=group_n,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        use_e8m0_scale=use_e8m0_scale,
        no_scale=no_scale,
        cluster_m=cluster_m, cluster_n=cluster_n,
        waves_per_eu=waves_per_eu,
    )

    a_flat = a_gpu.view(-1)
    b_flat = b_gpu.view(-1)
    scale_flat = scale_gpu.view(-1)
    c_flat = c_gpu.view(-1)

    launch_fn(c_flat, a_flat, b_flat, scale_flat, mpad, torch.cuda.current_stream())
    torch.cuda.synchronize()

    c_out = c_gpu[:, :M, :].cpu().float()
    ok = verify_output(c_out, ref.float(), rtol=rtol, atol=atol)
    if not ok:
        max_diff = (c_out - ref.float()).abs().max().item()
        print(f"  FAILED: max_diff={max_diff:.4f}")
        assert ok, f"Precision check FAILED: max_diff={max_diff:.4f}"
    print("  PASSED")

    if bench:
        for _ in range(10):
            launch_fn(c_flat, a_flat, b_flat, scale_flat, mpad, torch.cuda.current_stream())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            launch_fn(c_flat, a_flat, b_flat, scale_flat, mpad, torch.cuda.current_stream())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / bench_iters * 1e6
        flops = 2 * B * M * N * K
        tflops = flops / (us * 1e-6) / 1e12
        # Memory: A bf16 + B fp8 + scale fp32 + C bf16
        mem_bytes = (B * M * K * 2 + B * K * N * 1
                     + B * (K // 128) * (N // 128) * 4
                     + B * M * N * 2)
        bw_tbs = mem_bytes / (us * 1e-6) / 1e12
        print(f"  bench: {us:.2f} µs  {tflops:.2f} TFLOPS  {bw_tbs:.2f} TB/s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", [128, 256])
def test_a16w8_v4_shapes(M):
    """V4 shapes: B=16, K=4096, N=1024."""
    run_a16w8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=128, tile_n=128, tile_k=128,
        num_buffers=3,
    )


@pytest.mark.parametrize("M", [64, 128])
def test_a16w8_decode(M):
    """Decode-like shapes with small M. nb=3 confirmed best on AM (207,035 vs 214,886 sclk)."""
    run_a16w8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=64, tile_n=128, tile_k=128,
        num_buffers=3,
    )


@pytest.mark.parametrize("use_tdm_store", [True, False])
def test_a16w8_epilogue_variants(use_tdm_store):
    """Test both epilogue modes."""
    run_a16w8_test(
        B=4, M=128, N=128, K=256,
        group_k=128, group_n=128,
        tile_m=128, tile_n=128, tile_k=128,
        num_buffers=2,
        use_tdm_store=use_tdm_store,
    )


@pytest.mark.parametrize("num_buffers", [2, 3])
def test_a16w8_pipeline_depths(num_buffers):
    """Test double and triple buffer pipelining."""
    run_a16w8_test(
        B=8, M=256, N=128, K=512,
        group_k=128, group_n=128,
        tile_m=128, tile_n=128, tile_k=128,
        num_buffers=num_buffers,
    )


@pytest.mark.parametrize("M", [64, 128, 256])
def test_a16w8_e8m0_scale(M):
    """E8M0-only scale path (O2): uint8 scale, no residual mulf.

    Scale is pre-rounded to nearest 2^n. Reference uses rounded E8M0 scales.
    """
    tile_m = 64 if M <= 128 else 128
    run_a16w8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=tile_m, tile_n=128, tile_k=128,
        num_buffers=3,
        use_e8m0_scale=True,
        atol=5e-2, rtol=5e-2,
    )


@pytest.mark.parametrize("M", [64, 128, 256])
def test_a16w8_no_scale(M):
    """no_scale=True: E8M0=127 constant (scale=1.0), plain fp8→bf16 type conversion.

    arg_scale is unused. Reference: A @ B_fp8.float() without any scale multiply.
    This is the wo_a use case — fp8_e4m3fn weights are self-describing floats.
    """
    tile_m = 64 if M <= 128 else 128
    run_a16w8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=tile_m, tile_n=128, tile_k=128,
        num_buffers=3,
        no_scale=True,
        atol=5e-2, rtol=5e-2,
    )


@pytest.mark.parametrize("M", [64, 128])
def test_a16w8_cluster(M):
    """Cluster_n=4 A multicast: WGs sharing same M-tile load A once via MCAST.

    Uses cluster_n=4 (< gy=8 for N=1024/tile_n=128) — valid cluster size.
    Correctness: output must match non-cluster E8M0 reference.
    """
    tile_m = 64 if M <= 128 else 128
    run_a16w8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=tile_m, tile_n=128, tile_k=128,
        num_buffers=3,
        use_e8m0_scale=True,
        cluster_n=4,
        atol=5e-2, rtol=5e-2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="a16w8 BMM gfx1250 test/bench")
    parser.add_argument("-B", type=int, default=16)
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--group-k", type=int, default=128)
    parser.add_argument("--group-n", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--no-tdm-store", action="store_true")
    parser.add_argument("--e8m0-scale", action="store_true",
                        help="Use uint8 E8M0 scale (O2 optimization)")
    parser.add_argument("--no-scale", action="store_true",
                        help="no_scale mode: E8M0=127 constant, arg_scale unused")
    parser.add_argument("--cluster-n", type=int, default=1,
                        help="Cluster size along N (A multicast, default=1=disabled)")
    parser.add_argument("--cluster-m", type=int, default=1,
                        help="Cluster size along M (B multicast, default=1=disabled)")
    parser.add_argument("--waves-per-eu", type=int, default=None,
                        help="Occupancy hint (default=compiler chooses)")
    parser.add_argument("--l2-prefetch-distance", type=int, default=2,
                        help="L2 prefetch K-tile lookahead (0=disabled, default=2)")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-iters", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_a16w8_test(
        B=args.B, M=args.M, N=args.N, K=args.K,
        group_k=args.group_k, group_n=args.group_n,
        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        m_warp=args.m_warp, n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        use_tdm_store=not args.no_tdm_store,
        use_e8m0_scale=args.e8m0_scale,
        no_scale=args.no_scale,
        cluster_m=args.cluster_m, cluster_n=args.cluster_n,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
        bench=args.bench, bench_iters=args.bench_iters,
    )
