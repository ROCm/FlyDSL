#!/usr/bin/env python3
"""Unified MXFP4/MXFP8/A8W4 GEMM correctness tests for gfx1250.

Kernel implementation: kernels/gemm_fp8fp4_gfx1250.py
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
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm
from tests.kernels.utils import fp4_utils


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SCALE_BLOCK = 32


def preshuffle_e8m0_scale(scale: torch.Tensor, warp_tile: int,
                          scale_k_per_tile: int = 4,
                          WMMA_DIM: int = 16) -> torch.Tensor:
    """Preshuffle E8M0 scale: optional byte swap + interleave for WMMA access."""
    _, K_scale = scale.shape
    assert K_scale % 4 == 0, f"K_scale must be divisible by 4, got {K_scale}"
    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def random_fp8_data(rows: int, cols: int, *, device="cpu") -> torch.Tensor:
    """Generate random FP8/E4M3 data as uint8. Avoids NaN (0x7F/0xFF)."""
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8, device=device)


def _reference_scaled_gemm(a, b, a_scale, b_scale, M, N, K,
                           convert_fn, convert_fn_b=None):
    """Reference scaled GEMM: D = (A * A_scale) @ (B * B_scale)^T."""
    a_f32 = convert_fn(a.view(torch.uint8))[:M, :K]
    b_f32 = (convert_fn_b or convert_fn)(b.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K):
    return _reference_scaled_gemm(a_packed, b_packed, a_scale, b_scale,
                                  M, N, K, fp4_utils.mxfp4_to_f32)


def reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K):
    """Standard FP8 reference with SCALE_BLOCK=32."""
    return _reference_scaled_gemm(a, b, a_scale, b_scale,
                                  M, N, K, fp4_utils.fp8_e4m3_to_f32)


def reference_a8w4_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K):
    """Standard A8W4 reference: FP8 activation + FP4 weight, SCALE_BLOCK=32."""
    return _reference_scaled_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K,
                                  fp4_utils.fp8_e4m3_to_f32,
                                  convert_fn_b=fp4_utils.mxfp4_to_f32)


def _run_mxscale_gemm_test(
    data_format, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
    num_buffers, use_tdm_store, out_dtype,
    wave_specialized_tdm=False, use_scale_opsel=False,
    l2_prefetch_distance=0, cluster_m=1, cluster_n=1,
    inst_prefetch=False, waves_per_eu=None,
    expert_sched_mode=True,
):
    """Unified test body for FP4 and FP8."""
    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"

    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA_SCALE requires gfx1250, got {arch}")

    if wave_specialized_tdm:
        pytest.skip("Simulator does not support wave-specialized TDM")

    if use_scale_opsel and is_fp4:
        pytest.skip("FP4 32x16 WMMA scaleBType op_sel ignored by AM simulator")

    num_k_tiles = K // tile_k
    if num_buffers > 1 and num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    # FP8 256x256 + f32 + TDM store exceeds LDS
    if not is_fp4 and tile_m == 256 and tile_n == 256 \
       and out_dtype == "f32" and use_tdm_store:
        pytest.skip("256x256 tile with f32 TDM store exceeds LDS limit")

    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]

    torch.manual_seed(0)

    fmt_name = "A8W4" if is_a8w4 else ("MXFP4" if is_fp4 else "MXFP8")
    mcast_str = f", cluster=({cluster_m},{cluster_n})" \
        if cluster_m > 1 or cluster_n > 1 else ""
    tdm_str = ", tdm_store" if use_tdm_store else ", buffer_store"
    print(f"\nRunning {fmt_name} GEMM: M={M}, N={N}, K={K}, "
          f"tiles=({tile_m},{tile_n},{tile_k}), bufs={num_buffers}"
          f"{mcast_str}{tdm_str}, preshuffle, out={out_dtype}")

    # Generate data
    if is_a8w4:
        a = random_fp8_data(M, K)       # FP8 activation
        b = fp4_utils.random_fp4_packed(N, K)  # FP4 weight
    elif is_fp4:
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    else:
        a = random_fp8_data(M, K)
        b = random_fp8_data(N, K)
    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

    # Reference
    if is_a8w4:
        ref = reference_a8w4_gemm(a, b, a_scale, b_scale, M, N, K)
    elif is_fp4:
        ref = reference_mxfp4_gemm(a, b, a_scale, b_scale, M, N, K)
    else:
        ref = reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K)

    print(f"Ref stats: min={ref.min():.2f}, max={ref.max():.2f}, "
          f"mean={ref.mean():.2f}, std={ref.std():.2f}")

    # Preshuffle scales
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)

    # Preshuffle B data
    K_packed = K // 2 if (is_fp4 or is_a8w4) else K
    b = fp4_utils.preshuffle_b_16x16(b, N, K_packed)

    # Upload & launch
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(M, N, dtype=torch_out_dtype, device="cpu").cuda()

    launch_fn = compile_mxscale_gemm(
        data_format=data_format,
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        cluster_m=cluster_m, cluster_n=cluster_n,
        use_tdm_store=use_tdm_store,
        out_dtype=out_dtype,
        inst_prefetch=inst_prefetch,
        wave_specialized_tdm=wave_specialized_tdm,
        use_scale_opsel=use_scale_opsel,
        expert_sched_mode=expert_sched_mode,
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

    print(f"Out stats: min={c_out.float().min():.2f}, max={c_out.float().max():.2f}, "
          f"mean={c_out.float().mean():.2f}, std={c_out.float().std():.2f}")

    if c_out.float().abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    if out_dtype in ("bf16", "f16"):
        ref_cmp = ref.to(torch_out_dtype)
        c_out_f = c_out.float()
        ref_f = ref_cmp.float()
    else:
        c_out_f = c_out.float()
        ref_f = ref.float()

    diff = (c_out_f - ref_f).abs()
    print(f"Abs diff: max={diff.max():.4f}, mean={diff.mean():.4f}")

    cos_sim = torch.nn.functional.cosine_similarity(
        c_out_f.flatten().unsqueeze(0), ref_f.flatten().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Tolerances: FP4 is exact; FP8/A8W4 have FP accumulation error
    if is_fp4:
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=1e-2)
        else:
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-5, atol=1e-8)
    else:
        # FP8 and A8W4: standard SCALE_BLOCK=32 reference
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-2, atol=5e-2)
        else:
            atol = max(1e-2, K * 0.6)
            torch.testing.assert_close(c_out_f, ref_f, rtol=1e-3, atol=atol)
    print("PASSED")


# ── pytest parametrized tests ──

@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 128, 256, 128, 128, 128, 2, 2),
        (128, 128, 512, 128, 128, 128, 2, 2),
        (128, 128, 1024, 128, 128, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("wave_specialized_tdm", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                     num_buffers, use_tdm_store, out_dtype,
                     wave_specialized_tdm, use_scale_opsel):
    _run_mxscale_gemm_test(
        "fp4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        wave_specialized_tdm=wave_specialized_tdm,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 256, 256, 128, 256, 128, 2, 4),
        (256, 256, 256, 256, 256, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp8_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                     num_buffers, use_tdm_store, out_dtype, use_scale_opsel):
    _run_mxscale_gemm_test(
        "fp8", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp",
    [
        (128, 128, 256, 128, 128, 128, 2, 2),
        (128, 128, 512, 128, 128, 128, 2, 2),
        (128, 128, 1024, 128, 128, 128, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("use_scale_opsel", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_a8w4_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                    num_buffers, use_tdm_store, out_dtype, use_scale_opsel):
    _run_mxscale_gemm_test(
        "a8w4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        use_scale_opsel=use_scale_opsel)


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, cluster_m, cluster_n",
    [
        (256, 256, 256, 128, 128, 128, 2, 2, 2, 2),
        (1024, 1024, 1024, 128, 256, 128, 2, 4, 2, 2),
        (128, 256, 256, 128, 128, 128, 2, 2, 1, 2),
        (256, 128, 256, 128, 128, 128, 2, 2, 2, 1),
        (512, 512, 256, 128, 128, 128, 2, 2, 4, 4),
        (1024, 1024, 1024, 128, 256, 128, 2, 4, 4, 4),
        (512, 512, 512, 128, 128, 128, 2, 2, 2, 4),
        (512, 512, 512, 128, 128, 128, 2, 2, 4, 2),
    ],
)
@pytest.mark.parametrize("num_buffers", [2])
@pytest.mark.parametrize("use_tdm_store", [True, False])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_mxfp4_gemm_mcast(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
                            cluster_m, cluster_n, num_buffers, use_tdm_store,
                            out_dtype):
    _run_mxscale_gemm_test(
        "fp4", M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp,
        num_buffers, use_tdm_store, out_dtype,
        l2_prefetch_distance=2,
        cluster_m=cluster_m, cluster_n=cluster_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-format", type=str, default="fp4",
                        choices=["fp4", "fp8", "a8w4"])
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
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--out-dtype", type=str, default="bf16",
                        choices=["f32", "bf16", "f16"])
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--wave-spec-tdm", action="store_true", default=False)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--use-scale-opsel", action="store_true", default=False)
    parser.add_argument("--disable-expert-sched-mode", dest="expert_sched_mode",
                        action="store_false", default=True)
    args = parser.parse_args()

    _run_mxscale_gemm_test(
        args.data_format,
        args.M, args.N, args.K,
        args.tile_m, args.tile_n, args.tile_k,
        args.m_warp, args.n_warp,
        num_buffers=args.num_buffers,
        use_tdm_store=not args.no_tdm_store,
        out_dtype=args.out_dtype,
        wave_specialized_tdm=args.wave_spec_tdm,
        use_scale_opsel=args.use_scale_opsel,
        l2_prefetch_distance=args.l2_prefetch_distance,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        inst_prefetch=args.inst_prefetch,
        waves_per_eu=args.waves_per_eu,
        expert_sched_mode=args.expert_sched_mode,
    )
