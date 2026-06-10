#!/usr/bin/env python3
"""Correctness test for k_warp (K-parallel warp) GEMM on gfx1250 (FFM-Lite).

Uses kernels/gemm_fp8fp4_gfx1250_kpar.py which adds the k_warp compile option.
Each warp computes a K-slice of the same output tile, with an LDS reduce-add
epilogue.

Usage:
    python3 tests/kernels/test_gemm_kpar_ffm.py
    python3 tests/kernels/test_gemm_kpar_ffm.py --data-format fp8
    python3 tests/kernels/test_gemm_kpar_ffm.py -M 1 -N 512 -K 512
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import flydsl.compiler as flyc
from kernels.gemm_fp8fp4_gfx1250_unified import compile_fp8fp4_gemm
from tests.kernels.utils import fp4_utils

SCALE_BLOCK = 32


# ── helpers copied from test_gemm_fp8fp4_gfx1250.py ──

def _align_up(v, a):
    return ((v + a - 1) // a) * a


def preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=4, WMMA_DIM=16):
    rows, K_scale = scale.shape
    align = warp_tile
    if rows % align != 0:
        pad = _align_up(rows, align) - rows
        scale = torch.cat(
            [scale, torch.full((pad, K_scale), 127, dtype=scale.dtype, device=scale.device)],
            dim=0,
        )
    SCALES_PER_WMMA = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def reference_mxfp4_gemm(a_packed, b_packed, a_scale, b_scale, M, N, K):
    a_f32 = fp4_utils.mxfp4_to_f32(a_packed.view(torch.uint8))[:M, :K]
    b_f32 = fp4_utils.mxfp4_to_f32(b_packed.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K):
    a_f32 = fp4_utils.fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K]
    b_f32 = fp4_utils.fp8_e4m3_to_f32(b.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


def reference_a8w4_gemm(a_fp8, b_fp4, a_scale, b_scale, M, N, K):
    a_f32 = fp4_utils.fp8_e4m3_to_f32(a_fp8.view(torch.uint8))[:M, :K]
    b_f32 = fp4_utils.mxfp4_to_f32(b_fp4.view(torch.uint8))[:N, :K]
    a_sc = fp4_utils.e8m0_to_f32(a_scale.view(torch.uint8))
    b_sc = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))
    a_sc_exp = a_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
    b_sc_exp = b_sc.repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc_exp, (b_f32 * b_sc_exp).T)


# ── main test ──

def run_kpar_gemm_test(
    data_format="fp4",
    M=1,
    N=512,
    K=512,
    tile_m=16,
    tile_n=128,
    tile_k=256,
    m_warp=1,
    n_warp=1,
    k_warp=2,
    num_buffers=2,
    out_dtype="f32",
):
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm not available")
        return

    is_fp4 = data_format == "fp4"
    is_a8w4 = data_format == "a8w4"
    pack_a = 1 if (data_format in ("fp8", "a8w4")) else 2
    pack_b = 2 if (data_format in ("fp4", "a8w4")) else 1

    torch.manual_seed(42)

    print(f"\n{'='*60}")
    print(f"  k_warp GEMM test")
    print(f"  data_format={data_format}, scale_mode=mxscale")
    print(f"  M={M} N={N} K={K}")
    print(f"  tile=({tile_m},{tile_n},{tile_k}), m_warp={m_warp}, n_warp={n_warp}, k_warp={k_warp}")
    print(f"  num_buffers={num_buffers}, out_dtype={out_dtype}")
    print(f"{'='*60}")

    # Generate inputs
    if is_a8w4:
        a = torch.randint(0, 126, (M, K), dtype=torch.uint8)
        b = fp4_utils.random_fp4_packed(N, K)
    elif is_fp4:
        a = fp4_utils.random_fp4_packed(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
    else:
        a = torch.randint(0, 126, (M, K), dtype=torch.uint8)
        b = torch.randint(0, 126, (N, K), dtype=torch.uint8)

    a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
    b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)
    a_scale_raw = a_scale.clone()
    b_scale_raw = b_scale.clone()

    # Reference
    if is_a8w4:
        ref = reference_a8w4_gemm(a, b, a_scale, b_scale, M, N, K)
    elif is_fp4:
        ref = reference_mxfp4_gemm(a, b, a_scale, b_scale, M, N, K)
    else:
        ref = reference_mxfp8_gemm(a, b, a_scale, b_scale, M, N, K)

    print(f"Ref:  min={ref.min():.4f}  max={ref.max():.4f}  mean={ref.mean():.4f}")

    # Pad M to tile_m (kernel needs M >= tile_m for OOB predication)
    padded_m = _align_up(M, tile_m)
    padded_k = K
    padded_n = N

    if a.shape[0] < padded_m:
        pad_a = torch.zeros(padded_m - M, K // pack_a, dtype=a.dtype)
        a = torch.cat([a, pad_a], dim=0)
        pad_as = torch.full((padded_m - M, K // SCALE_BLOCK), 127, dtype=a_scale.dtype)
        a_scale = torch.cat([a_scale, pad_as], dim=0)

    # Preshuffle scales (use warp_tile from m/n_warp, not k_warp)
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    skt = tile_k // SCALE_BLOCK
    a_scale = preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt)
    b_scale = preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt)

    # Preshuffle B
    K_packed_b = K // pack_b
    b = fp4_utils.preshuffle_b_16x16(b, N, K_packed_b)

    # Upload
    _dtype_map = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
    torch_out_dtype = _dtype_map[out_dtype]

    a_gpu = a.cuda()
    b_gpu = b.cuda()
    as_gpu = a_scale.cuda()
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch_out_dtype, device="cuda")

    # Compile k_warp kernel
    launch_fn = compile_fp8fp4_gemm(
        data_format=data_format,
        scale_mode="mxscale",
        N=padded_n,
        K=padded_k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        k_warp=k_warp,
        num_buffers=num_buffers,
        use_tdm_store=True,
        out_dtype=out_dtype,
        expert_sched_mode=True,
    )

    # Launch
    flyc.compile(
        launch_fn,
        c_gpu.contiguous(),
        a_gpu.contiguous(),
        b_gpu.contiguous(),
        as_gpu.contiguous(),
        bs_gpu.contiguous(),
        padded_m,
        padded_n,
        padded_k,
        padded_n,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    c_out = c_gpu[:M, :N].float().cpu()
    ref_f = ref.float()

    print(f"Out:  min={c_out.min():.4f}  max={c_out.max():.4f}  mean={c_out.mean():.4f}")

    if c_out.abs().max() < 1e-10:
        print("WARNING: kernel output is all zeros!")

    diff = (c_out - ref_f).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        c_out.flatten().unsqueeze(0).double(),
        ref_f.flatten().unsqueeze(0).double(),
    ).item()
    print(f"Diff: max={diff.max():.6f}  mean={diff.mean():.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    # Tolerances — aligned with test_gemm_fp8fp4_gfx1250.py
    peak = float(ref_f.abs().max())
    if is_fp4:
        if out_dtype in ("bf16", "f16"):
            torch.testing.assert_close(c_out, ref_f, rtol=2e-2, atol=max(1.0, peak * 5e-3))
        else:
            torch.testing.assert_close(c_out, ref_f, rtol=1e-5, atol=1e-4)
    elif is_a8w4:
        # Scale-range-aware tolerance matching _a8w4_tolerances() in the main test.
        a_exp = int(a_scale_raw.max().item()) - 127
        b_exp = int(b_scale_raw.max().item()) - 127
        peak_prod_exp = max(0, a_exp) + max(0, b_exp)
        if out_dtype in ("bf16", "f16"):
            rtol = min(5e-2, 1e-2 + 3e-3 * peak_prod_exp)
            atol = max(5e-2, K * (0.6 + 1.5 * peak_prod_exp))
        else:
            rtol = min(2e-2, 1e-3 + 2e-3 * peak_prod_exp)
            atol = max(1e-2, K * (0.6 + 0.55 * peak_prod_exp))
        torch.testing.assert_close(c_out, ref_f, rtol=rtol, atol=atol)
    else:
        atol = max(1e-2, K * 0.6)
        torch.testing.assert_close(c_out, ref_f, rtol=2e-2, atol=atol)

    print("PASSED ✓")


def main():
    p = argparse.ArgumentParser(description="k_warp GEMM correctness test on FFM-Lite")
    p.add_argument("--data-format", default="fp4", choices=["fp4", "fp8", "a8w4"])
    p.add_argument("-M", type=int, default=1)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("--tile-m", type=int, default=16)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--m-warp", type=int, default=1)
    p.add_argument("--n-warp", type=int, default=1)
    p.add_argument("--k-warp", type=int, default=2)
    p.add_argument("--num-buffers", type=int, default=2)
    p.add_argument("--out-dtype", default="f32", choices=["f32", "bf16", "f16"])
    args = p.parse_args()

    run_kpar_gemm_test(
        data_format=args.data_format,
        M=args.M,
        N=args.N,
        K=args.K,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        k_warp=args.k_warp,
        num_buffers=args.num_buffers,
        out_dtype=args.out_dtype,
    )


if __name__ == "__main__":
    main()
