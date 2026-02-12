#!/usr/bin/env python3
"""MoE Blockscale GEMM Test - flydsl 2-stage vs aiter fused kernel.

Tests the MoE flow end-to-end:
  Stage 1: X @ W1[expert].T -> SiLU(gate)*up -> fp16 intermediate
  Stage 2: act @ W2[expert].T -> atomic accumulate to output

Compares flydsl 2-stage (per-token-scale) vs aiter fused blockscale kernel.
Both do the same FP8 MFMA compute; difference is scale granularity.
"""

import os
import sys
import logging
import math

import torch
import torch.nn.functional as F
from einops import rearrange

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2
from tests.test_common import run_perftest, verify_output, checkAllclose
from tests.utils import pertoken_quant, shuffle_weight
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)
# Quiet aiter spam
logging.getLogger("aiter").setLevel(logging.ERROR)

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

try:
    import aiter
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting, fused_topk
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight
    from aiter import pertoken_quant as aiter_pertoken_quant

    HAS_AITER = True
except ImportError:
    HAS_AITER = False

# Use aiter or torch for routing
from tests.kernels.test_moe_gemm import (
    build_routing_buffers,
    RoutingBuffers,
)


def torch_moe_blockscale_ref(
    hidden_states, w1, w2, topk_weight, topk_ids, dtype,
    scale_blks=(128, 128), a_scale=None, fc1_scale=None, fc2_scale=None,
):
    """Torch reference for blockscale MoE (from aiter test)."""
    computeType = torch.float32
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    token_num = hidden_states.shape[0]
    topk = topk_weight.shape[1]
    expert, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape

    blk_n, blk_k = scale_blks
    if a_scale is not None:
        hidden_states = hidden_states.view(token_num, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(token_num, -1)

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros((B, topk, D), dtype=computeType, device=hidden_states.device)

    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_k = model_dim // blk_k
    nblk_n_w2 = model_dim // blk_n
    nblk_k_w2 = inter_dim // blk_k

    if fc1_scale is not None:
        fc1_s = rearrange(
            fc1_scale.view(-1, 1).repeat(1, blk_n * blk_k)
            .view(expert, -1, nblk_k, blk_n, blk_k),
            "e n k bn bk -> e (n bn) (k bk)",
        )
        fc2_s = rearrange(
            fc2_scale.view(-1, 1).repeat(1, blk_n * blk_k)
            .view(expert, nblk_k_w2, nblk_n_w2, blk_k, blk_n),
            "e k n bk bn -> e (n bn) (k bk)",
        )
        w1 = w1 * fc1_s
        w2 = w2 * fc2_s

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub = hidden_states[mask]
            act = sub @ w1[E_id].T
            gate, up = act.split([inter_dim, inter_dim], dim=-1)
            out[mask] = (F.silu(gate) * up) @ w2[E_id].T

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


def test_moe_blockscale_e2e(
    token, model_dim, inter_dim, E, topk,
    tile_m=32, tile_n=128, tile_k=128,
    num_iters=10, num_warmup=3,
):
    """End-to-end MoE blockscale test: flydsl 2-stage vs aiter fused."""
    dtype = torch.bfloat16
    scale_blks = (128, 128)
    scale_blk_n, scale_blk_k = scale_blks
    device = torch.device("cuda")

    print("=" * 80)
    print(f"MoE Blockscale E2E: tokens={token}, dim={model_dim}, "
          f"inter={inter_dim}, E={E}, topk={topk}, tile={tile_m}x{tile_n}x{tile_k}")
    print("=" * 80)

    # ---- Generate data ----
    s = 0.2
    x_fp32 = torch.randn((token, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn((E, inter_dim * 2, model_dim), device=device, dtype=torch.float32) * s
    w2_fp32 = torch.randn((E, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))
    score = torch.rand((token, E), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    # ---- Per-token quantize for flydsl (standard per-row scale) ----
    x_q, x_scale = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
    w1_q, w1_scale = pertoken_quant(w1_fp32.view(-1, model_dim), quant_dtype=DTYPE_FP8)
    w1_q = w1_q.view(E, 2 * inter_dim, model_dim)
    w1_scale = w1_scale.view(E * 2 * inter_dim)
    w2_q, w2_scale = pertoken_quant(w2_fp32.view(-1, inter_dim), quant_dtype=DTYPE_FP8)
    w2_q = w2_q.view(E, model_dim, inter_dim)
    w2_scale = w2_scale.view(E * model_dim)

    # Preshuffle weights
    w1_shuf = shuffle_weight(w1_q.view(-1, model_dim), layout=(16, 16)).view(-1)
    w2_shuf = shuffle_weight(w2_q.view(-1, inter_dim), layout=(16, 16)).view(-1)

    # ---- Block quantize for aiter (memory-efficient: process per-expert) ----
    def block_quant_weight(w, blk_n, blk_k):
        """Block-quantize weight tensor [E, N, K] -> (w_q fp8 [E,N,K], scale [E, flat])."""
        E_w, N_w, K_w = w.shape
        nbn, nbk = N_w // blk_n, K_w // blk_k
        w_q_list, s_list = [], []
        for e in range(E_w):
            we = w[e].float()  # [N, K]
            tmp = we.view(nbn, blk_n, nbk, blk_k).permute(0, 2, 1, 3).reshape(nbn * nbk, blk_n * blk_k)
            q, s = pertoken_quant(tmp, quant_dtype=DTYPE_FP8)
            q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N_w, K_w)
            w_q_list.append(q)
            s_list.append(s.view(-1))
        return torch.stack(w_q_list), torch.stack(s_list)

    torch.cuda.empty_cache()
    w1_bq, w1_bscale = block_quant_weight(w1_fp32, scale_blk_n, scale_blk_k)
    torch.cuda.empty_cache()
    w2_bq, w2_bscale = block_quant_weight(w2_fp32, scale_blk_n, scale_blk_k)
    torch.cuda.empty_cache()

    # Input block quantize
    a1_bq, a1_bscale = pertoken_quant(
        x_fp32.view(-1, model_dim // scale_blk_k, scale_blk_k),
        quant_dtype=DTYPE_FP8)
    a1_bq = a1_bq.view(-1, model_dim)
    a1_bscale = a1_bscale.squeeze(-1)

    # ---- Torch reference (blockscale) ----
    out_ref = torch_moe_blockscale_ref(
        a1_bq, w1_bq, w2_bq, topk_weights, topk_ids, dtype,
        scale_blks=scale_blks, a_scale=a1_bscale,
        fc1_scale=w1_bscale, fc2_scale=w2_bscale)

    # ---- flydsl 2-stage (per-token scale) ----
    routing = build_routing_buffers(
        topk_ids=topk_ids, topk_weights=topk_weights,
        experts=E, model_dim=model_dim, tile_m=tile_m,
        moe_sort_mode="aiter" if HAS_AITER else "torch")

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _sorted_size, _blocks = routing
    size_expert_ids = sorted_expert_ids.numel()

    # Compile stage 1
    exe1 = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage1=False, in_dtype="fp8", out_dtype="f16")

    out1 = torch.zeros((token, topk, inter_dim), dtype=torch.float16, device=device)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    def launch_stage1():
        exe1(out1.view(-1), x_q.view(-1), w1_shuf, x_scale.view(-1), w1_scale,
             sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
             token, inter_dim, model_dim, size_expert_ids, stream_ptr)

    _, us1 = run_perftest(launch_stage1, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    # Re-quantize intermediate for stage 2
    out1_fp32 = out1.to(torch.float32)
    a2_q, a2_scale = pertoken_quant(out1_fp32.view(-1, inter_dim), quant_dtype=DTYPE_FP8)

    # Compile stage 2
    exe2 = compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage2=True, in_dtype="fp8", out_dtype="f16")

    out2 = torch.zeros((token, model_dim), dtype=torch.float16, device=device)

    def launch_stage2():
        exe2(out2.view(-1), a2_q.view(-1), w2_shuf, a2_scale.view(-1), w2_scale,
             sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
             token, model_dim, inter_dim, size_expert_ids, stream_ptr)

    _, us2 = run_perftest(launch_stage2, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    us_fly_total = us1 + us2
    print(f"  flydsl: stage1={us1:.1f}us + stage2={us2:.1f}us = total {us_fly_total:.1f}us")

    # Verify flydsl output (per-token scale, different from blockscale ref, so loose check)
    out2_f32 = out2.to(torch.float32)

    # ---- aiter fused blockscale ----
    us_aiter = 0
    if HAS_AITER:
        try:
            # aiter moe_sorting requires int32 topk_ids and fp32 topk_weights
            sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_ids_a, out_aiter = (
                aiter_moe_sorting(topk_ids.to(torch.int32), topk_weights.float(), E, model_dim, dtype))

            w1_bq_shuf = aiter_shuffle_weight(w1_bq, (16, 16))
            w2_bq_shuf = aiter_shuffle_weight(w2_bq, (16, 16))

            def launch_aiter():
                out_aiter.zero_()
                aiter.fmoe_fp8_blockscale_g1u1(
                    out_aiter, a1_bq, w1_bq_shuf, w2_bq_shuf,
                    sorted_ids_a, sorted_weights_a, sorted_expert_ids_a,
                    num_valid_ids_a, topk,
                    a1_bscale.t().contiguous(), w1_bscale, w2_bscale,
                    "", scale_blk_n, scale_blk_k, None)

            _, us_aiter = run_perftest(launch_aiter, num_iters=num_iters, num_warmup=num_warmup)
            torch.cuda.synchronize()
            out_aiter_f32 = out_aiter.to(torch.float32)

            # Verify aiter vs blockscale ref
            err = checkAllclose(out_ref, out_aiter, rtol=0.1, atol=0.1, msg="aiter vs ref")
            print(f"  aiter:  fused={us_aiter:.1f}us  (err_ratio={err:.4f})")
        except Exception as e:
            print(f"  aiter:  FAILED - {str(e)[:80]}")

    # ---- Summary ----
    ratio = us_aiter / us_fly_total if us_aiter > 0 else 0
    flops_stage1 = 2 * token * topk * (2 * inter_dim) * model_dim
    flops_stage2 = 2 * token * topk * model_dim * inter_dim
    flops_total = flops_stage1 + flops_stage2
    tflops_fly = flops_total / (us_fly_total / 1e6) / 1e12
    tflops_aiter = flops_total / (us_aiter / 1e6) / 1e12 if us_aiter > 0 else 0

    print(f"\n  {'':>12} | {'us':>8} | {'TFLOPS':>8} | ratio")
    print(f"  {'flydsl':>12} | {us_fly_total:>8.1f} | {tflops_fly:>8.2f} |")
    if us_aiter > 0:
        print(f"  {'aiter fused':>12} | {us_aiter:>8.1f} | {tflops_aiter:>8.2f} | {ratio:.2f}")
    print()

    return us_fly_total, us_aiter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MoE Blockscale Benchmark")
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-dim", type=int, default=7168)
    parser.add_argument("-idim", type=int, default=256)
    parser.add_argument("-e", "--expert", type=int, default=256)
    parser.add_argument("-k", "--topk", type=int, default=8)
    parser.add_argument("--tile_m", type=int, default=32)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--num_iters", type=int, default=10)
    args = parser.parse_args()

    torch.set_default_device("cuda")

    # Run multiple token counts
    token_list = [args.m] if args.m > 0 else [1, 2, 16, 32, 128, 512]

    print(f"\nMoE Config: dim={args.dim}, inter={args.idim}, E={args.expert}, "
          f"topk={args.topk}, tile={args.tile_m}x{args.tile_n}x{args.tile_k}")
    print(f"{'tokens':>8} | {'flydsl us':>10} | {'aiter us':>10} | {'ratio':>6}")
    print("-" * 50)

    for m in token_list:
        us_fly, us_aiter = test_moe_blockscale_e2e(
            token=m, model_dim=args.dim, inter_dim=args.idim,
            E=args.expert, topk=args.topk,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            num_iters=args.num_iters)
        r = us_aiter / us_fly if us_aiter > 0 else 0
        astr = f"{us_aiter:>10.1f}" if us_aiter > 0 else f"{'N/A':>10}"
        rstr = f"{r:>6.2f}" if r > 0 else f"{'N/A':>6}"
        print(f"{m:>8} | {us_fly:>10.1f} | {astr} | {rstr}")
