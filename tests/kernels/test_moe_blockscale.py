#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

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
import flydsl.compiler as flyc

import torch
import torch.nn.functional as F
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.moe_blockscale_2stage import compile_moe_blockscale_gemm1, compile_moe_blockscale_gemm2
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
    from aiter import pertoken_quant as aiter_pertoken_quant
    from aiter.ops.quant import per_group_quant_hip

    HAS_AITER = True
except ImportError:
    HAS_AITER = False

# Use aiter or torch for routing
from tests.kernels.test_moe_gemm import (
    build_routing_buffers,
    RoutingBuffers,
)


def _expand_blockscale(scale_flat, E, nblk_n, nblk_k, blk_n, blk_k):
    """Expand [E, nblk_n*nblk_k] flat block scales to [E, N, K] element-wise."""
    return (
        scale_flat.view(-1, 1).repeat(1, blk_n * blk_k)
        .view(E, nblk_n, nblk_k, blk_n, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, nblk_n * blk_n, nblk_k * blk_k)
    )


def torch_stage1_blockscale_ref(
    hidden_states, w1, topk_ids, a_scale, w1_scale, scale_blks, inter_dim,
):
    """Torch reference for stage 1 only: returns [B, topk, inter_dim] (SiLU(gate)*up)."""
    computeType = torch.float32
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    B, D = hidden_states.shape
    topk = topk_ids.shape[1]
    E = w1.shape[0]
    blk_n, blk_k = scale_blks

    if a_scale is not None:
        hidden_states = hidden_states.view(B, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(B, -1)

    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_k_w1 = D // blk_k
    if w1_scale is not None:
        w1 = w1 * _expand_blockscale(w1_scale, E, nblk_n_w1, nblk_k_w1, blk_n, blk_k)

    hidden_states = hidden_states.view(B, 1, D).expand(-1, topk, -1)
    out = torch.zeros((B, topk, inter_dim), dtype=computeType, device=hidden_states.device)
    for e in range(E):
        mask = topk_ids == e
        if mask.sum():
            sub = hidden_states[mask]
            act = sub @ w1[e].T
            gate, up = act.split([inter_dim, inter_dim], dim=-1)
            out[mask] = F.silu(gate) * up
    return out


def torch_stage2_blockscale_ref(
    act_q, w2, topk_ids, topk_weights, a_scale, w2_scale,
    scale_blks, tokens, model_dim, inter_dim, topk,
):
    """Torch reference for stage 2 only: takes re-quantized intermediate, returns [B, model_dim]."""
    computeType = torch.float32
    blk_n, blk_k = scale_blks
    E = w2.shape[0]

    act = act_q.to(computeType)
    nblk_k_w2 = inter_dim // blk_k
    if a_scale is not None:
        act = act.view(-1, nblk_k_w2, blk_k) * a_scale.unsqueeze(-1)
        act = act.view(-1, inter_dim)

    w2f = w2.to(computeType)
    nblk_n_w2 = model_dim // blk_n
    if w2_scale is not None:
        w2f = w2f * _expand_blockscale(w2_scale, E, nblk_n_w2, nblk_k_w2, blk_n, blk_k)

    act_3d = act.view(tokens, topk, inter_dim)
    out = torch.zeros((tokens, topk, model_dim), dtype=computeType, device=act.device)
    for e in range(E):
        mask = topk_ids == e
        if mask.sum():
            out[mask] = act_3d[mask] @ w2f[e].T
    return (out * topk_weights.view(tokens, -1, 1)).sum(dim=1)


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
        # fc1_scale: [E, nblk_n_w1 * nblk_k] -> expand to [E, 2*inter_dim, model_dim]
        # "e n k bn bk -> e (n bn) (k bk)" = permute(0,1,3,2,4).reshape
        nblk_n_w1 = (2 * inter_dim) // blk_n
        fc1_s = (
            fc1_scale.view(-1, 1).repeat(1, blk_n * blk_k)
            .view(expert, nblk_n_w1, nblk_k, blk_n, blk_k)
            .permute(0, 1, 3, 2, 4)
            .reshape(expert, 2 * inter_dim, model_dim)
        )
        # fc2_scale: [E, nblk_n_w2 * nblk_k_w2] -> expand to [E, model_dim, inter_dim]
        # "e n k bn bk -> e (n bn) (k bk)" = permute(0,1,3,2,4).reshape
        fc2_s = (
            fc2_scale.view(-1, 1).repeat(1, blk_n * blk_k)
            .view(expert, nblk_n_w2, nblk_k_w2, blk_n, blk_k)
            .permute(0, 1, 3, 2, 4)
            .reshape(expert, model_dim, inter_dim)
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


@pytest.mark.parametrize(
    "token, model_dim, inter_dim, E, topk",
    [
        pytest.param(16, 7168, 256, 8, 2, id="small-E8"),
        pytest.param(32, 7168, 256, 8, 2, id="medium-E8"),
        pytest.param(32, 7168, 256, 256, 8, id="DS-V3-M32", marks=pytest.mark.large_shape),
        pytest.param(128, 7168, 256, 256, 8, id="DS-V3-M128", marks=pytest.mark.large_shape),
    ],
)
def test_moe_blockscale_e2e(
    token, model_dim, inter_dim, E, topk,
    tile_m=16, tile_n=256, tile_k=128,
    tile_n2=None,
    waves_per_eu=2,
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
    x_q, x_scale = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
    del x_fp32

    score = torch.rand((token, E), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    # Per-expert block-quantize: generate fp32 one expert at a time, block-quantize, discard fp32.
    def block_quant_expert(w_fp32, blk_n, blk_k):
        """Block-quantize single expert weight [N, K] -> (w_q fp8 [N,K], scale [flat])."""
        N_w, K_w = w_fp32.shape
        nbn, nbk = N_w // blk_n, K_w // blk_k
        tmp = w_fp32.float().view(nbn, blk_n, nbk, blk_k).permute(0, 2, 1, 3).reshape(nbn * nbk, blk_n * blk_k)
        q, sc = pertoken_quant(tmp, quant_dtype=DTYPE_FP8)
        q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N_w, K_w)
        return q, sc.view(-1)

    w1_bq_list, w1_bscale_list = [], []
    w2_bq_list, w2_bscale_list = [], []
    for e_i in range(E):
        w1e = torch.randn((2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
        q, sc = block_quant_expert(w1e, scale_blk_n, scale_blk_k)
        w1_bq_list.append(q)
        w1_bscale_list.append(sc)
        del w1e

        w2e = torch.randn((model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))
        q2, sc2 = block_quant_expert(w2e, scale_blk_n, scale_blk_k)
        w2_bq_list.append(q2)
        w2_bscale_list.append(sc2)
        del w2e

    w1_bq = torch.stack(w1_bq_list)
    w1_bscale = torch.stack(w1_bscale_list)
    w2_bq = torch.stack(w2_bq_list)
    w2_bscale = torch.stack(w2_bscale_list)
    del w1_bq_list, w1_bscale_list, w2_bq_list, w2_bscale_list
    torch.cuda.empty_cache()

    # Preshuffle block-quantized weights (shared by FlyDSL and aiter)
    w1_bq_shuf = shuffle_weight(w1_bq, layout=(16, 16))
    w2_bq_shuf = shuffle_weight(w2_bq, layout=(16, 16))
    w1_shuf = w1_bq_shuf.view(-1)
    w2_shuf = w2_bq_shuf.view(-1)

    # Input block quantize (from x_q fp8 -> re-block)
    x_f32_for_blk = x_q.float()
    a1_bq, a1_bscale = pertoken_quant(
        x_f32_for_blk.view(-1, model_dim // scale_blk_k, scale_blk_k),
        quant_dtype=DTYPE_FP8)
    a1_bq = a1_bq.view(-1, model_dim)
    a1_bscale = a1_bscale.squeeze(-1)
    del x_f32_for_blk

    # ---- Torch reference (2-stage with intermediate re-quant, matching FlyDSL pipeline) ----
    out1_torch_ref = torch_stage1_blockscale_ref(
        a1_bq, w1_bq, topk_ids, a1_bscale, w1_bscale, scale_blks, inter_dim)
    out1_torch_fp16 = out1_torch_ref.to(torch.float16)
    a2_ref_bq, a2_ref_bscale = pertoken_quant(
        out1_torch_fp16.float().view(-1, inter_dim // scale_blk_k, scale_blk_k),
        quant_dtype=DTYPE_FP8)
    a2_ref_bq = a2_ref_bq.view(-1, inter_dim)
    a2_ref_bscale = a2_ref_bscale.squeeze(-1)
    out_ref = torch_stage2_blockscale_ref(
        a2_ref_bq, w2_bq, topk_ids, topk_weights, a2_ref_bscale, w2_bscale,
        scale_blks, token, model_dim, inter_dim, topk)

    # ---- flydsl 2-stage (true blockscale) ----
    routing = build_routing_buffers(
        topk_ids=topk_ids, topk_weights=topk_weights,
        experts=E, model_dim=model_dim, tile_m=tile_m,
        moe_sort_mode="aiter" if HAS_AITER else "torch")

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _sorted_size, _blocks = routing
    size_expert_ids = sorted_expert_ids.numel()

    # Prepare blockscale tensors for FlyDSL kernels.
    # Use per_group_quant_hip(transpose_scale=True) so the quant kernel writes scale
    # directly in [nblk_k, token] memory layout — no separate .t().contiguous() needed.
    # Fallback to pertoken_quant + manual transpose when aiter is unavailable.
    nblk_k_w1 = model_dim // scale_blk_k
    w1_scale_fly = w1_bscale.view(-1)   # [E, nblk_n_w1 * nblk_k_w1] flat

    if HAS_AITER:
        # quant kernel writes transposed scale layout directly
        a1_bq, a1_scale_fly = per_group_quant_hip(
            x_q.to(torch.bfloat16), quant_dtype=DTYPE_FP8,
            group_size=scale_blk_k, transpose_scale=True)
        a1_scale_fly = a1_scale_fly.view(-1)   # [nblk_k_w1 * token] flat
    else:
        a1_scale_fly = a1_bscale.t().contiguous().view(-1)

    # Compile stage 1
    exe1 = compile_moe_blockscale_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        doweight_stage1=False, scale_block_k=scale_blk_k, out_dtype="f16",
        waves_per_eu=waves_per_eu)

    out1 = torch.zeros((token, topk, inter_dim), dtype=torch.float16, device=device)
    stream = torch.cuda.current_stream()

    compiled_exe1 = flyc.compile(exe1, out1.view(-1), a1_bq.view(-1), w1_shuf, a1_scale_fly, w1_scale_fly,
                                sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                                token, inter_dim, model_dim, size_expert_ids, stream)

    def launch_stage1():
        compiled_exe1(out1.view(-1), a1_bq.view(-1), w1_shuf, a1_scale_fly, w1_scale_fly,
                      sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                      token, inter_dim, model_dim, size_expert_ids, stream)

    _, us1 = run_perftest(launch_stage1, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    # Verify stage 1 independently (ref uses a1_bscale in original [token, nblk_k] layout)
    out1_ref = torch_stage1_blockscale_ref(
        a1_bq, w1_bq, topk_ids, a1_bscale, w1_bscale, scale_blks, inter_dim)
    err_s1 = checkAllclose(out1_ref.to(out1.dtype), out1, rtol=0.1, atol=0.1,
                           msg="flydsl-stage1 vs ref", printLog=False)
    print(f"  stage1 err_ratio = {err_s1:.4f}")

    # Re-quantize stage1 output for stage 2
    nblk_k_w2 = inter_dim // scale_blk_k
    w2_scale_fly = w2_bscale.view(-1)   # [E, nblk_n_w2 * nblk_k_w2] flat

    if HAS_AITER:
        a2_bq, a2_scale_fly = per_group_quant_hip(
            out1.to(torch.bfloat16).view(-1, inter_dim), quant_dtype=DTYPE_FP8,
            group_size=scale_blk_k, transpose_scale=True)
        a2_scale_fly = a2_scale_fly.view(-1)   # [nblk_k_w2 * token*topk] flat
        a2_bscale_2d = a2_scale_fly.view(nblk_k_w2, -1).t().contiguous()  # [token*topk, nblk_k_w2] for ref
    else:
        a2_bq, a2_bscale_2d = pertoken_quant(
            out1.float().view(-1, inter_dim // scale_blk_k, scale_blk_k), quant_dtype=DTYPE_FP8)
        a2_bq = a2_bq.view(-1, inter_dim)
        a2_bscale_2d = a2_bscale_2d.squeeze(-1)   # [token*topk, nblk_k_w2]
        a2_scale_fly = a2_bscale_2d.t().contiguous().view(-1)

    # Compile stage 2 (optionally with different tile_n)
    tile_n_s2 = tile_n2 if tile_n2 is not None else tile_n
    exe2 = compile_moe_blockscale_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, topk=topk,
        tile_m=tile_m, tile_n=tile_n_s2, tile_k=tile_k,
        doweight_stage2=True, scale_block_k=scale_blk_k, out_dtype="f16",
        waves_per_eu=waves_per_eu)

    out2 = torch.zeros((token, model_dim), dtype=torch.float16, device=device)

    compiled_exe2 = flyc.compile(exe2, out2.view(-1), a2_bq.view(-1), w2_shuf, a2_scale_fly, w2_scale_fly,
                                sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                                token, model_dim, inter_dim, size_expert_ids, stream)

    def launch_stage2():
        compiled_exe2(out2.view(-1), a2_bq.view(-1), w2_shuf, a2_scale_fly, w2_scale_fly,
                      sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                      token, model_dim, inter_dim, size_expert_ids, stream)

    _, us2 = run_perftest(launch_stage2, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    # Re-run once with clean output for verification (atomic accumulation requires pre-zeroed buffer)
    out2.zero_()
    launch_stage2()
    torch.cuda.synchronize()

    # Verify stage 2 independently
    out2_ref_s2 = torch_stage2_blockscale_ref(
        a2_bq, w2_bq, topk_ids, topk_weights, a2_bscale_2d, w2_bscale,
        scale_blks, token, model_dim, inter_dim, topk)
    err_s2 = checkAllclose(out2_ref_s2.to(out2.dtype), out2, rtol=0.1, atol=0.1,
                           msg="flydsl-stage2 vs ref", printLog=False)
    print(f"  stage2 err_ratio = {err_s2:.4f}")

    us_fly_total = us1 + us2
    print(f"  flydsl: stage1={us1:.1f}us + stage2={us2:.1f}us = total {us_fly_total:.1f}us")

    # Verify flydsl full pipeline vs blockscale torch reference
    err_fly = checkAllclose(out_ref.to(out2.dtype), out2, rtol=0.1, atol=0.1,
                            msg="flydsl vs ref", printLog=False)
    print(f"  flydsl: pipeline err_ratio vs ref = {err_fly:.4f}")

    # ---- aiter comparisons ----
    us_aiter_fused = 0
    us_ck_s1 = 0
    us_ck_s2 = 0
    if HAS_AITER:
        from aiter.ops.moe_op import ck_moe_stage1_fwd, ck_moe_stage2_fwd
        from aiter import QuantType, ActivationType

        nblk_n_w1 = (2 * inter_dim) // scale_blk_n
        nblk_n_w2 = model_dim // scale_blk_n

        ck_block_m = 32
        sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_ids_a, out_aiter = (
            aiter_moe_sorting(topk_ids.to(torch.int32), topk_weights.float(), E, model_dim, dtype, ck_block_m))

        # --- aiter fused 1-stage ---
        out_ref_aiter = torch_moe_blockscale_ref(
            a1_bq, w1_bq, w2_bq, topk_weights, topk_ids, dtype,
            scale_blks=scale_blks, a_scale=a1_bscale,
            fc1_scale=w1_bscale, fc2_scale=w2_bscale)
        try:
            # a1_scale_fly is [nblk_k_w1, token] contiguous — exactly what fmoe_fp8_blockscale_g1u1 expects
            a1_scale_aiter = a1_scale_fly.view(nblk_k_w1, token)

            def launch_aiter_fused():
                out_aiter.zero_()
                aiter.fmoe_fp8_blockscale_g1u1(
                    out_aiter, a1_bq, w1_bq_shuf, w2_bq_shuf,
                    sorted_ids_a, sorted_weights_a, sorted_expert_ids_a,
                    num_valid_ids_a, topk,
                    a1_scale_aiter, w1_bscale, w2_bscale,
                    "", scale_blk_n, scale_blk_k, None)

            _, us_aiter_fused = run_perftest(launch_aiter_fused, num_iters=num_iters, num_warmup=num_warmup)
            torch.cuda.synchronize()
            err_fused = checkAllclose(out_ref_aiter, out_aiter, rtol=0.1, atol=0.1,
                                      msg="aiter-fused vs ref", printLog=False)
            print(f"  aiter fused: {us_aiter_fused:.1f}us  (err_ratio={err_fused:.4f})")
        except Exception as e:
            print(f"  aiter fused: FAILED - {str(e)[:80]}")

        # --- aiter CK stage1 (blockscale) ---
        try:
            out_ck_s1 = torch.zeros((token, topk, inter_dim), dtype=torch.float16, device=device)
            w1_scale_ck = w1_bscale.view(E, nblk_n_w1, nblk_k_w1).contiguous()

            def launch_ck_s1():
                ck_moe_stage1_fwd(
                    hidden_states=a1_bq,
                    w1=w1_bq_shuf, w2=w2_bq_shuf,
                    sorted_token_ids=sorted_ids_a,
                    sorted_expert_ids=sorted_expert_ids_a,
                    num_valid_ids=num_valid_ids_a,
                    out=out_ck_s1,
                    topk=topk, kernelName="",
                    w1_scale=w1_scale_ck,
                    a1_scale=a1_bscale,
                    block_m=ck_block_m,
                    sorted_weights=None,
                    quant_type=QuantType.per_1x128,
                    activation=ActivationType.Silu,
                    dst_type=torch.float16)

            _, us_ck_s1 = run_perftest(launch_ck_s1, num_iters=num_iters, num_warmup=num_warmup)
            torch.cuda.synchronize()
            err_ck_s1 = checkAllclose(out1_ref.to(out_ck_s1.dtype), out_ck_s1, rtol=0.1, atol=0.1,
                                       msg="ck-stage1 vs ref", printLog=False)
            err_ck_s1_vs_fly = checkAllclose(out1.float(), out_ck_s1.float(), rtol=0.1, atol=0.1,
                                              msg="ck-stage1 vs flydsl", printLog=False)
            print(f"  ck stage1:  {us_ck_s1:.1f}us  (err_vs_ref={err_ck_s1:.4f}, err_vs_fly={err_ck_s1_vs_fly:.4f})")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ck stage1:  FAILED - {str(e)[:120]}")

        # --- aiter CK stage2 (blockscale) ---
        try:
            out_ck_s2 = torch.zeros((token, model_dim), dtype=torch.float16, device=device)
            w2_scale_ck = w2_bscale.view(E, nblk_n_w2, nblk_k_w2).contiguous()

            def launch_ck_s2():
                out_ck_s2.zero_()
                ck_moe_stage2_fwd(
                    inter_states=a2_bq.view(token, topk, inter_dim),
                    w1=w1_bq_shuf, w2=w2_bq_shuf,
                    sorted_token_ids=sorted_ids_a,
                    sorted_expert_ids=sorted_expert_ids_a,
                    num_valid_ids=num_valid_ids_a,
                    out=out_ck_s2,
                    topk=topk, kernelName="",
                    w2_scale=w2_scale_ck,
                    a2_scale=a2_bscale_2d,
                    block_m=ck_block_m,
                    sorted_weights=sorted_weights_a,
                    quant_type=QuantType.per_1x128,
                    activation=ActivationType.Silu)

            _, us_ck_s2 = run_perftest(launch_ck_s2, num_iters=num_iters, num_warmup=num_warmup)
            torch.cuda.synchronize()
            # Re-run once for clean verification
            out_ck_s2.zero_()
            launch_ck_s2()
            torch.cuda.synchronize()
            err_ck_s2 = checkAllclose(out2_ref_s2.to(out_ck_s2.dtype), out_ck_s2, rtol=0.1, atol=0.1,
                                       msg="ck-stage2 vs ref", printLog=False)
            err_ck_s2_vs_fly = checkAllclose(out2.float(), out_ck_s2.float(), rtol=0.1, atol=0.1,
                                              msg="ck-stage2 vs flydsl", printLog=False)
            print(f"  ck stage2:  {us_ck_s2:.1f}us  (err_vs_ref={err_ck_s2:.4f}, err_vs_fly={err_ck_s2_vs_fly:.4f})")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ck stage2:  FAILED - {str(e)[:120]}")

    # ---- Summary ----
    flops_stage1 = 2 * token * topk * (2 * inter_dim) * model_dim
    flops_stage2 = 2 * token * topk * model_dim * inter_dim
    flops_total = flops_stage1 + flops_stage2
    tflops_fly = flops_total / (us_fly_total / 1e6) / 1e12
    tflops_fused = flops_total / (us_aiter_fused / 1e6) / 1e12 if us_aiter_fused > 0 else 0
    tflops_ck_s1 = flops_stage1 / (us_ck_s1 / 1e6) / 1e12 if us_ck_s1 > 0 else 0
    tflops_ck_s2 = flops_stage2 / (us_ck_s2 / 1e6) / 1e12 if us_ck_s2 > 0 else 0

    print(f"\n  {'':>14} | {'us':>8} | {'TFLOPS':>8} | vs flydsl")
    print(f"  {'flydsl-s1':>14} | {us1:>8.1f} | {flops_stage1/(us1/1e6)/1e12:>8.2f} |")
    print(f"  {'flydsl-s2':>14} | {us2:>8.1f} | {flops_stage2/(us2/1e6)/1e12:>8.2f} |")
    print(f"  {'flydsl-total':>14} | {us_fly_total:>8.1f} | {tflops_fly:>8.2f} |")
    if us_aiter_fused > 0:
        r = us_aiter_fused / us_fly_total
        print(f"  {'aiter-fused':>14} | {us_aiter_fused:>8.1f} | {tflops_fused:>8.2f} | {r:.2f}x")
    if us_ck_s1 > 0:
        r1 = us_ck_s1 / us1
        print(f"  {'ck-stage1':>14} | {us_ck_s1:>8.1f} | {tflops_ck_s1:>8.2f} | {r1:.2f}x vs fly-s1")
    if us_ck_s2 > 0:
        r2 = us_ck_s2 / us2
        print(f"  {'ck-stage2':>14} | {us_ck_s2:>8.1f} | {tflops_ck_s2:>8.2f} | {r2:.2f}x vs fly-s2")
    if us_ck_s1 > 0 and us_ck_s2 > 0:
        ck_total = us_ck_s1 + us_ck_s2
        tflops_ck_total = flops_total / (ck_total / 1e6) / 1e12
        r_total = ck_total / us_fly_total
        print(f"  {'ck-total':>14} | {ck_total:>8.1f} | {tflops_ck_total:>8.2f} | {r_total:.2f}x")
    print()

    return us_fly_total, us_aiter_fused


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MoE Blockscale Benchmark (FlyDSL vs aiter-fused vs aiter-ck2s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=0,
                        help="Token count (0 = sweep [16,32,128,256,1024])")
    parser.add_argument("-dim",  type=int, default=7168)
    parser.add_argument("-idim", type=int, default=2048)
    parser.add_argument("-e", "--expert", type=int, default=32)
    parser.add_argument("-k", "--topk",   type=int, default=8)
    parser.add_argument("--tile_m",    type=int, default=64)
    parser.add_argument("--tile_n",    type=int, default=128)
    parser.add_argument("--tile_n2",   type=int, default=0,
                        help="tile_n for stage2 (0 = same as tile_n)")
    parser.add_argument("--tile_k",    type=int, default=128)
    parser.add_argument("--wpe",       type=int, default=2,
                        help="waves_per_eu (0 = disabled)")
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup",type=int, default=5)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    wpe = args.wpe if args.wpe > 0 else None

    token_list = [args.m] if args.m > 0 else [128, 16384]

    print(f"\nMoE Blockscale: dim={args.dim}, inter={args.idim}, E={args.expert}, "
          f"topk={args.topk}, tile={args.tile_m}x{args.tile_n}x{args.tile_k}, wpe={wpe}")
    print(f"{'tokens':>8} | {'flydsl':>10} | {'aiter-fused':>12} | {'aiter-ck2s':>12}")
    print("-" * 52)

    tn2 = args.tile_n2 if args.tile_n2 > 0 else None
    for m in token_list:
        us_fly, us_aiter = test_moe_blockscale_e2e(
            token=m, model_dim=args.dim, inter_dim=args.idim,
            E=args.expert, topk=args.topk,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            tile_n2=tn2,
            waves_per_eu=wpe,
            num_iters=args.num_iters, num_warmup=args.num_warmup)
        print(f"{m:>8} | {us_fly:>10.1f} |")
