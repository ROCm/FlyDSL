# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Combined a8w4-interleave production-path dispatch + comparison harness.

Mirrors aiter's gfx950 production dispatch for the per-1x32 (mxfp4-weight) MoE
INTERLEAVE / Swiglu path (aiter/fused_moe.py):

    q_dtype_a = bf16  if M <  AITER_BF16_FP8_MOE_BOUND (256)   # a16w4 (mixed_moe)
              = fp8   if M >= 256                              # a8w4  (mxfp_moe)

Our combined dispatch routes:
    token <  256 -> OUR a16w4 (bf16 A x mxfp4 W)  [moe_2stage_a16wmix]
    token >= 256 -> OUR a8w4  (fp8  A x mxfp4 W)  [mxfp_moe, interleave]

Both regimes consume the SAME on-device W1/W2 tensor layout (standard
``shuffle_weight`` + ``e8m0_shuffle``); interleave is purely an internal kernel
N-column addressing choice in mxfp_moe gemm1, so no boundary re-shuffle is needed.

Correctness: ours vs a torch reference (bf16 A x mxfp4 W for M<256; fp4-quant A x
mxfp4 W for M>=256). This is the correctness the aiter dispatch is *supposed* to
provide; aiter's own small-M half runs its BROKEN a16w4 gemm1.

Run (from repo root, env sourced, GPU pinned):
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python -m kernels.moe.moe_2stage_a16wmix.compare_interleave_sweep
"""

import argparse
import math
import statistics

import torch

# Reuse the vetted test helpers (quant, routing, torch ref, launch glue).
from tests.kernels.test_moe_gemm import (
    _per_1x32_fp4_quant,
    _per_1x32_mxfp8_quant,
    build_routing_buffers,
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    flydsl_mxfp4_gemm1,
    flydsl_mxfp4_gemm2,
)
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.kernels.utils import gemm_common_utils as gcu
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight

BOUND = 256  # aiter AITER_BF16_FP8_MOE_BOUND default


def _cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8).item()


def _logits_diff(x, y):
    # Same metric as tests.test_common.verify_output's calc_diff: 1 - 2<x,y>/(|x|^2+|y|^2).
    x, y = x.double().flatten(), y.double().flatten()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - (2 * (x * y).sum() / denom)).item()


def _median_us(fn, iters=3, warmup=2, inner=10):
    # run_perftest requires num_iters > 1; take median of `iters` independent
    # timed runs, each averaging `inner` launches after `warmup` warmups.
    vals = []
    for _ in range(iters):
        _, us = run_perftest(fn, num_iters=inner, num_warmup=warmup)
        vals.append(float(us))
    return statistics.median(vals)


def _prep_weights(w1_fp32, w2_fp32, experts, model_dim, inter_dim, dev):
    N_OUT = 2 * inter_dim
    w1_q, w1_scale = _per_1x32_fp4_quant(w1_fp32.reshape(experts * N_OUT, model_dim))
    w2_q, w2_scale = _per_1x32_fp4_quant(w2_fp32.reshape(experts * model_dim, inter_dim))
    w1_shuf = shuffle_weight(w1_q.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    w2_shuf = shuffle_weight(w2_q.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    w1_scale_1d = gcu.e8m0_shuffle(w1_scale.view(experts * N_OUT, model_dim // 32)).view(torch.uint8).contiguous()
    w2_scale_1d = gcu.e8m0_shuffle(w2_scale.view(experts * model_dim, inter_dim // 32)).view(torch.uint8).contiguous()
    return dict(
        w1_q=w1_q,
        w2_q=w2_q,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_shuf=w1_shuf,
        w2_shuf=w2_shuf,
        w1_scale_1d=w1_scale_1d,
        w2_scale_1d=w2_scale_1d,
    )


def _run_a16w4(tokens, model_dim, inter_dim, experts, topk, BM, x_fp32, W, routing, iters):
    """token<256 regime: our a16w4 (bf16 A x mxfp4 W). Returns (out, s1_us, s2_us)."""
    dev = x_fp32.device
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, _ = routing
    sorted_token_ids = sorted_token_ids.to(dev)
    sorted_weights = sorted_weights.to(dev)
    sorted_expert_ids = sorted_expert_ids.to(dev)
    num_valid_ids = num_valid_ids.to(dev)
    cumsum = num_valid_ids.to(torch.int32).contiguous()
    m_indices = sorted_token_ids.to(torch.int32).contiguous()
    x_bf16 = x_fp32.to(torch.bfloat16).contiguous()

    inter_sorted = torch.zeros(sorted_size, inter_dim, dtype=torch.bfloat16, device=dev)

    def _g1():
        flydsl_a16w4_gemm1(
            a_bf16=x_bf16,
            w1_u8=W["w1_shuf"],
            w1_scale_u8=W["w1_scale_1d"],
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            m_indices=m_indices,
            inter_sorted_bf16=inter_sorted,
            n_tokens=tokens,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=BM,
            tile_n=256 if inter_dim % 256 == 0 else 128,
            tile_k=256,
        )

    out_buf = torch.zeros(tokens * model_dim, dtype=torch.bfloat16, device=dev)

    def _g2():
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=inter_sorted,
            w2_u8=W["w2_shuf"],
            w2_scale_u8=W["w2_scale_1d"],
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_token_ids,
            sorted_weights=sorted_weights,
            flat_out=out_buf,
            M_logical=tokens,
            max_sorted=sorted_size,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            tile_m=BM,
            tile_n=256,
            tile_k=256,
        )

    s1 = _median_us(_g1, iters)
    s2 = _median_us(_g2, iters)
    out_buf.zero_()
    _g1()
    _g2()
    torch.cuda.synchronize()
    return out_buf.view(tokens, model_dim).float(), s1, s2


def _run_a8w4(tokens, model_dim, inter_dim, experts, topk, BM, x_fp32, W, routing, iters, interleave=True):
    """token>=256 regime: our a8w4 (fp8 A x mxfp4 W, interleave). Returns (out, s1_us, s2_us)."""
    dev = x_fp32.device
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, _ = routing
    sorted_token_ids = sorted_token_ids.to(dev)
    sorted_weights = sorted_weights.to(dev)
    sorted_expert_ids = sorted_expert_ids.to(dev)
    num_valid_ids = num_valid_ids.to(dev)

    x_q, x_scale = _per_1x32_mxfp8_quant(x_fp32)
    cumsum = num_valid_ids.to(torch.int32).contiguous()
    m_indices = (sorted_token_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    x_scale_sort = (
        gcu.moe_mxfp4_sort(
            x_scale[:tokens].view(tokens, 1, -1),
            sorted_ids=sorted_token_ids,
            num_valid_ids=num_valid_ids,
            token_num=tokens,
            block_size=BM,
        )
        .view(torch.uint8)
        .contiguous()
    )

    scale_cols = inter_dim // 32
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    aqout = torch.zeros(sorted_size, inter_dim // 2, dtype=torch.uint8, device=dev)
    ascaleout = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
    hidden = torch.zeros(tokens, model_dim, dtype=torch.bfloat16, device=dev)

    def _g1():
        flydsl_mxfp4_gemm1(
            a_quant=x_q.view(torch.uint8).contiguous(),
            a_scale_sorted_shuffled=x_scale_sort,
            w1_u8=W["w1_shuf"],
            w1_scale_u8=W["w1_scale_1d"],
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            m_indices=m_indices,
            inter_sorted_quant=aqout,
            inter_sorted_shuffled_scale=ascaleout,
            hidden_states=hidden,
            n_tokens=tokens,
            BM=BM,
            use_nt=(BM == 32),
            inline_quant=False,
            interleave=interleave,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            a_dtype="fp8",
        )

    out_buf = torch.zeros(tokens * model_dim, dtype=torch.bfloat16, device=dev)

    def _g2():
        flydsl_mxfp4_gemm2(
            inter_sorted_quant=aqout,
            inter_sorted_shuffled_scale=ascaleout,
            w2_u8=W["w2_shuf"],
            w2_scale_u8=W["w2_scale_1d"],
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_token_ids,
            sorted_weights=sorted_weights,
            flat_out=out_buf,
            M_logical=tokens,
            max_sorted=sorted_size,
            BM=BM,
            use_nt=True,
            epilog="atomic",
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
        )

    _g1()  # populate aqout/ascaleout for stage2 timing
    torch.cuda.synchronize()
    s1 = _median_us(_g1, iters)
    s2 = _median_us(_g2, iters)
    out_buf.zero_()
    _g1()
    _g2()
    torch.cuda.synchronize()
    return out_buf.view(tokens, model_dim).float(), s1, s2


def _ref_a16(x_fp32, W, topk_ids, topk_weights, model_dim, inter_dim):
    x_bf16 = x_fp32.to(torch.bfloat16).contiguous()
    ref1 = torch_moe_gemm1(
        x_bf16,
        W["w1_q"],
        None,
        W["w1_scale"],
        topk_ids.long(),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=False,
    )
    return torch_moe_gemm2(
        ref1.to(torch.bfloat16),
        W["w2_q"],
        None,
        W["w2_scale"],
        topk_ids.long(),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=True,
    )


def _ref_bf16_dense(x_fp32, W, topk_ids, topk_weights, model_dim, inter_dim):
    """Trustworthy fidelity reference: bf16 A x (dequantized) mxfp4 W, no A-quant,
    no fp4 intermediate requant. This is the same yardstick aiter's test uses
    (logits_diff vs a bf16 dense moe). Both regimes are compared against it so the
    reported cosine is a real fidelity number, not the near-no-op e2e-test gate."""
    x_bf16 = x_fp32.to(torch.bfloat16).contiguous()
    ref1 = torch_moe_gemm1(
        x_bf16,
        W["w1_q"],
        None,
        W["w1_scale"],
        topk_ids.long(),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=False,
    )
    return torch_moe_gemm2(
        ref1.to(torch.bfloat16),
        W["w2_q"],
        None,
        W["w2_scale"],
        topk_ids.long(),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=True,
    )


def _ref_a8(x_fp32, W, topk_ids, topk_weights, tokens, model_dim, inter_dim, topk):
    x_q, x_scale = _per_1x32_mxfp8_quant(x_fp32)
    ref1 = torch_moe_gemm1(
        x_q,
        W["w1_q"],
        x_scale,
        W["w1_scale"],
        topk_ids.long(),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=False,
    )
    a2_q, a2_scale = _per_1x32_fp4_quant(ref1.reshape(tokens * topk, inter_dim))
    return torch_moe_gemm2(
        a2_q.view(tokens, topk, -1),
        W["w2_q"],
        a2_scale.view(tokens, topk, -1),
        W["w2_scale"],
        topk_ids.long(),
        topk_weights,
        model_dim=model_dim,
        doweight_stage2=True,
    )


def run_sweep(model_dim, inter_dim, experts, topk, token_list, iters, cos_bar, seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    s = 0.2
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32) * s
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=dev, dtype=torch.float32) * (s / math.sqrt(inter_dim))
    W = _prep_weights(w1_fp32, w2_fp32, experts, model_dim, inter_dim, dev)

    print(
        f"\n=== Combined a8w4-interleave sweep  MoE {model_dim}x{inter_dim} E{experts} k{topk} "
        f"(boundary={BOUND}) ==="
    )
    hdr = (
        f"{'token':>7} | {'regime':>6} | {'s1(us)':>9} {'s2(us)':>9} {'tot(us)':>9} | "
        f"{'e2e_pass':>8} {'cos_bf16':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for tokens in token_list:
        BM = 32
        x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * s
        score = torch.rand((tokens, experts), device=dev, dtype=torch.float32)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
        routing = build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=BM,
            moe_sort_mode="torch",
        )

        if tokens < BOUND:
            regime = "a16w4"
            out, s1, s2 = _run_a16w4(tokens, model_dim, inter_dim, experts, topk, BM, x_fp32, W, routing, iters)
            ref = _ref_a16(x_fp32, W, topk_ids, topk_weights, model_dim, inter_dim)
        else:
            regime = "a8w4"
            out, s1, s2 = _run_a8w4(tokens, model_dim, inter_dim, experts, topk, BM, x_fp32, W, routing, iters)
            ref = _ref_a8(x_fp32, W, topk_ids, topk_weights, tokens, model_dim, inter_dim, topk)

        # e2e_pass: the SAME (near-no-op) gate the shipped pytest e2e uses.
        ok = bool(verify_output(out, ref, rtol=0.5, atol=0.5, logits_diff_threshold=1))
        # cos_bf16: TRUSTWORTHY fidelity vs a bf16-dense moe (aiter's yardstick).
        ref_bf = _ref_bf16_dense(x_fp32, W, topk_ids, topk_weights, model_dim, inter_dim)
        cos_bf = _cos(out, ref_bf)
        rows.append((tokens, regime, s1, s2, s1 + s2, ok, cos_bf))
        mark = "  <<< 256 boundary" if tokens == BOUND else ""
        print(
            f"{tokens:>7} | {regime:>6} | {s1:>9.2f} {s2:>9.2f} {s1 + s2:>9.2f} | "
            f"{'PASS' if ok else 'FAIL':>8} {cos_bf:>9.4f}{mark}"
        )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dim", type=int, default=3584)
    ap.add_argument("--inter_dim", type=int, default=512)
    ap.add_argument("--experts", type=int, default=896)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--tokens", type=str, default="1,16,128,256,1024,4096,16384")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cos_bar", type=float, default=0.95)
    args = ap.parse_args()
    tl = [int(t) for t in args.tokens.split(",") if t.strip()]
    run_sweep(args.model_dim, args.inter_dim, args.experts, args.topk, tl, args.iters, args.cos_bar)
