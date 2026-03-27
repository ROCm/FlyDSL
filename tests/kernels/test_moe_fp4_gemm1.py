#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness + perf test for MoE BF16×MXFP4 gemm1 kernel.

Tests the FlyDSL MoE FP4 gemm1 kernel against a Torch reference:
  - Reference: dequantize MXFP4 weights to FP32, run BF16 matmul, apply SwiGLU/SiLU
  - Kernel: compile_moe_fp4_gemm1 from kernels/moe_fp4_gemm1.py

Architecture: gfx950 only (MI350, required for mfma_scale_f32_16x16x128_f8f6f4).
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in [os.path.join(_REPO_ROOT, "build", "python_packages"), _REPO_ROOT]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils
from tests.test_common import verify_output, run_perftest

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

ARCH = str(get_rocm_arch())
if not ARCH.startswith("gfx950"):
    pytest.skip(f"MoE FP4 GEMM requires gfx950, got {ARCH}.", allow_module_level=True)

from kernels.moe_fp4_gemm1 import compile_moe_fp4_gemm1


# ── MoE routing helper (same as test_moe_gemm.py) ────────────────────────────

def moe_sorting_torch(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-native MoE sorting that mirrors aiter's moe_sorting output."""
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert_id in range(num_experts):
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = int(token_id.numel())
        blocks = (tokens_num + block_size - 1) // block_size
        tokens_padded = blocks * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            (topk_id.to(torch.int32) << 24) | token_id.to(torch.int32)
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ].float()
        sorted_ids_begin += tokens_padded
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + blocks] = int(expert_id)
        sorted_expert_ids_begin += blocks

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = M
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


# ── Reference implementation ─────────────────────────────────────────────────

def torch_moe_fp4_gemm1_ref(
    x: torch.Tensor,           # [tokens, model_dim] bf16
    w: torch.Tensor,           # [experts, 2*inter_dim, model_dim] fp4 (packed i8)
    scale_w: torch.Tensor,     # [experts, 2*inter_dim, model_dim//32] fp32 (E8M0)
    topk_ids: torch.Tensor,    # [tokens, topk] int64
    topk_weights: torch.Tensor, # [tokens, topk] float32
    *,
    inter_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
    activation: str = "swiglu",
    doweight: bool = False,
) -> torch.Tensor:
    """Reference: dequantize FP4 weights, BF16 matmul, SwiGLU/SiLU, scatter."""
    tokens, model_dim = x.shape
    topk = topk_ids.shape[1]
    experts = w.shape[0]
    device = x.device

    # Dequantize B: MXFP4 (E2M1) → FP32
    # w shape: [experts, 2*inter_dim, model_dim] packed as i8 (2 fp4/byte)
    w_dq_list = []
    for e in range(experts):
        # fp4_utils.mxfp4_to_f32 expects [N, K] packed i8
        w_e_i8 = w[e]  # [2*inter_dim, model_dim//2] i8
        w_e_f32 = fp4_utils.mxfp4_to_f32(w_e_i8)  # [2*inter_dim, model_dim]
        # Apply E8M0 scales: scale_w[e] shape [2*inter_dim, model_dim//32]
        sc_e = fp4_utils.e8m0_to_f32(scale_w[e])   # [2*inter_dim, model_dim//32]
        sc_e_expanded = sc_e.repeat_interleave(32, dim=1)  # [2*inter_dim, model_dim]
        w_e_scaled = w_e_f32 * sc_e_expanded        # [2*inter_dim, model_dim]
        w_dq_list.append(w_e_scaled)

    # Output: [tokens, topk, inter_dim] in out_dtype
    out = torch.zeros((tokens, topk, inter_dim), dtype=out_dtype, device=device)

    x_f32 = x.float()

    def _swish(t):
        return t * torch.sigmoid(t)

    for tok_i in range(tokens):
        for slot_i in range(topk):
            e = int(topk_ids[tok_i, slot_i].item())
            w_e = w_dq_list[e]                          # [2*inter_dim, model_dim]
            gate_w = w_e[:inter_dim]                    # [inter_dim, model_dim]
            up_w = w_e[inter_dim:]                      # [inter_dim, model_dim]
            x_tok = x_f32[tok_i]                        # [model_dim]
            gate = x_tok @ gate_w.T                     # [inter_dim]
            up = x_tok @ up_w.T                         # [inter_dim]
            if activation == "swiglu":
                y = _swish(gate) * up
            else:
                y = torch.sigmoid(gate) * gate * up     # silu: same as swiglu
            if doweight:
                y = y * float(topk_weights[tok_i, slot_i].item())
            out[tok_i, slot_i] = y.to(out_dtype)

    return out


# ── Test helpers ──────────────────────────────────────────────────────────────

def _prepare_fp4_weights(w_fp32: torch.Tensor):
    """Quantize and shuffle weight tensor [N, K] → packed FP4 + scales."""
    # Quantize to MXFP4 (E2M1), per-1x32 scale blocks
    w_q_i8, scale_e8m0, _ = fp4_utils.per_1x32_f4_quant(w_fp32)  # [N, K//2], [N, K//32]
    # Shuffle for preshuffle GEMM layout
    w_shuffled = fp4_utils.shuffle_weight_w4(w_q_i8, 16, gate_up=False, moe_gemm=False)
    scale_shuffled = fp4_utils.shuffle_scale_w4(scale_e8m0, 1, False)
    return w_q_i8, scale_e8m0, w_shuffled, scale_shuffled


def run_moe_fp4_gemm1_test(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 256,
    activation: str = "swiglu",
    out_dtype_str: str = "bf16",
    doweight: bool = False,
    bench_iters: int = 0,
    bench_warmup: int = 2,
    atol: float = 0.15,
    rtol: float = 0.15,
):
    device = torch.device("cuda")
    torch_out_dtype = torch.bfloat16 if out_dtype_str == "bf16" else torch.float16

    # Random activations (BF16)
    x = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device)

    # Random topk routing
    topk_ids = torch.zeros((tokens, topk), dtype=torch.int64, device=device)
    for i in range(tokens):
        topk_ids[i] = torch.randperm(experts, device=device)[:topk]
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device=device, dtype=torch.float32), dim=-1
    )

    # Random FP32 weights for each expert [2*inter_dim, model_dim]
    w_fp32 = torch.randn(experts, 2 * inter_dim, model_dim, device=device, dtype=torch.float32)

    # Quantize and shuffle per-expert weight tiles
    w_q_all = []        # [experts, 2*inter_dim, model_dim//2] i8 (unshuffled, for reference)
    scale_all = []      # [experts, 2*inter_dim, model_dim//32] e8m0 (unshuffled)
    w_shuf_all = []     # [experts, 2*inter_dim, model_dim//2] i8 (shuffled, for kernel)
    sc_shuf_all = []    # shuffled scales
    for e in range(experts):
        w_q, sc, w_shuf, sc_shuf = _prepare_fp4_weights(w_fp32[e])
        w_q_all.append(w_q)
        scale_all.append(sc)
        w_shuf_all.append(w_shuf)
        sc_shuf_all.append(sc_shuf)

    # Stack into [experts, 2*inter_dim, model_dim//2]
    w_q_stacked = torch.stack(w_q_all, dim=0).to(device)
    scale_stacked = torch.stack(scale_all, dim=0).to(device)
    w_shuf_stacked = torch.stack(w_shuf_all, dim=0).contiguous().to(device)
    sc_shuf_stacked = torch.stack(sc_shuf_all, dim=0).contiguous().to(device)

    # Reference output
    ref_out = torch_moe_fp4_gemm1_ref(
        x, w_q_stacked, scale_stacked, topk_ids, topk_weights,
        inter_dim=inter_dim, out_dtype=torch_out_dtype,
        activation=activation, doweight=doweight,
    )  # [tokens, topk, inter_dim]

    # MoE routing (sorted format)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = moe_sorting_torch(
        topk_ids.int(), topk_weights, num_experts=experts, block_size=tile_m
    )
    size_expert_ids = int(sorted_expert_ids.shape[0])

    # Kernel output buffer: [tokens, topk, inter_dim]
    kernel_out = torch.zeros((tokens, topk, inter_dim), dtype=torch_out_dtype, device=device)

    # Flatten w/scale for kernel (kernel expects [experts*2*inter_dim, model_dim//2] contiguous)
    w_kernel = w_shuf_stacked.view(-1).contiguous()
    sc_kernel = sc_shuf_stacked.view(-1).contiguous()

    # scale_x: per-token f32 scale for A (BF16 inputs have no per-block quantization).
    # The kernel loads this as: buffer_load(sx_rsrc, token_id, vec_width=1, dtype=T.f32)
    # where token_id is the decoded token index (i32). Shape: [tokens] float32, all 1.0.
    scale_x = torch.ones((tokens,), dtype=torch.float32, device=device)

    # Compile kernel
    launch_fn = compile_moe_fp4_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        out_dtype=out_dtype_str, activation=activation, doweight_stage1=doweight,
        use_cshuffle_epilog=False,
    )

    def _launch():
        launch_fn(
            kernel_out.view(-1),
            x.view(-1),
            w_kernel,
            scale_x,
            sc_kernel,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids[:1].contiguous(),
            tokens,                     # i32_tokens_in
            inter_dim,                  # i32_inter_in
            model_dim,                  # i32_k_in
            size_expert_ids,            # i32_size_expert_ids_in
            torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(_launch)
    compiled_fn()
    torch.cuda.synchronize()

    # Compare: ref_out[tokens, topk, inter_dim] vs kernel_out[tokens, topk, inter_dim]
    ok = verify_output(kernel_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    assert ok, (
        f"MoE FP4 gemm1 output mismatch!\n"
        f"  tokens={tokens}, model_dim={model_dim}, inter_dim={inter_dim}, experts={experts}, topk={topk}\n"
        f"  tile=({tile_m},{tile_n},{tile_k}), activation={activation}\n"
        f"  max_err={((kernel_out.float() - ref_out.float()).abs().max().item()):.4f}"
    )
    print(f"[PASS] tokens={tokens}, model_dim={model_dim}, inter_dim={inter_dim}, "
          f"experts={experts}, topk={topk}, tile=({tile_m},{tile_n},{tile_k}), "
          f"activation={activation}")

    if bench_iters > 0:
        _, us = run_perftest(
            compiled_fn, num_iters=bench_iters, num_warmup=bench_warmup
        )
        # Approximate FLOPS: 2 * tokens * 2*inter_dim * model_dim
        flops = 2 * tokens * 2 * inter_dim * model_dim
        tflops = flops / (us / 1e6) / 1e12
        print(f"  Perf: {us:.1f} us, {tflops:.2f} TFLOPS")


# ── Pytest parametrize ────────────────────────────────────────────────────────

@pytest.mark.parametrize("activation", ["swiglu", "silu"])
@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk,tile_m,tile_n,tile_k",
    [
        # model_dim must be >= 2*tile_k for ping-pong pipeline
        (16, 512, 128, 4, 2, 16, 128, 256),
        (32, 512, 128, 4, 2, 32, 128, 256),
        pytest.param(64, 512, 256, 4, 2, 64, 128, 256, marks=pytest.mark.large_shape),
    ],
)
def test_moe_fp4_gemm1(
    tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, activation
):
    run_moe_fp4_gemm1_test(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        activation=activation,
        out_dtype_str="bf16",
    )


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE BF16×MXFP4 gemm1 test/bench")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--model_dim", type=int, default=512)  # must be >= 2*tile_k=512
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--tile_m", type=int, default=16)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=256)
    parser.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["bf16", "f16"])
    parser.add_argument("--doweight", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=0)
    parser.add_argument("--num_warmup", type=int, default=2)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    run_moe_fp4_gemm1_test(
        tokens=args.tokens,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        activation=args.activation,
        out_dtype_str=args.out_dtype,
        doweight=args.doweight,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
    )
