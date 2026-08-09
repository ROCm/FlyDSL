#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Standalone MoE stage2 (gemm2) benchmark for the fp8 new-vs-legacy A/B toggle.
#
# NOT a pytest file. Self-contained: reuses the reference input-build/routing
# logic from the deleted test harness run_moe_stage2 (copied inline), and the
# math reference helpers from tests/* (pertoken_quant, shuffle_weight,
# torch_moe_gemm1/2), which are NOT part of the deleted test.
#
# The A/B variant is selected by the env var MOE_FORCE_LEGACY_G2_FP8:
#   unset / "0" -> new layout-API fp8 path (_build_moe_gemm2_fp8)
#   "1"         -> legacy body
# See kernels/moe/moe_gemm_2stage/gemm2.py::compile_moe_gemm2.

import argparse
import math
import os
import statistics
import sys
from typing import Optional, Tuple

import torch

# Ensure kernels.* resolves from CWD (/tmp/pr947) and tests.* helpers import.
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

# Kernel under test.
from kernels.moe.moe_gemm_2stage import compile_moe_gemm2  # noqa: E402

# Reference math helpers (these live in tests/*, not the deleted test file).
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2  # noqa: E402
from tests.utils import pertoken_quant, shuffle_weight  # noqa: E402

ARCH = get_rocm_arch()
# gfx950 (MI350) uses OCP standard float8_e4m3fn; older MI300 uses fnuz.
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


# ---------------------------------------------------------------------------
# Routing / sorting: pure-torch path copied verbatim from the deleted test
# (moe_sorting_torch_native + build_routing_buffers, torch mode only).
# ---------------------------------------------------------------------------
def moe_sorting_torch_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for aiter's moe_sorting (torch-native path)."""
    assert topk_ids.is_cuda and topk_weights.is_cuda
    device = topk_ids.device
    M, topk = topk_ids.shape

    max_num_tokens_padded = int(topk_ids.numel() + int(num_experts) * int(block_size) - int(topk))
    max_num_m_blocks = int((max_num_tokens_padded + int(block_size) - 1) // int(block_size))

    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(int(num_experts)):
        token_id, topk_id = torch.where(topk_ids == expertId)
        tokensNum = int(token_id.numel())
        sorted_expert_ids_num = int((tokensNum + int(block_size) - 1) // int(block_size))
        tokensNumPad = int(sorted_expert_ids_num * int(block_size))
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokensNum] = (topk_id.to(torch.int32) << 24) | token_id.to(
            torch.int32
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokensNum] = topk_weights[token_id, topk_id].to(
            torch.float32
        )
        sorted_ids_begin = int(sorted_ids_begin + tokensNumPad)
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num] = int(
            expertId - skip_expert_num
        )
        sorted_expert_ids_begin = int(sorted_expert_ids_begin + sorted_expert_ids_num)

    num_tokens_post_pad[0] = int(sorted_ids_begin)
    num_tokens_post_pad[1] = int(topk_ids.shape[0])
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def build_routing_buffers_torch(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    tile_m: int,
):
    """Torch-native routing buffers (moe_sort_mode='torch' equivalent)."""
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = moe_sorting_torch_native(
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
        num_experts=int(experts),
        block_size=int(tile_m),
    )
    num_valid_ids = num_tokens_post_pad[:1].contiguous()
    sorted_size = int(sorted_token_ids.numel())
    blocks = int(sorted_expert_ids.numel())
    return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, blocks


# ---------------------------------------------------------------------------
# Cosine similarity for correctness ("fast but wrong" detector).
# ---------------------------------------------------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a @ b) / denom)


# ---------------------------------------------------------------------------
# Build all fp8 stage2 inputs. Mirrors run_moe_stage2(in_dtype="fp8").
# ---------------------------------------------------------------------------
def build_inputs(args):
    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))

    tokens = args.tokens
    model_dim = args.model_dim
    inter_dim = args.inter_dim
    experts = args.experts
    topk = args.topk
    tile_m = args.tile_m
    tile_k = args.tile_k

    # Tiling sanity (matches run_moe_stage2 guards).
    if model_dim % args.tile_n != 0:
        raise ValueError(f"model_dim ({model_dim}) must be divisible by tile_n ({args.tile_n}).")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim ({inter_dim}) must be divisible by tile_k ({tile_k}).")
    if (tile_m * tile_k) % 256 != 0:
        raise ValueError(f"tile_m*tile_k must be divisible by 256, got {tile_m}*{tile_k}.")
    if ((tile_m * tile_k) // 256) % 4 != 0:
        raise ValueError("(tile_m*tile_k)/256 must be divisible by 4.")

    s = 0.2
    x_fp32 = torch.rand((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.rand((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(model_dim)
    )
    w2_fp32 = torch.rand((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(inter_dim)
    )

    # Routing: deterministic torch topk + softmax.
    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = build_routing_buffers_torch(
        topk_ids=topk_ids, topk_weights=topk_weights, experts=experts, tile_m=tile_m
    )

    # Quantize (fp8 per-token/per-row).
    x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
    w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)

    # Preshuffle weights (on unpacked tensor).
    w2_shuffled = shuffle_weight(w2_q)

    # Stage2 input A2 = quantized reference stage1 output.
    # doweight applied in stage2 for the default flow (doweight_stage1=False).
    doweight_stage1 = False
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
    out1_ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )  # [tokens, topk, inter] fp32
    a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=DTYPE_FP8)

    # Flatten for kernel.
    w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
    scale_w2_flat = scale_w2.view(experts * model_dim, 1)
    w2_kernel = w2_shuffled_flat.contiguous().view(-1)

    a2_scale_1d = a2_scale.view(-1).contiguous()
    w2_scale_1d = scale_w2_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    out_s = str(args.out_dtype).strip().lower()
    if out_s in ("f16", "fp16", "half"):
        out_torch_dtype = torch.float16
    elif out_s in ("bf16", "bfloat16"):
        out_torch_dtype = torch.bfloat16
    elif out_s in ("f32", "fp32", "float"):
        out_torch_dtype = torch.float32
    else:
        raise ValueError(f"out_dtype must be 'f16','bf16','f32', got {args.out_dtype!r}")

    doweight_stage2 = not doweight_stage1

    return {
        "device": device,
        "a2_q": a2_q,
        "a2_scale": a2_scale,
        "w2_q": w2_q,
        "scale_w2": scale_w2,
        "w2_kernel": w2_kernel,
        "a2_scale_1d": a2_scale_1d,
        "w2_scale_1d": w2_scale_1d,
        "sorted_token_ids": sorted_token_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "sorted_weights_1d": sorted_weights_1d,
        "num_valid_ids": num_valid_ids,
        "blocks": blocks,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "out_torch_dtype": out_torch_dtype,
        "doweight_stage2": doweight_stage2,
    }


def make_launch(inp, args, out_buf):
    """Compile the stage2 kernel and return (launch_fn, exe)."""
    exe = compile_moe_gemm2(
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        in_dtype="fp8",
        out_dtype=args.out_dtype,
        group_size=-1,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        doweight_stage2=bool(inp["doweight_stage2"]),
        accumulate=bool(args.accumulate),
        scale_is_bf16=False,
    )

    def _args(o):
        return (
            o,
            inp["a2_q"].view(-1),
            inp["w2_kernel"].view(-1),
            inp["a2_scale_1d"],
            inp["w2_scale_1d"],
            inp["sorted_token_ids"],
            inp["sorted_expert_ids"],
            inp["sorted_weights_1d"],
            inp["num_valid_ids"],
            args.tokens,
            args.model_dim,
            args.inter_dim,
            int(inp["blocks"]),
            torch.cuda.current_stream(),
        )

    compiled = exe
    if hasattr(flyc, "compile"):
        compiled = flyc.compile(exe, *_args(out_buf))

    def launch(o):
        compiled(*_args(o))

    return launch, exe


def time_kernel(launch, out_buf, args) -> Tuple[float, float, float]:
    """Return (median_us, min_us, p90_us) over args.iters timed iterations.

    Atomic accumulate mode requires a pre-zeroed output. We zero the output
    OUTSIDE the timed region for each iteration by recording an event pair
    that brackets only the launch. The zeroing is enqueued on the same stream
    before start-event so it never overlaps the measured window.
    """
    accumulate = bool(args.accumulate)
    # Warmup.
    for _ in range(args.warmup):
        if accumulate:
            out_buf.zero_()
        launch(out_buf)
    torch.cuda.synchronize()

    times_us = []
    for _ in range(args.iters):
        if accumulate:
            out_buf.zero_()
            torch.cuda.synchronize()  # ensure zeroing done before timing window
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch(out_buf)
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)  # ms -> us

    times_us.sort()
    median = statistics.median(times_us)
    mn = times_us[0]
    p90 = times_us[min(len(times_us) - 1, int(round(0.9 * (len(times_us) - 1))))]
    return median, mn, p90


def main():
    ap = argparse.ArgumentParser(description="Standalone MoE stage2 (gemm2) fp8 benchmark")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--model_dim", type=int, default=7168)
    ap.add_argument("--inter_dim", type=int, default=256)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tile_m", type=int, default=16)
    ap.add_argument("--tile_n", type=int, default=256)
    ap.add_argument("--tile_k", type=int, default=128)
    ap.add_argument("--out_dtype", type=str, default="f16")
    acc = ap.add_mutually_exclusive_group()
    acc.add_argument("--accumulate", dest="accumulate", action="store_true", help="atomic accumulate mode (default)")
    acc.add_argument("--reduce", dest="accumulate", action="store_false", help="reduce mode (accumulate=False)")
    ap.set_defaults(accumulate=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available.", file=sys.stderr)
        sys.exit(1)

    variant = "legacy" if os.environ.get("MOE_FORCE_LEGACY_G2_FP8", "0") == "1" else "new"
    shape_id = (
        f"t{args.tokens}_md{args.model_dim}_id{args.inter_dim}_E{args.experts}_"
        f"tk{args.topk}_bm{args.tile_m}_tn{args.tile_n}_tk{args.tile_k}_"
        f"{'atomic' if args.accumulate else 'reduce'}_{args.out_dtype}"
    )

    inp = build_inputs(args)
    # Atomic mode reduces into [tokens, model_dim]; reduce mode scatters per
    # (token, topk-slot) into [tokens, topk, model_dim] and the host reduces
    # over the topk dim afterwards. Allocate the shape the kernel writes.
    if bool(args.accumulate):
        out = torch.zeros((args.tokens, args.model_dim), device=inp["device"], dtype=inp["out_torch_dtype"])
    else:
        out = torch.zeros(
            (args.tokens * args.topk, args.model_dim), device=inp["device"], dtype=inp["out_torch_dtype"]
        )

    launch, exe = make_launch(inp, args, out)

    # ---- Correctness: single clean launch into a zeroed output ----
    out.zero_()
    launch(out)
    torch.cuda.synchronize()
    ref = torch_moe_gemm2(
        inp["a2_q"],
        inp["w2_q"],
        inp["a2_scale"],
        inp["scale_w2"],
        inp["topk_ids"].to(torch.int64),
        inp["topk_weights"],
        model_dim=args.model_dim,
        doweight_stage2=bool(inp["doweight_stage2"]),
    )
    if bool(args.accumulate):
        got = out
    else:
        # Reduce mode: kernel wrote [tokens, topk, model_dim]; sum over the
        # topk slot dim to match the reference's [tokens, model_dim] reduction.
        got = out.view(args.tokens, args.topk, args.model_dim).sum(dim=1)
    cos = cosine_sim(got, ref)

    # ---- Perf: timed into the same buffer (re-zeroed each iter for atomic) ----
    median, mn, p90 = time_kernel(launch, out, args)

    print(
        f"[bench_g2] shape={shape_id} variant={variant} arch={ARCH} | "
        f"median={median:.1f}us min={mn:.1f}us p90={p90:.1f}us | cos={cos:.5f}"
    )
    print(f"SUMMARY shape={shape_id} variant={variant} median_us={median:.1f} cos={cos:.5f}")


if __name__ == "__main__":
    main()
