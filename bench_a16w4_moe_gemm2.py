#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Standalone benchmark harness for compile_a16w4_moe_gemm2.
# NOT committed — profiling only.
#
# Two-phase profiling workflow (avoids aiter CK JIT module conflict with rocprofv3):
#
#   # 1. Build inputs (imports aiter; no profiler attached):
#   PYTHONPATH=. python bench_a16w4_moe_gemm2.py --prepare --tokens 32
#
#   # 2. Profile (loads saved tensors; imports only flydsl+torch):
#   PYTHONPATH=. python bench_a16w4_moe_gemm2.py --profile --tokens 32
#
# Full sweep (no profiler):
#   PYTHONPATH=. python bench_a16w4_moe_gemm2.py
#   PYTHONPATH=. python bench_a16w4_moe_gemm2.py --tokens 64

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
for _p in [os.path.join(_REPO_ROOT, "build", "python_packages"), _REPO_ROOT]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from flydsl.runtime.device import get_rocm_arch
from kernels.mixed_moe_gemm_2stage import compile_a16w4_moe_gemm2

ARCH = get_rocm_arch()

# GPT-OSS / typical MoE config — matches bench_s2_flydsl.py
MODEL_DIM = 3072
INTER_DIM = 3072
MODEL_DIM_PAD = 192
INTER_DIM_PAD = 192
EXPERTS = 128
TOPK = 4
TILE_M = 16
TILE_N = 128
TILE_K = 256

TOKEN_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]

_STATE_PATH = "/tmp/bench_a16w4_inputs_{tokens}.pt"


def _state_path(tokens: int) -> str:
    return _STATE_PATH.format(tokens=tokens)


def prepare_inputs(tokens: int, seed: int = 0):
    """Build inputs using aiter and save to disk. Run this BEFORE --profile."""
    from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
    from aiter.fused_moe import moe_sorting
    from aiter import QuantType
    import aiter

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tile_m = TILE_M

    vm = MODEL_DIM - MODEL_DIM_PAD
    vi = INTER_DIM - INTER_DIM_PAD

    a2_bf16 = torch.randn((tokens * TOPK, INTER_DIM), device=device, dtype=dtype) * 0.2
    a2_bf16[:, vi:] = 0.0

    w2_fp32 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), device=device, dtype=torch.float32) / 10
    w2_fp32[:, vm:, :] = 0.0
    w2_fp32[:, :, vi:] = 0.0

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w2_qt_raw, w2_scale = torch_quant(w2_fp32.to(dtype), quant_dtype=aiter.dtypes.fp4x2)
    w2_qt_raw = w2_qt_raw.view(EXPERTS, MODEL_DIM, INTER_DIM // 2)

    w2_qt_shuf = shuffle_weight_a16w4(w2_qt_raw, 16, False).view(torch.uint8).contiguous()
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, EXPERTS, False).view(torch.uint8).contiguous()

    topk_ids = torch.zeros((tokens, TOPK), device=device, dtype=torch.int32)
    s, e = 0, TOPK
    for t in range(tokens):
        topk_ids[t, :] = torch.arange(s, e, device=device) % EXPERTS
        s = e % EXPERTS
        e = s + TOPK
    topk_weights = torch.ones((tokens, TOPK), device=device, dtype=torch.float32) / TOPK

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, EXPERTS, MODEL_DIM, dtype, tile_m
    )
    needed = sorted_expert_ids.shape[0] * tile_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full((needed - sorted_ids.shape[0],), tokens, dtype=sorted_ids.dtype, device=device)
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat([
            sorted_weights,
            torch.zeros(pad.shape[0], dtype=sorted_weights.dtype, device=device),
        ])

    out = torch.zeros((tokens, MODEL_DIM), device=device, dtype=dtype)
    no_bias = torch.empty(0, device=device, dtype=dtype)

    state = dict(
        out=out.cpu(),
        a2_bf16=a2_bf16.cpu(),
        w2_qt_shuf=w2_qt_shuf.cpu(),
        w2_scale_shuf=w2_scale_shuf.cpu(),
        sorted_ids=sorted_ids.cpu(),
        sorted_expert_ids=sorted_expert_ids.cpu(),
        sorted_weights=sorted_weights.cpu(),
        num_valid_ids=num_valid_ids.cpu(),
        no_bias=no_bias.cpu(),
        tokens=tokens,
        sorted_size=sorted_ids.shape[0],
    )
    path = _state_path(tokens)
    torch.save(state, path)
    print(f"Saved inputs for tokens={tokens} → {path}")
    print(f"  sorted_ids: {sorted_ids.shape}, sorted_expert_ids: {sorted_expert_ids.shape}")


def _load_inputs(tokens: int):
    """Load pre-built inputs from disk. No aiter imports."""
    path = _state_path(tokens)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved inputs for tokens={tokens}. Run with --prepare --tokens {tokens} first."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    return (
        state["out"].to(device),
        state["a2_bf16"].to(device),
        state["w2_qt_shuf"].to(device),
        state["w2_scale_shuf"].to(device),
        state["sorted_ids"].to(device),
        state["sorted_expert_ids"].to(device),
        state["sorted_weights"].to(device),
        state["num_valid_ids"].to(device),
        state["no_bias"].to(device),
        int(state["tokens"]),
        MODEL_DIM,
        INTER_DIM,
        int(state["sorted_size"]),
    )


def _compile():
    return compile_a16w4_moe_gemm2(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=TOPK,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        doweight_stage2=True,
        out_dtype="bf16",
        accumulate=True,
        model_dim_pad=MODEL_DIM_PAD,
        inter_dim_pad=INTER_DIM_PAD,
    )


def _run_shape(exe, inputs, *, num_warmup=3, num_iters=20):
    stream = torch.cuda.current_stream()
    for _ in range(num_warmup):
        exe(*inputs, stream)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        exe(*inputs, stream)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_iters * 1e6


def benchmark_sweep(num_warmup: int = 3, num_iters: int = 20):
    print(f"A16W4 MoE GEMM Stage2 — {ARCH}")
    print(f"Config: E={EXPERTS}, topk={TOPK}, model_dim={MODEL_DIM}(pad={MODEL_DIM_PAD}), "
          f"inter_dim={INTER_DIM}(pad={INTER_DIM_PAD})")
    print(f"Tiles: tile_m={TILE_M}, tile_n={TILE_N}, tile_k={TILE_K}")
    print()

    exe = _compile()

    header = f"{'tokens':>8}  {'latency_us':>12}  {'TFLOPS':>10}"
    print(header)
    print("-" * len(header))

    for tokens in TOKEN_SWEEP:
        try:
            inputs = _load_inputs(tokens)
        except FileNotFoundError:
            # Fall back to building inline (needs aiter)
            from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
            from aiter.fused_moe import moe_sorting
            from aiter import QuantType
            import aiter
            prepare_inputs(tokens)
            inputs = _load_inputs(tokens)

        lat_us = _run_shape(exe, inputs, num_warmup=num_warmup, num_iters=num_iters)
        flops = 2 * tokens * TOPK * INTER_DIM * MODEL_DIM
        tflops = flops / (lat_us * 1e-6) / 1e12
        print(f"{tokens:>8}  {lat_us:>12.2f}  {tflops:>10.3f}")


def benchmark_profile(tokens: int, num_warmup: int = 5):
    """Single instrumented call after warmup. No aiter imports — loads tensors from disk."""
    inputs = _load_inputs(tokens)   # move before _compile so HIP context is up first
    exe = _compile()
    stream = torch.cuda.current_stream()

    for _ in range(num_warmup):
        exe(*inputs, stream)
    torch.cuda.synchronize()

    exe(*inputs, stream)
    torch.cuda.synchronize()
    time.sleep(0.5)  # let rocprofv3 flush counter data before process exits


def main():
    parser = argparse.ArgumentParser(description="Benchmark compile_a16w4_moe_gemm2")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--prepare", action="store_true", help="Build + save inputs (needs aiter). Run before --profile.")
    parser.add_argument("--profile", action="store_true", help="Profile mode: load saved inputs, single call. Run under rocprofv3.")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_iters", type=int, default=20)
    args = parser.parse_args()

    if "gfx95" not in ARCH:
        print(f"WARNING: A16W4 requires gfx950+, current arch = {ARCH}")

    if args.prepare:
        token_list = [args.tokens] if args.tokens else TOKEN_SWEEP
        for t in token_list:
            prepare_inputs(t)
        return

    if args.profile:
        tokens = args.tokens or 64
        print(f"Profile mode: tokens={tokens}")
        benchmark_profile(tokens, num_warmup=args.num_warmup)
        return

    if args.tokens is not None:
        exe = _compile()
        inputs = _load_inputs(args.tokens)
        lat_us = _run_shape(exe, inputs, num_warmup=args.num_warmup, num_iters=args.num_iters)
        flops = 2 * args.tokens * TOPK * INTER_DIM * MODEL_DIM
        tflops = flops / (lat_us * 1e-6) / 1e12
        print(f"tokens={args.tokens}  latency={lat_us:.2f} us  TFLOPS={tflops:.3f}")
        return

    benchmark_sweep(num_warmup=args.num_warmup, num_iters=args.num_iters)


if __name__ == "__main__":
    main()
