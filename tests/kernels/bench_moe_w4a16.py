"""
Benchmark: FlyDSL vs sglang MoE kernels (W4A16 group-quant and bf16)

Usage:
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16 --bf16
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16 --group-size 128

Notes:
  - FlyDSL W4A16 only supports group_size=32 (int4 preshuffle constraint).
    When --group-size != 32, FlyDSL falls back to per-row scale (group_size=-1).
  - sglang uses Triton kernels; group_size is configurable (16, 32, 128, ...).
"""
import argparse
import math
import os
import sys
import time
import gc

import torch

# ---------------------------------------------------------------------------
# sglang mock setup (needed when running outside sglang server)
# ---------------------------------------------------------------------------
from sglang.srt import server_args
from unittest.mock import MagicMock

mock_args = MagicMock()
mock_args.enable_deterministic_inference = False
server_args._global_server_args = mock_args

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import inplace_fused_experts

# ---------------------------------------------------------------------------
# FlyDSL imports
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.kernels.test_moe_gemm import run_moe_stage1, run_moe_stage2


# ---------------------------------------------------------------------------
# sglang benchmark
# ---------------------------------------------------------------------------
def benchmark_sglang(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    *,
    w4a16: bool = True,
    group_size: int = 32,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> float:
    """Benchmark sglang Triton fused MoE kernel.

    Returns: average latency in milliseconds.
    """
    device = "cuda"
    K1 = model_dim          # stage1 K (activation hidden dim)
    N1 = 2 * inter_dim      # stage1 N (gate + up)
    K2 = inter_dim           # stage2 K (intermediate dim)
    N2 = model_dim           # stage2 N (output hidden dim)

    torch.manual_seed(42)
    topk_weights = torch.ones(tokens, topk, dtype=torch.float32, device=device) / topk
    topk_ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32, device=device)

    if w4a16:
        # Packed int4 weights: 2 int4 values per byte -> K//2 bytes
        # sglang shape convention: (E, N, K_packed) where K_packed = K_original // 2
        w1 = torch.randint(0, 256, (experts, N1, K1 // 2), dtype=torch.uint8, device=device).to(torch.int8)
        w2 = torch.randint(0, 256, (experts, N2, K2 // 2), dtype=torch.uint8, device=device).to(torch.int8)
        # Per-group scale: (E, K_original // group_size, N) -- Opt0 cache-friendly layout
        # NOTE: group_size is in terms of original (unpacked) elements.
        w1_scale = torch.ones(experts, K1 // group_size, N1, dtype=torch.float32, device=device)
        w2_scale = torch.ones(experts, K2 // group_size, N2, dtype=torch.float32, device=device)
        block_shape = [0, group_size]
    else:
        # bf16 weights (no quantization)
        w1 = torch.randn(experts, N1, K1, dtype=torch.bfloat16, device=device) * 0.2
        w2 = torch.randn(experts, N2, K2, dtype=torch.bfloat16, device=device) * 0.2
        w1_scale, w2_scale = None, None
        block_shape = None

    # Warmup
    for _ in range(num_warmup):
        hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device) * 0.2
        inplace_fused_experts(
            hidden, w1, w2, topk_weights, topk_ids,
            use_int4_w4a16=w4a16,
            w1_scale=w1_scale, w2_scale=w2_scale,
            block_shape=block_shape,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        hidden = torch.randn(tokens, model_dim, dtype=torch.bfloat16, device=device) * 0.2
        inplace_fused_experts(
            hidden, w1, w2, topk_weights, topk_ids,
            use_int4_w4a16=w4a16,
            w1_scale=w1_scale, w2_scale=w2_scale,
            block_shape=block_shape,
        )
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) / num_iters * 1000
    return elapsed_ms


# ---------------------------------------------------------------------------
# FlyDSL benchmark
# ---------------------------------------------------------------------------
def benchmark_flydsl(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    *,
    w4a16: bool = True,
    group_size: int = 32,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> float:
    """Benchmark FlyDSL MoE stage1 + stage2 kernels.

    Returns: average latency in milliseconds (stage1 + stage2).
    """
    device = "cuda"
    in_dtype = "int4_bf16" if w4a16 else "fp16"

    # FlyDSL only supports group_size=32 for W4A16; fall back to per-row otherwise.
    if w4a16 and group_size == 32:
        flydsl_group_size = 32
    else:
        flydsl_group_size = -1  # per-row scale

    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device=device, dtype=torch.float32) * 0.2
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, device=device, dtype=torch.float32) * 0.2
    w2 = torch.randn(experts, model_dim, inter_dim, device=device, dtype=torch.float32) * 0.2
    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=torch.int64)
    topk_weights = torch.ones(tokens, topk, device=device, dtype=torch.float32) / topk

    out1, us1 = run_moe_stage1(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=128, tile_k=128,
        doweight_stage1=False,
        in_dtype=in_dtype,
        group_size=flydsl_group_size,
        num_iters=num_iters, num_warmup=num_warmup,
        skip_ref=True,
        x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        return_outputs=True,
        compare_aiter_ck=False,
    )

    # Prepare stage2 input
    if w4a16:
        a2 = out1.to(torch.bfloat16)
    else:
        a2 = out1.to(torch.float16)

    _, us2 = run_moe_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=256, tile_k=128,
        doweight_stage1=False,
        in_dtype=in_dtype,
        group_size=flydsl_group_size,
        num_iters=num_iters, num_warmup=num_warmup,
        skip_ref=True,
        w2_fp32_in=w2,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        a2_fp8_in=a2, a2_scale_in=None,
        return_outputs=True,
        compare_aiter_ck=False,
    )

    return (us1 + us2) / 1000  # convert us to ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL vs sglang MoE kernels (W4A16 group-quant and bf16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bf16", action="store_true", help="Benchmark bf16 instead of W4A16")
    parser.add_argument("--tokens", type=int, default=5231)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=512)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=32,
                        help="Group size for W4A16 quantization (sglang: any; FlyDSL: 32 only)")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    tokens = args.tokens
    model_dim = args.model_dim
    inter_dim = args.inter_dim
    experts = args.experts
    topk = args.topk
    group_size = args.group_size
    w4a16 = not args.bf16
    dtype_name = "W4A16" if w4a16 else "bf16"

    M = tokens * topk
    # Stage1: M x (2*inter_dim) x model_dim  +  Stage2: M x model_dim x inter_dim
    total_flops = M * (2 * inter_dim) * model_dim * 2 + M * model_dim * inter_dim * 2

    print("=" * 70)
    print(f"    FlyDSL vs sglang {dtype_name} MoE Benchmark (group_size={group_size})")
    print("=" * 70)
    print(f"\nParams: tokens={tokens}, model_dim={model_dim}, inter_dim={inter_dim}")
    print(f"        experts={experts}, topk={topk}, M_effective={M}")
    if w4a16:
        flydsl_gs = 32 if group_size == 32 else -1
        print(f"        sglang group_size={group_size}, FlyDSL group_size={flydsl_gs}")
    print(f"Total FLOPs: {total_flops / 1e12:.4f} TFLOPs\n")

    # --- FlyDSL ---
    print(f"Running FlyDSL {dtype_name}...")
    flydsl_ms = benchmark_flydsl(
        tokens, model_dim, inter_dim, experts, topk,
        w4a16=w4a16, group_size=group_size,
        num_warmup=args.num_warmup, num_iters=args.num_iters,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # --- sglang ---
    print(f"Running sglang {dtype_name}...")
    sglang_ms = benchmark_sglang(
        tokens, model_dim, inter_dim, experts, topk,
        w4a16=w4a16, group_size=group_size,
        num_warmup=args.num_warmup, num_iters=args.num_iters,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # --- Results ---
    print("\n" + "=" * 70)
    print("                         RESULTS")
    print("=" * 70)
    flydsl_tflops = total_flops / (flydsl_ms / 1000) / 1e12
    sglang_tflops = total_flops / (sglang_ms / 1000) / 1e12
    print(f"\n{'Kernel':<20s} {'Latency':>10s} {'TFLOPS':>10s}")
    print("-" * 42)
    print(f"{'FlyDSL ' + dtype_name:<20s} {flydsl_ms:>8.2f} ms {flydsl_tflops:>8.2f}")
    print(f"{'sglang ' + dtype_name:<20s} {sglang_ms:>8.2f} ms {sglang_tflops:>8.2f}")

    ratio = sglang_ms / flydsl_ms
    print()
    if ratio > 1:
        print(f">>> FlyDSL is {ratio:.2f}x faster <<<")
    else:
        print(f">>> sglang is {1 / ratio:.2f}x faster <<<")
    print()


if __name__ == "__main__":
    main()
