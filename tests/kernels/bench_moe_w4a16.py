"""
Benchmark: FlyDSL MoE kernels (W4A16 group-quant, bf16, hybrid)

Usage:
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16 --bf16
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16 --hybrid-w2-bf16
  HIP_VISIBLE_DEVICES=4 python -m tests.kernels.bench_moe_w4a16 --group-size 128

Notes:
  - FlyDSL W4A16 only supports group_size=32 (int4 preshuffle constraint).
    When --group-size != 32, FlyDSL falls back to per-row scale (group_size=-1).
  - hybrid-w2-bf16: stage1 W4A16, stage2 bf16. Compares against pure W4A16.
"""
import argparse
import os
import sys
import time
import gc

import torch

# ---------------------------------------------------------------------------
# FlyDSL imports
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.kernels.test_moe_gemm import run_moe_stage1, run_moe_stage2


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
    in_dtype: str = "int4_bf16",
    group_size: int = 32,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> float:
    """Benchmark FlyDSL MoE stage1 + stage2 kernels.

    Returns: average latency in milliseconds (stage1 + stage2).
    """
    device = "cuda"

    # FlyDSL only supports group_size=32 for W4A16; fall back to per-row otherwise.
    if in_dtype == "int4_bf16" and group_size == 32:
        flydsl_group_size = 32
    else:
        flydsl_group_size = -1  # per-row scale (also used for bf16)

    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device=device, dtype=torch.float32) * 0.2
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, device=device, dtype=torch.float32) * 0.2
    w2 = torch.randn(experts, model_dim, inter_dim, device=device, dtype=torch.float32) * 0.2
    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=torch.int64)
    topk_weights = torch.ones(tokens, topk, device=device, dtype=torch.float32) / topk

    out1, us1 = run_moe_stage1(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=64, tile_k=128,
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

    # Prepare stage2 input: bf16 activations for bf16-based dtypes, fp16 otherwise.
    if in_dtype in ("int4_bf16", "bf16"):
        a2 = out1.to(torch.bfloat16)
    else:
        a2 = out1.to(torch.float16)

    # Free stage1 output to reclaim memory before stage2 allocation.
    del out1
    torch.cuda.empty_cache()

    _, us2 = run_moe_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=256, tile_k=128,
        doweight_stage1=False,
        in_dtype=in_dtype,
        group_size=flydsl_group_size,
        num_iters=num_iters, num_warmup=num_warmup,
        skip_ref=True,
        x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        a2_fp8_in=a2, a2_scale_in=None,
        return_outputs=True,
        compare_aiter_ck=False,
    )

    return (us1 + us2) / 1000  # convert us to ms


def benchmark_flydsl_hybrid_w2_bf16(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    *,
    group_size: int = 32,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> tuple[float, float, float]:
    """Benchmark FlyDSL hybrid: stage1 W4A16 + stage2 bf16.

    Returns: (stage1_ms, stage2_ms, total_ms).
    """
    device = "cuda"
    flydsl_group_size = 32 if group_size == 32 else -1

    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device=device, dtype=torch.float32) * 0.2
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, device=device, dtype=torch.float32) * 0.2
    w2 = torch.randn(experts, model_dim, inter_dim, device=device, dtype=torch.float32) * 0.2
    topk_ids = torch.randint(0, experts, (tokens, topk), device=device, dtype=torch.int64)
    topk_weights = torch.ones(tokens, topk, device=device, dtype=torch.float32) / topk

    # Stage1: W4A16
    out1, us1 = run_moe_stage1(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=64, tile_k=128,
        doweight_stage1=False,
        in_dtype="int4_bf16",
        group_size=flydsl_group_size,
        num_iters=num_iters, num_warmup=num_warmup,
        skip_ref=True,
        x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        return_outputs=True,
        compare_aiter_ck=False,
    )

    a2 = out1.to(torch.bfloat16)
    del out1
    torch.cuda.empty_cache()

    # Stage2: bf16
    _, us2 = run_moe_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=64, tile_n=256, tile_k=128,
        doweight_stage1=False,
        in_dtype="bf16",
        group_size=-1,
        num_iters=num_iters, num_warmup=num_warmup,
        skip_ref=True,
        x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        a2_fp8_in=a2, a2_scale_in=None,
        return_outputs=True,
        compare_aiter_ck=False,
    )

    s1_ms = us1 / 1000
    s2_ms = us2 / 1000
    return s1_ms, s2_ms, s1_ms + s2_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlyDSL MoE kernels (W4A16 group-quant, bf16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument("--bf16", action="store_true", help="Benchmark pure bf16 instead of W4A16")
    dtype_group.add_argument("--hybrid-w2-bf16", action="store_true",
                             help="Benchmark hybrid: stage1 W4A16, stage2 bf16 (compare vs pure W4A16)")
    parser.add_argument("--tokens", type=int, default=5231)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=512)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=32,
                        help="Group size for W4A16 quantization (FlyDSL: 32 only)")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    tokens = args.tokens
    model_dim = args.model_dim
    inter_dim = args.inter_dim
    experts = args.experts
    topk = args.topk
    group_size = args.group_size

    # Determine in_dtype and display name.
    is_hybrid = args.hybrid_w2_bf16
    if args.bf16:
        flydsl_in_dtype = "bf16"
        dtype_name = "bf16"
    elif is_hybrid:
        flydsl_in_dtype = "int4_bf16"  # used for pure W4A16 baseline
        dtype_name = "hybrid(s1:W4A16+s2:bf16)"
    else:
        flydsl_in_dtype = "int4_bf16"
        dtype_name = "W4A16"

    M = tokens * topk
    # Stage1: M x (2*inter_dim) x model_dim  +  Stage2: M x model_dim x inter_dim
    total_flops = M * (2 * inter_dim) * model_dim * 2 + M * model_dim * inter_dim * 2

    print("=" * 70)
    print(f"    FlyDSL MoE Benchmark: {dtype_name} (group_size={group_size})")
    print("=" * 70)
    print(f"\nParams: tokens={tokens}, model_dim={model_dim}, inter_dim={inter_dim}")
    print(f"        experts={experts}, topk={topk}, M_effective={M}")
    print(f"Total FLOPs: {total_flops / 1e12:.4f} TFLOPs\n")

    if is_hybrid:
        # --- Hybrid: stage1 W4A16 + stage2 bf16 ---
        print("Running FlyDSL hybrid (stage1=W4A16, stage2=bf16)...")
        h_s1, h_s2, hybrid_ms = benchmark_flydsl_hybrid_w2_bf16(
            tokens, model_dim, inter_dim, experts, topk,
            group_size=group_size,
            num_warmup=args.num_warmup, num_iters=args.num_iters,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # --- Baseline: pure W4A16 ---
        print("Running FlyDSL pure W4A16 (baseline)...")
        w4a16_ms = benchmark_flydsl(
            tokens, model_dim, inter_dim, experts, topk,
            in_dtype="int4_bf16", group_size=group_size,
            num_warmup=args.num_warmup, num_iters=args.num_iters,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # --- Results ---
        print("\n" + "=" * 70)
        print("                         RESULTS")
        print("=" * 70)
        hybrid_tflops = total_flops / (hybrid_ms / 1000) / 1e12
        w4a16_tflops = total_flops / (w4a16_ms / 1000) / 1e12
        print(f"\n{'Kernel':<30s} {'Stage1':>10s} {'Stage2':>10s} {'Total':>10s} {'TFLOPS':>10s}")
        print("-" * 72)
        print(f"{'FlyDSL hybrid(W4A16+bf16)':<30s} {h_s1:>8.2f} ms {h_s2:>8.2f} ms {hybrid_ms:>8.2f} ms {hybrid_tflops:>8.2f}")
        print(f"{'FlyDSL pure W4A16':<30s} {'—':>10s} {'—':>10s} {w4a16_ms:>8.2f} ms {w4a16_tflops:>8.2f}")

        diff_ms = hybrid_ms - w4a16_ms
        pct = diff_ms / w4a16_ms * 100
        print()
        if diff_ms > 0:
            print(f">>> Hybrid is {diff_ms:.2f} ms slower ({pct:+.1f}%) than pure W4A16 <<<")
        else:
            print(f">>> Hybrid is {-diff_ms:.2f} ms faster ({pct:+.1f}%) than pure W4A16 <<<")
        print()
    else:
        # --- FlyDSL ---
        print(f"Running FlyDSL {dtype_name}...")
        flydsl_ms = benchmark_flydsl(
            tokens, model_dim, inter_dim, experts, topk,
            in_dtype=flydsl_in_dtype, group_size=group_size,
            num_warmup=args.num_warmup, num_iters=args.num_iters,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # --- Results ---
        print("\n" + "=" * 70)
        print("                         RESULTS")
        print("=" * 70)
        flydsl_tflops = total_flops / (flydsl_ms / 1000) / 1e12
        print(f"\n{'Kernel':<20s} {'Latency':>10s} {'TFLOPS':>10s}")
        print("-" * 42)
        print(f"{'FlyDSL ' + dtype_name:<20s} {flydsl_ms:>8.2f} ms {flydsl_tflops:>8.2f}")
        print()


if __name__ == "__main__":
    main()
