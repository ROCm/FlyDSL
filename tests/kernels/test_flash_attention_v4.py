#!/usr/bin/env python3
"""Flash Attention V4 (Multi-Wave MFMA) kernel test and benchmark for FlyDSL.

Tests the V4 multi-wave Flash Attention kernel against PyTorch SDPA reference.
Optionally compares performance with V3 kernels.

Usage:
    python tests/kernels/test_flash_attention_v4.py
    python tests/kernels/test_flash_attention_v4.py --seq_len 512 --head_dim 128
    python tests/kernels/test_flash_attention_v4.py --compare-v3
"""

import sys
import os
import argparse
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

import flydsl
from kernels.flash_attention_v4 import build_flash_attention_v4_module, KERNEL_NAME


def pytorch_ref_attention(q, k, v, causal=True):
    """PyTorch SDPA reference.  q/k/v: (B, S, H, D) float32 -> (B, S, H, D)."""
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def bench_gpu_us(fn, warmup=10, iters=50):
    """Benchmark a GPU function, return average microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000


def run_config(batch, seq_len, num_heads, head_dim, dtype, causal,
               warmup, iters, v3_exe=None):
    """Run one configuration. Returns dict with results."""
    device = "cuda"
    results = {}

    # V4 requires seq_len divisible by BLOCK_M=64, head_dim by 16, head_dim >= 64
    if seq_len % 64 != 0:
        results["err"] = f"seq_len ({seq_len}) must be divisible by 64"
        return results
    if head_dim % 16 != 0 or head_dim < 64:
        results["err"] = f"head_dim ({head_dim}) must be >= 64 and divisible by 16"
        return results

    try:
        m = build_flash_attention_v4_module(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str="f16",
        )
        exe = flydsl.compile(m)
    except Exception as e:
        results["err"] = f"compile: {e}"
        import traceback
        traceback.print_exc()
        return results

    B, S, H, D = batch, seq_len, num_heads, head_dim
    q_4d = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k_4d = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v_4d = torch.randn(B, S, H, D, dtype=dtype, device=device)

    q_flat = q_4d.contiguous().view(-1)
    k_flat = k_4d.contiguous().view(-1)
    v_flat = v_4d.contiguous().view(-1)
    o_flat = torch.zeros_like(q_flat)

    try:
        exe(q_flat, k_flat, v_flat, o_flat, B, S)
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback
        traceback.print_exc()
        return results

    # PyTorch reference
    ref_4d = pytorch_ref_attention(
        q_4d.float(), k_4d.float(), v_4d.float(), causal=causal
    ).to(dtype)
    ref_flat = ref_4d.contiguous().view(-1)

    # Correctness
    o_f32 = o_flat.float()
    ref_f32 = ref_flat.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(
        o_f32.view(-1, D), ref_f32.view(-1, D), dim=1
    )
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos

    atol = 1e-2
    results["passed"] = max_err < atol and min_cos > 0.99

    # Benchmark V4
    try:
        def kernel_fn():
            o_flat.zero_()
            exe(q_flat, k_flat, v_flat, o_flat, B, S)

        us = bench_gpu_us(kernel_fn, warmup=warmup, iters=iters)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        tflops = flops / (us * 1e-6) / 1e12
        results["us"] = us
        results["tflops"] = tflops
    except Exception as e:
        results["bench_err"] = str(e)

    # Benchmark V3 for comparison
    if v3_exe is not None:
        try:
            o_v3 = torch.zeros_like(q_flat)

            def v3_fn():
                o_v3.zero_()
                v3_exe(q_flat, k_flat, v_flat, o_v3, B, S)

            v3_us = bench_gpu_us(v3_fn, warmup=warmup, iters=iters)
            v3_tflops = flops / (v3_us * 1e-6) / 1e12
            results["v3_us"] = v3_us
            results["v3_tflops"] = v3_tflops
        except Exception as e:
            results["v3_bench_err"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Flash Attention V4 (Multi-Wave MFMA) FlyDSL Test/Benchmark"
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16"]
    )
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compare-v3", action="store_true",
                        help="Also benchmark V3 for comparison")
    args = parser.parse_args()

    causal = not args.no_causal
    dtype = torch.float16
    causal_str = "causal" if causal else "non-causal"

    print("=" * 120)
    print(f"FlyDSL Flash Attention V4 Multi-Wave MFMA ({causal_str}, fp16)")
    print(f"  BLOCK_M=64, BLOCK_N=16, 4 waves (256 threads), mfma_f32_16x16x16f16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 120)

    if args.seq_len or args.head_dim or args.batch:
        configs = [(
            args.batch or 1,
            args.seq_len or 128,
            args.num_heads or 8,
            args.head_dim or 128,
        )]
    else:
        configs = [
            (1, 64,   8,  128),
            (1, 128,  8,  128),
            (1, 256, 32,  128),
            (1, 512, 32,  128),
            (2, 128,  8,  128),
        ]

    # Pre-compile V3 if comparing
    v3_exes = {}
    if args.compare_v3:
        from kernels.flash_attention_v3 import build_flash_attention_v3_module
        for _, _, nh, hd in configs:
            key = (nh, hd)
            if key not in v3_exes:
                try:
                    m = build_flash_attention_v3_module(
                        num_heads=nh, head_dim=hd,
                        causal=causal, dtype_str="f16",
                    )
                    v3_exes[key] = flydsl.compile(m)
                except Exception:
                    v3_exes[key] = None

    if args.compare_v3:
        hdr = (
            f"{'Config':>38s} | {'Status':>6s} | {'MaxErr':>8s} "
            f"{'MinCos':>8s} | {'V4(us)':>10s} {'V4 TFLOPS':>9s} | "
            f"{'V3(us)':>10s} {'V3 TFLOPS':>9s} | {'Speedup':>7s}"
        )
    else:
        hdr = (
            f"{'Config':>38s} | {'Status':>6s} | {'MaxErr':>8s} "
            f"{'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
        )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    all_passed = True
    for batch, seq_len, nh, hd in configs:
        tag = f"B={batch} S={seq_len} H={nh} D={hd}"
        try:
            v3_exe = v3_exes.get((nh, hd)) if args.compare_v3 else None
            r = run_config(
                batch, seq_len, nh, hd, dtype, causal,
                warmup=args.warmup, iters=args.iters,
                v3_exe=v3_exe,
            )
            if "err" in r:
                print(f"{tag:>38s} | {'ERROR':>6s} | {r['err'][:60]}")
                all_passed = False
                continue

            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False

            v4_us = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
            v4_tf = f"{r['tflops']:>9.3f}" if "tflops" in r else "      N/A"

            if args.compare_v3 and "v3_us" in r:
                v3_us = f"{r['v3_us']:>10.1f}"
                v3_tf = f"{r['v3_tflops']:>9.3f}"
                speedup = r["v3_us"] / r["us"] if r.get("us") else 0
                sp_s = f"{speedup:>6.2f}x"
                print(
                    f"{tag:>38s} | {status:>6s} | "
                    f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | "
                    f"{v4_us} {v4_tf} | {v3_us} {v3_tf} | {sp_s}"
                )
            else:
                print(
                    f"{tag:>38s} | {status:>6s} | "
                    f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | "
                    f"{v4_us} {v4_tf}"
                )
        except Exception as e:
            print(f"{tag:>38s} | {'ERROR':>6s} | {str(e)[:60]}")
            all_passed = False

    print("=" * 120)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
