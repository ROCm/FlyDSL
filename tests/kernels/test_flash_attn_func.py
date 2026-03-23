#!/usr/bin/env python3
"""flash_attn_func kernel test and benchmark for FlyDSL.

Tests flash_attn_func against PyTorch SDPA.
"""

import os
import sys
import argparse
import hashlib
import random
from pathlib import Path
import logging

# Configure logging to show INFO level messages (required for kernel name display)
logging.basicConfig(level=logging.INFO)

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

from kernels.flash_attn_func import (
    KERNEL_NAME,
    build_flash_attn_func_module,
    select_flash_attn_func_path,
)
from tests.test_common import run_perftest

# Tensor initialization range (uniform distribution)
UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123
FLASH_ATTN_FUNC_KERNEL_CONFIG = {
    "waves_per_eu": int(os.getenv("FLYDSL_WAVES_PER_EU", "2")),
    "flat_work_group_size": 512,  # only flat_work_group_size=512 is supported
    "daz": True,
}

# (batch, seq_len, num_heads, head_dim)
DEFAULT_CONFIGS = [
    (8, 128, 64, 128),
    (8, 256, 64, 128),
    (8, 512, 64, 128),
    (1, 128, 64, 128),
    (1, 256, 64, 128),
    (1, 512, 64, 128),
    (1, 1024, 64, 128),

    (1, 2048, 64, 128),
    (1, 4096, 64, 128),
    (1, 8192, 64, 128),

    (1, 2048, 32, 128),
    (1, 4096, 32, 128),
    (1, 8192, 32, 128),

    (1, 2048, 16, 128),
    (1, 4096, 16, 128),
    (1, 8192, 16, 128),

    (1, 2048, 8, 128),
    (1, 4096, 8, 128),
    (1, 8192, 8, 128),
]


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility across all RNG sources."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pytorch_ref_attention(q, k, v, causal=True):
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 hash of a tensor's raw bytes."""
    return hashlib.md5(
        tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: list = None,
) -> dict:
    """Compare two numpy arrays and compute various difference metrics.

    Args:
        arr1: First input array (result), will be cast to float32.
        arr2: Second input array (reference), will be cast to float32.
        k: Number of top differences to report.
        thresholds: Difference magnitude buckets for histogram.

    Returns:
        Dictionary with top_k_diff, threshold_stats, nan_info, max_diff, max_diff_thr.
    """
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: arr1 {arr1.shape} vs arr2 {arr2.shape}")

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = int(np.sum(nan_mask1))
        print(f"  Warning: result contains {result['nan_info']['arr1_nan_count']} NaN values")
    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = int(np.sum(nan_mask2))
        print(f"  Warning: reference contains {result['nan_info']['arr2_nan_count']} NaN values")

    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = (diff / (1.0 + np.abs(arr2))).max()
    result["max_diff"] = float(diff.max())
    result["max_diff_thr"] = float(max_diff_thr)

    print(f"  diff.abs.max = {diff.max():.6f}")
    print(f"  diff.abs.mean = {diff.mean():.6f}")
    print(f"  max_diff_thr (rel) = {max_diff_thr:.6e}")

    # Find top k differences
    flat_diff = diff.flatten()
    actual_k = min(k, len(flat_diff))
    top_k_indices = np.argpartition(flat_diff, -actual_k)[-actual_k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    print(f"  Top-{actual_k} differences:")
    for i in range(actual_k):
        idx = tuple(dim[i] for dim in orig_indices)
        entry = {
            "value": float(diff[idx]),
            "position": idx,
            "arr1_value": float(arr1[idx]),
            "arr2_value": float(arr2[idx]),
        }
        result["top_k_diff"].append(entry)
        print(f"    [{idx}] result={arr1[idx]:.6f}, ref={arr2[idx]:.6f}, diff={diff[idx]:.6f}")

    # Compute threshold statistics
    print(f"  Threshold distribution ({total_elements} elements):")
    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i], thresholds[i + 1]
        count = int(np.sum((diff >= lower) & (diff < upper)))
        pct = 100.0 * count / total_elements
        result["threshold_stats"].append(
            {"range": f"[{lower:.0e}, {upper:.0e})", "count": count, "percentage": pct}
        )
        print(f"    [{lower:.0e}, {upper:.0e}): {count:>8d} ({pct:6.2f}%)")

    count = int(np.sum(diff >= thresholds[-1]))
    pct = 100.0 * count / total_elements
    result["threshold_stats"].append(
        {"range": f">={thresholds[-1]:.0e}", "count": count, "percentage": pct}
    )
    print(f"    >={thresholds[-1]:.0e}       : {count:>8d} ({pct:6.2f}%)")

    return result


def run_config(
    batch, seq_len, num_heads, head_dim, dtype, causal, warmup, iters,
    seed=DEFAULT_SEED, dtype_str="f16", verbose=True,
):
    device = "cuda"
    results = {}
    active_path = select_flash_attn_func_path(
        num_heads=num_heads, head_dim=head_dim, causal=causal, dtype_str=dtype_str
    )
    results["active_path"] = active_path

    if seq_len % 128 != 0:
        results["err"] = f"seq_len ({seq_len}) must be divisible by 128 for flash_attn_func"
        return results
    if head_dim % 32 != 0 or head_dim < 64:
        results["err"] = f"head_dim ({head_dim}) must be >= 64 and divisible by 32"
        return results

    try:
        exe = build_flash_attn_func_module(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
            flat_work_group_size=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("flat_work_group_size"),
            daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
        )
    except Exception as e:
        results["err"] = f"build: {e}"
        import traceback

        traceback.print_exc()
        return results

    B, S, H, D = batch, seq_len, num_heads, head_dim
    setup_seed(seed)
    q_4d = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    k_4d = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    v_4d = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)

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

    ref_4d = pytorch_ref_attention(q_4d.float(), k_4d.float(), v_4d.float(), causal=causal).to(dtype)
    ref_flat = ref_4d.contiguous().view(-1)

    o_f32 = o_flat.float()
    ref_f32 = ref_flat.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.view(-1, D), ref_f32.view(-1, D), dim=1)
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < 1e-2 and min_cos > 0.99

    if verbose:
        tag = f"B={B} S={S} H={H} D={D}"
        print(f"  [{tag}] active_path = {active_path}")
        result_md5 = compute_md5(o_flat)
        ref_md5 = compute_md5(ref_flat)
        print(f"  [{tag}] result_md5 = {result_md5}")
        print(f"  [{tag}] ref_md5    = {ref_md5}")
        if result_md5 == ref_md5:
            print(f"  [{tag}] MD5 match: EXACT (bit-identical)")
        else:
            print(f"  [{tag}] MD5 match: DIFFER (not bit-identical)")

        print(f"  [{tag}] --- compare_arrays ---")
        compare_arrays(
            o_flat.to(torch.float32).detach().cpu().numpy(),
            ref_flat.to(torch.float32).detach().cpu().numpy(),
        )

    try:
        def kernel_fn():
            exe(q_flat, k_flat, v_flat, o_flat, B, S)

        _, us = run_perftest(kernel_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        tflops = flops / (us * 1e-6) / 1e12
        results["us"] = us
        results["tflops"] = tflops
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_aiter_bench(
    batch, seq_len, nheads, head_dim, dtype, causal,
    warmup, iters, seed=DEFAULT_SEED, backend="ck",
):
    """Run CK or ASM kernel via aiter and return {tflops, max_err, us}."""
    try:
        import aiter
    except ImportError:
        return {"err": "aiter not installed"}

    if backend == "asm" and dtype != torch.bfloat16:
        return {"skip": True}

    os.environ["AITER_BF16_FWD_BACKEND"] = backend
    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    try:
        out = aiter.flash_attn_func(
            q, k, v, 0.0, None, causal, (-1, -1),
            None, None, True,
            return_lse=True, return_attn_probs=True, how_v3_bf16_cvt=2,
        )[0]
        torch.cuda.synchronize()
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal).to(dtype)
    max_err = (out.float() - ref.float()).abs().max().item()
    results["max_err"] = max_err

    try:
        def bench_fn():
            aiter.flash_attn_func(
                q, k, v, 0.0, None, causal, (-1, -1),
                None, None, True,
                return_lse=True, return_attn_probs=True, how_v3_bf16_cvt=2,
            )

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def _fmt_result(r):
    """Format: 'Time(us) TFLOPS MaxErr'."""
    if r.get("skip"):
        return f"{'--':>10s} {'--':>8s} {'--':>8s}"
    if "err" in r:
        return f"{'--':>10s} {'ERR':>8s} {'--':>8s}"
    us = f"{r['us']:>10.1f}" if "us" in r else f"{'N/A':>10s}"
    tf = f"{r['tflops']:>8.3f}" if "tflops" in r else f"{'N/A':>8s}"
    err = f"{r['max_err']:>8.2e}" if "max_err" in r else f"{'N/A':>8s}"
    return f"{us} {tf} {err}"


def _fmt_cmp(fly_r, other_r):
    """Format FlyDSL vs other: 'TFLOPS% MaxErr-ratio'."""
    if other_r.get("skip") or "err" in other_r or "err" in fly_r:
        return f"{'--':>7s} {'--':>6s}"
    fly_tf = fly_r.get("tflops")
    oth_tf = other_r.get("tflops")
    fly_err = fly_r.get("max_err")
    oth_err = other_r.get("max_err")
    if fly_tf and oth_tf and oth_tf > 0:
        pct = f"{fly_tf / oth_tf * 100:>6.1f}%"
    else:
        pct = f"{'N/A':>7s}"
    if fly_err is not None and oth_err is not None and oth_err > 0:
        ratio = f"{fly_err / oth_err:>5.2f}x"
    else:
        ratio = f"{'N/A':>6s}"
    return f"{pct} {ratio}"


def main():
    parser = argparse.ArgumentParser(description="flash_attn_func FlyDSL Test/Benchmark")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument("--causal", action="store_true", dest="causal")
    causal_group.add_argument("--no-causal", action="store_false", dest="causal")
    parser.set_defaults(causal=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["fp16", "bf16"],
        help="Data type: fp16 or bf16 (default: both)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare FlyDSL vs CK vs ASM performance (requires aiter)",
    )
    args = parser.parse_args()

    dtype_map = {"fp16": (torch.float16, "f16"), "bf16": (torch.bfloat16, "bf16")}
    dtypes_to_test = [args.dtype] if args.dtype else ["bf16", "fp16"]
    causals_to_test = [args.causal] if args.causal is not None else [True, False]

    if args.batch or args.seq_len or args.num_heads or args.head_dim:
        configs = [(args.batch or 1, args.seq_len or 128, args.num_heads or 8, args.head_dim or 128)]
    else:
        configs = DEFAULT_CONFIGS

    causal_desc = {True: "causal", False: "non-causal", None: "causal+non-causal"}[args.causal]
    dtype_desc = args.dtype or "bf16+fp16"

    if args.compare:
        # ---- Comparison mode: FlyDSL vs CK vs ASM ----
        print("=" * 130)
        print(f"FlyDSL vs CK vs ASM  ({causal_desc}, {dtype_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  FlyDSL opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        print(f"  CK: bf16+fp16, ASM: bf16 only")
        print("=" * 130)
        print("Running benchmarks ...")

        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, hd in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    tag = f"B={batch} S={seq_len} H={nh} D={hd} {dtype_key} {causal_tag}"
                    print(f"  {tag} ...", flush=True)

                    fly_r = run_config(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, dtype_str=dtype_str, verbose=False,
                    )
                    ck_r = run_aiter_bench(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, backend="ck",
                    )
                    asm_r = run_aiter_bench(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, backend="asm",
                    )
                    rows.append((tag, fly_r, ck_r, asm_r))

        col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s}"
        cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
        hdr1 = (
            f"{'Config':>45s} | {'FlyDSL':^28s} | {'CK':^28s} | {'ASM':^28s}"
            f" | {'Fly/CK':^14s} | {'Fly/ASM':^14s}"
        )
        hdr2 = (
            f"{'':>45s} | {col} | {col} | {col}"
            f" | {cmp_col} | {cmp_col}"
        )
        sep = "-" * len(hdr2)
        print(f"\n{hdr1}")
        print(hdr2)
        print(sep)
        for tag, fly_r, ck_r, asm_r in rows:
            print(
                f"{tag:>45s} | {_fmt_result(fly_r)} | "
                f"{_fmt_result(ck_r)} | {_fmt_result(asm_r)}"
                f" | {_fmt_cmp(fly_r, ck_r)} | {_fmt_cmp(fly_r, asm_r)}"
            )
        print("=" * len(hdr2))

    else:
        # ---- Normal FlyDSL test mode ----
        print("=" * 130)
        print(f"FlyDSL flash_attn_func ({causal_desc}, {dtype_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Kernel opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        print("=" * 130)

        hdr = (
            f"{'Config/Path':>66s} | {'Status':>6s} | {'MaxErr':>8s} "
            f"{'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
        )
        print(f"\n{hdr}")
        print("-" * len(hdr))

        all_passed = True
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, hd in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    tag = f"B={batch} S={seq_len} H={nh} D={hd} {dtype_key} {causal_tag}"
                    try:
                        r = run_config(
                            batch, seq_len, nh, hd, dtype, causal,
                            warmup=args.warmup, iters=args.iters,
                            seed=args.seed, dtype_str=dtype_str,
                        )
                        if "err" in r:
                            cfg_path = f"{tag} / {r.get('active_path', 'unknown')}"
                            print(f"{cfg_path:>66s} | {'ERROR':>6s} | {r['err'][:60]}")
                            all_passed = False
                            continue

                        status = "PASS" if r["passed"] else "FAIL"
                        if not r["passed"]:
                            all_passed = False
                        cfg_path = f"{tag} / {r.get('active_path', 'unknown')}"

                        us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
                        tf_s = f"{r['tflops']:>9.3f}" if "tflops" in r else "      N/A"
                        print(
                            f"{cfg_path:>66s} | {status:>6s} | "
                            f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | "
                            f"{us_s} {tf_s}"
                        )
                    except Exception as e:
                        print(f"{tag:>66s} | {'ERROR':>6s} | {str(e)[:60]}")
                        all_passed = False

        print("=" * 130)
        if all_passed:
            print("All tests PASSED")
        else:
            print("Some tests FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
