#!/usr/bin/env python3
"""flash_attn_func kernel test and benchmark for FlyDSL.

Tests flash_attn_func against PyTorch SDPA.
"""

import os
import sys
import argparse
import csv
import hashlib
import math
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
    (4, 8192, 64, 128),

    (1, 2048, 32, 128),
    (1, 4096, 32, 128),
    (1, 8192, 32, 128),
    (8, 8192, 32, 128),

    (1, 2048, 16, 128),
    (1, 4096, 16, 128),
    (1, 8192, 16, 128),
    (16, 8192, 16, 128),

    (1, 2048, 8, 128),
    (1, 4096, 8, 128),
    (1, 8192, 8, 128),
    (32, 8192, 8, 128),
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
    """Run true CK or true ASM kernel via aiter and return {tflops, max_err, us}."""
    try:
        import aiter
    except ImportError:
        return {"err": "aiter not installed"}

    if backend == "asm" and dtype != torch.bfloat16:
        return {"skip": True}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "ck":
        def aiter_forward():
            return aiter.mha_fwd(
                q,
                k,
                v,
                0.0,
                softmax_scale,
                causal,
                -1,
                -1,
                0,
                True,
                False,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    elif backend == "asm":
        def aiter_forward():
            return aiter.fmha_v3_fwd(
                q,
                k,
                v,
                0.0,
                softmax_scale,
                causal,
                -1,
                -1,
                True,
                False,
                2,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    else:
        return {"err": f"unsupported backend: {backend}"}

    try:
        out = aiter_forward()[0]
        torch.cuda.synchronize()
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal).to(dtype)
    max_err = (out.float() - ref.float()).abs().max().item()
    results["max_err"] = max_err

    try:
        def bench_fn():
            aiter_forward()

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
    tf = f"{r['tflops']:>8.1f}" if "tflops" in r else f"{'N/A':>8s}"
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


def _gpu_short_name():
    """Extract short GPU name, e.g. 'AMD Instinct MI308X' -> 'MI308X'."""
    return torch.cuda.get_device_name(0).split()[-1]


def _csv_val(r, key):
    """Extract a value from result dict for CSV, formatted to match console."""
    if r.get("skip") or "err" in r:
        return ""
    v = r.get(key)
    if v is None:
        return ""
    if key in ("us", "tflops"):
        return f"{v:.1f}"
    if key == "max_err":
        return f"{v:.2e}"
    if key == "min_cos":
        return f"{v:.5f}"
    return v


def _csv_cmp(fly_r, other_r):
    """Compute (tflops_pct_str, maxerr_ratio_str) for CSV, formatted to match console."""
    if other_r.get("skip") or "err" in other_r or "err" in fly_r:
        return ("", "")
    ft, ot = fly_r.get("tflops"), other_r.get("tflops")
    pct = f"{ft / ot * 100:.1f}%" if ft and ot and ot > 0 else ""
    fe, oe = fly_r.get("max_err"), other_r.get("max_err")
    rat = f"{fe / oe:.2f}x" if fe is not None and oe is not None and oe > 0 else ""
    return (pct, rat)


def _write_cmp_csv(csv_path, data_rows, avg_rows):
    """Write compare-mode results to CSV."""
    header = [
        "B", "S", "H", "D", "dtype", "causal",
        "FlyDSL_Time(us)", "FlyDSL_TFLOPS", "FlyDSL_MaxErr",
        "CK_Time(us)", "CK_TFLOPS", "CK_MaxErr",
        "ASM_Time(us)", "ASM_TFLOPS", "ASM_MaxErr",
        "Fly/CK_TFLOPS%", "Fly/CK_MaxErr_ratio",
        "Fly/ASM_TFLOPS%", "Fly/ASM_MaxErr_ratio",
    ]
    def _metrics(fr, cr, ar):
        fck = _csv_cmp(fr, cr)
        fasm = _csv_cmp(fr, ar)
        return [
            _csv_val(fr, "us"), _csv_val(fr, "tflops"), _csv_val(fr, "max_err"),
            _csv_val(cr, "us"), _csv_val(cr, "tflops"), _csv_val(cr, "max_err"),
            _csv_val(ar, "us"), _csv_val(ar, "tflops"), _csv_val(ar, "max_err"),
            fck[0], fck[1], fasm[0], fasm[1],
        ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, fr, cr, ar in data_rows:
            w.writerow(list(cfg) + _metrics(fr, cr, ar))
        for label, fa, ca, aa in avg_rows:
            w.writerow([label, "", "", "", "", ""] + _metrics(fa, ca, aa))


def _write_normal_csv(csv_path, data_rows, avg_rows):
    """Write normal-mode results to CSV."""
    header = ["B", "S", "H", "D", "dtype", "causal", "Path", "Status",
              "MaxErr", "MinCos", "Time(us)", "TFLOPS"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, path, status, r in data_rows:
            w.writerow(list(cfg) + [
                path, status,
                _csv_val(r, "max_err"), _csv_val(r, "min_cos"),
                _csv_val(r, "us"), _csv_val(r, "tflops"),
            ])
        for label, avg in avg_rows:
            w.writerow([label, "", "", "", "", "", "", "--",
                _csv_val(avg, "max_err"), _csv_val(avg, "min_cos"),
                _csv_val(avg, "us"), _csv_val(avg, "tflops"),
            ])


def _avg_results(results_list, keys=("us", "tflops", "max_err")):
    """Average valid results over the specified keys."""
    valid = [r for r in results_list if not r.get("skip") and "err" not in r]
    if not valid:
        return {"skip": True}
    avg = {}
    for key in keys:
        vals = [r[key] for r in valid if key in r]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def _tag_group(cfg):
    """Extract (dtype_key, causal_tag) from config tuple."""
    return cfg[4], cfg[5]


def _print_grouped_avgs(rows, tag_fn, print_avg_fn):
    """Print grouped averages: all, then dtype x causal, dtype-only, causal-only."""
    print_avg_fn("AVG (all)", rows)
    seen_dtypes, seen_causals = [], []
    for row in rows:
        dk, ct = tag_fn(row)
        if dk not in seen_dtypes:
            seen_dtypes.append(dk)
        if ct not in seen_causals:
            seen_causals.append(ct)
    if len(seen_dtypes) > 1 and len(seen_causals) > 1:
        for dk in seen_dtypes:
            for ct in seen_causals:
                subset = [r for r in rows if tag_fn(r) == (dk, ct)]
                if subset:
                    print_avg_fn(f"AVG ({dk} {ct})", subset)
    if len(seen_dtypes) > 1:
        for dk in seen_dtypes:
            subset = [r for r in rows if tag_fn(r)[0] == dk]
            if subset:
                print_avg_fn(f"AVG ({dk})", subset)
    if len(seen_causals) > 1:
        for ct in seen_causals:
            subset = [r for r in rows if tag_fn(r)[1] == ct]
            if subset:
                print_avg_fn(f"AVG ({ct})", subset)


_CFG_HDR = f"{'B':>4s} {'S':>6s} {'H':>4s} {'D':>4s} {'dtype':>5s} {'causal':>8s}"
_CFG_W = len(_CFG_HDR)
_PATH_W = 20


def _fmt_cfg(cfg):
    """Format config tuple (B, S, H, D, dtype, causal) as fixed-width columns."""
    B, S, H, D, dt, cs = cfg
    return f"{B:>4d} {S:>6d} {H:>4d} {D:>4d} {dt:>5s} {cs:>8s}"


def _fmt_normal_row(cfg, path, status, r):
    """Format one row for normal test mode."""
    cfg_s = _fmt_cfg(cfg) if isinstance(cfg, tuple) else f"{cfg:>{_CFG_W}s}"
    path_s = f"  {path:<{_PATH_W}s}" if path else f"  {'':<{_PATH_W}s}"
    prefix = f"{cfg_s}{path_s}"
    if "err" in r:
        return f"{prefix} | {'ERROR':>6s} | {r['err'][:60]}"
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    return (
        f"{prefix} | {status:>6s} | "
        f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | "
        f"{us_s} {tf_s}"
    )


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
                    cfg = (batch, seq_len, nh, hd, dtype_key, causal_tag)
                    print(f"  {_fmt_cfg(cfg)} ...", flush=True)

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
                    rows.append((cfg, fly_r, ck_r, asm_r))

        col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s}"
        cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
        hdr1 = (
            f"{_CFG_HDR} | {'FlyDSL':^28s} | {'CK':^28s} | {'ASM':^28s}"
            f" | {'Fly/CK':^14s} | {'Fly/ASM':^14s}"
        )
        hdr2 = (
            f"{'':>{_CFG_W}s} | {col} | {col} | {col}"
            f" | {cmp_col} | {cmp_col}"
        )
        sep = "-" * len(hdr2)
        print(f"\n{hdr1}")
        print(hdr2)
        print(sep)
        for cfg, fly_r, ck_r, asm_r in rows:
            print(
                f"{_fmt_cfg(cfg)} | {_fmt_result(fly_r)} | "
                f"{_fmt_result(ck_r)} | {_fmt_result(asm_r)}"
                f" | {_fmt_cmp(fly_r, ck_r)} | {_fmt_cmp(fly_r, asm_r)}"
            )

        cmp_avg_rows = []

        def _cmp_avg(label, subset):
            fa = _avg_results([f for _, f, _, _ in subset])
            ca = _avg_results([c for _, _, c, _ in subset])
            aa = _avg_results([a for _, _, _, a in subset])
            print(
                f"{label:>{_CFG_W}s} | {_fmt_result(fa)} | "
                f"{_fmt_result(ca)} | {_fmt_result(aa)}"
                f" | {_fmt_cmp(fa, ca)} | {_fmt_cmp(fa, aa)}"
            )
            cmp_avg_rows.append((label, fa, ca, aa))

        print(sep)
        _print_grouped_avgs(rows, lambda r: _tag_group(r[0]), _cmp_avg)
        print("=" * len(hdr2))

        csv_path = f"fmha_perf_compare_{_gpu_short_name()}.csv"
        _write_cmp_csv(csv_path, rows, cmp_avg_rows)
        print(f"Results saved to: {csv_path}")

    else:
        # ---- Normal FlyDSL test mode ----
        print("=" * 130)
        print(f"FlyDSL flash_attn_func ({causal_desc}, {dtype_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Kernel opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        print("=" * 130)

        hdr = (
            f"{_CFG_HDR}  {'Path':<{_PATH_W}s} | {'Status':>6s} | {'MaxErr':>8s} "
            f"{'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
        )
        print(f"\n{hdr}")
        print("-" * len(hdr))

        all_passed = True
        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, hd in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    cfg = (batch, seq_len, nh, hd, dtype_key, causal_tag)
                    try:
                        r = run_config(
                            batch, seq_len, nh, hd, dtype, causal,
                            warmup=args.warmup, iters=args.iters,
                            seed=args.seed, dtype_str=dtype_str,
                        )
                        path = ""
                        if "err" in r:
                            print(_fmt_normal_row(cfg, path, "ERROR", r))
                            all_passed = False
                            rows.append((cfg, path, "ERROR", r))
                            continue

                        status = "PASS" if r["passed"] else "FAIL"
                        if not r["passed"]:
                            all_passed = False
                        print(_fmt_normal_row(cfg, path, status, r))
                        rows.append((cfg, path, status, r))
                    except Exception as e:
                        print(_fmt_normal_row(cfg, "", "ERROR", {"err": str(e)}))
                        all_passed = False
                        rows.append((cfg, "", "ERROR", {"err": str(e)}))

        # ---- Summary table ----
        print(f"\n{hdr}")
        print("-" * len(hdr))
        for cfg, path, status, r in rows:
            print(_fmt_normal_row(cfg, path, status, r))

        normal_avg_rows = []

        def _normal_avg_fn(label, subset):
            avg = _avg_results(
                [r for _, _, _, r in subset],
                keys=("max_err", "min_cos", "us", "tflops"),
            )
            if not avg.get("skip"):
                print(_fmt_normal_row(label, "", "--", avg))
                normal_avg_rows.append((label, avg))

        print("-" * len(hdr))
        _print_grouped_avgs(rows, lambda r: _tag_group(r[0]), _normal_avg_fn)
        print("=" * len(hdr))

        csv_path = f"fmha_perf_{_gpu_short_name()}.csv"
        _write_normal_csv(csv_path, rows, normal_avg_rows)
        print(f"Results saved to: {csv_path}")
        if all_passed:
            print("All tests PASSED")
        else:
            print("Some tests FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
