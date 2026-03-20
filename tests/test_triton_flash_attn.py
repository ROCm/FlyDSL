#!/usr/bin/env python3
"""Triton Flash Attention kernel test and benchmark.

Tests the Triton attn_fwd kernel against PyTorch SDPA, using the same test
framework and parameter configs as FlyDSL's test_flash_attn_func.py.

Usage:
    cd /home/yanguahe/code/my_asm_code/FlyDSL
    FLIR_LOG_MORE=1 HIP_VISIBLE_DEVICES=0 python3 tests/test_triton_flash_attn.py
    FLIR_LOG_MORE=1 HIP_VISIBLE_DEVICES=0 python3 tests/test_triton_flash_attn.py \\
        --batch 1 --seq_len 8192 --num_heads 64 --head_dim 128
"""

import os
import sys
import argparse
import hashlib
import importlib.util
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

os.environ.setdefault("FLIR_LOG_MORE", "1")

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

import torch
import torch.nn.functional as F
import numpy as np
import triton  # noqa: F401 — populate sys.modules before exec_module

from tests.test_common import run_perftest

# Load flash-attention.py (hyphenated filename requires importlib).
# triton/fa/utils/ must precede tests/ on sys.path because tests/utils.py
# shadows it (single file vs package).
_fa_path = Path("/home/yanguahe/code/triton_perf/triton/fa/flash-attention.py")
_fa_dir = str(_fa_path.resolve().parent)
if _fa_dir in sys.path:
    sys.path.remove(_fa_dir)
sys.path.insert(0, _fa_dir)

_spec = importlib.util.spec_from_file_location("flash_attention", _fa_path)
_fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa)

sys.path.remove(_fa_dir)

MetaData = _fa.MetaData
attention = _fa.attention

UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pytorch_ref_attention(q, k, v, causal=True):
    """Reference: PyTorch SDPA in BSHD layout."""
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


def compute_md5(tensor: torch.Tensor) -> str:
    return hashlib.md5(
        tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, k: int = 5,
                   thresholds: list = None) -> dict:
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: arr1 {arr1.shape} vs arr2 {arr2.shape}")

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = int(np.sum(nan_mask1))
        print(f"  Warning: result contains {result['nan_info']['arr1_nan_count']} NaN values")
    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = int(np.sum(nan_mask2))
        print(f"  Warning: reference contains {result['nan_info']['arr2_nan_count']} NaN values")

    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = (diff / (1.0 + np.abs(arr2))).max()
    result["max_diff"] = float(diff.max())
    result["max_diff_thr"] = float(max_diff_thr)

    print(f"  diff.abs.max = {diff.max():.6f}")
    print(f"  diff.abs.mean = {diff.mean():.6f}")
    print(f"  max_diff_thr (rel) = {max_diff_thr:.6e}")

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


def run_config(batch, seq_len, num_heads, head_dim, dtype, causal, warmup,
               iters, layout="bshd", seed=DEFAULT_SEED):
    device = "cuda"
    results = {}

    B, S, H, D = batch, seq_len, num_heads, head_dim
    setup_seed(seed)

    sm_scale = D ** -0.5
    metadata = MetaData(sm_scale=sm_scale)
    metadata.max_seqlens_q = S
    metadata.max_seqlens_k = S
    metadata.layout = layout
    if causal:
        metadata.need_causal()

    if layout == "bshd":
        q = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        k = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        v = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    elif layout == "bhsd":
        q = torch.empty(B, H, S, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        k = torch.empty(B, H, S, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        v = torch.empty(B, H, S, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    else:
        results["err"] = f"Unsupported layout: {layout}"
        return results

    o = torch.empty_like(q)

    try:
        tri_out, _, _ = attention(q, k, v, o, metadata)
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback
        traceback.print_exc()
        return results

    if layout == "bshd":
        ref_4d = pytorch_ref_attention(q, k, v, causal=causal).to(dtype)
    elif layout == "bhsd":
        q_bshd = q.transpose(1, 2)
        k_bshd = k.transpose(1, 2)
        v_bshd = v.transpose(1, 2)
        ref_bshd = pytorch_ref_attention(q_bshd, k_bshd, v_bshd, causal=causal).to(dtype)
        ref_4d = ref_bshd.transpose(1, 2)

    o_f32 = tri_out.float()
    ref_f32 = ref_4d.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(
        o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1
    )
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    try:
        torch.testing.assert_close(tri_out, ref_4d, atol=2e-2, rtol=2e-2)
        results["passed"] = True
    except AssertionError:
        results["passed"] = False

    tag = f"B={B} S={S} H={H} D={D}"
    result_md5 = compute_md5(tri_out)
    ref_md5 = compute_md5(ref_4d)
    print(f"  [{tag}] result_md5 = {result_md5}")
    print(f"  [{tag}] ref_md5    = {ref_md5}")
    if result_md5 == ref_md5:
        print(f"  [{tag}] MD5 match: EXACT (bit-identical)")
    else:
        print(f"  [{tag}] MD5 match: DIFFER (not bit-identical)")

    print(f"  [{tag}] --- compare_arrays ---")
    compare_arrays(
        tri_out.to(torch.float32).detach().cpu().numpy(),
        ref_4d.to(torch.float32).detach().cpu().numpy(),
    )

    try:
        def kernel_fn():
            attention(q, k, v, o, metadata)

        _, us = run_perftest(kernel_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        tflops = flops / (us * 1e-6) / 1e12
        results["us"] = us
        results["tflops"] = tflops
    except Exception as e:
        results["bench_err"] = str(e)
        import traceback
        traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Triton Flash Attention Test/Benchmark"
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--layout", type=str, default="bshd",
                        choices=["bshd", "bhsd"])
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    causal = not args.no_causal
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    layout = args.layout

    print("=" * 130)
    print(f"Triton Flash Attention ({'causal' if causal else 'non-causal'}, {args.dtype}, layout={layout})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 130)

    if args.seq_len or args.head_dim or args.batch:
        configs = [(
            args.batch or 1,
            args.seq_len or 128,
            args.num_heads or 8,
            args.head_dim or 128,
        )]
    else:
        configs = [
            (1, 128, 8, 128),
            (1, 128, 64, 128),
            (1, 256, 32, 128),
            (1, 512, 32, 128),
            (2, 128, 8, 128),
            (1, 8192, 64, 128),
        ]

    hdr = (
        f"{'Config':>40s} | {'Status':>6s} | {'MaxErr':>8s} "
        f"{'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    all_passed = True
    for batch, seq_len, nh, hd in configs:
        tag = f"B={batch} S={seq_len} H={nh} D={hd}"
        try:
            r = run_config(
                batch, seq_len, nh, hd, dtype, causal,
                warmup=args.warmup, iters=args.iters,
                layout=layout, seed=args.seed,
            )
            if "err" in r:
                print(f"{tag:>40s} | {'ERROR':>6s} | {r['err'][:60]}")
                all_passed = False
                continue

            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False

            us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
            tf_s = f"{r['tflops']:>9.3f}" if "tflops" in r else "      N/A"
            print(
                f"{tag:>40s} | {status:>6s} | "
                f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | "
                f"{us_s} {tf_s}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{tag:>40s} | {'ERROR':>6s} | {str(e)[:60]}")
            all_passed = False

    print("=" * 130)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
