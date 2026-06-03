#!/usr/bin/env python3
"""flash_attn_func kernel test and benchmark for FlyDSL.

Tests flash_attn_func against PyTorch SDPA.
"""

import os
import sys
import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import shutil
import tarfile
import urllib.request
from contextlib import contextmanager
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

from kernels.flash_attn_generic import (
    build_flash_attn_func_module,
)
from tests.test_common import run_perftest

# Tensor initialization range (uniform distribution)
UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123
FLASH_ATTN_FUNC_KERNEL_CONFIG = {
    "waves_per_eu": int(os.getenv("FLYDSL_WAVES_PER_EU", "2")),
    "daz": True,
    "dualwave_swp_lazy_rescale": os.getenv("FLYDSL_DUALWAVE_SWP_LAZY_RESCALE", "1") == "1",
    "dualwave_swp_setprio": os.getenv("FLYDSL_DUALWAVE_SWP_SETPRIO", "1") == "1",
    "dualwave_swp_debug_lazy_counts": os.getenv("FLYDSL_DUALWAVE_SWP_DEBUG_LAZY_COUNTS", "0") == "1",
    "dualwave_swp_enable_stagger": os.getenv("FLYDSL_DUALWAVE_SWP_STAGGER", "1") == "1",
}

# (batch, seq_len, num_heads, num_kv_heads, head_dim)
# num_kv_heads == num_heads -> MHA; num_kv_heads < num_heads -> GQA/MQA.
DEFAULT_CONFIGS = [
    (8, 128, 64, 64, 128),
    (8, 256, 64, 64, 128),
    (8, 512, 64, 64, 128),
    (1, 128, 64, 64, 128),
    (1, 256, 64, 64, 128),
    (1, 384, 64, 64, 128),
    (1, 512, 64, 64, 128),
    (1, 1024, 64, 64, 128),

    (1, 2048, 64, 64, 128),
    (1, 4096, 64, 64, 128),
    (1, 8192, 64, 64, 128),
    (4, 8192, 64, 64, 128),

    (1, 2048, 32, 32, 128),
    (1, 4096, 32, 32, 128),
    (1, 8192, 32, 32, 128),
    (8, 8192, 32, 32, 128),

    (1, 2048, 16, 16, 128),
    (1, 4096, 16, 16, 128),
    (1, 8192, 16, 16, 128),
    (16, 8192, 16, 16, 128),

    (1, 2048, 8, 8, 128),
    (1, 4096, 8, 8, 128),
    (1, 8192, 8, 8, 128),
    (32, 8192, 8, 8, 128),
    (16, 8192, 64, 64, 128),

    # GQA configs (num_kv_heads < num_heads).
    (16, 8192, 64, 8, 128),
]


def _maybe_configure_custom_llvm_tools() -> None:
    """Prefer the custom mlir-opt build used for peak FlashAttention perf."""
    if os.getenv("FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM", "1").lower() in ("0", "false", "no", "off"):
        return
    if os.getenv("FLYDSL_COMPILE_LLVM_DIR"):
        llvm_dir = Path(os.environ["FLYDSL_COMPILE_LLVM_DIR"]).expanduser()
        if not (llvm_dir / "bin" / "mlir-opt").is_file():
            raise RuntimeError(
                f"FLYDSL_COMPILE_LLVM_DIR={llvm_dir} does not contain bin/mlir-opt. "
                "Set FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 to use bundled LLVM."
            )
        return

    candidates = []
    for env_name in ("FLYDSL_FLASH_ATTN_FUNC_LLVM_DIR", "FLYDSL_CUSTOM_LLVM_TOOLS_DIR"):
        raw = os.getenv(env_name)
        if raw:
            candidates.append(Path(raw).expanduser())

    archive_name = None
    extract_dir = None
    config_path = _repo / "thirdparty" / "custom-llvm-tools.json"
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
            archive_prefix = cfg.get("archive_name", "flydsl-mlir-tools")
            llvm_ref = cfg.get("llvm_ref", "")
            if len(llvm_ref) >= 12:
                archive_stem = f"{archive_prefix}-{llvm_ref[:12]}-manylinux_2_28-x86_64"
                archive_name = f"{archive_stem}.tar.gz"
                extract_dir = _repo / "build-fly" / "custom-llvm-tools" / archive_stem
                candidates.extend(
                    [
                        extract_dir,
                        _repo / ".cache" / "custom-llvm-tools" / archive_stem,
                    ]
                )
        except Exception as exc:
            print(f"[flash_attn_func] failed to read custom LLVM tool config: {exc}")

    candidates.extend(
        [
            _repo / "build-fly" / "custom-llvm-tools",
            _repo / ".cache" / "custom-llvm-tools",
        ]
    )

    for path in candidates:
        if (path / "bin" / "mlir-opt").is_file():
            os.environ["FLYDSL_COMPILE_LLVM_DIR"] = str(path)
            print(f"[flash_attn_func] using custom LLVM tools: {path}")
            return

    if archive_name and extract_dir:
        root_dir = extract_dir.parent
        archive_path = root_dir / archive_name
        root_dir.mkdir(parents=True, exist_ok=True)

        if not archive_path.is_file():
            https_uri = f"https://rocm.frameworks-devreleases.amd.com/llvm-tools/gfx942-gfx950/{archive_name}"
            print(f"[flash_attn_func] fetching custom LLVM tools: {archive_name}")
            try:
                urllib.request.urlretrieve(https_uri, archive_path)
            except Exception as exc:
                archive_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"failed to download custom LLVM tools from {https_uri}: {exc}. "
                    "Set FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 to use bundled LLVM."
                ) from exc

        if archive_path.is_file():
            try:
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to extract custom LLVM tools from {archive_path}: {exc}. "
                    "Set FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 to use bundled LLVM."
                ) from exc

        if (extract_dir / "bin" / "mlir-opt").is_file():
            os.environ["FLYDSL_COMPILE_LLVM_DIR"] = str(extract_dir)
            print(f"[flash_attn_func] using custom LLVM tools: {extract_dir}")
            return

    raise RuntimeError(
        "custom LLVM tools not found or missing bin/mlir-opt. "
        "Set FLYDSL_COMPILE_LLVM_DIR to an LLVM tools directory, "
        "or set FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 to use bundled LLVM."
    )


@contextmanager
def _custom_llvm_tools_env():
    prev_llvm_dir = os.environ.get("FLYDSL_COMPILE_LLVM_DIR")
    _maybe_configure_custom_llvm_tools()
    try:
        yield
    finally:
        if prev_llvm_dir is None:
            os.environ.pop("FLYDSL_COMPILE_LLVM_DIR", None)
        else:
            os.environ["FLYDSL_COMPILE_LLVM_DIR"] = prev_llvm_dir


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
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        assert nh_q % nh_kv == 0, (
            f"num_heads ({nh_q}) must be divisible by num_kv_heads ({nh_kv})"
        )
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    score_elems = q_t.shape[0] * q_t.shape[1] * q_t.shape[2] * k_t.shape[2]
    if score_elems > 128 * 1024 * 1024:
        return pytorch_ref_attention_chunked(q_t, k_t, v_t, causal=causal).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


@torch.no_grad()
def pytorch_ref_attention_chunked(q_t, k_t, v_t, causal=True):
    """Compute reference attention in Q chunks to avoid large SDPA workspaces."""
    B, H, S, D = q_t.shape
    max_score_elems = 64 * 1024 * 1024
    chunk_size = max(1, min(S, max_score_elems // max(B * H * S, 1)))
    out = torch.empty((B, H, S, D), device=q_t.device, dtype=torch.float32)
    k_trans = k_t.transpose(-1, -2).contiguous()
    scale = 1.0 / math.sqrt(D)
    key_idx = torch.arange(S, device=q_t.device).view(1, 1, 1, S)

    for q_start in range(0, S, chunk_size):
        q_end = min(q_start + chunk_size, S)
        q_chunk = q_t[:, :, q_start:q_end, :]
        scores = torch.matmul(q_chunk, k_trans) * scale
        if causal:
            q_idx = torch.arange(q_start, q_end, device=q_t.device).view(1, 1, -1, 1)
            scores = scores.masked_fill(key_idx > q_idx, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, q_start:q_end, :] = torch.matmul(probs, v_t)

    return out


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
    seed=DEFAULT_SEED, dtype_str="f16", verbose=True, num_kv_heads=None,
):
    device = "cuda"
    results = {}

    if seq_len % 128 != 0:
        results["err"] = f"seq_len ({seq_len}) must be divisible by 128 for flash_attn_func"
        return results
    if head_dim % 32 != 0 or head_dim < 64:
        results["err"] = f"head_dim ({head_dim}) must be >= 64 and divisible by 32"
        return results
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
        results["err"] = (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        return results

    try:
        with _custom_llvm_tools_env():
            exe = build_flash_attn_func_module(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                dtype_str=dtype_str,
                waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
                daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
                num_kv_heads=num_kv_heads,
                dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
                dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
                dualwave_swp_debug_lazy_counts=FLASH_ATTN_FUNC_KERNEL_CONFIG[
                    "dualwave_swp_debug_lazy_counts"
                ],
                dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
            )
    except Exception as e:
        results["err"] = f"build: {e}"
        import traceback

        traceback.print_exc()
        return results

    B, S, H, D = batch, seq_len, num_heads, head_dim
    H_KV = num_kv_heads
    setup_seed(seed)
    q_4d = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    k_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    v_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    trigger_lazy_else = os.getenv("FLYDSL_DUALWAVE_SWP_TRIGGER_LAZY_ELSE", "0") == "1"
    debug_lazy_counts = FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_debug_lazy_counts"]
    if trigger_lazy_else:
        q_4d.fill_(1.0)
        k_4d.zero_()
        if S >= 128:
            k_4d[:, 64:128, :, :].fill_(80.0)
        print(
            "[DUALWAVE_SWP_LAZY_ELSE_DEBUG] constructed Q=1, K tile0=0, "
            "K tile1=80 to force row_max - m_row > 8",
            flush=True,
        )

    q_flat = q_4d.contiguous().view(-1)
    k_flat = k_4d.contiguous().view(-1)
    v_flat = v_4d.contiguous().view(-1)
    o_flat = torch.zeros_like(q_flat)
    debug_counts = (
        torch.zeros(2, dtype=torch.float32, device=device) if debug_lazy_counts else None
    )

    try:
        if debug_lazy_counts:
            exe(q_flat, k_flat, v_flat, o_flat, B, S, debug_counts=debug_counts)
        else:
            exe(q_flat, k_flat, v_flat, o_flat, B, S)
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback

        traceback.print_exc()
        return results

    if debug_lazy_counts:
        counts = debug_counts.detach().cpu().tolist()
        all_below_true_count = int(counts[0])
        all_below_false_count = int(counts[1])
        results["all_below_true_count"] = all_below_true_count
        results["all_below_false_count"] = all_below_false_count
        print(
            "[DUALWAVE_SWP_LAZY_COUNTS] "
            f"all_below_true_count = {all_below_true_count}, "
            f"all_below_false_count = {all_below_false_count}",
            flush=True,
        )

    ref_4d = pytorch_ref_attention(q_4d.float(), k_4d.float(), v_4d.float(), causal=causal).to(dtype)
    ref_flat = ref_4d.contiguous().view(-1)

    o_f32 = o_flat.float()
    ref_f32 = ref_flat.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1)
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
            if debug_lazy_counts:
                exe(q_flat, k_flat, v_flat, o_flat, B, S, debug_counts=debug_counts)
            else:
                exe(q_flat, k_flat, v_flat, o_flat, B, S)

        # Warm up ROCTracer/torch.profiler itself so the measured run_perftest
        # below is not biased by first-profiler-session setup overhead.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                kernel_fn()
            torch.cuda.synchronize()

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
    warmup, iters, seed=DEFAULT_SEED, backend="ck", num_kv_heads=None,
):
    """Run true aiter_ck or true aiter_asm kernel via aiter and return {tflops, max_err, us}."""
    try:
        import aiter
    except Exception:
        return {"err": "aiter not installed"}

    if backend == "asm" and dtype != torch.bfloat16:
        return {"skip": True}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    H_KV = num_kv_heads if num_kv_heads is not None else H
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "ck":
        def aiter_forward():
            return aiter.mha_fwd(
                q,                  # q
                k,                  # k
                v,                  # v
                0.0,                # dropout_p
                softmax_scale,      # softmax_scale
                causal,             # is_causal
                -1,                 # window_size_left
                -1,                 # window_size_right
                0,                  # sink_size
                True,               # return_softmax_lse
                False,              # return_dropout_randval
                cu_seqlens_q=None,
                cu_seqlens_kv=None,
                out=None,
                bias=None,
                alibi_slopes=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                gen=None,
            )
    elif backend == "asm":
        def aiter_forward():
            return aiter.fmha_v3_fwd(
                q,                  # q
                k,                  # k
                v,                  # v
                0.0,                # dropout_p
                softmax_scale,      # softmax_scale
                causal,             # is_causal
                -1,                 # window_size_left
                -1,                 # window_size_right
                True,               # return_softmax_lse
                False,              # return_dropout_randval
                2,                  # how_v3_bf16_cvt
                out=None,
                bias=None,
                alibi_slopes=None,
                gen=None,
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

        # Warm up ROCTracer/torch.profiler itself so the measured run_perftest
        # below is not biased by first-profiler-session setup overhead.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_opus_attn_bench(
    batch, seq_len, nheads, head_dim, dtype, causal,
    warmup, iters, seed=DEFAULT_SEED, num_kv_heads=None,
):
    """Run the local opus_attn GQA kernels and return {tflops, max_err, us}."""
    if dtype != torch.bfloat16:
        return {"skip": True}
    if head_dim not in (128, 512):
        return {"skip": True}

    H_KV = num_kv_heads if num_kv_heads is not None else nheads
    if nheads % H_KV != 0:
        return {"err": f"num_heads ({nheads}) must be divisible by num_kv_heads ({H_KV})"}

    if head_dim == 128:
        q_tile, kv_tile, num_warps = 32, 64, 8
    else:
        q_tile, kv_tile, num_warps = 16, 32, 8
    if (seq_len % kv_tile) != 0 or (seq_len // kv_tile) < 6:
        return {"skip": True}
    if (seq_len % (q_tile * num_warps)) != 0:
        return {"skip": True}

    opus_dir = _repo / "opus_attn"
    if str(opus_dir) not in sys.path:
        sys.path.insert(0, str(opus_dir))
    try:
        opus_attn = importlib.import_module("opus_attn")
    except Exception as e:
        return {"err": f"opus_attn import: {e}"}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    try:
        out = opus_attn.forward(q, k, v, causal=causal)
        torch.cuda.synchronize()
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"err": f"opus_attn: {e}"}

    ref = pytorch_ref_attention(q.float(), k.float(), v.float(), causal=causal).to(dtype)
    results["max_err"] = (out.float() - ref.float()).abs().max().item()

    try:
        out_bench = torch.empty_like(q)

        def bench_fn():
            opus_attn.forward_out(q, k, v, out_bench, causal=causal)

        # Warm up ROCTracer/torch.profiler itself so the measured run_perftest
        # below is not biased by first-profiler-session setup overhead.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_exp_isa_hand_asm_bench(
    batch, seq_len, nheads, head_dim, dtype, causal,
    warmup, iters, seed=DEFAULT_SEED, dtype_str="bf16", verbose=False, num_kv_heads=None,
):
    """Run exp_isa/flash_attn_opus.v1.s and return a run_config-compatible result dict."""
    if dtype != torch.bfloat16 or dtype_str != "bf16":
        return {"skip": True}
    if not causal:
        return {"skip": True}

    H_KV = num_kv_heads if num_kv_heads is not None else nheads
    if (nheads, H_KV, head_dim) != (64, 64, 128):
        return {"skip": True}
    if seq_len % 256 != 0:
        return {"skip": True}

    try:
        opus_asm = importlib.import_module("exp_isa.opus_asm")
    except Exception as e:
        return {"err": f"exp_isa.opus_asm import: {e}"}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    q_4d = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    try:
        out_4d = opus_asm.forward(q_4d, k_4d, v_4d, causal=causal)
        torch.cuda.synchronize()
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"err": f"exp_isa_opus: {e}"}

    ref_4d = pytorch_ref_attention(q_4d.float(), k_4d.float(), v_4d.float(), causal=causal).to(dtype)
    o_f32 = out_4d.float()
    ref_f32 = ref_4d.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1)
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < 1e-2 and min_cos > 0.99

    if verbose:
        tag = f"B={B} S={S} H={H} D={D}"
        result_md5 = compute_md5(out_4d)
        ref_md5 = compute_md5(ref_4d)
        print(f"  [{tag}] exp_isa_opus result_md5 = {result_md5}")
        print(f"  [{tag}] exp_isa_opus ref_md5    = {ref_md5}")
        print(f"  [{tag}] exp_isa_opus --- compare_arrays ---")
        compare_arrays(
            out_4d.to(torch.float32).detach().cpu().numpy(),
            ref_4d.to(torch.float32).detach().cpu().numpy(),
        )

    try:
        out_bench = torch.empty_like(q_4d)

        def bench_fn():
            opus_asm.forward_out(q_4d, k_4d, v_4d, out_bench, causal=causal)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

        _, us = run_perftest(bench_fn, num_iters=iters, num_warmup=warmup)
        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_exp_isa_fmha_bench(
    batch, seq_len, nheads, head_dim, dtype, causal,
    warmup, iters, seed=DEFAULT_SEED, dtype_str="bf16", verbose=False, num_kv_heads=None,
):
    """Run exp_isa MI350 256x64 FMHA asm kernels and return a run_config-compatible result dict."""
    if dtype != torch.bfloat16 or dtype_str != "bf16":
        return {"skip": True}

    H_KV = num_kv_heads if num_kv_heads is not None else nheads
    if head_dim != 128 or H_KV <= 0 or nheads % H_KV != 0:
        return {"skip": True}
    if causal and nheads % 8 != 0:
        return {"skip": True}
    if seq_len % 256 != 0:
        return {"skip": True}

    try:
        fmha_asm = importlib.import_module("exp_isa.fmha_asm")
    except Exception as e:
        return {"err": f"exp_isa.fmha_asm import: {e}"}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    q_4d = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    k_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    v_4d = torch.empty(B, S, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)

    try:
        out_4d = fmha_asm.forward(q_4d, k_4d, v_4d, causal=causal)
        torch.cuda.synchronize()
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"err": f"exp_isa_fmha: {e}"}

    ref_4d = pytorch_ref_attention(q_4d.float(), k_4d.float(), v_4d.float(), causal=causal).to(dtype)
    o_f32 = out_4d.float()
    ref_f32 = ref_4d.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1)
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < 1e-2 and min_cos > 0.99

    if verbose:
        tag = f"B={B} S={S} H={H} D={D}"
        result_md5 = compute_md5(out_4d)
        ref_md5 = compute_md5(ref_4d)
        print(f"  [{tag}] exp_isa_fmha result_md5 = {result_md5}")
        print(f"  [{tag}] exp_isa_fmha ref_md5    = {ref_md5}")
        print(f"  [{tag}] exp_isa_fmha --- compare_arrays ---")
        compare_arrays(
            out_4d.to(torch.float32).detach().cpu().numpy(),
            ref_4d.to(torch.float32).detach().cpu().numpy(),
        )

    try:
        out_bench = torch.empty_like(q_4d)

        def bench_fn():
            fmha_asm.forward_out(q_4d, k_4d, v_4d, out_bench, causal=causal)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=False,
            with_stack=False,
            with_modules=True,
        ):
            for _ in range(10):
                bench_fn()
            torch.cuda.synchronize()

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
    return _fmt_cmp_values(_cmp_values(fly_r, other_r))


def _cmp_values(fly_r, other_r):
    """Return numeric comparison values for one valid FlyDSL/comparator row."""
    if other_r.get("skip") or "err" in other_r or "err" in fly_r:
        return {"skip": True}
    fly_tf = fly_r.get("tflops")
    oth_tf = other_r.get("tflops")
    fly_err = fly_r.get("max_err")
    oth_err = other_r.get("max_err")
    result = {}
    if fly_tf and oth_tf and oth_tf > 0:
        result["tflops_pct"] = fly_tf / oth_tf * 100
    if fly_err is not None and oth_err is not None and oth_err > 0:
        result["max_err_ratio"] = fly_err / oth_err
    return result


def _fmt_cmp_values(cmp_r):
    """Format numeric comparison values."""
    if cmp_r.get("skip"):
        return f"{'--':>7s} {'--':>6s}"
    if "tflops_pct" in cmp_r:
        pct = f"{cmp_r['tflops_pct']:>6.1f}%"
    else:
        pct = f"{'N/A':>7s}"
    if "max_err_ratio" in cmp_r:
        ratio = f"{cmp_r['max_err_ratio']:>5.2f}x"
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
    return _csv_cmp_values(_cmp_values(fly_r, other_r))


def _csv_cmp_values(cmp_r):
    """Format numeric comparison values for CSV."""
    if cmp_r.get("skip"):
        return ("", "")
    pct = f"{cmp_r['tflops_pct']:.1f}%" if "tflops_pct" in cmp_r else ""
    rat = f"{cmp_r['max_err_ratio']:.2f}x" if "max_err_ratio" in cmp_r else ""
    return (pct, rat)


def _write_cmp_csv(csv_path, data_rows, avg_rows):
    """Write compare-mode results to CSV."""
    header = [
        "B", "S", "H", "Hkv", "D", "dtype", "causal",
        "FlyDSL_Time(us)", "FlyDSL_TFLOPS", "FlyDSL_MaxErr",
        "OPUS_Time(us)", "OPUS_TFLOPS", "OPUS_MaxErr",
        "aiter_ck_Time(us)", "aiter_ck_TFLOPS", "aiter_ck_MaxErr",
        "aiter_asm_Time(us)", "aiter_asm_TFLOPS", "aiter_asm_MaxErr",
        "Fly/OPUS_TFLOPS%", "Fly/OPUS_MaxErr_ratio",
        "Fly/aiter_ck_TFLOPS%", "Fly/aiter_ck_MaxErr_ratio",
        "Fly/aiter_asm_TFLOPS%", "Fly/aiter_asm_MaxErr_ratio",
    ]
    def _metrics(fr, or_, cr, ar, cmp_overrides=None):
        if cmp_overrides is None:
            fopus = _csv_cmp(fr, or_)
            fck = _csv_cmp(fr, cr)
            fasm = _csv_cmp(fr, ar)
        else:
            fopus, fck, fasm = cmp_overrides
        return [
            _csv_val(fr, "us"), _csv_val(fr, "tflops"), _csv_val(fr, "max_err"),
            _csv_val(or_, "us"), _csv_val(or_, "tflops"), _csv_val(or_, "max_err"),
            _csv_val(cr, "us"), _csv_val(cr, "tflops"), _csv_val(cr, "max_err"),
            _csv_val(ar, "us"), _csv_val(ar, "tflops"), _csv_val(ar, "max_err"),
            fopus[0], fopus[1], fck[0], fck[1], fasm[0], fasm[1],
        ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, fr, or_, cr, ar in data_rows:
            w.writerow(list(cfg) + _metrics(fr, or_, cr, ar))
        for avg_row in avg_rows:
            if len(avg_row) == 6:
                label, fa, oa, ca, aa, cmp_overrides = avg_row
            else:
                label, fa, oa, ca, aa = avg_row
                cmp_overrides = None
            # label + 6 empty cfg columns (S, H, Hkv, D, dtype, causal)
            w.writerow(
                [label, "", "", "", "", "", ""]
                + _metrics(fa, oa, ca, aa, cmp_overrides)
            )


def _write_normal_csv(csv_path, data_rows, avg_rows):
    """Write normal-mode results to CSV."""
    header = ["B", "S", "H", "Hkv", "D", "dtype", "causal", "Path", "Status",
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
            # label + 7 empty (S, H, Hkv, D, dtype, causal, Path) + Status + 4 metrics
            w.writerow([label, "", "", "", "", "", "", "", "--",
                _csv_val(avg, "max_err"), _csv_val(avg, "min_cos"),
                _csv_val(avg, "us"), _csv_val(avg, "tflops"),
            ])


def _valid_result(r):
    return not r.get("skip") and "err" not in r


def _avg_results(results_list, keys=("us", "tflops", "max_err")):
    """Average valid results over the specified keys."""
    valid = [r for r in results_list if _valid_result(r)]
    if not valid:
        return {"skip": True}
    avg = {}
    for key in keys:
        vals = [r[key] for r in valid if key in r]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def _avg_cmp_values(rows, fly_idx, other_idx):
    """Average per-row comparison values over rows where both sides are valid."""
    cmp_rows = [
        _cmp_values(row[fly_idx], row[other_idx])
        for row in rows
        if _valid_result(row[fly_idx]) and _valid_result(row[other_idx])
    ]
    if not cmp_rows:
        return {"skip": True}
    avg = {}
    for key in ("tflops_pct", "max_err_ratio"):
        vals = [r[key] for r in cmp_rows if key in r]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return avg


def _tag_group(cfg):
    """Extract (dtype_key, causal_tag) from config tuple (B, S, H, Hkv, D, dtype, causal)."""
    return cfg[5], cfg[6]


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


_CFG_HDR = (
    f"{'B':>4s} {'S':>6s} {'H':>4s} {'Hkv':>4s} {'D':>4s} "
    f"{'dtype':>5s} {'causal':>8s}"
)
_CFG_W = len(_CFG_HDR)
_PATH_W = 20


def _fmt_cfg(cfg):
    """Format config tuple (B, S, H, Hkv, D, dtype, causal) as fixed-width columns."""
    B, S, H, Hkv, D, dt, cs = cfg
    return f"{B:>4d} {S:>6d} {H:>4d} {Hkv:>4d} {D:>4d} {dt:>5s} {cs:>8s}"


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
    parser.add_argument(
        "--num_kv_heads", type=int, default=None,
        help="KV head count for GQA/MQA. Default = num_heads (MHA). "
             "Requires num_heads %% num_kv_heads == 0.",
    )
    parser.add_argument("--head_dim", type=int, default=None)
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument("--causal", action="store_true", dest="causal")
    causal_group.add_argument("--no-causal", action="store_false", dest="causal")
    parser.set_defaults(causal=None)
    parser.add_argument("--warmup", type=int, default=10)
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
        help="Compare FlyDSL vs OPUS vs aiter_ck vs aiter_asm performance (requires OPUS install and aiter)",
    )
    args = parser.parse_args()

    dtype_map = {"fp16": (torch.float16, "f16"), "bf16": (torch.bfloat16, "bf16")}
    dtypes_to_test = [args.dtype] if args.dtype else ["bf16", "fp16"]
    causals_to_test = [args.causal] if args.causal is not None else [True, False]

    if args.batch or args.seq_len or args.num_heads or args.head_dim or args.num_kv_heads:
        nh_single = args.num_heads or 8
        configs = [(
            args.batch or 1,
            args.seq_len or 128,
            nh_single,
            args.num_kv_heads if args.num_kv_heads is not None else nh_single,
            args.head_dim or 128,
        )]
    else:
        configs = DEFAULT_CONFIGS

    causal_desc = {True: "causal", False: "non-causal", None: "causal+non-causal"}[args.causal]
    dtype_desc = args.dtype or "bf16+fp16"

    if args.compare:
        # ---- Comparison mode: FlyDSL vs OPUS vs aiter_ck vs aiter_asm ----
        print("=" * 130)
        print(f"FlyDSL vs OPUS vs aiter_ck vs aiter_asm  ({causal_desc}, {dtype_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  FlyDSL opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        print(f"  OPUS: bf16 only, D=128/512; aiter_ck: bf16+fp16; aiter_asm: bf16 only")
        print("=" * 130)
        print("Running benchmarks ...")

        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, nh_kv_default, hd in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads (if set) overrides the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    cfg = (batch, seq_len, nh, nh_kv, hd, dtype_key, causal_tag)
                    print(f"  {_fmt_cfg(cfg)} ...", flush=True)

                    fly_r = run_config(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, dtype_str=dtype_str, verbose=False,
                        num_kv_heads=nh_kv,
                    )
                    opus_r = run_opus_attn_bench(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, num_kv_heads=nh_kv,
                    )
                    ck_r = run_aiter_bench(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, backend="ck",
                        num_kv_heads=nh_kv,
                    )
                    # asm_r = run_aiter_bench(
                    #     batch, seq_len, nh, hd, dtype, causal,
                    #     warmup=args.warmup, iters=args.iters,
                    #     seed=args.seed, backend="asm",
                    #     num_kv_heads=nh_kv,
                    # )
                    # asm_r = run_exp_isa_hand_asm_bench(
                    #     batch, seq_len, nh, hd, dtype, causal,
                    #     warmup=args.warmup, iters=args.iters,
                    #     seed=args.seed, dtype_str=dtype_str, verbose=False,
                    #     num_kv_heads=nh_kv,
                    # )
                    asm_r = run_exp_isa_fmha_bench(
                        batch, seq_len, nh, hd, dtype, causal,
                        warmup=args.warmup, iters=args.iters,
                        seed=args.seed, dtype_str=dtype_str, verbose=False,
                        num_kv_heads=nh_kv,
                    )
                    rows.append((cfg, fly_r, opus_r, ck_r, asm_r))

        col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s}"
        cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
        hdr1 = (
            f"{_CFG_HDR} | {'FlyDSL':^28s} | {'OPUS':^28s} | {'aiter_ck':^28s} | {'aiter_asm':^28s}"
            f" | {'Fly/OPUS':^14s} | {'Fly/aiter_ck':^14s} | {'Fly/aiter_asm':^14s}"
        )
        hdr2 = (
            f"{'':>{_CFG_W}s} | {col} | {col} | {col} | {col}"
            f" | {cmp_col} | {cmp_col} | {cmp_col}"
        )
        sep = "-" * len(hdr2)
        print(f"\n{hdr1}")
        print(hdr2)
        print(sep)
        for cfg, fly_r, opus_r, ck_r, asm_r in rows:
            print(
                f"{_fmt_cfg(cfg)} | {_fmt_result(fly_r)} | "
                f"{_fmt_result(opus_r)} | {_fmt_result(ck_r)} | {_fmt_result(asm_r)}"
                f" | {_fmt_cmp(fly_r, opus_r)} | {_fmt_cmp(fly_r, ck_r)}"
                f" | {_fmt_cmp(fly_r, asm_r)}"
            )

        cmp_avg_rows = []

        def _cmp_avg(label, subset):
            fa = _avg_results([f for _, f, _, _, _ in subset])
            oa = _avg_results([o for _, _, o, _, _ in subset])
            ca = _avg_results([c for _, _, _, c, _ in subset])
            aa = _avg_results([a for _, _, _, _, a in subset])
            fopus_cmp = _avg_cmp_values(subset, 1, 2)
            fck_cmp = _avg_cmp_values(subset, 1, 3)
            fasm_cmp = _avg_cmp_values(subset, 1, 4)
            print(
                f"{label:>{_CFG_W}s} | {_fmt_result(fa)} | "
                f"{_fmt_result(oa)} | {_fmt_result(ca)} | {_fmt_result(aa)}"
                f" | {_fmt_cmp_values(fopus_cmp)} | {_fmt_cmp_values(fck_cmp)}"
                f" | {_fmt_cmp_values(fasm_cmp)}"
            )
            cmp_avg_rows.append((
                label,
                fa,
                oa,
                ca,
                aa,
                (
                    _csv_cmp_values(fopus_cmp),
                    _csv_cmp_values(fck_cmp),
                    _csv_cmp_values(fasm_cmp),
                ),
            ))

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
                for batch, seq_len, nh, nh_kv_default, hd in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads (if set) overrides the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    cfg = (batch, seq_len, nh, nh_kv, hd, dtype_key, causal_tag)
                    try:
                        r = run_config(
                            batch, seq_len, nh, hd, dtype, causal,
                            warmup=args.warmup, iters=args.iters,
                            seed=args.seed, dtype_str=dtype_str,
                            num_kv_heads=nh_kv,
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
