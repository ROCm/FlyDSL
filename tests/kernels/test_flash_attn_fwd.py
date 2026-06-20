#!/usr/bin/env python3
"""flash_attn_func kernel test and benchmark for FlyDSL.

Tests flash_attn_func against PyTorch SDPA.
"""

import argparse
import csv
import hashlib
import logging
import math
import random
import sys
from pathlib import Path

# Configure logging to show INFO level messages (required for kernel name display)
logging.basicConfig(level=logging.INFO)

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

from kernels.flash_attn_interface import dualwave_splitk_workspace_elems, flydsl_flash_attn_func  # noqa: E402
from tests.test_common import run_perftest  # noqa: E402

# Tensor initialization range (uniform distribution)
UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123
# fp8 correctness gate (fixed; fp8 is lossy). These thresholds are intentionally
# not relaxed: an fp8 path that cannot meet them is a real failure, not a reason
# to move the gate.
FP8_MAX_ERR = 5e-2
FP8_MIN_COS = 0.98
# OCP e4m3fn (NOT the fnuz variant) end-to-end on gfx950.
FP8_DTYPE = torch.float8_e4m3fn
# Kernel config: populated from CLI args in main(); defaults here are only used
# if run_attn_config / _cfg_kw is called before main() (e.g. unit tests).
FLASH_ATTN_FUNC_KERNEL_CONFIG: dict = {
    "waves_per_eu": 2,
    "daz": True,
    "dualwave_swp_lazy_rescale": True,
    "dualwave_swp_setprio": True,
    "dualwave_swp_debug_lazy_counts": False,
    "dualwave_swp_enable_stagger": True,
}

# (batch, seq_len, num_heads, num_kv_heads, head_dim, num_kv_splits)
# num_kv_heads == num_heads -> MHA; num_kv_heads < num_heads -> GQA/MQA.
# num_kv_splits > 1 -> split-K path (gfx950 DUALWAVE_SWP only, seq_len >= 384, D=128).
DEFAULT_CONFIGS = [
    # set1
    (16, 8192, 64, 64, 128, 1),
    (16, 8192, 64, 8, 128, 1),
    (2, 1024, 64, 64, 128, 1),
    # set2
    (8, 128, 64, 64, 128, 1),
    (8, 256, 64, 64, 128, 1),
    (8, 512, 64, 64, 128, 1),
    (1, 128, 64, 64, 128, 1),
    (1, 256, 64, 64, 128, 1),
    (1, 384, 64, 64, 128, 1),
    (1, 512, 64, 64, 128, 1),
    (1, 1024, 64, 64, 128, 1),
    (1, 2048, 64, 64, 128, 1),
    (1, 4096, 64, 64, 128, 1),
    (1, 8192, 64, 64, 128, 1),
    (4, 8192, 64, 64, 128, 1),
    (1, 2048, 32, 32, 128, 1),
    (1, 4096, 32, 32, 128, 1),
    (1, 8192, 32, 32, 128, 1),
    (8, 8192, 32, 32, 128, 1),
    (1, 2048, 16, 16, 128, 1),
    (1, 4096, 16, 16, 128, 1),
    (1, 8192, 16, 16, 128, 1),
    (16, 8192, 16, 16, 128, 1),
    (1, 2048, 8, 8, 128, 1),
    (1, 4096, 8, 8, 128, 1),
    (1, 8192, 8, 8, 128, 1),
    (32, 8192, 8, 8, 128, 1),
    # set3
    (1, 8192, 2, 2, 128, 4),
    (1, 4096, 2, 2, 128, 4),
    (1, 2048, 4, 4, 128, 4),
    (1, 8192, 4, 4, 128, 2),
    # set4
    (1, 98144, 3, 3, 128, 5),
    (1, 147216, 3, 3, 128, 5),
    (1, 196288, 3, 3, 128, 5),
    (1, 245360, 3, 3, 128, 5),
    (1, 294432, 3, 3, 128, 5),
    (1, 12268, 24, 24, 128, 1),
    (1, 18402, 24, 24, 128, 1),
    (1, 24536, 24, 24, 128, 1),
    (1, 30670, 24, 24, 128, 2),
    (1, 36804, 24, 24, 128, 2),
    (1, 32768, 24, 24, 128, 1),
    (1, 32768, 32, 32, 128, 1),
    # set5
    (1, 64, 4, 4, 128, 1),
    (1, 30, 4, 4, 128, 1),
    (1, 1, 4, 4, 128, 1),
    (2, 7, 4, 4, 128, 1),
    (3, 31, 3, 3, 128, 1),
    (5, 33, 5, 5, 128, 1),
    (5, 63, 7, 7, 128, 1),
    (3, 65, 3, 3, 128, 1),
]

# Additional dense/varlen/cross-length cases.
# Row format: [seqlen_q, seqlen_kv, batch, num_heads, num_kv_heads, head_dim, num_kv_splits]
# - seqlen_kv is None: packed varlen self-attn, seqlen_q is per-batch Q/KV seqlens.
# - batch is an int: dense cross-length attention, seqlen_q/seqlen_kv are scalar lengths.
# - batch is None: packed varlen cross-length attention, seqlen_q/seqlen_kv are per-batch lists.
EXTRA_CONFIGS = [
    # varlen
    [[1024, 8192], None, None, 64, 64, 128, 1],
    [[512, 256, 1024, 128], None, None, 64, 64, 128, 1],  # uneven; MHA
    [[300, 700, 500], None, None, 32, 32, 128, 1],  # non-256/64-multiple
    [[1024, 1024], None, None, 64, 8, 128, 1],  # even, GQA
    [[1, 3, 31, 33, 63, 65], None, None, 16, 16, 128, 1],  # small + non-multiple
    # cross-length
    [31, 65, 1, 64, 8, 128, 1],
    [31, 100, 1, 64, 8, 128, 1],
    [31, 127, 1, 64, 8, 128, 1],
    [31, 1024, 1, 64, 8, 128, 1],
    [31, 8192, 1, 64, 8, 128, 1],
    [65, 31, 1, 64, 8, 128, 1],
    [65, 127, 1, 64, 8, 128, 1],
    [65, 1024, 1, 64, 8, 128, 1],
    [65, 8192, 1, 64, 8, 128, 1],
    [100, 31, 1, 64, 8, 128, 1],
    [100, 127, 1, 64, 8, 128, 1],
    [100, 8192, 1, 64, 8, 128, 1],
    [127, 31, 1, 64, 8, 128, 1],
    [127, 1024, 1, 64, 8, 128, 1],
    [127, 8192, 1, 64, 8, 128, 1],
    [1024, 31, 1, 64, 8, 128, 1],
    [1024, 100, 1, 64, 8, 128, 1],
    [1024, 8192, 1, 64, 8, 128, 1],
    [8192, 65, 1, 64, 8, 128, 1],
    [8192, 127, 1, 64, 8, 128, 1],
    [8192, 1024, 1, 64, 8, 128, 1],
    # varlen cross-length
    [[1024, 8192], [8192, 1024], None, 64, 64, 128, 1],
    [[512, 256, 1024, 128], [256, 512, 512, 256], None, 64, 8, 128, 1],
    [[300, 700, 500], [700, 300, 500], None, 32, 32, 128, 1],  # non-multiple
    [[1024, 31], [31, 1024], None, 64, 8, 128, 1],  # extreme q>>kv/q<<kv
    [[1, 65, 127, 333], [200, 64, 31, 100], None, 16, 16, 128, 1],
]


def _short_label(value):
    label = str(value)
    return label if len(label) <= 24 else label[:21] + "..."


def _extra_case_from_config(row):
    seqlen_q, seqlen_kv, batch, nh, nh_kv, hd, kv_splits = row
    if seqlen_kv is None:
        return {
            "sq_label": _short_label(seqlen_q),
            "skv_label": _short_label(seqlen_q),
            "nh": nh,
            "nh_kv": nh_kv,
            "hd": hd,
            "kv_splits": kv_splits,
            "kwargs": {"varlen_seqlens_q": list(seqlen_q)},
        }
    if batch is not None:
        return {
            "sq_label": f"[{seqlen_q}]",
            "skv_label": f"[{seqlen_kv}]",
            "nh": nh,
            "nh_kv": nh_kv,
            "hd": hd,
            "kv_splits": kv_splits,
            "kwargs": {"batch": batch, "seqlen_q": seqlen_q, "seqlen_kv": seqlen_kv},
        }
    return {
        "sq_label": _short_label(seqlen_q),
        "skv_label": _short_label(seqlen_kv),
        "nh": nh,
        "nh_kv": nh_kv,
        "hd": hd,
        "kv_splits": kv_splits,
        "kwargs": {"varlen_seqlens_q": list(seqlen_q), "varlen_seqlens_kv": list(seqlen_kv)},
    }


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
        assert nh_q % nh_kv == 0, f"num_heads ({nh_q}) must be divisible by num_kv_heads ({nh_kv})"
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
    max_score_elems = 1024 * 1024 * 1024  # 1 GiB → larger chunks, fewer kernel launches
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


@torch.no_grad()
def pytorch_ref_attention_qkv_diff(q, k, v, causal=True):
    """Reference for seqlen_q != seqlen_kv with a BOTTOM-RIGHT aligned causal mask.

    q: [B,Sq,H,D]; k,v: [B,Skv,Hkv,D]. Row r keeps keys [0, r+delta] with
    delta = Skv - Sq (so the mask hugs the bottom-right corner); an all-masked
    row outputs 0. Chunked over Q to bound the score matrix memory.
    """
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        assert nh_q % nh_kv == 0, f"num_heads ({nh_q}) must be divisible by num_kv_heads ({nh_kv})"
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    B, H, Sq, D = q_t.shape
    Skv = k_t.shape[2]
    delta = Skv - Sq
    scale = 1.0 / math.sqrt(D)
    k_trans = k_t.transpose(-1, -2).contiguous()
    out = torch.empty((B, H, Sq, D), device=q_t.device, dtype=torch.float32)
    chunk = max(1, min(Sq, (64 * 1024 * 1024) // max(B * H * Skv, 1)))
    key_idx = torch.arange(Skv, device=q_t.device).view(1, 1, 1, Skv)
    for s0 in range(0, Sq, chunk):
        s1 = min(s0 + chunk, Sq)
        scores = torch.matmul(q_t[:, :, s0:s1, :], k_trans) * scale
        if causal:
            q_idx = torch.arange(s0, s1, device=q_t.device).view(1, 1, -1, 1)
            scores = scores.masked_fill(key_idx > q_idx + delta, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)  # all-masked row -> 0 output
        out[:, :, s0:s1, :] = torch.matmul(probs, v_t)
    return out.transpose(1, 2)


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 hash of a tensor's raw bytes."""
    return hashlib.md5(tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()


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
        result["threshold_stats"].append({"range": f"[{lower:.0e}, {upper:.0e})", "count": count, "percentage": pct})
        print(f"    [{lower:.0e}, {upper:.0e}): {count:>8d} ({pct:6.2f}%)")

    count = int(np.sum(diff >= thresholds[-1]))
    pct = 100.0 * count / total_elements
    result["threshold_stats"].append({"range": f">={thresholds[-1]:.0e}", "count": count, "percentage": pct})
    print(f"    >={thresholds[-1]:.0e}       : {count:>8d} ({pct:6.2f}%)")

    return result


def _cfg_kw():
    """Return flydsl_flash_attn_func kwargs from the global kernel config."""
    return dict(
        waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
        daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
        dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
        dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
        dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
    )


def _flops(Sq, Skv, H, D, B, causal):
    """Compute FLOPs for one config (bottom-right causal or non-causal)."""
    delta = Skv - Sq
    if causal:
        valid = sum(min(max(r + delta + 1, 0), Skv) for r in range(Sq))
    else:
        valid = Sq * Skv
    return 4.0 * valid * D * H * B


def _acc_metric(o_f32, ref_f32, D, compare_mode=False):
    """Return (max_err, min_cos, passed) with zero-row-safe cosine.

    compare_mode: skip cosine (expensive for large configs); min_cos returned
    as None and passed is based on max_err only.
    """
    max_err = (o_f32 - ref_f32).abs().max().item()
    if compare_mode:
        return max_err, None, bool(max_err < 1e-2)
    res_rows = o_f32.reshape(-1, D)
    ref_rows = ref_f32.reshape(-1, D)
    nz = ref_rows.norm(dim=1) > 1e-6
    if bool(nz.all()):
        # All rows non-zero (typical for self-attn): compute cosine on views,
        # no fancy-index copies. For large B*S*H this avoids allocating GBs of
        # temporary tensors through boolean-mask index selection.
        min_cos = F.cosine_similarity(res_rows, ref_rows, dim=1).min().item()
        zero_ok = True
    else:
        min_cos = F.cosine_similarity(res_rows[nz], ref_rows[nz], dim=1).min().item() if bool(nz.any()) else 1.0
        zero_ok = res_rows[~nz].abs().max().item() < 1e-2 if bool((~nz).any()) else True
    passed = bool(max_err < 1e-2 and min_cos > 0.99 and zero_ok)
    return max_err, min_cos, passed


def run_attn_config(
    num_heads,
    head_dim,
    dtype,
    causal,
    warmup,
    iters,
    *,
    batch=1,
    seqlen_q=None,
    seqlen_kv=None,
    varlen_seqlens_q=None,
    varlen_seqlens_kv=None,
    num_kv_heads=None,
    num_kv_splits=1,
    seed=DEFAULT_SEED,
    dtype_str="bf16",
    verbose=False,
    trigger_lazy_else=False,
    compare_mode=False,
    precomputed_ref=None,
):
    """Unified flash-attention test/bench function.

    Modes (mutually exclusive):
    - dense self-attn:       seqlen_q set, varlen_seqlens_q is None, seqlen_kv is None.
    - dense cross-attn:      seqlen_q set, seqlen_kv set (may differ), varlen_seqlens_q is None.
    - varlen self-attn:      varlen_seqlens_q set, varlen_seqlens_kv is None.
    - varlen cross-attn:     varlen_seqlens_q and varlen_seqlens_kv both set.
    - split-K:               seqlen_q set, num_kv_splits > 1 (dense only, gfx950).

    compare_mode: when True, skip cosine computation (expensive for large B*S*H) and
    use pytorch_ref_attention (fast path) for dense self-attn instead of the
    general cross-attn reference.

    Returns a result dict with keys: max_err, [min_cos], passed, [us, tflops], [all_below_true/false_count].
    On skippable shapes (split-K constraint violated): returns {'skip': True}.
    On build/exec error: returns {'err': <str>}.
    """
    results = {}
    device = "cuda"
    varlen = varlen_seqlens_q is not None
    splitk = num_kv_splits > 1

    if num_kv_heads is None:
        num_kv_heads = num_heads
    H, D, H_KV = num_heads, head_dim, num_kv_heads
    debug_lazy = FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_debug_lazy_counts"]

    # ── split-K early-exit guard (mirrors run_splitk_config logic) ───────────
    if splitk:
        if D != 128 or dtype_str not in ("bf16", "f16") or (seqlen_q is not None and seqlen_q < 384):
            return {"skip": True}
        ws_elems = dualwave_splitk_workspace_elems(batch, H, seqlen_q, int(num_kv_splits), head_dim=D)
        if ws_elems * 4 >= 0xFFFFFFFF:
            return {"skip": True}

    setup_seed(seed)

    # ── tensor construction ──────────────────────────────────────────────────
    if varlen:
        vl_q = list(varlen_seqlens_q)
        vl_kv = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else vl_q
        B = len(vl_q)
        cuq = [0]
        [cuq.append(cuq[-1] + s) for s in vl_q]
        cukv = [0]
        [cukv.append(cukv[-1] + s) for s in vl_kv]
        total_q, total_kv = cuq[-1], cukv[-1]
        Sq = max(vl_q)
        cu_q_t = torch.tensor(cuq, dtype=torch.int32, device=device)
        cu_kv_t = torch.tensor(cukv, dtype=torch.int32, device=device)
        q_t = torch.empty(total_q, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        k_t = torch.empty(total_kv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        v_t = torch.empty(total_kv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        cross = any(vl_q[b] != vl_kv[b] for b in range(B))
        max_seqlen_kv = max(vl_kv)
    else:
        B, Sq = batch, seqlen_q
        Skv = seqlen_kv if seqlen_kv is not None else Sq
        cu_q_t = cu_kv_t = None
        cross = False
        max_seqlen_kv = None
        q_t = torch.empty(B, Sq, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        k_t = torch.empty(B, Skv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        v_t = torch.empty(B, Skv, H_KV, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        # TRIGGER_LAZY_ELSE: construct adversarial Q=1/K special input for debug.
        if trigger_lazy_else:
            q_t.fill_(1.0)
            k_t.zero_()
            if Sq >= 128:
                k_t[:, 64:128, :, :].fill_(80.0)
            print(
                "[DUALWAVE_SWP_LAZY_ELSE_DEBUG] constructed Q=1, K tile0=0, " "K tile1=80 to force row_max - m_row > 8",
                flush=True,
            )

    debug_counts = torch.zeros(2, dtype=torch.float32, device=device) if debug_lazy else None
    o_t = torch.zeros_like(q_t)

    # ── kernel launch ────────────────────────────────────────────────────────
    try:
        flydsl_flash_attn_func(
            q_t,
            k_t,
            v_t,
            causal=causal,
            num_kv_heads=H_KV,
            cu_seqlens_q=cu_q_t,
            cu_seqlens_kv=cu_kv_t,
            max_seqlen_q=Sq if varlen else None,
            max_seqlen_kv=max_seqlen_kv if varlen else None,
            cross_seqlen=cross if varlen else None,
            num_kv_splits=int(num_kv_splits),
            out=o_t,
            debug_counts=debug_counts,
            **_cfg_kw(),
        )
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback

        traceback.print_exc()
        return results

    if debug_lazy and debug_counts is not None:
        counts = debug_counts.detach().cpu().tolist()
        results["all_below_true_count"] = int(counts[0])
        results["all_below_false_count"] = int(counts[1])
        print(
            f"[DUALWAVE_SWP_LAZY_COUNTS] all_below_true={int(counts[0])}, " f"all_below_false={int(counts[1])}",
            flush=True,
        )

    # ── reference ───────────────────────────────────────────────────────────
    # precomputed_ref: shared reference tensor supplied by the caller (compare mode)
    # so that FlyDSL, aiter_ck, and aiter_asm all use the same single ref computation.
    # When not provided: compute here per mode.
    #   Dense self-attn → pytorch_ref_attention (no nan_to_num / +delta overhead).
    #   All other modes → pytorch_ref_attention_qkv_diff (handles delta≠0, zero rows).
    _self_attn = not varlen and (seqlen_kv is None or seqlen_kv == seqlen_q)
    if precomputed_ref is not None:
        ref_t = precomputed_ref
    elif varlen:
        ref_t = torch.empty(total_q, H, D, dtype=dtype, device=device)
        for b in range(B):
            qb = q_t[cuq[b] : cuq[b + 1]].unsqueeze(0).float()
            kb = k_t[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            vb = v_t[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            ref_fn = pytorch_ref_attention if vl_q[b] == vl_kv[b] else pytorch_ref_attention_qkv_diff
            ref_t[cuq[b] : cuq[b + 1]] = ref_fn(qb, kb, vb, causal=causal).to(dtype).squeeze(0)
    elif _self_attn:
        ref_t = pytorch_ref_attention(q_t.float(), k_t.float(), v_t.float(), causal=causal).to(dtype)
    else:
        ref_t = pytorch_ref_attention_qkv_diff(q_t.float(), k_t.float(), v_t.float(), causal=causal).to(dtype)

    o_f32 = o_t.contiguous().reshape(-1).float()
    ref_f32 = ref_t.contiguous().reshape(-1).float()
    max_err, min_cos, passed = _acc_metric(o_f32, ref_f32, D, compare_mode=compare_mode)
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    if min_cos is not None:
        results["min_cos"] = min_cos
    results["passed"] = passed

    if verbose:
        o_flat = o_t.reshape(-1)
        ref_flat = ref_t.reshape(-1)
        tag = f"B={B} Sq={Sq} H={H} D={D}"
        rm = compute_md5(o_flat)
        rm2 = compute_md5(ref_flat)
        print(f"  [{tag}] result_md5 = {rm}")
        print(f"  [{tag}] ref_md5    = {rm2}")
        if rm == rm2:
            print(f"  [{tag}] MD5 match: EXACT (bit-identical)")
        else:
            print(f"  [{tag}] MD5 match: DIFFER (not bit-identical)")
        print(f"  [{tag}] --- compare_arrays ---")
        compare_arrays(
            o_flat.to(torch.float32).detach().cpu().numpy(),
            ref_flat.to(torch.float32).detach().cpu().numpy(),
        )

    # ── benchmark ────────────────────────────────────────────────────────────
    try:
        if varlen:
            flops = sum(_flops(vl_q[b], vl_kv[b], H, D, 1, causal) for b in range(B))
        else:
            flops = _flops(Sq, Skv, H, D, B, causal)

        def kernel_fn():
            flydsl_flash_attn_func(
                q_t,
                k_t,
                v_t,
                causal=causal,
                num_kv_heads=H_KV,
                cu_seqlens_q=cu_q_t,
                cu_seqlens_kv=cu_kv_t,
                max_seqlen_q=Sq if varlen else None,
                max_seqlen_kv=max_seqlen_kv if varlen else None,
                cross_seqlen=cross if varlen else None,
                num_kv_splits=int(num_kv_splits),
                out=o_t,
                debug_counts=debug_counts,
                **_cfg_kw(),
            )

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
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


def run_aiter_bench(
    batch,
    seq_len,
    nheads,
    head_dim,
    dtype,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    backend="ck",
    num_kv_heads=None,
    precomputed_ref=None,
    seqlen_kv=None,
    varlen_seqlens_q=None,
    varlen_seqlens_kv=None,
):
    """Run true aiter_ck or true aiter_asm kernel via aiter and return {tflops, max_err, us}."""
    try:
        import aiter
    except Exception:
        return {"err": "aiter not installed"}

    varlen = varlen_seqlens_q is not None
    if backend == "asm" and dtype != torch.bfloat16:
        return {"skip": True}
    if backend == "asm" and (varlen or (seqlen_kv is not None and seqlen_kv != seq_len)):
        return {"skip": True}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    H, D = nheads, head_dim
    H_KV = num_kv_heads if num_kv_heads is not None else H
    if varlen:
        vl_q = list(varlen_seqlens_q)
        vl_kv = list(varlen_seqlens_kv) if varlen_seqlens_kv is not None else vl_q
        B = len(vl_q)
        S = max(vl_q)
        Skv = max(vl_kv)
        cuq = [0]
        [cuq.append(cuq[-1] + s) for s in vl_q]
        cukv = [0]
        [cukv.append(cukv[-1] + s) for s in vl_kv]
        total_q, total_kv = cuq[-1], cukv[-1]
        cu_q_t = torch.tensor(cuq, dtype=torch.int32, device="cuda")
        cu_kv_t = torch.tensor(cukv, dtype=torch.int32, device="cuda")
        q_pack = torch.empty(total_q, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        k_pack = torch.empty(total_kv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        v_pack = torch.empty(total_kv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        q = torch.zeros(B, S, H, D, dtype=dtype, device="cuda")
        k = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
        v = torch.zeros(B, Skv, H_KV, D, dtype=dtype, device="cuda")
        for b in range(B):
            q[b, : vl_q[b]] = q_pack[cuq[b] : cuq[b + 1]]
            k[b, : vl_kv[b]] = k_pack[cukv[b] : cukv[b + 1]]
            v[b, : vl_kv[b]] = v_pack[cukv[b] : cukv[b + 1]]
    else:
        B, S, Skv = batch, seq_len, seqlen_kv if seqlen_kv is not None else seq_len
        cu_q_t = cu_kv_t = None
        q = torch.empty(B, S, H, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        k = torch.empty(B, Skv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
        v = torch.empty(B, Skv, H_KV, D, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "ck":

        def aiter_forward():
            return aiter.mha_fwd(
                q,  # q
                k,  # k
                v,  # v
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                0,  # sink_size
                True,  # return_softmax_lse
                False,  # return_dropout_randval
                cu_seqlens_q=cu_q_t,
                cu_seqlens_kv=cu_kv_t,
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
                q,  # q
                k,  # k
                v,  # v
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                True,  # return_softmax_lse
                False,  # return_dropout_randval
                2,  # how_v3_bf16_cvt
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
        import traceback

        traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    if precomputed_ref is not None:
        ref = precomputed_ref
    elif varlen:
        ref = torch.empty(total_q, H, D, dtype=dtype, device="cuda")
        for b in range(B):
            qb = q_pack[cuq[b] : cuq[b + 1]].unsqueeze(0).float()
            kb = k_pack[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            vb = v_pack[cukv[b] : cukv[b + 1]].unsqueeze(0).float()
            ref_fn = pytorch_ref_attention if vl_q[b] == vl_kv[b] else pytorch_ref_attention_qkv_diff
            ref[cuq[b] : cuq[b + 1]] = ref_fn(qb, kb, vb, causal=causal).to(dtype).squeeze(0)
    else:
        ref_fn = (
            pytorch_ref_attention if (seqlen_kv is None or seqlen_kv == seq_len) else pytorch_ref_attention_qkv_diff
        )
        ref = ref_fn(q.float(), k.float(), v.float(), causal=causal).to(dtype)
    if varlen:
        out_cmp = torch.empty(total_q, H, D, dtype=out.dtype, device="cuda")
        for b in range(B):
            out_cmp[cuq[b] : cuq[b + 1]] = out[b, : vl_q[b]]
    else:
        out_cmp = out
    max_err = (out_cmp.float() - ref.float()).abs().max().item()
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
        if varlen:
            flops = sum(_flops(vl_q[b], vl_kv[b], H, D, 1, causal) for b in range(B))
        else:
            flops = _flops(S, Skv, H, D, B, causal)
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        results["bench_err"] = str(e)

    return results


# ── fp8 (e4m3fn) support ─────────────────────────────────────────────────────
# fp8 forward attention. Inputs Q/K/V are pre-quantized to e4m3fn on the host
# (no in-kernel quantization); per-tensor shape-[1] fp32 descales restore scale.
# QK logits use effective_logit_scale = sm_scale * q_descale * k_descale; the PV
# contribution / output applies v_descale. Accumulation (QK logits, online
# softmax max/sum, PV) stays fp32; output is bf16. The correctness reference
# dequantizes the SAME e4m3fn Q/K/V and applies the descales before the SDPA
# reference -- comparing against un-dequantized fp8 inputs would be an invalid
# reference.


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def quantize_per_tensor_fp8(x):
    """Per-tensor quantize a float tensor to e4m3fn + a shape-[1] fp32 descale.

    Mirrors aiter.ops.quant.per_tensor_quant: descale = amax / fp8_max, the
    stored fp8 value is round(x / descale), and dequant is fp8_value * descale.
    Uses aiter's helper when available (so the harness and the aiter comparator
    share identical quantization), with a numerically identical torch fallback.
    """
    try:
        from aiter import dtypes as _adtypes
        from aiter import per_tensor_quant as _ptq

        x_fp8, descale = _ptq(x, quant_dtype=_adtypes.fp8)
        # Enforce e4m3fn (not fnuz) and the expected per-tensor descale shape.
        if x_fp8.dtype != FP8_DTYPE:
            raise ValueError(f"aiter per_tensor_quant produced {x_fp8.dtype}, expected {FP8_DTYPE}")
        return x_fp8.contiguous(), descale.to(torch.float32).view(1).contiguous()
    except ImportError:
        fp8_max = torch.finfo(FP8_DTYPE).max
        amax = x.abs().max().to(torch.float32)
        descale = (amax / fp8_max).clamp(min=1e-12).view(1)
        x_fp8 = (x.to(torch.float32) / descale).to(FP8_DTYPE)
        return x_fp8.contiguous(), descale.to(torch.float32).contiguous()


def _dequant_fp8(x_fp8, descale):
    """Dequantize e4m3fn back to float32: fp8_value * descale."""
    return x_fp8.to(torch.float32) * descale.to(torch.float32)


def run_fp8_config(
    batch,
    seq_len,
    num_heads,
    head_dim,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    verbose=True,
    num_kv_heads=None,
    num_kv_splits=1,
):
    """Run the FlyDSL fp8 (e4m3fn) forward path and validate vs a dequantized-input
    SDPA reference at the fixed fp8 gate (max_err < 5e-2 and min_cos > 0.98).

    Unsupported fp8 configurations (non-gfx950, head_dim != 128, split-K) raise a
    clear error that is surfaced as an ERROR row (never a silent SKIP). Returns a
    run_config-compatible dict so it prints through the same summary table.
    """
    device = "cuda"
    results = {}

    if num_kv_heads is None:
        num_kv_heads = num_heads

    # fp8 split-K is not implemented. Reject it explicitly rather than silently
    # running a dense fp8 forward while the config row advertises kv_sp>1 (which
    # would validate the wrong path).
    if int(num_kv_splits) > 1:
        results["err"] = f"fp8 split-K (num_kv_splits={num_kv_splits}) is not implemented (dense fp8 only)"
        return results

    # fp8 forward is gfx950-only and head_dim==128 only. Reject anything else
    # up-front with a clear, specific error (surfaced as an ERROR row) rather
    # than a SKIP that would mask a real failure.
    try:
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        gpu_arch = ""
    if not gpu_arch.startswith("gfx950"):
        results["err"] = f"fp8 requires gfx950 (got '{gpu_arch or 'unknown'}')"
        return results
    if head_dim != 128:
        results["err"] = f"fp8 requires head_dim == 128 (got {head_dim})"
        return results
    if num_heads % num_kv_heads != 0:
        results["err"] = f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        return results
    if seq_len < 1:
        results["err"] = f"seq_len ({seq_len}) must be >= 1"
        return results

    B, S, H, D = batch, seq_len, num_heads, head_dim
    H_KV = num_kv_heads
    setup_seed(seed)

    # Host bf16 master tensors -> per-tensor e4m3fn + shape-[1] fp32 descales.
    q_bf16 = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    k_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    v_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
    q_fp8, q_descale = quantize_per_tensor_fp8(q_bf16)
    k_fp8, k_descale = quantize_per_tensor_fp8(k_bf16)
    v_fp8, v_descale = quantize_per_tensor_fp8(v_bf16)

    # Build the FlyDSL fp8 module through the PUBLIC builder (the same entry users
    # call); it routes gfx950 + D=128 + fp8 to the dual-wave SWP path and rejects
    # unsupported fp8 configs with a clear error. The fp8 kernel ABI adds
    # q/k/v_descale (forwarded as launch kwargs). Imported locally because the
    # module-level imports use the higher-level flash_attn_interface wrapper.
    from kernels.flash_attn_generic import build_flash_attn_func_module

    try:
        exe = build_flash_attn_func_module(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str="fp8",
            waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
            daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
            num_kv_heads=num_kv_heads,
            dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
            dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
            dualwave_swp_debug_lazy_counts=False,
            dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
        )
    except Exception as e:
        results["err"] = f"build: {e}"
        return results

    o_bf16 = torch.zeros(B, S, H, D, dtype=torch.bfloat16, device=device)
    q_flat = q_fp8.contiguous().view(-1)
    k_flat = k_fp8.contiguous().view(-1)
    v_flat = v_fp8.contiguous().view(-1)
    o_flat = o_bf16.contiguous().view(-1)

    # Public fp8 ABI: pre-quantized fp8 Q/K/V + per-tensor q/k/v_descale only. The
    # kernel runs QK on fp8 MFMA and dequantizes V in-kernel for the high-precision
    # PV; no production data is routed through debug/varlen slots.
    fp8_exec_kwargs = dict(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)

    try:
        exe(q_flat, k_flat, v_flat, o_flat, B, S, **fp8_exec_kwargs)
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        return results

    # Reference: dequantize the SAME e4m3fn Q/K/V (applying descales) and run the
    # high-precision SDPA reference the bf16 path uses.
    ref_4d = pytorch_ref_attention(
        _dequant_fp8(q_fp8, q_descale),
        _dequant_fp8(k_fp8, k_descale),
        _dequant_fp8(v_fp8, v_descale),
        causal=causal,
    )
    ref_flat = ref_4d.to(torch.float32).contiguous().view(-1)

    o_f32 = o_flat.float()
    ref_f32 = ref_flat.float()
    max_err = (o_f32 - ref_f32).abs().max().item()
    mean_err = (o_f32 - ref_f32).abs().mean().item()
    cos_sim = F.cosine_similarity(o_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1)
    min_cos = cos_sim.min().item()
    results["max_err"] = max_err
    results["mean_err"] = mean_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS

    if verbose:
        tag = f"B={B} S={S} H={H} D={D} fp8"
        print(f"  [{tag}] --- compare_arrays ---")
        compare_arrays(
            o_f32.detach().cpu().numpy(),
            ref_f32.detach().cpu().numpy(),
        )

    try:

        def kernel_fn():
            # Time the SAME public-ABI call as the correctness run: descales are
            # keyword-only on the launcher (positional args after B,S are
            # stride_kv_n/stride_q_n/head_dim_runtime), so they MUST be passed as
            # kwargs or they would silently bind to the stride/head_dim slots.
            exe(q_flat, k_flat, v_flat, o_flat, B, S, **fp8_exec_kwargs)

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
        results["us"] = us
        results["tflops"] = flops / (us * 1e-6) / 1e12
    except Exception as e:
        # A failed timing path must not be reportable as a clean PASS-with-N/A row.
        # Keep the correctness numbers visible but mark the row not-passed so the
        # summary surfaces the failure (see the status logic in main()).
        results["bench_err"] = str(e)
        results["passed"] = False

    return results


def aiter_asm_fp8_dispatch_ok(batch, seq_len, num_heads, num_kv_heads, head_dim):
    """Predicate: does this dense fp8 shape reach aiter's NATIVE gfx950 fp8 ASM
    kernel (fwd_hd128_fp8*.co, dtype fp8bf16, bf16_cvt=0)?

    Mirrors the aiter dispatch gate for native fp8 ASM: head_dim == 128, GQA
    ratio (num_heads / num_kv_heads) a power of two, and seqlen_q > 128. When
    this is False the aiter dispatcher falls back to CK, which must be labeled
    honestly per-shape (never reported as 'aiter asm fp8').
    """
    if head_dim != 128 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return False
    if not _is_pow2(num_heads // num_kv_heads):
        return False
    return seq_len > 128


def run_aiter_fp8_bench(
    batch,
    seq_len,
    nheads,
    head_dim,
    causal,
    warmup,
    iters,
    seed=DEFAULT_SEED,
    backend="asm",
    num_kv_heads=None,
):
    """Run aiter's fp8 forward and return {tflops, max_err, us, label}.

    backend="asm": drive the NATIVE gfx950 fp8 ASM kernel via
      aiter.ops.mha.fmha_v3_fwd(..., how_v3_bf16_cvt=0) directly. This is the
      genuine native-fp8 path (#2911), NOT the bf16-convert path (bf16_cvt!=0)
      and NOT a CK/triton fallback. Shapes that do not meet the native-asm gate
      are SKIPPED (so the asm column never silently substitutes CK for asm).
    backend="ck": secondary comparison via aiter.mha_fwd with descales.
    """
    try:
        import aiter
        from aiter.ops.mha import fmha_v3_fwd
    except Exception:
        return {"err": "aiter not installed"}

    if num_kv_heads is None:
        num_kv_heads = nheads
    asm_ok = aiter_asm_fp8_dispatch_ok(batch, seq_len, nheads, num_kv_heads, head_dim)
    if backend == "asm" and not asm_ok:
        # Native fp8 ASM kernel is not selected for this shape -> SKIP rather
        # than fall back to CK and mislabel it as asm.
        return {"skip": True}

    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    B, S, H, D = batch, seq_len, nheads, head_dim
    H_KV = num_kv_heads
    q_bf16 = torch.empty(B, S, H, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    k_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    v_bf16 = torch.empty(B, S, H_KV, D, dtype=torch.bfloat16, device="cuda").uniform_(*UNIFORM_RANGE)
    q_fp8, q_descale = quantize_per_tensor_fp8(q_bf16)
    k_fp8, k_descale = quantize_per_tensor_fp8(k_bf16)
    v_fp8, v_descale = quantize_per_tensor_fp8(v_bf16)
    softmax_scale = 1.0 / math.sqrt(D)

    if backend == "asm":
        results["label"] = "aiter_asm_fp8"

        def aiter_forward():
            out = torch.empty((B, S, H, D), device="cuda", dtype=torch.bfloat16)
            return fmha_v3_fwd(
                q_fp8,
                k_fp8,
                v_fp8,
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale (descales applied separately)
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                False,  # return_softmax_lse
                False,  # return_dropout_randval
                0,  # how_v3_bf16_cvt = 0 -> native fp8 (NOT bf16-convert)
                out,
                None,  # bias
                None,  # alibi_slopes
                q_descale,
                k_descale,
                v_descale,
                None,  # gen
            )

    elif backend == "ck":
        # CK fp8 is the labeled secondary comparison. Label honestly so a shape
        # that DID meet the asm gate is never silently reported as the headline.
        results["label"] = "aiter_ck_fp8" if asm_ok else "aiter_ck_fp8(fallback)"

        def aiter_forward():
            # CK fp8 needs an explicit bf16 output tensor; without it the op
            # infers an fp8 output and rejects ("invalid argument for fmha_fwd").
            out = torch.empty((B, S, H, D), device="cuda", dtype=torch.bfloat16)
            return aiter.mha_fwd(
                q_fp8,
                k_fp8,
                v_fp8,
                0.0,  # dropout_p
                softmax_scale,  # softmax_scale
                causal,  # is_causal
                -1,  # window_size_left
                -1,  # window_size_right
                0,  # sink_size
                False,  # return_softmax_lse
                False,  # return_dropout_randval
                cu_seqlens_q=None,
                cu_seqlens_kv=None,
                out=out,
                bias=None,
                alibi_slopes=None,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                gen=None,
            )

    else:
        return {"err": f"unsupported backend: {backend}"}

    try:
        res = aiter_forward()
        out = res[0] if isinstance(res, (tuple, list)) else res
        torch.cuda.synchronize()
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"err": f"{backend}: {e}"}

    # Compare against the dequantized-input SDPA reference (same e4m3fn inputs).
    # Compute the FULL fixed fp8 gate (max_err AND min_cos) so a claim that an
    # aiter fp8 row is within the gate is provable, not half-checked.
    ref = pytorch_ref_attention(
        _dequant_fp8(q_fp8, q_descale),
        _dequant_fp8(k_fp8, k_descale),
        _dequant_fp8(v_fp8, v_descale),
        causal=causal,
    )
    out_f32 = out.float()
    ref_f32 = ref.float()
    max_err = (out_f32 - ref_f32).abs().max().item()
    min_cos = F.cosine_similarity(out_f32.reshape(-1, D), ref_f32.reshape(-1, D), dim=1).min().item()
    results["max_err"] = max_err
    results["min_cos"] = min_cos
    results["passed"] = max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS

    try:

        def bench_fn():
            aiter_forward()

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
    """Format: 'Time(us) TFLOPS MaxErr MinCos St'.

    MinCos + a PASS/FAIL status are shown whenever the row carries the fixed-gate
    fields (fp8 comparator rows set min_cos/passed); for rows without them the
    extra columns render as '--' so the bf16/f16 layout is unchanged in width.
    """
    if r.get("skip"):
        return f"{'--':>10s} {'--':>8s} {'--':>8s} {'--':>7s} {'--':>4s}"
    if "err" in r:
        return f"{'--':>10s} {'ERR':>8s} {'--':>8s} {'--':>7s} {'--':>4s}"
    us = f"{r['us']:>10.1f}" if "us" in r else f"{'N/A':>10s}"
    tf = f"{r['tflops']:>8.1f}" if "tflops" in r else f"{'N/A':>8s}"
    err = f"{r['max_err']:>8.2e}" if "max_err" in r else f"{'N/A':>8s}"
    cos = f"{r['min_cos']:>7.4f}" if "min_cos" in r else f"{'--':>7s}"
    st = ("PASS" if r.get("passed") else "FAIL") if "passed" in r else "--"
    return f"{us} {tf} {err} {cos} {st:>4s}"


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
        "B",
        "S",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "kv_sp",
        "FlyDSL_Time(us)",
        "FlyDSL_TFLOPS",
        "FlyDSL_MaxErr",
        "FlyDSL_MinCos",
        "FlyDSL_Status",
        "aiter_ck_Time(us)",
        "aiter_ck_TFLOPS",
        "aiter_ck_MaxErr",
        "aiter_ck_MinCos",
        "aiter_ck_Status",
        "aiter_asm_Time(us)",
        "aiter_asm_TFLOPS",
        "aiter_asm_MaxErr",
        "aiter_asm_MinCos",
        "aiter_asm_Status",
        "Fly/aiter_ck_TFLOPS%",
        "Fly/aiter_ck_MaxErr_ratio",
        "Fly/aiter_asm_TFLOPS%",
        "Fly/aiter_asm_MaxErr_ratio",
    ]

    def _status_val(r):
        return ("PASS" if r.get("passed") else "FAIL") if "passed" in r else ""

    def _metrics(fr, cr, ar, cmp_overrides=None):
        if cmp_overrides is None:
            fck = _csv_cmp(fr, cr)
            fasm = _csv_cmp(fr, ar)
        else:
            fck, fasm = cmp_overrides
        return [
            _csv_val(fr, "us"),
            _csv_val(fr, "tflops"),
            _csv_val(fr, "max_err"),
            _csv_val(fr, "min_cos"),
            _status_val(fr),
            _csv_val(cr, "us"),
            _csv_val(cr, "tflops"),
            _csv_val(cr, "max_err"),
            _csv_val(cr, "min_cos"),
            _status_val(cr),
            _csv_val(ar, "us"),
            _csv_val(ar, "tflops"),
            _csv_val(ar, "max_err"),
            _csv_val(ar, "min_cos"),
            _status_val(ar),
            fck[0],
            fck[1],
            fasm[0],
            fasm[1],
        ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, fr, cr, ar in data_rows:
            w.writerow(list(cfg) + _metrics(fr, cr, ar))
        for avg_row in avg_rows:
            if len(avg_row) == 5:
                label, fa, ca, aa, cmp_overrides = avg_row
            else:
                label, fa, ca, aa = avg_row
                cmp_overrides = None
            # label + 7 empty cfg columns (S, H, Hkv, D, dtype, causal, kv_sp)
            w.writerow([label, "", "", "", "", "", "", ""] + _metrics(fa, ca, aa, cmp_overrides))


def _write_normal_csv(csv_path, data_rows, avg_rows):
    """Write normal-mode results to CSV."""
    header = [
        "B",
        "S",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "kv_sp",
        "Path",
        "Status",
        "MaxErr",
        "MinCos",
        "Time(us)",
        "TFLOPS",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cfg, path, status, r in data_rows:
            w.writerow(
                list(cfg)
                + [
                    path,
                    status,
                    _csv_val(r, "max_err"),
                    _csv_val(r, "min_cos"),
                    _csv_val(r, "us"),
                    _csv_val(r, "tflops"),
                ]
            )
        for label, avg in avg_rows:
            # label + 8 empty (S, H, Hkv, D, dtype, causal, kv_sp, Path) + Status + 4 metrics
            w.writerow(
                [
                    label,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "--",
                    _csv_val(avg, "max_err"),
                    _csv_val(avg, "min_cos"),
                    _csv_val(avg, "us"),
                    _csv_val(avg, "tflops"),
                ]
            )


def _write_varlen_cmp_csv(csv_path, data_rows):
    """Write compare-mode varlen / cross-length results to CSV."""
    header = [
        "Sq",
        "Skv",
        "H",
        "Hkv",
        "D",
        "dtype",
        "causal",
        "FlyDSL_Time(us)",
        "FlyDSL_TFLOPS",
        "FlyDSL_MaxErr",
        "aiter_ck_Time(us)",
        "aiter_ck_TFLOPS",
        "aiter_ck_MaxErr",
        "Fly/aiter_ck_TFLOPS%",
        "Fly/aiter_ck_MaxErr_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, fly_r, ck_r in data_rows:
            fck = _csv_cmp(fly_r, ck_r)
            w.writerow(
                [
                    sq,
                    skv,
                    nh,
                    nh_kv,
                    hd,
                    dtype_key,
                    causal_tag,
                    _csv_val(fly_r, "us"),
                    _csv_val(fly_r, "tflops"),
                    _csv_val(fly_r, "max_err"),
                    _csv_val(ck_r, "us"),
                    _csv_val(ck_r, "tflops"),
                    _csv_val(ck_r, "max_err"),
                    fck[0],
                    fck[1],
                ]
            )


def _write_varlen_normal_csv(csv_path, data_rows):
    """Write normal-mode varlen / cross-length results to CSV."""
    header = ["Sq", "Skv", "H", "Hkv", "D", "dtype", "causal", "Status", "MaxErr", "MinCos", "Time(us)", "TFLOPS"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, status, r in data_rows:
            w.writerow(
                [
                    sq,
                    skv,
                    nh,
                    nh_kv,
                    hd,
                    dtype_key,
                    causal_tag,
                    status,
                    _csv_val(r, "max_err"),
                    _csv_val(r, "min_cos"),
                    _csv_val(r, "us"),
                    _csv_val(r, "tflops"),
                ]
            )


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
    """Extract (dtype_key, causal_tag) from config tuple (B, S, H, Hkv, D, dtype, causal, kv_sp)."""
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


_CFG_HDR = f"{'B':>4s} {'S':>6s} {'H':>4s} {'Hkv':>4s} {'D':>4s} {'dtype':>5s} {'causal':>8s} {'kv_sp':>5s}"
_CFG_W = len(_CFG_HDR)
_PATH_W = 20


def _fmt_cfg(cfg):
    """Format config tuple (B, S, H, Hkv, D, dtype, causal, kv_sp) as fixed-width columns."""
    B, S, H, Hkv, D, dt, cs, ksp = cfg
    return f"{B:>4d} {S:>6d} {H:>4d} {Hkv:>4d} {D:>4d} {dt:>5s} {cs:>8s} {ksp:>5d}"


def _fmt_normal_row(cfg, path, status, r):
    """Format one row for normal test mode."""
    cfg_s = _fmt_cfg(cfg) if isinstance(cfg, tuple) else f"{cfg:>{_CFG_W}s}"
    path_s = f"  {path:<{_PATH_W}s}" if path else f"  {'':<{_PATH_W}s}"
    prefix = f"{cfg_s}{path_s}"
    if "err" in r:
        return f"{prefix} | {'ERROR':>6s} | {r['err'][:60]}"
    if r.get("skip"):
        return f"{prefix} | {'SKIP':>6s} | n/a"
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    return f"{prefix} | {status:>6s} | " f"{r['max_err']:>8.2e} {r['min_cos']:>8.5f} | " f"{us_s} {tf_s}"


_EXTRA_HDR = f"  {'Sq':<24} {'Skv':<24} {'H':>4} {'Hkv':>4} {'D':>4} {'dtype':>6} {'causal':>8}"
_EXTRA_W = len(_EXTRA_HDR)


def _fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag):
    return f"  {sq:<24} {skv:<24} {nh:>4} {nh_kv:>4} {hd:>4} {dtype_key:>6} {causal_tag:>8}"


def _fmt_extra_cmp_row(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, fly_r, ck_r):
    return f"{_fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag)} | {_fmt_result(fly_r)} | {_fmt_result(ck_r)} | {_fmt_cmp(fly_r, ck_r)}"


def _fmt_extra_normal_row(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag, status, r):
    prefix = _fmt_extra_prefix(sq, skv, nh, nh_kv, hd, dtype_key, causal_tag)
    if "err" in r:
        return f"{prefix} | {'ERROR':>6s} | {r['err'][:60]}"
    if r.get("skip"):
        return f"{prefix} | {'SKIP':>6s} | n/a"
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    min_cos = r.get("min_cos")
    min_cos_s = f"{min_cos:>8.5f}" if min_cos is not None else f"{'N/A':>8s}"
    return f"{prefix} | {status:>6s} | {r['max_err']:>8.2e} {min_cos_s} | {us_s} {tf_s}"


def _fmt_extra_cmp_avg_row(label, fly_r, ck_r, fly_ck_cmp):
    return f"{label:>{_EXTRA_W}s} | {_fmt_result(fly_r)} | {_fmt_result(ck_r)} | {_fmt_cmp_values(fly_ck_cmp)}"


def _fmt_extra_normal_avg_row(label, r):
    if r.get("skip"):
        return None
    us_s = f"{r['us']:>10.1f}" if "us" in r else "       N/A"
    tf_s = f"{r['tflops']:>9.1f}" if "tflops" in r else "      N/A"
    min_cos = r.get("min_cos")
    min_cos_s = f"{min_cos:>8.5f}" if min_cos is not None else f"{'N/A':>8s}"
    return f"{label:>{_EXTRA_W}s} | {'--':>6s} | {r['max_err']:>8.2e} {min_cos_s} | {us_s} {tf_s}"


def main():
    parser = argparse.ArgumentParser(description="flash_attn_func FlyDSL Test/Benchmark")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=None,
        help="KV head count for GQA/MQA. Default = num_heads (MHA). " "Requires num_heads %% num_kv_heads == 0.",
    )
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument(
        "--num_kv_splits",
        type=int,
        default=1,
        help="Split-K factor for the gfx950 DUALWAVE_SWP kernel. >1 runs the split-K "
        "path (+combine kernel) via run_splitk_config; D=128 bf16/f16, seq_len >= 384.",
    )
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument("--causal", action="store_true", dest="causal")
    causal_group.add_argument("--no-causal", action="store_false", dest="causal")
    parser.set_defaults(causal=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["fp16", "bf16", "fp8"],
        help="Data type: fp16, bf16, or fp8 (e4m3fn). Default: bf16+fp16 (fp8 must be requested explicitly).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FlyDSL vs aiter_ck vs aiter_asm performance (requires aiter)",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Run additional varlen/cross-length configs from EXTRA_CONFIGS",
    )
    # ── Kernel build options (override defaults without env vars) ──────────────
    parser.add_argument(
        "--waves-per-eu",
        type=int,
        default=2,
        dest="waves_per_eu",
        help="waves_per_eu occupancy hint passed to the FlyDSL kernel builder (default: 2)",
    )
    parser.add_argument(
        "--no-lazy-rescale",
        action="store_false",
        dest="dualwave_swp_lazy_rescale",
        help="Disable the DUALWAVE_SWP lazy online-softmax rescale (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_lazy_rescale=True)
    parser.add_argument(
        "--no-setprio",
        action="store_false",
        dest="dualwave_swp_setprio",
        help="Disable s_setprio scheduling hints in the DUALWAVE_SWP kernel (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_setprio=True)
    parser.add_argument(
        "--debug-lazy-counts",
        action="store_true",
        dest="dualwave_swp_debug_lazy_counts",
        help="Enable lazy-rescale branch counters (dualwave_swp_debug_lazy_counts=True, disabled by default)",
    )
    parser.add_argument(
        "--no-stagger",
        action="store_false",
        dest="dualwave_swp_enable_stagger",
        help="Disable wave-group phase stagger in the DUALWAVE_SWP kernel (enabled by default)",
    )
    parser.set_defaults(dualwave_swp_enable_stagger=True)
    parser.add_argument(
        "--trigger-lazy-else",
        action="store_true",
        dest="trigger_lazy_else",
        help="Construct adversarial inputs (Q=1, K tile0=0, K tile1=80) to force the "
        "lazy-rescale else-branch (row_max - m_row > 8); dense mode only, for debugging",
    )
    args = parser.parse_args()

    # Build kernel config from parsed args (no env-var reads).
    FLASH_ATTN_FUNC_KERNEL_CONFIG.update(
        {
            "waves_per_eu": args.waves_per_eu,
            "dualwave_swp_lazy_rescale": args.dualwave_swp_lazy_rescale,
            "dualwave_swp_setprio": args.dualwave_swp_setprio,
            "dualwave_swp_debug_lazy_counts": args.dualwave_swp_debug_lazy_counts,
            "dualwave_swp_enable_stagger": args.dualwave_swp_enable_stagger,
        }
    )

    # For fp8 the "torch dtype" entry is the dtype the host-side bf16 master
    # tensors are drawn in before per-tensor quantization to e4m3fn; the FlyDSL
    # dtype_str is "fp8". fp8 is never part of the default (bf16+fp16) sweep --
    # it must be requested explicitly so the unchanged dtype coverage is the
    # default behavior.
    dtype_map = {
        "fp16": (torch.float16, "f16"),
        "bf16": (torch.bfloat16, "bf16"),
        "fp8": (torch.bfloat16, "fp8"),
    }
    dtypes_to_test = [args.dtype] if args.dtype else ["bf16", "fp16"]
    causals_to_test = [args.causal] if args.causal is not None else [True, False]

    if args.batch or args.seq_len or args.num_heads or args.head_dim or args.num_kv_heads:
        nh_single = args.num_heads or 8
        configs = [
            (
                args.batch or 1,
                args.seq_len or 128,
                nh_single,
                args.num_kv_heads if args.num_kv_heads is not None else nh_single,
                args.head_dim or 128,
                args.num_kv_splits,
            )
        ]
    else:
        configs = DEFAULT_CONFIGS

    causal_desc = {True: "causal", False: "non-causal", None: "causal+non-causal"}[args.causal]
    dtype_desc = args.dtype or "bf16+fp16"
    extra_cases = (
        [_extra_case_from_config(row) for row in EXTRA_CONFIGS] if args.extra and configs is DEFAULT_CONFIGS else []
    )

    if args.compare:
        # ---- Comparison mode: FlyDSL vs aiter_ck vs aiter_asm ----
        print("=" * 130)
        print(f"FlyDSL vs aiter_ck vs aiter_asm  ({causal_desc}, {dtype_desc})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if args.num_kv_splits > 1:
            print(
                f"  FlyDSL column: split-K path (num_kv_splits={args.num_kv_splits}); "
                f"D!=128 / non-bf16,f16 / seq_len<384 / ws>4GiB configs SKIP"
            )
        print(f"  FlyDSL opts: {FLASH_ATTN_FUNC_KERNEL_CONFIG}")
        if "fp8" in dtypes_to_test:
            print(
                "  fp8 mode: aiter_asm column = NATIVE gfx950 fp8 ASM (fmha_v3_fwd, "
                "how_v3_bf16_cvt=0); SKIP where the native-asm gate (hdim=128, pow2 "
                "GQA, seqlen_q>128) is not met. aiter_ck column = aiter_ck fp8 (mha_fwd "
                "with descales), secondary."
            )
        else:
            print("  aiter_ck: bf16+fp16, aiter_asm: bf16 only (how_v3_bf16_cvt=2, bf16-convert)")
        print("=" * 130)
        print("Running benchmarks ...")

        rows = []
        for dtype_key in dtypes_to_test:
            dtype, dtype_str = dtype_map[dtype_key]
            for causal in causals_to_test:
                for batch, seq_len, nh, nh_kv_default, hd, cfg_kv_splits in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads / --num_kv_splits (if set) override the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    kv_splits = args.num_kv_splits if args.num_kv_splits > 1 else cfg_kv_splits
                    cfg = (batch, seq_len, nh, nh_kv, hd, dtype_key, causal_tag, kv_splits)
                    print(f"  {_fmt_cfg(cfg)} ...", flush=True)

                    # Compute reference once (shared by FlyDSL, aiter_ck, aiter_asm).
                    # All three use the same seed → same Q/K/V → identical reference.
                    setup_seed(args.seed)
                    _q = torch.empty(batch, seq_len, nh, hd, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
                    _k = torch.empty(batch, seq_len, nh_kv, hd, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
                    _v = torch.empty(batch, seq_len, nh_kv, hd, dtype=dtype, device="cuda").uniform_(*UNIFORM_RANGE)
                    shared_ref = pytorch_ref_attention(_q.float(), _k.float(), _v.float(), causal=causal).to(dtype)
                    del _q, _k, _v

                    try:
                        if dtype_str == "fp8":
                            fly_r = run_fp8_config(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                verbose=False,
                                num_kv_heads=nh_kv,
                                num_kv_splits=kv_splits,
                            )
                        else:
                            # upstream unified run_config/run_splitk_config into
                            # run_attn_config (handles split-K via num_kv_splits).
                            fly_r = run_attn_config(
                                nh,
                                hd,
                                dtype,
                                causal,
                                args.warmup,
                                args.iters,
                                batch=batch,
                                seqlen_q=seq_len,
                                num_kv_heads=nh_kv,
                                num_kv_splits=kv_splits,
                                seed=args.seed,
                                dtype_str=dtype_str,
                                trigger_lazy_else=args.trigger_lazy_else,
                                compare_mode=True,
                                precomputed_ref=shared_ref,
                            )
                    except Exception as _fly_err:
                        print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)}: {_fly_err}", flush=True)
                        fly_r = {"err": str(_fly_err)}
                    if dtype_str == "fp8":
                        ck_r = run_aiter_fp8_bench(
                            batch,
                            seq_len,
                            nh,
                            hd,
                            causal,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                            backend="ck",
                            num_kv_heads=nh_kv,
                        )
                        asm_r = run_aiter_fp8_bench(
                            batch,
                            seq_len,
                            nh,
                            hd,
                            causal,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                            backend="asm",
                            num_kv_heads=nh_kv,
                        )
                    else:
                        ck_r = run_aiter_bench(
                            batch,
                            seq_len,
                            nh,
                            hd,
                            dtype,
                            causal,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                            backend="ck",
                            num_kv_heads=nh_kv,
                            precomputed_ref=shared_ref,
                        )
                        asm_r = run_aiter_bench(
                            batch,
                            seq_len,
                            nh,
                            hd,
                            dtype,
                            causal,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                            backend="asm",
                            num_kv_heads=nh_kv,
                            precomputed_ref=shared_ref,
                        )
                    rows.append((cfg, fly_r, ck_r, asm_r))

        col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s} {'MinCos':>7s} {'St':>4s}"
        _col_w = len(col)
        cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
        hdr1 = (
            f"{_CFG_HDR} | {'FlyDSL':^{_col_w}s} | {'aiter_ck':^{_col_w}s} | {'aiter_asm':^{_col_w}s}"
            f" | {'Fly/aiter_ck':^14s} | {'Fly/aiter_asm':^14s}"
        )
        hdr2 = f"{'':>{_CFG_W}s} | {col} | {col} | {col}" f" | {cmp_col} | {cmp_col}"
        sep = "-" * len(hdr2)
        print(f"\n{hdr1}")
        print(hdr2)
        print(sep)
        for cfg, fly_r, ck_r, asm_r in rows:
            print(
                f"{_fmt_cfg(cfg)} | {_fmt_result(fly_r)} | "
                f"{_fmt_result(ck_r)} | {_fmt_result(asm_r)}"
                f" | {_fmt_cmp(fly_r, ck_r)}"
                f" | {_fmt_cmp(fly_r, asm_r)}"
            )

        cmp_avg_rows = []

        def _cmp_avg(label, subset):
            fa = _avg_results([f for _, f, _, _ in subset])
            ca = _avg_results([c for _, _, c, _ in subset])
            aa = _avg_results([a for _, _, _, a in subset])
            fck_cmp = _avg_cmp_values(subset, 1, 2)
            fasm_cmp = _avg_cmp_values(subset, 1, 3)
            print(
                f"{label:>{_CFG_W}s} | {_fmt_result(fa)} | "
                f"{_fmt_result(ca)} | {_fmt_result(aa)}"
                f" | {_fmt_cmp_values(fck_cmp)}"
                f" | {_fmt_cmp_values(fasm_cmp)}"
            )
            cmp_avg_rows.append(
                (
                    label,
                    fa,
                    ca,
                    aa,
                    (
                        _csv_cmp_values(fck_cmp),
                        _csv_cmp_values(fasm_cmp),
                    ),
                )
            )

        print(sep)
        _print_grouped_avgs(rows, lambda r: _tag_group(r[0]), _cmp_avg)
        print("=" * len(hdr2))

        csv_path = f"fmha_perf_compare_{_gpu_short_name()}.csv"
        _write_cmp_csv(csv_path, rows, cmp_avg_rows)
        print(f"Results saved to: {csv_path}")

        if extra_cases:
            print("=" * 130)
            print("Additional dense/varlen/cross-length cases: FlyDSL vs aiter_ck")
            print("=" * 130)
            col = f"{'Time(us)':>10s} {'TFLOPS':>8s} {'MaxErr':>8s}"
            cmp_col = f"{'TFLOPS':>7s} {'MaxErr':>6s}"
            xhdr1 = f"{_EXTRA_HDR} | " f"{'FlyDSL':^28} | {'aiter_ck':^28} | {'Fly/CK':^14}"
            xhdr2 = f"{'':>{_EXTRA_W}} | {col} | {col} | {cmp_col}"
            varlen_cmp_rows = []
            for dtype_key in dtypes_to_test:
                dtype, dtype_str = dtype_map[dtype_key]
                for causal in causals_to_test:
                    ctag = "causal" if causal else "nocausal"
                    for case in extra_cases:
                        nh = case["nh"]
                        nh_kv_eff = args.num_kv_heads if args.num_kv_heads is not None else case["nh_kv"]
                        hd = case["hd"]
                        kv_splits = case.get("kv_splits", 1)
                        kwargs = dict(case["kwargs"])
                        pre = _fmt_extra_prefix(case["sq_label"], case["skv_label"], nh, nh_kv_eff, hd, dtype_key, ctag)
                        print(f"{pre} ...", flush=True)
                        try:
                            fly_r = run_attn_config(
                                nh,
                                hd,
                                dtype,
                                causal,
                                args.warmup,
                                args.iters,
                                num_kv_heads=nh_kv_eff,
                                num_kv_splits=kv_splits,
                                seed=args.seed,
                                dtype_str=dtype_str,
                                compare_mode=True,
                                **kwargs,
                            )
                        except Exception as _fly_err:
                            print(
                                f"    [FlyDSL unsupported] Sq={case['sq_label']} Skv={case['skv_label']}: {_fly_err}",
                                flush=True,
                            )
                            fly_r = {"err": str(_fly_err)}
                        ck_r = run_aiter_bench(
                            kwargs.get("batch", 1),
                            kwargs.get("seqlen_q", max(kwargs.get("varlen_seqlens_q", [1]))),
                            nh,
                            hd,
                            dtype,
                            causal,
                            args.warmup,
                            args.iters,
                            seed=args.seed,
                            backend="ck",
                            num_kv_heads=nh_kv_eff,
                            seqlen_kv=kwargs.get("seqlen_kv"),
                            varlen_seqlens_q=kwargs.get("varlen_seqlens_q"),
                            varlen_seqlens_kv=kwargs.get("varlen_seqlens_kv"),
                        )
                        varlen_cmp_rows.append(
                            (
                                case["sq_label"],
                                case["skv_label"],
                                nh,
                                nh_kv_eff,
                                hd,
                                dtype_key,
                                ctag,
                                fly_r,
                                ck_r,
                            )
                        )
            print("\n" + xhdr1)
            print(xhdr2)
            print("  " + "-" * (len(xhdr2) - 2))
            for sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, fly_r, ck_r in varlen_cmp_rows:
                print(_fmt_extra_cmp_row(sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, fly_r, ck_r))
            print("  " + "-" * (len(xhdr2) - 2))

            def _extra_cmp_avg(label, subset):
                fly_avg = _avg_results([row[7] for row in subset])
                ck_avg = _avg_results([row[8] for row in subset])
                fly_ck_cmp = _avg_cmp_values(subset, 7, 8)
                print(_fmt_extra_cmp_avg_row(label, fly_avg, ck_avg, fly_ck_cmp))

            _print_grouped_avgs(varlen_cmp_rows, lambda r: (r[5], r[6]), _extra_cmp_avg)
            print("=" * len(xhdr2))
            varlen_csv_path = f"fmha_varlen_perf_compare_{_gpu_short_name()}.csv"
            _write_varlen_cmp_csv(varlen_csv_path, varlen_cmp_rows)
            print(f"Varlen results saved to: {varlen_csv_path}")

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
                for batch, seq_len, nh, nh_kv_default, hd, cfg_kv_splits in configs:
                    causal_tag = "causal" if causal else "nocausal"
                    # CLI --num_kv_heads / --num_kv_splits (if set) override the per-config default.
                    nh_kv = args.num_kv_heads if args.num_kv_heads is not None else nh_kv_default
                    kv_splits = args.num_kv_splits if args.num_kv_splits > 1 else cfg_kv_splits
                    cfg = (batch, seq_len, nh, nh_kv, hd, dtype_key, causal_tag, kv_splits)
                    try:
                        if dtype_str == "fp8":
                            r = run_fp8_config(
                                batch,
                                seq_len,
                                nh,
                                hd,
                                causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                                verbose=False,
                                num_kv_heads=nh_kv,
                                num_kv_splits=kv_splits,
                            )
                        else:
                            # upstream unified run_config/run_splitk_config into
                            # run_attn_config (handles split-K via num_kv_splits).
                            r = run_attn_config(
                                nh,
                                hd,
                                dtype,
                                causal,
                                args.warmup,
                                args.iters,
                                batch=batch,
                                seqlen_q=seq_len,
                                num_kv_heads=nh_kv,
                                num_kv_splits=kv_splits,
                                seed=args.seed,
                                dtype_str=dtype_str,
                                verbose=True,
                                trigger_lazy_else=args.trigger_lazy_else,
                            )
                        path = ""
                        if "err" in r:
                            print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)}: {r['err']}", flush=True)
                            print(_fmt_normal_row(cfg, path, "ERROR", r))
                            all_passed = False
                            rows.append((cfg, path, "ERROR", r))
                            continue
                        if r.get("skip"):
                            print(_fmt_normal_row(cfg, path, "SKIP", r))
                            rows.append((cfg, path, "SKIP", r))
                            continue

                        # A timing-path failure (bench_err) is surfaced as BENCHERR
                        # so a correctness-passing row whose benchmark crashed is
                        # never reported as a clean pass.
                        if r.get("bench_err"):
                            print(f"    [FlyDSL bench failed] {_fmt_cfg(cfg)}: {r['bench_err']}", flush=True)
                            status = "BENCHERR"
                            all_passed = False
                        else:
                            status = "PASS" if r["passed"] else "FAIL"
                            if not r["passed"]:
                                all_passed = False
                        print(_fmt_normal_row(cfg, path, status, r))
                        rows.append((cfg, path, status, r))
                    except Exception as e:
                        print(f"    [FlyDSL unsupported] {_fmt_cfg(cfg)}: {e}", flush=True)
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

        extra_ok = True
        if extra_cases:
            print("=" * 130)
            print("Additional dense/varlen/cross-length cases: FlyDSL vs reference")
            print("=" * 130)
            xhdr = (
                f"{_EXTRA_HDR} | " f"{'Status':>6s} | {'MaxErr':>8s} {'MinCos':>8s} | {'Time(us)':>10s} {'TFLOPS':>8s}"
            )
            varlen_rows = []
            for dtype_key in dtypes_to_test:
                dtype, dtype_str = dtype_map[dtype_key]
                for causal in causals_to_test:
                    ctag = "causal" if causal else "nocausal"
                    for case in extra_cases:
                        nh = case["nh"]
                        nh_kv_eff = args.num_kv_heads if args.num_kv_heads is not None else case["nh_kv"]
                        hd = case["hd"]
                        kv_splits = case.get("kv_splits", 1)
                        kwargs = dict(case["kwargs"])
                        pre = _fmt_extra_prefix(case["sq_label"], case["skv_label"], nh, nh_kv_eff, hd, dtype_key, ctag)
                        print(f"{pre} ...", flush=True)
                        try:
                            r = run_attn_config(
                                nh,
                                hd,
                                dtype,
                                causal,
                                args.warmup,
                                args.iters,
                                num_kv_heads=nh_kv_eff,
                                num_kv_splits=kv_splits,
                                seed=args.seed,
                                dtype_str=dtype_str,
                                verbose=True,
                                **kwargs,
                            )
                        except Exception as e:
                            print(f"{pre} RAISED: {e}")
                            varlen_rows.append(
                                (
                                    case["sq_label"],
                                    case["skv_label"],
                                    nh,
                                    nh_kv_eff,
                                    hd,
                                    dtype_key,
                                    ctag,
                                    "ERROR",
                                    {"err": str(e)},
                                )
                            )
                            extra_ok = False
                            continue
                        if "err" in r:
                            print(f"{pre} ERR: {r['err']}")
                            varlen_rows.append(
                                (case["sq_label"], case["skv_label"], nh, nh_kv_eff, hd, dtype_key, ctag, "ERROR", r)
                            )
                            extra_ok = False
                            continue
                        if r.get("skip"):
                            print(f"{pre} SKIP")
                            varlen_rows.append(
                                (case["sq_label"], case["skv_label"], nh, nh_kv_eff, hd, dtype_key, ctag, "SKIP", r)
                            )
                            continue
                        passed = bool(r.get("passed", False))
                        status = "PASS" if passed else "FAIL"
                        extra_ok = extra_ok and passed
                        varlen_rows.append(
                            (case["sq_label"], case["skv_label"], nh, nh_kv_eff, hd, dtype_key, ctag, status, r)
                        )
            print("\n" + xhdr)
            print("  " + "-" * (len(xhdr) - 2))
            for sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, status, r in varlen_rows:
                print(_fmt_extra_normal_row(sq, skv, nh, nh_kv_eff, hd, dtype_key, ctag, status, r))
            print("  " + "-" * (len(xhdr) - 2))

            def _extra_normal_avg(label, subset):
                avg = _avg_results(
                    [row[8] for row in subset],
                    keys=("max_err", "min_cos", "us", "tflops"),
                )
                avg_row = _fmt_extra_normal_avg_row(label, avg)
                if avg_row is not None:
                    print(avg_row)

            _print_grouped_avgs(varlen_rows, lambda r: (r[5], r[6]), _extra_normal_avg)
            print("=" * len(xhdr))
            varlen_csv_path = f"fmha_varlen_perf_{_gpu_short_name()}.csv"
            _write_varlen_normal_csv(varlen_csv_path, varlen_rows)
            print(f"Varlen results saved to: {varlen_csv_path}")

        if all_passed and extra_ok:
            print("All tests PASSED")
        else:
            print("Some tests FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
