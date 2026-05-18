#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for MoE token sorting kernel.

Validates the FlyDSL GPU kernel against:
  1. Python reference implementation (moe_sorting_reference)
  2. aiter/CK kernel (if available)

Usage:
    FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=./ pytest tests/kernels/test_moe_sorting.py -v
    FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=./ python tests/kernels/test_moe_sorting.py
"""

import argparse
import os
import sys

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import is_rdna_arch  # noqa: E402

if is_rdna_arch():
    pytest.skip("MoE sorting kernel requires CDNA (MI300X/MI350X).", allow_module_level=True)

from kernels.moe_sorting_kernel import (  # noqa: E402
    UNIT_SIZE,
    moe_sorting_flydsl,
)

WARMUP_ITERS = 3
RUN_BENCH = os.environ.get("MOE_SORTING_BENCH", "0") == "1"


def _call_flydsl(topk_ids, topk_weights, E, model_dim=4096, topk=None, unit_size=UNIT_SIZE, expert_mask=None):
    """Test helper: allocates outputs and calls moe_sorting_flydsl (CK-compatible API)."""
    if topk is None:
        topk = topk_ids.shape[1]
    T = topk_ids.shape[0]
    max_padded = T * topk + E * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size
    device = topk_ids.device
    s_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    s_w = torch.empty(max_padded, dtype=torch.float32, device=device)
    s_eids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    nv = torch.empty(2, dtype=torch.int32, device=device)
    buf = torch.empty((T, model_dim), dtype=torch.bfloat16, device=device)
    return moe_sorting_flydsl(topk_ids, topk_weights, s_ids, s_w, s_eids, nv, buf, E, unit_size, expert_mask)


BENCH_ITERS = 20
BENCH_WARMUP = 10
BENCH_MEASURE = 50


# ---------------------------------------------------------------------------
# CPU reference implementation
# ---------------------------------------------------------------------------
def moe_sorting_reference(topk_ids, topk_weights, num_experts, unit_size=UNIT_SIZE, expert_mask=None):
    """Pure-Python reference matching the CK/aiter packed-ID format."""
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * unit_size - topk
    max_num_m_blocks = (max_num_tokens_padded + unit_size - 1) // unit_size

    sentinel = (topk << 24) | M
    sorted_ids = torch.full((max_num_tokens_padded,), sentinel, dtype=torch.int32, device=device)
    sorted_weights = torch.zeros((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)
    num_valid_ids = torch.zeros(2, dtype=torch.int32, device=device)

    enabled = expert_mask.cpu().tolist() if expert_mask is not None else None

    ids_cursor = 0
    expert_ids_cursor = 0
    skip_expert_num = 0
    for eid in range(num_experts):
        if enabled is not None and not enabled[eid]:
            skip_expert_num += 1
            continue
        token_id, topk_pos = torch.where(topk_ids == eid)
        count = token_id.numel()
        if count == 0:
            continue
        num_blocks = (count + unit_size - 1) // unit_size
        padded = num_blocks * unit_size
        sorted_ids[ids_cursor : ids_cursor + count] = (topk_pos << 24) | token_id
        sorted_weights[ids_cursor : ids_cursor + count] = topk_weights[token_id, topk_pos]
        ids_cursor += padded
        sorted_expert_ids[expert_ids_cursor : expert_ids_cursor + num_blocks] = eid - skip_expert_num
        expert_ids_cursor += num_blocks

    num_valid_ids[0] = ids_cursor
    num_valid_ids[1] = M
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_topk_ids(T, E, topk, device="cuda"):
    """Generate random topk_ids and topk_weights for testing.

    Each token gets *unique* expert assignments (no duplicate expert IDs per
    token), matching the real MoE router constraint.  The mesh can only store
    one topk_slot per (token, expert) pair, so duplicates would silently drop
    assignments.
    """
    assert topk <= E, f"topk={topk} must be <= E={E}"
    topk_ids = torch.zeros(T, topk, dtype=torch.int32, device=device)
    for t in range(T):
        perm = torch.randperm(E, device=device)[:topk]
        topk_ids[t] = perm.to(torch.int32)
    topk_weights = torch.rand(T, topk, dtype=torch.float32, device=device)
    return topk_ids, topk_weights


def check_sorted_ids(
    ref_ids, gpu_ids, num_padded, topk, M, label="sorted_ids", topk_ids=None, gpu_eids=None, unit_size=UNIT_SIZE
):
    """Compare sorted_ids up to num_padded, ignoring padding sentinels.

    When topk_ids and gpu_eids are provided, falls back to per-expert-block
    validation: verifies each non-sentinel packed ID in a block maps to the
    expert declared by sorted_expert_ids (catches cross-expert permutations).
    """
    sentinel = (topk << 24) | M
    ref_slice = ref_ids[:num_padded]
    gpu_slice = gpu_ids[:num_padded]

    mask = ref_slice != sentinel
    n_valid = mask.sum().item()

    if n_valid == 0:
        print(f"  [{label}] no valid tokens (all padding) — OK")
        return True

    ref_valid = ref_slice[mask]
    gpu_valid = gpu_slice[mask]

    if torch.equal(ref_valid, gpu_valid):
        print(f"  [{label}] exact match ({n_valid} valid entries)")
        return True

    mismatch = (ref_valid != gpu_valid).sum().item()
    print(f"  [{label}] WARNING: {mismatch}/{n_valid} entries differ (checking per-expert blocks)")

    # Per-expert-block validation: verify each packed ID is in the correct expert block
    if topk_ids is not None and gpu_eids is not None:
        n_blocks = num_padded // unit_size
        topk_ids_cpu = topk_ids.cpu()
        gpu_slice_cpu = gpu_slice.cpu()
        gpu_eids_cpu = gpu_eids.cpu()
        ref_slice_cpu = ref_slice.cpu()
        bad_blocks = []
        for blk in range(n_blocks):
            start = blk * unit_size
            end = start + unit_size
            expert_id = gpu_eids_cpu[blk].item()
            if expert_id < 0:
                continue
            blk_gpu = set()
            blk_ref = set()
            for i in range(start, end):
                g = gpu_slice_cpu[i].item()
                r = ref_slice_cpu[i].item()
                if g != sentinel:
                    tok = g & 0xFFFFFF
                    topk_pos = g >> 24
                    if tok < M and topk_pos < topk:
                        assigned_expert = topk_ids_cpu[tok, topk_pos].item()
                        if assigned_expert != expert_id:
                            bad_blocks.append((blk, expert_id, tok, topk_pos, assigned_expert))
                    blk_gpu.add(g)
                if r != sentinel:
                    blk_ref.add(r)
            if blk_gpu != blk_ref and not bad_blocks:
                bad_blocks.append((blk, expert_id, -1, -1, -1))
        if not bad_blocks:
            print(f"  [{label}] per-expert-block validated ({n_blocks} blocks) — OK")
            return True
        print(f"  [{label}] FAIL: {len(bad_blocks)} block(s) have cross-expert errors")
        for blk, eid, tok, tpos, actual in bad_blocks[:5]:
            if tok >= 0:
                print(f"    block {blk}: expert_id={eid}, token {tok} topk_pos {tpos} -> expert {actual}")
            else:
                print(f"    block {blk}: expert_id={eid}, set mismatch")
        return False

    # Fallback: global set equality (no topk_ids/gpu_eids provided)
    ref_set = set(ref_valid.cpu().tolist())
    gpu_set = set(gpu_valid.cpu().tolist())
    if ref_set == gpu_set:
        print(f"  [{label}] set-equal (order differs) — OK")
        return True

    missing = ref_set - gpu_set
    extra = gpu_set - ref_set
    print(f"  [{label}] MISMATCH (missing={len(missing)}, extra={len(extra)})")
    diff_mask = ref_valid != gpu_valid
    diff_indices = diff_mask.nonzero(as_tuple=True)[0][:10]
    for idx in diff_indices:
        r = ref_valid[idx].item()
        g = gpu_valid[idx].item()
        r_tok, r_topk = r & 0xFFFFFF, r >> 24
        g_tok, g_topk = g & 0xFFFFFF, g >> 24
        print(f"    idx={idx.item()}: ref=({r_tok},{r_topk}) gpu=({g_tok},{g_topk})")
    return False


def check_sorted_weights(
    ref_w, gpu_w, ref_ids, topk, M, atol=1e-5, label="sorted_weights", gpu_ids=None, num_padded=None
):
    """Compare sorted_weights, masking padding entries.

    When gpu_ids is provided and position-by-position comparison fails,
    falls back to per-entry validation: checks that each GPU (packed_id, weight)
    pair matches the reference by packed_id lookup (handles non-deterministic
    order from atomic scatter).
    """
    sentinel = (topk << 24) | M
    # Limit to num_padded if provided (entries beyond are uninitialized)
    check_range = num_padded if num_padded is not None else len(ref_ids)
    ref_slice = ref_ids[:check_range]
    mask = ref_slice != sentinel
    n_valid = mask.sum().item()
    if n_valid == 0:
        return True
    ref_valid = ref_w[:check_range][mask]
    gpu_valid = gpu_w[:check_range][mask]
    max_err = (ref_valid - gpu_valid).abs().max().item()
    ok = max_err < atol
    if ok:
        print(f"  [{label}] max_err={max_err:.2e} (OK)")
        return True
    # Position-by-position failed; try per-entry validation if gpu_ids provided
    if gpu_ids is not None:
        # Build lookup: packed_id -> expected weight from ref
        ref_lut = {}
        for i in range(check_range):
            pid = ref_ids[i].item()
            if pid != sentinel:
                ref_lut[pid] = ref_w[i].item()
        # Check each GPU entry within the padded range
        gpu_slice = gpu_ids[:check_range]
        max_pair_err = 0.0
        n_pair_checked = 0
        for i in range(check_range):
            gpid = gpu_slice[i].item()
            if gpid == sentinel:
                continue
            n_pair_checked += 1
            if gpid in ref_lut:
                err = abs(gpu_w[i].item() - ref_lut[gpid])
                max_pair_err = max(max_pair_err, err)
            else:
                max_pair_err = float("inf")
                break
        if n_pair_checked == n_valid and max_pair_err < atol:
            print(f"  [{label}] max_pair_err={max_pair_err:.2e} (OK, order differs)")
            return True
    status = "FAIL"
    print(f"  [{label}] max_err={max_err:.2e} ({status})")
    return False


def check_expert_ids(ref_eids, gpu_eids, label="sorted_expert_ids", num_valid_blocks=None):
    """Compare sorted_expert_ids within valid range.

    When num_valid_blocks is provided, compares only that many blocks
    (entries beyond are uninitialized garbage). Otherwise falls back to
    masking by ref_eids != -1 (for Python reference comparisons).
    """
    if num_valid_blocks is not None:
        n_valid = num_valid_blocks
        ref_valid = ref_eids[:n_valid]
        gpu_valid = gpu_eids[:n_valid]
    else:
        mask = ref_eids != -1
        n_valid = mask.sum().item()
        if n_valid == 0:
            return True
        ref_valid = ref_eids[mask]
        gpu_valid = gpu_eids[mask]
    ok = torch.equal(ref_valid, gpu_valid)
    status = "OK" if ok else "FAIL"
    print(f"  [{label}] {n_valid} blocks ({status})")
    if not ok:
        diff = (ref_valid != gpu_valid).nonzero(as_tuple=True)[0][:10]
        for idx in diff:
            print(f"    block {idx.item()}: ref={ref_valid[idx].item()} gpu={gpu_valid[idx].item()}")
    return ok


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------
def run_test(T, E, topk, unit_size=UNIT_SIZE, max_tokens=None):
    """Run a single MoE sorting test case.

    Returns (passed: bool, gpu_time_us: float or None).
    """
    # Let moe_sorting_flydsl auto-select decode/prefill path.
    # max_tokens is only needed for explicit decode-path override.
    from kernels.moe_sorting_kernel import _compute_sub_tokens

    sub_tokens = _compute_sub_tokens(E)
    DECODE_MAX_T = 16
    path = "decode" if T <= min(sub_tokens, DECODE_MAX_T) else "prefill"

    if max_tokens is None and path == "decode":
        max_tokens = max(T, 8)
        max_tokens = ((max_tokens + 7) // 8) * 8

    print(f"\n{'='*60}")
    print(f"Test: T={T}, E={E}, topk={topk}, unit_size={unit_size}, path={path}")
    print(f"{'='*60}")

    torch.manual_seed(42 + T * 1000 + E * 10 + topk)
    topk_ids, topk_weights = generate_topk_ids(T, E, topk)

    # --- Reference ---
    ref_ids, ref_w, ref_eids, ref_nvalid = moe_sorting_reference(topk_ids, topk_weights, E, unit_size)

    # --- FlyDSL GPU kernel ---
    try:
        gpu_ids, gpu_w, gpu_eids, gpu_nvalid, gpu_moe_buf = _call_flydsl(
            topk_ids,
            topk_weights,
            E,
            model_dim=4096,
            topk=topk,
            unit_size=unit_size,
        )
    except Exception as e:
        print(f"  [FAIL] Kernel launch failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None

    torch.cuda.synchronize()

    # --- Validate ---
    passed = True

    # 1. num_valid_ids
    nv_ok = torch.equal(ref_nvalid, gpu_nvalid)
    print(f"  [num_valid_ids] ref={ref_nvalid.tolist()} gpu={gpu_nvalid.tolist()} ({'OK' if nv_ok else 'FAIL'})")
    passed &= nv_ok

    num_padded = ref_nvalid[0].item()

    # 2. sorted_ids (per-expert-block validation)
    passed &= check_sorted_ids(
        ref_ids, gpu_ids, num_padded, topk, T, topk_ids=topk_ids, gpu_eids=gpu_eids, unit_size=unit_size
    )

    # 3. sorted_weights
    passed &= check_sorted_weights(ref_w, gpu_w, ref_ids, topk, T, gpu_ids=gpu_ids, num_padded=num_padded)

    # 4. sorted_expert_ids
    passed &= check_expert_ids(ref_eids, gpu_eids)

    # 5. moe_buf should be zeroed
    moe_buf_zero = (gpu_moe_buf.view(torch.int32) == 0).all().item()
    print(f"  [moe_buf_zeroed] {'OK' if moe_buf_zero else 'FAIL'}")
    passed &= moe_buf_zero

    # --- Benchmark (opt-in via MOE_SORTING_BENCH=1) ---
    gpu_time_us = None
    if passed and RUN_BENCH:
        for _ in range(WARMUP_ITERS):
            _call_flydsl(topk_ids, topk_weights, E, model_dim=4096, topk=topk, unit_size=unit_size)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(BENCH_ITERS):
            _call_flydsl(topk_ids, topk_weights, E, model_dim=4096, topk=topk, unit_size=unit_size)
        end.record()
        torch.cuda.synchronize()
        gpu_time_us = start.elapsed_time(end) * 1000.0 / BENCH_ITERS  # ms → us
        print(f"  [perf] {gpu_time_us:.2f} us/call ({path})")

    status = "PASSED" if passed else "FAILED"
    print(f"  >>> {status}")
    return passed, gpu_time_us


# ---------------------------------------------------------------------------
# Test with aiter reference (optional)
# ---------------------------------------------------------------------------
def run_test_vs_aiter(T, E, topk, unit_size=UNIT_SIZE, max_tokens=None):
    """Compare FlyDSL kernel against aiter GPU kernel (if available)."""
    try:
        from aiter.fused_moe import moe_sorting as aiter_moe_sorting
    except ImportError:
        print("  [SKIP] aiter not available for cross-validation")
        return None, None

    torch.manual_seed(42 + T * 1000 + E * 10 + topk)
    topk_ids, topk_weights = generate_topk_ids(T, E, topk)

    print(f"\n  [vs aiter] T={T}, E={E}, topk={topk}")

    # aiter reference
    aiter_ids, aiter_w, aiter_eids, aiter_nvalid, _ = aiter_moe_sorting(
        topk_ids,
        topk_weights,
        E,
        model_dim=4096,
        moebuf_dtype=torch.bfloat16,
        block_size=unit_size,
    )

    # FlyDSL (auto-dispatches decode/prefill)
    fly_ids, fly_w, fly_eids, fly_nvalid, _ = _call_flydsl(
        topk_ids,
        topk_weights,
        E,
        model_dim=4096,
        topk=topk,
        unit_size=unit_size,
    )
    torch.cuda.synchronize()

    # Compare
    nv_ok = torch.equal(aiter_nvalid, fly_nvalid)
    num_padded = aiter_nvalid[0].item()
    num_valid_blocks = num_padded // unit_size
    ids_ok = check_sorted_ids(aiter_ids, fly_ids, num_padded, topk, T, "sorted_ids(vs_aiter)")
    w_ok = check_sorted_weights(
        aiter_w, fly_w, aiter_ids, topk, T, label="sorted_weights(vs_aiter)", gpu_ids=fly_ids, num_padded=num_padded
    )
    e_ok = check_expert_ids(aiter_eids, fly_eids, "sorted_expert_ids(vs_aiter)", num_valid_blocks=num_valid_blocks)

    passed = nv_ok and ids_ok and w_ok and e_ok
    return passed, None


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------
DECODE_CONFIGS = [
    # (T, E, topk) — decode path (small T)
    (1, 256, 8),
    (1, 32, 5),
    (4, 256, 8),
    (8, 256, 8),
    (16, 256, 8),
    (32, 256, 8),
    (64, 256, 8),
    # Edge cases
    (1, 8, 2),
    (7, 32, 5),  # odd T, topk not power of 2
    (31, 64, 6),  # prime T, topk not power of 2
    # Production E > 256 (DECODE_BLOCK=512) — core coverage
    (1, 257, 9),  # DeepSeek-R1 (256 routed + 1 shared)
    (16, 257, 9),
    (16, 513, 9),  # Qwen3.5 (512 routed + 1 shared)
]

DECODE_CONFIGS_FULL = DECODE_CONFIGS + [
    # Extended production coverage (large_shape — CI skips by default)
    (8, 257, 9),
    (1, 385, 7),  # DeepSeek-V4 (384 routed + 1 shared)
    (16, 385, 7),
    (1, 513, 9),  # Qwen3.5
    (1, 128, 4),  # Qwen3-MoE
    (16, 129, 7),  # Qwen3-Next (128 + 1 shared)
    (16, 161, 7),  # GLM-4-MoE (160 + 1 shared)
]


PREFILL_CONFIGS = [
    # (T, E, topk) — prefill path (large T, HBM workspace)
    (128, 256, 8),
    (512, 256, 8),
    (1024, 256, 8),
    (2048, 256, 8),
    # Production E > 256 — core coverage
    (1024, 257, 9),  # DeepSeek-R1
    (1024, 513, 9),  # Qwen3.5
]

PREFILL_CONFIGS_FULL = PREFILL_CONFIGS + [
    # Extended (large_shape — CI skips by default)
    (4096, 256, 8),
    (8192, 256, 8),
    (16384, 256, 8),
    (16384, 257, 9),
    (1024, 385, 7),  # DeepSeek-V4
    (16384, 385, 7),
    (16384, 513, 9),
]


@pytest.mark.parametrize("T,E,topk", DECODE_CONFIGS)
def test_moe_sorting_decode(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting failed for T={T}, E={E}, topk={topk}"


@pytest.mark.large_shape
@pytest.mark.parametrize("T,E,topk", [c for c in DECODE_CONFIGS_FULL if c not in DECODE_CONFIGS])
def test_moe_sorting_decode_full(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting failed for T={T}, E={E}, topk={topk}"


@pytest.mark.parametrize("T,E,topk", PREFILL_CONFIGS)
def test_moe_sorting_prefill(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting (prefill) failed for T={T}, E={E}, topk={topk}"


@pytest.mark.large_shape
@pytest.mark.parametrize("T,E,topk", [c for c in PREFILL_CONFIGS_FULL if c not in PREFILL_CONFIGS])
def test_moe_sorting_prefill_full(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting (prefill) failed for T={T}, E={E}, topk={topk}"


def run_test_ep(T, E, topk, mask_ratio=0.5, unit_size=UNIT_SIZE):
    """Run MoE sorting test with expert_mask (EP mode)."""
    from kernels.moe_sorting_kernel import _compute_sub_tokens

    sub_tokens = _compute_sub_tokens(E)
    DECODE_MAX_T = 16
    if T <= min(sub_tokens, DECODE_MAX_T):
        path = "decode"
    else:
        path = "prefill"

    print(f"\n{'='*60}")
    print(f"EP Test: T={T}, E={E}, topk={topk}, mask_ratio={mask_ratio}, path={path}")
    print(f"{'='*60}")

    torch.manual_seed(42 + T * 1000 + E * 10 + topk + int(mask_ratio * 100))
    topk_ids, topk_weights = generate_topk_ids(T, E, topk)

    if mask_ratio == 0.0:
        expert_mask = torch.zeros(E, dtype=torch.int32, device="cuda")
    elif mask_ratio == 1.0:
        expert_mask = torch.ones(E, dtype=torch.int32, device="cuda")
    else:
        expert_mask = (torch.rand(E, device="cuda") < mask_ratio).to(torch.int32)
        if expert_mask.sum() == 0:
            expert_mask[0] = 1

    n_enabled = expert_mask.sum().item()
    print(f"  expert_mask: {n_enabled}/{E} experts enabled")

    ref_ids, ref_w, ref_eids, ref_nvalid = moe_sorting_reference(
        topk_ids, topk_weights, E, unit_size, expert_mask=expert_mask
    )

    try:
        gpu_ids, gpu_w, gpu_eids, gpu_nvalid, gpu_moe_buf = _call_flydsl(
            topk_ids,
            topk_weights,
            E,
            model_dim=4096,
            topk=topk,
            unit_size=unit_size,
            expert_mask=expert_mask,
        )
    except Exception as e:
        print(f"  [FAIL] Kernel launch failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    torch.cuda.synchronize()

    passed = True
    nv_ok = torch.equal(ref_nvalid, gpu_nvalid)
    print(f"  [num_valid_ids] ref={ref_nvalid.tolist()} gpu={gpu_nvalid.tolist()} ({'OK' if nv_ok else 'FAIL'})")
    passed &= nv_ok

    num_padded = ref_nvalid[0].item()
    passed &= check_sorted_ids(
        ref_ids, gpu_ids, num_padded, topk, T, topk_ids=topk_ids, gpu_eids=gpu_eids, unit_size=unit_size
    )
    passed &= check_sorted_weights(ref_w, gpu_w, ref_ids, topk, T, gpu_ids=gpu_ids, num_padded=num_padded)
    passed &= check_expert_ids(ref_eids, gpu_eids)

    moe_buf_zero = (gpu_moe_buf.view(torch.int32) == 0).all().item()
    print(f"  [moe_buf_zeroed] {'OK' if moe_buf_zero else 'FAIL'}")
    passed &= moe_buf_zero

    status = "PASSED" if passed else "FAILED"
    print(f"  >>> {status}")
    return passed


EP_CONFIGS = [
    # (T, E, topk, mask_ratio)
    (4, 256, 8, 0.5),  # decode path
    (8, 256, 8, 0.3),  # decode path, sparse
    (64, 256, 8, 0.5),  # prefill path
    (128, 256, 8, 0.7),  # prefill path
    (2048, 256, 8, 0.5),  # prefill path
    (4, 256, 8, 1.0),  # all enabled (should match non-EP)
    (64, 256, 8, 1.0),  # all enabled, prefill
    (4, 256, 8, 0.0),  # all masked (empty output)
    # Production E>256 with EP
    (8, 257, 9, 0.5),  # DeepSeek-R1 decode + EP
    (1024, 257, 9, 0.5),  # DeepSeek-R1 prefill + EP
    (8, 513, 9, 0.5),  # Qwen3.5 decode + EP
    (1024, 513, 9, 0.5),  # Qwen3.5 prefill + EP (E > K4_BLOCK)
]


@pytest.mark.parametrize("T,E,topk,mask_ratio", EP_CONFIGS)
def test_moe_sorting_ep(T, E, topk, mask_ratio):
    passed = run_test_ep(T, E, topk, mask_ratio)
    assert passed, f"EP test failed: T={T}, E={E}, topk={topk}, mask_ratio={mask_ratio}"


@pytest.mark.parametrize(
    "T,E,topk",
    [
        (1, 256, 8),
        (8, 256, 8),
    ],
)
def test_moe_sorting_vs_aiter(T, E, topk):
    result, _ = run_test_vs_aiter(T, E, topk)
    if result is None:
        pytest.skip("aiter not available")
    assert result, f"FlyDSL vs aiter mismatch for T={T}, E={E}, topk={topk}"


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------
def bench_eager_us(fn, warmup=BENCH_WARMUP, iters=BENCH_MEASURE, flush_l2=True):
    """Per-iteration CUDA events timer with L2 flush and median latency."""
    flush_buf = None
    if flush_l2:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2_bytes = getattr(props, "L2_cache_size", 4 * 1024 * 1024)
        flush_buf = torch.empty(max(l2_bytes * 2, 8 * 1024 * 1024), dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    latencies = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters))
    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        latencies = [x for x in latencies if lo <= x <= hi] or latencies
    del flush_buf
    return latencies[len(latencies) // 2]


def bench_graph_us(fn, warmup=BENCH_WARMUP, iters=BENCH_MEASURE):
    """CUDA graph benchmark — amortizes kernel launch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
    except RuntimeError:
        return None  # graph capture not supported

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def run_bench_comparison(token_sweep=None):
    """Benchmark FlyDSL vs CK (aiter) across T values in eager and graph modes."""
    try:
        from aiter.fused_moe import moe_sorting as aiter_moe_sorting
    except ImportError:
        print("  aiter not available, skipping CK comparison")
        aiter_moe_sorting = None

    E, topk, model_dim = 256, 8, 4096
    if token_sweep is None:
        token_sweep = [1, 4, 8, 16, 32, 64, 128, 512, 2048, 4096, 8192, 16384]

    from kernels.moe_sorting_kernel import _compute_sub_tokens

    sub_tokens = _compute_sub_tokens(E)

    print(f"\n{'=' * 110}")
    print(f"  MoE Sorting Benchmark: FlyDSL vs CK (E={E}, topk={topk}, unit_size={UNIT_SIZE})")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  CUs: {props.multi_processor_count}, decode threshold: T<={sub_tokens}")
    print(f"  Modes: eager (with L2 flush, median of {BENCH_MEASURE}), graph ({BENCH_MEASURE} replays)")
    print(f"{'=' * 110}")
    print(
        f"{'T':>6s} | {'Path':>7s} | {'FLY eager':>10s} | {'FLY graph':>10s} | "
        f"{'CK eager':>10s} | {'CK graph':>10s} | {'Eager':>7s} | {'Graph':>7s}"
    )
    print("-" * 110)

    for T in token_sweep:
        torch.manual_seed(42)
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(T)]).to(torch.int32)
        topk_weights = torch.rand(T, topk, dtype=torch.float32, device="cuda")

        path = "decode" if T <= sub_tokens else "prefill"

        # Pre-allocate outputs to avoid per-call torch.empty overhead
        max_num_tokens_padded = T * topk + E * UNIT_SIZE - topk
        max_num_m_blocks = (max_num_tokens_padded + UNIT_SIZE - 1) // UNIT_SIZE
        fly_sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device="cuda")
        fly_sorted_w = torch.empty(max_num_tokens_padded, dtype=torch.float32, device="cuda")
        fly_sorted_eids = torch.empty(max_num_m_blocks, dtype=torch.int32, device="cuda")
        fly_nvalid = torch.empty(2, dtype=torch.int32, device="cuda")

        fly_moe_buf_2d = torch.empty((T, model_dim), dtype=torch.bfloat16, device="cuda")

        def fly_fn():
            moe_sorting_flydsl(
                topk_ids,
                topk_weights,
                fly_sorted_ids,
                fly_sorted_w,
                fly_sorted_eids,
                fly_nvalid,
                fly_moe_buf_2d,
                E,
                UNIT_SIZE,
            )

        fly_eager = bench_eager_us(fly_fn)
        fly_graph = bench_graph_us(fly_fn)

        ck_eager, ck_graph = None, None
        if aiter_moe_sorting is not None:

            def ck_fn():
                aiter_moe_sorting(
                    topk_ids, topk_weights, E, model_dim=model_dim, moebuf_dtype=torch.bfloat16, block_size=UNIT_SIZE
                )

            ck_eager = bench_eager_us(ck_fn)
            ck_graph = bench_graph_us(ck_fn)

        def fmt(v):
            return f"{v:8.1f}us" if v is not None else "       N/A"

        def ratio(a, b):
            if a is None or b is None or b == 0:
                return "    N/A"
            r = a / b
            return f"  {r:.2f}x"

        print(
            f"{T:>6d} | {path:>7s} | {fmt(fly_eager)} | {fmt(fly_graph)} | "
            f"{fmt(ck_eager)} | {fmt(ck_graph)} | "
            f"{ratio(fly_eager, ck_eager)} | {ratio(fly_graph, ck_graph)}"
        )

    print("=" * 110)
    print("  Ratio < 1.0 = FlyDSL faster. Eager includes launch overhead. Graph amortizes it.")
    print()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MoE sorting kernel test & benchmark")
    parser.add_argument("-T", type=int, default=None, help="Token count")
    parser.add_argument("-E", type=int, default=None, help="Number of experts")
    parser.add_argument("-k", "--topk", type=int, default=None, help="Top-k")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--aiter", action="store_true", help="Compare with aiter")
    parser.add_argument("--bench", action="store_true", help="Run benchmark sweep (eager + graph, FlyDSL vs CK)")
    parser.add_argument(
        "--bench-tokens", type=str, default=None, help="Comma-separated T values for bench (default: all)"
    )
    args = parser.parse_args()

    if args.bench:
        token_sweep = None
        if args.bench_tokens:
            token_sweep = [int(t) for t in args.bench_tokens.split(",")]
        run_bench_comparison(token_sweep=token_sweep)
        return

    if args.T is not None:
        E = args.E or 256
        topk = args.topk or 8
        configs = [(args.T, E, topk)]
    elif args.all:
        configs = DECODE_CONFIGS + PREFILL_CONFIGS
    else:
        configs = [
            (1, 256, 8),
            (8, 256, 8),
            (32, 256, 8),
            (128, 256, 8),
            (512, 256, 8),
        ]

    total = 0
    failures = 0
    results = []

    for T, E, topk in configs:
        passed, time_us = run_test(T, E, topk)
        total += 1
        if not passed:
            failures += 1
        results.append({"T": T, "E": E, "topk": topk, "passed": passed, "us": time_us})

        if args.aiter:
            aiter_ok, _ = run_test_vs_aiter(T, E, topk)
            if aiter_ok is False:
                failures += 1

    print(f"\n{'='*60}")
    print(f"Results: {total - failures}/{total} passed")
    if failures:
        print(f"FAILURES: {failures}")
    else:
        print("ALL TESTS PASSED")
    print(f"{'='*60}")

    for r in results:
        t_str = f"{r['us']:.1f}us" if r["us"] else "N/A"
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  T={r['T']:>6d} E={r['E']:>3d} topk={r['topk']} {status} {t_str}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
