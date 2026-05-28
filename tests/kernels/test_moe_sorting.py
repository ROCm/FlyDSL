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
    _supports_fused_oneshot,
    moe_softmax_sort_flydsl,
    moe_sorting_flydsl,
)
from kernels.topk_gating_softmax_kernel import (  # noqa: E402
    build_topk_gating_softmax_module,
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
    # Let moe_sorting_flydsl auto-select oneshot/multiphase path.
    # max_tokens is only needed for explicit oneshot-path override.
    from kernels.moe_sorting_kernel import BLOCK_SIZE, _compute_sub_tokens

    sub_tokens = _compute_sub_tokens(E)
    ONESHOT_MAX_T = min(sub_tokens, max(16, BLOCK_SIZE // max(topk, E // 8)))
    path = "oneshot" if T <= min(sub_tokens, ONESHOT_MAX_T) else "multiphase"

    if max_tokens is None and path == "oneshot":
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

    # FlyDSL (auto-dispatches oneshot/multiphase)
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
ONESHOT_CONFIGS = [
    # (T, E, topk) — oneshot path (small T)
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
    # Production E > 256 (ONESHOT_BLOCK=512) — core coverage
    (1, 257, 9),  # DeepSeek-R1 (256 routed + 1 shared)
    (16, 257, 9),
    (16, 513, 9),  # Qwen3.5 (512 routed + 1 shared)
]

ONESHOT_CONFIGS_FULL = ONESHOT_CONFIGS + [
    # Extended production coverage (large_shape — CI skips by default)
    (8, 257, 9),
    (1, 385, 7),  # DeepSeek-V4 (384 routed + 1 shared)
    (16, 385, 7),
    (1, 513, 9),  # Qwen3.5
    (1, 128, 4),  # Qwen3-MoE
    (16, 129, 7),  # Qwen3-Next (128 + 1 shared)
    (16, 161, 7),  # GLM-4-MoE (160 + 1 shared)
]


MULTIPHASE_CONFIGS = [
    # (T, E, topk) — multiphase path (large T, HBM workspace)
    (128, 256, 8),
    (512, 256, 8),
    (1024, 256, 8),
    (2048, 256, 8),
    # Production E > 256 — core coverage
    (1024, 257, 9),  # DeepSeek-R1
    (1024, 513, 9),  # Qwen3.5
]

MULTIPHASE_CONFIGS_FULL = MULTIPHASE_CONFIGS + [
    # Extended (large_shape — CI skips by default)
    (4096, 256, 8),
    (8192, 256, 8),
    (16384, 256, 8),
    (16384, 257, 9),
    (1024, 385, 7),  # DeepSeek-V4
    (16384, 385, 7),
    (16384, 513, 9),
]


@pytest.mark.parametrize("T,E,topk", ONESHOT_CONFIGS)
def test_moe_sorting_oneshot(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting failed for T={T}, E={E}, topk={topk}"


@pytest.mark.large_shape
@pytest.mark.parametrize("T,E,topk", [c for c in ONESHOT_CONFIGS_FULL if c not in ONESHOT_CONFIGS])
def test_moe_sorting_oneshot_full(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting failed for T={T}, E={E}, topk={topk}"


@pytest.mark.parametrize("T,E,topk", MULTIPHASE_CONFIGS)
def test_moe_sorting_multiphase(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting (multiphase) failed for T={T}, E={E}, topk={topk}"


@pytest.mark.large_shape
@pytest.mark.parametrize("T,E,topk", [c for c in MULTIPHASE_CONFIGS_FULL if c not in MULTIPHASE_CONFIGS])
def test_moe_sorting_multiphase_full(T, E, topk):
    passed, _ = run_test(T, E, topk)
    assert passed, f"MoE sorting (multiphase) failed for T={T}, E={E}, topk={topk}"


def run_test_ep(T, E, topk, mask_ratio=0.5, unit_size=UNIT_SIZE):
    """Run MoE sorting test with expert_mask (EP mode)."""
    from kernels.moe_sorting_kernel import BLOCK_SIZE, _compute_sub_tokens

    sub_tokens = _compute_sub_tokens(E)
    ONESHOT_MAX_T = min(sub_tokens, max(16, BLOCK_SIZE // max(topk, E // 8)))
    if T <= min(sub_tokens, ONESHOT_MAX_T):
        path = "oneshot"
    else:
        path = "multiphase"

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
    (4, 256, 8, 0.5),  # oneshot path
    (8, 256, 8, 0.3),  # oneshot path, sparse
    (64, 256, 8, 0.5),  # multiphase path
    (128, 256, 8, 0.7),  # multiphase path
    (2048, 256, 8, 0.5),  # multiphase path
    (4, 256, 8, 1.0),  # all enabled (should match non-EP)
    (64, 256, 8, 1.0),  # all enabled, multiphase
    (4, 256, 8, 0.0),  # all masked (empty output)
    # Production E>256 with EP
    (8, 257, 9, 0.5),  # DeepSeek-R1 oneshot + EP
    (1024, 257, 9, 0.5),  # DeepSeek-R1 multiphase + EP
    (8, 513, 9, 0.5),  # Qwen3.5 oneshot + EP
    (1024, 513, 9, 0.5),  # Qwen3.5 multiphase + EP (E > K4_BLOCK)
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
# Fused softmax+top-K+sort tests (moe_softmax_sort_flydsl)
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def _call_softmax_sort_fused(
    gating_logits, E, topk, dtype_str, *, model_dim=4096, unit_size=UNIT_SIZE, expert_mask=None, renormalize=True
):
    """Allocate outputs and call moe_softmax_sort_flydsl. Mirrors
    `_call_flydsl` but takes raw gating logits and dispatches through the
    fused entry point."""
    M = gating_logits.shape[0]
    max_padded = M * topk + E * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size
    device = gating_logits.device
    s_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    s_w = torch.empty(max_padded, dtype=torch.float32, device=device)
    s_eids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    nv = torch.empty(2, dtype=torch.int32, device=device)
    buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)
    return moe_softmax_sort_flydsl(
        gating_logits,
        s_ids,
        s_w,
        s_eids,
        nv,
        buf,
        E,
        topk,
        dtype_str,
        unit_size=unit_size,
        expert_mask=expert_mask,
        renormalize=renormalize,
    )


def _two_kernel_reference(
    gating_logits, E, topk, dtype_str, *, model_dim=4096, unit_size=UNIT_SIZE, expert_mask=None, renormalize=True
):
    """Run gating + sort as two separate kernels; return the same output
    tuple as the fused path. Used as the regression oracle for the fused
    kernel: anything the fused kernel produces must match what these two
    kernels produce on the same gating logits."""
    M = gating_logits.shape[0]
    device = gating_logits.device

    topk_weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    tei = torch.empty((M, topk), dtype=torch.int32, device=device)

    launch_topk = build_topk_gating_softmax_module(
        num_experts=E,
        topk=topk,
        dtype_str=dtype_str,
        renormalize=renormalize,
    )
    stream = torch.cuda.current_stream()
    launch_topk(gating_logits, topk_weights, topk_ids, tei, M, stream=stream)

    max_padded = M * topk + E * unit_size - topk
    max_blocks = (max_padded + unit_size - 1) // unit_size
    s_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    s_w = torch.empty(max_padded, dtype=torch.float32, device=device)
    s_eids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    nv = torch.empty(2, dtype=torch.int32, device=device)
    buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)
    return moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        s_ids,
        s_w,
        s_eids,
        nv,
        buf,
        E,
        unit_size,
        expert_mask=expert_mask,
    )


def _check_outputs_equal(ref_tuple, fused_tuple, *, topk, M, unit_size, label):
    """Compare the 5-tuple outputs of the two paths. Returns True on success.

    `sorted_ids` may legitimately differ in order within each expert's
    padded block (the sort is bag-of-tokens within an expert). We compare
    set-equality per expert block and exact equality for everything else.
    """
    ref_ids, ref_w, ref_eids, ref_nv, ref_buf = ref_tuple
    fused_ids, fused_w, fused_eids, fused_nv, fused_buf = fused_tuple

    passed = True

    nv_ok = torch.equal(ref_nv, fused_nv)
    print(
        f"  [{label}/num_valid_ids] ref={ref_nv.tolist()} fused={fused_nv.tolist()} " f"({'OK' if nv_ok else 'FAIL'})"
    )
    passed &= nv_ok

    num_padded = ref_nv[0].item()
    passed &= check_sorted_ids(ref_ids, fused_ids, num_padded, topk, M, f"{label}/sorted_ids")
    passed &= check_sorted_weights(
        ref_w,
        fused_w,
        ref_ids,
        topk,
        M,
        gpu_ids=fused_ids,
        num_padded=num_padded,
        label=f"{label}/sorted_weights",
    )
    # Both paths leave the trailing blocks of `sorted_expert_ids`
    # uninitialised (they're allocated via torch.empty in both the fused
    # and reference launchers), so compare only the in-range blocks.
    num_valid_blocks = num_padded // unit_size
    passed &= check_expert_ids(
        ref_eids,
        fused_eids,
        f"{label}/sorted_expert_ids",
        num_valid_blocks=num_valid_blocks,
    )

    buf_zero = (fused_buf.view(torch.int32) == 0).all().item()
    print(f"  [{label}/moe_buf_zeroed] {'OK' if buf_zero else 'FAIL'}")
    passed &= buf_zero

    return passed


def _run_softmax_sort_fused_test(T, E, topk, dtype_str, *, renormalize=True, unit_size=UNIT_SIZE, model_dim=4096):
    """Generate gating logits, run both paths, compare. Returns bool."""
    print(f"\n{'=' * 60}")
    print(f"Fused softmax_sort test: T={T}, E={E}, topk={topk}, " f"dtype={dtype_str}, renorm={renormalize}")
    print(f"{'=' * 60}")

    torch.manual_seed(42 + T * 1000 + E * 10 + topk + hash(dtype_str) % 100)
    torch_dtype = _TORCH_DTYPE[dtype_str]

    # Generate logits in fp32, then quantise to the kernel dtype so the
    # reference path sees identical bytes. Without this, bf16/f16 boundary
    # ties at the top-K cutoff can swing differently in fp32 vs quantised
    # arithmetic and produce spurious mismatches.
    gating_fp32 = torch.rand((T, E), device="cuda", dtype=torch.float32) * 4.0 - 2.0
    gating_dev = gating_fp32.to(torch_dtype).contiguous()

    fused_out = _call_softmax_sort_fused(
        gating_dev,
        E,
        topk,
        dtype_str,
        model_dim=model_dim,
        unit_size=unit_size,
        renormalize=renormalize,
    )
    ref_out = _two_kernel_reference(
        gating_dev,
        E,
        topk,
        dtype_str,
        model_dim=model_dim,
        unit_size=unit_size,
        renormalize=renormalize,
    )
    torch.cuda.synchronize()

    passed = _check_outputs_equal(
        ref_out,
        fused_out,
        topk=topk,
        M=T,
        unit_size=unit_size,
        label=f"fused(T={T},E={E},k={topk},{dtype_str})",
    )
    print(f"  >>> {'PASSED' if passed else 'FAILED'}")
    return passed


FUSED_ONESHOT_CONFIGS = [
    # (T, E, topk, dtype)
    (1, 256, 8, "bf16"),
    (4, 256, 8, "bf16"),
    (8, 256, 8, "bf16"),
    (16, 256, 8, "bf16"),
    (1, 128, 4, "bf16"),
    (8, 128, 4, "bf16"),
    (1, 32, 5, "bf16"),
    (4, 32, 5, "bf16"),
    # dtype coverage
    (8, 256, 8, "f16"),
    (8, 256, 8, "f32"),
    # Edge cases
    (7, 256, 8, "bf16"),  # M not a multiple of TOKENS_PER_BLOCK
    (13, 256, 8, "bf16"),  # arbitrary M < TOKENS_PER_BLOCK
]


@pytest.mark.parametrize("T,E,topk,dtype_str", FUSED_ONESHOT_CONFIGS)
def test_moe_softmax_sort_fused_oneshot(T, E, topk, dtype_str):
    assert _supports_fused_oneshot(E, topk, dtype_str), (
        f"Test config {E=}/{topk=}/{dtype_str=} not supported by fused oneshot; " "check FUSED_ONESHOT_CONFIGS"
    )
    passed = _run_softmax_sort_fused_test(T, E, topk, dtype_str)
    assert passed, (
        f"moe_softmax_sort_flydsl fused oneshot mismatch for " f"T={T}, E={E}, topk={topk}, dtype={dtype_str}"
    )


@pytest.mark.parametrize(
    "T,E,topk,dtype_str",
    [
        # M > FUSED_ONESHOT_MAX_T forces the fused entry to take its 2-kernel fallback
        # (separate gating + moe_sorting launches). The gating layout must
        # still be supported.
        (32, 256, 8, "bf16"),
        (64, 256, 8, "bf16"),
        (128, 256, 8, "bf16"),
        (1024, 256, 8, "bf16"),
    ],
)
def test_moe_softmax_sort_fallback(T, E, topk, dtype_str):
    """The fused-path entry must remain correct when it falls back to the
    2-kernel chain because M exceeds the oneshot bound."""
    passed = _run_softmax_sort_fused_test(T, E, topk, dtype_str)
    assert passed, (
        f"moe_softmax_sort_flydsl fallback path mismatch for " f"T={T}, E={E}, topk={topk}, dtype={dtype_str}"
    )


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


def bench_kernel_us(fn, warmup=BENCH_WARMUP, iters=BENCH_MEASURE):
    """Pure on-device kernel time (per invocation, microseconds).

    Uses ``torch.profiler`` (CUPTI on CUDA, roctracer on ROCm) to capture
    per-kernel begin/end timestamps from the GPU command processor itself.
    The returned figure sums every GPU kernel that ran during ``iters``
    invocations of ``fn`` and divides by ``iters``.

    Compared to ``bench_graph_us``:
      - ``bench_graph_us`` measures end-to-end CUDA-graph replay latency,
        which still includes graph-replay overhead and any inter-kernel
        dispatch gaps on the GPU command processor.
      - ``bench_kernel_us`` measures only the wall time the GPU is actually
        executing kernels — i.e. the floor on kernel runtime, with launch
        and dispatch effects removed.

    For multi-kernel paths (e.g. unfused gating + sort) this returns the
    sum of all per-kernel durations, which is the right comparison point
    for fusion: it isolates how much on-device compute fusion saved,
    independent of dispatch / scheduler effects.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
    except Exception:
        return None

    total_us = 0.0
    for k in prof.key_averages():
        # Sum events with non-zero on-device dwell. This naturally excludes
        # host-side stubs like hipModuleLaunchKernel / cudaLaunchKernel and
        # hipDeviceSynchronize / cudaDeviceSynchronize, whose self_device_time
        # is 0 because no work runs on the GPU under their name.
        sd = getattr(k, "self_device_time_total", 0.0)
        if sd > 0:
            total_us += sd
    return total_us / iters


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
    except (ImportError, AttributeError) as e:
        print(f"  aiter not available ({type(e).__name__}: {e}), skipping CK comparison")
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
    print(f"  CUs: {props.multi_processor_count}, oneshot threshold: T<={sub_tokens}")
    print(
        f"  Modes: eager (with L2 flush, median of {BENCH_MEASURE}), graph ({BENCH_MEASURE} replays), "
        f"kernel (on-device dwell)"
    )
    print(f"{'=' * 140}")
    print(
        f"{'T':>6s} | {'Path':>7s} | "
        f"{'FLY eager':>10s} | {'FLY graph':>10s} | {'FLY kern':>10s} | "
        f"{'CK eager':>10s} | {'CK graph':>10s} | {'CK kern':>10s} | "
        f"{'Eager':>7s} | {'Graph':>7s} | {'Kern':>7s}"
    )
    print("-" * 140)

    for T in token_sweep:
        torch.manual_seed(42)
        topk_ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(T)]).to(torch.int32)
        topk_weights = torch.rand(T, topk, dtype=torch.float32, device="cuda")

        path = "oneshot" if T <= sub_tokens else "multiphase"

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
        fly_kernel = bench_kernel_us(fly_fn)

        ck_eager, ck_graph, ck_kernel = None, None, None
        if aiter_moe_sorting is not None:

            def ck_fn():
                aiter_moe_sorting(
                    topk_ids, topk_weights, E, model_dim=model_dim, moebuf_dtype=torch.bfloat16, block_size=UNIT_SIZE
                )

            ck_eager = bench_eager_us(ck_fn)
            ck_graph = bench_graph_us(ck_fn)
            ck_kernel = bench_kernel_us(ck_fn)

        def fmt(v):
            return f"{v:8.1f}us" if v is not None else "       N/A"

        def ratio(a, b):
            if a is None or b is None or b == 0:
                return "    N/A"
            r = a / b
            return f"  {r:.2f}x"

        print(
            f"{T:>6d} | {path:>7s} | "
            f"{fmt(fly_eager)} | {fmt(fly_graph)} | {fmt(fly_kernel)} | "
            f"{fmt(ck_eager)} | {fmt(ck_graph)} | {fmt(ck_kernel)} | "
            f"{ratio(fly_eager, ck_eager)} | {ratio(fly_graph, ck_graph)} | "
            f"{ratio(fly_kernel, ck_kernel)}"
        )

    print("=" * 140)
    print("  Ratio < 1.0 = FlyDSL faster. Eager includes host launch overhead;")
    print("  Graph amortizes launch but includes dispatch gaps; Kern is pure on-GPU kernel time.")
    print()


def run_fused_bench_comparison(token_sweep=None, dtype_str="bf16", num_experts=256, topk=8, model_dim=4096):
    """Benchmark fused softmax+top-K+sort vs the unfused 2-kernel chain.

    Measures the routing stage (gating + sort) end-to-end across a range of
    M values, in eager mode, CUDA-graph mode, and pure on-device kernel
    time (profiler-based).

    Parameters
    ----------
    token_sweep : list[int] | None
        M values to sweep. Default: [1, 4, 8, 16, 32, 64].
    dtype_str   : 'bf16' | 'f16' | 'f32'.
    num_experts : E for the MoE router (e.g. 256 for DeepSeek R1).
    topk        : Experts per token (e.g. 8 for DeepSeek R1).
    model_dim   : Hidden size — sets `moe_buf` size, which controls the
                  blocks-1..N zero-pass cost portion of the kernel
                  (DeepSeek R1: 7168).
    """
    E = num_experts
    if token_sweep is None:
        # Decode regime is where fusion applies. Include a few sizes above
        # FUSED_ONESHOT_MAX_T=16 so the fallback path is also exercised.
        token_sweep = [1, 4, 8, 16, 32, 64]

    torch_dtype = _TORCH_DTYPE[dtype_str]

    print(f"\n{'=' * 145}")
    print(
        f"  MoE Fused Routing Benchmark: fused vs (gating + moe_sorting) "
        f"(E={E}, topk={topk}, dtype={dtype_str}, "
        f"model_dim={model_dim}, unit_size={UNIT_SIZE})"
    )
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(
        f"  Modes: eager (with L2 flush, median of {BENCH_MEASURE}), "
        f"graph ({BENCH_MEASURE} replays), kernel (sum of on-device kernel time)"
    )
    print(f"{'=' * 145}")
    print(
        f"{'T':>6s} | {'Path':>9s} | "
        f"{'unfused eager':>14s} | {'fused eager':>13s} | "
        f"{'unfused graph':>14s} | {'fused graph':>13s} | "
        f"{'unfused kern':>13s} | {'fused kern':>12s} | "
        f"{'Eager':>7s} | {'Graph':>7s} | {'Kern':>7s}"
    )
    print("-" * 145)

    # Cache the standalone gating launcher so the unfused path doesn't pay
    # compile time inside the measured region.
    launch_topk = build_topk_gating_softmax_module(
        num_experts=E,
        topk=topk,
        dtype_str=dtype_str,
        renormalize=True,
    )

    for T in token_sweep:
        torch.manual_seed(42)
        gating_logits = (
            (torch.rand((T, E), device="cuda", dtype=torch.float32) * 4.0 - 2.0).to(torch_dtype).contiguous()
        )

        max_num_tokens_padded = T * topk + E * UNIT_SIZE - topk
        max_num_m_blocks = (max_num_tokens_padded + UNIT_SIZE - 1) // UNIT_SIZE
        sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device="cuda")
        sorted_w = torch.empty(max_num_tokens_padded, dtype=torch.float32, device="cuda")
        sorted_eids = torch.empty(max_num_m_blocks, dtype=torch.int32, device="cuda")
        nvalid = torch.empty(2, dtype=torch.int32, device="cuda")
        moe_buf_2d = torch.empty((T, model_dim), dtype=torch.bfloat16, device="cuda")

        # Unfused: separate gating + sort tensors
        u_topk_w = torch.empty((T, topk), dtype=torch.float32, device="cuda")
        u_topk_ids = torch.empty((T, topk), dtype=torch.int32, device="cuda")
        u_tei = torch.empty((T, topk), dtype=torch.int32, device="cuda")

        def unfused_fn():
            stream = torch.cuda.current_stream()
            launch_topk(gating_logits, u_topk_w, u_topk_ids, u_tei, T, stream=stream)
            moe_sorting_flydsl(
                u_topk_ids,
                u_topk_w,
                sorted_ids,
                sorted_w,
                sorted_eids,
                nvalid,
                moe_buf_2d,
                E,
                UNIT_SIZE,
            )

        def fused_fn():
            moe_softmax_sort_flydsl(
                gating_logits,
                sorted_ids,
                sorted_w,
                sorted_eids,
                nvalid,
                moe_buf_2d,
                E,
                topk,
                dtype_str,
                unit_size=UNIT_SIZE,
            )

        # Warm up both paths once before measurement (covers compile cache).
        unfused_fn()
        fused_fn()
        torch.cuda.synchronize()

        unfused_eager = bench_eager_us(unfused_fn)
        fused_eager = bench_eager_us(fused_fn)
        unfused_graph = bench_graph_us(unfused_fn)
        fused_graph = bench_graph_us(fused_fn)
        unfused_kernel = bench_kernel_us(unfused_fn)
        fused_kernel = bench_kernel_us(fused_fn)

        path = "fused" if T <= 16 else "fallback"

        def fmt(v, w=12):
            return f"{v:{w}.1f}us" if v is not None else f"{'N/A':>{w + 2}s}"

        def ratio(unfused, fused):
            if unfused is None or fused is None or fused == 0:
                return "    N/A"
            # Speedup: how much faster is fused vs unfused? >1.0 = fused wins.
            r = unfused / fused
            return f"  {r:.2f}x"

        print(
            f"{T:>6d} | {path:>9s} | "
            f"{fmt(unfused_eager)} | {fmt(fused_eager)} | "
            f"{fmt(unfused_graph)} | {fmt(fused_graph)} | "
            f"{fmt(unfused_kernel, 11)} | {fmt(fused_kernel, 10)} | "
            f"{ratio(unfused_eager, fused_eager)} | "
            f"{ratio(unfused_graph, fused_graph)} | "
            f"{ratio(unfused_kernel, fused_kernel)}"
        )

    print("=" * 145)
    print("  Ratio > 1.0 = fused faster. Eager includes host launch overhead (2 launches vs 1);")
    print("  Graph amortizes launch but still includes inter-kernel dispatch gaps;")
    print("  Kern is pure on-GPU kernel time (sum of per-kernel device dwell, via torch.profiler).")
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
    parser.add_argument("--bench", action="store_true", help="Run benchmark sweep (eager + graph + kern, FlyDSL vs CK)")
    parser.add_argument(
        "--bench-fused", action="store_true", help="Run fused-vs-unfused benchmark for moe_softmax_sort_flydsl"
    )
    parser.add_argument(
        "--bench-tokens", type=str, default=None, help="Comma-separated T values for bench (default: all)"
    )
    parser.add_argument("--bench-experts", type=int, default=256, help="Num experts E for --bench-fused (default 256)")
    parser.add_argument("--bench-topk", type=int, default=8, help="topk for --bench-fused (default 8)")
    parser.add_argument(
        "--bench-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "f16", "f32"],
        help="Gating-logits dtype for --bench-fused (default bf16)",
    )
    parser.add_argument(
        "--bench-model-dim",
        type=int,
        default=4096,
        help="Hidden size (moe_buf width) for --bench-fused " "(DeepSeek R1: 7168; default 4096)",
    )
    args = parser.parse_args()

    if args.bench:
        token_sweep = None
        if args.bench_tokens:
            token_sweep = [int(t) for t in args.bench_tokens.split(",")]
        run_bench_comparison(token_sweep=token_sweep)
        return

    if args.bench_fused:
        token_sweep = None
        if args.bench_tokens:
            token_sweep = [int(t) for t in args.bench_tokens.split(",")]
        run_fused_bench_comparison(
            token_sweep=token_sweep,
            dtype_str=args.bench_dtype,
            num_experts=args.bench_experts,
            topk=args.bench_topk,
            model_dim=args.bench_model_dim,
        )
        return

    if args.T is not None:
        E = args.E or 256
        topk = args.topk or 8
        configs = [(args.T, E, topk)]
    elif args.all:
        configs = ONESHOT_CONFIGS + MULTIPHASE_CONFIGS
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
