#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel correctness tests.

Calls ``build_fused_rope_cache_module`` directly (no aiter wrapper) and
validates Q/K rotation outputs and KV cache writes against a pure-PyTorch
reference.  When AITER is available (installed or reachable via AITER_REPO),
results are also cross-checked against the Triton reference implementation.

Test dimensions covered
-----------------------
Model configs (QH, KH, D):
  - Llama-8B  TP1: QH=32, KH=8,  D=128
  - Llama-8B  TP8: QH=4,  KH=1,  D=128
  - Llama-70B TP1: QH=64, KH=8,  D=128
  - Llama-70B TP8: QH=8,  KH=1,  D=128
  - Llama-405B TP1: QH=128,KH=8, D=128
  - Llama-405B TP8: QH=16, KH=1, D=128
  - Qwen3-72B TP1: QH=64, KH=4,  D=128
  - Qwen3-72B TP8: QH=8,  KH=1,  D=128
  - GPT-OSS   TP1: QH=64, KH=8,  D=64
  - GPT-OSS   TP8: QH=8,  KH=1,  D=64

Token counts: T=1 (decode), T=32, T=128 (prefill)
KV cache layouts: flash_layout=True / False
Scale: apply_scale=True (fp8 cache) / False (bf16/f16 cache)
Position dtype: i32 / i64 (i64 uses .view(i32) stride-2 indexing)
Cos/sin dim: reuse_freqs_front_part=True (half-dim) / False (full-dim)

Usage
-----
    # Fast CI — correctness only (GPT-OSS 120B TP=8, 10 tests):
    PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s

    # All models × TPs (multi-model sweep):
    FLYDSL_ALL_MODELS=1 PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s

    # With benchmarking + optional AITER comparison:
    FLYDSL_BENCH=1 AITER_REPO=../aiter PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s

    # CLI — all models:
    PYTHONPATH=./ python tests/kernels/test_fused_rope_cache.py --all-models

    # CLI — with benchmark + AITER comparison:
    FLYDSL_BENCH=1 AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/test_fused_rope_cache.py --all-models
"""

import os
import sys

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch as _get_rocm_arch
from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module

# ---------------------------------------------------------------------------
# Skip if no GPU
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

# ---------------------------------------------------------------------------
# Optional AITER Triton cross-check
# AITER_REPO env var: path to aiter repo root (added to sys.path if set).
# Falls back to installed aiter package.
# ---------------------------------------------------------------------------
_AITER_REPO = os.environ.get("AITER_REPO", "")
if _AITER_REPO and _AITER_REPO not in sys.path:
    sys.path.insert(0, _AITER_REPO)

try:
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache as _aiter_rope

    HAS_AITER = True
except ImportError:
    HAS_AITER = False


def _bench_gpu_us(fn, warmup: int = 20, iters: int = 200) -> float:
    """Measure GPU kernel time via CUDA events (true device time, no Python-loop overhead)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # ms → µs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 16

# fp8 dtype: gfx95x (MI350/MI355X) uses e4m3fn; gfx94x (MI300X) uses e4m3fnuz.
# RDNA (gfx10xx/gfx11xx/gfx12xx): fp8 KV cache is a CDNA production feature;
# cvt_pk_fp8_f32 produces a different bit encoding on RDNA, so fp8 cache tests
# are skipped there to avoid false failures from dtype mismatches.
_ARCH = str(_get_rocm_arch())
_IS_RDNA = not _ARCH.startswith("gfx9")
FP8_DTYPE = torch.float8_e4m3fn if "gfx95" in _ARCH else torch.float8_e4m3fnuz
MAX_POS = 8192
X_SIZE = 16  # x-pack factor in non-flash key cache layout

# Default atol per dtype
_ATOL = {"bf16": 1e-2, "f16": 5e-3}

# ---------------------------------------------------------------------------
# Kernel compilation cache
# Keyed by all build-time parameters so each unique config compiles once.
# ---------------------------------------------------------------------------
_kernel_cache: dict = {}


def _get_launch_fn(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
    flash_layout: bool,
    dtype_str: str,
    apply_scale: bool,
    reuse_freqs_front_part: bool,
    pos_dtype: str,
):
    key = (head_dim, num_q_heads, num_kv_heads, block_size,
           flash_layout, dtype_str, apply_scale, reuse_freqs_front_part, pos_dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = build_fused_rope_cache_module(
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            is_neox=True,
            flash_layout=flash_layout,
            dtype_str=dtype_str,
            apply_scale=apply_scale,
            reuse_freqs_front_part=reuse_freqs_front_part,
            pos_dtype=pos_dtype,
        )
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _rope_ref(q, k, v, cos_cache, sin_cache, positions, slot_mapping,
               key_cache, value_cache, block_size, flash_layout,
               reuse_freqs_front_part):
    """Pure-PyTorch NeoX RoPE + KV cache reference.

    Operates in native dtype (bf16/f16) to match GPU hardware rounding.
    Half-dim cos/sin are broadcast over the full head as [cos, cos] / [sin, sin].
    """
    half_dim = cos_cache.shape[-1]  # D//2 when reuse_freqs=True, else D
    dtype = q.dtype

    # Index into cos/sin cache by position
    cos = cos_cache[positions.long()].unsqueeze(1).to(dtype)  # [T, 1, cols]
    sin = sin_cache[positions.long()].unsqueeze(1).to(dtype)

    # Expand half-dim to full-dim if reuse_freqs_front_part=True
    if reuse_freqs_front_part:
        # cos/sin shape: [T, 1, D//2] → replicate to [T, 1, D]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    # NeoX rotation: q_out = [q1*cos - q2*sin,  q2*cos + q1*sin]
    head_dim = q.shape[-1]
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]

    q_out = torch.cat([q1 * cos[..., :head_dim // 2] - q2 * sin[..., :head_dim // 2],
                       q2 * cos[..., head_dim // 2:] + q1 * sin[..., head_dim // 2:]], dim=-1)
    k_out = torch.cat([k1 * cos[..., :head_dim // 2] - k2 * sin[..., :head_dim // 2],
                       k2 * cos[..., head_dim // 2:] + k1 * sin[..., head_dim // 2:]], dim=-1)

    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()

    for i, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        bi = slot // block_size
        bp = slot % block_size
        if flash_layout:
            key_cache_out[bi, bp] = k_out[i]
            value_cache_out[bi, bp] = v[i]
        else:
            # key_cache: [num_blocks, KH, D//x, block_size, x]
            k_row = k_out[i]  # [KH, D]
            key_cache_out[bi, :, :, bp, :] = k_row.view(k_row.shape[0], k_row.shape[1] // X_SIZE, X_SIZE)
            # value_cache: [num_blocks, KH, D, block_size]
            value_cache_out[bi, :, :, bp] = v[i]

    return q_out, k_out, key_cache_out, value_cache_out


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def run_test(
    num_tokens: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
    apply_scale: bool = False,
    reuse_freqs_front_part: bool = True,
    pos_dtype: str = "i32",
    negative_slots: bool = False,
    block_size: int = BLOCK_SIZE,
    max_pos: int = MAX_POS,
    bench: bool = False,
):
    """Build kernel, run it, and compare against reference (and AITER if available).

    Returns (passed, max_errors_dict).
    When bench=True or FLYDSL_BENCH=1, also prints FlyDSL vs AITER timing.
    """
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    num_blocks = max(32, (num_tokens + block_size - 1) // block_size + 4)
    half_dim = head_dim // 2
    cos_sin_cols = half_dim if reuse_freqs_front_part else head_dim

    launch_fn = _get_launch_fn(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        flash_layout=flash_layout,
        dtype_str=dtype_str,
        apply_scale=apply_scale,
        reuse_freqs_front_part=reuse_freqs_front_part,
        pos_dtype=pos_dtype,
    )

    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_q_heads, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
    cos_cache = torch.randn(max_pos, cos_sin_cols, device=device, dtype=torch_dtype)
    sin_cache = torch.randn(max_pos, cos_sin_cols, device=device, dtype=torch_dtype)

    # Positions: i32 or i64 (i64 stored as int64 but kernel reads via stride-2 i32 view)
    positions_i32 = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    if pos_dtype == "i64":
        # The kernel expects positions as int64 tensor but reads each element
        # as two consecutive i32 words, taking only the low word (little-endian).
        positions_tensor = positions_i32.to(torch.int64)
    else:
        positions_tensor = positions_i32

    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
    if negative_slots:
        slot_mapping[1::2] = -1

    if pos_dtype == "i64":
        slot_mapping_tensor = slot_mapping.to(torch.int64)
    else:
        slot_mapping_tensor = slot_mapping

    if flash_layout:
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                                 device=device, dtype=torch_dtype)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                                   device=device, dtype=torch_dtype)
    else:
        key_cache = torch.zeros(num_blocks, num_kv_heads, head_dim // X_SIZE, block_size, X_SIZE,
                                 device=device, dtype=torch_dtype)
        value_cache = torch.zeros(num_blocks, num_kv_heads, head_dim, block_size,
                                   device=device, dtype=torch_dtype)

    if apply_scale:
        # fp8 cache: allocate as fp8 type for storage, but kernel uses raw buffer_ops.
        # Scales must be 1-D tensors (FlyDSL requires at least one dimension).
        kc_fp8 = torch.zeros_like(key_cache).to(FP8_DTYPE)
        vc_fp8 = torch.zeros_like(value_cache).to(FP8_DTYPE)
        kv_scale = 0.1  # round-trip friendly: maps bf16 range into fp8 range
        k_scale = torch.tensor([kv_scale], dtype=torch.float32, device=device)
        v_scale = torch.tensor([kv_scale], dtype=torch.float32, device=device)
    else:
        kc_fp8 = key_cache
        vc_fp8 = value_cache
        k_scale = torch.ones(1, dtype=torch.float32, device=device)
        v_scale = torch.ones(1, dtype=torch.float32, device=device)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    stream = torch.cuda.current_stream()
    launch_fn(
        q, k, v,
        positions_tensor, cos_cache, sin_cache,
        slot_mapping_tensor,
        kc_fp8, vc_fp8,
        q_out, k_out,
        num_tokens,
        k_scale, v_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    # Reference (bf16/f16 path only — fp8 correctness checked separately)
    q_ref, k_ref, kc_ref, vc_ref = _rope_ref(
        q, k, v,
        cos_cache, sin_cache,
        positions_i32,  # always i32 for reference indexing
        slot_mapping,   # always i32
        key_cache.clone(), value_cache.clone(),
        block_size,
        flash_layout=flash_layout,
        reuse_freqs_front_part=reuse_freqs_front_part,
    )

    atol = _ATOL[dtype_str]
    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()

    if not apply_scale:
        kc_err = (kc_fp8.float() - kc_ref.float()).abs().max().item()
        vc_err = (vc_fp8.float() - vc_ref.float()).abs().max().item()
        passed = q_err < atol and k_err < atol and kc_err < atol and vc_err < atol
        errs = {"q": q_err, "k": k_err, "kc": kc_err, "vc": vc_err}
    else:
        # fp8: dequantize the written cache with per-tensor scales and compare
        # against the bf16 reference. This catches packing/indexing/scaling bugs
        # and validates that negative-slot entries are left unchanged (the reference
        # preserves zeros for skipped slots, so kc_ref/vc_ref already encodes that).
        kc_deq = kc_fp8.to(torch.float32) * k_scale.float()
        vc_deq = vc_fp8.to(torch.float32) * v_scale.float()
        kc_err = (kc_deq - kc_ref.float()).abs().max().item()
        vc_err = (vc_deq - vc_ref.float()).abs().max().item()
        # fp8 e4m3 quantization error bound: 0.5 * (binade step at the max stored value).
        # For a value v stored with scale s, the fp8 input is v/s. The binade step at
        # x is x * 2^(1-mbits) = x/4 for e4m3 (3 mantissa bits). Dequant error ≤
        # 0.5 * (v/s)/4 * s = v/8. So max error ≤ max(|kc_ref|) / 8.
        kc_max = kc_ref.float().abs().max().item()
        vc_max = vc_ref.float().abs().max().item()
        kc_atol = max(1e-3, kc_max / 8.0)
        vc_atol = max(1e-3, vc_max / 8.0)
        passed = q_err < atol and k_err < atol and kc_err < kc_atol and vc_err < vc_atol
        errs = {"q": q_err, "k": k_err, "kc": kc_err, "vc": vc_err,
                "kc_atol": kc_atol, "vc_atol": vc_atol}

    do_bench = bench or os.environ.get("FLYDSL_BENCH", "0") == "1"

    # AITER cross-check (and optional benchmark)
    if HAS_AITER and not negative_slots and not apply_scale:
        # AITER Triton wrapper expects int64 slots/positions and 4D cos/sin
        slots_i64 = slot_mapping.to(torch.int64)
        pos_i64 = positions_i32.to(torch.int64)
        cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)  # [max_pos, 1, 1, cols]
        sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)

        kc_aiter = key_cache.clone().zero_()
        vc_aiter = value_cache.clone().zero_()
        q_aiter = torch.empty_like(q)
        k_aiter = torch.empty_like(k)

        _aiter_rope(
            q, k, v, kc_aiter, vc_aiter,
            slots_i64, pos_i64, cos_4d, sin_4d,
            k_scale, v_scale,
            is_neox=True, flash_layout=flash_layout,
            apply_scale=False, offs=None,
            q_out=q_aiter, k_out=k_aiter, output_zeros=False,
        )
        torch.cuda.synchronize()

        q_vs_aiter = (q_out.float() - q_aiter.float()).abs().max().item()
        k_vs_aiter = (k_out.float() - k_aiter.float()).abs().max().item()
        kc_vs_aiter = (kc_fp8.float() - kc_aiter.float()).abs().max().item()
        vc_vs_aiter = (vc_fp8.float() - vc_aiter.float()).abs().max().item()
        errs["aiter_q"] = q_vs_aiter
        errs["aiter_k"] = k_vs_aiter
        errs["aiter_kc"] = kc_vs_aiter
        errs["aiter_vc"] = vc_vs_aiter

        if do_bench:
            def _run_fly():
                launch_fn(
                    q, k, v, positions_tensor, cos_cache, sin_cache,
                    slot_mapping_tensor, kc_fp8, vc_fp8, q_out, k_out,
                    num_tokens, k_scale, v_scale,
                    stream=torch.cuda.current_stream(),
                )

            def _run_aiter():
                _aiter_rope(
                    q, k, v, kc_aiter, vc_aiter,
                    slots_i64, pos_i64, cos_4d, sin_4d,
                    k_scale, v_scale,
                    is_neox=True, flash_layout=flash_layout,
                    apply_scale=False, offs=None,
                    q_out=q_aiter, k_out=k_aiter, output_zeros=False,
                )

            fly_us = _bench_gpu_us(_run_fly)
            aiter_us = _bench_gpu_us(_run_aiter)
            speedup = aiter_us / fly_us if fly_us > 0 else 0.0
            errs["fly_us"] = fly_us
            errs["aiter_us"] = aiter_us
            errs["speedup"] = speedup

    return passed, errs


# ===========================================================================
# Category 1: Core decode configs (T=1) — fast CI gate
# ===========================================================================

@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (32, 8, 128),   # Llama-8B TP1
    (4,  1, 128),   # Llama-8B TP8
    (64, 8, 128),   # Llama-70B TP1
    (8,  1, 128),   # Llama-70B TP8
    (64, 8,  64),   # GPT-OSS TP1
    (8,  1,  64),   # GPT-OSS TP8
], ids=[
    "Llama8B-TP1", "Llama8B-TP8",
    "Llama70B-TP1", "Llama70B-TP8",
    "GPTOSS-TP1", "GPTOSS-TP8",
])
def test_decode_flash(num_q_heads, num_kv_heads, head_dim):
    """T=1 decode, flash layout, bf16 — core correctness gate."""
    passed, errs = run_test(
        num_tokens=1, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=True, dtype_str="bf16",
    )
    assert passed, f"FAILED: {errs}"


# ===========================================================================
# Category 2: Flash layout, all token sizes, bf16
# ===========================================================================

@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (32,  8, 128),   # Llama-8B TP1
    (4,   1, 128),   # Llama-8B TP8
    (64,  8, 128),   # Llama-70B TP1
    (8,   1, 128),   # Llama-70B TP8
    (128, 8, 128),   # Llama-405B TP1
    (16,  1, 128),   # Llama-405B TP8
    (64,  4, 128),   # Qwen3-72B TP1
    (8,   1, 128),   # Qwen3-72B TP8 (same shape as Llama-70B TP8)
    (64,  8,  64),   # GPT-OSS TP1
    (8,   1,  64),   # GPT-OSS TP8
], ids=[
    "Llama8B-TP1", "Llama8B-TP8",
    "Llama70B-TP1", "Llama70B-TP8",
    "Llama405B-TP1", "Llama405B-TP8",
    "Qwen72B-TP1", "Qwen72B-TP8",
    "GPTOSS-TP1", "GPTOSS-TP8",
])
def test_flash_bf16(num_tokens, num_q_heads, num_kv_heads, head_dim):
    """Flash layout, bf16, all supported model configs and token sizes."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=True, dtype_str="bf16",
    )
    assert passed, f"FAILED (T={num_tokens}): {errs}"


# ===========================================================================
# Category 3: Non-flash layout
# ===========================================================================

@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (32, 8, 128),   # Llama-8B TP1
    (4,  1, 128),   # Llama-8B TP8
    (8,  1, 128),   # Llama-70B TP8
    (8,  1,  64),   # GPT-OSS TP8
], ids=["Llama8B-TP1", "Llama8B-TP8", "Llama70B-TP8", "GPTOSS-TP8"])
def test_nonflash_bf16(num_tokens, num_q_heads, num_kv_heads, head_dim):
    """Non-flash (ATOM-default) layout, bf16."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=False, dtype_str="bf16",
    )
    assert passed, f"FAILED (T={num_tokens}): {errs}"


# ===========================================================================
# Category 4: f16 dtype
# ===========================================================================

@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_f16(num_tokens, flash_layout):
    """f16 dtype — Llama-8B TP8 representative config."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=128,
        num_q_heads=4, num_kv_heads=1,
        flash_layout=flash_layout, dtype_str="f16",
    )
    assert passed, f"FAILED (T={num_tokens} flash={flash_layout}): {errs}"


# ===========================================================================
# Category 5: pos_dtype — i32 vs i64 (stride-2 indexing)
# ===========================================================================

@pytest.mark.parametrize("pos_dtype", ["i32", "i64"])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_pos_dtype(pos_dtype, num_tokens):
    """Position tensor dtype: i32 direct vs i64 stride-2 view.

    The i64 path reads each int64 as two consecutive i32 words (low word only),
    which is the same physical value on little-endian AMD GPUs.
    """
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=128,
        num_q_heads=8, num_kv_heads=1,
        flash_layout=True, dtype_str="bf16",
        pos_dtype=pos_dtype,
    )
    assert passed, f"FAILED (pos_dtype={pos_dtype} T={num_tokens}): {errs}"


# ===========================================================================
# Category 6: reuse_freqs_front_part — half-dim vs full-dim cos/sin
# ===========================================================================

@pytest.mark.parametrize("reuse_freqs_front_part", [True, False],
                          ids=["half_dim", "full_dim"])
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_reuse_freqs(reuse_freqs_front_part, num_tokens, flash_layout):
    """Cos/sin shape: half-dim [max_pos, D//2] vs full-dim [max_pos, D]."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=128,
        num_q_heads=8, num_kv_heads=1,
        flash_layout=flash_layout, dtype_str="bf16",
        reuse_freqs_front_part=reuse_freqs_front_part,
    )
    assert passed, f"FAILED (reuse={reuse_freqs_front_part} T={num_tokens}): {errs}"


# ===========================================================================
# Category 7: Negative slots (slot < 0 skips KV cache write)
# ===========================================================================

@pytest.mark.parametrize("num_tokens", [4, 32])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_negative_slots(num_tokens, flash_layout):
    """Odd-indexed slots set to -1; those KV cache positions must remain zero."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=128,
        num_q_heads=8, num_kv_heads=1,
        flash_layout=flash_layout, dtype_str="bf16",
        negative_slots=True,
    )
    assert passed, f"FAILED (T={num_tokens} flash={flash_layout}): {errs}"


# ===========================================================================
# Category 8: fp8 KV cache (apply_scale=True) — finite-value sanity check
# ===========================================================================

@pytest.mark.skipif(_IS_RDNA, reason="fp8 KV cache is a CDNA production feature; "
                    "cvt_pk_fp8_f32 bit encoding differs on RDNA (gfx10xx/gfx11xx/gfx12xx)")
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (8, 1,  64),   # GPT-OSS TP8
    (8, 1, 128),   # Llama TP8
], ids=["GPTOSS-TP8", "Llama-TP8"])
def test_fp8_cache(num_tokens, flash_layout, num_q_heads, num_kv_heads, head_dim):
    """fp8 KV cache path: Q/K rotation correct, cache values finite."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=flash_layout, dtype_str="bf16",
        apply_scale=True,
    )
    assert passed, f"FAILED (T={num_tokens} flash={flash_layout}): {errs}"


# ===========================================================================
# Category 9: Cross-parameter sweep (opt-in via FLYDSL_ALL_MODELS=1)
# ===========================================================================

_ALL_CONFIGS = [
    ("Llama8B-TP1",   32,  8, 128),
    ("Llama8B-TP8",    4,  1, 128),
    ("Llama70B-TP1",  64,  8, 128),
    ("Llama70B-TP8",   8,  1, 128),
    ("Llama405B-TP1", 128, 8, 128),
    ("Llama405B-TP8",  16, 1, 128),
    ("Qwen72B-TP1",   64,  4, 128),
    ("Qwen72B-TP8",    8,  1, 128),
    ("GPTOSS-TP1",    64,  8,  64),
    ("GPTOSS-TP8",     8,  1,  64),
]


@pytest.mark.parametrize("model,num_q_heads,num_kv_heads,head_dim",
                          _ALL_CONFIGS, ids=[c[0] for c in _ALL_CONFIGS])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
@pytest.mark.parametrize("reuse_freqs_front_part", [True, False], ids=["half_cos", "full_cos"])
@pytest.mark.parametrize("pos_dtype", ["i32", "i64"])
@pytest.mark.skipif(os.environ.get("FLYDSL_ALL_MODELS", "0") != "1",
                    reason="Full sweep skipped; set FLYDSL_ALL_MODELS=1 to run")
def test_full_sweep(model, num_q_heads, num_kv_heads, head_dim,
                    num_tokens, flash_layout, reuse_freqs_front_part, pos_dtype):
    """Cross-parameter correctness sweep over all models × layouts × dtypes × pos_dtype."""
    passed, errs = run_test(
        num_tokens=num_tokens, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=flash_layout, dtype_str="bf16",
        reuse_freqs_front_part=reuse_freqs_front_part,
        pos_dtype=pos_dtype,
    )
    assert passed, (f"FAILED ({model} T={num_tokens} flash={flash_layout} "
                    f"reuse={reuse_freqs_front_part} pos={pos_dtype}): {errs}")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    all_models = "--all-models" in sys.argv
    do_bench = os.environ.get("FLYDSL_BENCH", "0") == "1"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AITER cross-check: {'enabled' if HAS_AITER else 'disabled (set AITER_REPO or install aiter)'}")
    print(f"Benchmark: {'enabled' if do_bench else 'disabled (set FLYDSL_BENCH=1)'}")
    failures = 0

    def _run(label, **kwargs):
        global failures
        passed, errs = run_test(**kwargs, bench=do_bench)
        status = "PASS" if passed else "FAIL"
        bench_str = ""
        if "fly_us" in errs:
            bench_str = (f"  FlyDSL={errs['fly_us']:.1f}us "
                         f"AITER={errs['aiter_us']:.1f}us "
                         f"speedup={errs['speedup']:.2f}x")
        kc_str = f" kc={errs['kc']:.4f}" if "kc" in errs else ""
        vc_str = f" vc={errs['vc']:.4f}" if "vc" in errs else ""
        print(f"  [{status}] {label}: q={errs['q']:.4f} k={errs['k']:.4f}{kc_str}{vc_str}{bench_str}")
        if not passed:
            failures += 1

    configs = _ALL_CONFIGS if all_models else [
        ("GPTOSS-TP8", 8, 1, 64),
    ]

    print("\n=== Category 1: decode (T=1) flash bf16 ===")
    for name, qh, kh, hd in configs:
        _run(f"{name} T=1", num_tokens=1, head_dim=hd,
             num_q_heads=qh, num_kv_heads=kh, flash_layout=True)

    print("\n=== Category 2: prefill (T=32, T=128) flash bf16 ===")
    for name, qh, kh, hd in configs:
        for T in [32, 128]:
            _run(f"{name} T={T}", num_tokens=T, head_dim=hd,
                 num_q_heads=qh, num_kv_heads=kh, flash_layout=True)

    print("\n=== Category 3: non-flash bf16 ===")
    nonflash = _ALL_CONFIGS if all_models else [("Llama8B-TP8", 4, 1, 128), ("GPTOSS-TP8", 8, 1, 64)]
    for name, qh, kh, hd in nonflash:
        for T in [1, 32]:
            _run(f"{name} T={T}", num_tokens=T, head_dim=hd,
                 num_q_heads=qh, num_kv_heads=kh, flash_layout=False)

    print("\n=== Category 5: pos_dtype i32 vs i64 ===")
    for pos_dtype in ["i32", "i64"]:
        for T in [1, 32, 128]:
            _run(f"pos_dtype={pos_dtype} T={T}", num_tokens=T, head_dim=128,
                 num_q_heads=8, num_kv_heads=1, flash_layout=True, pos_dtype=pos_dtype)

    print("\n=== Category 6: reuse_freqs_front_part ===")
    for reuse in [True, False]:
        for flash in [True, False]:
            _run(f"reuse={reuse} flash={flash}", num_tokens=32, head_dim=128,
                 num_q_heads=8, num_kv_heads=1, flash_layout=flash,
                 reuse_freqs_front_part=reuse)

    print("\n=== Category 8: fp8 cache ===")
    for flash in [True, False]:
        for T in [1, 32]:
            _run(f"fp8 flash={flash} T={T}", num_tokens=T, head_dim=128,
                 num_q_heads=8, num_kv_heads=1, flash_layout=flash, apply_scale=True)

    print(f"\n{'='*60}")
    if failures == 0:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"{failures} TESTS FAILED")
        sys.exit(1)
