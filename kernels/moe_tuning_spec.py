# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Locked specification for the MXFP4 MoE 2-stage tuning campaign on gfx950.

This is the single source of truth for the campaign's fixed parameters: the
target model shapes, the token sweep grid, the measurement protocol, the
win/no-regression predicates, the MFU denominator, and the routing-distribution
set used in correctness checks.  The measurement harness and the (later)
shape->config dispatch both import from here so the numbers live in exactly one
place.

All values are fixed inputs locked by the user before the campaign began; do not
change them as part of tuning.  Tuning changes tile configs, not these gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# --- MFU denominator -------------------------------------------------------
# Empirically measured fp4 GEMM ceiling on the target MI350X (gfx950, 256 CU,
# sclk max 2200 MHz).  MFU = effective_TFLOPS / FP4_PEAK_TFLOPS.
FP4_PEAK_TFLOPS = 4523.0

# --- Win margins (the win-margin policy) ---------------------------------------------------
WIN_MARGIN = 0.10  # 10% relative improvement required to claim a win.
# Large-shape (tokens >= LARGE_TOKEN_MIN): tuned_MFU >= baseline_MFU * (1 + WIN_MARGIN).
# Small-token (tokens <= SMALL_TOKEN_MAX): tuned_us <= baseline_us * (1 - WIN_MARGIN)
#   AND (baseline_us - tuned_us) >= ABS_US_BAND.

# --- No-regression tolerance + protocol (the no-regression policy) ----------------------------
REGRESSION_REL = 0.02  # 2% relative.
ABS_US_BAND = 2.0  # microseconds; default absolute floor (tokens >= 128).

# Regime-aware absolute floor (user-approved amendment).  On this shared node the
# small/low-token absolute latency is tiny (~30-300 us) and run-to-run jitter is
# ~3-7 us even after the in-protocol controls are exhausted (faithful L2-flush
# argument rotation, repeated measurement, AND harness-verified clock pinning).
# This is irreducible measurement noise at tiny absolute latency, not a harness
# defect: under the 8 us small-token floor the residual a4w4 repeatability
# instability is confined to a single mid-token point (token 128, under the strict
# 2 us tokens>=128 floor) plus the e2e guardrail outlier (token 64) -- i.e. the
# small-token (<=64) kernel-path band is satisfied; tokens >= 128 keep the strict
# 2 us floor.  8 us is still far below the small-token win threshold (>= 10% AND
# >= 2 us; 10% of even the smallest ~127 us point is ~12.7 us), so widening the
# band does NOT weaken win detection.  Floor is regime-aware: 8 us for
# tokens <= SMALL_TOKEN_MAX, 2 us otherwise.
SMALL_TOKEN_ABS_US_BAND = 8.0


def abs_floor_us(token: int) -> float:
    """Regime-aware absolute floor for the no-regression / repeatability band.

    8 us for the small-token regime (tokens <= SMALL_TOKEN_MAX), 2 us otherwise.
    Used together with the 2% relative term as ``max(2%, abs_floor_us(token))``.
    """
    return SMALL_TOKEN_ABS_US_BAND if token <= SMALL_TOKEN_MAX else ABS_US_BAND


WARMUP_ITERS = 10
BENCH_ITERS = 100
# Reported statistics per point.
REPORT_STATS = ("median", "p95")
# Protocol flags (recorded with every measurement; runs under other settings are
# non-comparable).
GRAPH_CAPTURE = False
L2_FLUSH_PER_ITER = True
CLOCKS_PINNED = True

# --- Token regimes (the win-margin policy / the target-bucket policy) -----------------------------------------
LARGE_TOKEN_MIN = 4096  # MFU regime.
SMALL_TOKEN_MAX = 64  # latency regime.
# Predeclared MFU target buckets (the target-bucket policy): the two largest in-sweep tokens.
MFU_TARGET_BUCKETS: Tuple[int, ...] = (16384, 32768)

# --- Token grids (the token-grid policy) ---------------------------------------------------
TOKEN_GRID_FULL: Tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
TOKEN_GRID_GPTOSS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# --- Routing distributions for correctness (the routing-distribution policy) -------------------------
ROUTING_DISTRIBUTIONS: Tuple[str, ...] = (
    "default",
    "uniform",
    "expert_skewed",
    "few_active",
    "all_active",
    "sentinel_padding",
)

# --- Node environment (the node/shape policy) ----------------------------------------------
TARGET_ARCH = "gfx950"


@dataclass(frozen=True)
class ModelShape:
    """One target MoE model shape and its in-scope quant dtypes.

    ``dtypes`` are the activation x weight quant aliases in scope for this loop:
    ``"a4w4"`` (fp4 x fp4) and/or ``"a8w4"`` (fp8 x fp4).  ``i4`` is out of scope.
    ``token_grid`` is the sweep used for this model (the token-grid policy).
    """

    name: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act: str  # "silu" or "swiglu"
    dtypes: Tuple[str, ...]
    token_grid: Tuple[int, ...]


# The four target models (the node/shape policy + plan workload table).  DeepSeek V4 is a8w4
# only; i4 (Kimi a16wi4) is excluded from this loop.
MODELS: Tuple[ModelShape, ...] = (
    ModelShape("deepseek_v3", 7168, 256, 257, 9, "silu", ("a4w4", "a8w4"), TOKEN_GRID_FULL),
    ModelShape("deepseek_v4", 7168, 512, 385, 7, "silu", ("a8w4",), TOKEN_GRID_FULL),
    ModelShape("kimi_k2", 7168, 256, 384, 8, "silu", ("a4w4", "a8w4"), TOKEN_GRID_FULL),
    ModelShape("gpt_oss", 3072, 3072, 128, 4, "swiglu", ("a4w4", "a8w4"), TOKEN_GRID_GPTOSS),
)

# Map a quant alias to the activation operand dtype passed to the kernel builder
# (the weight operand is fp4 in both in-scope cases).
DTYPE_ALIAS_TO_A_DTYPE = {"a4w4": "fp4", "a8w4": "fp8"}

# --- Correctness quarantine (non-fp4-activation e2e is environment-blocked) ---
# Controlled evidence (direct aiter test_fmoe, each model's true activation, both
# gate modes, token=16) shows the failing axis is the ACTIVATION operand being
# non-fp4:
#   a4w4  (fp4 activation):  logits_diff ~1e-5  -> PASS (all models, both gates)
#   a8w4  (fp8 activation):  logits_diff ~0.98  -> FAIL (DS V3/V4, Kimi; both gates)
#   a16w4 (bf16 activation): logits_diff ~0.98  -> FAIL (DS V3; both gates)
#   GPT-OSS a8w4 Swiglu+INTERLEAVE: ~6e-6       -> PASS (lone non-fp4-act pass;
#     aiter selects a different runtime q_dtype_a/fuse-quant path there)
# fp8 AND bf16 activation both fail with fp4 weight; only fp4 activation passes.
# Note: aiter test_fmoe passes the SAME activation/gate to BOTH its torch
# reference and the kernel, so the activation choice alone cannot explain the
# mismatch.
#
# Root cause is an activation-dtype-dependent wrapper/layout CONTRACT mismatch in
# the aiter e2e path, NOT a proven FlyDSL kernel math bug -- this checkout's own
# tests/kernels/test_moe_gemm.py --in_dtype a8w4 passes with --skip_ref false.
# For non-fp4 activation aiter preps weights via shuffle_weight_a16w4 /
# shuffle_scale_a16w4 and its reference sets a2_scale=None (no stage1->stage2 A2
# requant), while the FlyDSL mixed stage2 kernel expects a pre-scattered A2 E8M0
# scale (mixed_moe_gemm_2stage.py); this checkout's own 2-stage harness does
# requantize A2 and passes.  Reconciling this is aiter-environment integration
# work, outside the GEMM-tuning scope.
#
# All a8w4 (model, dtype) pairs are therefore QUARANTINED until the e2e a8w4
# correctness path is validated.  Their rows are kept for provenance but excluded
# from the validated baseline and from any win claim -- a genuine correctness
# block, not a silent scope reduction.
QUARANTINED_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("deepseek_v3", "a8w4"),
    ("deepseek_v4", "a8w4"),
    ("kimi_k2", "a8w4"),
    ("gpt_oss", "a8w4"),
)


def is_quarantined(model: str, dtype: str) -> bool:
    """True if (model, dtype) is correctness-quarantined (see QUARANTINED_SHAPES)."""
    return (model, dtype) in QUARANTINED_SHAPES


def validated_models():
    """Yield (ModelShape, dtype) pairs that are NOT correctness-quarantined."""
    for m in MODELS:
        for dtype in m.dtypes:
            if not is_quarantined(m.name, dtype):
                yield m, dtype


def validated_point_keys() -> set:
    """(model, dtype, act, token) keys for the correctness-passing subset.

    This is the workload the validated baseline must fully cover; the quarantined
    a8w4 shapes are excluded until their correctness path is fixed.
    """
    keys = set()
    for m, dtype in validated_models():
        for token in m.token_grid:
            keys.add((m.name, dtype, m.act, str(token)))
    return keys


def is_large_token(token: int) -> bool:
    """True if ``token`` is in the large-shape MFU regime (tokens >= 4096)."""
    return token >= LARGE_TOKEN_MIN


def is_small_token(token: int) -> bool:
    """True if ``token`` is in the small-token latency regime (tokens <= 64)."""
    return token <= SMALL_TOKEN_MAX


def is_regression(baseline_us: float, tuned_us: float, token: int = None) -> bool:
    """No-regression gate (the no-regression policy): regression iff BOTH the
    relative AND absolute bands are exceeded — ``tuned > baseline*1.02`` AND
    ``tuned-baseline > abs_floor``.

    The absolute floor is regime-aware (``abs_floor_us(token)``): 8 us for
    tokens <= SMALL_TOKEN_MAX, 2 us otherwise.  When ``token`` is None the strict
    2 us floor is used (back-compatible).  Applied per point on BOTH the
    kernel-path and e2e metrics; a point is a regression if either metric regresses.
    """
    floor = ABS_US_BAND if token is None else abs_floor_us(token)
    return (tuned_us > baseline_us * (1.0 + REGRESSION_REL)) and ((tuned_us - baseline_us) > floor)


def is_large_shape_win(baseline_mfu: float, tuned_mfu: float) -> bool:
    """Large-shape win gate (the win-margin policy): ``tuned_MFU >= baseline_MFU * 1.10``."""
    return tuned_mfu >= baseline_mfu * (1.0 + WIN_MARGIN)


def is_small_token_win(baseline_us: float, tuned_us: float) -> bool:
    """Small-token win gate (the win-margin policy): both a relative and an absolute floor —
    ``tuned_us <= baseline_us*0.90`` AND ``(baseline_us - tuned_us) >= 2us``.

    The absolute floor rejects sub-microsecond percentage-only claims.
    """
    return (tuned_us <= baseline_us * (1.0 - WIN_MARGIN)) and ((baseline_us - tuned_us) >= ABS_US_BAND)


def effective_tflops(token: int, model_dim: int, inter_dim: int, topk: int, combined_us: float) -> float:
    """Combined effective TFLOPS per the aiter test_moe_2stage formula:
    ``token*model_dim*inter_dim*3*topk*2 / us`` (us in microseconds).
    """
    return token * model_dim * inter_dim * 3 * topk * 2 / combined_us / 1e6


def mfu(effective_tflops_value: float) -> float:
    """MFU = effective TFLOPS / fp4 peak (4523 TFLOPS)."""
    return effective_tflops_value / FP4_PEAK_TFLOPS
