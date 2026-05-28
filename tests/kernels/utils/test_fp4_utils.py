#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``tests.kernels.utils.fp4_utils``.

Today these are CPU-only (no MFMA exercised). The most important coverage
is ``f32_to_e8m0_even`` vs. the legacy ``f32_to_e8m0(amax / max_normal)``
path: the two recipes diverge by one E8M0 step on the majority of inputs
when ``max_normal`` is not exactly ``2^emax`` (i.e. for E2M3/E3M2/E4M3).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.kernels.utils import fp4_utils  # noqa: E402

# (dtype_name, emax, mbits, max_normal) -- straight from the OCP MX v1.0 spec.
_OCP_MX_TYPES = [
    ("E2M1", 2, 1, 6.0),  # MXFP4   max = 4 * 1.5
    ("E2M3", 2, 3, 7.5),  # MXFP6   max = 4 * 1.875
    ("E3M2", 4, 2, 28.0),  # MXFP6   max = 16 * 1.75
    ("E4M3", 8, 3, 448.0),  # MXFP8   max = 256 * 1.75
    ("E5M2", 15, 2, 57344.0),
]


def _e8m0_to_f32(byte_u8: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 -> float32, OCP convention. byte=0 reserved (0+), byte=255 NaN."""
    bits = byte_u8.to(torch.int64) << 23
    f = bits.to(torch.int32).view(torch.float32)
    f = torch.where(byte_u8 == 0, torch.full_like(f, 2.0**-126), f)
    f = torch.where(byte_u8 == 0xFF, torch.full_like(f, float("nan")), f)
    return f


def _reference_even_round_byte(amax: float, emax: int, mbits: int) -> int:
    """Single-element reference: log2-floor-with-half-ULP-round-to-even.

    Independent re-derivation of the OCP §6.3 formula in plain python; used
    to cross-check the vectorized ``f32_to_e8m0_even`` against arithmetic
    that doesn't share its bit-manipulation tricks.
    """
    if amax == 0.0:
        return 1  # smallest normal scale, matches Quark even_round
    if math.isnan(amax):
        return 0xFF
    # Add half-ULP at f32 mantissa bit (23 - mbits - 1), mask off mantissa.
    import struct

    bits = struct.unpack("<I", struct.pack("<f", amax))[0]
    bits = (bits + (1 << (23 - mbits - 1))) & 0x7F800000
    rounded = struct.unpack("<f", struct.pack("<I", bits))[0]
    if rounded == 0.0:
        return 0
    k = int(math.floor(math.log2(rounded))) - emax
    return max(0, min(254, k + 127))


@pytest.mark.parametrize("dtype_name,emax,mbits,max_normal", _OCP_MX_TYPES)
def test_f32_to_e8m0_even_matches_python_reference(dtype_name, emax, mbits, max_normal):
    """Vectorized impl matches the per-element python reference bit-for-bit."""
    g = torch.Generator(device="cpu").manual_seed(0)
    # Sample amax across many binades + a few exact powers of 2 (tie points).
    amax = torch.cat(
        [
            torch.rand(2048, generator=g, dtype=torch.float32) * 8.0,
            torch.rand(2048, generator=g, dtype=torch.float32) * max_normal * 4,
            # exact powers of 2 — tie points where rounding mode matters
            torch.tensor([2.0**k for k in range(-10, 12)], dtype=torch.float32),
            torch.tensor([0.0, max_normal, max_normal * 0.5], dtype=torch.float32),
        ]
    )
    got = fp4_utils.f32_to_e8m0_even(amax, emax=emax, mbits=mbits).view(torch.uint8)
    want = torch.tensor(
        [_reference_even_round_byte(a.item(), emax, mbits) for a in amax],
        dtype=torch.uint8,
    )
    mism = (got.int() - want.int() != 0).sum().item()
    assert mism == 0, (
        f"{dtype_name}: {mism}/{amax.numel()} byte mismatches vs python reference\n"
        f"first 10 amax / got / want:\n"
        + "\n".join(
            f"  amax={a.item():.6g}  got={g.item():3d}  want={w.item():3d}"
            for a, g, w in zip(amax[:10], got[:10], want[:10])
        )
    )


@pytest.mark.parametrize("dtype_name,emax,mbits,max_normal", _OCP_MX_TYPES)
def test_e8m0_even_scale_within_one_binade_of_optimal(dtype_name, emax, mbits, max_normal):
    """OCP §6.3 ``even_round`` rounds ``log2(amax)`` round-to-nearest-even,
    so the decoded scale ``S = 2^k`` is within one binade of the optimal
    ``amax / max_normal``: ``0.5 <= S / (amax / max_normal) <= 2.0`` modulo
    the ``(1 + max_mant)`` factor of ``max_normal``. Equivalently
    ``S * max_normal`` falls in ``[amax / 2, 2 * amax]``.

    This is the right invariant for spec-compliant even_round — the
    stronger "S * max_normal >= amax" (no saturation) is NOT guaranteed
    by RTNE because round-down can win at tie points. The earlier
    ``f32_to_e8m0(amax / max_normal)`` path biases low by roughly one step
    on most inputs and would fail the upper bound for many blocks.
    """
    g = torch.Generator(device="cpu").manual_seed(1)
    amax = torch.rand(4096, generator=g, dtype=torch.float32) * max_normal * 3 + 1e-6
    byte = fp4_utils.f32_to_e8m0_even(amax, emax=emax, mbits=mbits).view(torch.uint8)
    scale = _e8m0_to_f32(byte)
    ratio = (scale * max_normal) / amax
    # Round-to-nearest in log2 space ⇒ ratio in [1/(1+max_mant_frac), 2*max_mant_frac].
    # Tight numerical bound is 0.5 < ratio < 2; the tie may land exactly at
    # the edges, so use a small slack.
    n_lo = (ratio < 0.5 - 1e-5).sum().item()
    n_hi = (ratio > 2.0 + 1e-5).sum().item()
    assert n_lo == 0 and n_hi == 0, (
        f"{dtype_name}: even_round produced scale outside [0.5, 2.0] of "
        f"amax/max_normal: {n_lo} below, {n_hi} above. "
        f"Range observed: [{ratio.min().item():.4g}, {ratio.max().item():.4g}]"
    )


def test_f32_to_e8m0_legacy_off_by_one_on_e2m3():
    """Document the existing ``f32_to_e8m0(amax / max_normal)`` bias.

    The legacy path is consistently one E8M0 step low for ~60-70% of random
    amax inputs in E2M3 because 1/7.5 lies in (0.5, 1) of a binade. This
    test fails-loudly only if the legacy function's behavior changes; the
    intent is to pin down *why* ``f32_to_e8m0_even`` exists.
    """
    emax, mbits, max_normal = 2, 3, 7.5
    g = torch.Generator(device="cpu").manual_seed(2)
    amax = torch.rand(4096, generator=g, dtype=torch.float32) * max_normal
    even = fp4_utils.f32_to_e8m0_even(amax, emax=emax, mbits=mbits).view(torch.uint8)
    legacy = fp4_utils.f32_to_e8m0((amax / max_normal).clamp_(min=2**-126))
    legacy = legacy.view(torch.uint8)
    diff = even.int() - legacy.int()
    # Every diverging byte should be even == legacy + 1 (legacy is biased low).
    bad_dirs = ((diff != 0) & (diff != 1)).sum().item()
    assert bad_dirs == 0, (
        f"Unexpected disagreement direction: {bad_dirs} bytes diverged in "
        f"directions other than +1. Distribution: {torch.unique(diff, return_counts=True)}"
    )
    n_mism = (diff != 0).sum().item()
    # Loose lower bound: on uniform amax in [0, max_normal] we see ~60%+.
    assert n_mism > amax.numel() // 3, (
        f"Expected ≥1/3 of bytes to disagree (legacy off-by-one on E2M3); "
        f"got {n_mism}/{amax.numel()}. Has f32_to_e8m0 been changed?"
    )
