# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types and helpers shared across MoE FlyDSL kernel modules."""

from enum import Enum

import flydsl.expr as fx


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


# ── Vector bit-reinterpretation helpers ──────────────────────────────────────
# Repack integer lanes into the element types consumed by MFMA using
# Vector.from_elements and bitcast, shared by the 2-stage MoE kernels.


def i64_to_v4f16(x_i64):
    """Reinterpret one i64 lane as vector<4xf16>."""
    v1 = fx.Vector.from_elements([x_i64], fx.Int64)
    return v1.bitcast(fx.Float16)


def i64_to_v4i16(x_i64):
    """Reinterpret one i64 lane as vector<4xi16> (bf16 bit pattern)."""
    v1 = fx.Vector.from_elements([x_i64], fx.Int64)
    return v1.bitcast(fx.Int16)


def i64x2_to_v8f16(lo, hi):
    """Reinterpret two i64 lanes as vector<8xf16>."""
    v2 = fx.Vector.from_elements([lo, hi], fx.Int64)
    return v2.bitcast(fx.Float16)


def i64x2_to_v8bf16(lo, hi):
    """Reinterpret two i64 lanes as vector<8xbf16>."""
    v2 = fx.Vector.from_elements([lo, hi], fx.Int64)
    return v2.bitcast(fx.BFloat16)


def i64x4_to_i32x8(x0, x1, x2, x3):
    """Reinterpret four i64 lanes as vector<8xi32>."""
    v4 = fx.Vector.from_elements([x0, x1, x2, x3], fx.Int64)
    return v4.bitcast(fx.Int32)
