# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types and helpers shared across MoE FlyDSL kernel modules."""

from enum import Enum

from flydsl._mlir.dialects import vector
from flydsl.expr import as_ir_value
from flydsl.expr.typing import T


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
# Thin vector.from_elements + vector.bitcast wrappers that repack packed integer
# lanes into the vector element types consumed by MFMA. Pure (only their args and
# the module-level vector / T are referenced), shared by the 2-stage MoE kernels.


def i64_to_v4f16(x_i64):
    """Reinterpret one i64 lane as vector<4xf16>."""
    v1 = vector.from_elements(T.vec(1, T.i64), [as_ir_value(x_i64)])
    return vector.bitcast(T.f16x4, as_ir_value(v1))


def i64_to_v4i16(x_i64):
    """Reinterpret one i64 lane as vector<4xi16> (bf16 bit pattern)."""
    v1 = vector.from_elements(T.vec(1, T.i64), [as_ir_value(x_i64)])
    return vector.bitcast(T.i16x4, as_ir_value(v1))


def i64x2_to_v8f16(lo, hi):
    """Reinterpret two i64 lanes as vector<8xf16>."""
    v2 = vector.from_elements(T.i64x2, [as_ir_value(lo), as_ir_value(hi)])
    return vector.bitcast(T.f16x8, as_ir_value(v2))


def i64x2_to_v8bf16(lo, hi):
    """Reinterpret two i64 lanes as vector<8xbf16>."""
    v2 = vector.from_elements(T.i64x2, [as_ir_value(lo), as_ir_value(hi)])
    return vector.bitcast(T.bf16x8, as_ir_value(v2))


def i64x4_to_i32x8(x0, x1, x2, x3):
    """Reinterpret four i64 lanes as vector<8xi32>."""
    v4 = vector.from_elements(T.vec(4, T.i64), [as_ir_value(x0), as_ir_value(x1), as_ir_value(x2), as_ir_value(x3)])
    return vector.bitcast(T.vec(8, T.i32), as_ir_value(v4))
