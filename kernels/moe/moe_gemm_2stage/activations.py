# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Elementwise activation helpers for MoE stage1 (device fast paths)."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T


def silu(x):
    # device fast path:
    #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
    #   sig = rcp(1 + emu)                   -> v_rcp_f32
    #   y = x * sig
    #
    # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
    # sequences that introduce extra compares/cndmasks.
    t = x * (-1.4426950408889634)  # -log2(e)
    emu = rocdl.exp2(T.f32, t)
    den = 1.0 + emu
    sig = rocdl.rcp(T.f32, den)
    return x * sig


def gelu_tanh(x):
    # GeLU tanh approx: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
    # Expand tanh via exp(-2|y|) in [0,1] (non-positive exponent avoids fp32
    # overflow).
    half = fx.Float32(0.5)
    one = fx.Float32(1.0)
    two = fx.Float32(2.0)
    zero = fx.Float32(0.0)
    x3 = x * x * x
    y = fx.Float32(0.7978845608) * (x + fx.Float32(0.044715) * x3)
    abs_y = fx.Float32(y).maximumf(zero - y)
    e = fx.exp(fx.Float32(-2.0) * abs_y)
    den = one + e
    # 1 + tanh(y): y>=0 -> 2/den ; y<0 -> 2*e/den
    numerator = (y > zero).select(two, two * e)
    return half * x * (numerator * (one / den))
