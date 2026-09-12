# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared f32 gate/up activations with stable rounding and scheduling."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from kernels.common.tensor_shim import _to_raw as _raw

LOG2E = 1.4426950408889634


def sigmoid_batch(xs):
    """Emit all exponentials before their reciprocals to preserve batch scheduling."""
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(x * fx.Float32(-LOG2E)))) for x in xs]
    return [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]


def sigmoid_f32(g):
    return sigmoid_batch([g])[0]


def clamp_gate_up(g, u, neg_limit):
    """Upper-bound the gate and symmetrically clamp the up value."""
    return -fx.max(-g, neg_limit), fx.max(-fx.max(-u, neg_limit), neg_limit)


def silu_mul_batch(gs, us):
    sig = sigmoid_batch(gs)
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def tanh_batch(xs):
    """Sign-restored tanh with exp2 and reciprocal operations grouped by stage."""
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    es = []
    for x in xs:
        abs_x = fx.max(x, -x)
        es.append(fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * neg_two_log2e))))
    recips = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e))) for e in es]
    zero = fx.Float32(0.0)
    out = []
    for i, x in enumerate(xs):
        tanh_abs = (fx.Float32(1.0) - es[i]) * recips[i]
        out.append((x > zero).select(tanh_abs, -tanh_abs))
    return out


def tanh_f32(x):
    return tanh_batch([x])[0]


def situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp):
    gate = beta * tanh_f32(g * beta_rcp) * sigmoid_f32(g)
    up = lbeta * tanh_f32(u * lbeta_rcp)
    return gate * up


def situ_mul_batch(gs, us, beta, beta_rcp, lbeta, lbeta_rcp, neg_clamp_limit):
    """SiTU with an upper gate clamp and symmetric up clamp; limits stay runtime."""
    out = []
    for i in range(len(gs)):
        g, u = clamp_gate_up(gs[i], us[i], neg_clamp_limit)
        out.append(situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp))
    return out
