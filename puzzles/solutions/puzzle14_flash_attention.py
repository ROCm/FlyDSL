# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 14 (solution) - Flash attention forward, one head (out = softmax(QK^T/sqrt(D)) V).

This puzzle teaches the *online softmax* that makes flash attention memory-cheap:
the softmax normalization is folded into a single streaming pass over the keys, so
the full (S, S) score matrix is never materialized.

Layout choice for clarity: one thread owns one query row i (S = 128 rows, 128
threads). Each thread keeps its running max m, running denominator l, and output
accumulator acc[0:D] in registers, then streams over all keys j:

    s      = (q_i . k_j) * scale
    m_new  = max(m, s)
    alpha  = exp2((m - m_new) * log2e)        # rescale old stats
    p      = exp2((s - m_new) * log2e)
    l      = l * alpha + p
    acc    = acc * alpha + p * v_j
    m      = m_new

Finally o_i = acc / l. No cross-thread reduction is needed because each thread's
row is independent - the online recurrence is the whole point.

Concepts: online/streaming softmax, running max/sum rescale, register accumulator.
Book: Chapter 6 + Chapter 7; reference kernels/attention/.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

VEC = 4
LOG2E = 1.4426950408889634


def build(S: int, D: int):
    assert D % VEC == 0
    d_tiles = D // VEC
    scale = float(D) ** -0.5

    @flyc.kernel
    def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor):
        i = fx.thread_idx.x  # this thread's query row
        fm = arith.FastMathFlags.fast

        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        K_buf = fx.rocdl.make_buffer_tensor(K)
        V_buf = fx.rocdl.make_buffer_tensor(V)
        O_buf = fx.rocdl.make_buffer_tensor(O)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        def load_row(buf, row):
            div = fx.logical_divide(fx.slice(buf, (row, None)), fx.make_layout(VEC, 1))
            vecs = []
            for t in range_constexpr(d_tiles):
                r = fx.make_rmem_tensor(VEC, fx.Float32)
                fx.copy_atom_call(atom, fx.slice(div, (None, t)), r)
                vecs.append(fx.memref_load_vec(r))
            return vecs

        q = load_row(Q_buf, i)
        acc = [fx.full(VEC, fx.Float32(0.0), fx.Float32) for _ in range_constexpr(d_tiles)]
        m = fx.Float32(float("-inf"))
        l = fx.Float32(0.0)

        for j in range_constexpr(S):
            k = load_row(K_buf, j)
            s = fx.Float32(0.0)
            for t in range_constexpr(d_tiles):
                s = s + (q[t] * k[t]).reduce(ReductionOp.ADD, fastmath=fm)
            s = s * scale

            m_new = m.maximumf(s)
            alpha = fmath.exp2((m - m_new) * LOG2E, fastmath=fm)
            p = fmath.exp2((s - m_new) * LOG2E, fastmath=fm)
            l = l * alpha + p

            v = load_row(V_buf, j)
            for t in range_constexpr(d_tiles):
                acc[t] = acc[t] * alpha + v[t] * p
            m = m_new

        inv_l = 1.0 / l
        o_div = fx.logical_divide(fx.slice(O_buf, (i, None)), fx.make_layout(VEC, 1))
        for t in range_constexpr(d_tiles):
            r = fx.make_rmem_tensor(VEC, fx.Float32)
            fx.memref_store_vec(acc[t] * inv_l, r)
            fx.copy_atom_call(atom, r, fx.slice(o_div, (None, t)))

    @flyc.jit
    def attn(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        attn_kernel(Q, K, V, O).launch(grid=(1, 1, 1), block=(S, 1, 1), stream=stream)

    return attn


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    s, d = q.shape
    o = torch.zeros(s, d, dtype=torch.float32, device=q.device)
    build(s, d)(q, k, v, o, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return o


if __name__ == "__main__":
    from puzzles import common

    s, d = 128, 64
    q = torch.randn(s, d, dtype=torch.float32).cuda()
    k = torch.randn(s, d, dtype=torch.float32).cuda()
    v = torch.randn(s, d, dtype=torch.float32).cuda()
    common.check(run(q, k, v), common.ref_flash_attention(q, k, v), name="puzzle14 flash_attn", atol=5e-2, rtol=5e-2)
