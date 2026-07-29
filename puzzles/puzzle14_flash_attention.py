# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 14 - Flash attention forward, one head (out = softmax(QK^T/sqrt(D)) V).

GOAL: compute attention with the *online softmax* that makes flash attention
memory-cheap. Instead of materializing the full (S, S) score matrix and running a
separate softmax pass, fold the normalization into a single streaming pass over
the keys, keeping only running stats in registers.

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

CONCEPTS: online/streaming softmax, running max/sum rescale, register accumulator.
READ: FlyDSL for Dummies - Chapter 6 + Chapter 7; reference kernels/attention/.
HINT: a nested load_row(buf, row) helper that returns a list of d_tiles float4
vectors (read-only capture) keeps the streaming loop tidy. Use exp2 with LOG2E
rather than exp; hold m/l as scalars and acc as a list of VEC-wide vectors.
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

        # ==== YOUR CODE HERE ====
        # 1. Buffer tensors for Q/K/V/O; a 128b copy atom:
        #      atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        # 2. Nested read-only helper load_row(buf, row) -> list of d_tiles float4:
        #      div = fx.logical_divide(fx.slice(buf, (row, None)), fx.make_layout(VEC, 1))
        #      for t in range_constexpr(d_tiles): copy_atom_call into rmem, memref_load_vec.
        # 3. q = load_row(Q_buf, i); acc = [fx.full(VEC, 0.0) ...]; m = -inf; l = 0.
        # 4. Stream over keys j in range_constexpr(S):
        #      k = load_row(K_buf, j); s = sum_t (q[t]*k[t]).reduce(ADD); s *= scale
        #      m_new = m.maximumf(s); alpha = exp2((m-m_new)*LOG2E); p = exp2((s-m_new)*LOG2E)
        #      l = l*alpha + p; v = load_row(V_buf, j); acc[t] = acc[t]*alpha + v[t]*p; m = m_new
        # 5. inv_l = 1.0 / l; store acc[t] * inv_l to row i of O via the same atom.
        raise NotImplementedError("Fill in attn_kernel")

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
