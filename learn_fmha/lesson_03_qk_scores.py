# SPDX-License-Identifier: Apache-2.0
"""Lesson 03 — Attention GEMM1: the scores matrix S = K @ Q^T.

Goal: the first of attention's two GEMMs. Given Q[q, hd] and K[kv, hd] (bf16), compute
the raw attention scores. We use ONE 16x16x16-based MFMA K-loop over head_dim (HD=64,
so 4 K-steps) for a 16-query x 16-kv tile.

Layout choice that matters for the WHOLE kernel:
  The MFMA computes A @ B^T. We set A = K[kv, hd], B = Q[q, hd]. The result is therefore
      C = K @ Q^T   with shape [kv, q]   (row = kv, col = q).
  We deliberately want S laid out as [kv=row, q=col]. Why? Because the softmax (Lesson
  04) reduces over kv, and this layout puts a lane's kv-values in its own 4 accumulator
  registers (across the C-fragment), making the max/sum reductions cheap. The cost is
  that GEMM2 (Lesson 05) will then need kv as its *contraction* index in a different
  place -> the "P-transpose problem." Every attention kernel pays a transpose somewhere;
  choosing S=[kv,q] pays it on the P side, which turns out cheaper. (CK Tile makes the
  same choice in its QRKSVS pipeline.)

Recall the C-fragment layout from Lesson 01 (mfma 16x16x16):
  lane (k_outer=lane//16, mn=lane%16) gets C[k_outer*4 + e, mn] for e=0..3.
  Here C row = kv, C col = q. So lane holds S[kv = k_outer*4+e, q = mn].

sm_scale (1/sqrt(hd)) is folded in at store time (in real attention it multiplies S
before softmax; here we apply it to show where it goes).

### Idiomatic note
The QK MFMA is declared once as a typed atom (`make_mma_atom` + `fly.mma_atom_call_ssa`),
the same naming used for the copy atom. We keep the explicit per-lane loads and the
C-fragment store hand-written because the S=[kv,q] layout choice — which element each lane
ends up holding — is the whole point of this lesson and is invisible under `fx.gemm`.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_03_qk_scores.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly

BQ = 16  # queries in the tile
BKV = 16  # keys in the tile
HD = 64
KSTEPS = HD // 16


@flyc.kernel(known_block_size=[64, 1, 1])
def qk_kernel(Q: fx.Tensor, K: fx.Tensor, S: fx.Tensor, sm_scale: fx.Constexpr[float]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rS = fx.buffer_ops.create_buffer_resource(S)

    # IDIOMATIC: name the QK MFMA (A=K, B=Q -> S = K @ Q^T) once.
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)

    acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
    for ks in fx.range_constexpr(KSTEPS):
        k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
        # A = K[kv=mn, hd], B = Q[q=mn, hd]; both [row, hd] row-major.
        k_off = mn * fx.Int32(HD) + k0
        q_off = mn * fx.Int32(HD) + k0
        k_vec = fx.buffer_ops.buffer_load(rK, k_off, vec_width=4, dtype=fx.BFloat16)
        q_vec = fx.buffer_ops.buffer_load(rQ, q_off, vec_width=4, dtype=fx.BFloat16)
        acc = fly.mma_atom_call_ssa(
            [f32x4],
            mma_atom,
            fx.Vector(k_vec).bitcast(fx.Int16).ir_value(),  # A = K
            fx.Vector(q_vec).bitcast(fx.Int16).ir_value(),  # B = Q  (instruction does K @ Q^T)
            acc,
        )
    accv = fx.Vector(acc)
    # lane holds S[kv = k_outer*4+e, q = mn]; store with sm_scale folded in.
    for e in fx.range_constexpr(4):
        kv = k_outer * fx.Int32(4) + fx.Int32(e)
        q = mn
        s = fx.Float32(accv[e]) * fx.Float32(sm_scale)
        fx.buffer_ops.buffer_store(s.ir_value(), rS, (kv * fx.Int32(BQ) + q).ir_value())


@flyc.jit
def run_qk(Q: fx.Tensor, K: fx.Tensor, S: fx.Tensor, sm_scale: fx.Constexpr[float], stream: fx.Stream = fx.Stream(None)):
    qk_kernel(Q, K, S, sm_scale).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(BQ, HD, dtype=torch.bfloat16).cuda()
    K = torch.randn(BKV, HD, dtype=torch.bfloat16).cuda()
    S = torch.zeros(BKV, BQ, dtype=torch.float32).cuda()  # [kv, q]
    sm = 1.0 / HD**0.5

    run_qk(Q, K, S, sm, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ref = (K.float() @ Q.float().T) * sm  # [kv, q]
    err = (S - ref).abs().max().item()
    print(f"S layout = [kv, q]   max abs err = {err:.4f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")
