# SPDX-License-Identifier: Apache-2.0
"""Lesson 05 — GEMM2 (O = P @ V) and the first FULLY FUSED attention tile (bf16).

Goal: finish attention. We have S=[kv,q] (Lesson 03) and softmax P (Lesson 04) in
registers; now do the second GEMM  O[q, d] = sum_kv P[q, kv] * V[kv, d]  and write the
output. Single tile: BQ=16 queries, BKV=16 keys, HD=64 (QK contraction), hd_v=64 output
(=> DT=4 output d-tiles). One wavefront, NO LDS. End-to-end fused attention.

### GEMM2 orientation
MFMA computes result[m,n] = sum_k A[m,k] B[n,k]. We want O[q,d] = sum_kv P[q,kv] V[kv,d].
Set m=q, n=d, k=kv:  A = P[q, kv],  B[d, kv] = V[kv, d]  (so B = V^T).
  - A operand: lane(k_outer,mn) holds A[q=mn, kv=k_outer*4+e].
  - B operand: lane(k_outer,mn) holds B[d=mn, kv=k_outer*4+e] = V[kv=k_outer*4+e, d=mn]
        (V stored [kv,d] row-major => 4 loads at stride hd_v).
  - C result: lane(k_outer,mn) holds O[q=k_outer*4+e, d=mn].
P doesn't depend on d, so we reuse the same A-fragment for all DT=4 output d-tiles.

### The "P-transpose problem" — and why it DOESN'T bite us here (important!)
GEMM2 needs the A-operand P[q=mn, kv=k_outer*4+e]. Softmax (Lesson 04) left this lane
holding P[kv=k_outer*4+e, q=mn] — which is THE SAME 4 SCALARS. For the 16x16x16 MFMA the
output C-fragment layout (rows grouped in 4s by k_outer) exactly matches the input
A-fragment layout (K grouped in 4s by k_outer), so **P is already in the right registers —
no transpose needed.**

That is a LUCKY property of this MFMA shape. The production fp8 kernel uses mfma_32x32x16,
whose C-fragment scatters kv differently than its A-fragment wants -> there the transpose
is REAL and must be done (through LDS, or via ds_bpermute). We meet that head-on in
Lesson 07 (fp8) and fix it in Lessons 11-12. Teaching point: **whether you pay a transpose
depends on the MFMA shape you chose** — it is not fundamental to attention.

### Idiomatic note
Both GEMMs use ONE typed MMA atom (`make_mma_atom` + `fly.mma_atom_call_ssa`) — declared
once, called for QK and again per output d-tile for PV. The softmax stays direct (Lesson
04). The explicit fragment loads/stores are kept on purpose: the lesson's whole point is
that GEMM2's A-operand P[q,kv] is the SAME 4 registers softmax produced (no transpose for
this 16x16x16 shape) — a fact you can only SEE in the hand-mapped lane layout, not under
`fx.gemm`.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_05_pv_fused.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly

BQ = 16
BKV = 16
HD = 64       # QK head dim (contraction of GEMM1)
HDV = 64      # V head dim (output width)
KSTEPS = HD // 16
DT = HDV // 16
LOG2E = 1.4426950408889634


@flyc.kernel(known_block_size=[64, 1, 1])
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, sm_scale: fx.Constexpr[float]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # IDIOMATIC: one typed 16x16x16 bf16 MFMA atom serves BOTH attention GEMMs.
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)

    # --- GEMM1: S[kv=k_outer*4+e, q=mn] ---
    acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
    for ks in fx.range_constexpr(KSTEPS):
        k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
        k_vec = fx.buffer_ops.buffer_load(rK, mn * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16)
        q_vec = fx.buffer_ops.buffer_load(rQ, mn * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16)
        acc = fly.mma_atom_call_ssa(
            [f32x4],
            mma_atom,
            fx.Vector(k_vec).bitcast(fx.Int16).ir_value(),
            fx.Vector(q_vec).bitcast(fx.Int16).ir_value(),
            acc,
        )
    sv = [fx.Float32(fx.Vector(acc)[e]) * fx.Float32(sm_scale) for e in range(4)]

    # --- softmax over kv (Lesson 04) -> p[e] = P[kv=k_outer*4+e, q=mn], normalized ---
    m = sv[0]
    for e in fx.range_constexpr(3):
        m = m.maximumf(sv[e + 1])
    m = m.maximumf(m.shuffle_xor(fx.Int32(16), fx.Int32(64)))
    m = m.maximumf(m.shuffle_xor(fx.Int32(32), fx.Int32(64)))
    safe_m = (m < fx.Float32(-1.0e38)).select(fx.Float32(0.0), m)
    p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[e] - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
    l = p[0]
    for e in fx.range_constexpr(3):
        l = l + p[e + 1]
    l = l + l.shuffle_xor(fx.Int32(16), fx.Int32(64))
    l = l + l.shuffle_xor(fx.Int32(32), fx.Int32(64))
    inv_l = fx.Float32(1.0) / l
    p_norm = [p[e] * inv_l for e in range(4)]

    # --- GEMM2: A = P (already register-resident, no transpose), B = V^T, per d-tile ---
    # A-fragment as bf16 vector<4>: lane holds P[q=mn, kv=k_outer*4+e] = p_norm[e].
    a_bf16 = fx.Vector.from_elements([p_norm[e].to(fx.BFloat16) for e in range(4)], fx.BFloat16)
    a_i16 = fx.Vector(a_bf16).bitcast(fx.Int16)
    for dt in fx.range_constexpr(DT):
        # B = V^T: lane holds V[kv=k_outer*4+e, d=dt*16+mn], stored [kv, d] row-major (stride HDV).
        d_col = fx.Int32(dt * 16) + mn
        b_bf16 = fx.Vector.from_elements(
            [fx.buffer_ops.buffer_load(rV, (k_outer * fx.Int32(4) + fx.Int32(e)) * fx.Int32(HDV) + d_col,
                                       vec_width=1, dtype=fx.BFloat16) for e in range(4)],
            fx.BFloat16,
        )
        b_i16 = fx.Vector(b_bf16).bitcast(fx.Int16)
        o = fly.mma_atom_call_ssa(
            [f32x4],
            mma_atom,
            a_i16.ir_value(),
            b_i16.ir_value(),
            fx.Vector.filled(4, 0.0, fx.Float32).ir_value(),
        )
        ov = fx.Vector(o)
        # C result: lane holds O[q=k_outer*4+e, d=dt*16+mn]
        for e in fx.range_constexpr(4):
            q = k_outer * fx.Int32(4) + fx.Int32(e)
            d = fx.Int32(dt * 16) + mn
            fx.buffer_ops.buffer_store(fx.Float32(ov[e]).ir_value(), rO, (q * fx.Int32(HDV) + d).ir_value())


@flyc.jit
def run_attn(Q, K, V, O, sm_scale: fx.Constexpr[float], stream: fx.Stream = fx.Stream(None)):
    attn_kernel(Q, K, V, O, sm_scale).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(BQ, HD, dtype=torch.bfloat16).cuda()
    K = torch.randn(BKV, HD, dtype=torch.bfloat16).cuda()
    V = torch.randn(BKV, HDV, dtype=torch.bfloat16).cuda()
    O = torch.zeros(BQ, HDV, dtype=torch.float32).cuda()
    sm = 1.0 / HD**0.5

    run_attn(Q, K, V, O, sm, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    S = (Q.float() @ K.float().T) * sm     # [q, kv]
    P = torch.softmax(S, dim=1)            # over kv
    ref = P @ V.float()                    # [q, d]
    err = (O - ref).abs().max().item()
    print(f"fused attention  max abs err = {err:.4f}  ->  {'PASS' if err < 5e-2 else 'FAIL'}")
