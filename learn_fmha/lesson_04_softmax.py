# SPDX-License-Identifier: Apache-2.0
"""Lesson 04 — Softmax over the scores (the cross-lane reduction).

Goal: given S = [kv, q] from Lesson 03, compute per-query softmax weights
P[kv, q] = exp(S[kv,q] - max_kv S) / sum_kv exp(...). The reduction is over kv FOR EACH
query q. This lesson is about HOW a reduction works when the values you reduce live in
different LANES — the single trickiest mechanic in attention.

This lesson does softmax over ONE 16-kv tile (the whole S fits in registers). The
"online"/streaming version that folds many kv-tiles with a running max/sum is Lesson 06
(it needs the runtime kv-loop). The math here is the per-tile core of that.

### Where the kv values live (from Lesson 01's C-fragment layout)
S is laid out [kv=row, q=col]; lane (k_outer=lane//16, mn=lane%16) holds the 4 values
S[kv = k_outer*4 + e, q = mn] for e=0..3. So for a FIXED query q=mn, its 16 kv values
are split:
  - 4 of them are in THIS lane's own registers (e=0..3, the k_outer this lane belongs to),
  - the other 12 are in lanes {mn+16, mn+32, mn+48} (the other k_outer groups).

### The reduction in two stages
1. **Intra-lane:** reduce this lane's own 4 registers (a plain max / sum over e=0..3).
2. **Cross-lane:** combine the 4 k_outer groups (lanes differing only in bits 4 and 5 of
   the lane id). `shuffle_xor(mask, width)` lets a lane read another lane's value where
   `other_lane = my_lane XOR mask`. k_outer = lane//16, so XOR 16 flips k_outer bit0
   (0<->1, 2<->3) and XOR 32 flips k_outer bit1 (0<->2, 1<->3). Doing both merges all 4
   groups. After the two shuffles, every lane in column mn holds the SAME reduced value.

This is exactly the production pattern (real kernel uses mfma_32x32x16 -> 16 regs/lane +
ONE shuffle_xor(32); here 4 regs/lane + TWO shuffles because the 16-wide MFMA splits kv
across 4 lane-groups instead of 2).

### Numerical stability
Subtract the max before exp (standard). We also carry the all-masked guard idea: if the
max is -inf (no valid kv), force it to 0 so exp(-inf - 0) = 0 instead of NaN. Not needed
for this dense tile but introduced here because Lesson 06's causal masking will need it.

### Idiomatic note
GEMM1 reuses the typed MMA atom (`make_mma_atom` + `fly.mma_atom_call_ssa`). The softmax
itself — the cross-lane max/sum reduction — stays as DIRECT register + `shuffle_xor` code:
the layout-algebra skill explicitly says reductions stay direct, and the lane-shuffle
mechanic is exactly what this lesson teaches.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_04_softmax.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly

BQ = 16
BKV = 16
HD = 64
KSTEPS = HD // 16
LOG2E = 1.4426950408889634


@flyc.kernel(known_block_size=[64, 1, 1])
def softmax_kernel(Q: fx.Tensor, K: fx.Tensor, P: fx.Tensor, sm_scale: fx.Constexpr[float]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rP = fx.buffer_ops.create_buffer_resource(P)

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)

    # --- GEMM1 (same as Lesson 03): sv[e] = S[kv=k_outer*4+e, q=mn] * sm_scale ---
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

    # --- softmax over kv for this lane's query column q=mn ---
    # 1) intra-lane max over the 4 own kv
    m = sv[0]
    for e in fx.range_constexpr(3):
        m = m.maximumf(sv[e + 1])
    # 2) cross-lane: merge the 4 k_outer groups (XOR 16 then XOR 32)
    m = m.maximumf(m.shuffle_xor(fx.Int32(16), fx.Int32(64)))
    m = m.maximumf(m.shuffle_xor(fx.Int32(32), fx.Int32(64)))
    # all-masked guard (harmless here; needed once causal masking arrives in Lesson 06)
    m_is_neg = m < fx.Float32(-1.0e38)
    safe_m = m_is_neg.select(fx.Float32(0.0), m)

    # exp via fast rocdl.exp2: exp(x) = 2^(x*log2e)
    p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[e] - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
    # sum over kv: intra-lane then cross-lane
    l = p[0]
    for e in fx.range_constexpr(3):
        l = l + p[e + 1]
    l = l + l.shuffle_xor(fx.Int32(16), fx.Int32(64))
    l = l + l.shuffle_xor(fx.Int32(32), fx.Int32(64))
    inv_l = fx.Float32(1.0) / l

    for e in fx.range_constexpr(4):
        kv = k_outer * fx.Int32(4) + fx.Int32(e)
        pw = p[e] * inv_l
        fx.buffer_ops.buffer_store(pw.ir_value(), rP, (kv * fx.Int32(BQ) + mn).ir_value())


@flyc.jit
def run_softmax(Q, K, P, sm_scale: fx.Constexpr[float], stream: fx.Stream = fx.Stream(None)):
    softmax_kernel(Q, K, P, sm_scale).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(BQ, HD, dtype=torch.bfloat16).cuda()
    K = torch.randn(BKV, HD, dtype=torch.bfloat16).cuda()
    P = torch.zeros(BKV, BQ, dtype=torch.float32).cuda()
    sm = 1.0 / HD**0.5

    run_softmax(Q, K, P, sm, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    S = (K.float() @ Q.float().T) * sm  # [kv, q]
    ref = torch.softmax(S, dim=0)  # softmax over kv
    err = (P - ref).abs().max().item()
    print(f"P = softmax_kv(S)   max abs err = {err:.5f}  ->  {'PASS' if err < 1e-2 else 'FAIL'}")
