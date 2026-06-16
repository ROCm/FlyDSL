# SPDX-License-Identifier: Apache-2.0
"""Lesson 06 — Streaming attention: runtime kv-loop, ONLINE softmax, causal mask.

Goal: real attention processes a long key sequence in TILES, folding each tile into a
running result with the flash-attention "online softmax." We also add the causal mask.
This turns Lesson 05's single-tile kernel into one that handles arbitrary seqlen_k with a
RUNTIME loop. One query tile (BQ=16) vs all kv tiles (BKV=16 each), one wavefront.

### Online softmax (the flash trick)
We can't see all kv before normalizing. So we keep, per query, a running max `m` and
running denominator `l`, and the unnormalized output accumulator `o`. For each kv-tile:
  m_new = max(m_old, max_kv tile)
  corr  = exp(m_old - m_new)            # rescale factor for what we already accumulated
  p     = exp(s - m_new)                # this tile's weights (unnormalized)
  l     = l*corr + sum(p)
  o     = o*corr + p @ V_tile           # rescale old o, add this tile
At the end: O = o / l. This is mathematically exact, never materializes the full S.

### FlyDSL runtime loop
Compile-time loops use range_constexpr (unrolled). A RUNTIME-bounded loop (n_kv depends
on seqlen) uses scf.for via:
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=[m, l, o0..o3]):
        ... use st[0]=m, st[1]=l, st[2+d]=o_d ...
        st = yield [m_new, l_new, o0_new, ...]
The loop-carried state (m, l, o-tiles) MUST be threaded through init= and yield — there
are no mutable captured variables across iterations.

### Causal mask
Query at absolute row `qrow` may attend to key `kv` only if `kv <= qrow + (sk - sq)`.
The `(sk - sq)` offset aligns the diagonal when the key sequence is longer than the query
sequence (prefix KV). Forgetting it silently breaks any case with sk != sq. Masked entries
get score -inf -> p=0. The all-masked guard (Lesson 04) prevents NaN for fully-masked rows.

### Idiomatic style (flydsl-layout-algebra skill)
Both GEMMs go through a single typed MFMA atom: `fx.make_mma_atom(fx.rocdl.MFMA(16,16,16, bf16))`
driven by `fly.mma_atom_call_ssa`, instead of repeating the raw `mfma_f32_16x16x16bf16_1k_`
intrinsic. The global LOADS stay direct: the per-lane causal-masked, strided kv gathers are the
skill's "jagged / bespoke register packing" stay-direct case, so the zipped_divide / tiled_copy
layout partitioning is deliberately not applied to them.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_06_causal_seqloop.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly

BQ = 16
BKV = 16
HD = 64
HDV = 64
KSTEPS = HD // 16
DT = HDV // 16
LOG2E = 1.4426950408889634


@flyc.kernel(known_block_size=[64, 1, 1])
def attn_kernel(Q, K, V, O, sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    neg_inf = fx.Float32(-3.0e38)

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # IDIOMATIC (flydsl-layout-algebra skill, "tiled-MMA" recipe): declare the 16x16x16 bf16 MFMA
    # ONCE as a typed layout-API atom and drive BOTH GEMMs through `fly.mma_atom_call_ssa`, instead
    # of repeating the raw `rocdl.mfma_f32_16x16x16bf16_1k_` intrinsic at each call site. The MFMA
    # shape/dtype is named once; the call sites read as a matmul op. Operands are the SAME i16-bitcast
    # vec(4) fragments as the raw intrinsic, so the lane dataflow is unchanged.
    # The GLOBAL LOADS stay direct (create_buffer_resource + buffer_load): per-lane causal-masked,
    # strided kv gathers are the skill's "jagged / bespoke register packing" stay-direct case — the
    # zipped_divide/tiled_copy partitioning does not express them cleanly.
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

    def _mfma(a_i16, b_i16, acc):
        return fly.mma_atom_call_ssa([f32x4], _mma_atom, a_i16, b_i16, acc)

    # this lane's query column q = mn (single q-tile at row 0..15). Preload Q fragment (reused).
    q_packs = []
    for ks in fx.range_constexpr(KSTEPS):
        k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
        q_packs.append(fx.buffer_ops.buffer_load(rQ, mn * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16))

    # per-lane causal bound for query qrow = mn: valid kv <= eff_bound
    qrow = mn
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_i - fx.Int32(1)).select(cb, sk_i - fx.Int32(1))
    else:
        eff_bound = sk_i - fx.Int32(1)

    n_kv = (sk_i + fx.Int32(BKV - 1)) // fx.Int32(BKV)
    m0 = fx.Float32(-3.0e38)
    l0 = fx.Float32(0.0)
    o0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(DT)]
    init = [m0, l0] + o0
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BKV)

        # GEMM1 for this kv-tile: sv[e] = S[kv = kv0 + k_outer*4 + e, q = mn]
        acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
        for ks in fx.range_constexpr(KSTEPS):
            k0 = fx.Int32(ks * 16) + k_outer * fx.Int32(4)
            kvrow = kv0 + k_outer * fx.Int32(4)  # base row; the 4 contiguous-K load is along hd
            # K[kv = kv0 + k_outer*4 + (which?)]... we load K row = kv0 + (this lane's kv block).
            # lane's A-fragment row is kv = kv0 + k_outer*4 + e is the OUTPUT row, not load row.
            # For mfma A=K[kv=mn?]. Careful: in GEMM1 A=K, lane(k_outer,mn) loads K[row=mn, k].
            # Output C row = k_outer*4+e. So the kv that ends in this lane's reg e is kv0+k_outer*4+e,
            # but the K ROW we LOAD is mn-indexed. We must load K[kv = kv0 + mn].
            k_row = kv0 + mn
            k_row_safe = (k_row < sk_i).select(k_row, fx.Int32(0))
            k_vec = fx.buffer_ops.buffer_load(rK, k_row_safe * fx.Int32(HD) + k0, vec_width=4, dtype=fx.BFloat16)
            acc = _mfma(
                fx.Vector(k_vec).bitcast(fx.Int16).ir_value(),
                fx.Vector(q_packs[ks]).bitcast(fx.Int16).ir_value(),
                acc,
            )
        # apply scale + causal mask
        sv = []
        for e in fx.range_constexpr(4):
            kv = kv0 + k_outer * fx.Int32(4) + fx.Int32(e)
            s = fx.Float32(fx.Vector(acc)[e]) * fx.Float32(sm_scale)
            sv.append((kv <= eff_bound).select(s, neg_inf))

        # online softmax update
        m_loc = sv[0]
        for e in fx.range_constexpr(3):
            m_loc = m_loc.maximumf(sv[e + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(16), fx.Int32(64)))
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(32), fx.Int32(64)))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)

        p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[e] - safe_m) * fx.Float32(LOG2E)))) for e in range(4)]
        l_loc = p[0]
        for e in fx.range_constexpr(3):
            l_loc = l_loc + p[e + 1]
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(16), fx.Int32(64))
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(32), fx.Int32(64))
        l_run = l_run * corr + l_loc

        # GEMM2: o += V^T @ P, output O[d=row, q=col=mn]. We put d on the row and q on the COLUMN
        # so the running normalizer l (indexed by this lane's query q=mn) lines up with the output
        # column at epilogue time. (If you instead make q the output ROW, each lane's inv_l would
        # belong to the wrong query — a classic flash-attention orientation bug.)
        # MFMA result[m=d,n=q]=sum_k A[m,k]B[n,k]: A[d,kv]=V[kv,d] (=V^T), B[q,kv]=P[q,kv].
        b_bf16 = fx.Vector.from_elements([p[e].to(fx.BFloat16) for e in range(4)], fx.BFloat16)  # B=P[q=mn,kv=k_outer*4+e]
        b_i16 = fx.Vector(b_bf16).bitcast(fx.Int16)
        corr4 = fx.Vector.filled(4, fx.Float32(corr), fx.Float32)
        new_o = []
        for dt in fx.range_constexpr(DT):
            # A = V^T: lane holds A[d=mn? no — A[m=d,k=kv]] -> lane(k_outer,mn) holds A[d=mn, kv=k_outer*4+e]
            #          = V[kv=k_outer*4+e, d=dt*16+mn].
            d_row = fx.Int32(dt * 16) + mn  # the d this lane represents (output row index family)
            a_elems = []
            for e in fx.range_constexpr(4):
                kv = kv0 + k_outer * fx.Int32(4) + fx.Int32(e)
                kv_ok = kv < sk_i
                kv_safe = kv_ok.select(kv, fx.Int32(0))
                vval = fx.buffer_ops.buffer_load(rV, kv_safe * fx.Int32(HDV) + d_row, vec_width=1, dtype=fx.BFloat16)
                a_elems.append(kv_ok.select(vval, fx.BFloat16(0.0)))
            a_bf16 = fx.Vector.from_elements(a_elems, fx.BFloat16)
            a_i16 = fx.Vector(a_bf16).bitcast(fx.Int16)
            o_resc = (fx.Vector(o_acc[dt]) * corr4).ir_value()
            o_new = _mfma(a_i16.ir_value(), b_i16.ir_value(), o_resc)
            new_o.append(fx.Vector(o_new))
        st = yield [m_new, l_run] + new_o

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    # l_run is this lane's normalizer for query q = mn. With O[d=row, q=col], every output element
    # this lane writes has q = mn, so one inv_l applies to all of them.
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    for dt in fx.range_constexpr(DT):
        ov = fx.Vector(o_acc[dt])
        for e in fx.range_constexpr(4):
            d = fx.Int32(dt * 16) + k_outer * fx.Int32(4) + fx.Int32(e)  # C row = d within this dt-block
            q = mn  # C result col = q = mn
            fx.buffer_ops.buffer_store((fx.Float32(ov[e]) * inv_l).ir_value(), rO, (q * fx.Int32(HDV) + d).ir_value())


@flyc.jit
def run_attn(Q, K, V, O, sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    attn_kernel(Q, K, V, O, sq, sk, sm_scale, causal).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    for (sq, sk, causal) in [(16, 64, 1), (16, 64, 0), (16, 48, 1)]:
        Q = torch.randn(sq, HD, dtype=torch.bfloat16).cuda()
        K = torch.randn(sk, HD, dtype=torch.bfloat16).cuda()
        V = torch.randn(sk, HDV, dtype=torch.bfloat16).cuda()
        O = torch.zeros(sq, HDV, dtype=torch.float32).cuda()
        sm = 1.0 / HD**0.5
        run_attn(Q, K, V, O, sq, sk, sm, causal, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        S = (Q.float() @ K.float().T) * sm
        if causal:
            qi = torch.arange(sq).view(-1, 1).cuda()
            ki = torch.arange(sk).view(1, -1).cuda()
            S = S.masked_fill(ki > qi + (sk - sq), float("-inf"))
        ref = torch.softmax(S, dim=1) @ V.float()
        err = (O - ref).abs().max().item()
        print(f"sq={sq} sk={sk} causal={causal}  err={err:.4f}  {'PASS' if err < 5e-2 else 'FAIL'}")
