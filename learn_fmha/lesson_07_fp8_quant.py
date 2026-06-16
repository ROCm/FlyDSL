# SPDX-License-Identifier: Apache-2.0
"""Lesson 07 — fp8 (e4m3 FNUZ): quantization, descale, p_scale, and the REAL P-transpose.

Goal: take the bf16 streaming attention of Lesson 06 and change ONE thing — the dtype —
to fp8 e4m3 FNUZ (the gfx942 fp8 format). This is the bridge to the production kernel.
The fp8 switch forces four new things, each a teaching point:

  (1) fp8 MFMA: mfma_f32_16x16x32_fp8_fp8 (K=32, not 16). Each lane feeds 8 fp8 (=i64).
  (2) fp8 PACKING: there is NO `.to(fp8)` that lowers — it emits a bare arith.truncf that
      the backend rejects. You MUST use rocdl.cvt_pk_fp8_f32(res, a, b, old, word_sel),
      which packs TWO f32 -> two fp8 bytes into an i32 (4 calls -> an i64 of 8 fp8).
  (3) DESCALE: fp8 has ~2 decimal digits, so Q/K/V are quantized with per-tensor (here)
      scales and the kernel multiplies the raw-fp8 MFMA result by the descales. Q/K
      descale is applied to S AFTER the QK MFMA; V descale folds into the output epilogue.
  (4) p_scale (shown but set to 1.0): P (the probabilities) are tiny; multiplying by a
      per-head p_scale before the fp8 cast keeps them in e4m3 range, then it cancels in
      O/l. Implemented as +log2(p_scale) in the exp bias. We keep it = 1 here (its effect
      is pure quantization precision) but wire the slot so the structure matches production.

### THE REAL P-TRANSPOSE (this is why fp8 is harder than bf16)
With the 16x16x32 fp8 MFMA, GEMM2's P operand (B) needs 8 CONTIGUOUS kv per lane:
B[q=mn, kv = k_outer*8 + e], e=0..7. But GEMM1/softmax produced P with only 4 kv per
lane (kv = k_outer*4 + e) AND grouped differently. 4 != 8, wrong grouping -> the registers
do NOT line up (unlike Lesson 05's lucky 16x16x16 case). So we MUST transpose P.
Simplest fix (this lesson): round-trip P through LDS — store it q-major as fp8
(p_lds[q*BKV + kv]), barrier, reload 8 contiguous kv per lane as the i64 B-operand.
Lessons 11-12 make this cheaper (V-transpose / register ds_bpermute); Lesson 17 removes
the V side entirely with column-major V.

Single q-tile (16 q) vs all kv (16/tile), one wavefront. BKV=32 per kv-tile (one fp8
MFMA K-step covers 32 kv).

### Idiomatic style (flydsl-layout-algebra skill)
The 16x16x32 fp8 MFMA is declared ONCE as a typed atom -- `fx.make_mma_atom(fx.rocdl.MFMA(16,16,32,
f8))` -- and both GEMMs run through `fly.mma_atom_call_ssa`, instead of repeating the raw
`mfma_f32_16x16x32_fp8_fp8_` intrinsic. Everything fp8-specific stays direct on purpose (the skill's
"hand-rolled register packing the algebra does not express cleanly" criterion): cvt_pk_fp8_f32
packing, i32/i64 dword loads, per-tensor descale, and the P-transpose through LDS.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_07_fp8_quant.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

BQ = 16
BKV = 32  # one fp8 MFMA K-step covers 32 kv
HD = 64
HDV = 64
KSTEPS = HD // 32  # GEMM1 contraction over hd, fp8 K=32 -> 2 steps
DT = HDV // 16
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="lesson07_smem")
_P_BYTES = BQ * BKV  # q-major P scratch, fp8 (1 byte): 16*32 = 512 B
_alloc.ptr = _P_BYTES


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)
    mn = lane % fx.Int32(16)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    neg_inf = fx.Float32(-3.0e38)
    log2_pscale = fx.Float32(0.0)  # p_scale = 1 here; slot kept for structure parity

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    # IDIOMATIC (flydsl-layout-algebra skill, "tiled-MMA" recipe): declare the 16x16x32 fp8 MFMA
    # ONCE as a typed layout-API atom and drive BOTH GEMMs through `fly.mma_atom_call_ssa`, instead
    # of repeating the raw `rocdl.mfma_f32_16x16x32_fp8_fp8_` intrinsic. Operands stay the SAME packed
    # i64 fp8 fragments + vec(4) f32 accumulator, so lane dataflow is unchanged.
    # The fp8-SPECIFIC parts stay direct (skill: "hand-rolled register packing the algebra does not
    # express cleanly"): cvt_pk_fp8_f32 packing, i32/i64 dword loads, per-tensor descale, and the
    # P-transpose through LDS. The MFMA is the one piece that maps onto the typed atom cleanly.
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.typing.T.f8))

    def _mfma(a_i64, b_i64, acc):
        return fly.mma_atom_call_ssa([f32x4], _mma_atom, a_i64, b_i64, acc)

    # preload this lane's Q fragment (fp8, 8 per k-step as i64), reused across kv tiles.
    q_packs = []
    for ks in fx.range_constexpr(KSTEPS):
        off = mn * fx.Int32(HD) + fx.Int32(ks * 32) + k_outer * fx.Int32(8)
        w = fx.buffer_ops.buffer_load(rQ, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
        q_packs.append(fx.Vector(w).bitcast(fx.Int64)[0])

    qrow = mn
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_i - fx.Int32(1)).select(cb, sk_i - fx.Int32(1))
    else:
        eff_bound = sk_i - fx.Int32(1)

    n_kv = (sk_i + fx.Int32(BKV - 1)) // fx.Int32(BKV)
    qk_descale = fx.Float32(qd_s * kd_s * sm_scale)
    init = [fx.Float32(-3.0e38), fx.Float32(0.0)] + [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(DT)]
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BKV)

        # ---- GEMM1: TWO 16-kv sub-tiles fill the 32-kv tile ----
        # The fp8 MFMA is 16x16x32: M=16(kv), N=16(q), K=32(hd). So ONE MFMA outputs 16 kv.
        # To produce the 32 kv that GEMM2's K=32 contraction needs, we run GEMM1 TWICE
        # (sub=0: kv 0..15, sub=1: kv 16..31), each accumulating over hd (KSTEPS=2). This is
        # exactly why the production kernel has an inner "sub-tile" structure.
        # After both, this lane holds 8 scores: sv[sub*4+e] = S[kv = kv0+sub*16+k_outer*4+e, q=mn].
        sv = []
        for sub in fx.range_constexpr(2):
            acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
            for ks in fx.range_constexpr(KSTEPS):
                k_row = kv0 + fx.Int32(sub * 16) + mn
                k_row_safe = (k_row < sk_i).select(k_row, fx.Int32(0))
                off = k_row_safe * fx.Int32(HD) + fx.Int32(ks * 32) + k_outer * fx.Int32(8)
                kw = fx.buffer_ops.buffer_load(rK, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
                k_i64 = fx.Vector(kw).bitcast(fx.Int64)[0]
                acc = _mfma(k_i64.ir_value(), q_packs[ks].ir_value(), acc)
            for e in fx.range_constexpr(4):
                kv = kv0 + fx.Int32(sub * 16) + k_outer * fx.Int32(4) + fx.Int32(e)
                s = fx.Float32(fx.Vector(acc)[e]) * qk_descale
                sv.append((kv <= eff_bound).select(s, neg_inf))

        # ---- online softmax over all 8 own scores + cross-lane (4 k_outer groups) ----
        m_loc = sv[0]
        for i in fx.range_constexpr(7):
            m_loc = m_loc.maximumf(sv[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(16), fx.Int32(64)))
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(fx.Int32(32), fx.Int32(64)))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        p = [fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[i] - safe_m) * fx.Float32(LOG2E) + log2_pscale))) for i in range(8)]
        l_loc = p[0]
        for i in fx.range_constexpr(7):
            l_loc = l_loc + p[i + 1]
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(16), fx.Int32(64))
        l_loc = l_loc + l_loc.shuffle_xor(fx.Int32(32), fx.Int32(64))
        l_run = l_run * corr + l_loc

        # ---- THE P-TRANSPOSE through LDS ----
        # store this lane's 8 P as fp8 into p_lds[q*BKV + kv_in_tile], where for index i=sub*4+e the
        # kv_in_tile = sub*16 + k_outer*4 + e (covers all 32 kv 1:1). cvt_pk_fp8_f32 packs 2 f32 ->
        # 2 fp8 bytes into an i32. NOTE: recreate the SmemPtr view INSIDE the scf.for body — the loop
        # invalidates views created outside it (a FlyDSL scf.for + shared-memory gotcha).
        p_lds = SmemPtr(_alloc.get_base(), 0, fx.typing.T.i8, shape=(_P_BYTES,)).get()
        lo0 = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p[0].ir_value(), p[1].ir_value(), fx.Int32(0).ir_value(), False)
        w0 = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p[2].ir_value(), p[3].ir_value(), lo0, True)
        lo1 = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p[4].ir_value(), p[5].ir_value(), fx.Int32(0).ir_value(), False)
        w1 = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, p[6].ir_value(), p[7].ir_value(), lo1, True)
        p_i8 = fx.Vector(fx.Vector.from_elements([w0, w1], fx.Int32)).bitcast(fx.Int8)
        for i in fx.range_constexpr(8):
            sub = i // 4
            e = i % 4
            kv_in_tile = fx.Int32(sub * 16) + k_outer * fx.Int32(4) + fx.Int32(e)
            fx.Vector.from_elements([fx.Vector(p_i8)[i]], fx.Int8).store(p_lds, [fx.Index(mn * fx.Int32(BKV) + kv_in_tile)])
        fx.gpu.barrier()

        # ---- GEMM2: O[d=row, q=col=mn] += V^T @ P ----
        # B = P reloaded from LDS: lane needs 8 contiguous kv = k_outer*8 + e for q=mn.
        p_base = mn * fx.Int32(BKV) + k_outer * fx.Int32(8)
        p_vec8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), p_lds, [fx.Index(p_base)])
        p_i64 = fx.Vector(p_vec8).bitcast(fx.Int64)[0]
        corr4 = fx.Vector.filled(4, fx.Float32(corr), fx.Float32)
        new_o = []
        for dt in fx.range_constexpr(DT):
            d_row = fx.Int32(dt * 16) + mn  # A=V^T: lane holds V[kv=k_outer*8+e, d=dt*16+mn]
            v_elems = []
            for e in fx.range_constexpr(8):
                kv = kv0 + k_outer * fx.Int32(8) + fx.Int32(e)
                kv_ok = kv < sk_i
                kv_safe = kv_ok.select(kv, fx.Int32(0))
                vv = fx.buffer_ops.buffer_load(rV, kv_safe * fx.Int32(HDV) + d_row, vec_width=1, dtype=fx.Int8)
                v_elems.append(kv_ok.select(vv, fx.Int8(0)))
            v_i64 = fx.Vector(fx.Vector.from_elements(v_elems, fx.Int8)).bitcast(fx.Int64)[0]
            o_resc = (fx.Vector(o_acc[dt]) * corr4).ir_value()
            o_new = _mfma(v_i64.ir_value(), p_i64.ir_value(), o_resc)
            new_o.append(fx.Vector(o_new))
        fx.gpu.barrier()
        st = yield [m_new, l_run] + new_o

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    out_scale = fx.Float32(vd_s) * inv_l  # V descale folds into the output normalization
    for dt in fx.range_constexpr(DT):
        ov = fx.Vector(o_acc[dt])
        for e in fx.range_constexpr(4):
            d = fx.Int32(dt * 16) + k_outer * fx.Int32(4) + fx.Int32(e)
            q = mn
            o_bf = (fx.Float32(ov[e]) * out_scale).to(fx.BFloat16)
            fx.buffer_ops.buffer_store(o_bf.ir_value(), rO, (q * fx.Int32(HDV) + d).ir_value())


@flyc.jit
def run_attn(Q, K, V, O, qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
             sq: fx.Int32, sk: fx.Int32, sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
             stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, sm_scale, causal).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def _quant(t):
    s = t.abs().max().item() / 224.0  # e4m3 max ~448; use 224 for headroom
    q = (t / s).to(torch.float8_e4m3fnuz)
    return q, s


def _run_one(sq, sk, causal):
    torch.manual_seed(0)
    Qf = torch.randn(sq, HD)
    Kf = torch.randn(sk, HD)
    Vf = torch.randn(sk, HDV)
    Qq, qd = _quant(Qf)
    Kq, kd = _quant(Kf)
    Vq, vd = _quant(Vf)
    O = torch.zeros(sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD**0.5
    run_attn(Qq.cuda(), Kq.cuda(), Vq.cuda(), O, qd, kd, vd, sq, sk, sm, causal, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    # reference from the DEQUANTIZED fp8 inputs (so we measure the kernel, not quant noise)
    S = (Qq.float() * qd @ (Kq.float() * kd).T) * sm
    if causal:
        qi = torch.arange(sq).view(-1, 1)
        ki = torch.arange(sk).view(1, -1)
        S = S.masked_fill(ki > qi + (sk - sq), float("-inf"))
    ref = torch.softmax(S, dim=1) @ (Vq.float() * vd)
    err = (O.float().cpu() - ref).abs().max().item()
    print(f"sq={sq} sk={sk} causal={causal}  err={err:.4f}  {'PASS' if err < 6e-2 else 'FAIL'}")


if __name__ == "__main__":
    # GOTCHA (teaching point): the module-global SmemAllocator is finalize()d ONCE per process
    # (it has a `finalized` guard). If you JIT a SECOND shape in the SAME process, the kernel
    # recompiles but finalize is skipped -> "does not reference a valid global memref". So we run
    # each shape in its OWN subprocess. (Production tests/ck_check.py do exactly this — fork per
    # shape.) Run a single shape directly with:  python3 lesson_07_fp8_quant.py 16 64 1
    import subprocess
    import sys

    if len(sys.argv) == 4:
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        for shp in [(16, 64, 1), (16, 64, 0), (16, 48, 1)]:
            subprocess.run([sys.executable, __file__, *map(str, shp)], check=False)
