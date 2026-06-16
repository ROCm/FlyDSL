# SPDX-License-Identifier: Apache-2.0
"""Lesson 22c — The all-tricks kernel, MULTI-HEAD (fixes the grid).

Same kernel as lesson_22b, with ONE addition: a head dimension. 22b was single-head, so
its grid was only ceil(sq/BM) workgroups -> at sq1024 that's 8 workgroups on 80 CUs,
badly CU-starved (Lesson 08). Real attention has nq heads, and EACH (head, q-tile) is an
independent workgroup. So the grid becomes:
    grid = nq * ceil(sq / BM)
which is nq x bigger and actually fills the machine. This is the same grid the production
kernels use, and it's why their TFLOPS are far higher than 22b's at the same seqlen.

THE CHANGE (vs 22b): decode (head, q-tile) from the block index and offset every tensor
base by the head. Tensors are laid out [head, seq, hd] (V column-major [head, hd, seq]).
Everything else -- multiwave, column-V, register-P, causal-bound, fast-exp2 -- is identical
to 22b. MHA (one kv head per q head; no GQA, to stay readable).

All other tricks/omissions are exactly as in lesson_22b_all_tricks.py (read that first).

IDIOMATIC STYLE (same as 22b, matching kernels/fmha_prefill_fp8_layout.py): both GEMMs are
driven through a TYPED MMA ATOM (``fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fp8))`` +
``fly.mma_atom_call_ssa`` via the ``_mfma`` helper) instead of the raw
``fx.rocdl.mfma_f32_32x32x16_fp8_fp8`` intrinsic — the MFMA shape/dtype is declared ONCE. Per the
flydsl-layout-algebra skill, everything else stays DIRECT (the K/V cooperative gather + hand LDS
layout, the per-head base offsets, the online softmax, the ds_bpermute P-transpose + cvt_pk_fp8
packing, and the causal-bound math): those are bespoke register packing / swizzle / reductions the
algebra does not express. This is a readability swap of the GEMMs, not a re-layout of the kernel.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22c_multihead.py            # all shapes
      HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_22c_multihead.py 8 1024 1024 1   # nq sq sk causal
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

HD = 128
HDV = 128
KSTEPS = HD // 16    # GEMM1 hd contraction, 32x32x16 fp8 K=16 -> 8 steps
DT = HDV // 32       # GEMM2 d-tiles (4)
BN = 32              # kv per MFMA tile (32x32x16)
import os
NWAVES = int(os.environ.get("FMHA_NWAVES", "4"))
WAVE_ROWS = 32       # each wave owns 32 q-rows
BM = NWAVES * WAVE_ROWS
LOG2E = 1.4426950408889634

_alloc = SmemAllocator(None, arch="gfx942", global_sym_name="lesson22c_smem")
# Only K and V tiles in LDS (P is register-resident). Column-V: V stored [d, kv] in LDS.
_K_BYTES = BN * HD          # one kv-tile of K, [kv, hd] fp8
_V_BYTES = HDV * BN         # one kv-tile of V, [d, kv] fp8 (COLUMN-major)
_K_OFF = 0
_V_OFF = _K_OFF + _K_BYTES
_alloc.ptr = _V_OFF + _V_BYTES

_LGKMCNT0 = 0xC07F


@flyc.kernel
def attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
                qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
                sq: fx.Int32, sk: fx.Int32, nq: fx.Constexpr[int],
                sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int]):
    tid = fx.Int32(fx.thread_idx.x)
    wave = tid // fx.Int32(64)
    lane = tid % fx.Int32(64)
    blk = fx.Int32(fx.block_idx.x)
    sq_i = fx.Int32(sq)
    sk_i = fx.Int32(sk)
    f32t = fx.typing.T.f32
    _ar = fx.arith.unwrap
    neg_inf = fx.Float32(-3.0e38)
    off32 = fx.Int32(32)
    width64 = fx.Int32(64)

    q_local = lane % fx.Int32(32)   # 32x32x16: C col / q index
    half = lane // fx.Int32(32)     # 0/1: which half-group of kv

    # THE GRID FIX: grid = nq * ceil(sq/BM). Decode (head, q-tile) from the block index, then offset
    # each tensor base by the head. nq x more workgroups -> the 80 CUs actually fill up.
    num_q_tiles = (sq_i + fx.Int32(BM - 1)) // fx.Int32(BM)
    qtile = blk % num_q_tiles
    head = blk // num_q_tiles
    # per-head ELEMENT base offsets. Tensors: Q/K/O = [head, seq, hd]; V column-major = [head, hd, kv].
    q_hbase = head * (sq_i * fx.Int32(HD))
    k_hbase = head * (sk_i * fx.Int32(HD))
    v_hbase = head * (fx.Int32(HDV) * sk_i)
    o_hbase = head * (sq_i * fx.Int32(HDV))

    # workgroup handles q-tile `qtile`; wave owns its 32-row slice.
    wave_q0 = qtile * fx.Int32(BM) + wave * fx.Int32(WAVE_ROWS)

    rQ = fx.buffer_ops.create_buffer_resource(Q)
    rK = fx.buffer_ops.create_buffer_resource(K)
    rV = fx.buffer_ops.create_buffer_resource(V)
    rO = fx.buffer_ops.create_buffer_resource(O)

    qrow = wave_q0 + q_local
    qrow_safe = (qrow < sq_i).select(qrow, fx.Int32(0))
    # preload Q fragment: lane holds Q[qrow, hd = ks*16 + half*8 + e] (8 fp8 -> i64) per k-step.
    q_i64 = []
    for ks in fx.range_constexpr(KSTEPS):
        off = q_hbase + qrow_safe * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
        w = fx.buffer_ops.buffer_load(rQ, off // fx.Int32(4), vec_width=2, dtype=fx.Int32)
        q_i64.append(fx.Vector(w).bitcast(fx.Int64)[0])
    qk_descale = fx.Float32(qd_s * kd_s * sm_scale)

    # causal bound (Lesson 14): valid kv <= eff_bound for this lane's query row.
    sk_m1 = sk_i - fx.Int32(1)
    if fx.const_expr(causal != 0):
        cb = qrow + (sk_i - sq_i)
        eff_bound = (cb < sk_m1).select(cb, sk_m1)
    else:
        eff_bound = sk_m1

    # cooperative load: 256-thread (or fewer) coop store of the K/V tile into LDS.
    NTHREADS = NWAVES * 64
    NSLOT = 256  # 32 kv x 8 feature-groups of 16
    NPASS = (NSLOT + NTHREADS - 1) // NTHREADS
    is_h0 = half == fx.Int32(0)
    q_byte = q_local * fx.Int32(4)
    q32_byte = (q_local + fx.Int32(32)) * fx.Int32(4)

    def _cvt4(v0, v1, v2, v3):
        lo = fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v0).ir_value(), fx.Float32(v1).ir_value(), fx.Int32(0).ir_value(), False)
        return fx.rocdl.cvt_pk_fp8_f32(fx.typing.T.i32, fx.Float32(v2).ir_value(), fx.Float32(v3).ir_value(), lo, True)

    # IDIOMATIC (flydsl-layout-algebra skill, "tiled-MMA" recipe): declare the 32x32x16 fp8 MFMA
    # ONCE as a typed layout-API atom and drive BOTH GEMMs through `fly.mma_atom_call_ssa`, instead
    # of repeating the raw `rocdl.mfma_f32_32x32x16_fp8_fp8` intrinsic at each GEMM site. Same hybrid
    # scope as the production kernels/fmha_prefill_fp8_layout.py: the MFMA shape/dtype is named once,
    # the call sites read as a matmul op. The operands are the SAME packed-fp8 i64 / v16f32 acc as
    # before, so the lane dataflow (and the ds_bpermute P-transpose feeding GEMM2) is unchanged.
    f32x16 = fx.typing.T.vec(16, f32t)
    _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.typing.T.f8))

    def _mfma(a, b, c):
        a_raw = a.ir_value() if hasattr(a, "ir_value") else a
        b_raw = b.ir_value() if hasattr(b, "ir_value") else b
        c_raw = c.ir_value() if hasattr(c, "ir_value") else c
        return fly.mma_atom_call_ssa([f32x16], _mma_atom, a_raw, b_raw, c_raw)

    kv_local = lane % fx.Int32(32)

    # causal kv-tile cap (Lesson 14): skip fully-masked tiles.
    n_kv_full = (sk_i + fx.Int32(BN - 1)) // fx.Int32(BN)
    if fx.const_expr(causal == 0):
        n_kv = n_kv_full
    else:
        q_max = blk * fx.Int32(BM) + fx.Int32(BM - 1)
        kv_max = q_max + (sk_i - sq_i)
        n_kv_c = (kv_max + fx.Int32(BN)) // fx.Int32(BN)
        n_kv = (n_kv_c < n_kv_full).select(n_kv_c, n_kv_full)

    m0 = fx.Float32(-3.0e38)
    l0 = fx.Float32(0.0)
    o0 = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(DT)]
    init = [m0, l0] + o0
    for kt, st in range(fx.Index(0), fx.Index(n_kv), fx.Index(1), init=init):
        m_run = st[0]
        l_run = st[1]
        o_acc = [st[2 + d] for d in range(DT)]
        kv0 = fx.Int32(kt) * fx.Int32(BN)

        # ---- cooperative load K [kv,hd] and V [d,kv] (COLUMN-major) into LDS ----
        k_lds = SmemPtr(_alloc.get_base(), _K_OFF, fx.typing.T.i8, shape=(_K_BYTES,)).get()
        vt_lds = SmemPtr(_alloc.get_base(), _V_OFF, fx.typing.T.i8, shape=(_V_BYTES,)).get()
        for p in fx.range_constexpr(NPASS):
            slot = fx.Int32(p * NTHREADS) + tid
            ok = slot < fx.Int32(NSLOT)
            slot_s = ok.select(slot, fx.Int32(0))
            kv_row = slot_s // fx.Int32(8)       # 0..31
            cg = slot_s % fx.Int32(8)            # feature group of 16
            kvg = kv0 + kv_row
            kvg_s = (kvg < sk_i).select(kvg, fx.Int32(0))
            # K: [kv, hd] row-major -> wide 16B load of 16 contiguous hd.
            k_off = k_hbase + kvg_s * fx.Int32(HD) + cg * fx.Int32(16)
            kw = fx.buffer_ops.buffer_load(rK, k_off // fx.Int32(4), vec_width=4, dtype=fx.Int32)
            # V COLUMN-major: V stored [d, kv] -> for this (kv_row, d-group) load 16 d?? No: we want
            # kv contiguous for GEMM2. Store V into LDS as [d, kv]: 16 contiguous-d for this kv from a
            # column-major global V[d, kv] is a strided gather -> instead we load 16 contiguous KV for
            # a fixed d-group. Simpler & matches GEMM2: treat the coop tile as 32 d-rows x ... Here we
            # keep it simple: V global is [d, kv]; load 16 contiguous kv for d = cg*16-block? We need
            # all (d,kv). Use the SAME slot decomposition but read V[d=kv_row?]. To stay clear we load
            # V[d = cg*16 + (0..15), kv = kvg] is strided; instead store column tile directly:
            # global V[d, kv]: element (d, kv) at d*sk + kv. For LDS [d, kv] we copy contiguously.
            d_idx = kv_row                      # reuse 0..31 as a d-row index chunk base *... see note
            # (kept intentionally as the straightforward per-(d,kv) copy below)
            fx.Vector(kw).bitcast(fx.Int8).store(k_lds, [fx.Index(kv_row * fx.Int32(HD) + cg * fx.Int32(16))])
        # V coop load: copy the [HDV x BN] column tile. Each thread copies one (d, kv) run of 16 kv.
        for p in fx.range_constexpr((HDV * BN // 16 + NTHREADS - 1) // NTHREADS):
            s2 = fx.Int32(p * NTHREADS) + tid
            ok2 = s2 < fx.Int32(HDV * BN // 16)
            s2s = ok2.select(s2, fx.Int32(0))
            d_row = s2s // fx.Int32(BN // 16)      # which d (0..HDV-1)
            kvg16 = (s2s % fx.Int32(BN // 16)) * fx.Int32(16)  # kv base (0 or 16)
            # global column-V [d, kv]: row d, 16 contiguous kv starting kv0+kvg16
            gkv = kv0 + kvg16
            gkv_s = (gkv < sk_i).select(gkv, fx.Int32(0))
            voff = v_hbase + d_row * sk_i + gkv_s
            vw = fx.buffer_ops.buffer_load(rV, voff // fx.Int32(4), vec_width=4, dtype=fx.Int32)
            fx.Vector(vw).bitcast(fx.Int8).store(vt_lds, [fx.Index(d_row * fx.Int32(BN) + kvg16)])
        fx.gpu.barrier()

        # ---- GEMM1: S[kv,q] = K @ Q^T (8 k-steps over hd) ----
        acc = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        for ks in fx.range_constexpr(KSTEPS):
            k_elem = kv_local * fx.Int32(HD) + fx.Int32(ks * 16) + half * fx.Int32(8)
            kv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), k_lds, [fx.Index(k_elem)])
            k_i64 = fx.Vector(kv8).bitcast(fx.Int64)[0]
            acc = _mfma(k_i64, q_i64[ks], acc)  # GEMM1: S = K @ Q^T (typed atom)
        sv_raw = fx.Vector(acc)
        # 32x32x16 C-layout: lane holds 16 vals S[kv=(i//4)*8+half*4+i%4, q=q_local].
        sv = []
        for i in fx.range_constexpr(16):
            kv = kv0 + fx.Int32((i // 4) * 8) + half * fx.Int32(4) + fx.Int32(i % 4)
            s = fx.Float32(sv_raw[i]) * qk_descale
            sv.append((kv <= eff_bound).select(s, neg_inf))

        # ---- online softmax (Lesson 04/13): max, fast exp2, sum; one shuffle_xor (2 groups) ----
        m_loc = sv[0]
        for i in fx.range_constexpr(15):
            m_loc = m_loc.maximumf(sv[i + 1])
        m_loc = m_loc.maximumf(m_loc.shuffle_xor(off32, width64))
        m_new = m_run.maximumf(m_loc)
        m_is_neg = m_new < fx.Float32(-1.0e38)
        safe_m = m_is_neg.select(fx.Float32(0.0), m_new)
        corr = fx.Float32(fx.rocdl.exp2(f32t, _ar((m_run - safe_m) * fx.Float32(LOG2E))))
        corr = m_is_neg.select(fx.Float32(0.0), corr)
        p_vals = []
        l_loc = fx.Float32(0.0)
        for i in fx.range_constexpr(16):
            pv = fx.Float32(fx.rocdl.exp2(f32t, _ar((sv[i] - safe_m) * fx.Float32(LOG2E))))
            p_vals.append(pv)
            l_loc = l_loc + pv
        l_loc = l_loc + l_loc.shuffle_xor(off32, width64)
        l_run = l_run * corr + l_loc

        # ---- register-resident P transpose (Lesson 12): 4 ds_bpermute ----
        p_i64_s = []
        for s in fx.range_constexpr(2):
            pack0 = _cvt4(p_vals[s * 8 + 0], p_vals[s * 8 + 1], p_vals[s * 8 + 2], p_vals[s * 8 + 3])
            pack1 = _cvt4(p_vals[s * 8 + 4], p_vals[s * 8 + 5], p_vals[s * 8 + 6], p_vals[s * 8 + 7])
            h0_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack0))
            h0_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q_byte.ir_value(), pack1))
            h1_b0 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack0))
            h1_b1 = fx.Int32(fx.rocdl.ds_bpermute(fx.typing.T.i32, q32_byte.ir_value(), pack1))
            fx.rocdl.s_waitcnt(_LGKMCNT0)
            w0 = is_h0.select(h0_b0, h0_b1)
            w1 = is_h0.select(h1_b0, h1_b1)
            p_i64_s.append(fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0])

        # ---- GEMM2: O[d,q] += V^T @ P. COLUMN-V => V read is contiguous kv, NO transpose. ----
        corr_vec = fx.Vector.filled(16, fx.Float32(corr), fx.Float32)
        for dt in fx.range_constexpr(DT):
            o_acc[dt] = fx.Vector(o_acc[dt]) * corr_vec
        for dt in fx.range_constexpr(DT):
            acc2 = fx.Vector(o_acc[dt]).ir_value()
            d_col = fx.Int32(dt * 32) + (lane % fx.Int32(32))
            for s in fx.range_constexpr(2):
                # V LDS is [d, kv]: 8 contiguous kv for fixed d -> one wide read (the whole point).
                v_elem = d_col * fx.Int32(BN) + fx.Int32(s * 16) + half * fx.Int32(8)
                vv8 = fx.Vector.load(fx.typing.T.vec(8, fx.typing.T.i8), vt_lds, [fx.Index(v_elem)])
                v_i64 = fx.Vector(vv8).bitcast(fx.Int64)[0]
                acc2 = _mfma(v_i64, p_i64_s[s], acc2)  # GEMM2: O = V^T @ P (typed atom)
            o_acc[dt] = fx.Vector(acc2)
        fx.gpu.barrier()
        st = yield [m_new, l_run] + o_acc

    m_run = st[0]
    l_run = st[1]
    o_acc = [st[2 + d] for d in range(DT)]
    l_is_zero = l_run < fx.Float32(1.0e-30)
    inv_l = l_is_zero.select(fx.Float32(0.0), fx.Float32(1.0) / l_run)
    out_scale = fx.Float32(vd_s) * inv_l
    in_b = qrow < sq_i
    if in_b:
        for dt in fx.range_constexpr(DT):
            ov = fx.Vector(o_acc[dt]) * fx.Vector.filled(16, out_scale, fx.Float32)
            ov_bf = fx.Vector(ov).to(fx.BFloat16)
            for j in fx.range_constexpr(4):
                d = fx.Int32(dt * 32) + fx.Int32(j * 8) + half * fx.Int32(4)
                v4 = fx.Vector.from_elements([fx.Vector(ov_bf)[j * 4 + e] for e in range(4)], fx.BFloat16)
                o_idx = fx.Int32(o_hbase + qrow_safe * fx.Int32(HDV) + d)
                fx.buffer_ops.buffer_store(v4.ir_value(), rO, o_idx.ir_value())


@flyc.jit
def run_attn(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor,
             qd_s: fx.Constexpr[float], kd_s: fx.Constexpr[float], vd_s: fx.Constexpr[float],
             sq: fx.Int32, sk: fx.Int32, nq: fx.Constexpr[int],
             sm_scale: fx.Constexpr[float], causal: fx.Constexpr[int],
             grid_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _alloc.finalize()
    attn_kernel(Q, K, V, O, qd_s, kd_s, vd_s, sq, sk, nq, sm_scale, causal).launch(
        grid=(grid_blocks,), block=(NWAVES * 64,), stream=stream)


def _quant(t):
    s = t.abs().max().item() / 224.0
    return (t / s).to(torch.float8_e4m3fnuz), s


def _run_one(nq, sq, sk, causal, bench=False):
    torch.manual_seed(0)
    # tensors laid out [head, seq, hd]; V column-major [head, hd, kv].
    Qf = torch.randn(nq, sq, HD)
    Kf = torch.randn(nq, sk, HD)
    Vf = torch.randn(nq, sk, HDV)
    Qq, qd = _quant(Qf)
    Kq, kd = _quant(Kf)
    Vq, vd = _quant(Vf)
    Vcol = Vq.transpose(1, 2).contiguous()  # [head, hd, kv]
    O = torch.zeros(nq, sq, HDV, dtype=torch.bfloat16).cuda()
    sm = 1.0 / HD**0.5
    grid = nq * ((sq + BM - 1) // BM)
    args = [Qq.cuda(), Kq.cuda(), Vcol.cuda(), O, qd, kd, vd, sq, sk, nq, sm, causal, grid]
    run_attn(*args, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    # multi-head reference
    S = torch.einsum("hqd,hkd->hqk", Qq.float() * qd, Kq.float() * kd) * sm
    if causal:
        qi = torch.arange(sq).view(1, -1, 1)
        ki = torch.arange(sk).view(1, 1, -1)
        S = S.masked_fill(ki > qi + (sk - sq), float("-inf"))
    ref = torch.einsum("hqk,hkd->hqd", torch.softmax(S, dim=2), Vq.float() * vd)
    err = (O.float().cpu() - ref).abs().max().item()
    print(f"nq={nq} sq={sq} sk={sk} causal={causal} BM={BM} grid={grid}  err={err:.4f}  "
          f"{'PASS' if err < 6e-2 else 'FAIL'}", end="")
    if bench:
        # manual wall-clock (do_bench mis-times this kernel at large seq -> use a plain timed loop).
        import time
        for _ in range(5):
            run_attn(*args, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            run_attn(*args, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N * 1e3
        tf = nq * (2.0 * sq * sk * HD + 2.0 * sq * sk * HDV) / 2.0 / 1e9 / ms
        print(f"   {ms:.2f}ms  {tf:.1f} TF", end="")
    print()


if __name__ == "__main__":
    import subprocess
    import sys
    if len(sys.argv) >= 5:
        _run_one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                 bench=(len(sys.argv) > 5 and sys.argv[5] == "bench"))
    else:
        # correctness shapes, then a perf shape (nq=8 like the compare script) with bench.
        for shp in [("8", "256", "256", "1"), ("8", "1024", "1024", "1"),
                    ("4", "512", "768", "1"), ("8", "256", "256", "0"),
                    ("8", "16384", "16384", "1", "bench")]:
            subprocess.run([sys.executable, __file__, *shp], check=False)
