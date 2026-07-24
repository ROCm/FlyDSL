# SPDX-License-Identifier: MIT
# FlyDSL GQA sliding window attention kernel
#
# Each of the 8 waves independently computes attention for its own 32-row Q tile.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

MFMA_MASK = 0x08
VALU_MASK = 0x02
EXP_MASK = 0x400
RESCALE_THRESHOLD = 8.0


def build_gqa_attn(
    *,
    ATTN_B=16,
    ATTN_H=64,
    ATTN_H_KV=8,
    ATTN_D=128,
    waves_per_eu=2,
    sliding_window=None,
):
    GROUP_SIZE = ATTN_H // ATTN_H_KV
    Q_BLOCK_SIZE = 32
    KV_BLOCK_SIZE = 64

    NUM_WARPS = 8
    WARP_SIZE = 64

    # Sliding window (aiter semantics): sliding_window=(LEFT, RIGHT). With
    # seq_len_q == seq_len_kv, query row i attends keys j in [i-LEFT, i+RIGHT]
    if sliding_window is not None:
        swa_left, swa_right = sliding_window
        assert swa_left >= 0 and swa_right >= 0, "sliding_window (LEFT, RIGHT) must be >= 0"
        _q_rows_per_cta = Q_BLOCK_SIZE * NUM_WARPS
        _span = swa_left + (_q_rows_per_cta - 1) + swa_right
        _nt = -(-(_span + KV_BLOCK_SIZE) // KV_BLOCK_SIZE)  #
        NT_BAND = max(4, ((_nt + 3) // 4) * 4)
    else:
        swa_left = swa_right = None
        NT_BAND = None

    NUM_THREADS = WARP_SIZE * NUM_WARPS

    D = ATTN_D

    if D == 128:
        TEMPERATURE_SCALE = 0.08838834764 * 1.44269504089
    else:
        TEMPERATURE_SCALE = 0.125 * 1.44269504089

    BYTES_PER_THREAD = 16
    BYTES_PER_WARP = BYTES_PER_THREAD * WARP_SIZE  # 1024
    BYTES_PER_MEMCPY = BYTES_PER_THREAD * NUM_THREADS  # 8192

    _LDS_TILE_ELEMS = KV_BLOCK_SIZE * ATTN_D

    @fx.struct
    class SharedStorage:
        k0: fx.Array[fx.BFloat16, _LDS_TILE_ELEMS, 16]
        k1: fx.Array[fx.BFloat16, _LDS_TILE_ELEMS, 16]
        v0: fx.Array[fx.BFloat16, _LDS_TILE_ELEMS, 16]
        v1: fx.Array[fx.BFloat16, _LDS_TILE_ELEMS, 16]

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def attend_ker(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_stride1: fx.Int32,
        K_stride1: fx.Int32,
        V_stride1: fx.Int32,
        O_stride1: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
    ):
        f32 = fx.Float32
        bf16 = fx.BFloat16
        i32 = fx.Int32
        v8bf16_t = Vec.make_type(8, bf16)
        v4bf16_t = Vec.make_type(4, bf16)
        v16f32_t = Vec.make_type(16, f32)
        fm = arith.FastMathFlags.fast

        NEG_INF = f32(float("-inf"))
        NEG_FLOOR = f32(-30.0)

        def sgpr(x):
            # Hoist a wave-uniform value into an SGPR (readfirstlane).
            return i32(rocdl.readfirstlane(T.i32, _raw(i32(x))))

        def fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm)

        def fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm)

        def fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm)

        def fmaxf(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result

        def bcast16(scalar):
            return Vec.from_elements([fx.Float32(scalar)], f32).broadcast_to(16)

        def hw_exp2_scalar(x):
            return rocdl.exp2(f32.ir_type, _raw(x))

        def hw_exp2_v16(v16):
            src = Vec(v16)
            outs = [hw_exp2_scalar(src[k]) for k in range_constexpr(16)]
            return Vec.from_elements(outs, f32)

        # of the raw rocdl.mfma intrinsic. The atom call is lowered by the fly
        # atom->SSA passes, which the backend scheduler can reason about and

        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, bf16))

        def mfma(a_v8, b_v8, c_v16):
            return fly.mma_atom_call_ssa([v16f32_t], _mma_atom, _raw(a_v8), _raw(b_v8), _raw(c_v16))

        # ---------------- prefill offsets ----------------
        def prefill_offsets(is_k, row_stride):
            ST_ROWS = 32 if is_k else 8
            ST_COLS = 32
            ST_ROW_BYTES = ST_COLS * 2
            ST_BYTES = ST_ROWS * ST_ROW_BYTES
            ST_PER_ROW = 128 // ST_COLS
            offs = []
            for i in range_constexpr(2):
                lane_byte_off = lid * BYTES_PER_THREAD + wid * BYTES_PER_WARP + i * NUM_WARPS * BYTES_PER_WARP
                subtile_id = lane_byte_off // ST_BYTES
                subtile_row = subtile_id // ST_PER_ROW
                subtile_col = subtile_id % ST_PER_ROW
                sub_off = lane_byte_off % ST_BYTES
                row = sub_off // ST_ROW_BYTES
                col = (sub_off % ST_ROW_BYTES) // 2
                if const_expr(is_k):
                    offset = (row * 32 + col) * 2
                    s1 = ((offset % 1024) >> 9) << 5
                    s2 = ((offset % 2048) >> 10) << 4
                    sw = offset ^ s1 ^ s2
                else:
                    sw = (row * 32 + col) * 2
                sw_row = sw // ST_ROW_BYTES
                sw_col = (sw % ST_ROW_BYTES) // 2
                global_row = sw_row + subtile_row * ST_ROWS
                global_col = sw_col + subtile_col * ST_COLS
                # Element offsets (voffset); the BufferCopyLDS128b atom scales by
                # the bf16 element width to bytes.
                offs.append(global_row * row_stride + global_col)
            return offs

        # ---------------- group_load: global -> LDS ----------------
        # High-level DMA: a BufferCopyLDS128b copy atom + fx.copy lowers to the
        # same rocdl.raw_ptr_buffer_load_lds as the raw path, but the atom scales
        # element-indexed src/soffset by the bf16 element width. voffset/soffset
        # are therefore ELEMENTS here (raw path used bytes = elems * 2).
        _dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _lds_dma_ptr_ty = fx.PointerType.get(bf16.ir_type, 2, BYTES_PER_THREAD)

        def group_load(lds_base, tile, offsets, src_div, base_elems, row_stride):
            soff_elems = base_elems + tile * (KV_BLOCK_SIZE * row_stride)
            soff_elems = sgpr(soff_elems)
            for i in range_constexpr(2):
                lds_ptr = fx.inttoptr(_lds_dma_ptr_ty, i32(lds_base + i32(i * BYTES_PER_MEMCPY)))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(src_div, (None, i32(offsets[i])))
                fx.copy(_dma_atom, src, dst, soffset=i32(soff_elems))

        def load_k(tile, buf):
            group_load(k_lds_base[buf], tile, off_K, k_src_div[buf], k_base_elems, K_stride1)

        def load_v(tile, buf):
            group_load(v_lds_base[buf], tile, off_V, v_src_div[buf], v_base_elems, V_stride1)

        # ---------------- LDS -> registers ----------------
        # Provenance-preserving LDS read: add_offset off the per-buffer smem
        # iterator lowers to a getelementptr (not ptrtoint->inttoptr), so LLVM
        # alias analysis proves k0/k1/v0/v1 don't alias and the AMDGPU waitcnt
        # pass drops the conservative vmcnt(0) drains between double-buffered
        # DMAs and LDS reads (the job the alias-scope metadata did in v1).
        def _read_v8bf16(smem_ptr, byte_off):
            elem_off = i32(byte_off) // i32(2)  # bf16 elems; byte_off always even
            off = fx.add_offset(smem_ptr, fx.make_int_tuple(elem_off))
            return fx.make_view(off, fx.make_layout(8, 1)).load()

        def swizzle_k_expr(row, col):
            offset = (row * 32 + col) * 2
            s1 = ((offset % 1024) >> 9) << 5
            s2 = ((offset % 2048) >> 10) << 4
            return offset ^ s1 ^ s2

        def load_k_regs(buf):
            smem_ptr = k_smem[buf]
            row_offset = lid % 32
            col_offset = 8 * (lid // 32)
            ST_BYTES = 32 * 32 * 2
            kreg = [[None] * 8 for _ in range_constexpr(2)]
            for ii in range_constexpr(2):
                for jj in range_constexpr(4):
                    off_imm = (ii * 4 + jj) * ST_BYTES
                    for j in range_constexpr(2):
                        col = j * 16 + col_offset
                        byte_off = swizzle_k_expr(row_offset, col) + off_imm
                        kreg[ii][jj * 2 + j] = _read_v8bf16(smem_ptr, byte_off)
            return kreg  # [n=2][k=8] v8bf16

        # Transposing LDS read via the CDNA4 ds_read_tr16 copy atom, provenance-
        # preserving (getelementptr, no ptrtoint->inttoptr) so LLVM alias analysis
        # keeps k0/k1/v0/v1 disjoint and drops conservative vmcnt(0) drains.
        #
        # recast_iter the sw-shifted per-lane base ONCE: the RecastIterOp is a
        # fusion barrier that stops the nested-add_offset rewrite (MemrefLowering
        # .td) from merging the runtime sw into each per-read index. LLVM then
        # sees gep(gep(@sym, sw), const), so every per-read constant folds into
        # the ds_read offset: immediate off a single base VGPR -- no register
        # scatter, no spilling (the naive add_offset form spills 44 VGPRs).
        _v_tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), bf16)
        _v_tr_layout = fx.make_layout(4, 1)

        def load_v_regs(buf):
            smem_ptr = v_smem[buf]
            row_offset = ((lid % 16) // 4) + ((lid // 32) * 4)
            col_offset = ((lid % 4) * 4) + (16 * ((lid % 32) // 16))
            col_in_sub = col_offset % 32
            ST_BYTES = 8 * 32 * 2
            ST_PER_ROW = 4
            sw_elems = i32(row_offset * 32 + col_in_sub)  # sw/2, runtime
            base_ptr = fx.recast_iter(bf16, fx.add_offset(smem_ptr, fx.make_int_tuple(sw_elems)))
            vreg = [[None] * 4 for _ in range_constexpr(4)]
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    halves = []
                    for k in range_constexpr(2):
                        off = (i * 2 * ST_PER_ROW + j) * ST_BYTES + k * ST_PER_ROW * ST_BYTES
                        view = fx.make_view(
                            fx.add_offset(base_ptr, fx.make_int_tuple(off // 2)),
                            _v_tr_layout,
                        )
                        res = fly.copy_atom_call_ssa([v4bf16_t], _v_tr_atom, view)
                        halves.append(Vec(res))
                    vreg[i][j] = halves[0].shuffle(halves[1], list(range(8)))
            return vreg

        # ---------------- O store ----------------
        def store_o(o_reg):
            row_offset = lid % 32
            col_offset = 4 * (lid // 32)
            base = (
                batch_idx * (i32(seq_len_q) * (ATTN_H * ATTN_D))
                + tile_idx * (Q_BLOCK_SIZE * ATTN_H * ATTN_D)
                + head_idx * ATTN_D
            )
            o_store_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), bf16)
            for j in range_constexpr(4):
                ov = Vec(o_reg[j])
                for k in range_constexpr(4):
                    col = 32 * j + col_offset + k * 8
                    elem0 = k * 4  # (idx=k*2 float2) -> 4 f32 per k
                    elems = [ov[elem0 + e] for e in range_constexpr(4)]
                    # Vectorized store: 4 bf16 as one buffer_store_b64 (mirrors HIP
                    # store_o_global's buffer_store_b64), instead of 4 scalar shorts.
                    vbf = Vec.from_elements(elems, f32).to(bf16)
                    off = base + row_offset * O_stride1 + col
                    fx.memref_store_vec(vbf, o_store_reg)
                    fx.copy(_o_store_atom, o_store_reg, fx.slice(o_div, (None, i32(off))))

        def mma_AtB_QK(A, B, C):
            D = [None, None]
            for n in range_constexpr(2):
                acc = mfma(A[0][n], B[0][0], C[n])
                for k in range_constexpr(1, 8):
                    acc = mfma(A[k][n], B[k][0], acc)
                D[n] = acc
            return D

        def ov_slice(o_reg, vreg_k, att_bf_k):
            for n in range_constexpr(4):
                o_reg[n] = mfma(vreg_k[n], att_bf_k, o_reg[n])
            return o_reg

        def mma_AtB_OV_slice(D, A, B):
            for n in range_constexpr(4):
                D[n] = mfma(A[n], B, D[n])
            return D

        def mma_AtB_OV(C, A, B):
            D = [None] * 4
            for n in range_constexpr(4):
                acc = mfma(A[0][n], B[0], C[n])
                for k in range_constexpr(1, 4):
                    acc = mfma(A[k][n], B[k], acc)
                D[n] = acc
            return D

        # ---------------- softmax helpers ----------------
        # Cross-32-lane reduction peer via permlane32_swap
        def _permlane32_pair(scalar_f32):
            v_i32 = _raw(f32(scalar_f32).bitcast(i32))
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
            lhs_i32 = _llvm.extractvalue(T.i32, swapped, [0])
            rhs_i32 = _llvm.extractvalue(T.i32, swapped, [1])
            lhs = i32(lhs_i32).bitcast(f32)
            rhs = i32(rhs_i32).bitcast(f32)
            return lhs, rhs

        def col_max(att):
            lo = Vec(att[0])
            hi = Vec(att[1])
            mx = lo[0]
            for r in range_constexpr(1, 16):
                mx = fmaxf(mx, lo[r])
            for r in range_constexpr(16):
                mx = fmaxf(mx, hi[r])
            lhs, rhs = _permlane32_pair(fx.Float32(mx))
            return fmaxf(lhs, rhs)

        def mul_o(o_reg, scal):
            b = bcast16(scal)
            return [arith.mulf(_raw(o_reg[n]), _raw(b), fastmath=fm) for n in range_constexpr(4)]

        def sub_col(att, mx):
            b = bcast16(mx)
            return [arith.subf(_raw(att[n]), _raw(b), fastmath=fm) for n in range_constexpr(2)]

        def exp2_one(v16):
            return hw_exp2_v16(v16)

        def finish_scalar(h0_exp_v16, h1_sub_v16, norm):
            h0 = [Vec(h0_exp_v16)[r] for r in range_constexpr(16)]
            h1 = [hw_exp2_scalar(Vec(h1_sub_v16)[r]) for r in range_constexpr(16)]
            sm = h0[0]
            for r in range_constexpr(1, 16):
                sm = fadd(sm, h0[r])
            for r in range_constexpr(16):
                sm = fadd(sm, h1[r])
            lhs, rhs = _permlane32_pair(fx.Float32(sm))
            norm = fadd(norm, fadd(lhs, rhs))
            packs = [
                Vec.from_elements([h0[e] for e in range_constexpr(0, 8)], f32).to(bf16),
                Vec.from_elements([h0[e] for e in range_constexpr(8, 16)], f32).to(bf16),
                Vec.from_elements([h1[e] for e in range_constexpr(0, 8)], f32).to(bf16),
                Vec.from_elements([h1[e] for e in range_constexpr(8, 16)], f32).to(bf16),
            ]
            return packs, norm

        # of each memory block: a side-effecting no-op the scheduler cannot move
        # past, so the following buffer_load_lds DMAs stay pinned inside their
        # block instead of drifting up into the previous compute block (which
        # forced the compiler to insert conservative vmcnt(3/5/6) drains).
        def nop_anchor():
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_nop 7", "", has_side_effects=True)

        # Value anchors (mirror flash_attn_gfx950.py _anchor_v_o / _anchor_v_p): a
        # no-op inline-asm that reads and re-defines the values in place (identity
        # "=v,...,0,1,..." tie), pinning them at this source position. The scheduler
        # cannot hoist a later ds_read/DMA above the anchor, which stops the compiler
        # from inserting conservative vmcnt(3/5/6) partial drains around the O
        # accumulator and P packs.
        def _anchor_vals(vals):
            n = len(vals)
            raws = [_raw(v) for v in vals]
            elems = ", ".join(str(r.type) for r in raws)
            ret_ty = ir.Type.parse(f"!llvm.struct<({elems})>")
            outs = ",".join("=v" for _ in range_constexpr(n))
            ties = ",".join(str(k) for k in range_constexpr(n))
            ret = _llvm.inline_asm(ret_ty, raws, "", f"{outs},{ties}", has_side_effects=True)
            return [_llvm.extractvalue(raws[k].type, ret, [k]) for k in range_constexpr(n)]

        _USE_VALUE_ANCHORS = False

        def anchor_o(o_reg):
            if const_expr(not _USE_VALUE_ANCHORS):
                return o_reg
            return _anchor_vals(o_reg)  # 4 x v16f32

        def anchor_p(packs):
            if const_expr(not _USE_VALUE_ANCHORS):
                return packs
            return _anchor_vals(packs)  # 4 x v8bf16

        def sched_pairs(pairs, valu_cnt, group):
            for _p in range_constexpr(pairs):
                rocdl.sched_group_barrier(MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(VALU_MASK, valu_cnt, group)

        def sched_exp_pairs(pairs, exp_cnt, group):
            for _p in range_constexpr(pairs):
                rocdl.sched_group_barrier(MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(EXP_MASK, exp_cnt, group)

        # s_waitcnt via the ROCDL intrinsic (not inline asm). The AMDGPU
        # SIInsertWaitcnts backend pass understands rocdl.s_waitcnt and folds it
        # into its own analysis, so it does NOT re-insert conservative waits on
        # top (vmcnt(0) full-drains / extra partial lgkmcnt) the way it does for
        # an opaque inline-asm blob. gfx950 s_waitcnt bitfield encoding, matching
        _VMCNT_LO_MASK = 0xF
        _LGKMCNT_EXPCNT_BASE = 0x3F70  # vmcnt=0, expcnt=7(max), lgkmcnt=63(max)
        _VMCNT_HI_SHIFT = 14
        _VMCNT_HI_MASK = 0x3
        _LGKMCNT_0_ONLY = 0xC07F  # vmcnt=63(max), expcnt=7(max), lgkmcnt=0

        def wait_vmcnt(n):
            # vmcnt(n) only; leave lgkmcnt/expcnt maxed (no wait on those).
            val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
            rocdl.s_waitcnt(val)

        def wait_lgkmcnt0():
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)

        # ---------------- LDS ----------------
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_smem_0 = lds.k0.ptr
        k_smem_1 = lds.k1.ptr
        v_smem_0 = lds.v0.ptr
        v_smem_1 = lds.v1.ptr
        k_smem = [k_smem_0, k_smem_1]
        v_smem = [v_smem_0, v_smem_1]

        tid = fx.thread_idx.x
        # (wave, lane) decomposition: layout (NUM_WARPS, WARP_SIZE):(WARP_SIZE, 1)
        # maps tid -> (tid // WARP_SIZE, tid % WARP_SIZE).
        coord_wave_lane = fx.idx2crd(fx.Int32(tid), fx.make_layout((NUM_WARPS, WARP_SIZE), (WARP_SIZE, 1)))
        wid = i32(fx.get(coord_wave_lane, 0))
        lid = i32(fx.get(coord_wave_lane, 1))

        bx = fx.block_idx.x
        head_idx = (bx % ATTN_H_KV) * GROUP_SIZE + (bx // ATTN_H_KV)
        batch_idx = fx.block_idx.z
        head_idx_kv = head_idx // GROUP_SIZE
        block_tile_idx = fx.block_idx.y
        tile_idx = sgpr(block_tile_idx * NUM_WARPS + wid)
        # Hoist the wave id into an SGPR before deriving stagger.
        wid_uni = sgpr(wid)
        stagger = wid_uni // 4

        # Per-warp LDS destination bases, hoisted into SGPRs (wave-uniform), matching
        lds_warp_off = i32(wid * BYTES_PER_WARP)

        def _lds_base(smem_ptr):
            return sgpr(i32(fx.ptrtoint(smem_ptr)) + lds_warp_off)

        k_lds_base = [_lds_base(k_smem_0), _lds_base(k_smem_1)]
        v_lds_base = [_lds_base(v_smem_0), _lds_base(v_smem_1)]

        if const_expr(NT_BAND is not None):
            qs = block_tile_idx * i32(Q_BLOCK_SIZE * NUM_WARPS)  # first q row of CTA
            left_edge = qs - i32(swa_left)
            left_edge = (left_edge > i32(0)).select(left_edge, i32(0))
            right_edge = qs + i32((Q_BLOCK_SIZE * NUM_WARPS - 1) + swa_right)
            seq_m1 = i32(seq_len_kv) - i32(1)
            right_edge = (right_edge < seq_m1).select(right_edge, seq_m1)
            # First in-band tile, 64-aligned (>= 0 since qs >= 0).
            base0 = (left_edge // i32(KV_BLOCK_SIZE)) * i32(KV_BLOCK_SIZE)
            # In-band tile count for THIS CTA: ceil((right - base0 + 1)/64), rounded
            span_tiles = (right_edge - base0 + i32(KV_BLOCK_SIZE)) // i32(KV_BLOCK_SIZE)
            nt_ct = (span_tiles + i32(3)) // i32(4) * i32(4)
            nt_ct = (nt_ct > i32(4)).select(nt_ct, i32(4))
            nt_ct = (nt_ct < i32(NT_BAND)).select(nt_ct, i32(NT_BAND))
            nt_ct = sgpr(nt_ct)
            nt_rt = nt_ct

            max_base = i32(seq_len_kv) - nt_ct * i32(KV_BLOCK_SIZE)
            base_unclamped = (base0 < max_base).select(base0, max_base)
            base_unclamped = (base_unclamped > i32(0)).select(base_unclamped, i32(0))
            base_kv_row = sgpr(base_unclamped)
            swa_row_off = base_kv_row * i32(ATTN_H_KV * ATTN_D)
        else:
            nt_rt = i32(seq_len_kv) // i32(KV_BLOCK_SIZE)
            base_kv_row = i32(0)
            swa_row_off = i32(0)

        k_base_elems = batch_idx * (i32(seq_len_kv) * (ATTN_H_KV * ATTN_D)) + head_idx_kv * ATTN_D + swa_row_off
        v_base_elems = batch_idx * (i32(seq_len_kv) * (ATTN_H_KV * ATTN_D)) + head_idx_kv * ATTN_D + swa_row_off

        # ---------------- per-query band mask (aiter window_size) ----------------
        def mask_band(att, band_tile):
            if const_expr(NT_BAND is None):
                return att

            q_row = i32(tile_idx * i32(Q_BLOCK_SIZE)) + i32(lid % 32)

            kcol_base = base_kv_row + i32(band_tile) * i32(KV_BLOCK_SIZE) + i32((lid // 32) * 4)

            width = fx.Uint32(swa_left + swa_right)
            out = [None, None]

            for h in range_constexpr(2):
                src = Vec(att[h])
                elems = []
                rel_base = kcol_base + i32(32 * h) - q_row + i32(swa_left)

                for r in range_constexpr(16):
                    c = 8 * (r // 4) + (r % 4)
                    shifted = (rel_base + i32(c)).bitcast(fx.Uint32)
                    keep = shifted <= width
                    elems.append(keep.select(fx.Float32(src[r]), NEG_INF))

                out[h] = Vec.from_elements(elems, f32)
            return out

        # Divided buffer-tensor views for the G->LDS DMA copy atom. logical_divide
        # by (1,1) yields the ((1),(N)) shape fx.copy expects; slicing the second
        # mode selects the per-lane element voffset.
        k_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(K), fx.make_layout(1, 1))
        v_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(V), fx.make_layout(1, 1))
        k_src_div = [k_div, k_div]
        v_src_div = [v_div, v_div]
        # Q global->register load via a 128b buffer copy atom (v8bf16 per lane).
        q_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(Q), fx.make_layout(1, 1))
        _q_load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), bf16)
        # O register->global store via a 64b buffer copy atom (4 bf16 = b64 store).
        o_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(O), fx.make_layout(1, 1))
        _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), bf16)

        ZERO16 = Vec.filled(16, 0.0, f32)

        q_reg = [[None] * 8 for _ in range_constexpr(1)]
        q_reg_t = [[None] * 1 for _ in range_constexpr(8)]
        k_reg = [[None] * 8 for _ in range_constexpr(2)]
        k_reg_t = [[None] * 2 for _ in range_constexpr(8)]
        v_reg = [[None] * 4 for _ in range_constexpr(4)]
        o_reg = [ZERO16, ZERO16, ZERO16, ZERO16]
        att_block = [[None] * 2 for _ in range_constexpr(2)]
        att_block_bf16 = [None] * 2

        # scalar softmax accumulators (max_vec, norm_vec, scale_vec)
        max_vec = f32(float("-inf"))
        max_vec_prev = max_vec
        norm_vec = f32(0.0)
        scale_vec = f32(1.0)

        swizzled_offsets_K = prefill_offsets(True, K_stride1)
        swizzled_offsets_V = prefill_offsets(False, V_stride1)
        off_K = swizzled_offsets_K
        off_V = swizzled_offsets_V

        # ---------------- Load K[0] into shared ----------------
        load_k(0, 0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # ---------------- Load Q into registers ----------------
        q_row_offset = lid % 32
        q_col_offset = 8 * (lid // 32)
        q_base = (
            batch_idx * (i32(seq_len_q) * (ATTN_H * ATTN_D))
            + tile_idx * (Q_BLOCK_SIZE * ATTN_H * ATTN_D)
            + head_idx * ATTN_D
        )

        def _concat(lhs, rhs):
            lv = Vec(lhs)
            rv = Vec(rhs)
            return lv.shuffle(rv, list(range(lv.numel)) + [lv.numel + i for i in range(rv.numel)])

        q_raw = [None] * 8
        for j in range_constexpr(8):
            col = 16 * j + q_col_offset
            elem_off = q_base + q_row_offset * Q_stride1 + col
            q_raw[j] = fly.copy_atom_call_ssa([v8bf16_t], _q_load_atom, fx.slice(q_div, (None, i32(elem_off))))

        rocdl.sched_barrier(0)
        wait_vmcnt(0)
        rocdl.sched_barrier(0)
        # tree-concat 8 x v8 -> v64
        q16 = [_concat(q_raw[2 * p], q_raw[2 * p + 1]) for p in range_constexpr(4)]
        q32 = [_concat(q16[2 * p], q16[2 * p + 1]) for p in range_constexpr(2)]
        q_all = _concat(q32[0], q32[1])
        q_sc64 = Vec.from_elements([TEMPERATURE_SCALE], f32).broadcast_to(64)
        q_all_scaled = arith.mulf(q_all.to(f32), q_sc64, fastmath=fm)
        q_all_bf = Vec(Vec(q_all_scaled).to(bf16))
        for j in range_constexpr(8):
            qbf = q_all_bf.shuffle(q_all_bf, [j * 8 + e for e in range(8)])
            q_reg[0][j] = qbf
            # transpose_q (q_reg[0][j] -> q_reg_t[j][0]): identity relabel
            q_reg_t[j][0] = qbf

        # ---------------- Load K[1] into shared, V[0] into shared ----------------
        load_k(1, 1)
        load_v(0, 0)

        # ---------------- Load K[0] from shared to registers ----------------
        k_reg = load_k_regs(0)
        rocdl.sched_barrier(0)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # ---------------- QK[0] ----------------
        att_block[0] = [ZERO16, ZERO16]

        for i in range_constexpr(2):
            for j in range_constexpr(8):
                k_reg_t[j][i] = k_reg[i][j]

        att_block[0] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[0])
        att_block[0] = mask_band(att_block[0], 0)

        # ---------------- Partial softmax for QK[0] ----------------
        max_vec = (
            fx.Float32(fmaxf(col_max(att_block[0]), NEG_FLOOR))
            if const_expr(NT_BAND is not None)
            else fx.Float32(col_max(att_block[0]))
        )
        max_vec_prev = max_vec
        att_block[0] = sub_col(att_block[0], max_vec)
        att_block[0][0] = exp2_one(att_block[0][0])

        rocdl.sched_barrier(0)

        if stagger > 0:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        rocdl.sched_barrier(0)

        # ---- Load K[1] from shared, load K[2] into shared, load V[1] into shared ----
        nop_anchor()
        rocdl.sched_barrier(0)
        k_reg = load_k_regs(1)
        load_k(2, 0)
        load_v(1, 1)
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        # pending_scale: i1, deferred norm rescale flag (reference's int pending_scale).
        pending_scale = fx.Boolean(False)

        # Scale the 4 bf16 P packs (att_block_bf16) by a scalar corr
        def scale_packs(packs, corr_scalar):
            corr8 = Vec.from_elements([fx.Float32(corr_scalar)], f32).broadcast_to(8)
            out = []
            for p in range_constexpr(4):
                pf = Vec(packs[p]).to(f32)
                ps = arith.mulf(pf, corr8, fastmath=fm)
                out.append(Vec(ps).to(bf16))
            return out

        # Lazy-threshold rescale with deferred norm (mirrors reference block 2/6):
        def rescale_defer(att_buf, o_reg, max_prev, scale_old, packs):
            m_cur = col_max(att_buf)
            if const_expr(NT_BAND is not None):
                m_cur = fmaxf(m_cur, NEG_FLOOR)  # tile fully out of band -> -inf
            m_new = fmaxf(m_cur, max_prev)
            with arith.fastmath(fm):
                delta = fx.Float32(m_new) - fx.Float32(max_prev)
            not_ok = fx.Float32(delta) > RESCALE_THRESHOLD
            mask = rocdl.ballot(T.i64, not_ok)
            needs_rescale = fx.Int64(mask) != 0
            kept_max = needs_rescale.select(fx.Float32(m_new), fx.Float32(max_prev))

            o0, o1, o2, o3 = o_reg
            p0, p1, p2, p3 = packs
            scale_new = scale_old
            if needs_rescale:
                with arith.fastmath(fm):
                    scale_s = hw_exp2_scalar(fx.Float32(max_prev) - fx.Float32(m_new))
                    corr_v = bcast16(fx.Float32(scale_s))
                    o0 = Vec(o_reg[0]) * Vec(corr_v)
                    o1 = Vec(o_reg[1]) * Vec(corr_v)
                    o2 = Vec(o_reg[2]) * Vec(corr_v)
                    o3 = Vec(o_reg[3]) * Vec(corr_v)

                p0, p1, p2, p3 = scale_packs(packs, fx.Float32(scale_s))
                scale_new = fx.Float32(scale_s)
            o_new = [o0, o1, o2, o3]
            p_new = [p0, p1, p2, p3]
            return o_new, scale_new, needs_rescale, kept_max, p_new

        # ========================================================================
        # Body inlined directly in the rolled runtime loop, reading carried state
        # via _unflatten at the top and yielding new state via _flatten at the
        # bottom. A rolled loop keeps o_reg/k_reg as loop-carried phis (fixed registers across
        # the back-edge) instead of one giant unrolled live range that spills.
        # Buffer parities: odd tiles use buf1 for K-shared/V-store, even use buf0.
        # ========================================================================
        # Flatten/unflatten the carried state to/from a flat list of raw ir.Values,
        # as required by the runtime range(..., init=) loop-carried phi mechanism.
        def _flatten(k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale):
            flat = []
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    flat.append(k_reg[i][jj])
            flat.append(att0[0])
            flat.append(att0[1])
            for n in range_constexpr(4):
                flat.append(o_reg[n])
            flat.append(max_vec_prev)
            flat.append(norm_vec)
            flat.append(scale_vec)
            flat.append(pending_scale)
            return flat

        def _unflatten(flat):
            p = 0
            k_reg = [[None] * 8 for _ in range_constexpr(2)]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg[i][jj] = flat[p]
                    p += 1
            att0 = [flat[p], flat[p + 1]]
            p += 2
            o_reg = [flat[p + n] for n in range_constexpr(4)]
            p += 4
            max_vec_prev = fx.Float32(flat[p])
            p += 1
            norm_vec = fx.Float32(flat[p])
            p += 1
            scale_vec = fx.Float32(flat[p])
            p += 1
            pending_scale = flat[p]
            p += 1
            return k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale

        def _run_body(j, k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale):
            jm1 = j - 1
            jp1 = j + 1
            k_reg_t = [[None] * 2 for _ in range_constexpr(8)]

            # ---- Block 0: QK[odd] + finish softmax for QK[even] ----
            att1 = [ZERO16, ZERO16]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg_t[jj][i] = k_reg[i][jj]

            if pending_scale:
                norm_vec = norm_vec * scale_vec

            att1 = mma_AtB_QK(k_reg_t, q_reg_t, att1)
            att_block_bf16, norm_vec = finish_scalar(att0[0], att0[1], norm_vec)
            att_block_bf16 = anchor_p(att_block_bf16)
            sched_exp_pairs(6, 3, 1)
            sched_pairs(10, 5, 1)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 1: Load K[j] into shared (buf1), load V from shared (buf0) ----
            nop_anchor()
            rocdl.sched_barrier(0)
            load_k(j, 1)
            v_reg = load_v_regs(0)
            att1 = mask_band(att1, jm1 - 1)  # k_reg held band tile j-2 in block 0
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 2: A[even]*V, partial softmax for QK[odd] ----
            rocdl.s_setprio(1)
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[0], att_block_bf16[0])
            o_reg, scale_vec, pending_scale, max_vec, att_block_bf16 = rescale_defer(
                att1, o_reg, max_vec_prev, scale_vec, att_block_bf16
            )
            o_reg = anchor_o(o_reg)
            max_vec_prev = max_vec
            sched_pairs(4, 6, 2)

            o_reg = mma_AtB_OV_slice(o_reg, v_reg[1], att_block_bf16[1])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[2], att_block_bf16[2])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[3], att_block_bf16[3])
            att1 = sub_col(att1, max_vec)
            att1[0] = exp2_one(att1[0])
            sched_pairs(6, 6, 2)
            sched_exp_pairs(6, 3, 2)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 3: Load V[j-1] into shared (buf0), load K from shared (buf0) ----
            nop_anchor()
            rocdl.sched_barrier(0)
            load_v(jm1, 0)
            k_reg = load_k_regs(0)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 4: QK[even] + finish softmax for QK[odd] ----
            att0 = [ZERO16, ZERO16]
            for i in range_constexpr(2):
                for jj in range_constexpr(8):
                    k_reg_t[jj][i] = k_reg[i][jj]

            if pending_scale:
                norm_vec = norm_vec * scale_vec

            att0 = mma_AtB_QK(k_reg_t, q_reg_t, att0)
            att_block_bf16, norm_vec = finish_scalar(att1[0], att1[1], norm_vec)
            att_block_bf16 = anchor_p(att_block_bf16)
            sched_exp_pairs(6, 3, 3)
            sched_pairs(10, 5, 3)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 5: Load K[j+1] into shared (buf0), load V from shared (buf1) ----
            nop_anchor()
            rocdl.sched_barrier(0)
            load_k(jp1, 0)
            v_reg = load_v_regs(1)
            att0 = mask_band(att0, jm1)  # k_reg held band tile j-1 in block 4
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 6: A[odd]*V, partial softmax for QK[even] ----
            rocdl.s_setprio(1)
            mma_AtB_OV_slice(o_reg, v_reg[0], att_block_bf16[0])
            o_reg, scale_vec, pending_scale, max_vec, att_block_bf16 = rescale_defer(
                att0, o_reg, max_vec_prev, scale_vec, att_block_bf16
            )
            o_reg = anchor_o(o_reg)
            max_vec_prev = max_vec
            sched_pairs(4, 6, 4)
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[1], att_block_bf16[1])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[2], att_block_bf16[2])
            o_reg = mma_AtB_OV_slice(o_reg, v_reg[3], att_block_bf16[3])
            att0 = sub_col(att0, max_vec)
            att0[0] = exp2_one(att0[0])
            sched_pairs(6, 5, 4)
            sched_exp_pairs(6, 3, 4)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            # ---- Block 7: Load V[j] into shared (buf1), load K from shared (buf1) ----
            nop_anchor()
            rocdl.sched_barrier(0)
            load_v(j, 1)
            k_reg = load_k_regs(1)
            wait_lgkmcnt0()
            wait_vmcnt(4)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            return (k_reg, att0, o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale)

        init_flat = _flatten(k_reg, att_block[0], o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale)
        # Unroll x2 over a runtime range(..., init=) loop: driver steps j by 4,
        # body runs tiles (j, j+1) then (j+2, j+3). The i32 DSL bounds keep this a
        # runtime scf.for (Python-int bounds would trigger constexpr unrolling), and
        # loop-carried state flows through init=/yield instead of raw iter_args.
        UNROLL = 2
        for jv, iter_args in range(3, nt_rt - 1, 2 * UNROLL, init=init_flat):
            j0 = i32(jv)
            state = _unflatten(list(iter_args))
            for u in range_constexpr(UNROLL):
                j = j0 + i32(2 * u)
                state = _run_body(j, *state)
            loop_results = yield _flatten(*state)
        k_reg, att_block[0], o_reg, max_vec_prev, norm_vec, scale_vec, pending_scale = _unflatten(loop_results)

        # ====================================================================
        # (no lazy-threshold vote): every tile rescales o_reg/norm by exp2(prev-new).
        # ====================================================================
        # full OV: o_reg[n] += sum_kk v_reg[kk][n] * att_bf[kk] (4 contraction slices)
        def full_ov(o_reg, vreg, packs):
            for kk in range_constexpr(4):
                o_reg = ov_slice(o_reg, [vreg[kk][n] for n in range_constexpr(4)], packs[kk])
            return o_reg

        def rescale_uncond(att_buf, max_prev):
            m_cur = col_max(att_buf)
            if const_expr(NT_BAND is not None):
                m_cur = fmaxf(m_cur, NEG_FLOOR)
            m_new = fmaxf(m_cur, max_prev)
            scale_s = hw_exp2_scalar(arith.subf(_raw(max_prev), _raw(m_new), fastmath=fm))
            return fx.Float32(scale_s), fx.Float32(m_new)

        nt = nt_rt

        # ---- Block 0: QK[last odd] + finish softmax for last even ----
        att_block[1] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[1] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[1])

        if pending_scale:
            norm_vec = fmul(norm_vec, scale_vec)
        att_block_bf16, norm_vec = finish_scalar(att_block[0][0], att_block[0][1], norm_vec)
        sched_exp_pairs(6, 3, 5)
        sched_pairs(10, 5, 5)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 1: Load K[nt-1] into shared (buf1), load V from shared (buf0) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        load_k(nt - 1, 1)
        v_reg = load_v_regs(0)
        att_block[1] = mask_band(att_block[1], nt - i32(3))
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 2: A*V, partial softmax for last odd ----
        rocdl.s_setprio(1)
        o_reg = full_ov(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[1], max_vec_prev)
        max_vec_prev = max_vec
        att_block[1] = sub_col(att_block[1], max_vec)
        att_block[1][0] = exp2_one(att_block[1][0])
        sched_pairs(10, 5, 6)
        sched_exp_pairs(6, 3, 6)
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 3: Load V[nt-2] into shared (buf0), load K from shared (buf0) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        load_v(nt - 2, 0)
        k_reg = load_k_regs(0)
        wait_lgkmcnt0()
        wait_vmcnt(4)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 4: QK + finish softmax for the odd from block 2 ----
        att_block[0] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[0] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[0])
        # mask_band deferred to loading block 5 (consumed in block 6).
        norm_vec = fmul(norm_vec, scale_vec)
        att_block_bf16, norm_vec = finish_scalar(att_block[1][0], att_block[1][1], norm_vec)
        sched_exp_pairs(6, 3, 7)
        sched_pairs(10, 5, 7)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 5: Load V from shared (buf1) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        v_reg = load_v_regs(1)
        att_block[0] = mask_band(att_block[0], nt - i32(2))
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 6: A*V, partial softmax for the even from block 4 ----
        rocdl.s_setprio(1)
        o_reg = full_ov(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[0], max_vec_prev)
        max_vec_prev = max_vec
        att_block[0] = sub_col(att_block[0], max_vec)
        att_block[0][0] = exp2_one(att_block[0][0])
        sched_pairs(10, 5, 8)
        sched_exp_pairs(6, 3, 8)
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 7: Load V[nt-1] into shared (buf1), load K from shared (buf1) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        load_v(nt - 1, 1)
        k_reg = load_k_regs(1)
        wait_lgkmcnt0()
        wait_vmcnt(2)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 8: QK + finish softmax for the even from block 6 ----
        att_block[1] = [ZERO16, ZERO16]
        for i in range_constexpr(2):
            for jj in range_constexpr(8):
                k_reg_t[jj][i] = k_reg[i][jj]
        att_block[1] = mma_AtB_QK(k_reg_t, q_reg_t, att_block[1])
        # mask_band deferred to loading block 9 (consumed in block 10).
        norm_vec = fmul(norm_vec, scale_vec)
        att_block_bf16, norm_vec = finish_scalar(att_block[0][0], att_block[0][1], norm_vec)
        sched_exp_pairs(6, 3, 9)
        sched_pairs(10, 5, 9)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 9: Load V from shared (buf0) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        v_reg = load_v_regs(0)
        att_block[1] = mask_band(att_block[1], nt - i32(1))
        wait_lgkmcnt0()
        wait_vmcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 10: A*V, full softmax for the last QK (att_block[1]) ----
        o_reg = mma_AtB_OV(o_reg, v_reg, att_block_bf16)
        scale_vec, max_vec = rescale_uncond(att_block[1], max_vec_prev)
        max_vec_prev = max_vec
        att_block[1] = sub_col(att_block[1], max_vec)
        att_block[1][0] = exp2_one(att_block[1][0])
        sched_pairs(10, 5, 10)
        sched_exp_pairs(6, 3, 10)
        rocdl.sched_barrier(0)
        norm_vec = fmul(norm_vec, scale_vec)
        att_block_bf16, norm_vec = finish_scalar(att_block[1][0], att_block[1][1], norm_vec)
        rocdl.sched_barrier(0)
        o_reg = mul_o(o_reg, scale_vec)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 11: Load V from shared (buf1) ----
        nop_anchor()
        rocdl.sched_barrier(0)
        v_reg = load_v_regs(1)
        wait_lgkmcnt0()
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Block 12: Final A*V and normalize ----
        o_reg = mma_AtB_OV(o_reg, v_reg, att_block_bf16)

        inv = rocdl.rcp(T.f32, norm_vec)
        # Guard against a fully-masked row (norm == 0 -> rcp == inf).
        if const_expr(NT_BAND is not None):
            inv = arith.select(fx.Float32(norm_vec) > 0.0, inv, 0.0)
        o_reg = mul_o(o_reg, inv)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        # ---- Conclusion: store O and LSE ----
        if stagger == 0:
            rocdl.s_barrier()

        store_o(o_reg)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_stride1: fx.Int32,
        K_stride1: fx.Int32,
        V_stride1: fx.Int32,
        O_stride1: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = ATTN_H
        # grid_y from the runtime seq_len_q arg (ceil-div over the Q tile x wave fan-out).
        grid_y = (fx.Int32(seq_len_q) // fx.Int32(Q_BLOCK_SIZE) + fx.Int32(NUM_WARPS - 1)) // fx.Int32(NUM_WARPS)
        grid_z = ATTN_B
        attend_ker(
            Q,
            K,
            V,
            O,
            Q_stride1,
            K_stride1,
            V_stride1,
            O_stride1,
            seq_len_q,
            seq_len_kv,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{NUM_THREADS},{NUM_THREADS}",
            },
        ).launch(grid=(grid_x, grid_y, grid_z), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch
