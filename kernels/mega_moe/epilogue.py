# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Stage-1 fused-MoE group-GEMM1 epilogue: activation + gate/up combine + output-quant + scatter (Activation/OutputQuant/Scatter atoms + Gemm1Epilogue.run dispatch).

from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.common.kernels_common import _guard
from kernels.common.mma.mfma_epilogues import c_shuffle_epilog
from kernels.common.utils import exp2_amdgcn_scalar, fabs_f32, idx_to_ptr, lds_load, rcp_amdgcn_scalar, store_nt


def reduce_slice_k_partials(acc_list, *, lds_scratch, wid_k, local_tid, slice_k, vec4_f32):
    """Intra-CTA slice-K reduction: sum per-K-slice partial MFMA accumulators across the
    ``slice_k`` wave groups via LDS and broadcast the sum back to every group's registers.
    Returns the reduced ``acc_list`` (fresh vec4_f32 loaded from LDS).
    """
    n = len(acc_list)
    c_slots = arith.constant(n * 4, index=True)
    base = local_tid * c_slots
    positions = [base + arith.constant(a * 4, index=True) for a in range_constexpr(n)]
    for g in range_constexpr(slice_k):
        gpu.barrier()
        with _guard(arith.cmpi(CmpIPredicate.eq, wid_k, arith.constant(g, index=True))):
            for a in range_constexpr(n):
                if const_expr(g == 0):
                    vector.store(acc_list[a], lds_scratch, [positions[a]], alignment=16)
                else:
                    prev = vector.load_op(vec4_f32, lds_scratch, [positions[a]])
                    vector.store(arith.addf(prev, acc_list[a]), lds_scratch, [positions[a]], alignment=16)
    gpu.barrier()
    return [vector.load_op(vec4_f32, lds_scratch, [positions[a]]) for a in range_constexpr(n)]


def stage_srcmap_to_lds(
    *,
    tc,
    addr_disp,
    tx,
    lds_tid,
    sorted_rsrc,
    expert_rsrc,
    sorted_w_rsrc,
    wts_sorted_rsrc,
    atom_sw_out_rsrc,
    tile_m,
    sort_block_m,
    ca,
    static_tiles,
    fz_epr,
    fz_npes,
    fz_mtpr,
    fz_k,
    fz_tile_m,
    fz_rank,
):
    """atom_contract (always on): emit the dispatch srcmap into ``lds_tid`` (so the epilogue can
    write a2 at the ATOM logical row t*topk+s) plus optional compact / static-tile metadata
    (sorted_token_ids / combine weights / sorted_expert_ids) that stage-2 gemm2+combine consumes.
    """
    bx_m = tc.bx_m
    bx = tc.bx
    expert_i32 = tc.expert_global
    ef = tc.ef
    kf = tc.kf
    cnt_ef = tc.cnt_ef

    c_tile_m_idx = arith.constant(tile_m, index=True)
    tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, c_tile_m_idx)
    with _guard(tid_in_range):
        tid_row = bx_m + tx
        # atom_contract: lds_tid = dispatch srcmap[slot] = (k_slot<<24)|src_global -> epilogue writes a2@logical.
        atom_srcmap_addr = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(addr_disp),
            arith.constant(19),
            vec_width=1,
            dtype=T.i64,
        )
        tid_val = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(atom_srcmap_addr),
            tid_row,
            vec_width=1,
            dtype=T.i32,
        )
        tid_vec1 = vector.from_elements(T.vec(1, T.i32), [tid_val])
        vector.store(tid_vec1, lds_tid, [tx])
        if const_expr(ca):
            # compact+atom: srcmap[real]=src_enc, srcmap[pad]=sentinel; emit _sti (disp 40) + _sw_atom (disp 20).
            ca_sti_addr = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(addr_disp),
                arith.constant(40),
                vec_width=1,
                dtype=T.i64,
            )
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            ca_row = bx_m_i32 + arith.index_cast(T.i32, tx)
            buffer_ops.buffer_store(
                tid_val,
                buffer_ops.create_buffer_resource_from_addr(ca_sti_addr),
                ca_row,
                offset_is_bytes=False,
            )
            ca_t = tid_val & arith.constant(0xFFFFFF, type=T.i32)
            ca_real = arith.cmpi(CmpIPredicate.ult, ca_t, arith.constant(fz_npes * fz_mtpr, type=T.i32))
            ca_s = tid_val >> arith.constant(24, type=T.i32)
            ca_logw = ca_t * arith.constant(fz_k, type=T.i32) + ca_s
            ca_wt = buffer_ops.buffer_load(sorted_w_rsrc, tid_row, vec_width=1, dtype=T.f32)
            # sorted-row weight (stage2 doweight): real rows get the recv wt, padding 0.
            ca_wt_sorted = arith.select(ca_real, ca_wt, arith.constant(0.0, type=T.f32))
            buffer_ops.buffer_store(ca_wt_sorted, wts_sorted_rsrc, ca_row, offset_is_bytes=False)
            with _guard(ca_real):
                buffer_ops.buffer_store(ca_wt, atom_sw_out_rsrc, ca_logw, offset_is_bytes=False)
        if const_expr(static_tiles and fz_epr <= 64):
            # Unconditional per-row emit of compact sorted_token_ids + combine weight (outside store_pair to avoid stale padding rows); padding rows get sentinel t = npes*mtpr.
            tx_i32_e = arith.index_cast(T.i32, tx)
            pos_e = kf * arith.constant(fz_tile_m, type=T.i32) + tx_i32_e
            real_e = arith.cmpi(CmpIPredicate.ult, pos_e, cnt_ef)
            bx_i32_e = arith.index_cast(T.i32, bx)
            crow_e = bx_i32_e * arith.constant(sort_block_m, type=T.i32) + tx_i32_e
            sent_e = arith.constant(fz_npes * fz_mtpr, type=T.i32)
            sti_out_e = arith.select(real_e, tid_val, sent_e)
            buffer_ops.buffer_store(sti_out_e, sorted_rsrc, crow_e, offset_is_bytes=False)
            # recv routing weight feeds two sinks: combine wts_buf at logical t*topk+s (real only), stage2 doweight at sorted row _crow_e (padding -> 0).
            t_we = tid_val & arith.constant(0xFFFFFF, type=T.i32)
            s_we = tid_val >> arith.constant(24, type=T.i32)
            log_we = t_we * arith.constant(fz_k, type=T.i32) + s_we
            wt_e = buffer_ops.buffer_load(sorted_w_rsrc, tid_row, vec_width=1, dtype=T.f32)
            wt_e_sorted = arith.select(real_e, wt_e, arith.constant(0.0, type=T.f32))
            buffer_ops.buffer_store(wt_e_sorted, wts_sorted_rsrc, crow_e, offset_is_bytes=False)
            with _guard(real_e):
                buffer_ops.buffer_store(wt_e, atom_sw_out_rsrc, log_we, offset_is_bytes=False)

    if const_expr(static_tiles and fz_epr <= 64):
        # static-tiles: emit LOCAL expert id (_ef) one entry per compact tile at index `bx`, matching gemm2's expert_ids[bx_m // sort_block_m] read.
        tx0 = arith.cmpi(CmpIPredicate.eq, tx, arith.constant(0, index=True))
        with _guard(tx0):
            bx_i32_se = arith.index_cast(T.i32, bx)
            buffer_ops.buffer_store(ef, expert_rsrc, bx_i32_se, offset_is_bytes=False)

    if const_expr(ca):
        # compact+atom: emit _se_atom (disp 41) = LOCAL expert id, one entry per compact tile at index `bx`, matching gemm2's expert_ids[bx_m // sort_block_m] read.
        tx0_ca = arith.cmpi(CmpIPredicate.eq, tx, arith.constant(0, index=True))
        with _guard(tx0_ca):
            ca_se_addr = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(addr_disp),
                arith.constant(41),
                vec_width=1,
                dtype=T.i64,
            )
            ca_se_rsrc = buffer_ops.create_buffer_resource_from_addr(ca_se_addr)
            ca_le2 = expert_i32 - arith.constant(fz_rank * fz_epr, type=T.i32)
            bx_i32_ca = arith.index_cast(T.i32, bx)
            buffer_ops.buffer_store(ca_le2, ca_se_rsrc, bx_i32_ca, offset_is_bytes=False)


class Activation:
    """silu + bias-add + gate/up combine -> acc[]. INTERLEAVE folds gate/up from a single
    even/odd accumulator list; SEPARATED combines two lists.
    """

    def __init__(
        self,
        *,
        gate_up_interleave,
        num_acc_n,
        m_repeat,
        pack_N,
        vec4_f32,
        f32,
    ):
        self.gate_up_interleave = gate_up_interleave
        self.num_acc_n = num_acc_n
        self.m_repeat = m_repeat
        self.pack_N = pack_N
        self.vec4_f32 = vec4_f32
        self.f32 = f32

    def _silu_elem(self, g):
        """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
        f32 = self.f32
        neg_log2e = arith.constant(-1.4426950408889634, type=f32)
        t = g * neg_log2e
        emu = exp2_amdgcn_scalar(t)
        one = arith.constant(1.0, type=f32)
        den = one + emu
        sig = rcp_amdgcn_scalar(den)
        return g * sig

    def _silu_mul_vec4(self, gate_v4, up_v4):
        """Element-wise silu(gate) * up on vec4_f32."""
        result_elems = []
        for ei in range_constexpr(4):
            g = vector.extract(gate_v4, static_position=[ei], dynamic_position=[])
            u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
            result_elems.append(self._silu_elem(g) * u)
        return vector.from_elements(self.vec4_f32, result_elems)

    def merge(self, acc_gate, acc_up):
        """activation + gate/up combine -> acc[]. Returns (acc, gui_out_n)."""
        num_acc_n = self.num_acc_n
        m_repeat = self.m_repeat

        acc = None
        gui_out_n = None
        if const_expr(self.gate_up_interleave):
            gui_out_n = num_acc_n // self.pack_N
            acc = [None] * (gui_out_n * m_repeat)
            for mi in range_constexpr(m_repeat):
                for ni in range_constexpr(gui_out_n):
                    g_idx = mi * num_acc_n + ni * self.pack_N
                    u_idx = g_idx + 1
                    out_idx = mi * gui_out_n + ni
                    acc[out_idx] = self._silu_mul_vec4(acc_gate[g_idx], acc_gate[u_idx])
        else:
            acc = [None] * (int(num_acc_n) * int(m_repeat))
            for mi in range_constexpr(m_repeat):
                for ni in range_constexpr(num_acc_n):
                    aidx = mi * num_acc_n + ni
                    acc[aidx] = self._silu_mul_vec4(acc_gate[aidx], acc_up[aidx])
        return acc, gui_out_n


class OutputQuant:
    """store_pair's quant core: abs-max reduce -> e8m0 -> _f32_to_e2m1 / cvt_pk_fp8 packing
    + a2-scale (e8m0) swizzle emit. Constant locals materialised in the ctor to preserve trace op order.
    """

    def __init__(
        self,
        *,
        need_fp4,
        need_fp8,
        need_sort,
        e_vec,
        num_shuffle_steps,
        shuffle_dists,
        sorted_scale_cols_i32,
        sorted_scale_rsrc,
        sort_block_m,
    ):
        self.need_fp4 = need_fp4
        self.need_fp8 = need_fp8
        self.need_sort = need_sort
        self.e_vec = e_vec
        self.num_shuffle_steps = num_shuffle_steps
        self.shuffle_dists = shuffle_dists
        self.sorted_scale_rsrc = sorted_scale_rsrc
        self.sort_block_m = sort_block_m

        # ~40 arith.constant locals (materialised here to preserve op order).
        self._c0_i32 = arith.constant(0, type=T.i32)
        self._c1_i32 = arith.constant(1, type=T.i32)
        self._c2_i32 = arith.constant(2, type=T.i32)
        self._c3_i32 = arith.constant(3, type=T.i32)
        self._c4_i32 = arith.constant(4, type=T.i32)
        self._c5_i32 = arith.constant(5, type=T.i32)
        self._c7_i32 = arith.constant(7, type=T.i32)
        self._c15_i32 = arith.constant(15, type=T.i32)
        self._c21_i32 = arith.constant(21, type=T.i32)
        self._c23_i32 = arith.constant(23, type=T.i32)
        self._c28_i32 = arith.constant(28, type=T.i32)
        self._c31_i32 = arith.constant(31, type=T.i32)
        self._c32_i32 = arith.constant(32, type=T.i32)
        self._c64_i32 = arith.constant(64, type=T.i32)
        self._c126_i32 = arith.constant(126, type=T.i32)
        self._c127_i32 = arith.constant(127, type=T.i32)
        self._c254_i32 = arith.constant(254, type=T.i32)
        self._c256_i32 = arith.constant(256, type=T.i32)
        self._c0xFF_i32 = arith.constant(0xFF, type=T.i32)
        self._c0x200000_i32 = arith.constant(0x200000, type=T.i32)
        self._c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)
        self._c0x400000_i32 = arith.constant(0x400000, type=T.i32)
        self._c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)
        self._c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
        self._c0_f32 = arith.constant(0.0, type=T.f32)

        self._c8_i32 = arith.constant(8, type=T.i32)
        fp_headroom = 2 if need_fp4 else (8 if need_fp8 else 0)
        self._c_headroom_i32 = arith.constant(fp_headroom, type=T.i32)

        if const_expr(need_sort):
            self._n32_sort = sorted_scale_cols_i32 * self._c32_i32

    def _f32_to_e2m1(self, qx_f32):
        """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
        qx = qx_f32.bitcast(T.i32)
        s = qx & self._c0x80000000_i32
        e = (qx >> self._c23_i32) & self._c0xFF_i32
        m = qx & self._c0x7FFFFF_i32
        adj_exp = arith.maxsi(self._c126_i32 - e, self._c0_i32)
        m_denorm = (self._c0x400000_i32 | (m >> self._c1_i32)) >> adj_exp
        is_denorm = arith.cmpi(CmpIPredicate.ult, e, self._c127_i32)
        m = arith.select(is_denorm, m_denorm, m)
        e = arith.maxsi(e - self._c126_i32, self._c0_i32)
        combined = (e << self._c2_i32) | (m >> self._c21_i32)
        rounded = (combined + self._c1_i32) >> self._c1_i32
        e2m1 = arith.minui(rounded, self._c7_i32)
        return (s >> self._c28_i32) | e2m1

    def quantize(self, *, frag, idx_to_llvm_ptr, row_byte_base, col_g0, row_local, bx):
        """Quant + pack + store the fp4/fp8 fragment, and (if _need_sort) emit the swizzled
        e8m0 a2-scale byte for gemm2.
        """
        e_vec = self.e_vec
        frag_vals = []
        for i in range_constexpr(e_vec):
            frag_vals.append(vector.extract(frag, static_position=[i], dynamic_position=[]))

        local_max = self._c0_f32
        for i in range_constexpr(e_vec):
            abs_v = fabs_f32(frag_vals[i])
            local_max = arith.maximumf(local_max, abs_v)

        for si in range_constexpr(self.num_shuffle_steps):
            off = arith.constant(self.shuffle_dists[si], type=T.i32)
            peer = local_max.shuffle_xor(off, self._c64_i32)
            local_max = arith.maximumf(local_max, peer)

        max_i32 = local_max.bitcast(T.i32)
        max_rounded = (max_i32 + self._c0x200000_i32) & self._c0xFF800000_i32
        exp_field = max_rounded >> self._c23_i32
        e8m0_biased = arith.maxsi(exp_field - self._c_headroom_i32, self._c0_i32)

        quant_exp = self._c254_i32 - e8m0_biased
        quant_scale = (quant_exp << self._c23_i32).bitcast(T.f32)

        if const_expr(self.need_fp4):
            fp4_vals = []
            for i in range_constexpr(e_vec):
                scaled_v = frag_vals[i] * quant_scale
                fp4_vals.append(self._f32_to_e2m1(scaled_v))

            packed_i32 = fp4_vals[0] | (fp4_vals[1] << self._c4_i32)
            for k in range_constexpr(1, e_vec // 2):
                byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << self._c4_i32)
                packed_i32 = packed_i32 | (byte_k << arith.constant(k * 8, type=T.i32))

            ptr_addr_idx = row_byte_base + col_g0 / arith.constant(2, index=True)
            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
            pack_bytes = e_vec // 2
            if const_expr(pack_bytes == 1):
                store_nt(arith.TruncIOp(T.i8, packed_i32), out_ptr_v, alignment=1)
            elif const_expr(pack_bytes == 2):
                store_nt(arith.TruncIOp(T.i16, packed_i32), out_ptr_v, alignment=2)
            else:
                store_nt(packed_i32, out_ptr_v, alignment=4)

        elif const_expr(self.need_fp8):
            scaled_vals = []
            for i in range_constexpr(e_vec):
                scaled_vals.append(frag_vals[i] * quant_scale)

            ptr_addr_idx = row_byte_base + col_g0
            if const_expr(e_vec <= 4):
                packed_i32 = self._c0_i32
                for w in range_constexpr(e_vec // 2):
                    packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled_vals[2 * w],
                        scaled_vals[2 * w + 1],
                        packed_i32,
                        w,
                    )
                out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                if const_expr(e_vec == 2):
                    store_nt(arith.TruncIOp(T.i16, packed_i32), out_ptr_v, alignment=2)
                else:
                    store_nt(packed_i32, out_ptr_v, alignment=4)
            else:
                for wg in range_constexpr(e_vec // 4):
                    b = wg * 4
                    packed_w = self._c0_i32
                    packed_w = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled_vals[b],
                        scaled_vals[b + 1],
                        packed_w,
                        0,
                    )
                    packed_w = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled_vals[b + 2],
                        scaled_vals[b + 3],
                        packed_w,
                        1,
                    )
                    word_ptr = ptr_addr_idx + arith.constant(wg * 4, index=True)
                    out_ptr_v = idx_to_llvm_ptr(word_ptr)
                    store_nt(packed_w, out_ptr_v, alignment=4)

        if const_expr(self.need_sort):
            col_g0_i32 = arith.index_cast(T.i32, col_g0)
            is_scale_writer = arith.cmpi(CmpIPredicate.eq, col_g0_i32 & self._c31_i32, self._c0_i32)
            with _guard(is_scale_writer):
                # atom_contract: a2-scale lives at the COMPACT sorted row (bx = compact tile index, NOT sparse slot bx_m), matching stage2 gemm2's scale[compact_row] read.
                bx_i32_sc = arith.index_cast(T.i32, bx)
                rl_i32_sc = arith.index_cast(T.i32, row_local)
                row_i32_s = bx_i32_sc * arith.constant(self.sort_block_m, type=T.i32) + rl_i32_sc
                col_s_i32 = col_g0_i32 >> self._c5_i32
                d0 = row_i32_s >> self._c5_i32
                d1 = (row_i32_s >> self._c4_i32) & self._c1_i32
                d2 = row_i32_s & self._c15_i32
                d3 = col_s_i32 >> self._c3_i32
                d4 = (col_s_i32 >> self._c2_i32) & self._c1_i32
                d5 = col_s_i32 & self._c3_i32
                byte_off = (
                    d0 * self._n32_sort
                    + d3 * self._c256_i32
                    + d5 * self._c64_i32
                    + d2 * self._c4_i32
                    + d4 * self._c2_i32
                    + d1
                )
                e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                buffer_ops.buffer_store(
                    e8m0_i8,
                    self.sorted_scale_rsrc,
                    byte_off,
                    offset_is_bytes=True,
                )


class Scatter:
    """write_row_to_lds + precompute_row (address regime) + raw store path; delegates the
    fp4/fp8 quant core to an ``OutputQuant`` instance.
    """

    def __init__(
        self,
        *,
        contiguous_io,
        out_base_idx,
        out_row_stride,
        lds_tid,
        mask24_i32,
        num_valid_i32,
        topk_i32_v,
        tokens_i32_v,
        topk,
        fz_npes,
        fz_mtpr,
        fz_k,
        static_tiles,
        fz_epr,
        fz_tile_m,
        cnt_ef,
        kf,
        bx,
        quant,
    ):
        self.contiguous_io = contiguous_io
        self.out_base_idx = out_base_idx
        self.out_row_stride = out_row_stride
        self.lds_tid = lds_tid
        self.mask24_i32 = mask24_i32
        self.num_valid_i32 = num_valid_i32
        self.topk_i32_v = topk_i32_v
        self.tokens_i32_v = tokens_i32_v
        self.topk = topk
        self.fz_npes = fz_npes
        self.fz_mtpr = fz_mtpr
        self.fz_k = fz_k
        self.static_tiles = static_tiles
        self.fz_epr = fz_epr
        self.fz_tile_m = fz_tile_m
        self.cnt_ef = cnt_ef
        self.kf = kf
        self.bx = bx
        self.quant = quant

    def write_row_to_lds(
        self,
        *,
        acc,
        mi: int,
        ii: int,
        row_in_tile,
        row,
        row_base_lds,
        col_base_local,
        num_acc_n: int,
        lds_out,
    ):
        for ni in range_constexpr(num_acc_n):
            col_local = col_base_local + (ni * 16)
            acc_idx = mi * num_acc_n + ni
            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
            lds_idx = row_base_lds + col_local
            vec1_f32 = T.vec(1, T.f32)
            v1 = vector.from_elements(vec1_f32, [v])
            vector.store(v1, lds_out, [lds_idx], alignment=4)

    def precompute_row(self, *, row_local, row):
        row_i32 = arith.index_cast(T.i32, row)
        # atom_contract: `row` is the SPARSE fixed-slot address (not compact [0,num_valid)), so do NOT gate by row<num_valid; validity comes from t_ok/s_ok below (padding is sentinel-masked).
        row_valid = arith.cmpi(CmpIPredicate.uge, row_i32, arith.constant(0, type=T.i32))
        fused2 = lds_load(self.lds_tid, row_local)
        t = fused2 & self.mask24_i32
        s = fused2 >> 24
        # atom_contract: t = src_global in [0, npes*mtpr) (padding sentinel-masked); s bound and logical-row stride use fz_k (fuse_topk), NOT the kernel's topk=1.
        t_bound = arith.constant(self.fz_npes * self.fz_mtpr)
        s_bound = arith.constant(self.fz_k)
        ts_stride = self.fz_k
        t_ok = arith.cmpi(CmpIPredicate.ult, t, t_bound)
        s_ok = arith.cmpi(CmpIPredicate.ult, s, s_bound)
        row_valid = arith.andi(row_valid, arith.andi(t_ok, s_ok))
        if const_expr(self.static_tiles and self.fz_epr <= 64):
            # in-kernel padding mask: row is REAL iff position-in-expert = _kf*tile_m + row_local < ll_count[_ef].
            rl_i32_m = arith.index_cast(T.i32, row_local)
            pos_m = self.kf * arith.constant(self.fz_tile_m, type=T.i32) + rl_i32_m
            real_m = arith.cmpi(CmpIPredicate.ult, pos_m, self.cnt_ef)
            row_valid = arith.andi(row_valid, real_m)
        t_idx = arith.index_cast(T.index, t)
        s_idx = arith.index_cast(T.index, s)
        ts_idx = t_idx * arith.constant(ts_stride, index=True) + s_idx
        row_byte_base = self.out_base_idx + ts_idx * arith.constant(self.out_row_stride, index=True)
        return ((fused2, row_byte_base), row_valid)

    def store_pair(self, *, row_local, row, row_ctx, col_pair0, col_g0, frag):
        fused, row_byte_base = row_ctx
        self.quant.quantize(
            frag=frag,
            idx_to_llvm_ptr=idx_to_ptr,
            row_byte_base=row_byte_base,
            col_g0=col_g0,
            row_local=row_local,
            bx=self.bx,
        )


class Gemm1Epilogue:
    """Activation + gate/up combine + output-quant + scatter, wired through the ``c_shuffle_epilog`` callback seam via ``run``'s 2-way dispatch."""

    def __init__(self, *, activation, quant, scatter):
        self.activation = activation
        self.quant = quant
        self.scatter = scatter

    def run(
        self,
        *,
        acc,
        gui_out_n,
        gate_up_interleave,
        tile_m,
        tile_n,
        e_vec,
        cshuffle_nlane,
        total_threads,
        m_repeat,
        num_acc_n,
        tx,
        lane_div_16,
        lane_mod_16,
        bx_m,
        by_n,
        n_tile_base,
        lds_out,
        lds_out_B,
        frag_elem,
    ):
        # `acc` / `gui_out_n` come from `activation.merge` (called before OutputQuant consts, to preserve op order).
        scatter = self.scatter

        def write_row_to_lds(**kw):
            return scatter.write_row_to_lds(acc=acc, **kw)

        def store_pair(**kw):
            return scatter.store_pair(**kw)

        if const_expr(gate_up_interleave):
            # gui: acc has activation applied, halved N
            gui_eff_n = gui_out_n
            gui_tile_n = tile_n // 2
            gui_cshuffle_nlane = min(32, gui_tile_n // e_vec)
            gui_by_n = by_n / arith.constant(2, index=True)
            gui_n_tile_base = n_tile_base / arith.constant(2, index=True)
            c_shuffle_epilog(
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=gui_tile_n,
                e_vec=e_vec,
                cshuffle_nlane=gui_cshuffle_nlane,
                block_size=total_threads,
                m_repeat=m_repeat,
                num_acc_n=gui_eff_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=gui_by_n,
                n_tile_base=gui_n_tile_base,
                lds_out=lds_out,
                frag_elem_type=frag_elem,
                write_row_to_lds=write_row_to_lds,
                precompute_row=scatter.precompute_row,
                store_pair=store_pair,
            )
        else:
            c_shuffle_epilog(
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=e_vec,
                cshuffle_nlane=cshuffle_nlane,
                block_size=total_threads,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=frag_elem,
                write_row_to_lds=write_row_to_lds,
                precompute_row=scatter.precompute_row,
                store_pair=store_pair,
                lds_out_split=lds_out_B,
            )
