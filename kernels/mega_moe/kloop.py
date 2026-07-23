# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""K-loop compute atoms for the fused stage-1 GEMM (``compile_fused_moe_gemm1``)."""

from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.common.layout_utils import crd2idx
from kernels.common.mma.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    swizzle_xor16,
)
from kernels.common.utils import int_to_ptr, lds_base_index


class G2SLoaderX(object):
    """Per-K-step A gather (global VMEM -> LDS). use_async_copy selects the async
    buffer_load_lds DMA (prefetch_to_lds) vs load_tile + store_to_lds."""

    def __init__(
        self,
        *,
        x_rsrc,
        x_elem,
        vec16_elems,
        vec16_x,
        a_elem_bytes,
        elem_bytes,
        c_a_pack,
        num_x_loads,
        x_load_bytes,
        k_blocks16,
        layout_lds,
        total_threads,
        wave_id,
        use_async_copy,
        eff_lds_stride=None,
        tile_m=None,
        lds_group_base=None,
    ):
        self._x_rsrc = x_rsrc
        self._x_elem = x_elem
        self._vec16_elems = vec16_elems
        self._vec16_x = vec16_x
        self._a_elem_bytes = a_elem_bytes
        self._elem_bytes = elem_bytes
        self._c_a_pack = c_a_pack
        self._num_x_loads = num_x_loads
        self._x_load_bytes = x_load_bytes
        self._k_blocks16 = k_blocks16
        self._layout_lds = layout_lds
        # slice_k>1: total_threads/wave_id are K-slice group counts; lds_group_base offsets this group's X buffer.
        self._total_threads = total_threads
        self._wave_id = wave_id
        self._has_group_base = lds_group_base is not None
        self._lds_group_base = lds_group_base if lds_group_base is not None else arith.index(0)
        # per-tile gather lists (set by for_tile)
        self._x_row_base_div4 = None
        self._x_col_local_i32 = None
        self._x_row_local = None
        # async-copy config
        if const_expr(use_async_copy):
            self._dma_bytes = 16
            self._wave_size = 64
            eff_bytes_per_buffer = int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
            self._num_dma_loads = max(1, eff_bytes_per_buffer // (total_threads * self._dma_bytes))

    def for_tile(self, x_row_base_div4, x_col_local_i32, x_row_local):
        self._x_row_base_div4 = x_row_base_div4
        self._x_col_local_i32 = x_col_local_i32
        self._x_row_local = x_row_local

    def _load_x(self, idx_i32):
        a_elem_bytes = self._a_elem_bytes
        idx_elem = idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
        return buffer_copy_gmem16_dwordx4(
            buffer_ops,
            vector,
            elem_type=self._x_elem,
            idx_i32=idx_elem,
            rsrc=self._x_rsrc,
            vec_elems=self._vec16_elems,
        )

    def load_tile(self, base_k):
        a_elem_bytes = self._a_elem_bytes
        c_a_pack = self._c_a_pack
        x_row_base_div4 = self._x_row_base_div4
        x_col_local_i32 = self._x_col_local_i32
        base_k_div4 = ((base_k / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)) / arith.index(4)
        parts = []
        for i in range_constexpr(self._num_x_loads):
            idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
            x_vec = self._load_x(idx_i32)
            parts.append(vector.bitcast(T.vec(4, T.i32), x_vec))
        return parts

    def store_to_lds(self, vec_x_in_parts, lds_buffer):
        elem_bytes = self._elem_bytes
        x_load_bytes = self._x_load_bytes
        x_row_local = self._x_row_local
        x_col_local_i32 = self._x_col_local_i32
        for i in range_constexpr(self._num_x_loads):
            row_local = x_row_local[i]
            col_local_i32 = x_col_local_i32[i]
            if const_expr(x_load_bytes == 16):
                lds_store_16b_xor16(
                    arith,
                    vector,
                    lds_memref=lds_buffer,
                    vec16_ty=self._vec16_x,
                    layout_lds=self._layout_lds,
                    row_local=row_local,
                    col_local_i32=col_local_i32,
                    tx_c4=arith.index(4),
                    k_blocks16=self._k_blocks16,
                    lds_base=self._lds_group_base,
                    vec_part_i32x4=vec_x_in_parts[i],
                    elem_bytes=elem_bytes,
                )

    def _dma_x_tile_to_lds(self, base_k, lds_buffer):
        elem_bytes = self._elem_bytes
        c_a_pack = self._c_a_pack
        k_blocks16 = self._k_blocks16
        total_threads = self._total_threads
        wave_id = self._wave_id
        x_rsrc = self._x_rsrc
        x_row_base_div4 = self._x_row_base_div4
        x_col_local_i32 = self._x_col_local_i32
        x_row_local = self._x_row_local
        dma_bytes = self._dma_bytes
        wave_size = self._wave_size
        num_dma_loads = self._num_dma_loads
        c4_idx = arith.index(4)
        base_k_div4 = ((base_k / c_a_pack) * arith.constant(int(elem_bytes), index=True)) / arith.index(4)

        lds_ptr_i64 = None
        for i in range_constexpr(num_dma_loads):
            row_local_i = x_row_local[i]
            col_local_i32_i = x_col_local_i32[i]
            col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte_idx = row_k_dw * c4_idx + col_local_sw
            global_offset = arith.index_cast(T.i32, global_byte_idx)

            if const_expr(i == 0):
                lds_addr = lds_base_index(lds_buffer) + wave_id * arith.constant(wave_size * dma_bytes, index=True)
                # slice_k>1: shift into this K-slice group's X buffer (wid_k*input_elems bytes).
                if const_expr(self._has_group_base):
                    lds_addr = lds_addr + self._lds_group_base
                lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
            else:
                lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * dma_bytes, type=T.i64)

            lds_ptr = int_to_ptr(lds_ptr_i64, 3)

            rocdl.raw_ptr_buffer_load_lds(
                x_rsrc,
                lds_ptr,
                arith.constant(dma_bytes, type=T.i32),
                global_offset,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            )

    def prefetch_to_lds(self, base_k, lds_buffer):
        self._dma_x_tile_to_lds(base_k, lds_buffer)


class S2RLoaderA(object):
    """LDS -> register A subtile loads. Keeps the is_f8_a 256b (4x i64) vs fp4
    128b (2x i64) branch."""

    def __init__(
        self,
        *,
        vec16_x,
        vec2_i64,
        elem_bytes,
        a_elem_vec_pack,
        is_f8_a,
        k_blocks16,
        layout_lds,
        row_a_lds,
        col_offset_base,
        m_repeat,
        k_unroll,
        lds_group_base=None,
    ):
        self._vec16_x = vec16_x
        self._vec2_i64 = vec2_i64
        self._elem_bytes = elem_bytes
        self._a_elem_vec_pack = a_elem_vec_pack
        self._is_f8_a = is_f8_a
        self._k_blocks16 = k_blocks16
        self._layout_lds = layout_lds
        self._row_a_lds = row_a_lds
        self._col_offset_base = col_offset_base
        self._m_repeat = m_repeat
        self._k_unroll = k_unroll
        # slice_k>1: add this K-slice group's X-buffer base (wid_k*input_elems); slice_k==1 -> None.
        self._has_group_base = lds_group_base is not None
        self._lds_group_base = lds_group_base

    def lds_load_packs_k64(self, curr_row_a_lds, col_base, lds_buffer):
        elem_bytes = self._elem_bytes
        k_blocks16 = self._k_blocks16
        vec16_x = self._vec16_x
        vec2_i64 = self._vec2_i64
        col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
        col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / arith.index(2))
        idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], self._layout_lds)
        if const_expr(self._has_group_base):
            idx_a16 = idx_a16 + self._lds_group_base
        loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
        a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
        a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
        a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
        return a0, a1

    def prefetch_full_a_from_lds(self, lds_buffer, ku_limit=None):
        """Load entire A tile from LDS into registers before compute."""
        a_elem_vec_pack = self._a_elem_vec_pack
        col_offset_base = self._col_offset_base
        is_f8_a = self._is_f8_a
        m_repeat = self._m_repeat
        row_a_lds = self._row_a_lds
        if ku_limit is None:
            ku_limit = self._k_unroll
        a_regs = []
        for k_idx in range_constexpr(ku_limit):
            col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
            for mi_idx in range_constexpr(m_repeat):
                mi_val = arith.constant(mi_idx * 16, index=True)
                curr_row = row_a_lds + mi_val
                a0, a1 = self.lds_load_packs_k64(curr_row, col_base, lds_buffer)
                if const_expr(is_f8_a):
                    a2, a3 = self.lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                    a_regs.append((a0, a1, a2, a3))
                else:
                    a_regs.append((a0, a1))
        return a_regs

    def load_a_subtile(self, k_idx, mi_idx, lds_buffer):
        """Load a single A sub-tile from LDS (one ds_read)."""
        a_elem_vec_pack = self._a_elem_vec_pack
        col_offset_base = self._col_offset_base
        is_f8_a = self._is_f8_a
        row_a_lds = self._row_a_lds
        col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
        mi_val = arith.constant(mi_idx * 16, index=True)
        curr_row = row_a_lds + mi_val
        a0, a1 = self.lds_load_packs_k64(curr_row, col_base, lds_buffer)
        if const_expr(is_f8_a):
            a2, a3 = self.lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
            return (a0, a1, a2, a3)
        else:
            return (a0, a1)


class BLoader(object):
    """Per-K-step fp4 weight VMEM -> register loads (gate & up). When
    gate_up_interleave the up list is None (single-B pipe)."""

    def __init__(
        self,
        *,
        w_rsrc,
        layout_b,
        w_elem_type,
        vec2_i64,
        b_elem_bytes,
        kpack_bytes,
        b_nt,
        lane_div_16,
        num_acc_n,
        k_unroll,
        gate_up_interleave,
    ):
        self._w_rsrc = w_rsrc
        self._layout_b = layout_b
        self._w_elem_type = w_elem_type
        self._vec2_i64 = vec2_i64
        self._b_elem_bytes = b_elem_bytes
        self._kpack_bytes = kpack_bytes
        self._b_nt = b_nt
        self._lane_div_16 = lane_div_16
        self._num_acc_n = num_acc_n
        self._k_unroll = k_unroll
        self._gate_up_interleave = gate_up_interleave
        # per-tile block lists (set by for_tile)
        self._gate_n_blk_list = None
        self._gate_n_intra_list = None
        self._up_n_blk_list = None
        self._up_n_intra_list = None

    def for_tile(self, gate_n_blk_list, gate_n_intra_list, up_n_blk_list, up_n_intra_list):
        self._gate_n_blk_list = gate_n_blk_list
        self._gate_n_intra_list = gate_n_intra_list
        self._up_n_blk_list = up_n_blk_list
        self._up_n_intra_list = up_n_intra_list

    def load_b_packs_k64(self, base_k, ku, n_blk, n_intra):
        b_elem_bytes = self._b_elem_bytes
        b_nt = self._b_nt
        kpack_bytes = self._kpack_bytes
        lane_div_16 = self._lane_div_16
        layout_b = self._layout_b
        w_rsrc = self._w_rsrc
        c64 = arith.constant(64, index=True)
        base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
        k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
        k1 = lane_div_16
        coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
        idx_pack = crd2idx(coord_pack, layout_b)
        vec_elems = kpack_bytes // int(b_elem_bytes)
        b16 = _buffer_load_vec(
            buffer_ops,
            vector,
            w_rsrc,
            idx_pack,
            elem_type=self._w_elem_type(),
            vec_elems=vec_elems,
            elem_bytes=b_elem_bytes,
            offset_in_bytes=(b_elem_bytes == 1),
            cache_modifier=b_nt,
        )
        b_i64x2 = vector.bitcast(self._vec2_i64, b16)
        b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
        b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
        return b0, b1

    def load_b_tile(self, base_k, ku_limit=None):
        """Load B tiles. Returns (gate_b_tile, up_b_tile).
        When gate_up_interleave, up_b_tile is None."""
        gate_n_blk_list = self._gate_n_blk_list
        gate_n_intra_list = self._gate_n_intra_list
        up_n_blk_list = self._up_n_blk_list
        up_n_intra_list = self._up_n_intra_list
        gate_up_interleave = self._gate_up_interleave
        num_acc_n = self._num_acc_n
        if ku_limit is None:
            ku_limit = self._k_unroll
        gate_b_tile = []
        up_b_tile = [] if (not gate_up_interleave) else None
        for ku in range_constexpr(ku_limit):
            g_packs0, g_packs1 = [], []
            u_packs0, u_packs1 = [], []
            for ni in range_constexpr(num_acc_n):
                gb0, gb1 = self.load_b_packs_k64(base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni])
                g_packs0.append(gb0)
                g_packs1.append(gb1)
                if const_expr(not gate_up_interleave):
                    ub0, ub1 = self.load_b_packs_k64(base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni])
                    u_packs0.append(ub0)
                    u_packs1.append(ub1)
            gate_b_tile.append((g_packs0, g_packs1))
            if const_expr(not gate_up_interleave):
                up_b_tile.append((u_packs0, u_packs1))
        return gate_b_tile, up_b_tile


class KScaleLoader(object):
    """Per-K-step A / B micro-scale reads (`load_a_scale_i32`,
    `prefetch_ab_scale_tile`)."""

    def __init__(
        self,
        *,
        sw_rsrc,
        layout_b_scale,
        lds_a_scale,
        lane_mod_16,
        m_repeat_packed,
        num_acc_n_packed,
        gate_up_interleave,
        raw_sni=None,
        raw_klane_sh=None,
        sx_rsrc=None,
        sc_cp_vec=None,
        sc_cp_iters=None,
        total_threads=None,
        tx=None,
    ):
        self._sw_rsrc = sw_rsrc
        self._layout_b_scale = layout_b_scale
        self._lds_a_scale = lds_a_scale
        self._lane_mod_16 = lane_mod_16
        self._m_repeat_packed = m_repeat_packed
        self._num_acc_n_packed = num_acc_n_packed
        self._gate_up_interleave = gate_up_interleave
        self._raw_sni = raw_sni
        self._raw_klane_sh = raw_klane_sh
        # raw_a_scale LDS-staging invariants (stage_a_scale_to_lds)
        self._sx_rsrc = sx_rsrc
        self._sc_cp_vec = sc_cp_vec
        self._sc_cp_iters = sc_cp_iters
        self._total_threads = total_threads
        self._tx = tx
        # per-tile (set by for_tile)
        self._gate_scale_bases = None
        self._up_scale_bases = None
        self._rearrange_a_scale = None
        self._rearrange_b_scale = None

    def for_tile(
        self,
        *,
        gate_scale_bases,
        up_scale_bases,
        rearrange_a_scale,
        rearrange_b_scale,
    ):
        self._gate_scale_bases = gate_scale_bases
        self._up_scale_bases = up_scale_bases
        self._rearrange_a_scale = rearrange_a_scale
        self._rearrange_b_scale = rearrange_b_scale

    def stage_a_scale_to_lds(self, bx_m):
        """Stage this tile's full A activation-scale into LDS once, before the K-loop, to keep
        these cooperative VMEM loads out of the in-flight X-DMA vmcnt chain; one workgroup
        barrier publishes it to all waves."""
        sc_row_base = bx_m * arith.constant(self._raw_sni, index=True)
        sc_tx = self._tx * arith.constant(self._sc_cp_vec, index=True)
        for j in range_constexpr(self._sc_cp_iters):
            sc_f = sc_tx + arith.constant(j * self._total_threads * self._sc_cp_vec, index=True)
            sc_v = buffer_ops.buffer_load(
                self._sx_rsrc,
                sc_row_base + sc_f,
                vec_width=self._sc_cp_vec,
                dtype=T.i32,
                cache_modifier=0,
            )
            if const_expr(self._sc_cp_vec == 1):
                sc_v = vector.from_elements(T.vec(1, T.i32), [sc_v])
            vector.store(sc_v, self._lds_a_scale, [sc_f], alignment=self._sc_cp_vec * 4)
        gpu.barrier()

    def load_a_scale_i32(self, mi, koff):
        # Read the two vec2-i32 A-scale from LDS (prologue-staged, no swizzle) and repack to CK layout.
        raw_sni = self._raw_sni
        raw_klane_sh = self._raw_klane_sh
        lane_mod_16 = self._lane_mod_16
        layout_b_scale = self._layout_b_scale
        lds_a_scale = self._lds_a_scale
        k1 = koff // layout_b_scale.stride_k0
        ca = k1 * arith.constant(2, index=True)
        # Row-local = global row - bx_m = _mi*32 + lane_mod_16 (and +16); col = _ca.
        rl0 = arith.constant(mi * 32, index=True) + lane_mod_16
        rl16 = arith.constant(mi * 32 + 16, index=True) + lane_mod_16
        idx0 = rl0 * arith.constant(raw_sni, index=True) + ca
        idx16 = rl16 * arith.constant(raw_sni, index=True) + ca
        v0 = vector.load_op(T.vec(2, T.i32), lds_a_scale, [idx0])
        v1 = vector.load_op(T.vec(2, T.i32), lds_a_scale, [idx16])
        ia = vector.extract(v0, static_position=[0], dynamic_position=[])
        ib = vector.extract(v0, static_position=[1], dynamic_position=[])
        ic = vector.extract(v1, static_position=[0], dynamic_position=[])
        id = vector.extract(v1, static_position=[1], dynamic_position=[])
        m8 = arith.constant(0xFF, type=T.i32)
        ba = arith.andi(arith.shrui(ia, raw_klane_sh), m8)
        bc = arith.andi(arith.shrui(ic, raw_klane_sh), m8)
        bb = arith.andi(arith.shrui(ib, raw_klane_sh), m8)
        bd = arith.andi(arith.shrui(id, raw_klane_sh), m8)
        return arith.ori(
            arith.ori(ba, arith.shli(bc, arith.constant(8, type=T.i32))),
            arith.ori(arith.shli(bb, arith.constant(16, type=T.i32)), arith.shli(bd, arith.constant(24, type=T.i32))),
        )

    def prefetch_ab_scale_tile(self, base_k, ku_packed_limit):
        gate_up_interleave = self._gate_up_interleave
        m_repeat_packed = self._m_repeat_packed
        num_acc_n_packed = self._num_acc_n_packed
        layout_b_scale = self._layout_b_scale
        sw_rsrc = self._sw_rsrc
        gate_scale_bases = self._gate_scale_bases
        up_scale_bases = self._up_scale_bases
        a_scale_tile = []
        gate_b_scale = []
        up_b_scale = [] if (not gate_up_interleave) else None
        for ku in range_constexpr(ku_packed_limit):
            k_off = (ku + base_k) * layout_b_scale.stride_k0
            for mi in range_constexpr(m_repeat_packed):
                s = self._rearrange_a_scale(self.load_a_scale_i32(mi, k_off))
                a_scale_tile.append(vector.from_elements(T.vec(1, T.i32), [s]))
            for ni in range_constexpr(num_acc_n_packed):
                gs = buffer_ops.buffer_load(
                    sw_rsrc,
                    gate_scale_bases[ni] + k_off,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=0,
                )
                gs = self._rearrange_b_scale(gs)
                gate_b_scale.append(vector.from_elements(T.vec(1, T.i32), [gs]))
                if const_expr(not gate_up_interleave):
                    us = buffer_ops.buffer_load(
                        sw_rsrc,
                        up_scale_bases[ni] + k_off,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=0,
                    )
                    us = self._rearrange_b_scale(us)
                    up_b_scale.append(vector.from_elements(T.vec(1, T.i32), [us]))
        return [a_scale_tile, gate_b_scale, up_b_scale]


class MfmaScaleGU(object):
    """gate+up scaled MFMA (dual accumulator, fp8/fp4 A) on the raw
    rocdl.mfma_scale_f32_16x16x128_f8f6f4 op with runtime cbsz/blgp and per-iteration opsel
    (opsel_a = ikxdl*pack_M+imxdl, opsel_b = ikxdl*pack_N+inxdl); fp8-A packs 256b, fp4-A 128b."""

    def __init__(
        self,
        *,
        is_f8_a,
        cbsz,
        blgp,
        pack_M,
        pack_N,
        pack_K,
        m_repeat,
        m_repeat_packed,
        num_acc_n,
        num_acc_n_packed,
        k_unroll,
        vec4_f32,
        gate_up_interleave,
    ):
        self._is_f8_a = is_f8_a
        self._cbsz = cbsz
        self._blgp = blgp
        self._pack_M = pack_M
        self._pack_N = pack_N
        self._pack_K = pack_K
        self._m_repeat = m_repeat
        self._m_repeat_packed = m_repeat_packed
        self._num_acc_n = num_acc_n
        self._num_acc_n_packed = num_acc_n_packed
        self._k_unroll = k_unroll
        self._vec4_f32 = vec4_f32
        self._gate_up_interleave = gate_up_interleave
        self._single_b_pipe = gate_up_interleave

    @staticmethod
    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
        vec4_i64 = T.vec(4, T.i64)
        vec8_i32 = T.vec(8, T.i32)
        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
        return vector.bitcast(vec8_i32, v4)

    def compute_tile(
        self,
        acc_gate_in,
        acc_up_in,
        gate_b_tile_in,
        up_b_tile_in,
        a_tile_regs,
        a_scale=None,
        gate_b_scale=None,
        up_b_scale=None,
        *,
        ku_count=None,
    ):
        cbsz = self._cbsz
        blgp = self._blgp
        is_f8_a = self._is_f8_a
        pack_K = self._pack_K
        pack_M = self._pack_M
        pack_N = self._pack_N
        m_repeat = self._m_repeat
        m_repeat_packed = self._m_repeat_packed
        num_acc_n = self._num_acc_n
        num_acc_n_packed = self._num_acc_n_packed
        vec4_f32 = self._vec4_f32
        gate_up_interleave = self._gate_up_interleave
        pack_i64x4_to_i32x8 = self.pack_i64x4_to_i32x8
        if ku_count is None:
            ku_count = self._k_unroll

        gate_list = list(acc_gate_in)
        single_b = gate_up_interleave
        up_list = None if single_b else list(acc_up_in)
        mfma_res_ty = vec4_f32

        c0_i64 = arith.constant(0, type=T.i64)

        eff_packed = (ku_count + pack_K - 1) // pack_K
        # B-major: fix B (ni), cycle A (mi) -- B stays in registers while A is repacked per mi.
        for ku128 in range_constexpr(eff_packed):
            for ni in range_constexpr(num_acc_n_packed):
                gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                gate_bs_val = vector.extract(
                    gate_bs_i32,
                    static_position=[0],
                    dynamic_position=[],
                )
                if const_expr(not single_b):
                    up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                    up_bs_val = vector.extract(up_bs_i32, static_position=[0], dynamic_position=[])
                for ikxdl in range_constexpr(pack_K):
                    k_idx = ku128 * pack_K + ikxdl
                    if const_expr(k_idx < ku_count):
                        gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                        if const_expr(not single_b):
                            up_bp0, up_bp1 = up_b_tile_in[k_idx]
                        for inxdl in range_constexpr(pack_N):
                            ni_idx = ni * pack_N + inxdl
                            gb0 = gate_bp0[ni_idx]
                            gb1 = gate_bp1[ni_idx]
                            gb128 = pack_i64x4_to_i32x8(gb0, gb1, c0_i64, c0_i64)
                            if const_expr(not single_b):
                                ub0 = up_bp0[ni_idx]
                                ub1 = up_bp1[ni_idx]
                                ub128 = pack_i64x4_to_i32x8(ub0, ub1, c0_i64, c0_i64)
                            for mi in range_constexpr(m_repeat_packed):
                                a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                a_scale_val = vector.extract(
                                    a_scale_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                for imxdl in range_constexpr(pack_M):
                                    mi_idx = mi * pack_M + imxdl
                                    a_reg_idx = k_idx * m_repeat + mi_idx
                                    if const_expr(is_f8_a):
                                        a0, a1, a2, a3 = a_tile_regs[a_reg_idx]
                                        a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                    else:
                                        a0, a1 = a_tile_regs[a_reg_idx]
                                        a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                                    acc_idx = mi_idx * num_acc_n + ni_idx
                                    gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            gb128,
                                            gate_list[acc_idx],
                                            cbsz,
                                            blgp,
                                            ikxdl * pack_M + imxdl,
                                            a_scale_val,
                                            ikxdl * pack_N + inxdl,
                                            gate_bs_val,
                                        ],
                                    )
                                    if const_expr(not single_b):
                                        up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                            mfma_res_ty,
                                            [
                                                a128,
                                                ub128,
                                                up_list[acc_idx],
                                                cbsz,
                                                blgp,
                                                ikxdl * pack_M + imxdl,
                                                a_scale_val,
                                                ikxdl * pack_N + inxdl,
                                                up_bs_val,
                                            ],
                                        )
        return gate_list, up_list

    def compute_bmajor_mfma_phase(
        self,
        all_a_tiles,
        gate_b_single,
        up_b_single,
        a_scale_vals,
        gate_bs_val,
        up_bs_val,
        gate_list,
        up_list,
        k_idx,
        ni_idx,
        ikxdl,
        inxdl,
    ):
        """B-major MFMA: fix one B (ni), pack it once, cycle all A tiles (mi). When
        _single_b_pipe, up_b_single is None."""
        cbsz = self._cbsz
        blgp = self._blgp
        is_f8_a = self._is_f8_a
        pack_M = self._pack_M
        pack_N = self._pack_N
        m_repeat = self._m_repeat
        m_repeat_packed = self._m_repeat_packed
        num_acc_n = self._num_acc_n
        vec4_f32 = self._vec4_f32
        single_b_pipe = self._single_b_pipe
        pack = self.pack_i64x4_to_i32x8
        c0_i64 = arith.constant(0, type=T.i64)

        mfma_res_ty = vec4_f32
        gb128 = pack(gate_b_single[0], gate_b_single[1], c0_i64, c0_i64)
        if const_expr(not single_b_pipe):
            ub128 = pack(up_b_single[0], up_b_single[1], c0_i64, c0_i64)

        for mi_p in range_constexpr(m_repeat_packed):
            a_scale_val = a_scale_vals[mi_p]
            for imxdl in range_constexpr(pack_M):
                mi_idx = mi_p * pack_M + imxdl
                a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                if const_expr(is_f8_a):
                    a128 = pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                else:
                    a128 = pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

                acc_idx = mi_idx * num_acc_n + ni_idx
                gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty,
                    [
                        a128,
                        gb128,
                        gate_list[acc_idx],
                        cbsz,
                        blgp,
                        ikxdl * pack_M + imxdl,
                        a_scale_val,
                        ikxdl * pack_N + inxdl,
                        gate_bs_val,
                    ],
                )
                if const_expr(not single_b_pipe):
                    up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [
                            a128,
                            ub128,
                            up_list[acc_idx],
                            cbsz,
                            blgp,
                            ikxdl * pack_M + imxdl,
                            a_scale_val,
                            ikxdl * pack_N + inxdl,
                            up_bs_val,
                        ],
                    )


class KLoop(object):
    """Deep software-pipelined K-loop driver: prologue prime, ping/pong steady loop of
    interleaved half-iterations, and odd/even drain, with MFMA lagging a full half-iteration
    behind the loads (sched_barrier / s_waitcnt / s_setprio fence choreography)."""

    def __init__(
        self,
        *,
        g2s_x,
        s2r_a,
        b_loader,
        k_scale,
        mfma,
        lds_x_ping,
        lds_x_pong,
        sw_rsrc,
        layout_b_scale,
        barrier,
        acc_init,
        k_base_idx,
        k_dim,
        tile_k,
        pack_K,
        pack_N,
        k_unroll,
        k_unroll_packed,
        m_repeat,
        m_repeat_packed,
        num_acc_n,
        num_acc_n_packed,
        single_b_pipe,
        use_async_copy,
        isched,
        ck_rate,
        vmcnt_before_barrier,
        pipe_n_phases,
        pp_mfma,
        pp_a_reads,
        pp_b_loads,
        pp_has_scale,
        gate_n_blk_list,
        gate_n_intra_list,
        up_n_blk_list,
        up_n_intra_list,
        gate_scale_bases,
        up_scale_bases,
        rearrange_a_scale,
        rearrange_b_scale,
    ):
        self._g2s_x = g2s_x
        self._s2r_a = s2r_a
        self._b_loader = b_loader
        self._k_scale = k_scale
        self._mfma = mfma
        self._lds_x_ping = lds_x_ping
        self._lds_x_pong = lds_x_pong
        self._sw_rsrc = sw_rsrc
        self._layout_b_scale = layout_b_scale
        self._barrier = barrier
        self._acc_init = acc_init
        self._k_base_idx = k_base_idx
        self._k_dim = k_dim
        self._tile_k = tile_k
        self._pack_K = pack_K
        self._pack_N = pack_N
        self._k_unroll = k_unroll
        self._k_unroll_packed = k_unroll_packed
        self._m_repeat = m_repeat
        self._m_repeat_packed = m_repeat_packed
        self._num_acc_n = num_acc_n
        self._num_acc_n_packed = num_acc_n_packed
        self._single_b_pipe = single_b_pipe
        self._use_async_copy = use_async_copy
        self._isched = isched
        self._ck_rate = ck_rate
        self._vmcnt_before_barrier = vmcnt_before_barrier
        self._pipe_n_phases = pipe_n_phases
        self._pp_mfma = pp_mfma
        self._pp_a_reads = pp_a_reads
        self._pp_b_loads = pp_b_loads
        self._pp_has_scale = pp_has_scale
        self._gate_n_blk_list = gate_n_blk_list
        self._gate_n_intra_list = gate_n_intra_list
        self._up_n_blk_list = up_n_blk_list
        self._up_n_intra_list = up_n_intra_list
        self._gate_scale_bases = gate_scale_bases
        self._up_scale_bases = up_scale_bases
        self._rearrange_a_scale = rearrange_a_scale
        self._rearrange_b_scale = rearrange_b_scale

    def _interleaved_half(
        self,
        lds_read,
        lds_write,
        next_k_dma_py,
        next_k_load,
        prev_a_tile,
        prev_gate_w,
        prev_up_w,
        prev_a_scale,
        prev_gate_bs,
        prev_up_bs,
        acc_gate,
        acc_up,
    ):
        """One interleaved half-iteration (deep pipeline): DMA targets lds_write (the OTHER
        buffer) while ds_read uses lds_read (DMA'd in the previous half); MFMAs run on prev
        data."""
        # rebind loop-invariants
        k_base_idx = self._k_base_idx
        pack_K = self._pack_K
        pack_N = self._pack_N
        layout_b_scale = self._layout_b_scale
        vmcnt_before_barrier = self._vmcnt_before_barrier
        barrier = self._barrier
        use_async_copy = self._use_async_copy
        k_dim = self._k_dim
        prefetch_x_to_lds = self._g2s_x.prefetch_to_lds
        load_x_tile = self._g2s_x.load_tile
        store_x_tile_to_lds = self._g2s_x.store_to_lds
        load_a_subtile = self._s2r_a.load_a_subtile
        load_b_packs_k64 = self._b_loader.load_b_packs_k64
        load_a_scale_i32 = self._k_scale.load_a_scale_i32
        compute_bmajor_mfma_phase = self._mfma.compute_bmajor_mfma_phase
        m_repeat = self._m_repeat
        m_repeat_packed = self._m_repeat_packed
        num_acc_n = self._num_acc_n
        num_acc_n_packed = self._num_acc_n_packed
        k_unroll = self._k_unroll
        single_b_pipe = self._single_b_pipe
        isched = self._isched
        ck_rate = self._ck_rate
        pipe_n_phases = self._pipe_n_phases
        pp_mfma = self._pp_mfma
        pp_a_reads = self._pp_a_reads
        pp_b_loads = self._pp_b_loads
        pp_has_scale = self._pp_has_scale
        sw_rsrc = self._sw_rsrc
        gate_scale_bases = self._gate_scale_bases
        up_scale_bases = self._up_scale_bases
        rearrange_a_scale = self._rearrange_a_scale
        rearrange_b_scale = self._rearrange_b_scale
        gate_n_blk_list = self._gate_n_blk_list
        gate_n_intra_list = self._gate_n_intra_list
        up_n_blk_list = self._up_n_blk_list
        up_n_intra_list = self._up_n_intra_list
        abs_k = k_base_idx + arith.constant(next_k_load, index=True)
        bk = abs_k // arith.constant(2, index=True)
        sk = abs_k // arith.constant(pack_K * 128, index=True)
        k_off = sk * layout_b_scale.stride_k0

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(vmcnt_before_barrier)
        barrier()
        rocdl.sched_barrier(0)

        # DMA A to OTHER buffer (for next half), non-blocking
        abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
        if const_expr(use_async_copy and next_k_dma_py < int(k_dim)):
            prefetch_x_to_lds(abs_k_dma, lds_write)
        if const_expr(not use_async_copy):
            x_regs = load_x_tile(abs_k_dma)

        # extract previous scale values
        prev_asvs = []
        for mi_p in range_constexpr(m_repeat_packed):
            prev_asvs.append(
                vector.extract(
                    prev_a_scale[mi_p],
                    static_position=[0],
                    dynamic_position=[],
                )
            )
        prev_gsv_list = []
        for gs_ni in range_constexpr(num_acc_n_packed):
            prev_gsv_list.append(
                vector.extract(
                    prev_gate_bs[gs_ni],
                    static_position=[0],
                    dynamic_position=[],
                )
            )
        if const_expr(not single_b_pipe):
            prev_usv_list = []
            for us_ni in range_constexpr(num_acc_n_packed):
                prev_usv_list.append(
                    vector.extract(
                        prev_up_bs[us_ni],
                        static_position=[0],
                        dynamic_position=[],
                    )
                )

        # execute phases from unified schedule
        a_all = {}
        b_gate_all = {}
        b_up_all = {}

        for p in range_constexpr(pipe_n_phases):
            # Scale VMEM loads (phase 0 only)
            if const_expr(pp_has_scale[p]):
                new_as_list = []
                for mi_p in range_constexpr(m_repeat_packed):
                    new_as_list.append(rearrange_a_scale(load_a_scale_i32(mi_p, k_off)))
                new_gs_list = []
                for gs_ni in range_constexpr(num_acc_n_packed):
                    gs_raw = buffer_ops.buffer_load(
                        sw_rsrc,
                        gate_scale_bases[gs_ni] + k_off,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=0,
                    )
                    new_gs_list.append(rearrange_b_scale(gs_raw))
                if const_expr(not single_b_pipe):
                    new_us_list = []
                    for us_ni in range_constexpr(num_acc_n_packed):
                        us_raw = buffer_ops.buffer_load(
                            sw_rsrc,
                            up_scale_bases[us_ni] + k_off,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=0,
                        )
                        new_us_list.append(rearrange_b_scale(us_raw))

            # B VMEM loads
            for b_j in range_constexpr(len(pp_b_loads[p])):
                b_type, b_ku, b_ni = pp_b_loads[p][b_j]
                if const_expr(b_type == "gate"):
                    b_gate_all[(b_ku, b_ni)] = load_b_packs_k64(
                        bk,
                        b_ku,
                        gate_n_blk_list[b_ni],
                        gate_n_intra_list[b_ni],
                    )
                else:
                    b_up_all[(b_ku, b_ni)] = load_b_packs_k64(
                        bk,
                        b_ku,
                        up_n_blk_list[b_ni],
                        up_n_intra_list[b_ni],
                    )

            # A ds_reads
            if const_expr(isched == 0):
                rocdl.sched_barrier(0)
            for a_j in range_constexpr(len(pp_a_reads[p])):
                ak, ami = pp_a_reads[p][a_j]
                a_all[(ak, ami)] = load_a_subtile(
                    ak,
                    ami,
                    lds_read,
                )
            if const_expr(isched == 0):
                rocdl.sched_barrier(0)

            # MFMAs on prev data
            rocdl.s_setprio(1)
            for m_j in range_constexpr(len(pp_mfma[p])):
                k_idx, ni_idx, ikxdl, inxdl, ku128 = pp_mfma[p][m_j]
                ni_packed_idx = ni_idx // pack_N
                up_b_single = (
                    (
                        prev_up_w[k_idx][0][ni_idx],
                        prev_up_w[k_idx][1][ni_idx],
                    )
                    if not single_b_pipe
                    else None
                )
                compute_bmajor_mfma_phase(
                    prev_a_tile,
                    (
                        prev_gate_w[k_idx][0][ni_idx],
                        prev_gate_w[k_idx][1][ni_idx],
                    ),
                    up_b_single,
                    prev_asvs,
                    prev_gsv_list[ni_packed_idx],
                    (prev_usv_list[ni_packed_idx] if not single_b_pipe else None),
                    acc_gate,
                    acc_up,
                    k_idx,
                    ni_idx,
                    ikxdl,
                    inxdl,
                )
            rocdl.s_setprio(0)
            if const_expr(isched >= 2):
                # CK-style: interleave this phase's loads into its MFMA stream (R MFMAs/load) to hide VMEM/DS latency.
                na_s = len(pp_a_reads[p])
                nb_s = len(pp_b_loads[p])
                if const_expr(isched == 3):
                    # mode 3: alternate A(ds)/B(vmem) evenly across the MFMAs
                    for i_s in range_constexpr(max(na_s, nb_s)):
                        if const_expr(i_s < na_s):
                            rocdl.sched_mfma(ck_rate)
                            rocdl.sched_dsrd(1)
                        if const_expr(i_s < nb_s):
                            rocdl.sched_mfma(ck_rate)
                            rocdl.sched_vmem(1)
                else:
                    for _ in range_constexpr(na_s):
                        rocdl.sched_mfma(ck_rate)
                        rocdl.sched_dsrd(1)
                    for _ in range_constexpr(nb_s):
                        rocdl.sched_mfma(ck_rate)
                        rocdl.sched_vmem(1)
                rocdl.sched_mfma(256)
            rocdl.sched_barrier(0)

        # assemble loaded data for next half-iteration
        cur_a_tile = []
        for k in range_constexpr(k_unroll):
            for mi in range_constexpr(m_repeat):
                cur_a_tile.append(a_all[(k, mi)])

        cur_gate_w = []
        cur_up_w = None if single_b_pipe else []
        for ku in range_constexpr(k_unroll):
            g_packs0, g_packs1 = [], []
            u_packs0, u_packs1 = [], []
            for ni in range_constexpr(num_acc_n):
                g = b_gate_all[(ku, ni)]
                g_packs0.append(g[0])
                g_packs1.append(g[1])
                if const_expr(not single_b_pipe):
                    u = b_up_all[(ku, ni)]
                    u_packs0.append(u[0])
                    u_packs1.append(u[1])
            cur_gate_w.append((g_packs0, g_packs1))
            if const_expr(not single_b_pipe):
                cur_up_w.append((u_packs0, u_packs1))

        cur_a_scale = []
        for mi_p in range_constexpr(m_repeat_packed):
            cur_a_scale.append(
                vector.from_elements(
                    T.vec(1, T.i32),
                    [new_as_list[mi_p]],
                )
            )
        cur_gate_bs = []
        for gs_ni in range_constexpr(num_acc_n_packed):
            cur_gate_bs.append(vector.from_elements(T.vec(1, T.i32), [new_gs_list[gs_ni]]))
        if const_expr(not single_b_pipe):
            cur_up_bs = []
            for us_ni in range_constexpr(num_acc_n_packed):
                cur_up_bs.append(vector.from_elements(T.vec(1, T.i32), [new_us_list[us_ni]]))
        else:
            cur_up_bs = None

        if const_expr(not use_async_copy):
            store_x_tile_to_lds(x_regs, lds_write)

        return (
            cur_a_tile,
            cur_gate_w,
            cur_up_w,
            cur_a_scale,
            cur_gate_bs,
            cur_up_bs,
            acc_gate,
            acc_up,
        )

    def run(self, a_scale_pong, gate_bs_pong, up_bs_pong):
        """Prime -> ping/pong steady loop -> odd/even drain. Receives the k0 B/A-scale prefetch
        (a_scale_pong/gate_bs_pong/up_bs_pong); returns (acc_gate, acc_up)."""
        # rebind loop-invariants
        acc_init = self._acc_init
        num_acc_n = self._num_acc_n
        m_repeat = self._m_repeat
        single_b_pipe = self._single_b_pipe
        k_base_idx = self._k_base_idx
        tile_k = self._tile_k
        pack_K = self._pack_K
        k_unroll = self._k_unroll
        k_dim = self._k_dim
        use_async_copy = self._use_async_copy
        barrier = self._barrier
        lds_x_ping = self._lds_x_ping
        lds_x_pong = self._lds_x_pong
        prefetch_x_to_lds = self._g2s_x.prefetch_to_lds
        load_x_tile = self._g2s_x.load_tile
        store_x_tile_to_lds = self._g2s_x.store_to_lds
        prefetch_full_a_from_lds = self._s2r_a.prefetch_full_a_from_lds
        load_b_tile = self._b_loader.load_b_tile
        compute_tile = self._mfma.compute_tile

        def prefetch_ab_scale_tile(base_k, ku_packed_limit=self._k_unroll_packed):
            return self._k_scale.prefetch_ab_scale_tile(base_k, ku_packed_limit)

        acc_gate = [acc_init] * num_acc_n * m_repeat
        acc_up = [acc_init] * num_acc_n * m_repeat if not single_b_pipe else None

        k1 = k_base_idx + arith.constant(tile_k, index=True)
        rocdl.sched_barrier(0)
        if const_expr(use_async_copy):
            prefetch_x_to_lds(k1, lds_x_ping)
        else:
            x_regs_prime = load_x_tile(k1)
            store_x_tile_to_lds(x_regs_prime, lds_x_ping)

        k0_b = k_base_idx // arith.constant(2, index=True)
        gate_w0, up_w0 = load_b_tile(k0_b)
        # Prime the deep pipeline: DMA K=tile_k -> ping (1 tile ahead)
        if const_expr(use_async_copy):
            rocdl.s_waitcnt(0)
        gpu.barrier()
        rocdl.sched_barrier(0)
        a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(6)

        num_k_tiles_py = int(k_dim) // int(tile_k)
        odd_k_tiles = (num_k_tiles_py % 2) == 1
        tail_tiles = 1 if odd_k_tiles else 2
        k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
        if const_expr(k_main2_py < 0):
            k_main2_py = 0

        gate_w_pong = gate_w0
        up_w_pong = up_w0

        rocdl.sched_barrier(0)

        if const_expr(k_main2_py > 0):
            for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                next_k_load_1 = k_iv_py + tile_k
                next_k_load_2 = k_iv_py + tile_k * 2
                next_k_dma_1 = k_iv_py + tile_k * 2
                next_k_dma_2 = k_iv_py + tile_k * 3

                # Half 1: read ping (DMA'd prev half), DMA->pong, MFMA(pong)
                (
                    a_tile_ping,
                    gate_w_ping,
                    up_w_ping,
                    a_scale_ping,
                    gate_bs_ping,
                    up_bs_ping,
                    acc_gate,
                    acc_up,
                ) = self._interleaved_half(
                    lds_x_ping,
                    lds_x_pong,
                    next_k_dma_1,
                    next_k_load_1,
                    a_tile_pong,
                    gate_w_pong,
                    up_w_pong,
                    a_scale_pong,
                    gate_bs_pong,
                    up_bs_pong,
                    acc_gate,
                    acc_up,
                )

                # Half 2: read pong (DMA'd Half 1), DMA->ping, MFMA(ping)
                (
                    a_tile_pong,
                    gate_w_pong,
                    up_w_pong,
                    a_scale_pong,
                    gate_bs_pong,
                    up_bs_pong,
                    acc_gate,
                    acc_up,
                ) = self._interleaved_half(
                    lds_x_pong,
                    lds_x_ping,
                    next_k_dma_2,
                    next_k_load_2,
                    a_tile_ping,
                    gate_w_ping,
                    up_w_ping,
                    a_scale_ping,
                    gate_bs_ping,
                    up_bs_ping,
                    acc_gate,
                    acc_up,
                )

        if const_expr(odd_k_tiles):
            acc_gate, acc_up = compute_tile(
                acc_gate,
                acc_up,
                gate_w_pong,
                up_w_pong,
                a_tile_pong,
                a_scale_pong,
                gate_bs_pong,
                up_bs_pong,
                ku_count=k_unroll,
            )
        else:
            k_tail_rel = arith.constant(k_dim - tile_k, index=True)
            k_tail1 = k_base_idx + k_tail_rel
            x_regs_ping = []
            if const_expr(use_async_copy):
                prefetch_x_to_lds(k_tail1, lds_x_ping)
            else:
                x_regs_ping = load_x_tile(k_tail1)
            gate_w_ping, up_w_ping = load_b_tile(k_tail1 // arith.constant(2, index=True))
            a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                k_tail1 // arith.constant(pack_K * 128, index=True)
            )
            acc_gate, acc_up = compute_tile(
                acc_gate,
                acc_up,
                gate_w_pong,
                up_w_pong,
                a_tile_pong,
                a_scale_pong,
                gate_bs_pong,
                up_bs_pong,
            )
            if const_expr(not use_async_copy):
                store_x_tile_to_lds(x_regs_ping, lds_x_ping)
            rocdl.s_waitcnt(0)
            barrier()
            a_tile_ping = prefetch_full_a_from_lds(lds_x_ping)
            acc_gate, acc_up = compute_tile(
                acc_gate,
                acc_up,
                gate_w_ping,
                up_w_ping,
                a_tile_ping,
                a_scale_ping,
                gate_bs_ping,
                up_bs_ping,
                ku_count=k_unroll,
            )

        return acc_gate, acc_up
