# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SageAttention CDNA helpers: traits, context, loaders, GEMM, softmax, store."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_f32_32x32x16_fp8_fp8 as _ods_mfma_f32_32x32x16_fp8_fp8,
)
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_i32_32x32x16_i8 as _ods_mfma_i32_32x32x16_i8,
)
from flydsl.expr import arith, buffer_ops, const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.attention.sage_attn_utils import (
    _LDS_CAP_BYTES,
    _extract_aligned_pointer,
    _f32_to_bf16_trunc,
    _i32_pair_to_i64,
    _lds_vec_load,
    _lds_vec_store,
    _lds_vec_store_elem,
    _pointer_load,
)
from kernels.common.kernels_common import get_warp_size


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


class _SageTraits:
    """Compile-time constants for one SageAttention CDNA launch config."""

    def __init__(
        self,
        *,
        BLOCK_M,
        BLOCK_N,
        BLOCK_SIZE,
        CAUSAL,
        D_CHUNKS,
        GROUPS,
        HEAD_DIM,
        KV_STRIDE_TOKEN,
        K_NEEDS_GUARD,
        K_STEPS_QK,
        K_STRIDE,
        LDS_K_BYTES,
        LDS_K_TOTAL_BYTES,
        LDS_V_BYTES,
        LDS_V_TOTAL_BYTES,
        MFMA_K_FP8,
        MFMA_K_INT8,
        MFMA_M,
        MFMA_N,
        NUM_BATCHES_K,
        NUM_BATCHES_V,
        NUM_KV_HEADS,
        NUM_PIPE_STAGES,
        NUM_Q_HEADS,
        PV_K_STEPS,
        Q_SCALE_BLOCK_M,
        ROWS_PER_BATCH_K,
        ROWS_PER_BATCH_V,
        ROWS_PER_WAVE,
        STRIDE_TOKEN,
        THREADS_PER_ROW_K,
        THREADS_PER_ROW_V,
        VEC_WIDTH_K,
        VEC_WIDTH_V,
        V_NEEDS_GUARD,
        V_STRIDE,
        V_TRANSPOSED,
        WARP_SIZE,
        _DEFER_SCALE,
        _LPT_SCHED,
        _dma_any,
        _dma_k,
        _dma_xor,
        lds_base_offset,
        use_bias,
    ):
        """Store all launch-config constants as attributes."""
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_SIZE = BLOCK_SIZE
        self.CAUSAL = CAUSAL
        self.D_CHUNKS = D_CHUNKS
        self.GROUPS = GROUPS
        self.HEAD_DIM = HEAD_DIM
        self.KV_STRIDE_TOKEN = KV_STRIDE_TOKEN
        self.K_NEEDS_GUARD = K_NEEDS_GUARD
        self.K_STEPS_QK = K_STEPS_QK
        self.K_STRIDE = K_STRIDE
        self.LDS_K_BYTES = LDS_K_BYTES
        self.LDS_K_TOTAL_BYTES = LDS_K_TOTAL_BYTES
        self.LDS_V_BYTES = LDS_V_BYTES
        self.LDS_V_TOTAL_BYTES = LDS_V_TOTAL_BYTES
        self.MFMA_K_FP8 = MFMA_K_FP8
        self.MFMA_K_INT8 = MFMA_K_INT8
        self.MFMA_M = MFMA_M
        self.MFMA_N = MFMA_N
        self.NUM_BATCHES_K = NUM_BATCHES_K
        self.NUM_BATCHES_V = NUM_BATCHES_V
        self.NUM_KV_HEADS = NUM_KV_HEADS
        self.NUM_PIPE_STAGES = NUM_PIPE_STAGES
        self.NUM_Q_HEADS = NUM_Q_HEADS
        self.PV_K_STEPS = PV_K_STEPS
        self.Q_SCALE_BLOCK_M = Q_SCALE_BLOCK_M
        self.ROWS_PER_BATCH_K = ROWS_PER_BATCH_K
        self.ROWS_PER_BATCH_V = ROWS_PER_BATCH_V
        self.ROWS_PER_WAVE = ROWS_PER_WAVE
        self.STRIDE_TOKEN = STRIDE_TOKEN
        self.THREADS_PER_ROW_K = THREADS_PER_ROW_K
        self.THREADS_PER_ROW_V = THREADS_PER_ROW_V
        self.VEC_WIDTH_K = VEC_WIDTH_K
        self.VEC_WIDTH_V = VEC_WIDTH_V
        self.V_NEEDS_GUARD = V_NEEDS_GUARD
        self.V_STRIDE = V_STRIDE
        self.V_TRANSPOSED = V_TRANSPOSED
        self.WARP_SIZE = WARP_SIZE
        self._DEFER_SCALE = _DEFER_SCALE
        self._LPT_SCHED = _LPT_SCHED
        self._dma_any = _dma_any
        self._dma_k = _dma_k
        self._dma_xor = _dma_xor
        self.lds_base_offset = lds_base_offset
        self.use_bias = use_bias


class _SageKernel:
    """SageAttention CDNA emitter: helpers, prologue, epilogue.

    Methods read kernel SSA off ``self`` and constants off ``self.t``.
    """

    def __init__(self, t, allocator):
        """Bind config traits and the shared-memory allocator."""
        self.t = t
        self.allocator = allocator

    def mfma_i32_k16(self, result_type, operands):
        a, b, c = operands[:3]
        cbsz = operands[3] if len(operands) > 3 else 0
        abid = operands[4] if len(operands) > 4 else 0
        blgp = operands[5] if len(operands) > 5 else 0
        return _ods_mfma_i32_32x32x16_i8(
            res=result_type,
            a=_llvm_value(a),
            b=_llvm_value(b),
            c=_llvm_value(c),
            cbsz=cbsz,
            abid=abid,
            blgp=blgp,
        ).result

    def mfma_fp8_k16(self, result_type, operands):
        a, b, c = (operands[0], operands[1], operands[2])
        cbsz = operands[3] if len(operands) > 3 else 0
        abid = operands[4] if len(operands) > 4 else 0
        blgp = operands[5] if len(operands) > 5 else 0
        a_v = _llvm_value(a)
        b_v = _llvm_value(b)
        c_v = _llvm_value(c)
        return _ods_mfma_f32_32x32x16_fp8_fp8(
            res=result_type, a=a_v, b=b_v, c=c_v, cbsz=cbsz, abid=abid, blgp=blgp
        ).result

    def _fadd(self, a, b):
        return arith.addf(_raw(a), _raw(b), fastmath=self.fm_fast)

    def _fsub(self, a, b):
        return arith.subf(_raw(a), _raw(b), fastmath=self.fm_fast)

    def _fmul(self, a, b):
        return arith.mulf(_raw(a), _raw(b), fastmath=self.fm_fast)

    def _fmax(self, a, b):
        return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=self.fm_fast).result

    def _ffma(self, a, b, c):
        return math.fma(_raw(a), _raw(b), _raw(c), fastmath=self.fm_fast)

    def _fdiv(self, a, b):
        return arith.divf(_raw(a), _raw(b), fastmath=self.fm_fast)

    def q_global_idx(self, token_idx, col):
        token = self.batch_idx * self.seq_len_q_v + token_idx
        return token * self.t.STRIDE_TOKEN + self.head_q_idx * self.t.HEAD_DIM + col

    def kv_global_idx(self, token_idx, col):
        token = self.batch_idx * self.seq_len_k_v + token_idx
        return token * self.t.KV_STRIDE_TOKEN + self.head_kv_idx * self.t.HEAD_DIM + col

    def _load_gm_i64(self, div, elem_idx):
        v8i8 = fly.copy_atom_call_ssa([self.v8i8_type], self.load_atom_64_i8, fx.slice(div, (None, fx.Int32(elem_idx))))
        return fx.Vector(v8i8).bitcast(fx.Int64)[0]

    def _load_ptr_f32(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
        return _pointer_load(T.f32, gep)

    def _load_ptr_f32_vec4(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
        return _pointer_load(self.v4f32_type, gep)

    def _store_gm_bf16x4(self, div, elem_idx, bf16_vec):
        pack = Vec(bf16_vec).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.o_store_reg)
        fx.copy(self.store_atom_64, self.o_store_reg, fx.slice(div, (None, fx.Int32(elem_idx))))

    def _bpermute_f32(self, value, src_lane):
        value_i32 = arith.bitcast(T.i32, _raw(value))
        lane_byte_idx = fx.Int32(src_lane * 4)
        result_i32 = rocdl.ds_bpermute(T.i32, _raw(lane_byte_idx), value_i32)
        return arith.bitcast(T.f32, result_i32)

    def reduction_peer32(self, v_f32):
        return fx.Float32(v_f32).shuffle_xor(self.shuf32_i32, self.width_i32)

    def row_max_reduce(self, local_max):
        """Reduce max across the 2 klane halves within wave64."""
        return self._fmax(local_max, self.reduction_peer32(local_max))

    def row_sum_reduce(self, local_sum):
        """Reduce sum across the 2 klane halves within wave64."""
        return self._fadd(local_sum, self.reduction_peer32(local_sum))

    def _coop_load_k_dma(self, tile_start, buf_off):
        assert not self.t.K_NEEDS_GUARD, "K DMA path requires ROWS_PER_BATCH_K<=BLOCK_N"
        DMA_BYTES_K = self.t.VEC_WIDTH_K
        buf_off_i64 = fx.Int64(buf_off).ir_value()
        wave_byte = rocdl.readfirstlane(T.i64, fx.Int64(self.wave_id * (self.t.WARP_SIZE * DMA_BYTES_K)).ir_value())
        base_ptr = self._lds_as3_base(self.lds)
        base_ptr = buffer_ops.get_element_ptr(base_ptr, buf_off_i64)
        base_ptr = buffer_ops.get_element_ptr(base_ptr, wave_byte)
        size_i32 = fx.Int32(DMA_BYTES_K).ir_value()
        c0 = fx.Int32(0).ir_value()
        aux1 = fx.Int32(1).ir_value()
        _CPR_K = self.t.HEAD_DIM // self.t.VEC_WIDTH_K
        _XMASK_K = _CPR_K - 1
        for batch in range_constexpr(self.t.NUM_BATCHES_K):
            row_offset = batch * self.t.ROWS_PER_BATCH_K
            row_idx_raw = tile_start + self.load_row_k_batch + row_offset
            if const_expr(self.t._dma_xor):
                tile_row = self.load_row_k_batch + row_offset
                phys_chunk = self.load_lane_k
                g_chunk = fx.Int64(phys_chunk) ^ fx.Int64(tile_row) & fx.Int64(_XMASK_K)
                g_col = g_chunk * self.t.VEC_WIDTH_K
                g_idx = self.kv_global_idx(row_idx_raw, g_col)
            else:
                g_idx = self.kv_global_idx(row_idx_raw, self.load_col_k_base)
            voff = fx.Int32(g_idx).ir_value()
            if const_expr(batch == 0):
                lds_ptr = base_ptr
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    base_ptr, static_byte_offset=batch * self.t.BLOCK_SIZE * DMA_BYTES_K
                )
            rocdl.raw_ptr_buffer_load_lds(self.k_rsrc, lds_ptr, size_i32, voff, c0, c0, aux1)

    def coop_load_k(self, tile_start, buf_off):
        tile_start = fx.Int64(tile_start)
        if const_expr(self.t._dma_k):
            self._coop_load_k_dma(tile_start, buf_off)
            return
        for batch in range_constexpr(self.t.NUM_BATCHES_K):
            row_offset = batch * self.t.ROWS_PER_BATCH_K
            row_idx_raw = tile_start + self.load_row_k_batch + row_offset
            if const_expr(self.t.K_NEEDS_GUARD):
                row_valid = self.load_row_k_batch < self.t.BLOCK_N
                do_load = row_valid
            else:
                do_load = True
            if do_load:
                g_idx = self.kv_global_idx(row_idx_raw, self.load_col_k_base)
                lds_row = self.load_row_k_batch + row_offset
                lds_idx = buf_off + lds_row * self.t.K_STRIDE + self.load_col_k_base
                v4i32 = fly.copy_atom_call_ssa(
                    [self.v4i32_type], self.load_atom_128, fx.slice(self.k_div, (None, fx.Int32(g_idx)))
                )
                vec = fx.Vector(v4i32).bitcast(fx.Int8)
                _lds_vec_store(vec, self.lds, lds_idx)

    def coop_load_v(self, tile_start, buf_off):
        """Cooperatively load V (FP8 raw bytes) from global into LDS (gfx942 column-major)."""
        tile_start = fx.Int64(tile_start)
        for batch in range_constexpr(self.t.NUM_BATCHES_V):
            row_offset = batch * self.t.ROWS_PER_BATCH_V
            row_idx_raw = tile_start + self.load_row_v_batch + row_offset
            if const_expr(self.t.V_NEEDS_GUARD):
                row_valid = self.load_row_v_batch < self.t.BLOCK_N
                do_load = row_valid
            else:
                do_load = True
            if do_load:
                if const_expr(self.t.V_TRANSPOSED):
                    d_idx = self.tid % self.t.HEAD_DIM
                    k_group = self.tid // self.t.HEAD_DIM
                    tile_idx = tile_start // self.t.BLOCK_N
                    g_byte_idx = (
                        (
                            (self.batch_idx * self.t.NUM_KV_HEADS + self.head_kv_idx) * self.num_k_blocks_per_head
                            + tile_idx
                        )
                        * self.t.HEAD_DIM
                        + d_idx
                    ) * self.t.BLOCK_N + k_group * 16
                    g_dword_i32 = fx.Int32(g_byte_idx >> 2).ir_value()
                    v4i32 = buffer_ops.buffer_load(self.v_rsrc, g_dword_i32, vec_width=4, dtype=T.i32)
                    raw_v = fx.Vector(v4i32).bitcast(fx.Int8)
                    v_off = buf_off + d_idx * self.t.V_STRIDE + k_group * 16
                    _lds_vec_store(raw_v, self.lds_v, v_off)
                else:
                    g_idx = self.kv_global_idx(row_idx_raw, self.load_col_v_base)
                    g_dword_i32 = fx.Int32(g_idx >> 2).ir_value()
                    v4i32 = buffer_ops.buffer_load(self.v_rsrc, g_dword_i32, vec_width=4, dtype=T.i32)
                    raw_v = fx.Vector(v4i32).bitcast(fx.Int8)
                    lds_col = self.load_row_v_batch + row_offset
                    for di in range_constexpr(self.t.VEC_WIDTH_V):
                        d_idx = self.load_col_v_base + di
                        v_off = buf_off + d_idx * self.t.V_STRIDE + lds_col
                        b_i8 = fx.Vector(raw_v)[di]
                        _lds_vec_store_elem(b_i8, self.lds_v, v_off, fx.Int8)

    def load_k_frag(self, kv_block_row, ks, buf_off):
        """Load K fragment from LDS for the QK MFMA."""
        if const_expr(self.t._dma_xor):
            _CPR_K = self.t.HEAD_DIM // self.t.VEC_WIDTH_K
            _XMASK_K = _CPR_K - 1
            g_chunk = fx.Int64(ks)
            phys_chunk = g_chunk ^ fx.Int64(kv_block_row) & fx.Int64(_XMASK_K)
            k_col = phys_chunk * self.t.VEC_WIDTH_K + self.klane * 8
        else:
            k_col = fx.Int64(ks * self.t.MFMA_K_INT8) + self.klane * 8
        lds_idx = buf_off + kv_block_row * self.t.K_STRIDE + k_col
        v8i8 = _lds_vec_load(self.v8i8_type, self.lds, lds_idx)
        k_i64v = fx.Vector(v8i8).bitcast(fx.Int64)
        return fx.Vector(k_i64v)[0]

    def load_v_frag_fp8(self, pks, dc, buf_off):
        """Load a V fragment (FP8) from LDS for the PV MFMA."""
        d_col = dc * self.t.MFMA_M + self.lane32
        kv_k_start = fx.Int64(pks * self.t.MFMA_K_FP8) + self.klane * (self.t.MFMA_K_FP8 // 2)
        v_off = buf_off + d_col * self.t.V_STRIDE + kv_k_start
        v8i8 = _lds_vec_load(self.v8i8_type, self.lds_v, v_off)
        v_i64v = fx.Vector(v8i8).bitcast(fx.Int64)
        return fx.Vector(v_i64v)[0]

    def f32_to_bf16_trunc(self, f32_raw):
        """Bitwise f32 → bf16 truncation (upper 16 bits)."""
        return _f32_to_bf16_trunc(f32_raw)

    def _buf_off(self, buf_idx_i32, stride_index):
        """Return byte offset zero or stride for the selected pipeline buffer."""
        is_one = fx.Int32(buf_idx_i32) == 1
        return fx.Int64(is_one.select(stride_index, self.ZERO_INDEX))

    def _gather_p_i64_from_pwords(self, p_words_2d, pks):
        """Gather 8 fp8 bytes (i64 B operand) for mfma_f32_32x32x16_fp8."""
        width_i32 = fx.Int32(64).ir_value()
        words = []
        for j in range_constexpr(2):
            selected = None
            for k in range_constexpr(2):
                k_pos = pks * self.t.MFMA_K_FP8 + k * 8 + j * 4
                st = k_pos // self.t.MFMA_N
                rem = k_pos % self.t.MFMA_N
                sk = rem // 4 % 2
                w_idx = rem // 8
                xor_off = fx.Int32((k ^ sk) * self.t.MFMA_M).ir_value()
                w_cand = fx.Int32(p_words_2d[st][w_idx]).shuffle_xor(xor_off, width_i32)
                if k == 0:
                    selected = w_cand
                else:
                    is_k = self.klane_i32 == k
                    selected = is_k.select(w_cand, selected)
            words.append(selected)
        return _i32_pair_to_i64(words[0], words[1])

    def _emit_qk_softmax_pquant(self, kv_block_start_arg, k_buf_off_arg, m_in, l_in):
        """Emit QK MFMA + scale + mask + online softmax + P-quant for one KV tile.
        Returns (m_new, l_new, corr, p_words_2d) as IR values.
        """
        kv_block_start_arg = fx.Int64(kv_block_start_arg)
        s_accs_loc = [_raw(self.c_zero_v16i32) for _ in range(self.N_SUBTILES)]
        for ks in range_constexpr(self.t.K_STEPS_QK):
            for st in range_constexpr(self.N_SUBTILES):
                kv_row_loc = self.lane32 + st * self.t.MFMA_N
                k_frag = self.load_k_frag(kv_row_loc, ks, k_buf_off_arg)
                s_accs_loc[st] = self.mfma_i32_k16(
                    self.v16i32_type, [k_frag, self.q_packs[ks], s_accs_loc[st], 0, 0, 0]
                )
        kv_tile_idx_loc = kv_block_start_arg // self.t.BLOCK_N
        max_kv_tile = self.num_k_blocks_per_head - 1
        kv_tile_safe = (kv_tile_idx_loc < self.num_k_blocks_per_head).select(kv_tile_idx_loc, max_kv_tile)
        k_descale_base_loc = (
            self.batch_idx * self.t.NUM_KV_HEADS * self.num_k_blocks_per_head
            + self.head_kv_idx * self.num_k_blocks_per_head
            + kv_tile_safe
        )
        k_ds_loc = fx.Float32(self._load_ptr_f32(self.kds_ptr, k_descale_base_loc))
        qk_scale_loc = self._fmul(self.q_ds, k_ds_loc)
        s_f32_loc = []
        if const_expr(self.t._DEFER_SCALE):
            for st in range_constexpr(self.N_SUBTILES):
                s_f32_vec = Vec(s_accs_loc[st]).to(fx.Float32)
                for elem in range_constexpr(self.ELEMS_PER_TILE):
                    s_f32_loc.append(s_f32_vec[elem])
        else:
            qk_scale_v16_loc = Vec.from_elements([qk_scale_loc], fx.Float32).broadcast_to(16)
            for st in range_constexpr(self.N_SUBTILES):
                s_f32_vec = Vec(s_accs_loc[st]).to(fx.Float32)
                s_scaled_vec = Vec(arith.mulf(s_f32_vec, qk_scale_v16_loc, fastmath=self.fm_fast))
                for elem in range_constexpr(self.ELEMS_PER_TILE):
                    s_f32_loc.append(s_scaled_vec[elem])
        if const_expr(self.t.use_bias):
            s_biased = list(s_f32_loc)
            bias_base = (
                (self.batch_idx * self.t.NUM_Q_HEADS + self.head_q_idx) * self.q_scale_num_blocks
                + self.q_scale_tile_idx
            ) * self.seq_len_k_v
            bias_col = kv_block_start_arg + self.lane
            bias_col_safe = (bias_col < self.seq_len_k_v).select(bias_col, fx.Int64(0))
            bias_lane = fx.Float32(self._load_ptr_f32(self.bias_ptr, bias_base + bias_col_safe))
            for st in range_constexpr(self.N_SUBTILES):
                for elem in range_constexpr(self.ELEMS_PER_TILE):
                    idx = st * self.ELEMS_PER_TILE + elem
                    msub = elem // 4
                    erem = elem % 4
                    bias_src_lane = fx.Int64(st * self.t.MFMA_N + msub * 8 + erem) + self.klane * 4
                    bias = self._bpermute_f32(bias_lane, bias_src_lane)
                    s_biased[idx] = self._fadd(s_biased[idx], bias)
            s_f32_loc = s_biased
        kv_start_i32_loc = fx.Int32(kv_block_start_arg)
        if const_expr(self.t.CAUSAL):
            s_named = list(s_f32_loc)
            for st in range_constexpr(self.N_SUBTILES):
                for elem in range_constexpr(self.ELEMS_PER_TILE):
                    idx = st * self.ELEMS_PER_TILE + elem
                    msub = elem // 4
                    erem = elem % 4
                    kv_col_i32 = (
                        kv_start_i32_loc
                        + fx.Int32(st * self.t.MFMA_N)
                        + fx.Int32(msub * 8)
                        + self.klane_off_i32
                        + fx.Int32(erem)
                    )
                    out_of_range = kv_col_i32 >= self.seq_len_k_i32
                    out_of_range = out_of_range | (kv_col_i32 > self.q_row_i32)
                    s_named[idx] = out_of_range.select(self.c_neg_inf, s_named[idx])
            s_f32_loc = s_named
        else:
            tile_end = kv_block_start_arg + self.t.BLOCK_N
            tile_oob = tile_end > self.seq_len_k_v
            s_base = s_f32_loc

            def _mask_oob_values():
                _llvm.inline_asm(None, [], "", "", has_side_effects=True)
                masked = []
                for st in range_constexpr(self.N_SUBTILES):
                    for elem in range_constexpr(self.ELEMS_PER_TILE):
                        idx = st * self.ELEMS_PER_TILE + elem
                        msub = elem // 4
                        erem = elem % 4
                        kv_col_i32 = (
                            kv_start_i32_loc
                            + fx.Int32(st * self.t.MFMA_N)
                            + fx.Int32(msub * 8)
                            + self.klane_off_i32
                            + fx.Int32(erem)
                        )
                        out_of_range = kv_col_i32 >= self.seq_len_k_i32
                        masked.append(_raw(out_of_range.select(self.c_neg_inf, s_base[idx])))
                return masked

            @flyc.jit
            def _apply_seq_mask():
                if tile_oob:
                    return _mask_oob_values()
                return [_raw(v) for v in s_base]

            s_f32_loc = list(_apply_seq_mask())
        local_max_l = s_f32_loc[0]
        for r in range_constexpr(self.NUM_S_VALS - 1):
            local_max_l = self._fmax(local_max_l, s_f32_loc[r + 1])
        row_max_l = self.row_max_reduce(local_max_l)
        if const_expr(self.t._DEFER_SCALE):
            row_max_l = self._fmul(row_max_l, qk_scale_loc)
        m_new_l = self._fmax(m_in, row_max_l)
        diff_m_l = self._fsub(m_in, m_new_l)
        corr_l = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_l))
        neg_max_l = self._fsub(self.c_zero_f, m_new_l)
        p_vals_l = []
        local_sum_l = _raw(self.c_zero_f)
        for r in range_constexpr(self.NUM_S_VALS):
            if const_expr(self.t._DEFER_SCALE):
                diff = self._ffma(s_f32_loc[r], qk_scale_loc, neg_max_l)
            else:
                diff = self._fadd(s_f32_loc[r], neg_max_l)
            p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
            p_vals_l.append(p)
            local_sum_l = self._fadd(local_sum_l, p)
        tile_sum_l = self.row_sum_reduce(local_sum_l)
        l_new_l = self._fadd(self._fmul(corr_l, l_in), tile_sum_l)
        c0_i32_l = fx.Int32(0).ir_value()
        p_words_l = []
        for st in range_constexpr(self.N_SUBTILES):
            sub = []
            for w in range_constexpr(4):
                s0 = _raw(p_vals_l[st * self.ELEMS_PER_TILE + w * 4 + 0])
                s1 = _raw(p_vals_l[st * self.ELEMS_PER_TILE + w * 4 + 1])
                s2 = _raw(p_vals_l[st * self.ELEMS_PER_TILE + w * 4 + 2])
                s3 = _raw(p_vals_l[st * self.ELEMS_PER_TILE + w * 4 + 3])
                packed = c0_i32_l
                packed = rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, packed, 0)
                packed = rocdl.cvt_pk_fp8_f32(T.i32, s2, s3, packed, 1)
                sub.append(packed)
            p_words_l.append(sub)
        return (m_new_l, l_new_l, corr_l, p_words_l)

    def _run_pv_mfma(self, p_words, cur_v_off, o_accs):
        for pks in range_constexpr(self.t.PV_K_STEPS):
            p_i64 = self._gather_p_i64_from_pwords(p_words, pks)
            for dc in range_constexpr(self.t.D_CHUNKS):
                v_i64 = self.load_v_frag_fp8(pks, dc, cur_v_off)
                o_accs[dc] = self.mfma_fp8_k16(self.v16f32_type, [v_i64, p_i64, o_accs[dc], 0, 0, 0])
        return o_accs

    def _lds_as3_base(self, lds_view):
        base_idx = _memref.extract_aligned_pointer_as_index(_llvm_value(lds_view))
        base_i64 = fx.Int64(base_idx).ir_value()
        return buffer_ops.create_llvm_ptr(base_i64, address_space=3)

    def setup(
        self,
        Q,
        K,
        V,
        O,  # noqa: E741
        Q_descale,
        K_descale,
        V_descale,
        Bias,
        batch_size,
        seq_len_q,
        seq_len_k,
        num_q_blocks,
    ):
        """Emit kernel prologue: pointers, LDS, tile indices, Q packs, constants."""
        self.fm_fast = arith.FastMathFlags.fast
        self.q_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(Q), fx.make_layout(1, 1))
        self.o_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(O), fx.make_layout(1, 1))
        qds_ptr = _extract_aligned_pointer(Q_descale)
        self.kds_ptr = _extract_aligned_pointer(K_descale)
        self.vds_ptr = _extract_aligned_pointer(V_descale)
        if const_expr(self.t.use_bias):
            self.bias_ptr = _extract_aligned_pointer(Bias)
        bs_v_for_rsrc = fx.Int64(batch_size)
        slk_v_for_rsrc = fx.Int64(seq_len_k)
        num_k_blocks_for_rsrc = (slk_v_for_rsrc + self.t.BLOCK_N - 1) // self.t.BLOCK_N
        kv_total_bytes = bs_v_for_rsrc * slk_v_for_rsrc * (self.t.NUM_KV_HEADS * self.t.HEAD_DIM)
        v_total_bytes = (
            bs_v_for_rsrc * (self.t.NUM_KV_HEADS * self.t.HEAD_DIM) * num_k_blocks_for_rsrc * self.t.BLOCK_N
            if self.t.V_TRANSPOSED
            else kv_total_bytes
        )
        self.k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=kv_total_bytes)
        self.v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=v_total_bytes)
        self.k_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(K), fx.make_layout(1, 1))
        self.load_atom_64_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int8)
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        register_as = int(fx.AddressSpace.Register)
        self.o_store_reg_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(2, 1), register_as)
        self.o_store_reg_lay = fx.make_layout(2, 1)
        self.o_store_reg = fx.memref_alloca(self.o_store_reg_ty, self.o_store_reg_lay)
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v16i32_type = Vec.make_type(16, fx.Int32)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v4f32_type = Vec.make_type(4, fx.Float32)
        self.v8i8_type = Vec.make_type(8, fx.Int8)
        self.seq_len_q_v = fx.Int64(seq_len_q)
        self.seq_len_k_v = fx.Int64(seq_len_k)
        base_ptr = self.allocator.get_base()
        self.lds = SmemPtr(base_ptr, self.t.lds_base_offset, T.i8, shape=(self.t.LDS_K_TOTAL_BYTES,)).get()
        self.lds_v = SmemPtr(
            base_ptr, self.t.lds_base_offset + self.t.LDS_K_TOTAL_BYTES, T.i8, shape=(self.t.LDS_V_TOTAL_BYTES,)
        ).get()
        block_id = fx.Int64(gpu.block_idx.x)
        self.tid = fx.Int64(gpu.thread_idx.x)
        self.wave_id = self.tid // self.t.WARP_SIZE
        self.lane = self.tid % self.t.WARP_SIZE
        self.lane32 = self.lane % self.t.MFMA_M
        self.klane = self.lane // self.t.MFMA_M
        wave_q_offset = self.wave_id * self.t.ROWS_PER_WAVE
        self.head_q_idx = block_id % self.t.NUM_Q_HEADS
        batch_q_tile_id = block_id // self.t.NUM_Q_HEADS
        num_q_tiles = (self.seq_len_q_v + self.t.BLOCK_M - 1) // self.t.BLOCK_M
        q_tile_raw = batch_q_tile_id % num_q_tiles
        self.batch_idx = batch_q_tile_id // num_q_tiles
        if const_expr(self.t._LPT_SCHED):
            q_tile_idx = num_q_tiles - 1 - q_tile_raw
        else:
            q_tile_idx = q_tile_raw
        q_start = q_tile_idx * self.t.BLOCK_M
        self.head_kv_idx = self.head_q_idx // self.t.GROUPS
        self.load_row_k_batch = self.tid // self.t.THREADS_PER_ROW_K
        self.load_lane_k = self.tid % self.t.THREADS_PER_ROW_K
        self.load_col_k_base = self.load_lane_k * self.t.VEC_WIDTH_K
        self.load_row_v_batch = self.tid // self.t.THREADS_PER_ROW_V
        load_lane_v = self.tid % self.t.THREADS_PER_ROW_V
        self.load_col_v_base = load_lane_v * self.t.VEC_WIDTH_V
        self.q_row = q_start + wave_q_offset + self.lane32
        self.q_row_i32 = fx.Int32(self.q_row)
        self.q_in_bounds = self.q_row < self.seq_len_q_v
        q_row_safe = self.q_in_bounds.select(self.q_row, fx.Int64(0))
        c_zero_i64 = fx.Int64(0)
        self.q_packs = []
        for ks in range_constexpr(self.t.K_STEPS_QK):
            q_col = fx.Int64(ks * self.t.MFMA_K_INT8) + self.klane * 8
            g_idx = self.q_global_idx(q_row_safe, q_col)
            half = fx.Int64(self._load_gm_i64(self.q_div, g_idx))
            self.q_packs.append(self.q_in_bounds.select(half, c_zero_i64))
        self.num_k_blocks_per_head = (self.seq_len_k_v + self.t.BLOCK_N - 1) // self.t.BLOCK_N
        self.q_scale_num_blocks = fx.Int64(num_q_blocks)
        q_scale_tile_raw = (q_start + wave_q_offset) // self.t.Q_SCALE_BLOCK_M
        self.q_scale_tile_idx = (q_scale_tile_raw < self.q_scale_num_blocks).select(
            q_scale_tile_raw, self.q_scale_num_blocks - 1
        )
        q_descale_base = (
            self.batch_idx * self.t.NUM_Q_HEADS * self.q_scale_num_blocks
            + self.head_q_idx * self.q_scale_num_blocks
            + self.q_scale_tile_idx
        )
        self.q_ds = fx.Float32(self._load_ptr_f32(qds_ptr, q_descale_base))
        self.v_descale_base = (self.batch_idx * self.t.NUM_KV_HEADS + self.head_kv_idx) * self.t.HEAD_DIM
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_zero_f = fx.Float32(0.0)
        self.c_one_f = fx.Float32(1.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        self.c_zero_v16i32 = Vec.filled(16, 0, fx.Int32)
        self.shuf32_i32 = fx.Int32(32)
        self.width_i32 = fx.Int32(self.t.WARP_SIZE)
        _q_end = q_start + self.t.BLOCK_M
        if const_expr(self.t.CAUSAL):
            self.kv_upper = (_q_end < self.seq_len_k_v).select(_q_end, self.seq_len_k_v)
        else:
            self.kv_upper = self.seq_len_k_v
        self.K_BUF1_OFF = fx.Int64(self.t.LDS_K_BYTES)
        self.V_BUF1_OFF = fx.Int64(self.t.LDS_V_BYTES)
        self.ZERO_INDEX = fx.Int64(0)
        self.N_SUBTILES = self.t.BLOCK_N // self.t.MFMA_N
        self.ELEMS_PER_TILE = 16
        self.NUM_S_VALS = self.N_SUBTILES * self.ELEMS_PER_TILE
        self.klane_i32 = fx.Int32(self.klane)
        self.klane_off_i32 = self.klane_i32 * fx.Int32(4)
        self.seq_len_k_i32 = fx.Int32(self.seq_len_k_v)

    def write_output(self, loop_results, _FINAL_OFF_L, _FINAL_OFF_O_ACCS):
        """Normalize accumulators by 1/l and apply V descale; store guarded by caller."""
        l_final = loop_results[_FINAL_OFF_L]
        o_finals = [loop_results[_FINAL_OFF_O_ACCS + dc] for dc in range_constexpr(self.t.D_CHUNKS)]
        inv_l = self._fdiv(self.c_one_f, l_final)
        inv_l_fp8 = _raw(inv_l)
        for dc in range_constexpr(self.t.D_CHUNKS):
            for msub in range_constexpr(4):
                d_col_base = fx.Int64(dc * self.t.MFMA_M + msub * 8) + self.klane * 4
                v_ds_vec = Vec(
                    self._load_ptr_f32_vec4(
                        self.vds_ptr, self.v_descale_base + dc * self.t.MFMA_M + msub * 8 + self.klane * 4
                    )
                )
                bf16_elems = []
                for erem in range_constexpr(4):
                    i_pos = msub * 4 + erem
                    v_ds = v_ds_vec[erem]
                    o_elem = fx.Vector(o_finals[dc])[i_pos].ir_value()
                    scale = self._fmul(inv_l_fp8, v_ds)
                    o_norm = self._fmul(o_elem, scale)
                    bf16_elems.append(self.f32_to_bf16_trunc(o_norm))
                o_vec = Vec.from_elements(bf16_elems, fx.BFloat16).ir_value()
                o_global = self.q_global_idx(self.q_row, d_col_base)
                self._store_gm_bf16x4(self.o_div, o_global, o_vec)


def _detect_gpu_name():
    """Return the GPU marketing name (e.g. 'AMD Instinct MI308X') or ''."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0) or ""
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "Marketing Name" in line and "MI" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _require_mi308x(gpu_arch):
    """Raise unless running on an AMD MI308X (gfx942); this kernel targets MI308X only."""
    gpu_name = _detect_gpu_name()
    is_gfx942 = isinstance(gpu_arch, str) and gpu_arch.startswith("gfx942")
    is_mi308x = "MI308" in gpu_name.upper()
    if not (is_gfx942 and is_mi308x):
        raise RuntimeError(
            f"SageAttention CDNA kernel supports only AMD MI308X (gfx942); "
            f"got arch={gpu_arch!r} name={gpu_name!r}. Other hardware is not supported."
        )


def _make_sage_launch_config(
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=False,
    block_m=None,
    block_n=None,
    flat_work_group_size=None,
    use_bias=False,
    v_transposed: bool = False,
    bias_block_m: int | None = None,
    path_tag: str = "auto",
):
    """Compute traits, allocator, tags, and launch dimensions for one kernel build."""
    gpu_arch = get_hip_arch()
    _require_mi308x(gpu_arch)
    WARP_SIZE = get_warp_size(gpu_arch)
    MFMA_M = 32
    MFMA_N = 32
    path_tag = f"{path_tag}_w256"
    MFMA_K_INT8 = 16
    MFMA_K_FP8 = 16
    ROWS_PER_WAVE = MFMA_M
    BLOCK_M = block_m if block_m is not None else 256
    BLOCK_N = block_n if block_n is not None else 128
    Q_SCALE_BLOCK_M = bias_block_m if bias_block_m is not None else BLOCK_M
    V_TRANSPOSED = bool(v_transposed)
    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    K_STEPS_QK = head_dim // MFMA_K_INT8
    D_CHUNK = MFMA_N
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEPS = BLOCK_N // MFMA_K_FP8
    assert head_dim % MFMA_K_INT8 == 0, f"head_dim {head_dim} must be divisible by {MFMA_K_INT8}"
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert BLOCK_N % MFMA_K_FP8 == 0
    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    GROUPS = num_q_heads // num_kv_heads
    STRIDE_TOKEN = NUM_Q_HEADS * HEAD_DIM
    KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM
    _dma_env = os.environ.get("SAGE_LDS_DMA", "auto")
    if _dma_env == "auto":
        _dma_env = "0"
    _dma_k = _dma_env in ("1", "k", "kv")
    _dma_any = _dma_k
    _cpr_k_host = head_dim // 16
    _cpr_k_is_p2 = _cpr_k_host > 0 and _cpr_k_host & _cpr_k_host - 1 == 0
    _dma_xor = _dma_any and _cpr_k_is_p2 and (os.environ.get("SAGE_LDS_DMA_XOR", "1") not in ("0", ""))
    if const_expr(_dma_any):
        path_tag = f"{path_tag}_dma{_dma_env}" + ("x" if _dma_xor else "")
    _defer_env = os.environ.get("SAGE_DEFER_SCALE", "auto")
    if _defer_env in ("0", ""):
        _DEFER_SCALE = False
    elif _defer_env == "force":
        _DEFER_SCALE = True
    else:
        _DEFER_SCALE = not CAUSAL or BLOCK_M <= 128
    if use_bias:
        _DEFER_SCALE = False
    if const_expr(_DEFER_SCALE):
        path_tag = f"{path_tag}_ds"
    _lpt_env = os.environ.get("SAGE_LPT_SCHED", "auto")
    _LPT_SCHED = _lpt_env not in ("0", "")
    if const_expr(_LPT_SCHED):
        path_tag = f"{path_tag}_lpt"
    path_tag = f"{path_tag}_{'causal' if CAUSAL else 'nc'}"
    _default_lds_pad = "8" if V_TRANSPOSED else "4"
    _LDS_PAD = int(os.environ.get("SAGE_LDS_PAD", _default_lds_pad))
    if _LDS_PAD < 0 or _LDS_PAD % 4 != 0:
        raise ValueError("SAGE_LDS_PAD must be a non-negative multiple of 4")
    path_tag = f"{path_tag}_pad{_LDS_PAD}"
    if const_expr(_dma_k):
        K_STRIDE = HEAD_DIM
    else:
        K_STRIDE = HEAD_DIM + _LDS_PAD
    V_STRIDE = BLOCK_N + _LDS_PAD
    VEC_WIDTH_K = 16
    VEC_WIDTH_V = 16
    THREADS_PER_ROW_K = HEAD_DIM // VEC_WIDTH_K
    THREADS_PER_ROW_V = HEAD_DIM // VEC_WIDTH_V
    ROWS_PER_BATCH_K = BLOCK_SIZE // THREADS_PER_ROW_K
    ROWS_PER_BATCH_V = BLOCK_SIZE // THREADS_PER_ROW_V
    if ROWS_PER_BATCH_K >= BLOCK_N:
        NUM_BATCHES_K = 1
        K_NEEDS_GUARD = ROWS_PER_BATCH_K > BLOCK_N
    else:
        NUM_BATCHES_K = BLOCK_N // ROWS_PER_BATCH_K
        K_NEEDS_GUARD = False
    if V_TRANSPOSED:
        NUM_BATCHES_V = 1
        V_NEEDS_GUARD = False
    elif ROWS_PER_BATCH_V >= BLOCK_N:
        NUM_BATCHES_V = 1
        V_NEEDS_GUARD = ROWS_PER_BATCH_V > BLOCK_N
    else:
        NUM_BATCHES_V = BLOCK_N // ROWS_PER_BATCH_V
        V_NEEDS_GUARD = False
    LDS_K_BYTES = BLOCK_N * K_STRIDE * 1
    LDS_V_BYTES = HEAD_DIM * V_STRIDE * 1
    _pipe_env = os.environ.get("SAGE_PIPE_STAGES", "2")
    NUM_PIPE_STAGES = 1 if _pipe_env in ("0", "1") else 2
    LDS_K_TOTAL_BYTES = LDS_K_BYTES * NUM_PIPE_STAGES
    LDS_V_TOTAL_BYTES = LDS_V_BYTES * NUM_PIPE_STAGES
    LDS_TOTAL_BYTES = LDS_K_TOTAL_BYTES + LDS_V_TOTAL_BYTES
    if NUM_PIPE_STAGES == 2 and LDS_TOTAL_BYTES > _LDS_CAP_BYTES:
        NUM_PIPE_STAGES = 1
        LDS_K_TOTAL_BYTES = LDS_K_BYTES
        LDS_V_TOTAL_BYTES = LDS_V_BYTES
        LDS_TOTAL_BYTES = LDS_K_TOTAL_BYTES + LDS_V_TOTAL_BYTES
    if NUM_PIPE_STAGES == 2:
        path_tag = f"{path_tag}_pipe2"
    else:
        path_tag = f"{path_tag}_pipe1"
    LDS_TOTAL_I8_ELEMS = LDS_TOTAL_BYTES
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name=f"sage_attn_cdna_smem_M{BLOCK_M}N{BLOCK_N}_{path_tag}"
    )
    lds_base_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_base_offset + LDS_TOTAL_I8_ELEMS

    t = _SageTraits(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=BLOCK_SIZE,
        CAUSAL=CAUSAL,
        D_CHUNKS=D_CHUNKS,
        GROUPS=GROUPS,
        HEAD_DIM=HEAD_DIM,
        KV_STRIDE_TOKEN=KV_STRIDE_TOKEN,
        K_NEEDS_GUARD=K_NEEDS_GUARD,
        K_STEPS_QK=K_STEPS_QK,
        K_STRIDE=K_STRIDE,
        LDS_K_BYTES=LDS_K_BYTES,
        LDS_K_TOTAL_BYTES=LDS_K_TOTAL_BYTES,
        LDS_V_BYTES=LDS_V_BYTES,
        LDS_V_TOTAL_BYTES=LDS_V_TOTAL_BYTES,
        MFMA_K_FP8=MFMA_K_FP8,
        MFMA_K_INT8=MFMA_K_INT8,
        MFMA_M=MFMA_M,
        MFMA_N=MFMA_N,
        NUM_BATCHES_K=NUM_BATCHES_K,
        NUM_BATCHES_V=NUM_BATCHES_V,
        NUM_KV_HEADS=NUM_KV_HEADS,
        NUM_PIPE_STAGES=NUM_PIPE_STAGES,
        NUM_Q_HEADS=NUM_Q_HEADS,
        PV_K_STEPS=PV_K_STEPS,
        Q_SCALE_BLOCK_M=Q_SCALE_BLOCK_M,
        ROWS_PER_BATCH_K=ROWS_PER_BATCH_K,
        ROWS_PER_BATCH_V=ROWS_PER_BATCH_V,
        ROWS_PER_WAVE=ROWS_PER_WAVE,
        STRIDE_TOKEN=STRIDE_TOKEN,
        THREADS_PER_ROW_K=THREADS_PER_ROW_K,
        THREADS_PER_ROW_V=THREADS_PER_ROW_V,
        VEC_WIDTH_K=VEC_WIDTH_K,
        VEC_WIDTH_V=VEC_WIDTH_V,
        V_NEEDS_GUARD=V_NEEDS_GUARD,
        V_STRIDE=V_STRIDE,
        V_TRANSPOSED=V_TRANSPOSED,
        WARP_SIZE=WARP_SIZE,
        _DEFER_SCALE=_DEFER_SCALE,
        _LPT_SCHED=_LPT_SCHED,
        _dma_any=_dma_any,
        _dma_k=_dma_k,
        _dma_xor=_dma_xor,
        lds_base_offset=lds_base_offset,
        use_bias=use_bias,
    )
    _cache_tag = (
        gpu_arch,
        BLOCK_M,
        BLOCK_N,
        CAUSAL,
        HEAD_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        _DEFER_SCALE,
        _LPT_SCHED,
        _dma_k,
        _dma_xor,
        V_TRANSPOSED,
        use_bias,
        NUM_PIPE_STAGES,
        path_tag,
    )
    return dict(
        t=t,
        allocator=allocator,
        path_tag=path_tag,
        cache_tag=_cache_tag,
        gpu_arch=gpu_arch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_Q_HEADS=NUM_Q_HEADS,
    )


class SageKernelContext:
    """Runtime kernel state container; helpers read/write through ``core``."""

    def __init__(self, core: "_SageKernel"):
        self.core = core
        self.t = core.t


class SageKLoader:
    def __init__(self, ctx: SageKernelContext):
        self.ctx = ctx

    def coop_load_k(self, tile_start, buf_off):
        return self.ctx.core.coop_load_k(tile_start, buf_off)


class SageVLoader:
    def __init__(self, ctx: SageKernelContext):
        self.ctx = ctx

    def coop_load_v(self, tile_start, buf_off):
        return self.ctx.core.coop_load_v(tile_start, buf_off)


class SageQKSoftmaxHelper:
    def __init__(self, ctx: SageKernelContext):
        self.ctx = ctx

    def emit_qk_softmax_pquant(self, kv_block_start, k_buf_off, m_in, l_in):
        return self.ctx.core._emit_qk_softmax_pquant(kv_block_start, k_buf_off, m_in, l_in)


class SagePVGemmHelper:
    def __init__(self, ctx: SageKernelContext):
        self.ctx = ctx

    def run_pv_mfma(self, p_words, cur_v_off, o_accs):
        return self.ctx.core._run_pv_mfma(p_words, cur_v_off, o_accs)


class SageStoreHelper:
    def __init__(self, ctx: SageKernelContext):
        self.ctx = ctx

    def write_output(self, loop_results, final_off_l, final_off_o_accs):
        return self.ctx.core.write_output(loop_results, final_off_l, final_off_o_accs)


class SageKernelFacade:
    """Facade wiring context, helpers, and the core emitter."""

    def __init__(self, t, allocator):
        self.core = _SageKernel(t, allocator)
        self.t = t
        self.ctx = SageKernelContext(self.core)
        self.k_loader = SageKLoader(self.ctx)
        self.v_loader = SageVLoader(self.ctx)
        self.qk_softmax = SageQKSoftmaxHelper(self.ctx)
        self.pv_gemm = SagePVGemmHelper(self.ctx)
        self.store = SageStoreHelper(self.ctx)

    def setup(self, *args, **kwargs):
        return self.core.setup(*args, **kwargs)

    @property
    def q_in_bounds(self):
        return self.core.q_in_bounds

    def __getattr__(self, name):
        return getattr(self.core, name)
