# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SageAttention kernel for AMD MI308X (CDNA gfx942)."""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_f32_32x32x16_fp8_fp8 as _ods_mfma_f32_32x32x16_fp8_fp8,
)
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_i32_32x32x16_i8 as _ods_mfma_i32_32x32x16_i8,
)
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_i32_32x32x32_i8 as _ods_mfma_i32_32x32x32_i8,
)
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_scale_f32_32x32x64_f8f6f4 as _ods_mfma_scale_f32_32x32x64_f8f6f4,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.attention.sage_attn_utils import (
    _LDS_CAP_BYTES,
    _extract_aligned_pointer,
    _llvm_value,
    _pointer_load,
    _pointer_store,
)
from kernels.attention.sage_attn_utils import _fadd as _u_fadd
from kernels.attention.sage_attn_utils import _ffma as _u_ffma
from kernels.attention.sage_attn_utils import _fmax as _u_fmax
from kernels.attention.sage_attn_utils import _fmul as _u_fmul
from kernels.attention.sage_attn_utils import _fsub as _u_fsub
from kernels.common.kernels_common import _if_else, _if_then, get_warp_size


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
        _GFX942_W256,
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
        self._GFX942_W256 = _GFX942_W256
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

    def mfma_i32_k32(self, result_type, operands):
        a, b, c = (operands[0], operands[1], operands[2])
        cbsz = operands[3] if len(operands) > 3 else 0
        abid = operands[4] if len(operands) > 4 else 0
        blgp = operands[5] if len(operands) > 5 else 0
        a_v = _llvm_value(a)
        b_v = _llvm_value(b)
        c_v = _llvm_value(c)
        if const_expr(self.t._GFX942_W256):
            return _ods_mfma_i32_32x32x16_i8(
                res=result_type, a=a_v, b=b_v, c=c_v, cbsz=cbsz, abid=abid, blgp=blgp
            ).result
        return _ods_mfma_i32_32x32x32_i8(res=result_type, a=a_v, b=b_v, c=c_v, cbsz=cbsz, abid=abid, blgp=blgp).result

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

    def mfma_fp8_k64(self, result_type, a, b, c, scale_a, scale_b):
        """rocdl.mfma.scale.f32.32x32x64.f8f6f4 (fp8*fp8) wrapper.
        a,b: vec<8xi32> per lane; c: vec<16xf32>; scale_a/b i32 e8m0 (127=1.0).
        """
        a_v = _llvm_value(a)
        b_v = _llvm_value(b)
        c_v = _llvm_value(c)
        return _ods_mfma_scale_f32_32x32x64_f8f6f4(
            res=result_type, a=a_v, b=b_v, c=c_v, cbsz=0, blgp=0, opselA=0, scaleA=scale_a, opselB=0, scaleB=scale_b
        ).result

    def _fadd(self, a, b):
        return _u_fadd(a, b, self.fm_fast)

    def _fsub(self, a, b):
        return _u_fsub(a, b, self.fm_fast)

    def _fmul(self, a, b):
        return _u_fmul(a, b, self.fm_fast)

    def _fmax(self, a, b):
        return _u_fmax(a, b, self.fm_fast)

    def _ffma(self, a, b, c):
        return _u_ffma(a, b, c, self.fm_fast)

    def q_global_idx(self, token_idx, col):
        token = self.batch_idx * self.seq_len_q_v + token_idx
        return token * self.t.STRIDE_TOKEN + self.head_q_idx * self.t.HEAD_DIM + col

    def kv_global_idx(self, token_idx, col):
        token = self.batch_idx * self.seq_len_k_v + token_idx
        return token * self.t.KV_STRIDE_TOKEN + self.head_kv_idx * self.t.HEAD_DIM + col

    def _load_ptr_i8_vec16(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
        return _pointer_load(self.v16i8_type, gep)

    def _load_ptr_i8_vec8(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
        return _pointer_load(self.v8i8_type, gep)

    def _load_ptr_i64(self, ptr, base_idx):
        """Load 8 contiguous bytes from `ptr+base_idx` as a scalar i64
        (8xi8 packed). Used for MFMA i8 operands which require packed i64.
        """
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.i8)
        return _pointer_load(T.i64, gep)

    def _load_ptr_f8_vec8(self, ptr, base_idx):
        return self._load_ptr_i8_vec8(ptr, base_idx)

    def _store_ptr_bf16(self, ptr, base_idx, val):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.bf16)
        _pointer_store(val, gep)

    def _load_ptr_f32(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
        return _pointer_load(T.f32, gep)

    def _load_ptr_f32_vec4(self, ptr, base_idx):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.f32)
        return _pointer_load(self.v4f32_type, gep)

    def _bpermute_f32(self, value, src_lane):
        value_i32 = arith.bitcast(T.i32, _raw(value))
        lane_byte_idx = fx.Int32(arith.unwrap(arith.index_cast(T.i32, _raw(src_lane * 4))))
        result_i32 = rocdl.ds_bpermute(T.i32, _raw(lane_byte_idx), value_i32)
        return fx.Float32(arith.bitcast(T.f32, result_i32))

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
        buf_off_i64 = arith.unwrap(arith.index_cast(T.i64, _raw(buf_off)))
        wave_byte = rocdl.readfirstlane(
            T.i64, arith.unwrap(arith.index_cast(T.i64, self.wave_id * fx.Index(self.t.WARP_SIZE * DMA_BYTES_K)))
        )
        base_ptr = self._lds_as3_base(self.lds)
        base_ptr = buffer_ops.get_element_ptr(base_ptr, buf_off_i64)
        base_ptr = buffer_ops.get_element_ptr(base_ptr, wave_byte)
        size_i32 = arith.constant(DMA_BYTES_K, type=T.i32)
        c0 = arith.constant(0, type=T.i32)
        aux1 = arith.constant(1, type=T.i32)
        _CPR_K = self.t.HEAD_DIM // self.t.VEC_WIDTH_K
        _XMASK_K = _CPR_K - 1
        for batch in range_constexpr(self.t.NUM_BATCHES_K):
            row_offset = batch * self.t.ROWS_PER_BATCH_K
            row_idx_raw = tile_start + self.load_row_k_batch + row_offset
            if const_expr(self.t._dma_xor):
                tile_row = self.load_row_k_batch + row_offset
                phys_chunk = self.load_lane_k
                g_chunk = ArithValue(phys_chunk) ^ ArithValue(tile_row) & fx.Index(_XMASK_K)
                g_col = fx.Index(g_chunk) * fx.Index(self.t.VEC_WIDTH_K)
                g_idx = self.kv_global_idx(row_idx_raw, g_col)
            else:
                g_idx = self.kv_global_idx(row_idx_raw, self.load_col_k_base)
            voff = arith.unwrap(arith.index_cast(T.i32, _raw(g_idx)))
            if const_expr(batch == 0):
                lds_ptr = base_ptr
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    base_ptr, static_byte_offset=batch * self.t.BLOCK_SIZE * DMA_BYTES_K
                )
            rocdl.raw_ptr_buffer_load_lds(self.k_rsrc, lds_ptr, size_i32, voff, c0, c0, aux1)

    def coop_load_k(self, tile_start, buf_off):
        if const_expr(self.t._dma_k):
            self._coop_load_k_dma(tile_start, buf_off)
            return
        for batch in range_constexpr(self.t.NUM_BATCHES_K):
            row_offset = batch * self.t.ROWS_PER_BATCH_K
            row_idx_raw = tile_start + self.load_row_k_batch + row_offset
            if const_expr(self.t.K_NEEDS_GUARD):
                row_valid = self.load_row_k_batch < fx.Index(self.t.BLOCK_N)
                do_load = row_valid
            else:
                do_load = True
            if do_load:
                g_idx = self.kv_global_idx(row_idx_raw, self.load_col_k_base)
                lds_row = self.load_row_k_batch + row_offset
                lds_idx = buf_off + lds_row * self.t.K_STRIDE + self.load_col_k_base
                g_dword_i32 = arith.unwrap(arith.index_cast(T.i32, _raw(g_idx >> fx.Index(2))))
                v4i32 = buffer_ops.buffer_load(self.k_rsrc, g_dword_i32, vec_width=4, dtype=T.i32)
                vec = vector.bitcast(self.v16i8_type, v4i32)
                Vec(vec).store(self.lds, [lds_idx])

    def coop_load_v(self, tile_start, buf_off):
        """Cooperatively load V (FP8 raw bytes) from global into LDS (gfx942 column-major)."""
        for batch in range_constexpr(self.t.NUM_BATCHES_V):
            row_offset = batch * self.t.ROWS_PER_BATCH_V
            row_idx_raw = tile_start + self.load_row_v_batch + row_offset
            if const_expr(self.t.V_NEEDS_GUARD):
                row_valid = self.load_row_v_batch < fx.Index(self.t.BLOCK_N)
                do_load = row_valid
            else:
                do_load = True
            if do_load:
                if const_expr(self.t.V_TRANSPOSED):
                    d_idx = self.tid % fx.Index(self.t.HEAD_DIM)
                    k_group = self.tid // fx.Index(self.t.HEAD_DIM)
                    tile_idx = tile_start // fx.Index(self.t.BLOCK_N)
                    g_byte_idx = (
                        (
                            (self.batch_idx * self.t.NUM_KV_HEADS + self.head_kv_idx) * self.num_k_blocks_per_head
                            + tile_idx
                        )
                        * fx.Index(self.t.HEAD_DIM)
                        + d_idx
                    ) * fx.Index(self.t.BLOCK_N) + k_group * fx.Index(16)
                    g_dword_i32 = arith.unwrap(arith.index_cast(T.i32, _raw(g_byte_idx >> fx.Index(2))))
                    v4i32 = buffer_ops.buffer_load(self.v_rsrc, g_dword_i32, vec_width=4, dtype=T.i32)
                    raw_v = vector.bitcast(self.v16i8_type, v4i32)
                    v_off = buf_off + d_idx * self.t.V_STRIDE + k_group * fx.Index(16)
                    Vec(raw_v).store(self.lds_v, [v_off])
                else:
                    g_idx = self.kv_global_idx(row_idx_raw, self.load_col_v_base)
                    g_dword_i32 = arith.unwrap(arith.index_cast(T.i32, _raw(g_idx >> fx.Index(2))))
                    v4i32 = buffer_ops.buffer_load(self.v_rsrc, g_dword_i32, vec_width=4, dtype=T.i32)
                    raw_v = vector.bitcast(self.v16i8_type, v4i32)
                    lds_col = self.load_row_v_batch + row_offset
                    for di in range_constexpr(self.t.VEC_WIDTH_V):
                        d_idx = self.load_col_v_base + di
                        v_off = buf_off + d_idx * self.t.V_STRIDE + lds_col
                        b_i8 = vector.extract(raw_v, static_position=[di], dynamic_position=[])
                        Vec.from_elements([b_i8], fx.Int8).store(self.lds_v, [v_off])

    def load_k_frag(self, kv_block_row, ks, buf_off):
        """Load K fragment from LDS for the QK MFMA."""
        if const_expr(self.t._dma_xor):
            _CPR_K = self.t.HEAD_DIM // self.t.VEC_WIDTH_K
            _XMASK_K = _CPR_K - 1
            if const_expr(self.t._GFX942_W256):
                g_chunk = fx.Index(ks)
                phys_chunk = ArithValue(g_chunk) ^ ArithValue(kv_block_row) & fx.Index(_XMASK_K)
                k_col = fx.Index(phys_chunk) * fx.Index(self.t.VEC_WIDTH_K) + self.klane * fx.Index(8)
            else:
                g_chunk = fx.Index(ks * 2) + self.klane
                phys_chunk = ArithValue(g_chunk) ^ ArithValue(kv_block_row) & fx.Index(_XMASK_K)
                k_col = fx.Index(phys_chunk) * fx.Index(self.t.VEC_WIDTH_K)
        elif const_expr(self.t._GFX942_W256):
            k_col = fx.Index(ks * self.t.MFMA_K_INT8) + self.klane * 8
        else:
            k_col = fx.Index(ks * self.t.MFMA_K_INT8) + self.klane * 16
        lds_idx = buf_off + fx.Index(kv_block_row * self.t.K_STRIDE) + k_col
        if const_expr(self.t._GFX942_W256):
            v8i8 = Vec.load(self.v8i8_type, self.lds, [lds_idx])
            k_i64v = vector.bitcast(Vec.make_type(1, fx.Int64), v8i8)
            return vector.extract(k_i64v, static_position=[0], dynamic_position=[])
        v16i8 = Vec.load(self.v16i8_type, self.lds, [lds_idx])
        return vector.bitcast(self.v4i32_type, v16i8)

    def load_v_frag_fp8(self, pks, dc, buf_off, iter_lane_addr_i32):
        """Load a V fragment (FP8) from LDS for the PV MFMA (gfx942 vector ds_read)."""
        d_col = dc * self.t.MFMA_M + self.lane32
        kv_k_start = fx.Index(pks * self.t.MFMA_K_FP8) + self.klane * fx.Index(self.t.MFMA_K_FP8 // 2)
        v_off = buf_off + d_col * self.t.V_STRIDE + kv_k_start
        if const_expr(self.t.MFMA_K_FP8 == 16):
            v8i8 = Vec.load(self.v8i8_type, self.lds_v, [v_off])
            v_i64v = vector.bitcast(Vec.make_type(1, fx.Int64), v8i8)
            return vector.extract(v_i64v, static_position=[0], dynamic_position=[])
        lo_v16i8 = Vec.load(self.v16i8_type, self.lds_v, [v_off])
        hi_v16i8 = Vec.load(self.v16i8_type, self.lds_v, [v_off + 16])
        lo_v4i32 = vector.bitcast(self.v4i32_type, lo_v16i8)
        hi_v4i32 = vector.bitcast(self.v4i32_type, hi_v16i8)
        elems = []
        for w in range_constexpr(4):
            elems.append(vector.extract(lo_v4i32, static_position=[w], dynamic_position=[]))
        for w in range_constexpr(4):
            elems.append(vector.extract(hi_v4i32, static_position=[w], dynamic_position=[]))
        return Vec.from_elements(elems, fx.Int32).ir_value()

    def f32_to_bf16_trunc(self, f32_raw):
        """Bitwise f32 → bf16 truncation (upper 16 bits)."""
        i32_val = arith.BitcastOp(T.i32, f32_raw).result
        i16_val = arith.TruncIOp(T.i16, arith.ShRUIOp(i32_val, arith.constant(16, type=T.i32)).result).result
        return arith.BitcastOp(T.bf16, i16_val).result

    def _buf_off(self, buf_idx_i32, stride_index):
        """buf_idx ∈ {0,1} → byte offset in {0, stride}, returned as fx.Index."""
        is_one = ArithValue(fx.Int32(buf_idx_i32) == fx.Int32(1))
        return fx.Index(is_one.select(stride_index, self.ZERO_INDEX))

    def _i32_pair_to_i64(self, lo, hi):
        lo64 = arith.extui(T.i64, _raw(lo))
        hi64 = arith.shli(arith.extui(T.i64, _raw(hi)), arith.constant(32, type=T.i64))
        return arith.ori(lo64, hi64)

    def _gather_p_i64_from_pwords(self, p_words_2d, pks):
        """Gather 8 fp8 bytes (i64 B operand) for mfma_f32_32x32x16_fp8."""
        width_i32 = arith.constant(64, type=T.i32)
        words = []
        for j in range_constexpr(2):
            selected = None
            for k in range_constexpr(2):
                k_pos = pks * self.t.MFMA_K_FP8 + k * 8 + j * 4
                st = k_pos // self.t.MFMA_N
                rem = k_pos % self.t.MFMA_N
                sk = rem // 4 % 2
                w_idx = rem // 8
                xor_off = arith.constant((k ^ sk) * self.t.MFMA_M, type=T.i32)
                w_cand = fx.Int32(p_words_2d[st][w_idx]).shuffle_xor(xor_off, width_i32)
                if k == 0:
                    selected = w_cand
                else:
                    is_k = arith.cmpi(arith.CmpIPredicate.eq, self.klane_i32, arith.constant(k, type=T.i32))
                    selected = ArithValue(is_k).select(w_cand, selected)
            words.append(selected)
        return self._i32_pair_to_i64(words[0], words[1])

    def _emit_qk_softmax_pquant(self, kv_block_start_arg, k_buf_off_arg, m_in, l_in):
        """Emit QK MFMA + scale + mask + online softmax + P-quant for one KV tile.
        Returns (m_new, l_new, corr, p_words_2d) as IR values.
        """
        s_accs_loc = [_raw(self.c_zero_v16i32) for _ in range(self.N_SUBTILES)]
        for ks in range_constexpr(self.t.K_STEPS_QK):
            for st in range_constexpr(self.N_SUBTILES):
                kv_row_loc = self.lane32 + st * self.t.MFMA_N
                k_frag = self.load_k_frag(kv_row_loc, ks, k_buf_off_arg)
                if const_expr(self.t._GFX942_W256):
                    s_accs_loc[st] = self.mfma_i32_k32(
                        self.v16i32_type, [k_frag, self.q_packs[ks], s_accs_loc[st], 0, 0, 0]
                    )
                else:
                    s_accs_loc[st] = self.mfma_i32_k32(
                        self.v16i32_type, [k_frag, self.q_packs[ks], s_accs_loc[st], 0, 0, 0]
                    )
        kv_tile_idx_loc = kv_block_start_arg // self.t.BLOCK_N
        max_kv_tile = self.num_k_blocks_per_head - fx.Index(1)
        kv_tile_safe = fx.Index(
            ArithValue(kv_tile_idx_loc < self.num_k_blocks_per_head).select(kv_tile_idx_loc, max_kv_tile)
        )
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
            bias_col_safe = fx.Index(ArithValue(bias_col < self.seq_len_k_v).select(bias_col, fx.Index(0)))
            bias_lane = fx.Float32(self._load_ptr_f32(self.bias_ptr, bias_base + bias_col_safe))
            for st in range_constexpr(self.N_SUBTILES):
                for elem in range_constexpr(self.ELEMS_PER_TILE):
                    idx = st * self.ELEMS_PER_TILE + elem
                    msub = elem // 4
                    erem = elem % 4
                    bias_src_lane = fx.Index(st * self.t.MFMA_N + msub * 8 + erem) + self.klane * 4
                    bias = self._bpermute_f32(bias_lane, bias_src_lane)
                    s_biased[idx] = self._fadd(s_biased[idx], bias)
            s_f32_loc = s_biased
        kv_start_i32_loc = fx.Int32(arith.unwrap(arith.index_cast(T.i32, _raw(kv_block_start_arg))))
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
                    out_of_range = ArithValue(kv_col_i32 >= self.seq_len_k_i32)
                    out_of_range = out_of_range | ArithValue(kv_col_i32 > self.q_row_i32)
                    s_named[idx] = out_of_range.select(self.c_neg_inf, s_named[idx])
            s_f32_loc = s_named
        else:
            tile_end = kv_block_start_arg + fx.Index(self.t.BLOCK_N)
            tile_oob_av = ArithValue(tile_end > self.seq_len_k_v)
            cond_i1 = _raw(tile_oob_av)
            f32_ty = ir.F32Type.get()
            result_types = [f32_ty] * self.NUM_S_VALS
            if_op = _scf.IfOp(cond_i1, result_types, has_else=True, loc=ir.Location.unknown())
            with _if_then(if_op):
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
                        out_of_range = ArithValue(kv_col_i32 >= self.seq_len_k_i32)
                        masked.append(_raw(out_of_range.select(self.c_neg_inf, s_f32_loc[idx])))
                _scf.YieldOp(masked)
            with _if_else(if_op):
                _scf.YieldOp([_raw(v) for v in s_f32_loc])
            s_f32_loc = list(if_op.results)
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
        c0_i32_l = arith.constant(0, type=T.i32)
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

    def _run_pv_mfma(self, p_words, cur_v_off, o_accs, v_iter_lane_addr_i32):
        if const_expr(self.t._GFX942_W256):
            for pks in range_constexpr(self.t.PV_K_STEPS):
                p_i64 = self._gather_p_i64_from_pwords(p_words, pks)
                for dc in range_constexpr(self.t.D_CHUNKS):
                    v_i64 = self.load_v_frag_fp8(pks, dc, cur_v_off, v_iter_lane_addr_i32)
                    o_accs[dc] = self.mfma_fp8_k16(self.v16f32_type, [v_i64, p_i64, o_accs[dc], 0, 0, 0])
        else:
            from flydsl._mlir.dialects._rocdl_ops_gen import permlane32_swap as _permlane32_swap_op

            _struct_ty_2xi32 = ir.Type.parse("!llvm.struct<(i32, i32)>")
            for pks in range_constexpr(self.t.PV_K_STEPS):
                v8_elems = []
                for w in range_constexpr(4):
                    a_w = _raw(p_words[pks * 2][w])
                    b_w = _raw(p_words[pks * 2 + 1][w])
                    swapped = _permlane32_swap_op(_struct_ty_2xi32, old=a_w, src=b_w, fi=False, bound_control=True)
                    lo_word = _llvm.extractvalue(T.i32, swapped, [0])
                    hi_word = _llvm.extractvalue(T.i32, swapped, [1])
                    v8_elems.append(lo_word)
                    v8_elems.append(hi_word)
                p_pack_v8i32 = Vec.from_elements(v8_elems, fx.Int32).ir_value()
                scale_127 = arith.constant(127, type=T.i32)
                for dc in range_constexpr(self.t.D_CHUNKS):
                    v_frag = self.load_v_frag_fp8(pks, dc, cur_v_off, v_iter_lane_addr_i32)
                    o_accs[dc] = self.mfma_fp8_k64(
                        self.v16f32_type, v_frag, p_pack_v8i32, o_accs[dc], scale_127, scale_127
                    )
        return o_accs

    def _lds_as3_base(self, lds_view):
        base_idx = _memref.extract_aligned_pointer_as_index(_llvm_value(lds_view))
        base_i64 = arith.unwrap(arith.index_cast(T.i64, base_idx))
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
        q_ptr = _extract_aligned_pointer(Q)
        self.o_ptr = _extract_aligned_pointer(O)
        qds_ptr = _extract_aligned_pointer(Q_descale)
        self.kds_ptr = _extract_aligned_pointer(K_descale)
        self.vds_ptr = _extract_aligned_pointer(V_descale)
        if const_expr(self.t.use_bias):
            self.bias_ptr = _extract_aligned_pointer(Bias)
        bs_v_for_rsrc = fx.Index(batch_size)
        slk_v_for_rsrc = fx.Index(seq_len_k)
        num_k_blocks_for_rsrc = (slk_v_for_rsrc + fx.Index(self.t.BLOCK_N - 1)) // fx.Index(self.t.BLOCK_N)
        kv_total_bytes = bs_v_for_rsrc * slk_v_for_rsrc * fx.Index(self.t.NUM_KV_HEADS * self.t.HEAD_DIM)
        v_total_bytes = (
            bs_v_for_rsrc
            * fx.Index(self.t.NUM_KV_HEADS * self.t.HEAD_DIM)
            * num_k_blocks_for_rsrc
            * fx.Index(self.t.BLOCK_N)
            if self.t.V_TRANSPOSED
            else kv_total_bytes
        )
        self.k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=kv_total_bytes)
        self.v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=v_total_bytes)
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v16i32_type = Vec.make_type(16, fx.Int32)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v4f32_type = Vec.make_type(4, fx.Float32)
        self.v8i8_type = Vec.make_type(8, fx.Int8)
        self.v16i8_type = Vec.make_type(16, fx.Int8)
        self.seq_len_q_v = fx.Index(seq_len_q)
        self.seq_len_k_v = fx.Index(seq_len_k)
        base_ptr = self.allocator.get_base()
        self.lds = SmemPtr(base_ptr, self.t.lds_base_offset, T.i8, shape=(self.t.LDS_K_TOTAL_BYTES,)).get()
        self.lds_v = SmemPtr(
            base_ptr, self.t.lds_base_offset + self.t.LDS_K_TOTAL_BYTES, T.i8, shape=(self.t.LDS_V_TOTAL_BYTES,)
        ).get()
        block_id = fx.Index(gpu.block_idx.x)
        self.tid = fx.Index(gpu.thread_idx.x)
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
            q_tile_idx = num_q_tiles - fx.Index(1) - q_tile_raw
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
        self.q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(self.q_row), _raw(self.seq_len_q_v))
        q_row_safe = fx.Index(ArithValue(self.q_in_bounds).select(self.q_row, fx.Index(0)))
        _zero_v4i32_ir = Vec.filled(4, 0, fx.Int32).ir_value()
        c_zero_i64 = arith.constant(0, type=T.i64)
        self.q_packs = []
        for ks in range_constexpr(self.t.K_STEPS_QK):
            if const_expr(self.t._GFX942_W256):
                q_col = fx.Index(ks * self.t.MFMA_K_INT8) + self.klane * 8
                g_idx = self.q_global_idx(q_row_safe, q_col)
                half = self._load_ptr_i64(q_ptr, g_idx)
                self.q_packs.append(ArithValue(self.q_in_bounds).select(half, c_zero_i64))
            else:
                q_col = fx.Index(ks * self.t.MFMA_K_INT8) + self.klane * 16
                g_idx = self.q_global_idx(q_row_safe, q_col)
                v16i8 = self._load_ptr_i8_vec16(q_ptr, g_idx)
                v4i32 = vector.bitcast(self.v4i32_type, v16i8)
                self.q_packs.append(ArithValue(self.q_in_bounds).select(v4i32, _zero_v4i32_ir))
        self.num_k_blocks_per_head = (self.seq_len_k_v + self.t.BLOCK_N - 1) // self.t.BLOCK_N
        self.q_scale_num_blocks = fx.Index(num_q_blocks)
        q_scale_tile_raw = (q_start + wave_q_offset) // self.t.Q_SCALE_BLOCK_M
        self.q_scale_tile_idx = ArithValue(q_scale_tile_raw < self.q_scale_num_blocks).select(
            q_scale_tile_raw, self.q_scale_num_blocks - fx.Index(1)
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
            self.kv_upper = fx.Index(ArithValue(_q_end < self.seq_len_k_v).select(_q_end, self.seq_len_k_v))
        else:
            self.kv_upper = self.seq_len_k_v
        self.K_BUF1_OFF = fx.Index(self.t.LDS_K_BYTES)
        self.V_BUF1_OFF = fx.Index(self.t.LDS_V_BYTES)
        self.ZERO_INDEX = fx.Index(0)
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
        inv_l = arith.divf(_raw(self.c_one_f), _raw(l_final), fastmath=self.fm_fast)
        inv_l_fp8 = _raw(inv_l)
        for dc in range_constexpr(self.t.D_CHUNKS):
            for msub in range_constexpr(4):
                d_col_base = fx.Index(dc * self.t.MFMA_M) + fx.Index(msub * 8) + self.klane * 4
                v_ds_vec = Vec(
                    self._load_ptr_f32_vec4(
                        self.vds_ptr,
                        self.v_descale_base + fx.Index(dc * self.t.MFMA_M) + fx.Index(msub * 8) + self.klane * 4,
                    )
                )
                bf16_elems = []
                for erem in range_constexpr(4):
                    i_pos = msub * 4 + erem
                    v_ds = v_ds_vec[erem]
                    o_elem = vector.extract(o_finals[dc], static_position=[i_pos], dynamic_position=[])
                    scale = arith.mulf(_raw(inv_l_fp8), _raw(v_ds), fastmath=self.fm_fast)
                    o_norm = arith.mulf(o_elem, scale, fastmath=self.fm_fast)
                    bf16_elems.append(self.f32_to_bf16_trunc(o_norm))
                o_vec = Vec.from_elements(bf16_elems, fx.BFloat16).ir_value()
                o_global = self.q_global_idx(self.q_row, d_col_base)
                self._store_ptr_bf16(self.o_ptr, o_global, o_vec)


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


def build_sage_attn_cdna_module(
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=False,
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    use_bias=False,
    use_fp8_pv=False,
    split_k: int = 1,
    gfx942_w256: bool | None = None,
    v_transposed: bool = False,
    bias_block_m: int | None = None,
):
    """Build FlyDSL sage attention kernel for AMD MI308X (CDNA gfx942).

    Uses Int8 MFMA for QK^T and BF16 MFMA for PV, matching triton sage v1 precision.
    """
    gpu_arch = get_hip_arch()
    _require_mi308x(gpu_arch)
    _IS_GFX942 = gpu_arch.startswith("gfx942")
    _GFX942_COMPACT = False
    _w256_env = os.environ.get("SAGE_GFX942_W256", "auto")
    if gfx942_w256 is None:
        if _w256_env == "auto":
            _GFX942_W256 = True
        elif _w256_env in ("compact", "c256"):
            _GFX942_W256 = False
            _GFX942_COMPACT = True
        else:
            _GFX942_W256 = _IS_GFX942 and _w256_env not in ("0", "")
            _GFX942_COMPACT = False
    else:
        _GFX942_W256 = _IS_GFX942 and bool(gfx942_w256)
        _GFX942_COMPACT = False
    if _IS_GFX942 and (not _GFX942_W256):
        raise ValueError("gfx942 requires the w256 SageAttention path")
    WARP_SIZE = get_warp_size(gpu_arch)
    MFMA_M = 32
    MFMA_N = 32
    if _GFX942_W256:
        path_tag = f"{path_tag}_w256"
        MFMA_K_INT8 = 16
        MFMA_K_FP8 = 16
    else:
        MFMA_K_INT8 = 32
        MFMA_K_FP8 = 64
    ROWS_PER_WAVE = MFMA_M
    BLOCK_M = block_m if block_m is not None else 256
    BLOCK_N = block_n if block_n is not None else 128
    Q_SCALE_BLOCK_M = bias_block_m if bias_block_m is not None else BLOCK_M
    SPLIT_K = max(1, int(split_k))
    IS_SPLIT_K = SPLIT_K > 1
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
    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)
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
    _default_lds_pad = "8" if _IS_GFX942 and V_TRANSPOSED else "4" if _IS_GFX942 else "16"
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
    NUM_PIPE_STAGES = 1 if _pipe_env in ("0", "1") or IS_SPLIT_K else 2
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
    if IS_SPLIT_K:
        path_tag = f"{path_tag}_sk{SPLIT_K}"
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
        _GFX942_W256=_GFX942_W256,
        _LPT_SCHED=_LPT_SCHED,
        _dma_any=_dma_any,
        _dma_k=_dma_k,
        _dma_xor=_dma_xor,
        lds_base_offset=lds_base_offset,
        use_bias=use_bias,
    )
    _impl = _SageKernel(t, allocator)
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
        _GFX942_W256,
        _dma_k,
        _dma_xor,
        V_TRANSPOSED,
        use_bias,
        use_fp8_pv,
        NUM_PIPE_STAGES,
        SPLIT_K,
        path_tag,
    )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def sage_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        Bias: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
    ):
        const_expr(_cache_tag)
        _impl.setup(Q, K, V, O, Q_descale, K_descale, V_descale, Bias, batch_size, seq_len_q, seq_len_k, num_q_blocks)
        if const_expr(_impl.t.NUM_PIPE_STAGES == 2):
            _impl.coop_load_k(_impl.ZERO_INDEX, _impl.ZERO_INDEX)
            _impl.coop_load_v(_impl.ZERO_INDEX, _impl.ZERO_INDEX)
            _impl.coop_load_k(fx.Index(_impl.t.BLOCK_N), _impl.K_BUF1_OFF)
            _impl.coop_load_v(fx.Index(_impl.t.BLOCK_N), _impl.V_BUF1_OFF)
            gpu.barrier()
            c0_i32_init = arith.constant(0, type=T.i32)
            init_args = [c0_i32_init, _raw(_impl.c_neg_inf), _raw(_impl.c_zero_f)]
            for _ in range_constexpr(_impl.t.D_CHUNKS):
                init_args.append(_raw(_impl.c_zero_v16f32))
            _OFF_CUR_BUF = 0
            _OFF_M = 1
            _OFF_L = 2
            _OFF_O_ACCS = 3
            loop_results = init_args
            for kv_block_start, inner_iter_args in range(0, _impl.kv_upper, _impl.t.BLOCK_N, init=init_args):
                cur_buf_i32 = inner_iter_args[_OFF_CUR_BUF]
                m_running = inner_iter_args[_OFF_M]
                l_running = inner_iter_args[_OFF_L]
                o_accs = [inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(_impl.t.D_CHUNKS)]
                cur_k_off = _impl._buf_off(cur_buf_i32, _impl.K_BUF1_OFF)
                cur_v_off = _impl._buf_off(cur_buf_i32, _impl.V_BUF1_OFF)
                next_buf_i32 = arith.XOrIOp(_raw(cur_buf_i32), arith.constant(1, type=T.i32)).result
                m_new, l_new, corr, p_words = _impl._emit_qk_softmax_pquant(
                    kv_block_start, cur_k_off, m_running, l_running
                )
                corr_vec16 = Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
                for dc in range_constexpr(_impl.t.D_CHUNKS):
                    o_accs[dc] = _impl._fmul(o_accs[dc], corr_vec16)
                kv_block_after_next = kv_block_start + fx.Index(2 * _impl.t.BLOCK_N)
                o_accs = _impl._run_pv_mfma(p_words, cur_v_off, o_accs, None)
                gpu.barrier()
                _impl.coop_load_k(kv_block_after_next, cur_k_off)
                _impl.coop_load_v(kv_block_after_next, cur_v_off)
                _yield_args = [next_buf_i32, _raw(m_new), _raw(l_new)]
                for dc in range_constexpr(_impl.t.D_CHUNKS):
                    _yield_args.append(o_accs[dc])
                loop_results = yield _yield_args
            _FINAL_OFF_L = _OFF_L
            _FINAL_OFF_O_ACCS = _OFF_O_ACCS
        else:
            init_args = [_raw(_impl.c_neg_inf), _raw(_impl.c_zero_f)]
            for _ in range_constexpr(_impl.t.D_CHUNKS):
                init_args.append(_raw(_impl.c_zero_v16f32))
            _OFF_M = 0
            _OFF_L = 1
            _OFF_O_ACCS = 2
            loop_results = init_args
            for kv_block_start, inner_iter_args in range(0, _impl.kv_upper, _impl.t.BLOCK_N, init=init_args):
                m_running = inner_iter_args[_OFF_M]
                l_running = inner_iter_args[_OFF_L]
                o_accs = [inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(_impl.t.D_CHUNKS)]
                _impl.coop_load_k(kv_block_start, _impl.ZERO_INDEX)
                _impl.coop_load_v(kv_block_start, _impl.ZERO_INDEX)
                gpu.barrier()
                m_new, l_new, corr, p_words = _impl._emit_qk_softmax_pquant(
                    kv_block_start, _impl.ZERO_INDEX, m_running, l_running
                )
                corr_vec16 = Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
                for dc in range_constexpr(_impl.t.D_CHUNKS):
                    o_accs[dc] = _impl._fmul(o_accs[dc], corr_vec16)
                o_accs = _impl._run_pv_mfma(p_words, _impl.ZERO_INDEX, o_accs, None)
                gpu.barrier()
                _yield_args = [_raw(m_new), _raw(l_new)]
                for dc in range_constexpr(_impl.t.D_CHUNKS):
                    _yield_args.append(o_accs[dc])
                loop_results = yield _yield_args
            _FINAL_OFF_L = _OFF_L
            _FINAL_OFF_O_ACCS = _OFF_O_ACCS
        if _impl.q_in_bounds:
            _impl.write_output(loop_results, _FINAL_OFF_L, _FINAL_OFF_O_ACCS)

    @flyc.jit
    def launch_sage_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        Bias: fx.Tensor,
        O_acc: fx.Tensor,
        M_out: fx.Tensor,
        L_out: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bs_idx = fx.Index(batch_size)
        slq_idx = fx.Index(seq_len_q)
        num_q_tiles = (slq_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_Q_HEADS
        launcher = sage_attn_kernel(
            Q, K, V, O, Q_descale, K_descale, V_descale, Bias, batch_size, seq_len_q, seq_len_k, num_q_blocks
        )
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get("denormal-fp-math-f32"), ir.StringAttr.get("preserve-sign,preserve-sign")]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_sage_attn(*args, **kwargs)

    def _compile(
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
        stream=None,  # noqa: E741
    ):
        with CompilationContext.compile_hints(_compile_hints):
            return flyc.compile(
                launch_sage_attn,
                Q,
                K,
                V,
                O,
                Q_descale,
                K_descale,
                V_descale,
                Bias,
                batch_size,
                seq_len_q,
                seq_len_k,
                num_q_blocks,
                fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch
