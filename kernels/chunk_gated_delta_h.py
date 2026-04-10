# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Gated Delta Net K5 hidden-state recurrence kernel using the @flyc.kernel API.

Mirrors the Triton `chunk_gated_delta_rule_fwd_kernel_h_opt3` from ATOM/FLA,
rewritten in FlyDSL for AMD GPUs (gfx942/gfx950).

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

import functools
import math

import torch
import triton

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl, buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as math_dialect, llvm as _llvm
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _llvm_exp2_f32(x):
    """Emit llvm.exp2.f32 intrinsic directly (maps to single v_exp_f32 on AMD)."""
    x_raw = _to_raw(x)
    return _llvm.call_intrinsic(
        ir.F32Type.get(), "llvm.exp2.f32", [x_raw], [], []
    )


def _fast_exp(x):
    """exp(x) via exp2(x * log2(e)) using the LLVM intrinsic."""
    log2e = arith.constant(_LOG2E, type=T.f32)
    return _llvm_exp2_f32(arith.mulf(x, log2e))


def _mfma_bf16_16x16x32(a_bf16x8, b_bf16x8, acc_f32x4):
    """Single mfma_f32_16x16x32_bf16 instruction."""
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0
    ).res


# ── Utility helpers ──────────────────────────────────────────────────────

def _prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]


@functools.lru_cache(maxsize=8)
def _prepare_chunk_offsets(cu_seqlens_id, chunk_size, device):
    cu_seqlens = torch._dynamo.utils.get_fake_value(cu_seqlens_id) if hasattr(torch._dynamo, 'utils') else None
    return None


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    lens = _prepare_lens(cu_seqlens)
    return torch.cat([
        cu_seqlens.new_tensor([0]),
        triton.cdiv(lens, chunk_size),
    ]).cumsum(-1)


# ── Compile the kernel ───────────────────────────────────────────────────

def compile_chunk_gated_delta_h(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
):
    """Compile the GDN K5 kernel.

    Returns a @flyc.jit function:
        launch_fn(k, v, w, v_new, g, h, h0, ht,
                  cu_seqlens, chunk_offsets,
                  T_val, T_flat, N_val, stream)
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_C_FRAG = 4

    M_REPEAT = BT // WMMA_M
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # ── LDS layout ──
    # w tile: [BT, 64] bf16 row-major, one K-block at a time
    LDS_W_STRIDE = 64  # elements per row
    LDS_W_ELEMS = BT * LDS_W_STRIDE
    LDS_W_BYTES = LDS_W_ELEMS * 2

    # k tile: [BT, 64] bf16 row-major (stored as [BT,64], read transposed via ds_read_b64_tr_b16)
    LDS_K_STRIDE = 64
    LDS_K_ELEMS = BT * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    # gated v_new: [BT, BV] bf16 row-major
    LDS_VN_STRIDE = BV
    LDS_VN_ELEMS = BT * LDS_VN_STRIDE
    LDS_VN_BYTES = LDS_VN_ELEMS * 2  # bf16

    # h snapshot: [K, BV] bf16 row-major
    LDS_H_STRIDE = BV
    LDS_H_ELEMS = K * LDS_H_STRIDE
    LDS_H_BYTES = LDS_H_ELEMS * 2

    # w and k are used in different phases, so they can share the same LDS region
    LDS_WK_BYTES = max(LDS_W_BYTES, LDS_K_BYTES)

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="gdn_h_smem")
    lds_wk_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_wk_offset + LDS_WK_BYTES
    lds_vn_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_vn_offset + LDS_VN_BYTES
    lds_h_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_h_offset + LDS_H_BYTES

    # Cooperative load parameters
    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    @flyc.kernel(name="chunk_gdn_fwd_h_opt3")
    def gdn_h_kernel(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = arith.index_cast(T.i32, gpu.block_id("x"))
        i_nh = arith.index_cast(T.i32, gpu.block_id("y"))
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))

        vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
        if USE_INITIAL_STATE:
            h0_ = GTensor(h0_tensor, dtype=T.f32, shape=(-1,))
        if STORE_FINAL_STATE:
            ht_ = GTensor(ht_tensor, dtype=T.f32, shape=(-1,))

        if IS_VARLEN:
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # ── LDS views ──
        lds_base_ptr = allocator.get_base()

        # w/k tile (shared region, bf16)
        lds_wk_ptr = SmemPtr(lds_base_ptr, lds_wk_offset, T.bf16, shape=(max(LDS_W_ELEMS, LDS_K_ELEMS),))
        lds_wk = STensor(lds_wk_ptr, dtype=T.bf16, shape=(max(LDS_W_ELEMS, LDS_K_ELEMS),))

        # gated v_new (bf16)
        lds_vn_ptr = SmemPtr(lds_base_ptr, lds_vn_offset, T.bf16, shape=(LDS_VN_ELEMS,))
        lds_vn = STensor(lds_vn_ptr, dtype=T.bf16, shape=(LDS_VN_ELEMS,))

        # h snapshot (bf16)
        lds_h_ptr = SmemPtr(lds_base_ptr, lds_h_offset, T.bf16, shape=(LDS_H_ELEMS,))
        lds_h = STensor(lds_h_ptr, dtype=T.bf16, shape=(LDS_H_ELEMS,))

        # ── Cooperative load decomposition ──
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # ── XOR swizzle: col ^ ((row & 7) << 3) at 8-element granularity for bf16 ──
        def _xor_swizzle(row, col):
            return col ^ ((row & fx.Int32(0x7)) << fx.Int32(3))

        def _xor_swizzle_idx(row, col):
            return col ^ ((row & arith.index(0x7)) << arith.index(3))

        # ── LDS vector read helper (generates ds_read_b128 for 8xbf16) ──
        v8bf16_type = T.vec(8, T.bf16)
        lds_wk_memref = lds_wk_ptr.get()

        def _lds_vec_read_bf16x8(elem_idx):
            return vector.load_op(v8bf16_type, lds_wk_memref, [elem_idx])

        # ── ds_read_b64_tr_b16 helper (gfx950) ──
        v4bf16_type = T.vec(4, T.bf16)

        def _ds_read_tr_bf16x4(lds_byte_offset):
            byte_idx = arith.index_cast(T.index, lds_byte_offset)
            byte_i64 = arith.index_cast(T.i64, byte_idx)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result

        # ds_read_b64_tr_b16 lane decomposition
        tr_k_group = (lane % fx.Int32(16)) // fx.Int32(4)
        tr_col_sub = lane % fx.Int32(4)
        tr_col_half = (lane % fx.Int32(32)) // fx.Int32(16)
        lane_div_32 = lane // fx.Int32(32)

        # ── Prologue: compute bos, T_local, NT, boh ──
        if IS_VARLEN:
            bos = cu_[fx.Index(i_n)]
            eos = cu_[fx.Index(i_n) + fx.Index(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Index(i_n)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = i_n * NT

        # ── Base pointer offsets (element counts) ──
        # h: [B, NT, H, K, V] — base = (boh*H + i_h) * K * V
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(K * V)
        stride_h = fx.Int32(H * K * V)

        # k: [B, T, Hg, K] — base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_base = (bos * fx.Int32(Hg) + i_h // fx.Int32(gqa_ratio)) * fx.Int32(K)
        stride_k = fx.Int32(Hg * K)

        if WU_CONTIGUOUS:
            if IS_VARLEN:
                v_base = (i_h * T_flat + bos) * fx.Int32(V)
                w_base = (i_h * T_flat + bos) * fx.Int32(K)
            else:
                v_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
                w_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(K)
            stride_v = fx.Int32(V)
            stride_w = fx.Int32(K)
        else:
            v_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            w_base = (bos * fx.Int32(H) + i_h) * fx.Int32(K)
            stride_v = fx.Int32(H * V)
            stride_w = fx.Int32(H * K)

        if IS_VARLEN:
            vn_base = (i_h * T_flat + bos) * fx.Int32(V)
        else:
            vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)

        if USE_INITIAL_STATE:
            h0_base = (i_nh * fx.Int32(K * V))
        if STORE_FINAL_STATE:
            ht_base = (i_nh * fx.Int32(K * V))

        # ── MFMA lane mapping for 16x16 tiles ──
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # index-typed versions for LDS addressing
        wid_idx = arith.index_cast(T.index, wid)
        lane_n_idx = arith.index_cast(T.index, lane_n)
        lane_m_base_idx = arith.index_cast(T.index, lane_m_base)

        # ── Initialize h accumulators ──
        acc_zero = arith.constant_vector(0.0, T.f32x4)

        # h_accs[kb][nr] = f32x4 accumulator for k-block kb, v-repeat nr
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # ── Load initial state if provided ──
        if USE_INITIAL_STATE:
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    h0_elems = []
                    for elem_i in range_constexpr(4):
                        h0_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        h0_off = h0_base + h0_row * fx.Int32(V) + h0_col
                        h0_elems.append(h0_[fx.Index(h0_off)])
                    loaded_vec = vector.from_elements(T.f32x4, h0_elems)
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = arith.addf(h_accs[acc_idx], loaded_vec)

        # ── Main chunk loop ──
        init_state = [_to_raw(v) for v in h_accs]
        c_zero = arith.index(0)
        c_one = arith.index(1)
        nt_idx = arith.index_cast(T.index, NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state)
            i_t_i32 = arith.index_cast(T.i32, i_t)

            # ── 1. Store h snapshot to global + LDS ──
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    h_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    lds_h_col = fx.Int32(nr * 16) + lane_n

                    for elem_i in range_constexpr(4):
                        f32_val = vector.extract(acc_val, static_position=[elem_i], dynamic_position=[])
                        bf16_val = arith.trunc_f(T.bf16, f32_val)

                        h_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        h_off = h_base + i_t_i32 * stride_h + h_row * fx.Int32(V) + h_col
                        h_[fx.Index(h_off)] = bf16_val

                        lds_h_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        lds_h_idx = lds_h_row * fx.Int32(BV) + lds_h_col
                        lds_h[fx.Index(lds_h_idx)] = bf16_val

            gpu.barrier()

            # ── 2. Delta correction: b_v = w @ h, then v_new = u - b_v ──
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(arith.constant_vector(0.0, T.f32x4))

            K_STEPS_PER_BLOCK = 64 // WMMA_K

            # ── Prefetch w[0] into registers ──
            w_prefetch = []
            w_prefetch_lds = []
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t_i32 * fx.Int32(BT) + row
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + fx.Int32(0 * 64) + load_col_base
                w_prefetch.append(w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))
                w_prefetch_lds.append(row * fx.Int32(LDS_W_STRIDE) + load_col_base)

            for kb in range_constexpr(NUM_K_BLOCKS):
                # ── Store prefetched w[kb] to LDS ──
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    lds_wk.vec_store((fx.Index(w_prefetch_lds[batch]),), w_prefetch[batch], LOAD_VEC_WIDTH)

                gpu.barrier()

                # ── MFMA: w (A from LDS_wk) × h (B from LDS_h) ──
                # Overlap: issue next K-block's global loads during MFMA
                if kb + 1 < NUM_K_BLOCKS:
                    w_prefetch = []
                    w_prefetch_lds = []
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                        safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                        g_off = w_base + safe_row * stride_w + fx.Int32((kb + 1) * 64) + load_col_base
                        w_prefetch.append(w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))
                        w_prefetch_lds.append(row * fx.Int32(LDS_W_STRIDE) + load_col_base)

                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    w_lds_row_idx = wid_idx * arith.index(16) + lane_n_idx
                    w_lds_col_idx = arith.index(ks * WMMA_K) + lane_m_base_idx * arith.index(8)
                    w_lds_idx = w_lds_row_idx * arith.index(LDS_W_STRIDE) + w_lds_col_idx
                    a_frag = _lds_vec_read_bf16x8(w_lds_idx)

                    global_ks = kb * K_STEPS_PER_BLOCK + ks

                    for nr in range_constexpr(N_REPEAT):
                        h_k_row = fx.Int32(global_ks * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group
                        h_v_col = fx.Int32(nr * 16) + tr_col_sub * fx.Int32(4)
                        h_lds_elem = h_k_row * fx.Int32(BV) + h_v_col
                        h_lds_byte = h_lds_elem * fx.Int32(2) + fx.Int32(lds_h_offset)

                        h_lo = _ds_read_tr_bf16x4(h_lds_byte)
                        h_hi = _ds_read_tr_bf16x4(h_lds_byte + fx.Int32(4 * BV * 2))
                        b_frag = vector.shuffle(h_lo, h_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                        bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])

                gpu.barrier()

            # v_new = u - b_v  (per warp's M-tile only)
            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bt_row_raw = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                    u_row_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, u_bt_row_raw, T_local)
                    safe_u_row = arith.select(u_row_in_bounds, u_bt_row_raw, fx.Int32(0))
                    u_off = v_base + safe_u_row * stride_v + u_col
                    u_bf16 = v_[fx.Index(u_off)]
                    u_f32_elems.append(arith.extf(T.f32, u_bf16))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                vn_frags.append(arith.subf(u_f32, bv_val))

            # ── 2b. Store v_new (pre-gating) for output ──
            if SAVE_NEW_VALUE:
                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    vn_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    for elem_i in range_constexpr(4):
                        vn_bt_row = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        vn_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, vn_bt_row, T_local)
                        _if_vn = scf.IfOp(vn_in_bounds)
                        with ir.InsertionPoint(_if_vn.then_block):
                            f32_v = vector.extract(vn_val, static_position=[elem_i], dynamic_position=[])
                            bf16_v = arith.trunc_f(T.bf16, f32_v)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            vn_[fx.Index(vn_off)] = bf16_v
                            scf.YieldOp([])

            # ── 3. Gating ──
            if USE_G:
                next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
                last_idx_raw = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, next_chunk_end, T_local),
                    next_chunk_end,
                    T_local,
                ) - fx.Int32(1)

                g_last_off = (bos + last_idx_raw) * fx.Int32(H) + i_h
                g_last = g_[fx.Index(g_last_off)]
                exp_g_last = _fast_exp(g_last)

                # Gate v_new: each f32x4 element corresponds to a different BT row
                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    gate_vec = arith.constant_vector(0.0, T.f32x4)
                    for elem_i in range_constexpr(4):
                        abs_row = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                        safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                        g_row_off = (bos + safe_row) * fx.Int32(H) + i_h
                        g_row = g_[fx.Index(g_row_off)]
                        gate = _fast_exp(arith.subf(g_last, g_row))
                        gate_masked = arith.select(in_bounds, gate, arith.constant(0.0, type=T.f32))
                        gate_vec = vector.insert(gate_masked, gate_vec, static_position=[elem_i], dynamic_position=[])
                    vn_frags[nr] = arith.mulf(vn_val, gate_vec)

                # Scale h: h *= exp(g_last)
                exp_g_last_vec = arith.constant_vector(0.0, T.f32x4)
                for ei in range_constexpr(4):
                    exp_g_last_vec = vector.insert(exp_g_last, exp_g_last_vec, static_position=[ei], dynamic_position=[])

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = arith.mulf(h_accs_in[acc_idx], exp_g_last_vec)

            # ── 3b. Store gated v_new to LDS (bf16) for k^T @ v_new reload ──
            for nr in range_constexpr(N_REPEAT):
                vn_val = vn_frags[nr]
                lds_col = fx.Int32(nr * 16) + lane_n
                for elem_i in range_constexpr(4):
                    f32_v = vector.extract(vn_val, static_position=[elem_i], dynamic_position=[])
                    bf16_v = arith.trunc_f(T.bf16, f32_v)
                    lds_row = wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                    lds_idx = lds_row * fx.Int32(LDS_VN_STRIDE) + lds_col
                    lds_vn[fx.Index(lds_idx)] = bf16_v

            gpu.barrier()

            # ── 4. State update: h += k^T @ v_new_gated ──
            BT_STEPS = BT // WMMA_K

            # ── Prefetch k[0] into registers ──
            k_prefetch = []
            k_prefetch_lds = []
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t_i32 * fx.Int32(BT) + row
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                g_off = k_base + safe_row * stride_k + fx.Int32(0 * 64) + load_col_base
                k_prefetch.append(k_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))
                k_prefetch_lds.append(row * fx.Int32(LDS_K_STRIDE) + load_col_base)

            for kb in range_constexpr(NUM_K_BLOCKS):
                # ── Store prefetched k[kb] to LDS ──
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    lds_wk.vec_store((fx.Index(k_prefetch_lds[batch]),), k_prefetch[batch], LOAD_VEC_WIDTH)

                gpu.barrier()

                # Issue next K-block's global loads during MFMA
                if kb + 1 < NUM_K_BLOCKS:
                    k_prefetch = []
                    k_prefetch_lds = []
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)
                        safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                        g_off = k_base + safe_row * stride_k + fx.Int32((kb + 1) * 64) + load_col_base
                        k_prefetch.append(k_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))
                        k_prefetch_lds.append(row * fx.Int32(LDS_K_STRIDE) + load_col_base)

                # ── MFMA: k^T (A from LDS_wk via ds_read_b64_tr_b16) × v_new (B from LDS_vn) ──
                for bt_s in range_constexpr(BT_STEPS):
                    k_col_tr = wid * fx.Int32(16) + tr_col_sub * fx.Int32(4)
                    bt_row_tr = fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group
                    k_lds_elem = bt_row_tr * fx.Int32(LDS_K_STRIDE) + k_col_tr
                    k_lds_byte = k_lds_elem * fx.Int32(2) + fx.Int32(lds_wk_offset)

                    k_lo = _ds_read_tr_bf16x4(k_lds_byte)
                    k_hi = _ds_read_tr_bf16x4(k_lds_byte + fx.Int32(4 * LDS_K_STRIDE * 2))
                    k_a_frag = vector.shuffle(k_lo, k_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                    for nr in range_constexpr(N_REPEAT):
                        # Read v_new B-operand from LDS_vn via ds_read_b64_tr_b16
                        vn_bt_row = fx.Int32(bt_s * WMMA_K) + lane_m_base * fx.Int32(8) + tr_k_group
                        vn_v_col = fx.Int32(nr * 16) + tr_col_sub * fx.Int32(4)
                        vn_lds_elem = vn_bt_row * fx.Int32(LDS_VN_STRIDE) + vn_v_col
                        vn_lds_byte = vn_lds_elem * fx.Int32(2) + fx.Int32(lds_vn_offset)

                        vn_lo = _ds_read_tr_bf16x4(vn_lds_byte)
                        vn_hi = _ds_read_tr_bf16x4(vn_lds_byte + fx.Int32(4 * BV * 2))
                        vn_b_frag = vector.shuffle(vn_lo, vn_hi, [0, 1, 2, 3, 4, 5, 6, 7])

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x32(k_a_frag, vn_b_frag, h_accs_in[acc_idx])

                gpu.barrier()

            results = yield [_to_raw(v) for v in h_accs_in]

        h_accs_final = list(results)

        # ── Epilogue: store final state ──
        if STORE_FINAL_STATE:
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_final[acc_idx]

                    ht_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    for elem_i in range_constexpr(4):
                        f32_val = vector.extract(acc_val, static_position=[elem_i], dynamic_position=[])
                        ht_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(elem_i)
                        ht_off = ht_base + ht_row * fx.Int32(V) + ht_col
                        ht_[fx.Index(ht_off)] = f32_val

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        launcher = gdn_h_kernel(
            k_tensor, v_tensor, w_tensor, v_new_tensor, g_tensor,
            h_tensor, h0_tensor, ht_tensor,
            cu_seqlens_tensor, chunk_offsets_tensor,
            T_val, T_flat, N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h


# ── Python wrapper (matches Triton interface) ────────────────────────────

_compiled_kernels = {}


def chunk_gated_delta_rule_fwd_h_flydsl(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    wu_contiguous: bool = True,
    BV: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 wrapper matching the Triton opt3 interface."""
    B, T, Hg, K = k.shape
    BT = chunk_size

    if wu_contiguous:
        H = w.shape[1]
        V = u.shape[-1]
        T_flat = w.shape[2]
    else:
        H = u.shape[-2]
        V = u.shape[-1]
        T_flat = w.shape[1]

    if BV <= 0:
        BV = min(V, 16)

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        NT = sum(triton.cdiv(int(l), BT) for l in lens.tolist())
        chunk_offsets = torch.cat([
            cu_seqlens.new_tensor([0]),
            triton.cdiv(lens, BT),
        ]).cumsum(-1).to(torch.int32)

    assert K <= 256

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new_buf = k.new_empty(B, H, T_flat, V, dtype=u.dtype)
    v_new = v_new_buf if save_new_value else None

    # Compile kernel with these specific parameters
    cache_key = (K, V, BT, BV, H, Hg,
                 g is not None, initial_state is not None,
                 output_final_state, save_new_value,
                 cu_seqlens is not None, wu_contiguous)

    if cache_key not in _compiled_kernels:
        _compiled_kernels[cache_key] = compile_chunk_gated_delta_h(
            K=K, V=V, BT=BT, BV=BV, H=H, Hg=Hg,
            USE_G=(g is not None),
            USE_INITIAL_STATE=(initial_state is not None),
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=save_new_value,
            IS_VARLEN=(cu_seqlens is not None),
            WU_CONTIGUOUS=wu_contiguous,
        )

    launch_fn = _compiled_kernels[cache_key]

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H

    # Prepare dummy tensors for optional params
    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    cu_arg = cu_seqlens.to(torch.int32) if cu_seqlens is not None else dummy.to(torch.int32)
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)

    stream = torch.cuda.current_stream()

    launch_fn(
        k, u, w, vn_arg, g_arg,
        h, h0_arg, ht_arg,
        cu_arg, co_arg,
        T, T_flat, N,
        grid_v, grid_nh,
        stream,
    )

    return h, v_new, final_state


__all__ = [
    "compile_chunk_gated_delta_h",
    "chunk_gated_delta_rule_fwd_h_flydsl",
]
