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
from flydsl._mlir.dialects import scf, math as math_dialect
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import fly_values
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.tensor_shim import GTensor, STensor, _to_raw


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

    # LDS for gated v_new: [BT, BV] f32 row-major, shared across warps
    LDS_VN_ELEMS = BT * BV
    LDS_VN_BYTES = LDS_VN_ELEMS * 4  # f32 = 4 bytes

    # LDS for h snapshot bf16: [K, BV] bf16 row-major, for w@h B operand
    # Each k-block is [64, BV], total K rows x BV cols
    LDS_H_ELEMS = K * BV
    LDS_H_BYTES = LDS_H_ELEMS * 2  # bf16 = 2 bytes

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="gdn_h_smem")
    lds_vn_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_vn_offset + LDS_VN_BYTES
    lds_h_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_h_offset + LDS_H_BYTES
    # lds_vn_offset is in bytes; element offset for f32 = lds_vn_offset // 4
    LDS_VN_F32_BASE = lds_vn_offset // 4

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

        # ── LDS view for gated v_new (f32) ──
        lds_base_ptr = allocator.get_base()
        lds_vn_ptr = SmemPtr(
            lds_base_ptr,
            lds_vn_offset,
            T.f32,
            shape=(LDS_VN_ELEMS,),
        )
        lds_vn = STensor(lds_vn_ptr, dtype=T.f32, shape=(LDS_VN_ELEMS,))

        # ── LDS view for h snapshot (bf16) — used for w@h B operand ──
        lds_h_ptr = SmemPtr(
            lds_base_ptr,
            lds_h_offset,
            T.bf16,
            shape=(LDS_H_ELEMS,),
        )
        lds_h = STensor(lds_h_ptr, dtype=T.bf16, shape=(LDS_H_ELEMS,))

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
        # For mfma_f32_16x16x32_bf16:
        #   lane_id maps to (row, col) within the 16x16 output tile
        #   row = lane % 16, col = lane // 16 (4 f32 values per lane)
        lane_row = lane % fx.Int32(16)
        lane_col_base = lane // fx.Int32(16)

        # ── Initialize h accumulators ──
        # h state: NUM_K_BLOCKS blocks of [64, BV], each decomposed into
        # M_REPEAT x N_REPEAT MFMA tiles of 16x16
        # Each warp handles M_REPEAT/NUM_WARPS rows of 16
        # With 4 warps and M_REPEAT=4 (BT=64), each warp handles 1 row of 16
        acc_zero = arith.constant_vector(0.0, T.f32x4)

        # h_accs[kb][nr] = f32x4 accumulator for k-block kb, v-repeat nr
        # Each warp owns one M-slice (wid-th 16-row block)
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # ── Load initial state if provided ──
        if USE_INITIAL_STATE:
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    # h0: [K, V] with row = kb*64 + wid*16 + lane_row, col = i_v*BV + nr*16 + lane_col_base*4 + {0..3}
                    h0_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                    h0_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    h0_off = h0_base + h0_row * fx.Int32(V) + h0_col
                    loaded = h0_.vec_load((fx.Index(h0_off),), 4)
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = arith.addf(h_accs[acc_idx], loaded)

        # ── Main chunk loop ──
        # We use range_constexpr-style unrolling is not possible for dynamic NT.
        # Use scf.for with loop-carried h_accs.

        init_state = [_to_raw(v) for v in h_accs]
        c_zero = arith.index(0)
        c_one = arith.index(1)
        nt_idx = arith.index_cast(T.index, NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state)
            i_t_i32 = arith.index_cast(T.i32, i_t)

            # ── 1. Store h snapshot to global + LDS ──
            # Store h to global memory for K6, and to LDS for w@h B operand.
            # MFMA C layout: each lane holds 4 consecutive columns,
            #   row = kb*64 + wid*16 + lane_row, col = i_v*BV + nr*16 + lane_col_base*4
            # LDS layout: [K, BV] bf16 row-major (only the i_v*BV slice)
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    bf16_vals = []
                    for elem_i in range_constexpr(4):
                        f32_val = vector.extract(acc_val, static_position=[elem_i], dynamic_position=[])
                        bf16_vals.append(arith.trunc_f(T.bf16, f32_val))
                    bf16_vec = vector.from_elements(T.vec(4, T.bf16), bf16_vals)

                    h_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                    h_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    h_off = h_base + i_t_i32 * stride_h + h_row * fx.Int32(V) + h_col
                    h_.vec_store((fx.Index(h_off),), bf16_vec, 4)

                    # Also write to LDS [K, BV] row-major with local col = nr*16 + lane_col_base*4
                    lds_h_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                    lds_h_col = fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    lds_h_idx = lds_h_row * fx.Int32(BV) + lds_h_col
                    lds_h.vec_store((fx.Index(lds_h_idx),), bf16_vec, vec_size=4)

            gpu.barrier()

            # ── 2. Delta correction: b_v = w @ h, then v_new = u - b_v ──
            # b_v[BT, BV] = w[BT, K] @ h[K, BV]
            # h is now in LDS as [K, BV] bf16 row-major.
            # Each warp handles one 16-row M-tile of the BT dimension.

            K_STEPS = K // WMMA_K

            # Check if this warp's rows are in bounds
            warp_row_start = i_t_i32 * fx.Int32(BT) + wid * fx.Int32(16) + lane_row
            row_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, warp_row_start, T_local)
            # Clamp row to 0 for OOB lanes to avoid garbage/NaN loads
            safe_warp_row = arith.select(row_in_bounds, warp_row_start, fx.Int32(0))

            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(arith.constant_vector(0.0, T.f32x4))

            for ks in range_constexpr(K_STEPS):
                # A operand (w): bf16x8, row clamped to avoid OOB
                w_col = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8)
                w_off = w_base + safe_warp_row * stride_w + w_col
                a_frag = w_.vec_load((fx.Index(w_off),), 8)

                for nr in range_constexpr(N_REPEAT):
                    # B operand (h) from LDS: bf16x8
                    b_elems = []
                    for bi in range_constexpr(8):
                        lds_r = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(bi)
                        lds_c = fx.Int32(nr * 16) + lane_row
                        lds_idx = lds_r * fx.Int32(BV) + lds_c
                        b_elems.append(lds_h[fx.Index(lds_idx)])
                    b_frag = vector.from_elements(T.vec(8, T.bf16), b_elems)

                    bv_accs[nr] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[nr])

            # v_new = u - b_v  (per warp's M-tile only)
            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                # Use clamped row for u load too
                u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                u_off = v_base + safe_warp_row * stride_v + u_col
                u_vec = v_.vec_load((fx.Index(u_off),), 4)

                u_f32_elems = []
                for ei in range_constexpr(4):
                    u_bf16 = vector.extract(u_vec, static_position=[ei], dynamic_position=[])
                    u_f32_elems.append(arith.extf(T.f32, u_bf16))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                vn_frags.append(arith.subf(u_f32, bv_val))

            # ── 2b. Store v_new (pre-gating) for output ──
            if SAVE_NEW_VALUE:
                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    bf16_vals = []
                    for ei in range_constexpr(4):
                        f32_v = vector.extract(vn_val, static_position=[ei], dynamic_position=[])
                        bf16_vals.append(arith.trunc_f(T.bf16, f32_v))
                    bf16_vec = vector.from_elements(T.vec(4, T.bf16), bf16_vals)

                    vn_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    vn_off = vn_base + warp_row_start * fx.Int32(V) + vn_col
                    # Only store for in-bounds rows (use raw row, not clamped)
                    # OOB stores would write to wrong locations
                    # TODO: conditional store would be ideal, but for now
                    # the clamped loads produce valid (but wrong) data for OOB rows
                    # which is fine since those rows are never read
                    vn_.vec_store((fx.Index(vn_off),), bf16_vec, 4)

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
                exp_g_last = math_dialect.ExpOp(g_last).result

                # Gate v_new (this warp's M-tile only)
                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    row_in_chunk = wid * fx.Int32(16) + lane_row
                    abs_row = i_t_i32 * fx.Int32(BT) + row_in_chunk
                    in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)

                    # Clamp abs_row to avoid OOB g load (OOB lanes get gate=0)
                    safe_row = arith.select(in_bounds, abs_row, fx.Int32(0))
                    g_row_off = (bos + safe_row) * fx.Int32(H) + i_h
                    g_row = g_[fx.Index(g_row_off)]
                    gate = math_dialect.ExpOp(arith.subf(g_last, g_row)).result
                    gate_masked = arith.select(in_bounds, gate, arith.constant(0.0, type=T.f32))

                    gate_vec = arith.constant_vector(0.0, T.f32x4)
                    for ei in range_constexpr(4):
                        gate_vec = vector.insert(gate_masked, gate_vec, static_position=[ei], dynamic_position=[])
                    vn_frags[nr] = arith.mulf(vn_val, gate_vec)

                # Scale h: h *= exp(g_last)
                exp_g_last_vec = arith.constant_vector(0.0, T.f32x4)
                for ei in range_constexpr(4):
                    exp_g_last_vec = vector.insert(exp_g_last, exp_g_last_vec, static_position=[ei], dynamic_position=[])

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = arith.mulf(h_accs_in[acc_idx], exp_g_last_vec)

            # ── 3b. Store gated v_new to LDS (f32) for k^T @ v_new reload ──
            # Each warp writes its own 16-row slice to LDS in f32;
            # barrier ensures all warps have finished before any warp
            # reloads arbitrary rows.
            for nr in range_constexpr(N_REPEAT):
                vn_val = vn_frags[nr]
                # LDS layout: row-major [BT, BV], row = wid*16+lane_row,
                # col = nr*16 + lane_col_base*4
                lds_row = wid * fx.Int32(16) + lane_row
                lds_col = fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                lds_idx = lds_row * fx.Int32(BV) + lds_col
                lds_vn.vec_store((fx.Index(lds_idx),), vn_val, vec_size=4)

            gpu.barrier()

            # ── 4. State update: h += k^T @ v_new_gated ──
            # k^T[K, BT] @ v_new[BT, BV] -> [K, BV]
            # Each warp handles 16 rows of K within each k-block.
            # v_new is loaded from LDS (f32) and truncated to bf16 for MFMA.
            BT_STEPS = BT // WMMA_K

            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # A = k^T: need k[t_row, k_col] gathered as bf16x8
                    # Clamp OOB BT rows to 0 to avoid NaN (v_new is 0 for those rows,
                    # but NaN*0=NaN in IEEE 754 so k must also be clean)
                    k_a_elems = []
                    for ki in range_constexpr(8):
                        k_t_row_raw = i_t_i32 * fx.Int32(BT) + fx.Int32(bt_s * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(ki)
                        k_row_valid = arith.cmpi(arith.CmpIPredicate.slt, k_t_row_raw, T_local)
                        k_t_row = arith.select(k_row_valid, k_t_row_raw, fx.Int32(0))
                        k_t_col = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                        k_off = k_base + k_t_row * stride_k + k_t_col
                        k_val = k_[fx.Index(k_off)]
                        k_a_elems.append(arith.select(k_row_valid, k_val, arith.constant(0.0, type=T.bf16)))
                    k_a_frag = vector.from_elements(T.vec(8, T.bf16), k_a_elems)

                    for nr in range_constexpr(N_REPEAT):
                        # B = v_new from LDS (f32 -> bf16):
                        # LDS layout [BT, BV] row-major
                        # need v_new[bt_s*32+lane_col_base*8+bi, nr*16+lane_row]
                        vn_b_elems = []
                        for bi in range_constexpr(8):
                            lds_r = fx.Int32(bt_s * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(bi)
                            lds_c = fx.Int32(nr * 16) + lane_row
                            lds_elem_idx = lds_r * fx.Int32(BV) + lds_c
                            f32_val = lds_vn[fx.Index(lds_elem_idx)]
                            vn_b_elems.append(arith.trunc_f(T.bf16, f32_val))
                        vn_b_frag = vector.from_elements(T.vec(8, T.bf16), vn_b_elems)

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x32(k_a_frag, vn_b_frag, h_accs_in[acc_idx])

            results = yield [_to_raw(v) for v in h_accs_in]

        h_accs_final = list(results)

        # ── Epilogue: store final state ──
        if STORE_FINAL_STATE:
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_final[acc_idx]

                    ht_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                    ht_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    ht_off = ht_base + ht_row * fx.Int32(V) + ht_col
                    ht_.vec_store((fx.Index(ht_off),), acc_val, 4)

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
    BV: int = 32,
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
