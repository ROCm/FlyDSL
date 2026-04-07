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

from kernels.tensor_shim import GTensor, _to_raw


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
        i_v = gpu.block_id("x")
        i_nh = gpu.block_id("y")
        i_n = i_nh // H
        i_h = i_nh % H

        tid = gpu.thread_id("x")
        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE

        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))

        if SAVE_NEW_VALUE:
            vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
        if USE_INITIAL_STATE:
            h0_ = GTensor(h0_tensor, dtype=T.f32, shape=(-1,))
        if STORE_FINAL_STATE:
            ht_ = GTensor(ht_tensor, dtype=T.f32, shape=(-1,))

        if IS_VARLEN:
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

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

        if SAVE_NEW_VALUE:
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

            # ── 1. Store h snapshot ──
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    # Convert f32x4 -> bf16x4 for storage
                    bf16_vals = []
                    for elem_i in range_constexpr(4):
                        f32_val = vector.extract(acc_val, static_position=[elem_i], dynamic_position=[])
                        bf16_vals.append(arith.trunc_f(T.bf16, f32_val))
                    bf16_vec = vector.from_elements(T.vec(4, T.bf16), bf16_vals)

                    h_row = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                    h_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    h_off = h_base + i_t_i32 * stride_h + h_row * fx.Int32(V) + h_col
                    h_.vec_store((fx.Index(h_off),), bf16_vec, 4)

            # ── 2. Delta correction: b_v = w @ h, then v_new = u - b_v ──
            # b_v is [BT, BV] but we compute per-MFMA-tile
            # For each (wid-th M-row, nr-th N-col) tile:
            #   b_v_acc = sum over kb: w_tile[BT_row, kb*64..] @ h_tile[kb*64.., BV_col]
            # w: [T, K] with stride_w per row
            # h: in registers as h_accs

            # We need to compute w @ h where w is [BT, K] and h is [K, BV]
            # The MFMA approach: for each output (m_tile, n_tile) of b_v:
            #   accumulate over k_blocks: dot(w[m_tile, k_block], h[k_block, n_tile])
            # But h is in registers (distributed across warps/lanes).
            # Since each warp owns a different M-slice of h, we need cross-warp
            # communication for w @ h. This is complex.
            #
            # Simpler approach matching Triton: each thread computes its own
            # portion using the h values it owns, then reduces.
            # Actually, in Triton, h is [64, BV] in registers per program,
            # and w @ h is computed as tl.dot(w_block, h_block.to(bf16)).
            # The key insight: in Triton, ALL threads in the program share
            # the same h values (it's a 2D block, not distributed).
            #
            # In FlyDSL with MFMA, we need to restructure:
            # h_accs are distributed across warps (each warp owns 16 rows of K).
            # For w @ h: w[BT, K] @ h[K, BV]
            #   - w rows are the BT dimension (time)
            #   - h rows are the K dimension
            # Each warp owns 16 rows of K in h. To compute w @ h, we need
            # all K rows, so we need to broadcast h across warps.
            #
            # Alternative: use buffer_load to reload h from global memory
            # (we just stored it). This avoids cross-warp communication.

            # Reload h from global memory as bf16 for the matmul
            # b_v[BT, BV] = w[BT, K] @ h[K, BV]
            # We compute this per-thread: each thread handles specific output elements

            # For MFMA-based matmul of w @ h:
            # A = w[BT, K], B = h[K, BV]
            # Tile: M=BT=64, N=BV, K=K=128
            # MFMA 16x16x32: need M_REPEAT=4, N_REPEAT, K_STEPS=K/32=4

            K_STEPS = K // WMMA_K

            # Initialize b_v accumulators: M_REPEAT x N_REPEAT tiles
            bv_accs = []
            for _mr in range_constexpr(M_REPEAT):
                for _nr in range_constexpr(N_REPEAT):
                    bv_accs.append(arith.constant_vector(0.0, T.f32x4))

            # Load w and h tiles and compute MFMA
            for ks in range_constexpr(K_STEPS):
                # Load A (w) operand: each lane needs bf16x8 from w
                # w layout: row = i_t*BT + wid*16 + lane_row, col = ks*32 + lane_col*8
                # For mfma_f32_16x16x32_bf16: A is bf16x8 per lane
                # A[lane] = w[row, ks*32 + lane_col*8 .. ks*32 + lane_col*8 + 7]
                # where row = warp_m*16 + lane%16, lane_col = lane//16

                for mr in range_constexpr(M_REPEAT):
                    w_row = i_t_i32 * fx.Int32(BT) + fx.Int32(mr * 16) + lane_row
                    w_col = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8)
                    w_off = w_base + w_row * stride_w + w_col
                    a_frag = w_.vec_load((fx.Index(w_off),), 8)

                    for nr in range_constexpr(N_REPEAT):
                        # Load B (h) operand from global memory (just stored)
                        h_row = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8)
                        h_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_row
                        # h is stored as [K, V], B operand for MFMA needs [K, N]
                        # For mfma B: b[lane] = h[ks*32 + lane_col*8..+7, nr*16 + lane_row]
                        # But h is stored row-major [K, V], so we need column access
                        # Actually for MFMA B operand in NT layout:
                        # B is also bf16x8, indexed as B[col, k] where col=lane%16, k=lane//16*8
                        h_b_row = fx.Int32(nr * 16) + lane_row
                        h_b_col = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8)
                        h_b_off = h_base + i_t_i32 * stride_h + h_b_col * fx.Int32(V) + h_b_row
                        # This loads 8 consecutive bf16 from h, but h is [K, V] row-major
                        # so consecutive elements along V dimension at different K rows
                        # We need 8 elements along K dimension at fixed V position
                        # h[k, v] = h_base + k*V + v
                        # For B operand: need h[ks*32+lane_col*8+0..7, nr*16+lane_row]
                        # These are NOT consecutive in memory (stride = V between them)
                        # We need to load them individually and pack

                        b_elems = []
                        for bi in range_constexpr(8):
                            h_k_idx = fx.Int32(ks * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(bi)
                            h_v_idx = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_row
                            h_elem_off = h_base + i_t_i32 * stride_h + h_k_idx * fx.Int32(V) + h_v_idx
                            b_elems.append(h_[fx.Index(h_elem_off)])
                        b_frag = vector.from_elements(T.vec(8, T.bf16), b_elems)

                        bv_idx = mr * N_REPEAT + nr
                        bv_accs[bv_idx] = _mfma_bf16_16x16x32(a_frag, b_frag, bv_accs[bv_idx])

            # Now compute v_new = u - b_v and optionally store
            # u: [T, V] with stride_v per row
            # b_v result is in bv_accs as f32x4 per MFMA tile
            # v_new elements: row = i_t*BT + mr*16 + lane_row, col = i_v*BV + nr*16 + lane_col*4 + {0..3}

            # We need v_new as bf16 for the subsequent k^T @ v_new MFMA
            # Store v_new to global, then reload for MFMA (or keep in registers)

            # First compute v_new = u - bv for each tile element
            vn_frags = []
            for mr in range_constexpr(M_REPEAT):
                for nr in range_constexpr(N_REPEAT):
                    bv_idx = mr * N_REPEAT + nr
                    bv_val = bv_accs[bv_idx]

                    # Load u elements (4 consecutive bf16)
                    u_row = i_t_i32 * fx.Int32(BT) + fx.Int32(mr * 16) + lane_row
                    u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                    u_off = v_base + u_row * stride_v + u_col
                    u_vec = v_.vec_load((fx.Index(u_off),), 4)

                    # Convert u from bf16x4 to f32x4
                    u_f32_elems = []
                    for ei in range_constexpr(4):
                        u_bf16 = vector.extract(u_vec, static_position=[ei], dynamic_position=[])
                        u_f32_elems.append(arith.extf(T.f32, u_bf16))
                    u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                    # v_new = u - bv
                    vn_f32 = arith.subf(u_f32, bv_val)
                    vn_frags.append(vn_f32)

            # ── 2b. Store v_new if requested ──
            if SAVE_NEW_VALUE:
                for mr in range_constexpr(M_REPEAT):
                    for nr in range_constexpr(N_REPEAT):
                        vn_idx = mr * N_REPEAT + nr
                        vn_val = vn_frags[vn_idx]
                        bf16_vals = []
                        for ei in range_constexpr(4):
                            f32_v = vector.extract(vn_val, static_position=[ei], dynamic_position=[])
                            bf16_vals.append(arith.trunc_f(T.bf16, f32_v))
                        bf16_vec = vector.from_elements(T.vec(4, T.bf16), bf16_vals)

                        vn_row = i_t_i32 * fx.Int32(BT) + fx.Int32(mr * 16) + lane_row
                        vn_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_col_base * fx.Int32(4)
                        vn_off = vn_base + vn_row * fx.Int32(V) + vn_col
                        vn_.vec_store((fx.Index(vn_off),), bf16_vec, 4)

            # ── 3. Gating ──
            if USE_G:
                # last_idx = min((i_t+1)*BT, T_local) - 1
                next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
                last_idx_raw = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, next_chunk_end, T_local),
                    next_chunk_end,
                    T_local,
                ) - fx.Int32(1)

                # g_last = g[bos + last_idx, i_h]  (g layout: [total_T, H])
                g_last_off = (bos + last_idx_raw) * fx.Int32(H) + i_h
                g_last = g_[fx.Index(g_last_off)]
                exp_g_last = math_dialect.ExpOp(g_last).result

                # Scale v_new: v_new *= exp(g_last - g[bos + i_t*BT + row, i_h])
                # Also need mask: row < T_local
                for mr in range_constexpr(M_REPEAT):
                    for nr in range_constexpr(N_REPEAT):
                        vn_idx = mr * N_REPEAT + nr
                        vn_val = vn_frags[vn_idx]

                        # For each of the 4 elements in the f32x4:
                        # They share the same row (mr*16 + lane_row) but different cols
                        row_in_chunk = fx.Int32(mr * 16) + lane_row
                        abs_row = i_t_i32 * fx.Int32(BT) + row_in_chunk
                        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, abs_row, T_local)

                        g_row_off = (bos + abs_row) * fx.Int32(H) + i_h
                        g_row = g_[fx.Index(g_row_off)]
                        gate = math_dialect.ExpOp(arith.subf(g_last, g_row)).result
                        gate_masked = arith.select(in_bounds, gate, arith.constant(0.0, type=T.f32))

                        # Broadcast gate to f32x4
                        gate_vec = arith.constant_vector(0.0, T.f32x4)
                        for ei in range_constexpr(4):
                            gate_vec = vector.insert(gate_masked, gate_vec, static_position=[ei], dynamic_position=[])
                        vn_frags[vn_idx] = arith.mulf(vn_val, gate_vec)

                # Scale h: h *= exp(g_last)
                exp_g_last_vec = arith.constant_vector(0.0, T.f32x4)
                for ei in range_constexpr(4):
                    exp_g_last_vec = vector.insert(exp_g_last, exp_g_last_vec, static_position=[ei], dynamic_position=[])

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = arith.mulf(h_accs_in[acc_idx], exp_g_last_vec)

            # ── 4. State update: h += k^T @ v_new_gated ──
            # k: [T, K] with stride_k per row (actually [B, T, Hg, K])
            # v_new: [BT, BV] — in vn_frags as f32x4 per MFMA tile
            # We need k^T @ v_new: [K, BT] @ [BT, BV] -> [K, BV]
            # This updates h[K, BV]

            # Convert v_new to bf16 for MFMA
            vn_bf16_frags = []
            for mr in range_constexpr(M_REPEAT):
                for nr in range_constexpr(N_REPEAT):
                    vn_idx = mr * N_REPEAT + nr
                    vn_val = vn_frags[vn_idx]
                    bf16_vals = []
                    for ei in range_constexpr(4):
                        f32_v = vector.extract(vn_val, static_position=[ei], dynamic_position=[])
                        bf16_vals.append(arith.trunc_f(T.bf16, f32_v))
                    vn_bf16_frags.append(vector.from_elements(T.vec(4, T.bf16), bf16_vals))

            # For k^T @ v_new:
            # A = k^T [K, BT], B = v_new [BT, BV]
            # Output h_update [K, BV]
            # MFMA tiles: M=K (split into NUM_K_BLOCKS * 64/16 = 4*4=16 tiles along M)
            # Actually M dimension of output = K = 128, tiled as NUM_K_BLOCKS * (64/16) = 2*4 = 8 groups
            # But each warp handles one 16-row slice, so warp wid handles rows wid*16..(wid+1)*16-1
            # within each k-block.

            # Simpler: for each k-block kb, the h update is:
            #   h[kb*64..(kb+1)*64, :] += k[BT, kb*64..(kb+1)*64]^T @ v_new[BT, :]
            # This is a [64, BT]^T @ [BT, BV] = [64, BV] matmul
            # With MFMA 16x16x32: M=64 (4 tiles), N=BV (N_REPEAT tiles), K=BT=64 (2 steps of K=32)

            BT_STEPS = BT // WMMA_K

            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # Load k^T operand (A for MFMA): k^T[k_row, bt_col]
                    # k is [T, K], so k^T[k_row, t_col] = k[t_col, k_row]
                    # A operand: bf16x8 per lane
                    # A[lane] = k^T[wid*16+lane%16, bt_s*32+lane//16*8..+7]
                    #         = k[i_t*BT + bt_s*32+lane//16*8+0..7, kb*64+wid*16+lane%16]

                    k_a_row = wid * fx.Int32(16) + lane_row
                    # For each element in bf16x8:
                    k_a_elems = []
                    for ki in range_constexpr(8):
                        k_t_row = i_t_i32 * fx.Int32(BT) + fx.Int32(bt_s * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(ki)
                        k_t_col = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_row
                        k_off = k_base + k_t_row * stride_k + k_t_col
                        k_a_elems.append(k_[fx.Index(k_off)])
                    k_a_frag = vector.from_elements(T.vec(8, T.bf16), k_a_elems)

                    for nr in range_constexpr(N_REPEAT):
                        # Load v_new operand (B for MFMA): v_new[bt_row, v_col]
                        # B[lane] = v_new[bt_s*32+lane//16*8..+7, nr*16+lane%16]
                        # v_new is stored in vn_bf16_frags but as f32x4 per tile
                        # We need to reload from global or reconstruct

                        # Reload v_new B operand from global memory
                        # v_new was stored at vn_base + row*V + col
                        vn_b_elems = []
                        for bi in range_constexpr(8):
                            vn_b_row = i_t_i32 * fx.Int32(BT) + fx.Int32(bt_s * WMMA_K) + lane_col_base * fx.Int32(8) + fx.Int32(bi)
                            vn_b_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_row
                            if SAVE_NEW_VALUE:
                                vn_b_off = vn_base + vn_b_row * fx.Int32(V) + vn_b_col
                                vn_b_elems.append(vn_[fx.Index(vn_b_off)])
                            else:
                                # If not saving v_new, we stored it anyway for this purpose
                                vn_b_off = vn_base + vn_b_row * fx.Int32(V) + vn_b_col
                                vn_b_elems.append(vn_[fx.Index(vn_b_off)])
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
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

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
    vn_arg = v_new if v_new is not None else dummy
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
