#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""WMMA GEMM kernel for RDNA3 / RDNA3.5 (gfx11*, wave32).

Ported from rdna_f16_gemm.py (gfx120x). Same algorithm (4-warp double-
buffered LDS ping-pong, 128x128x32 tiles, swizzled grid mapping) but
adapted for the legacy v16-operand WMMA ABI used by RDNA3/RDNA3.5:

  * Input operands (A, B) are vector<16> instead of vector<8>; each
    lane carries 16 contiguous K-elements of one M (or N) row. Lanes
    0-15 carry distinct rows; lanes 16-31 carry duplicates of the same
    rows lanes 0-15 read. We just have all lanes do the LDS loads —
    duplicate loads are wasted bandwidth but simpler than a wave-half
    broadcast.
    TODO(perf): lanes 16-31 could ``ds_swizzle_b32`` XOR 16 broadcast
    from lanes 0-15 to halve LDS read bandwidth.

  * Accumulator (C/D) is still vector<8>, but the per-lane row mapping
    differs from gfx12: lane L holds D[2*si + (L/16)][L%16], i.e. even
    rows in lanes 0-15 and odd rows in lanes 16-31. The store-back loop
    uses ``g_row = base + 2*si + klane`` instead of the gfx12
    ``g_row = base + 8*klane + si``.

Computes C[M,N] = A[M,K] @ B_T[N,K]^T (same interface as
``rdna_f16_gemm.create_wmma_gemm_module``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=4,
    reg_n=4,
    reg_k=2,
    waves_m=2,
    waves_n=2,
    group_m=8,
    a_k_pad=8,
    b_k_pad=8,
):
    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_f16_gemm requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "Use rdna_f16_gemm.create_wmma_gemm_module on gfx120* (RDNA4)."
        )

    BLOCK_M = WMMA_M * reg_m * waves_m  # 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 128
    BLOCK_K = WMMA_K * reg_k  # 32
    NUM_WAVES = waves_m * waves_n  # 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k >= 2 and reg_k % 2 == 0

    LOAD_VEC = 8  # 8 bf16 = 128-bit GMEM/LDS load
    A_TILE_ELEMS = BLOCK_M * BLOCK_K
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)
    B_TILE_ELEMS = BLOCK_N * BLOCK_K
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)

    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE
    LDS_TOTAL = 2 * LDS_ONE_BUF

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2, "Need at least 2 K-tiles for prefetch pipeline"

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    def _wmma_op(a_vec, b_vec, acc):
        # On gfx11 the WMMA intrinsic takes v16 inputs (and v8 accumulator).
        if is_bf16:
            a_i16 = a_vec.bitcast(fx.Int16)
            b_i16 = b_vec.bitcast(fx.Int16)
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_i16, b_i16, acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    elem_dtype = fx.BFloat16 if is_bf16 else fx.Float16

    # ── Shared-memory storage for double-buffered A+B LDS tiles ──────────
    # One flat bf16/f16 array; v8 chunks are addressed by byte_offset // 2
    # (element-index = byte_offset / sizeof(elem)) inside the kernel.
    # 16-byte alignment so the underlying buffer is suitable for v8 loads
    # (8 * 2 bytes = 16 bytes).
    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
    ):
        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr  # i8-base aliased as elem_dtype*

        # ── v8 load/store helpers — element-indexed (v8_idx = byte_offset // 2 // 8) ──
        # Mirrors fp8_gemm_utils.S2RLoader._vec_load_16xf8: byte-offset the
        # pointer, recast to the element dtype, project into a v8 view.
        def _v8_load(v8_idx):
            elem_off = fx.Int32(v8_idx * 8)  # v8 chunks are 8 elements wide
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            typed_ptr = fx.recast_iter(elem_dtype, ptr_off)
            return fx.make_view(typed_ptr, fx.make_layout(8, 1)).load()

        def _v8_store(v8_idx, value):
            elem_off = fx.Int32(v8_idx * 8)
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            typed_ptr = fx.recast_iter(elem_dtype, ptr_off)
            fx.make_view(typed_ptr, fx.make_layout(8, 1)).store(value)

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane = tid % 32
        # On gfx11 the v16 ABI has lanes 16-31 mirror lanes 0-15, so the
        # M (or N) row is selected by ``lane % 16`` only. No klane shift
        # in the K dimension — each lane carries all 16 K-elements.
        lane16 = lane % 16
        klane = lane // 16  # used only for the gfx11 accumulator store-back

        # Swizzle workgroup mapping for L2 locality
        effective_group_m = min(group_m, grid_m)
        num_pid_in_group = effective_group_m * grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * effective_group_m
        group_size_m = effective_group_m

        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % group_size_m)
        bid_n = pid_in_group // group_size_m

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        tile_m0 = bid_m * BLOCK_M
        tile_n0 = bid_n * BLOCK_N

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(arg_bt, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)

        # ============================================================
        # Pre-compute GMEM offsets and LDS addresses (same as gfx12)
        # ============================================================
        a_lds_info = []
        for al in range_constexpr(NUM_A_LOADS):
            a_lin = tid * LOAD_VEC + (al * THREADS_PER_BLOCK * LOAD_VEC)
            a_load_row = a_lin // BLOCK_K
            a_load_col = a_lin % BLOCK_K
            lds_rel = a_load_row * BLOCK_K_PAD_A + a_load_col
            g_row = tile_m0 + a_load_row
            a_lds_info.append((g_row, a_load_col, lds_rel))

        b_lds_info = []
        for bl in range_constexpr(NUM_B_LOADS):
            b_lin = tid * LOAD_VEC + (bl * THREADS_PER_BLOCK * LOAD_VEC)
            b_load_row = b_lin // BLOCK_K
            b_load_col = b_lin % BLOCK_K
            lds_rel = LDS_A_SIZE + b_load_row * BLOCK_K_PAD_B + b_load_col
            g_row = tile_n0 + b_load_row
            b_lds_info.append((g_row, b_load_col, lds_rel))

        def _gmem_load(k_base):
            raw_data = []
            for al in range_constexpr(NUM_A_LOADS):
                g_row, a_load_col, _ = a_lds_info[al]
                g_col = k_base + a_load_col
                elem_off = g_row * K + g_col
                f32_off = elem_off // 2
                a_raw = buffer_ops.buffer_load(a_rsrc, f32_off, vec_width=4, dtype=fx.Float32)
                raw_data.append(a_raw)

            for bl in range_constexpr(NUM_B_LOADS):
                g_row, b_load_col, _ = b_lds_info[bl]
                g_col = k_base + b_load_col
                elem_off = g_row * K + g_col
                f32_off = elem_off // 2
                b_raw = buffer_ops.buffer_load(bt_rsrc, f32_off, vec_width=4, dtype=fx.Float32)
                raw_data.append(b_raw)

            return raw_data

        def _lds_store(raw_data, buf_offset):
            for al in range_constexpr(NUM_A_LOADS):
                _, _, lds_rel = a_lds_info[al]
                a_vec = raw_data[al].bitcast(fx.BFloat16 if is_bf16 else fx.Float16)
                lds_idx = buf_offset + lds_rel
                _v8_store(lds_idx // 8, a_vec)

            for bl in range_constexpr(NUM_B_LOADS):
                _, _, lds_rel = b_lds_info[bl]
                b_vec = raw_data[NUM_A_LOADS + bl].bitcast(fx.BFloat16 if is_bf16 else fx.Float16)
                lds_idx = buf_offset + lds_rel
                _v8_store(lds_idx // 8, b_vec)

        # ============================================================
        # LDS read helpers — v16 by concatenating two v8 loads
        # ============================================================
        # gfx11's v16 operand has element layout: lane L (L%16) carries 16
        # contiguous K-elements of row (lane%16). So per WMMA K-tile we
        # need 16 K-elements, stored as two contiguous v8 chunks at
        # offsets ``col_lo = 16*rk`` and ``col_hi = 16*rk + 8``.
        _concat16_mask = list(range(16))  # shuffle mask for v8 ++ v8 → v16

        def _load_b_from_lds(rk, buf_offset):
            vecs = []
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lds_idx_lo = buf_offset + LDS_A_SIZE + row * BLOCK_K_PAD_B + col_lo
                lds_idx_hi = buf_offset + LDS_A_SIZE + row * BLOCK_K_PAD_B + col_hi
                v_lo = _v8_load(lds_idx_lo // 8)
                v_hi = _v8_load(lds_idx_hi // 8)
                vecs.append(v_lo.shuffle(v_hi, _concat16_mask))
            return vecs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lds_idx_lo = buf_offset + row * BLOCK_K_PAD_A + col_lo
            lds_idx_hi = buf_offset + row * BLOCK_K_PAD_A + col_hi
            v_lo = _v8_load(lds_idx_lo // 8)
            v_hi = _v8_load(lds_idx_hi // 8)
            return v_lo.shuffle(v_hi, _concat16_mask)

        def _barrier():
            # gfx11 barrier — split signal/wait and s_wait_dscnt are gfx12+.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _do_compute_rk(accs_in, rk, buf_offset):
            new_accs = list(accs_in)
            b_vecs = _load_b_from_lds(rk, buf_offset)
            for rm in range_constexpr(reg_m):
                a_vec = _load_a_single_from_lds(rk, rm, buf_offset)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_op(
                        a_vec,
                        b_vecs[rn],
                        new_accs[idx],
                    )
            return new_accs

        zero_acc = fx.full(8, 0.0, fx.Float32)
        accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

        c_lds_buf_stride = LDS_ONE_BUF

        # --- PROLOGUE ---
        prologue_data = _gmem_load(0)
        _lds_store(prologue_data, 0)
        _barrier()

        n_acc = reg_m * reg_n
        init_state = list(accs)

        for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
            s_accs = list(state[:n_acc])

            read_off = iv % 2 * c_lds_buf_stride
            write_off = (1 - iv % 2) * c_lds_buf_stride

            next_k = (iv + 1) * BLOCK_K
            next_data = _gmem_load(next_k)

            for rk in range_constexpr(reg_k):
                s_accs = _do_compute_rk(s_accs, rk, read_off)

            _lds_store(next_data, write_off)
            _barrier()

            results = yield list(s_accs)

        accs = list(results[:n_acc])

        last_read_off = ((num_k_tiles - 1) % 2) * c_lds_buf_stride
        for rk in range_constexpr(reg_k):
            accs = _do_compute_rk(accs, rk, last_read_off)

        # ============================================================
        # Store results to GMEM (gfx11 layout: stride-2 rows)
        # ============================================================
        # gfx11 v8f32 acc layout: lane L holds D[2*si + (L/16)][L%16]
        # for si in 0..7 — i.e. lanes 0-15 carry even rows, lanes 16-31
        # carry odd rows of the same 16 columns.
        for rm in range_constexpr(reg_m):
            for rn in range_constexpr(reg_n):
                idx = rm * reg_n + rn
                wmma_m_off = wave_m * (reg_m * WMMA_M) + 16 * rm
                wmma_n_off = wave_n * (reg_n * WMMA_N) + 16 * rn
                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + 2 * si + klane
                    g_col = tile_n0 + wmma_n_off + lane16
                    val = accs[idx][si]
                    if const_expr(out_dtype == "bf16"):
                        val = val.to(fx.BFloat16)
                    elif const_expr(out_dtype == "f16"):
                        val = val.to(fx.Float16)
                    elem_off = g_row * N + g_col
                    buffer_ops.buffer_store(val, c_rsrc, elem_off)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream,
    ):
        c1 = 1
        total_blocks = grid_m * grid_n
        bk = THREADS_PER_BLOCK

        launcher = wmma_gemm_kernel(arg_c, arg_a, arg_bt)
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(bk, c1, c1),
            stream=stream,
        )

    return launch_gemm, BLOCK_M, BLOCK_N, BLOCK_K
