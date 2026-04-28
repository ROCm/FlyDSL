#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA4 (gfx120x, wave32).

4-warp LDS kernel inspired by Triton's 93 TFLOPS approach.

Architecture:
- 128x128x32 tiles, 4 warps (128 threads), 2x2 warp layout
- Each warp: 4 M-repeats x 4 N-repeats (64x64 output per warp)
- 2 K-steps per iteration (K=32, WMMA_K=16) -> 32 WMMAs per iter
- Double-buffered LDS (ping-pong): compute from buf[cur], prefetch to buf[1-cur]
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- K-padding on LDS stores for bank conflict avoidance

LDS layout (per buffer):
  A tile: 128 rows x (32+pad) cols x 2B, stored row-major
  B tile: 128 rows x (32+pad) cols x 2B, stored row-major
  Total per buffer: ~20KB, double-buffered: ~40KB

Pipeline: split GMEM load / LDS store with double buffering

Computes C[M,N] = A[M,K] @ B_T[N,K]^T
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr, const_expr
from flydsl.runtime.device import get_rocm_arch
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm


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
    reg_m=4,  # M-repeats per warp
    reg_n=4,  # N-repeats per warp
    reg_k=2,  # K-steps per tile (32/16=2)
    waves_m=2,  # warps in M dimension
    waves_n=2,  # warps in N dimension
    group_m=8,
    a_k_pad=8,  # K-padding for A in LDS (bank conflict avoidance)
    b_k_pad=8,  # K-padding for B in LDS
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 16*4*2 = 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
    BLOCK_K = WMMA_K * reg_k  # 16*2 = 32
    NUM_WAVES = waves_m * waves_n  # 2*2 = 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k >= 2 and reg_k % 2 == 0

    # Loading: each thread loads 8 bf16 elements per load (128 bits = buffer_load_b128)
    LOAD_VEC = 8
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4096/(128*8) = 4
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4
    TOTAL_LOADS = NUM_A_LOADS + NUM_B_LOADS  # 8

    # LDS layout with K-padding for bank conflict avoidance
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elements
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elements
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE  # 10240 elements = 20KB
    LDS_TOTAL = 2 * LDS_ONE_BUF  # 20480 elements = 40KB

    gpu_arch = get_rocm_arch()

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2, "Need at least 2 K-tiles for prefetch pipeline"

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    def _in_elem_ty():
        return T.bf16 if is_bf16 else T.f16

    def _out_elem_ty():
        return T.f32 if out_dtype == "f32" else T.bf16

    def _wmma_op(result_type, a_vec, b_vec, acc, v8i16_ty):
        if is_bf16:
            a_i16 = vector.bitcast(v8i16_ty, a_vec)
            b_i16 = vector.bitcast(v8i16_ty, b_vec)
            return rocdl.wmma_f32_16x16x16_bf16(result_type, a_i16, b_i16, arith.unwrap(acc)).result
        else:
            return rocdl.wmma_f32_16x16x16_f16(
                result_type, arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)
            ).result

    elem_bytes = 2  # bf16/f16 are both 2 bytes
    allocator = SmemAllocator(None, arch=gpu_arch)
    # Reserve LDS space (allocate_array needs an ir.Type, but we're outside MLIR
    # context here; manually compute offset instead).
    lds_byte_offset = allocator._align(allocator.ptr, elem_bytes)
    allocator.ptr = lds_byte_offset + LDS_TOTAL * elem_bytes

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
    ):
        in_ir_ty = T.bf16 if is_bf16 else T.f16
        v8_in_ty = T.vec(8, in_ir_ty)
        v4f32_ty = T.f32x4
        v8f32_ty = T.vec(8, T.f32)
        i16_ty = T.i16
        v8i16_ty = T.vec(8, i16_ty)

        from flydsl.utils.smem_allocator import SmemPtr

        lds_base = allocator.get_base()
        lds_view = SmemPtr(lds_base, lds_byte_offset, in_ir_ty, shape=(LDS_TOTAL,)).get()

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        c32 = fx.Index(32)
        c16 = fx.Index(16)
        c8 = fx.Index(8)
        c2 = fx.Index(2)
        wave_id = tid // c32
        lane = tid % c32
        lane16 = lane % c16
        klane = lane // c16
        base8 = klane * c8

        # Swizzle workgroup mapping for L2 locality
        effective_group_m = min(group_m, grid_m)
        c_grid_n = fx.Index(grid_n)
        c_group_m = fx.Index(effective_group_m)
        num_pid_in_group = c_group_m * c_grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * c_group_m
        group_size_m = c_group_m

        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % group_size_m)
        bid_n = pid_in_group // group_size_m

        # 2x2 warp layout
        c_wn = fx.Index(waves_n)
        wave_m = wave_id // c_wn
        wave_n = wave_id % c_wn

        tile_m0 = bid_m * fx.Index(BLOCK_M)
        tile_n0 = bid_n * fx.Index(BLOCK_N)

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(arg_bt, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)

        # ============================================================
        # Pre-compute GMEM offsets and LDS addresses
        # ============================================================
        a_lds_info = []
        for al in range_constexpr(NUM_A_LOADS):
            a_lin = tid * fx.Index(LOAD_VEC) + fx.Index(al * THREADS_PER_BLOCK * LOAD_VEC)
            a_load_row = a_lin // fx.Index(BLOCK_K)
            a_load_col = a_lin % fx.Index(BLOCK_K)
            lds_rel = a_load_row * fx.Index(BLOCK_K_PAD_A) + a_load_col
            g_row = tile_m0 + a_load_row
            a_lds_info.append((g_row, a_load_col, lds_rel))

        b_lds_info = []
        for bl in range_constexpr(NUM_B_LOADS):
            b_lin = tid * fx.Index(LOAD_VEC) + fx.Index(bl * THREADS_PER_BLOCK * LOAD_VEC)
            b_load_row = b_lin // fx.Index(BLOCK_K)
            b_load_col = b_lin % fx.Index(BLOCK_K)
            lds_rel = fx.Index(LDS_A_SIZE) + b_load_row * fx.Index(BLOCK_K_PAD_B) + b_load_col
            g_row = tile_n0 + b_load_row
            b_lds_info.append((g_row, b_load_col, lds_rel))

        # ============================================================
        # Phase 1: Issue GMEM loads (non-blocking), return raw data
        # ============================================================
        def _gmem_load(k_base):
            """Issue buffer_loads for A+B tile. Returns list of raw v4f32."""
            raw_data = []
            for al in range_constexpr(NUM_A_LOADS):
                g_row, a_load_col, _ = a_lds_info[al]
                g_col = k_base + a_load_col
                elem_off = g_row * fx.Index(K) + g_col
                f32_off = elem_off // c2
                a_raw = buffer_ops.buffer_load(a_rsrc, f32_off, vec_width=4, dtype=T.f32)
                raw_data.append(a_raw)

            for bl in range_constexpr(NUM_B_LOADS):
                g_row, b_load_col, _ = b_lds_info[bl]
                g_col = k_base + b_load_col
                elem_off = g_row * fx.Index(K) + g_col
                f32_off = elem_off // c2
                b_raw = buffer_ops.buffer_load(bt_rsrc, f32_off, vec_width=4, dtype=T.f32)
                raw_data.append(b_raw)

            return raw_data  # [a0, a1, a2, a3, b0, b1, b2, b3] -- 8 x v4f32

        # ============================================================
        # Phase 2: Store loaded data to LDS
        # ============================================================
        def _lds_store(raw_data, buf_offset):
            """Store previously loaded data to LDS at buf_offset."""
            for al in range_constexpr(NUM_A_LOADS):
                _, _, lds_rel = a_lds_info[al]
                a_vec = vector.bitcast(v8_in_ty, raw_data[al])
                lds_idx = buf_offset + lds_rel
                vector.store(a_vec, lds_view, [lds_idx])

            for bl in range_constexpr(NUM_B_LOADS):
                _, _, lds_rel = b_lds_info[bl]
                b_vec = vector.bitcast(v8_in_ty, raw_data[NUM_A_LOADS + bl])
                lds_idx = buf_offset + lds_rel
                vector.store(b_vec, lds_view, [lds_idx])

        # ============================================================
        # LDS read helpers -- row-major with K-padding
        # ============================================================
        def _load_a_from_lds(rk, buf_offset):
            """Load A WMMA operands from LDS for K-step rk."""
            vecs = []
            col_base = fx.Index(rk * WMMA_K) + base8
            for rm in range_constexpr(reg_m):
                row = wave_m * fx.Index(reg_m * WMMA_M) + fx.Index(rm * WMMA_M) + lane16
                lds_idx = buf_offset + row * fx.Index(BLOCK_K_PAD_A) + col_base
                a_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                vecs.append(a_raw)
            return vecs

        def _load_b_from_lds(rk, buf_offset):
            """Load B WMMA operands from LDS for K-step rk."""
            vecs = []
            col_base = fx.Index(rk * WMMA_K) + base8
            for rn in range_constexpr(reg_n):
                row = wave_n * fx.Index(reg_n * WMMA_N) + fx.Index(rn * WMMA_N) + lane16
                lds_idx = buf_offset + fx.Index(LDS_A_SIZE) + row * fx.Index(BLOCK_K_PAD_B) + col_base
                b_raw = vector.load_op(v8_in_ty, lds_view, [lds_idx])
                vecs.append(b_raw)
            return vecs

        def _barrier():
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                constraints="",
                has_side_effects=True,
            )

        def _do_compute_rk(accs_in, rk, buf_offset):
            """Compute all WMMAs for one K-step.

            Pattern: load all B first, then for each A load 1 A -> 4 WMMAs.
            This keeps register pressure low: only 4 B + 1 A + 16 accs live.
            """
            new_accs = list(accs_in)
            # Load all B operands for this K-step first
            b_vecs = _load_b_from_lds(rk, buf_offset)
            # Then load A one at a time and do reg_n WMMAs per A
            for rm in range_constexpr(reg_m):
                a_vec = _load_a_single_from_lds(rk, rm, buf_offset)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_op(
                        v8f32_ty,
                        a_vec,
                        b_vecs[rn],
                        new_accs[idx],
                        v8i16_ty,
                    )
            return new_accs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            """Load a single A WMMA operand from LDS for K-step rk, repeat rm_val."""
            col_base = fx.Index(rk * WMMA_K) + base8
            row = wave_m * fx.Index(reg_m * WMMA_M) + fx.Index(rm_val * WMMA_M) + lane16
            lds_idx = buf_offset + row * fx.Index(BLOCK_K_PAD_A) + col_base
            return vector.load_op(v8_in_ty, lds_view, [lds_idx])

        # ============================================================
        # Initialize accumulators -- 4x4 = 16 accumulators
        # ============================================================
        zero_acc = arith.constant_vector(0.0, v8f32_ty)
        accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

        # ============================================================
        # DOUBLE-BUFFERED PIPELINE WITH SPLIT LOAD/STORE
        # ============================================================

        c_lds_buf_stride = fx.Index(LDS_ONE_BUF)
        c_zero = fx.Index(0)

        # --- PROLOGUE ---
        prologue_data = _gmem_load(c_zero)
        _lds_store(prologue_data, c_zero)
        _barrier()

        # --- MAIN LOOP: kt=0..num_k_tiles-2 (SCF loop) ---
        # Loop-carried: accs (reg_m*reg_n accumulators)
        n_acc = reg_m * reg_n
        init_state = list(accs)

        for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
            s_accs = list(state[:n_acc])

            # Ping-pong: even iterations read buf0/write buf1, odd reversed
            read_off = iv % fx.Index(2) * c_lds_buf_stride
            write_off = (fx.Index(1) - iv % fx.Index(2)) * c_lds_buf_stride

            # 1. Issue GMEM loads for next tile (non-blocking)
            next_k = (iv + fx.Index(1)) * fx.Index(BLOCK_K)
            next_data = _gmem_load(next_k)

            # 2. Compute from current read buffer
            for rk in range_constexpr(reg_k):
                s_accs = _do_compute_rk(s_accs, rk, read_off)

            # 3. Store loaded data to write buffer
            _lds_store(next_data, write_off)

            # 4. Barrier
            _barrier()

            results = yield list(s_accs)

        accs = list(results[:n_acc])

        # --- EPILOGUE: Last tile in LDS ---
        # After num_k_tiles-1 iterations, last written buffer is the read buffer
        last_read_off = fx.Index((num_k_tiles - 1) % 2) * c_lds_buf_stride
        for rk in range_constexpr(reg_k):
            accs = _do_compute_rk(accs, rk, last_read_off)

        # ============================================================
        # Store results to GMEM
        # ============================================================
        c_layout_n = fx.Index(N)
        for rm in range_constexpr(reg_m):
            for rn in range_constexpr(reg_n):
                idx = rm * reg_n + rn
                wmma_m_off = wave_m * fx.Index(reg_m * WMMA_M) + fx.Index(rm * WMMA_M)
                wmma_n_off = wave_n * fx.Index(reg_n * WMMA_N) + fx.Index(rn * WMMA_N)
                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + fx.Index(si)
                    g_col = tile_n0 + wmma_n_off + lane16
                    val = vector.extract(accs[idx], static_position=[si], dynamic_position=[])
                    if const_expr(out_dtype == "bf16"):
                        val = arith.trunc_f(T.bf16, val)
                    elem_off = g_row * c_layout_n + g_col
                    buffer_ops.buffer_store(val, c_rsrc, elem_off)

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

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
