# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_8wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu#L1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def compile_fp8_gemm(*, M: int, N: int, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256):
    BLOCK_K = 128

    assert M >= 1
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_lds_cur0_alloc.ptr = a_lds_size
    A_lds_cur1_alloc.ptr = a_lds_size
    A_lds_next0_alloc.ptr = a_lds_size
    A_lds_next1_alloc.ptr = a_lds_size
    B_lds_cur0_alloc.ptr = b_lds_size
    B_lds_cur1_alloc.ptr = b_lds_size
    B_lds_next0_alloc.ptr = b_lds_size
    B_lds_next1_alloc.ptr = b_lds_size

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        MfmaAccum_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        Vec16_t = Vec.make_type(16, fx.Float8E4M3FN)
        SharedPtr_t = fx.PointerType.get(F8_IR_t, 2, 512)

        a_cur0 = SmemPtr(A_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_cur1 = SmemPtr(A_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next0 = SmemPtr(A_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next1 = SmemPtr(A_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()

        b_cur0 = SmemPtr(B_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_cur1 = SmemPtr(B_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next0 = SmemPtr(B_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next1 = SmemPtr(B_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        block_m = fx.block_idx.x // N_BLOCKS
        block_n = fx.block_idx.x % N_BLOCKS

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        def _make_fp8_buf_tensor(arg_i8):
            t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
            iter_i8 = fx.get_iter(t_i8)
            f8_buf_ptr_ty = fx.PointerType.get(
                elem_ty=F8_IR_t,
                address_space=TargetAddressSpace.BufferDesc,
                alignment=fx.PointerType(iter_i8.type).alignment,
            )
            iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
            return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))

        gA = _make_fp8_buf_tensor(A)
        gB = _make_fp8_buf_tensor(B_T)
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False)
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False)
        A_rsrc = fx.logical_divide(gA, fx.make_layout(1, 1))
        B_rsrc = fx.logical_divide(gB, fx.make_layout(1, 1))
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            wave_offset = (wave_id // 2) * 16 * K
            row = ((wave_id % 2) * 64 + lane_id) // 8
            col = (lane_id % 8) * 16
            swz_row, swz_col = _swizzle_128(row, col)
            for round in range_constexpr(N_LDS_ROUNDS):
                offsets.append(wave_offset + (round * 64 + swz_row) * K + swz_col)
            return offsets

        g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        def _load_lds(gl_src_div, lds_dst, k_offset, gl_offsets, n_steps):
            def _lds_dst_at():
                base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
                offset_idx = base_idx + fx.Index(wave_id * 1024 + step * 8192)
                offset_i64 = arith_dialect.index_cast(fx.T.i64(), offset_idx)
                lds_ptr = fx.inttoptr(SharedPtr_t, offset_i64)
                return fx.make_view(lds_ptr, fx.make_layout(1, 1))

            for step in range_constexpr(n_steps):
                src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[step])))
                dst = _lds_dst_at()
                fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _pack_i32x4_i32x8(lo, hi):
            # Pack two i32x4 as one i32x8
            return lo.shuffle(hi, list(range(8)))

        def _load_a_rt(lds_src, wave_offset):
            frag = []
            for i in range_constexpr(N_TILES_A):
                halves = []
                for k_i in range_constexpr(2):
                    row = lane_id % 16
                    col = (lane_id // 16) * 16 + k_i * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _load_b_rt(lds_src, wave_offset):
            frag = []
            for i in range_constexpr(N_TILES_B):
                halves = []
                for k_i in range_constexpr(2):
                    row = lane_id % 16
                    col = (lane_id // 16) * 16 + k_i * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        reg_f32_4_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(4, 1), 3)
        reg_f32_1_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), 3)
        reg_bf16_1_ty = fx.MemRefType.get(fx.T.bf16(), fx.LayoutType.get(1, 1), 3)

        def _store_C_scaled(c_frag, base_row, base_col):
            def _load_scale_vec4(row):
                r = fx.memref_alloca(reg_f32_4_ty, fx.make_layout(4, 1))
                fx.copy(scale_atom_4, fx.slice(sa_div, (None, fx.Int32(row))), r)
                return Vec(fx.memref_load_vec(r))

            def _load_scale_scalar(col):
                r = fx.memref_alloca(reg_f32_1_ty, fx.make_layout(1, 1))
                fx.copy(scale_atom_1, fx.slice(sb_div, (None, fx.Int32(col))), r)
                return Vec(fx.memref_load_vec(r))[0]

            def _store_bf16(value_bf16, c_index):
                r = fx.memref_alloca(reg_bf16_1_ty, fx.make_layout(1, 1))
                fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), r)
                fx.copy(out_atom_1, r, fx.slice(c_div, (None, fx.Int32(c_index))))

            a_scales = [_load_scale_vec4(base_row + i * 16 + (lane_id // 16) * 4) for i in range_constexpr(N_TILES_A)]
            b_scales = [_load_scale_scalar(base_col + i * 16 + lane_id % 16) for i in range_constexpr(N_TILES_B)]
            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(4):
                        scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                        _store_bf16(scaled, (row + i) * N + col)

        def _c_idx(i, j):
            return i * N_TILES_B + j

        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

        def _mfma_ABt_all(a, b, c):
            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = fly_dialect.mma_atom_call_ssa(
                        [MfmaAccum_t], mma_atom, a[i], b[j], c[_c_idx(i, j)]
                    )
            return c

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        # 2x2 config of 4x2 (instead of 4x4 in 4wave) 16x16 sub-tiles
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        global_offsets = _compute_global_swizzle()

        _load_lds(B_rsrc, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_A)
        _load_lds(B_rsrc, b_cur1, B1_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_cur1, A1_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_A)

        if wave_m == 1:
            rocdl.s_barrier()

        _wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        _load_lds(B_rsrc, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_A)
        _load_lds(B_rsrc, b_next1, B1_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_B)

        _wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = _load_b_rt(b_cur0, wave_n_offset * BLOCK_K)
            a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
            _load_lds(A_rsrc, a_next1, A1_gl_offset + (k + 1) * BLOCK_K, global_offsets, N_LDS_STEPS_A)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
            _load_lds(B_rsrc, b_cur0, B0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_B)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
            _load_lds(A_rsrc, a_cur0, A0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_A)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            _load_lds(B_rsrc, b_cur1, B1_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_B)
            _wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        b0_frag = _load_b_rt(b_cur0, wave_n_offset * BLOCK_K)
        a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = _load_b_rt(b_next0, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
        _wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale and store back to gmem
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        _store_C_scaled(c00_frag, base_row + 0, base_col + 0)
        _store_C_scaled(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_C_scaled(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_C_scaled(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir

        A_lds_cur0_alloc.finalized = False
        A_lds_cur1_alloc.finalized = False
        A_lds_next0_alloc.finalized = False
        A_lds_next1_alloc.finalized = False
        B_lds_cur0_alloc.finalized = False
        B_lds_cur1_alloc.finalized = False
        B_lds_next0_alloc.finalized = False
        B_lds_next1_alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            A_lds_cur0_alloc.finalize()
            A_lds_cur1_alloc.finalize()
            A_lds_next0_alloc.finalize()
            A_lds_next1_alloc.finalize()
            B_lds_cur0_alloc.finalize()
            B_lds_cur1_alloc.finalize()
            B_lds_next0_alloc.finalize()
            B_lds_next1_alloc.finalize()
        grid_x = ((M + BLOCK_M - 1) // BLOCK_M) * (N // BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
