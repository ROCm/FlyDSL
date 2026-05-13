# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, and bf16 stores go through the layout API
(``fx.rocdl.make_buffer_tensor`` + ``fx.copy`` with ``BufferCopyLDS128b``
/ ``BufferCopy{16,32,128}b``). MFMAs use ``fly.mma_atom_call_ssa`` so
the chained Vec(4, f32) accumulator stays on AGPR. The XOR swizzle and
the 8-buffer LDS pipeline ping-pong are kept as direct arithmetic to
preserve the original kernel's interleaved-cluster scheduling.

Optional preshuffle for B. ``compile_fp8_gemm(..., b_preshuffled=True)``
consumes B in the same on-disk layout as
``preshuffle_gemm_v2``/``shuffle_weight((16, 16))`` so a single
offline-shuffled weight tensor can feed either kernel. The host
helper :func:`preshuffle_b` performs the permute. LDS layout, MFMA
fragment shape, and wave assignment are unchanged.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import fly as _fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as _memref_dialect
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TgtAS
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` into the layout consumed when
    ``compile_fp8_gemm(..., b_preshuffled=True)``.

    On-disk format matches ``preshuffle_gemm.shuffle_weight((16, 16))``
    / ``preshuffle_gemm_v2``: N-major outermost
    ``((nlane=16, n_outer=N/16), (kpack=16, klane=4, k0=K/64))``
    with strides ``((16, K*16), (1, 256, 1024))``. Each 64-K-col block
    of 16 N rows = 1 KB is contiguous; one MFMA_Scale_128 K-tile = 2
    consecutive k0 blocks = 2 KB.

    Equivalent to ``b.reshape(N/16, 16, K/64, 4, 16).permute(0, 2, 3, 1, 4)``.
    """
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    if num_wg <= SWIZZLE_THRESHOLD or num_wg % NUM_XCDS != 0:
        return _divmod(wgid, num_pid_n)

    intra_xcd, xcd = _divmod(wgid, NUM_XCDS)
    wgid = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod(wgid, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


def compile_fp8_gemm(
    *,
    M: int,
    N: int,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
    b_preshuffled: bool = False,
):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert BLOCK_M >= 64 and BLOCK_N >= 64
    assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K
    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

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

    @flyc.kernel
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

        a_cur0 = SmemPtr(A_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_cur1 = SmemPtr(A_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next0 = SmemPtr(A_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next1 = SmemPtr(A_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()

        b_cur0 = SmemPtr(B_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_cur1 = SmemPtr(B_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next0 = SmemPtr(B_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next1 = SmemPtr(B_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()

        _AS_SHARED = 2
        _shared_ptr_ty = fx.PointerType.get(F8_IR_t, _AS_SHARED, 512)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(M // BLOCK_M, N // BLOCK_N)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        A_K_STEP = BLOCK_K
        # N-major preshuffle matches preshuffle_gemm_v2 on-disk format:
        # ``((nlane=16, n_outer=N/16), (kpack=16, klane=4, k0=K/64))``
        # with strides ``((16, K*16), (1, 256, 1024))``. Tile base is
        # unchanged (``(tile_j*BLOCK_N // 16) * K*16 = tile_j*BLOCK_N*K``);
        # one K-tile of 128 K-cols = 2 k0 entries forward = 2 KB.
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K

        # A/B come in as torch.int8 (PyTorch fp8 view restriction); recast
        # the buffer-desc pointer's element type to fp8 so typed copy
        # atoms (BufferCopyLDS128b) accept them.
        def _make_fp8_buf_tensor(arg_i8):
            t_i8 = fx.rocdl.make_buffer_tensor(arg_i8)
            iter_i8 = fx.get_iter(t_i8)
            f8_buf_ptr_ty = fx.PointerType.get(
                elem_ty=F8_IR_t,
                address_space=_TgtAS.BufferDesc,
                alignment=fx.PointerType(iter_i8.type).alignment,
            )
            iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
            return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))

        gA = _make_fp8_buf_tensor(A)
        gB = _make_fp8_buf_tensor(B_T)
        gC = fx.rocdl.make_buffer_tensor(C)
        gSA = fx.rocdl.make_buffer_tensor(A_scale)
        gSB = fx.rocdl.make_buffer_tensor(B_scale)
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        # XOR bits 4..6 of the tile-local linear offset with bits 8..10.
        def _swizzle_128(row, col):
            offset = row * BLOCK_K + col
            swz = ((offset % (16 * BLOCK_K)) >> 8) << 4
            swizzled = offset ^ swz
            return swizzled // BLOCK_K, swizzled % BLOCK_K

        def _compute_global_swizzle(preshuffled):
            # Two thread→tile mappings:
            #   * Non-preshuffled (row-major B / A): ``lane//8 → row``,
            #     ``lane%8 → col_chunk``. 8 lanes share a row at adjacent
            #     16-byte col chunks → contiguous 128 B per lane-group in DRAM.
            #     LDS write at ``lane*16`` produces row-major-within-wave LDS.
            #     XOR swizzle reduces ds_read bank conflicts.
            #   * Preshuffled (preshuffle_gemm_v2 N-major layout for B,
            #     ``((nlane=16, n_outer=N/16), (kpack=16, klane=4, k0=K/64))``
            #     with strides ``((16, K*16), (1, 256, 1024))``): swap to
            #     ``lane%8 → row``, ``lane//8 → col_chunk``. 8 lanes now span
            #     NLANE=0..7 at the same (n_outer, klane, k0) → 128 B
            #     contiguous in DRAM. LDS write at ``lane*16`` produces
            #     col-major-within-wave LDS where each 128-byte LDS row gets
            #     exactly 8 lanes (32 banks fully utilised) → no bank conflict
            #     without any additional swizzle.
            offsets = []
            for round in range_constexpr(max(N_TILES_A, N_TILES_B)):
                if const_expr(preshuffled):
                    row = lane_id % 8 + wave_id * 8 + round * 32
                    col = (lane_id // 8) * 16
                    offsets.append(
                        (row // 16) * (K * 16)
                        + (row % 16) * 16
                        + (col // 64) * 1024
                        + ((col % 64) // 16) * 256
                        + (col % 16)
                    )
                else:
                    row = lane_id // 8 + wave_id * 8 + round * 32
                    col = (lane_id % 8) * 16
                    r, c = _swizzle_128(row, col)
                    offsets.append(r * K + c)
            return offsets

        def _compute_lds_swizzle(wave_idx, n_tiles, preshuffled=False):
            # LDS read byte for (R, C) within one sub-buffer:
            #   * Non-preshuffled: ``byte = R*BLOCK_K + C_swizzled`` (XOR
            #     swizzle on col by R-bits to avoid bank conflicts).
            #   * Preshuffled: ``byte = (R//8)*1024 + (R%8)*16 + (C//16)*128``.
            #     col-major-within-wave-portion; ``C`` is always a multiple of
            #     16 here (``C = (lane//16)*16 + step*64``) so ``C%16 == 0``
            #     drops out. 8 lanes per 128-byte LDS row utilise all 32 banks,
            #     producing the MI350-optimal 2-way ds_read_b128 bank pattern.
            lds_swz = []
            for row_offset in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    if const_expr(preshuffled):
                        swz.append((row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128)
                    else:
                        r, c = _swizzle_128(row, col)
                        swz.append(r * BLOCK_K + c)
                lds_swz.append(swz)
            return lds_swz

        # G->LDS atom: 128 bits per thread = 16 fp8 elements. The atom
        # state carries the runtime ``soffset`` set to ``k_offset``.
        g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        # LDS dst pointers for ``buffer_load_lds`` go through
        # ``extract_aligned_pointer_as_index + add + inttoptr`` to break
        # LLVM's alias chain on the LDS sub-buffer symbols; otherwise the
        # AMDGPU backend inserts defensive ``s_waitcnt vmcnt(N)`` between
        # G->LDS writes and the subsequent ``ds_read``.
        def _lds_dst_at(lds_dst_mem, byte_offset_runtime):
            base_idx = _memref_dialect.extract_aligned_pointer_as_index(lds_dst_mem)
            offset_idx = base_idx + fx.Index(byte_offset_runtime)
            offset_i64 = _arith_dialect.index_cast(fx.T.i64(), offset_idx)
            lds_ptr = fx.inttoptr(_shared_ptr_ty, offset_i64)
            return fx.make_view(lds_ptr, fx.make_layout(1, 1))

        def _load_lds(gl_src_div, lds_dst_mem, k_offset, gl_offsets, n_tiles):
            assert len(gl_offsets) >= n_tiles
            for step in range_constexpr(n_tiles):
                src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[step])))
                dst = _lds_dst_at(lds_dst_mem, wave_id * 1024 + step * 4096)
                fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _load_one_lds(gl_src_div, lds_dst_mem, k_offset, gl_offsets, tile_idx):
            assert len(gl_offsets) > tile_idx
            src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[tile_idx])))
            dst = _lds_dst_at(lds_dst_mem, wave_id * 1024 + tile_idx * 4096)
            fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _pack_i32x4_i32x8(lo, hi):
            return lo.shuffle(hi, list(range(8)))

        def _load_rt(lds_src, wave_idx, n_tiles, preshuffled=False):
            frag = []
            for i in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    if const_expr(preshuffled):
                        # C%16 == 0 by construction; drop it.
                        byte = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                    else:
                        r, c = _swizzle_128(row, col)
                        byte = r * BLOCK_K + c
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(byte)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _load_one_rt(lds_src, lds_swz, row, k):
            v = Vec.load(Vec16_t, lds_src, [fx.Index(lds_swz[row][k])])
            return v.bitcast(fx.Int32)

        def _c_idx(i, j):
            return i * N_TILES_B + j

        # The C++ AddressSpace enum prepends Generic=0, so the Python
        # AddressSpace.Register value (2) maps to Shared on the C++ side.
        # Pass the C++ integer (3) directly to MemRefType.get.
        _AS_REG = 3
        scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        reg_f32_4_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(4, 1), _AS_REG)
        reg_f32_1_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), _AS_REG)
        reg_bf16_1_ty = fx.MemRefType.get(fx.T.bf16(), fx.LayoutType.get(1, 1), _AS_REG)

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

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        # MFMA via ``fly.mma_atom_call_ssa``. The atom carries scale_a /
        # scale_b state (default 0x7F7F7F7F = no scaling). Returns a
        # chained Vec(4, f32) SSA so the accumulator stays on AGPR.
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

        def _mfma(a_val, b_val, c_val):
            return _fly_dialect.mma_atom_call_ssa([MfmaAccum_t], mma_atom, a_val, b_val, c_val)

        def _mfma_ABt_all(a, b, c):
            assert len(a) == N_TILES_A
            assert len(b) == N_TILES_B
            assert len(c) == N_TILES_A * N_TILES_B

            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = _mfma(a[i], b[j], c[_c_idx(i, j)])
            return c

        def _mfma_ABt_one(a, b, c, m, n):
            assert m < N_TILES_A and n < N_TILES_B

            c[_c_idx(m, n)] = _mfma(a[m], b[n], c[_c_idx(m, n)])
            return c

        def _interleaved_cluster(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c,
            lds_src_preshuffled=False,
        ):
            # 64x64 output via 4x4 MFMAs, with per-tile G→LDS and LDS→reg
            # loads interleaved between MFMAs to hide latency.
            rt_dst = []

            c = _mfma_ABt_one(a, b, c, 0, 0)
            c = _mfma_ABt_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(wave_idx, n_tiles_lds, preshuffled=lds_src_preshuffled)
            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 0, 0)

            c = _mfma_ABt_one(a, b, c, 0, 2)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 0, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 0, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 1, 0)

            c = _mfma_ABt_one(a, b, c, 1, 0)
            c = _mfma_ABt_one(a, b, c, 1, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 1, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 1, 2)
            c = _mfma_ABt_one(a, b, c, 1, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 2, 0)

            c = _mfma_ABt_one(a, b, c, 2, 0)
            c = _mfma_ABt_one(a, b, c, 2, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 2, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 2, 2)
            c = _mfma_ABt_one(a, b, c, 2, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 3, 0)

            c = _mfma_ABt_one(a, b, c, 3, 0)
            c = _mfma_ABt_one(a, b, c, 3, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 3, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 3, 2)
            c = _mfma_ABt_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c,
            lds_src_preshuffled=False,
        ):
            _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_tiles_lds)
            rt_dst = _load_rt(lds_src, wave_idx, n_tiles_rt, preshuffled=lds_src_preshuffled)
            c = _mfma_ABt_all(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c,
            lds_src_preshuffled=False,
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )
            else:
                return _compute_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        gl_off_a = _compute_global_swizzle(preshuffled=False)
        gl_off_b = _compute_global_swizzle(b_preshuffled)

        # Prologue: 8-buffer LDS pipeline pre-fill.
        _load_lds(ga_div, a_cur0, A0_gl_offset + 0 * A_K_STEP, gl_off_a, N_TILES_A)
        _load_lds(gb_div, b_cur0, B0_gl_offset + 0 * B_K_STEP, gl_off_b, N_TILES_B)
        _load_lds(gb_div, b_cur1, B1_gl_offset + 0 * B_K_STEP, gl_off_b, N_TILES_B)
        _load_lds(ga_div, a_cur1, A1_gl_offset + 0 * A_K_STEP, gl_off_a, N_TILES_A)

        _load_lds(ga_div, a_next0, A0_gl_offset + 1 * A_K_STEP, gl_off_a, N_TILES_A)
        _load_lds(gb_div, b_next0, B0_gl_offset + 1 * B_K_STEP, gl_off_b, N_TILES_B)
        _load_lds(gb_div, b_next1, B1_gl_offset + 1 * B_K_STEP, gl_off_b, N_TILES_B)
        _load_lds(ga_div, a_next1, A1_gl_offset + 1 * A_K_STEP, gl_off_a, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = _load_rt(a_cur0, wave_i, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = _load_rt(b_cur0, wave_j, N_TILES_B, preshuffled=b_preshuffled)

        for k in range_constexpr(K_ITERS - 2):
            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                ga_div,
                A0_gl_offset + (k + 2) * A_K_STEP,
                gl_off_a,
                wave_j,
                b_cur1,
                N_TILES_A,
                N_TILES_B,
                a0_frag,
                b0_frag,
                c00_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0,
                gb_div,
                B0_gl_offset + (k + 2) * B_K_STEP,
                gl_off_b,
                wave_i,
                a_cur1,
                N_TILES_B,
                N_TILES_A,
                a0_frag,
                b1_frag,
                c01_frag,
            )

            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1,
                gb_div,
                B1_gl_offset + (k + 2) * B_K_STEP,
                gl_off_b,
                wave_i,
                a_next0,
                N_TILES_B,
                N_TILES_A,
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1,
                ga_div,
                A1_gl_offset + (k + 2) * A_K_STEP,
                gl_off_a,
                wave_j,
                b_next0,
                N_TILES_A,
                N_TILES_B,
                a1_frag,
                b1_frag,
                c11_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B, preshuffled=b_preshuffled)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = _load_rt(a_next0, wave_i, N_TILES_A)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        b0_frag = _load_rt(b_next0, wave_j, N_TILES_B, preshuffled=b_preshuffled)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        _wait_barrier(0)
        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B, preshuffled=b_preshuffled)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

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
        grid_x = (M * N) // (BLOCK_M * BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
