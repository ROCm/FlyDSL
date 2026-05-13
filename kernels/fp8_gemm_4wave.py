# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, bf16 stores, and the per-atom MFMA all go
through the layout API (``fx.rocdl.make_buffer_tensor`` + ``fx.copy``
+ ``fx.gemm``). The XOR swizzle and the 8-buffer LDS pipeline are
kept as direct arithmetic to preserve the kernel's interleaved
cluster scheduling.

LDS storage uses 8 named ``fx.get_dyn_shared`` bases carved into one
dyn-shared region; the ``fly-attach-lds-alias-scope`` MLIR pass
attaches per-symbol alias scopes so AMDGPU's SI Wait Counter pass
treats cross-buffer accesses as no-alias.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TgtAS
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _preshuffle_2d(t):
    """K-major outermost preshuffle for a row-major ``(rows, K)`` tensor.

    Output shape ``(K//128, rows//16, 16, 128)`` is K-major outermost so
    every 128-K-col atom slab of ``rows * 128`` bytes lives contiguously.
    """
    rows, k = t.shape[-2:]
    assert rows % 16 == 0 and k % 128 == 0, f"need rows%16==0 and K%128==0, got rows={rows} K={k}"
    return t.reshape(rows // 16, 16, k // 128, 128).permute(2, 0, 1, 3).contiguous()


def preshuffle_a(a):
    """Permute row-major ``A`` ``(M, K)`` into the layout used when
    ``a_preshuffled=True``. Same K-major outermost packing as
    :func:`preshuffle_b`."""
    return _preshuffle_2d(a)


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` into the layout used when
    ``b_preshuffled=True``. Same K-major outermost packing as
    :func:`preshuffle_a`."""
    return _preshuffle_2d(b_t)


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
    a_preshuffled: bool = False,
    b_preshuffled: bool = False,
):
    """4-wave FP8 GEMM kernel compile entry point.

    When ``a_preshuffled`` / ``b_preshuffled`` is True, the matching
    operand is assumed to be already permuted by
    :func:`preshuffle_a` / :func:`preshuffle_b` into a K-major
    outermost layout (``(K//128, rows//16, 16, 128)``). LDS layout,
    MFMA operand layout and wave assignment are unchanged.
    """
    BLOCK_K = 128  # MFMA_Scale 16x16x128 atom; 4-wave 2x2 layout needs BLOCK >= 64.
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert BLOCK_M >= 64 and BLOCK_N >= 64
    assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K
    # 16-row 16x128 atom tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # 8 disjoint sub-buffers within one dyn-shared region. Each named
    # ``fx.get_dyn_shared`` emits a distinct LDS global so the
    # ``fly-attach-lds-alias-scope`` pass can give it its own alias
    # scope.
    _LDS_SUBBUFS = [
        ("A_lds_cur_0", 0 * a_lds_size),
        ("A_lds_cur_1", 1 * a_lds_size),
        ("A_lds_next_0", 2 * a_lds_size),
        ("A_lds_next_1", 3 * a_lds_size),
        ("B_lds_cur_0", 4 * a_lds_size + 0 * b_lds_size),
        ("B_lds_cur_1", 4 * a_lds_size + 1 * b_lds_size),
        ("B_lds_next_0", 4 * a_lds_size + 2 * b_lds_size),
        ("B_lds_next_1", 4 * a_lds_size + 3 * b_lds_size),
    ]
    _TOTAL_LDS_BYTES = 4 * a_lds_size + 4 * b_lds_size

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)
        F8_IR_t = fx.Float8E4M3FN.ir_type

        _AS_SHARED = 2
        _shared_f8_ptr_ty = fx.PointerType.get(F8_IR_t, _AS_SHARED, 512)
        _shared_i32_ptr_ty = fx.PointerType.get(fx.T.i32(), _AS_SHARED, 512)

        _lds_int = {
            name: fx.ptrtoint(fx.get_dyn_shared(sym_name=name))
            for name, _ in _LDS_SUBBUFS
        }
        _lds_off = dict(_LDS_SUBBUFS)

        a_cur0, a_cur1 = "A_lds_cur_0", "A_lds_cur_1"
        a_next0, a_next1 = "A_lds_next_0", "A_lds_next_1"
        b_cur0, b_cur1 = "B_lds_cur_0", "B_lds_cur_1"
        b_next0, b_next1 = "B_lds_next_0", "B_lds_next_1"

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(M // BLOCK_M, N // BLOCK_N)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        # K-major preshuffle: tile's slot in k_outer=0 slab is
        # (tile_row // 16) * 2048 = tile_row * 128; the per-K-iter step
        # then jumps to the next ``rows * BLOCK_K`` byte slab.
        if const_expr(a_preshuffled):
            A0_gl_offset = (tile_i * BLOCK_M) * BLOCK_K
            A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * BLOCK_K
            A_K_STEP = M * BLOCK_K
        else:
            A0_gl_offset = (tile_i * BLOCK_M) * K
            A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
            A_K_STEP = BLOCK_K
        if const_expr(b_preshuffled):
            B0_gl_offset = (tile_j * BLOCK_N) * BLOCK_K
            B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * BLOCK_K
            B_K_STEP = N * BLOCK_K
        else:
            B0_gl_offset = (tile_j * BLOCK_N) * K
            B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K
            B_K_STEP = BLOCK_K

        # A/B arrive as torch.int8 (PyTorch fp8 view limitation); recast
        # the buffer-desc element type to fp8 so BufferCopyLDS128b takes
        # them.
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

        # XOR 3 bits of dim-0 (row, bit-1 base) with 3 bits of dim-1
        # (col, bit-4 base). Same as the manual
        # ((offset>>8)<<4) ^ offset; shared between LDS and global
        # access via outer layouts with different strides.
        _swz_attr = fx.CoordSwizzleType.get(3, 1, [0], 4, [1])
        _swz_shape = (LDS_BLOCK_M, BLOCK_K)
        _coord_swz = fx.make_composed_layout(
            fx.static(_swz_attr), fx.make_identity_layout(_swz_shape)
        )
        _lds_swz_layout = fx.make_composed_layout(
            fx.make_layout(_swz_shape, (BLOCK_K, 1)), _coord_swz
        )
        # Preshuffled (K-major outermost): per K_iter all rows of one K
        # atom (= ``rows * BLOCK_K`` bytes) live in a contiguous slab;
        # within the slab, ``(outer, inner, k_inner)`` is row-major. The
        # k_outer step is fed via ``soffset`` (``k * rows * BLOCK_K``).
        def _preshuffle_layout():
            return fx.make_composed_layout(
                fx.make_layout(
                    ((16, LDS_BLOCK_M // 16), BLOCK_K),
                    ((BLOCK_K, 16 * BLOCK_K), 1),
                ),
                _coord_swz,
            )

        _gl_swz_layout_a = (
            _preshuffle_layout()
            if const_expr(a_preshuffled)
            else fx.make_composed_layout(fx.make_layout(_swz_shape, (K, 1)), _coord_swz)
        )
        _gl_swz_layout_b = (
            _preshuffle_layout()
            if const_expr(b_preshuffled)
            else fx.make_composed_layout(fx.make_layout(_swz_shape, (K, 1)), _coord_swz)
        )

        def _compute_global_swizzle(layout):
            offsets = []
            for round in range_constexpr(max(N_TILES_A, N_TILES_B)):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                offsets.append(fx.crd2idx((row, col), layout).to_py_value())
            return offsets

        # 128 bits per thread = 16 fp8 elements; soffset carries the
        # runtime k offset.
        g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        # Routed through ptrtoint + add + inttoptr instead of the
        # natural fx.add_offset+fx.recast_iter chain: empirically the
        # natural route compiles ~5-9% slower on BLOCK=256 shapes
        # (probably because the extra int_tuple / recast_iter ops
        # survive canonicalization and disrupt the AMDGPU back-end's
        # common-base + offset pattern matching).
        def _lds_dst_at(name, byte_offset_runtime):
            off = _lds_int[name] + fx.Int32(_lds_off[name] + byte_offset_runtime)
            ptr = fx.inttoptr(_shared_f8_ptr_ty, off)
            return fx.make_view(ptr, fx.make_layout(1, 1))

        def _load_lds(gl_src_div, name, k_offset, gl_offsets, n_tiles):
            assert len(gl_offsets) >= n_tiles
            for step in range_constexpr(n_tiles):
                src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[step])))
                dst = _lds_dst_at(name, wave_id * 1024 + step * 4096)
                fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _load_one_lds(gl_src_div, name, k_offset, gl_offsets, tile_idx):
            assert len(gl_offsets) > tile_idx
            src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[tile_idx])))
            dst = _lds_dst_at(name, wave_id * 1024 + tile_idx * 4096)
            fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _pack_i32x4_i32x8(lo, hi):
            return lo.shuffle(hi, list(range(8)))

        # 16 fp8 == 4 i32; load as i32x4 because LLVM has no
        # vector<16xf8>.
        def _vec_load_lds_i32x4(name, fp8_elem_offset):
            off = _lds_int[name] + fx.Int32(_lds_off[name] + fp8_elem_offset)
            ptr = fx.inttoptr(_shared_i32_ptr_ty, off)
            view = fx.make_view(ptr, fx.make_layout(4, 1))
            return Vec(fx.memref_load_vec(view))

        def _load_rt(name, wave_idx, n_tiles):
            frag = []
            for i in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    halves.append(_vec_load_lds_i32x4(name, fx.crd2idx((row, col), _lds_swz_layout).to_py_value()))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _load_one_rt(name, wave_idx, n_tiles, row_idx, k):
            row = wave_idx * (n_tiles * 16) + row_idx * 16 + lane_id % 16
            col = (lane_id // 16) * 16 + k * 64
            return _vec_load_lds_i32x4(name, fx.crd2idx((row, col), _lds_swz_layout).to_py_value())

        def _c_idx(i, j):
            return i * N_TILES_B + j

        # C++ AddressSpace enum prepends Generic=0; pass the C++ index
        # (3 = Register) directly to MemRefType.get to avoid the
        # Python AddressSpace.Register (=2) being read as Shared.
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

        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

        # MFMA goes through fx.gemm. The Vec operands are spilled into
        # register-memref fragments around each call; the alloca /
        # store / load round trip is folded away by
        # ``fly-convert-atom-call-to-ssa-form`` +
        # ``fly-promote-regmem-to-vectorssa``, leaving a plain
        # ``llvm.amdgcn.mfma.scale.f32.16x16x128`` chained on
        # ``<4 x float>`` SSA so ISel keeps the accumulator on AGPR.
        a_reg_ty = fx.MemRefType.get(fx.T.i32(), fx.LayoutType.get(8, 1), _AS_REG)
        b_reg_ty = fx.MemRefType.get(fx.T.i32(), fx.LayoutType.get(8, 1), _AS_REG)
        c_reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(4, 1), _AS_REG)
        tiled_mma_single = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((2, 2, 1), (1, 2, 0)),
        )

        def _mfma(a_vec, b_vec, c_vec):
            a_mem = fx.memref_alloca(a_reg_ty, fx.make_layout(8, 1))
            b_mem = fx.memref_alloca(b_reg_ty, fx.make_layout(8, 1))
            c_mem = fx.memref_alloca(c_reg_ty, fx.make_layout(4, 1))
            fx.memref_store_vec(a_vec, a_mem)
            fx.memref_store_vec(b_vec, b_mem)
            fx.memref_store_vec(c_vec, c_mem)
            fx.gemm(tiled_mma_single, c_mem, a_mem, b_mem, c_mem)
            return Vec(fx.memref_load_vec(c_mem))

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

        def _interleaved_cluster(lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c):
            # 4x4 MFMAs over 64x64, with per-tile G->LDS and LDS->reg
            # loads interleaved between MFMAs to hide latency.
            rt_dst = []

            c = _mfma_ABt_one(a, b, c, 0, 0)
            c = _mfma_ABt_one(a, b, c, 0, 1)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 0, 0)

            c = _mfma_ABt_one(a, b, c, 0, 2)

            rt_dst_1 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 0, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 0, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 1, 0)

            c = _mfma_ABt_one(a, b, c, 1, 0)
            c = _mfma_ABt_one(a, b, c, 1, 1)

            rt_dst_1 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 1, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 1, 2)
            c = _mfma_ABt_one(a, b, c, 1, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 2, 0)

            c = _mfma_ABt_one(a, b, c, 2, 0)
            c = _mfma_ABt_one(a, b, c, 2, 1)

            rt_dst_1 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 2, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 2, 2)
            c = _mfma_ABt_one(a, b, c, 2, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 3, 0)

            c = _mfma_ABt_one(a, b, c, 3, 0)
            c = _mfma_ABt_one(a, b, c, 3, 1)

            rt_dst_1 = _load_one_rt(lds_src, wave_idx, n_tiles_lds, 3, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 3, 2)
            c = _mfma_ABt_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c
        ):
            _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_tiles_lds)
            rt_dst = _load_rt(lds_src, wave_idx, n_tiles_rt)
            c = _mfma_ABt_all(a, b, c)
            return c, rt_dst

        def _compute_block(lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c
                )
            else:
                return _compute_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        gl_off_a = _compute_global_swizzle(_gl_swz_layout_a)
        gl_off_b = _compute_global_swizzle(_gl_swz_layout_b)

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

        b0_frag = _load_rt(b_cur0, wave_j, N_TILES_B)

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
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = _load_rt(a_next0, wave_i, N_TILES_A)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        b0_frag = _load_rt(b_next0, wave_j, N_TILES_B)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        _wait_barrier(0)
        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)
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
        grid_x = (M * N) // (BLOCK_M * BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            smem=_TOTAL_LDS_BYTES,
            stream=stream,
        )

    return launch_gemm
