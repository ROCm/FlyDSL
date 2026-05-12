# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 4-Wave GEMM kernel — algorithm of ROCm/FlyDSL PR #505 with all
global-memory IO expressed via the FlyDSL layout API. Bit-for-bit
identical final ISA to PR's raw rocdl path on MI350X.

Layout-API surface used (vs PR's raw rocdl path):

* ``fx.rocdl.make_buffer_tensor`` for A, B, C, A_scale, B_scale (replaces
  ``buffer_ops.create_buffer_resource``).
* ``fx.recast_iter`` to retype the i8-viewed FP8 buffers back to
  ``Float8E4M3FN`` so the typed copy atoms accept them.
* ``fx.logical_divide`` + ``fx.slice`` for per-thread 1:1 buffer-desc
  memref views on the global SOURCE side of G→LDS (PR's per-thread
  ``voffset`` becomes the slice index).
* ``fx.make_copy_atom`` with ``BufferCopyLDS128b`` (G→LDS),
  ``BufferCopy128b`` (vec4 A scale), ``BufferCopy32b`` (scalar B scale)
  and ``BufferCopy16b`` (bf16 output stores).
* ``fx.copy_atom_call`` with ``atom.set_value("soffset", ...)`` carrying
  PR's per-call ``k_offset`` through the atom state struct.
* ``fx.memref_alloca`` / ``fx.memref_load_vec`` / ``fx.memref_store_vec``
  for the per-thread accumulator scratch in the epilogue.

CRITICAL — LDS dst pointer construction
---------------------------------------

The G→LDS copy atom requires a 1:1 shared memref as its destination.
The "obvious" way (``fx.slice(fx.logical_divide(lds_view, (1,1)),
(None, byte_offset))``) lowers to ``getelementptr i8, ptr addrspace(3)
@A_lds_cur_0, i32 byte_offset``. The GEP keeps ``@A_lds_cur_0`` in the
LLVM alias chain, so the AMDGPU backend sees ``buffer_load_lds writes
@A_lds_cur_0`` and ``ds_read reads @A_lds_cur_0`` as MAY-alias and
defensively inserts ``s_waitcnt vmcnt(N)`` between them — even though
our explicit ``_wait_barrier(...)`` already paces the LDS pipeline.
Empirically that adds ~76 extra ``vmcnt(20/24/28)`` instructions per
8192³ kernel and drops TFLOPS by ~4%.

The fix (``_lds_dst_at`` helper) constructs the per-step LDS dst pointer
the same way PR does:

    base_idx = memref.extract_aligned_pointer_as_index(lds_dst)
    offset_i64 = arith.index_cast(i64, base_idx + byte_offset)
    lds_ptr = fx.inttoptr(!fly.ptr<f8, shared>, offset_i64)
    dst    = fx.make_view(lds_ptr, fx.make_layout(1, 1))

The ``inttoptr`` is opaque to LLVM's alias analysis, so ``buffer_load_lds``
and ``ds_read`` no longer appear to alias. The scheduler trusts our
manual synchronization, drops the redundant ``vmcnt`` waits, and the
final ISA matches PR's bit-for-bit (15412 lines, 0 diff).

What stays raw (and *why*):

* The MFMA hot loop keeps each ``Vec(4, f32)`` accumulator as a chained
  SSA value through raw ``rocdl.mfma_scale_f32_16x16x128_f8f6f4`` calls,
  because routing the 256-wide accumulator through ``fx.gemm`` lowers to
  memref load/store and forces the LLVM AMDGCN backend onto VGPR-dest
  MFMA + ``v_accvgpr_*`` shuffles (~70-80% of PR TFLOPS).
* LDS→Reg uses ``Vec.load`` directly with the manual swizzle, which is
  what the MFMA accumulator chain consumes.
* ``s_waitcnt vmcnt(N) ; s_barrier`` is emitted via ``_llvm.inline_asm``
  to pace the multi-buffer LDS pipeline exactly as PR does.

Public entry point is ``compile_fp8_gemm_4wave`` (renamed from PR's
``compile_fp8_gemm``) so existing tests continue to import the same
symbol.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


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
        # naive mapping
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


def compile_fp8_gemm_4wave(
    *,
    M: int,
    N: int,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
):
    # fixed for MFMA 16x16x128
    BLOCK_K = 128

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    # The base mfma atom is 16x16, we use 4 waves in a 2x2 config so the
    # block size must be at least 64 to keep this config
    assert BLOCK_M >= 64
    assert BLOCK_N >= 64

    assert N % BLOCK_N == 0
    assert M % BLOCK_M == 0
    assert K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    # this is actually the number of 16-row tiles in a BLOCK_M x BLOCK_N tile
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16

    # Each accumulator is 4 floats (depends on MFMA atom)
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

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        MfmaAccum_t = Vec.make_type(4, fx.Float32)
        # Initial value for the C register tile
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

        # Helpers for PR-style LDS dst pointer construction (see _lds_dst_at
        # below). NOTE: pass C++'s integer for the address space directly;
        # Python's AddressSpace.Shared enum value (=1) maps to C++'s Global (=1).
        from flydsl._mlir.dialects import memref as _memref_dialect
        from flydsl._mlir.dialects import arith as _arith_dialect

        _AS_SHARED = 2  # C++ Fly_AddressSpace::Shared
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
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K

        # Layout-API buffer-desc tensors for both G→LDS (BufferCopyLDS128b)
        # and the epilogue (BufferCopy{16,32,128}b). Replaces PR's raw
        # buffer_ops.{create_buffer_resource,buffer_load,buffer_store} +
        # rocdl.raw_ptr_buffer_load_lds.
        # A/B come in as torch.int8 (PyTorch fp8 view restriction); recast
        # the buffer-desc pointer's element type to fp8 so the BufferCopyLDS128b
        # atom (which requires src/dst element-type match) can consume them.
        from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TgtAS

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

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(max(N_TILES_A, N_TILES_B)):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                a, b = _swizzle_128(row, col)
                offsets.append(a * K + b)
            return offsets

        def _compute_lds_swizzle(wave_idx, n_tiles):
            lds_swz = []
            for row_offset in range_constexpr(n_tiles):
                # Each wave loads a 16x128 tile of A/B, max_steps is the
                # max number of tiles we want to load.
                row = wave_idx * (n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    swz_row, swz_col = _swizzle_128(row, col)
                    swz.append(swz_row * 128 + swz_col)
                lds_swz.append(swz)
            return lds_swz

        # G→LDS atom (128 bits per thread = 16 fp8 elems). State carries
        # (soffset, imm_offset); we set soffset = k_offset per call to
        # match PR's `rocdl.raw_ptr_buffer_load_lds(..., soffset, ...)`.
        # Pass valBits=128 explicitly so the atom matches the LDS-128b op.
        g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        # IMPORTANT — LDS dst pointer construction matches PR's
        # ptrtoint+add+inttoptr style instead of going through fx.slice on
        # a logical_divide tensor. The fx.slice path lowers to a GEP from
        # @A_lds_cur_0, which keeps the LDS sub-buffer's symbol in LLVM's
        # alias chain. LLVM AMDGPU then sees `buffer_load_lds writes into
        # @A_lds_cur_0` and `ds_read reads from @A_lds_cur_0` as aliasing,
        # and inserts conservative `s_waitcnt vmcnt(N)` between them — even
        # though our explicit `_wait_barrier(...)` already pacing the LDS
        # pipeline. The inttoptr path produces an opaque pointer that
        # breaks LLVM's alias chain, so the scheduler trusts our manual
        # synchronization and does not insert extra vmcnt waits.
        def _lds_dst_at(lds_dst_mem, byte_offset_runtime):
            base_idx = _memref_dialect.extract_aligned_pointer_as_index(lds_dst_mem)
            offset_idx = base_idx + fx.Index(byte_offset_runtime)
            offset_i64 = _arith_dialect.index_cast(fx.T.i64(), offset_idx)
            lds_ptr = fx.inttoptr(_shared_ptr_ty, offset_i64)
            return fx.make_view(lds_ptr, fx.make_layout(1, 1))

        def _load_lds(gl_src_div, lds_dst_mem, k_offset, gl_offsets, n_tiles):
            assert len(gl_offsets) >= n_tiles
            atom = g2lds_atom.set_value({"soffset": fx.Int32(k_offset)})
            for step in range_constexpr(n_tiles):
                src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[step])))
                dst = _lds_dst_at(lds_dst_mem, wave_id * 1024 + step * 4096)
                fx.copy_atom_call(atom, src, dst)

        def _load_one_lds(gl_src_div, lds_dst_mem, k_offset, gl_offsets, tile_idx):
            assert len(gl_offsets) > tile_idx
            atom = g2lds_atom.set_value({"soffset": fx.Int32(k_offset)})
            src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[tile_idx])))
            dst = _lds_dst_at(lds_dst_mem, wave_id * 1024 + tile_idx * 4096)
            fx.copy_atom_call(atom, src, dst)

        def _pack_i32x4_i32x8(lo, hi):
            # Pack two i32x4 as one i32x8
            return lo.shuffle(hi, list(range(8)))

        def _load_rt(lds_src, wave_idx, n_tiles):
            # Load n_tiles 16x128 fragments from LDS to registers
            # Each 16x128 fragment requires 2 i32x4 (2 ds_read_b128)
            frag = []
            for i in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz)])
                    halves.append(v.bitcast(fx.Int32))  # i32x4
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))  # i32x8
            return frag

        def _load_one_rt(lds_src, lds_swz, row, k):
            # Load half of a 16x128 tile from LDS to registers
            v = Vec.load(Vec16_t, lds_src, [fx.Index(lds_swz[row][k])])
            return v.bitcast(fx.Int32)  # return a i32x4

        def _c_idx(i, j):
            return i * N_TILES_B + j

        # Atoms + register memref types for the layout-API epilogue.
        # NOTE: C++ AddressSpace adds a Generic=0 ahead of Python's enum, so
        # AddressSpace.Register (Python = 2) gets misinterpreted as Shared (C++ = 2).
        # Pass C++'s integer value (3) directly.
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
                fx.copy_atom_call(scale_atom_4, fx.slice(sa_div, (None, fx.Int32(row))), r)
                return Vec(fx.memref_load_vec(r))

            def _load_scale_scalar(col):
                r = fx.memref_alloca(reg_f32_1_ty, fx.make_layout(1, 1))
                fx.copy_atom_call(scale_atom_1, fx.slice(sb_div, (None, fx.Int32(col))), r)
                return Vec(fx.memref_load_vec(r))[0]

            def _store_bf16(value_bf16, c_index):
                r = fx.memref_alloca(reg_bf16_1_ty, fx.make_layout(1, 1))
                fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), r)
                fx.copy_atom_call(out_atom_1, r, fx.slice(c_div, (None, fx.Int32(c_index))))

            a_scales = [
                _load_scale_vec4(base_row + i * 16 + (lane_id // 16) * 4)
                for i in range_constexpr(N_TILES_A)
            ]
            b_scales = [
                _load_scale_scalar(base_col + i * 16 + lane_id % 16)
                for i in range_constexpr(N_TILES_B)
            ]
            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    # this is a constant and is equal to the number of
                    # floats each thread holds for a 16x16 out tile
                    # (depends on mfma atom used)
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

        def _mfma_ABt_all(a, b, c):
            assert len(a) == N_TILES_A
            assert len(b) == N_TILES_B
            assert len(c) == N_TILES_A * N_TILES_B

            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        MfmaAccum_t,
                        [a[i], b[j], c[_c_idx(i, j)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                    )
            return c

        def _mfma_ABt_one(a, b, c, m, n):
            assert m < N_TILES_A and n < N_TILES_B

            c[_c_idx(m, n)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                MfmaAccum_t,
                [a[m], b[n], c[_c_idx(m, n)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
            )
            return c

        def _interleaved_cluster(lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c):
            # Compute a 64x64 output tile using 4x4 MFMA instructions
            # returns the updated accumulator and the next fragment loaded from lds_src
            rt_dst = []

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 0, 0)
            c = _mfma_ABt_one(a, b, c, 0, 1)
            # rocdl.sched_barrier(0)

            lds_swz = _compute_lds_swizzle(wave_idx, n_tiles_lds)
            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 0, 0)

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 0, 2)
            # rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 0, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 0, 3)
            # rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 1, 0)

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 1, 0)
            c = _mfma_ABt_one(a, b, c, 1, 1)
            # rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 1, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 1, 2)
            c = _mfma_ABt_one(a, b, c, 1, 3)
            # rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 2, 0)

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 2, 0)
            c = _mfma_ABt_one(a, b, c, 2, 1)
            # rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 2, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 2, 2)
            c = _mfma_ABt_one(a, b, c, 2, 3)
            # rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 3, 0)

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 3, 0)
            c = _mfma_ABt_one(a, b, c, 3, 1)
            # rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 3, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            # rocdl.sched_barrier(0)
            c = _mfma_ABt_one(a, b, c, 3, 2)
            c = _mfma_ABt_one(a, b, c, 3, 3)
            # rocdl.sched_barrier(0)

            return c, rt_dst

        def _compute_cluster(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c
        ):
            _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_tiles_lds)
            rt_dst = _load_rt(lds_src, wave_idx, n_tiles_rt)
            c = _mfma_ABt_all(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, a, b, c
                )
            else:
                return _compute_cluster(
                    lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(ga_div, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(gb_div, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(gb_div, b_cur1, B1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(ga_div, a_cur1, A1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)

        # Issue load for next tile
        _load_lds(ga_div, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(gb_div, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(gb_div, b_next1, B1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(ga_div, a_next1, A1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)

        # So far we issued 4 loads for A and 4 loads for B, each load
        # requires N_TILES_A/B memory ops. That would be
        # 4 * TILES_A + 4 * TILES_B but since we need a_cur0 it's 3*TILES_A
        _wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))  # wait for a_cur0

        a0_frag = _load_rt(a_cur0, wave_i, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))  # wait for b_cur0

        b0_frag = _load_rt(b_cur0, wave_j, N_TILES_B)

        for k in range_constexpr(K_ITERS - 2):
            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))  # 2 loads in-flight for each of A/B

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                ga_div,
                A0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
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
                B0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
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
                B1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
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
                A1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_j,
                b_next0,
                N_TILES_A,
                N_TILES_B,
                a1_frag,
                b1_frag,
                c11_frag,
            )

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 2
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)

        # rocdl.sched_barrier(0)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        # rocdl.sched_barrier(0)

        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)

        # rocdl.sched_barrier(0)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        # rocdl.sched_barrier(0)

        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))

        a0_frag = _load_rt(a_next0, wave_i, N_TILES_A)

        # rocdl.sched_barrier(0)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        # rocdl.sched_barrier(0)

        b0_frag = _load_rt(b_next0, wave_j, N_TILES_B)

        # rocdl.sched_barrier(0)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        # rocdl.sched_barrier(0)

        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 1
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)

        _wait_barrier(0)

        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)

        # rocdl.sched_barrier(0)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        # rocdl.sched_barrier(0)

        # rocdl.sched_barrier(0)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        # rocdl.sched_barrier(0)

        # rocdl.sched_barrier(0)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        # rocdl.sched_barrier(0)

        # rocdl.sched_barrier(0)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        # rocdl.sched_barrier(0)

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
        from flydsl.compiler.kernel_function import CompilationContext

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
