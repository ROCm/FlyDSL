#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""HipKittens-aligned BF16 GEMM kernel.

Design: 8 warps / 512 threads, WARPS_M=2 x WARPS_N=4,
A+B both through LDS with cooperative loading, XOR-16 swizzle,
2-stage (128KB) ping-pong with 2 K-tiles per stage,
mfma_f32_16x16x32_bf16 on gfx950.

Tile: 256x256, K-tile=32 per chunk (2 chunks per stage = 64 K elements).
LDS: 128KB total (2 stages x 64KB), gfx950 only (160KB LDS).

HK alignment:
- 16 MFMAs per phase (2 K-tiles accumulated)
- ds_read_b128 with constant offsets from 2 base VGPRs (one per stage)
- s_setprio(1)/s_setprio(0) around MFMA groups
- lgkmcnt-only waits (no vmcnt in inner loop)
"""

from __future__ import annotations
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, gpu, vector, rocdl, buffer_ops
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.compiler.kernel_function import CompilationContext

from kernels.mfma_preshuffle_pipeline import (
    swizzle_xor16,
    tile_chunk_coord_i32,
)

def _range_constexpr(n):
    return fx.range_constexpr(n)


def compile_hk_bf16_gemm(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 32,
    warps_m: int = 2,
    warps_n: int = 4,
    waves_per_eu: Optional[int] = 2,
    out_dtype: str = "bf16",
):
    elem_bytes = 2
    num_warps = warps_m * warps_n
    total_threads = num_warps * 64
    wave_size = 64

    tile_k_bytes = tile_k * elem_bytes
    warp_tile_m = tile_m // warps_m
    warp_tile_n = tile_n // warps_n
    m_repeat = warp_tile_m // 16
    num_acc_n = warp_tile_n // 16
    n_accs = m_repeat * num_acc_n
    m_half = m_repeat // 2
    n_half = num_acc_n // 2

    load_bytes = 16
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    bytes_b_per_tile = tile_n * tile_k * elem_bytes
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    num_a_dma = bytes_per_thread_a // load_bytes
    num_b_dma = bytes_per_thread_b // load_bytes

    lds_k_dim = tile_k
    k_blocks16 = tile_k_bytes // 16
    tile_k_dwords = tile_k_bytes // 4
    lds_a_elems = tile_m * tile_k
    lds_b_elems = tile_n * tile_k
    lds_a_bytes = lds_a_elems * elem_bytes
    lds_b_bytes = lds_b_elems * elem_bytes

    # mi-tile stride = 16 * lds_k_dim = 512 elements = 1024 bytes
    _mi_tile_stride = 16 * lds_k_dim

    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _use_mfma_k32 = _is_gfx950
    _out_is_bf16 = out_dtype == "bf16"

    def _out_elem():
        return T.bf16 if _out_is_bf16 else T.f16

    # --- 2 SmemAllocators: pong/ping, each holds 2 K-tiles (a_k0 + b_k0 + a_k1 + b_k1) ---
    # Total per stage = 4 * 16KB = 64KB. Two stages = 128KB < 160KB.
    # All sub-regions in one allocator share the same LDS base → 1 base VGPR per stage.
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    def _alloc_stage(alloc):
        """Allocate 4 sub-regions: a_k0, b_k0, a_k1, b_k1. Returns offsets."""
        a0 = alloc._align(alloc.ptr, 16); alloc.ptr = a0 + lds_a_bytes
        b0 = alloc._align(alloc.ptr, 16); alloc.ptr = b0 + lds_b_bytes
        a1 = alloc._align(alloc.ptr, 16); alloc.ptr = a1 + lds_a_bytes
        b1 = alloc._align(alloc.ptr, 16); alloc.ptr = b1 + lds_b_bytes
        return a0, b0, a1, b1

    pong_a0_off, pong_b0_off, pong_a1_off, pong_b1_off = _alloc_stage(allocator_pong)
    ping_a0_off, ping_b0_off, ping_a1_off, ping_b1_off = _alloc_stage(allocator_ping)

    _tile_k_div4 = tile_k * elem_bytes // 4
    num_k_tiles = K // tile_k
    num_k_pairs = num_k_tiles // 2

    # Sub-region element offsets within each stage (for ds_read offset computation)
    # These become constant ds_read byte offsets when multiplied by elem_bytes.
    _off_a0 = pong_a0_off // elem_bytes  # 0
    _off_b0 = pong_b0_off // elem_bytes  # lds_a_elems
    _off_a1 = pong_a1_off // elem_bytes  # lds_a_elems + lds_b_elems
    _off_b1 = pong_b1_off // elem_bytes  # lds_a_elems + lds_b_elems + lds_a_elems

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def hk_bf16_gemm_v2(arg_c, arg_a, arg_b, i32_m, i32_n):
        c_m = arith.index_cast(T.index, i32_m)
        c_n = arith.index_cast(T.index, i32_n)
        tx = gpu.thread_id("x")

        # ---- XCD remap + WGM grouping ----
        _raw_wgid = gpu.block_id("x")
        _NUM_XCDS = 8; _CHUNK = 64
        _num_pid_m = (M + tile_m - 1) // tile_m if M > 0 else 1
        _num_pid_n = N // tile_n if N > 0 else 1
        _NUM_WGS = _num_pid_m * _num_pid_n
        _block = _NUM_XCDS * _CHUNK
        _limit = (_NUM_WGS // _block) * _block
        _xcd = _raw_wgid & (_NUM_XCDS - 1)
        _local_pid = _raw_wgid >> 3
        _chunk_idx = _local_pid >> 6
        _pos_in_chunk = _local_pid & (_CHUNK - 1)
        _new_wgid = _chunk_idx * _block + _xcd * _CHUNK + _pos_in_chunk
        from flydsl._mlir.dialects._arith_enum_gen import CmpIPredicate
        _wgid = arith.select(
            arith.cmpi(CmpIPredicate.sgt, _raw_wgid, arith.index(_limit)),
            _raw_wgid, _new_wgid)
        _WGM = 8
        _num_wgid_in_group = _WGM * _num_pid_n
        _group_id = _wgid // _num_wgid_in_group
        _first_pid_m = _group_id * _WGM
        _local_id = _wgid % _num_wgid_in_group
        _group_size_m = arith.minsi(
            arith.index(_num_pid_m) - _first_pid_m, arith.index(_WGM))
        bx = _first_pid_m + arith.remsi(_local_id, _group_size_m)
        by = arith.divsi(_local_id, _group_size_m)

        # ---- LDS memrefs ----
        # ONE get_base() per allocator — all memrefs share the same base pointer
        # This ensures LLVM knows DMA writes and ds_reads alias correctly (no spurious vmcnt)
        base_pong = allocator_pong.get_base()
        base_ping = allocator_ping.get_base()

        # Sub-region memrefs for DMA (separate shapes for m0 computation)
        lds_pong_a0 = SmemPtr(base_pong, pong_a0_off, T.bf16, shape=(lds_a_elems,)).get()
        lds_pong_b0 = SmemPtr(base_pong, pong_b0_off, T.bf16, shape=(lds_b_elems,)).get()
        lds_pong_a1 = SmemPtr(base_pong, pong_a1_off, T.bf16, shape=(lds_a_elems,)).get()
        lds_pong_b1 = SmemPtr(base_pong, pong_b1_off, T.bf16, shape=(lds_b_elems,)).get()
        lds_ping_a0 = SmemPtr(base_ping, ping_a0_off, T.bf16, shape=(lds_a_elems,)).get()
        lds_ping_b0 = SmemPtr(base_ping, ping_b0_off, T.bf16, shape=(lds_b_elems,)).get()
        lds_ping_a1 = SmemPtr(base_ping, ping_a1_off, T.bf16, shape=(lds_a_elems,)).get()
        lds_ping_b1 = SmemPtr(base_ping, ping_b1_off, T.bf16, shape=(lds_b_elems,)).get()

        # ---- Buffer resources ----
        _a_nrec = arith.index_cast(T.i64, c_m * (K * elem_bytes))
        _b_nrec = arith.index_cast(T.i64, c_n * (K * elem_bytes))
        _c_nrec = arith.index_cast(T.i64, c_m * c_n * 2)
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False, num_records_bytes=_a_nrec)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=False, num_records_bytes=_b_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False, num_records_bytes=_c_nrec)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane decomposition ----
        layout_wave_lane = fx.make_layout((num_warps, wave_size), (wave_size, 1))
        coord_wl = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        layout_wave2d = fx.make_layout((warps_m, warps_n), (warps_n, 1))
        coord_wave2d = fx.idx2crd(wave_id, layout_wave2d)
        wave_m_id = fx.get(coord_wave2d, 0)
        wave_n_id = fx.get(coord_wave2d, 1)
        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        col_offset_base_bytes = lane_div_16 * 8 * elem_bytes
        row_a_base = wave_m_id * warp_tile_m + lane_mod_16
        row_b_base = wave_n_id * warp_tile_n + lane_mod_16

        # ---- DMA helpers ----
        from flydsl._mlir.dialects import memref as memref_dialect
        c4 = fx.Index(4)
        dma_bytes = 16 if _is_gfx950 else 4
        dma_dwords = dma_bytes // 4
        tx_dma_base = tx * dma_dwords
        layout_a_dma = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        layout_b_dma = fx.make_layout((tile_n, tile_k_dwords), (tile_k_dwords, 1))
        _K_bytes = K * elem_bytes
        _lds_k_dim_c = fx.Index(lds_k_dim)
        _k_blocks16_c = arith.index(k_blocks16)

        def _dma_tile(rsrc, base_row, base_k_div4, lds_buffer, layout_dma, num_dma,
                     lds_byte_offset=0):
            """lds_byte_offset: additional byte offset within the full stage memref."""
            lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
            if lds_byte_offset > 0:
                lds_base_idx = lds_base_idx + arith.index(lds_byte_offset)
            wave_off = rocdl.readfirstlane(
                T.i64, arith.index_cast(T.i64, wave_id * arith.index(wave_size * dma_bytes)))
            for i in _range_constexpr(num_dma):
                coord = tile_chunk_coord_i32(
                    arith, tx_i32_base=tx_dma_base, i=i,
                    total_threads=total_threads, layout_tile_div4=layout_dma,
                    chunk_i32=dma_dwords)
                row_local, col_dw = coord
                col_bytes = col_dw * c4
                col_swz = swizzle_xor16(row_local, col_bytes, _k_blocks16_c)
                row_global = base_row + row_local
                global_byte_idx = row_global * _K_bytes + (base_k_div4 * c4 + col_swz)
                global_offset = arith.index_cast(T.i32, global_byte_idx)
                if i == 0:
                    lds_ptr_base = buffer_ops.create_llvm_ptr(
                        arith.index_cast(T.i64, lds_base_idx), address_space=3)
                    lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, wave_off)
                else:
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr, static_byte_offset=total_threads * dma_bytes)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, lds_ptr,
                    arith.constant(dma_bytes, type=T.i32),
                    global_offset,
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                    arith.constant(0, type=T.i32),
                )

        def _dma_k_pair(k_pair_div4, lds_full):
            """DMA 2 K-tiles into one full-stage memref's 4 sub-regions."""
            k0 = k_pair_div4
            k1 = k_pair_div4 + _tile_k_div4
            _dma_tile(a_rsrc, bx_m, k0, lds_full, layout_a_dma, num_a_dma,
                      lds_byte_offset=pong_a0_off)
            _dma_tile(b_rsrc, by_n, k0, lds_full, layout_b_dma, num_b_dma,
                      lds_byte_offset=pong_b0_off)
            _dma_tile(a_rsrc, bx_m, k1, lds_full, layout_a_dma, num_a_dma,
                      lds_byte_offset=pong_a1_off)
            _dma_tile(b_rsrc, by_n, k1, lds_full, layout_b_dma, num_b_dma,
                      lds_byte_offset=pong_b1_off)

        def dma_pong(k_pair_div4):
            _dma_k_pair(k_pair_div4, lds_pong_full)

        def dma_ping(k_pair_div4):
            _dma_k_pair(k_pair_div4, lds_ping_full)

        def _make_dma_interleaved(k_pair_div4, lds_full):
            """Return dma_fn(phase_idx) issuing 2 DMA per phase through the SAME memref."""
            k0 = k_pair_div4
            k1 = k_pair_div4 + _tile_k_div4
            def _dma_phase(phase_idx):
                if phase_idx == 0:
                    _dma_tile(a_rsrc, bx_m, k0, lds_full, layout_a_dma, num_a_dma,
                              lds_byte_offset=pong_a0_off)
                elif phase_idx == 1:
                    _dma_tile(b_rsrc, by_n, k0, lds_full, layout_b_dma, num_b_dma,
                              lds_byte_offset=pong_b0_off)
                elif phase_idx == 2:
                    _dma_tile(a_rsrc, bx_m, k1, lds_full, layout_a_dma, num_a_dma,
                              lds_byte_offset=pong_a1_off)
                elif phase_idx == 3:
                    _dma_tile(b_rsrc, by_n, k1, lds_full, layout_b_dma, num_b_dma,
                              lds_byte_offset=pong_b1_off)
            return _dma_phase

        # ---- LDS read: use ONE stage memref for ALL sub-regions ----
        # All reads from one stage share the same base VGPR.
        # Sub-region offsets (a0=0, b0=lds_a_elems, a1=..., b1=...) become
        # constant element offsets added to the row-based index.

        # Stage-wide memref (all 4 sub-regions in one memref)
        _stage_elems = allocator_pong.ptr // elem_bytes  # total elements per stage

        # Full-stage memrefs for ds_reads — SAME base_pong/base_ping (no duplicate get_base!)
        lds_pong_full = SmemPtr(base_pong, 0, T.bf16, shape=(_stage_elems,)).get()
        lds_ping_full = SmemPtr(base_ping, 0, T.bf16, shape=(_stage_elems,)).get()

        # Element offsets for each sub-region within the stage
        _off_a0 = pong_a0_off // elem_bytes  # 0
        _off_b0 = pong_b0_off // elem_bytes  # lds_a_elems = 8192
        _off_a1 = pong_a1_off // elem_bytes  # 16384
        _off_b1 = pong_b1_off // elem_bytes  # 24576

        def _lds_load_pack(base_idx, lds_stage_full, const_elem_off):
            """Load bf16x8 from stage memref at base_idx + const_elem_off.
            base_idx: dynamic VGPR (row_base * lds_k_dim + col_swz, computed ONCE)
            const_elem_off: Python int constant (sub_region + mi * 16 * lds_k_dim)
            """
            idx = base_idx + const_elem_off
            loaded = vector.load_op(T.bf16x8, lds_stage_full, [idx])
            v_i64x2 = vector.bitcast(T.i64x2, loaded)
            a0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        # ---- MFMA ----
        def _make_mfma():
            mfma_res_ty = T.f32x4
            if _use_mfma_k32:
                mfma_fn = rocdl.mfma_f32_16x16x32_bf16
                def _v8(lo, hi):
                    return vector.bitcast(T.bf16x8, vector.from_elements(T.i64x2, [lo, hi]))
                def k32(acc, a0, a1, b0, b1):
                    return mfma_fn(mfma_res_ty, [_v8(a0, a1), _v8(b0, b1), acc, 0, 0, 0])
            else:
                mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
                def _v4(x):
                    return vector.bitcast(T.i16x4, vector.from_elements(T.vec(1, T.i64), [x]))
                def k32(acc, a0, a1, b0, b1):
                    mid = mfma_fn(mfma_res_ty, [_v4(a0), _v4(b0), acc, 0, 0, 0])
                    return mfma_fn(mfma_res_ty, [_v4(a1), _v4(b1), mid, 0, 0, 0])
            return k32

        mfma_k32 = _make_mfma()

        # ---- HK compute: 4 phases with data reuse across phases ----
        # Phase 0: load A_half0 (4×2k=8) + B_half0 (2×2k=4) = 12 reads, 16 MFMAs
        # Phase 1: reuse A_half0, load B_half1 (2×2k=4) = 4 reads, 16 MFMAs
        # Phase 2: load A_half1 (4×2k=8), reuse B_half0 = 8 reads, 16 MFMAs
        # Phase 3: reuse A_half1, reuse B_half1 = 0 reads, 16 MFMAs
        # Total: 24 reads per K-pair (matches HK), not 48.

        def _mfma_phase(accs, a_packs_k0, a_packs_k1, b_packs_k0, b_packs_k1,
                        mi_off, ni_off, mi_cnt, ni_cnt):
            """16 MFMAs: 8 from k0 + 8 from k1, using pre-loaded A/B packs."""
            current_accs = list(accs)
            rocdl.s_setprio(1)
            for mi_local in _range_constexpr(mi_cnt):
                a0, a1 = a_packs_k0[mi_local]
                for ni_local in _range_constexpr(ni_cnt):
                    b0, b1 = b_packs_k0[ni_local]
                    mi = mi_off + mi_local
                    ni = ni_off + ni_local
                    current_accs[mi * num_acc_n + ni] = mfma_k32(
                        current_accs[mi * num_acc_n + ni], a0, a1, b0, b1)
            for mi_local in _range_constexpr(mi_cnt):
                a0, a1 = a_packs_k1[mi_local]
                for ni_local in _range_constexpr(ni_cnt):
                    b0, b1 = b_packs_k1[ni_local]
                    mi = mi_off + mi_local
                    ni = ni_off + ni_local
                    current_accs[mi * num_acc_n + ni] = mfma_k32(
                        current_accs[mi * num_acc_n + ni], a0, a1, b0, b1)
            rocdl.s_setprio(0)
            return current_accs

        # Pre-compute base indices ONCE — these are the 2 dynamic VGPRs shared by ALL reads.
        # Swizzle is invariant across mi-tiles because (mi*16) % k_blocks16 == 0.
        def _compute_base_idx(row_base):
            col_swz_bytes = swizzle_xor16(row_base, col_offset_base_bytes, _k_blocks16_c)
            col_swz_elem = col_swz_bytes // 2
            return row_base * _lds_k_dim_c + col_swz_elem

        def _load_b_half(ni_off, lds_full, b_base_idx):
            """Load n_half B packs from both K-tiles. 2+2=4 reads.
            B is loaded FIRST (HK pattern: src1 data read before src0)."""
            b0 = []; b1 = []
            for ni_local in _range_constexpr(n_half):
                off = (ni_off + ni_local) * _mi_tile_stride
                b0.append(_lds_load_pack(b_base_idx, lds_full, _off_b0 + off))
                b1.append(_lds_load_pack(b_base_idx, lds_full, _off_b1 + off))
            return b0, b1

        def _load_a_half(mi_off, lds_full, a_base_idx):
            """Load m_half A packs from both K-tiles. 4+4=8 reads.
            A is loaded SECOND (after B, matching HK pattern)."""
            a0 = []; a1 = []
            for mi_local in _range_constexpr(m_half):
                off = (mi_off + mi_local) * _mi_tile_stride
                a0.append(_lds_load_pack(a_base_idx, lds_full, _off_a0 + off))
                a1.append(_lds_load_pack(a_base_idx, lds_full, _off_a1 + off))
            return a0, a1

        # HK lgkmcnt(8): wait for ds_reads, leave 8 DMA events pending
        # Encoding: vmcnt=63(max), expcnt=7(max), lgkmcnt=8
        _LGKMCNT_8 = (0x3 << 14) | (8 << 8) | (0x7 << 4) | 0xF  # 0xC87F

        def _wait_reads_then_barrier(has_dma):
            """HK waitcnt pattern: lgkmcnt(N) → barrier → lgkmcnt(0)."""
            if has_dma:
                rocdl.s_waitcnt(_LGKMCNT_8)  # wait for ds_reads, DMA still pending
            rocdl.s_barrier()
            rocdl.s_waitcnt(0xC07F)  # lgkmcnt(0) — drain all

        def compute_tile_2k(accs, lds_full, dma_fn=None):
            """4 phases × 16 MFMAs = 64 MFMAs, with data reuse + LDS prefetch.
            HK pattern: next-phase ds_reads issued BEFORE inter-phase barrier,
            so reads overlap with barrier sync and complete before MFMAs.
            All ds_reads from ONE stage memref → shared base VGPR.
            """
            # Compute 2 base index VGPRs ONCE — shared across ALL phases and reads
            a_base_idx = _compute_base_idx(row_a_base)
            b_base_idx = _compute_base_idx(row_b_base)

            # Phase 0: 12 reads (B first, then A — HK pattern) + 2 DMA + MFMAs
            b_h0_k0, b_h0_k1 = _load_b_half(0, lds_full, b_base_idx)
            a_h0_k0, a_h0_k1 = _load_a_half(0, lds_full, a_base_idx)
            rocdl.sched_barrier(0)
            if dma_fn is not None:
                dma_fn(0)
            rocdl.sched_barrier(0)
            _wait_reads_then_barrier(dma_fn is not None)
            accs = _mfma_phase(accs, a_h0_k0, a_h0_k1, b_h0_k0, b_h0_k1,
                               0, 0, m_half, n_half)

            # Prefetch Phase 1 B_half1 reads (4) BEFORE inter-phase barrier
            b_h1_k0, b_h1_k1 = _load_b_half(n_half, lds_full, b_base_idx)
            rocdl.sched_barrier(0)
            if dma_fn is not None:
                dma_fn(1)
            rocdl.sched_barrier(0)

            # Phase 1: barrier (reads already in-flight) + MFMAs
            _wait_reads_then_barrier(dma_fn is not None)
            accs = _mfma_phase(accs, a_h0_k0, a_h0_k1, b_h1_k0, b_h1_k1,
                               0, n_half, m_half, n_half)

            # Prefetch Phase 2 A_half1 reads (8) BEFORE inter-phase barrier
            a_h1_k0, a_h1_k1 = _load_a_half(m_half, lds_full, a_base_idx)
            rocdl.sched_barrier(0)
            if dma_fn is not None:
                dma_fn(2)
            rocdl.sched_barrier(0)

            # Phase 2: barrier (reads in-flight) + MFMAs, reuse B_half0
            _wait_reads_then_barrier(dma_fn is not None)
            accs = _mfma_phase(accs, a_h1_k0, a_h1_k1, b_h0_k0, b_h0_k1,
                               m_half, 0, m_half, n_half)

            # Phase 3: no new reads needed (reuse A_half1 + B_half1), just DMA
            if dma_fn is not None:
                dma_fn(3)
                rocdl.sched_barrier(0)
                _wait_reads_then_barrier(True)
            else:
                rocdl.s_barrier()
            accs = _mfma_phase(accs, a_h1_k0, a_h1_k1, b_h1_k0, b_h1_k1,
                               m_half, n_half, m_half, n_half)
            return accs

        # ---- Epilogue ----
        def store_output(final_accs):
            m_base = bx_m + wave_m_id * warp_tile_m
            n_base = by_n + wave_n_id * warp_tile_n
            row_off = lane_div_16 * 4
            col_off = lane_mod_16
            for mi in _range_constexpr(m_repeat):
                col_base = n_base + col_off
                for ii in _range_constexpr(4):
                    row_g = m_base + (mi * 16) + row_off + ii
                    idx_base = row_g * c_n + col_base
                    for ni in _range_constexpr(num_acc_n):
                        acc_val = final_accs[mi * num_acc_n + ni]
                        val_f32 = vector.extract(acc_val, static_position=[ii], dynamic_position=[])
                        val_out = arith.trunc_f(_out_elem(), val_f32)
                        buffer_ops.buffer_store(val_out, c_rsrc, idx_base + (ni * 16))

        # ---- Main loop: 2-stage ping-pong, 2 K-tiles per stage ----
        zero_f32x4 = vector.broadcast(T.f32x4, arith.constant(0.0, type=T.f32))
        init_accs = [zero_f32x4 for _ in range(n_accs)]

        # Prologue: DMA K-pair 0 into pong
        dma_pong(arith.index(0))
        rocdl.s_waitcnt(0xC07F)  # lgkmcnt(0)
        rocdl.s_barrier()

        _2k = 2 * _tile_k_div4  # stride per K-pair in div4 units

        if num_k_pairs <= 1:
            accs = compute_tile_2k(init_accs, lds_pong_full)
        elif num_k_pairs == 2:
            dma_ping(arith.index(1 * _2k))
            accs = compute_tile_2k(init_accs, lds_pong_full)
            rocdl.s_waitcnt(0xC07F)
            rocdl.s_barrier()
            accs = compute_tile_2k(accs, lds_ping_full)
        else:
            # General case: 2x unroll (128 MFMAs per scf.for iteration = matches HK)
            num_loop_iters = (num_k_pairs - 1) // 2
            has_tail = (num_k_pairs - 1) % 2 == 1
            init_state = init_accs
            results = init_state

            for pair_iv, inner in fx.range(0, num_loop_iters, 1, init=init_state):
                accs_in = list(inner)
                k_ping = (2 * pair_iv + 1) * _2k
                k_pong = (2 * pair_iv + 2) * _2k

                # Compute from pong, interleave DMA→ping (same memref for read+write)
                dma_to_ping = _make_dma_interleaved(k_ping, lds_ping_full)
                accs_out = compute_tile_2k(accs_in, lds_pong_full, dma_fn=dma_to_ping)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()

                # Compute from ping, interleave DMA→pong
                dma_to_pong = _make_dma_interleaved(k_pong, lds_pong_full)
                accs_out = compute_tile_2k(accs_out, lds_ping_full, dma_fn=dma_to_pong)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()

                results = yield accs_out

            accs = list(results)

            if has_tail:
                tail_k = arith.index((num_k_pairs - 1) * _2k)
                dma_ping(tail_k)
                accs = compute_tile_2k(accs, lds_pong_full)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()
                accs = compute_tile_2k(accs, lds_ping_full)
            else:
                accs = compute_tile_2k(accs, lds_pong_full)

        store_output(accs)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        i32_m: fx.Int32, i32_n: fx.Int32, stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n
        launcher = hk_bf16_gemm_v2(arg_c, arg_a, arg_b, i32_m, i32_n)
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        launcher.launch(grid=(gx * gy, 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_gemm
