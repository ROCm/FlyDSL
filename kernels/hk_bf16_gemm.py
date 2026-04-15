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

    # --- 2 SmemAllocators: pong/ping ---
    # Each stage: A_interleaved(32KB) + B_interleaved(32KB) = 64KB.
    # A_interleaved layout: [mi0_k0(1024B), mi0_k1(1024B), mi1_k0(1024B), ...]
    # This gives ds_read offsets 0, 1024, 2048, ..., 7168 from ONE base (HK pattern).
    # Total: 2 stages × 64KB = 128KB < 160KB.
    _mi_block_bytes = 16 * tile_k * elem_bytes  # 1024
    _a_interleaved_bytes = m_repeat * 2 * _mi_block_bytes  # 16 mi × 2 k × 1024 = 32KB
    _b_interleaved_bytes = num_acc_n * 2 * _mi_block_bytes  # 4 ni × 2 k × 1024 = 8KB... wait
    # Actually: A has m_repeat mi-tiles, B has num_acc_n ni-tiles (warp-level)
    # But LDS stores the FULL tile (all warps), not per-warp.
    # tile_m / 16 = 16 mi-tiles for A, tile_n / 16 = 16 ni-tiles for B
    _num_mi_tiles = tile_m // 16  # 16
    _num_ni_tiles = tile_n // 16  # 16
    _a_interleaved_bytes = _num_mi_tiles * 2 * _mi_block_bytes  # 16 × 2 × 1024 = 32KB
    _b_interleaved_bytes = _num_ni_tiles * 2 * _mi_block_bytes  # 16 × 2 × 1024 = 32KB

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    def _alloc_stage(alloc):
        """Allocate 2 interleaved regions: A (32KB) + B (32KB)."""
        a_off = alloc._align(alloc.ptr, 16); alloc.ptr = a_off + _a_interleaved_bytes
        b_off = alloc._align(alloc.ptr, 16); alloc.ptr = b_off + _b_interleaved_bytes
        return a_off, b_off

    pong_a_off, pong_b_off = _alloc_stage(allocator_pong)
    ping_a_off, ping_b_off = _alloc_stage(allocator_ping)

    _tile_k_div4 = tile_k * elem_bytes // 4
    num_k_tiles = K // tile_k
    num_k_pairs = num_k_tiles // 2

    # Interleaved element offsets: k0 at even blocks, k1 at odd blocks
    # mi-tile stride = 2 * _mi_block_bytes / elem_bytes = 2 * 512 = 1024 elements = 2048 bytes
    # k-chunk offset = _mi_block_bytes / elem_bytes = 512 elements = 1024 bytes
    _interleaved_mi_stride = 2 * _mi_block_bytes // elem_bytes  # 1024 elems = 2048 bytes
    _interleaved_k_offset = _mi_block_bytes // elem_bytes  # 512 elems = 1024 bytes
    # B region starts after A
    _b_region_elem_off = _a_interleaved_bytes // elem_bytes  # 16384 elems

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

        # ---- LDS memrefs (interleaved layout: 2 regions per stage) ----
        # ONE base per allocator. A and B each get one memref for ds_reads → 2 base VGPRs.
        base_pong = allocator_pong.get_base()
        base_ping = allocator_ping.get_base()

        _a_interleaved_elems = _a_interleaved_bytes // elem_bytes  # 16384
        _b_interleaved_elems = _b_interleaved_bytes // elem_bytes  # 16384

        # ONE memref per allocator (like preshuffle). A/B offset encoded in index, not memref view.
        _stage_elems = allocator_pong.ptr // elem_bytes
        lds_pong = SmemPtr(base_pong, 0, T.bf16, shape=(_stage_elems,)).get()
        lds_ping = SmemPtr(base_ping, 0, T.bf16, shape=(_stage_elems,)).get()

        # B region element offset within stage (A starts at 0)
        _b_elem_off = pong_b_off // elem_bytes

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

        # ---- DMA: interleaved k0/k1 layout ----
        # Thread mapping for interleaved DMA:
        # Sequential LDS position T*16 maps to interleaved block:
        #   block = T*16 / 1024 = T // 64
        #   mi = block // 2
        #   k_idx = block % 2
        #   local_pos = (T*16 % 1024) / 16 = T % 64
        #   row_in_mi = local_pos // 4
        #   k_group = local_pos % 4
        # Global read: row = base_row + mi*16 + row_in_mi
        #              K_pos = k_start + k_idx * tile_k + k_group * 8
        from flydsl._mlir.dialects import memref as memref_dialect
        c4 = fx.Index(4)
        dma_bytes = 16 if _is_gfx950 else 4
        dma_dwords = dma_bytes // 4
        _K_bytes = K * elem_bytes
        _lds_k_dim_c = fx.Index(lds_k_dim)
        _k_blocks16_c = arith.index(k_blocks16)

        # Interleaved DMA: loads both k0 and k1 for one matrix in interleaved order.
        # 512 threads × 4 loads × 16 bytes = 32KB per matrix (covers both K-tiles).
        _num_interleaved_dma = 4  # 32KB / (512 * 16) = 4 loads per thread

        def _dma_matrix_interleaved(rsrc, base_row, k_pair_div4, lds_buf, dim_tiles,
                                    lds_byte_offset=0):
            """DMA both k0+k1 for one matrix into interleaved LDS layout.
            dim_tiles: number of mi/ni tiles (tile_m/16 or tile_n/16).
            LDS layout: [mi0_k0(1024B), mi0_k1(1024B), mi1_k0(1024B), ...]
            """
            lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_buf)
            if lds_byte_offset > 0:
                lds_base_idx = lds_base_idx + arith.index(lds_byte_offset)
            wave_off = rocdl.readfirstlane(
                T.i64, arith.index_cast(T.i64, wave_id * arith.index(wave_size * dma_bytes)))

            k0_div4 = k_pair_div4
            k1_div4 = k_pair_div4 + _tile_k_div4
            _K_bytes_idx = fx.Index(_K_bytes)

            for i in _range_constexpr(_num_interleaved_dma):
                # Sequential position in LDS
                seq_pos = tx * dma_dwords + i * total_threads * dma_dwords
                # Decompose: block → (mi, k_idx), within block → (row_in_mi, k_group)
                block_dwords = _mi_block_bytes // 4  # 256 dwords per block
                block_id = seq_pos // block_dwords
                local_dw = seq_pos % block_dwords
                mi = block_id // 2
                k_idx = block_id % 2
                row_in_mi = local_dw // tile_k_dwords  # local_dw // 16
                k_group_dw = local_dw % tile_k_dwords  # local_dw % 16

                # Global address
                row_global = base_row + mi * 16 + row_in_mi
                col_bytes = k_group_dw * c4
                # k0 or k1 offset
                k_base = k0_div4 + k_idx * _tile_k_div4
                global_byte_idx = row_global * _K_bytes_idx + (k_base * c4 + col_bytes)
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

        def dma_pong(k_pair_div4):
            _dma_matrix_interleaved(a_rsrc, bx_m, k_pair_div4, lds_pong, _num_mi_tiles)
            _dma_matrix_interleaved(b_rsrc, by_n, k_pair_div4, lds_pong, _num_ni_tiles,
                                    lds_byte_offset=pong_b_off)

        def dma_ping(k_pair_div4):
            _dma_matrix_interleaved(a_rsrc, bx_m, k_pair_div4, lds_ping, _num_mi_tiles)
            _dma_matrix_interleaved(b_rsrc, by_n, k_pair_div4, lds_ping, _num_ni_tiles,
                                    lds_byte_offset=ping_b_off)

        def _make_dma_interleaved(k_pair_div4, lds_buf):
            """Return dma_fn(phase_idx) issuing DMA loads spread across phases."""
            # Determine B offset based on which stage this is
            b_off = pong_b_off if lds_buf is lds_pong else ping_b_off
            def _dma_phase(phase_idx):
                if phase_idx == 0:
                    _dma_matrix_interleaved(a_rsrc, bx_m, k_pair_div4, lds_buf, _num_mi_tiles)
                elif phase_idx == 1:
                    _dma_matrix_interleaved(b_rsrc, by_n, k_pair_div4, lds_buf, _num_ni_tiles,
                                            lds_byte_offset=b_off)
            return _dma_phase

        # ---- LDS read: use ONE stage memref for ALL sub-regions ----
        # All reads from one stage share the same base VGPR.
        # Sub-region offsets (a0=0, b0=lds_a_elems, a1=..., b1=...) become
        # constant element offsets added to the row-based index.

        # Stage-wide memref (all 4 sub-regions in one memref)

        # Full-stage memrefs for ds_reads — SAME base_pong/base_ping (no duplicate get_base!)

        # Element offsets for each sub-region within the stage


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

        # Sub-region offsets expressed as virtual row offsets (divisible by lds_k_dim).
        # Adding these to row_base before row*lds_k_dim+col_swz lets LLVM fold them
        # into ds_read offset field (same pattern as preshuffle_gemm).

        def _lds_load_pack(row, lds_buf):
            """Load bf16x8 from lds_buf at row * lds_k_dim + col_base.
            No XOR swizzle — linear row-major. Within-wave bank conflicts = 0.
            Cross-wave conflicts (occupancy=2) are unavoidable regardless of swizzle.
            LLVM can fold mi*16*lds_k_dim into ds_read offset → 2 base VGPRs.
            """
            col_base_elem = col_offset_base_bytes // 2
            idx = row * _lds_k_dim_c + col_base_elem
            loaded = vector.load_op(T.bf16x8, lds_buf, [idx])
            v_i64x2 = vector.bitcast(T.i64x2, loaded)
            a0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        # Interleaved offsets: k0 at even mi-tile positions, k1 at odd
        # For mi-tile i: k0 at i * 2 * _mi_tile_stride, k1 at i * 2 * _mi_tile_stride + _mi_tile_stride
        # This gives offsets 0, 512, 1024, 1536, 2048, ... (in elements)
        # = 0, 1024, 2048, 3072, 4096, ... (in bytes) — matches HK!

        # In interleaved layout, virtual row for (mi_tile, k_idx):
        #   virtual_row = mi * 32 + k_idx * 16 + lane_mod_16
        # where 32 = 2 blocks × 16 rows/block (k0+k1 per mi-tile)
        # This gives ds_read offsets: mi*32*32 + k_idx*16*32 = mi*1024 + k_idx*512
        # In bytes: mi*2048 + k_idx*1024 — matches HK pattern!

        def _load_b_half(ni_off, lds_stage):
            """Load n_half B packs from interleaved LDS. B loaded FIRST."""
            b0 = []; b1 = []
            for ni_local in _range_constexpr(n_half):
                ni = ni_off + ni_local
                # B data starts at _b_elem_off within the stage
                b_row = row_b_base + ni * 32 + _b_elem_off // lds_k_dim
                b0.append(_lds_load_pack(b_row, lds_stage))
                b1.append(_lds_load_pack(b_row + 16, lds_stage))
            return b0, b1

        def _load_a_half(mi_off, lds_stage):
            """Load m_half A packs from interleaved LDS. A loaded SECOND."""
            a0 = []; a1 = []
            for mi_local in _range_constexpr(m_half):
                mi = mi_off + mi_local
                # A data starts at offset 0 within the stage
                a_row = row_a_base + mi * 32
                a0.append(_lds_load_pack(a_row, lds_stage))
                a1.append(_lds_load_pack(a_row + 16, lds_stage))
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

        def compute_tile_2k(accs, lds_stage, dma_fn=None):
            """4 phases × 16 MFMAs = 64 MFMAs, with data reuse + LDS prefetch.
            HK pattern: next-phase ds_reads issued BEFORE inter-phase barrier,
            so reads overlap with barrier sync and complete before MFMAs.
            All ds_reads from ONE stage memref → shared base VGPR.
            """
            # Phase 0: 12 reads (B first, then A — HK pattern) + 2 DMA + MFMAs
            b_h0_k0, b_h0_k1 = _load_b_half(0, lds_stage)
            a_h0_k0, a_h0_k1 = _load_a_half(0, lds_stage)
            rocdl.sched_barrier(0)
            if dma_fn is not None:
                dma_fn(0)
            rocdl.sched_barrier(0)
            _wait_reads_then_barrier(dma_fn is not None)
            accs = _mfma_phase(accs, a_h0_k0, a_h0_k1, b_h0_k0, b_h0_k1,
                               0, 0, m_half, n_half)

            # Prefetch Phase 1 B_half1 reads (4) BEFORE inter-phase barrier
            b_h1_k0, b_h1_k1 = _load_b_half(n_half, lds_stage)
            rocdl.sched_barrier(0)
            if dma_fn is not None:
                dma_fn(1)
            rocdl.sched_barrier(0)

            # Phase 1: barrier (reads already in-flight) + MFMAs
            _wait_reads_then_barrier(dma_fn is not None)
            accs = _mfma_phase(accs, a_h0_k0, a_h0_k1, b_h1_k0, b_h1_k1,
                               0, n_half, m_half, n_half)

            # Prefetch Phase 2 A_half1 reads (8) BEFORE inter-phase barrier
            a_h1_k0, a_h1_k1 = _load_a_half(m_half, lds_stage)
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
            accs = compute_tile_2k(init_accs, lds_pong)
        elif num_k_pairs == 2:
            dma_ping(arith.index(1 * _2k))
            accs = compute_tile_2k(init_accs, lds_pong)
            rocdl.s_waitcnt(0xC07F)
            rocdl.s_barrier()
            accs = compute_tile_2k(accs, lds_ping)
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
                dma_to_ping = _make_dma_interleaved(k_ping, lds_ping)
                accs_out = compute_tile_2k(accs_in, lds_pong, dma_fn=dma_to_ping)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()

                # Compute from ping, interleave DMA→pong
                dma_to_pong = _make_dma_interleaved(k_pong, lds_pong)
                accs_out = compute_tile_2k(accs_out, lds_ping, dma_fn=dma_to_pong)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()

                results = yield accs_out

            accs = list(results)

            if has_tail:
                tail_k = arith.index((num_k_pairs - 1) * _2k)
                dma_ping(tail_k)
                accs = compute_tile_2k(accs, lds_pong)
                rocdl.s_waitcnt(0xC07F)
                rocdl.s_barrier()
                accs = compute_tile_2k(accs, lds_ping)
            else:
                accs = compute_tile_2k(accs, lds_pong)

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

    return flyc.compile(launch_gemm)
