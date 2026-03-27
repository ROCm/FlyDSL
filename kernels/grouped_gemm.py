# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Grouped FP8 GEMM kernel (M-grouped contiguous layout).

API matching DeepGEMM's m_grouped_fp8_gemm_nt_contiguous:
  - A: [M_total, K] FP8 - concatenated rows from all groups
  - scale_a: [scale_k, M_total] FP32 - per-token, per-128K scales (transposed)
  - B: [num_groups, N, K] FP8 - one weight matrix per group
  - scale_b: [num_groups, scale_n, scale_k] FP32 - per-block scales
  - D: [M_total, N] BF16 - output
  - grouped_layout: [M_total] INT32 - maps each row to group ID (-1 for padding)

Block scaling granularity (matching DeepGEMM):
  - A: (1, 128) - per-token, per-128-K-elements
  - B: (128, 128) - per-128-N, per-128-K block

This is Step 0 (baseline): single-buffered LDS, no advanced optimizations.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import math as math_dialect
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue

from kernels.mfma_preshuffle_pipeline import crd2idx


@functools.lru_cache(maxsize=128)
def compile_grouped_fp8_gemm(
    *,
    n: int,
    k: int,
    num_groups: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
):
    """Compile grouped FP8 GEMM kernel and return the JIT launcher.

    Args:
        n: N dimension (output columns per group)
        k: K dimension (reduction dimension)
        num_groups: Number of groups (experts)
        tile_m: M tile size (default 128)
        tile_n: N tile size (default 128)
        tile_k: K tile size (default 128)
        scale_block_k: K-dimension scale block size (default 128)
        scale_block_n: N-dimension scale block size (default 128)
        out_dtype: Output data type ("bf16" or "f16")

    Returns:
        JIT launcher function.
    """
    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx95")
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_grouped_gemm")

    # Validate parameters
    if k % tile_k != 0:
        raise ValueError(f"k ({k}) must be divisible by tile_k ({tile_k})")
    if n % tile_n != 0:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_k % scale_block_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by scale_block_k ({scale_block_k})")
    if tile_n % scale_block_n != 0:
        raise ValueError(f"tile_n ({tile_n}) must be divisible by scale_block_n ({scale_block_n})")

    # Output type
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")
    out_mlir = lambda: T.bf16 if out_dtype == "bf16" else T.f16

    # Compile-time constants
    total_threads = 256
    elem_bytes = 1  # FP8
    num_k_tiles = k // tile_k
    scale_k = k // scale_block_k
    scale_n = n // scale_block_n
    sb_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    k_unroll = tile_k // 64  # K64-byte micro-steps (for K32 MFMA pairs)

    # LDS allocation (single-buffered for baseline)
    lds_a_bytes = tile_m * tile_k * elem_bytes
    lds_alloc_bytes = lds_a_bytes
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    # Module name for caching
    module_name = (
        f"grouped_fp8_gemm_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_baseline"
    ).replace("-", "_")

    # Thread -> tile element mapping for A loads
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    bytes_per_thread_a = bytes_a_per_tile // total_threads

    @flyc.kernel(name=module_name)
    def grouped_fp8_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_grouped_layout: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
    ):
        # Convert runtime parameters to index type
        m_in = arith.index_cast(T.index, i32_m)
        n_in = arith.index_cast(T.index, i32_n)
        k_in = arith.index_cast(T.index, i32_k)
        num_groups_in = arith.index_cast(T.index, i32_num_groups)

        # Thread and block IDs
        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-block index

        # Block positions
        bx_m = bx * fx.Index(tile_m)
        by_n = by * fx.Index(tile_n)

        # Wave/lane decomposition (256 threads = 4 waves x 64 lanes)
        layout_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        # Lane decomposition for MFMA (lane_id -> lane_div_16, lane_mod_16)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # LDS setup
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_alloc_offset, T.f8, shape=(tile_m * tile_k,)).get()
        lds_stride = tile_k
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(lds_stride, 1))

        # Buffer resources
        a_nbytes = m_in * k_in
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, a_nbytes)
        )

        b_nbytes = num_groups_in * n_in * k_in
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, b_nbytes)
        )

        d_nbytes = m_in * n_in * fx.Index(2)  # bf16/f16 = 2 bytes
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=arith.index_cast(T.i64, d_nbytes)
        )

        # Scale buffers
        # scale_a: [scale_k, M] - transposed layout
        sa_nbytes = fx.Index(scale_k) * m_in * fx.Index(4)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, sa_nbytes)
        )

        # scale_b: [num_groups, scale_n, scale_k]
        sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * 4)
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, sb_nbytes)
        )

        # grouped_layout: [M]
        gl_nbytes = m_in * fx.Index(4)
        gl_rsrc = buffer_ops.create_buffer_resource(
            arg_grouped_layout, max_size=False, num_records_bytes=arith.index_cast(T.i64, gl_nbytes)
        )

        # Load group ID for this M-block (use first row of tile)
        group_id_i32 = buffer_ops.buffer_load(gl_rsrc, bx_m, vec_width=1, dtype=T.i32)
        is_valid = arith.cmpi(arith.CmpIPredicate.sge, group_id_i32, fx.Int32(0))

        # Early exit for invalid blocks (padding rows)
        _if_valid = scf.IfOp(is_valid)
        with ir.InsertionPoint(_if_valid.then_block):
            group_idx = arith.index_cast(T.index, group_id_i32)

            # MFMA tiling constants
            m_repeat = tile_m // 16  # 8 for tile_m=128
            num_waves = 4
            n_per_wave = tile_n // num_waves  # 32 for tile_n=128
            num_acc_n = n_per_wave // 16  # 2 for n_per_wave=32

            # Initialize accumulators (FP32)
            acc_init = arith.constant_vector(0.0, T.f32x4)
            num_accs = m_repeat * num_acc_n
            accs = [acc_init] * num_accs

            # Wave's N-tile base
            wave_mod_4 = wave_id % fx.Index(4)
            n_tile_base = wave_mod_4 * fx.Index(n_per_wave)

            # Precompute N-block indices for scale_b
            c_scale_block_n = fx.Index(scale_block_n)
            c_scale_k = fx.Index(scale_k)
            n_block_for_scale = []
            for ni in range_constexpr(num_acc_n):
                col_base = by_n + n_tile_base + arith.index(ni * 16)
                n_blk = col_base // c_scale_block_n
                n_block_for_scale.append(n_blk)

            # A load mapping: thread -> (row, col) in tile
            tile_k_div16 = tile_k // 16
            layout_a_tile = fx.make_layout((tile_m, tile_k_div16), stride=(tile_k_div16, 1))
            loads_per_thread = bytes_per_thread_a // 16  # 16-byte loads

            # Main K-loop
            c_scale_block_k = fx.Index(scale_block_k)
            c_tile_k = fx.Index(tile_k)

            for k_tile_idx in range_constexpr(num_k_tiles):
                k_base = fx.Index(k_tile_idx * tile_k)

                # ===== Load A tile to LDS =====
                for load_idx in range_constexpr(loads_per_thread):
                    lin_idx = tx * fx.Index(loads_per_thread) + fx.Index(load_idx)
                    coord = fx.idx2crd(lin_idx, layout_a_tile)
                    row_local = fx.get(coord, 0)
                    col_local_16 = fx.get(coord, 1)
                    col_local = col_local_16 * fx.Index(16)

                    # Global A index
                    row_global = bx_m + row_local
                    a_idx = row_global * k_in + k_base + col_local

                    # Load 16 bytes (16 FP8 elements)
                    a_vec = buffer_ops.buffer_load(a_rsrc, a_idx, vec_width=4, dtype=T.i32)

                    # Store to LDS
                    lds_coord = (row_local, col_local)
                    lds_idx = crd2idx(lds_coord, layout_lds)
                    a_vec_f8 = vector.bitcast(T.vec(16, T.f8), a_vec)
                    vector.store(a_vec_f8, lds_a, [lds_idx])

                gpu.barrier()

                # ===== Compute MFMA tiles =====
                # For each scale block in this K-tile
                for sb in range_constexpr(sb_per_tile):
                    kb = fx.Index(k_tile_idx * sb_per_tile + sb)  # Global K-block index

                    # Load scale_a for this K-block (per-token scale)
                    # scale_a layout: [scale_k, M] transposed
                    sa_base = kb * m_in
                    s_a_vecs = []
                    row_off_base = lane_div_16 * fx.Index(4)
                    for mi in range_constexpr(m_repeat):
                        s_a_row = []
                        for ii in range_constexpr(4):
                            row_in_tile = arith.index(mi * 16) + row_off_base + fx.Index(ii)
                            row_global = bx_m + row_in_tile
                            sa_idx = sa_base + row_global
                            s_a_val = buffer_ops.buffer_load(sa_rsrc, sa_idx, vec_width=1, dtype=T.f32)
                            s_a_row.append(s_a_val)
                        s_a_vec4 = vector.from_elements(T.f32x4, s_a_row)
                        s_a_vecs.append(s_a_vec4)

                    # Load scale_b for this K-block
                    # scale_b layout: [num_groups, scale_n, scale_k]
                    sb_group_offset = group_idx * fx.Index(scale_n * scale_k)
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        sb_idx = sb_group_offset + n_block_for_scale[ni] * c_scale_k + kb
                        s_b_val = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
                        s_b_vals.append(s_b_val)

                    # MFMA computation for this scale block
                    # K64 micro-steps within scale block
                    ku_per_sb = scale_block_k // 64

                    for ku_local in range_constexpr(ku_per_sb):
                        ku = sb * ku_per_sb + ku_local
                        k_offset_bytes = ku * 64  # Byte offset within tile

                        for mi in range_constexpr(m_repeat):
                            # Load A from LDS (16 bytes = 2 x 8 bytes for K32 MFMA pair)
                            row_a_lds = lane_mod_16 + arith.index(mi * 16)
                            col_a_base = lane_div_16 * fx.Index(16) + fx.Index(k_offset_bytes)

                            # Load 16 bytes and split into two i64 for K32 MFMAs
                            lds_coord_a = (row_a_lds, col_a_base)
                            lds_idx_a = crd2idx(lds_coord_a, layout_lds)
                            a16 = vector.load_op(T.vec(16, T.f8), lds_a, [lds_idx_a])
                            a_i64x2 = vector.bitcast(T.i64x2, a16)
                            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni

                                # Load B from global memory
                                # B layout: [num_groups, N, K] with K-major
                                b_group_off = group_idx * (n_in * k_in)
                                b_col = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
                                b_k_base = k_base + fx.Index(k_offset_bytes) + lane_div_16 * fx.Index(16)

                                b_idx0 = b_group_off + b_col * k_in + b_k_base
                                b_idx1 = b_idx0 + fx.Index(8)

                                # Load 8 bytes each for the two K32 MFMAs
                                b0_i32x2 = buffer_ops.buffer_load(b_rsrc, b_idx0, vec_width=2, dtype=T.i32)
                                b1_i32x2 = buffer_ops.buffer_load(b_rsrc, b_idx1, vec_width=2, dtype=T.i32)

                                b0_i64 = vector.extract(
                                    vector.bitcast(T.vec(1, T.i64), b0_i32x2),
                                    static_position=[0], dynamic_position=[]
                                )
                                b1_i64 = vector.extract(
                                    vector.bitcast(T.vec(1, T.i64), b1_i32x2),
                                    static_position=[0], dynamic_position=[]
                                )

                                # Two K32 MFMAs
                                mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
                                mfma_mid = mfma_fn(T.f32x4, [a0, b0_i64, acc_init, 0, 0, 0])
                                mfma_result = mfma_fn(T.f32x4, [a1, b1_i64, mfma_mid, 0, 0, 0])

                                # Apply scales: accum += mfma_result * scale_a * scale_b
                                s_a_v4 = s_a_vecs[mi]
                                s_b_bc = vector.broadcast(T.f32x4, s_b_vals[ni])
                                scaled = ArithValue(mfma_result) * ArithValue(s_a_v4)
                                accs[acc_idx] = math_dialect.fma(scaled, s_b_bc, accs[acc_idx])

                gpu.barrier()

            # ===== Epilogue: store results =====
            c_n = n_in
            lane_div_16_mul4 = lane_div_16 * fx.Index(4)

            for mi in range_constexpr(m_repeat):
                for ii in range_constexpr(4):
                    row_off = lane_div_16_mul4 + fx.Index(ii)
                    row_in_tile = arith.index(mi * 16) + row_off
                    row_global = bx_m + row_in_tile

                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        col_base = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16

                        # Extract scalar from accumulator
                        val_f32 = vector.extract(accs[acc_idx], static_position=[ii], dynamic_position=[])
                        val_out = arith.trunc_f(out_mlir(), val_f32)

                        # Store to D
                        d_idx = row_global * c_n + col_base
                        buffer_ops.buffer_store(val_out, d_rsrc, d_idx)

            scf.YieldOp([])

    # ===== JIT Launcher =====
    @flyc.jit
    def launch_grouped_fp8_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_grouped_layout: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid dimensions
        m_in = arith.index_cast(T.index, i32_m)
        n_in = arith.index_cast(T.index, i32_n)
        gx = n_in // fx.Index(tile_n)  # N-blocks
        gy = (m_in + fx.Index(tile_m - 1)) // fx.Index(tile_m)  # M-blocks (ceil)

        launcher = grouped_fp8_gemm_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_grouped_layout,
            i32_m,
            i32_n,
            i32_k,
            i32_num_groups,
        )
        launcher.launch(grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_grouped_fp8_gemm
