# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Masked Grouped FP8 GEMM kernel (M-grouped masked layout).

API matching DeepGEMM's m_grouped_fp8_gemm_nt_masked:
  - A: [G, expected_m, K] FP8 - padded activation tensor per group
  - scale_a: [G, scale_k, expected_m] - per-token, per-128K scales (transposed).
    uint8 (E8M0) on gfx950 (HW scaling); FP32 on gfx942 (SW scaling).
  - B: [G, N, K] FP8 - one weight matrix per group
  - scale_b: [G, scale_n, scale_k] - per-block scales.
    uint8 (E8M0) on gfx950; FP32 on gfx942.
  - D: [G, expected_m, N] BF16 - padded output tensor per group
  - masked_m: [G] INT32 - tracks the actual number of valid tokens per group
  - expected_m: INT32 - the padded capacity (max_m) for the M dimension

Block scaling granularity (matching DeepGEMM's 1D2D configuration):
  - A: (1, 128) - per-token, per-128-K-elements
  - B: (128, 128) - per-128-N, per-128-K block
"""

import functools

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

from kernels.grouped_gemm_blockscale_common import (
    compute_compile_constants,
    make_a_tile_loaders,
    make_b_loader,
    make_compute_tile,
    make_hot_loop_scheduler,
    make_lds_loader,
    make_prefetch_scales,
    out_mlir_for,
    pack_i64x4_to_i32x8,
    setup_lds_allocation,
    validate_params,
)
from kernels.mfma_epilogues import mfma_epilog
from kernels.mfma_preshuffle_pipeline import (
    crd2idx,
    lds_store_16b_xor16,
    load_b_pack_k32,
    make_preshuffle_b_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


@functools.lru_cache(maxsize=128)
def compile_grouped_gemm_blockscale_masked(
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
    waves_per_eu: int | None = None,
):
    """Compile masked grouped FP8 GEMM kernel and return the JIT launcher.

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

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_masked_grouped_gemm")

    validate_params(
        n=n, k=k, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=scale_block_k, scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    out_mlir = out_mlir_for(out_dtype)

    _c = compute_compile_constants(
        n=n, k=k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=scale_block_k, scale_block_n=scale_block_n,
    )
    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    scale_k = _c.scale_k
    scale_n = _c.scale_n
    sb_per_tile = _c.sb_per_tile
    k_unroll = _c.k_unroll
    kpack_bytes = _c.kpack_bytes
    tile_k_bytes = _c.tile_k_bytes
    tile_k_dwords = _c.tile_k_dwords
    bytes_a_per_tile = _c.bytes_a_per_tile
    bytes_per_thread_a = _c.bytes_per_thread_a
    a_load_bytes = _c.a_load_bytes
    chunk_i32_a = _c.chunk_i32_a
    num_a_loads = _c.num_a_loads

    lds_alloc_offset, lds_tile_elems = setup_lds_allocation(
        allocator=allocator, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, elem_bytes=elem_bytes,
    )

    # Module name for caching
    module_name = (
        f"grouped_gemm_blockscale_masked_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_pingpong"
    ).replace("-", "_")

    @flyc.kernel(name=module_name)
    def grouped_gemm_blockscale_masked_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_expected_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
    ):
        # Convert runtime parameters to index type
        # In the masked kernel, expected_m acts as our padded max capacity per group.
        m_in = arith.index_cast(T.index, i32_expected_m)
        n_in = arith.index_cast(T.index, i32_n)
        k_in = arith.index_cast(T.index, i32_k)
        num_groups_in = arith.index_cast(T.index, i32_num_groups)

        # Thread and 3D block IDs
        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-block index
        bz = gpu.block_id("z")  # Group ID index
        group_idx = bz

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

        # LDS setup: single memref for both ping-pong buffers
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k,)).get()
        lds_stride = tile_k
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(lds_stride, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_elems)

        # CShuffle epilogue LDS (aliased from same base, bf16 element type)
        lds_out = SmemPtr(base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)).get()

        # Buffer resources
        a_nbytes = num_groups_in * m_in * k_in
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, a_nbytes)
        )

        b_nbytes = num_groups_in * n_in * k_in
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, b_nbytes)
        )

        d_nbytes = num_groups_in * m_in * n_in * fx.Index(2)  # bf16/f16 = 2 bytes
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=arith.index_cast(T.i64, d_nbytes)
        )

        # Scale buffers — gfx950 HW E8M0 path consumes int8 (one byte/scale,
        # pre-packed on host); gfx942 SW path consumes f32.
        scale_byte_size = 1 if _is_gfx950 else 4

        # scale_a: [G, scale_k, max_m]
        sa_nbytes = num_groups_in * fx.Index(scale_k) * m_in * fx.Index(scale_byte_size)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, sa_nbytes)
        )

        # scale_b: [G, scale_n, scale_k]
        sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * scale_byte_size)
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, sb_nbytes)
        )

        # masked_m: [G]
        mask_nbytes = num_groups_in * fx.Index(4)
        mask_rsrc = buffer_ops.create_buffer_resource(
            arg_masked_m, max_size=False, num_records_bytes=arith.index_cast(T.i64, mask_nbytes)
        )

        # Early exit for invalid blocks that fall entirely within the padded garbage
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        valid_m_i32 = buffer_ops.buffer_load(mask_rsrc, group_idx, vec_width=1, dtype=T.i32)
        is_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, valid_m_i32)

        _if_valid = scf.IfOp(is_valid)
        with ir.InsertionPoint(_if_valid.then_block):

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

            # B preshuffle layout: total N = num_groups * N (all groups concatenated)
            c_n_total = num_groups_in * n_in
            b_layout = make_preshuffle_b_layout(
                arith, c_n=c_n_total, c_k=k_in,
                kpack_bytes=kpack_bytes, elem_bytes=elem_bytes,
            )
            layout_b = b_layout.layout_b

            # Decompose global N column into (n_blk, n_intra) for preshuffle layout
            c_n0 = c_n_total // fx.Index(16)
            c_n0_i32 = arith.index_cast(T.i32, c_n0)
            layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
            n_blk_list = []
            n_intra_list = []
            group_n_off = group_idx * n_in  # N-offset for this group in concatenated B
            for ni in range_constexpr(num_acc_n):
                col_global = group_n_off + by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
                coord_ni = fx.idx2crd(col_global, layout_n_blk_intra)
                n_blk_list.append(fx.get(coord_ni, 0))
                n_intra_list.append(fx.get(coord_ni, 1))

            (prefetch_a_tile, store_a_tile_to_lds,
             a_row_local, a_col_local_i32, k_blocks16) = make_a_tile_loaders(
                a_rsrc=a_rsrc, lds_a=lds_a, layout_lds=layout_lds,
                bx_m=bx_m, tx=tx,
                tile_m=tile_m, tile_k=tile_k,
                tile_k_bytes=tile_k_bytes, tile_k_dwords=tile_k_dwords,
                chunk_i32_a=chunk_i32_a, num_a_loads=num_a_loads,
                total_threads=total_threads, elem_bytes=elem_bytes,
                k_in=k_in, m_in=m_in, group_idx=group_idx,
            )

            lds_load_packs_k64 = make_lds_loader(
                lds_a=lds_a, layout_lds=layout_lds, k_blocks16=k_blocks16,
            )

            load_b_tile = make_b_loader(
                arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
                n_blk_list=n_blk_list, n_intra_list=n_intra_list,
                lane_div_16=lane_div_16, kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes, k_unroll=k_unroll, num_acc_n=num_acc_n,
            )

            # Base coordinates for A0 prefetch (mi=0, ku=0)
            row_a_lds_base = lane_mod_16  # mi=0
            col_offset_base_bytes = lane_div_16 * fx.Index(16)  # ku=0

            mfma_res_ty = T.f32x4

            ku_per_sb = scale_block_k // 64
            rocdl.sched_barrier(0)

            hot_loop_scheduler = make_hot_loop_scheduler(
                _is_gfx950=_is_gfx950, sb_per_tile=sb_per_tile,
                m_repeat=m_repeat, num_acc_n=num_acc_n,
                k_unroll=k_unroll, num_a_loads=num_a_loads,
                ku_per_sb=ku_per_sb,
            )

            sa_group_off = group_idx * c_scale_k * m_in  # 3D scale_a Offset

            prefetch_scales = make_prefetch_scales(
                _is_gfx950=_is_gfx950, sa_rsrc=sa_rsrc, sb_rsrc=sb_rsrc,
                group_idx=group_idx, scale_n=scale_n, scale_k=scale_k,
                c_scale_k=c_scale_k, n_block_for_scale=n_block_for_scale,
                bx_m=bx_m, lane_mod_16=lane_mod_16, m_in=m_in,
                sb_per_tile=sb_per_tile, m_repeat=m_repeat, num_acc_n=num_acc_n,
                sa_group_off=sa_group_off,
            )

            compute_tile = make_compute_tile(
                _is_gfx950=_is_gfx950, lds_load_packs_k64=lds_load_packs_k64,
                sa_rsrc=sa_rsrc, sb_rsrc=sb_rsrc,
                group_idx=group_idx, scale_n=scale_n, scale_k=scale_k,
                c_scale_k=c_scale_k, n_block_for_scale=n_block_for_scale,
                bx_m=bx_m, lane_mod_16=lane_mod_16, lane_div_16=lane_div_16,
                m_in=m_in, sb_per_tile=sb_per_tile, m_repeat=m_repeat,
                num_acc_n=num_acc_n, ku_per_sb=ku_per_sb,
                col_offset_base_bytes=col_offset_base_bytes,
                mfma_res_ty=mfma_res_ty, acc_init=acc_init,
                sa_group_off=sa_group_off,
            )

            # ===== Ping-pong K-loop =====
            # Prologue: prefetch first A tile into VGPRs, store to LDS, load B + scales
            a_regs0 = prefetch_a_tile(0)
            store_a_tile_to_lds(a_regs0, lds_base_pong)
            b_tile_pong = load_b_tile(fx.Index(0))
            scales_pong_pf = prefetch_scales(0)
            gpu.barrier()

            # Prefetch first A pack from pong (hides LDS latency behind upcoming VMEM)
            a0_prefetch_pong = lds_load_packs_k64(row_a_lds_base, col_offset_base_bytes, lds_base_pong)

            for k_pair in range_constexpr(0, num_k_tiles, 2):
                # Prefetch next scales BEFORE B-tile VMEM (per moe-2stage pattern:
                # scale-load latency hides behind heavy B VMEM); then A+B regs.
                if k_pair + 1 < num_k_tiles:
                    scales_ping_pf = prefetch_scales(k_pair + 1)
                    a_regs_ping = prefetch_a_tile(k_pair + 1)
                    b_tile_ping = load_b_tile(fx.Index((k_pair + 1) * tile_k))

                # Compute current tile from pong LDS
                accs = compute_tile(accs, k_pair, lds_base_pong, b_tile_pong, scales_pong_pf,
                                    a0_prefetch=a0_prefetch_pong)
                a0_prefetch_pong = None

                # Store next A to LDS (ds_write after compute)
                if k_pair + 1 < num_k_tiles:
                    store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                gpu.barrier()

                if k_pair + 1 < num_k_tiles:
                    # Prefetch first A pack from ping
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds_base, col_offset_base_bytes, lds_base_ping)

                    # Prefetch next scales + A+B
                    if k_pair + 2 < num_k_tiles:
                        scales_pong_pf = prefetch_scales(k_pair + 2)
                        a_regs_pong = prefetch_a_tile(k_pair + 2)
                        b_tile_pong = load_b_tile(fx.Index((k_pair + 2) * tile_k))

                    # Compute current tile from ping LDS
                    accs = compute_tile(accs, k_pair + 1, lds_base_ping, b_tile_ping, scales_ping_pf,
                                        a0_prefetch=a0_prefetch_ping)
                    a0_prefetch_ping = None

                    # Store next A to LDS
                    if k_pair + 2 < num_k_tiles:
                        store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                    gpu.barrier()

                    # Prefetch first A pack from pong for next iteration
                    if k_pair + 2 < num_k_tiles:
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds_base, col_offset_base_bytes, lds_base_pong)

            # ===== Epilogue: CShuffle vectorized stores =====
            c_n = n_in
            d_group_off = group_idx * m_in * n_in  # 3D D Offset
            vec1_out = T.vec(1, out_mlir())
            e_vec = 4 if (tile_n % (32 * 4)) == 0 else 2

            def write_row_to_lds(
                *, mi, ii, row_in_tile, row,
                row_base_lds, col_base_local, num_acc_n, lds_out,
            ):
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + (ni * 16)
                    acc_idx = mi * num_acc_n + ni
                    acc = accs[acc_idx]
                    val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                    v_out = arith.trunc_f(out_mlir(), val)
                    lds_idx = row_base_lds + col_local
                    v1 = vector.from_elements(vec1_out, [v_out])
                    vector.store(v1, lds_out, [lds_idx], alignment=2)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                idx_out = d_group_off + row * c_n + col_g0
                byte_off = idx_out * 2
                if e_vec == 4:
                    frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                    buffer_ops.buffer_store(
                        frag_i32x2, d_rsrc, byte_off, offset_is_bytes=True
                    )
                else:
                    frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                    frag_i32 = vector.extract(
                        frag_i32x1, static_position=[0], dynamic_position=[]
                    )
                    buffer_ops.buffer_store(
                        frag_i32, d_rsrc, byte_off, offset_is_bytes=True
                    )

            mfma_epilog(
                use_cshuffle=True,
                arith=arith, vector=vector, gpu=gpu,
                range_constexpr=range_constexpr,
                tile_m=tile_m, tile_n=tile_n, e_vec=e_vec,
                m_repeat=m_repeat, num_acc_n=num_acc_n,
                tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=out_mlir(),
                write_row_to_lds=write_row_to_lds,
                store_pair=store_pair,
            )

            scf.YieldOp([])

    # ===== JIT Launcher =====
    @flyc.jit
    def launch_grouped_gemm_blockscale_masked(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_expected_m: fx.Int32,
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
        max_m_in = arith.index_cast(T.index, i32_expected_m)
        n_in = arith.index_cast(T.index, i32_n)
        num_groups_in = arith.index_cast(T.index, i32_num_groups)

        gx = n_in // fx.Index(tile_n)  # N-blocks
        gy = (max_m_in + fx.Index(tile_m - 1)) // fx.Index(tile_m)  # M-blocks (ceil)
        gz = num_groups_in

        launcher = grouped_gemm_blockscale_masked_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_masked_m,
            i32_expected_m,
            i32_n,
            i32_k,
            i32_num_groups,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        launcher.launch(grid=(gx, gy, gz), block=(total_threads, 1, 1), stream=stream)

    return launch_grouped_gemm_blockscale_masked