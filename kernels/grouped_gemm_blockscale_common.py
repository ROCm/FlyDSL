# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared building blocks for the grouped FP8 blockscale GEMM kernels.

Used by both `kernels/grouped_gemm_blockscale_contiguous.py` and
`kernels/grouped_gemm_blockscale_masked.py`. Holds the parts of the two
kernels that are byte-identical (parameter validation, compile-time
scalar constants, helper closures) so they live in one place.
"""

from collections import namedtuple

import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector
from flydsl.expr import range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir.dialects import math as math_dialect

from kernels.mfma_preshuffle_pipeline import (
    crd2idx,
    lds_store_16b_xor16,
    load_b_pack_k32,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


CompileConstants = namedtuple(
    "CompileConstants",
    [
        "total_threads",
        "elem_bytes",
        "num_k_tiles",
        "scale_k",
        "scale_n",
        "sb_per_tile",
        "k_unroll",
        "kpack_bytes",
        "tile_k_bytes",
        "tile_k_dwords",
        "bytes_a_per_tile",
        "bytes_per_thread_a",
        "a_load_bytes",
        "chunk_i32_a",
        "num_a_loads",
    ],
)


def validate_params(*, n, k, tile_n, tile_k, scale_block_k, scale_block_n, out_dtype):
    """Validate the divisibility constraints and out_dtype choice shared by
    both grouped GEMM blockscale kernels."""
    if k % tile_k != 0:
        raise ValueError(f"k ({k}) must be divisible by tile_k ({tile_k})")
    if n % tile_n != 0:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_k % scale_block_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by scale_block_k ({scale_block_k})")
    if tile_n % scale_block_n != 0:
        raise ValueError(f"tile_n ({tile_n}) must be divisible by scale_block_n ({scale_block_n})")
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")


def out_mlir_for(out_dtype):
    """Return a zero-arg callable that yields the MLIR element type for the
    chosen output dtype. Matches the original local `out_mlir` lambda exactly
    so MLIR emission is unchanged."""
    return lambda: T.bf16 if out_dtype == "bf16" else T.f16


def compute_compile_constants(*, n, k, tile_m, tile_n, tile_k, scale_block_k, scale_block_n):
    """Compute the compile-time scalar constants shared by both kernels.

    Returns a `CompileConstants` namedtuple. Pure-Python — no MLIR ops emitted.
    """
    total_threads = 256
    elem_bytes = 1  # FP8
    num_k_tiles = k // tile_k
    scale_k = k // scale_block_k
    scale_n = n // scale_block_n
    sb_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    k_unroll = tile_k // 64  # K64-byte micro-steps (for K32 MFMA pairs)
    kpack_bytes = 16  # 16-byte packs for FP8

    tile_k_bytes = tile_k * elem_bytes
    tile_k_dwords = tile_k_bytes // 4
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16  # 16-byte loads (dwordx4)
    chunk_i32_a = a_load_bytes // 4  # 4 dwords per load
    num_a_loads = bytes_per_thread_a // a_load_bytes

    return CompileConstants(
        total_threads=total_threads,
        elem_bytes=elem_bytes,
        num_k_tiles=num_k_tiles,
        scale_k=scale_k,
        scale_n=scale_n,
        sb_per_tile=sb_per_tile,
        k_unroll=k_unroll,
        kpack_bytes=kpack_bytes,
        tile_k_bytes=tile_k_bytes,
        tile_k_dwords=tile_k_dwords,
        bytes_a_per_tile=bytes_a_per_tile,
        bytes_per_thread_a=bytes_per_thread_a,
        a_load_bytes=a_load_bytes,
        chunk_i32_a=chunk_i32_a,
        num_a_loads=num_a_loads,
    )


def setup_lds_allocation(*, allocator, tile_m, tile_k, tile_n, elem_bytes):
    """Reserve LDS for ping-pong A tiles and the CShuffle epilogue output.

    The ping-pong A buffers and the FP16/BF16 epilogue output share the same
    LDS arena (alias), so we reserve the max of the two. Returns
    `(lds_alloc_offset, lds_tile_elems)` where `lds_tile_elems` is the
    A-element stride between the ping and pong halves.
    """
    lds_a_bytes = tile_m * tile_k * elem_bytes
    lds_pingpong_bytes = 2 * lds_a_bytes
    lds_out_bytes = tile_m * tile_n * 2  # bf16/f16 = 2 bytes per element
    lds_total_bytes = max(lds_pingpong_bytes, lds_out_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_total_bytes
    lds_tile_elems = tile_m * tile_k  # element offset between ping and pong
    return lds_alloc_offset, lds_tile_elems


def make_a_tile_loaders(
    *,
    a_rsrc,
    lds_a,
    layout_lds,
    bx_m,
    tx,
    tile_m,
    tile_k,
    tile_k_bytes,
    tile_k_dwords,
    chunk_i32_a,
    num_a_loads,
    total_threads,
    elem_bytes,
    k_in,
    m_in=None,
    group_idx=None,
):
    """Build the prefetch + LDS-store closures for the A tile.

    Returns `(prefetch_a_tile, store_a_tile_to_lds, a_row_local,
    a_col_local_i32, k_blocks16)`. When `m_in` and `group_idx` are both
    None (contig path) no group offset is emitted; when both are provided
    (masked path), `group_idx * m_in * (k_in/4)` is added as the leading
    term inside `prefetch_a_tile`, exactly matching the original masked
    code so the resulting MLIR (and ISA) is byte-identical. `k_blocks16`
    is returned for reuse by the downstream LDS-load helper.
    """
    layout_a_tile_div4 = fx.make_layout(
        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
    )
    c_chunk_a = fx.Index(chunk_i32_a)
    tx_i32_base = tx * c_chunk_a
    _k_div4_factor = k_in // fx.Index(4)
    if m_in is not None and group_idx is not None:
        a_tile_offset_div4 = group_idx * m_in * _k_div4_factor  # 3D A Offset
    else:
        a_tile_offset_div4 = None
    k_blocks16 = arith.index(tile_k_bytes // 16)
    c4_bytes = fx.Index(4)

    a_row_local = []
    a_col_local_i32 = []
    for i in range_constexpr(num_a_loads):
        row_local, col_local_i32 = tile_chunk_coord_i32(
            arith, tx_i32_base=tx_i32_base, i=i,
            total_threads=total_threads,
            layout_tile_div4=layout_a_tile_div4,
            chunk_i32=chunk_i32_a,
        )
        a_row_local.append(row_local)
        a_col_local_i32.append(col_local_i32)

    def prefetch_a_tile(k_tile_idx_py):
        """Load A tile from global memory into VGPRs."""
        base_k_div4 = fx.Index(k_tile_idx_py * tile_k_dwords)
        parts = []
        for i in range_constexpr(num_a_loads):
            row_global = bx_m + a_row_local[i]
            if a_tile_offset_div4 is None:
                idx_i32 = row_global * _k_div4_factor + base_k_div4 + a_col_local_i32[i]
            else:
                idx_i32 = a_tile_offset_div4 + row_global * _k_div4_factor + base_k_div4 + a_col_local_i32[i]
            a_vec = buffer_ops.buffer_load(a_rsrc, idx_i32, vec_width=4, dtype=T.i32)
            parts.append(vector.bitcast(T.i32x4, a_vec))
        return parts

    def store_a_tile_to_lds(a_parts, lds_base):
        """Write prefetched A tile from VGPRs into LDS with XOR16 swizzle."""
        for i in range_constexpr(num_a_loads):
            lds_store_16b_xor16(
                arith, vector,
                lds_memref=lds_a, vec16_ty=T.f8x16,
                layout_lds=layout_lds,
                row_local=a_row_local[i],
                col_local_i32=a_col_local_i32[i],
                tx_c4=c4_bytes, k_blocks16=k_blocks16,
                lds_base=lds_base,
                vec_part_i32x4=a_parts[i], elem_bytes=elem_bytes,
            )

    return prefetch_a_tile, store_a_tile_to_lds, a_row_local, a_col_local_i32, k_blocks16


def make_lds_loader(*, lds_a, layout_lds, k_blocks16):
    """Build the LDS-side A K64 pack loader.

    Returns `lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base)`
    which loads 16B from LDS with the XOR16 swizzle and returns the two
    i64 halves.
    """
    def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
        col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
        idx_a16 = crd2idx((curr_row_a_lds, col_base_swz_bytes), layout_lds) + lds_base
        loaded_a16 = vector.load_op(T.vec(16, T.f8), lds_a, [idx_a16])
        a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
        a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
        a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
        return a0, a1

    return lds_load_packs_k64


def make_b_loader(
    *,
    arg_b,
    b_rsrc,
    layout_b,
    n_blk_list,
    n_intra_list,
    lane_div_16,
    kpack_bytes,
    elem_bytes,
    k_unroll,
    num_acc_n,
):
    """Build the B-tile loader closure.

    Returns `load_b_tile(base_k)` which loads all B packs for one K-tile,
    returning a list of length `k_unroll` where each entry is
    `(packs_half0[ni], packs_half1[ni])` for one K64 micro-step.
    """
    def load_b_pack(base_k, ki_step, ni):
        return load_b_pack_k32(
            buffer_ops, arith, vector,
            arg_b=arg_b, b_rsrc=b_rsrc,
            layout_b=layout_b,
            base_k=base_k, ki_step=ki_step,
            n_blk=n_blk_list[ni],
            n_intra=n_intra_list[ni],
            lane_div_16=lane_div_16,
            elem_type=T.f8,
            kpack_bytes=kpack_bytes,
            elem_bytes=elem_bytes,
        )

    def load_b_tile(base_k):
        b_tile = []
        for ku in range_constexpr(k_unroll):
            packs0 = []
            packs1 = []
            for ni in range_constexpr(num_acc_n):
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                b0 = load_b_pack(base_k, ki0, ni)
                b1 = load_b_pack(base_k, ki1, ni)
                packs0.append(b0)
                packs1.append(b1)
            b_tile.append((packs0, packs1))
        return b_tile

    return load_b_tile


def pack_i64x4_to_i32x8(x0, x1, x2, x3):
    """Pack four i64 values into a single i32x8 vector via i64x4 bitcast.

    Used to assemble the K=128 MFMA A/B operands on gfx950.
    """
    v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
    return vector.bitcast(T.vec(8, T.i32), v4)


def make_hot_loop_scheduler(
    *,
    _is_gfx950,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    k_unroll,
    num_a_loads,
    ku_per_sb,
):
    """Build the per-tile sched_group_barrier scheduler closure.

    Emits the dsrd / mfma / vmem_rd / dswr group barriers in the order
    matching the MoE stage-2 pattern. Returns a zero-arg closure to be
    invoked once per K-tile body inside the ping-pong loop.
    """
    def hot_loop_scheduler():
        mfma_group = num_acc_n
        if _is_gfx950:
            total_mfma = sb_per_tile * m_repeat * mfma_group
        else:
            total_mfma = k_unroll * m_repeat * mfma_group * 2
        rocdl.sched_group_barrier(rocdl.mask_dsrd, ku_per_sb * m_repeat, 0)
        rocdl.sched_group_barrier(rocdl.mask_mfma, total_mfma, 1)
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, num_a_loads, 2)
        rocdl.sched_group_barrier(rocdl.mask_dswr, num_a_loads, 3)
        rocdl.sched_barrier(0)

    return hot_loop_scheduler


def make_prefetch_scales(
    *,
    _is_gfx950,
    sa_rsrc,
    sb_rsrc,
    group_idx,
    scale_n,
    scale_k,
    c_scale_k,
    n_block_for_scale,
    bx_m,
    lane_mod_16,
    m_in,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    sa_group_off=None,
):
    """Build the cross-tile E8M0 scale prefetch closure (gfx950 HW path).

    Returns `prefetch_scales(k_tile_idx_py)` that returns
    `(sa_pf, sb_pf)` — outer index = sb (sb_per_tile), inner =
    m_repeat / num_acc_n. Returns None on the gfx942 SW path (where
    scales are loaded inside compute_tile instead).

    `sa_group_off` is None for the contig path (no addition emitted)
    and `group_idx * c_scale_k * m_in` for the masked path. Using a
    Python `is None` guard keeps the contig MLIR identical to the
    pre-extraction code.
    """
    def prefetch_scales(k_tile_idx_py):
        if not _is_gfx950:
            return None
        sa_pf = []
        sb_pf = []
        sb_group_offset = group_idx * fx.Index(scale_n * scale_k)
        for sb in range_constexpr(sb_per_tile):
            kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)
            if sa_group_off is None:
                sa_base_pf = kb * m_in
            else:
                sa_base_pf = sa_group_off + kb * m_in

            sa_sb = []
            for mi in range_constexpr(m_repeat):
                sa_row = bx_m + arith.index(mi * 16) + lane_mod_16
                sa_idx = sa_base_pf + sa_row
                sa_i8 = buffer_ops.buffer_load(sa_rsrc, sa_idx, vec_width=1, dtype=T.i8)
                sa_e8m0 = ArithValue(sa_i8).extui(T.i32)
                sa_sb.append(sa_e8m0)
            sa_pf.append(sa_sb)

            sb_sb = []
            for ni in range_constexpr(num_acc_n):
                sb_idx = sb_group_offset + n_block_for_scale[ni] * c_scale_k + kb
                sb_i8 = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.i8)
                sb_i32 = ArithValue(sb_i8).extui(T.i32)
                sb_e8m0 = rocdl.readfirstlane(T.i32, sb_i32)
                sb_sb.append(sb_e8m0)
            sb_pf.append(sb_sb)
        return (sa_pf, sb_pf)

    return prefetch_scales


def make_compute_tile(
    *,
    _is_gfx950,
    lds_load_packs_k64,
    sa_rsrc,
    sb_rsrc,
    group_idx,
    scale_n,
    scale_k,
    c_scale_k,
    n_block_for_scale,
    bx_m,
    lane_mod_16,
    lane_div_16,
    m_in,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    ku_per_sb,
    col_offset_base_bytes,
    mfma_res_ty,
    acc_init,
    sa_group_off=None,
):
    """Build the per-K-tile compute closure.

    Returns `compute_tile(accs_in, k_tile_idx_py, lds_base, b_tile_in,
    scales_pf, *, a0_prefetch=None)` which advances the accumulators by
    one K-tile of MFMA work. `scales_pf` is the prefetched scales for
    the gfx950 HW path; None for the gfx942 SW path (which loads scales
    locally inside the closure).

    `sa_group_off` is None for the contig path (no addition emitted)
    and `group_idx * c_scale_k * m_in` for the masked path; only the
    gfx942 SW path uses it.
    """
    def compute_tile(accs_in, k_tile_idx_py, lds_base, b_tile_in, scales_pf, *, a0_prefetch=None):
        current_accs = list(accs_in)

        for sb in range_constexpr(sb_per_tile):
            kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)

            s_a_vecs = []
            s_b_vals = []
            if not _is_gfx950:
                if sa_group_off is None:
                    sa_base = kb * m_in
                else:
                    sa_base = sa_group_off + kb * m_in
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

                sb_group_offset = group_idx * fx.Index(scale_n * scale_k)
                for ni in range_constexpr(num_acc_n):
                    sb_idx = sb_group_offset + n_block_for_scale[ni] * c_scale_k + kb
                    s_b_val = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
                    s_b_val = rocdl.readfirstlane(T.f32, s_b_val)
                    s_b_vals.append(s_b_val)

            if _is_gfx950:
                sa_pf, sb_pf = scales_pf
                sa_e8m0_list = sa_pf[sb]
                sb_e8m0_list = sb_pf[sb]

                ku0 = sb * ku_per_sb
                ku1 = ku0 + 1
                b0_packs0, b0_packs1 = b_tile_in[ku0]
                b1_packs0, b1_packs1 = b_tile_in[ku1]
                col_base0 = col_offset_base_bytes + fx.Index(ku0 * 64)
                col_base1 = col_offset_base_bytes + fx.Index(ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = lane_mod_16 + arith.index(mi * 16)
                    if a0_prefetch is not None and sb == 0 and mi == 0:
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b0_packs0[ni], b0_packs1[ni],
                            b1_packs0[ni], b1_packs1[ni],
                        )
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a128, b128, current_accs[acc_idx],
                             0, 0, 0, sa_e8m0_list[mi], 0, sb_e8m0_list[ni]],
                        )
            else:
                for ku_local in range_constexpr(ku_per_sb):
                    ku = sb * ku_per_sb + ku_local
                    k_offset_bytes = ku * 64
                    b_packs0, b_packs1 = b_tile_in[ku]

                    for mi in range_constexpr(m_repeat):
                        if a0_prefetch is not None and sb == 0 and ku_local == 0 and mi == 0:
                            a0, a1 = a0_prefetch
                        else:
                            row_a_lds = lane_mod_16 + arith.index(mi * 16)
                            col_a_base_bytes = lane_div_16 * fx.Index(16) + fx.Index(k_offset_bytes)
                            a0, a1 = lds_load_packs_k64(row_a_lds, col_a_base_bytes, lds_base)

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni

                            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
                            mfma_mid = mfma_fn(T.f32x4, [a0, b_packs0[ni], acc_init, 0, 0, 0])
                            mfma_result = mfma_fn(T.f32x4, [a1, b_packs1[ni], mfma_mid, 0, 0, 0])

                            s_a_v4 = s_a_vecs[mi]
                            s_b_bc = vector.broadcast(T.f32x4, s_b_vals[ni])
                            scaled = ArithValue(mfma_result) * ArithValue(s_a_v4)
                            current_accs[acc_idx] = math_dialect.fma(scaled, s_b_bc, current_accs[acc_idx])

        return current_accs

    return compute_tile


def make_pingpong_kloop(
    *,
    num_k_tiles,
    tile_k,
    prefetch_a_tile,
    store_a_tile_to_lds,
    load_b_tile,
    prefetch_scales,
    compute_tile,
    hot_loop_scheduler,
    lds_load_packs_k64,
    lds_base_pong,
    lds_base_ping,
    row_a_lds_base,
    col_offset_base_bytes,
):
    """Build the ping-pong K-loop driver.

    Returns `run_kloop(accs)` which advances `accs` through all
    K-tiles using the prologue + alternating ping/pong stages.
    Loop body is byte-identical between contig and masked, so this
    factory has no offset parameters.
    """
    def run_kloop(accs):
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

            # Store next A to LDS (ds_write after compute, overlaps with trailing MFMAs)
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

        return accs

    return run_kloop


def make_epilogue_writers(
    *,
    accs,
    d_rsrc,
    out_mlir,
    e_vec,
    c_n,
    d_group_off=None,
):
    """Build the CShuffle-epilogue writer closures.

    Returns `(write_row_to_lds, store_pair)` to be passed to
    `mfma_epilog`. `d_group_off` is None for the contig path (no
    addition emitted) and `group_idx * m_in * n_in` for the masked
    path. Using a Python `is None` guard keeps the contig MLIR
    identical to the pre-extraction code.
    """
    vec1_out = T.vec(1, out_mlir())

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
        if d_group_off is None:
            idx_out = row * c_n + col_g0
        else:
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

    return write_row_to_lds, store_pair
