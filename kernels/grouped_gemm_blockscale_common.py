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
from flydsl.expr import arith, buffer_ops, vector
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from kernels.mfma_preshuffle_pipeline import (
    lds_store_16b_xor16,
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
