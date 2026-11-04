# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Fused split-K reduction shared by the K-split workgroups in one cluster."""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from kernels.common.tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

from .gemm_common_gfx1250 import workgroup_barrier

CPOL_DEVICE = 16
CPOL_STORE_DEVICE = CPOL_DEVICE | 3
UNROLL = 32
MAX_PARTIAL_VECTORS = 128  # 512 dwords per load batch, including split-K 8.


@ASTRewriter.transform
def emit_fused_splitk_epilogue(
    *,
    elem,
    tid,
    block,
    tile_m,
    tile_n,
    lds_base_ptr,
    partials,
    out,
    c_off,
    ldc64,
    c_lds_row,
    split_k,
    split_idx,
    mn_oob,
    flat_tile,
    bounded_m=False,
    n_oob=None,
):
    """Reduce disjoint row ranges using all of a tile's split-K workgroups.

    The launch clusters every K split of a tile together in grid.z, so the
    cluster barrier alone orders the exchange: no flags, no spinning, and no
    resident-grid precondition.  Each workgroup owns ``tile_m // split_k``
    rows, keeps its own stripe in LDS, and publishes only the peer rows --
    the caller has already rotated its LDS rows so those are contiguous.
    At ``split_k == 2`` the reduced stripe goes straight to global and the
    scratch holds peer rows only; deeper splits keep the padded plane and a
    final LDS + TDM store.  The sum stays in FP32 and in split order.
    """
    partial_elem = fx.Float32
    partial_bytes = 4
    vec = 4
    reduce_m = tile_m // split_k
    reduce_row = split_idx * reduce_m
    peer_m = tile_m - reduce_m
    lean = split_k == 2
    lanes_per_row = tile_n // vec
    rows_per_iter = block // lanes_per_row
    partial_vectors = MAX_PARTIAL_VECTORS // (block // 128)
    unroll = min(UNROLL, partial_vectors // split_k, reduce_m // rows_per_iter)
    row0 = tid // lanes_per_row
    col = (tid % lanes_per_row) * vec
    plane = fx.Int64((peer_m if lean else tile_m) * tile_n)
    partial_base = fx.recast_iter(partial_elem, partials) + fx.Int64(flat_tile) * split_k * plane
    lds_partial = fx.recast_iter(partial_elem, lds_base_ptr)
    lds_out = fx.recast_iter(elem, lds_base_ptr)

    # Each scratch plane holds only the rotated peer rows, including on an M
    # tail; the output descriptor discards the invalid logical rows.
    partial_out = partial_base + fx.Int64(split_idx) * plane
    # A store cannot drain a padded LDS tile, so the tile is the PADDED width and the
    # view's inner extent the real one -- boundary_check then drops the pad columns.
    OUT_TILE = (peer_m, c_lds_row)
    dst = fx.Tensor(fx.make_view(partial_out, fx.make_layout((peer_m, tile_n), (fx.Int64(tile_n), 1))))
    store, out_coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
        fx.rocdl.TensorStore(cache_modifier=CPOL_STORE_DEVICE),
        dst,
        fx.make_layout(OUT_TILE, (c_lds_row, 1)),
        OUT_TILE,
        num_warps=block // 32,
    )
    out_blk = fx.zipped_divide(out_coord, OUT_TILE)[None, (0, None)]
    lds_box = fx.Tensor(fx.make_view(lds_partial, fx.make_layout((OUT_TILE, 1), ((c_lds_row, 1), peer_m * c_lds_row))))
    tOs, tOg = fx.rocdl.cdna5.tdm_partition(store, fx.rocdl.wave_id(), fx.make_layout(block // 32, 1), lds_box, out_blk)
    fx.copy(store, tOs[None, 0], tOg[None, 0])
    fx.rocdl.s_wait_tensorcnt(0)
    cluster.cluster_barrier()

    output_base = fx.recast_iter(elem, out) + c_off + fx.Int64(reduce_row) * ldc64
    remaining = mn_oob - reduce_row
    valid_rows = (remaining > 0).select(remaining, fx.Int32(0))
    if const_expr(lean and bounded_m and n_oob is None):
        output_buffer = ptr_buf_tensor(
            output_base,
            elem,
            unit_elems=vec,
            num_records_bytes=fx.Int64(valid_rows) * ldc64 * 2,
        )
    if const_expr(lean):
        # Both peer stripes start at offset zero, so the descriptor needs no
        # split-index term.
        peer_buffer = ptr_buf_tensor(
            partial_base + fx.Int64(1 - split_idx) * plane,
            partial_elem,
            unit_elems=vec,
            num_records_bytes=fx.Int64(reduce_m * tile_n * partial_bytes),
        )
    else:
        # Select the cyclic peer stripe in each uniform buffer base. Keeping
        # that offset out of the per-vector indices avoids vector arithmetic.
        buffers = [
            ptr_buf_tensor(
                partial_base
                + fx.Int64(s) * plane
                + fx.Int64((split_idx + split_k - s - 1) & (split_k - 1)) * (reduce_m * tile_n),
                partial_elem,
                unit_elems=vec,
                num_records_bytes=fx.Int64(reduce_m * tile_n * partial_bytes),
            )
            for s in range_constexpr(split_k)
        ]

    def _local(batch, u):
        return fx.ptr_load(
            lds_partial + (peer_m + row0 + (batch * unroll + u) * rows_per_iter) * c_lds_row + col,
            result_type=T.vec(vec, partial_elem.ir_type),
        )

    for batch in range(reduce_m // (rows_per_iter * unroll)):
        partial_indices = [tid + (batch * unroll + u) * block for u in range_constexpr(unroll)]
        if const_expr(lean):
            # A two-term FP32 sum is commutative, so local-then-peer matches
            # split order bit for bit and needs no branch.
            peer_parts = [
                buf_copy_load(peer_buffer, idx, partial_elem, vec, cache_modifier=CPOL_DEVICE)
                for idx in partial_indices
            ]
            values = [[_local(batch, u), peer_parts[u]] for u in range_constexpr(unroll)]
        else:
            local_parts = [_local(batch, u) for u in range_constexpr(unroll)]
            parts = [
                [fx.make_rmem_tensor(vec, partial_elem) for s in range_constexpr(split_k)]
                for u in range_constexpr(unroll)
            ]
            # Uniform branches skip the unpublished local stripe completely.
            # Stage all peer loads before converting or summing the fragments.
            for s in range_constexpr(split_k):
                for u in range_constexpr(unroll):
                    parts[u][s].store(local_parts[u])
                if split_idx != fx.Int32(s):
                    for u in range_constexpr(unroll):
                        parts[u][s].store(
                            buf_copy_load(
                                buffers[s],
                                partial_indices[u],
                                partial_elem,
                                vec,
                                cache_modifier=CPOL_DEVICE,
                            )
                        )
            values = [[parts[u][s].load() for s in range_constexpr(split_k)] for u in range_constexpr(unroll)]
        for u in range_constexpr(unroll):
            acc = values[u][0]
            for s in range_constexpr(1, split_k):
                acc = acc + values[u][s]
            row = row0 + (batch * unroll + u) * rows_per_iter
            if const_expr(not lean):
                fx.ptr_store(acc.to(elem), lds_out + row * c_lds_row + col)
            elif const_expr(n_oob is not None):
                if row < valid_rows:
                    if col + vec <= n_oob:
                        fx.ptr_store(acc.to(elem), output_base + row * ldc64 + col)
                    else:
                        for j in range_constexpr(vec):
                            if col + j < n_oob:
                                fx.ptr_store(acc[j].to(elem), output_base + row * ldc64 + col + j)
            elif const_expr(bounded_m):
                buf_copy_store(
                    output_buffer,
                    row * fx.Int32(ldc64 // vec) + col // vec,
                    acc.to(elem),
                    elem,
                    vec,
                )
            else:
                fx.ptr_store(acc.to(elem), output_base + row * ldc64 + col)

    if const_expr(not lean):
        workgroup_barrier(use_cluster=False)
        # The view carries this store's bounds now, so BOTH clamps live in its shape:
        # valid_rows on M and n_oob on N. Dropping the N one writes the tile's pad
        # columns past column N and corrupts the next row.
        OUT_TILE2 = (reduce_m, c_lds_row)
        out_n = tile_n if n_oob is None else fx.max(0, fx.min(n_oob, tile_n))
        tile_out = fx.Tensor(fx.make_view(output_base, fx.make_layout((valid_rows, out_n), (ldc64, 1))))
        store_atom, out_coord2 = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorStore(),
            tile_out,
            fx.make_layout(OUT_TILE2, (c_lds_row, 1)),
            OUT_TILE2,
            num_warps=block // 32,
        )
        out_blk2 = fx.zipped_divide(out_coord2, OUT_TILE2)[None, (0, None)]
        lds_box2 = fx.Tensor(
            fx.make_view(lds_out, fx.make_layout((OUT_TILE2, 1), ((c_lds_row, 1), reduce_m * c_lds_row)))
        )
        tO2s, tO2g = fx.rocdl.cdna5.tdm_partition(
            store_atom, fx.rocdl.wave_id(), fx.make_layout(block // 32, 1), lds_box2, out_blk2
        )
        fx.copy(store_atom, tO2s[None, 0], tO2g[None, 0])
        fx.rocdl.s_wait_tensorcnt(0)
