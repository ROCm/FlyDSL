# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K partial reduction epilogue for the gfx1250 A8W8 GEMM."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

DEFAULT_BLOCK = 128
VEC = 8  # 16 bytes per thread and slice: one BufferCopy128b in/out.


def compile_gemm_a8w8_splitk_reduce(
    *,
    split_k: int,
    out_dtype_str: str = "bf16",
    block: int = DEFAULT_BLOCK,
    unroll: int = 0,
):
    if split_k <= 1:
        raise ValueError(f"split_k must be greater than one, got {split_k}")
    if out_dtype_str not in ("bf16", "f16"):
        raise ValueError(f"unsupported output dtype {out_dtype_str!r}")
    if block not in (64, 128, 256):
        raise ValueError(f"block must be one of 64, 128, or 256, got {block}")
    unroll = unroll or 256 // block
    if unroll <= 0:
        raise ValueError(f"unroll must be positive, got {unroll}")
    return _compile_gemm_a8w8_splitk_reduce(split_k, out_dtype_str, block, unroll)


@functools.lru_cache(maxsize=32)
def _compile_gemm_a8w8_splitk_reduce(split_k: int, out_dtype_str: str, block: int, unroll: int):
    is_f16 = out_dtype_str == "f16"
    tile = block * VEC
    span = tile * unroll

    @flyc.kernel(known_block_size=[block, 1, 1])
    def reduce_kernel(partials: fx.Pointer, out: fx.Pointer, i32_total: fx.Int32):
        elem = fx.Float16 if is_f16 else fx.BFloat16
        vec_f32, vec_out = T.vec(VEC, T.f32), T.vec(VEC, elem.ir_type)
        tid, blk = gpu.thread_id("x"), gpu.block_id("x")
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem)
        slice_bytes = fx.Int64(i32_total) * fx.Int64(2)

        def _view(ptr_i64):
            ptr_ty = fx.PointerType.get(elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
            view = fx.make_view(
                fx.inttoptr(ptr_ty, ptr_i64),
                fx.make_layout((1, i32_total), (i32_total, 1)),
            )
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=slice_bytes)

        partial_base = fx.Int64(ptrtoint(partials))
        partial_bufs = [_view(partial_base + fx.Int64(s) * slice_bytes) for s in range_constexpr(split_k)]
        out_buf = _view(fx.Int64(ptrtoint(out)))

        tile_mn, tv_layout = fx.make_layout_tv(
            fx.make_layout((1, block), (1, 1)),
            fx.make_layout((1, VEC), (1, 1)),
        )
        thread_copy = fx.make_tiled_copy(atom, tv_layout, tile_mn).get_slice(tid)
        base = fx.Int32(blk) * fx.Int32(unroll)

        def _part(buf, u):
            return fx.slice(fx.zipped_divide(buf, tile_mn), (None, (0, base + u)))

        srcs = [[thread_copy.partition_S(_part(buf, u)) for buf in partial_bufs] for u in range_constexpr(unroll)]
        frags = [[fx.make_fragment_like(src) for src in row] for row in srcs]

        # Maximize load-to-use distance: issue every slice load before arithmetic.
        for u in range_constexpr(unroll):
            for s in range_constexpr(split_k):
                fx.copy(atom, srcs[u][s], frags[u][s])

        for u in range_constexpr(unroll):
            acc = fx.Vector(fx.memref_load_vec(frags[u][0])).extf(vec_f32)
            for s in range_constexpr(1, split_k):
                acc = acc + fx.Vector(fx.memref_load_vec(frags[u][s])).extf(vec_f32)
            dst = thread_copy.partition_D(_part(out_buf, u))
            out_frag = fx.make_fragment_like(dst)
            fx.memref_store_vec(acc.truncf(vec_out), out_frag)
            fx.copy(atom, out_frag, dst)

    @flyc.jit
    def launch(partials: fx.Pointer, out: fx.Pointer, i32_total: fx.Int32, stream: fx.Stream):
        n_tiles = (i32_total + fx.Int32(span - 1)) // fx.Int32(span)
        reduce_kernel(partials, out, i32_total).launch(
            grid=(n_tiles, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["compile_gemm_a8w8_splitk_reduce"]
