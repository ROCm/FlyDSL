# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Split-K partial reduction epilogue for the gfx1250 a8w8 preshuffle GEMM."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from kernels.common.kernels_common import format_kernel_name

BLOCK = 512
ELEM_BYTES = 2  # bf16 / f16 output


@functools.lru_cache(maxsize=32)
def compile_gemm_a8w8_splitk_reduce(
    *,
    split_k: int,
    out_dtype_str: str = "bf16",
    unroll: int = 0,
    partial_dtype_str: str | None = None,
):
    partial_dtype_str = partial_dtype_str or out_dtype_str
    if partial_dtype_str not in ("f32", "bf16", "f16") or out_dtype_str not in ("bf16", "f16"):
        raise ValueError("partials must be f32/bf16/f16 and output must be bf16/f16")
    if split_k < 1:
        raise ValueError("split_k must be positive")
    input_bytes = 4 if partial_dtype_str == "f32" else 2
    vec = 16 // input_bytes
    unroll = unroll or max(1, 4 // split_k)
    if unroll < 1 or unroll & (unroll - 1):
        raise ValueError(f"unroll must be a power of two, got {unroll}")
    is_f16 = out_dtype_str == "f16"
    span = BLOCK * vec * unroll
    name = format_kernel_name(f"gemm_a8w8_splitk_reduce_{partial_dtype_str}_to_{out_dtype_str}_sk{split_k}_u{unroll}")

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def reduce_kernel(
        partials: fx.Pointer,
        out: fx.Pointer,
        i64_run: fx.Int64,
        i32_ld: fx.Int32,
        i64_slice_bytes: fx.Int64,
    ):
        elem = fx.Float16 if is_f16 else fx.BFloat16
        input_elem = {"f32": fx.Float32, "bf16": fx.BFloat16, "f16": fx.Float16}[partial_dtype_str]
        tid = fx.thread_idx.x
        tile, run = fx.block_idx.x, fx.block_idx.y
        input_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), input_elem)
        output_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b() if input_bytes == 4 else fx.rocdl.BufferCopy128b(), elem
        )

        off = fx.Int64(tile) * span
        rest = i64_run - off
        valid = fx.Int32(fx.min(rest, span))
        row = fx.Int64(run) * fx.Int64(i32_ld) + off

        def _view(ptr, dtype, elem_bytes):
            pt = fx.PointerType.get(dtype.ir_type, address_space=fx.AddressSpace.Global, alignment=elem_bytes)
            view = fx.make_view(fx.recast_iter(pt, ptr), fx.make_layout((1, span), (span, 1)))
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=valid * elem_bytes)

        pbase = fx.recast_iter(fx.Int8, partials) + row * input_bytes
        pbufs = [_view(pbase + s * i64_slice_bytes, input_elem, input_bytes) for s in range_constexpr(split_k)]
        obuf = _view(fx.recast_iter(fx.Int8, out) + row * ELEM_BYTES, elem, ELEM_BYTES)

        threads = fx.make_layout((1, BLOCK), (1, 1))
        values = fx.make_layout((1, vec), (1, 1))
        input_copy = fx.make_tiled_copy_tv(input_atom, threads, values)
        input_thr = input_copy.get_slice(tid)
        output_thr = fx.make_tiled_copy_tv(output_atom, threads, values).get_slice(tid)

        def _part(buf, u):
            return fx.slice(fx.zipped_divide(buf, input_copy.tile_mn), (None, (0, u)))

        srcs = [[input_thr.partition_S(_part(pbuf, u)) for pbuf in pbufs] for u in range_constexpr(unroll)]
        frags = [[fx.make_fragment_like(s) for s in row_srcs] for row_srcs in srcs]
        # Issue every load of the block's span before any arithmetic.
        for u in range_constexpr(unroll):
            for s in range_constexpr(split_k):
                fx.copy(input_atom, srcs[u][s], frags[u][s])
        for u in range_constexpr(unroll):
            acc = fx.memref_load_vec(frags[u][0]).to(fx.Float32)
            for s in range_constexpr(1, split_k):
                acc = acc + fx.memref_load_vec(frags[u][s]).to(fx.Float32)
            dst = output_thr.partition_D(_part(obuf, u))
            ofrag = fx.make_fragment_like(dst)
            fx.memref_store_vec(acc.to(elem), ofrag)
            fx.copy(output_atom, ofrag, dst)

    @flyc.jit
    def launch(
        partials: fx.Pointer,
        out: fx.Pointer,
        i64_run: fx.Int64,
        i32_runs: fx.Int32,
        i32_ld: fx.Int32,
        i64_slice_bytes: fx.Int64,
        stream: fx.Stream,
    ):
        n_tiles = fx.ceildiv(i64_run, span)
        reduce_kernel(partials, out, i64_run, i32_ld, i64_slice_bytes).launch(
            grid=(n_tiles, fx.Int64(i32_runs), 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
