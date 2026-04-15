# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused silu_and_mul + MXFP4 quantization + sorted-scale write kernel (FlyDSL).

Designed for split-K MOE stage1 post-processing:

  input   : tmp_out  (token_num * topk, inter_dim * 2) bf16
  sorted  : sorted_token_ids (sorted_len,) i32 -- packed (token<<0 | slot<<24)
            num_valid_ids    (1,) i32
  output  : out_fp4          raw byte buffer -- FP4x2 packed, row stride = inter_dim//2
            out_scale_sorted raw byte buffer -- tiled E8M0 scale (same layout as moe_mxfp4_sort)

Grid:  (num_sorted_rows, 1, 1)  -- one workgroup per sorted row (including blockM padding).
Block: (BLOCK_THREADS, 1, 1)

Each workgroup:
  1. Loads sorted_token_ids[bid] -> (token_id, slot_id) -> row = token_id * topk + slot_id
  2. If bid < num_valid_ids (valid row):
     a. Reads gate = tmp_out[row, 0:inter_dim], up = tmp_out[row, inter_dim:2*inter_dim]
     b. Computes silu(gate) * up in f32
     c. Per-1x32 MXFP4 quant -> writes packed FP4 + E8M0 scale in tiled layout
  3. If bid >= num_valid_ids (blockM padding row):
     a. Writes zero FP4 bytes to out_fp4
     b. Writes zero E8M0 scale to out_scale_sorted (keeps tiled layout consistent)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, rocdl, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl.expr import buffer_ops, math as fx_math

BLOCK_THREADS = 256
WARP_SIZE = 64


def _make_scale_tiled_layout(scale_cols_i32):
    """Build hierarchical 2-D layout for sorted E8M0 scale bytes.

    Uses flydsl's hierarchical shape to express the tiled decomposition::

        row  -> (row%16,  (row//16)%2,  row//32)
        col  -> (col%4,   (col//4)%2,   col//8)

    flydsl decomposes hierarchical shapes innermost-first (left-to-right),
    so shape ``(16, 2, N)`` yields ``(idx%16, (idx//16)%2, idx//32)``.

    Strides: ``(4, 1, n32_sort, 64, 2, 256)`` where
    ``n32_sort = scale_cols * 32``.
    """
    n32_sort = scale_cols_i32 * arith.constant(32, type=T.i32)
    return fx.make_layout(
        ((16, 2, 32), (4, 2, 8)),
        stride=((4, 1, n32_sort), (64, 2, arith.constant(256, type=T.i32))),
    )


def _scale_byte_offset(layout_scale, row, col32):
    """Compute byte offset for one E8M0 scale element via layout algebra."""
    result = fx.crd2idx(fx.make_coord(row, col32), layout_scale)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
        scalar = arith.index_cast(T.index, scalar)
    return ArithValue(scalar)


def build_silu_and_mul_fq_module(inter_dim: int, topk: int):
    """Return a JIT launcher for fused silu_and_mul + mxfp4 quant + scale sort.

    Parameters
    ----------
    inter_dim : int
        Output columns of stage1 (after activation). Input has inter_dim*2 cols.
        Must be divisible by 32 (MXFP4 block size).
    topk : int
        Number of expert slots per token.
    """
    assert inter_dim % 32 == 0, f"inter_dim={inter_dim} must be divisible by 32"

    scale_cols = inter_dim // 32
    ELEMS_PER_THREAD = (inter_dim + BLOCK_THREADS - 1) // BLOCK_THREADS
    VEC = max(ELEMS_PER_THREAD, 2)
    if VEC % 2 != 0:
        VEC += 1
    assert 32 % VEC == 0, f"VEC={VEC} must divide 32 evenly"
    THREADS_PER_QUANT_BLK = 32 // VEC
    SHUFFLE_DISTS = []
    d = 1
    while d < THREADS_PER_QUANT_BLK:
        SHUFFLE_DISTS.append(d)
        d *= 2

    elem_bytes_bf16 = 2
    input_row_bytes = inter_dim * 2 * elem_bytes_bf16
    fp4_row_bytes = inter_dim // 2
    up_byte_offset_static = inter_dim * elem_bytes_bf16

    @flyc.kernel
    def silu_and_mul_fq_kernel(
        x: fx.Tensor,
        out_fp4: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        token_num: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        c2 = arith.constant(2, type=i32)
        c4 = arith.constant(4, type=i32)

        scale_cols_i32 = arith.constant(scale_cols, type=i32)
        inter_dim_i32 = arith.constant(inter_dim, type=i32)
        topk_i32 = arith.constant(topk, type=i32)

        layout_scale = _make_scale_tiled_layout(scale_cols_i32)

        in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_fp4, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(out_scale_sorted, max_size=True)
        tid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, c0, vec_width=1, dtype=i32)
        token_num_i32 = ArithValue(token_num)
        bid_i32 = ArithValue(bid)

        row_in_range = bid_i32 < num_valid
        fused_tid_val = buffer_ops.buffer_load(tid_rsrc, bid_i32, vec_width=1, dtype=i32)
        token_id = fused_tid_val & 0xFFFFFF
        slot_id = ArithValue(fused_tid_val) >> 24
        t_ok = token_id < token_num_i32
        s_ok = slot_id < topk_i32
        is_valid = row_in_range & (t_ok & s_ok)

        def _f32_to_e2m1(qx_f32):
            """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
            qx = qx_f32.bitcast(i32)
            s = qx & 0x80000000
            e = (qx >> 23) & 0xFF
            m = qx & 0x7FFFFF
            c126 = arith.constant(126, type=i32)
            adj_exp = arith.maxsi(c126 - e, c0)
            m_denorm = (0x400000 | (m >> c1)) >> adj_exp
            is_denorm = e < arith.constant(127, type=i32)
            m = is_denorm.select(m_denorm, m)
            e = arith.maxsi(e - c126, c0)
            combined = (e << c2) | (m >> 21)
            rounded = (combined + c1) >> c1
            e2m1 = arith.minui(rounded, arith.constant(7, type=i32))
            return (s >> 28) | e2m1

        thread_id = ArithValue(tid)
        COLS_PER_ITER = BLOCK_THREADS * VEC
        c0_f32 = arith.constant(0.0, type=f32)

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = thread_id * VEC + iter_idx * COLS_PER_ITER

            if col0 < inter_dim_i32:
                if is_valid:
                    in_row = token_id * topk_i32 + slot_id

                    in_byte_base = in_row * input_row_bytes
                    gate_byte = in_byte_base + col0 * elem_bytes_bf16
                    up_byte = gate_byte + up_byte_offset_static
                    gate_dw = gate_byte >> c2
                    up_dw = up_byte >> c2
                    vec_dw = VEC * elem_bytes_bf16 // 4

                    gate_raw = buffer_ops.buffer_load(
                        in_rsrc, gate_dw, vec_width=vec_dw, dtype=i32
                    )
                    up_raw = buffer_ops.buffer_load(
                        in_rsrc, up_dw, vec_width=vec_dw, dtype=i32
                    )

                    vec_bf16_ty = T.vec(VEC, T.bf16)
                    vec_f32_ty = T.vec(VEC, f32)
                    if vec_dw == 1:
                        gate_vec = vector.from_elements(T.vec(1, i32), [gate_raw])
                        up_vec = vector.from_elements(T.vec(1, i32), [up_raw])
                        gate_bf16 = vector.bitcast(vec_bf16_ty, gate_vec)
                        up_bf16 = vector.bitcast(vec_bf16_ty, up_vec)
                    else:
                        gate_bf16 = vector.bitcast(vec_bf16_ty, gate_raw)
                        up_bf16 = vector.bitcast(vec_bf16_ty, up_raw)
                    gate_f32 = gate_bf16.extf(vec_f32_ty)
                    up_f32 = up_bf16.extf(vec_f32_ty)

                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    c1_f32 = arith.constant(1.0, type=f32)
                    act_vals = []
                    for vi in range_constexpr(VEC):
                        g = vector.extract(gate_f32, static_position=[vi], dynamic_position=[])
                        u = vector.extract(up_f32, static_position=[vi], dynamic_position=[])
                        t = g * neg_log2e
                        emu = ArithValue(rocdl.exp2(f32, t))
                        den = c1_f32 + emu
                        sig = ArithValue(rocdl.rcp(f32, den))
                        act_vals.append(g * sig * u)

                    local_max = c0_f32
                    for vi in range_constexpr(VEC):
                        abs_v = fx_math.absf(act_vals[vi])
                        local_max = local_max.maximumf(abs_v)

                    for sh_dist in SHUFFLE_DISTS:
                        peer = local_max.shuffle_xor(
                            arith.constant(sh_dist, type=i32),
                            arith.constant(64, type=i32),
                        )
                        local_max = local_max.maximumf(peer)

                    max_i32_v = local_max.bitcast(i32)
                    max_rounded = (max_i32_v + 0x200000) & 0xFF800000
                    exp_field = max_rounded >> 23
                    e8m0_biased = arith.maxsi(exp_field - c2, c0)

                    quant_exp = arith.constant(254, type=i32) - e8m0_biased
                    quant_scale = (quant_exp << 23).bitcast(f32)

                    fp4_vals = []
                    for vi in range_constexpr(VEC):
                        fp4_vals.append(_f32_to_e2m1(act_vals[vi] * quant_scale))

                    packed_i32 = fp4_vals[0] | (fp4_vals[1] << c4)
                    for k in range_constexpr(1, VEC // 2):
                        byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << c4)
                        packed_i32 = packed_i32 | (byte_k << (k * 8))

                    fp4_byte_off = in_row * fp4_row_bytes + (col0 >> c1)
                    _pack_bytes = VEC // 2
                    if _pack_bytes == 1:
                        buffer_ops.buffer_store(
                            arith.trunci(T.i8, packed_i32),
                            out_rsrc, fp4_byte_off, offset_is_bytes=True,
                        )
                    elif _pack_bytes == 2:
                        buffer_ops.buffer_store(
                            arith.trunci(T.i16, packed_i32),
                            out_rsrc, fp4_byte_off, offset_is_bytes=True,
                        )
                    else:
                        buffer_ops.buffer_store(
                            packed_i32, out_rsrc, fp4_byte_off, offset_is_bytes=True,
                        )

                    if (col0 & 31) == c0:
                        col_s = col0 >> 5
                        s_off = _scale_byte_offset(layout_scale, bid_i32, col_s)
                        buffer_ops.buffer_store(
                            arith.trunci(T.i8, e8m0_biased),
                            scale_rsrc, s_off, offset_is_bytes=True,
                        )
                else:
                    if (col0 & 31) == c0:
                        col_s = col0 >> 5
                        s_off = _scale_byte_offset(layout_scale, bid_i32, col_s)
                        buffer_ops.buffer_store(
                            arith.trunci(T.i8, c0),
                            scale_rsrc, s_off, offset_is_bytes=True,
                        )

    @flyc.jit
    def launch_silu_and_mul_fq(
        x: fx.Tensor,
        out_fp4: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_sorted_rows)
        launcher = silu_and_mul_fq_kernel(
            x, out_fp4, out_scale_sorted, sorted_ids, num_valid_ids, token_num
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_silu_and_mul_fq
