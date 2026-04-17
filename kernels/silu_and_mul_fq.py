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
from flydsl.expr import buffer_ops, math as fx_math
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir

from kernels.kernels_common import get_warp_size

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()


def _make_scale_tiled_layout(scale_cols_val):
    """Build hierarchical 2-D layout for sorted E8M0 scale bytes.

    Uses flydsl's hierarchical shape to express the tiled decomposition::

        row  -> (row%16,  (row//16)%2,  row//32)
        col  -> (col%4,   (col//4)%2,   col//8)

    flydsl decomposes hierarchical shapes innermost-first (left-to-right),
    so shape ``(16, 2, N)`` yields ``(idx%16, (idx//16)%2, idx//32)``.

    Strides: ``(4, 1, n32_sort, 64, 2, 256)`` where
    ``n32_sort = scale_cols * 32``.
    """
    n32_sort = scale_cols_val * 32
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

    fp4_row_bytes = inter_dim // 2
    _pack_bytes = VEC // 2

    @flyc.kernel
    def silu_and_mul_fq_kernel(
        x: fx.Tensor,
        out_fp4: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        token_num: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        vec_f32_ty = T.vec(VEC, T.f32)

        # ── Layout API: buffer-backed tensor for structured input ────
        X_buf = fx.rocdl.make_buffer_tensor(x)
        copy_atom_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy(VEC * 16), 16  # bf16 = 16 bits
        )
        vec_reg_ty = fx.MemRefType.get(
            T.bf16, fx.LayoutType.get(VEC, 1), fx.AddressSpace.Register
        )
        vec_reg_lay = fx.make_layout(VEC, 1)

        def load_vec(div_tensor, idx):
            r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
            fx.copy_atom_call(copy_atom_vec, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        # ── Buffer resources for flat byte buffers and scalar loads ───
        out_rsrc = buffer_ops.create_buffer_resource(out_fp4, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(out_scale_sorted, max_size=True)
        tid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        bid_i32 = ArithValue(bid)

        fused_tid_val = buffer_ops.buffer_load(tid_rsrc, bid_i32, vec_width=1, dtype=T.i32)
        token_id = fused_tid_val & 0xFFFFFF
        slot_id = fused_tid_val >> 24
        is_valid = (bid_i32 < num_valid) & (token_id < ArithValue(token_num)) & (slot_id < topk)

        layout_scale = _make_scale_tiled_layout(
            ArithValue(arith.constant(scale_cols, type=T.i32))
        )

        def _store_scale(scale_rsrc, layout_scale, bid_i32, col0, val_i8):
            if (col0 & 31) == fx.Int32(0):
                s_off = _scale_byte_offset(layout_scale, bid_i32, col0 >> 5)
                buffer_ops.buffer_store(
                    val_i8, scale_rsrc, s_off, offset_is_bytes=True,
                )

        def _f32_to_e2m1(qx_f32):
            """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
            qx = qx_f32.bitcast(T.i32)
            s = qx & 0x80000000
            e = (qx >> 23) & 0xFF
            m = qx & 0x7FFFFF
            c0_i32 = arith.constant(0, type=T.i32)
            c126 = arith.constant(126, type=T.i32)
            adj_exp = arith.maxsi(c126 - e, c0_i32)
            m_denorm = (0x400000 | (m >> 1)) >> adj_exp
            m = (e < arith.constant(127, type=T.i32)).select(m_denorm, m)
            e = arith.maxsi(e - c126, c0_i32)
            rounded = ((e << 2) | (m >> 21)) + 1 >> 1
            e2m1 = arith.minui(rounded, arith.constant(7, type=T.i32))
            return (s >> 28) | e2m1

        thread_id = ArithValue(tid)
        COLS_PER_ITER = BLOCK_THREADS * VEC
        c0_f32 = arith.constant(0.0, type=T.f32)

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = thread_id * VEC + iter_idx * COLS_PER_ITER

            if col0 < inter_dim:
                if is_valid:
                    in_row = token_id * topk + slot_id

                    row_x = fx.slice(X_buf, (in_row, None))
                    row_div = fx.logical_divide(row_x, fx.make_layout(VEC, 1))
                    tile_idx = tid + iter_idx * BLOCK_THREADS

                    gate_f32 = load_vec(row_div, tile_idx).extf(vec_f32_ty)
                    up_f32 = load_vec(row_div, tile_idx + inter_dim // VEC).extf(vec_f32_ty)

                    # ── SiLU(gate) * up ──────────────────────────────
                    neg_log2e = arith.constant(-1.4426950408889634, type=T.f32)
                    c1_f32 = arith.constant(1.0, type=T.f32)
                    act_vals = []
                    for vi in range_constexpr(VEC):
                        g = vector.extract(gate_f32, static_position=[vi], dynamic_position=[])
                        u = vector.extract(up_f32, static_position=[vi], dynamic_position=[])
                        emu = ArithValue(rocdl.exp2(T.f32, g * neg_log2e))
                        sig = ArithValue(rocdl.rcp(T.f32, c1_f32 + emu))
                        act_vals.append(g * sig * u)

                    # ── Per-32-block max for E8M0 scale ──────────────
                    local_max = c0_f32
                    for vi in range_constexpr(VEC):
                        local_max = local_max.maximumf(fx_math.absf(act_vals[vi]))

                    for sh_dist in SHUFFLE_DISTS:
                        local_max = local_max.maximumf(
                            local_max.shuffle_xor(fx.Int32(sh_dist), fx.Int32(WARP_SIZE))
                        )

                    # ── Quantize to FP4 ──────────────────────────────
                    exp_field = ((local_max.bitcast(T.i32) + 0x200000) & 0xFF800000) >> 23
                    e8m0_biased = arith.maxsi(
                        exp_field - arith.constant(2, type=T.i32),
                        arith.constant(0, type=T.i32),
                    )
                    quant_scale = (
                        (arith.constant(254, type=T.i32) - e8m0_biased) << 23
                    ).bitcast(T.f32)

                    fp4_vals = []
                    for vi in range_constexpr(VEC):
                        fp4_vals.append(_f32_to_e2m1(act_vals[vi] * quant_scale))

                    packed_i32 = fp4_vals[0] | (fp4_vals[1] << 4)
                    for k in range_constexpr(1, VEC // 2):
                        byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << 4)
                        packed_i32 = packed_i32 | (byte_k << (k * 8))

                    fp4_byte_off = in_row * fp4_row_bytes + (col0 >> 1)
                    _pack_type = {1: T.i8, 2: T.i16}.get(_pack_bytes, T.i32)
                    packed = arith.trunci(_pack_type, packed_i32) if _pack_bytes < 4 else packed_i32
                    buffer_ops.buffer_store(
                        packed, out_rsrc, fp4_byte_off, offset_is_bytes=True,
                    )

                    _store_scale(scale_rsrc, layout_scale, bid_i32, col0,
                                 arith.trunci(T.i8, e8m0_biased))
                else:
                    _store_scale(scale_rsrc, layout_scale, bid_i32, col0,
                                 arith.constant(0, type=T.i8))

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

        idx_rows = ArithValue(num_sorted_rows).index_cast(T.index)
        launcher = silu_and_mul_fq_kernel(
            x, out_fp4, out_scale_sorted, sorted_ids, num_valid_ids, token_num
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_silu_and_mul_fq
