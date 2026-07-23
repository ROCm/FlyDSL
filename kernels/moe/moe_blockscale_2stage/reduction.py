# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE block-scale 2-stage MFMA kernels (stage1 / stage2 / reduction). :: _reduction"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, vector
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common import buffer_ops
from kernels.common.kernels_common import _if_else, _if_then


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            valid_mask [tokens, topk] (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    When use_mask=True, only sums slots where valid_mask[t,k]=1.
    Used in conjunction with compile_moe_blockscale_gemm2(accumulate=False) to avoid atomic contention.
    """
    get_rocm_arch()
    ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8

    masked = "masked" if use_mask else ""

    module_name = f"bs_moe_reduce_topk{topk}_{dtype_str}{masked}"

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    compute_type = lambda: T.f32
    i8_type = lambda: T.i8

    def elem_type():
        ty = T.f32 if elem_type_tag == "f32" else (T.f16 if elem_type_tag == "f16" else T.bf16)
        return ty() if callable(ty) else ty

    if True:

        @flyc.kernel(name=module_name)
        def moe_reduction_kernel(
            X: fx.Tensor,
            Y: fx.Tensor,
            valid_mask: fx.Tensor,
            i32_m_tokens: fx.Int32,
        ):
            m_tokens = fx.Index(i32_m_tokens)
            c_topk = fx.Index(topk)
            c_model_dim = fx.Index(model_dim)
            mask_nbytes_idx = m_tokens * c_topk
            elem_bits = 32 if dtype_str == "f32" else 16
            copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
            n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32
            # Buffer-backed tensors via layout API (all dtypes)
            X_buf = fx.rocdl.make_buffer_tensor(X)
            Y_buf = fx.rocdl.make_buffer_tensor(Y)
            # Scalar buffer resources for tail path and mask
            x_rsrc = buffer_ops.create_buffer_resource(X, max_size=True)
            y_rsrc = buffer_ops.create_buffer_resource(Y, max_size=True)
            mask_rsrc = buffer_ops.create_buffer_resource(valid_mask, max_size=False, num_records_bytes=mask_nbytes_idx)

            token_idx = gpu.block_id("x")
            tile_idx = gpu.block_id("y")
            tid = gpu.thread_id("x")

            # Guard: token in range (Index is unsigned → auto ult)
            tok_ok = token_idx < m_tokens
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                # Guard: any work in bounds (Index < → ult)
                col_ok = col_base < c_model_dim
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    # Fast path: full vector in-bounds (Index <= → ule)
                    end_ok = col_base + c_vecw <= c_model_dim
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        # ── Vector path via layout API (all dtypes) ──
                        # fx.copy auto-iterates when atom width < VEC_WIDTH
                        # (e.g. f32: BufferCopy128b handles 4, fx.copy issues 2 calls for 8)
                        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                        vec_type_c = T.vec(copy_vec_width, compute_type())
                        vec_type_e = T.vec(copy_vec_width, elem_type())

                        acc_vecs = [vector.broadcast(vec_type_c, fx.Float32(0.0).ir_value()) for _ in range(n_sub)]
                        elem_dtype = fx.Numeric.from_ir_type(elem_type())

                        tok_i32 = fx.Int32(token_idx)
                        tile_i32 = fx.Int32(tile_idx)
                        tid_i32 = fx.Int32(tid)

                        for k in range_constexpr(topk):
                            # X[token, k, :] → tile → thread's VEC_WIDTH slice
                            x_row = X_buf[tok_i32, fx.Int32(k), None]
                            x_tiled = fx.logical_divide(x_row, fx.make_layout(tile_cols, 1))
                            x_div = fx.logical_divide(x_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                            x_thread = x_div[None, tid_i32]

                            if const_expr(use_mask):
                                m_idx_i32 = fx.Int32(token_idx * c_topk + fx.Index(k))
                                mv = buffer_ops.buffer_load(mask_rsrc, m_idx_i32, vec_width=1, dtype=i8_type())
                                mv_ok = mv != fx.Int8(0)

                            if const_expr(n_sub > 1):
                                x_inner = fx.logical_divide(x_thread, fx.make_layout(copy_vec_width, 1))
                            for si in range_constexpr(n_sub):
                                src = x_inner[None, fx.Int32(si)] if n_sub > 1 else x_thread
                                r = fx.make_rmem_tensor(copy_vec_width, elem_dtype)
                                fx.copy_atom_call(copy_atom, src, r)
                                vec_e = fx.memref_load_vec(r)

                                if const_expr(use_mask):
                                    zero_e = vector.broadcast(vec_type_e, arith.constant(0.0, type=elem_type()))
                                    vec_e = mv_ok.select(vec_e, zero_e)

                                if const_expr(elem_bits < 32):
                                    vec_c = vec_e.extf(vec_type_c)
                                else:
                                    vec_c = vec_e
                                acc_vecs[si] = acc_vecs[si] + vec_c

                        # ── Store results ──
                        if const_expr(n_sub > 1):
                            y_row = Y_buf[tok_i32, None]
                            y_tiled = fx.logical_divide(y_row, fx.make_layout(tile_cols, 1))
                            y_div = fx.logical_divide(y_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                            y_inner = fx.logical_divide(y_div[None, tid_i32], fx.make_layout(copy_vec_width, 1))

                        for si in range_constexpr(n_sub):
                            out_vec = acc_vecs[si]
                            if const_expr(elem_bits < 32):
                                out_vec = out_vec.truncf(vec_type_e)

                            if const_expr(n_sub > 1):
                                dst = y_inner[None, fx.Int32(si)]
                            else:
                                y_row = Y_buf[tok_i32, None]
                                y_tiled = fx.logical_divide(y_row, fx.make_layout(tile_cols, 1))
                                y_div = fx.logical_divide(y_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1))
                                dst = y_div[None, tid_i32]

                            r_out = fx.make_rmem_tensor(copy_vec_width, elem_dtype)
                            fx.memref_store_vec(out_vec, r_out)
                            fx.copy_atom_call(copy_atom, r_out, dst)

                    with _if_else(_if_full):
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            lane_ok = col < c_model_dim
                            _if_lane = scf.IfOp(lane_ok)
                            with _if_then(_if_lane):
                                a = arith.constant(0.0, type=compute_type())
                                token_base = token_idx * c_topk
                                for k in range_constexpr(topk):
                                    k_idx = fx.Index(k)
                                    x_idx_i32 = fx.Int32((token_base + k_idx) * c_model_dim + col)
                                    if const_expr(use_mask):
                                        m_idx_i32 = fx.Int32(token_base + k_idx)
                                        mv = buffer_ops.buffer_load(mask_rsrc, m_idx_i32, vec_width=1, dtype=i8_type())
                                        v = (mv != fx.Int8(0)).select(
                                            buffer_ops.buffer_load(x_rsrc, x_idx_i32, vec_width=1, dtype=elem_type()),
                                            arith.constant(0.0, type=elem_type()),
                                        )
                                    else:
                                        v = buffer_ops.buffer_load(x_rsrc, x_idx_i32, vec_width=1, dtype=elem_type())
                                    if const_expr(dtype_str in ("f16", "bf16")):
                                        v = v.extf(compute_type())
                                    a = a + v

                                out = a
                                if const_expr(dtype_str in ("f16", "bf16")):
                                    out = out.truncf(elem_type())
                                y_idx_i32 = fx.Int32(token_idx * c_model_dim + col)
                                buffer_ops.buffer_store(out, y_rsrc, y_idx_i32)

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(i32_m_tokens)
        moe_reduction_kernel(X, Y, valid_mask, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction
