"""Stage2 combine kernel for mla_decode_fp8_v4 — gfx1250 flydsl.

Combines per-split partial outputs (mid_o + mid_lse) from Stage1
``mla_decode_fp8_v4_kernel`` into the final O via online log-sum-exp.

Inputs (from Stage1):
  mid_o   : fp32 flat, logical (total_q, num_kv_splits, Hq, D_V)
            mid_o[q, sp, h] is the per-split normalized output
            (= local acc_s / e_sum_s).
  mid_lse : fp32 flat, logical (total_q, num_kv_splits, Hq)
            mid_lse[q, sp, h] = local_max + log(local_e_sum) (natural log).

Output:
  O : bf16 (total_q, Hq, D_V) flat.

Algorithm (per (q, h)):
    e_max = -inf; e_sum = 0; acc = 0
    for sp in [0, num_kv_splits):
        tv      = mid_o[q, sp, h, :]            # vec<D_V xf32>
        tlogic  = mid_lse[q, sp, h]             # scalar
        n_e_max  = max(tlogic, e_max)
        old_scale = exp(e_max - n_e_max)
        acc     = acc * old_scale + exp(tlogic - n_e_max) * tv
        e_sum    = e_sum * old_scale + exp(tlogic - n_e_max)
        e_max    = n_e_max
    O[q, h, :] = (acc / e_sum).to(bf16)

For num_kv_splits=1: acc=tv, e_sum=1 → O = mid_o (just bf16 cast).

Grid : (total_q, Hq, 1).  Block : (WAVE_SIZE=32, 1, 1) — single wave per WG.
Each lane handles cols_per_lane = D_V / WAVE_SIZE columns of D_V.
"""

from typing import Any, Callable, Optional
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as _std_arith,
    math as _mlir_math,
)
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


WAVE_SIZE = 32

# Natural log helpers via exp2 (gfx1250 has hardware exp2; natural exp uses
# `exp(x) = exp2(x * log2(e))`).
_LOG2E = 1.4426950408889634


def _bf16_vec_ty(n):
    return ir.VectorType.get([n], ir.BF16Type.get())


def _f32_vec_ty(n):
    return ir.VectorType.get([n], ir.F32Type.get())


def compile_mla_decode_fp8_v4_stage2(
    head_dim_v: int = 512,
    num_q_heads: int = 128,
    num_kv_splits: int = 1,
) -> Callable[..., Any]:
    """Compile Stage2 combine kernel.

    Always applies V4 attn_sink ref form post-combine:
        O *= 1 / (1 + exp(sink[h] - lse_combined))
    Pass attn_sink = -inf to disable (exp(-inf)=0 → scale=1, identity).
    Current V4 weights init attn_sink = -inf so disabling is the default.
    """
    assert head_dim_v % WAVE_SIZE == 0, (
        f"head_dim_v={head_dim_v} must be a multiple of WAVE_SIZE={WAVE_SIZE}"
    )
    cols_per_lane = head_dim_v // WAVE_SIZE
    # We chunk the per-lane vec<cols_per_lane xf32> load/store into vec<4xf32>
    # pieces (= ds_load_b128/buffer_load_dwordx4 friendly).
    assert cols_per_lane % 4 == 0, (
        f"cols_per_lane={cols_per_lane} must be a multiple of 4"
    )
    n_chunks = cols_per_lane // 4

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).lower().startswith("gfx12"):
        raise RuntimeError(
            f"compile_mla_decode_fp8_v4_stage2: requires GFX12, got {gpu_arch!r}"
        )

    # ===================================================================
    # Kernel
    # ===================================================================
    @flyc.kernel
    def stage2_kernel(
        mid_o: fx.Tensor,    # fp32 flat (total_q * num_kv_splits * Hq * D_V)
        mid_lse: fx.Tensor,  # fp32 flat (total_q * num_kv_splits * Hq)
        attn_sink: fx.Tensor,# fp32 (Hq,) — V4 attn_sink (-inf to disable)
        O: fx.Tensor,        # bf16 flat (total_q * Hq * D_V)
    ):
        # ---------------- Type aliases ----------------
        ty_4xf32 = T.vec(4, T.f32)
        ty_clxf32 = _f32_vec_ty(cols_per_lane)
        ty_clxbf16 = _bf16_vec_ty(cols_per_lane)

        # ---------------- HBM resources ----------------
        mid_o_rsrc = buffer_ops.create_buffer_resource(mid_o, max_size=True)
        mid_lse_rsrc = buffer_ops.create_buffer_resource(mid_lse, max_size=True)
        sink_rsrc = buffer_ops.create_buffer_resource(attn_sink, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)

        # ---------------- Compile-time index values ----------------
        head_dim_v_i = arith.index(head_dim_v)
        num_q_heads_i = arith.index(num_q_heads)
        num_kv_splits_i = arith.index(num_kv_splits)
        cols_per_lane_i = arith.index(cols_per_lane)
        log2e_const = arith.constant(_LOG2E, type=T.f32)

        # ---------------- Thread / block indexing ----------------
        lane = gpu.thread_id("x")
        cur_q = gpu.block_id("x")    # in [0, total_q)
        cur_h = gpu.block_id("y")    # in [0, Hq)

        col_base = arith.muli(lane, cols_per_lane_i)   # head_dim col offset

        # ---------------- Helpers ----------------
        def _exp(x):
            # exp(x) = exp2(x * log2e)
            x_log2 = arith.mulf(x, log2e_const)
            return _mlir_math.Exp2Op(_raw(x_log2)).result

        def _load_tv(mid_o_base):
            """Read vec<cols_per_lane xf32> as n_chunks × vec<4xf32>, shuffle-concat.

            mid_o_base = element offset to start of this lane's slice.
            """
            chunks = []
            for c in range_constexpr(n_chunks):
                off = arith.addi(mid_o_base, arith.index(c * 4))
                chunks.append(buffer_ops.buffer_load(
                    mid_o_rsrc, off, vec_width=4, dtype=T.f32,
                ))
            # Pairwise shuffle-concat into vec<cols_per_lane xf32>.
            # n_chunks must be a power of 2 for the recursion below; if not,
            # we fall back to a left-fold.
            cur = chunks[0]
            cur_w = 4
            for c in range_constexpr(n_chunks - 1):
                nxt = chunks[c + 1]
                cur = vector.shuffle(cur, nxt, list(range(cur_w + 4)))
                cur_w += 4
            return cur

        def _store_o(out_bf16, o_base):
            """Write vec<cols_per_lane xbf16> as n_chunks × vec<4xbf16>."""
            for c in range_constexpr(n_chunks):
                start = c * 4
                chunk_bf16 = vector.shuffle(
                    out_bf16, out_bf16, list(range(start, start + 4)),
                )
                off = arith.addi(o_base, arith.index(start))
                # bf16 buffer_store: vec<4xbf16> = 8 bytes = 2 i32. Reinterpret
                # via bitcast to vec<2xi32> for buffer_store.
                ty_2xi32 = T.vec(2, T.i32)
                chunk_i32 = vector.bitcast(ty_2xi32, chunk_bf16)
                buffer_ops.buffer_store(chunk_i32, o_rsrc, off // arith.index(2))

        # ---------------- Main combine loop ----------------
        # Initialize accumulator state per lane.
        neg_inf = arith.constant(-float("inf"), type=T.f32)
        zero_f = arith.constant(0.0, type=T.f32)
        acc = arith.constant_vector(0.0, ty_clxf32)
        e_max = neg_inf
        e_sum = zero_f

        # Per-split q,h base in mid_o / mid_lse.
        # mid_o_off = ((cur_q * num_kv_splits + sp) * Hq + cur_h) * D_V + col_base
        # mid_lse_off = (cur_q * num_kv_splits + sp) * Hq + cur_h
        cur_qh = arith.addi(
            arith.muli(cur_q, num_q_heads_i),
            cur_h,
        )

        for sp in range_constexpr(num_kv_splits):
            sp_i = arith.index(sp)
            # offsets
            qsp = arith.addi(
                arith.muli(cur_q, num_kv_splits_i),
                sp_i,
            )
            mid_lse_off = arith.addi(
                arith.muli(qsp, num_q_heads_i),
                cur_h,
            )
            mid_o_base = arith.addi(
                arith.muli(
                    arith.addi(arith.muli(qsp, num_q_heads_i), cur_h),
                    head_dim_v_i,
                ),
                col_base,
            )

            tv = _load_tv(mid_o_base)
            tlogic = buffer_ops.buffer_load(
                mid_lse_rsrc, mid_lse_off, vec_width=1, dtype=T.f32,
            )

            n_e_max = arith.maximumf(tlogic, e_max)
            old_scale = _exp(arith.subf(e_max, n_e_max))
            new_factor = _exp(arith.subf(tlogic, n_e_max))
            old_scale_v = vector.broadcast(ty_clxf32, old_scale)
            new_factor_v = vector.broadcast(ty_clxf32, new_factor)
            acc = arith.addf(
                arith.mulf(acc, old_scale_v),
                arith.mulf(tv, new_factor_v),
            )
            e_sum = arith.addf(arith.mulf(e_sum, old_scale), new_factor)
            e_max = n_e_max

        # Normalize: out = acc / e_sum
        inv_esum = arith.divf(arith.constant(1.0, type=T.f32), e_sum)
        inv_esum_v = vector.broadcast(ty_clxf32, inv_esum)
        out_f32 = arith.mulf(acc, inv_esum_v)

        # v2.5 attn_sink ref form:
        #   lse_combined = e_max + log(e_sum) (= e_max + log2(e_sum)/log2e)
        #   sink_scale = 1 / (1 + exp(sink[h] - lse))
        # Pass sink=-inf to disable (exp(-inf)=0 → scale=1.0).
        sink_h = buffer_ops.buffer_load(
            sink_rsrc, cur_h, vec_width=1, dtype=T.f32,
        )
        log_esum = arith.divf(_mlir_math.Log2Op(_raw(e_sum)).result, log2e_const)
        lse_combined = arith.addf(e_max, log_esum)
        delta = arith.subf(sink_h, lse_combined)
        sink_scale = arith.divf(
            arith.constant(1.0, type=T.f32),
            arith.addf(arith.constant(1.0, type=T.f32), _exp(delta)),
        )
        sink_scale_v = vector.broadcast(ty_clxf32, sink_scale)
        out_f32 = arith.mulf(out_f32, sink_scale_v)

        out_bf16 = _std_arith.TruncFOp(ty_clxbf16, _raw(out_f32)).result

        # Store to O[cur_q, cur_h, col_base : col_base+cols_per_lane]
        o_base = arith.addi(
            arith.muli(cur_qh, head_dim_v_i),
            col_base,
        )
        _store_o(out_bf16, o_base)

    # ===================================================================
    # JIT launcher
    # ===================================================================
    @flyc.jit
    def launch_stage2(
        mid_o: fx.Tensor,
        mid_lse: fx.Tensor,
        attn_sink: fx.Tensor,
        O: fx.Tensor,
        total_q: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, total_q.ir_value())
        gy = arith.index(num_q_heads)
        gz = arith.index(1)
        stage2_kernel(mid_o, mid_lse, attn_sink, O).launch(
            grid=(gx, gy, gz),
            block=(WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch_stage2
