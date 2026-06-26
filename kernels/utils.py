# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Basics for the layout-API MXFP4 MoE gemm: shape constants, K-derived size
formulas, raw pointer/LDS helpers, and the e8m0 / SwiGLU quant math. The MMA + A/B
data movement live in ``moegemm``; compile + launch in ``moe_dispatcher``.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, rocdl
from flydsl.expr.typing import T

# -- shape constants -----------------------------------------------------------
# gemm1 (up/gate-proj) KIMI defaults: D_HIDDEN (K, contraction), D_INTER (output),
# NE (#experts), TOPK. Per-shape values come from the compile args.
NE_DEFAULT, K_DEFAULT, INTER_DEFAULT, TOPK_DEFAULT = 385, 7168, 512, 9
# gemm2 (down-proj) KIMI defaults: contraction K = inter_dim; N_OUT = model_dim.
MAX_M = 655360
NE = 385
K = 512
N_OUT = 7168
# tiling (BM-independent).
BN = BK = 256
KH_TILE = BK // 2  # 128 packed-fp4 bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64  # e8m0 scale-layout K-independent stride
LOG2E = 1.4426950408889634
_PTR3 = "!llvm.ptr<3>"


# -- K-derived sizes (parametrized over the contraction dim K = inter_dim) -----
def k_half_for(k):
    return k // 2  # packed-fp4 bytes along K (KIMI: 256)


def k_tiles_total_for(k):
    return k // BK  # KIMI: 2


def kunroll_for(k):
    # streaming main-loop trip count: kUnroll = K_TILES_TOTAL - kStages.
    return k_tiles_total_for(k) - kStages


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64  # KIMI: 128


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64  # KIMI: 128


# -- shape-parametrized sizes (NE/N_OUT vary per instance; N_OUT % 256 == 0) ----
def num_n_blocks_for(n_out):
    return n_out // 256


def kbs_per_expert_dw_for(n_out, k=K):
    return (n_out // 16 // 2) * kbs_stride_n0_dw_for(k)


def kmchunks(BM):
    return 1 if const_expr(BM == 16) else BM // 16


# -- raw / pointer / LDS helpers ----------------------------------------------
def _raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _lds_ptr3(base_i32, byte_off_i32):
    """ptr<3> = inttoptr(i64(base_i32 + byte_off_i32))."""
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32 + byte_off_i32)))


def _lds_base_ptr3(lds_view):
    """One ptr<3> for the LDS base; offsets via GEP."""
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<3>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_base_ptr1(addr_i64):
    """One ptr<1> base from a raw i64 device address (bare data_ptr() kernarg)."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row) = (row & 14) << 3 (fp4 A tile)."""
    return (row & fx.Int32(14)) << fx.Int32(3)


def _lds_swizzle_mask_f8(row):
    """lds_swizzle_mask<ROW_BYTES=256>(row) = (row & 15) << 4 (fp8 A tile)."""
    return (row & fx.Int32(15)) << fx.Int32(4)


# -- e8m0 / SwiGLU quant math -------------------------------------------------
def _silu_mul_batch(gs, us):
    """silu(g)*u via exp2/rcp (matches HIP silu_mul_fast)."""
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _fabs_f32(x):
    """fabsf via sign-bit clear (FlyDSL has no arith.absf)."""
    abs_bits = _raw(x).bitcast(T.i32) & _raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def _e8m0_from_amax(amax_f32, dtype_max=6.0):
    """(e8m0_i32, quant_scale_f32) = ceil_pow2(amax/dtype_max) clamped to 254.
    dtype_max is the output format's max magnitude (fp4 e2m1 = 6, fp8 e4m3 = 448)."""
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / dtype_max)).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, _raw(bexp), _raw(fx.Int32(254)))
    e8m0 = fx.Int32(arith.select(lt, _raw(bexp), _raw(fx.Int32(254))))
    qscale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale
