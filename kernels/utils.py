# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Basics for the layout-API MXFP4 MoE gemm: shape/size constants, pointer/LDS
helpers, e8m0 + SwiGLU quant math. MMA + data movement live in ``moegemm``."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, rocdl
from flydsl.expr.typing import T

# -- shape constants (KIMI defaults; per-shape values come from the compile args) --
NE = 385  # #experts
TOPK_DEFAULT = 9
H_DEFAULT = 7168  # model_dim: gemm1 D_HIDDEN (contraction) / gemm2 N_OUT (output)
INTER_DEFAULT = 512  # inter_dim: gemm1 D_INTER (output) / gemm2 D_INTER (contraction)
MAX_M = 655360
# tiling (BM-independent).
BN = BK = 256
KH_TILE = BK // 2  # 128 packed-fp4 bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64  # e8m0 scale-layout K-independent stride
LOG2E = 1.4426950408889634


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


def kbs_per_expert_dw_for(n_out, k=INTER_DEFAULT):
    return (n_out // 16 // 2) * kbs_stride_n0_dw_for(k)


def kmchunks(BM):
    return 1 if const_expr(BM == 16) else BM // 16


# -- raw / pointer / LDS helpers ----------------------------------------------
def raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(raw(a), raw(cc)))


def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    """LDS dst view for a buffer_load_lds DMA (align 16 for 128b, 4 for 32b chunks).
    Gotcha: FlyDSL AddressSpace.Shared = LDS (enum 2, NOT LLVM addrspace 3)."""
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def global_base_ptr1(addr_i64):
    """One ptr<1> base from a raw i64 device address (bare data_ptr() kernarg)."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), raw(fx.Int64(addr_i64)))


def gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=raw(byte_off_i32), elem_type=T.i8)


def global_ptr1(arg, byte_off_i32):
    return gep1(global_base_ptr1(arg), byte_off_i32)


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS
    (ptr[i] / ptr[i] = v), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, raw(fx.Int64(arg)))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base address; index in ELEMENTS
    (ptr[i] / ptr[i] = v), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at BYTE offset from the i32 LDS base; mirrors raw
    llvm.load(result_type, gep_i8(base, off)). result_type may be vector or scalar."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row) = (row & 14) << 3 (fp4 A tile)."""
    return (row & fx.Int32(14)) << fx.Int32(3)


def lds_swizzle_mask_f8(row):
    """lds_swizzle_mask<ROW_BYTES=256>(row) = (row & 15) << 4 (fp8 A tile)."""
    return (row & fx.Int32(15)) << fx.Int32(4)


# -- e8m0 / SwiGLU quant math -------------------------------------------------
def silu_mul_batch(gs, us):
    """silu(g)*u via exp2/rcp (matches HIP silu_mul_fast)."""
    e = [fx.Float32(rocdl.exp2(T.f32, raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def fabs_f32(x):
    """fabsf via sign-bit clear (FlyDSL has no arith.absf)."""
    abs_bits = raw(x).bitcast(T.i32) & raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def e8m0_from_amax(amax_f32, dtype_max=6.0):
    """(e8m0_i32, quant_scale_f32) = ceil_pow2(amax/dtype_max) clamped to 254.
    dtype_max is the output format's max magnitude (fp4 e2m1 = 6, fp8 e4m3 = 448)."""
    wi = fx.Int32(raw(amax_f32 * fx.Float32(1.0 / dtype_max)).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, raw(bexp), raw(fx.Int32(254)))
    e8m0 = fx.Int32(arith.select(lt, raw(bexp), raw(fx.Int32(254))))
    qscale = fx.Float32(raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale
