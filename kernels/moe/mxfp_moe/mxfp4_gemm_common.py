# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, T
from kernels.common import buffer_ops
from kernels.common.layout_utils import crd2idx

from . import dpp_utils

_PTR3 = "!llvm.ptr<3>"
kStages = 2
kBS_stride_k0_dw = 64
LOG2E = 1.4426950408889634


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


# A elem selects the f8f6f4 cbsz for the scaled MFMA atom: fp4 (e2m1) -> cbsz 4,
# fp8 (e4m3) -> cbsz 0. B (blgp) is always mxfp4.
_A_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype):
    """16 (opsel_a, opsel_b) scaled 16x16x128 MFMA atoms; A elem = fp4/fp8, B = fp4."""
    elem_a = _A_ELEM[a_dtype]
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
        )
        for osa in range(4)
        for osb in range(4)
    }


def _global_i32_buffer_view(addr_i64, num_bytes):
    # fx.copy's BufferCopy/BufferCopyLDS atoms take soffset as an element count, not
    # the bytes buffer_ops.buffer_load's soffset_bytes expected.
    # make_layout's dynamic-shape leaf must be i32/i64, not fx.Index.
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=num_bytes_i64)


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(_global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1))


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _lds_base_ptr3(lds_view):
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    # Plain scalar read: fx pointer index, no tiling/register-fragment machinery.
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_load(tiles, idx):
    # Atom/types must be built with an active MLIR trace context, not as globals.
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy_atom_call(atom, fx.slice(tiles, (None, idx)), r)
    return r.load()[0]


def _global_scalar_tiles(addr_i64, numeric_cls, num_elems):
    ptr_ty = fx.PointerType.get(
        numeric_cls.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=numeric_cls.width // 8,
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    flat = fx.make_view(ptr, fx.make_layout(num_elems, 1))
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _scalar_store(tiles, idx, value, numeric_cls):
    atom = fx.make_copy_atom(fx.UniversalCopy(numeric_cls.width), numeric_cls)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), numeric_cls)
    r.store(fx.Vector.from_elements([numeric_cls(value)], numeric_cls))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    idx_coords = [fx.Int64(c) for c in coords]
    return fx.Int32(crd2idx(idx_coords, layout))


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(addr_i64)), num_records_bytes=num_records_bytes)


def _lds_swizzle_mask(row):
    return (row & fx.Int32(14)) << fx.Int32(3)


lds_swizzle_mask = _lds_swizzle_mask


def lds_swizzle_mask_f8(row):
    return (row & 15) << 4


def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    elem_ty = T.i32 if elem_ty is None else elem_ty
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    ptr = fx.inttoptr(ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(ptr, fx.make_layout(1, 1))


def global_typed_ptr(arg, elem_ty, align=4):
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_dma_atom_128():
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def flat_buffer_view(arg, base_elems, elem_ty, *, align, elem_bytes, fold=True, num_records_bytes=None):
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if fold:
        base = fx.Int32(fx.rocdl.readfirstlane(T.i32, base_elems))
        off_i64 = fx.Uint32(base).to(fx.Uint64).bitcast(fx.Int64)
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg) + off_i64 * fx.Int64(elem_bytes))
    else:
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg))
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout((1, 1), (1, 1))))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


# A-LDS bank-conflict swizzle as layout algebra: over the flat 16-byte-block index
# (row*8 + block_col) the manual XOR mask equals SwizzleType(3, 0, 4) (verified bit-for-bit).
_A_LDS_BLOCKS_PER_ROW = 8  # 128-byte fp4 row / 16-byte block


def _a_lds_swz_block_layout(rows):
    """Composed layout over 16-byte A-LDS blocks: (rows, 8) row-major, swizzled S<3,0,4>.

    crd2idx((row, block_col), <this>) == the manual `(byte_off ^ mask) // 16` block index.
    """
    return fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 0, 4)),
        fx.make_layout((rows, _A_LDS_BLOCKS_PER_ROW), (_A_LDS_BLOCKS_PER_ROW, 1)),
    )


def _a_lds_swz_block_idx(swz_layout, row, block_col):
    """Swizzled 16-byte-block index for (row, block_col) via layout algebra."""
    return fx.Int32(crd2idx([fx.Int64(row), fx.Int64(block_col)], swz_layout))


def _fabs_f32(x):
    return fmath.absf(x)


fabs_f32 = _fabs_f32


_FMT_MAX_FP4 = 6.0
_FMT_MAX_FP8_E4M3 = 448.0


def _e8m0_roundup(amax_f32, fmt_max):
    wi = (amax_f32 * fx.Float32(1.0 / fmt_max)).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    return (bexp < fx.Int32(254)).select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32, _FMT_MAX_FP4)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


def _e8m0_from_amax_fp8(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32, _FMT_MAX_FP8_E4M3)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


def _inline_e8m0(amax_u16_i32):
    f32 = ((fx.Int32(amax_u16_i32) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(fx.Float32)
    return _e8m0_roundup(f32, _FMT_MAX_FP4)


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = T.vec(2, T.i16)
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _umax_i32(a, b):
    is_gt = arith.cmpi(arith.CmpIPredicate.ugt, _raw(a), _raw(b))
    return fx.Int32(arith.select(is_gt, _raw(a), _raw(b)))


def _inline_dpp_quad_amax(a32):
    a32 = fx.Int32(_raw(a32))
    s1 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0xB1, 0xF, 0xF, True))
    a32 = _umax_i32(a32, s1)
    s2 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0x4E, 0xF, 0xF, True))
    return _umax_i32(a32, s2)


def k_half_for(k):
    return k // 2


def k_tiles_total_for(k, BK):
    return k // BK


def kunroll_for(k, BK):
    return k_tiles_total_for(k, BK) - kStages


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64


def num_n_blocks_for(n, BN):
    return n // BN


def kbs_c_n1_for(n):
    return n // 16 // 2


def kbs_per_expert_dw_for(n, k):
    return kbs_c_n1_for(n) * kbs_stride_n0_dw_for(k)


def bq_bytes_for(ne, n, k):
    return ne * n * k_half_for(k)


def bscale_bytes_for(ne, n, k):
    return ne * kbs_per_expert_dw_for(n, k) * 4


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4
