# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, T

kStages = 2
kBS_stride_k0_dw = 64


_A_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _lds_swizzle_mask(row, row_bytes=128):
    """XOR16 swizzle for an FP4 LDS row of `row_bytes`; permutes its 16-byte columns."""
    assert row_bytes in (64, 128), f"unsupported FP4 LDS row width {row_bytes}"
    return (row & fx.Int32(2 * (row_bytes // 16) - 2)) << fx.Int32(3)


def lds_swizzle_mask_f8(row, row_bytes):
    """XOR16 swizzle for an FP8 LDS row whose width is 128 or 256 bytes."""
    return (row & (row_bytes // 16 - 1)) << 4


def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    """LDS dst view for buffer_load_lds DMA (AddressSpace.Shared = LDS enum 2, not addrspace 3)."""
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def global_typed_ptr(arg, elem_ty, align=4, *, byte_offset=None):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int64(arg))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def lds_typed_ptr(base_i32, elem_ty, align=4, *, byte_offset=None):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int32(base_i32))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(base_i32, elem_ir_ty, align=align, byte_offset=byte_off_i32)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_dma_atom_128():
    """BufferCopyLDS128b copy-atom (16B global->LDS DMA chunk)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def flat_buffer_view(arg, base_elems, elem_ty, *, align, elem_bytes, fold=True, num_records_bytes=None):
    """Flat buffer-tensor view over a RAW i64 addr; fold=True folds wave-uniform base to a VGPR voffset, fold=False keeps per-lane offset + num_records_bytes for OOB-zero."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if fold:
        base = fx.Uint32(fx.rocdl.readfirstlane(T.i32, base_elems))
        off_i64 = fx.Uint64(base)
        base_iter = fx.inttoptr(
            ptr_ty,
            fx.Uint64(arg) + off_i64 * fx.Uint64(elem_bytes),
        )
    else:
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg))
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout((1, 1), (1, 1))))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def _fabs_f32(x):
    return fmath.absf(x)
