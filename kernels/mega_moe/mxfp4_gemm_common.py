# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Shared LDS / buffer helpers for the ported aiter mxmoe gemm2 (mxmoe_gemm_v2).

Faithful copy of aiter/ops/flydsl/kernels/mxfp4_gemm_common.py, trimmed to the subset the gemm2
down-proj path needs (the fp8-route-quant DPP helpers are dropped: not used by gemm2/epilog)."""

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

kStages = 2
kBS_stride_k0_dw = 64


def _lds_swizzle_mask(row):
    return (row & fx.Int32(14)) << fx.Int32(3)


def lds_swizzle_mask_f8(row):
    """lds_swizzle_mask<ROW_BYTES=256>(row) = (row & 15) << 4 (fp8 A tile)."""
    return (row & 15) << 4


def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    """LDS dst view for buffer_load_lds DMA (AddressSpace.Shared = LDS enum 2, not addrspace 3)."""
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_dma_atom_128():
    """BufferCopyLDS128b copy-atom (16B global->LDS DMA chunk)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def flat_buffer_view(arg, base_elems, elem_ty, *, align, elem_bytes, fold=True, num_records_bytes=None):
    """Flat buffer view over an i64 address.

    ``fold=True`` folds a wave-uniform base into a VGPR offset; otherwise the
    per-lane offset and byte bound provide OOB-zero behavior.
    """
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


def _fabs_f32(x):
    return fmath.absf(x)
