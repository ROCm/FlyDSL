# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir.dialects import arith, fly
from ..._mlir.dialects._fly_enum_gen import AddressSpace
from ..._mlir.dialects.fly import LayoutType, PointerType
from ..._mlir.dialects.fly import MemRefType as FlyMemRefType
from ..._mlir.dialects.fly_rocdl import CopyOpCDNA3BufferLDSTType, MmaAtomCDNA3_MFMAType
from ..primitive import (
    get_iter,
    get_layout,
    make_ptr,
    make_view,
)
from ..typing import Tensor

BufferCopy = lambda bit_size: CopyOpCDNA3BufferLDSTType.get(bit_size)  # noqa: E731
BufferCopy32b = lambda: CopyOpCDNA3BufferLDSTType.get(32)  # noqa: E731
BufferCopy64b = lambda: CopyOpCDNA3BufferLDSTType.get(64)  # noqa: E731
BufferCopy128b = lambda: CopyOpCDNA3BufferLDSTType.get(128)  # noqa: E731


def MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        # default to f32
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaAtomCDNA3_MFMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def make_buffer_tensor(tensor: Tensor) -> Tensor:
    def _elem_bit_width(elem_ty):
        if hasattr(elem_ty, "width"):
            return int(elem_ty.width)
        return 0

    MAX_BUFFER_SIZE = 0xFFFFFFFF

    memref_val = tensor.value
    memref_ty = FlyMemRefType(memref_val.type)
    elem_ty = memref_ty.element_type
    layout_ty = LayoutType(memref_ty.layout)

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    elem_bits = _elem_bit_width(elem_ty)
    elem_bytes = elem_bits // 8 if elem_bits > 0 else 1

    if layout_ty.is_static:
        cosize_val = fly.cosize(layout)
        cosize_ty = fly.IntTupleType(cosize_val.type)
        num_records_bytes = cosize_ty.static_value * elem_bytes
        if num_records_bytes > MAX_BUFFER_SIZE:
            num_records_bytes = MAX_BUFFER_SIZE
    else:
        num_records_bytes = MAX_BUFFER_SIZE

    i16_type = ir.IntegerType.get_signless(16)
    i32_type = ir.IntegerType.get_signless(32)
    i64_type = ir.IntegerType.get_signless(64)

    stride_val = arith.ConstantOp(i16_type, ir.IntegerAttr.get(i16_type, 0)).result
    num_records_val = arith.ConstantOp(i64_type, ir.IntegerAttr.get(i64_type, num_records_bytes)).result
    flags_val_int = (7 << 12) | (4 << 15)
    flags_val = arith.ConstantOp(i32_type, ir.IntegerAttr.get(i32_type, flags_val_int)).result

    src_ptr_ty = PointerType(ptr.type)
    buf_ptr_ty = PointerType.get(
        elem_ty=elem_ty,
        address_space=int(AddressSpace.BufferDesc),
        alignment=src_ptr_ty.alignment,
    )
    buf_ptr = make_ptr(buf_ptr_ty, [ptr, stride_val, num_records_val, flags_val])

    return make_view(buf_ptr, layout)
