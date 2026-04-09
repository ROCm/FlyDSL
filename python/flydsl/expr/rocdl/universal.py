# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from ..._mlir.dialects import arith, fly
from ..._mlir.dialects._fly_enum_gen import AddressSpace
from ..._mlir.dialects.fly import PointerType
from ..._mlir.dialects.fly_rocdl import CopyOpCDNA3BufferCopyType, MmaOpCDNA3_MFMAType
from ..._mlir.extras import types as T
from ..primitive import (
    get_iter,
    get_layout,
    make_ptr,
    make_view,
)
from ..typing import Tensor


def BufferCopy(bit_size):
    """Create a CDNA3 buffer copy atom.

    Current atom state:
    - `soffset` (`i32`)

    Update the state with `copy_atom.set_value("soffset", value)`.
    """
    return CopyOpCDNA3BufferCopyType.get(bit_size)


# BufferCopy aliases for convenience
BufferCopy8b = lambda: CopyOpCDNA3BufferCopyType.get(8)
BufferCopy16b = lambda: CopyOpCDNA3BufferCopyType.get(16)
BufferCopy32b = lambda: CopyOpCDNA3BufferCopyType.get(32)
BufferCopy64b = lambda: CopyOpCDNA3BufferCopyType.get(64)
BufferCopy128b = lambda: CopyOpCDNA3BufferCopyType.get(128)


def MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        # default to f32
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpCDNA3_MFMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpGFX1250_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def make_buffer_tensor(tensor: Tensor, max_size: bool = True) -> Tensor:
    def _elem_bit_width(elem_ty):
        if hasattr(elem_ty, "width"):
            return int(elem_ty.width)
        return 0

    MAX_BUFFER_SIZE = 0xFFFFFFFF

    elem_ty = tensor.element_type

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    elem_bits = _elem_bit_width(elem_ty)
    elem_bytes = elem_bits // 8 if elem_bits > 0 else 1

    if max_size:
        # Always use max buffer size to handle dynamic tensor shapes safely.
        # The JIT cache may reuse a compiled kernel for tensors of different
        # leading dimensions; a static num_records from the traced shape would
        # silently truncate larger tensors.  This matches the behavior of
        # buffer_ops.create_buffer_resource(max_size=True).
        num_records_bytes = MAX_BUFFER_SIZE
    elif layout.is_static:
        cosize = fly.cosize(layout)
        num_records_bytes = cosize.get_static_leaf_int * elem_bytes
        if num_records_bytes > MAX_BUFFER_SIZE:
            num_records_bytes = MAX_BUFFER_SIZE
    else:
        num_records_bytes = MAX_BUFFER_SIZE

    stride_val = arith.ConstantOp(T.i16(), ir.IntegerAttr.get(T.i16(), 0)).result
    num_records_val = arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), num_records_bytes)).result
    from ..buffer_ops import _get_buffer_flags

    flags_val_int = _get_buffer_flags()
    flags_val = arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), flags_val_int)).result

    src_ptr_ty = PointerType(ptr.type)
    buf_ptr_ty = PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=int(AddressSpace.BufferDesc),
        alignment=src_ptr_ty.alignment,
    )
    buf_ptr = make_ptr(buf_ptr_ty, [ptr, stride_val, num_records_val, flags_val])

    return make_view(buf_ptr, layout)
