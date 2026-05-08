# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import (
    MmaOpGFX1250_WMMAType,
    MmaOpRDNA3_WMMAType,
)
from ..._mlir.dialects import arith, fly
from ..._mlir.dialects._fly_enum_gen import AddressSpace
from ..._mlir.dialects.fly import AtomicOp, PointerType
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA3BufferAtomicType,
    CopyOpCDNA3BufferCopyLDSType,
    CopyOpCDNA3BufferCopyType,
    MmaOpCDNA3_MFMAType,
    TargetAddressSpace,
)
from ..._mlir.extras import types as T
from ...runtime.device import get_rocm_arch
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
    - `soffset` (`i32`), default zero
    """
    return CopyOpCDNA3BufferCopyType.get(bit_size)


# BufferCopy aliases for convenience
BufferCopy8b = lambda: CopyOpCDNA3BufferCopyType.get(8)
BufferCopy16b = lambda: CopyOpCDNA3BufferCopyType.get(16)
BufferCopy32b = lambda: CopyOpCDNA3BufferCopyType.get(32)
BufferCopy64b = lambda: CopyOpCDNA3BufferCopyType.get(64)
BufferCopy128b = lambda: CopyOpCDNA3BufferCopyType.get(128)


def BufferCopyLDS(bit_size):
    """Create a CDNA3 buffer-to-LDS copy atom.

    Only supports BufferDesc -> Shared address space direction.

    Current atom state:
    - `soffset` (`i32`), default zero
    - `imm_offset` (`i32`), default zero
    """
    return CopyOpCDNA3BufferCopyLDSType.get(bit_size)


BufferCopyLDS32b = lambda: CopyOpCDNA3BufferCopyLDSType.get(32)
BufferCopyLDS64b = lambda: CopyOpCDNA3BufferCopyLDSType.get(64)
BufferCopyLDS128b = lambda: CopyOpCDNA3BufferCopyLDSType.get(128)


def BufferAtomic(atomic_op, val_type):
    """Create a CDNA3 buffer atomic copy atom.

    Current atom state:
    - `soffset` (`i32`), default zero
    """
    ty = val_type.ir_type if hasattr(val_type, "ir_type") else val_type
    return CopyOpCDNA3BufferAtomicType.get(int(atomic_op), ty)


BufferAtomicAdd = lambda val_type: BufferAtomic(AtomicOp.Add, val_type)
BufferAtomicMax = lambda val_type: BufferAtomic(AtomicOp.Max, val_type)
BufferAtomicMin = lambda val_type: BufferAtomic(AtomicOp.Min, val_type)
BufferAtomicPkAdd = lambda val_type: BufferAtomic(AtomicOp.Add, T.vector(2, val_type.ir_type))


def MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        # default to f32
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpCDNA3_MFMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def WMMA_GFX1250(m, n, k, elem_ty_ab, elem_ty_acc=None):
    """Create a GFX1250-class WMMA atom (MI450, K=4/32/64/128, FP8/FP4/scaled OK)."""
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpGFX1250_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def WMMA_RDNA3(m, n, k, elem_ty_ab, elem_ty_acc=None):
    """Create an RDNA 3 / 3.5 WMMA atom (gfx1100/gfx1150).

    Only M=N=K=16 with F16/BF16/IU8/IU4 input dtypes is supported. FP8/FP4,
    scaled WMMA, sparse SWMMAC, and K != 16 require gfx12+ and are rejected
    by the C++ ``verify`` of ``MmaOpRDNA3_WMMAType``.
    """
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpRDNA3_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def _is_rdna3_arch(arch: str) -> bool:
    """Return True for arch strings that match RDNA 3 / RDNA 3.5 (gfx110x / gfx115x)."""
    return arch.startswith("gfx110") or arch.startswith("gfx115")


def WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    """Create a wave32 WMMA atom appropriate for the current GPU arch.

    Dispatches to :func:`WMMA_RDNA3` for gfx110x / gfx115x (RDNA 3 / 3.5,
    K=16-only intrinsics) and to :func:`WMMA_GFX1250` for gfx12+ MI450-class
    chips (K=4/32/64/128, FP8/FP4, scaled). Use the explicit ``WMMA_RDNA3`` /
    ``WMMA_GFX1250`` constructors when a kernel must pin a specific atom
    regardless of the runtime arch (e.g. tests).
    """
    try:
        arch = str(get_rocm_arch())
    except Exception:
        arch = ""
    if _is_rdna3_arch(arch):
        return WMMA_RDNA3(m, n, k, elem_ty_ab, elem_ty_acc)
    return WMMA_GFX1250(m, n, k, elem_ty_ab, elem_ty_acc)


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
        address_space=TargetAddressSpace.BufferDesc,
        alignment=src_ptr_ty.alignment,
    )
    buf_ptr = make_ptr(buf_ptr_ty, [ptr, stride_val, num_records_val, flags_val])

    return make_view(buf_ptr, layout)
