# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX11_WMMAType, MmaOpGFX1250_WMMAType
from ..._mlir.dialects.fly import AtomicOp, PointerType
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA3BufferAtomicType,
    CopyOpCDNA3BufferCopyLDSType,
    CopyOpCDNA3BufferCopyType,
    MmaOpCDNA3_MFMAType,
    TargetAddressSpace,
)
from ..._mlir.extras import types as T
from ..primitive import cosize, get_iter, get_layout, get_scalar, make_ptr, make_view
from ..typing import Int16, Int32, Int64, Tensor


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


def WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None):
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc

    # Arch-aware dispatch:
    #   * RDNA3 / RDNA3.5 (gfx1100..gfx1152) use the legacy v16-operand WMMA ABI.
    #   * RDNA4 (gfx1250)                    uses the new v8-operand ABI.
    from ...runtime.device import get_rocm_arch

    arch = (get_rocm_arch() or "").lower()
    if arch.startswith("gfx11"):
        return MmaOpGFX11_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)
    if arch.startswith("gfx12"):
        return MmaOpGFX1250_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)
    raise ValueError(
        f"WMMA is not available on target arch {arch!r}; " "supported: gfx11xx (RDNA3 / RDNA3.5) and gfx12xx (RDNA4). "
    )


def make_buffer_tensor(
    tensor: Tensor,
    max_size: bool = True,
    *,
    num_records_bytes=None,
) -> Tensor:
    """Wrap ``tensor`` in a buffer-resource view for hardware OOB-checked
    loads / stores.

    Args:
        tensor: Source tensor with a layout.
        max_size: If ``True`` (default) use ``0xFFFFFFFF`` as the descriptor
            size -- safe coarse OOB checking that survives any cache reuse.
        num_records_bytes: Explicit byte count for the descriptor.  Required
            when ``max_size=False``.  Should be computed from runtime tensor
            extents (e.g. ``M * N * elem_bytes`` where ``M``/``N`` are
            runtime ``fx.Int32``) so the cached kernel stays valid across
            shapes.

    Why ``num_records_bytes`` is required when ``max_size=False``:
        Auto-deriving from ``cosize(layout) * elem_bytes`` bakes the static
        layout's shape values into IR.  The JIT cache key does not include
        shape by default, so the same compiled artifact is reused for calls
        with different shapes -- and the first shape's ``num_records`` then
        silently truncates OOB reads / writes on the others.  This is the
        PR #551 fp8gemm regression mode; the guard makes the unsafe pattern
        impossible at compile time.
    """
    elem_ty = tensor.element_type

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    if num_records_bytes is not None:
        # Always materialise as Int64: the ROCDL ``make.buffer.rsrc`` op
        # requires an i64 num_records operand, and runtime expressions like
        # ``fx.Int32(M) * N * elem_bytes`` produce i32 which would otherwise
        # overflow for buffers > 2GB / fail verification.
        if isinstance(num_records_bytes, int):
            num_records_bytes = Int64(num_records_bytes)
        elif hasattr(num_records_bytes, "to") and not isinstance(num_records_bytes, Int64):
            num_records_bytes = num_records_bytes.to(Int64)
        elif not hasattr(num_records_bytes, "ir_value"):
            num_records_bytes = Int64(num_records_bytes)
    elif max_size:
        MAX_BUFFER_SIZE = 0xFFFFFFFF
        num_records_bytes = Int64(MAX_BUFFER_SIZE)
    else:
        raise ValueError(
            "make_buffer_tensor(..., max_size=False) without "
            "num_records_bytes auto-derives num_records from cosize(layout), "
            "baking the static shape into IR.  The JIT cache key does not "
            "include shape by default, so a kernel compiled for shape A is "
            "reused for shape B with A's num_records baked in -- silently "
            "truncating OOB reads / writes.  Pass num_records_bytes "
            "explicitly (computed from runtime tensor extents) or use "
            "max_size=True for the safe coarse OOB path."
        )

    from ..buffer_ops import _get_buffer_flags

    buf_ptr_ty = PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=ptr.alignment,
    )
    buf_ptr = make_ptr(
        buf_ptr_ty,
        [
            ptr,
            Int16(0).ir_value(),
            num_records_bytes.ir_value(),
            Int32(_get_buffer_flags()).ir_value(),
        ],
    )
    return make_view(buf_ptr, layout)
