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


def BufferCopy(bit_size, cache_modifier=0):
    """Create a CDNA3 buffer copy atom (cache_modifier: 0=cached, 2=nt).

    Current atom state:
    - `soffset` (`i32`), default zero
    """
    return CopyOpCDNA3BufferCopyType.get(bit_size, cache_modifier)


BufferCopy8b = lambda cache_modifier=0: CopyOpCDNA3BufferCopyType.get(8, cache_modifier)
BufferCopy16b = lambda cache_modifier=0: CopyOpCDNA3BufferCopyType.get(16, cache_modifier)
BufferCopy32b = lambda cache_modifier=0: CopyOpCDNA3BufferCopyType.get(32, cache_modifier)
BufferCopy64b = lambda cache_modifier=0: CopyOpCDNA3BufferCopyType.get(64, cache_modifier)
BufferCopy128b = lambda cache_modifier=0: CopyOpCDNA3BufferCopyType.get(128, cache_modifier)


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


def WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None, **kwargs):
    """Create an arch-appropriate WMMA atom.

    Supported kwargs (gfx11 integer paths only — iu8 / iu4):
        sign_a (bool, default False): treat A operand as signed.
        sign_b (bool, default False): treat B operand as signed.
        clamp  (bool, default False): saturate integer accumulator.
    These are forwarded verbatim to MmaOpGFX11_WMMAType.get(); the ROCDL
    intrinsic's verify() will reject them on fp16/bf16 paths.
    The gfx12 (RDNA4) path does not expose these knobs yet and will raise
    if any are passed as True.
    Future WMMA ops for new architectures should extend kwargs here rather
    than growing the positional signature.
    """
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
        return MmaOpGFX11_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc, **kwargs)
    if arch.startswith("gfx12"):
        if any(kwargs.get(k) for k in ("sign_a", "sign_b", "clamp")):
            raise ValueError("sign_a/sign_b/clamp are not supported on the gfx12 (RDNA4) WMMA path yet")
        return MmaOpGFX1250_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc)
    raise ValueError(
        f"WMMA is not available on target arch {arch!r}; supported: gfx11xx (RDNA3 / RDNA3.5) and gfx12xx (RDNA4). "
    )


def make_buffer_tensor(
    tensor: Tensor,
    max_size: bool = True,
    *,
    num_records_bytes=None,
) -> Tensor:
    """Wrap ``tensor`` in a buffer-resource view for hardware OOB-checked
    loads / stores.

    ``max_size=True`` (default) sets the descriptor to ``0xFFFFFFFF``.
    Pass ``num_records_bytes`` when the byte count is a compile-time
    constant (folds to a constant in IR).  Otherwise with ``max_size=False``
    it is derived at runtime from ``cosize(layout) * elem_bytes``.
    """
    elem_ty = tensor.element_type

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    if num_records_bytes is not None:
        # Coerce to i64: ROCDL make.buffer.rsrc requires an i64 num_records
        # operand.  Int64(...) handles Python int, other fx Integer types
        # (e.g. fx.Int32(M) * N), and raw ir.Value with i32/index/float types
        # -- emitting the appropriate extension / cast.  Idempotent when the
        # input is already Int64.
        if not isinstance(num_records_bytes, Int64):
            num_records_bytes = Int64(num_records_bytes)
    elif max_size:
        num_records_bytes = Int64(0xFFFFFFFF)
    else:
        elem_bits = elem_ty.width
        if elem_bits % 8 == 0:
            num_records_bytes = Int64(get_scalar(cosize(layout)) * (elem_bits // 8))
        else:
            num_records_bytes = Int64((get_scalar(cosize(layout)) * elem_bits + 7) // 8)

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
