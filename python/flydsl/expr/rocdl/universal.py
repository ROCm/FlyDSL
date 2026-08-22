# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import (
    MmaOpGFX11_WMMAType,
    MmaOpGFX120X_WMMAType,
    MmaOpGFX1250_WMMAType,
)
from ..._mlir.dialects import fly_rocdl
from ..._mlir.dialects import rocdl as mlir_rocdl
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
from ..meta import dsl_loc_tracing
from ..primitive import cosize, get_iter, get_layout, get_scalar, make_ptr, make_view
from ..typing import (
    AddressSpace,
    Int16,
    Int32,
    Int64,
    Pointer,
    Tensor,
    is_generic_address_space,
    is_target_address_space,
)
from . import cdna3, rdna3, rdna4
from .utils import normalize_s_waitcnt_field


@dsl_loc_tracing
def s_waitcnt(bitfield=None, *, vmcnt=None, lgkmcnt=None, expcnt=None):
    """Wait for named counters, or emit a legacy raw wait-counter bitfield."""
    if bitfield is not None:
        if vmcnt is not None or lgkmcnt is not None or expcnt is not None:
            raise TypeError("legacy raw s_waitcnt bitfield cannot be combined with non-None keyword arguments")
        return mlir_rocdl.s_waitcnt(normalize_s_waitcnt_field("bitfield", bitfield, 0xFFFF))

    arch = get_rocm_arch()
    if arch.startswith(("gfx942", "gfx950")):
        return cdna3.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt, expcnt=expcnt)
    if arch.startswith("gfx11"):
        return rdna3.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt, expcnt=expcnt)
    if arch.startswith("gfx120"):
        return rdna4.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt, expcnt=expcnt)
    raise ValueError(
        f"s_waitcnt is not supported on target arch {arch!r}; supported: gfx942 (CDNA3), gfx950 (CDNA4), gfx11xx (RDNA3 / RDNA3.5), and gfx120x (RDNA4). "
    )


@dsl_loc_tracing
def asyncmark():
    """Close the current group of async operations (``rocdl.asyncmark``).

    Async LDS DMA copies (the ``*LoadAsyncLDS`` atoms, gfx1250 TDM / async global loads) are not
    tracked by the compiler's automatic wait insertion. Group them with this and drain them with
    :func:`wait_asyncmark`.
    """
    return mlir_rocdl.asyncmark()


@dsl_loc_tracing
def wait_asyncmark(count=0):
    """Wait until at most ``count`` async groups remain outstanding.

    ``count`` must be a compile time value: a Python ``int`` or a static DSL integer. ``count=0``
    drains every outstanding group; ``None`` maps to the maximum, which waits for nothing.
    """
    n = normalize_s_waitcnt_field("count", count, 0xFFFF)
    return mlir_rocdl.wait_asyncmark(ir.IntegerAttr.get(ir.IntegerType.get_signless(16), n))


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

    ``bit_size`` must be 32 or 128. Only supports BufferDesc -> Shared address space direction.

    This atom is synchronous in the sense that the compiler inserts the ``vmcnt`` wait for you
    before the staged LDS data is read. If you want to insert ``vmcnt``, use the async counterparts
    instead: :func:`flydsl.expr.rocdl.cdna4.BufferLoadAsyncLDS` (same direction and state) or
    :func:`flydsl.expr.rocdl.cdna4.GlobalLoadAsyncLDS` (Global -> Shared). Those require explicit
    ``fx.rocdl.asyncmark()`` / ``fx.rocdl.wait_asyncmark(n)`` tracking.

    Current atom state:
    - `soffset` (`i32`), default zero
    - `imm_offset` (`i32`), default zero
    """
    return CopyOpCDNA3BufferCopyLDSType.get(bit_size)


BufferCopyLDS32b = lambda: CopyOpCDNA3BufferCopyLDSType.get(32)
BufferCopyLDS128b = lambda: CopyOpCDNA3BufferCopyLDSType.get(128)


def BufferCopyLDS64b():
    """Create a 64-bit CDNA3 buffer-to-LDS copy atom.

    .. deprecated::
        There is no 8-byte LDS DMA instruction on any AMD target: the transfer
        widths are 1/2/4 bytes, plus 12/16 bytes on gfx950. This entry point
        previously produced an atom that passed verification and then silently
        failed instruction selection, so the copy never happened. It now raises.

        Use :func:`BufferCopyLDS32b` or :func:`BufferCopyLDS128b` (gfx950)
        instead. Kept as a named export for one deprecation window; see
        ``docs/api_stability.md`` section 3.
    """
    raise ValueError(
        "BufferCopyLDS64b is deprecated and unsupported: there is no 8-byte LDS DMA "
        "instruction on any AMD target (widths are 1/2/4 bytes, plus 12/16 bytes on "
        "gfx950). It previously verified but silently failed instruction selection. "
        "Use BufferCopyLDS32b, or BufferCopyLDS128b on gfx950."
    )


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

    Supported kwargs:
        sign_a  (bool, default False): treat A operand as signed (iu8/iu4 only).
        sign_b  (bool, default False): treat B operand as signed (iu8/iu4 only).
        clamp   (bool, default False): saturate integer accumulator (iu8/iu4 only).
        mod_c   (int,  default 0):     I16 C-operand modifier (gfx1250 only).
        reuse_a (bool, default False): operand-reuse scheduler hint (gfx1250 only).
        reuse_b (bool, default False): operand-reuse scheduler hint (gfx1250 only).
    Forwarded to the arch-specific WMMA atom (MmaOpGFX11_WMMAType on gfx11,
    MmaOpGFX120X_WMMAType on gfx120x, MmaOpGFX1250_WMMAType on gfx1250); the
    atom's verify() rejects sign_a/sign_b/clamp on the float (fp16/bf16/fp8)
    paths, where the intrinsic has no such operands. Future WMMA ops for new
    architectures should extend kwargs here rather than growing the positional
    signature.
    """
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = ir.F32Type.get()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc

    # Arch-aware dispatch:
    #   * RDNA3 / RDNA3.5 (gfx1100..gfx1152) use the legacy v16-operand WMMA ABI.
    #   * RDNA4 (gfx1200 / gfx1201) and gfx1250 share the v8-operand ABI but not
    #     the instruction shapes: RDNA4 has the gfx11 16x16x16 forms, gfx1250 has
    #     16x16x32 (plus fp8 K=64/128) with mods/reuse operands. They therefore
    #     get separate atoms, matched on the disjoint gfx120x / gfx1250 prefixes
    #     rather than on a shared gfx12 one.

    arch = get_rocm_arch() or ""
    if arch.startswith("gfx11"):
        return MmaOpGFX11_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc, **kwargs)
    if arch.startswith("gfx1250"):
        return MmaOpGFX1250_WMMAType.get(
            m,
            n,
            k,
            ty_ab,
            ty_ab,
            ty_acc,
            sign_a=bool(kwargs.get("sign_a", False)),
            sign_b=bool(kwargs.get("sign_b", False)),
            clamp=bool(kwargs.get("clamp", False)),
            mod_c=int(kwargs.get("mod_c", 0)),
            reuse_a=bool(kwargs.get("reuse_a", False)),
            reuse_b=bool(kwargs.get("reuse_b", False)),
        )
    if arch.startswith("gfx120"):
        return MmaOpGFX120X_WMMAType.get(
            m,
            n,
            k,
            ty_ab,
            ty_ab,
            ty_acc,
            sign_a=bool(kwargs.get("sign_a", False)),
            sign_b=bool(kwargs.get("sign_b", False)),
            clamp=bool(kwargs.get("clamp", False)),
        )
    raise ValueError(
        f"WMMA is not available on target arch {arch!r}; supported: gfx11xx (RDNA3 / RDNA3.5), gfx120x (RDNA4), and gfx1250. "
    )


def make_buffer_ptr(ptr: Pointer, num_records_bytes=None):
    """Construct a new buffer-resource (``BufferDesc``) pointer from a global
    pointer, for hardware OOB-checked loads / stores.

    ``num_records_bytes`` is the descriptor byte count.  When ``None``
    (default) it falls back to the max size ``0xFFFFFFFF``.
    """
    if not is_generic_address_space(ptr.address_space, AddressSpace.Global):
        raise ValueError(f"make_buffer_ptr requires a global-address-space pointer, got {ptr.address_space}")

    elem_ty = ptr.element_type

    if num_records_bytes is None:
        num_records_bytes = Int64(0xFFFFFFFF)
    elif not isinstance(num_records_bytes, Int64):
        # Coerce to i64: ROCDL make.buffer.rsrc requires an i64 num_records operand.
        num_records_bytes = Int64(num_records_bytes)

    from ...runtime.device import is_rdna_arch

    arch = get_rocm_arch()
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)

    buf_ptr_ty = PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=ptr.alignment,
    )
    return make_ptr(
        buf_ptr_ty,
        [
            ptr,
            Int16(0).ir_value(),
            num_records_bytes.ir_value(),
            Int32(flags).ir_value(),
        ],
    )


def make_buffer_tensor(
    tensor: Tensor,
    max_size: bool = True,
    *,
    num_records_bytes=None,
) -> Tensor:
    """Construct a new buffer-resource-backed tensor from a global-pointer
    tensor, for hardware OOB-checked loads / stores and buffer_copy atoms
    (CDNA buffer copy); layout is unchanged. For the gfx1250 TDM DMA use
    :func:`make_tdm_atom` instead — TDM needs a raw VA, not a buffer resource.

    ``max_size=True`` (default) sets the descriptor to ``0xFFFFFFFF``.
    Pass ``num_records_bytes`` when the byte count is a compile-time
    constant (folds to a constant in IR).  Otherwise with ``max_size=False``
    it is derived at runtime from ``cosize(layout) * elem_bytes``.
    """
    elem_ty = tensor.element_type

    ptr = get_iter(tensor)
    layout = get_layout(tensor)

    if num_records_bytes is None and not max_size:
        # Derive the byte count from the layout footprint.
        elem_bits = elem_ty.width
        if elem_bits % 8 == 0:
            num_records_bytes = Int64(get_scalar(cosize(layout)) * (elem_bits // 8))
        else:
            num_records_bytes = Int64((get_scalar(cosize(layout)) * elem_bits + 7) // 8)

    buf_ptr = make_buffer_ptr(ptr, num_records_bytes=num_records_bytes)
    return make_view(buf_ptr, layout)


@dsl_loc_tracing
def get_buffer_rsrc(ptr: Pointer):
    """Extract the raw ROCDL buffer resource (``!llvm.ptr<8>``) from a
    buffer-descriptor pointer.

    ``ptr`` must be a buffer-descriptor pointer, e.g. the value produced by
    :func:`make_buffer_ptr` or the iterator of a :func:`make_buffer_tensor`
    result.
    """
    if not is_target_address_space(ptr.address_space, TargetAddressSpace.BufferDesc):
        raise ValueError(f"get_buffer_rsrc requires a buffer-descriptor pointer, got {ptr.address_space}")

    return fly_rocdl.get_buffer_rsrc(ptr)
