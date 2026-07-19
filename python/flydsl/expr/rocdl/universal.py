# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import (
    MmaOpGFX11_WMMAType,
    MmaOpGFX1250_WMMAScaleType,
    MmaOpGFX1250_WMMAType,
)
from ..._mlir.dialects import fly_rocdl
from ..._mlir.dialects.fly import AtomicOp, PointerType
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA3BufferAtomicType,
    CopyOpCDNA3BufferCopyLDSType,
    CopyOpCDNA3BufferCopyType,
    CopyOpGFX1250TDMType,
    MmaOpCDNA3_MFMAType,
    TargetAddressSpace,
)
from ..._mlir.extras import types as T
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

    Supported kwargs (integer paths only — iu8 / iu4):
        sign_a (bool, default False): treat A operand as signed.
        sign_b (bool, default False): treat B operand as signed.
        clamp  (bool, default False): saturate integer accumulator.
    Forwarded to the arch-specific WMMA atom (MmaOpGFX11_WMMAType on gfx11,
    MmaOpGFX1250_WMMAType on gfx12 / gfx1250); the atom's verify() rejects them
    on the float (fp16/bf16/fp8) paths, where the intrinsic has no such operands.
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
    #   * RDNA4 (gfx12xx, e.g. gfx1201) and gfx1250 use the new v8-operand ABI;
    #     both route through MmaOpGFX1250_WMMAType via the gfx12 prefix below.
    #     (gfx1250 is its own arch, not RDNA4, but shares this WMMA atom.)
    from ...runtime.device import get_rocm_arch

    arch = (get_rocm_arch() or "").lower()
    if arch.startswith("gfx11"):
        return MmaOpGFX11_WMMAType.get(m, n, k, ty_ab, ty_ab, ty_acc, **kwargs)
    if arch.startswith("gfx12"):
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
        )
    raise ValueError(
        f"WMMA is not available on target arch {arch!r}; supported: gfx11xx (RDNA3 / RDNA3.5), gfx12xx (RDNA4), and gfx1250. "
    )


def WMMAScale(
    m,
    n,
    k,
    elem_ty_a,
    elem_ty_b=None,
    elem_ty_acc=None,
    *,
    opsel_a=0,
    opsel_b=0,
    mod_c=0,
    reuse_a=False,
    reuse_b=False,
    block_size=32,
):
    """Create a gfx1250 MX-scaled WMMA atom (E8M0 block scale) for the unified
    f8/f6/f4 operand format. Per-operand scales are atom state (``scale_a`` /
    ``scale_b``); ``opsel_a`` / ``opsel_b`` are forwarded as the intrinsic's
    ``scaleAType`` / ``scaleBType`` operands (the scale-format / lane selector,
    not an output opsel). ``mod_c`` (i16 C-operand modifier) and ``reuse_a`` /
    ``reuse_b`` (operand-reuse scheduler hints) are forwarded to V_WMMA_SCALE.

    ``block_size`` selects the MX block size (elements per shared E8M0 scale):
    ``32`` (default) uses V_WMMA_SCALE with i32 scale state; ``16`` uses
    V_WMMA_SCALE16 with i64 scale state.
    """
    ty_a = elem_ty_a.ir_type if hasattr(elem_ty_a, "ir_type") else elem_ty_a
    if elem_ty_b is None:
        ty_b = ty_a
    else:
        ty_b = elem_ty_b.ir_type if hasattr(elem_ty_b, "ir_type") else elem_ty_b
    ty_acc = (
        ir.F32Type.get()
        if elem_ty_acc is None
        else (elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc)
    )
    return MmaOpGFX1250_WMMAScaleType.get(
        m,
        n,
        k,
        ty_a,
        ty_b,
        ty_acc,
        opsel_a=opsel_a,
        opsel_b=opsel_b,
        mod_c=mod_c,
        reuse_a=reuse_a,
        reuse_b=reuse_b,
        block_size=block_size,
    )


def TDM(
    rank,
    num_warps,
    pad_interval=0,
    pad_amount=0,
    cache_modifier=0,
    atomic_barrier=False,
    early_timeout=False,
):
    """Create a gfx1250 N-D TDM (Tensor Data Mover) Global<->LDS copy atom *type*.

    ``rank`` is the tensor/tile rank (1-5). Direction is inferred at lowering from
    which side is Global vs Shared; the tile shape is compile-time on the operand
    layout. ``pad_interval`` / ``pad_amount`` (elements) add LDS row padding on the
    load path.

    ``atomic_barrier`` (descriptor bit 18, HW auto-barrier) and ``early_timeout``
    (bit 21, multicast-load GL1 knob) set compile-time descriptor config bits.

    The tile descriptor (global base pointer, per-dim extent for out-of-bounds
    handling, per-dim stride) plus the MCAST ``workgroup_mask`` are runtime atom
    state set via ``fx.atom.set_value``. :func:`make_tdm_atom` builds the atom and
    populates that descriptor from a tensor in one call.
    """
    return CopyOpGFX1250TDMType.get(
        rank,
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
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

    from ...runtime.device import get_rocm_arch, is_rdna_arch

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


def make_tdm_atom(
    tensor: Tensor,
    tensor_extents,
    strides=None,
    *,
    num_warps,
    pad_interval=0,
    pad_amount=0,
    cache_modifier=0,
    atomic_barrier=False,
    early_timeout=False,
) -> object:
    """Build a gfx1250 N-D TDM copy atom carrying ``tensor``'s tile descriptor.

    The atom holds the global tile descriptor as runtime state: base pointer, the
    tensor's per-dim extent (for hardware out-of-bounds handling: load zero-fill,
    store drop), and per-dim strides. Reuse the atom across a tile loop; to move to
    the next tile re-set ``base`` (or bump ``imm_offset`` via
    :func:`advance_tdm_atom`).

    ``tensor_extents`` is a list of the tensor's per-dim extent in tensor dim order
    ``[dim0(outermost) .. dim_{rank-1}(innermost)]`` (rank = ``len(tensor_extents)``,
    1-5); each entry is a Python ``int`` or an ``i32`` / ``index`` runtime value (or
    any ``fx`` integer), and ``None`` means no clamp on that axis (INT32_MAX).
    ``strides`` is an optional list of per-dim strides in elements (same order);
    the innermost stride is assumed 1 and ignored, so entries for dims 0..rank-2
    are used. ``None`` (or a ``None`` entry) falls back to the tile memref's static
    layout stride; pass it explicitly for a tile with a dynamic outer stride (else
    the descriptor stride faults rather than silently reading stride 0).

    Issue the copy with ``fx.copy_atom_call(atom, global_tile, lds)``. NOTE: the
    global operand's *pointer is unused* — only its layout (tile shape) and address
    space (copy direction) are read; the base comes from the ``base`` state.
    """
    from ..primitive import atom_set_value, make_copy_atom

    NO_CLAMP = 0x7FFFFFFF
    STRIDE_UNSET = -0x80000000  # matches kOuterStrideUnset in CopyAtom.cpp

    extents = list(tensor_extents)
    rank = len(extents)
    if not 1 <= rank <= 5:
        raise ValueError(f"make_tdm_atom: rank must be in [1, 5], got {rank}")
    strides = list(strides) if strides is not None else [None] * rank
    if len(strides) != rank:
        raise ValueError(f"make_tdm_atom: expected {rank} strides, got {len(strides)}")

    copy_op = CopyOpGFX1250TDMType.get(
        rank,
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
    )
    atom = make_copy_atom(copy_op, tensor.element_type)
    atom = atom_set_value(atom, "base", get_iter(tensor))
    for i in range(rank):
        ext = (
            Int32(NO_CLAMP)
            if extents[i] is None
            else (extents[i] if isinstance(extents[i], Int32) else Int32(extents[i]))
        )
        atom = atom_set_value(atom, f"extent_{i}", ext)
    for i in range(rank - 1):  # innermost stride assumed 1, not stored
        st = (
            Int64(STRIDE_UNSET)
            if strides[i] is None
            else (strides[i] if isinstance(strides[i], Int64) else Int64(strides[i]))
        )
        atom = atom_set_value(atom, f"stride_{i}", st)
    return atom


def advance_tdm_atom(atom, byte_offset) -> object:
    """Return a TDM atom with its global byte offset (``imm_offset``) set.

    The offset is added to the ``base`` pointer in i64 at lowering (carry-safe),
    so a K-reduction loop can advance the tile by bumping this single scalar
    instead of re-deriving and re-setting the ``base`` pointer each iteration.
    ``byte_offset`` is the cumulative byte delta from ``base`` (typically
    ``k_tile * k_stride_bytes``); it replaces (does not accumulate onto) any
    previously-set ``imm_offset``. Accepts a Python ``int`` or an ``i64`` / any
    ``fx`` integer value.
    """
    from ..primitive import atom_set_value

    off = byte_offset if isinstance(byte_offset, Int64) else Int64(byte_offset)
    return atom_set_value(atom, "imm_offset", off)


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
