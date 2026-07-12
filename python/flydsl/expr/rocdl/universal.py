# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import (
    MmaOpGFX11_WMMAType,
    MmaOpGFX1250_WMMAScaleType,
    MmaOpGFX1250_WMMAType,
)
from ..._mlir.dialects.fly import AtomicOp, PointerType
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA3BufferAtomicType,
    CopyOpCDNA3BufferCopyLDSType,
    CopyOpCDNA3BufferCopyType,
    CopyOpGFX1250TDM2DType,
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
):
    """Create a gfx1250 MX-scaled WMMA atom (E8M0 block scale) for the unified
    f8/f6/f4 operand format. Per-operand scales are atom state (``scale_a`` /
    ``scale_b``, i32); ``opsel_a`` / ``opsel_b`` are forwarded as the intrinsic's
    ``scaleAType`` / ``scaleBType`` operands (the scale-format / lane selector,
    not an output opsel). ``mod_c`` (i16 C-operand modifier) and ``reuse_a`` /
    ``reuse_b`` (operand-reuse scheduler hints) are forwarded to V_WMMA_SCALE.
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
    )


def TDM2D(
    num_warps,
    pad_interval=0,
    pad_amount=0,
    cache_modifier=0,
    atomic_barrier=False,
    early_timeout=False,
):
    """Create a gfx1250 2D TDM (Tensor Data Mover) Global<->LDS copy atom *type*.

    Direction is inferred at lowering from which side is Global vs Shared; the tile
    shape is compile-time on the operand layout. ``pad_interval`` / ``pad_amount``
    (elements) add LDS row padding on the load path.

    ``atomic_barrier`` (descriptor bit 18, HW auto-barrier) and ``early_timeout``
    (bit 21, multicast-load GL1 knob) set compile-time descriptor config bits.

    The tile descriptor (global base pointer, runtime 2D extent for out-of-bounds
    handling, and outer stride) plus the MCAST ``workgroup_mask`` are runtime atom
    state set via ``fx.atom.set_value``. :func:`make_tdm_atom` builds the atom and
    populates that descriptor from a tensor in one call.
    """
    return CopyOpGFX1250TDM2DType.get(
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
    )


def make_buffer_tensor(
    tensor: Tensor,
    max_size: bool = True,
    *,
    num_records_bytes=None,
) -> Tensor:
    """Wrap ``tensor`` in a buffer-resource view for hardware OOB-checked
    loads / stores (CDNA buffer copy). For the gfx1250 TDM DMA use
    :func:`make_tdm_atom` instead — TDM needs a raw VA, not a buffer resource.

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


def make_tdm_atom(
    tensor: Tensor,
    tensor_extents=None,
    outer_stride=None,
    *,
    num_warps,
    pad_interval=0,
    pad_amount=0,
    cache_modifier=0,
    atomic_barrier=False,
    early_timeout=False,
) -> object:
    """Build a gfx1250 2D TDM copy atom carrying ``tensor``'s tile descriptor.

    The atom holds the global tile descriptor as runtime state: base pointer, the
    tensor's 2D extent (for hardware out-of-bounds handling: load zero-fill, store
    drop), and the outer stride in elements. Reuse the atom across a tile loop; to
    move to the next tile re-set ``base`` via ``fx.atom.set_value(atom, "base",
    ptr)``.

    ``tensor_extents`` is an optional list in tensor dim order ``[outer, inner]``;
    each entry is a Python ``int`` or an ``i32`` / ``index`` runtime value (or any
    ``fx`` integer), and ``None`` (or an omitted trailing dim) means no clamp on
    that axis (INT32_MAX). ``outer_stride`` is the runtime outer stride in
    elements; ``None`` falls back to the tile memref's static layout stride.

    Issue the copy with ``fx.copy_atom_call(atom, global_tile, lds)``: the global
    operand supplies only the tile shape and the copy direction.
    """
    from ..primitive import atom_set_value, make_copy_atom

    NO_CLAMP = 0x7FFFFFFF

    extents = list(tensor_extents) if tensor_extents is not None else []
    if len(extents) > 2:
        raise ValueError(f"make_tdm_atom: at most 2 extents (TDM is 2D), got {len(extents)}")
    extents += [None] * (2 - len(extents))  # pad to [outer, inner]

    def _i32(v):
        return v if isinstance(v, Int32) else Int32(v)

    outer = Int32(NO_CLAMP) if extents[0] is None else _i32(extents[0])
    inner = Int32(NO_CLAMP) if extents[1] is None else _i32(extents[1])
    stride = Int32(0) if outer_stride is None else _i32(outer_stride)

    copy_op = CopyOpGFX1250TDM2DType.get(
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
    )
    atom = make_copy_atom(copy_op, tensor.element_type)
    atom = atom_set_value(atom, "base", get_iter(tensor))
    atom = atom_set_value(atom, "outer_extent", outer)
    atom = atom_set_value(atom, "inner_extent", inner)
    atom = atom_set_value(atom, "outer_stride", stride)
    return atom
