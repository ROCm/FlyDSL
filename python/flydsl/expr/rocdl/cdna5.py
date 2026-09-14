# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CDNA5 / gfx1250 ROCDL atom builders."""

from ..._mlir import ir
from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAScaleType
from ..._mlir.dialects import fly_rocdl
from ..typing import Int32, Int64, Tensor

__all__ = [
    "WMMAScale",
    "TensorLoad",
    "TensorStore",
    "TDM",
    "make_tdm_atom",
    "make_tiled_tdm_atom",
    "tdm_partition",
]


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


class TensorLoad:
    """CDNA5 TDM Global -> LDS DMA (``TENSOR_LOAD_TO_LDS``).

    Current atom state:
    - `workgroup_mask` (i32): the workgroup mask.
    - `early_timeout` (i32): the early timeout mask.
    - `atomic_barrier_addr` (shared ptr): *which* LDS barrier this copy arrives on.
      *Whether* it arrives on one is the atom's type, so only ``atomic_barrier=True``
      has the field.
    - `boundary_check` (int_tuple): per mode boundary check, congruent with the global tensor.
    """

    def __init__(self, cache_modifier=0):
        self.cache_modifier = cache_modifier


class TensorStore:
    """CDNA5 TDM LDS -> Global DMA (``TENSOR_STORE_FROM_LDS``).

    Current atom state:
    - `atomic_barrier_addr` (shared ptr): *which* LDS barrier this copy arrives on.
      *Whether* it arrives on one is the atom's type, so only ``atomic_barrier=True``
      has the field.
    - `boundary_check` (int_tuple): per mode boundary check, congruent with the global tensor.
    """

    def __init__(self, cache_modifier=0):
        self.cache_modifier = cache_modifier


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

    The global base pointer comes from the ``copy_atom_call`` global operand; the
    per-dim extent (OOB), per-dim stride, ``imm_offset`` (K-loop tile bump), and the
    MCAST ``workgroup_mask`` are runtime atom state set via ``fx.atom.set_value``.
    :func:`make_tdm_atom` builds the atom and populates the descriptor from a tensor.
    """
    return fly_rocdl.CopyOpGFX1250TDMType.get(
        rank,
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
    )


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

    The global base pointer comes from the ``copy_atom_call`` global operand (not
    atom state); the atom carries the tensor's per-dim extent (for hardware
    out-of-bounds handling: load zero-fill, store drop) and per-dim strides. Reuse
    the atom across a tile loop; advance the tile via the ``imm_offset`` state
    (``fx.copy(atom, gt, dst, imm_offset=...)``) or by advancing the global operand.

    ``tensor_extents`` is a list of the tensor's per-dim extent in tensor dim order
    ``[dim0(outermost) .. dim_{rank-1}(innermost)]`` (rank = ``len(tensor_extents)``,
    1-5); each entry is a Python ``int`` or an ``i32`` / ``index`` runtime value (or
    any ``fx`` integer), and ``None`` means no clamp on that axis (INT32_MAX).
    ``strides`` is an optional list of per-dim strides in elements (same order);
    the innermost stride is assumed 1 and ignored, so entries for dims 0..rank-2
    are used. ``None`` (or a ``None`` entry) falls back to the tile memref's static
    layout stride; pass it explicitly for a tile whose true (or dynamic) outer
    stride differs from the packed tile-internal stride.

    Issue the copy with ``fx.copy_atom_call(atom, global_tile, lds)``: the global
    operand supplies both the copy direction (address space) and the base pointer.
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

    copy_op = fly_rocdl.CopyOpGFX1250TDMType.get(
        rank,
        num_warps,
        pad_interval,
        pad_amount,
        cache_modifier,
        atomic_barrier=atomic_barrier,
        early_timeout=early_timeout,
    )
    atom = make_copy_atom(copy_op, tensor.element_type)
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


def make_tiled_tdm_atom(
    op,
    tensor: Tensor,
    smem_layout,
    tdm_tile,
    num_warps=1,
    *,
    init_boundary_check=True,
    atomic_barrier=False,
    internal_type=None,
):
    """Build a wave-scoped CDNA5 TDM copy atom and its coordinate tensor.

    * ``op`` — a ``TensorLoad(...)`` or ``TensorStore(...)`` instance.
    * ``tensor`` — the global tensor.
    * ``smem_layout`` — the LDS tile layout.
    * ``tdm_tile`` — the tiler: how many elements to take from each global mode.
    * ``num_warps`` — how many warps of the workgroup split this tile. The same
      number must be handed to :func:`tdm_partition` as the size of its warp
      layout.
    * ``init_boundary_check`` — The *initial* ``boundary_check`` state.
    * ``atomic_barrier`` — whether this atom arrives on the atomic barrier when finished.
    * ``internal_type`` — the unit the *descriptor* counts in, which may be
      wider than the tensor's element (its width must be a multiple). It
      is what lets a sub-byte element ride on a ``data_size`` the hardware can
      encode.

    Example:
        Loading a 128x64 tile of a row-major ``gA`` into LDS.

            sA_layout = fx.make_layout((128, 64), (64, 1))
            atom, mA = make_tiled_tdm_atom(TensorLoad(), gA, sA_layout, (128, 64))

            mA = fx.zipped_divide(mA, (128, 64))[None, (bid_x, bid_y)]
            sA = fx.Tensor(fx.make_view(smem_ptr, sA_layout))

            tAsA, tAgA = tdm_partition(atom, warp_coord, warp_layout, sA, mA)
            fx.copy(atom, tAgA, tAsA)

    Choosing ``sA_layout``:
        The layouts below all hold that same 128x64 tile and differ only in how
        it sits in LDS.

            # Plain row-major. No skip, so the atom carries no padding fields.
            fx.make_layout((128, 64), (64, 1))

            # 8 elements of slack after every 64-element row -- the usual bank-conflict
            # dodge. The atom picks it up as `padInterval = 64, padAmount = 8`.
            fx.make_layout((128, 64), (72, 1))

            # The same addresses with M split 8x16.
            fx.make_layout(((8, 16), 64), ((72, 576), 1))

            # This pads once every 8 rows (`padInterval = 512, padAmount = 64`)
            # instead of once every row.
            fx.make_layout(((8, 16), 64), ((64, 576), 1))

        An LDS tile may also be column-major, but that is a property it has to share
        with the tensor: the innermost descriptor dim is the one TDM reads
        contiguously from global memory, so a column-major tile wants a column-major
        ``gA`` and is refused over the row-major one above.
    """
    from ..primitive import make_tile

    if not isinstance(op, (TensorLoad, TensorStore)):
        raise TypeError(
            f"make_tiled_tdm_atom: first argument must be a TensorLoad() or " f"TensorStore() instance, got {op!r}"
        )

    smem_layout = smem_layout.layout if isinstance(smem_layout, Tensor) else smem_layout
    # An `!fly.tile` operand, like `smem_layout`: it is entirely static, so it lives in the
    # value's type and the derivation reads it there.
    tiler = tdm_tile if isinstance(tdm_tile, ir.Value) else make_tile(*tdm_tile)

    common = dict(
        init_boundary_check=ir.BoolAttr.get(init_boundary_check),
        num_warps=num_warps,
        cache_modifier=op.cache_modifier,
        atomic_barrier=bool(atomic_barrier),
        internal_type=(
            None
            if internal_type is None
            else (internal_type.ir_type if hasattr(internal_type, "ir_type") else internal_type)
        ),
    )
    if isinstance(op, TensorLoad):
        atom, tdm_tensor = fly_rocdl.make_tiled_tdm_load_atom(tensor, smem_layout, tiler, **common)
    else:
        atom, tdm_tensor = fly_rocdl.make_tiled_tdm_store_atom(tensor, smem_layout, tiler, **common)
    return atom, tdm_tensor


def tdm_partition(
    atom,
    warp_coord,
    warp_layout,
    stensor,
    gtensor,
):
    """Cut an LDS tile and a coordinate tile into the calls the atom makes.

    Both tiles come out shaped ``((ATOM), (ITER))`` -- mode 0 is one call's worth of
    values and mode 1 counts the calls.

    ``warp_coord`` / ``warp_layout`` say how the warps split the tile: each
    issues one instruction over its own share, and the assembled tile belongs to
    the whole workgroup. Pass ``0`` and ``make_layout(1)`` when a single warp
    does the copy. There is no thread index and no per-lane slice, so within a
    warp every lane sees the same partition.
    """
    from ..primitive import composition, crd2idx, size
    from ..typing import static

    n_warps = size(warp_layout).unpack()

    layout_V = static(fly_rocdl.tdm_partition_layout(atom.type, stensor.type, gtensor.type, n_warps))
    if n_warps == 1:
        return composition(stensor, layout_V), composition(gtensor, layout_V)

    # The multicast coordinate is sliced out of the middle mode: the warps take equal
    # contiguous chunks of the LDS order, and this one is `warp_coord`'s.
    warp_id = crd2idx(warp_coord, warp_layout)
    return (
        composition(stensor, layout_V)[None, warp_id, None],
        composition(gtensor, layout_V)[None, warp_id, None],
    )
