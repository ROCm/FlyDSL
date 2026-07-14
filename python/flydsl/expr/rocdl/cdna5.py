# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx1250-specific ROCDL atom builders (N-D TDM whole-tile Global<->LDS copy)."""

from ..._mlir.dialects.fly_rocdl import CopyOpGFX1250TDMType
from ..typing import Int32, Int64, Tensor


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
    return CopyOpGFX1250TDMType.get(
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
