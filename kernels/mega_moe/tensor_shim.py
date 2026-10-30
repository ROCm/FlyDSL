# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec
from kernels.common.tensor_shim import (  # noqa: F401  # _preload_compiled/_run_compiled are re-exported
    BUF_VIEW_MAX_ELEMS,
    _preload_compiled,
    _run_compiled,
    buf_base_i64,
)

_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}


def ptr_buf_tensor(
    ptr,
    elem=fx.Int32,
    n=BUF_VIEW_MAX_ELEMS,
    unit_elems=1,
    num_records_bytes=None,
    unit_stride=None,
):
    """Buffer-resource (V#) view of *ptr*, so ``t[i]`` / ``fx.slice`` index it.

    *ptr* is an fx.Pointer kernel arg or a raw i64 address (see
    :func:`buf_base_i64`).

    Keeps the addressing `buffer_ops` used: descriptor in SGPRs, 32-bit voffset
    per access. Indexing a plain typed pointer instead builds a full 64-bit
    address in VGPRs on every access.

    ``unit_elems`` sets the access width and hence the rank:
      1  -> flat ``(n,)``; ``t[i]`` is one element. No atom, no fragment.
      >1 -> ``(n, unit_elems)``; ``fx.slice(t, (u, None))`` is one wide access
            and ``fx.copy`` over it emits a single ``buffer_load_dwordx{2,4}``.
            A vector *element type* would be the obvious spelling for this but
            MLIR rejects it (``AlignAttr`` takes integer/float only), so the
            width lives in the layout.

    ``unit_stride`` is the element distance between consecutive units and
    defaults to ``unit_elems`` (units tile the buffer, so ``u`` counts whole
    units). Pass 1 for a wide access at an arbitrary *element* offset: units
    then overlap -- meaningless to iterate but exact for the one slice a caller
    takes, and the only way to express e.g. a dwordx4 store at a row base that
    is only dword-aligned. It also sets the pointer alignment, which is all such
    an access can rely on.

    ``num_records_bytes`` may be a runtime value, for a hardware OOB check that
    zero-fills rather than reading stale bytes.
    """
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = fx.make_layout((n,), (1,)) if unit_elems == 1 else fx.make_layout((n, unit_elems), (unit_stride, 1))
    pt = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(pt, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def buf_copy_atom(unit_bytes, elem=fx.Int32, cache_modifier=0):
    """Copy atom for a ``unit_bytes``-wide buffer access (0=cached, 2=nt)."""
    return fx.make_copy_atom(_BUF_COPY_ATOM[unit_bytes](cache_modifier), elem)


def _buf_copy_slice(buffer, index, unit_elems):
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


def buf_copy_load(buffer, index, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Load one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fx.copy(
        buf_copy_atom(unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier),
        _buf_copy_slice(buffer, index, unit_elems),
        fragment,
    )
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


def buf_copy_store(buffer, index, value, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Store one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    fx.copy(
        buf_copy_atom(unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier),
        fragment,
        _buf_copy_slice(buffer, index, unit_elems),
    )
