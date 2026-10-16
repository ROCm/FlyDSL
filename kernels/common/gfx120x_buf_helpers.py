# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared gfx120x buffer-pointer helpers for FlyDSL kernel builders.

Canonical home for ``ptr_buf_tensor``, ``buf_copy_load`` / ``buf_copy_store``,
``buf_base_i64``, and ``kernel_signature``. Used by the gfx120x quant,
elementwise, norm, RoPE, and AdaLN kernel modules.

These are compile-time kernel helpers, not launch functions. They do not
check the architecture. Each gfx120x launch calls ``require_gfx120x``.

"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import ptrtoint
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# Nominal extent for a raw pointer view; the descriptor bound is the real OOB guard.
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF

__all__ = [
    "BUF_VIEW_MAX_ELEMS",
    "kernel_signature",
    "buf_base_i64",
    "ptr_buf_tensor",
    "buf_copy_load",
    "buf_copy_store",
]


def kernel_signature(**params: object) -> str:
    """Render specialization parameters into a legal kernel-name suffix."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


def buf_base_i64(base: object) -> fx.Int64:
    """Return the byte address of a pointer or tensor-like kernel argument."""
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    if isinstance(raw.type, (ir.IntegerType, ir.IndexType)):
        return fx.Int64(base)
    aligned = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return fx.Int64(llvm.PtrToIntOp(T.i64, aligned).result)


def ptr_buf_tensor(
    ptr: object,
    elem: object = fx.Int32,
    n: object = BUF_VIEW_MAX_ELEMS,
    unit_elems: int = 1,
    num_records_bytes: object = None,
    unit_stride: object = None,
) -> fx.Tensor:
    """Create a buffer-resource view over a raw device pointer or tensor-like arg."""
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = fx.make_layout((n,), (1,)) if unit_elems == 1 else fx.make_layout((n, unit_elems), (unit_stride, 1))
    ptr_type = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(ptr_type, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}


def _buf_copy_slice(buffer: object, index: object, unit_elems: object) -> fx.Tensor:
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


def buf_copy_load(
    buffer: object, index: object, elem: object = fx.Int32, unit_elems: int = 1, cache_modifier: int = 0
) -> object:
    """Load one scalar/vector unit through an AMD buffer-copy atom."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    atom = fx.make_copy_atom(_BUF_COPY_ATOM[unit_elems * (elem.width // 8)](cache_modifier), elem)
    fx.copy(atom, _buf_copy_slice(buffer, index, unit_elems), fragment)
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


def buf_copy_store(
    buffer: object, index: object, value: object, elem: object = fx.Int32, unit_elems: int = 1, cache_modifier: int = 0
) -> None:
    """Store one scalar/vector unit through an AMD buffer-copy atom."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    atom = fx.make_copy_atom(_BUF_COPY_ATOM[unit_elems * (elem.width // 8)](cache_modifier), elem)
    fx.copy(atom, fragment, _buf_copy_slice(buffer, index, unit_elems))
