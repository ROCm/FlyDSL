# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import ptrtoint
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

_PRELOAD_COMPILE_LOCK = threading.RLock()


_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}

# Nominal extent of a raw-pointer buffer view. The real bound is the V#
# num_records field, which make_buffer_tensor(max_size=True) pins to 0xFFFFFFFF
# bytes, so this only has to be large enough not to constrain any caller's
# indices.
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


def buf_base_i64(base):
    """i64 base address of *base*: an fx pointer, a tensor/memref, or an address.

    Lets a caller narrow a descriptor to one row (``ptr + row * row_bytes``)
    without first materialising a pointer for it.
    """
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    if isinstance(raw.type, (ir.IntegerType, ir.IndexType)):
        return fx.Int64(base)
    # tensor / memref kernel arg: take its aligned base pointer.
    aligned = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return fx.Int64(llvm.PtrToIntOp(T.i64, aligned).result)


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


def _compile_and_run(exe, *args):
    """``flyc.compile(exe, *args)``: compiles **and** executes the kernel, and
    returns its ``CompiledFunction`` (None under compile-only)."""
    try:
        return flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _run_compiled(exe, *args, specialization_key=None):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.

    A specialization key gives a multi-constexpr JitFunction one compiled
    callable per configuration. Factory-style launchers should leave it unset.
    """
    if specialization_key is None:
        cf = getattr(exe, "_cf", None)
    else:
        cache = getattr(exe, "_cf_by_specialization", None)
        cf = cache.get(specialization_key) if cache is not None else None
    if cf is not None:
        cf(*args)
        return
    cf = _compile_and_run(exe, *args)
    if specialization_key is None:
        exe._cf = cf
    else:
        if not hasattr(exe, "_cf_by_specialization"):
            exe._cf_by_specialization = {}
        exe._cf_by_specialization[specialization_key] = cf


def _preload_compiled(exe, *args):
    """Materialize a JIT artifact without dispatching its GPU kernels.

    Newer FlyDSL versions expose ``JitFunction.preload()`` for this operation.
    Older supported versions provide the same no-dispatch behavior through the
    public ``COMPILE_ONLY`` mode.  Calling ``flyc.compile()`` without either
    guard executes the launcher and is unsafe for preload callers that pass
    placeholder pointers.
    """
    preload = getattr(exe, "preload", None)
    if callable(preload):
        return preload(*args)

    with _PRELOAD_COMPILE_LOCK:
        old_compile_only = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            return flyc.compile(exe, *args)
        finally:
            if old_compile_only is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = old_compile_only
