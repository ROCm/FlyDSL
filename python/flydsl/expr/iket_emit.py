# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device-side record emission for :mod:`flydsl.expr.iket`.

Split from ``iket.py`` so the public API stays readable: this module holds the
per-kernel prologue (slot reservation, wave identity, block filtering) and the
per-event store.

Cost model.  The prologue runs once per wave and does the only atomic; every event
after it is a timestamp plus the record store.  Claiming a slot per *event* would put
an agent-scope atomic on a mainloop's hot path, which dominates the cost of everything
else in the instrumentation.
"""

from __future__ import annotations

from typing import Any

from . import iket as _iket

_I32 = "i32"
_I64 = "i64"
_PTR1 = "!llvm.ptr<1>"


class _KernelState:
    """Per-kernel trace state, rebuilt for each traced kernel body."""

    __slots__ = ("base", "next_index", "buf_ptr", "hw_id", "xcc_id", "recording", "names")

    def __init__(self):
        self.base = None  # i32 SSA: first slot this wave owns
        self.next_index = 0  # Python int: events emitted so far in this trace
        self.buf_ptr = None  # !llvm.ptr<1> SSA: the trace buffer parameter
        self.hw_id = None
        self.xcc_id = None
        self.recording = None  # i1 SSA or None when every workgroup records
        self.names: dict[str, int] = {}


_STATE: dict[int, _KernelState] = {}


def _enclosing(name: str):
    """Nearest enclosing op with the given MLIR name, from the insertion point."""
    from .._mlir import ir

    op = ir.InsertionPoint.current.block.owner
    while op is not None and getattr(op, "name", "") != name:
        op = getattr(op, "parent", None)
    return op


def _state() -> _KernelState:
    """State for the kernel currently being traced.

    Keyed on the enclosing ``gpu.func``'s symbol name so a nested region (an ``scf.for``
    body, a guarded branch) shares the prologue emitted at kernel entry instead of
    re-running it.

    The key must be the symbol, not ``id(op)``: the MLIR Python bindings hand back a
    fresh wrapper object on every lookup, so identity changes between calls even for the
    same operation, and every event would start a new kernel's state.
    """
    key = (_kernel_symbol(), _context_token())
    if key not in _STATE:
        _STATE.clear()  # one kernel is traced at a time
        _STATE[key] = _KernelState()
    return _STATE[key]


def _context_token() -> int:
    """Identity of the MLIR context the current insertion point belongs to.

    The symbol name alone is not a sufficient key: two compilations can trace kernels
    with the same name (very common in tests), and reusing the first one's cached SSA
    values inside the second one's context dereferences freed memory -- a segfault, not
    an exception.  Pairing the name with the live context makes a stale entry unreachable
    instead of catastrophic.
    """
    from .._mlir import ir

    return id(ir.Context.current)


def _kernel_symbol() -> str:
    """Symbol name of the ``gpu.func`` being traced, or a stable fallback."""
    from .._mlir import ir

    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        attrs = getattr(op, "attributes", None)
        if attrs is not None and "sym_name" in attrs:
            return str(attrs["sym_name"])
        op = getattr(op, "parent", None)
    return "<anonymous>"


def reset_state() -> None:
    """Drop cached state; the runtime calls this between compilations."""
    _STATE.clear()


def name_table() -> dict[str, int]:
    """Name table for the kernel just traced, for the host-side trace writer."""
    for st in _STATE.values():
        return dict(st.names)
    return {}


def event_id(name: str | None) -> int:
    """Index *name* in this kernel's name table, assigning on first use.

    The table is compile-time state serialised beside the trace; the device only writes
    the integer.  There is no cap on name count or length: the id is a 24-bit record
    field, not a packed encoding.
    """
    if name is None:
        return 0  # reserved for range_pop, which carries no name
    st = _state()
    if name not in st.names:
        st.names[name] = len(st.names) + 1
    return st.names[name]


def _const(value: int, ty_str: str):
    from .._mlir import ir
    from .._mlir.dialects import llvm as _llvm

    ty = ir.Type.parse(ty_str)
    return _llvm.ConstantOp(ty, ir.IntegerAttr.get(ty, value)).result


def _gep_bytes(ptr, byte_offset):
    """Byte-offset GEP on a global-address-space pointer."""
    from .._mlir import ir
    from .._mlir.dialects import llvm as _llvm

    return _llvm.GEPOp(
        ir.Type.parse(_PTR1),
        ptr,
        [byte_offset],
        [-2147483648],  # kDynamicIndex: the single index is dynamic
        ir.Type.parse("i8"),
        _llvm.GEPNoWrapFlags.none,
    ).result


def _block_filter():
    """Wave-uniform predicate for ``FLYDSL_IKET_BLOCKS``, or None to record everything.

    Filtering on the device -- rather than during post-processing -- means a
    non-recording wave pays one comparison at entry and writes nothing at all.
    """
    from .._mlir.dialects import arith as _arith
    from ..utils import env
    from .gpu import block_id

    spec = (env.iket.blocks or "").strip()
    if not spec or spec == "all":
        return None

    st = _state()
    if spec.startswith("xcc:"):
        want = _const(int(spec[4:]), _I32)
        return _arith.CmpIOp(_arith.CmpIPredicate.eq, st.xcc_id, want).result

    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"FLYDSL_IKET_BLOCKS must be 'x,y,z', 'xcc:N' or 'all'; got {spec!r}")

    cond = None
    for axis, want in zip(("x", "y", "z"), parts):
        got = block_id(axis)
        got = got.ir_value() if hasattr(got, "ir_value") else got
        term = _arith.CmpIOp(_arith.CmpIPredicate.eq, got, _const(int(want), _I32)).result
        cond = term if cond is None else _arith.AndIOp(cond, term).result
    return cond


def _prologue() -> _KernelState:
    """Emit the once-per-wave prologue, or return the state if it already ran.

    The prologue is emitted at the **entry block of the kernel**, not at whichever event
    site happens to come first.  Its values (the slot base, the buffer address, the wave
    ids) are used by every later event, so emitting them inside an ``scf.for`` body or a
    guarded branch would leave them failing to dominate their uses outside that region --
    an "operand does not dominate this use" verifier error whose cause is far from the
    line that triggers it.
    """
    from .._mlir import ir
    from .._mlir.dialects import llvm as _llvm
    from ..utils import env
    from .rocdl import inline_asm

    st = _state()
    if st.base is not None:
        return st

    entry = _kernel_entry_block()
    ctx = ir.InsertionPoint.at_block_begin(entry) if entry is not None else _null_ctx()
    with ctx:
        return _emit_prologue_body(st, env, inline_asm, ir, _llvm, entry)


def _null_ctx():
    import contextlib

    return contextlib.nullcontext()


def _kernel_entry_block():
    """Entry block of the enclosing kernel function, or None if not inside one.

    The kernel op reports its *symbol* as ``.name`` (e.g. ``"probe_0"``), not
    ``"gpu.func"``, so it is identified by position instead: the last op on the way up
    before ``gpu.module``.
    """
    from .._mlir import ir

    op = ir.InsertionPoint.current.block.owner
    last = None
    while op is not None:
        if getattr(op, "name", "") == "gpu.module":
            return last.regions[0].blocks[0] if last is not None else None
        last = op
        op = getattr(op, "parent", None)
    return None


def _emit_prologue_body(st, env, inline_asm, ir, _llvm, entry):

    # HW_ID and XCC_ID are wave-invariant -- waves are CU-resident on CDNA -- so they are
    # read once and reused by every event.
    st.hw_id = inline_asm.s_getreg_hw_id()
    st.xcc_id = inline_asm.s_getreg_xcc_id()

    # The buffer arrives as the trailing implicit kernel parameter (see
    # compiler/kernel_function.py). Binding a device global instead would work, but it
    # needs a post_load_processors callback, which marks the kernel non-disk-cacheable.
    st.buf_ptr = _trace_buffer_param(entry)

    # One atomic per wave, not per event: claim a contiguous slot range up front. The
    # cursor lives in the buffer's reserved slot 0 rather than a device global, so the
    # host reads the same memory the kernel wrote -- a separately allocated cursor reads
    # zero forever, with no error anywhere.
    per_wave = _const(int(env.iket.events_per_wave), _I32)
    st.base = _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, st.buf_ptr, per_wave, _llvm.AtomicOrdering.monotonic).result

    st.recording = _block_filter()
    return st


def _trace_buffer_param(entry):
    """The trailing ``!llvm.ptr<1>`` parameter carrying the trace buffer.

    *entry* is the kernel's entry block, resolved before the insertion point moved into
    it -- once inside, the parent chain no longer reaches the gpu.func.
    """
    if entry is None:
        raise RuntimeError("iket record emission must happen inside a @flyc.kernel body")

    args = list(entry.arguments)
    if not args:
        raise RuntimeError(
            "iket trace buffer parameter is missing; the kernel was traced without "
            "tracing enabled at jit-argument construction time"
        )
    # The parameter is a !fly.ptr (matching the host-side PointerJitArg); the record
    # stores need the LLVM pointer behind it.
    from .typing import Pointer

    return Pointer(args[-1]).llvm_ptr


def _payload_value(payload: Any):
    """Coerce a payload to a single i64, or return a zero constant when absent.

    Uses the single-value accessor rather than a multi-value-tolerant helper, so a
    mistakenly-passed RangeToken raises at the call site instead of silently producing a
    malformed record.
    """
    from .._mlir import ir
    from .._mlir.dialects import arith as _arith

    if payload is None:
        return _const(0, _I64)

    if hasattr(payload, "ir_value"):
        value = payload.ir_value()
    elif isinstance(payload, bool):
        return _const(int(payload), _I64)
    elif isinstance(payload, int):
        return _const(payload, _I64)
    elif isinstance(payload, float):
        from .numeric import Float64

        value = Float64(payload).ir_value()
    else:
        value = payload

    if not isinstance(value, ir.Value):
        raise TypeError(f"iket payload must coerce to a single value, got {type(payload).__name__}")

    ty = str(value.type)
    i64 = ir.Type.parse(_I64)
    if ty == _I64:
        return value
    if ty.startswith("i"):
        width = int(ty[1:])
        return value if width == 64 else _arith.ExtUIOp(i64, value).result
    if ty in ("f32", "f64"):
        bits = _arith.BitcastOp(ir.Type.parse("i32" if ty == "f32" else _I64), value).result
        return bits if ty == "f64" else _arith.ExtUIOp(i64, bits).result
    raise TypeError(f"unsupported iket payload type {ty}")


def emit_record(event_name: str | None, kind: int, *, payload: Any = None, start_slot=None):
    """Write one 32-byte record.  Returns the slot i32 for START events."""
    from .._mlir import ir
    from .._mlir.dialects import arith as _arith
    from .._mlir.dialects import llvm as _llvm
    from .._mlir.dialects import scf
    from ..utils import env
    from .rocdl import inline_asm

    st = _prologue()

    per_wave = int(env.iket.events_per_wave)
    index = st.next_index
    st.next_index += 1
    eid = event_id(event_name)

    slot = _arith.AddIOp(st.base, _const(index, _I32)).result

    # A wave that emits more events than it reserved would write into the next wave's
    # range, so the store is skipped once the reservation is exhausted.  The host sees
    # the shortfall through the cursor and rejects the trace rather than trimming it.
    if index >= per_wave:
        return slot

    def _store_body():
        # Slot 0 is the cursor, so records start one record-length in.
        slot_1 = _arith.AddIOp(slot, _const(1, _I32)).result
        byte_off = _arith.MulIOp(slot_1, _const(_iket.RECORD_BYTES, _I32)).result
        rec_ptr = _gep_bytes(st.buf_ptr, byte_off)

        ts = inline_asm.s_memrealtime()
        packed = (eid & 0xFFFFFF) | ((kind & 0xFF) << 24)
        start = start_slot._slot if start_slot is not None else _const(0, _I32)

        for offset, value in (
            (0, ts),
            (8, _payload_value(payload)),
            (16, st.hw_id),
            (20, st.xcc_id),
            (24, _const(packed, _I32)),
            (28, start),
        ):
            field = _gep_bytes(rec_ptr, _const(offset, _I32))
            _llvm.StoreOp(value, field)

    # One record per wave, not 64: the first *active* lane writes it. It must be the
    # first active lane rather than lane 0 -- inside a divergent region lane 0 may be
    # inactive, and the event would be dropped with no trace of the loss.
    lane = _mbcnt()
    leader = _first_active_lane()
    is_leader = _arith.CmpIOp(_arith.CmpIPredicate.eq, lane, leader).result
    cond = is_leader if st.recording is None else _arith.AndIOp(is_leader, st.recording).result

    # Runtime bound: the cursor is allowed to run past capacity so the host can see how
    # much was wanted, but the STORE must not. Without this the kernel writes outside the
    # buffer once enough waves have claimed ranges -- an illegal memory access that takes
    # the whole process down, not a truncated trace.
    capacity = _const(_buffer_capacity_slots(), _I32)
    in_bounds = _arith.CmpIOp(_arith.CmpIPredicate.slt, slot, capacity).result
    cond = _arith.AndIOp(cond, in_bounds).result

    if_op = scf.IfOp(cond, [], has_else=False)
    if len(if_op.regions[0].blocks) == 0:
        if_op.regions[0].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        _store_body()
        scf.YieldOp([])

    return slot


def _buffer_capacity_slots() -> int:
    """Record slots the buffer holds, excluding the reserved cursor slot."""
    from ..utils import env

    return max(int(env.iket.buffer_bytes) // _iket.RECORD_BYTES - 1, 1)


def _mbcnt():
    """Lane index within the wave."""
    from .gpu import lane_id

    value = lane_id()
    return value.ir_value() if hasattr(value, "ir_value") else value


def _first_active_lane():
    """Index of the lowest active lane, as an i32."""
    from .._mlir import ir
    from .._mlir.dialects import arith as _arith
    from .._mlir.dialects import llvm as _llvm
    from .rocdl import ballot

    i64 = ir.Type.parse(_I64)
    mask = ballot(i64, _const(1, "i1"))
    mask = mask.ir_value() if hasattr(mask, "ir_value") else mask
    if str(mask.type) != _I64:
        mask = _arith.ExtUIOp(i64, mask).result
    # cttz(mask) -- the mask is never zero inside an executing region.
    zeros = _llvm.CountTrailingZerosOp(mask, True).result
    return _arith.TruncIOp(ir.Type.parse(_I32), zeros).result


# The host-side buffer for the current traced run. The jit layer fills the kernel's
# trailing pointer argument from it, and collect() reads records back through it.
_BUFFER = None


def ensure_buffer():
    """Allocate the process-wide trace buffer on first use."""
    global _BUFFER
    if _BUFFER is None:
        from ..utils import iket_runtime

        _BUFFER = iket_runtime.allocate()
    return _BUFFER


def current_buffer():
    """The trace buffer, or None if nothing has been traced yet."""
    return _BUFFER
