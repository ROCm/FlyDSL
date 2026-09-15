# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device-side record emission for :mod:`flydsl.expr.ktrace`.

Split from ``ktrace.py`` so the public API stays readable: this module holds the
per-kernel prologue (slot reservation, wave identity, block filtering) and the
per-event store.

Cost model.  The prologue runs once per wave and does the only atomic; every event
after it is a timestamp plus the record store.  Claiming a slot per *event* would put
an agent-scope atomic on a mainloop's hot path, which dominates the cost of everything
else in the instrumentation.
"""

from __future__ import annotations

from typing import Any

from .. import ktrace as _ktrace

_I32 = "i32"
_I64 = "i64"
_PTR1 = "!llvm.ptr<1>"


class _KernelState:
    """Per-kernel trace state, rebuilt for each traced kernel body."""

    __slots__ = ("base", "buf_ptr", "hw_id", "xcc_id", "recording")

    def __init__(self):
        self.base = None  # i32 SSA: first slot this wave owns
        self.buf_ptr = None  # !llvm.ptr<1> SSA: the trace buffer parameter
        self.hw_id = None
        self.xcc_id = None
        self.recording = None  # i1 SSA or None when every workgroup records


_STATE: dict[int, _KernelState] = {}

# Names survive beyond the compilation that assigned them, because the host reads the
# table back AFTER the launch -- and on a JIT cache hit the kernel body is never re-traced,
# so ``_STATE`` is empty by then while the device still writes the ids baked into the
# cached binary.  Without this, a second process sharing the disk cache decodes a full
# record list against {} and ``summarize`` silently reports zero phases.
#
# Ids are global rather than per-kernel: the device writes only the integer, and the host
# has no way to tell which kernel a record came from, so two kernels numbering their own
# names from 1 would make id 4 ambiguous.  Assigning from one counter keeps every id
# decodable no matter which kernels a process traced.
_NAMES: dict[str, int] = {}

# Reverse of _NAMES, kept alongside it so merge_names can spot an id claimed by two names.
# The trace writer inverts _NAMES the same way, so a collision here is exactly a record
# that would render under the wrong phase name.
_ID_OWNER: dict[int, str] = {}


def _state() -> _KernelState:
    """State for the kernel currently being traced.

    Keyed on the enclosing ``gpu.func``'s symbol name so a nested region (an ``scf.for``
    body, a guarded branch) shares the prologue emitted at kernel entry instead of
    re-running it.

    The key must be the symbol, not ``id(op)``: the MLIR Python bindings hand back a
    fresh wrapper object on every lookup, so identity changes between calls even for the
    same operation, and every event would start a new kernel's state.

    Seeing a new key drops the previous kernel's entry.  That is safe only because a
    kernel body is traced atomically -- ``KernelFunction._emit_kernel`` sets
    ``_current = self`` and clears it in a ``finally``, with the whole body traced in
    between, and FlyDSL has no kernel-calls-kernel form -- so events never interleave
    between two kernels.  If that ever changes, this clear becomes a live bug: the
    returning kernel would find ``base is None``, re-run ``_prologue()``, and emit a
    SECOND slot claim for a wave that already owns a range, permanently reserving the
    first range and inflating the cursor toward the overflow threshold.
    """
    key = (_kernel_symbol(), _context_token())
    if key not in _STATE:
        _STATE.clear()
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
    from ..._mlir import ir

    return id(ir.Context.current)


def _kernel_symbol() -> str:
    """Symbol name of the ``gpu.func`` being traced, or a stable fallback."""
    from ..._mlir import ir

    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        attrs = getattr(op, "attributes", None)
        if attrs is not None and "sym_name" in attrs:
            return str(attrs["sym_name"])
        op = getattr(op, "parent", None)
    return "<anonymous>"


def name_table() -> dict[str, int]:
    """Every event name seen in this process, for the host-side trace writer.

    Deliberately not per-kernel: see :data:`_NAMES`.  A record carries only its id, so the
    table must stay valid for ids assigned by an earlier compilation -- including one whose
    kernel was served from the disk cache in this process and never re-traced.
    """
    return dict(_NAMES)


def merge_names(names: dict[str, int]) -> None:
    """Adopt ids restored from a cached artifact.

    The restored ids are baked into that binary and cannot be renumbered, so they win over
    any binding this process assigned.  ``event_id`` allocates above every id in use, so a
    name assigned *after* a merge can never collide with a restored one.

    A name assigned *before* the merge can, though, and that case has to be caught rather
    than resolved: trace a fresh kernel (``epilogue`` -> 1), then hit the disk cache for
    another whose binary baked ``mainloop`` -> 1, and both names now claim id 1.  The
    trace writer inverts this table, so one of them would simply disappear and its records
    would render under the other's name -- a plausible-looking timeline for the wrong
    phase.  Neither id can be renumbered: each is already compiled into a binary that will
    write it.  So the trace is unrecoverable and the read must fail loudly.
    """
    for name, ident in names.items():
        owner = _ID_OWNER.get(ident)
        if owner is not None and owner != name:
            raise RuntimeError(
                f"ktrace event id {ident} is claimed by both {owner!r} and {name!r}. "
                "Two kernels compiled with different name tables are live in this process "
                "(typically one traced fresh and one served from the disk cache), and "
                "their records can no longer be told apart. Re-run with "
                "FLYDSL_RUNTIME_ENABLE_CACHE=0 to trace both from source."
            )
        _NAMES[name] = ident
        _ID_OWNER[ident] = name


def event_id(name: str | None) -> int:
    """Index *name* in the process-wide name table, assigning on first use.

    The table is compile-time state serialised beside the trace; the device only writes
    the integer.  There is no cap on name count or length: the id is a 24-bit record
    field, not a packed encoding.
    """
    if name is None:
        return 0  # reserved for range_pop, which carries no name
    if name not in _NAMES:
        # Above the max, not len()+1: ids restored from a cached artifact (see
        # merge_names) need not be contiguous, so len()+1 could hand out an id that is
        # already bound to another name and alias two phases under one id.
        _NAMES[name] = max(_NAMES.values(), default=0) + 1
        _ID_OWNER[_NAMES[name]] = name
    return _NAMES[name]


def _const(value: int, ty_str: str):
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    ty = ir.Type.parse(ty_str)
    return _llvm.ConstantOp(ty, ir.IntegerAttr.get(ty, value)).result


def _gep_bytes(ptr, byte_offset):
    """Byte-offset GEP on a global-address-space pointer."""
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    return _llvm.GEPOp(
        ir.Type.parse(_PTR1),
        ptr,
        [byte_offset],
        [-2147483648],  # kDynamicIndex: the single index is dynamic
        ir.Type.parse("i8"),
        _llvm.GEPNoWrapFlags.none,
    ).result


def _block_filter():
    """Wave-uniform predicate for ``FLYDSL_KTRACE_BLOCKS``, or None to record everything.

    Filtering on the device -- rather than during post-processing -- means a
    non-recording wave pays one comparison at entry and writes nothing at all.
    """
    from ..._mlir.dialects import arith as _arith
    from ...utils import env
    from ..gpu import block_id

    spec = (env.ktrace.blocks or "").strip()
    if not spec or spec == "all":
        return None

    st = _state()
    if spec.startswith("xcc:"):
        want = _const(int(spec[4:]), _I32)
        return _arith.CmpIOp(_arith.CmpIPredicate.eq, st.xcc_id, want).result

    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"FLYDSL_KTRACE_BLOCKS must be 'x,y,z', 'xcc:N' or 'all'; got {spec!r}")

    cond = None
    for axis, want in zip(("x", "y", "z"), parts):
        got = _as_i32(block_id(axis))
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
    from ..._mlir import ir

    st = _state()
    if st.base is not None:
        return st

    entry = _kernel_entry_block()
    ctx = ir.InsertionPoint.at_block_begin(entry) if entry is not None else _null_ctx()
    with ctx:
        return _emit_prologue_body(st, entry)


def _null_ctx():
    import contextlib

    return contextlib.nullcontext()


def _kernel_entry_block():
    """Entry block of the enclosing kernel function, or None if not inside one.

    The kernel op reports its *symbol* as ``.name`` (e.g. ``"probe_0"``), not
    ``"gpu.func"``, so it is identified by position instead: the last op on the way up
    before ``gpu.module``.
    """
    from ..._mlir import ir

    op = ir.InsertionPoint.current.block.owner
    last = None
    while op is not None:
        if getattr(op, "name", "") == "gpu.module":
            return last.regions[0].blocks[0] if last is not None else None
        last = op
        op = getattr(op, "parent", None)
    return None


def _emit_prologue_body(st, entry):
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith
    from ..._mlir.dialects import llvm as _llvm
    from ...utils import env
    from . import inline_asm

    # HW_ID and XCC_ID are wave-invariant -- waves are CU-resident on CDNA -- so they are
    # read once and reused by every event.
    st.hw_id = inline_asm.s_getreg_hw_id()
    st.xcc_id = inline_asm.s_getreg_xcc_id()

    # The buffer arrives as the trailing implicit kernel parameter (see
    # compiler/kernel_function.py). Binding a device global instead would work, but it
    # needs a post_load_processors callback, which marks the kernel non-disk-cacheable.
    st.buf_ptr = _trace_buffer_param(entry)

    # The filter is computed BEFORE the claim: a filtered-out wave must not claim slots
    # either. A realistic GEMM launches thousands of waves, and if every one claimed its
    # range the cursor would run past the buffer even though only a handful ever record
    # -- the claim itself then overflows, which is an illegal memory access rather than a
    # truncated trace.
    st.recording = _block_filter()

    # One atomic per wave, not per event: claim a contiguous slot range up front. The
    # cursor lives in the buffer's reserved slot 0 rather than a device global, so the
    # host reads the same memory the kernel wrote -- a separately allocated cursor reads
    # zero forever, with no error anywhere.
    per_wave = _const(int(env.ktrace.events_per_wave), _I32)

    def _claim():
        claimed = _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, st.buf_ptr, per_wave, _llvm.AtomicOrdering.monotonic).result
        # The atomic returns the OLD cursor, so the first wave would get base 0 -- whose
        # counter word is the global cursor itself, and every event would corrupt it.
        # Shift every base past the reserved cursor slot so slot 0 is the cursor alone.
        return _arith.AddIOp(claimed, _const(1, _I32)).result

    if st.recording is None:
        st.base = _claim()
    else:
        from ..._mlir.dialects import scf

        claim_if = scf.IfOp(st.recording, [ir.Type.parse(_I32)], has_else=True)
        for region in claim_if.regions:
            if len(region.blocks) == 0:
                region.blocks.append(*[])
        with ir.InsertionPoint(claim_if.regions[0].blocks[0]):
            scf.YieldOp([_claim()])
        with ir.InsertionPoint(claim_if.regions[1].blocks[0]):
            # Non-recording waves never store, so any base works; 0 costs no atomic.
            scf.YieldOp([_const(0, _I32)])
        st.base = claim_if.results[0]

    return st


def _trace_buffer_param(entry):
    """The trailing ``!llvm.ptr<1>`` parameter carrying the trace buffer.

    *entry* is the kernel's entry block, resolved before the insertion point moved into
    it -- once inside, the parent chain no longer reaches the gpu.func.
    """
    if entry is None:
        raise RuntimeError("ktrace record emission must happen inside a @flyc.kernel body")

    args = list(entry.arguments)
    if not args:
        raise RuntimeError(
            "trace buffer parameter is missing; the kernel was traced without "
            "tracing enabled at jit-argument construction time"
        )
    # The parameter is a !fly.ptr (matching the host-side PointerJitArg); the record
    # stores need the LLVM pointer behind it.
    from ..typing import Pointer

    return Pointer(args[-1]).llvm_ptr


def _payload_value(payload: Any):
    """Coerce a payload to a single i64, or return a zero constant when absent.

    Uses the single-value accessor rather than a multi-value-tolerant helper, so a
    mistakenly-passed RangeToken raises at the call site instead of silently producing a
    malformed record.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith

    if payload is None:
        return _const(0, _I64)

    if hasattr(payload, "ir_value"):
        value = payload.ir_value()
    elif isinstance(payload, bool):
        return _const(int(payload), _I64)
    elif isinstance(payload, int):
        return _const(payload, _I64)
    elif isinstance(payload, float):
        from ..numeric import Float64

        value = Float64(payload).ir_value()
    else:
        value = payload

    if not isinstance(value, ir.Value):
        raise TypeError(f"ktrace payload must coerce to a single value, got {type(payload).__name__}")

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
    raise TypeError(f"unsupported ktrace payload type {ty}")


def emit_record(event_name: str | None, kind: int, *, payload: Any = None, start_slot=None):
    """Write one 32-byte record.  Returns the slot i32 for START events."""
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith
    from ..._mlir.dialects import llvm as _llvm
    from ..._mlir.dialects import scf
    from ...utils import env
    from . import inline_asm

    st = _prologue()

    eid = event_id(event_name)

    def _store_body(slot):
        """Write the record and return the absolute slot it occupies.

        The returned value is what a token range pairs on, so it must be the record's own
        position -- the same index the host decodes -- not the pre-offset `slot`.
        """
        # Layout: slot 0 is the global cursor; slot `base` is this wave's counter word;
        # its records follow at base+1+local. `slot` is already base+local, so one more
        # slot skips exactly the counter.
        slot_1 = _arith.AddIOp(slot, _const(1, _I32)).result
        byte_off = _arith.MulIOp(slot_1, _const(_ktrace.RECORD_BYTES, _I32)).result
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

        return slot_1

    # One record per wave, not 64: the first *active* lane writes it. It must be the
    # first active lane rather than lane 0 -- inside a divergent region lane 0 may be
    # inactive, and the event would be dropped with no trace of the loss.
    lane = _mbcnt()
    leader = _first_active_lane()
    is_leader = _arith.CmpIOp(_arith.CmpIPredicate.eq, lane, leader).result
    cond = is_leader if st.recording is None else _arith.AndIOp(is_leader, st.recording).result

    # The slot index is computed at RUNTIME, not from a compile-time counter: a call site
    # inside a loop is traced once but executes every iteration, and a constant index would
    # make all iterations overwrite the same slot, leaving only the last -- silently, and
    # precisely for the per-k-tile annotation this feature exists to support.
    #
    # Both the counter bump and the store sit inside the leader guard: bumping from all 64
    # lanes would consume 64 slots per event while only one lane writes a record. The guard
    # yields the slot so range_start can hand it to its range_end -- token ranges pair by
    # the START record's own slot, so two ranges open at once must not share a value.
    if_op = scf.IfOp(cond, [ir.Type.parse(_I32)], has_else=True)
    for region in if_op.regions:
        if len(region.blocks) == 0:
            region.blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        # The counter is the wave's own first reserved word, so the atomic is uncontended.
        ctr_ptr = _gep_bytes(st.buf_ptr, _arith.MulIOp(st.base, _const(_ktrace.RECORD_BYTES, _I32)).result)
        local = _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add, ctr_ptr, _const(1, _I32), _llvm.AtomicOrdering.monotonic
        ).result
        slot = _arith.AddIOp(st.base, local).result

        # Two runtime bounds: the wave must stay inside its own reservation, and the slot
        # inside the buffer. The counter still counts past both, so the host sees how much
        # was wanted; only the STORE is suppressed. Without this the kernel writes outside
        # the buffer once enough waves have claimed ranges -- an illegal memory access that
        # takes the process down, not a truncated trace.
        #
        # UNSIGNED compares: `local` and `slot` come from an i32 atomic add that is never
        # saturated, so across enough recording waves the cursor passes 2^31 and `slot`
        # goes negative as a signed i32. A signed `slt` is then unconditionally true and
        # the guard inverts from "in bounds" to "always pass", GEPing to a negative byte
        # offset -- exactly the out-of-bounds write this check exists to stop. The host
        # reads the cursor as c_uint32, so unsigned is also the interpretation the two
        # sides already agree on.
        per_wave = _const(int(env.ktrace.events_per_wave) - 1, _I32)
        within_wave = _arith.CmpIOp(_arith.CmpIPredicate.ult, local, per_wave).result
        capacity = _const(_buffer_capacity_slots(), _I32)
        in_bounds = _arith.CmpIOp(_arith.CmpIPredicate.ult, slot, capacity).result
        store_ok = _arith.AndIOp(within_wave, in_bounds).result

        # The record's own slot is what range_end pairs on, so it is yielded out of the
        # bounds guard too. A suppressed store must yield a value NO record can ever
        # carry: yielding the unoffset `slot` aliases the record written one event
        # earlier, whose absolute slot is that same number (a store at local-1 lands at
        # base+local). A range_end carrying that stale id then pops an unrelated open
        # range and reports a fabricated duration against the wrong phase.
        store_if = scf.IfOp(store_ok, [ir.Type.parse(_I32)], has_else=True)
        for region in store_if.regions:
            if len(region.blocks) == 0:
                region.blocks.append(*[])
        with ir.InsertionPoint(store_if.regions[0].blocks[0]):
            scf.YieldOp([_store_body(slot)])
        with ir.InsertionPoint(store_if.regions[1].blocks[0]):
            scf.YieldOp([_const(_ktrace.UNPAIRED_SLOT, _I32)])
        scf.YieldOp([store_if.results[0]])
    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
        # Non-leader lanes never write a record, so they must yield the same unpairable id
        # as a suppressed store. Yielding the wave base instead would hand every non-leader
        # lane a real slot number -- the wave's own counter word -- and a range_end built
        # from it would pair against whatever record later occupies that slot.
        scf.YieldOp([_const(_ktrace.UNPAIRED_SLOT, _I32)])

    return if_op.results[0]


def _buffer_capacity_slots() -> int:
    """Record slots the buffer holds, excluding the reserved cursor slot."""
    from ...utils import env

    return max(int(env.ktrace.buffer_bytes) // _ktrace.RECORD_BYTES - 1, 1)


def _as_i32(value):
    """Normalise a DSL scalar to an ``i32`` SSA value.

    ``lane_id()`` and ``block_id()`` yield ``index`` in some kernels and ``i32`` in
    others, yet both are compared against ``i32`` values.  Without this the mismatch
    surfaces only as a module-verifier error naming the kernel's first ktrace call site,
    far from the comparison that is actually ill-typed.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith

    value = value.ir_value() if hasattr(value, "ir_value") else value
    ty = str(value.type)
    i32 = ir.Type.parse(_I32)
    if ty == _I32:
        return value
    if ty == "index":
        return _arith.IndexCastOp(i32, value).result
    if not ty.startswith("i"):
        raise TypeError(f"expected an integer or index value for an ktrace comparison, got {ty}")
    width = int(ty[1:])
    if width == 32:
        return value
    return _arith.ExtUIOp(i32, value).result if width < 32 else _arith.TruncIOp(i32, value).result


def _mbcnt():
    """Lane index within the wave, as an i32."""
    from ..gpu import lane_id

    return _as_i32(lane_id())


def _first_active_lane():
    """Index of the lowest active lane, as an i32."""
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith
    from ..._mlir.dialects import llvm as _llvm
    from . import ballot

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
        from ...utils import ktrace_runtime

        _BUFFER = ktrace_runtime.allocate()
    return _BUFFER


def current_buffer():
    """The trace buffer, or None if nothing has been traced yet."""
    return _BUFFER
