# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side support for :mod:`flydsl.expr.ktrace`.

The device-side record emission that used to live here -- the per-kernel prologue and
the per-event store -- is now ``convert-fly-ktrace-to-rocdl``; the frontend emits
``fly_ktrace`` annotations and the pass expands them. What remains is what the host
still owns: the event-name table the trace writer decodes against, the payload
coercion the ops need for their ``Optional<I64>`` operand, and the trace buffer.
"""

from __future__ import annotations

from typing import Any

from .. import ktrace as _ktrace

_I64 = "i64"

# Names survive beyond the compilation that assigned them, because the host reads the
# table back AFTER the launch -- and on a JIT cache hit the kernel body is never
# re-traced, so nothing re-registers them while the device still writes the ids baked
# into the cached binary.  Without this, a second process sharing the disk cache decodes
# a full record list against {} and ``summarize`` silently reports zero phases.
#
# Ids are global rather than per-kernel: the device writes only the integer, and the host
# has no way to tell which kernel a record came from, so two kernels numbering their own
# names from 1 would make id 4 ambiguous.  The pass numbers from one counter per module
# for that reason; this table is where those ids accumulate across compilations.
_NAMES: dict[str, int] = {}

# Reverse of _NAMES, kept alongside it so merge_names can spot an id claimed by two names.
# The trace writer inverts _NAMES the same way, so a collision here is exactly a record
# that would render under the wrong phase name.
_ID_OWNER: dict[int, str] = {}


def _const(value: int, ty_str: str):
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    ty = ir.Type.parse(ty_str)
    return _llvm.ConstantOp(ty, ir.IntegerAttr.get(ty, value)).result


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


def _payload_value(payload: Any):
    """Coerce a payload to a single i64.

    Never called with None: the op declares ``Optional<I64>``, so an absent payload
    omits the operand (see ``_dialect_payload``) and the pass materialises the zero.

    Uses the single-value accessor rather than a multi-value-tolerant helper, so a
    mistakenly-passed RangeToken raises at the call site instead of silently producing a
    malformed record.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith

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


def _buffer_capacity_slots() -> int:
    """Record slots the buffer holds, excluding the reserved cursor slot."""
    from ...utils import env

    return max(int(env.ktrace.buffer_bytes) // _ktrace.RECORD_BYTES - 1, 1)


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
