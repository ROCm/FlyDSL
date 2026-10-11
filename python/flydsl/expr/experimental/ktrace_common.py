# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Target-independent event names and payload operands for trace annotations."""

from __future__ import annotations

from typing import Any

_I64 = "i64"

# A record carries an event id rather than its name. The ids must remain valid
# across compiled kernels and disk-cache hits in one process.
_NAMES: dict[str, int] = {}
_ID_OWNER: dict[int, str] = {}


def name_table() -> dict[str, int]:
    """Return every event name bound to a live compiled kernel."""
    return dict(_NAMES)


def merge_names(names: dict[str, int]) -> None:
    """Adopt ids from a compiled artifact without changing existing bindings."""
    # Validate the whole table first: a failed merge must not publish half of it.
    for name, ident in names.items():
        previous = _NAMES.get(name)
        if previous is not None and previous != ident:
            raise RuntimeError(
                f"ktrace event name {name!r} has ids {previous} and {ident} in live binaries. "
                "Records from the first binary would lose their name if this binding were "
                "replaced. Re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0 to trace both from source."
            )
        owner = _ID_OWNER.get(ident)
        if owner is not None and owner != name:
            raise RuntimeError(
                f"ktrace event id {ident} is claimed by both {owner!r} and {name!r}. "
                "Two kernels compiled with different name tables are live in this process "
                "(typically one traced fresh and one served from the disk cache), and "
                "their records can no longer be told apart. Re-run with "
                "FLYDSL_RUNTIME_ENABLE_CACHE=0 to trace both from source."
            )

    for name, ident in names.items():
        _NAMES[name] = ident
        _ID_OWNER[ident] = name


def _const(value: int, ty_str: str):
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    ty = ir.Type.parse(ty_str)
    return _llvm.ConstantOp(ty, ir.IntegerAttr.get(ty, value)).result


def payload_value(payload: Any):
    """Coerce a supplied scalar payload to one i64 annotation operand."""
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
    if ty == "index":
        return _arith.IndexCastOp(i64, value).result
    if ty.startswith("i"):
        width = int(ty[1:])
        return value if width == 64 else _arith.ExtUIOp(i64, value).result
    if ty in ("f32", "f64"):
        bits = _arith.BitcastOp(ir.Type.parse("i32" if ty == "f32" else _I64), value).result
        return bits if ty == "f64" else _arith.ExtUIOp(i64, bits).result
    raise TypeError(f"unsupported ktrace payload type {ty}")
