# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import threading
from typing import Any

_I64 = "i64"

_NAMES: dict[str, int] = {}
_ID_OWNER: dict[int, str] = {}
_NAMES_LOCK = threading.RLock()


class EventNameConflict(RuntimeError):
    pass


def name_assignment_lock():
    return _NAMES_LOCK


def name_table() -> dict[str, int]:
    with _NAMES_LOCK:
        return dict(_NAMES)


def merge_names(names: dict[str, int]) -> None:
    with _NAMES_LOCK:
        _merge_names(names)


def _merge_names(names: dict[str, int]) -> None:
    for name, ident in names.items():
        previous = _NAMES.get(name)
        if previous is not None and previous != ident:
            raise EventNameConflict(
                f"ktrace event name {name!r} has ids {previous} and {ident} in live binaries. "
                "Records from the first binary would lose their name if this binding were "
                "replaced. Re-run with FLYDSL_RUNTIME_ENABLE_CACHE=0 to trace both from source."
            )
        owner = _ID_OWNER.get(ident)
        if owner is not None and owner != name:
            raise EventNameConflict(
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


def payload_operand(payload: Any):
    from ..._mlir import ir
    from ..._mlir.dialects import arith as _arith

    signed = getattr(payload, "signed", True)
    if isinstance(payload, bool):
        return _const(int(payload), _I64), "bool"
    if isinstance(payload, int):
        if not -(1 << 63) <= payload < (1 << 64):
            raise ValueError("ktrace integer payload must fit in 64 bits")
        return _const(payload, _I64), "int64" if payload < (1 << 63) else "uint64"
    if hasattr(payload, "ir_value"):
        value = payload.ir_value()
    elif isinstance(payload, float):
        from ..numeric import Float64

        value = Float64(payload).ir_value()
    else:
        value = payload

    if not isinstance(value, ir.Value):
        raise TypeError(f"ktrace payload must coerce to a single value, got {type(payload).__name__}")

    ty = str(value.type)
    i64 = ir.Type.parse(_I64)
    if ty == "index":
        return _arith.IndexCastOp(i64, value).result, "int64"
    if ty.startswith("i"):
        width = int(ty[1:])
        if width > 64:
            raise TypeError(f"ktrace integer payload must fit in 64 bits, got {ty}")
        if width == 1:
            return _arith.ExtUIOp(i64, value).result, "bool"
        extend = _arith.ExtSIOp if signed else _arith.ExtUIOp
        bits = value if width == 64 else extend(i64, value).result
        return bits, "int64" if signed else "uint64"
    if ty in ("f16", "bf16"):
        value = _arith.ExtFOp(ir.Type.parse("f32"), value).result
        ty = "f32"
    if ty in ("f32", "f64"):
        bits = _arith.BitcastOp(ir.Type.parse("i32" if ty == "f32" else _I64), value).result
        return (bits, "float64") if ty == "f64" else (_arith.ExtUIOp(i64, bits).result, "float32")
    raise TypeError(f"unsupported ktrace payload type {ty}")
