# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral trace annotation discovery and static schedule planning."""

from .._mlir import ir

MAX_EVENTS = 1 << 20
_OPTION_FIELDS = ("version", "mode", "blocks", "exclude", "max_events", "max_blocks", "hardware")


def option(options, name):
    """Read named fields while accepting older tuple-shaped cache entries."""

    if hasattr(options, name):
        return getattr(options, name)
    return dict(zip(_OPTION_FIELDS, options))[name]


def selected_blocks(ctx):
    """Resolve capture-level block selection against the current JIT scope."""

    override = option(ctx.trace_spec["options"], "blocks")
    return ctx.trace_blocks if override == "jit" else override


def _op(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else owner


def _integer(value, env):
    # FlyDSL integer wrappers overload == to emit IR. Compare raw SSA values on
    # the host while reconstructing a compile-time schedule.
    for key, number in env:
        if ir.Value.__eq__(value, key):
            return number
    op = _op(value)
    if not hasattr(op, "name"):
        raise ValueError("flytrace requires constant bounds and statically reconstructible integer payloads")
    if op.name == "arith.constant":
        return int(ir.IntegerAttr(op.attributes["value"]).value)
    args = list(op.operands)
    if op.name in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci"):
        result = _integer(args[0], env)
        if ir.IntegerType.isinstance(value.type):
            bits = ir.IntegerType(value.type).width
            result &= (1 << bits) - 1
            if op.name != "arith.extui" and result >= 1 << (bits - 1):
                result -= 1 << bits
        return result
    operations = {
        "arith.addi": lambda a, b: a + b,
        "arith.subi": lambda a, b: a - b,
        "arith.muli": lambda a, b: a * b,
    }
    if op.name in operations:
        return operations[op.name](*[_integer(arg, env) for arg in args])
    raise ValueError(f"flytrace cannot reconstruct payload from {op.name}; use constants or loop indices")


def _contains(block):
    return any(
        op.operation.name == "fly.trace_event" or any(_contains(child) for region in op.regions for child in region.blocks)
        for op in block.operations
    )


def _metadata(op):
    kind = ir.StringAttr(op.attributes["kind"]).value
    range_id = None
    if kind.startswith(("range_start:", "range_end:")):
        kind, encoded_id = kind.split(":", 1)
        range_id = int(encoded_id)
    metadata = {
        "name": ir.StringAttr(op.attributes["event_name"]).value,
        "kind": kind,
    }
    if range_id is not None:
        metadata["range_id"] = range_id
    return metadata


def static_layout(block):
    """Build a compact schedule for constant structured loops."""

    items, total = [], 0
    for view in list(block.operations):
        op = view.operation
        if op.name == "fly.trace_event":
            items.append(("event", op, total))
            total += 1
        elif op.name == "scf.for":
            child, count = static_layout(op.regions[0].blocks[0])
            if not count:
                continue
            lo, hi, step = (_integer(value, ()) for value in list(op.operands)[:3])
            if step <= 0:
                raise ValueError("flytrace requires positive constant loop steps")
            trips = max(0, (hi - lo + step - 1) // step)
            if total + trips * count > MAX_EVENTS:
                raise ValueError("flytrace static schedule exceeds 2**20 events per wave")
            items.append(("loop", op, total, lo, hi, step, child, count))
            total += trips * count
        elif any(_contains(child) for region in op.regions for child in region.blocks):
            raise ValueError(f"flytrace annotations inside {op.name} require the dynamic recorder")
    if total > MAX_EVENTS:
        raise ValueError("flytrace static schedule exceeds 2**20 events per wave")
    return items, total


def static_schema(items, env=()):
    """Expand a static layout into the event sequence used by the decoder."""

    result = []
    for item in items:
        if item[0] == "event":
            op = item[1]
            payload = _integer(op.operands[0], env) if len(op.operands) else None
            if payload is not None and not -(1 << 31) <= payload < (1 << 31):
                raise ValueError("flytrace reconstructed payload must fit signed int32")
            result.append({**_metadata(op), "payload": payload})
        else:
            _, op, _, lo, hi, step, child, _ = item
            induction = op.regions[0].blocks[0].arguments[0]
            for number in range(lo, hi, step):
                result.extend(static_schema(child, env + ((induction, number),)))
    return result


def dynamic_sites(block):
    """Return trace operations in stable source/region order."""

    result = []
    for view in list(block.operations):
        op = view.operation
        if op.name == "fly.trace_event":
            result.append(op)
        for region in op.regions:
            for child in region.blocks:
                result.extend(dynamic_sites(child))
    return result


def dynamic_schema(sites):
    return [
        {
            "site": site,
            **_metadata(op),
            "has_payload": bool(len(op.operands)),
        }
        for site, op in enumerate(sites)
    ]


def grid_metadata(grid):
    """Return static launch dimensions, or None for a runtime grid."""

    from ..expr.typing import as_ir_value

    values = tuple(as_ir_value(value, keep_static=True) for value in grid)
    return values if all(isinstance(value, int) for value in values) else None
