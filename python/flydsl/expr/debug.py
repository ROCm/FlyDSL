# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Opt-in device-side runtime checks."""

from .._mlir import ir
from .._mlir.dialects import arith, gpu, llvm, scf, vector
from ..utils import env
from .meta import dsl_loc_tracing
from .typing import as_ir_value

__all__ = ["device_assert"]


def device_assert(condition, message="", mask=None):
    """Assert a boolean condition on the device when debug assertions are enabled.

    Set ``FLYDSL_DEBUG_DEVICE_ASSERT=1`` before compiling or invoking the JIT.
    Disabled assertions emit no IR. Arguments are still evaluated by Python.

    ``condition`` must be a scalar bool/i1 or a fixed-size vector of i1 values.
    ``mask`` may be a scalar boolean or a boolean vector matching the condition;
    masked-off elements are ignored. Any failing element triggers one diagnostic
    per calling thread, with source and block/thread coordinates, followed by a
    device trap. The diagnostic is best effort and execution cannot be resumed.

    This checks values owned by one thread; it does not synchronize threads or
    reduce across a wave. Python ``assert`` retains its existing semantics.

    Example::

        fx.device_assert(index < length, "index exceeds input length", mask=active)
    """
    if not isinstance(message, str):
        raise TypeError("device_assert message must be a compile-time string")
    if "\0" in message:
        raise ValueError("device_assert message must not contain NUL characters")
    if not env.debug.device_assert:
        return
    _device_assert(condition, message, mask)


def _boolean_value(value, name):
    value = as_ir_value(value)
    if not isinstance(value, ir.Value):
        raise TypeError(f"device_assert {name} must be a boolean scalar or vector")
    ty = value.type
    elem_ty = ir.VectorType(ty).element_type if isinstance(ty, ir.VectorType) else ty
    if not isinstance(elem_ty, ir.IntegerType) or ir.IntegerType(elem_ty).width != 1:
        raise TypeError(f"device_assert {name} must have boolean (i1) elements, got {ty}")
    if isinstance(ty, ir.VectorType) and ir.VectorType(ty).scalable:
        raise TypeError("device_assert does not support scalable vectors")
    return value


@dsl_loc_tracing
def _device_assert(condition, message, mask):
    # Device intrinsics and traps must not accidentally be emitted in a launcher.
    parent = ir.InsertionPoint.current.block.owner
    while parent is not None and parent.operation.name != "gpu.func":
        parent = parent.parent
    if parent is None:
        raise ValueError("device_assert must be called inside a GPU kernel")

    cond = _boolean_value(condition, "condition")
    if mask is not None:
        active = _boolean_value(mask, "mask")
        if active.type != cond.type:
            if isinstance(cond.type, ir.VectorType) and active.type == ir.IntegerType.get_signless(1):
                active = vector.BroadcastOp(cond.type, active).result
            else:
                raise TypeError("device_assert mask must be scalar or match the condition's vector shape")
        # Inactive elements satisfy the assertion, including an entirely inactive thread.
        one = arith.ConstantOp(
            ir.IntegerType.get_signless(1), ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)
        ).result
        if isinstance(cond.type, ir.VectorType):
            one = vector.BroadcastOp(cond.type, one).result
        cond = arith.OrIOp(cond, arith.XOrIOp(active, one)).result
    if isinstance(cond.type, ir.VectorType):
        cond = vector.ReductionOp(ir.IntegerType.get_signless(1), vector.CombiningKind.AND, cond).result

    one = arith.ConstantOp(ir.IntegerType.get_signless(1), ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)).result
    failed = arith.XOrIOp(cond, one).result
    location = ir.Location.current
    while isinstance(location, (ir.CallSiteLoc, ir.NameLoc)):
        location = location.callee if isinstance(location, ir.CallSiteLoc) else location.child_loc
    source = f"{location.filename}:{location.start_line}" if isinstance(location, ir.FileLineColLoc) else "<unknown>"
    # Escape user text for printf; it must never be interpreted as a format string.
    prefix = f"FlyDSL device assertion failed at {source}: {message}".replace("%", "%%")
    fail_op = scf.IfOp(failed, [], has_else=False)
    with ir.InsertionPoint(fail_op.then_block):
        coords = [gpu.block_id(axis) for axis in ("x", "y", "z")]
        coords += [gpu.thread_id(axis) for axis in ("x", "y", "z")]
        coords = [arith.IndexCastOp(ir.IntegerType.get_signless(64), coord).result for coord in coords]
        gpu.PrintfOp(prefix + " [block=(%lld,%lld,%lld), thread=(%lld,%lld,%lld)]\n", coords)
        llvm.Trap()
        scf.YieldOp([])
