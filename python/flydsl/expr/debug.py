# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Opt-in device-side runtime checks."""

from .._mlir import ir
from .._mlir.dialects import arith, gpu, llvm, scf, vector
from ..utils import env
from .meta import dsl_loc_tracing
from .typing import as_ir_value

__all__ = ["device_assert"]


def device_assert(condition, message=""):
    """Assert a boolean condition on the device when debug assertions are enabled.

    Set ``FLYDSL_DEBUG_DEVICE_ASSERT=1`` before compiling or invoking the JIT.
    Disabled assertions emit no IR. Arguments are still evaluated by Python.

    ``condition`` must be a scalar bool/i1 or a fixed-size vector of i1 values.
    Any failing element triggers one diagnostic per calling thread, with source
    and block/thread coordinates, followed by a device trap. The diagnostic is
    best effort and execution cannot be resumed.

    This checks values owned by one thread; it does not synchronize threads or
    reduce across a wave. Python ``assert`` retains its existing semantics.

    Example::

        fx.device_assert(index < length, "index exceeds input length")
    """
    if not env.debug.device_assert:
        return
    _device_assert(condition, message)


def _boolean_value(value, name):
    # Unwraps a FlyDSL value and accepts Scalar i1 or fixed-size Vector<i1>. Raises TypeError otherwise.
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
def _device_assert(condition, message):
    from ..compiler.diagnostics import location_chain

    parent = ir.InsertionPoint.current.block.owner
    # Find the nearest enclosing gpu.func so we don't emit device intrinsics in a host launcher.
    while parent is not None and parent.operation.name != "gpu.func":
        parent = parent.parent
    if parent is None:
        raise ValueError("device_assert must be called inside a GPU kernel")

    cond = _boolean_value(condition, "condition")
    bool_type = ir.IntegerType.get_signless(1)
    true = arith.ConstantOp(bool_type, ir.IntegerAttr.get(bool_type, 1)).result
    if isinstance(cond.type, ir.VectorType):
        true = vector.BroadcastOp(cond.type, true).result
    failed = arith.XOrIOp(cond, true).result
    if isinstance(cond.type, ir.VectorType):
        failed = vector.ReductionOp(bool_type, vector.CombiningKind.OR, failed).result

    frames = location_chain(ir.Location.current)
    source = f"{frames[0].filename}:{frames[0].line}" if frames else "<unknown>"
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
