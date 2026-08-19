# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Helpers shared by every cooperative algorithm, at any scope.

Anything here is target-neutral and never dispatched: it is plain glue that
warp- and block-scope algorithms both need. Algorithm logic belongs in the
scope subpackages, not here.

Every algorithm passes *op* straight through to these three, so the shape of a
custom op is not baked into any signature above this file.
"""

from ...expr.arith import max as _max
from ...expr.arith import min as _min
from ...expr.gpu import num_warp_threads, thread_idx
from ...expr.typing import ReductionOp, Vector


def require_power_of_two(value, what):
    if not isinstance(value, int) or value < 1 or (value & (value - 1)):
        raise ValueError(f"{what} must be a power of two, got {value!r}")


def resolve_warp_width(width, what):
    if width is None:
        width = num_warp_threads()
    require_power_of_two(width, what)
    return width


def combine(op, lhs, rhs):
    if op is ReductionOp.ADD:
        return lhs + rhs
    if op is ReductionOp.MUL:
        return lhs * rhs
    if op is ReductionOp.MAX:
        return _max(lhs, rhs)
    if op is ReductionOp.MIN:
        return _min(lhs, rhs)
    # TODO: support more binary reduction ops or lambda fns.
    raise TypeError(f"unsupported ReductionOp, got {op!r}")


def _representable_extreme(dtype, lowest):
    """The lowest or highest value *dtype* can hold.

    Floats use an infinity rather than the finite lowest/highest: a finite bound
    is itself a legal input, so a lane holding it would be indistinguishable
    from an empty one, and an infinity is exactly what ``fx.max`` / ``fx.min``
    leave unchanged against anything else.
    """
    if dtype.is_float:
        return dtype(float("-inf") if lowest else float("inf"))
    if dtype.signed:
        half = 1 << (dtype.width - 1)
        return dtype(-half if lowest else half - 1)
    return dtype(0 if lowest else (1 << dtype.width) - 1)


def identity(op, dtype):
    """The value that leaves *op* unchanged: ``identity(op, t) ⊕ x == x``.

    An exclusive scan needs it for the first thread, which has nothing in front
    of it.
    """
    if op is ReductionOp.ADD:
        return dtype(0)
    if op is ReductionOp.MUL:
        return dtype(1)
    if op is ReductionOp.MAX:
        return _representable_extreme(dtype, lowest=True)
    if op is ReductionOp.MIN:
        return _representable_extreme(dtype, lowest=False)
    # TODO: support more binary reduction ops
    raise TypeError(f"unsupported ReductionOp, got {op!r}")


def seed(value, op, init):
    """Fold *init* into a scan result; ``None`` leaves it alone.

    A scan puts :func:`identity` in front of the first thread, and
    ``init ⊕ identity == init``, so seeding the inclusive and the exclusive
    form is the same one operation applied to both.
    """
    return value if init is None else combine(op, init, value)


def thread_partial(value, op):
    """Fold a per-thread ``Vector`` down to one scalar; pass a scalar through."""
    return value.reduce(op) if isinstance(value, Vector) else value


def linear_thread_id(block_size):
    """Linear thread index within the block, matching ``gpu.thread_id`` ordering."""
    tid = thread_idx.x
    if block_size is None:
        return tid
    dim_x, dim_y, dim_z = block_size
    if dim_y > 1 or dim_z > 1:
        tid = tid + thread_idx.y * dim_x + thread_idx.z * (dim_x * dim_y)
    return tid
