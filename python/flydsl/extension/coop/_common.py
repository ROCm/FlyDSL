# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Helpers shared by every cooperative algorithm, at any scope.

Anything here is target-neutral and never dispatched: it is plain glue that
warp- and block-scope algorithms both need. Algorithm logic belongs in the
scope subpackages, not here.

Operators, identities and recursive record movement share one protocol here.
"""

from ...expr.arith import max as _max
from ...expr.arith import min as _min
from ...expr.gpu import lane_id, num_warp_threads, shuffle, thread_idx
from ...expr.typing import Boolean, Numeric, ReductionOp, Vector
from ._values import (
    _as_items,
    _from_items,
    _is_items,
    _item_dtype,
    _items_dtype,
    _normalize_value,
    _record_cast,
    _record_default,
    _record_select,
    _record_shuffle,
)


def _cast_value(dtype, value):
    """Convert a scalar seed, preserving an already constructed record."""
    if isinstance(value, dtype):
        return value
    if isinstance(value, Vector) and issubclass(dtype, Vector):
        return _record_cast(value, dtype)
    if issubclass(dtype, Numeric):
        return dtype(value)
    return dtype(**value) if isinstance(value, dict) else _record_default(dtype, value)


def _convert_value(value, dtype):
    """Apply a specialization's element type to a scalar or blocked item range."""
    value = _normalize_value(value)
    if _is_items(value, dtype):
        return _from_items([_convert_item(value[i], dtype) for i in range(len(value))], dtype=dtype, like=value)
    return _convert_item(value, dtype)


def _convert_item(value, dtype):
    """Convert one declared element without splatting or reshaping its data."""
    value = _normalize_value(value)
    if issubclass(dtype, Numeric):
        if isinstance(value, (int, float, bool, Numeric)):
            return dtype(value)
        raise TypeError(f"expected a scalar {dtype.__name__} item, got {type(value).__name__}")
    if issubclass(dtype, Vector) and not isinstance(value, Vector):
        raise TypeError(f"expected one complete {dtype.__name__} Vector item, got {type(value).__name__}")
    return _record_cast(value, dtype)


def _normalize_columns(value):
    """Use the same numeric Vector or Struct tuple for both warp API forms."""
    return _from_items(value) if isinstance(value, (tuple, list)) else _normalize_value(value)


def _cast_seed(value, init):
    """Convert a scalar or per-column seed to the input item type."""
    if isinstance(value, (tuple, list)):
        dtype = _items_dtype(value)
        if _is_items(init):
            seeds = _as_items(init)
            if len(seeds) != len(value):
                raise ValueError("scan seed must have the same number of items as the input")
            return _from_items([_cast_value(dtype, item) for item in seeds], like=value)
        return _cast_value(dtype, init)
    return _cast_value(_item_dtype(value), _normalize_columns(init))


def _select_value(condition, lhs, rhs):
    if isinstance(lhs, Vector) and isinstance(rhs, Numeric):
        rhs = Vector.filled_like(lhs, rhs)
    if isinstance(lhs, Numeric):
        return condition.select(lhs, rhs)
    return _record_select(condition, lhs, rhs)


def _shuffle_value(value, offset, width, mode="idx"):
    """Keep numeric shuffle instructions; move records field by field."""
    # Keep packed numeric vectors intact (e.g. two f16 values per shuffle).
    if isinstance(value, (Numeric, Vector)) and value.dtype.width not in (1, 128):
        return shuffle(value, offset, width, mode=mode)
    source = offset
    if mode == "up":
        source = lane_id() - offset
    elif mode == "down":
        source = lane_id() + offset
    elif mode == "xor":
        source = lane_id() ^ offset
    return _record_shuffle(value, source, width)


def _require_power_of_two(value, what):
    if not isinstance(value, int) or value < 1 or (value & (value - 1)):
        raise ValueError(f"{what} must be a power of two, got {value!r}")


def _resolve_warp_width(width, what):
    warp_threads = num_warp_threads()
    if width is None:
        return warp_threads
    _require_power_of_two(width, what)
    if width > warp_threads:
        raise ValueError(f"{what} must not exceed the target's {warp_threads}-lane warp, got {width}")
    return width


def _combine(op, lhs, rhs):
    if isinstance(lhs, (tuple, list)) or isinstance(rhs, (tuple, list)):
        template = lhs if _is_items(lhs) else rhs
        left = _as_items(lhs) if _is_items(lhs) else (lhs,) * len(template)
        right = _as_items(rhs) if _is_items(rhs) else (rhs,) * len(template)
        if len(left) != len(right):
            raise ValueError("combining tiles requires the same shape")
        return _from_items([_combine(op, a, b) for a, b in zip(left, right)], like=template)
    # Boolean reductions preserve their input type: addition/maximum implement
    # any, and multiplication/minimum implement all, without integer widening.
    if isinstance(lhs, Boolean) and isinstance(rhs, Boolean):
        if op in (ReductionOp.ADD, ReductionOp.MAX):
            return lhs | rhs
        if op in (ReductionOp.MUL, ReductionOp.MIN):
            return lhs & rhs
    if op is ReductionOp.ADD:
        return lhs + rhs
    if op is ReductionOp.MUL:
        return lhs * rhs
    if op is ReductionOp.MAX:
        return _max(lhs, rhs)
    if op is ReductionOp.MIN:
        return _min(lhs, rhs)
    if callable(op):
        result = _normalize_value(op(lhs, rhs))
        if _item_dtype(result) is not _item_dtype(lhs):
            raise TypeError("a reduction/scan operator must return the complete input element type")
        return result
    raise TypeError(f"expected ReductionOp or a binary callable, got {op!r}")


def _is_commutative(op):
    """Custom operators opt in to algorithms that reorder operands."""
    return isinstance(op, ReductionOp) or getattr(op, "commutative", False) is True


def _require_commutative(op, algorithm):
    if not _is_commutative(op):
        raise ValueError(f"{algorithm} requires a commutative operator; declare op.commutative = True")


def _representable_extreme(dtype, lowest):
    """The lowest or highest value *dtype* can hold.

    Floats use an infinity rather than the finite lowest/highest: a finite bound
    is itself a legal input, so a lane holding it would be indistinguishable
    from an empty one, and an infinity is exactly what ``fx.max`` / ``fx.min``
    leave unchanged against anything else.
    """
    if issubclass(dtype, Vector):
        return dtype(_representable_extreme(dtype(0).dtype, lowest))
    if dtype.is_float:
        return dtype(float("-inf") if lowest else float("inf"))
    if dtype.width == 1:
        return dtype(not lowest)
    if dtype.signed:
        half = 1 << (dtype.width - 1)
        return dtype(-half if lowest else half - 1)
    return dtype(0 if lowest else (1 << dtype.width) - 1)


def _identity(op, dtype, explicit=None):
    """The value that leaves *op* unchanged: ``_identity(op, t) ⊕ x == x``.

    Exclusive scans use it for a defined first output. Custom operations may
    instead omit it, in which case the first unseeded output is unspecified.
    """
    if explicit is not None:
        return _cast_value(dtype, explicit(dtype) if callable(explicit) else explicit)
    if op is ReductionOp.ADD:
        return dtype(0)
    if op is ReductionOp.MUL:
        return dtype(1)
    if op is ReductionOp.MAX:
        return _representable_extreme(dtype, lowest=True)
    if op is ReductionOp.MIN:
        return _representable_extreme(dtype, lowest=False)
    neutral = getattr(op, "identity", None)
    if neutral is not None:
        return _cast_value(dtype, neutral(dtype) if callable(neutral) else neutral)
    raise TypeError("this operation needs an identity: pass identity= or define op.identity(dtype)")


def _optional_identity(op, dtype, explicit=None):
    """Return None for a semigroup (a callable without an identity)."""
    if explicit is not None or isinstance(op, ReductionOp) or getattr(op, "identity", None) is not None:
        return _identity(op, dtype, explicit)
    return None


def _validate_valid_items(valid_items, size):
    if isinstance(valid_items, int) and not 0 <= valid_items <= size:
        raise ValueError(f"valid_items must be between 0 and {size}, got {valid_items}")


def _validate_scalar_valid_items(value, valid_items, size, dtype=None):
    """Reduce/scan guarded overloads accept one element per thread."""
    if valid_items is not None and _is_items(value, dtype):
        raise TypeError("valid_items is supported only for a single item per thread; omit it for an item range")
    _validate_valid_items(valid_items, size)


def _seed(value, op, init):
    """Fold *init* into a scan result; ``None`` leaves it alone.

    A scan puts :func:`identity` in front of the first thread, and
    ``init ⊕ identity == init``, so seeding the inclusive and the exclusive
    form is the same one operation applied to both.
    """
    return value if init is None else _combine(op, _cast_seed(value, init), value)


def _thread_partial(value, op, dtype=None):
    """Fold a per-thread Numeric/record item range, preserving item order."""
    if not _is_items(value, dtype):
        return _normalize_value(value)
    if isinstance(value, Vector) and value.dtype is not Boolean and isinstance(op, ReductionOp):
        # Preserve the native vector reduction tree for existing numeric tiles.
        return value.reduce(op)
    items = _as_items(value, dtype)
    partial = items[0]
    for item in items[1:]:
        partial = _combine(op, partial, item)
    return partial


def _linear_thread_id(block_size):
    """Linear thread index within the block, matching ``gpu.thread_id`` ordering."""
    tid = thread_idx.x
    if block_size is None:
        return tid
    dim_x, dim_y, dim_z = block_size
    if dim_y > 1 or dim_z > 1:
        tid = tid + thread_idx.y * dim_x + thread_idx.z * (dim_x * dim_y)
    return tid


# Retain the helper spellings used by the existing block implementations.
require_power_of_two = _require_power_of_two
resolve_warp_width = _resolve_warp_width
combine = _combine
identity = _identity
seed = _seed
thread_partial = _thread_partial
linear_thread_id = _linear_thread_id
