# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Head and tail flags for adjacent runs in a blocked sequence."""

import inspect
from functools import partial

from ....compiler import jit
from ....compiler.jit_function import JitFunction
from ....expr.numeric import Int32, Integer, Numeric
from ....expr.primitive import ptrtoint
from ....expr.typing import Pointer, ReductionOp, Vector
from .._values import _flatten_record, _from_items, _is_items, _normalize_value, _record_default, _record_select
from .shuffle import BlockShuffle

__all__ = [
    "BlockDiscontinuity",
]


def _different(left, right):
    """Compare complete elements: any Vector component or Struct field may differ."""
    if isinstance(left, Numeric):
        return left != right
    if isinstance(left, Pointer):
        return ptrtoint(left) != ptrtoint(right)
    if isinstance(left, Vector):
        return (left != right).to(Int32).reduce(ReductionOp.MAX)
    result = Int32(0)
    for a, b in zip(_flatten_record(left), _flatten_record(right)):
        result = result | Int32(_different(a, b))
    return result


def _signature_target(op):
    """Expose the Python signature while retaining partial and method bindings."""
    if isinstance(op, JitFunction):
        return op.func
    if isinstance(op, partial):
        return partial(_signature_target(op.func), *op.args, **op.keywords)
    if inspect.ismethod(op) and isinstance(op.__self__, JitFunction) and op.__func__ is JitFunction.__call__:
        return op.__self__.func
    if isinstance(getattr(type(op), "__call__", None), JitFunction):
        return _signature_target(op.__call__)
    return op


def _indexed(op):
    # Resolve once before generating calls. User exceptions never participate
    # in overload selection and must propagate without retrying the predicate.
    try:
        signature = inspect.signature(_signature_target(op))
    except (TypeError, ValueError) as exc:
        raise TypeError("flag_op must expose a signature accepting (left, right[, right_index])") from exc
    try:
        signature.bind(None, None, None)
        return True
    except TypeError:
        try:
            signature.bind(None, None)
        except TypeError as exc:
            raise TypeError("flag_op must accept (left, right) or (left, right, right_index)") from exc
    return False


def _apply(op, left, right, index, indexed):
    result = _normalize_value(op(left, right, index) if indexed else op(left, right))
    if not isinstance(result, (bool, int, Integer)):
        raise TypeError("flag_op must return one scalar predicate per complete element")
    return Int32(result)


@jit
def _apply_boundary(active, op, left, right, index, indexed):
    result = Int32(1)
    if active:
        result = _apply(op, left, right, index, indexed)
    return result


class BlockDiscontinuity(BlockShuffle):
    """Flag run boundaries using ``flag_op(left, right)`` (default inequality).

    Specialize with ``BlockDiscontinuity[dtype, block_size, algorithm]``. The parameters
    below are compile-time positional arguments to ``BlockDiscontinuity[...]``, in bracket
    order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. This is the predicate
            operand type; output flags are Int32.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        algorithm: Optional member of BlockShuffleAlgorithm selecting the implementation
            policy. Defaults to ``BlockShuffleAlgorithm.SHARED_MEMORY`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    The default predicate compares complete elements: Vector and Struct elements
    differ when any component or field differs.
    Flags are Int32, including for floating-point inputs. Unspecified tile
    boundaries always produce a flag without evaluating the predicate on the device.
    ``flag_heads_and_tails`` shares one
    publication and one barrier between both results.
    All threads participate with equal
    per-thread item counts in blocked order. Synchronize before reusing storage.
    A predicate may accept a third argument: the flattened index of its right
    operand, including ``tile_size`` for a supplied tile successor.
    Plain functions, JIT functions, bound methods, callable objects, and partial
    applications use their effective Python signature. If both two and three
    positional arguments bind, the three-argument form is selected. The
    signature must be inspectable and accept one of these forms; otherwise
    compilation fails before generating predicate calls. The original callable
    is invoked without trial calls or exception-based retries.

    Attributes:
        dtype: Specialized value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        algorithm: Selected ``BlockShuffleAlgorithm`` storage policy.
        SharedStorage: Shared-memory struct type inherited from ``BlockShuffle``,
            with ``first`` and ``last`` arrays of ``block_threads`` items each.

    Examples:
        # T0..T3 supply x={[2,2], [2,5], [5,7], [7,7]}; T4..T63 supply [7,7].
        P = fx.coop.BlockDiscontinuity[fx.Int32, 64, fx.coop.BlockShuffleAlgorithm.SHARED_MEMORY]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        heads, tails = P.flag_heads_and_tails(x, storage=storage)
        # Flattened heads are 1 at indices 0,3,5; tails are 1 at indices 2,4,127.
        # Every other flag is zero.

        different = lambda left, right: left != right
        fx.barrier()
        heads, tails, previous = P.flag_heads_and_tails(
            x, different, storage=storage, predecessor=2, successor=7, return_predecessors=True
        )
        # Boundary values continue the runs: only heads 3,5 and tails 2,4 remain set.
        # T0..T3 previous: {[2,2], [2,2], [5,5], [7,7]}; T4..T63: [7,7].
        fx.barrier()
        heads, previous = P.flag_heads(
            x, different, storage=storage, predecessor=2, return_predecessors=True
        )
        fx.barrier()
        tails = P.flag_tails(x, different, storage=storage, successor=7)
        # Separate calls give the same flags and predecessors as the combined call.
    """

    @classmethod
    def _flags(cls, value, flag_op, storage, predecessor, successor, heads, tails, return_predecessors=False):
        indexed = _indexed(flag_op)
        items, tid, left, right = cls._neighbors(value, storage)
        if predecessor is not None:
            left = _record_select(tid == 0, _record_default(cls.dtype, predecessor), left)
        if successor is not None:
            right = _record_select(tid == cls.block_threads - 1, _record_default(cls.dtype, successor), right)
        head_result = None
        tail_result = None
        if heads:
            if predecessor is None:
                first = _apply_boundary(tid > 0, flag_op, left, items[0], tid * len(items), indexed)
            else:
                first = _apply(flag_op, left, items[0], tid * len(items), indexed)
            out = [
                first,
                *[_apply(flag_op, items[i - 1], items[i], tid * len(items) + i, indexed) for i in range(1, len(items))],
            ]
            head_result = Vector.from_elements(out) if _is_items(value, cls.dtype) else out[0]
        if tails:
            if successor is None:
                last = _apply_boundary(
                    tid < cls.block_threads - 1, flag_op, items[-1], right, (tid + 1) * len(items), indexed
                )
            else:
                last = _apply(flag_op, items[-1], right, (tid + 1) * len(items), indexed)
            out = [
                *[
                    _apply(flag_op, items[i], items[i + 1], tid * len(items) + i + 1, indexed)
                    for i in range(len(items) - 1)
                ],
                last,
            ]
            tail_result = Vector.from_elements(out) if _is_items(value, cls.dtype) else out[0]
        if return_predecessors:
            predecessors = [left, *items[:-1]]
            predecessors = _from_items(predecessors, like=value) if _is_items(value, cls.dtype) else predecessors[0]
            return head_result, tail_result, predecessors
        return head_result, tail_result

    @classmethod
    def flag_heads(
        cls,
        value,
        flag_op=_different,
        *,
        storage,
        predecessor=None,
        return_predecessors: bool = False,
    ):
        """Flag items that start a run relative to their predecessors.

        Args:
            value: Per-thread input value with same-type items in blocked order.
            flag_op: Callable ``flag_op(left, right)`` or
                ``flag_op(left, right, right_index)``; defaults to inequality.
                May be JIT-decorated. Returns one scalar Boolean or integer flag
                per complete element. Three arguments are preferred when both
                forms bind; exceptions from the predicate propagate unchanged.
            storage: Shared instance of ``SharedStorage`` used by the block.
            predecessor: Item preceding the tile. If omitted, its first head
                flag is always one.
            return_predecessors: Whether to also return the predecessor of each
                input item. Without an explicit predecessor, the first item's
                returned predecessor is the block's last input item.

        Returns:
            One Int32 flag for a single T input, or an Int32 Vector of item flags
            for an input sequence. Vector components inside T do not add flags.
            If ``return_predecessors`` is true, returns
            ``(head_flags, predecessors)``; predecessors retain the input dtype.
        """
        result = cls._flags(value, flag_op, storage, predecessor, None, True, False, return_predecessors)
        return (result[0], result[2]) if return_predecessors else result[0]

    @classmethod
    def flag_tails(
        cls,
        value,
        flag_op=_different,
        *,
        storage,
        successor=None,
    ):
        """Flag items that end a run relative to their successors.

        Args:
            value: Per-thread input value with same-type items in blocked order.
            flag_op: Callable ``flag_op(left, right)`` or
                ``flag_op(left, right, right_index)``; defaults to inequality.
                May be JIT-decorated. Returns one scalar Boolean or integer flag
                per complete element; three arguments are preferred when both bind.
            storage: Shared instance of ``SharedStorage`` used by the block.
            successor: Item following the tile. If omitted, its last tail flag
                is always one.

        Returns:
            One Int32 flag for a single T input, or an Int32 Vector with one
            flag per input item. Vector components inside T do not add flags.
        """
        return cls._flags(value, flag_op, storage, None, successor, False, True)[1]

    @classmethod
    def flag_heads_and_tails(
        cls,
        value,
        flag_op=_different,
        *,
        storage,
        predecessor=None,
        successor=None,
        return_predecessors: bool = False,
    ):
        """Compute both run boundaries using one shared-memory publication.

        Args:
            value: Per-thread input value with same-type items in blocked order.
            flag_op: Callable ``flag_op(left, right)`` or
                ``flag_op(left, right, right_index)``; defaults to inequality.
                May be JIT-decorated. Returns one scalar Boolean or integer flag
                per complete element; three arguments are preferred when both bind.
            storage: Shared instance of ``SharedStorage`` used by the block.
            predecessor: Item preceding the tile, or ``None`` to force the first
                head flag to one.
            successor: Item following the tile, or ``None`` to force the last
                tail flag to one.
            return_predecessors: Whether to append each item's predecessor to
                the result tuple. An unspecified first predecessor wraps to the
                block's last input item.

        Returns:
            ``(head_flags, tail_flags)`` or, when requested,
            ``(head_flags, tail_flags, predecessors)``. Flags are ``Int32``
            scalars for a single T or Vectors for item sequences; predecessors
            preserve complete input elements and their sequence representation.
        """
        return cls._flags(value, flag_op, storage, predecessor, successor, True, True, return_predecessors)
