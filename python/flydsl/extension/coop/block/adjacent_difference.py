# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Differences between adjacent items of a blocked sequence."""

from ....compiler import jit
from ....expr.numeric import Integer
from .._common import _combine, _validate_valid_items
from .._values import _from_items, _is_items, _normalize_value, _record_cast, _record_default, _record_select
from .shuffle import BlockShuffle

__all__ = [
    "BlockAdjacentDifference",
]


def _subtract(current, neighbor):
    return current - neighbor


@jit
def _apply_if(active, op, current, neighbor, output_dtype):
    result = _record_cast(current, output_dtype)
    if active:
        # Difference results may have a different type from their operands.
        # Keep the reduction/scan type contract in _combine unchanged.
        difference = _normalize_value(op(current, neighbor)) if callable(op) else _combine(op, current, neighbor)
        result = _record_cast(difference, output_dtype)
    return result


class BlockAdjacentDifference(BlockShuffle):
    """Compute left/right differences, with optional tile boundary values.

    Specialize with ``BlockAdjacentDifference[dtype, block_size, algorithm]``. The
    parameters below are compile-time positional arguments to
    ``BlockAdjacentDifference[...]``, in bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. This is the operand type
            T; output_dtype is selected per call.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        algorithm: Optional member of BlockShuffleAlgorithm selecting the implementation
            policy. Defaults to ``BlockShuffleAlgorithm.SHARED_MEMORY`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    ``op(current, neighbor)`` defaults to subtraction. Without an explicit
    predecessor/successor, the first/last output is the corresponding input
    converted to the output type. ``output_dtype`` selects one complete output
    element type U and defaults to the input type T. Both operands and supplied
    boundary values remain T; operation results and passthrough inputs convert
    to U using native element conversion. Vector element shapes must match;
    unsupported conversions fail during compilation. The operator is not
    invoked at runtime for missing neighbors or invalid items, and is never
    called speculatively to infer U. A difference need not be associative.
    Items use blocked ownership.
    All block threads participate with equal per-thread item counts. Each call
    publishes boundary items and synchronizes; synchronize again before
    reusing its storage.

    Attributes:
        dtype: Specialized value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        algorithm: Selected ``BlockShuffleAlgorithm`` storage policy.
        SharedStorage: Shared-memory struct type inherited from ``BlockShuffle``,
            with ``first`` and ``last`` arrays of ``block_threads`` items each.

    Examples:
        # T0..T3 supply x={[2,5], [9,10], [15,21], [22,30]}; T4..T63 supply [30,30].
        P = fx.coop.BlockAdjacentDifference[fx.Int32, 64, fx.coop.BlockShuffleAlgorithm.SHARED_MEMORY]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y = P.subtract_left(x, storage=storage)
        # T0..T3 receive {[2,3], [4,1], [5,6], [1,8]}; all remaining outputs are zero.

        # Only the first six items participate in these differences; others pass through.
        difference = lambda current, neighbor: current - neighbor
        fx.barrier()
        left = P.subtract_left(x, difference, storage=storage, predecessor=1, valid_items=6)
        fx.barrier()
        right = P.subtract_right(x, difference, storage=storage, successor=25, valid_items=6)
        # T0..T3 left: {[1,3], [4,1], [5,6], [22,30]}.
        # T0..T3 right: {[-3,-4], [-1,-5], [-6,-4], [22,30]}.
        # T4..T63 retain [30,30] in both outputs; successor applies at valid item 5.
        fx.barrier()
        left = P.subtract_left_partial_tile(x, 6, difference, storage=storage, predecessor=1)
        fx.barrier()
        right = P.subtract_right_partial_tile(x, 6, difference, storage=storage, successor=25)
        # The named partial-tile methods produce the same results.

        fx.barrier()
        half = P.subtract_left(
            x, lambda a, b: (a.to(fx.Float32) - b.to(fx.Float32)) * 0.5,
            storage=storage, output_dtype=fx.Float32,
        )
        # T0 receives [2.0, 1.5]: the first input passes through as Float32.
    """

    @classmethod
    def subtract_left(
        cls,
        value,
        op=_subtract,
        *,
        storage,
        predecessor=None,
        valid_items: int | Integer | None = None,
        output_dtype: type | None = None,
    ):
        """Apply an operation to each item and its left neighbor.

        Args:
            value: Input value of the specialized dtype, owned in blocked order.
            op: Callable ``op(current, left)`` on two complete input elements;
                defaults to subtraction. Its result must convert to the output
                element type. Native ``ReductionOp`` values are also accepted.
            storage: Shared instance of ``SharedStorage`` used by the block.
            predecessor: Item preceding the tile, or ``None`` to leave the
                tile's first output equal to its converted input.
            valid_items: Number of valid items in the flattened tile, or ``None``
                for the full tile. Inputs outside this prefix are returned
                after conversion to the output dtype; all threads still participate.
                The count is uniform and lies in ``[0, block_threads * item_count]``;
                runtime bounds are the caller's responsibility.
            output_dtype: Compile-time output element type U, or ``None`` to use
                the specialized input type T. Does not change the operands or
                shared storage. No implicit element splitting or reshaping occurs.

        Returns:
            One U for a single T input, or the same number of U items for an
            input sequence. Scalar sequences return a Vector; Vector/Struct
            sequences return a tuple. An input SoV is retained only when its
            element type is unchanged. Invalid items do not invoke ``op``.
        """
        items, tid, left, _ = cls._neighbors(value, storage)
        _validate_valid_items(valid_items, cls.block_threads * len(items))
        limit = cls.block_threads * len(items) if valid_items is None else valid_items
        if predecessor is not None:
            left = _record_select(tid == 0, _record_default(cls.dtype, predecessor), left)
        out = []
        for i, item in enumerate(items):
            index = tid * len(items) + i
            active = index < limit
            if predecessor is None:
                active = active & (index > 0)
            out.append(
                _apply_if(
                    active,
                    op,
                    item,
                    left if i == 0 else items[i - 1],
                    cls.dtype if output_dtype is None else output_dtype,
                )
            )
        return _from_items(out, like=value) if _is_items(value, cls.dtype) else out[0]

    @classmethod
    def subtract_right(
        cls,
        value,
        op=_subtract,
        *,
        storage,
        successor=None,
        valid_items: int | Integer | None = None,
        output_dtype: type | None = None,
    ):
        """Apply an operation to each item and its right neighbor.

        Args:
            value: Input value of the specialized dtype, owned in blocked order.
            op: Callable ``op(current, right)`` on two complete input elements;
                defaults to subtraction. Its result must convert to the output
                element type. Native ``ReductionOp`` values are also accepted.
            storage: Shared instance of ``SharedStorage`` used by the block.
            successor: Item following the valid tile, or ``None`` to leave the
                last valid output equal to its converted input.
            valid_items: Valid prefix length in flattened blocked order, or
                ``None`` for the full tile. Remaining inputs pass through
                after conversion to the output dtype. A supplied successor applies
                at the valid prefix end. The count is uniform and lies in
                ``[0, block_threads * item_count]``; enforce runtime bounds.
            output_dtype: Compile-time output element type U, or ``None`` to use
                the specialized input type T. Operands and shared storage remain T;
                native conversions preserve complete element boundaries and shapes.

        Returns:
            One U for a single T input, or the same number of U items for an
            input sequence. Scalar sequences return a Vector; Vector/Struct
            sequences return a tuple. An input SoV is retained only when its
            element type is unchanged. Invalid items do not invoke ``op``.
        """
        items, tid, _, right = cls._neighbors(value, storage)
        _validate_valid_items(valid_items, cls.block_threads * len(items))
        limit = cls.block_threads * len(items) if valid_items is None else valid_items
        out = []
        for i, item in enumerate(items):
            index = tid * len(items) + i
            neighbor = right if i == len(items) - 1 else items[i + 1]
            if successor is not None:
                neighbor = _record_select(index == limit - 1, _record_default(cls.dtype, successor), neighbor)
                active = index < limit
            else:
                active = index + 1 < limit
            out.append(_apply_if(active, op, item, neighbor, cls.dtype if output_dtype is None else output_dtype))
        return _from_items(out, like=value) if _is_items(value, cls.dtype) else out[0]

    @classmethod
    def subtract_left_partial_tile(
        cls,
        value,
        valid_items: int | Integer,
        op=_subtract,
        *,
        storage,
        predecessor=None,
        output_dtype: type | None = None,
    ):
        """Compute left differences for a valid prefix, converting other inputs.

        Args:
            value: Per-thread input value in blocked order.
            valid_items: Number of valid items across the block's flattened tile.
            op: Callable ``op(current, left)``; defaults to subtraction.
            storage: Shared instance of ``SharedStorage`` used by the block.
            predecessor: Item preceding the tile, or ``None`` to retain its
                first input item after output conversion.
            output_dtype: Compile-time output element type, defaulting to the
                input type. Both differences and passthrough items convert to it.

        Returns:
            Difference values as for ``subtract_left``.
        """
        return cls.subtract_left(
            value, op, storage=storage, predecessor=predecessor, valid_items=valid_items, output_dtype=output_dtype
        )

    @classmethod
    def subtract_right_partial_tile(
        cls,
        value,
        valid_items: int | Integer,
        op=_subtract,
        *,
        storage,
        successor=None,
        output_dtype: type | None = None,
    ):
        """Compute right differences using the end of a valid prefix.

        Args:
            value: Per-thread input value in blocked order.
            valid_items: Number of valid items across the block's flattened tile.
            op: Callable ``op(current, right)``; defaults to subtraction.
            storage: Shared instance of ``SharedStorage`` used by the block.
            successor: Item following the valid prefix, or ``None`` to retain
                the last valid input item after output conversion. Invalid inputs
                also pass through this conversion without invoking ``op``.
            output_dtype: Compile-time output element type, defaulting to the
                input type. Both differences and passthrough items convert to it.

        Returns:
            Difference values as for ``subtract_right``.
        """
        return cls.subtract_right(
            value, op, storage=storage, successor=successor, valid_items=valid_items, output_dtype=output_dtype
        )
