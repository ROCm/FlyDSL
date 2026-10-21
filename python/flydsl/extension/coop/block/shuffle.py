# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide thread offsets and adjacent items in blocked order."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.numeric import Integer
from ....expr.struct import Struct
from .._common import _linear_thread_id
from .._values import (
    _as_items,
    _from_items,
    _is_items,
    _record_default,
    _record_select,
    _shared_array,
    _shared_load,
    _shared_store,
)
from ._spec import BlockAlgorithmMeta, BlockPrimitive

__all__ = [
    "BlockShuffleAlgorithm",
    "BlockShuffle",
]


class BlockShuffleAlgorithm(enum.Enum):
    """Storage policies for block-wide shifts and rotations.

    Attributes:
        SHARED_MEMORY: Publish thread boundary items in shared memory and
            synchronize the block before reading neighboring values.
    """

    SHARED_MEMORY = "shared_memory"

    SharedMemory = SHARED_MEMORY


@jit
def _publish(first, last, tid, storage):
    _shared_store(storage.first, tid, first)
    _shared_store(storage.last, tid, last)
    barrier()


def _storage(dtype, threads, warp):
    return Struct["first" : _shared_array(dtype, threads), "last" : _shared_array(dtype, threads)]


class _BlockShuffleMeta(BlockAlgorithmMeta):
    _algorithms = BlockShuffleAlgorithm
    _shared_storage = {BlockShuffleAlgorithm.SHARED_MEMORY: _storage}

    def _default_algorithm_for(cls, target):
        return BlockShuffleAlgorithm.SHARED_MEMORY


class BlockShuffle(BlockPrimitive, metaclass=_BlockShuffleMeta):
    """Shift thread values or items in a blocked sequence.

    Specialize with ``BlockShuffle[dtype, block_size, algorithm]``. The parameters below are
    compile-time positional arguments to ``BlockShuffle[...]``, in bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        algorithm: Optional member of BlockShuffleAlgorithm selecting the implementation
            policy. Defaults to ``BlockShuffleAlgorithm.SHARED_MEMORY`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    ``offset`` and ``rotate`` move one complete element per thread. ``up`` and ``down``
    move one item in the flattened blocked sequence. A missing boundary
    value defaults to the calling thread's original boundary item.
    All block threads must participate with the same per-thread item count.
    Methods include a publication barrier; synchronize before reusing storage.

    Attributes:
        dtype: Specialized value type.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        algorithm: Selected ``BlockShuffleAlgorithm`` policy.
        SharedStorage: Shared-memory struct type with one ``first`` and one
            ``last`` item per block thread.

    Examples:
        # Thread t supplies x=[2*t, 2*t+1], a blocked sequence of 0 through 127.
        P = fx.coop.BlockShuffle[fx.Int32, 64, fx.coop.BlockShuffleAlgorithm.SHARED_MEMORY]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y, suffix = P.up(x, storage=storage, predecessor=-1, return_suffix=True)
        # Thread t receives [2*t-1, 2*t]; suffix=127 in every thread.
        fx.barrier()
        down, prefix = P.down(x, storage=storage, successor=128, return_prefix=True)
        # Thread t receives [2*t+1, 2*t+2]; prefix=0 in every thread.

        # For scalar input s=10*(t+1), offset fills the boundary and rotate wraps it.
        fx.barrier()
        shifted = P.offset(s, distance=1, storage=storage, fill=-1)
        fx.barrier()
        rotated = P.rotate(s, distance=1, storage=storage)
        # T0..T62 receive 10*(t+2) in both results; T63 receives -1 and 10 respectively.
    """

    @classmethod
    def _neighbors(cls, value, storage):
        cls._check(value)
        items = list(_as_items(value, cls.dtype))
        tid = _linear_thread_id(cls.block_size)
        _publish(items[0], items[-1], tid, storage)
        left = _record_default(cls.dtype, _shared_load(storage.last, (tid + cls.block_threads - 1) % cls.block_threads))
        right = _record_default(cls.dtype, _shared_load(storage.first, (tid + 1) % cls.block_threads))
        return items, tid, left, right

    @classmethod
    def offset(cls, value, distance: int | Integer = 1, *, storage, fill=None):
        """Read the value at the calling thread's index plus ``distance``.

        Args:
            value: One input value of the specialized dtype per thread.
            distance: Signed thread offset; positive values read higher indices.
            storage: Shared instance of ``SharedStorage`` used by the block.
            fill: Value for an out-of-block source, or ``None`` to retain the
                calling thread's input. Scalar fills broadcast to record fields.

        Returns:
            The selected value, with the same dtype as ``value``.
        """
        cls._check(value)
        if _is_items(value, cls.dtype):
            raise TypeError("offset expects one complete element per thread; use up/down for item ranges")
        tid = _linear_thread_id(cls.block_size)
        _publish(value, value, tid, storage)
        source = tid + distance
        valid = (source >= 0) & (source < cls.block_threads)
        moved = _record_default(cls.dtype, _shared_load(storage.first, valid.select(source, tid)))
        return _record_select(valid, moved, value if fill is None else _record_default(cls.dtype, fill))

    @classmethod
    def rotate(cls, value, distance: int | Integer = 1, *, storage):
        """Read a value at a cyclic thread offset within the block.

        Args:
            value: One input value of the specialized dtype per thread.
            distance: Signed thread offset, reduced modulo ``block_threads``.
            storage: Shared instance of ``SharedStorage`` used by the block.

        Returns:
            The value from thread ``(tid + distance) % block_threads``.
        """
        cls._check(value)
        if _is_items(value, cls.dtype):
            raise TypeError("rotate expects one complete element per thread")
        tid = _linear_thread_id(cls.block_size)
        _publish(value, value, tid, storage)
        # Reduce the distance before adding tid, so the sum cannot overflow.
        shift = (distance % cls.block_threads + cls.block_threads) % cls.block_threads
        source = (tid + shift) % cls.block_threads
        return _record_default(cls.dtype, _shared_load(storage.first, source))

    @classmethod
    def up(
        cls,
        value,
        *,
        storage,
        predecessor=None,
        return_suffix: bool = False,
    ):
        """Shift blocked items one position toward higher sequence indices.

        Args:
            value: Input value with same-type items in blocked ownership order.
            storage: Shared instance of ``SharedStorage`` used by the block.
            predecessor: Item inserted at the start of the block, or ``None``
                to retain the first input item.
            return_suffix: Whether to also return the block's last input item.

        Returns:
            Shifted values with the input item count and shape. If ``return_suffix`` is true, returns
            ``(shifted_items, block_suffix)``; the suffix is available to every
            thread.
        """
        items, tid, left, _ = cls._neighbors(value, storage)
        boundary = items[0] if predecessor is None else _record_default(cls.dtype, predecessor)
        result = [_record_select(tid == 0, boundary, left), *items[:-1]]
        result = _from_items(result, like=value) if _is_items(value, cls.dtype) else result[0]
        return (
            (result, _record_default(cls.dtype, _shared_load(storage.last, cls.block_threads - 1)))
            if return_suffix
            else result
        )

    @classmethod
    def down(
        cls,
        value,
        *,
        storage,
        successor=None,
        return_prefix: bool = False,
    ):
        """Shift blocked items one position toward lower sequence indices.

        Args:
            value: Input value with same-type items in blocked ownership order.
            storage: Shared instance of ``SharedStorage`` used by the block.
            successor: Item inserted at the end of the block, or ``None`` to
                retain the last input item.
            return_prefix: Whether to also return the block's first input item.

        Returns:
            Shifted values with the input item count and shape. If ``return_prefix`` is true, returns
            ``(shifted_items, block_prefix)``; the prefix is available to every
            thread.
        """
        items, tid, _, right = cls._neighbors(value, storage)
        boundary = items[-1] if successor is None else _record_default(cls.dtype, successor)
        result = [*items[1:], _record_select(tid == cls.block_threads - 1, boundary, right)]
        result = _from_items(result, like=value) if _is_items(value, cls.dtype) else result[0]
        return (result, _record_default(cls.dtype, _shared_load(storage.first, 0))) if return_prefix else result
