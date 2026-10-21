# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reusable decoding windows with bounded shared memory."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.numeric import Int32, Integer
from ....expr.primitive import range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array, ReductionOp, Vector
from .._common import _linear_thread_id
from .._values import (
    _as_items,
    _from_items,
    _record_default,
    _record_select,
    _shared_array,
    _shared_load,
    _shared_store,
)
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
from .scan import BlockScan

__all__ = [
    "BlockRunLengthDecodeAlgorithm",
    "BlockRunLengthDecode",
]


class BlockRunLengthDecodeAlgorithm(enum.Enum):
    """Initialization policies for reusable block run decoders.

    Attributes:
        SCAN: Prefix-scan run lengths to obtain run endpoints, then decode
            windows using shared per-thread endpoint and value scratch space.
    """

    SCAN = "scan"

    Scan = SCAN


@jit
def _publish(value, end, storage, tid):
    _shared_store(storage.values, tid, value)
    storage.ends[tid] = end
    barrier()


@jit
def _publish_thread_ends(end, storage, tid):
    storage.thread_ends[tid] = end
    barrier()


@jit
def _find_run(position, ends, runs):
    low = Int32(0)
    high = Int32(runs)
    for step in range_constexpr(runs.bit_length()):
        mid = (low + high) // 2
        end = ends[(mid < runs).select(mid, Int32(runs - 1))].to(position.dtype)
        advance = (mid < runs) & (end <= position)
        low = advance.select(mid + 1, low)
        high = advance.select(high, mid)
    return (low < runs).select(low, Int32(runs - 1))


def _storage(dtype, threads, warp, count):
    dtype, offset = dtype if isinstance(dtype, tuple) else (dtype, Int32)
    if not isinstance(offset, type) or not issubclass(offset, Integer):
        raise TypeError("decoded offset dtype must be an integer type")
    return Struct[
        "values" : _shared_array(dtype, threads),
        "ends" : Array[offset, threads],
        "thread_ends" : Array[offset, threads],
        "scan" : BlockScan[offset, threads].SharedStorage,
    ]


class _Decoder:
    """Reusable per-thread run state returned by block decoder initialization.

    Run values and boundaries remain in thread-local state. Every ``decode``
    call restores its shared index, allowing completed calls from different
    decoder states to reuse the same scratch storage. All block threads must
    participate uniformly. ``run_length_decode`` is an alias for ``decode``.

    Attributes:
        cls: Specialized ``BlockRunLengthDecode`` class.
        values: Per-thread run values in blocked order.
        starts: Inclusive global starting offset of each local run.
        ends: Exclusive global ending offset of each local run.
        total: Exclusive bound for decoded positions, in the offset dtype.
        storage: Shared scratch storage used by decoding calls.
    """

    def __init__(self, cls, values, starts, ends, total, storage):
        """Retain initialized run values and boundaries for later windows.

        Args:
            cls: Specialized block decoder class.
            values: Per-thread sequence of complete run values.
            starts: Per-thread vector of inclusive run starting offsets.
            ends: Per-thread vector of exclusive run ending offsets.
            total: Exclusive global decoded-position bound.
            storage: Shared instance of the specialized ``SharedStorage``.
        """
        self.cls, self.values, self.starts, self.ends = cls, values, starts, ends
        self.total, self.storage = total, storage

    def decode(self, *, decoded_items_per_thread: int = 1, offset: int | Integer = 0, default=0):
        """Decode a blocked output window without rescanning run lengths.

        Args:
            decoded_items_per_thread: Positive Python integer giving the output
                vector extent for every thread.
            offset: Global starting position of the decoded window, converted
                to the specialized offset dtype. Thread ``t`` receives positions
                ``offset + t * decoded_items_per_thread + i``.
            default: Fill for positions outside every run or at/above ``total``.
                Scalar fills broadcast to record fields.

        Returns:
            ``(values, relative_offsets, total)``. The decoded value contains ``decoded_items_per_thread`` items;
            relative
            offsets are an integer vector of positions within each matched run.
            Unmatched positions contain ``default`` and offset ``-1`` (the
            maximum representable value for an unsigned offset dtype). ``total``
            is the initialized bound, including the maximum-value sentinel when
            offset initialization omitted it. The call finishes with a barrier.
        """
        cls, storage = self.cls, self.storage
        dtype = cls.value_dtype or Int32
        if not isinstance(decoded_items_per_thread, int) or decoded_items_per_thread < 1:
            raise ValueError("decoded_items_per_thread must be a positive Python int")
        tid = _linear_thread_id(cls.block_size)
        # Restore the index on every call. Separate decoder states can share
        # scratch storage while retaining their local run data in registers.
        _publish_thread_ends(self.ends[-1], storage, tid)
        positions = [
            dtype(offset) + (tid * decoded_items_per_thread + i).to(dtype) for i in range(decoded_items_per_thread)
        ]
        owners = [_find_run(position, storage.thread_ends, cls.block_threads) for position in positions]
        out = [_record_default(cls.dtype, default) for _ in positions]
        relative = [dtype(-1) for _ in positions]
        for run in range(cls.runs_per_thread):
            _publish(self.values[run], self.starts[run], storage, tid)
            starts = [storage.ends[owner].to(dtype) for owner in owners]
            # All reads must finish before the same scratch slots hold ends.
            barrier()
            _publish(self.values[run], self.ends[run], storage, tid)
            for i, position in enumerate(positions):
                end = storage.ends[owners[i]].to(dtype)
                valid = (position >= starts[i]) & (position < end) & (position < self.total)
                out[i] = _record_select(
                    valid, _record_default(cls.dtype, _shared_load(storage.values, owners[i])), out[i]
                )
                relative[i] = valid.select(position - starts[i], relative[i])
            barrier()
        return _from_items(out), Vector.from_elements(relative), self.total

    run_length_decode = decode


class _BlockRunLengthDecodeMeta(BlockTileAlgorithmMeta):
    _supports_pairs = True
    _parameter_name = "runs_per_thread"
    _algorithms = BlockRunLengthDecodeAlgorithm
    _shared_storage = {BlockRunLengthDecodeAlgorithm.SCAN: _storage}

    def _default_algorithm_for(cls, target):
        return BlockRunLengthDecodeAlgorithm.SCAN

    def __call__(
        cls,
        values,
        lengths,
        *,
        storage,
        decoded_items_per_thread: int = 1,
        offset: int | Integer = 0,
        default=0,
    ):
        """Implement :meth:`BlockRunLengthDecode.__call__`."""
        state = cls.init_with_run_lengths(values, lengths, storage=storage)
        return state.decode(decoded_items_per_thread=decoded_items_per_thread, offset=offset, default=default)


class BlockRunLengthDecode(BlockPrimitive, metaclass=_BlockRunLengthDecodeMeta):
    """Decode globally ordered runs into reusable blocked output windows.

    Specialize with ``BlockRunLengthDecode[dtype, block_size, runs_per_thread, algorithm]``.
    The parameters below are compile-time positional arguments to
    ``BlockRunLengthDecode[...]``, in bracket order.

    Args:
        dtype: Required complete run-value type, including Vector or Struct, or
            (value_dtype, offset_dtype) to select an integer offset type. A single value
            type uses Int32 offsets. The offset type controls run lengths, run offsets,
            relative offsets and decoded totals; all offset arithmetic must fit that type.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        runs_per_thread: Required positive Python int giving the number of input runs owned
            by each thread in blocked order. The block provides block_threads *
            runs_per_thread runs. This fixes the input run count; decoded_items_per_thread
            selects the output window size separately at call time.
        algorithm: Optional member of BlockRunLengthDecodeAlgorithm selecting the
            implementation policy. Defaults to ``BlockRunLengthDecodeAlgorithm.SCAN`` when
            omitted. It fixes the specialization and its SharedStorage layout.

    ``init_with_run_lengths`` and ``init_with_run_offsets``
    return a state with ``decode`` for successive windows without rescanning.
    Offsets specify run beginnings. Supply ``total`` to bound the last run;
    otherwise it extends to the maximum representable offset. The callable
    convenience interface initializes from lengths and decodes one window.
    Results are (values, relative offsets, total). Out-of-window entries use
    default and relative offset -1 (the maximum value for unsigned offsets).
    Run lengths must be nonnegative, and supplied run starts must be
    nondecreasing in flattened blocked order. Arithmetic must fit the selected
    offset dtype; these value constraints are not checked. All initialization
    and decoding calls require uniform participation by the block and include
    barriers so completed calls can reuse shared scratch.

    Attributes:
        dtype: Specialized decoded value type.
        value_dtype: Explicit integer offset dtype, or ``None`` for ``Int32``.
        block_size: Normalized ``(x, y, z)`` block shape.
        block_threads: Total number of participating block threads.
        runs_per_thread: Positive compile-time number of input runs per thread.
        algorithm: Selected ``BlockRunLengthDecodeAlgorithm`` policy.
        SharedStorage: Shared-memory struct type containing ``values``, ``ends``,
            and ``thread_ends`` arrays sized by block threads, plus ``scan``
            scratch. Storage does not scale with the decoded output window.
        scan_type: Block scan primitive used to initialize from run lengths.

    Examples:
        # T0..T3 own run values {7,4,9,2}, lengths {2,1,3,2}, and start offsets {0,2,3,6}.
        # T4..T63 own zero-length runs: value=0, length=0, start offset=8.
        P = fx.coop.BlockRunLengthDecode[fx.Int32, 64, 1, fx.coop.BlockRunLengthDecodeAlgorithm.SCAN]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y, relative, total = P(values, lengths, storage=storage, decoded_items_per_thread=2)
        # T0..T3 y: {[7,7], [4,9], [9,9], [2,2]}; relative: {[0,1], [0,0], [1,2], [0,1]}.
        # T4..T63 receive y=[0,0], relative=[-1,-1]. Every thread receives total=8.
        fx.barrier()
        tail, relative, total = P(
            values, lengths, storage=storage, decoded_items_per_thread=2, offset=5, default=-1
        )
        # T0 receives tail=[9,2], relative=[2,0]; T1 receives [2,-1], [1,-1].
        # T2..T63 receive [-1,-1] in both vectors; total remains 8.
        fx.barrier()
        state = P.init_with_run_lengths(values, lengths, storage=storage)
        tail, relative, total = state.decode(decoded_items_per_thread=2, offset=5, default=-1)
        fx.barrier()
        state = P.init_with_run_offsets(values, offsets, storage=storage, total=8)
        tail, relative, total = state.decode(decoded_items_per_thread=2, offset=5, default=-1)
        # Reusable states initialized from lengths or offsets give the same later window.
    """

    scan_type = BlockScan

    @classmethod
    def __call__(
        cls,
        values,
        lengths,
        *,
        storage,
        decoded_items_per_thread: int = 1,
        offset: int | Integer = 0,
        default=0,
    ):
        """Initialize from run lengths and decode one blocked output window.

        Args:
            values: ``runs_per_thread`` run values of the specialized dtype,
                owned in blocked order.
            lengths: Integer value with one nonnegative length per local run. Lengths
                and their sum must fit the specialized offset dtype.
            storage: Shared instance of ``SharedStorage`` used by all threads.
            decoded_items_per_thread: Positive Python integer giving the output
                item count for each thread.
            offset: Starting position in the globally expanded run sequence.
            default: Fill for positions outside the decoded sequence; scalar
                fills broadcast to record fields.

        Returns:
            ``(decoded_values, relative_offsets, total_length)``. The first two
            result values each contain ``decoded_items_per_thread`` items per thread. Relative offsets use the
            configured integer dtype and contain ``-1`` for unmatched positions
            (its maximum value when unsigned). The total is available to every
            thread in the same offset dtype.

        Raises:
            TypeError: The operator must be specialized with ``BlockRunLengthDecode[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, runs_per_thread]")
        return type(cls).__call__(
            cls,
            values,
            lengths,
            storage=storage,
            decoded_items_per_thread=decoded_items_per_thread,
            offset=offset,
            default=default,
        )

    @classmethod
    def _inputs(cls, values, offsets):
        cls._check(values)
        dtype = cls.value_dtype or Int32
        values = _from_items(_as_items(values, cls.dtype))
        offsets = Vector.from_elements(_as_items(offsets))
        if offsets.numel != cls.runs_per_thread or offsets.dtype.is_float:
            raise TypeError("run lengths/offsets must contain runs_per_thread integer values")
        return values, Vector.from_elements([item.to(dtype) for item in offsets])

    @classmethod
    def init_with_run_lengths(cls, values, lengths, *, storage) -> "_Decoder":
        """Scan run lengths and retain state for repeated decoding windows.

        Args:
            values: ``runs_per_thread`` run values of the specialized dtype per
                thread in blocked order.
            lengths: Integer scalar or list/tuple/Vector containing
                ``runs_per_thread`` nonnegative lengths. Values and their sum must fit
                the specialized offset dtype; zero-length runs emit no items.
            storage: Shared instance of ``SharedStorage`` used by the block.

        Returns:
            A decoder state with ``decode`` and its ``run_length_decode`` alias.
            Its total bound is the sum of all run lengths, and subsequent windows
            reuse the initialized starts and ends without another scan.
        """
        values, lengths = cls._inputs(values, lengths)
        dtype = cls.value_dtype or Int32
        scan = cls.scan_type[dtype, cls.block_size]
        ends, total = scan.inclusive_with_aggregate(lengths, ReductionOp.ADD, storage=storage.scan)
        starts = Vector.from_elements([ends[i] - lengths[i] for i in range(cls.runs_per_thread)])
        barrier()
        return _Decoder(cls, values, starts, ends, total, storage)

    @classmethod
    def init_with_run_offsets(
        cls,
        values,
        offsets,
        *,
        storage,
        total: int | Integer | None = None,
    ) -> "_Decoder":
        """Initialize reusable state from global run starting offsets.

        Args:
            values: ``runs_per_thread`` run values of the specialized dtype per
                thread in blocked order.
            offsets: Integer scalar or list/tuple/Vector with one starting offset
                per local run. Starts must be nondecreasing
                across the flattened block sequence and fit the offset dtype.
                Each run ends at the next start; equal starts describe empty runs.
            storage: Shared instance of ``SharedStorage`` used by the block.
            total: Exclusive end of the final run, at least its starting offset.
                ``None`` uses the maximum representable offset as the bound;
                decoding then reports this sentinel as its total.

        Returns:
            A decoder state with ``decode`` and its ``run_length_decode`` alias.
            Positions before the first run or at/above the total bound decode
            to the requested fill value and an invalid relative offset.
        """
        values, starts = cls._inputs(values, offsets)
        dtype = cls.value_dtype or Int32
        if total is None:
            total = (1 << (dtype.width - (1 if dtype.signed else 0))) - 1
        total = dtype(total)
        tid = _linear_thread_id(cls.block_size)
        _publish_thread_ends(starts[0], storage, tid)
        next_start = storage.thread_ends[(tid + 1) % cls.block_threads].to(dtype)
        end = (tid == cls.block_threads - 1).select(total, next_start)
        ends = Vector.from_elements([*list(starts)[1:], end])
        barrier()
        return _Decoder(cls, values, starts, ends, total, storage)
