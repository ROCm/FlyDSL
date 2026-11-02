# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expose portable collectives under the same names as the dispatched API.

``fx.coop.universal.warp_reduce`` always uses the portable warp implementation.
``fx.coop.warp_reduce`` may select a backend override when its requirements are
met. Both expose the same argument and result contracts; floating-point
rounding can depend on the reduction tree.

The block subclasses in this module also route their internal warp operations
and scans through portable implementations. This namespace supports explicit
implementation selection and comparison in correctness tests and benchmarks.
"""

from types import SimpleNamespace

from . import block as _block
from . import warp as _warp
from .block import reduce as _block_reduce
from .block import scan as _block_scan
from .block.adjacent_difference import *
from .block.discontinuity import *
from .block.exchange import *
from .block.histogram import *
from .block.load import *
from .block.merge_sort import *
from .block.radix_rank import BlockRadixRank as _BlockRadixRank
from .block.radix_rank import BlockRadixRankAlgorithm as BlockRadixRankAlgorithm
from .block.radix_sort import BlockRadixSort as _BlockRadixSort
from .block.radix_sort import BlockRadixSortAlgorithm as BlockRadixSortAlgorithm
from .block.run_length_decode import BlockRunLengthDecode as _BlockRunLengthDecode
from .block.run_length_decode import BlockRunLengthDecodeAlgorithm as BlockRunLengthDecodeAlgorithm
from .block.shuffle import *
from .block.store import *
from .block.topk import BlockTopK as _BlockTopK
from .block.topk import BlockTopKAlgorithm as BlockTopKAlgorithm
from .warp import reduce as _warp_reduce
from .warp import scan as _warp_scan
from .warp.bitonic_sort import *
from .warp.exchange import *
from .warp.load import *
from .warp.merge_sort import *
from .warp.reduce_batched import *
from .warp.store import *

WarpReduce = _warp_reduce.WarpReduce
WarpScan = _warp_scan.WarpScan

warp_head_segmented_reduce = _warp_reduce.warp_head_segmented_reduce
warp_tail_segmented_reduce = _warp_reduce.warp_tail_segmented_reduce
warp_broadcast = _warp_scan.warp_broadcast
warp_reduce = _warp_reduce.warp_reduce
warp_inclusive_scan = _warp_scan.warp_inclusive_scan
warp_exclusive_scan = _warp_scan.warp_exclusive_scan
warp_scan = _warp_scan.warp_scan
warp_scan_with_aggregate = _warp_scan.warp_scan_with_aggregate


# What the block classes below fold through, in place of the dispatched warp
# namespace. Only the two names block scope actually reaches for are here, so a
# warp primitive that gains a block-scope caller has to be added deliberately.
_UNIVERSAL_WARP = SimpleNamespace(
    warp_reduce=warp_reduce,
    warp_inclusive_scan=warp_inclusive_scan,
    warp_scan_with_aggregate=warp_scan_with_aggregate,
)


# The policy enums describe what an algorithm does, not how it is compiled, so
# they are the dispatched ones rather than copies: a caller must be able to pass
# ``fx.coop.BlockReduceAlgorithm.RAKING`` to either spelling of ``BlockReduce``.
BlockReduceAlgorithm = _block_reduce.BlockReduceAlgorithm
BlockScanAlgorithm = _block_scan.BlockScanAlgorithm


class BlockReduce(_block_reduce.BlockReduce):
    """Reduce a block using portable warp operations for every policy.

    Specialize with ``BlockReduce[dtype, block_size, algorithm]``. The parameters below are
    compile-time positional arguments to ``BlockReduce[...]``, in bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. The aggregate has this
            type. Structured elements require a callable operator.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        algorithm: Optional member of BlockReduceAlgorithm selecting the implementation
            policy. Defaults to ``BlockReduceAlgorithm.WARP_REDUCTIONS`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    Specialization, arguments, result ownership and shared-storage requirements
    are those of :class:`~flydsl.extension.coop.block.reduce.BlockReduce`.

    Examples:
        # x in thread t is [2*t+1, 2*t+2], so the block owns integers 1 through 128.
        P = fx.coop.universal.BlockReduce[fx.Int32, 64, fx.coop.universal.BlockReduceAlgorithm.WARP_REDUCTIONS]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        y = P(x, fx.ReductionOp.ADD, storage=storage)
        # Every thread receives y=8256.
        fx.barrier()
        partial = P(x[0], fx.ReductionOp.ADD, storage=storage, valid_items=5, identity=fx.Int32(0))
        # Every thread receives partial=25, the sum of the first five threads' odd values.
    """

    warp_ops = _UNIVERSAL_WARP


class BlockScan(_block_scan.BlockScan):
    """Scan a block using portable warp operations for every policy.

    Specialize with ``BlockScan[dtype, block_size, algorithm]``. The parameters below are
    compile-time positional arguments to ``BlockScan[...]``, in bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. Prefixes, seeds and the
            aggregate use this type. Structured elements require a callable operator.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        algorithm: Optional member of BlockScanAlgorithm selecting the implementation
            policy. Defaults to ``BlockScanAlgorithm.WARP_SCANS`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    Specialization, arguments, callbacks and shared-storage requirements are
    those of :class:`~flydsl.extension.coop.block.scan.BlockScan`.

    Examples:
        # Compute prefix sums of 256 ones, with four consecutive items per thread. The scan continues
        # across thread boundaries.

        P = fx.coop.universal.BlockScan[fx.Int32, 64, fx.coop.universal.BlockScanAlgorithm.WARP_SCANS]
        storage = fx.SharedAllocator().allocate(P.SharedStorage).peek()
        exclusive = P.exclusive(x, fx.ReductionOp.ADD, init=0, storage=storage)
        fx.barrier()
        inclusive = P.inclusive(x, fx.ReductionOp.ADD, storage=storage)

        # Tables below show T0..T3; each of T4..T63 also supplies four ones.
        # Data           | T0        | T1        | T2           | T3
        # ---------------+-----------+-----------+--------------+--------------
        # x in (blocked) | [1,1,1,1] | [1,1,1,1] | [1,1,1,1]    | [1,1,1,1]
        # exclusive      | [0,1,2,3] | [4,5,6,7] | [8,9,10,11]  | [12,13,14,15]
        # inclusive      | [1,2,3,4] | [5,6,7,8] | [9,10,11,12] | [13,14,15,16]

        # The corresponding exclusive_with_aggregate and inclusive_with_aggregate methods also return 256
        # to every thread. With 128 threads and four ones per thread, the last thread receives
        # [508,509,510,511] from the exclusive scan.

        # Guarded scans take one item per thread and count contributing threads.
        fx.barrier()
        inclusive, aggregate = P.inclusive_with_aggregate(
            x[0], fx.ReductionOp.ADD, storage=storage, init=10, valid_items=6
        )
        # T0..T5 receive 11..16; the remaining threads receive 16. aggregate=6.

        # A callback supplies one prefix for the complete array tile.
        def previous_prefix(aggregate):
            return fx.Int32(10)

        fx.barrier()
        exclusive, aggregate = P.exclusive_with_aggregate(
            x, fx.ReductionOp.ADD, storage=storage, prefix_callback=previous_prefix
        )
        # aggregate=256; T0 receives [10,11,12,13], T1 receives [14,15,16,17].
    """

    warp_ops = _UNIVERSAL_WARP


class BlockRadixSort(_BlockRadixSort):
    """Sort a block by radix digits using portable internal scans.

    Specialize with ``BlockRadixSort[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockRadixSort[...]``, in
    bracket order.

    Args:
        dtype: Required key element type for keys only, or (key_dtype, value_dtype) for
            key/value pairs. Each type describes one complete element, including Vector or
            Struct elements; payload types are independent of key types. Keys require
            numeric radix fields; supply a decomposer at call time for structured keys.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        items_per_thread: Required positive Python int giving the number of complete items
            owned by each thread. The block tile contains block_threads * items_per_thread
            items. This count excludes element components/fields and need not be a power of
            two. Pair payloads have the same per-thread count as keys.
        algorithm: Optional member of BlockRadixSortAlgorithm selecting the implementation
            policy. Defaults to ``BlockRadixSortAlgorithm.RADIX_SCAN`` when omitted. It
            fixes the specialization and its SharedStorage layout.

    The API and bit-ordering contract are those of
    :class:`~flydsl.extension.coop.block.radix_sort.BlockRadixSort`.

    Examples:
        # T0..T3 supply keys={[3,-1], [2,-3], [0,2], [-1,1]}; T4..T63 supply [4,4].
        # Thread t supplies payload values=[2*t, 2*t+1].
        P = fx.coop.universal.BlockRadixSort[fx.Int32, 64, 2]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        y = P(keys, storage=storage)
        # T0..T3 receive {[-3,-1], [-1,0], [1,2], [2,3]}; T4..T63 receive [4,4].

        # Sort descending by the lowest two transformed bits and return striped pairs.
        Q = fx.coop.universal.BlockRadixSort[(fx.Int32, fx.Int32), 64, 2, fx.coop.universal.BlockRadixSortAlgorithm.RADIX_DIGIT]
        scratch = allocator.allocate(Q.SharedStorage).peek()
        sorted_keys, sorted_values = Q(
            keys, values, storage=scratch, descending=True, begin_bit=0, end_bit=2,
            decomposer=lambda key: (key,), radix_bits=2, blocked_to_striped=True,
        )
        # The logical sorted prefix is [3,-1,-1,2,2,-3,1,0], followed by 120 fours.
        # Its payload prefix is [0,1,6,2,5,3,7,4], followed by positions 8 through 127.
        # Striped T0..T3 keys: {[3,4], [-1,4], [-1,4], [2,4]}; values: {[0,64], [1,65], [6,66], [2,67]}.
        fx.barrier()
        sorted_keys, sorted_values = Q.sort_blocked_to_striped(
            keys, values, storage=scratch, descending=True, begin_bit=0, end_bit=2,
            decomposer=lambda key: (key,), radix_bits=2,
        )
        # The named method returns the same striped pairs.
    """

    scan_type = BlockScan


class BlockRadixRank(_BlockRadixRank):
    """Rank a block's radix digits using portable internal scans.

    Specialize with ``BlockRadixRank[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockRadixRank[...]``, in
    bracket order.

    Args:
        dtype: Required complete input element type, such as a Numeric scalar, concrete
            Vector, or Struct. Vector components and Struct fields belong to one element;
            outer per-thread item sequences are not part of dtype. Raw-bit ranking requires
            integer Numeric keys; other key types require a digit_extractor at call time.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        items_per_thread: Required positive Python int giving the number of complete items
            owned by each thread. The block tile contains block_threads * items_per_thread
            items. This count excludes element components/fields and need not be a power of
            two.
        algorithm: Optional member of BlockRadixRankAlgorithm selecting the implementation
            policy. Defaults to ``BlockRadixRankAlgorithm.BASIC`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    The API, digit extraction and input layouts are those of
    :class:`~flydsl.extension.coop.block.radix_rank.BlockRadixRank`.

    Histogram capacity defaults to eight bits. Use
    ``BlockRadixRank.with_radix_bits(bits)`` before bracket specialization
    to change it; this is separate from the bracket parameters.

    Examples:
        # T0..T3 supply blocked keys={[2,0], [1,2], [0,1], [2,0]}; T4..T63 supply [3,3].
        P = fx.coop.universal.BlockRadixRank[fx.Int32, 64, 2, fx.coop.universal.BlockRadixRankAlgorithm.BASIC]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        ranks = P(keys, storage=storage, begin_bit=0, end_bit=2)
        # T0..T3 ranks: {[5,0], [3,6], [1,4], [7,2]}; T4..T63 retain ranks [2*t,2*t+1].

        # MATCH consumes warp-striped input. Exchange the same sequence into that layout.
        E = fx.coop.universal.BlockExchange[fx.Int32, 64, 2]
        exchange_storage = allocator.allocate(E.SharedStorage).peek()
        striped_keys = E.blocked_to_warp_striped(keys, storage=exchange_storage)
        Q = fx.coop.universal.BlockRadixRank.with_radix_bits(2)[fx.Int32, 64, 2, fx.coop.universal.BlockRadixRankAlgorithm.MATCH]
        scratch = allocator.allocate(Q.SharedStorage).peek()

        def save_counts(counts):
            # bin_counts is a separate writable tensor of 64 Int32 elements.
            bin_counts[fx.thread_idx.x] = counts[0]

        ranks, offsets = Q(
            striped_keys, storage=scratch, descending=True, begin_bit=0, end_bit=2,
            digit_extractor=lambda key: key & 3, radix_bits=2,
            count_callback=save_counts, with_prefix=True,
        )
        # bin_counts starts [3,2,3,120], followed by 60 zeros.
        # Offsets in T0..T3 are {[125], [123], [120], [0]}; later offsets are padding to ignore.
        # In logical input order, the first eight ranks are [120,125,123,121,126,124,122,127].
        # The other keys, all digit 3, receive ranks 0 through 119 in their original order.
        fx.barrier()
        ranks, offsets = Q.rank(
            striped_keys, storage=scratch, descending=True, begin_bit=0, end_bit=2,
            digit_extractor=lambda key: key & 3, radix_bits=2,
            count_callback=save_counts, with_prefix=True,
        )
        # The named rank entry point returns the same results in warp-striped ownership.
    """

    scan_type = BlockScan


class BlockTopK(_BlockTopK):
    """Select a block's smallest or largest keys using portable internal scans.

    Specialize with ``BlockTopK[dtype, block_size, items_per_thread, algorithm]``. The
    parameters below are compile-time positional arguments to ``BlockTopK[...]``, in bracket
    order.

    Args:
        dtype: Required key element type for keys only, or (key_dtype, value_dtype) for
            key/value pairs. Each type describes one complete element, including Vector or
            Struct elements; payload types are independent of key types. MERGE accepts a
            call-time compare_op for custom key ordering; AIR requires Numeric scalar keys.
        block_size: Required launch shape: a positive Python int for (x, 1, 1), or three
            positive Python int extents (x, y, z). Linear thread order varies x fastest. The
            shape must match the kernel launch. A thread count smaller than the target
            physical warp must be a power of two and becomes the logical warp width; other
            counts must be multiples of the physical warp width. The launch and allocated
            SharedStorage must fit the target limits.
        items_per_thread: Required positive Python int giving the number of complete items
            owned by each thread. The block tile contains block_threads * items_per_thread
            items. This count excludes element components/fields and need not be a power of
            two. Pair payloads have the same per-thread count as keys.
        algorithm: Optional member of BlockTopKAlgorithm selecting the implementation
            policy. Defaults to ``BlockTopKAlgorithm.MERGE`` when omitted. It fixes the
            specialization and its SharedStorage layout.

    The API and policy-dependent output ordering are those of
    :class:`~flydsl.extension.coop.block.topk.BlockTopK`.

    Examples:
        # T0..T3 supply keys={[4,1], [7,3], [6,2], [8,5]}; T4..T63 supply [0,0].
        # Thread t supplies payload values=[2*t,2*t+1].
        P = fx.coop.universal.BlockTopK[fx.Int32, 64, 2, fx.coop.universal.BlockTopKAlgorithm.MERGE]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        y, valid = P(keys, 3, storage=storage, largest=True)
        # The first three blocked slots contain [8,7,6]; only those slots have valid=True.
        # All other output values are unspecified. AIR would retain input order [7,6,8].

        Q = fx.coop.universal.BlockTopK[(fx.Int32, fx.Int32), 64, 2, fx.coop.universal.BlockTopKAlgorithm.MERGE]
        scratch = allocator.allocate(Q.SharedStorage).peek()
        selected, payload, valid = Q(
            keys, 3, values, storage=scratch, largest=False, num_valid=6, compare_op=lambda a, b: a < b,
        )
        # Valid keys are [1,2,3], with payloads [1,5,3], in the first three blocked slots.
        fx.barrier()
        selected, payload, valid = Q.min_pairs(keys, values, 3, num_valid=6, storage=scratch)
        fx.barrier()
        selected, payload, valid = Q.max_pairs(keys, values, 3, num_valid=6, storage=scratch)
        # min_pairs gives [1,2,3] / [1,5,3]; max_pairs gives [7,6,4] / [2,4,0].
        fx.barrier()
        selected, valid = P.min_keys(keys, 3, num_valid=6, storage=storage)
        fx.barrier()
        selected, valid = P.max_keys(keys, 3, num_valid=6, storage=storage)
        # Key-only methods give the same key prefixes. Only the first three slots are valid.
    """

    scan_type = BlockScan


class BlockRunLengthDecode(_BlockRunLengthDecode):
    """Decode block-local runs using portable internal scans.

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

    The API and initialized decode-window lifetime are those of
    :class:`~flydsl.extension.coop.block.run_length_decode.BlockRunLengthDecode`.

    Examples:
        # T0..T3 own run values {7,4,9,2}, lengths {2,1,3,2}, and start offsets {0,2,3,6}.
        # T4..T63 own zero-length runs: value=0, length=0, start offset=8.
        P = fx.coop.universal.BlockRunLengthDecode[fx.Int32, 64, 1, fx.coop.universal.BlockRunLengthDecodeAlgorithm.SCAN]
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


__all__ = [
    *_warp.__all__,
    *_block.__all__,
]
