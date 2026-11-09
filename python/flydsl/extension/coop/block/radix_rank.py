# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Blocked histogram ranks and warp-striped match-based radix ranks."""

import enum
import functools

from ...._mlir.dialects.gpu import BarrierScope
from ....compiler import jit
from ....expr.gpu import barrier, lane_id, shuffle_idx
from ....expr.llvm import atomic_add, atomic_or
from ....expr.math import ctpop
from ....expr.numeric import Int32, Numeric, Uint8, Uint16, Uint64
from ....expr.primitive import const_expr, range_constexpr
from ....expr.struct import Struct
from ....expr.typing import Array, ReductionOp, Vector
from .._common import _linear_thread_id
from .._values import _as_items, _is_items, _shared_load
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta, _dispatch_target, _Public
from .radix_sort import _bit_range, _digit
from .scan import BlockScan

__all__ = [
    "BlockRadixRankAlgorithm",
    "BlockRadixRank",
]


def _extract_digit(key, extractor, radix_bits):
    digit = extractor(key)
    if isinstance(digit, int):
        if not 0 <= digit < (1 << radix_bits):
            raise ValueError("digit_extractor must return a digit in [0, 2**radix_bits)")
    elif not isinstance(digit, Numeric) or digit.dtype.is_float:
        raise TypeError("digit_extractor must return an integer digit")
    return Int32(digit)


class BlockRadixRankAlgorithm(enum.Enum):
    """Ranking policies with blocked or warp-striped input layouts.

    All policies assign distinct stable ranks, preserving their declared input
    layout's order among equal digits.

    Attributes:
        BASIC: Consume blocked tiles, count digits in packed per-thread
            shared counters and scan all of them at once. This is the
            default policy.
        MEMOIZE: Like BASIC, but keep each thread's raking segment of
            counters in registers between the scan's two sweeps.
        MATCH: Consume warp-striped tiles, form match masks with shuffles,
            and publish counts after computing local ranks.
        MATCH_EARLY_COUNTS_ANY: Publish atomic histograms before local ranking,
            then use shuffle-based match masks on warp-striped inputs.
        MATCH_EARLY_COUNTS_ATOMIC_OR: Publish early atomic histograms, then
            build shared atomic-OR match masks on warp-striped inputs.
        EARLY_COUNTS: Alias of ``MATCH_EARLY_COUNTS_ANY``.
    """

    BASIC = "basic"
    MEMOIZE = "memoize"
    MATCH = "match"
    MATCH_EARLY_COUNTS_ANY = "match_early_counts_any"
    MATCH_EARLY_COUNTS_ATOMIC_OR = "match_early_counts_atomic_or"
    EARLY_COUNTS = "match_early_counts_any"

    Basic = BASIC
    Memoize = MEMOIZE
    Match = MATCH
    MatchEarlyCountsAny = MATCH_EARLY_COUNTS_ANY
    MatchEarlyCountsAtomicOr = MATCH_EARLY_COUNTS_ATOMIC_OR
    EarlyCounts = EARLY_COUNTS


# BASIC/MEMOIZE count into one 16-bit counter per (digit, thread), packed two
# to a 32-bit word, and scan all of them at once. At most this many counter
# words per thread (16 digits, about 36 bytes of shared memory per thread);
# wider digits rank in stable sub-passes.
_COUNTER_ROWS = 8


def _counter_bits(threads, count):
    """The widest digit whose counters fit ``_COUNTER_ROWS`` words per thread."""
    return (_COUNTER_ROWS * (2 if threads * count < 1 << 16 else 1)).bit_length() - 1


def _counter_plan(threads, count, bins):
    """Compile-time layout of the per-thread digit counters of BASIC/MEMOIZE.

    Digit ``d`` lives in counter row ``d % rows`` and 16-bit field
    ``d // rows`` of each thread's word, so one 32-bit scan in row-major
    ``(row, thread)`` order scans every field at once. A tile of 65536 or more
    items needs whole 32-bit counters. Each thread rakes ``segment``
    consecutive words; an even row count gets a zero padding row so that
    segment is odd and the raking reads hit distinct banks.

    Returns ``(packing, rows, segment)``.
    """
    packing = min(2 if threads * count < 1 << 16 else 1, bins)
    rows = bins // packing
    return packing, rows, rows + 1 if rows % 2 == 0 else rows


def _storage(dtype, threads, warp, count, match=False, masks=False, radix_bits=8):
    warps = (threads + warp - 1) // warp
    index_dtype = Uint16 if threads * count <= 65536 else Int32
    digit_dtype = Uint8 if radix_bits <= 8 else Uint16 if radix_bits <= 16 else Int32
    # Running offsets take LDS atomic adds; the atomic-OR policy, whose mask
    # table dominates its storage, keeps 16-bit offsets updated by a write.
    offset_dtype = Uint16 if masks and warp * count < 65536 else Int32
    bins = 1 << radix_bits
    digit_words = (threads * count * digit_dtype.width + 63) // 64
    index_words = (threads * count * index_dtype.width + 63) // 64
    widest = 1 << min(radix_bits, _counter_bits(threads, count))
    counter_words = 1 if match else (_counter_plan(threads, count, widest)[2] * threads + 1) // 2
    scratch_words = max(digit_words + index_words, warps * bins if masks else 1, counter_words)
    Digits = Array[digit_dtype, threads * count]
    Indices = Array[index_dtype, threads * count]
    Masks = Array[Uint64, warps * bins if masks else 1]
    Words = Array[Int32, 2 * scratch_words]
    Halves = Array[Uint16, 4 * scratch_words]

    @Struct
    class Storage:
        scratch: Array[Uint64, scratch_words]
        histogram: Array[Int32, warps * bins if match else 1]
        offsets: Array[offset_dtype, warps * bins if match else 1]
        prefix: Array[Int32, bins]
        scan: BlockScan[Int32, threads].SharedStorage

        # BASIC/MEMOIZE digit counters, as words and as their 16-bit fields.
        @property
        def counters(self):
            return Words.__peek_from_ptr__(self.scratch.ptr)

        @property
        def counter_fields(self):
            return Halves.__peek_from_ptr__(self.scratch.ptr)

        # Typed views reuse scratch bytes at non-overlapping lifetimes.
        @property
        def digits(self):
            return Digits.__peek_from_ptr__(self.scratch.ptr)

        @property
        def indices(self):
            digit_words = (Digits.__dsl_size_of__() + 7) // 8
            return Indices.__peek_from_ptr__(self.scratch.ptr + digit_words)

        @property
        def masks(self):
            return Masks.__peek_from_ptr__(self.scratch.ptr)

    return Storage


def _match_storage(dtype, threads, warp, count, radix_bits=8):
    return _storage(dtype, threads, warp, count, match=True, radix_bits=radix_bits)


def _atomic_match_storage(dtype, threads, warp, count, radix_bits=8):
    return _storage(dtype, threads, warp, count, match=True, masks=True, radix_bits=radix_bits)


# Every per-warp row below (histogram, offsets, masks) is read and written by
# its own warp alone until _read_counts, so ordering inside a row needs only a
# subgroup barrier, never the block's.


@jit
def _initialize(storage, tid, width, bins, early, masks):
    warp_id, lane = tid // width, tid % width
    for i in range_constexpr((bins + width - 1) // width):
        index = lane + i * width
        if index < bins:
            storage.offsets[warp_id * bins + index] = storage.offsets.dtype(0)
            if const_expr(early):
                storage.histogram[warp_id * bins + index] = Int32(0)
            if const_expr(masks):
                storage.masks[warp_id * bins + index] = Uint64(0)
    barrier(scope=BarrierScope.Subgroup)


@jit
def _early_histogram(digits, storage, warp_id, bins, count, masks, lanes_below, lane):
    for i in range_constexpr(count):
        if const_expr(masks is None):
            atomic_add(storage.histogram.ptr + warp_id * bins + digits[i], Int32(1))
        else:
            # One add per distinct digit: its first lane adds all of the
            # digit's lanes, so skewed digits do not serialize on one address.
            if lanes_below(masks[i], lane) == 0:
                atomic_add(storage.histogram.ptr + warp_id * bins + digits[i], Int32(ctpop(masks[i])))
    barrier()


def _read_counts(source, tid, threads, warps, bins, tracked):
    """Each tracked digit's total over the warps' *source* rows, and every warp's exclusive prefix in it.

    Both stay in registers: only the thread that owns a digit uses them.
    Padding digits past the last bin count zero.
    """
    totals, fronts = [], []
    for i in range(tracked):
        digit = tid * tracked + i
        valid = None if threads * tracked == bins else digit < bins
        safe = digit if valid is None else valid.select(digit, Int32(0))
        total, front = Int32(0), []
        for w in range(warps):
            front.append(total)
            total = total + Int32(_shared_load(source, w * bins + safe))
        totals.append(total if valid is None else valid.select(total, Int32(0)))
        fronts.append(front)
    return Vector.from_elements(totals), fronts


@jit
def _write_prefix(storage, prefix, counts, fronts, tid, bins, tracked, descending, tile_items):
    """Publish each owned digit's start and, per warp, where that warp's run of the digit starts."""
    for i in range_constexpr(tracked):
        digit = tid * tracked + i
        if digit < bins:
            start = tile_items - prefix[i] - counts[i] if const_expr(descending) else prefix[i]
            storage.prefix[digit] = start
            for w in range_constexpr(len(fronts[i])):
                storage.histogram[w * bins + digit] = start + fronts[i][w]
    barrier()


def _shuffle_match(digit, width, bits):
    """Mask of the logical warp's lanes holding *digit*: one shuffle per lane."""
    mask = Uint64(0)
    for source in range(width):
        peer = shuffle_idx(digit, (lane_id() // width) * width + source, width)
        mask = mask | (peer == digit).select(Uint64(1) << source, Uint64(0))
    return mask


def _mask_rank(mask, lane):
    """Set bits of *mask* below logical *lane*."""
    return Int32(ctpop(mask & ((Uint64(1) << lane) - Uint64(1))))


@jit
def _match_rounds(digits, storage, tid, width, bins, bits, count, use_atomic_or, masks, match_any, lanes_below):
    """Warp-local ranks of each item in turn, from running per-warp digit offsets.

    Offsets start at zero. The first lane of each digit adds the digit's match
    count after every lane has read the running offset; an atomic add needs
    no value from that read, so the items' offset reads stay independent.
    The atomic-OR policy waits for its mask read anyway and writes instead.
    """
    warp_id, lane = tid // width, tid % width
    result = []
    for i in range_constexpr(count):
        digit = digits[i]
        if const_expr(use_atomic_or):
            atomic_or(storage.masks.ptr + warp_id * bins + digit, Uint64(1) << lane)
            barrier(scope=BarrierScope.Subgroup)
            mask = Uint64(storage.masks[warp_id * bins + digit])
        elif const_expr(masks is not None):
            mask = masks[i]
        else:
            mask = match_any(digit, width, bits)
        local_rank = lanes_below(mask, lane)
        previous = Int32(0)
        if const_expr(i > 0):
            previous = Int32(_shared_load(storage.offsets, warp_id * bins + digit))
        result.append(previous + local_rank)
        barrier(scope=BarrierScope.Subgroup)
        if local_rank == 0:
            if const_expr(i > 0 and not use_atomic_or):
                atomic_add(storage.offsets.ptr + warp_id * bins + digit, Int32(ctpop(mask)))
            else:
                storage.offsets[warp_id * bins + digit] = (previous + Int32(ctpop(mask))).to(storage.offsets.dtype)
            if const_expr(use_atomic_or):
                # Each digit's first lane empties its mask for the next item.
                storage.masks[warp_id * bins + digit] = Uint64(0)
        barrier(scope=BarrierScope.Subgroup)
    return Vector.from_elements(result)


@jit
def _add_offsets(ranks, digits, storage, tid, width, bins, count):
    result = []
    for i in range_constexpr(count):
        digit = digits[i]
        offset = Int32(storage.histogram[(tid // width) * bins + digit])
        result.append(ranks[i] + offset)
    return Vector.from_elements(result)


@jit
def _publish(digits, indices, ranks, storage, count):
    for i in range_constexpr(count):
        storage.digits[ranks[i]] = digits[i].to(storage.digits.dtype)
        storage.indices[ranks[i]] = indices[i].to(storage.indices.dtype)
    barrier()


@jit
def _inverse(indices, positions, storage, count):
    for i in range_constexpr(count):
        storage.indices[indices[i]] = positions[i].to(storage.indices.dtype)
    barrier()


def _counter_slot(digit, column, plan, threads):
    """Index of *digit*'s counter in *column*: a 16-bit field, or a word without packing."""
    packing, rows, _ = plan
    word = (digit & (rows - 1)) * threads + column
    return word * 2 + ((digit >> (rows.bit_length() - 1)) & 1) if packing == 2 else word


def _counter_ranks(digits, storage, scan, tid, threads, bins, descending, memoize, tracked, with_prefix):
    """Rank blocked digits with one packed counter per (digit, thread) and one block scan.

    A thread's own items are ranked in registers: ``local[i]`` counts its
    earlier items with the same digit, and every item writes the thread's
    total for its digit, so duplicates store the same value and no counter is
    read back before the scan. Each thread zeroes only its own counter
    column, which needs no barrier before its own writes. Thread ``t``
    rakes ``segment`` consecutive words, so the exclusive scan in
    ``(digit, thread)`` order leaves, at each thread's counter, the number of
    items ranked before that thread's first item of the digit; thread 0's
    column holds every digit's start.
    """
    count = len(digits)
    plan = _counter_plan(threads, count, bins)
    packing, _, segment = plan
    words, fields = storage.counters, storage.counter_fields
    ordered = [Int32(bins - 1) - d if descending else d for d in digits]
    local, total = [Int32(0)] * count, [Int32(1)] * count
    for i in range(count):
        for j in range(i):
            same = Int32(ordered[i] == ordered[j])
            local[i], total[i], total[j] = local[i] + same, total[i] + same, total[j] + same

    def load(digit, column):
        index = _counter_slot(digit, column, plan, threads)
        return _shared_load(fields, index).to(Int32) if packing == 2 else Int32(words[index])

    for r in range(segment):
        words[r * threads + tid] = Int32(0)
    for i in range(count):
        index = _counter_slot(ordered[i], tid, plan, threads)
        if packing == 2:
            fields[index] = total[i].to(Uint16)
        else:
            words[index] = total[i]
    barrier()
    raked = [Int32(words[tid * segment + k]) for k in range(segment)]
    partial = raked[0]
    for value in raked[1:]:
        partial = partial + value
    running, aggregate = scan.exclusive_with_aggregate(partial, ReductionOp.ADD, storage=storage.scan)
    if packing == 2:
        # Every high-field digit follows all low-field digits.
        running = running + (aggregate << 16)
    for k in range(segment):
        value = raked[k] if memoize else Int32(words[tid * segment + k])
        words[tid * segment + k] = running
        running = running + value
    barrier()
    ranks = [load(ordered[i], tid) + local[i] for i in range(count)]
    # Starts of the digits whose offsets this thread returns, in the requested order.
    prefix = [
        load(Int32(bins - 1 - (tid * tracked + i) % bins) if descending else Int32((tid * tracked + i) % bins), 0)
        for i in range(tracked if with_prefix else 0)
    ]
    return Vector.from_elements(ranks), Vector.from_elements(prefix) if with_prefix else None


@jit
def _digit_offsets(digits, storage, scan, tid, threads, bins, descending, tile_items):
    """Start of each digit this thread reports, from a block histogram and one scan.

    BASIC/MEMOIZE return these when sub-passes rank a digit too wide for
    their counters.
    """
    tracked = (bins + threads - 1) // threads
    for i in range_constexpr(tracked):
        if tid + i * threads < bins:
            storage.prefix[tid + i * threads] = Int32(0)
    barrier()
    for i in range_constexpr(len(digits)):
        atomic_add(storage.prefix.ptr + digits[i], Int32(1))
    barrier()
    counts = []
    for i in range_constexpr(tracked):
        digit = tid * tracked + i
        valid = digit < bins
        counts.append(valid.select(Int32(storage.prefix[valid.select(digit, Int32(0))]), Int32(0)))
    counts = Vector.from_elements(counts)
    starts = scan.exclusive(counts, ReductionOp.ADD, storage=storage.scan)
    # The scan's barrier orders every histogram read before these writes.
    for i in range_constexpr(tracked):
        digit = tid * tracked + i
        if digit < bins:
            storage.prefix[digit] = tile_items - starts[i] - counts[i] if const_expr(descending) else starts[i]
    barrier()
    return Vector.from_elements([Int32(storage.prefix[(tid * tracked + i) % bins]) for i in range_constexpr(tracked)])


class _BlockRadixRankMeta(BlockTileAlgorithmMeta):
    _algorithms = BlockRadixRankAlgorithm
    _shared_storage = {
        BlockRadixRankAlgorithm.BASIC: _storage,
        BlockRadixRankAlgorithm.MEMOIZE: _storage,
        BlockRadixRankAlgorithm.MATCH: _match_storage,
        BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ANY: _match_storage,
        BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ATOMIC_OR: _atomic_match_storage,
    }

    def _default_algorithm_for(cls, target):
        return BlockRadixRankAlgorithm.BASIC

    def __call__(
        cls,
        keys,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        digit_extractor=None,
        radix_bits: int | None = None,
        count_callback=None,
        with_prefix: bool = False,
    ):
        """Implement :meth:`BlockRadixRank.__call__`."""
        return cls.rank(
            keys,
            storage=storage,
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            digit_extractor=digit_extractor,
            radix_bits=radix_bits,
            count_callback=count_callback,
            with_prefix=with_prefix,
        )


class BlockRadixRank(BlockPrimitive, metaclass=_BlockRadixRankMeta):
    """Assign stable ranks by unsigned raw bits or a custom digit extractor.

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

    Histogram capacity defaults to eight bits; call
    ``BlockRadixRank.with_radix_bits(bits)`` before specialization to change it.
    The allocation must fit the target's shared-memory limit.

    BASIC and MEMOIZE consume blocked tiles. They count each thread's digits
    in packed 16-bit shared counters and rank every bin with one block scan;
    MEMOIZE keeps its raking segment in registers instead of reading it twice.
    Digits wider than four bits rank in stable sub-passes, and ``with_prefix``
    then counts the whole digit in a block histogram.
    MATCH and both early-count policies consume warp-striped tiles: within each
    physical warp, item index precedes lane index. MATCH forms per-item masks using portable shuffles;
    early-count policies first publish atomic histograms, then use shuffles or
    shared atomic-OR masks. Equal digits retain their policy's input order.

    Without an extractor, keys must be integer Numeric values and sorting uses
    their unsigned raw bits, including for signed integer types. Wider ranges
    use successive stable digit passes. ``digit_extractor(key)`` instead accepts
    key values and requires an explicit ``radix_bits`` within capacity. The extractor receives the original key
    value without casting and returns an integer digit in
    ``[0, 2**radix_bits)``. It defines one digit pass independently of key width;
    it does not provide natural floating-point ordering or a numeric cast.
    Use BlockRadixSort for ordered signed or floating-point sorting.

    MATCH policies support ``count_callback(counts)``. Each thread receives
    ``ceil(radix_digits / block_threads)`` counts in blocked digit order, with
    unused entries zero-filled. Early callbacks run before local ranks are
    computed. For a single digit pass, ``with_prefix=True`` also returns the
    corresponding digit offsets; padding offsets wrap to valid digit indices
    and must be ignored. Raw-bit multipass calls reject ``with_prefix=True``.

    All block threads must participate with uniform options and the same
    allocation of ``SharedStorage``. Synchronize before reusing that storage.

    Attributes:
        dtype: Specialized key element type.
        block_size: Specialized ``(x, y, z)`` shape.
        block_threads: Number of participating threads.
        items_per_thread: Number of keys supplied by each thread.
        algorithm: Ranking policy and its required input layout.
        configured_radix_bits: Histogram capacity in bits; defaults to eight.
        SharedStorage: Shared-memory Struct type sized for the specialization.

    Examples:
        # T0..T3 supply blocked keys={[2,0], [1,2], [0,1], [2,0]}; T4..T63 supply [3,3].
        P = fx.coop.BlockRadixRank[fx.Int32, 64, 2, fx.coop.BlockRadixRankAlgorithm.BASIC]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        ranks = P(keys, storage=storage, begin_bit=0, end_bit=2)
        # T0..T3 ranks: {[5,0], [3,6], [1,4], [7,2]}; T4..T63 retain ranks [2*t,2*t+1].

        # MATCH consumes warp-striped input. Exchange the same sequence into that layout.
        E = fx.coop.BlockExchange[fx.Int32, 64, 2]
        exchange_storage = allocator.allocate(E.SharedStorage).peek()
        striped_keys = E.blocked_to_warp_striped(keys, storage=exchange_storage)
        Q = fx.coop.BlockRadixRank.with_radix_bits(2)[fx.Int32, 64, 2, fx.coop.BlockRadixRankAlgorithm.MATCH]
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

    scan_type = _Public("BlockScan")
    configured_radix_bits = 8

    @classmethod
    def __call__(
        cls,
        keys,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        digit_extractor=None,
        radix_bits: int | None = None,
        count_callback=None,
        with_prefix: bool = False,
    ):
        """Assign each input key its stable zero-based rank across the block tile.

        BASIC and MEMOIZE require blocked input; MATCH policies require warp-striped
        input. Ranks refer to the original inputs in that layout, preserve equal-key
        order, and do not rearrange keys. All threads must participate and synchronize
        before reusing shared storage.

        Args:
            keys: Key value matching the specialized dtype and item count.
                Without ``digit_extractor``, the dtype must be integer Numeric.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean reversing digit order while retaining
                input order among equal digits.
            begin_bit: Nonnegative Python integer giving the first included raw
                bit. With an extractor, only the one-pass range consistency is used.
            end_bit: First excluded raw bit; defaults to the integer key width,
                or ``begin_bit + radix_bits`` when a digit width is supplied.
                Raw-bit ranges must satisfy ``0 <= begin_bit <= end_bit <= width``.
            digit_extractor: Optional callable applied to the original key and
                returning an integer digit in ``[0, 2**radix_bits)``. Requires
                explicit ``radix_bits`` and ``end_bit == begin_bit + radix_bits``.
                Runtime digit bounds are the caller's responsibility.
            radix_bits: Optional positive Python integer no larger than
                ``configured_radix_bits`` selecting the digit pass width. Without
                it, raw-bit calls choose a policy-dependent pass width. Required
                for extractors. A wider explicit raw-bit range may span many passes.
            count_callback: Optional ``callback(counts)`` for MATCH policies, called
                on every thread once per pass. Counts are Int32 Vector entries for
                this thread's consecutive digit bins, with zero-filled padding.
                BASIC and MEMOIZE reject callbacks. Early-count policies invoke
                it before local ranking; MATCH invokes
                it after. Its return value is ignored. The requested bit range
                must fit ``configured_radix_bits`` when using callbacks.
            with_prefix: Compile-time boolean requesting ``(ranks, digit_offsets)``
                for a single digit pass. Offsets locate bins in the requested
                ascending or descending order. The bit range must fit configured
                capacity. Multipass raw-bit calls reject it with ValueError.

        Returns:
            One Int32 rank for a single key, or an Int32 Vector with one rank
            per input item. Vector components inside a key do not add ranks.
            Ranks form a permutation of ``[0, block_threads * items_per_thread)``.
            For a single pass with ``with_prefix=True``, also returns an Int32
            Vector with ``ceil(2**pass_bits / block_threads)`` digit offsets per
            thread in blocked digit order. Entries beyond the number of bins wrap
            to valid bin offsets and must be ignored.

        Raises:
            TypeError: The operator must be specialized with ``BlockRadixRank[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(
            cls,
            keys,
            storage=storage,
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            digit_extractor=digit_extractor,
            radix_bits=radix_bits,
            count_callback=count_callback,
            with_prefix=with_prefix,
        )

    @classmethod
    def _match_any(cls, digit, width, bits):
        """Mask of the logical warp's lanes whose *digit* equals this lane's; target overrides may replace it."""
        return _shuffle_match(digit, width, bits)

    @classmethod
    def _early_masks(cls, digits, width, bits):
        """Match masks of every item before the early counts, or None to count lane by lane.

        Portable shuffle masks cost too many registers to hold through the
        early phase; a target with cheap masks may return them for reuse.
        """
        return None

    @classmethod
    def _lanes_below(cls, mask, lane):
        """Number of *mask* lanes below logical *lane*; target overrides may replace it."""
        return _mask_rank(mask, lane)

    @classmethod
    def with_radix_bits(cls, radix_bits: int) -> "type[BlockRadixRank]":
        """Configure histogram capacity before specializing dtype, block, and items.

        Shared-memory usage grows exponentially with the number of digit bits.
        The resulting allocation must fit the target's actual shared-memory limit.
        This configures capacity; call-time ``radix_bits`` selects a pass width.

        Args:
            radix_bits: Positive Python integer smaller than 31 giving the maximum
                digit width supported by the specialization's shared storage.

        Returns:
            An unspecialized BlockRadixRank subclass configured for the requested
            capacity. Apply ``[dtype, block_size, items_per_thread, algorithm]``
            to it before invoking a rank collective.
        """
        target = _dispatch_target(cls)
        if target is not cls:
            return target.with_radix_bits(radix_bits)
        if cls.block_threads is not None:
            raise TypeError("configure radix bits before specializing the block shape")
        if not isinstance(radix_bits, int) or radix_bits < 1:
            raise ValueError("radix_bits must be a positive Python int")
        if radix_bits >= 31:
            raise ValueError("radix digit space must fit signed 32-bit bin counters (radix_bits < 31)")
        factories = {
            algorithm: functools.partial(factory, radix_bits=radix_bits)
            for algorithm, factory in _BlockRadixRankMeta._shared_storage.items()
        }
        return type(
            f"{cls.__name__}[radix_bits={radix_bits}]",
            (cls,),
            {"configured_radix_bits": radix_bits, "_shared_storage": factories},
        )

    @classmethod
    def rank(
        cls,
        keys,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        digit_extractor=None,
        radix_bits: int | None = None,
        count_callback=None,
        with_prefix: bool = False,
    ):
        """Assign each input key its stable zero-based rank across the block tile.

        BASIC and MEMOIZE require blocked input; MATCH policies require warp-striped
        input. Ranks refer to the original inputs in that layout, preserve equal-key
        order, and do not rearrange keys. All threads must participate and synchronize
        before reusing shared storage.

        Args:
            keys: Key value matching the specialized dtype and item count.
                Without ``digit_extractor``, the dtype must be integer Numeric.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean reversing digit order while retaining
                input order among equal digits.
            begin_bit: Nonnegative Python integer giving the first included raw
                bit. With an extractor, only the one-pass range consistency is used.
            end_bit: First excluded raw bit; defaults to the integer key width,
                or ``begin_bit + radix_bits`` when a digit width is supplied.
                Raw-bit ranges must satisfy ``0 <= begin_bit <= end_bit <= width``.
            digit_extractor: Optional callable applied to the original key and
                returning an integer digit in ``[0, 2**radix_bits)``. Requires
                explicit ``radix_bits`` and ``end_bit == begin_bit + radix_bits``.
                Runtime digit bounds are the caller's responsibility.
            radix_bits: Optional positive Python integer no larger than
                ``configured_radix_bits`` selecting the digit pass width. Without
                it, raw-bit calls choose a policy-dependent pass width. Required
                for extractors. A wider explicit raw-bit range may span many passes.
            count_callback: Optional ``callback(counts)`` for MATCH policies, called
                on every thread once per pass. Counts are Int32 Vector entries for
                this thread's consecutive digit bins, with zero-filled padding.
                BASIC and MEMOIZE reject callbacks. Early-count policies invoke
                it before local ranking; MATCH invokes
                it after. Its return value is ignored. The requested bit range
                must fit ``configured_radix_bits`` when using callbacks.
            with_prefix: Compile-time boolean requesting ``(ranks, digit_offsets)``
                for a single digit pass. Offsets locate bins in the requested
                ascending or descending order. The bit range must fit configured
                capacity. Multipass raw-bit calls reject it with ValueError.

        Returns:
            One Int32 rank for a single key, or an Int32 Vector with one rank
            per input item. Vector components inside a key do not add ranks.
            Ranks form a permutation of ``[0, block_threads * items_per_thread)``.
            For a single pass with ``with_prefix=True``, also returns an Int32
            Vector with ``ceil(2**pass_bits / block_threads)`` digit offsets per
            thread in blocked digit order. Entries beyond the number of bins wrap
            to valid bin offsets and must be ignored.
        """
        cls._check(keys)
        if radix_bits is not None:
            if not isinstance(radix_bits, int) or not 1 <= radix_bits <= cls.configured_radix_bits:
                raise ValueError(
                    "radix_bits exceeds configured capacity; use BlockRadixRank.with_radix_bits(bits) first"
                )
            end_bit = begin_bit + radix_bits if end_bit is None else end_bit
        elif digit_extractor is not None:
            raise ValueError("digit_extractor requires radix_bits")
        if digit_extractor is None:
            if not issubclass(cls.dtype, Numeric) or cls.dtype.is_float:
                raise TypeError("noninteger BlockRadixRank keys require a digit_extractor and radix_bits")
            end_bit = _bit_range(cls.dtype, begin_bit, end_bit)
        else:
            if not isinstance(begin_bit, int) or begin_bit < 0 or end_bit != begin_bit + radix_bits:
                raise ValueError("digit_extractor requires one digit pass: end_bit == begin_bit + radix_bits")
        match = cls.algorithm not in (BlockRadixRankAlgorithm.BASIC, BlockRadixRankAlgorithm.MEMOIZE)
        if count_callback is not None and not match:
            raise ValueError("count callbacks require a MATCH radix rank algorithm")
        if (count_callback is not None or with_prefix) and end_bit - begin_bit > cls.configured_radix_bits:
            raise ValueError("digit callbacks and prefixes require sufficient configured_radix_bits")
        vector = _is_items(keys, cls.dtype)
        current = _as_items(keys, cls.dtype)
        count, threads, width = cls.items_per_thread, cls.block_threads, cls.warp_threads
        tid = _linear_thread_id(cls.block_size)
        active = ((threads - (tid // width) * width) < width).select(threads - (tid // width) * width, Int32(width))
        positions = Vector.from_elements(
            [
                (tid // width) * width * count + i * active + tid % width if match else tid * count + i
                for i in range(count)
            ]
        )
        indices = positions
        original_ranks = positions
        scan = cls.scan_type[Int32, cls.block_size]
        output_prefix = None
        # Four-bit passes fit the counters of BASIC/MEMOIZE; MATCH supports
        # the complete eight-bit histogram in one pass.
        step = radix_bits or (
            max(1, end_bit - begin_bit)
            if count_callback is not None or with_prefix
            else min(cls.configured_radix_bits, 8 if match else 4)
        )
        # An extractor defines one digit; sub-passes below take bit fields of it.
        extracted = (
            None if digit_extractor is None else [_extract_digit(k, digit_extractor, radix_bits) for k in current]
        )
        single = len(range(begin_bit, end_bit, step)) <= 1
        if with_prefix and not single:
            raise ValueError("with_prefix requires one digit pass: radix_bits must cover end_bit - begin_bit")
        if not match and step > _counter_bits(threads, count):
            # BASIC/MEMOIZE rank a digit wider than their counters in stable
            # sub-passes; a histogram supplies the whole digit's offsets.
            if with_prefix:
                whole = extracted or [_digit(key, begin_bit, end_bit - begin_bit, False) for key in current]
                output_prefix = _digit_offsets(
                    whole, storage, scan, tid, threads, 1 << (end_bit - begin_bit), descending, threads * count
                )
            step = -(-step // -(-step // _counter_bits(threads, count)))
        passes = list(range(begin_bit, end_bit, step)) or [begin_bit]
        for pass_index, bit in enumerate(passes):
            bits = min(step, end_bit - bit)
            bins = 1 << bits
            if extracted is None:
                digits = Vector.from_elements([_digit(key, bit, bits, False) for key in current])
            elif len(passes) == 1:
                digits = Vector.from_elements(extracted)
            else:
                digits = Vector.from_elements([(d >> (bit - begin_bit)) & (bins - 1) for d in extracted])
            if pass_index > 0:
                _publish(digits, positions, original_ranks, storage, count)
                digits = Vector.from_elements([Int32(_shared_load(storage.digits, p)) for p in positions])
                indices = Vector.from_elements([Int32(_shared_load(storage.indices, p)) for p in positions])
                barrier()
            if match:
                early = cls.algorithm is not BlockRadixRankAlgorithm.MATCH
                atomic_masks = cls.algorithm is BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ATOMIC_OR
                tracked = (bins + threads - 1) // threads
                # Masks a target can afford to hold through the early phase
                # make the early counts one add per digit instead of per lane.
                masks = cls._early_masks(digits, width, bits) if early and not atomic_masks else None
                _initialize(storage, tid, width, bins, early, atomic_masks)
                if early:
                    _early_histogram(digits, storage, tid // width, bins, count, masks, cls._lanes_below, tid % width)
                    counts, fronts = _read_counts(storage.histogram, tid, threads, cls.num_warps, bins, tracked)
                    if count_callback is not None:
                        count_callback(counts)
                ranks = _match_rounds(
                    digits,
                    storage,
                    tid,
                    width,
                    bins,
                    bits,
                    count,
                    atomic_masks,
                    masks,
                    cls._match_any,
                    cls._lanes_below,
                )
                if not early:
                    # Each warp's final running offsets are its digit counts.
                    barrier()
                    counts, fronts = _read_counts(storage.offsets, tid, threads, cls.num_warps, bins, tracked)
                    if count_callback is not None:
                        count_callback(counts)
                prefix = scan.exclusive(counts, ReductionOp.ADD, storage=storage.scan)
                _write_prefix(storage, prefix, counts, fronts, tid, bins, tracked, descending, threads * count)
                ranks = _add_offsets(ranks, digits, storage, tid, width, bins, count)
                output_prefix = Vector.from_elements(
                    [Int32(storage.prefix[(tid * tracked + i) % bins]) for i in range(tracked)]
                )
            else:
                # MEMOIZE keeps each raking segment in registers between the
                # upsweep and the downsweep; BASIC reads it again.
                tracked = (bins + threads - 1) // threads
                ranks, pass_prefix = _counter_ranks(
                    digits,
                    storage,
                    scan,
                    tid,
                    threads,
                    bins,
                    descending,
                    cls.algorithm is BlockRadixRankAlgorithm.MEMOIZE,
                    tracked,
                    with_prefix and len(passes) == 1,
                )
                if len(passes) > 1:
                    # _inverse reuses the counter bytes every thread just read.
                    barrier()
                else:
                    output_prefix = pass_prefix
            if len(passes) == 1:
                result = ranks if vector else ranks[0]
                barrier()
                return (result, output_prefix) if with_prefix else result
            _inverse(indices, ranks, storage, count)
            original_ranks = Vector.from_elements([Int32(_shared_load(storage.indices, p)) for p in positions])
            barrier()
        result = original_ranks if vector else original_ranks[0]
        barrier()
        return (result, output_prefix) if with_prefix else result
