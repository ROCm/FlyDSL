# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Blocked histogram ranks and warp-striped match-based radix ranks."""

import enum
import functools

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
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
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
        BASIC: Consume blocked tiles and build each bin predicate on demand.
            This is the default policy.
        MEMOIZE: Consume blocked tiles and cache all bin predicates in registers.
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


def _storage(dtype, threads, warp, count, match=False, masks=False, radix_bits=8):
    warps = (threads + warp - 1) // warp
    index_dtype = Uint16 if threads * count <= 65536 else Int32
    digit_dtype = Uint8 if radix_bits <= 8 else Uint16 if radix_bits <= 16 else Int32
    offset_dtype = Uint16 if warp * count < 65536 else Int32
    bins = 1 << radix_bits
    digit_words = (threads * count * digit_dtype.width + 63) // 64
    index_words = (threads * count * index_dtype.width + 63) // 64
    scratch_words = max(digit_words + index_words, warps * bins if masks else 1)
    Digits = Array[digit_dtype, threads * count]
    Indices = Array[index_dtype, threads * count]
    Masks = Array[Uint64, warps * bins if masks else 1]

    @Struct
    class Storage:
        scratch: Array[Uint64, scratch_words]
        histogram: Array[Int32, warps * bins if match else 1]
        offsets: Array[offset_dtype, warps * bins if match else 1]
        counts: Array[Int32, bins if match else 1]
        prefix: Array[Int32, bins]
        scan: BlockScan[Int32, threads].SharedStorage

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


@jit
def _initialize(storage, tid, threads, warps, bins):
    for i in range_constexpr((warps * bins + threads - 1) // threads):
        index = tid + i * threads
        if index < warps * bins:
            storage.histogram[index] = Int32(0)
            storage.offsets[index] = storage.offsets.dtype(0)
    barrier()


@jit
def _early_histogram(digits, storage, warp_id, bins, count):
    for i in range_constexpr(count):
        atomic_add(storage.histogram.ptr + warp_id * bins + digits[i], Int32(1))
    barrier()


@jit
def _read_counts(storage, tid, threads, warps, bins, tracked):
    result = []
    for i in range_constexpr(tracked):
        digit = tid * tracked + i
        total = Int32(0)
        if digit < bins:
            for w in range_constexpr(warps):
                total += storage.histogram[w * bins + digit]
            storage.counts[digit] = total
        result.append(total)
    barrier()
    return Vector.from_elements(result)


@jit
def _write_prefix(storage, prefix, tid, bins, tracked, descending, tile_items):
    for i in range_constexpr(tracked):
        digit = tid * tracked + i
        if digit < bins:
            count = Int32(storage.counts[digit])
            storage.prefix[digit] = tile_items - prefix[i] - count if const_expr(descending) else prefix[i]
    barrier()


@jit
def _match_rounds(digits, storage, tid, threads, width, bins, count, early, use_atomic_or):
    warp_id, lane = tid // width, tid % width
    remaining = threads - warp_id * width
    active = (remaining < width).select(remaining, Int32(width))
    result = []
    for i in range_constexpr(count):
        digit = digits[i]
        if const_expr(use_atomic_or):
            for b in range_constexpr((((threads + width - 1) // width) * bins + threads - 1) // threads):
                index = tid + b * threads
                if index < ((threads + width - 1) // width) * bins:
                    storage.masks[index] = Uint64(0)
            barrier()
            atomic_or(storage.masks.ptr + warp_id * bins + digit, Uint64(1) << lane)
            barrier()
            mask = Uint64(storage.masks[warp_id * bins + digit])
        else:
            mask = Uint64(0)
            for source in range_constexpr(width):
                peer = shuffle_idx(digit, (lane_id() // width) * width + source, width)
                match = (peer == digit) & (source < active)
                mask = mask | match.select(Uint64(1) << source, Uint64(0))
        local_rank = Int32(ctpop(mask & ((Uint64(1) << lane) - Uint64(1))))
        matches = Int32(ctpop(mask))
        previous = Int32(_shared_load(storage.offsets, warp_id * bins + digit))
        result.append(previous + local_rank)
        barrier()
        if local_rank == 0:
            storage.offsets[warp_id * bins + digit] = (previous + matches).to(storage.offsets.dtype)
            if const_expr(not early):
                storage.histogram[warp_id * bins + digit] = previous + matches
        barrier()
    return Vector.from_elements(result)


@jit
def _add_offsets(ranks, digits, storage, tid, width, bins, warps, count):
    result = []
    for i in range_constexpr(count):
        digit = digits[i]
        offset = Int32(storage.prefix[digit])
        for w in range_constexpr(warps):
            if w < tid // width:
                offset += storage.histogram[w * bins + digit]
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

    BASIC and MEMOIZE consume blocked tiles. BASIC forms a bin histogram on
    demand; MEMOIZE caches all bin predicates in registers before scanning.
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

    scan_type = BlockScan
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
        # Four-bit passes cap the scan work of BASIC/MEMOIZE; MATCH supports
        # the complete eight-bit histogram in one pass.
        step = radix_bits or (
            max(1, end_bit - begin_bit)
            if count_callback is not None or with_prefix
            else min(cls.configured_radix_bits, 8 if match else 4)
        )
        passes = list(range(begin_bit, end_bit, step)) or [begin_bit]
        if with_prefix and len(passes) > 1:
            raise ValueError("with_prefix requires one digit pass: radix_bits must cover end_bit - begin_bit")
        for pass_index, bit in enumerate(passes):
            bits = min(step, end_bit - bit)
            bins = 1 << bits
            digits = Vector.from_elements(
                [
                    (
                        _extract_digit(key, digit_extractor, bits)
                        if digit_extractor is not None
                        else _digit(key, bit, bits, False)
                    )
                    for key in current
                ]
            )
            if pass_index > 0:
                _publish(digits, positions, original_ranks, storage, count)
                digits = Vector.from_elements([Int32(_shared_load(storage.digits, p)) for p in positions])
                indices = Vector.from_elements([Int32(_shared_load(storage.indices, p)) for p in positions])
                barrier()
            if match:
                early = cls.algorithm is not BlockRadixRankAlgorithm.MATCH
                _initialize(storage, tid, threads, cls.num_warps, bins)
                if early:
                    _early_histogram(digits, storage, tid // width, bins, count)
                    counts = _read_counts(storage, tid, threads, cls.num_warps, bins, (bins + threads - 1) // threads)
                    if count_callback is not None:
                        count_callback(counts)
                ranks = _match_rounds(
                    digits,
                    storage,
                    tid,
                    threads,
                    width,
                    bins,
                    count,
                    early,
                    cls.algorithm is BlockRadixRankAlgorithm.MATCH_EARLY_COUNTS_ATOMIC_OR,
                )
                tracked = (bins + threads - 1) // threads
                if not early:
                    counts = _read_counts(storage, tid, threads, cls.num_warps, bins, tracked)
                    if count_callback is not None:
                        count_callback(counts)
                prefix = scan.exclusive(counts, ReductionOp.ADD, storage=storage.scan)
                _write_prefix(storage, prefix, tid, bins, tracked, descending, threads * count)
                ranks = _add_offsets(ranks, digits, storage, tid, width, bins, cls.num_warps, count)
                output_prefix = Vector.from_elements(
                    [Int32(storage.prefix[(tid * tracked + i) % bins]) for i in range(tracked)]
                )
            else:
                # MEMOIZE retains every bin predicate across the histogram
                # scans; BASIC constructs only the current bin's predicates.
                cached = (
                    [[Int32(digit == b) for digit in digits] for b in range(bins)]
                    if cls.algorithm is BlockRadixRankAlgorithm.MEMOIZE
                    else None
                )
                result, offset = [Int32(0)] * count, Int32(0)
                for b in range(bins):
                    digit = bins - 1 - b if descending else b
                    flags = cached[digit] if cached is not None else [Int32(key == digit) for key in digits]
                    prefix, total = scan.exclusive_with_aggregate(
                        sum(flags, Int32(0)), ReductionOp.ADD, storage=storage.scan
                    )
                    local = Int32(0)
                    for i in range(count):
                        result[i] = (flags[i] != 0).select(offset + prefix + local, result[i])
                        local += flags[i]
                    storage.prefix[digit] = offset
                    offset += total
                    barrier()
                ranks = Vector.from_elements(result)
                tracked = (bins + threads - 1) // threads
                output_prefix = Vector.from_elements(
                    [Int32(storage.prefix[(tid * tracked + i) % bins]) for i in range(tracked)]
                )
            if len(passes) == 1:
                result = ranks if vector else ranks[0]
                barrier()
                return (result, output_prefix) if with_prefix else result
            _inverse(indices, ranks, storage, count)
            original_ranks = Vector.from_elements([Int32(_shared_load(storage.indices, p)) for p in positions])
            barrier()
        result = original_ranks
        barrier()
        return result if vector else result[0]
