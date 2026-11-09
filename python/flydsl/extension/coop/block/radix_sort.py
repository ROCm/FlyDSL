# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Stable single-bit and multi-bit LSD radix sorting through block scans."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.numeric import Int32, Numeric, Uint8, Uint16, Uint32, Uint64, Uint128
from ....expr.primitive import add_offset, const_expr, inttoptr, ptr_load, ptrtoint, range_constexpr
from ....expr.struct import Struct, is_struct_type
from ....expr.typing import Array, Pointer, ReductionOp, Vector
from .._common import _linear_thread_id
from .._sorting import _pack, _unpack
from .._values import _flatten_record, _item_dtype, _leaf_fields, _normalize_value, _rebuild_record
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta, _Public
from .scan import BlockScan

__all__ = [
    "BlockRadixSortAlgorithm",
    "BlockRadixSort",
]

_UNSIGNED = {1: Uint8, 8: Uint8, 16: Uint16, 32: Uint32, 64: Uint64, 128: Uint128}


def _ordered_bits(key, transform=True):
    dtype = key.dtype
    unsigned = _UNSIGNED[dtype.width]
    if dtype.width == 1:
        return key.select(Uint8(1), Uint8(0))
    bits = key.bitcast(unsigned)
    if not transform or (not dtype.is_float and not dtype.signed):
        return bits
    sign = unsigned(1 << (dtype.width - 1))
    if dtype.is_float:
        bits = (key == dtype(0)).select(unsigned(0), bits)
        return ((bits & sign) != unsigned(0)).select(~bits, bits ^ sign)
    return bits ^ sign


def _fields(key, decomposer):
    if decomposer is None and not isinstance(key, Numeric):
        raise TypeError("composite radix keys require a decomposer returning numeric scalar fields")
    fields = (key,) if decomposer is None else decomposer(key)
    if not isinstance(fields, (tuple, list)) or not fields:
        raise TypeError("decomposer must return a nonempty tuple/list of numeric fields, most significant first")
    fields = tuple(_normalize_value(field) for field in fields)
    for field in fields:
        if not isinstance(field, Numeric) or field.dtype.width not in _UNSIGNED:
            raise TypeError("radix fields must be supported numeric scalar values")
    return fields


def _bit_range(dtype, begin_bit, end_bit):
    width = dtype if isinstance(dtype, int) else dtype.width
    end_bit = width if end_bit is None else end_bit
    if not isinstance(begin_bit, int) or not isinstance(end_bit, int) or not 0 <= begin_bit <= end_bit <= width:
        raise ValueError("expected 0 <= begin_bit <= end_bit <= decomposed key width")
    return end_bit


def _digit(key, begin, bits, transform=True, decomposer=None):
    result = Uint32(0)
    offset = 0
    for field in reversed(_fields(key, decomposer)):
        lo, hi = max(begin, offset), min(begin + bits, offset + field.dtype.width)
        if lo < hi:
            part = _ordered_bits(field, transform) >> (lo - offset)
            part = part & part.dtype((1 << (hi - lo)) - 1)
            result = result | (part.to(Uint32) << (lo - begin))
        offset += field.dtype.width
    return result.to(Int32)


class BlockRadixSortAlgorithm(enum.Enum):
    """Default digit-width policies for :class:`BlockRadixSort`.

    An explicit ``radix_bits`` overrides the policy's default digit width. An
    LSD radix sort's result does not depend on the digit width, so a backend may
    rank with a different width where that is faster: gfx9 picks 1- to 3-bit
    digits for both policies.

    Attributes:
        RADIX_SCAN: Perform one stable binary partition per bit by default.
        RADIX_DIGIT: Use four-bit digits by default, scanning per-thread
            histograms and scattering once per digit. The histogram scan
            iterates over bins at runtime.
    """

    RADIX_SCAN = "radix_scan"
    RADIX_DIGIT = "radix_digit"

    RadixScan = RADIX_SCAN
    RadixDigit = RADIX_DIGIT


def _columns(items):
    """Split items into 32-bit word columns; return the columns and their inverse."""
    first = items[0]
    if isinstance(first, Pointer):
        columns, rebuild = _columns([ptrtoint(item).to(Uint64) for item in items])
        return columns, lambda c: [type(item)(inttoptr(item.type, a)) for item, a in zip(items, rebuild(c))]
    if isinstance(first, (Vector,)) or is_struct_type(_item_dtype(first)):
        if isinstance(first, Vector):
            parts = [_columns([item[i] for item in items]) for i in range(first.numel)]
        else:
            fields = [_flatten_record(item) for item in items]
            parts = [_columns([f[i] for f in fields]) for i in range(len(fields[0]))]

        def rebuild_parts(c):
            leaves, at = [], 0
            for columns, rebuild in parts:
                leaves.append(rebuild(c[at : at + len(columns)]))
                at += len(columns)
            if not isinstance(first, Vector):
                dtype = _item_dtype(first)
                return [_rebuild_record(dtype, [leaf[i] for leaf in leaves], items[i]) for i in range(len(items))]
            return [
                Vector(
                    Vector.from_elements([leaf[i] for leaf in leaves], dtype=first.dtype),
                    shape=first.shape,
                    dtype=first.dtype,
                )
                for i in range(len(items))
            ]

        return [column for columns, _ in parts for column in columns], rebuild_parts
    dtype = _item_dtype(first)
    unsigned = _UNSIGNED[dtype.width]
    if dtype.width == 1:
        return [[item.select(Uint32(1), Uint32(0)) for item in items]], lambda c: [
            w.to(unsigned).to(dtype) for w in c[0]
        ]
    columns = [[(item.bitcast(unsigned) >> bit).to(Uint32) for item in items] for bit in range(0, dtype.width, 32)]

    def rebuild_scalar(c):
        out = []
        for i in range(len(items)):
            bits = c[0][i].to(unsigned)
            for k in range(1, len(c)):
                bits = bits | (c[k][i].to(unsigned) << (32 * k))
            out.append(bits.bitcast(dtype))
        return out

    return columns, rebuild_scalar


def _exchange(columns, ranks, tid, storage, threads, count, striped):
    """Scatter every word column to *ranks* at once; read back blocked or striped positions.

    Each column owns a plane of ``storage.words``, so keys and payloads share one
    barrier. Columns beyond the planes take further rounds. No barrier trails the
    read-back: the next pass synchronizes in its scan before it scatters, and the
    sort synchronizes once after its last read-back.
    """
    tile = threads * count
    planes = storage.words.size // tile
    out = []
    for first in range(0, len(columns), planes):
        if first:
            barrier()  # the previous round's reads precede this round's scatter
        group = columns[first : first + planes]
        # Item-major order keeps an item's plane stores adjacent, so a backend can
        # pair them (gfx9: ds_write2st64_b32).
        for i in range(count):
            for c, column in enumerate(group):
                storage.words[c * tile + ranks[i]] = column[i]
        barrier()
        for c in range(len(group)):
            if striped:
                out.append([Uint32(storage.words[c * tile + tid + i * threads]) for i in range(count)])
            else:
                row = ptr_load(add_offset(storage.words.ptr, c * tile + tid * count), Vector[Uint32, count])
                out.append([Uint32(row[i]) for i in range(count)])
    return out


# Scatter planes beyond this many bytes go in further rounds.
_PLANE_BUDGET = 32 * 1024


def _words_per_item(dtype):
    """32-bit word columns of one item (0 when unknown)."""
    words = 0
    try:
        for _, leaf in _leaf_fields(dtype):
            if issubclass(leaf, Pointer):
                words += 2
            elif issubclass(leaf, Vector):
                return 0
            else:
                words += max(1, leaf.width // 32)
    except TypeError:
        return 0
    return words


def _storage(dtype, threads, warp, count):
    keys, values = dtype if isinstance(dtype, tuple) else (dtype, None)
    words = _words_per_item(keys) + (_words_per_item(values) if values is not None else 0)
    tile = threads * count
    planes = max(1, min(words or 1, _PLANE_BUDGET // (4 * tile)))
    return Struct[
        "words" : Array[Uint32, planes * tile, 16],
        "scan" : BlockScan[Int32, threads].SharedStorage,
    ]


def _histogram_scan(scan, histogram, storage):
    return scan.exclusive_with_aggregate(histogram, ReductionOp.ADD, storage=storage.scan)


@jit
def _digit_ranks(digits, scan, storage, descending, bins):
    """Stable multi-bin distribution with a runtime loop over digit bins."""
    count = digits.numel
    ranks = Vector.filled(count, 0, Int32)
    offset = Int32(0)
    for bin in range(bins):
        digit = bins - 1 - bin if const_expr(descending) else bin
        flags = Vector.from_elements([Int32(key == digit) for key in digits])
        histogram = flags.reduce(ReductionOp.ADD)
        prefix, total = _histogram_scan(scan, histogram, storage)
        local = Int32(0)
        result = []
        for i in range_constexpr(count):
            result.append((flags[i] != 0).select(offset + prefix + local, ranks[i]))
            local += flags[i]
        ranks = Vector.from_elements(result)
        offset += total
        barrier()
    return ranks


class _BlockRadixSortMeta(BlockTileAlgorithmMeta):
    _supports_pairs = True
    _algorithms = BlockRadixSortAlgorithm
    _shared_storage = {algorithm: _storage for algorithm in BlockRadixSortAlgorithm}

    def _default_algorithm_for(cls, target):
        return BlockRadixSortAlgorithm.RADIX_SCAN

    def __call__(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        decomposer=None,
        radix_bits: int | None = None,
        blocked_to_striped: bool = False,
    ):
        """Implement :meth:`BlockRadixSort.__call__`."""
        return cls._sort(
            keys, values, storage, descending, begin_bit, end_bit, decomposer, radix_bits, blocked_to_striped
        )


class BlockRadixSort(BlockPrimitive, metaclass=_BlockRadixSortMeta):
    """Stably radix-sort blocked keys with optional payloads and striped output.

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

    All tile items participate.

    ``RADIX_SCAN`` uses one-bit passes and ``RADIX_DIGIT`` defaults to four-bit
    passes; ``radix_bits`` overrides either policy. Signed integers flip their
    sign bit. Floating keys invert negative bit patterns and flip the sign bit
    of nonnegative patterns, with signed zeros normalized to equal sort keys.
    NaNs follow the resulting bit order, including sign and payload; no separate
    NaN placement rule is imposed. Equivalent transformed keys retain their
    original blocked order in either direction. Original key bits are preserved.

    A decomposer returns numeric fields in most-significant-first order. Bit
    ranges address the concatenation of their transformed representations,
    with bit zero in the last field. Fields may be Boolean or 8-, 16-, 32-,
    64-, or 128-bit Numeric values supported by the target. Partial bit ranges
    sort only those transformed bits and preserve order for equal subkeys.

    All threads must call with uniform options and a common allocation of the
    specialization's ``SharedStorage``. Synchronize the block before reusing
    storage. Blocked position ``t * items_per_thread + i`` becomes striped
    position ``t + i * block_threads`` when striped output is requested.

    Attributes:
        dtype: Specialized key element type.
        value_dtype: Specialized payload type, or ``None``.
        block_size: Specialized ``(x, y, z)`` shape.
        block_threads: Number of participating threads.
        items_per_thread: Number of input keys per thread.
        algorithm: Policy providing the default digit width.
        SharedStorage: Shared-memory Struct type required by the specialization.

    Examples:
        # T0..T3 supply keys={[3,-1], [2,-3], [0,2], [-1,1]}; T4..T63 supply [4,4].
        # Thread t supplies payload values=[2*t, 2*t+1].
        P = fx.coop.BlockRadixSort[fx.Int32, 64, 2]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        y = P(keys, storage=storage)
        # T0..T3 receive {[-3,-1], [-1,0], [1,2], [2,3]}; T4..T63 receive [4,4].

        # Sort descending by the lowest two transformed bits and return striped pairs.
        Q = fx.coop.BlockRadixSort[(fx.Int32, fx.Int32), 64, 2, fx.coop.BlockRadixSortAlgorithm.RADIX_DIGIT]
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

    scan_type = _Public("BlockScan")

    @classmethod
    def __call__(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        decomposer=None,
        radix_bits: int | None = None,
        blocked_to_striped: bool = False,
    ):
        """Stably sort a complete blocked tile by transformed key bits.

        See :class:`BlockRadixSort` for signed, floating, and decomposed key order.
        All block threads must participate; synchronize before reusing storage.

        Args:
            keys: Key value matching the specialized key dtype and item count.
            values: Optional payload with the same item count and single-item/sequence
                form as ``keys``; its element shape may differ. Must match the
                pair specialization's value dtype.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean selecting descending transformed order.
            begin_bit: Python integer giving the first included transformed bit.
            end_bit: Python integer giving the first excluded bit, or ``None`` for
                the complete decomposed width. Must satisfy
                ``0 <= begin_bit <= end_bit <= decomposed_width``.
            decomposer: Optional callable exposing a nonempty sequence of numeric key fields, most significant
                first. Required when keys need field extraction; fields determine ordering while original keys are
                returned.
            radix_bits: Python integer in ``[1, 30]`` giving bits per pass, or
                ``None`` for the policy default. Bin work grows exponentially with
                the actual pass width.
            blocked_to_striped: Compile-time boolean requesting striped output,
                where thread ``t`` receives sorted positions ``t + i * block_threads``.

        Returns:
            Sorted keys, or ``(keys, values)`` for pairs, in blocked or requested
            striped layout. Results retain the item count and complete element types. An empty bit range
            preserves input order while still applying any requested layout change.

        Raises:
            TypeError: The operator must be specialized with ``BlockRadixSort[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(
            cls,
            keys,
            values,
            storage=storage,
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            radix_bits=radix_bits,
            blocked_to_striped=blocked_to_striped,
        )

    @classmethod
    def _sort(cls, keys, values, storage, descending, begin_bit, end_bit, decomposer, radix_bits, striped):
        cls._check(keys)
        items, payload, vector = _unpack(keys, values, cls.value_dtype, cls.dtype)
        if payload is not None and cls.value_dtype is None:
            raise TypeError("specialize [(key_dtype, value_dtype), block_size, items_per_thread] for pairs")
        end_bit = _bit_range(sum(field.dtype.width for field in _fields(items[0], decomposer)), begin_bit, end_bit)
        if radix_bits is None:
            radix_bits = 1 if cls.algorithm is BlockRadixSortAlgorithm.RADIX_SCAN else 4
        if not isinstance(radix_bits, int) or radix_bits < 1:
            raise ValueError("radix_bits must be a positive Python int")
        if radix_bits >= 31:
            raise ValueError("radix digit space must fit signed 32-bit bin counters (radix_bits < 31)")
        count = cls.items_per_thread
        tid = _linear_thread_id(cls.block_size)
        scan = cls.scan_type[Int32, cls.block_size]
        passes = list(range(begin_bit, end_bit, radix_bits))
        if not passes and not striped:
            return _pack(items, payload, vector, keys, values)
        key_columns, rebuild_keys = _columns(items)
        value_columns, rebuild_values = _columns(payload) if payload is not None else ([], None)
        # An empty bit range still moves the tile once when striped output is requested.
        for index, bit in enumerate(passes or [None]):
            if bit is None:
                ranks = [tid * count + i for i in range(count)]
            else:
                bits = min(radix_bits, end_bit - bit)
                digits = Vector.from_elements([_digit(item, bit, bits, True, decomposer) for item in items])
                if bits == 1:
                    flags = Vector.from_elements(
                        [Int32((digit != 0) if descending else (digit == 0)) for digit in digits]
                    )
                    prefix, total = scan.exclusive_with_aggregate(flags, ReductionOp.ADD, storage=storage.scan)
                    ranks = Vector.from_elements(
                        [(flags[i] != 0).select(prefix[i], total + tid * count + i - prefix[i]) for i in range(count)]
                    )
                else:
                    ranks = _digit_ranks(digits, scan, storage, descending, 1 << bits)
            # The last pass reads the requested layout straight out of its scatter.
            last = index == max(len(passes), 1) - 1
            columns = _exchange(
                key_columns + value_columns, ranks, tid, storage, cls.block_threads, count, striped and last
            )
            key_columns, value_columns = columns[: len(key_columns)], columns[len(key_columns) :]
            items = rebuild_keys(key_columns)
        # Every thread has read the tile back before any thread returns.
        barrier()
        if payload is not None:
            payload = rebuild_values(value_columns)
        return _pack(items, payload, vector, keys, values)

    @classmethod
    def sort_blocked_to_striped(
        cls,
        keys,
        values=None,
        *,
        storage,
        descending: bool = False,
        begin_bit: int = 0,
        end_bit: int | None = None,
        decomposer=None,
        radix_bits: int | None = None,
    ):
        """Stably sort blocked input and distribute the result in striped order.

        Thread ``t`` receives sorted positions ``t + i * block_threads``. Ordering
        follows :class:`BlockRadixSort`; all threads must participate and synchronize
        before reusing the shared allocation.

        Args:
            keys: Key value matching the specialization.
            values: Optional payload matching the key item count and single-item/
                sequence form, with the specialized value dtype. Element shapes may differ.
            storage: Shared instance of the specialization's ``SharedStorage``.
            descending: Compile-time boolean selecting descending key order.
            begin_bit: Inclusive bit offset in the sortable representation.
            end_bit: Exclusive bit offset, or ``None`` for all remaining bits.
            decomposer: Optional key-field decomposition; see ``__call__``.
            radix_bits: Optional compile-time digit width; see ``__call__``.

        Returns:
            Striped sorted keys, or ``(keys, values)`` for pairs. Results retain the item count and complete element types.
        """
        return cls(
            keys,
            values,
            storage=storage,
            descending=descending,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            radix_bits=radix_bits,
            blocked_to_striped=True,
        )

    @classmethod
    def sort_descending(
        cls,
        keys,
        values=None,
        *,
        storage,
        begin_bit: int = 0,
        end_bit: int | None = None,
        decomposer=None,
        radix_bits: int | None = None,
        blocked_to_striped: bool = False,
    ):
        """Shorthand for ``BlockRadixSort[...](keys, values, descending=True, ...)``."""
        return cls(
            keys,
            values,
            storage=storage,
            descending=True,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            radix_bits=radix_bits,
            blocked_to_striped=blocked_to_striped,
        )

    @classmethod
    def sort_descending_blocked_to_striped(
        cls,
        keys,
        values=None,
        *,
        storage,
        begin_bit: int = 0,
        end_bit: int | None = None,
        decomposer=None,
        radix_bits: int | None = None,
    ):
        """Shorthand for ``sort_blocked_to_striped(keys, values, descending=True, ...)``."""
        return cls.sort_blocked_to_striped(
            keys,
            values,
            storage=storage,
            descending=True,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            radix_bits=radix_bits,
        )
