# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Sorted merge selection and adaptive radix selection of block tiles."""

import enum

from ....compiler import jit
from ....expr.gpu import barrier
from ....expr.numeric import Int32, Integer, Numeric
from ....expr.primitive import range_constexpr
from ....expr.typing import ReductionOp, Vector
from .._common import _linear_thread_id
from .._sorting import _pack, _unpack
from ._spec import BlockPrimitive, BlockTileAlgorithmMeta
from .merge_sort import BlockMergeSort
from .merge_sort import _storage as _merge_storage
from .radix_sort import _exchange_words, _ordered_bits
from .radix_sort import _storage as _radix_storage
from .scan import BlockScan

__all__ = [
    "BlockTopKAlgorithm",
    "BlockTopK",
]


class BlockTopKAlgorithm(enum.Enum):
    """Selection policies for :class:`BlockTopK`.

    Attributes:
        MERGE: Stably sort the tile and mark its first ``k`` outputs valid.
            Supports numeric order and custom strict comparators. This is the
            default policy.
        AIR: Refine a numeric radix threshold from most significant to least
            significant bits, then stably compact selected items. The selected
            prefix preserves input order and is not sorted by key.
    """

    MERGE = "merge"
    AIR = "air"

    Merge = MERGE
    Air = AIR


def _storage(dtype, threads, warp, count):
    return _merge_storage(dtype, threads, warp, count)


@jit
def _threshold(ordered, k, valid, scan, storage, count, width):
    """Refine the bucket containing item k, inspecting bits most significant first."""
    threshold = ordered.dtype(0)
    mask = ordered.dtype(0)
    remaining = k
    for bit in range_constexpr(width - 1, -1, -1):
        bit_mask = ordered.dtype(1 << bit)
        zero_count = Int32(0)
        for i in range_constexpr(count):
            in_bucket = ((ordered[i] & mask) == threshold) & (valid[i] != 0)
            zero_count += Int32(in_bucket & ((ordered[i] & bit_mask) == 0))
        unused, zeros = scan.exclusive_with_aggregate(zero_count, ReductionOp.ADD, storage=storage.scan)
        take_one = remaining > zeros
        threshold = take_one.select(threshold | bit_mask, threshold)
        remaining = take_one.select(remaining - zeros, remaining)
        mask = mask | bit_mask
        barrier()
    return threshold, remaining


class _BlockTopKMeta(BlockTileAlgorithmMeta):
    _supports_pairs = True
    _algorithms = BlockTopKAlgorithm
    _shared_storage = {BlockTopKAlgorithm.MERGE: _storage, BlockTopKAlgorithm.AIR: _radix_storage}

    def _default_algorithm_for(cls, target):
        return BlockTopKAlgorithm.MERGE

    def __call__(
        cls,
        keys,
        k: int | Integer,
        values=None,
        *,
        storage,
        largest: bool = True,
        num_valid: int | Integer | None = None,
        compare_op=None,
    ):
        """Implement :meth:`BlockTopK.__call__`."""
        cls._check(keys)
        total = cls.block_threads * cls.items_per_thread
        if isinstance(k, int) and not 0 <= k <= total:
            raise ValueError("k must be in [0, tile_items]")
        if num_valid is None:
            num_valid = total
        if isinstance(num_valid, int) and not 0 <= num_valid <= total:
            raise ValueError("num_valid must be in [0, tile_items]")
        if isinstance(k, int) and isinstance(num_valid, int) and k > num_valid:
            raise ValueError("k must not exceed num_valid")
        items, payload, vector = _unpack(keys, values, cls.value_dtype, cls.dtype)
        if payload is not None and cls.value_dtype is None:
            raise TypeError("specialize [(key_dtype, value_dtype), block_size, items_per_thread] for pairs")
        tid, count = _linear_thread_id(cls.block_size), cls.items_per_thread
        flags = [(tid * count + i < k) for i in range(count)]
        if cls.algorithm is BlockTopKAlgorithm.MERGE:
            dtype = (cls.dtype, cls.value_dtype) if cls.value_dtype is not None else cls.dtype
            sort = BlockMergeSort[dtype, cls.block_size, count]
            result = sort(
                keys, values, storage=storage, descending=largest, compare_op=compare_op, valid_items=num_valid
            )
            valid = Vector.from_elements(flags) if vector else flags[0]
            return (*result, valid) if payload is not None else (result, valid)
        if compare_op is not None:
            raise TypeError("AIR uses numeric radix order; custom comparators require MERGE")
        if not issubclass(cls.dtype, Numeric):
            raise TypeError("AIR requires numeric scalar keys; composite keys with a comparator require MERGE")
        ordered = Vector.from_elements([~_ordered_bits(key) if largest else _ordered_bits(key) for key in items])
        validity = Vector.from_elements([Int32(tid * count + i < num_valid) for i in range(count)])
        scan = cls.scan_type[Int32, cls.block_size]
        threshold, remaining = _threshold(ordered, Int32(k), validity, scan, storage, count, ordered.dtype.width)
        equal = Vector.from_elements([Int32((ordered[i] == threshold) & (validity[i] != 0)) for i in range(count)])
        duplicate_rank = scan.exclusive(equal, ReductionOp.ADD, storage=storage.scan)
        barrier()
        selected = Vector.from_elements(
            [
                Int32(
                    (validity[i] != 0)
                    & ((ordered[i] < threshold) | ((ordered[i] == threshold) & (duplicate_rank[i] < remaining)))
                )
                for i in range(count)
            ]
        )
        prefix, selected_count = scan.exclusive_with_aggregate(selected, ReductionOp.ADD, storage=storage.scan)
        barrier()
        ranks = Vector.from_elements(
            [(selected[i] != 0).select(prefix[i], selected_count + tid * count + i - prefix[i]) for i in range(count)]
        )
        positions = [tid * count + i for i in range(count)]
        items = _exchange_words(items, ranks, positions, storage)
        if payload is not None:
            payload = _exchange_words(payload, ranks, positions, storage)
        result = _pack(items, payload, vector, keys, values)
        valid = Vector.from_elements(flags) if vector else flags[0]
        return (*result, valid) if payload is not None else (result, valid)


class BlockTopK(BlockPrimitive, metaclass=_BlockTopKMeta):
    """Select a blocked prefix of ``k`` keys, optionally carrying values.

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

    Input and output use blocked order: thread ``t`` owns flattened positions
    ``t * items_per_thread + i``. ``k`` and ``num_valid`` count flattened items
    and may be uniform runtime scalars satisfying
    ``0 <= k <= num_valid <= block_threads * items_per_thread``. Only outputs
    marked valid belong to the selection.

    MERGE, the default policy, produces a sorted prefix and supports a strict
    weak comparator. Numeric comparison uses ``<`` or ``>``, without a special
    NaN order. AIR refines a radix threshold, then stably compacts selected
    items in original input order; it does not sort the tile. Both select ties
    in original blocked order. AIR compares signed and floating keys by ordered
    bit representations: signed zeros are equivalent, and NaNs follow their
    transformed bit order, including sign and payload. Original key bits remain
    unchanged. Supply a comparator to MERGE when another NaN order is required.

    All block threads must participate with uniform options and a common
    allocation of the specialization's ``SharedStorage``. Synchronize the block
    before reusing storage for another collective.

    Attributes:
        dtype: Specialized key element type.
        value_dtype: Specialized payload type, or ``None`` for keys only.
        block_size: Specialized ``(x, y, z)`` shape.
        block_threads: Number of participating threads.
        items_per_thread: Number of input keys supplied by each thread.
        algorithm: Selected MERGE or AIR policy.
        SharedStorage: Shared-memory Struct type required by the specialization.

    Examples:
        # T0..T3 supply keys={[4,1], [7,3], [6,2], [8,5]}; T4..T63 supply [0,0].
        # Thread t supplies payload values=[2*t,2*t+1].
        P = fx.coop.BlockTopK[fx.Int32, 64, 2, fx.coop.BlockTopKAlgorithm.MERGE]
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(P.SharedStorage).peek()
        y, valid = P(keys, 3, storage=storage, largest=True)
        # The first three blocked slots contain [8,7,6]; only those slots have valid=True.
        # All other output values are unspecified. AIR would retain input order [7,6,8].

        Q = fx.coop.BlockTopK[(fx.Int32, fx.Int32), 64, 2, fx.coop.BlockTopKAlgorithm.MERGE]
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

    @classmethod
    def __call__(
        cls,
        keys,
        k: int | Integer,
        values=None,
        *,
        storage,
        largest: bool = True,
        num_valid: int | Integer | None = None,
        compare_op=None,
    ):
        """Select the largest or smallest ``k`` valid keys from a blocked tile.

        MERGE returns a sorted prefix; AIR stably compacts selected inputs without
        sorting. Equal keys are selected in original blocked order. All threads
        must participate and synchronize before reusing storage.

        Args:
            keys: Key value matching the specialized key dtype and item count.
            k: Uniform Python integer or runtime scalar counting selected elements
                across the whole block. Must satisfy ``0 <= k <= num_valid``;
                runtime bounds are the caller's responsibility.
            values: Optional payload with the same item count and single-item/sequence
                form as ``keys``; its element shape may differ. Must match the
                pair specialization's value dtype.
            storage: Shared instance of the specialization's ``SharedStorage``.
            largest: Compile-time boolean selecting the largest keys when ``True``
                or the smallest keys when ``False``.
            num_valid: Uniform Python integer or runtime scalar counting valid
                leading input elements in flattened blocked order. Defaults to the
                full tile and must lie in ``[0, block_threads * items_per_thread]``.
            compare_op: Optional strict weak ordering ``compare_op(a, b)`` for
                MERGE, true when ``a`` precedes ``b`` in ascending order. Required
                for record keys. AIR rejects custom comparators and uses numeric
                radix order; see :class:`BlockTopK` for floating-point behavior.

        Returns:
            ``(keys, valid)`` or ``(keys, values, valid)`` when carrying payloads.
            ``valid`` marks exactly the first ``k`` flattened output positions. Key and payload results retain their item counts and complete element types. Only
            entries marked valid belong to the result.

        Raises:
            TypeError: The operator must be specialized with ``BlockTopK[...]`` before use.
        """
        if cls.block_threads is None:
            raise TypeError(
                f"specialize {cls.__name__} first, e.g. {cls.__name__}[dtype, block_size, items_per_thread]"
            )
        return type(cls).__call__(
            cls, keys, k, values, storage=storage, largest=largest, num_valid=num_valid, compare_op=compare_op
        )

    @classmethod
    def max_keys(
        cls,
        keys,
        k: int | Integer,
        num_valid: int | Integer | None = None,
        *,
        storage,
    ):
        """Select the largest ``k`` keys using the specialized policy.

        MERGE returns sorted keys; AIR preserves selected input order without
        sorting. These convenience methods use the policy's numeric order and do
        not accept a comparator. All threads must participate and synchronize
        before reusing storage.

        Args:
            keys: Key value matching the specialization and supporting numeric ordering.
            k: Uniform selected-item count across the entire block, with
                ``0 <= k <= num_valid``. May be a Python integer or runtime scalar.
            num_valid: Uniform number of valid leading input elements in blocked
                order, or ``None`` for the full tile. Must be in
                ``[0, block_threads * items_per_thread]``; enforce runtime bounds.
            storage: Shared instance of the specialization's ``SharedStorage``.

        Returns:
            ``(keys, valid)``, with ``valid`` marking the first ``k`` flattened output positions. Key and payload results
            retain their item counts and complete element types.
            Only marked positions belong to the selected result.
        """
        return cls(keys, k, storage=storage, largest=True, num_valid=num_valid)

    @classmethod
    def min_keys(
        cls,
        keys,
        k: int | Integer,
        num_valid: int | Integer | None = None,
        *,
        storage,
    ):
        """Select the smallest ``k`` keys using the specialized policy.

        MERGE returns sorted keys; AIR preserves selected input order without
        sorting. These convenience methods use the policy's numeric order and do
        not accept a comparator. All threads must participate and synchronize
        before reusing storage.

        Args:
            keys: Key value matching the specialization and supporting numeric ordering.
            k: Uniform selected-item count across the entire block, with
                ``0 <= k <= num_valid``. May be a Python integer or runtime scalar.
            num_valid: Uniform number of valid leading input elements in blocked
                order, or ``None`` for the full tile. Must be in
                ``[0, block_threads * items_per_thread]``; enforce runtime bounds.
            storage: Shared instance of the specialization's ``SharedStorage``.

        Returns:
            ``(keys, valid)``, with ``valid`` marking the first ``k`` flattened output positions. Key and payload results
            retain their item counts and complete element types.
            Only marked positions belong to the selected result.
        """
        return cls(keys, k, storage=storage, largest=False, num_valid=num_valid)

    @classmethod
    def max_pairs(
        cls,
        keys,
        values,
        k: int | Integer,
        num_valid: int | Integer | None = None,
        *,
        storage,
    ):
        """Select the largest ``k`` keys using the specialized policy.

        MERGE returns sorted keys; AIR preserves selected input order without
        sorting. These convenience methods use the policy's numeric order and do
        not accept a comparator. All threads must participate and synchronize
        before reusing storage.

        Args:
            keys: Key value matching the specialization and supporting numeric ordering.
            values: Payload tile matching the key item count and single-item/sequence
                form, with the specialized value dtype. Element shapes may differ.
                Requires a key/value pair specialization.
            k: Uniform selected-item count across the entire block, with
                ``0 <= k <= num_valid``. May be a Python integer or runtime scalar.
            num_valid: Uniform number of valid leading input elements in blocked
                order, or ``None`` for the full tile. Must be in
                ``[0, block_threads * items_per_thread]``; enforce runtime bounds.
            storage: Shared instance of the specialization's ``SharedStorage``.

        Returns:
            ``(keys, values, valid)``, with ``valid`` marking the first ``k`` flattened output positions. Result
            values retain the input item count and shape.
            Only marked positions belong to the selected result.
        """
        return cls(keys, k, values, storage=storage, largest=True, num_valid=num_valid)

    @classmethod
    def min_pairs(
        cls,
        keys,
        values,
        k: int | Integer,
        num_valid: int | Integer | None = None,
        *,
        storage,
    ):
        """Select the smallest ``k`` keys using the specialized policy.

        MERGE returns sorted keys; AIR preserves selected input order without
        sorting. These convenience methods use the policy's numeric order and do
        not accept a comparator. All threads must participate and synchronize
        before reusing storage.

        Args:
            keys: Key value matching the specialization and supporting numeric ordering.
            values: Payload tile matching the key item count and single-item/sequence
                form, with the specialized value dtype. Element shapes may differ.
                Requires a key/value pair specialization.
            k: Uniform selected-item count across the entire block, with
                ``0 <= k <= num_valid``. May be a Python integer or runtime scalar.
            num_valid: Uniform number of valid leading input elements in blocked
                order, or ``None`` for the full tile. Must be in
                ``[0, block_threads * items_per_thread]``; enforce runtime bounds.
            storage: Shared instance of the specialization's ``SharedStorage``.

        Returns:
            ``(keys, values, valid)``, with ``valid`` marking the first ``k`` flattened output positions. Result
            values retain the input item count and shape.
            Only marked positions belong to the selected result.
        """
        return cls(keys, k, values, storage=storage, largest=False, num_valid=num_valid)
