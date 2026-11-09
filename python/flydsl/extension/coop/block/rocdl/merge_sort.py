# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockMergeSort``: the block's runs come from a bitonic network in registers.

The portable sort forms one run per thread and merges runs through shared
memory, every pass a binary search per item. On gfx9 the calls with 32-bit
integer keys in their numeric order (no comparator, every slot valid, a
power-of-two item count, at most one 32-bit payload word) instead run a
bitonic network over blocked positions ``p = tid * count + i``:

- **Inside a warp** a stage pairs registers of one thread (static) or lanes
  ``xor 1, 2, 4, 8`` (DPP), ``xor 16`` (``ds_swizzle``) or ``xor 32``
  (``ds_bpermute``): no shared memory and no barrier.
- **Across warps** one shared-memory exchange serves up to two stages: each
  thread reads its whole group and runs those stages locally. Slots are
  item-major ("striped"), so a wave's lanes touch consecutive words; blocked
  slots put them ``count`` words apart.

With a power-of-two warp count the network spans the block and the merge
passes have nothing left to do; otherwise it sorts each warp's tile and the
portable passes merge the warps.

The result is the portable one bit for bit. Key-only integer keys need no
tie-break -- equal keys are equal bits -- so a compare-exchange is
``min``/``max``. With payloads each key carries its input position as the
tie (complemented for descending order), one 64-bit compare orders the pair,
and the order is total: the network's result is the stable sort. Payloads
never move during the network; they are gathered by position at the end.
Every value handed back is a ``min``/``max``/select or a load, never a bare
DPP move a caller's arithmetic could fold.
"""

from ....._mlir.dialects import gpu
from .....expr import arith as _arith
from .....expr.gpu import barrier
from .....expr.numeric import Int32, Int64, Integer, Numeric, Uint32, Uint64
from ...._dispatch import override
from ..._values import _item_dtype, _record_cast, _shared_load, _shared_store
from ...warp.rocdl._common import _is_gfx9
from ...warp.rocdl.bitonic_sort import _xor
from .. import merge_sort as _portable

__all__ = ["BlockMergeSort"]


def _is_word(value):
    return isinstance(value, Numeric) and value.dtype.width == 32


def _covers(items, payload, count, compare_op, valid_items):
    """Integer 32-bit keys in their numeric order, every slot valid, a power-of-two item count."""
    if compare_op is not None or valid_items is not None or count & (count - 1):
        return False
    if not (_is_word(items[0]) and issubclass(items[0].dtype, Integer)):
        return False
    return payload is None or _is_word(payload[0])


def _ordered(key, tie):
    """One 64-bit integer ordered as (key, tie): the key in its own signedness, the tie unsigned below it."""
    wide = Int64 if key.dtype.signed else Uint64
    return (key.to(wide) << 32) | tie.to(Uint64).to(wide)


def _select(first, lhs, rhs):
    """``first ? lhs : rhs`` per component, for a trace-time bool or a runtime predicate."""
    if isinstance(first, bool):
        return lhs if first else rhs
    return tuple(first.select(a, b) for a, b in zip(lhs, rhs))


class _Order:
    """Compare-exchange of network elements: ``(key,)`` or ``(key, tie)`` tuples."""

    def __init__(self, descending):
        self.descending = descending

    def precedes(self, x, y):
        wx, wy = _ordered(*x), _ordered(*y)
        return (wx > wy) if self.descending else (wx < wy)

    def pair(self, a, b):
        """``(earlier, later)`` of two elements."""
        if len(a) == 1:
            x, y = a[0], b[0]
            lo, hi = x.dtype(_arith.min(x, y)), x.dtype(_arith.max(x, y))
            return ((hi,), (lo,)) if self.descending else ((lo,), (hi,))
        b_first = self.precedes(b, a)
        return _select(b_first, b, a), _select(b_first, a, b)

    def exchange(self, mine, partner, first):
        """The element this position keeps: the earlier of the pair when *first*, else the later."""
        if len(mine) == 1:
            earlier, later = self.pair(mine, partner)
            return _select(first, earlier, later)
        take = self.precedes(partner, mine) == first
        return _select(take, partner, mine)


def _in_warp_stage(elems, position, count, span, d, order):
    """One network stage whose pairs lie in one logical warp: in-thread (d < count) or across lanes."""
    out = list(elems)
    for i in range(count):
        if d < count:
            j = i ^ d
            if j < i:
                continue
            # i, the lower position, keeps the earlier entry when its span bit is clear.
            first = (i & span) == 0 if span < count else (position(i) & span) == 0
            earlier, later = order.pair(elems[i], elems[j])
            out[i], out[j] = _select(first, earlier, later), _select(first, later, earlier)
            continue
        p = position(i)
        first = ((p & span) == 0) == ((p & d) == 0)
        partner = tuple(_xor(c, d // count) for c in elems[i])
        out[i] = order.exchange(elems[i], partner, first)
    return out


def _slot(position, count, threads):
    """Item-major shared slot of a tile position: a wave's lanes land on consecutive words."""
    return (position % count) * threads + position // count


def _pick(slots, bits):
    """``slots[sum(bits[j] << j)]`` for runtime bits."""
    level = list(slots)
    for j in range(len(bits) - 1, -1, -1):
        half = len(level) // 2
        level = [_select(bits[j], level[g + half], level[g]) for g in range(half)]
    return level[0]


def _group_stage(elems, storage, tid, count, threads, span, distances, order, first_exchange):
    """Network stages whose distances cross warps, through one shared-memory exchange.

    Every thread publishes its elements, then reads the ``2**k`` entries of its
    group (positions that differ only in the *distances* bits), runs the ``k``
    stages on them locally and keeps its own. Positions of a group share the
    span bit, so the direction is the same for all of them -- and for the
    whole wave, since the distances are warp-index bits.
    """
    keyed = len(elems[0]) == 2
    if not first_exchange:
        barrier()  # the previous exchange's reads come first
    for i in range(count):
        slot = _slot(tid * count + i, count, threads)
        _shared_store(storage.slots, slot, elems[i][0])
        if keyed:
            _shared_store(storage.values, slot, elems[i][1].bitcast(storage.values.dtype))
    barrier()
    k = len(distances)
    ascending = ((tid * count) & span) == 0
    out = []
    for i in range(count):
        p = tid * count + i
        base = p & ~sum(distances)
        group = []
        for g in range(1 << k):
            slot = _slot(base + sum(distances[j] for j in range(k) if g >> j & 1), count, threads)
            key = _record_cast(_shared_load(storage.slots, slot), _item_dtype(elems[i][0]))
            if keyed:
                group.append((key, Uint32(_shared_load(storage.values, slot).bitcast(Uint32))))
            else:
                group.append((key,))
        for j in range(k):
            bit = 1 << j
            for g in range(1 << k):
                if g & bit:
                    continue
                earlier, later = order.pair(group[g], group[g | bit])
                group[g], group[g | bit] = _select(ascending, earlier, later), _select(ascending, later, earlier)
        out.append(_pick(group, [(p & distances[j]) != 0 for j in range(k)]))
    return out


def _bitonic(items, payload, storage, tid, width, threads, descending, block_network):
    """Bitonic sort of each warp's tile, or (*block_network*) of the whole block tile."""
    count = len(items)
    order = _Order(descending)
    lane = tid % width
    keyed = payload is not None
    if keyed:
        ties = [Uint32(tid * count + i) for i in range(count)]
        if descending:
            ties = [~t for t in ties]
        elems = [(key, tie) for key, tie in zip(items, ties)]
    else:
        elems = [(key,) for key in items]
    if block_network:
        total = threads * count
        position = lambda i: tid * count + i  # noqa: E731
    else:
        # Every warp sorts its own tile in the same direction.
        total = width * count
        position = lambda i: lane * count + i  # noqa: E731
    warp_tile = width * count
    exchanged = False
    span = 2
    while span <= total:
        d = span // 2
        cross = []
        while d >= warp_tile:
            cross.append(d)
            d //= 2
        for c in range(0, len(cross), 2):
            elems = _group_stage(elems, storage, tid, count, threads, span, cross[c : c + 2], order, not exchanged)
            exchanged = True
        while d:
            elems = _in_warp_stage(elems, position, count, span, d, order)
            d //= 2
        span *= 2
    items = [e[0] for e in elems]
    if keyed:
        scope = gpu.BarrierScope.Workgroup if exchanged else gpu.BarrierScope.Subgroup
        if exchanged:
            barrier()  # the last exchange's reads come first
        for i in range(count):
            _shared_store(storage.values, _slot(tid * count + i, count, threads), payload[i])
        barrier(scope=scope)
        sources = [Int32(e[1] if not descending else ~e[1]) for e in elems]
        payload = [
            _record_cast(_shared_load(storage.values, _slot(s, count, threads)), _item_dtype(payload[0]))
            for s in sources
        ]
        if not block_network:
            # The gathers finish before the merge passes stage payloads into the same slots.
            barrier(scope=scope)
    return items, payload


class BlockMergeSort(_portable.BlockMergeSort):
    __doc__ = _portable.BlockMergeSort.__doc__

    @override
    @classmethod
    def _sort_runs(cls, items, payload, valid, tid, storage, descending, compare_op, valid_items):
        width, count, threads = cls.warp_threads, cls.items_per_thread, cls.block_threads
        if not (_is_gfx9() and _covers(items, payload, count, compare_op, valid_items)):
            return super()._sort_runs(items, payload, valid, tid, storage, descending, compare_op, valid_items)
        warps = threads // width
        block_network = warps & (warps - 1) == 0
        items, payload = _bitonic(items, payload, storage, tid, width, threads, descending, block_network)
        return items, payload, (threads if block_network else width) * count
