# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockExchange``: wide blocked-side accesses on a conflict-free LDS layout.

LDS geometry of gfx9 (measured on gfx942 with ``SQ_LDS_BANK_CONFLICT``): 32 banks
of 4 bytes. A wave64 access is served in passes of 128 bytes over *consecutive*
lanes -- 32 lanes for b32, 16 for b64, 8 for b128 -- and every further dword one
pass sends to a busy bank costs one cycle. Both numbers are class attributes, so
a target with another geometry overrides them.

With ``SHARED_MEMORY`` and a scalar dtype of 8 to 128 bits:

- The blocked side moves a thread's items with the widest access its tile allows
  (``ds_write/read_b64/b128``; the portable storage is 16-byte aligned).
- When a thread's tile spans ``n >= 2`` such accesses (``n`` a power of two, e.g.
  8 or 16 Int32, 4 or 8 Int64), consecutive lanes are ``n`` accesses apart and
  collide within a pass. XORing the access index with the 128-byte row index
  separates them. Unlike padding, the swizzle keeps the storage size, the 16-byte
  alignment of every access and the ``read2st64`` immediate offsets of the striped
  side (there the XOR term only depends on lane bits).

Every other call -- one item per thread, ``WARP_TIME_SLICING``, records, vectors,
pointers, guarded and flagged scatters, non-gfx9 targets -- runs the portable code.
"""

from ....._mlir.dialects import gpu
from .....compiler import jit
from .....expr.gpu import barrier
from .....expr.numeric import Index, Numeric
from .....expr.primitive import ptr_load, ptr_store
from .....expr.typing import Vector
from ...._dispatch import override
from ..._common import _linear_thread_id
from ..._values import _as_items, _from_items, _is_items, _shared_load, _shared_store
from ...warp.rocdl._common import _is_gfx9
from .. import exchange as _exchange

__all__ = ["BlockExchange"]

_SHARED_MEMORY = _exchange.BlockExchangeAlgorithm.SHARED_MEMORY


def _log2(x):
    return x.bit_length() - 1


class _TileLayout:
    """Element-index layout of a full exchange tile.

    A thread's blocked items form ``chunks`` accesses of ``vector`` elements. With
    ``chunks`` a power of two above one, the access index's low ``log2(m)`` bits
    are XORed with the bits that select the pass, ``m = min(chunks, lanes per
    pass)``: a pass then sees ``m`` distinct bank groups instead of one.
    """

    def __init__(self, dtype, count, pass_bytes, max_access):
        size = dtype.width // 8
        tile = count * size
        access = min(max_access, tile & -tile)
        self.vector = access // size
        self.chunks = tile // access
        lanes = max(1, pass_bytes // access)
        self.mask = 0
        n = self.chunks
        if n > 1 and n & (n - 1) == 0:
            m = min(n, lanes)
            self.mask = m - 1
            self.low = _log2(self.vector)
            self.shift = self.low + max(_log2(lanes), _log2(n))
            self.span = 1 << (self.shift + _log2(m))

    def swizzle(self, index):
        if not self.mask:
            return index
        return index ^ (((index >> self.shift) & self.mask) << self.low)

    def at(self, base, offset):
        """Physical element of ``base + offset`` (``offset`` a Python int, kept as an immediate when possible)."""
        if not self.mask or offset % self.span == 0:
            return self.swizzle(base) + offset
        return self.swizzle(base + offset)

    def chunk_offsets(self, base):
        """Physical first element of each of a thread's blocked accesses (``base`` = tid * count)."""
        if not self.mask:
            return [base + j * self.vector for j in range(self.chunks)]
        # base is a multiple of the tile, so the pass bits do not depend on j and
        # the XOR term lands in base's zero bits: access j is access 0 XOR j.
        first = base | (((base >> self.shift) & self.mask) << self.low)
        return [first if j == 0 else first ^ (j * self.vector) for j in range(self.chunks)]


@jit
def _scatter_store(slots, rank, slot, size, item):
    if (rank >= 0) & (rank < size):
        _shared_store(slots, slot, item)


class BlockExchange(_exchange.BlockExchange):
    __doc__ = _exchange.BlockExchange.__doc__

    _LDS_PASS_BYTES = 128  # 32 banks x 4 bytes per conflict-free pass
    _LDS_MAX_ACCESS_BYTES = 16  # ds_read/write_b128

    @classmethod
    def _layout(cls):
        """The gfx9 tile layout, or None when the portable exchange runs instead."""
        cls._check()
        dtype = cls.dtype
        if cls.algorithm is not _SHARED_MEMORY or cls.items_per_thread == 1 or not _is_gfx9():
            return None
        if not (isinstance(dtype, type) and issubclass(dtype, Numeric)) or issubclass(dtype, Index):
            return None
        if dtype.width < 8 or dtype.width & (dtype.width - 1):
            return None
        return _TileLayout(dtype, cls.items_per_thread, cls._LDS_PASS_BYTES, cls._LDS_MAX_ACCESS_BYTES)

    @classmethod
    def _store_blocked(cls, layout, slots, tid, items):
        n = layout.vector
        for j, at in enumerate(layout.chunk_offsets(tid * cls.items_per_thread)):
            part = items[j * n : (j + 1) * n]
            if n == 1:
                _shared_store(slots, at, part[0])
            else:
                ptr_store(Vector.from_elements(part, dtype=cls.dtype), slots.ptr + at)

    @classmethod
    def _load_blocked(cls, layout, slots, tid):
        n = layout.vector
        out = []
        for at in layout.chunk_offsets(tid * cls.items_per_thread):
            if n == 1:
                out.append(_shared_load(slots, at))
            else:
                vector = ptr_load(slots.ptr + at, result_type=Vector.make_type(n, cls.dtype))
                # Signless loads lose unsigned metadata; restore the declared element type.
                out.extend(cls.dtype(vector[i]) for i in range(n))
        return out

    @classmethod
    def _striped(cls, tid, i):
        return tid, i * cls.block_threads

    @classmethod
    def _warp_striped(cls, tid, i):
        warp = cls.warp_threads
        return (tid // warp) * (warp * cls.items_per_thread) + tid % warp, i * warp

    @classmethod
    def _move(cls, value, storage, layout, from_blocked, index, scope):
        cls._check(value)
        items = _as_items(value, cls.dtype)
        tid = _linear_thread_id(cls.block_size)
        slots = storage.slots
        count = cls.items_per_thread
        if from_blocked:
            cls._store_blocked(layout, slots, tid, items)
            barrier(scope=scope)
            out = [_shared_load(slots, layout.at(*index(tid, i))) for i in range(count)]
        else:
            for i, item in enumerate(items):
                _shared_store(slots, layout.at(*index(tid, i)), item)
            barrier(scope=scope)
            out = cls._load_blocked(layout, slots, tid)
        return _from_items(out, dtype=cls.dtype, like=value) if _is_items(value, cls.dtype) else out[0]

    @override
    @classmethod
    def blocked_to_striped(cls, value, *, storage):
        layout = cls._layout()
        if layout is None:
            return super().blocked_to_striped(value, storage=storage)
        return cls._move(value, storage, layout, True, cls._striped, gpu.BarrierScope.Workgroup)

    @override
    @classmethod
    def striped_to_blocked(cls, value, *, storage):
        layout = cls._layout()
        if layout is None:
            return super().striped_to_blocked(value, storage=storage)
        return cls._move(value, storage, layout, False, cls._striped, gpu.BarrierScope.Workgroup)

    @override
    @classmethod
    def blocked_to_warp_striped(cls, value, *, storage):
        layout = cls._layout()
        if layout is None:
            return super().blocked_to_warp_striped(value, storage=storage)
        return cls._move(value, storage, layout, True, cls._warp_striped, gpu.BarrierScope.Subgroup)

    @override
    @classmethod
    def warp_striped_to_blocked(cls, value, *, storage):
        layout = cls._layout()
        if layout is None:
            return super().warp_striped_to_blocked(value, storage=storage)
        return cls._move(value, storage, layout, False, cls._warp_striped, gpu.BarrierScope.Subgroup)

    @override
    @classmethod
    def scatter_to_blocked(cls, value, ranks, *, storage):
        layout = cls._layout()
        if layout is None:
            return super().scatter_to_blocked(value, ranks, storage=storage)
        cls._check(value)
        count = cls.items_per_thread
        ranks = Vector.from_elements(_as_items(ranks))
        if ranks.numel != count or ranks.dtype.is_float:
            raise TypeError("ranks must contain items_per_thread integer entries")
        items = _as_items(value, cls.dtype)
        tid = _linear_thread_id(cls.block_size)
        slots = storage.slots
        size = cls.block_threads * count
        for i, item in enumerate(items):
            # Out-of-tile ranks are ignored, as in the portable scatter.
            _scatter_store(slots, ranks[i], layout.swizzle(ranks[i]), size, item)
        barrier()
        out = cls._load_blocked(layout, slots, tid)
        return _from_items(out, dtype=cls.dtype, like=value) if _is_items(value, cls.dtype) else out[0]


for _name in (
    "blocked_to_striped",
    "striped_to_blocked",
    "blocked_to_warp_striped",
    "warp_striped_to_blocked",
    "scatter_to_blocked",
):
    getattr(BlockExchange, _name).__func__.__doc__ = getattr(_exchange.BlockExchange, _name).__doc__
