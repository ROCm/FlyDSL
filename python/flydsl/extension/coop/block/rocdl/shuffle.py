# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockShuffle``: single-wave blocks shift with DPP instead of shared memory.

``up`` / ``down`` take their neighbors from ``_WaveNeighbors`` and, in a
single-wave block, the block suffix / prefix from one ``v_readlane``.
``offset`` and ``rotate`` by a constant distance of one are the same neighbor
moves. Every other call runs the portable code.
"""

from ...._dispatch import override
from ..._values import _is_items, _record_default, _record_select
from .. import shuffle as _shuffle
from ._neighbors import _lane_value, _WaveNeighbors

__all__ = ["BlockShuffle"]


def _unit(distance):
    return isinstance(distance, int) and not isinstance(distance, bool) and distance in (-1, 1)


class BlockShuffle(_WaveNeighbors, _shuffle.BlockShuffle):
    __doc__ = _shuffle.BlockShuffle.__doc__

    @override
    @classmethod
    def offset(cls, value, distance=1, *, storage, fill=None):
        if _unit(distance):
            cls._check(value)
            if not _is_items(value, cls.dtype) and cls._wave_neighbors_apply([value]):
                _, tid, left, right = cls._neighbors(value, storage, left=distance < 0, right=distance > 0)
                edge = tid == 0 if distance < 0 else tid == cls.block_threads - 1
                keep = value if fill is None else _record_default(cls.dtype, fill)
                return _record_select(edge, keep, left if distance < 0 else right)
        return super().offset(value, distance, storage=storage, fill=fill)

    @override
    @classmethod
    def rotate(cls, value, distance=1, *, storage):
        if _unit(distance):
            cls._check(value)
            if not _is_items(value, cls.dtype) and cls._wave_neighbors_apply([value]):
                _, _, left, right = cls._neighbors(value, storage, left=distance < 0, right=distance > 0)
                return left if distance < 0 else right
        return super().rotate(value, distance, storage=storage)

    @override
    @classmethod
    def up(cls, value, *, storage, predecessor=None, return_suffix=False):
        items = cls._single_wave_items(value) if return_suffix else None
        if items is None:
            return super().up(value, storage=storage, predecessor=predecessor, return_suffix=return_suffix)
        # A single wave publishes nothing: the suffix is one readlane.
        shifted = super().up(value, storage=storage, predecessor=predecessor)
        return shifted, _lane_value(items[-1], cls.block_threads - 1)

    @override
    @classmethod
    def down(cls, value, *, storage, successor=None, return_prefix=False):
        items = cls._single_wave_items(value) if return_prefix else None
        if items is None:
            return super().down(value, storage=storage, successor=successor, return_prefix=return_prefix)
        shifted = super().down(value, storage=storage, successor=successor)
        return shifted, _lane_value(items[0], 0)


for _name in ("offset", "rotate", "up", "down"):
    getattr(BlockShuffle, _name).__func__.__doc__ = getattr(_shuffle.BlockShuffle, _name).__doc__
