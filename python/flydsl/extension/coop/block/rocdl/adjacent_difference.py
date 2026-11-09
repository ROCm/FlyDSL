# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockAdjacentDifference``: single-wave blocks read neighbors with DPP (see ``_neighbors``)."""

from .. import adjacent_difference as _adjacent
from ._neighbors import _WaveNeighbors

__all__ = ["BlockAdjacentDifference"]


class BlockAdjacentDifference(_WaveNeighbors, _adjacent.BlockAdjacentDifference):
    __doc__ = _adjacent.BlockAdjacentDifference.__doc__
