# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockDiscontinuity``: single-wave blocks read neighbors with DPP (see ``_neighbors``)."""

from .. import discontinuity as _discontinuity
from ._neighbors import _WaveNeighbors

__all__ = ["BlockDiscontinuity"]


class BlockDiscontinuity(_WaveNeighbors, _discontinuity.BlockDiscontinuity):
    __doc__ = _discontinuity.BlockDiscontinuity.__doc__
