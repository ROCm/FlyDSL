# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cooperative algorithms over the threads of one kernel launch.

Those names are dispatched: on a target with an override, ``fx.coop.warp_reduce``
is the target's. ``fx.coop.universal`` is the same surface with dispatch turned
off — see :mod:`flydsl.extension.coop.universal`.
"""

from . import block as block
from . import universal as universal
from . import warp as warp
from .block import *
from .warp import *

__all__ = [
    *warp.__all__,
    *block.__all__,
]
