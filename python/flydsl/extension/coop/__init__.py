# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cooperative algorithms over the threads of one kernel launch.

Each scope has one module per primitive family. Block collectives share the
specialization machinery in ``block/_spec.py``. ROCm warp overrides live in
``warp/rocdl.py``; every portable implementation remains available in
``universal``.

That surface is flat — ``fx.coop.<name>`` — so callers never spell the scope
out twice (``fx.coop.warp_reduce``, not ``fx.coop.warp.warp_reduce``).

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
