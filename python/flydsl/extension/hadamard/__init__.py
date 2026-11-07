# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp- and block-scope Fast Walsh-Hadamard Transform (FWHT) with target dispatch.

``fx.hadamard.<name>`` selects the target-accelerated implementation when one
exists; ``fx.hadamard.universal.<name>`` always runs the portable implementation.
"""

from .._dispatch import Dispatcher
from . import universal as universal
from .universal import BlockFWHT as _BlockFWHT
from .universal import warp_fwht

__all__ = [
    "BlockFWHT",
    "warp_fwht",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

_dispatch.dispatch_all(globals(), ["warp_fwht"])


class BlockFWHT(_BlockFWHT):
    __doc__ = _BlockFWHT.__doc__

    _dispatcher = _dispatch
