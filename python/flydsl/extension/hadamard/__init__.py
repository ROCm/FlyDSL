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

# Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see universal.py.
# from .universal import warp_fwht_e4m3x4, warp_fwht_e5m2x4

__all__ = [
    "BlockFWHT",
    "warp_fwht",
    # "warp_fwht_e4m3x4",
    # "warp_fwht_e5m2x4",
]

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

# Include "warp_fwht_e4m3x4" and "warp_fwht_e5m2x4" once ROCm/FlyDSL#1376 lands.
_dispatch.dispatch_all(globals(), ["warp_fwht"])


class BlockFWHT(_BlockFWHT):
    __doc__ = _BlockFWHT.__doc__

    _dispatcher = _dispatch
