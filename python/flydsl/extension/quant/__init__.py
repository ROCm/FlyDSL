# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Micro-scaled and sub-byte quantization codecs with target overrides.

``fx.quant.<name>`` selects the implementation best suited to the active
compilation target; ``fx.quant.universal.<name>`` always runs the portable
bit-exact reference implementation.
"""

from .._dispatch import Dispatcher
from . import universal
from .universal import *  # noqa: F401,F403

__all__ = list(universal.__all__)

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

_dispatch.dispatch_all(globals(), __all__)
