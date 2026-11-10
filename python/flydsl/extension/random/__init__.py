# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral random-number algorithms with optional target overrides.

``fx.random.<name>`` runs the implementation chosen for the compilation
target; ``fx.random.universal.<name>`` always runs the portable one.
"""

from .._dispatch import Dispatcher
from . import universal
from .universal import *  # noqa: F401,F403

__all__ = list(universal.__all__)

_dispatch = Dispatcher(__name__, targets={"rocm": "rocdl"})
__getattr__ = _dispatch.load_target

_dispatch.dispatch_all(globals(), __all__)
