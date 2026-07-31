# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral random-number algorithms with optional target overrides.

``fx.random.<name>`` runs the implementation chosen for the compilation
target; ``fx.random.universal.<name>`` always runs the portable one.
"""

from .._library import Library
from . import universal
from .universal import *  # noqa: F401,F403

__all__ = list(universal.__all__)

_library = Library(__name__, targets={"rocm": "rocdl"})
__getattr__ = _library.load_target

_library.dispatch_all(globals(), __all__)
