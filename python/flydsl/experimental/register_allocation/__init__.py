# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register-memory allocation with optional register class and number placement.

Reached as ``fx.experimental.register_allocation``, or imported with
``from flydsl.experimental import register_allocation as fx_regalloc``;
``rocdl`` holds the AMDGPU register class descriptors (``fx_regalloc.rocdl.VGPR32``).
"""

from . import register as register
from . import rocdl as rocdl
from .register import *

__all__ = [
    *register.__all__,
]
