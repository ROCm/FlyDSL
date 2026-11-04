# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Placement of register memory at chosen numbers of a register class.

Placement is a register allocation hint, not a guarantee; see
docs/experimental/register_allocation.md. ``rocdl`` holds the AMDGPU register classes.
"""

from . import register as register
from . import rocdl as rocdl
from .register import *

__all__ = [
    *register.__all__,
]
