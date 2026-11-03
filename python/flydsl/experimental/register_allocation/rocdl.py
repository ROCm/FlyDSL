# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""AMDGPU register class descriptors for :mod:`flydsl.experimental.register_allocation`.

The suffix is the width of one class member, which is also the unit of
placement numbers.
"""

from .register import RegisterClass

__all__ = [
    "AGPR32",
    "VGPR32",
    "SGPR32",
]

AGPR32 = RegisterClass("amdgcn", "AGPR_32", 32)
VGPR32 = RegisterClass("amdgcn", "VGPR_32", 32)
SGPR32 = RegisterClass("amdgcn", "SGPR_32", 32)
