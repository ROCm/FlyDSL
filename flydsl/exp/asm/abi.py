# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
ABI/shared policy helpers for the ASM backend.

This module centralizes decisions that must be consistent across:
- Metadata emission (.amdhsa_* directives)
- Kernel IR ABI setup (KernelABI precolored registers)

In particular:
- whether we request the system VGPR workitem id (flat tid in v0)
- how that depends on the logical workgroup size
"""

from __future__ import annotations

from typing import Tuple

from .utils import normalize_wg_size


def system_vgpr_workitem_id_from_wg_size(wg_size: Tuple[int, int, int]) -> int:
    """Return the .amdhsa_system_vgpr_workitem_id value to request.

    Current policy (matches existing working behavior in tests):
    - single-wave workgroup (y==1 and z==1): request 0
    - multi-wave workgroup (y>1 or z>1): request 1 (flat workitem id in v0)
    """
    wg_x, wg_y, wg_z = normalize_wg_size(wg_size)
    return 1 if (wg_y > 1 or wg_z > 1) else 0
