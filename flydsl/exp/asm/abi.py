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
    - single-wave workgroup (total threads <= 64): request 0
    - multi-wave workgroup (total threads > 64): request 1 (flat workitem id in v0)

    NOTE:
    A workgroup can be multi-wave even when (y,z)==(1,1), e.g. wg_size=(256,1,1).
    Those kernels require the flat workitem id to correctly distinguish waves.
    """
    wg_x, wg_y, wg_z = normalize_wg_size(wg_size)
    total_threads = int(wg_x) * int(wg_y) * int(wg_z)
    return 1 if total_threads > 64 else 0
