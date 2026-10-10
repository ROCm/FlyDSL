# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Kernel-facing names for the gfx120x checks in ``flydsl.runtime.device``.

The implementation lives next to ``is_rdna_arch`` and ``get_warp_size``.
This module only re-exports it so kernel imports stay stable.
"""

from flydsl.runtime.device import get_gcn_arch, is_gfx120x, is_gfx120x_arch, require_gfx120x

__all__ = ["get_gcn_arch", "is_gfx120x_arch", "is_gfx120x", "require_gfx120x"]
