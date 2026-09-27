# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi-K3 model-specific fused kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernels.kimi_k3.full_layer import KimiK3KdaFullLayer
    from kernels.kimi_k3.kda import KimiK3KdaAttention

__all__ = ["KimiK3KdaAttention", "KimiK3KdaFullLayer"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "KimiK3KdaAttention":
        from kernels.kimi_k3.kda import KimiK3KdaAttention

        return KimiK3KdaAttention
    if name == "KimiK3KdaFullLayer":
        from kernels.kimi_k3.full_layer import KimiK3KdaFullLayer

        return KimiK3KdaFullLayer
    raise AttributeError(name)
