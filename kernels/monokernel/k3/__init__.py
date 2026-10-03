# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Public API for the Kimi-K3 decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernels.monokernel.k3.op import KimiK3MonoKernel, KimiK3StagedAgenticOp
    from kernels.monokernel.weights import LayerWeights

__all__ = ["KimiK3MonoKernel", "KimiK3StagedAgenticOp", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "KimiK3MonoKernel":
        from kernels.monokernel.k3.op import KimiK3MonoKernel

        return KimiK3MonoKernel
    if name == "KimiK3StagedAgenticOp":
        from kernels.monokernel.k3.op import KimiK3StagedAgenticOp

        return KimiK3StagedAgenticOp
    if name == "LayerWeights":
        from kernels.monokernel.weights import LayerWeights

        return LayerWeights
    raise AttributeError(name)
