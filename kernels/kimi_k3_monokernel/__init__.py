# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Public API for the Kimi-K3 decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernels.kimi_k3_monokernel.op import KimiK3MonoKernel
    from kernels.monokernel.weights import LayerWeights

__all__ = ["KimiK3MonoKernel", "KimiK3CompileConfig", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "KimiK3CompileConfig":
        from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig

        return KimiK3CompileConfig
    if name == "KimiK3MonoKernel":
        from kernels.kimi_k3_monokernel.op import KimiK3MonoKernel

        return KimiK3MonoKernel
    if name == "LayerWeights":
        from kernels.monokernel.weights import LayerWeights

        return LayerWeights
    raise AttributeError(name)
