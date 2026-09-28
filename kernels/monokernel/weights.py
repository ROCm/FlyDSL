# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared host-side weight container for model-specific MonoKernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.monokernel.config import GLM5_CONFIG, LayerConfig


@dataclass
class LayerWeights:
    """One tensor-parallel rank's weights and model geometry."""

    heads: int
    t: dict[str, torch.Tensor]
    config: LayerConfig = GLM5_CONFIG
    rank: int = 0
    npes: int = 1


__all__ = ["LayerWeights"]
