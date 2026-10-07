# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public API for the DeepSeek-V4 decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.monokernel.config import MoeMode

if TYPE_CHECKING:
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

__all__ = ["MoeMode", "Dsv4MonoKernel"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "Dsv4MonoKernel":
        from kernels.monokernel.dsv4.op import Dsv4MonoKernel

        return Dsv4MonoKernel
    raise AttributeError(name)
