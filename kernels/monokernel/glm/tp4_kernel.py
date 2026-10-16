# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compatibility entry point; TP4 and TP8 use the same kernel builder."""

from kernels.monokernel.glm.kernel import build_glm5_monokernel as _build

__all__ = ["build_glm5_monokernel"]


def build_glm5_monokernel(*args, **kwargs):
    """Retain the TP4 builder's BF16 RoPE-table contract."""
    kwargs.setdefault("rope_dtype", "bf16")
    return _build(*args, **kwargs)
