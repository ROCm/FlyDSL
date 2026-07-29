# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MegaMoEV2 fused MoE operator.

Public symbols are lazy-loaded so that importing a light submodule
(e.g. ``kernels.mega_moe.gemm1``) does not pull in the full operator's
heavy dependencies (``mori.shmem`` / ``torch.distributed``) via the package.
"""

import importlib

_LAZY = {
    "MegaMoEV2": "mega_moe",
    "compile_gemm1": "gemm1",
    "gemm1_kernel": "gemm1",
}

__all__ = list(_LAZY)


def __getattr__(name):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
