# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MegaMoEV2 fused MoE operator with lazy public imports."""

import importlib

_LAZY = {
    "MegaMoEConfig": "mega_moe_config",
    "MegaMoEV2": "mega_moe",
    "Stage1Config": "mega_moe_config",
    "Stage2Config": "mega_moe_config",
    "compile_gemm1": "gemm1",
    "convert_aiter_lqq_to_megamoe": "quant",
    "gemm1_kernel": "gemm1",
    "moe_smooth_per_token_scaled_quant": "quant",
    "select_mega_moe_config": "mega_moe_config",
    "smooth_per_token_scaled_quant": "quant",
}

__all__ = list(_LAZY)


def __getattr__(name):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
