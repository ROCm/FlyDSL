# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""RDNA4 MXFP4 weights with native FP8 arithmetic and FP32 accumulation.

Host configuration selection imports without a GPU or compiler dependency.
The tensor API loads on first use.
"""

from .dispatch import choose_config, rank_configurations

__all__ = ["PreparedMXFP4", "prepare_weights", "gemm", "choose_config", "rank_configurations"]


def __getattr__(name):
    if name in ("PreparedMXFP4", "prepare_weights", "gemm"):
        from . import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
