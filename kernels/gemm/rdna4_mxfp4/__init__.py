# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""RDNA4 MXFP4 GEMM; host selection imports without Torch or compiler dependencies."""

from importlib import import_module

from .heuristic import choose_config, rank_configurations

_TENSOR_API = (
    "PreparedMXFP4",
    "prepare_weights",
    "gemm",
    "get_device_capabilities",
    "get_warmup_keys",
    "compile_gemm",
)
__all__ = [*_TENSOR_API, "choose_config", "rank_configurations"]


def __getattr__(name):
    if name in _TENSOR_API:
        module = import_module(".entrypoint", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
