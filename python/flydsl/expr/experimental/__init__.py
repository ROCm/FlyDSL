# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from typing import TYPE_CHECKING

_EXPERIMENTAL_MODULES = {
    "ktrace": ".ktrace",
}

if TYPE_CHECKING:
    from . import ktrace as ktrace


def __getattr__(name: str):
    module_name = _EXPERIMENTAL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(_EXPERIMENTAL_MODULES))
