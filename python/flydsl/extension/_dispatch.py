# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Lazy target overrides for target-neutral extension libraries."""

import importlib
from functools import wraps
from types import ModuleType
from typing import Callable, Mapping

from ..compiler.backends import compile_backend_name


class Dispatcher:
    """Dispatch a library's universal functions to optional target modules.

    ``targets`` maps compiler backend IDs to public target namespaces. For
    example, ``{"rocm": "rocdl"}`` maps the ROCm compiler backend to the
    sibling ``<library>.rocdl`` module.
    """

    def __init__(self, package: str, *, targets: Mapping[str, str]):
        self.package = package
        self.targets = dict(targets)
        self._resolved_impls = {}

    def _resolve(self, backend: str, name: str, universal_impl: Callable) -> Callable:
        key = (backend, name)
        if key not in self._resolved_impls:
            implementation = universal_impl
            target_name = self.targets.get(backend)
            if target_name is not None:
                module = self.load_target(target_name)
                if name in getattr(module, "__all__", ()):
                    implementation = getattr(module, name)
            self._resolved_impls[key] = implementation
        return self._resolved_impls[key]

    def dispatch(self, universal_impl: Callable) -> Callable:
        """Wrap a universal implementation with cached target override lookup."""

        name = universal_impl.__name__

        @wraps(universal_impl)
        def wrapper(*args, **kwargs):
            implementation = self._resolve(compile_backend_name(), name, universal_impl)
            return implementation(*args, **kwargs)

        return wrapper

    def dispatch_all(self, namespace: dict, names) -> None:
        """Wrap every name in ``names`` with target override lookup."""

        for name in names:
            namespace[name] = self.dispatch(namespace[name])

    def load_target(self, name: str) -> ModuleType:
        """Load one explicitly declared target namespace."""

        if name not in self.targets.values():
            raise AttributeError(f"module {self.package!r} has no attribute {name!r}")
        return importlib.import_module(f".{name}", self.package)
