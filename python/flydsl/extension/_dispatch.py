# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Lazy target overrides for target-neutral extension libraries.

A library either dispatches its public functions (``Dispatcher.dispatch_all``)
or its public classes (``Dispatcher.public_class``). A dispatched class
specializes into the target module's same-named class, which subclasses the
portable one and replaces only what its target does better; anything else it
inherits.
"""

import importlib
from functools import wraps
from types import ModuleType
from typing import Callable, Mapping

from ..compiler.backends import compile_backend_name


def dispatch_target(cls):
    """The compile backend's override of a public class, or *cls* itself.

    Only a class that declares ``_dispatch_class`` in its own body dispatches:
    a subclass derived from it (a specialization, or a configured variant such
    as ``with_radix_bits``) must keep its own configuration rather than be
    swapped for the target's.
    """
    if not cls.__dict__.get("_dispatch_class", False):
        return cls
    return cls._dispatcher._resolve(compile_backend_name(), cls.__name__, cls)


def override(method):
    """Mark a target class's method as replacing one a base class defines.

    ``check_overrides`` turns a marked method whose base has lost that name --
    renamed or removed in the portable class -- into an error when the class is
    created, instead of an override that silently stops running.
    """
    function = method.__func__ if isinstance(method, (classmethod, staticmethod)) else method
    setattr(function, "__override__", True)
    return method


def check_overrides(cls):
    """Raise if any class in *cls*'s MRO marks an ``override`` that no later class defines."""
    mro = cls.__mro__
    for index, owner in enumerate(mro):
        for name, attribute in vars(owner).items():
            function = attribute.__func__ if isinstance(attribute, (classmethod, staticmethod)) else attribute
            if getattr(function, "__override__", False) and not any(name in vars(base) for base in mro[index + 1 :]):
                raise TypeError(f"{owner.__qualname__}.{name} is marked override, but no base class defines it")


class Dispatcher:
    """Dispatch a library's universal functions or classes to optional target modules.

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

    def public_class(self, portable):
        """The public spelling of *portable*: specializing it specializes the target's same-named override."""
        namespace = {
            "__doc__": portable.__doc__,
            "__module__": self.package,
            "__qualname__": portable.__name__,
            "_dispatcher": self,
            "_dispatch_class": True,
        }
        return type(portable)(portable.__name__, (portable,), namespace)

    def load_target(self, name: str) -> ModuleType:
        """Load one explicitly declared target namespace."""

        if name not in self.targets.values():
            raise AttributeError(f"module {self.package!r} has no attribute {name!r}")
        return importlib.import_module(f".{name}", self.package)
