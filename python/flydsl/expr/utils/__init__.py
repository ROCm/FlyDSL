# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from typing import Any, Callable

from .print_typst import print_typst

__all__ = [
    "print_typst",
    "lazy_classattr",
]


class lazy_classattr:
    """A class-level attribute computed lazily by a zero-arg function.

    Reading the attribute on the class (``Owner.attr``) or on an instance calls
    ``fn()`` and returns its result; nothing is cached, so each access re-runs
    ``fn``.

    Two reasons it is a descriptor rather than a ``@property`` on a metaclass:

    * **Deferral is required, caching is wrong.** It exists for values that can
      only be built inside a transient context (e.g. an MLIR type, valid only
      under the active ``ir.Context``). Caching the first result would leak it
      into later contexts, so ``__get__`` recomputes every time.
    * **The metaclass route is unavailable.** It is meant for classes whose
      metaclass cannot be subclassed (such as ``ir.Value``), which rules out
      hosting a metaclass property; a plain descriptor in the class dict still
      fires on class access.
    """

    def __init__(self, fn: Callable[[], Any]):
        self._fn = fn
        self._name = "<unnamed>"

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, owner=None):
        return self._fn()

    def __repr__(self):
        return f"<{type(self).__name__} {self._name!r}>"
