# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Pluggable GPU compile-backend registry.

Usage::

    from flydsl.compiler.backends import get_backend, register_backend

    backend = get_backend()          # resolve from FLYDSL_COMPILE_BACKEND (default: rocm)
    backend = get_backend("rocm")    # explicit

    register_backend("my_hw", MyBackendFactory)  # third-party extension
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Type

from ...utils import env
from .base import BaseBackend, GPUTarget

_registry: Dict[str, Type[BaseBackend]] = {}


def register_backend(name: str, backend_cls: type, *, force: bool = False) -> None:
    """Register a backend class under *name* (case-insensitive).

    ``backend_cls`` must be a concrete subclass of ``BaseBackend``.
    Raises ``ValueError`` if *name* is already registered, unless
    ``force=True`` (useful during development / experimentation).
    """
    key = name.lower()
    if key in _registry and not force:
        raise ValueError(
            f"Compile backend '{name}' is already registered. "
            f"Use force=True to override (not recommended in production)."
        )
    _registry[key] = backend_cls


def compile_backend_name() -> str:
    """Return the active backend id from env (default ``'rocm'``)."""
    return (env.compile.backend or "rocm").lower()


@lru_cache(maxsize=4)
def _make_backend(name: str, arch: str) -> BaseBackend:
    """Internal: create and cache a backend instance for *(name, arch)*.

    Both *name* and *arch* must already be resolved (non-empty) so that
    the ``lru_cache`` key is deterministic and won't become stale when
    environment variables change after the first call.
    """
    if name not in _registry:
        available = ", ".join(sorted(_registry)) or "(none)"
        raise ValueError(
            f"Unknown compile backend '{name}'. Registered backends: {available}"
        )
    backend_cls = _registry[name]
    target = backend_cls.make_target(arch)
    return backend_cls(target)


def get_backend(name: Optional[str] = None, *, arch: str = "") -> BaseBackend:
    """Resolve a backend instance.

    *name* defaults to ``FLYDSL_COMPILE_BACKEND`` env var (or ``'rocm'``).
    *arch* overrides the auto-detected architecture when non-empty.

    Compile/runtime pairing (``FLYDSL_COMPILE_BACKEND`` vs ``FLYDSL_RUNTIME_KIND``)
    is validated on each :class:`~flydsl.compiler.jit_function.JitFunction` call
    (via :func:`flydsl.runtime.device_runtime.ensure_compile_runtime_pairing_from_env`)
    and again on first :func:`flydsl.runtime.device_runtime.get_device_runtime`,
    not here.
    """
    if name is None:
        name = compile_backend_name()
    name = name.lower()
    if not arch:
        backend_cls = _registry.get(name)
        if backend_cls is None:
            available = ", ".join(sorted(_registry)) or "(none)"
            raise ValueError(
                f"Unknown compile backend '{name}'. Registered backends: {available}"
            )
        arch = backend_cls.detect_target().arch
    return _make_backend(name, arch)


# -- auto-register built-in backends ------------------------------------
from .rocm import RocmBackend  # noqa: E402

register_backend("rocm", RocmBackend)

__all__ = [
    "BaseBackend",
    "GPUTarget",
    "compile_backend_name",
    "get_backend",
    "register_backend",
]
