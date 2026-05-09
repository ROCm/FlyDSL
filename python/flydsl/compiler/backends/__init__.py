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
        if name in _import_errors:
            raise ImportError(
                f"Compile backend '{name}' failed to import"
            ) from _import_errors[name]
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
            if name in _import_errors:
                raise ImportError(
                    f"Compile backend '{name}' failed to import"
                ) from _import_errors[name]
            available = ", ".join(sorted(_registry)) or "(none)"
            raise ValueError(
                f"Unknown compile backend '{name}'. Registered backends: {available}"
            )
        arch = backend_cls.detect_target().arch
    return _make_backend(name, arch)


# -- auto-discover built-in backends (Triton-style directory scan) --------
_import_errors: Dict[str, Exception] = {}


def _discover_backends() -> None:
    """Scan this package for concrete :class:`BaseBackend` subclasses.

    Import errors are stored in ``_import_errors`` and surfaced when the
    user actually requests that backend (see ``_make_backend``).
    """
    import importlib
    from pathlib import Path

    root = Path(__file__).parent
    for item in sorted(root.iterdir()):
        name = item.stem
        if name.startswith("_") or name == "base":
            continue
        if not (item.suffix == ".py" or (item.is_dir() and (item / "__init__.py").exists())):
            continue
        try:
            mod = importlib.import_module(f".{name}", __package__)
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if (
                    isinstance(cls, type)
                    and issubclass(cls, BaseBackend)
                    and cls is not BaseBackend
                    and not getattr(cls, "__abstractmethods__", None)
                ):
                    register_backend(name, cls, force=False)
                    break
        except Exception as exc:
            _import_errors[name] = exc


_discover_backends()


def _discover_entry_point_backends() -> None:
    """Discover third-party backends via ``entry_points(group='flydsl.backends')``.

    Each entry point should point to a concrete :class:`BaseBackend` subclass.
    The entry point *name* becomes the backend name. Built-in backends
    (already registered by ``_discover_backends``) take precedence.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    eps = entry_points()
    # Python 3.12+: eps.select(); 3.9-3.11: dict-like
    if hasattr(eps, "select"):
        backend_eps = eps.select(group="flydsl.backends")
    else:
        backend_eps = eps.get("flydsl.backends", [])
    for ep in backend_eps:
        name = ep.name.lower()
        if name in _registry:
            continue  # built-in takes precedence
        try:
            cls = ep.load()
            register_backend(name, cls, force=False)
        except Exception as exc:
            _import_errors[name] = exc


_discover_entry_point_backends()

__all__ = [
    "BaseBackend",
    "GPUTarget",
    "compile_backend_name",
    "get_backend",
    "register_backend",
]
