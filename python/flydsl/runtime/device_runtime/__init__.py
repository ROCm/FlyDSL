# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Per-process GPU device runtime.

Exactly one :class:`DeviceRuntime` implementation is active per process.
It must match the selected compile backend (e.g. ``rocm`` compile backend ↔
``rocm`` runtime / HIP).

Environment:

* ``FLYDSL_RUNTIME_KIND`` — selects the built-in runtime implementation
  (default: ``rocm``). Must agree with ``FLYDSL_COMPILE_BACKEND`` via
  :data:`COMPILE_BACKEND_TO_RUNTIME_KIND` (and optional extension mappings).
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from ...utils import env
from .base import DeviceRuntime
from .rocm import RocmDeviceRuntime

# Compile-backend id -> device-runtime kind (single string namespace).
COMPILE_BACKEND_TO_RUNTIME_KIND: Dict[str, str] = {
    "rocm": "rocm",
}

_EXTRA_MAPPINGS: Dict[str, str] = {}

_builtin_runtimes: Dict[str, Type[DeviceRuntime]] = {
    "rocm": RocmDeviceRuntime,
}

_runtime_cls_override: Optional[Type[DeviceRuntime]] = None
_instance: Optional[DeviceRuntime] = None


def register_compile_runtime_mapping(compile_backend: str, runtime_kind: str) -> None:
    """Map a compile-backend id to a device-runtime *kind*.

    Use when a third-party :func:`flydsl.compiler.backends.register_backend`
    targets an existing runtime stack (e.g. custom name → ``rocm``).
    """
    _EXTRA_MAPPINGS[compile_backend.strip().lower()] = runtime_kind.strip().lower()


def register_device_runtime(
    cls: Type[DeviceRuntime],
    *,
    kind: Optional[str] = None,
    force: bool = False,
) -> None:
    """Register a custom :class:`DeviceRuntime` class for the whole process.

    Must be called before the first :func:`get_device_runtime` if replacing the
    default. Raises if a runtime instance already exists, unless ``force=True``.

    If *kind* is set, the class is also registered under that runtime *kind* for
    ``FLYDSL_RUNTIME_KIND`` (and should match the compile-backend mapping).
    """
    global _runtime_cls_override, _instance
    if _instance is not None and not force:
        raise RuntimeError(
            "Cannot register a device runtime after get_device_runtime() "
            "has been called (unless force=True)."
        )
    if _runtime_cls_override is not None and not force:
        raise ValueError(
            "A custom device runtime class is already registered "
            "(use force=True to replace)."
        )
    _runtime_cls_override = cls
    if kind is not None:
        _builtin_runtimes[kind.strip().lower()] = cls


def _expected_runtime_kind_for_compile_backend(compile_backend_id: str) -> str:
    key = compile_backend_id.strip().lower()
    if key in _EXTRA_MAPPINGS:
        return _EXTRA_MAPPINGS[key]
    if key in COMPILE_BACKEND_TO_RUNTIME_KIND:
        return COMPILE_BACKEND_TO_RUNTIME_KIND[key]
    raise ValueError(
        f"No device-runtime kind mapped for compile backend {compile_backend_id!r}. "
        "Register a mapping with register_compile_runtime_mapping()."
    )


def _resolve_runtime_class() -> Type[DeviceRuntime]:
    if _runtime_cls_override is not None:
        return _runtime_cls_override
    kind = (env.runtime.kind or "rocm").strip().lower()
    cls = _builtin_runtimes.get(kind)
    if cls is None:
        known = ", ".join(sorted(_builtin_runtimes)) or "(none)"
        raise ValueError(
            f"Unknown FLYDSL_RUNTIME_KIND={kind!r}. Built-in kinds: {known}"
        )
    return cls


def _selected_runtime_kind_from_env() -> str:
    """Runtime kind implied by env / registration, without instantiating :class:`DeviceRuntime`."""
    if _runtime_cls_override is not None:
        return str(_runtime_cls_override.kind)
    return (env.runtime.kind or "rocm").strip().lower()


def ensure_compile_runtime_pairing_from_env(compile_backend_id: str) -> None:
    """Raise if *compile_backend_id* does not match the configured runtime kind.

    Uses only environment and registration state — does not construct
    :class:`DeviceRuntime`. Suitable for compiler paths (e.g. ``COMPILE_ONLY``)
    where initializing the runtime is unnecessary.
    """
    expected = _expected_runtime_kind_for_compile_backend(compile_backend_id)
    actual = _selected_runtime_kind_from_env()
    if actual != expected:
        raise RuntimeError(
            f"Compile backend {compile_backend_id!r} requires device runtime kind "
            f"{expected!r}, but FLYDSL_RUNTIME_KIND (and registration) resolve to "
            f"{actual!r}. "
            f"Align FLYDSL_COMPILE_BACKEND with FLYDSL_RUNTIME_KIND (and extension "
            f"mappings), or use a matching pair of register_backend / "
            f"register_device_runtime."
        )


def ensure_compile_runtime_compatible(
    compile_backend_id: str,
    *,
    runtime: Optional[DeviceRuntime] = None,
) -> None:
    """Raise if *compile_backend_id* does not match the active runtime kind."""
    expected = _expected_runtime_kind_for_compile_backend(compile_backend_id)
    rt = runtime if runtime is not None else get_device_runtime()
    if rt.kind != expected:
        raise RuntimeError(
            f"Compile backend {compile_backend_id!r} requires device runtime kind "
            f"{expected!r}, but the active runtime is {rt.kind!r}. "
            f"Align FLYDSL_COMPILE_BACKEND with FLYDSL_RUNTIME_KIND (and extension "
            f"mappings), or use a matching pair of register_backend / "
            f"register_device_runtime."
        )


def _active_compile_backend_id() -> str:
    """Mirror :func:`flydsl.compiler.backends.compile_backend_name` without importing ``compiler``."""
    return (env.compile.backend or "rocm").lower()


def get_device_runtime() -> DeviceRuntime:
    """Return the single process-wide :class:`DeviceRuntime` instance.

    Compile/runtime pairing runs once when the singleton is first created (see
    :func:`ensure_compile_runtime_pairing_from_env`), not on every call — the
    active backend and runtime kind are treated as fixed for the process after
    that point.
    """
    global _instance
    if _instance is None:
        ensure_compile_runtime_pairing_from_env(_active_compile_backend_id())
        cls = _resolve_runtime_class()
        _instance = cls()
    return _instance


__all__ = [
    "COMPILE_BACKEND_TO_RUNTIME_KIND",
    "DeviceRuntime",
    "RocmDeviceRuntime",
    "ensure_compile_runtime_compatible",
    "ensure_compile_runtime_pairing_from_env",
    "get_device_runtime",
    "register_compile_runtime_mapping",
    "register_device_runtime",
]
