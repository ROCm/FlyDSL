# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Install-time compile/runtime capability (from CMake-generated ``_build_config``)."""

from __future__ import annotations

from typing import Tuple

_FALLBACK: Tuple[str, tuple[str, ...], tuple[str, ...]] = ("rocdl", ("rocm",), ("rocm",))


def install_caps() -> Tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return ``(TARGET_STACK, compile_backend_ids, runtime_kinds)`` for this install."""
    try:
        from flydsl import _build_config as bc
    except ImportError:
        return _FALLBACK
    stack = str(getattr(bc, "TARGET_STACK", _FALLBACK[0]))
    backends = tuple(getattr(bc, "ENABLED_COMPILE_BACKEND_IDS", ()) or ())
    runtimes = tuple(getattr(bc, "ENABLED_RUNTIME_KINDS", ()) or ())
    return (stack, backends, runtimes)


def ensure_compile_backend_in_build(name: str) -> None:
    name = name.strip().lower()
    stack, backends, _ = install_caps()
    if name not in backends:
        raise RuntimeError(
            f"This FlyDSL install targets stack {stack!r} and does not support compile backend {name!r}. "
            f"Allowed FLYDSL_COMPILE_BACKEND values: {sorted(backends)!r}."
        )


def ensure_runtime_kind_in_build(kind: str) -> None:
    kind = kind.strip().lower()
    stack, _, kinds = install_caps()
    if kind not in kinds:
        raise RuntimeError(
            f"This FlyDSL install targets stack {stack!r} and does not support device runtime kind {kind!r}. "
            f"Allowed FLYDSL_RUNTIME_KIND values: {sorted(kinds)!r}."
        )
