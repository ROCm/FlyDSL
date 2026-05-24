# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Annotation resolution helpers.

``from __future__ import annotations`` (and PEP 649 in future Python versions)
stores function annotations as strings rather than the resolved Python objects.
The JIT cache-key dispatch in ``jit_argument.py`` keys on the annotation's
identity (``hasattr(ann, "__get_c_pointers__")`` etc.), so unresolved string
annotations silently make runtime-typed parameters look like opaque scalars and
their values leak into the cache key.  This helper centralises the resolution
so ``jit_function`` and ``kernel_function`` stay in sync.
"""

import inspect
import warnings

_FALLBACK_WARNED = set()


def resolve_signature(func):
    """Return ``inspect.Signature`` with string annotations resolved.

    Uses ``inspect.signature(func, eval_str=True)`` so annotations stored as
    strings (under ``from __future__ import annotations``) are evaluated
    against the function's globals.  Only ``NameError`` triggers fallback —
    that's the one expected failure mode (e.g. ``TYPE_CHECKING``-guarded
    imports, forward references).  In that case we fall back to the raw
    signature and emit a one-shot warning per function, because the cache-key
    dispatch will then degrade silently (the exact bug this helper exists to
    avoid).
    """
    try:
        return inspect.signature(func, eval_str=True)
    except NameError as exc:
        key = getattr(func, "__qualname__", repr(func))
        if key not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(key)
            warnings.warn(
                f"FlyDSL: could not resolve string annotations for {key!r} "
                f"({exc}). Falling back to raw signature; runtime-typed "
                f"parameters may not be recognised and their values may leak "
                f"into the JIT cache key. Import the referenced types at "
                f"module top-level (outside TYPE_CHECKING), or drop "
                f"``from __future__ import annotations`` in this module.",
                stacklevel=2,
            )
        return inspect.signature(func)
