from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
import warnings

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler._annotations import _FALLBACK_WARNED, resolve_signature


@flyc.jit
def _future_runtime_launch(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    pass


def _cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._make_cache_key(bound.arguments)


def test_future_annotations_runtime_int32_ignores_value_in_cache_key():
    """With future-annotations, fx.Int32 must still be recognised as runtime
    so its value does not enter the cache key."""
    key1 = _cache_key(_future_runtime_launch, 1)
    key2 = _cache_key(_future_runtime_launch, 2)

    assert key1 == key2
    assert ("n", int) in key1
    assert ("n", (int, 1)) not in key1
    assert ("n", (int, 2)) not in key1


def test_resolve_signature_resolves_bare_name_under_future_annotations():
    """Bare-name annotations stored as strings by ``from __future__ import
    annotations`` must be resolved against the function's globals."""

    def g(n: fx.Int32):  # bare ref -> stringified to 'fx.Int32'
        return n

    # sanity: future-annotations turned it into a string
    assert g.__annotations__["n"] == "fx.Int32"

    sig = resolve_signature(g)
    assert sig.parameters["n"].annotation is fx.Int32


def test_resolve_signature_fallback_emits_warning_once():
    """When a string annotation cannot be resolved, fallback must warn
    (not silently degrade) and warn only once per function."""

    def f(x):
        return x

    # Inject an unresolvable bare-name annotation (simulating future-annotations
    # against a TYPE_CHECKING-only import).
    f.__annotations__["x"] = "NotARealTypeXYZ"

    _FALLBACK_WARNED.discard(getattr(f, "__qualname__", repr(f)))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig1 = resolve_signature(f)
        sig2 = resolve_signature(f)

    flydsl_warns = [x for x in w if "FlyDSL" in str(x.message)]
    assert len(flydsl_warns) == 1, f"expected 1 FlyDSL warning, got {len(flydsl_warns)}"
    # fallback returns raw signature with the annotation left as the string
    assert sig1.parameters["x"].annotation == "NotARealTypeXYZ"
    assert sig2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
