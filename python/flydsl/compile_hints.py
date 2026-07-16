# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared compile-hint validation, layering, and cache identity.

Compile hints have several producers (persistent JIT defaults, nested
thread-local overlays, and autotune candidates), but one effective value must
own both cache identity and compilation.  Layers are shallow: a value in a
later layer replaces the whole value at that key.  ``None`` inherits the
earlier layer.

Occupancy hints additionally use ``0`` as an explicit request to return to the
source/compiler baseline.  Zero is therefore retained while layers are being
merged and removed only when the final effective snapshot is resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

OCCUPANCY_HINT_KEYS = frozenset(("waves_per_eu", "maxnreg"))
_SCALAR_HINT_TYPES = (type(None), bool, int, float, str)


def normalize_fastmath_hint(flags):
    """Canonicalize every fastmath spelling accepted by the DSL context."""
    if flags is None:
        return None
    if isinstance(flags, str):
        return flags
    if isinstance(flags, (set, frozenset)):
        return ",".join(sorted(str(flag) for flag in flags))
    if isinstance(flags, (list, tuple)):
        return ",".join(str(flag) for flag in flags)
    return str(flags)


def snapshot_compile_hint(value, *, path="compile_hints"):
    """Validate and detach a deterministic compile-hint value.

    Backends may add new named hints without changing this module, but their
    values must use one stable grammar: scalar values, string-keyed mappings,
    lists, and tuples.  Rejecting arbitrary objects avoids mutable snapshots
    and ``repr``-based cache collisions.
    """
    if isinstance(value, Mapping):
        snapshot = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} mappings must have string keys, got {key!r}")
            snapshot[key] = snapshot_compile_hint(item, path=f"{path}[{key!r}]")
        return snapshot
    if isinstance(value, list):
        return [snapshot_compile_hint(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(snapshot_compile_hint(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} floats must be finite for a stable cache identity")
    if type(value) in _SCALAR_HINT_TYPES:
        return value
    raise TypeError(
        f"{path} must contain only scalar values, string-keyed mappings, lists, or tuples; "
        f"got {type(value).__name__}"
    )


def stable_hint_key(value):
    """Return a type-aware, insertion-order-independent cache identity."""
    if isinstance(value, Mapping):
        items = sorted(value.items())
        return (dict, tuple((stable_hint_key(key), stable_hint_key(item)) for key, item in items))
    if isinstance(value, list):
        return (list, tuple(stable_hint_key(item) for item in value))
    if isinstance(value, tuple):
        return (tuple, tuple(stable_hint_key(item) for item in value))
    return (type(value), repr(value))


def compile_hints_cache_key(hints: Mapping):
    """Return the canonical cache-key segment for an effective hint snapshot."""
    items = sorted(hints.items())
    return tuple((key, stable_hint_key(value)) for key, value in items)


def normalize_occupancy_hint(value, knob: str):
    """Validate an occupancy value while preserving an explicit zero reset."""

    def normalize_scalar(item):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{knob} must contain non-negative ints, got {item!r}")
        if item < 0:
            raise ValueError(f"{knob} must be >= 0, got {item}")
        return int(item)

    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized = {}
        for kernel_name, item in value.items():
            if not isinstance(kernel_name, str):
                raise TypeError(f"{knob} mapping keys must be kernel names, got {kernel_name!r}")
            normalized[kernel_name] = normalize_scalar(item)
        return normalized
    return normalize_scalar(value)


def merge_compile_hint_layers(*layers: Mapping | None) -> dict:
    """Shallow-merge layers without discarding explicit occupancy resets.

    Later layers win at the top level.  A ``None`` value means that the layer
    has no opinion for that key and therefore inherits the earlier value.
    Unknown keys are deliberately retained for future backend hints.
    """
    merged = {}
    for layer in layers:
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise TypeError(f"compile hints must be mappings, got {type(layer).__name__}")
        for key, value in layer.items():
            if type(key) is not str:
                raise TypeError(f"compile hint keys must be strings, got {key!r}")
            if value is None:
                continue
            if key in OCCUPANCY_HINT_KEYS:
                value = normalize_occupancy_hint(value, key)
            elif key == "fastmath":
                value = normalize_fastmath_hint(value)
            merged[key] = snapshot_compile_hint(value, path=f"compile_hints[{key!r}]")
    return merged


def _remove_occupancy_resets(canonical: dict) -> dict:
    for key in OCCUPANCY_HINT_KEYS:
        value = canonical.get(key)
        if isinstance(value, Mapping):
            value = {kernel_name: item for kernel_name, item in value.items() if item != 0}
            if value:
                canonical[key] = value
            else:
                canonical.pop(key, None)
        elif value == 0:
            canonical.pop(key, None)
    return canonical


def canonicalize_compile_hints(hints: Mapping | None) -> dict:
    """Validate and detach one final hint snapshot, removing occupancy resets."""
    return _remove_occupancy_resets(merge_compile_hint_layers(hints))


def resolve_compile_hints(*layers: Mapping | None) -> dict:
    """Resolve all layers into the detached snapshot used for cache and codegen."""
    return _remove_occupancy_resets(merge_compile_hint_layers(*layers))
