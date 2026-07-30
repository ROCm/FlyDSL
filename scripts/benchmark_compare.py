#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared, dependency-free benchmark regression policy helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path


@dataclass(frozen=True)
class Threshold:
    relative_pct: float
    absolute: float
    direction: str = "lower_better"


@dataclass(frozen=True)
class Regression:
    delta: float
    delta_pct: float
    regressed: bool


def compare_values(baseline: float, current: float, threshold: Threshold) -> Regression:
    """Compare values using the allreduce relative-AND-absolute rule."""
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if threshold.direction == "lower_better":
        delta = current - baseline
    elif threshold.direction == "higher_better":
        delta = baseline - current
    else:
        raise ValueError(f"unsupported metric direction: {threshold.direction!r}")
    delta_pct = delta / baseline * 100.0
    return Regression(
        delta=delta,
        delta_pct=delta_pct,
        regressed=delta_pct > threshold.relative_pct and delta > threshold.absolute,
    )


class ThresholdConfig:
    """Versioned per-architecture hard-gate allowlist."""

    def __init__(self, raw: dict):
        if raw.get("version") != 1:
            raise ValueError("benchmark threshold config must have version 1")
        self._architectures = raw.get("architectures", {})

    @classmethod
    def from_path(cls, path: Path) -> "ThresholdConfig":
        return cls(json.loads(path.read_text()))

    def match(self, *, arch: str, op: str, shape: str, dtype: str) -> Threshold | None:
        for entry in self._architectures.get(arch, []):
            if not fnmatch(op, entry.get("op", "*")):
                continue
            if not fnmatch(shape, entry.get("shape", "*")):
                continue
            if not fnmatch(dtype, entry.get("dtype", "*")):
                continue
            return Threshold(
                relative_pct=float(entry["relative_pct"]),
                absolute=float(entry["absolute_us"]),
                direction="lower_better",
            )
        return None

    def supports_arch(self, arch: str) -> bool:
        return arch in self._architectures
