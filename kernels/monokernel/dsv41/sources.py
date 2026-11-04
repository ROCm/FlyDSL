# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A digest of every source the V4.1 gfx1250 kernels are built from, part of
each kernel's name: FlyDSL's disk cache keys a build by the jitted function's
own source and closure, not by the helper modules it calls, so an edit to a
helper (``attention.py``, ``gemv.py``, ``*_gfx1250.py``) could otherwise launch
a stale binary -- observed: a fixed mailbox index still running its old code."""

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _files():
    yield from sorted(_ROOT.glob("*.py"))
    yield from sorted(_ROOT.parent.glob("*_gfx1250.py"))


def _digest() -> str:
    h = hashlib.sha256()
    for f in _files():
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


SOURCES = _digest()


def kernel_name(stem: str, **params) -> str:
    """``stem`` + the build's parameters + the source digest."""
    tail = "_".join(f"{k}{int(v) if isinstance(v, bool) else v}" for k, v in params.items())
    return f"{stem}_{tail}_{SOURCES}" if tail else f"{stem}_{SOURCES}"


def cache_tag(name: str, key=None) -> str:
    """A build's identity for FlyDSL's disk cache: every ``@flyc.jit`` launcher
    reads it (``_ = tag``) so the captured string -- the kernel name (sources
    digest) and the whole build key -- is part of the launcher's cache key.
    Without it the cache told builds apart by the launcher's source alone: all 13
    K2 builds of a precompile (S 1 .. 48, timeline, debug offset) hit one entry,
    and a benchmark's K2 at S > 1 ran the S = 1 binary (2026-10-27)."""
    return f"{name}|{key!r}"
