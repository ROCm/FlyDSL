# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fixtures for the arch-neutral attention tests.

The tests in this directory reach a backend only through `kernels/attention/dispatch.py`. Collection is safe on any
arch: the dispatch and config modules import nothing heavy, and builders are imported lazily.

`fwd_build` is a **session-scoped build memo** keyed by `(arch, meta, knob pins)`: tests that share a configuration
share its compile, which is what keeps the suite inside its build budget (a build costs far more than a launch at the
shapes used here).
"""

import pytest

_BUILDS = {}


@pytest.fixture(scope="session")
def backend():
    """The attention backend for the current arch; skips when there is none."""
    from kernels.attention import dispatch

    try:
        arch = dispatch.current_arch()
    except Exception as exc:  # no device / no flydsl runtime
        pytest.skip(f"cannot detect the GPU arch: {exc}")
    be = dispatch.backend_for(arch)
    if be is None:
        pytest.skip(f"no attention backend for arch {arch}")
    return be


@pytest.fixture(scope="session")
def arch():
    from kernels.attention import dispatch

    return dispatch.current_arch()


def _memo(kind, backend, arch, meta, pins):
    """One build per distinct *build*: keyed by the resolved traits and knobs, so metadata that differ only in the real
    head dim within a rung (same tile, same padding, same floor) share one compile."""
    knobs = getattr(backend, f"{kind}_knobs")(arch, **pins).resolve(meta)
    key = (kind, arch, backend.build_cache_key(getattr(backend, f"{kind}_traits")(meta, knobs), knobs))
    if key not in _BUILDS:
        _BUILDS[key] = getattr(backend, f"build_{kind}")(meta, knobs)
    return _BUILDS[key]


@pytest.fixture(scope="session")
def fwd_build(backend, arch):
    """`fwd_build(meta, **knob_pins)` -> the forward launcher (memoised for the session)."""

    def get(meta, **pins):
        return _memo("fwd", backend, arch, meta, pins)

    return get


@pytest.fixture(scope="session")
def bwd_build(backend, arch):
    """`bwd_build(meta, fwd=None, dq=None, dkdv=None)` -> a namespace of the three builders for one problem (`.fwd`, `.dq`,
    `.dkdv`), each memoised for the session. The dicts are per-kernel knob pins (`MFMA_ROWS=16`, ...)."""
    from types import SimpleNamespace

    def get(meta, fwd=None, dq=None, dkdv=None):
        return SimpleNamespace(
            fwd=_memo("fwd", backend, arch, meta, fwd or {}),
            dq=_memo("dq", backend, arch, meta, dq or {}),
            dkdv=_memo("dkdv", backend, arch, meta, dkdv or {}),
        )

    return get


@pytest.fixture(scope="session")
def build_count():
    """How many distinct builds the session has made so far (for the build-budget report)."""
    return lambda: len(_BUILDS)


def pytest_sessionfinish(session, exitstatus):
    if _BUILDS:
        print(f"\n[attention tests] distinct builds this session: {len(_BUILDS)}")
