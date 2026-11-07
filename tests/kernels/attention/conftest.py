# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fixtures for the arch-neutral attention tests.

The tests in this directory reach a backend only through
`kernels/attention/dispatch.py`. Collection is safe on any arch: the dispatch
and config modules import nothing heavy, and builders are imported lazily.
"""

import pytest


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
