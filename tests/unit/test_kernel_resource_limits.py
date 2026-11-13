# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""With the check_resource_limits compile hint, a kernel whose static shared memory exceeds
the target's limit fails to compile.

The check itself is the `fly-check-resource-limits` pass (tests/mlir); this pins that
@jit modules carry the target it reads and that the error points at the user's code.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.diagnostics import DSLCompileError
from flydsl.compiler.kernel_function import CompilationContext

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

LIMIT = 65536  # gfx942 LDS bytes per workgroup


@pytest.fixture(autouse=True)
def compile_for_gfx942(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


@flyc.kernel
def _shared_kernel(nbytes: fx.Constexpr[int]):
    fx.SharedAllocator().allocate(nbytes)


@flyc.jit
def _launch(nbytes: fx.Constexpr[int]):
    _shared_kernel(nbytes).launch(grid=1, block=64)


def test_shared_memory_over_limit_is_rejected():
    with pytest.raises(DSLCompileError) as excinfo, CompilationContext.compile_hints({"check_resource_limits": True}):
        _launch(LIMIT + 1)
    error = str(excinfo.value)
    assert f"uses {LIMIT + 1} bytes of shared memory, exceeding the limit of {LIMIT}" in error
    assert __file__ in error and "@flyc.kernel" in error
