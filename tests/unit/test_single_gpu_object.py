#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression tests for ROCm/FlyDSL#1054: exactly one code object per kernel.

Historically two independent sites attached a ``#rocdl.target`` to the same
``gpu.module``: module creation (bare chip-only target) and the
``rocdl-attach-target`` pass (carrying ``O``/``abi``/``fast``/``unsafe-math``/
``wave64`` and ``link_libs``).  ``gpu-module-to-binary`` serialized one object
per target and, with no offloading handler, MLIR selected the *first* — the
bare one — so every kernel was serialized twice and all compile options were
silently dropped.

These tests pin the fixed contract: exactly one ``#gpu.object`` whose target
carries the compile options, and the kernel still runs correctly.
"""

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest_skip = True
else:
    pytest_skip = False

import re

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

if pytest_skip:
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

pytestmark = [pytest.mark.rocm_lower, pytest.mark.l2_device]

# ──────────────────────────────────────────────────────────────
# Minimal copy kernel
# ──────────────────────────────────────────────────────────────


@flyc.kernel
def _vec_kernel(A: fx.Tensor):
    tid = fx.thread_idx.x
    tA = fx.logical_divide(A, fx.make_layout(64, 1))
    atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    reg = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(atom, fx.slice(tA, (None, tid)), reg)
    fx.copy_atom_call(atom, reg, fx.slice(tA, (None, tid)))


@flyc.jit
def _vec_launch(A: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _vec_kernel(A).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def _reset_jit_caches(jit_fn):
    jit_fn._call_state_cache.clear()
    jit_fn._mem_cache.clear()
    jit_fn._last_compiled = None
    jit_fn.manager_key = None
    jit_fn.cache_manager = None


def _gpu_objects(ir_text: str):
    """(count, targets) of #gpu.object entries inside the gpu.binary.

    The IR printer renders the object array in compact single-line form, so
    match the full op text (which contains the embedded ELF as an escaped
    string) rather than assuming pretty-printed brackets.
    """
    binary_m = re.search(r"gpu\.binary\s+@\w+\s+\[", ir_text)
    assert binary_m is not None, "no gpu.binary in compiled module"
    # One #gpu.object per code object; each carries exactly one #rocdl.target.
    targets = re.findall(r"#gpu\.object<(#rocdl\.target<[^>]*>)", ir_text)
    count = len(re.findall(r"#gpu\.object<", ir_text))
    return count, targets


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────


class TestSingleCodeObject:
    def test_exactly_one_gpu_object(self, monkeypatch):
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_vec_launch)

        A = torch.arange(64, dtype=torch.float32, device="cuda")
        exe = flyc.compile(_vec_launch)
        exe(A, torch.cuda.current_stream(A.device))
        torch.cuda.synchronize()

        _key, artifact = _vec_launch._last_compiled
        count, targets = _gpu_objects(artifact._ir_text)
        assert count == 1, f"expected exactly one #gpu.object, found {count}"
        assert len(targets) == 1 and "gfx" in targets[0]

    def test_compile_hints_reach_shipped_object(self, monkeypatch):
        """fast_fp_math/unsafe_fp_math must be visible on the (only) object."""
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_vec_launch)

        A = torch.arange(64, dtype=torch.float32, device="cuda")
        hints = {"fast_fp_math": True, "unsafe_fp_math": True}
        exe = flyc.compile[hints](_vec_launch)
        exe(A, torch.cuda.current_stream(A.device))
        torch.cuda.synchronize()

        _key, artifact = _vec_launch._last_compiled
        count, targets = _gpu_objects(artifact._ir_text)
        assert count == 1
        target_attr = targets[0]
        assert "fast" in target_attr, f"fast_fp_math not on shipped target: {target_attr}"
        assert "unsafe_math" in target_attr, f"unsafe_fp_math not on shipped target: {target_attr}"

    def test_hints_change_codegen_and_result_correct(self, monkeypatch):
        """The shipped object must actually carry the fast-math flags AND the
        kernel must still run correctly with them (end-to-end check)."""
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_vec_launch)

        A = torch.arange(64, dtype=torch.float32, device="cuda")
        exe = flyc.compile[{"fast_fp_math": True}](_vec_launch)
        exe(A, torch.cuda.current_stream(A.device))
        torch.cuda.synchronize()
        torch.testing.assert_close(A, torch.arange(64, dtype=torch.float32, device="cuda"))
