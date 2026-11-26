# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for compile_hints, CompileCallable, and llvm_options.

Validates:
  - LLVM cl::opt set/restore (bool, int, str) via Python bindings
  - llvm_options context manager scoping
  - CompileCallable subscript syntax: flyc.compile[hints]
  - compile_hints propagation through JitFunction → MlirCompiler → pipeline
"""

import ctypes
import gc
import math
import weakref

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device_runtime.rocm import _hip_library_names

# ──────────────────────────────────────────────────────────────
# Minimal kernel for compile-hint tests
# ──────────────────────────────────────────────────────────────


@flyc.kernel
def _noop_kernel():
    pass


@flyc.jit
def _noop_launch(stream: fx.Stream = fx.Stream(None)):
    _noop_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def _reset_jit_caches(jit_fn):
    jit_fn._mem_cache.clear()
    jit_fn._last_compiled = None
    jit_fn.cache_manager = None


# ──────────────────────────────────────────────────────────────
# Tests: LLVM option Python bindings
# ──────────────────────────────────────────────────────────────


class TestLLVMOptionBindings:
    """Test set_llvm_option_{bool,int,str} directly."""

    @staticmethod
    def _get_fly():
        from flydsl._mlir._mlir_libs import _mlirDialectsFly

        return _mlirDialectsFly

    def test_bool_round_trip(self):
        _fly = self._get_fly()
        # enable-post-misched defaults to true
        old = _fly.set_llvm_option_bool("enable-post-misched", False)
        assert old is True
        restored = _fly.set_llvm_option_bool("enable-post-misched", True)
        assert restored is False

    @pytest.mark.skip(reason="Temporarily disabled: opt-bisect-limit leaks LLVM BISECT logs across pytest.")
    def test_int_round_trip(self):
        _fly = self._get_fly()
        # Use a large limit so it doesn't affect compilation
        old = _fly.set_llvm_option_int("opt-bisect-limit", 2147483647)
        restored = _fly.set_llvm_option_int("opt-bisect-limit", old)
        assert restored == 2147483647

    def test_str_round_trip(self):
        _fly = self._get_fly()
        old = _fly.set_llvm_option_str("module-summary-dot-file", "/tmp/test.dot")
        assert old == ""
        restored = _fly.set_llvm_option_str("module-summary-dot-file", old)
        assert restored == "/tmp/test.dot"

    def test_unknown_option_raises(self):
        _fly = self._get_fly()
        with pytest.raises(RuntimeError, match="Unknown LLVM option"):
            _fly.set_llvm_option_bool("this-option-does-not-exist-xyz", True)


# ──────────────────────────────────────────────────────────────
# Tests: llvm_options context manager
# ──────────────────────────────────────────────────────────────


class TestLLVMOptionsContextManager:
    """Test the llvm_options context manager for scoped set/restore."""

    def test_bool_scoping(self):
        from flydsl.compiler.llvm_options import llvm_options

        _fly = TestLLVMOptionBindings._get_fly()

        # Get baseline
        baseline = _fly.set_llvm_option_bool("enable-post-misched", True)

        with llvm_options({"enable-post-misched": False}):
            # Inside context: should be False
            val = _fly.set_llvm_option_bool("enable-post-misched", False)
            assert val is False  # was set to False by context entry

        # After context: should be restored
        val = _fly.set_llvm_option_bool("enable-post-misched", baseline)
        assert val == baseline

    @pytest.mark.skip(reason="Temporarily disabled: opt-bisect-limit leaks LLVM BISECT logs across pytest.")
    def test_mixed_types(self):
        from flydsl.compiler.llvm_options import llvm_options

        with llvm_options(
            {
                "enable-post-misched": False,
                "opt-bisect-limit": 100,
                "module-summary-dot-file": "/tmp/test",
            }
        ):
            pass  # just verify no exceptions

    def test_invalid_type_raises(self):
        from flydsl.compiler.llvm_options import llvm_options

        with pytest.raises(TypeError, match="Unsupported type"):
            with llvm_options({"enable-post-misched": [1, 2, 3]}):
                pass

    def test_unknown_option_warns_and_is_skipped(self):
        from flydsl.compiler.llvm_options import llvm_options

        # Options moved to TableGen (not cl::opt) must not abort compilation.
        with pytest.warns(RuntimeWarning, match="Unknown LLVM option"):
            with llvm_options({"this-option-does-not-exist-xyz": True, "enable-post-misched": False}):
                pass
        # Known options in the same dict are still applied and restored.
        from flydsl._mlir._mlir_libs import _mlirDialectsFly

        assert _mlirDialectsFly.set_llvm_option_bool("enable-post-misched", True) is True


# ──────────────────────────────────────────────────────────────
# Tests: CompileCallable subscript
# ──────────────────────────────────────────────────────────────


class TestCompileCallable:
    """Test flyc.compile[hints] subscript syntax."""

    def test_subscript_returns_new_callable(self):
        c = flyc.compile[{"fast_fp_math": True}]
        assert hasattr(c, "_compile_hints")
        assert c._compile_hints == {"fast_fp_math": True}

    def test_subscript_non_dict_raises(self):
        with pytest.raises(TypeError, match="expects a dict"):
            flyc.compile["not_a_dict"]

    def test_deferred_compilation_sets_hints(self):
        """flyc.compile[hints](fn) without args should set hints on JitFunction."""
        hinted = flyc.compile[{"fast_fp_math": True, "unsafe_fp_math": True}](_noop_launch)
        assert hinted.compile_hints == {"fast_fp_math": True, "unsafe_fp_math": True}

    def test_bare_compile_no_hints(self):
        """flyc.compile(fn) without subscript should not set hints."""

        @flyc.jit
        def _fresh(stream: fx.Stream = fx.Stream(None)):
            _noop_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        result = flyc.compile(_fresh)
        assert result.compile_hints == {}


# ──────────────────────────────────────────────────────────────
# Tests: compile_hints pipeline propagation
# ──────────────────────────────────────────────────────────────


class TestCompileHintsPropagation:
    """Test that compile_hints flow through to the compilation pipeline."""

    def test_fp_math_reaches_pipeline(self, monkeypatch):
        """Verify fast_fp_math/unsafe_fp_math reach the backend pipeline."""
        from flydsl.compiler.backends import rocm

        captured = {}
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_noop_launch)

        orig = rocm.RocmBackend.pipeline_fragments

        def patched(self, *, compile_hints):
            captured["hints"] = dict(compile_hints)
            return orig(self, compile_hints=compile_hints)

        monkeypatch.setattr(rocm.RocmBackend, "pipeline_fragments", patched)

        exe = flyc.compile[{"fast_fp_math": True, "unsafe_fp_math": True}](_noop_launch)
        exe()

        assert captured["hints"].get("fast_fp_math") is True
        assert captured["hints"].get("unsafe_fp_math") is True

    def test_llvm_options_in_compile_hints(self):
        """Verify llvm_options key is accepted and doesn't crash."""
        _reset_jit_caches(_noop_launch)
        exe = flyc.compile[
            {
                "llvm_options": {"enable-post-misched": False},
            }
        ](_noop_launch)
        exe()  # should compile and run without error


class TestCacheDisabledRegression:
    """Regression coverage for FLYDSL_RUNTIME_ENABLE_CACHE=0 launch path."""

    def test_cache_disabled_keeps_compiled_artifact_alive(self, monkeypatch):
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_noop_launch)

        stream = torch.cuda.current_stream()

        _noop_launch(stream)
        torch.cuda.synchronize()
        assert _noop_launch.cache_manager is None
        assert len(_noop_launch._mem_cache) == 1

        artifact = next(iter(_noop_launch._mem_cache.values()))
        artifact_ref = weakref.ref(artifact)
        _, last_artifact = _noop_launch._last_compiled
        assert artifact is last_artifact

        del artifact
        del last_artifact
        _noop_launch._last_compiled = None
        gc.collect()
        kept_alive = artifact_ref()
        assert kept_alive is not None

        # The second launch should reuse the in-process artifact even when disk
        # cache is disabled.
        _noop_launch(stream)
        torch.cuda.synchronize()
        assert next(iter(_noop_launch._mem_cache.values())) is kept_alive

    def test_cache_disabled_run_perftest_does_not_crash(self, monkeypatch):
        from tests.test_common import run_perftest

        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        _reset_jit_caches(_noop_launch)

        stream = torch.cuda.current_stream()
        _, avg_us = run_perftest(
            lambda: (_noop_launch(stream), torch.cuda.synchronize()),
            num_iters=5,
            num_warmup=1,
        )

        assert math.isfinite(avg_us)
        assert avg_us >= 0
        assert len(_noop_launch._mem_cache) == 1


# ──────────────────────────────────────────────────────────────
# Tests: max_flat_work_group_size at launch
# ──────────────────────────────────────────────────────────────


@flyc.kernel
def _iota_kernel(out: fx.Tensor):
    t = fx.thread_idx.x
    out[t] = fx.Float32(t) + 1.0


@flyc.jit
def _runtime_block_launch(out: fx.Tensor, threads: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _iota_kernel(out).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)


def _take_hip_error():
    """Read and clear HIP's last error, so a refused launch cannot fail a later torch call."""
    for name in _hip_library_names():
        try:
            return ctypes.CDLL(name).hipGetLastError()
        except OSError:
            continue
    pytest.fail("no HIP library to read the last error from")


class TestMaxFlatWorkGroupSize:
    """A runtime block compiles for at most 256 work-items unless max_flat_work_group_size raises it."""

    @pytest.mark.parametrize("hint", [None, 512])
    def test_runtime_block_of_512(self, monkeypatch, hint):
        monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        # flyc.compile[hints] stores its hints on the launcher; restore them after the test.
        monkeypatch.setattr(_runtime_block_launch, "compile_hints", {})
        _reset_jit_caches(_runtime_block_launch)
        launch = _runtime_block_launch
        if hint is not None:
            launch = flyc.compile[{"max_flat_work_group_size": hint}](_runtime_block_launch)
        out = torch.zeros(512, device="cuda")

        launch(out, 512)
        hip_error = _take_hip_error()
        torch.cuda.synchronize()

        if hint is None:
            # HIP refuses the launch and nothing runs. A JIT launch does not raise, so read HIP's error.
            assert hip_error != 0
            assert int(torch.count_nonzero(out)) == 0
        else:
            assert hip_error == 0
            torch.testing.assert_close(out.cpu(), torch.arange(1, 513, dtype=torch.float32))
