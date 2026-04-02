# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for compile_hints, CompileCallable, and llvm_options.

Validates:
  - LLVM cl::opt set/restore (bool, int, str) via Python bindings
  - llvm_options context manager scoping
  - CompileCallable subscript syntax: flyc.compile[hints]
  - compile_hints propagation through JitFunction → MlirCompiler → pipeline
"""
import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx


# ──────────────────────────────────────────────────────────────
# Minimal kernel for compile-hint tests
# ──────────────────────────────────────────────────────────────

@flyc.kernel
def _noop_kernel():
    pass


@flyc.jit
def _noop_launch(stream: fx.Stream = fx.Stream(None)):
    _noop_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


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

    def test_mixed_types(self):
        from flydsl.compiler.llvm_options import llvm_options

        with llvm_options({
            "enable-post-misched": False,
            "opt-bisect-limit": 100,
            "module-summary-dot-file": "/tmp/test",
        }):
            pass  # just verify no exceptions

    def test_invalid_type_raises(self):
        from flydsl.compiler.llvm_options import llvm_options

        with pytest.raises(TypeError, match="Unsupported type"):
            with llvm_options({"enable-post-misched": [1, 2, 3]}):
                pass


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
        """Verify fast_fp_math/unsafe_fp_math appear in rocdl-attach-target."""
        from flydsl.compiler.backends import rocm
        captured = {}

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
        exe = flyc.compile[{
            "llvm_options": {"enable-post-misched": False},
        }](_noop_launch)
        exe()  # should compile and run without error
