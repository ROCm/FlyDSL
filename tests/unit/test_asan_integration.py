# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for AddressSanitizer (ASan) integration.

These tests verify that the ASan environment variables and backend
configuration are properly wired up. Actual ASan functionality
requires a GPU with xnack+ support.
"""

import os
import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils import env
from flydsl.compiler.backends import get_backend


class TestAsanEnvironmentVariables:
    """Test ASan environment variable handling."""

    def test_enable_asan_default_is_false(self):
        """By default, ASan should be disabled."""
        assert env.debug.enable_asan is False

    def test_asan_instrument_lds_default_is_false(self):
        """By default, LDS instrumentation should be disabled."""
        assert env.debug.asan_instrument_lds is False

    def test_enable_asan_env_var_parsing(self, monkeypatch):
        """FLYDSL_DEBUG_ENABLE_ASAN=1 should enable ASan."""
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_ASAN", "1")
        # Create new env manager to read updated env
        from flydsl.utils.env import DebugEnvManager

        debug_env = DebugEnvManager()
        assert debug_env.enable_asan is True

    def test_enable_asan_env_var_true_string(self, monkeypatch):
        """FLYDSL_DEBUG_ENABLE_ASAN=true should enable ASan."""
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_ASAN", "true")
        from flydsl.utils.env import DebugEnvManager

        debug_env = DebugEnvManager()
        assert debug_env.enable_asan is True

    def test_enable_asan_env_var_yes_string(self, monkeypatch):
        """FLYDSL_DEBUG_ENABLE_ASAN=yes should enable ASan."""
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_ASAN", "yes")
        from flydsl.utils.env import DebugEnvManager

        debug_env = DebugEnvManager()
        assert debug_env.enable_asan is True


class TestAsanBackendConfiguration:
    """Test ASan backend configuration."""

    def test_backend_hash_includes_asan_status(self, monkeypatch):
        """Backend hash should differ when ASan is enabled."""
        backend = get_backend()

        # Without ASan
        monkeypatch.setattr(env.debug, "enable_asan", False)
        hash_without_asan = backend.hash()

        # With ASan
        monkeypatch.setattr(env.debug, "enable_asan", True)
        hash_with_asan = backend.hash()

        assert hash_without_asan != hash_with_asan
        assert ":asan" in hash_with_asan

    def test_gpu_module_targets_with_asan(self, monkeypatch):
        """GPU module targets should include xnack+ when ASan is enabled."""
        backend = get_backend()

        # Mock ASan enabled
        monkeypatch.setattr(env.debug, "enable_asan", True)
        targets = backend.gpu_module_targets()

        # Check that xnack+ is in the target
        assert len(targets) == 1
        assert ":xnack+" in targets[0]

    def test_pipeline_fragments_includes_asan_options(self, monkeypatch):
        """Pipeline fragments should include ASan compiler options."""
        backend = get_backend()

        monkeypatch.setattr(env.debug, "enable_asan", True)
        fragments = backend.pipeline_fragments(compile_hints={})

        # Find gpu-module-to-binary fragment
        binary_frag = None
        for frag in fragments:
            if "gpu-module-to-binary" in frag:
                binary_frag = frag
                break

        assert binary_frag is not None
        assert "-fsanitize=address" in binary_frag
        assert "-shared-libsan" in binary_frag


class TestAsanCacheKey:
    """Test ASan integration with JIT cache key."""

    @flyc.kernel
    def _simple_kernel(x: fx.Tensor):
        pass

    @flyc.jit
    def _simple_launch(x: fx.Tensor):
        TestAsanCacheKey._simple_kernel(x).launch(grid=(1,), block=(64,))

    def test_cache_key_includes_asan_flag(self, monkeypatch):
        """Cache key should include ASan flag when enabled."""
        jit_fn = TestAsanCacheKey._simple_launch
        jit_fn._ensure_sig()

        bound = jit_fn._sig.bind(None)  # Pass None as tensor placeholder
        bound.apply_defaults()

        # Without ASan
        monkeypatch.setattr(env.debug, "enable_asan", False)
        key_without_asan = jit_fn._make_cache_key(bound.arguments)

        # With ASan
        monkeypatch.setattr(env.debug, "enable_asan", True)
        key_with_asan = jit_fn._make_cache_key(bound.arguments)

        # Keys should be different
        assert key_without_asan != key_with_asan

        # ASan key should contain asan marker
        assert any("_asan_" in str(part) for part in key_with_asan)


@pytest.mark.skipif(
    os.environ.get("FLYDSL_DEBUG_ENABLE_ASAN") != "1",
    reason="ASan runtime test requires FLYDSL_DEBUG_ENABLE_ASAN=1 and xnack+ GPU",
)
class TestAsanRuntime:
    """Test ASan runtime functionality (requires xnack+ GPU)."""

    def test_asan_library_resolution(self):
        """Test that ASan runtime library can be found."""
        from flydsl.compiler.jit_executor import _find_asan_runtime_lib

        asan_lib = _find_asan_runtime_lib()
        # Should either find a library or return empty string
        assert asan_lib == "" or "asan" in asan_lib.lower()
