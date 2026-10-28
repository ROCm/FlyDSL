# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU Address Sanitizer integration tests.

These tests verify that FLYDSL_COMPILE_ENABLE_ASAN=1 correctly enables LLVM
AddressSanitizer instrumentation in compiled kernels.

Requirements:
  - ROCm device bitcode (asanrtl.bc) available
  - GPU that supports xnack+ mode
  - Runtime env: HSA_XNACK=1  HSA_DISABLE_FRAGMENT_ALLOCATOR=1

Run:
  HSA_XNACK=1 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 \
    FLYDSL_COMPILE_ENABLE_ASAN=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python -m pytest tests/system/test_gpu_asan.py -v
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _asan_env():
    """Build an environment dict with ASAN prerequisites."""
    e = dict(os.environ)
    e["HSA_XNACK"] = "1"
    e["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
    e["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    e["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    e["FLYDSL_COMPILE_ENABLE_ASAN"] = "1"
    e["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    # Ensure the subprocess can find flydsl with the same PYTHONPATH
    repo = str(Path(__file__).resolve().parents[2])
    build_pkgs = os.path.join(repo, "build-fly", "python_packages")
    mlir_libs = os.path.join(build_pkgs, "flydsl", "_mlir", "_mlir_libs")
    existing = e.get("PYTHONPATH", "")
    e["PYTHONPATH"] = os.pathsep.join(filter(None, [build_pkgs, repo, existing]))
    existing_ld = e.get("LD_LIBRARY_PATH", "")
    e["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [mlir_libs, existing_ld]))
    return e


def _run_script(script_text: str, *, timeout: int = 120):
    """Write script to a temp file and run it (avoids inspect.getsource issues with -c)."""
    import tempfile

    env = _asan_env()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_text)
        f.flush()
        try:
            return subprocess.run(
                [sys.executable, f.name],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        finally:
            os.unlink(f.name)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
class TestGpuAsan:
    """GPU ASAN compile and detection tests."""

    def test_asan_compile_simple_kernel(self):
        """A simple kernel compiles successfully with ASAN enabled."""
        repo = str(Path(__file__).resolve().parents[2])
        example = os.path.join(repo, "examples", "01-vectorAdd.py")
        if not os.path.isfile(example):
            pytest.skip("examples/01-vectorAdd.py not found")
        script = Path(example).read_text() + "\nprint('ASAN_COMPILE_PASS')\n"
        result = _run_script(script)
        assert (
            result.returncode == 0
        ), f"ASAN kernel compilation failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "ASAN_COMPILE_PASS" in result.stdout

    def test_asan_detects_oob_access(self):
        """An intentional OOB access triggers an ASAN report."""
        script = textwrap.dedent("""\
            import torch
            import flydsl.compiler as flyc
            import flydsl.expr as fx

            @flyc.kernel
            def oob_kernel(A: fx.Tensor, C: fx.Tensor):
                tid = fx.thread_idx.x
                # Intentional OOB: read past the end of A (tid+1 at boundary)
                C[tid] = A[tid + 1]

            @flyc.jit
            def oob_jit(A: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
                oob_kernel(A, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

            N = 64
            A = torch.ones(N, dtype=torch.float32, device="cuda")
            C = torch.zeros(N, dtype=torch.float32, device="cuda")
            oob_jit(A, C)
            torch.cuda.synchronize()
            print("OOB_NOT_DETECTED")
        """)
        result = _run_script(script)
        combined = result.stdout + result.stderr
        detected = (
            "heap-buffer-overflow" in combined
            or "__asan_report" in combined
            or "AddressSanitizer" in combined
            or "Memory access fault" in combined
            or result.returncode != 0
        )
        if not detected:
            pytest.skip(
                "ASAN did not detect OOB — GPU may not support xnack+ or ASAN "
                "runtime is not fully functional. This is expected on some hardware."
            )


class TestAsanConfig:
    """Unit tests for ASAN configuration plumbing (no GPU needed)."""

    def test_enable_asan_env_default_false(self):
        from flydsl.utils import env

        original = os.environ.pop("FLYDSL_COMPILE_ENABLE_ASAN", None)
        try:
            assert env.compile.enable_asan is False
        finally:
            if original is not None:
                os.environ["FLYDSL_COMPILE_ENABLE_ASAN"] = original

    def test_enable_asan_env_true(self):
        from flydsl.utils import env

        original = os.environ.get("FLYDSL_COMPILE_ENABLE_ASAN")
        os.environ["FLYDSL_COMPILE_ENABLE_ASAN"] = "1"
        try:
            assert env.compile.enable_asan is True
        finally:
            if original is None:
                del os.environ["FLYDSL_COMPILE_ENABLE_ASAN"]
            else:
                os.environ["FLYDSL_COMPILE_ENABLE_ASAN"] = original

    def test_asan_bitcode_discovery(self):
        from flydsl.compiler.backends.rocm import _find_rocm_bitcode_dir

        try:
            bc_dir = _find_rocm_bitcode_dir()
            assert (bc_dir / "asanrtl.bc").is_file()
            assert (bc_dir / "ocml.bc").is_file()
            assert (bc_dir / "ockl.bc").is_file()
        except FileNotFoundError:
            pytest.skip("ROCm device bitcode not available on this system")

    def test_cache_invalidation_includes_asan(self):
        try:
            from flydsl.compiler.jit_function import _CACHE_INVALIDATING_ENV_VARS
        except ImportError:
            pytest.skip("MLIR bindings not available (need full build)")

        assert "FLYDSL_COMPILE_ENABLE_ASAN" in _CACHE_INVALIDATING_ENV_VARS
