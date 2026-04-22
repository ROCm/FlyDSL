# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Verify upstream gpu-to-llvm behavior with asyncObject vs asyncDependencies.

MLIR IR-level tests that document how the upstream gpu-to-llvm pass handles
different async patterns. These tests use fly-opt directly (no GPU required).
"""
import os
import subprocess
import textwrap

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _find_fly_opt():
    if "FLY_OPT" in os.environ:
        return os.environ["FLY_OPT"]
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dir = os.environ.get("FLY_BUILD_DIR", os.path.join(repo_root, "build-fly"))
    candidate = os.path.join(build_dir, "bin", "fly-opt")
    if os.path.isfile(candidate):
        return candidate
    import shutil
    return shutil.which("fly-opt") or candidate


FLY_OPT = _find_fly_opt()
GPU_TO_LLVM = "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}"


def _run_pass(mlir_ir: str, pipeline: str = GPU_TO_LLVM) -> subprocess.CompletedProcess:
    if not os.path.isfile(FLY_OPT):
        pytest.skip(f"fly-opt not found at {FLY_OPT}")
    return subprocess.run(
        [FLY_OPT, f"--pass-pipeline=builtin.module({pipeline})"],
        input=mlir_ir,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestUpstreamGpuToLLVM:
    """Upstream gpu-to-llvm pass behavior on various async patterns."""

    def test_async_object_stream_is_silently_dropped(self):
        """asyncObject (%stream) is IGNORED by upstream gpu-to-llvm.

        The stream pointer disappears from the lowered IR — the kernel
        launches on null stream instead of the user's stream.
        """
        ir_text = textwrap.dedent("""\
            module attributes {gpu.container_module} {
              gpu.binary @kb [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "">]
              func.func @f(%arg0: !llvm.ptr, %stream: !llvm.ptr) {
                %c1 = arith.constant 1 : index
                gpu.launch_func <%stream : !llvm.ptr> @kb::@k
                    blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
                    args(%arg0 : !llvm.ptr)
                return
              }
            }
        """)
        result = _run_pass(ir_text)
        assert result.returncode == 0, f"Pass failed unexpectedly:\n{result.stderr}"
        out = result.stdout
        assert "mgpuStreamCreate" not in out
        assert "gpu.launch_func  @kb::@k" in out or "gpu.launch_func @kb::@k" in out, (
            f"Expected asyncObject to be dropped.\nOutput:\n{out}"
        )

    def test_async_deps_require_gpu_async_token_type(self):
        """asyncDependencies MUST be !gpu.async.token, not !llvm.ptr."""
        ir_text = textwrap.dedent("""\
            module attributes {gpu.container_module} {
              gpu.binary @kb [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "">]
              func.func @f(%arg0: !llvm.ptr, %stream: !llvm.ptr) {
                %c1 = arith.constant 1 : index
                %token = gpu.launch_func async [%stream] @kb::@k
                    blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
                    args(%arg0 : !llvm.ptr)
                gpu.wait [%token]
                return
              }
            }
        """)
        result = _run_pass(ir_text)
        assert result.returncode != 0
        assert "expects different type" in result.stderr or "gpu.async.token" in result.stderr

    def test_async_token_chain_destroys_stream(self):
        """async token chain causes mgpuStreamCreate + mgpuStreamDestroy."""
        ir_text = textwrap.dedent("""\
            module attributes {gpu.container_module} {
              gpu.binary @kb [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "">]
              func.func @f(%arg0: !llvm.ptr) {
                %c1 = arith.constant 1 : index
                %t0 = gpu.wait async
                %t1 = gpu.launch_func async [%t0] @kb::@k
                    blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
                    args(%arg0 : !llvm.ptr)
                gpu.wait [%t1]
                return
              }
            }
        """)
        result = _run_pass(ir_text)
        assert result.returncode == 0, f"Pass failed:\n{result.stderr}"
        out = result.stdout
        assert "mgpuStreamCreate" in out
        assert "mgpuStreamDestroy" in out
        assert "mgpuStreamSynchronize" in out

    def test_sync_launch_baseline(self):
        """Synchronous launch (no async) — no stream ops generated."""
        ir_text = textwrap.dedent("""\
            module attributes {gpu.container_module} {
              gpu.binary @kb [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "">]
              func.func @f(%arg0: !llvm.ptr) {
                %c1 = arith.constant 1 : index
                gpu.launch_func @kb::@k
                    blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
                    args(%arg0 : !llvm.ptr)
                return
              }
            }
        """)
        result = _run_pass(ir_text)
        assert result.returncode == 0, f"Pass failed:\n{result.stderr}"
        out = result.stdout
        assert "mgpuStreamCreate" not in out
        assert "mgpuStreamDestroy" not in out
