#!/usr/bin/env python3
"""Correctness + perf tests for the optimized RDNA 3 / 3.5 atom-based GEMM
kernel (``kernels/rdna3_gemm.py``).

Compared to ``test_rdna3_wmma_gemm.py`` (which validates the WMMA atom on
small single-warp PoC kernels), this file exercises the production-shape
kernel with multi-wave layout, LDS double-buffer, and software pipelining.
"""

import logging
import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna3_gemm import create_rdna3_gemm_module
from tests.test_common import verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna3():
    if not (ARCH.startswith("gfx115") or ARCH.startswith("gfx110")):
        pytest.skip(f"RDNA 3 / 3.5 GEMM tests require gfx110x or gfx115x, got {ARCH}")


# Sizes that exercise the full pipeline (require K >= 2*BLOCK_K=64 for
# the 2-stage ping-pong; M, N must be divisible by BLOCK_M = BLOCK_N = 128).


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 128, 64, id="128x128x64"),     # minimum (2 K-tiles)
        pytest.param(128, 128, 256, id="128x128x256"),   # 8 K-tiles
        pytest.param(256, 256, 256, id="256x256x256"),   # multi-block
        pytest.param(512, 512, 512, id="512x512x512"),
        pytest.param(1024, 1024, 1024, id="1024x1024x1024"),
    ],
)
def test_rdna3_gemm_f32_acc(M, N, K, dtype):
    """Optimized RDNA 3 / 3.5 atom-based GEMM with F32 accumulator."""
    _requires_rdna3()

    torch.manual_seed(42)
    torch_dtype = torch.float16 if dtype == "f16" else torch.bfloat16

    A = (torch.randn(M, K, dtype=torch_dtype, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch_dtype, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn, _, _, _ = create_rdna3_gemm_module(
        M, N, K, in_dtype=dtype, out_dtype="f32"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C, C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K, dtype",
    [
        pytest.param(128, 128, 256, "f16", id="f16_acc-128x128x256"),
        pytest.param(256, 256, 512, "f16", id="f16_acc-256x256x512"),
        pytest.param(128, 128, 256, "bf16", id="bf16_acc-128x128x256"),
        pytest.param(256, 256, 512, "bf16", id="bf16_acc-256x256x512"),
    ],
)
def test_rdna3_gemm_same_precision_acc(M, N, K, dtype):
    """Same-precision accumulator (f16->f16, bf16->bf16) exercising
    the f32-internal-acc fallback in MmaOpRDNA3_WMMAType."""
    _requires_rdna3()

    torch.manual_seed(42)
    torch_dtype = torch.float16 if dtype == "f16" else torch.bfloat16

    A = (torch.randn(M, K, dtype=torch_dtype, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch_dtype, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch_dtype, device="cuda")

    launch_fn, _, _, _ = create_rdna3_gemm_module(
        M, N, K, in_dtype=dtype, out_dtype=dtype
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A.float() @ B_T.float().T).to(torch_dtype)
    # Lower precision: looser bounds.
    atol = 0.1 if dtype == "f16" else 0.5
    rtol = 0.1 if dtype == "f16" else 0.1
    assert verify_output(C, C_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "waves_m, waves_n, reg_m, reg_n, reg_k",
    [
        (1, 1, 1, 1, 1),  # 1 wave32, single-WMMA — minimum config
        (1, 1, 4, 4, 2),  # 1 wave but big reg tile
        (2, 2, 2, 2, 2),  # 4 waves, modest reg
        (2, 2, 4, 4, 2),  # default production config
        (2, 2, 4, 4, 1),  # smaller K tile
    ],
)
def test_rdna3_gemm_wave_reg_configs(waves_m, waves_n, reg_m, reg_n, reg_k):
    """Sweep wave / register tile knobs to validate the parametrized
    multi-wave + multi-reg paths."""
    _requires_rdna3()

    BLOCK_M = 16 * reg_m * waves_m
    BLOCK_N = 16 * reg_n * waves_n
    BLOCK_K = 16 * reg_k
    # Pick the smallest valid problem for this config (need K >= 2*BLOCK_K
    # for the ping-pong pipeline, and M, N >= BLOCK_M, BLOCK_N).
    M, N, K = BLOCK_M, BLOCK_N, max(2 * BLOCK_K, 64)

    torch.manual_seed(42)
    A = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn, _, _, _ = create_rdna3_gemm_module(
        M, N, K, in_dtype="f16", out_dtype="f32",
        waves_m=waves_m, waves_n=waves_n,
        reg_m=reg_m, reg_n=reg_n, reg_k=reg_k,
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C, C_ref, atol=0.05, rtol=0.05)
