#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness for the gfx120 INT4 WMMA GEMM example.

C[M, N] = A[M, K] @ B[N, K].T. Inputs are packed with pack_int4. The
reference is the logical int64 matmul, atol=0.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from tests.kernels.int4_pack import pack_int4  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 INT4 GEMM requires gfx120*, got {_ARCH}", allow_module_level=True)

from kernels.gemm.rdna4_int4_gemm import choose_int4_tile, create_wmma_int4_gemm_module  # noqa: E402


def _packed_operands(M, N, K, seed=2026):
    torch.manual_seed(seed)
    logical_a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    logical_b = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    logical_a[0].fill_(-8)
    logical_a[1 % M].fill_(7)
    logical_a[2 % M].zero_()
    logical_b[0].fill_(7)
    logical_b[1 % N].fill_(-8)
    logical_b[2 % N].zero_()
    a = torch.from_numpy(pack_int4(logical_a.cpu().numpy())).contiguous()
    b = torch.from_numpy(pack_int4(logical_b.cpu().numpy())).contiguous()
    return logical_a, logical_b, a, b


@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 16, 32),
        (32, 16, 64),
        (16, 32, 64),
        (32, 48, 96),
        (48, 48, 96),
        (64, 64, 128),
        (128, 128, 512),
        (16, 16, 512),
    ],
)
def test_rdna4_int4_gemm_exact(M, N, K):
    logical_a, logical_b, a, b = _packed_operands(M, N, K)
    a = a.to("cuda")
    b = b.to("cuda")
    c = torch.full((M, N), -1, dtype=torch.int32, device="cuda")

    launch = create_wmma_int4_gemm_module(M, N, K)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = logical_a.to(torch.int64) @ logical_b.to(torch.int64).T
    torch.testing.assert_close(c.cpu(), ref.to(torch.int32), atol=0, rtol=0)


def test_rdna4_int4_tile_is_multiwave():
    block_m, block_n, block_k, waves_m, waves_n, reg_m, reg_n = choose_int4_tile(128, 128, 512)
    assert waves_m >= 2 and waves_n >= 2
    assert block_m >= 64 and block_n >= 64
    assert block_k < 512
    assert reg_m * reg_n >= 4
