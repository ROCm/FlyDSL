#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device correctness for gfx120x native iu4 WMMA GEMM.

One-smoke shapes vs torch unpack reference and vs unpack->iu8 fallback.
Default W4 paths (AWQ / SVD) stay unpack; this exercises the reusable GEMM.
"""

import os
import sys

import pytest  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_int4_codec import (  # noqa: E402
    pack_int4_row_major,
    unpack_int4_row_major,
)

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 iu4 GEMM requires gfx120*, got {_ARCH}", allow_module_level=True)

from kernels.gemm.rdna4_iu4_gemm import (  # noqa: E402
    iu4_gemm,
    pick_tile_config,
    shapes_ok_for_native_iu4,
)


def _torch_ref(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    *,
    signed: bool = True,
) -> torch.Tensor:
    a = unpack_int4_row_major(a_packed).to(torch.int32)
    b = unpack_int4_row_major(b_packed).to(torch.int32)
    if not signed:
        a = unpack_int4_row_major(a_packed).to(torch.uint8).to(torch.int32)
        # unsigned path: codec signed-extends; re-unpack as uint via mask
        from kernels.quant.rdna4_int4_codec import unpack_uint4_row_major

        a = unpack_uint4_row_major(a_packed).to(torch.int32)
        b = unpack_uint4_row_major(b_packed).to(torch.int32)
    acc = (a.to(torch.float32) @ b.to(torch.float32).T).to(torch.int32)
    if scale_a is None and scale_b is None:
        return acc
    sa = scale_a.reshape(-1, 1).to(torch.float32)
    sb = scale_b.reshape(1, -1).to(torch.float32)
    return acc.to(torch.float32) * sa * sb


@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
    ],
    ids=["16cubed", "32cubed", "64cubed"],
)
def test_iu4_gemm_i32_vs_torch(M: int, N: int, K: int) -> None:
    """Raw i32 accumulator vs torch unpack reference (one shape at a time)."""
    assert shapes_ok_for_native_iu4(M, N, K)
    torch.manual_seed(2026 + M + N + K)
    logical_a = torch.randint(-7, 8, (M, K), dtype=torch.int8)
    logical_b = torch.randint(-7, 8, (N, K), dtype=torch.int8)
    a_p = pack_int4_row_major(logical_a).contiguous().cuda()
    b_p = pack_int4_row_major(logical_b).contiguous().cuda()

    out = iu4_gemm(a_p, b_p, out_dtype=torch.int32, prefer_native=True)
    torch.cuda.synchronize()
    ref = (logical_a.to(torch.float32) @ logical_b.to(torch.float32).T).to(torch.int32)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_iu4_gemm_scaled_bf16_vs_torch() -> None:
    M, N, K = 64, 64, 64
    torch.manual_seed(7)
    logical_a = torch.randint(-7, 8, (M, K), dtype=torch.int8)
    logical_b = torch.randint(-7, 8, (N, K), dtype=torch.int8)
    a_p = pack_int4_row_major(logical_a).contiguous().cuda()
    b_p = pack_int4_row_major(logical_b).contiguous().cuda()
    sa = (torch.rand(M, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    sb = (torch.rand(N, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()

    out = iu4_gemm(a_p, b_p, sa, sb, out_dtype=torch.bfloat16, prefer_native=True)
    torch.cuda.synchronize()
    ref = _torch_ref(a_p.cpu(), b_p.cpu(), sa.cpu(), sb.cpu()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=4e-3, atol=2e-3)


def test_iu4_gemm_native_matches_unpack_iu8_fallback() -> None:
    """Native path vs prefer_native=False (unpack→iu8) on one smoke shape."""
    M, N, K = 64, 64, 64
    torch.manual_seed(11)
    logical_a = torch.randint(-7, 8, (M, K), dtype=torch.int8)
    logical_b = torch.randint(-7, 8, (N, K), dtype=torch.int8)
    a_p = pack_int4_row_major(logical_a).contiguous().cuda()
    b_p = pack_int4_row_major(logical_b).contiguous().cuda()
    sa = torch.ones(M, device="cuda", dtype=torch.float32)
    sb = torch.ones(N, device="cuda", dtype=torch.float32)

    native = iu4_gemm(a_p, b_p, sa, sb, out_dtype=torch.float32, prefer_native=True)
    torch.cuda.synchronize()
    fallback = iu4_gemm(a_p, b_p, sa, sb, out_dtype=torch.float32, prefer_native=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(native, fallback, rtol=1e-5, atol=1e-5)


def test_iu4_gemm_partial_tile() -> None:
    M, N, K = 37, 50, 48
    assert shapes_ok_for_native_iu4(M, N, K)
    torch.manual_seed(99)
    logical_a = torch.randint(-7, 8, (M, K), dtype=torch.int8)
    logical_b = torch.randint(-7, 8, (N, K), dtype=torch.int8)
    a_p = pack_int4_row_major(logical_a).contiguous().cuda()
    b_p = pack_int4_row_major(logical_b).contiguous().cuda()
    out = iu4_gemm(a_p, b_p, out_dtype=torch.int32, prefer_native=True)
    torch.cuda.synchronize()
    ref = (logical_a.to(torch.float32) @ logical_b.to(torch.float32).T).to(torch.int32)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_iu4_tile_selection() -> None:
    cfg = pick_tile_config(128, 128, 512)
    assert cfg.bm == 128 and cfg.bk >= 64
    cfg2 = pick_tile_config(32, 32, 64)
    assert cfg2.bm == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
