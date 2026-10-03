#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness for gfx120x FP8 e4m3 / e5m2 tensorwise scaled_mm."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.common.tensor_shim import _run_compiled  # noqa: E402
from kernels.gemm.rdna4_scaled_mm_fp8 import (  # noqa: E402
    build_scaled_mm_fp8_module,
    pick_tile_config,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 FP8 scaled-MM requires gfx120x, got {ARCH}", allow_module_level=True)


def _ptr(t: torch.Tensor) -> PointerJitArg:
    """Pass a byte-addressed device tensor to the raw-pointer launcher."""
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _run_scaled_mm(
    a: object,
    b_nk: object,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    out_dtype: torch.dtype | None = torch.bfloat16,
    e5m2: bool = False,
) -> torch.Tensor:
    m, k = a.shape
    n = b_nk.shape[0]
    cfg = pick_tile_config(m, n, k)
    skip_bounds = m % cfg.bm == 0 and n % cfg.bn == 0
    launch = build_scaled_mm_fp8_module(out_dtype_name(out_dtype), cfg, skip_bounds, e5m2=e5m2)
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    _run_compiled(
        launch,
        _ptr(a.view(torch.uint8)),
        _ptr(b_nk.view(torch.uint8)),
        _ptr(out.view(torch.uint8)),
        _ptr(scale_a),
        _ptr(scale_b),
        m,
        n,
        k,
        torch.cuda.current_stream(),
    )
    return out


def out_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[dtype]


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(130, 144, 128, id="ragged-130x144x128"),
    ],
)
def test_rdna4_scaled_mm_fp8(m: int, n: int, k: int) -> None:
    """Tensorwise e4m3 GEMM matches an f32 reference after the output cast."""
    torch.manual_seed(17)
    a_f32 = torch.randn((m, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_f32 = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    a = a_f32.to(torch.float8_e4m3fn).contiguous()
    b_nk = b_f32.to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = _run_scaled_mm(a, b_nk, scale_a, scale_b, e5m2=False)
    torch.cuda.synchronize()
    ref = (a.float() @ b_nk.float().T) * scale_a[0] * scale_b[0]
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.08)


def test_rdna4_scaled_mm_fp8_e5m2() -> None:
    """Tensorwise e5m2 GEMM matches f32 reference (bf8 / Float8E5M2 WMMA)."""
    torch.manual_seed(19)
    m = n = k = 64
    a_f32 = torch.randn((m, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_f32 = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    a = a_f32.to(torch.float8_e5m2).contiguous()
    b_nk = b_f32.to(torch.float8_e5m2).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = _run_scaled_mm(a, b_nk, scale_a, scale_b, e5m2=True)
    torch.cuda.synchronize()
    ref = (a.float() @ b_nk.float().T) * scale_a[0] * scale_b[0]
    torch.testing.assert_close(out.float(), ref, rtol=0.05, atol=0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
