# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device launches for gfx120x fixed product hosts.

Skips unless the device is gfx120x. Other architectures do not import the kernels
beyond the arch check inside the test body.
"""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


def _require_gfx120x() -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")


def test_w8a16_linear_matches_gemm_contract() -> None:
    _require_gfx120x()
    from kernels.gemm.rdna4_w8a16_linear import w8a16_linear

    torch.manual_seed(0)
    m, n, k = 32, 64, 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    scale = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    out = w8a16_linear(x, weight, scale)
    torch.cuda.synchronize()
    ref = (x.float() @ (weight.float() * scale.reshape(-1, 1)).T).to(x.dtype)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


def test_int8_linear_iu8() -> None:
    _require_gfx120x()
    from kernels.gemm.rdna4_int8_linear import int8_linear
    from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise

    torch.manual_seed(3)
    m, n, k = 32, 64, 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    w_scale = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    a8, x_scale = quantize_int8_rowwise(x)
    out = int8_linear(a8, weight, x_scale.reshape(-1), w_scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = ((a8.float() * x_scale.reshape(-1, 1).float()) @ (weight.float() * w_scale.reshape(-1, 1)).T).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


def test_scaled_mm_fp8_small() -> None:
    _require_gfx120x()
    from kernels.gemm.rdna4_scaled_mm_fp8 import scaled_mm_fp8

    torch.manual_seed(2)
    m = n = k = 64
    a = torch.randn((m, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn)
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)
    out = scaled_mm_fp8(a, b, scale_a, scale_b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float().T) * scale_a[0] * scale_b[0]
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.08)
