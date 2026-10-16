#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x per-tensor FP8 quantize / dequantize."""

import pytest
import torch

from kernels.quant.rdna4_fp8_quant import build_fp8_dequant_module, build_fp8_quant_module, dequantize_fp8
from tests.kernels._rdna4_test_utils import ptr, run

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rdna4_fp8_quant_dequant(dtype: torch.dtype) -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")
    n = 2048
    x = torch.randn(n, device="cuda", dtype=dtype)
    scale = (x.float().abs().amax() / 448.0).clamp_min(1e-8).reshape(1).contiguous()
    packed = torch.empty(n, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream()

    dtype_name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]
    run(
        build_fp8_quant_module(in_dtype=dtype_name),
        ptr(x),
        ptr(packed),
        ptr(scale),
        torch.tensor(n, device="cpu", dtype=torch.int32).item(),
        448.0,
        stream,
    )
    out = torch.empty_like(x)
    run(
        build_fp8_dequant_module(out_dtype=dtype_name),
        ptr(packed),
        ptr(out),
        ptr(scale),
        n,
        stream,
    )
    torch.cuda.synchronize()
    ref_q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    ref = ref_q.float() * scale
    assert torch.allclose(out.float(), ref, rtol=5e-2, atol=5e-2)


def test_fp8_quant_e5m2_ignores_wrong_lp_max() -> None:
    """e5m2 build ignores caller lp_max=448 and uses e5m2 max (57344)."""
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")
    n = 1024
    # Values that need e5m2 range (above e4m3 448).
    x = torch.full((n,), 2000.0, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    packed = torch.empty(n, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream()
    run(
        build_fp8_quant_module(in_dtype="bfloat16", e5m2=True),
        ptr(x),
        ptr(packed),
        ptr(scale),
        n,
        448.0,  # wrong on purpose — host must use 57344
        stream,
    )
    out = torch.empty_like(x)
    run(
        build_fp8_dequant_module(out_dtype="bfloat16", e5m2=True),
        ptr(packed),
        ptr(out),
        ptr(scale),
        n,
        stream,
    )
    torch.cuda.synchronize()
    # If clamped to 448, dequant would be ~448; with e5m2 max, ~2000 survives.
    assert float(out.float().abs().mean()) > 1000.0, float(out.float().abs().mean())


def test_dequantize_fp8_host_matches_builder() -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")
    n = 2048
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    scale = (x.float().abs().amax() / 448.0).clamp_min(1e-8).reshape(1).contiguous()
    packed = torch.empty(n, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.current_stream()
    run(build_fp8_quant_module(in_dtype="bfloat16"), ptr(x), ptr(packed), ptr(scale), n, 448.0, stream)
    out = dequantize_fp8(packed, scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref_q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    ref = ref_q.float() * scale
    assert torch.allclose(out.float(), ref, rtol=5e-2, atol=5e-2)
