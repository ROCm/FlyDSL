#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x SiLU*mul and chunk-2 SwiGLU."""

import pytest
import torch
import torch.nn.functional as F

from kernels.quant.rdna4_swiglu import build_silu_mul_module, build_swiglu_chunk_module
from tests.kernels._rdna4_test_utils import ptr, run

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


def _require_gfx120x() -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")


def test_rdna4_silu_mul() -> None:
    _require_gfx120x()
    m, n = 64, 2048
    gate = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    out = torch.empty_like(gate)
    run(build_silu_mul_module(), ptr(gate), ptr(up), ptr(out), m * n, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.allclose(out.float(), F.silu(gate.float()) * up.float(), rtol=3e-2, atol=3e-2)


def test_rdna4_swiglu_chunk() -> None:
    _require_gfx120x()
    m, half_w = 64, 2048
    x = torch.randn((m, half_w * 2), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, half_w), device="cuda", dtype=torch.bfloat16)
    run(
        build_swiglu_chunk_module(),
        ptr(x),
        ptr(out),
        m * half_w,
        half_w,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    gate, up = x.chunk(2, dim=-1)
    assert torch.allclose(out.float(), F.silu(gate.float()) * up.float(), rtol=3e-2, atol=3e-2)
