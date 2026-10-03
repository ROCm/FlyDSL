#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness / smoke for gfx120x stochastic FP8 select and bitcast paths."""

import pytest
import torch

from kernels.quant.rdna4_stoch_fp8 import build_stoch_fp8_module
from tests.kernels._rdna4_test_utils import ptr, run

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@pytest.mark.parametrize("n", [4096, 1_048_576])
def test_rdna4_stoch_fp8(n: int) -> None:
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")
    src = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    rng = torch.randint(0, 256, (n,), device="cuda", dtype=torch.uint8)
    path = "bitcast" if n >= 1_048_576 else "select"
    run(
        build_stoch_fp8_module(path=path),
        ptr(src),
        ptr(rng),
        n,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    out = rng.view(torch.float8_e4m3fn)
    assert torch.isfinite(out.float()).all().item()
