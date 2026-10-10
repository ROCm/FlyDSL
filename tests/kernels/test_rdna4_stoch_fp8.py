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


def test_stoch_fp8_uses_f32_not_fp16() -> None:
    """A fraction fp16 would round away must change the fp8 bin."""
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]
    if not arch.startswith("gfx120"):
        pytest.skip(f"requires gfx120x, got {arch!r}")
    # 1.0625 is an fp16 number. Subtracting 2**-12 is not. fp16 rounds it back
    # to 1.0625, and rng byte 128 then rounds that up to the next e4m3 bin.
    # The f32 value stays in the 1.0 bin.
    value = torch.tensor([1.0625 - 2.0**-12], device="cuda", dtype=torch.float32)
    rng = torch.tensor([128], device="cuda", dtype=torch.uint8)
    run(
        build_stoch_fp8_module(in_dtype="float32", path="bitcast"),
        ptr(value),
        ptr(rng),
        1,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    got = float(rng.view(torch.float8_e4m3fn).float())
    assert got == 1.0, got
