#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x SiLU*mul and chunk-2 SwiGLU."""

import pytest
import torch
import torch.nn.functional as F

from kernels.common.gfx120x_swiglu import build_silu_mul_module, build_swiglu_chunk_module
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


def test_swiglu_chunk_odd_half_stays_unpadded() -> None:
    """A half-width that is not a vector multiple stays on the caller's storage."""
    from kernels.common.gfx120x_swiglu import swiglu_chunk

    _require_gfx120x()
    torch.manual_seed(42)
    m, half_w = 8, 12  # 12 % 8 != 0
    x = torch.randn((m, half_w * 2), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = swiglu_chunk(x, stream=stream)
    stream.synchronize()
    gate, up = x.chunk(2, dim=-1)
    ref = F.silu(gate.float()) * up.float()
    assert out.shape == (m, half_w)
    assert x.shape == (m, half_w * 2)
    assert torch.allclose(out.float(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("n", [3, 12, 100])
def test_silu_mul_odd_numel_stays_unpadded(n: int, dtype: torch.dtype) -> None:
    from kernels.common.gfx120x_swiglu import silu_mul

    _require_gfx120x()
    torch.manual_seed(7 + n)
    gate = torch.randn((n,), device="cuda", dtype=dtype)
    up = torch.randn((n,), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = silu_mul(gate, up, stream=stream)
    stream.synchronize()
    ref = F.silu(gate.float()) * up.float()
    assert out.shape == (n,)
    assert torch.allclose(out.float(), ref, rtol=3e-2, atol=3e-2)


def test_swiglu_odd_half_does_not_pad(monkeypatch: pytest.MonkeyPatch) -> None:
    import kernels.common.gfx120x_pad as padmod
    from kernels.common.gfx120x_swiglu import swiglu_chunk

    _require_gfx120x()

    def _boom(*_args, **_kwargs):
        raise AssertionError("device_pad")

    monkeypatch.setattr(padmod, "device_pad", _boom)
    x = torch.randn((4, 18), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = swiglu_chunk(x, stream=stream)
    stream.synchronize()
    gate, up = x.chunk(2, dim=-1)
    ref = F.silu(gate.float()) * up.float()
    assert out.shape == (4, 9)
    assert torch.allclose(out.float(), ref, rtol=3e-2, atol=3e-2)


def test_swiglu_chunk_noncontiguous_out() -> None:
    """A strided destination is written through a contiguous temporary."""
    from kernels.common.gfx120x_swiglu import swiglu_chunk

    _require_gfx120x()
    torch.manual_seed(9)
    m, half = 8, 16
    x = torch.randn((m, half * 2), device="cuda", dtype=torch.bfloat16)
    buf = torch.empty((m, half * 2), device="cuda", dtype=torch.bfloat16)
    out = buf[:, :half]
    assert not out.is_contiguous()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        got = swiglu_chunk(x, out=out, stream=stream)
    stream.synchronize()
    gate, up = x.chunk(2, dim=-1)
    ref = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)
