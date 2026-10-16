#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Odd and wide gfx120x QK RMSNorm, tail RoPE, and one scale per row.

Activations and the weight are bf16. Quant is fp8 e4m3, fp8 e5m2, or int8.
Heads that are odd or wider than 64 reload each pair from global memory
after the sum-of-squares reduction.
"""

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

if not str(get_rocm_arch()).startswith("gfx120"):
    pytest.skip("requires gfx120x", allow_module_level=True)

from kernels.norm.qk_norm_rope_quant_gfx120x import qk_norm_rope_quant  # noqa: E402


def _expect(x, weight, cos, sin, positions, eps, qmax):
    """Pair math used by the kernel, including an odd tail's unused partner."""
    rows, head = x.shape
    ncos = int(cos.shape[-1])
    rope = ncos * 2
    nope = head - rope
    xf = x.float()
    y = xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) / head + eps) * weight.float()
    out = y.clone()
    amax = torch.zeros(rows, device=x.device, dtype=torch.float32)
    pos = positions.to(torch.long)
    for k in range((head + 1) // 2):
        e0 = 2 * k
        e1 = e0 + 1
        y0 = y[:, e0] if e0 < head else torch.zeros(rows, device=x.device)
        y1 = y[:, e1] if e1 < head else torch.zeros(rows, device=x.device)
        if rope:
            in0 = e0 >= nope
            in1 = e1 >= nope
            if in0:
                safe = (e0 - nope) // 2
            elif in1:
                safe = (e1 - nope) // 2
            else:
                safe = 0
            c = cos[pos, safe]
            s = sin[pos, safe]
            ry0 = y0 * c - y1 * s
            ry1 = y0 * s + y1 * c
            if in0:
                y0 = ry0
            if in1:
                y1 = ry1
        amax = torch.maximum(amax, torch.maximum(y0.abs(), y1.abs()))
        if e0 < head:
            out[:, e0] = y0
        if e1 < head:
            out[:, e1] = y1
    return out, amax.clamp_min(1e-12) / qmax


def _check(x, weight, cos, sin, pos, quant, qmax, dtype, eps):
    packed, scale = qk_norm_rope_quant(x, weight, cos, sin, pos, quant=quant, eps=eps)
    y, expect = _expect(x, weight, cos, sin, pos, eps, qmax)
    assert packed.shape == x.shape and packed.dtype == dtype
    assert scale.shape == (x.shape[0],) and scale.dtype == torch.float32
    assert (scale - expect).abs().max().item() < 1e-3, quant
    got = packed.float() * scale[:, None]
    ref_q = (y / expect[:, None]).to(dtype).float() * expect[:, None]
    err = (got - y).abs().max().item()
    bound = 4 * (ref_q - y).abs().max().item() + 1e-3
    assert err <= bound, (quant, x.shape[-1], err, bound)


@pytest.mark.parametrize(
    ("head", "rope", "quant", "qmax", "dtype", "eps", "strided"),
    [
        (65, 32, "fp8_e4m3fn", 448.0, torch.float8_e4m3fn, 1e-6, False),
        (65, 32, "fp8_e5m2", 57344.0, torch.float8_e5m2, 1e-6, False),
        (65, 32, "int8", 127.0, torch.int8, 1e-6, False),
        (7, 4, "fp8_e4m3fn", 448.0, torch.float8_e4m3fn, 1e-6, False),
        (71, 0, "fp8_e5m2", 57344.0, torch.float8_e5m2, 1e-5, False),
        (96, 96, "int8", 127.0, torch.int8, 1e-6, False),
        (192, 64, "fp8_e4m3fn", 448.0, torch.float8_e4m3fn, 1e-6, True),
    ],
)
def test_qk_reload_odd_and_wide(head, rope, quant, qmax, dtype, eps, strided) -> None:
    torch.manual_seed(head * 17 + rope + int(strided))
    rows = 4
    pos = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    if strided:
        buf = torch.randn(rows, head + 3, device="cuda", dtype=torch.bfloat16)
        x = buf[:, 3:]
        assert x.stride(0) != head
    else:
        x = torch.randn(rows, head, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(head, device="cuda", dtype=torch.bfloat16) + 0.25
    ncos = rope // 2
    cos = torch.randn(rows, ncos, device="cuda") if ncos else torch.empty(rows, 0, device="cuda")
    sin = torch.randn(rows, ncos, device="cuda") if ncos else torch.empty(rows, 0, device="cuda")
    _check(x, weight, cos, sin, pos, quant, qmax, dtype, eps)
