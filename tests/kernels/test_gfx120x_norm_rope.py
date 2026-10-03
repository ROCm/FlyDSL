#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness smoke tests for the gfx120x norm / RoPE / AdaLN kernels.

These tests call FlyDSL builders directly; they intentionally do not import
any application integration layer.
"""

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

if not str(get_rocm_arch()).startswith("gfx120"):
    pytest.skip("requires gfx120x", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402
from kernels.norm.adaln_gfx120x import build_adaln_module  # noqa: E402
from kernels.norm.rms_rope_gfx120x import build_rms_rope_module  # noqa: E402
from kernels.norm.rope_gfx120x import (  # noqa: E402
    build_rope_module,
    build_rope_qk_fused_module,
    build_rope_split_module,
)

EPS = 1e-6


def _ptr(tensor: object) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def _rope_freqs(rows: object, pairs: object, device: torch.device) -> torch.Tensor:
    freqs = torch.randn((rows, pairs, 2, 2), device=device, dtype=torch.float32)
    return freqs


def _rope_ref(x: object, freqs: object, *, split_half: bool = False) -> torch.Tensor:
    rows, hd = x.reshape(-1, x.shape[-1]).shape
    pairs = hd // 2
    xf = x.reshape(rows, hd).float()
    f = freqs.reshape(rows, pairs, 2, 2).float()
    if split_half:
        x0, x1 = xf[:, :pairs], xf[:, pairs:]
    else:
        xp = xf.reshape(rows, pairs, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
    y0 = f[..., 0, 0] * x0 + f[..., 0, 1] * x1
    y1 = f[..., 1, 0] * x0 + f[..., 1, 1] * x1
    if split_half:
        return torch.cat((y0, y1), dim=-1).reshape_as(x)
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


def _run_rope(
    builder: object, x: object, freqs: object, out: torch.Tensor | None, *, split_half: bool = False, qk: bool = False
) -> None:
    rows, hd = x.reshape(-1, x.shape[-1]).shape
    pairs = hd // 2
    stream = torch.cuda.current_stream()
    n_pairs_total = rows * pairs
    args = (_ptr(x), _ptr(freqs), _ptr(out)) if not qk else (_ptr(x), _ptr(x), _ptr(freqs), _ptr(out), _ptr(out))
    common = (n_pairs_total, pairs, 1, rows, 1, rows, 1, rows * pairs)
    if split_half:
        common = common + (hd,)
    full_args = args + common + (stream,)
    compiled = flyc.compile(builder, *full_args)
    compiled(*full_args)


def test_rope_interleaved_and_split_half() -> None:
    device = torch.device("cuda")
    rows, hd = 2, 64
    x = torch.randn((rows, hd), device=device, dtype=torch.bfloat16)
    freqs = _rope_freqs(rows, hd // 2, device)

    out = torch.empty_like(x)
    _run_rope(build_rope_module("bfloat16", 256), x, freqs, out)
    torch.testing.assert_close(out.float(), _rope_ref(x, freqs).float(), rtol=3e-2, atol=3e-2)

    out_split = torch.empty_like(x)
    _run_rope(
        build_rope_split_module("bfloat16", 256),
        x,
        freqs,
        out_split,
        split_half=True,
    )
    torch.testing.assert_close(out_split.float(), _rope_ref(x, freqs, split_half=True).float(), rtol=3e-2, atol=3e-2)


def test_rope_qk_fused() -> None:
    device = torch.device("cuda")
    rows, hd = 2, 64
    q = torch.randn((rows, hd), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = _rope_freqs(rows, hd // 2, device)
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    stream = torch.cuda.current_stream()
    pairs = hd // 2
    n_pairs_total = rows * pairs
    args = (
        _ptr(q),
        _ptr(k),
        _ptr(freqs),
        _ptr(oq),
        _ptr(ok),
        n_pairs_total,
        pairs,
        1,
        rows,
        1,
        rows,
        1,
        rows * pairs,
        stream,
    )
    builder = build_rope_qk_fused_module("bfloat16", 256)
    compiled = flyc.compile(builder, *args)
    compiled(*args)
    torch.testing.assert_close(oq.float(), _rope_ref(q, freqs).float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(ok.float(), _rope_ref(k, freqs).float(), rtol=3e-2, atol=3e-2)


def test_rms_rope() -> None:
    device = torch.device("cuda")
    rows, hd = 2, 64
    x = torch.randn((rows, hd), device=device, dtype=torch.bfloat16)
    scale = torch.randn((hd,), device=device, dtype=torch.bfloat16)
    freqs = _rope_freqs(rows, hd // 2, device)
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    builder = build_rms_rope_module(hd, "bfloat16", block_threads=32)
    args = (x, scale, freqs, out, rows, 1, EPS, stream)
    compiled = flyc.compile(builder, *args)
    compiled(*args)
    norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS) * scale.float()
    ref = _rope_ref(norm.to(x.dtype), freqs)
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)


def test_adaln_and_rms_adaln() -> None:
    device = torch.device("cuda")
    rows, width = 2, 256
    x = torch.randn((rows, width), device=device, dtype=torch.bfloat16)
    scale = torch.randn_like(x) * 0.1
    shift = torch.randn_like(x) * 0.1
    stream = torch.cuda.current_stream()
    for subtract_mean in (True, False):
        out = torch.empty_like(x)
        builder = build_adaln_module(width, "bfloat16", subtract_mean, block_threads=256)
        args = (x, scale, shift, out, rows, 1, 1, EPS, stream)
        compiled = flyc.compile(builder, *args)
        compiled(*args)
        xf = x.float()
        if subtract_mean:
            mean = xf.mean(-1, keepdim=True)
            rstd = torch.rsqrt((xf - mean).square().mean(-1, keepdim=True) + EPS)
            norm = (xf - mean) * rstd
        else:
            norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
        ref = norm * (1.0 + scale.float()) + shift.float()
        torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)
