#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GQA sliding-window attention correctness harness (gfx950 only).

Kernel implementation: ``kernels/attention/swa_gfx950.py``.

The kernel computes bf16 GQA attention with an aiter-style sliding window:
``sliding_window=(LEFT, RIGHT)`` keeps, for query row ``i``, keys ``j`` in
``[i-LEFT, i+RIGHT]`` (with ``seq_len_q == seq_len_kv``). Each of the 8 waves
in a CTA owns one 32-row Q tile, so a CTA covers 256 query rows.

Correctness is checked against an fp32 band-masked softmax reference.
"""

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.swa_gfx950 import build_gqa_attn  # noqa: E402

ARCH = str(get_rocm_arch())

# Kernel is specialized for D=128 GQA with H=32 / H_KV=16 (GROUP=2).
D = 128
H = 32
H_KV = 16
GROUP = H // H_KV
DTYPE = torch.bfloat16

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _band_mask(n, left, right, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j >= i - left) & (j <= i + right)


@torch.no_grad()
def _ref_fp32(q, k, v, left, right):
    """fp32 band-masked GQA softmax reference. q:[B,N,H,D], k/v:[B,N,H_KV,D]."""
    n = q.shape[1]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(GROUP, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(GROUP, dim=1)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / D**0.5)
    keep = _band_mask(n, left, right, q.device).view(1, 1, n, n)
    sc = sc.masked_fill(~keep, float("-inf"))
    p = torch.softmax(sc, dim=-1)
    out = torch.matmul(p, vf)
    return out.permute(0, 2, 1, 3).contiguous()


def _run_swa(B, N, left, right, seed=0):
    if ARCH != "gfx950":
        pytest.skip(f"SWA GQA attention requires gfx950, got {ARCH}")

    torch.manual_seed(seed)
    q = torch.randn(B, N, H, D, dtype=DTYPE, device="cuda")
    k = torch.randn(B, N, H_KV, D, dtype=DTYPE, device="cuda")
    v = torch.randn(B, N, H_KV, D, dtype=DTYPE, device="cuda")
    out = torch.zeros(B, N, H, D, dtype=DTYPE, device="cuda")

    launch = build_gqa_attn(ATTN_B=B, ATTN_H=H, ATTN_H_KV=H_KV, ATTN_D=D, sliding_window=(left, right))
    stream = torch.cuda.current_stream()
    args = (
        q.reshape(-1),
        k.reshape(-1),
        v.reshape(-1),
        out.reshape(-1),
        H * D,  # Q_stride1
        H_KV * D,  # K_stride1
        H_KV * D,  # V_stride1
        H * D,  # O_stride1
        N,  # seq_len_q
        N,  # seq_len_kv
        fx.Stream(stream),
    )
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = _ref_fp32(q, k, v, left, right)
    o_f32 = out.float()
    max_err = (ref - o_f32).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), o_f32.flatten(), dim=0).item()
    print(f"\n[swa_gfx950] B={B} N={N} win=({left},{right}) cos={cos:.6f} max_abs={max_err:.4f}")
    assert cos > 0.999, f"cosine {cos:.6f} <= 0.999 (max_abs={max_err:.4f})"


@pytest.mark.parametrize(
    "B, N, left, right",
    [
        pytest.param(2, 4096, 512, 0, id="B2_N4096_win512_0"),
        pytest.param(2, 8192, 2048, 0, id="B2_N8192_win2048_0"),
        pytest.param(1, 16384, 4096, 0, marks=pytest.mark.large_shape, id="B1_N16384_win4096_0"),
        pytest.param(2, 4096, 256, 256, id="B2_N4096_win256_256"),
        pytest.param(1, 8192, 1024, 0, id="B1_N8192_win1024_0"),
    ],
)
def test_swa_gqa_attention(B, N, left, right):
    _run_swa(B, N, left, right)


if __name__ == "__main__":
    for cfg in [
        (2, 4096, 512, 0),
        (2, 8192, 2048, 0),
        (1, 16384, 4096, 0),
        (2, 4096, 256, 256),
        (1, 8192, 1024, 0),
    ]:
        _run_swa(*cfg)
