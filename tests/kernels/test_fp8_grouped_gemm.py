#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""fp8 per-tensor GROUPED GEMM forward correctness + perf harness.

Kernel implementation --> ``kernels/gemm/fp8_grouped_gemm.py``.

The M-grouped forward computes, for each group g:
    out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ b_T[g]^T * a_scale * b_scale
where a is [M_total, K], b_T is [G, N, K] (NT), and group_offs [G+1] splits M.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.fp8_grouped_gemm import grouped_gemm_fp8_forward  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _tensorwise_quant(x, quant_dtype=FP8_DTYPE):
    """Per-tensor (scalar) symmetric fp8 quant. Returns (q, scale) with
    x ~= q.float() * scale."""
    finfo = torch.finfo(quant_dtype)
    amax = x.abs().max().clamp(min=1e-8)
    scale = (amax / finfo.max).to(torch.float32)
    q = (x / scale).clamp(finfo.min, finfo.max).to(quant_dtype)
    return q, scale


def _ref_grouped(a_q, b_q_t, a_scale, b_scale, group_offs, out_dtype=torch.float32):
    """Per-group torch reference: out[g] = (a_q.f32 * a_scale) @ (b_q_t[g].f32 * b_scale)^T."""
    M_total, _ = a_q.shape
    G, N, _ = b_q_t.shape
    out = torch.empty((M_total, N), device=a_q.device, dtype=out_dtype)
    a_f32 = a_q.to(torch.float32) * a_scale
    for g in range(G):
        lo = int(group_offs[g].item())
        hi = int(group_offs[g + 1].item())
        if hi <= lo:
            continue
        bg = b_q_t[g].to(torch.float32) * b_scale  # [N, K]
        out[lo:hi] = (a_f32[lo:hi] @ bg.T).to(out_dtype)
    return out


def _make_group_offs(group_ms, device):
    offs = torch.zeros(len(group_ms) + 1, dtype=torch.int64, device=device)
    offs[1:] = torch.tensor(group_ms, dtype=torch.int64, device=device).cumsum(0)
    return offs


def _run_grouped(group_ms, N, K, *, num_warmups=5, num_iters=20, verbose=True):
    """Run + verify a single (group_ms, N, K) grouped forward. Returns TFLOPS."""
    if "gfx95" not in ARCH:
        pytest.skip("fp8 grouped GEMM requires CDNA4 (gfx95*)")

    device = torch.device("cuda")
    G = len(group_ms)
    M_total = int(sum(group_ms))

    a_fp32 = torch.randn(M_total, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(G, N, K, device=device, dtype=torch.float32)

    a_q, a_scale = _tensorwise_quant(a_fp32)
    b_q_t, b_scale = _tensorwise_quant(b_fp32_t)
    a_q = a_q.contiguous()
    b_q_t = b_q_t.contiguous()

    group_offs = _make_group_offs(group_ms, device)
    c_ref = _ref_grouped(a_q, b_q_t, a_scale, b_scale, group_offs)

    if verbose:
        print(f"\n[fp8_grouped_gemm] G={G} M_total={M_total} N={N} K={K} groups={group_ms}")

    def _launch():
        return grouped_gemm_fp8_forward(a_q, b_q_t, a_scale, b_scale, group_offs, out_dtype=OUT_DTYPE)

    out = _launch()
    torch.cuda.synchronize()
    ok = verify_output(out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    assert ok, f"correctness failed for G={G} M_total={M_total} N={N} K={K}"

    _, us = run_perftest(lambda _o: _launch(), out, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()

    flops = 2 * M_total * N * K
    tflops = flops / (us / 1e6) / 1e12
    if verbose:
        print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS")
    return tflops


# (group_ms, N, K): a mix of aligned, ragged, single-token, and empty groups so the
# on-device group scan / K-tail / partial-M-tile clamps all get exercised.
_CASES = [
    pytest.param([256, 256, 256, 256], 512, 512, id="4x256_n512_k512"),
    pytest.param([300, 128, 700, 45], 1024, 768, id="ragged_n1024_k768"),
    pytest.param([1024, 0, 512, 1536], 2048, 2048, id="empty_group_n2048_k2048"),
    pytest.param([1, 255, 4096], 4096, 1024, id="single_token_n4096_k1024"),
    pytest.param([2048, 2048], 4096, 7168, id="2x2048_n4096_k7168"),
]


@pytest.mark.parametrize("group_ms, N, K", _CASES)
def test_fp8_grouped_gemm_forward(group_ms, N, K):
    _run_grouped(group_ms, N, K)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fp8 grouped GEMM forward benchmark")
    parser.add_argument("-G", type=int, default=8)
    parser.add_argument("-M", type=int, default=8192, help="total M (split evenly across G)")
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=7168)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--num_warmups", type=int, default=10)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    base = args.M // args.G
    gms = [base] * args.G
    gms[-1] += args.M - base * args.G  # remainder into the last group
    try:
        _run_grouped(gms, args.N, args.K, num_warmups=args.num_warmups, num_iters=args.num_iters)
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
