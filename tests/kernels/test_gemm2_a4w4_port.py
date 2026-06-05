#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Accuracy + performance test for the FlyDSL port of aiter's ``gemm2_a4w4``
MXFP4 MoE down-proj kernel (gfx950), over two specializations:

  * ``bm32``   -> ``mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC``
  * ``bm16nt`` -> ``mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT``
                 (the instance aiter's fused_moe actually selects in production)

Kernel under test: ``kernels.gemm2_a4w4_port.compile_gemm2_a4w4_port(BM, use_nt)``.

Tests (each parametrized over both variants):
  * ``test_smoke``          — compile + run, output finite/nonzero (no aiter).
  * ``test_accuracy_vs_hip``— bit-exact vs aiter's HIP gemm2 on identical bytes.
  * ``test_performance``    — CUDA-graph GPU-event time vs HIP + regression bound.

The kernel is pinned to the Kimi-K2.5 shape, so B_q is ~706 MB — heavy device
tests (l2_device).
"""

import logging
import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (os.path.join(_REPO_ROOT, "build", "python_packages"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm2_a4w4_port import (  # noqa: E402
    BSCALE_BYTES,
    N_OUT,
    NE,
    K,
    ascale_bytes,
    compile_gemm2_a4w4_port,
)

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = get_rocm_arch()
if "gfx95" not in ARCH:
    pytest.skip(
        f"gemm2_a4w4_port uses mfma_scale_f32_16x16x128_f8f6f4 (gfx950+); got {ARCH}",
        allow_module_level=True,
    )

try:
    import aiter  # noqa: F401
    from aiter.ops.mxfp4_moe import mxfp4_moe_gemm2_a4w4

    HAS_AITER = True
except Exception:
    HAS_AITER = False

_HIP = "mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_"
# (BM, use_nt, hip_kernel_name)
VARIANTS = [
    pytest.param(32, False, _HIP + "BM32_ATOMIC", id="bm32"),
    pytest.param(16, True, _HIP + "BM16_ATOMIC_NT", id="bm16nt"),
    pytest.param(32, True, _HIP + "BM32_ATOMIC_NT", id="bm32nt"),
]

# Compiled launchers reused across tests, keyed by (BM, use_nt).
_LAUNCH = {}


def _launcher(BM, use_nt):
    key = (BM, use_nt)
    if key not in _LAUNCH:
        _LAUNCH[key] = compile_gemm2_a4w4_port(BM=BM, use_nt=use_nt)
    return _LAUNCH[key]


def _make_inputs(srt: int, BM: int, seed: int = 0, const_scale: bool = True):
    """Build a valid input set for a given BM. ``const_scale`` pins e8m0 scales
    to 127 (2^0) so random fp4 data cannot overflow; unique sorted_token_ids make
    the bf16 atomic accumulation deterministic so HIP and the port are bit-exact.
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    assert srt % BM == 0
    mmb = srt // BM
    M = srt
    t = dict(
        aq=torch.randint(0, 256, (srt, K // 2), dtype=torch.uint8, device=dev, generator=g),
        ascale=torch.full((ascale_bytes(BM) // 4, 4), 127, dtype=torch.uint8, device=dev),
        bq=torch.randint(0, 256, (NE, N_OUT, K // 2), dtype=torch.uint8, device=dev, generator=g),
        bscale=torch.full((BSCALE_BYTES // 4, 4), 127, dtype=torch.uint8, device=dev),
        eids=torch.randint(0, NE, (mmb,), dtype=torch.int32, device=dev, generator=g),
        cumsum=torch.tensor([srt], dtype=torch.int32, device=dev),
        stids=torch.randperm(M, dtype=torch.int32, device=dev, generator=g),
        sweights=(torch.rand(srt, dtype=torch.float32, device=dev, generator=g) + 0.5),
    )
    if not const_scale:
        t["ascale"] = torch.randint(0, 256, t["ascale"].shape, dtype=torch.uint8, device=dev)
        t["bscale"] = torch.randint(0, 256, t["bscale"].shape, dtype=torch.uint8, device=dev)
    t["M"] = M
    t["mmb"] = mmb
    return t


def _compile_port(t, out, BM, use_nt):
    """flyc.compile EXECUTES the kernel once into the buffer it is given, so
    compile against a throwaway buffer; the returned callable is then run into
    the real (zeroed) output. The kernel atomic-accumulates, so reusing one
    buffer for both compile and run would double the result."""
    throwaway = torch.zeros_like(out)
    return flyc.compile(
        _launcher(BM, use_nt),
        t["aq"],
        t["ascale"],
        t["bq"],
        t["bscale"],
        t["eids"],
        t["cumsum"],
        t["stids"],
        t["sweights"],
        t["M"],
        t["mmb"],
        throwaway,
        torch.cuda.current_stream(),
    )


def _run_port(t, BM, use_nt, out=None):
    if out is None:
        out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
    launch = _compile_port(t, out, BM, use_nt)
    out.zero_()
    launch(
        t["aq"],
        t["ascale"],
        t["bq"],
        t["bscale"],
        t["eids"],
        t["cumsum"],
        t["stids"],
        t["sweights"],
        t["M"],
        t["mmb"],
        out,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return out, launch


def _run_hip(t, hip_name, out=None):
    if out is None:
        out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
    else:
        out.zero_()
    mxfp4_moe_gemm2_a4w4(
        t["cumsum"],
        t["aq"],
        t["ascale"],
        t["bq"],
        t["bscale"],
        t["stids"],
        t["eids"],
        t["sweights"],
        out,
        t["M"],
        t["M"],
        hip_name,
    )
    torch.cuda.synchronize()
    return out


def _graph_median_us(fn, warmup=8, replays=100, reps=20):
    """Median per-replay GPU time (us) via a CUDA graph. ``fn`` enqueues one
    kernel on the current stream; we warm up on a side stream, capture one
    ``fn()``, then time batches of graph replays with CUDA events (removes host
    launch overhead, isolates GPU-kernel time). The atomic accumulator saturates
    to inf over replays — does not affect the data-independent GEMM timing."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        for _ in range(replays):
            g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays * 1e3)  # ms total -> us/replay
    samples.sort()
    return samples[len(samples) // 2]


@pytest.mark.parametrize("BM,use_nt,hip_name", VARIANTS)
def test_smoke(BM, use_nt, hip_name):
    """Self-contained: the ported kernel compiles, runs, and produces a
    finite, non-zero output (no aiter / no reference needed)."""
    t = _make_inputs(srt=256, BM=BM, seed=1)
    out, _ = _run_port(t, BM, use_nt)
    assert torch.isfinite(out).all(), "port output has non-finite values"
    assert out.abs().sum().item() > 0, "port output is all zero"


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
@pytest.mark.parametrize("BM,use_nt,hip_name", VARIANTS)
@pytest.mark.parametrize("srt", [256, 1024])
def test_accuracy_vs_hip(srt, BM, use_nt, hip_name):
    """Bit-exact match against aiter's HIP gemm2 (same instance) on identical bytes."""
    t = _make_inputs(srt=srt, BM=BM, seed=2)
    out_hip = _run_hip(t, hip_name)
    out_port, _ = _run_port(t, BM, use_nt)

    assert torch.isfinite(out_hip).all() and torch.isfinite(out_port).all()
    if torch.equal(out_hip, out_port):
        return  # bit-exact
    a = out_hip.float().reshape(-1)
    b = out_port.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    raise AssertionError(f"port != HIP (BM={BM} nt={use_nt} srt={srt}): cosine={cos:.6f} max_abs_diff={max_abs:.6g}")


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
@pytest.mark.parametrize("BM,use_nt,hip_name", VARIANTS)
# srt = roundup(M*TOPK, BM) for test.py's KIMI-K2.5 M-list {4,8,16,32,64,128,256}
# plus a large context (M=16384 -> srt=147456). TOPK=9; values are multiples of
# both 16 and 32 so they are valid for both BM specializations.
@pytest.mark.parametrize("srt", [64, 96, 160, 288, 576, 1152, 2304, 147456])
def test_performance(srt, BM, use_nt, hip_name):
    """CUDA-graph (GPU-event timed) kernel performance vs HIP. Graph replay
    removes host launch overhead so this reflects pure GPU-kernel time. Reports
    the ratio and guards against a large regression."""
    t = _make_inputs(srt=srt, BM=BM, seed=3)
    out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
    launch = _compile_port(t, out, BM, use_nt)

    def f_port():
        launch(
            t["aq"],
            t["ascale"],
            t["bq"],
            t["bscale"],
            t["eids"],
            t["cumsum"],
            t["stids"],
            t["sweights"],
            t["M"],
            t["mmb"],
            out,
            torch.cuda.current_stream(),
        )

    def f_hip():
        mxfp4_moe_gemm2_a4w4(
            t["cumsum"],
            t["aq"],
            t["ascale"],
            t["bq"],
            t["bscale"],
            t["stids"],
            t["eids"],
            t["sweights"],
            out,
            t["M"],
            t["M"],
            hip_name,
        )

    hip_us = _graph_median_us(f_hip)
    port_us = _graph_median_us(f_port)
    ratio = port_us / hip_us
    _LOG.info(
        "gemm2 a4w4 [%s] srt=%d  HIP=%.1f us  port=%.1f us  port/HIP=%.2fx (cuda-graph)",
        f"bm{BM}{'_nt' if use_nt else ''}",
        srt,
        hip_us,
        port_us,
        ratio,
    )
    assert ratio < 1.5, f"port too slow vs HIP: {ratio:.2f}x ({port_us:.1f} vs {hip_us:.1f} us)"
