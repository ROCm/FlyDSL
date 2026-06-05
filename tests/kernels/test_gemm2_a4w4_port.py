#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Accuracy + performance test for the 1:1 HIP->FlyDSL port of aiter's
``gemm2_a4w4`` MXFP4 MoE down-proj kernel (gfx950, BM32 atomic instance).

Kernel under test: ``kernels.gemm2_a4w4_port`` (mirrors aiter PR #3470
``mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC``).

Three tests:
  * ``test_smoke``          — self-contained: compile + run, output finite/nonzero
                              (no aiter dependency).
  * ``test_accuracy_vs_hip``— bit-exact vs aiter's HIP gemm2 on identical bytes
                              (requires aiter; this is how the port was validated).
  * ``test_performance``    — wall-clock vs HIP at a GPU-saturating size
                              (requires aiter; loose regression bound + report).

The kernel is hardcoded to the production Kimi-K2.5 shape, so B_q is ~706 MB —
these are heavy device tests (l2_device).
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
    ASCALE_BYTES,
    BM,
    BSCALE_BYTES,
    N_OUT,
    NE,
    K,
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

HIP_KERNEL_NAME = "mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM32_ATOMIC"

# One compiled launcher reused across tests (compile is the expensive part).
_LAUNCH = None


def _launcher():
    global _LAUNCH
    if _LAUNCH is None:
        _LAUNCH = compile_gemm2_a4w4_port()
    return _LAUNCH


def _make_inputs(srt: int, seed: int = 0, const_scale: bool = True):
    """Build a valid input set. ``const_scale`` pins e8m0 scales to 127 (2^0) so
    random fp4 data cannot overflow to inf/nan; unique sorted_token_ids
    (one token per sorted row) make the bf16 atomic accumulation deterministic
    so HIP and the port are comparable bit-for-bit.
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    assert srt % BM == 0
    mmb = srt // BM
    M = srt
    t = dict(
        aq=torch.randint(0, 256, (srt, K // 2), dtype=torch.uint8, device=dev, generator=g),
        ascale=torch.full((ASCALE_BYTES // 4, 4), 127, dtype=torch.uint8, device=dev),
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


def _compile_port(t, out):
    """flyc.compile EXECUTES the kernel once into the buffer it is given, so
    compile against a throwaway buffer; the returned callable is then run into
    the real (zeroed) output. The kernel atomic-accumulates, so reusing one
    buffer for both compile and run would double the result."""
    throwaway = torch.zeros_like(out)
    return flyc.compile(
        _launcher(),
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


def _run_port(t, out=None):
    if out is None:
        out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
    launch = _compile_port(t, out)
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


def _run_hip(t, out=None):
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
        HIP_KERNEL_NAME,
    )
    torch.cuda.synchronize()
    return out


def _graph_median_us(fn, warmup=8, replays=100, reps=20):
    """Median per-replay GPU time (microseconds) using a HIP/CUDA graph.

    ``fn`` enqueues exactly one kernel on the current stream. We warm up on a
    side stream (to finish any lazy init before capture), capture a single
    ``fn()`` into a CUDA graph, then time batches of ``replays`` graph replays
    with CUDA events. Graph replay removes per-launch host overhead, and event
    timing measures pure GPU time, so this isolates the kernel itself.

    Note: the kernel atomic-accumulates into ``out`` and replays are not zeroed,
    so the accumulator saturates to inf over many replays. That does not affect
    the (data-independent) GEMM timing.
    """
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


def test_smoke():
    """Self-contained: the ported kernel compiles, runs, and produces a
    finite, non-zero output (no aiter / no reference needed)."""
    t = _make_inputs(srt=256, seed=1)
    out, _ = _run_port(t)
    assert torch.isfinite(out).all(), "port output has non-finite values"
    assert out.abs().sum().item() > 0, "port output is all zero"


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
@pytest.mark.parametrize("srt", [256, 1024])
def test_accuracy_vs_hip(srt):
    """Bit-exact match against aiter's HIP gemm2 on identical input bytes."""
    t = _make_inputs(srt=srt, seed=2)
    out_hip = _run_hip(t)
    out_port, _ = _run_port(t)

    assert torch.isfinite(out_hip).all() and torch.isfinite(out_port).all()
    if torch.equal(out_hip, out_port):
        return  # bit-exact
    a = out_hip.float().reshape(-1)
    b = out_port.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    raise AssertionError(f"port != HIP (srt={srt}): cosine={cos:.6f} max_abs_diff={max_abs:.6g}")


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
# srt = roundup(M*TOPK, BM) for test.py's KIMI-K2.5 M-list {4,8,16,32,64,128,256}
# plus a large context (M=16384 tokens -> srt=147456). TOPK=9, BM=32: the gemm2
# down-proj processes the expanded+padded tokens.
@pytest.mark.parametrize("srt", [64, 96, 160, 288, 576, 1152, 2304, 147456])
def test_performance(srt):
    """CUDA-graph (GPU-event timed) kernel performance vs HIP at a GPU-saturating
    size. Graph replay removes host launch overhead so this reflects pure
    GPU-kernel time. Reports the ratio and guards against a large regression."""
    t = _make_inputs(srt=srt, seed=3)
    out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
    launch = _compile_port(t, out)

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
            HIP_KERNEL_NAME,
        )

    hip_us = _graph_median_us(f_hip)
    port_us = _graph_median_us(f_port)
    ratio = port_us / hip_us
    _LOG.info(
        "gemm2 a4w4 srt=%d  HIP=%.1f us  port=%.1f us  port/HIP=%.2fx (cuda-graph)",
        srt,
        hip_us,
        port_us,
        ratio,
    )
    # Loose regression guard. Fail only on a large regression.
    assert ratio < 1.5, f"port too slow vs HIP: {ratio:.2f}x ({port_us:.1f} vs {hip_us:.1f} us)"
