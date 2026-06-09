#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Accuracy + performance test for the FlyDSL port of aiter's ``gemm2_a4w4``
MXFP4 MoE down-proj kernel (gfx950), over multiple shapes / specializations.

The mxfp4 gemm2 fixes K=512 (contraction = inter_dim); the port is parametrized
by shape (NE experts, N_OUT model_dim) and specialization (BM, kUseNT). Each
variant is validated against the matching prebuilt HIP instance
``mxfp4_moe_g2_a4w4_NE{NE}_H{N_OUT}_E512_TOPK9_BM{BM}_ATOMIC[_NT]``.

Variants (prebuilt HIP instances available: NE∈{257,385}, H=7168, BM∈{16,32}):
  ne385_bm32 / ne385_bm16nt / ne385_bm32nt — Kimi-K2.5 (NE=385)
  ne257_bm16nt / ne257_bm32nt              — NE=257 shape

Tests (parametrized over all variants):
  * ``test_smoke``          — compile + run, finite/nonzero (no aiter).
  * ``test_accuracy_vs_hip``— bit-exact vs aiter's HIP gemm2 (same instance).
  * ``test_performance``    — CUDA-graph GPU-event time vs HIP + regression bound.

B_q is ~hundreds of MB, so these are heavy device tests (l2_device).
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
    MAX_M,
    K,
    ascale_bytes,
    bscale_bytes_for,
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
    from aiter.ops.mxfp4_moe import mxfp4_moe_gemm2_a4w4, mxfp4_moe_gemm2_a4w4_mxfp4out

    HAS_AITER = True
except Exception:
    HAS_AITER = False


def _hip_name(NE, N_OUT, BM, use_nt, epilog):
    if epilog == "atomic":
        return f"mxfp4_moe_g2_a4w4_NE{NE}_H{N_OUT}_E512_TOPK9_BM{BM}_ATOMIC{'_NT' if use_nt else ''}"
    if epilog == "nonatomic":
        return f"mxfp4_moe_g2_a4w4_NE{NE}_H{N_OUT}_E512_BM{BM}_NONATOMIC"
    if epilog == "nonatomic_mxfp4":
        return f"mxfp4_moe_g2_a4w4_NE{NE}_H{N_OUT}_E512_BM{BM}_NONATOMIC_MXFP4OUT"
    raise ValueError(epilog)


def _variant(NE, N_OUT, BM, use_nt, epilog, vid):
    return pytest.param(NE, N_OUT, BM, use_nt, epilog, _hip_name(NE, N_OUT, BM, use_nt, epilog), id=vid)


# (NE, N_OUT, BM, use_nt, epilog, hip_name) — full coverage of every prebuilt
# HIP gemm2 instance for H=7168, E512:
#   * ATOMIC (bf16, weighted, one-tile-per-block): NE∈{257,385} × BM∈{16,32,64} × {ATOMIC, NT}
#   * NONATOMIC (bf16 flat per-sorted-row, persistent): NE∈{257,385} × BM128
#   * NONATOMIC_MXFP4OUT (fp4 q + e8m0 scale flat): NE385 × BM128
VARIANTS = (
    [
        _variant(NE, 7168, BM, use_nt, "atomic", f"ne{NE}_bm{BM}{'nt' if use_nt else ''}")
        for NE in (385, 257)
        for BM in (16, 32, 64)
        for use_nt in (False, True)
    ]
    + [_variant(NE, 7168, 128, False, "nonatomic", f"ne{NE}_bm128_nonatomic") for NE in (385, 257)]
    + [_variant(385, 7168, 128, False, "nonatomic_mxfp4", "ne385_bm128_mxfp4out")]
)

# Compiled launchers reused across tests, keyed by (BM, use_nt, NE, N_OUT).
_LAUNCH = {}


def _launcher(BM, use_nt, NE, N_OUT, epilog):
    key = (BM, use_nt, NE, N_OUT, epilog)
    if key not in _LAUNCH:
        _LAUNCH[key] = compile_gemm2_a4w4_port(BM=BM, use_nt=use_nt, NE=NE, N_OUT=N_OUT, epilog=epilog)
    return _LAUNCH[key]


def _out_tensors(srt, N_OUT, epilog):
    """Output buffer(s) for an epilog. mxfp4out writes packed fp4 q (N_OUT//2
    bytes/row) + e8m0 scale (N_OUT//32 bytes/row); all others write bf16. A
    1-elem dummy ``out_scale`` is passed for the non-mxfp4 epilogs (unused)."""
    dev = "cuda"
    if epilog == "nonatomic_mxfp4":
        out = torch.zeros(srt, N_OUT // 2, dtype=torch.uint8, device=dev)
        out_scale = torch.zeros(srt, N_OUT // 32, dtype=torch.uint8, device=dev)
    else:
        out = torch.zeros(srt, N_OUT, dtype=torch.bfloat16, device=dev)
        out_scale = torch.zeros(1, dtype=torch.uint8, device=dev)
    return out, out_scale


def _make_inputs(srt, BM, NE, N_OUT, seed=0, const_scale=True):
    """Build a valid input set for (BM, NE, N_OUT). ``const_scale`` pins e8m0
    scales to 127 (2^0); unique sorted_token_ids make the bf16 atomic
    accumulation deterministic so HIP and the port are bit-exact."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    assert srt % BM == 0
    mmb = srt // BM
    M = srt
    t = dict(
        aq=torch.randint(0, 256, (srt, K // 2), dtype=torch.uint8, device=dev, generator=g),
        ascale=torch.full((ascale_bytes(BM, MAX_M) // 4, 4), 127, dtype=torch.uint8, device=dev),
        bq=torch.randint(0, 256, (NE, N_OUT, K // 2), dtype=torch.uint8, device=dev, generator=g),
        bscale=torch.full((bscale_bytes_for(NE, N_OUT) // 4, 4), 127, dtype=torch.uint8, device=dev),
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
    t["N_OUT"] = N_OUT
    return t


def _compile_port(t, out, out_scale, BM, use_nt, NE, N_OUT, epilog):
    """flyc.compile EXECUTES the kernel once into the buffer it is given, so
    compile against throwaways; the returned callable is then run into the real
    (zeroed) output."""
    return flyc.compile(
        _launcher(BM, use_nt, NE, N_OUT, epilog),
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
        torch.zeros_like(out),
        torch.zeros_like(out_scale),
        torch.cuda.current_stream(),
    )


def _run_port(t, BM, use_nt, NE, N_OUT, epilog):
    out, out_scale = _out_tensors(t["M"], N_OUT, epilog)
    launch = _compile_port(t, out, out_scale, BM, use_nt, NE, N_OUT, epilog)
    out.zero_()
    out_scale.zero_()
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
        out_scale,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return out, out_scale, launch


def _run_hip(t, hip_name, epilog, NE, N_OUT):
    """Run the matching prebuilt HIP instance. Returns (out, out_scale) where
    out_scale is None for the bf16 (atomic/nonatomic) paths."""
    if epilog == "nonatomic_mxfp4":
        out_q = torch.zeros(t["M"], N_OUT // 2, dtype=torch.uint8, device="cuda")
        out_scale = torch.zeros(t["M"], N_OUT // 32, dtype=torch.uint8, device="cuda")
        mxfp4_moe_gemm2_a4w4_mxfp4out(
            t["cumsum"],
            t["aq"],
            t["ascale"],
            t["bq"],
            t["bscale"],
            t["eids"],
            out_q,
            out_scale,
            NE,
            N_OUT,
            K,
            t["M"],
        )
        torch.cuda.synchronize()
        return out_q, out_scale
    # atomic + nonatomic bf16 share the same op (kernelName selects the path)
    out = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")
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
    return out, None


def _graph_median_us(fn, warmup=8, replays=100, reps=20):
    """Median per-replay GPU time (us) via a CUDA graph. ``fn`` enqueues one
    kernel; we warm up on a side stream, capture one ``fn()``, then time batches
    of graph replays with CUDA events (removes host overhead, isolates GPU time).
    The atomic accumulator saturates to inf over replays — does not affect the
    data-independent GEMM timing."""
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


@pytest.mark.parametrize("NE,N_OUT,BM,use_nt,epilog,hip_name", VARIANTS)
def test_smoke(NE, N_OUT, BM, use_nt, epilog, hip_name):
    """Self-contained: the ported kernel compiles, runs, and produces a finite,
    non-zero output (no aiter / no reference needed)."""
    t = _make_inputs(srt=256, BM=BM, NE=NE, N_OUT=N_OUT, seed=1)
    out, out_scale, _ = _run_port(t, BM, use_nt, NE, N_OUT, epilog)
    if epilog == "nonatomic_mxfp4":
        assert (out != 0).any(), "port fp4 q output is all zero"
        assert (out_scale != 0).any(), "port e8m0 scale output is all zero"
    else:
        assert torch.isfinite(out).all(), "port output has non-finite values"
        assert out.abs().sum().item() > 0, "port output is all zero"


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
@pytest.mark.parametrize("NE,N_OUT,BM,use_nt,epilog,hip_name", VARIANTS)
@pytest.mark.parametrize("srt", [256, 1024])
def test_accuracy_vs_hip(srt, NE, N_OUT, BM, use_nt, epilog, hip_name):
    """Bit-exact match against aiter's HIP gemm2 (same instance) on identical bytes."""
    if srt % BM:
        pytest.skip(f"srt={srt} not a multiple of BM={BM}")
    t = _make_inputs(srt=srt, BM=BM, NE=NE, N_OUT=N_OUT, seed=2)
    out_hip, scale_hip = _run_hip(t, hip_name, epilog, NE, N_OUT)
    out_port, scale_port, _ = _run_port(t, BM, use_nt, NE, N_OUT, epilog)

    if epilog == "nonatomic_mxfp4":
        # fp4 q + e8m0 scale are integer byte streams: require exact equality.
        assert torch.equal(out_hip, out_port), (
            f"port fp4 q != HIP (NE={NE} srt={srt}): "
            f"{(out_hip != out_port).sum().item()}/{out_hip.numel()} bytes differ"
        )
        assert torch.equal(scale_hip, scale_port), (
            f"port e8m0 scale != HIP (NE={NE} srt={srt}): "
            f"{(scale_hip != scale_port).sum().item()}/{scale_hip.numel()} bytes differ"
        )
        return

    assert torch.isfinite(out_hip).all() and torch.isfinite(out_port).all()
    if torch.equal(out_hip, out_port):
        return  # bit-exact
    a = out_hip.float().reshape(-1)
    b = out_port.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    raise AssertionError(
        f"port != HIP (NE={NE} BM={BM} nt={use_nt} ep={epilog} srt={srt}): "
        f"cosine={cos:.6f} max_abs_diff={max_abs:.6g}"
    )


@pytest.mark.skipif(not HAS_AITER, reason="aiter required for the HIP gemm2 reference")
@pytest.mark.parametrize("NE,N_OUT,BM,use_nt,epilog,hip_name", VARIANTS)
# srt = roundup(M*TOPK, BM) for test.py's KIMI-K2.5 M-list {4,8,16,32,64,128,256}
# plus a large context (M=16384 -> srt=147456). Values are multiples of both 16
# and 32 so they are valid for both BM specializations.
@pytest.mark.parametrize("srt", [64, 96, 160, 288, 576, 1152, 2304, 147456])
def test_performance(srt, NE, N_OUT, BM, use_nt, epilog, hip_name):
    """CUDA-graph (GPU-event timed) kernel performance vs HIP. Graph replay
    removes host launch overhead so this reflects pure GPU-kernel time. Reports
    the ratio and guards against a large regression."""
    if srt % BM:
        pytest.skip(f"srt={srt} not a multiple of BM={BM}")
    t = _make_inputs(srt=srt, BM=BM, NE=NE, N_OUT=N_OUT, seed=3)
    out, out_scale = _out_tensors(srt, N_OUT, epilog)
    launch = _compile_port(t, out, out_scale, BM, use_nt, NE, N_OUT, epilog)

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
            out_scale,
            torch.cuda.current_stream(),
        )

    if epilog == "nonatomic_mxfp4":
        hq = torch.zeros(t["M"], N_OUT // 2, dtype=torch.uint8, device="cuda")
        hs = torch.zeros(t["M"], N_OUT // 32, dtype=torch.uint8, device="cuda")

        def f_hip():
            mxfp4_moe_gemm2_a4w4_mxfp4out(
                t["cumsum"], t["aq"], t["ascale"], t["bq"], t["bscale"], t["eids"], hq, hs, NE, N_OUT, K, t["M"]
            )

    else:
        hout = torch.zeros(t["M"], N_OUT, dtype=torch.bfloat16, device="cuda")

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
                hout,
                t["M"],
                t["M"],
                hip_name,
            )

    hip_us = _graph_median_us(f_hip)
    port_us = _graph_median_us(f_port)
    ratio = port_us / hip_us
    _LOG.info(
        "gemm2 a4w4 [NE%d bm%d%s %s] srt=%d  HIP=%.1f us  port=%.1f us  port/HIP=%.2fx (cuda-graph)",
        NE,
        BM,
        "_nt" if use_nt else "",
        epilog,
        srt,
        hip_us,
        port_us,
        ratio,
    )
    assert ratio < 1.5, f"port too slow vs HIP: {ratio:.2f}x ({port_us:.1f} vs {hip_us:.1f} us)"
