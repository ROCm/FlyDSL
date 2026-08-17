#!/usr/bin/env python3
"""Correctness test for the layout-API flex/flash attention forward (gfx950).

Compares flydsl_flex_attention_layout against torch scaled_dot_product_attention
(non-causal). gfx950-only (uses cdna4-era MFMA + the layout API); skipped elsewhere.

Phase 0 kernel constraints (see the kernel's make_flex_attn_param): block_m=32,
block_n multiple of 32, head_dim multiple of 32, seqlen_kv multiple of block_n.

pd2 stagger main loop uses ``emit_tile_stagger_kv`` internally (same path as manual
cluster emit); no separate parity test yet. ``_D2_SHAPES`` includes Skv=32 (single
KV tile when block_n=32); pd2 kernels with ``n_kv_tiles==1`` use ``emit_tile_stagger_kv`` + epilogue.
"""
import argparse
import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

import pytest  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flex_attention_layout_gfx950 import (  # noqa: E402
    flydsl_flex_attention_layout,
)
from tests.test_common import run_perftest  # noqa: E402

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="layout-API attention kernel targets gfx950",
)

_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}


def _sdpa_ref(q, k, v, scale):
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    out = F.scaled_dot_product_attention(qh, kh, vh, scale=scale, is_causal=False)
    return out.permute(0, 2, 1, 3).contiguous()


def _attn_tflops(B, Sq, Skv, H, D, us):
    flops = 4.0 * B * H * Sq * Skv * D
    return flops / (us * 1e-6) / 1e12


def _run(B, Sq, Skv, H, D, dtype_str, *, pipe_depth=1, num_groups=2, accurate_softmax=True):
    dtype = _DTYPES[dtype_str]
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)

    out = flydsl_flex_attention_layout(
        q, k, v, scale=scale, pipe_depth=pipe_depth, num_groups=num_groups,
        accurate_softmax=accurate_softmax,
    ).float()
    ref = _sdpa_ref(q, k, v, scale).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    return max_err, cos


def bench_pipeline_depths(B, Sq, Skv, H, D, dtype_str, *, warmup=10, iters=100):
    dtype = _DTYPES[dtype_str]
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)

    def _pd1():
        flydsl_flex_attention_layout(q, k, v, scale=scale, pipe_depth=1)

    def _ps2():
        flydsl_flex_attention_layout(q, k, v, scale=scale, pipe_depth=2, num_groups=2)

    _, us1 = run_perftest(_pd1, num_iters=iters, num_warmup=warmup)
    _, us3 = run_perftest(_ps2, num_iters=iters, num_warmup=warmup)
    return us1, us3


def print_pipeline_compare(B, Sq, Skv, H, D, dtype_str, *, warmup=10, iters=100):
    us1, us3 = bench_pipeline_depths(
        B, Sq, Skv, H, D, dtype_str, warmup=warmup, iters=iters,
    )
    tf1 = _attn_tflops(B, Sq, Skv, H, D, us1)
    tf3 = _attn_tflops(B, Sq, Skv, H, D, us3)
    print(
        f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}  "
        f"pd1={us1:.1f}us ({tf1:.2f} TFLOPS)  "
        f"ps2={us3:.1f}us ({tf3:.2f} TFLOPS)  "
        f"ps2/pd1={(us1 / us3 if us3 > 0 else float('nan')):.3f}x"
    )
    return us1, us3


_SHAPES = [
    (1, 64, 64, 4, 128),
    (1, 64, 128, 4, 128),
    (2, 128, 128, 8, 128),
    (1, 64, 64, 4, 64),
    (1, 128, 128, 4, 128),
]

_D2_SHAPES = [
    (1, 64, 32, 4, 128),
    (1, 64, 128, 4, 128),
    (1, 64, 64, 4, 64),
    (2, 128, 128, 8, 128),
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout(B, Sq, Skv, H, D, dtype_str):
    max_err, cos = _run(B, Sq, Skv, H, D, dtype_str, pipe_depth=1)
    # Per-column max softmax (no shuffle_xor in hot loop, flash-style).
    # Match flash tolerance: atol=8e-2.
    assert max_err < 8e-2 and cos > 0.98, (
        f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: max_err={max_err} cos={cos}"
    )


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout_approx_softmax(B, Sq, Skv, H, D, dtype_str):
    # Column softmax normalizes by a per-column sum instead of a per-row sum, so it
    # is intentionally inexact; this only guards against it degrading further.
    _, cos = _run(B, Sq, Skv, H, D, dtype_str, pipe_depth=1, accurate_softmax=False)
    assert cos > 0.95, f"approx B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: cos={cos}"


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _D2_SHAPES)
def test_flex_attention_layout_d2(B, Sq, Skv, H, D, dtype_str):
    max_err, cos = _run(B, Sq, Skv, H, D, dtype_str, pipe_depth=2)
    assert max_err < 3e-2 and cos > 0.99, (
        f"d2 B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: max_err={max_err} cos={cos}"
    )


def main():
    p = argparse.ArgumentParser(description="layout flex-attn correctness and/or pipeline bench")
    p.add_argument("--bench", action="store_true", help="compare pd1 vs ps2 (pipe_depth=2 stagger) performance")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seq_len_kv", type=int, default=None, help="defaults to --seq_len")
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "f16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--correctness", action="store_true", help="run correctness sweeps (default if no --bench)")
    args = p.parse_args()

    run_correctness = args.correctness or not args.bench
    if run_correctness:
        for (B, Sq, Skv, H, D) in _SHAPES:
            for dt in ["bf16", "f16"]:
                me, cos = _run(B, Sq, Skv, H, D, dt, pipe_depth=1)
                ok = me < 3e-2 and cos > 0.99
                print(
                    f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dt}: "
                    f"max_err={me:.4g} cos={cos:.5f} -> {'PASS' if ok else 'FAIL'}"
                )
        for (B, Sq, Skv, H, D) in _D2_SHAPES:
            me, cos = _run(B, Sq, Skv, H, D, "bf16", pipe_depth=2)
            ok = me < 3e-2 and cos > 0.99
            print(
                f"d2 B{B} Sq{Sq} Skv{Skv} H{H} D{D} bf16: "
                f"max_err={me:.4g} cos={cos:.5f} -> {'PASS' if ok else 'FAIL'}"
            )

    if args.bench:
        if not get_rocm_arch().startswith("gfx950"):
            print("Pipeline bench requires gfx950; skipping.")
            return
        Skv = args.seq_len_kv if args.seq_len_kv is not None else args.seq_len
        print(f"=== pipeline depth compare (warmup={args.warmup}, iters={args.iters}) ===")
        print_pipeline_compare(
            args.batch, args.seq_len, Skv, args.num_heads, args.head_dim, args.dtype,
            warmup=args.warmup, iters=args.iters,
        )
        print("=== _D2_SHAPES sweep ===")
        for B, Sq, Skv, H, D in _D2_SHAPES:
            print_pipeline_compare(B, Sq, Skv, H, D, "bf16", warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
