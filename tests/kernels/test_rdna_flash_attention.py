#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RDNA4 Flash Attention compatibility benchmark for AITER's FlyDSL kernel.

The FA kernel lives in AITER, but it is implemented using FlyDSL and should stay
compatible with FlyDSL compiler/runtime changes.  This benchmark is intended for
FlyDSL CI on RDNA4 runners and compares AITER's FlyDSL FA wrapper against
PyTorch SDPA for correctness and latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _add_aiter_repo_to_path() -> None:
    aiter_repo = os.environ.get("AITER_REPO", "").strip()
    if aiter_repo and os.path.isdir(aiter_repo) and aiter_repo not in sys.path:
        sys.path.insert(0, aiter_repo)


def _get_arch() -> str:
    try:
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    except Exception:
        return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown")


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("f16", "fp16", "float16"):
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _parse_shapes(value: str) -> list[tuple[int, int, int, str]]:
    shapes: list[tuple[int, int, int, str]] = []
    for item in value.replace(";", " ").split():
        seq_s, heads_s, head_dim_s, dtype_s = [part.strip() for part in item.split(",")]
        shapes.append((int(seq_s), int(heads_s), int(head_dim_s), dtype_s))
    return shapes


def _time_call(fn, repeats: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    times: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    assert out is not None
    return out, times


def _estimate_attention_tflops(batch: int, seq_len: int, heads: int, head_dim: int, seconds: float) -> float:
    # QK^T and PV dominate: 2 matmuls * 2 FLOPs per FMA.
    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    return flops / seconds / 1e12


def _estimate_io_tbps(batch: int, seq_len: int, heads: int, head_dim: int, dtype: torch.dtype, seconds: float) -> float:
    element_size = torch.empty((), dtype=dtype).element_size()
    # q/k/v reads plus output write. This is a conservative visible-IO estimate.
    bytes_moved = 4.0 * batch * seq_len * heads * head_dim * element_size
    return bytes_moved / seconds / 1e12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default=os.environ.get("RDNA_FA_SHAPES", "1024,12,128,bf16 8192,12,128,bf16"),
        help="Space- or semicolon-separated seq_len,num_heads,head_dim,dtype entries.",
    )
    parser.add_argument("--batch", type=int, default=int(os.environ.get("RDNA_FA_BATCH", "1")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("RDNA_FA_WARMUP", "3")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("RDNA_FA_ITERS", "7")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RDNA_FA_SEED", "2029")))
    parser.add_argument("--out-json", default=os.environ.get("RDNA_FA_OUT_JSON", ""))
    parser.add_argument("--allow-non-rdna4-skip", action="store_true", default=True)
    args = parser.parse_args()

    arch = _get_arch()
    if not torch.cuda.is_available():
        if arch.startswith("gfx120"):
            raise RuntimeError(
                "RDNA FA benchmark detected an RDNA4 device through FlyDSL, "
                "but PyTorch cannot initialize CUDA/HIP."
            )
        print("Skipping RDNA FA benchmark: CUDA/HIP device is not available.")
        return 0

    if not arch.startswith("gfx120"):
        if args.allow_non_rdna4_skip:
            print(f"Skipping RDNA FA benchmark: arch={arch!r} is not RDNA4/gfx120*.")
            return 0
        raise RuntimeError(f"RDNA FA benchmark requires gfx120*, got {arch!r}")

    _add_aiter_repo_to_path()
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_func
    except Exception as exc:
        raise RuntimeError(
            "AITER FlyDSL Flash Attention is not importable. "
            "Set AITER_REPO to an AITER checkout containing the RDNA FA kernel."
        ) from exc

    print(f"device {torch.cuda.get_device_name(0)} {arch}", flush=True)
    results = []

    for seq_len, heads, head_dim, dtype_name in _parse_shapes(args.shapes):
        dtype = _dtype_from_name(dtype_name)
        shape = (args.batch, seq_len, heads, head_dim)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + seq_len)
        q = torch.randn(shape, generator=gen, device="cuda", dtype=dtype)
        k = torch.randn(shape, generator=gen, device="cuda", dtype=dtype)
        v = torch.randn(shape, generator=gen, device="cuda", dtype=dtype)

        def run_aiter() -> torch.Tensor:
            return flydsl_flash_attn_func(q, k, v, causal=False)

        def run_sdpa() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2).contiguous()

        # First call compiles the FlyDSL kernel; exclude it from timing.
        run_aiter()
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            run_aiter()
            run_sdpa()
        torch.cuda.synchronize()

        out_aiter, aiter_times = _time_call(run_aiter, args.iters)
        out_sdpa, sdpa_times = _time_call(run_sdpa, args.iters)

        aiter_mean = statistics.mean(aiter_times)
        sdpa_mean = statistics.mean(sdpa_times)
        cos = F.cosine_similarity(
            out_aiter.float().reshape(-1, head_dim),
            out_sdpa.float().reshape(-1, head_dim),
            dim=1,
        )
        mae = float((out_aiter.float() - out_sdpa.float()).abs().mean().item())
        max_abs = float((out_aiter.float() - out_sdpa.float()).abs().max().item())
        mean_cos = float(cos.mean().item())
        min_cos = float(cos.min().item())
        speedup = sdpa_mean / aiter_mean
        tflops = _estimate_attention_tflops(args.batch, seq_len, heads, head_dim, aiter_mean)
        tbps = _estimate_io_tbps(args.batch, seq_len, heads, head_dim, dtype, aiter_mean)

        if mean_cos < 0.999 or mae > 5e-3:
            raise AssertionError(
                f"RDNA FA correctness failed for shape={shape}: "
                f"mean_cos={mean_cos:.6f}, min_cos={min_cos:.6f}, mae={mae:.6g}, max_abs={max_abs:.6g}"
            )

        item = {
            "shape": list(shape),
            "dtype": dtype_name,
            "aiter_mean_s": aiter_mean,
            "aiter_median_s": statistics.median(aiter_times),
            "sdpa_mean_s": sdpa_mean,
            "sdpa_median_s": statistics.median(sdpa_times),
            "speedup_vs_sdpa": speedup,
            "estimated_tflops": tflops,
            "estimated_tbps": tbps,
            "mean_cos": mean_cos,
            "min_cos": min_cos,
            "mae": mae,
            "max_abs": max_abs,
        }
        results.append(item)

        shape_tag = f"{seq_len}x{heads}x{head_dim}"
        print(
            "RDNA FA "
            f"{shape_tag} {dtype_name}: "
            f"aiter={aiter_mean * 1e3:.3f} ms, "
            f"sdpa={sdpa_mean * 1e3:.3f} ms, "
            f"speedup={speedup:.3f}x, "
            f"cos={mean_cos:.6f}, mae={mae:.6g}",
            flush=True,
        )
        print(f"Throughput: estimated {tflops:.3f} TFLOPS, BW: {tbps:.3f} TB/s", flush=True)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"arch": arch, "results": results}, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
