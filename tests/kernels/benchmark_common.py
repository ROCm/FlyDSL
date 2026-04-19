#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared helpers for optional perf comparison in GPU operator tests.

These tests are primarily correctness tests. Performance comparison (FlyDSL vs AIter)
is opt-in via environment variables so CI remains fast/stable.
"""

from __future__ import annotations

import os
import sys

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

# Make repo-root / src-layout packages importable when running as a module:
#   python -m tests.kernels.benchmark_common
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))  # FlyDSL/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

_EMBEDDED_FLYDSL = os.path.join(_REPO_ROOT, ".flydsl", "build", "python_packages", "flydsl")
if os.path.isdir(_EMBEDDED_FLYDSL) and _EMBEDDED_FLYDSL not in sys.path:
    sys.path.insert(0, _EMBEDDED_FLYDSL)


@dataclass(frozen=True)
class PerfRow:
    op: str
    shape: str
    dtype: str
    flydsl_gpu_us: Optional[float]
    aiter_gpu_us: Optional[float]

    @property
    def speedup_aiter_vs_flydsl(self) -> Optional[float]:
        if self.flydsl_gpu_us is None or self.aiter_gpu_us is None:
            return None
        return self.flydsl_gpu_us / self.aiter_gpu_us


def _fmt_us(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:,.1f}"


def print_perf_table(rows: List[PerfRow]) -> None:
    print("\n" + "=" * 100)
    print("Perf Compare (gpu us): FlyDSL vs AIter")
    print("=" * 100)
    print(f"{'op':10s} {'shape':18s} {'dtype':6s} {'FlyDSL(gpu us)':>14s} {'AIter(gpu us)':>14s} {'speedup':>10s}")
    for r in rows:
        sp = r.speedup_aiter_vs_flydsl
        sp_s = "-" if sp is None else f"{sp:,.2f}x"
        print(
            f"{r.op:10s} {r.shape:18s} {r.dtype:6s} {_fmt_us(r.flydsl_gpu_us):>14s} {_fmt_us(r.aiter_gpu_us):>14s} {sp_s:>10s}"
        )
    print("=" * 100 + "\n")


def bench_gpu_us_torch(fn: Callable[[], None], *, warmup: int = 20, iters: int = 200) -> float:
    """Measure device time using torch CUDA events (works for torch-launched kernels, incl. Triton)."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def maybe_enable_aiter() -> bool:
    """Best-effort make `import aiter` work.

    - If already importable: returns True.
    - Else: try inserting AITER_REPO into sys.path.
    """
    try:
        import aiter  # noqa: F401

        return True
    except Exception:
        pass

    # Do not assume any absolute default path; only enable via explicit env var.
    aiter_repo = os.environ.get("AITER_REPO", "").strip()
    if aiter_repo and os.path.isdir(aiter_repo):
        sys.path.insert(0, aiter_repo)
        try:
            import aiter  # noqa: F401

            return True
        except Exception:
            return False
    return False


def _parse_configs(s: str) -> List[Tuple[int, int, str]]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[int, int, str]] = []
    for part in s.split(";"):
        p = part.strip()
        if not p:
            continue
        m_s, n_s, dt = [x.strip() for x in p.split(",")]
        out.append((int(m_s), int(n_s), dt))
    return out


def _default_configs() -> List[Tuple[int, int, str]]:
    # Keep aligned with tests/kernels/test_{softmax,layernorm,rmsnorm}.py defaults.
    return [
        (64, 256, "f32"),
        (128, 1024, "f32"),
        (32, 128, "f16"),
        (64, 2000, "f32"),
        (16, 512, "bf16"),
        (1024, 8192, "bf16"),
        (32768, 8192, "bf16"),
    ]


def _default_wmma_configs() -> List[Tuple[int, int, str]]:
    """Default WMMA GEMM benchmark configs: (M, N=K, dtype)."""
    return [
        (256, 256, "bf16"),
        (1024, 1024, "bf16"),
        (2048, 2048, "bf16"),
        (4096, 4096, "bf16"),
    ]


def _default_fp8_configs() -> List[Tuple[int, int, str]]:
    """Default FP8 GEMM benchmark configs: (M, N=K, dtype='fp8')."""
    return [
        (32, 4096, "fp8"),
        (32, 8192, "fp8"),
        (128, 4096, "fp8"),
        (4096, 4096, "fp8"),
    ]


def _dtype_torch(dt: str):
    dt = dt.lower()
    import torch

    if dt in ("f32", "fp32", "float32"):
        return torch.float32, "f32"
    if dt in ("f16", "fp16", "float16"):
        return torch.float16, "f16"
    if dt in ("bf16", "bfloat16"):
        return torch.bfloat16, "bf16"
    raise ValueError(f"unsupported dtype: {dt}")


def _bench_flydsl_torch(*, op: str, M: int, N: int, dtype: str, warmup: int, iters: int) -> Optional[float]:
    """Build + compile FlyDSL kernel, then benchmark via torch CUDA events.

    This intentionally avoids hip-python / HIP driver calls, aligning with the
    style used by other tests (flydsl.compile + torch timing).
    """
    import torch
    import flydsl

    if not torch.cuda.is_available():
        return None

    torch_dtype, dt_norm = _dtype_torch(dtype)
    dtype = dt_norm

    if op == "softmax":
        from kernels.softmax_kernel import build_softmax_module

        # M is runtime; module construction uses a dummy M.
        # `flydsl.compile()` already has its own cache.
        m = build_softmax_module(1, N, dtype)
        exe = flydsl.compile(m)
        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)
        return bench_gpu_us_torch(lambda: exe(x, y, M), warmup=warmup, iters=iters)

    if op == "layernorm":
        from kernels.layernorm_kernel import build_layernorm_module

        m = build_layernorm_module(1, N, dtype)
        exe = flydsl.compile(m)
        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        gamma = torch.randn((N,), device="cuda", dtype=torch_dtype)
        beta = torch.randn((N,), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)
        return bench_gpu_us_torch(lambda: exe(x, gamma, beta, y, M), warmup=warmup, iters=iters)

    if op == "rmsnorm":
        from kernels.rmsnorm_kernel import build_rmsnorm_module

        m = build_rmsnorm_module(1, N, dtype)
        exe = flydsl.compile(m)
        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        gamma = torch.randn((N,), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)
        return bench_gpu_us_torch(lambda: exe(x, gamma, y, M), warmup=warmup, iters=iters)

    if op == "wmma_gemm":
        from kernels.rdna_f16_gemm import create_wmma_gemm_module

        K = N  # square by default; caller can override via config
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        launch, *_ = create_wmma_gemm_module(M, N, K, in_dtype=dtype, out_dtype="bf16")
        A = torch.randn(M, K, dtype=torch_dtype, device="cuda")
        B_T = torch.randn(N, K, dtype=torch_dtype, device="cuda")
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        return bench_gpu_us_torch(
            lambda: launch(C, A, B_T, torch.cuda.current_stream()),
            warmup=warmup,
            iters=iters,
        )

    if op == "wmma_fp8_gemm":
        from kernels.rdna_fp8_preshuffle_gemm import compile_fp8_gemm, preshuffle_b_fp8, fp8_quantize_per_token, fp8_quantize_per_channel

        K = N  # square by default
        torch.manual_seed(42)
        A_f32 = torch.randn(M, K, device="cuda") * 0.1
        B_f32 = torch.randn(K, N, device="cuda") * 0.1
        A_fp8, sa = fp8_quantize_per_token(A_f32)
        B_fp8, sb = fp8_quantize_per_channel(B_f32)
        B_shuf = preshuffle_b_fp8(B_fp8).view(torch.float32).contiguous()
        A_view = A_fp8.view(torch.float32).contiguous()
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        sa_t = sa.to(device="cuda", dtype=torch.float32).contiguous()
        sb_t = sb.to(device="cuda", dtype=torch.float32).contiguous()
        launch = compile_fp8_gemm(M=M, N=N, K=K)
        return bench_gpu_us_torch(
            lambda: launch(C, A_view, B_shuf, sa_t, sb_t, torch.cuda.current_stream()),
            warmup=warmup,
            iters=iters,
        )

    raise ValueError(f"unknown op: {op}")


def _bench_aiter(*, op: str, impl: str, M: int, N: int, dtype: str, warmup: int, iters: int) -> Optional[float]:
    """Benchmark AIter implementation.

    - impl=triton: uses aiter.ops.triton.*
    """
    if not maybe_enable_aiter():
        return None

    import torch

    torch_dtype, dt_norm = _dtype_torch(dtype)
    dtype = dt_norm
    impl = (impl or "triton").lower()

    try:
        import aiter
    except Exception:
        return None

    if impl == "triton":
        if op == "softmax":
            from aiter.ops.triton.softmax import softmax as fn

            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: fn(x), warmup=warmup, iters=iters)
        if op == "layernorm":
            from aiter.ops.triton.norm import layer_norm as fn

            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            w = torch.randn((N,), device="cuda", dtype=torch_dtype)
            b = torch.randn((N,), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: fn(x, w, b, 1e-5, None), warmup=warmup, iters=iters)
        if op == "rmsnorm":
            from aiter.ops.triton.rmsnorm import rms_norm as fn

            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            w = torch.randn((N,), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: fn(x, w, 1e-5), warmup=warmup, iters=iters)
        return None

    raise ValueError(f"unsupported AITER_IMPL={impl!r} (expected triton)")


def run_compare_sweep(
    *,
    configs: List[Tuple[int, int, str]],
    aiter_impl: str = "triton",
    warmup: int = 10,
    iters: int = 50,
) -> List[PerfRow]:
    rows: List[PerfRow] = []
    for M, N, dt in configs:
        shape = f"{M}x{N}"
        for op in ("softmax", "layernorm", "rmsnorm"):
            flydsl_us = None
            aiter_us = None
            try:
                flydsl_us = _bench_flydsl_torch(op=op, M=M, N=N, dtype=dt, warmup=warmup, iters=iters)
            except Exception:
                flydsl_us = None
            try:
                aiter_us = _bench_aiter(op=op, impl=aiter_impl, M=M, N=N, dtype=dt, warmup=warmup, iters=iters)
            except Exception:
                aiter_us = None
            rows.append(PerfRow(op=op, shape=shape, dtype=dt, flydsl_gpu_us=flydsl_us, aiter_gpu_us=aiter_us))
    return rows


def run_wmma_sweep(
    *,
    warmup: int = 10,
    iters: int = 50,
) -> List[PerfRow]:
    """Benchmark WMMA GEMM kernels (RDNA4 only) vs torch."""
    import torch

    rows: List[PerfRow] = []

    from flydsl.runtime.device import get_rocm_arch

    arch = get_rocm_arch()
    if not arch.startswith("gfx120"):
        return rows

    fail_count = 0

    # wmma_gemm (LDS bf16)
    for M, N, dt in _default_wmma_configs():
        K = N
        shape = f"{M}x{N}x{K}"
        flydsl_us = None
        torch_us = None
        try:
            flydsl_us = _bench_flydsl_torch(op="wmma_gemm", M=M, N=N, dtype=dt, warmup=warmup, iters=iters)
        except Exception as e:
            print(f"ERROR: wmma_gemm {shape} FAILED: {e}")
            fail_count += 1
        try:
            torch_dtype, _ = _dtype_torch(dt)
            A = torch.randn(M, K, dtype=torch_dtype, device="cuda")
            B = torch.randn(K, N, dtype=torch_dtype, device="cuda")
            C = torch.zeros(M, N, dtype=torch_dtype, device="cuda")
            torch_us = bench_gpu_us_torch(lambda: torch.mm(A, B, out=C), warmup=warmup, iters=iters)
        except Exception:
            pass  # torch reference failure is non-fatal
        rows.append(PerfRow(op="wmma_gemm", shape=shape, dtype=dt, flydsl_gpu_us=flydsl_us, aiter_gpu_us=torch_us))

    # wmma_fp8_gemm (A raw, B preshuffled)
    for M, N, dt in _default_fp8_configs():
        K = N
        shape = f"{M}x{N}x{K}"
        flydsl_us = None
        torch_us = None
        try:
            flydsl_us = _bench_flydsl_torch(op="wmma_fp8_gemm", M=M, N=N, dtype="bf16", warmup=warmup, iters=iters)
        except Exception as e:
            print(f"ERROR: fp8_gemm {shape} FAILED: {e}")
            fail_count += 1
        try:
            from kernels.rdna_fp8_preshuffle_gemm import fp8_quantize_per_token, fp8_quantize_per_channel

            A_f32 = torch.randn(M, K, device="cuda") * 0.1
            B_f32 = torch.randn(K, N, device="cuda") * 0.1
            A_fp8, sa = fp8_quantize_per_token(A_f32)
            B_fp8, sb = fp8_quantize_per_channel(B_f32)
            B_col = B_fp8.T.contiguous().T
            sa_t = sa.to(device="cuda", dtype=torch.float32).unsqueeze(1).contiguous()   # (M, 1)
            sb_t = sb.to(device="cuda", dtype=torch.float32).unsqueeze(0).contiguous()   # (1, N)
            torch_us = bench_gpu_us_torch(
                lambda: torch._scaled_mm(A_fp8, B_col, scale_a=sa_t, scale_b=sb_t, out_dtype=torch.bfloat16),
                warmup=warmup,
                iters=iters,
            )
        except Exception:
            pass  # torch reference failure is non-fatal
        rows.append(PerfRow(op="fp8_gemm", shape=shape, dtype="fp8", flydsl_gpu_us=flydsl_us, aiter_gpu_us=torch_us))

    if fail_count > 0:
        raise RuntimeError(f"{fail_count} RDNA WMMA benchmark(s) failed — see errors above")

    return rows


# ── MOE bench common helpers ──────────────────────────────────────────────

BENCH_WARMUP = 5
BENCH_ITERS = 20

BENCH_MODEL_CONFIGS = [
    # name,      model_dim, inter_dim, experts, topk
    ("DeepSeek-TP", 7168,    256,   257,  9),
    ("DeepSeek-EP", 7168,   2048,    32,  8),
    ("GPToss",      2880,   2880,   128,  4),
]

BENCH_DTYPE_TARGET_TILES = {
    # dtype: (tile_m, target_n, target_k, wmma_k)
    "fp4":  (16, 256, 512, 128),
    "fp8":  (16, 256, 512, 128),
    "a8w4": (16, 256, 512, 128),
    "fp16": (32,  64,  64,  32),
    "bf16": (32,  64,  64,  32),
}

BENCH_DEFAULT_TOKEN_SWEEP = [1, 4, 8, 32, 64, 128, 256]
_BENCH_SCALE_GROUP = 32


def bench_kernel_us(run_fn, warmup=10, iters=50, flush_l2=True, prep_fn=None):
    """Per-iteration CUDA events timer with optional L2 flush and median latency."""
    import torch

    flush_buf = None
    if flush_l2:
        l2_bytes = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "L2_cache_size", 4 * 1024 * 1024)
        alloc_bytes = max(l2_bytes * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        run_fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        start_ev[i].record()
        run_fn()
        end_ev[i].record()

    torch.cuda.synchronize()
    latencies = sorted(start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered

    del flush_buf
    return latencies[len(latencies) // 2]


def bench_best_tile(target, dim, align):
    """Largest value <= target that divides dim and is a multiple of align."""
    for v in range(target, 0, -align):
        if dim % v == 0:
            return v
    return None


def bench_resolve_tiles(in_dtype, model_dim, inter_dim):
    """Compute the largest valid (tile_m, tile_n1, tile_k1, tile_n2, tile_k2)
    for a given dtype and model shape, falling back from the target when
    dimensions don't divide evenly."""
    tile_m, target_n, target_k, wmma_k = BENCH_DTYPE_TARGET_TILES[in_dtype]

    tile_n1 = bench_best_tile(target_n, inter_dim, 16)
    tile_k1 = bench_best_tile(target_k, model_dim, wmma_k)
    tile_n2 = bench_best_tile(target_n, model_dim, 16)

    tile_k2 = None
    for k in range(target_k, 0, -wmma_k):
        if inter_dim % k != 0:
            continue
        total = tile_m * k
        if total % 256 == 0 and (total // 256) % 4 == 0:
            tile_k2 = k
            break

    if any(v is None for v in (tile_n1, tile_k1, tile_n2, tile_k2)):
        return None
    return (tile_m, tile_n1, tile_k1, tile_n2, tile_k2)


def bench_dtype_bpe(in_dtype):
    """Return (a_bpe, w_bpe, w_scale_bpg) for bandwidth accounting."""
    if in_dtype == "fp4":
        return 0.5, 0.5, 1
    if in_dtype == "a8w4":
        return 1, 0.5, 1
    if in_dtype == "fp8":
        return 1, 1, 1
    if in_dtype in ("fp16", "bf16"):
        return 2, 2, 0
    return 1, 1, 1


def bench_bytes_moved_stage1(tokens, topk, model_dim, inter_dim, experts, in_dtype):
    import math
    a_bpe, w_bpe, w_scale_bpg = bench_dtype_bpe(in_dtype)
    aE = min(tokens * topk, experts)
    b = 0
    b += tokens * model_dim * a_bpe
    b += aE * (2 * inter_dim) * model_dim * w_bpe
    b += aE * (2 * inter_dim) * math.ceil(model_dim / _BENCH_SCALE_GROUP) * w_scale_bpg
    b += tokens * topk * inter_dim * 2
    return int(b)


def bench_bytes_moved_stage2(tokens, topk, model_dim, inter_dim, experts, in_dtype):
    import math
    a_bpe, w_bpe, w_scale_bpg = bench_dtype_bpe(in_dtype)
    aE = min(tokens * topk, experts)
    b = 0
    b += tokens * topk * inter_dim * a_bpe
    b += aE * model_dim * inter_dim * w_bpe
    b += aE * model_dim * math.ceil(inter_dim / _BENCH_SCALE_GROUP) * w_scale_bpg
    b += tokens * topk * model_dim * 2
    return int(b)


def bench_print_banner(text):
    print(f"\n{'=' * 110}")
    print(f"  {text}")
    print(f"{'=' * 110}")


def bench_print_stage_header():
    print(f"{'Tokens':>7} {'M_eff':>7} {'Latency(us)':>12} {'TFLOPS':>9} "
          f"{'BW(TB/s)':>10} {'Util%':>7} {'Status':>8}")
    print("-" * 110)


def bench_print_stage_row(tokens, m_eff, us, tflops, tbps, util_pct, status):
    print(f"{tokens:>7} {m_eff:>7} {us:>10.1f}   {tflops:>8.2f} "
          f"{tbps:>9.3f}  {util_pct:>6.1f}% {status:>8}")


# ── MOE bench sweep system ─────────────────────────────────────────────────
# Generic benchmark sweep for MoE 2-stage kernels: sweeps model configs ×
# dtypes × token counts.  Callers provide the stage1/stage2 runner functions
# and data-setup helpers so this module stays kernel-agnostic.


def add_moe_bench_args(parser) -> None:
    """Register the ``--bench`` argument group on *parser*.

    Call this from the test script's ``if __name__ == '__main__':`` block so
    the user can ``python test_xxx.py --bench ...``.
    """
    import argparse  # noqa: F811 – local import to avoid top-level dep

    bench_group = parser.add_argument_group(
        "benchmark sweep",
        "Options for --bench mode (sweep model configs × dtypes × token counts)",
    )
    bench_group.add_argument("--bench", action="store_true", default=False,
                             help="Run benchmark sweep mode instead of normal test mode.")
    bench_group.add_argument("--bench-dtype", type=str, default=None,
                             help="Comma-separated dtypes for bench (default: all keys in BENCH_DTYPE_TARGET_TILES).")
    bench_group.add_argument("--bench-tokens", type=str, default=None,
                             help="Comma-separated token counts for bench (default: 1,4,8,32,64,128,256).")
    bench_group.add_argument("--bench-config", type=str, default=None,
                             help="Config name filter for bench (DeepSeek-TP, DeepSeek-EP, GPToss).")
    bench_group.add_argument("--bench-no-ref", action="store_true", default=False,
                             help="Skip correctness reference check in bench mode (pure perf).")
    bench_group.add_argument("--bench-warmup", type=int, default=BENCH_WARMUP,
                             help=f"Warmup iterations for bench (default: {BENCH_WARMUP}).")
    bench_group.add_argument("--bench-iters", type=int, default=BENCH_ITERS,
                             help=f"Measurement iterations for bench (default: {BENCH_ITERS}).")
    bench_group.add_argument("--bench-peak-tflops", type=float, default=0,
                             help="Peak TFLOPS for utilization calculation in bench mode.")


def moe_bench_config(
    name: str,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    in_dtype: str,
    token_list: List[int],
    check_ref: bool,
    peak_tflops: float,
    *,
    stage1_fn: Callable,
    stage2_fn: Callable,
    setup_data_fn: Callable,
    prepare_a2_fn: Callable,
    warmup: int = BENCH_WARMUP,
    iters: int = BENCH_ITERS,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
) -> None:
    """Benchmark a single (model, dtype) configuration across all token counts.

    Parameters
    ----------
    stage1_fn : callable
        ``run_moe_stage1(...)`` from the test harness.
    stage2_fn : callable
        ``run_moe_stage2(...)`` from the test harness.
    setup_data_fn : callable
        ``(tokens, model_dim, inter_dim, experts, topk, tile_m) -> (x, w1, w2, ids, wts, routing)``
    prepare_a2_fn : callable
        ``(out1_fp16, tokens, topk, inter_dim, in_dtype) -> (a2_q, a2_scale)``
    """
    import torch

    tiles = bench_resolve_tiles(in_dtype, model_dim, inter_dim)
    if tiles is None:
        bench_print_banner(f"{name}  |  {in_dtype}  |  dim={model_dim}  inter={inter_dim}")
        print(f"  SKIP: no valid tile for this shape (WMMA_K alignment)")
        return
    tile_m, tile_n1, tile_k1, tile_n2, tile_k2 = tiles

    bench_print_banner(f"{name}  |  {in_dtype}  |  dim={model_dim}  inter={inter_dim}  E={experts}  K={topk}")
    print(f"  Tiles: stage1=({tile_m},{tile_n1},{tile_k1})  stage2=({tile_m},{tile_n2},{tile_k2})")
    print(f"  Warmup={warmup}  Iters={iters}  RefCheck={'ON' if check_ref else 'OFF'}")
    print(
        f"  Knobs: use_tdm_store={bool(use_tdm_store)}  "
        f"inst_prefetch={bool(inst_prefetch)}  "
        f"wave_specialized_tdm={bool(wave_specialized_tdm)}"
    )
    if peak_tflops > 0:
        print(f"  Peak compute reference: {peak_tflops:.0f} TFLOPS")

    # ── Stage 1 ──
    print(f"\n  ── Stage 1 (gate+up: [{model_dim}] -> [{2*inter_dim}]) ──")
    bench_print_stage_header()

    s1_results = []
    for tok in token_list:
        torch.cuda.empty_cache()
        x, w1, w2, ids, wts, routing = setup_data_fn(tok, model_dim, inter_dim, experts, topk, tile_m)
        try:
            out1, us1 = stage1_fn(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype,
                tile_m=tile_m, tile_n=tile_n1, tile_k=tile_k1,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                return_outputs=True, skip_ref=(not check_ref),
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us1 = 0.0
            out1 = torch.zeros((tok, topk, inter_dim), device="cuda", dtype=torch.float16)
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        m_eff = tok * topk
        flops = 2 * tok * topk * (2 * inter_dim) * model_dim
        tflops = flops / (us1 / 1e6) / 1e12 if us1 > 0 else 0
        bm = bench_bytes_moved_stage1(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us1 / 1e6) if us1 > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        bench_print_stage_row(tok, m_eff, us1, tflops, tbps, util, status)
        s1_results.append((tok, m_eff, us1, tflops, tbps, status, out1))

    # ── Stage 2 atomic ──
    print(f"\n  ── Stage 2 atomic (down: [{inter_dim}] -> [{model_dim}]) ──")
    bench_print_stage_header()

    for tok, m_eff, _, _, _, s1_status, out1 in s1_results:
        torch.cuda.empty_cache()
        x, w1, w2, ids, wts, routing = setup_data_fn(tok, model_dim, inter_dim, experts, topk, tile_m)
        a2_q, a2_scale = prepare_a2_fn(out1, tok, topk, inter_dim, in_dtype)
        try:
            _, us2 = stage2_fn(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype, out_dtype="f16",
                tile_m=tile_m, tile_n=tile_n2, tile_k=tile_k2,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                a2_fp8_in=a2_q, a2_scale_in=a2_scale,
                return_outputs=True, skip_ref=(not check_ref),
                use_reduce=False,
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us2 = 0.0
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        flops = 2 * tok * topk * model_dim * inter_dim
        tflops = flops / (us2 / 1e6) / 1e12 if us2 > 0 else 0
        bm = bench_bytes_moved_stage2(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us2 / 1e6) if us2 > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        bench_print_stage_row(tok, m_eff, us2, tflops, tbps, util, status)

    # ── Stage 2 reduce ──
    print(f"\n  ── Stage 2 reduce (down: [{inter_dim}] -> [{model_dim}]) ──")
    bench_print_stage_header()

    for tok, m_eff, _, _, _, s1_status, out1 in s1_results:
        torch.cuda.empty_cache()
        x, w1, w2, ids, wts, routing = setup_data_fn(tok, model_dim, inter_dim, experts, topk, tile_m)
        a2_q, a2_scale = prepare_a2_fn(out1, tok, topk, inter_dim, in_dtype)
        try:
            _, us2r = stage2_fn(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype, out_dtype="f16",
                tile_m=tile_m, tile_n=tile_n2, tile_k=tile_k2,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                a2_fp8_in=a2_q, a2_scale_in=a2_scale,
                return_outputs=True, skip_ref=(not check_ref),
                use_reduce=True,
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us2r = 0.0
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        flops = 2 * tok * topk * model_dim * inter_dim
        tflops = flops / (us2r / 1e6) / 1e12 if us2r > 0 else 0
        bm = bench_bytes_moved_stage2(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us2r / 1e6) if us2r > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        bench_print_stage_row(tok, m_eff, us2r, tflops, tbps, util, status)

    del s1_results
    torch.cuda.empty_cache()


def moe_bench_main(
    args,
    *,
    stage1_fn: Callable,
    stage2_fn: Callable,
    setup_data_fn: Callable,
    prepare_a2_fn: Callable,
) -> None:
    """Entry point for ``--bench`` mode: sweep model configs × dtypes × token counts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args (must include the ``--bench-*`` group from ``add_moe_bench_args``
        and ``use_tdm_store``, ``inst_prefetch``, ``wave_specialized_tdm``).
    stage1_fn, stage2_fn, setup_data_fn, prepare_a2_fn :
        Kernel-specific callables (see ``moe_bench_config`` for signatures).
    """
    import time
    import torch

    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"

    warmup = args.bench_warmup
    iters = args.bench_iters

    dtypes = args.bench_dtype.split(",") if args.bench_dtype else list(BENCH_DTYPE_TARGET_TILES.keys())
    token_list = (
        [int(t) for t in args.bench_tokens.split(",")]
        if args.bench_tokens
        else BENCH_DEFAULT_TOKEN_SWEEP
    )
    check_ref = not args.bench_no_ref

    print("=" * 110)
    print("  AMD gfx1250 MOE GEMM Kernel Performance Benchmark")
    print(f"  PyTorch {torch.__version__}")
    print(f"  Device:  {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  CUs:     {props.multi_processor_count}")
    print(f"  Memory:  {props.total_memory / 1024**3:.0f} GB")
    print(f"  Warmup={warmup}  Iters={iters}  RefCheck={'ON' if check_ref else 'OFF'}")
    print(f"  Dtypes:  {dtypes}")
    print(f"  Tokens:  {token_list}")
    print("=" * 110)

    t_start = time.time()
    for cfg_name, mdim, idim, exp, topk in BENCH_MODEL_CONFIGS:
        if args.bench_config and args.bench_config not in cfg_name:
            continue
        for dt in dtypes:
            if dt not in BENCH_DTYPE_TARGET_TILES:
                print(f"\n  [SKIP] Unknown dtype: {dt}")
                continue
            try:
                moe_bench_config(
                    cfg_name, mdim, idim, exp, topk,
                    dt, token_list, check_ref, args.bench_peak_tflops,
                    stage1_fn=stage1_fn,
                    stage2_fn=stage2_fn,
                    setup_data_fn=setup_data_fn,
                    prepare_a2_fn=prepare_a2_fn,
                    warmup=warmup, iters=iters,
                    use_tdm_store=bool(args.use_tdm_store),
                    inst_prefetch=bool(args.inst_prefetch),
                    wave_specialized_tdm=bool(args.wave_specialized_tdm),
                )
            except Exception as e:
                print(f"\n  [ERROR] {cfg_name}/{dt}: {e}")
                import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    bench_print_banner(f"Done in {elapsed:.1f}s")


def main() -> None:
    # CLI entrypoint:
    #   BENCH_CONFIGS="M,N,dtype;..." AITER_IMPL=triton BENCH_WARMUP=10 BENCH_ITERS=50 python -m tests.kernels.benchmark_common
    configs = _parse_configs(os.environ.get("BENCH_CONFIGS", "")) or _default_configs()
    aiter_impl = os.environ.get("AITER_IMPL", "triton")
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    iters = int(os.environ.get("BENCH_ITERS", "50"))
    rows = run_compare_sweep(configs=configs, aiter_impl=aiter_impl, warmup=warmup, iters=iters)
    print_perf_table(rows)

    # WMMA GEMM benchmarks (RDNA4 only)
    wmma_rows = run_wmma_sweep(warmup=warmup, iters=iters)
    if wmma_rows:
        print("\n" + "=" * 100)
        print("Perf Compare (gpu us): FlyDSL WMMA vs torch (RDNA4)")
        print("=" * 100)
        print(f"{'op':10s} {'shape':18s} {'dtype':6s} {'FlyDSL(gpu us)':>14s} {'torch(gpu us)':>14s} {'speedup':>10s}")
        for r in wmma_rows:
            sp = r.speedup_aiter_vs_flydsl
            sp_s = "-" if sp is None else f"{sp:,.2f}x"
            print(
                f"{r.op:10s} {r.shape:18s} {r.dtype:6s} {_fmt_us(r.flydsl_gpu_us):>14s} {_fmt_us(r.aiter_gpu_us):>14s} {sp_s:>10s}"
            )
        print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
