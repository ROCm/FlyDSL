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
