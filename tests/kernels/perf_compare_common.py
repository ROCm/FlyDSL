#!/usr/bin/env python3
"""Shared helpers for optional perf comparison in GPU operator tests.

These tests are primarily correctness tests. Performance comparison (FLIR vs AIter)
is opt-in via environment variables so CI remains fast/stable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# Make repo-root / src-layout packages importable when running as a module:
#   python -m tests.kernels.perf_compare_common
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))  # FlyDSL/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

_EMBEDDED_FLYDSL = os.path.join(_REPO_ROOT, ".flir", "build", "python_packages", "flydsl")
if os.path.isdir(_EMBEDDED_FLYDSL) and _EMBEDDED_FLYDSL not in sys.path:
    sys.path.insert(0, _EMBEDDED_FLYDSL)


@dataclass(frozen=True)
class PerfRow:
    op: str
    shape: str
    dtype: str
    flir_gpu_us: Optional[float]
    aiter_gpu_us: Optional[float]

    @property
    def speedup_aiter_vs_flir(self) -> Optional[float]:
        if self.flir_gpu_us is None or self.aiter_gpu_us is None:
            return None
        return self.flir_gpu_us / self.aiter_gpu_us


def _fmt_us(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:,.1f}"


def print_perf_table(rows: List[PerfRow]) -> None:
    print("\n" + "=" * 100)
    print("Perf Compare (gpu us): FLIR vs AIter")
    print("=" * 100)
    print(f"{'op':10s} {'shape':18s} {'dtype':6s} {'FLIR(gpu us)':>14s} {'AIter(gpu us)':>14s} {'speedup':>10s}")
    for r in rows:
        sp = r.speedup_aiter_vs_flir
        sp_s = "-" if sp is None else f"{sp:,.2f}x"
        print(f"{r.op:10s} {r.shape:18s} {r.dtype:6s} {_fmt_us(r.flir_gpu_us):>14s} {_fmt_us(r.aiter_gpu_us):>14s} {sp_s:>10s}")
    print("=" * 100 + "\n")


def bench_gpu_us_hip(launch: Callable[[], None], *, warmup: int = 20, iters: int = 200) -> float:
    """Measure device time using HIP events (works for hipModuleLaunchKernel launches)."""
    try:
        from hip import hip
        from flydsl.runtime.hip_util import hip_check
    except Exception as e:
        raise RuntimeError("HIP python bindings are required for bench_gpu_us_hip") from e

    start = hip_check(hip.hipEventCreate())
    end = hip_check(hip.hipEventCreate())
    try:
        for _ in range(warmup):
            launch()
        hip_check(hip.hipDeviceSynchronize())
        hip_check(hip.hipEventRecord(start, 0))
        for _ in range(iters):
            launch()
        hip_check(hip.hipEventRecord(end, 0))
        hip_check(hip.hipEventSynchronize(end))
        ms = hip_check(hip.hipEventElapsedTime(start, end))
        return float(ms) * 1e3 / iters
    finally:
        hip_check(hip.hipEventDestroy(start))
        hip_check(hip.hipEventDestroy(end))


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
    - Else: try inserting AITER_REPO (default: /home/xzhu/aiter) into sys.path.
    """
    try:
        import aiter  # noqa: F401
        return True
    except Exception:
        pass

    aiter_repo = os.environ.get("AITER_REPO", "/home/xzhu/aiter")
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


def _bench_flydsl_hip(*, op: str, M: int, N: int, dtype: str, warmup: int, iters: int) -> Optional[float]:
    """Build + compile FlyDSL kernel, then benchmark launch via HIP events.

    Note: compile time is not included in timing.
    """
    try:
        import ctypes
        import torch
        from hip import hip
        from flydsl.runtime.hip_util import hip_check
        from tests.utils import compile_to_hsaco
    except Exception as e:
        raise RuntimeError("FlyDSL HIP benchmark requires torch+hip") from e

    torch_dtype, dt_norm = _dtype_torch(dtype)
    dtype = dt_norm

    if op == "softmax":
        from kernels.softmax_kernel import build_softmax_module, KERNEL_NAME as KNAME
        ctx = build_softmax_module(M, N, dtype)
        hsaco = compile_to_hsaco(ctx.module, kernel_name=KNAME)
        mod = hip_check(hip.hipModuleLoadData(hsaco))
        fn = hip_check(hip.hipModuleGetFunction(mod, KNAME.encode("utf-8")))

        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)

        blk = min(256, 1 if N == 0 else 2 ** ((N - 1).bit_length()))
        if blk < 32:
            blk = 32

        def launch():
            arg_ptrs = [ctypes.c_void_p(int(x.data_ptr())), ctypes.c_void_p(int(y.data_ptr()))]
            args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
            hip_check(
                hip.hipModuleLaunchKernel(
                    fn, M, 1, 1, blk, 1, 1, 0, 0, args, None
                )
            )

        return bench_gpu_us_hip(launch, warmup=warmup, iters=iters)

    if op == "layernorm":
        from kernels.layernorm_kernel import build_layernorm_module, BLOCK_THREADS
        ctx = build_layernorm_module(M, N, dtype)
        from tests.utils import compile_to_hsaco
        hsaco = compile_to_hsaco(ctx.module, kernel_name="layernorm")
        mod = hip_check(hip.hipModuleLoadData(hsaco))
        fn = hip_check(hip.hipModuleGetFunction(mod, b"layernorm_kernel"))

        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        gamma = torch.randn((N,), device="cuda", dtype=torch_dtype)
        beta = torch.randn((N,), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)

        def launch():
            arg_ptrs = [
                ctypes.c_void_p(int(x.data_ptr())),
                ctypes.c_void_p(int(gamma.data_ptr())),
                ctypes.c_void_p(int(beta.data_ptr())),
                ctypes.c_void_p(int(y.data_ptr())),
            ]
            args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
            hip_check(
                hip.hipModuleLaunchKernel(
                    fn, M, 1, 1, BLOCK_THREADS, 1, 1, 0, 0, args, None
                )
            )

        return bench_gpu_us_hip(launch, warmup=warmup, iters=iters)

    if op == "rmsnorm":
        from kernels.rmsnorm_kernel import build_rmsnorm_module, BLOCK_THREADS
        ctx = build_rmsnorm_module(M, N, dtype)
        from tests.utils import compile_to_hsaco
        hsaco = compile_to_hsaco(ctx.module, kernel_name="rmsnorm")
        mod = hip_check(hip.hipModuleLoadData(hsaco))
        fn = hip_check(hip.hipModuleGetFunction(mod, b"rmsnorm_kernel"))

        x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
        gamma = torch.randn((N,), device="cuda", dtype=torch_dtype)
        y = torch.empty((M, N), device="cuda", dtype=torch_dtype)

        def launch():
            arg_ptrs = [
                ctypes.c_void_p(int(x.data_ptr())),
                ctypes.c_void_p(int(gamma.data_ptr())),
                ctypes.c_void_p(int(y.data_ptr())),
            ]
            args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
            hip_check(
                hip.hipModuleLaunchKernel(
                    fn, M, 1, 1, BLOCK_THREADS, 1, 1, 0, 0, args, None
                )
            )

        return bench_gpu_us_hip(launch, warmup=warmup, iters=iters)

    raise ValueError(f"unknown op: {op}")


def _bench_aiter(*, op: str, impl: str, M: int, N: int, dtype: str, warmup: int, iters: int) -> Optional[float]:
    """Benchmark AIter implementation.

    - impl=triton: uses aiter.ops.triton.*
    - impl=hip: uses aiter.*2d_hip (may have limited shape support; best-effort)
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

    if impl == "hip":
        # Best-effort: these kernels are not guaranteed to support large M/N.
        if op == "softmax" and hasattr(aiter, "softmax2d_hip"):
            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: aiter.softmax2d_hip(x), warmup=warmup, iters=iters)
        if op == "layernorm" and hasattr(aiter, "layernorm2d_hip"):
            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            w = torch.randn((N,), device="cuda", dtype=torch_dtype)
            b = torch.randn((N,), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: aiter.layernorm2d_hip(x, w, b, 1e-5, None), warmup=warmup, iters=iters)
        if op == "rmsnorm" and hasattr(aiter, "rmsnorm2d_hip"):
            x = torch.randn((M, N), device="cuda", dtype=torch_dtype)
            w = torch.randn((N,), device="cuda", dtype=torch_dtype)
            return bench_gpu_us_torch(lambda: aiter.rmsnorm2d_hip(x, w, 1e-5, 0), warmup=warmup, iters=iters)
        return None

    raise ValueError(f"unsupported AITER_IMPL={impl!r} (expected triton|hip)")


def run_compare_sweep(
    *,
    configs: List[Tuple[int, int, str]],
    aiter_impl: str = "triton",
    warmup: int = 10,
    iters: int = 50,
) -> List[PerfRow]:
    rows: List[PerfRow] = []
    for (M, N, dt) in configs:
        shape = f"{M}x{N}"
        for op in ("softmax", "layernorm", "rmsnorm"):
            flir_us = None
            aiter_us = None
            try:
                flir_us = _bench_flydsl_hip(op=op, M=M, N=N, dtype=dt, warmup=warmup, iters=iters)
            except Exception:
                flir_us = None
            try:
                aiter_us = _bench_aiter(op=op, impl=aiter_impl, M=M, N=N, dtype=dt, warmup=warmup, iters=iters)
            except Exception:
                aiter_us = None
            rows.append(PerfRow(op=op, shape=shape, dtype=dt, flir_gpu_us=flir_us, aiter_gpu_us=aiter_us))
    return rows


def main() -> None:
    # CLI entrypoint:
    #   BENCH_CONFIGS="M,N,dtype;..." AITER_IMPL=triton BENCH_WARMUP=10 BENCH_ITERS=50 python -m tests.kernels.perf_compare_common
    configs = _parse_configs(os.environ.get("BENCH_CONFIGS", "")) or _default_configs()
    aiter_impl = os.environ.get("AITER_IMPL", "triton")
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    iters = int(os.environ.get("BENCH_ITERS", "50"))
    rows = run_compare_sweep(configs=configs, aiter_impl=aiter_impl, warmup=warmup, iters=iters)
    print_perf_table(rows)


if __name__ == "__main__":
    main()

