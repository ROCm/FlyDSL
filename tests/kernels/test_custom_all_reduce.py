#!/usr/bin/env python3
"""Custom all-reduce kernel tests.

This test file provides:
- **Single-process** correctness tests (default, always runs when CUDA/ROCm is available).
- **Optional multi-process distributed** correctness tests inspired by AIter's
  `tensor_model_parallel_all_reduce` harness (guarded by env vars).
"""

import os
import sys
from pathlib import Path

# Prefer embedded MLIR/flydsl to avoid mixing multiple runtimes.
_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "flydsl"
_embedded2 = _repo / ".flir" / "build" / "python_packages" / "flydsl"
_embedded_pick = _embedded if _embedded.exists() else _embedded2
if _embedded_pick.exists():
    os.environ.setdefault("FLYDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded_pick))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

from kernels.custom_all_reduce import init_custom_ar, meta_size


def _parse_tuple(s: str):
    s = (s or "").strip()
    if not s:
        raise ValueError("empty tuple string")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _normalize_dtype_arg(dtype_arg: str) -> str:
    """Accept both AIter-style names (fp16/bf16/fp32) and internal (f16/bf16/f32)."""
    d = (dtype_arg or "").strip().lower()
    if d in {"fp16", "f16"}:
        return "f16"
    if d in {"bf16"}:
        return "bf16"
    if d in {"fp32", "f32"}:
        return "f32"
    raise ValueError(f"unsupported dtype: {dtype_arg}")


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _dist_worker(rank: int, world_size: int, shape, dtype_str: str, with_graph: bool, port: int, profile: bool = False):
    import torch
    import torch.distributed as dist
    import time

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    if dtype_str == "f32":
        dtype = DTYPE_FP32
        atol = 1e-4
    elif dtype_str == "f16":
        dtype = DTYPE_FP16
        # NOTE: reference is fp32 sum; output is fp16, so error is dominated by fp16 quantization.
        # Scale by sqrt(world_size) since sum magnitude grows ~sqrt(ws) for random inputs.
        atol = 2e-3 * (float(world_size) ** 0.5) + 2e-3
    elif dtype_str == "bf16":
        dtype = DTYPE_BF16
        # bf16 has lower mantissa; scale tolerance similarly.
        atol = 2e-3 * float(world_size) + 2e-3
    else:
        raise ValueError(f"unsupported dtype_str: {dtype_str}")

    x = torch.randn(shape, device=device, dtype=dtype).contiguous()
    x_flat = x.reshape(-1).contiguous()

    # -------------------------------------------------------------------------
    # Route B (IPC): exchange HIP IPC handles for each rank's input buffer.
    # -------------------------------------------------------------------------
    from flydsl.runtime import ipc as fly_ipc

    my_ipc = fly_ipc.get_ipc_handle(x_flat)
    gathered_ipc = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_ipc, (my_ipc.handle, int(my_ipc.offset_bytes)), group=group)

    # Pack handles into CPU uint8 tensors (AIter-like shape).
    handles = []
    offsets = []
    for h_bytes, off in gathered_ipc:
        handles.append(torch.tensor(list(h_bytes), device="cpu", dtype=torch.uint8))
        offsets.append(int(off))

    # AIter-like meta/rank_data (we repurpose rank_data as the local registered buffer).
    meta = torch.empty((meta_size(),), device=device, dtype=torch.int8)
    rank_data = x_flat
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank, full_nvlink=False)

    # Warmup: align all ranks.
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    out = torch.empty_like(x_flat)

    # -------------------------------------------------------------------------
    # Profiling
    # - end-to-end: dist.all_gather + pack + kernel + sync
    # - kernel-only: just fa.all_reduce_reg (timed by cuda events)
    # -------------------------------------------------------------------------
    iters = int(os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_PROFILE_ITERS", "50"))
    warmup = int(os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_PROFILE_WARMUP", "5"))

    try:
        if with_graph:
            # NOTE: FlyDSL's current ROCm runtime path may call hipStreamSynchronize internally,
            # which is incompatible with HIP graph capture and can invalidate the capture state.
            # Keep the flag for CLI parity but run eager.
            print(
                f"[rank={rank}] WARN: --withGraph is not supported for FlyDSL custom_all_reduce on ROCm yet; running eager.",
                file=sys.stderr,
            )

        # Compile/warmup once (not included in steady-state timings).
        fa.all_reduce_reg(x_flat, out, open_fp8_quant=False)
        torch.cuda.synchronize()

        kernel_ms_list = []
        e2e_ms_list = []
        if profile:
            # Extra warmup iterations (excluded).
            for _ in range(warmup):
                fa.all_reduce_reg(x_flat, out, open_fp8_quant=False)
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            for _ in range(iters):
                t0 = time.perf_counter()

                start_evt.record()
                fa.all_reduce_reg(x_flat, out, open_fp8_quant=False)
                end_evt.record()
                torch.cuda.synchronize()

                t1 = time.perf_counter()
                kernel_ms_list.append(float(start_evt.elapsed_time(end_evt)))
                e2e_ms_list.append(float((t1 - t0) * 1e3))
        else:
            # Non-profiling path: run once (keep correctness coverage without spending time).
            fa.all_reduce_reg(x_flat, out, open_fp8_quant=False)
            torch.cuda.synchronize()

        torch.cuda.synchronize()

        # Correctness (one-shot, out of band):
        # build a deterministic fp32 reference by gathering fp16 inputs once (not profiled).
        gathered = [torch.empty_like(x_flat) for _ in range(world_size)]
        dist.all_gather(gathered, x_flat, group=group)
        ref_f32 = torch.zeros_like(x_flat, dtype=torch.float32)
        for t in gathered:
            ref_f32 += t.to(torch.float32)
        # Kernel output is fp16/bf16; compare in fp32.
        max_err = (out.to(torch.float32) - ref_f32).abs().max().item()
        assert max_err < atol, f"[rank={rank}] max_err={max_err:.3e} >= atol={atol}"

        if profile:
            # Report timings (rank0 prints).
            avg_kernel_ms = sum(kernel_ms_list) / max(1, len(kernel_ms_list))
            avg_e2e_ms = sum(e2e_ms_list) / max(1, len(e2e_ms_list))
            max_kernel_ms = max(kernel_ms_list) if kernel_ms_list else 0.0
            max_e2e_ms = max(e2e_ms_list) if e2e_ms_list else 0.0

            stats = {
                "rank": int(rank),
                "dtype": str(dtype_str),
                "shape": tuple(int(x) for x in shape),
                "iters": int(iters),
                "warmup": int(warmup),
                "avg_kernel_ms": float(avg_kernel_ms),
                "avg_e2e_ms": float(avg_e2e_ms),
                "max_kernel_ms": float(max_kernel_ms),
                "max_e2e_ms": float(max_e2e_ms),
            }

            gathered_stats = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, stats, group=group)
            if rank == 0:
                # Summarize
                gathered_stats = sorted(gathered_stats, key=lambda x: x["rank"])
                print(
                    f"[custom_all_reduce profile] dtype={dtype_str} shape={shape} world_size={world_size} iters={iters} warmup={warmup}",
                    flush=True,
                )
                for s in gathered_stats:
                    print(
                        f"  - rank{s['rank']}: kernel_avg={s['avg_kernel_ms']:.3f}ms (max={s['max_kernel_ms']:.3f}ms) "
                        f"e2e_avg={s['avg_e2e_ms']:.3f}ms (max={s['max_e2e_ms']:.3f}ms)",
                        flush=True,
                    )
                max_rank_kernel = max(s["avg_kernel_ms"] for s in gathered_stats)
                max_rank_e2e = max(s["avg_e2e_ms"] for s in gathered_stats)
                mean_kernel = sum(s["avg_kernel_ms"] for s in gathered_stats) / world_size
                mean_e2e = sum(s["avg_e2e_ms"] for s in gathered_stats) / world_size
                print(
                    f"  => kernel_avg: mean={mean_kernel:.3f}ms max_rank={max_rank_kernel:.3f}ms; "
                    f"e2e_avg: mean={mean_e2e:.3f}ms max_rank={max_rank_e2e:.3f}ms",
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_test(N: int, dtype_str: str, *, world_size: int = 1):
    torch.manual_seed(0)

    # Create a shim handle (API-shape compatibility with the C++ extension).
    meta = torch.empty((meta_size(),), device="cuda", dtype=torch.int8)
    rank_data = torch.empty((1,), device="cuda", dtype=torch.int8)
    handles = [torch.empty((1,), device="cpu", dtype=torch.uint8) for _ in range(world_size)]
    offsets = [0 for _ in range(world_size)]
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=0, full_nvlink=False)

    if dtype_str == "f32":
        dtype = DTYPE_FP32
        atol = 1e-4
    elif dtype_str == "f16":
        dtype = DTYPE_FP16
        atol = 1e-3  # allreduce is elementwise sum/copy; should be tight
    elif dtype_str == "bf16":
        dtype = DTYPE_BF16
        atol = 1e-2
    else:
        raise ValueError(f"unsupported dtype_str: {dtype_str}")

    if world_size == 1:
        x = torch.randn((N,), device="cuda", dtype=dtype).contiguous()
        y = torch.empty((N,), device="cuda", dtype=dtype)
        fa.all_reduce_reg(x, y, open_fp8_quant=False)
        torch.cuda.synchronize()
        max_err = (y.to(torch.float32) - x.to(torch.float32)).abs().max().item()
        assert max_err < atol, f"max_err={max_err:.3e} >= atol={atol}"

        # all_gather (world_size==1) is identity
        yg = torch.empty((N,), device="cuda", dtype=dtype)
        fa.all_gather_reg(x, yg)
        torch.cuda.synchronize()
        max_err = (yg.to(torch.float32) - x.to(torch.float32)).abs().max().item()
        assert max_err < atol, f"[gather] max_err={max_err:.3e} >= atol={atol}"
    else:
        xs = [torch.randn((N,), device="cuda", dtype=dtype).contiguous() for _ in range(world_size)]
        y = torch.empty((N,), device="cuda", dtype=dtype)
        fa.all_reduce_reg(xs, y, open_fp8_quant=False)
        torch.cuda.synchronize()
        expected = torch.stack(xs, dim=0).to(torch.float32).sum(dim=0)
        max_err = (y.to(torch.float32) - expected).abs().max().item()
        assert max_err < atol, f"max_err={max_err:.3e} >= atol={atol}"

        yg = torch.empty((world_size, N), device="cuda", dtype=dtype)
        fa.all_gather_reg(xs, yg)
        torch.cuda.synchronize()
        expected_g = torch.stack(xs, dim=0).to(dtype)
        max_err = (yg.to(torch.float32) - expected_g.to(torch.float32)).abs().max().item()
        assert max_err < atol, f"[gather] max_err={max_err:.3e} >= atol={atol}"


def test_all():
    shapes_env = os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_SHAPES", "").strip()
    
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            parts = [x.strip() for x in p.split(",")]
            if len(parts) == 2:
                n_s, dt = parts
                ws = 1
            else:
                n_s, dt, ws_s = parts
                ws = int(ws_s)
            configs.append((int(n_s), dt, ws))
    else:
        # Default: run one case (can override via FLYDSL_CUSTOM_ALL_REDUCE_SHAPES)
        configs = [
            # allreduce requires N to be multiple of 16B pack: f16/bf16 -> 8 elems, f32 -> 4 elems
            (256 * 8 + 16, "f16", 1),
        ]

    for N, dtype, ws in configs:
        run_test(N, dtype_str=dtype, world_size=ws)


def test_distributed_allreduce_optional():
    """Distributed correctness test (multi-GPU, multi-process).

    Enable via:
      - FLYDSL_CUSTOM_ALL_REDUCE_DIST=1
    Control via:
      - FLYDSL_CUSTOM_ALL_REDUCE_DIST_WS (default 8)
      - FLYDSL_CUSTOM_ALL_REDUCE_DIST_SHAPE (default "128,8192")
      - FLYDSL_CUSTOM_ALL_REDUCE_DIST_DTYPE (default "f16")
      - FLYDSL_CUSTOM_ALL_REDUCE_DIST_GRAPH (default 1)
    """
    if os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_DIST", "0") != "1":
        pytest.skip("distributed test disabled (set FLYDSL_CUSTOM_ALL_REDUCE_DIST=1 to enable)")

    ws = int(os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_DIST_WS", "8"))
    shape_s = os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_DIST_SHAPE", "128,8192")
    shape = tuple(int(x.strip()) for x in shape_s.split(",") if x.strip())
    dtype_str = os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_DIST_DTYPE", "f16").strip()
    with_graph = os.environ.get("FLYDSL_CUSTOM_ALL_REDUCE_DIST_GRAPH", "1") == "1"

    ng = torch.cuda.device_count()
    if ng < ws:
        pytest.skip(f"need >= {ws} GPUs, got {ng}")

    # allreduce kernel requires pack alignment; enforce in test input.
    numel = 1
    for d in shape:
        numel *= d
    pack = 8 if dtype_str in {"f16", "bf16"} else 4
    if numel % pack != 0:
        pytest.skip(f"shape numel must be multiple of {pack} for dtype {dtype_str}")

    port = _free_port()
    import torch.multiprocessing as mp

    mp.spawn(
        _dist_worker,
        args=(ws, shape, dtype_str, with_graph, port),
        nprocs=ws,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    from multiprocessing import freeze_support, set_start_method

    freeze_support()
    # Align with AIter harness: use spawn to avoid fork+CUDA issues.
    set_start_method("spawn", force=True)

    # FlyDSL demo: default to fp16 only (bf16 distributed path is not stable yet).
    l_dtype = ["fp16"]
    l_shape = [(128, 8192)]

    parser = argparse.ArgumentParser(description="custom all-reduce test runner")
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32", "f16", "bf16", "f32"],
        nargs="?",
        const=None,
        default=None,
        help="data type (fp16/bf16/fp32)",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=_parse_tuple,
        nargs="?",
        const=None,
        default=None,
        help="shape. e.g. -s 128,8192",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=8,
        help="tensor parallel world size (distributed path)",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="pipeline parallel size (kept for CLI parity; unused in FlyDSL demo)",
    )
    parser.add_argument(
        "--withGraph",
        action="store_true",
        help="enable CUDA Graph replay (distributed path)",
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="run single-process smoke (no torch.distributed)",
    )

    args = parser.parse_args()

    import torch

    if args.dtype is None:
        dtype_list = l_dtype
    else:
        # If user explicitly requests bf16, fail fast with a clear message.
        if _normalize_dtype_arg(args.dtype) == "bf16":
            raise SystemExit("bf16 distributed mode is not supported/stable for FlyDSL custom_all_reduce yet (use -d fp16)")
        dtype_list = [args.dtype]

    if args.shape is None:
        shape_list = l_shape
    else:
        shape_list = [args.shape]

    # CLI mode mirrors AIter's script: default runs distributed allreduce with graph.
    if args.single_process:
        # Use existing single-process coverage (uses env override inside test_all()).
        test_all()
        raise SystemExit(0)

    ws = int(args.tp_size)
    with_graph = bool(args.withGraph)
    import torch.multiprocessing as mp

    ng = torch.cuda.device_count()
    if ng < ws:
        raise SystemExit(f"need >= {ws} GPUs for --tp_size {ws}, got {ng}")

    for dtype_arg in dtype_list:
        dtype_str = _normalize_dtype_arg(dtype_arg)
        for shape in shape_list:
            port = _free_port()
            mp.spawn(
                _dist_worker,
                args=(ws, shape, dtype_str, with_graph, port, True),
                nprocs=ws,
                join=True,
            )


