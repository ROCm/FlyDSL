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
_repo = Path(__file__).resolve().parents[3]
_embedded = _repo / "build" / "python_packages" / "flydsl"
if _embedded.exists():
    os.environ.setdefault("FLYDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
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


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _dist_worker(rank: int, world_size: int, shape, dtype_str: str, with_graph: bool, port: int):
    import torch
    import torch.distributed as dist

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
        atol = 1e-3
    elif dtype_str == "bf16":
        dtype = DTYPE_BF16
        atol = 1e-2
    else:
        raise ValueError(f"unsupported dtype_str: {dtype_str}")

    x = torch.randn(shape, device=device, dtype=dtype).contiguous()
    x_flat = x.reshape(-1)

    # Gather all ranks' inputs onto each rank (so we can exercise FlyDSL packed-input mode per-rank).
    gathered = [torch.empty_like(x_flat) for _ in range(world_size)]
    dist.all_gather(gathered, x_flat, group=group)
    stacked = torch.stack(gathered, dim=0).contiguous()

    # AIter-like handle shape (we don't use IPC handles in the FlyDSL demo).
    meta = torch.empty((meta_size(),), device=device, dtype=torch.int8)
    rank_data = torch.empty((1,), device=device, dtype=torch.int8)
    handles = [torch.empty((1,), device="cpu", dtype=torch.uint8) for _ in range(world_size)]
    offsets = [0 for _ in range(world_size)]
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank, full_nvlink=False)

    # Warmup: align all ranks.
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    out = torch.empty_like(x_flat)

    if with_graph:
        # Graph capture requires avoiding per-call allocations. We pass the pre-stacked tensor directly.
        graph = torch.cuda.CUDAGraph()
        out.fill_(0)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fa.all_reduce_reg(stacked, out, open_fp8_quant=False)
        graph.replay()
    else:
        fa.all_reduce_reg(stacked, out, open_fp8_quant=False)

    torch.cuda.synchronize()

    # Reference: sum across ranks.
    ref = stacked.to(torch.float32).sum(dim=0)
    max_err = (out.to(torch.float32) - ref).abs().max().item()
    assert max_err < atol, f"[rank={rank}] max_err={max_err:.3e} >= atol={atol}"

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
    test_all()


