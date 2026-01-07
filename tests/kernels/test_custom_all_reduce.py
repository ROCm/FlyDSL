#!/usr/bin/env python3
"""Custom all-reduce (block sum + broadcast) kernel tests."""

import os
import sys
from pathlib import Path

# Prefer embedded MLIR/rocdsl to avoid mixing multiple runtimes.
_repo = Path(__file__).resolve().parents[3]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
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
    shapes_env = os.environ.get("ROCDSL_CUSTOM_ALL_REDUCE_SHAPES", "").strip()
    
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
        # Default: run one case (can override via ROCDSL_CUSTOM_ALL_REDUCE_SHAPES)
        configs = [
            # allreduce requires N to be multiple of 16B pack: f16/bf16 -> 8 elems, f32 -> 4 elems
            (256 * 8 + 16, "f16", 1),
        ]

    for N, dtype, ws in configs:
        run_test(N, dtype_str=dtype, world_size=ws)


if __name__ == "__main__":
    test_all()


