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

import flydsl

from kernels.custom_all_reduce import build_custom_all_reduce_module


def _reference_block_sum_broadcast(x_fp32: "torch.Tensor", *, block_size: int) -> "torch.Tensor":
    assert x_fp32.dtype == torch.float32
    n = x_fp32.numel()
    y = torch.empty((n,), device=x_fp32.device, dtype=torch.float32)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        s = x_fp32[start:end].sum()
        y[start:end] = s
    return y


def run_test(N: int, dtype_str: str, *, block_size: int = 256):
    m = build_custom_all_reduce_module(N, dtype_str=dtype_str, BLOCK_SIZE=block_size)
    exe = flydsl.compile(m)

    torch.manual_seed(0)
    x0 = torch.randn((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype_str == "f32":
        x = x0.contiguous()
        y = torch.empty((N,), device="cuda", dtype=DTYPE_FP32)
        atol = 1e-4
    elif dtype_str == "f16":
        x = x0.to(DTYPE_FP16).contiguous()
        y = torch.empty((N,), device="cuda", dtype=DTYPE_FP16)
        atol = 5e-2
    elif dtype_str == "bf16":
        x = x0.to(DTYPE_BF16).contiguous()
        y = torch.empty((N,), device="cuda", dtype=DTYPE_BF16)
        atol = 1e-1
    else:
        raise ValueError(f"unsupported dtype_str: {dtype_str}")

    exe(x, y)
    torch.cuda.synchronize()

    expected = _reference_block_sum_broadcast(x.to(torch.float32), block_size=block_size)
    res = y.to(torch.float32)

    max_err = (res - expected).abs().max().item()
    assert max_err < atol, f"max_err={max_err:.3e} >= atol={atol} (N={N}, dtype={dtype_str})"


def test_all():
    shapes_env = os.environ.get("ROCDSL_CUSTOM_ALL_REDUCE_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(n_s), dt))
    else:
        # Default: run one case (can override via ROCDSL_CUSTOM_ALL_REDUCE_SHAPES)
        configs = [
            (256 * 8 + 13, "f16"),
        ]

    for N, dtype in configs:
        run_test(N, dtype_str=dtype, block_size=256)


if __name__ == "__main__":
    test_all()


