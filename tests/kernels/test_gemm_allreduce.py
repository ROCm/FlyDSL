#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_WORK_ROOT = os.path.dirname(_REPO_ROOT)
_FORCED_BUILD_PKG = os.environ.get("FLYDSL_BUILD_PYTHON_PACKAGE", "")
_PYTHON_CANDIDATES = [
    _FORCED_BUILD_PKG,
    _REPO_ROOT,
    _WORK_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if not _FORCED_BUILD_PKG:
    _blocked = "/build-fly/python_packages"
    sys.path[:] = [p for p in sys.path if _blocked not in str(p)]

# Avoid deleting/re-importing flydsl when pytest/conftest has already imported
# MLIR bindings; double initialization can abort the process.
_flydsl_mod = sys.modules.get("flydsl")
if _flydsl_mod is not None and _FORCED_BUILD_PKG:
    _loaded_path = str(getattr(_flydsl_mod, "__file__", ""))
    if _loaded_path and _FORCED_BUILD_PKG not in _loaded_path:
        pytest.skip(
            "flydsl already loaded from non-build path; set a clean env for integration test",
            allow_module_level=True,
        )

from kernels.gemm_allreduce import GemmAllReduceConfig, build_gemm_allreduce_operator
from kernels.gemm_allreduce import make_flydsl_allreduce_fn
from tests.test_common import verify_output
from tests.utils import shuffle_weight


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _make_inputs(m: int, n: int, k: int, *, dtype=torch.bfloat16):
    torch.manual_seed(1234)
    a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    b = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    b_shuffled = shuffle_weight(b, layout=(16, 16))
    return a, b, b_shuffled


@pytest.mark.parametrize("m,n,k,tile_m,tile_n,tile_k", [(32, 64, 128, 32, 64, 128)])
def test_gemm_allreduce_identity(m, n, k, tile_m, tile_n, tile_k):
    cfg = GemmAllReduceConfig(
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype="bf16",
        out_dtype="bf16",
    )
    op = build_gemm_allreduce_operator(config=cfg)

    a, b_ref, b_shuffled = _make_inputs(m, n, k, dtype=torch.bfloat16)
    out = op(a, b_shuffled)
    torch.cuda.synchronize()

    ref = (a.float() @ b_ref.float().T)
    assert verify_output(out.float(), ref, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("m,n,k,tile_m,tile_n,tile_k", [(32, 64, 128, 32, 64, 128)])
def test_gemm_allreduce_mock_sum2(m, n, k, tile_m, tile_n, tile_k):
    def mock_allreduce(inp: torch.Tensor, out: torch.Tensor | None, stream_ptr: int | None) -> torch.Tensor:
        _ = stream_ptr
        if out is None:
            return inp * 2.0
        out.copy_(inp * 2.0)
        return out

    cfg = GemmAllReduceConfig(
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype="bf16",
        out_dtype="bf16",
    )
    op = build_gemm_allreduce_operator(config=cfg, allreduce_fn=mock_allreduce)

    a, b_ref, b_shuffled = _make_inputs(m, n, k, dtype=torch.bfloat16)
    out_buf = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out = op(a, b_shuffled, out=out_buf)
    torch.cuda.synchronize()

    assert out.data_ptr() == out_buf.data_ptr()
    ref = (a.float() @ b_ref.float().T) * 2.0
    assert verify_output(out.float(), ref, atol=0.1, rtol=0.1)


def _get_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _dist_gemm_allreduce_worker(rank: int, world_size: int, port: int, m: int, n: int, k: int):
    import importlib.util
    import types

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["FLYDSL_AITER_IMPL"] = "flydsl"

    # Root-level custom_all_reduce.py imports `kernels.custom_all_reduce_kernel`.
    # Load `/home/xzhu/custom_all_reduce_kernel.py` and alias it to that module path.
    kernel_src = os.path.join(_WORK_ROOT, "custom_all_reduce_kernel.py")
    spec = importlib.util.spec_from_file_location("kernels.custom_all_reduce_kernel", kernel_src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load custom_all_reduce_kernel from {kernel_src}")
    _car_k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_car_k)

    if "kernels" not in sys.modules:
        sys.modules["kernels"] = types.ModuleType("kernels")
    sys.modules["kernels.custom_all_reduce_kernel"] = _car_k

    from custom_all_reduce import init_custom_ar

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(f"cuda:{rank}")

        # Minimal inputs required by init_custom_ar; for flydsl impl, handles/meta
        # are not consumed as in aiter API but shape/length checks are enforced.
        meta = torch.empty(1, dtype=torch.uint8, device=device)
        rank_data = torch.empty(1, dtype=torch.uint8, device=device)
        handles = [0] * world_size
        offsets = list(range(world_size))
        ar = init_custom_ar(
            meta=meta,
            rank_data=rank_data,
            handles=handles,
            offsets=offsets,
            rank=rank,
            full_nvlink=True,
            out=None,
        )

        cfg = GemmAllReduceConfig(
            K=k,
            tile_m=32,
            tile_n=64,
            tile_k=128,
            in_dtype="bf16",
            out_dtype="bf16",
        )
        op = build_gemm_allreduce_operator(
            config=cfg,
            allreduce_fn=make_flydsl_allreduce_fn(ar),
        )

        torch.manual_seed(2026)
        a0 = (torch.randn(m, k, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b = (torch.randn(n, k, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        b_shuffled = shuffle_weight(b, layout=(16, 16))
        a = a0 * float(rank + 1)

        out = op(a, b_shuffled)
        torch.cuda.synchronize(device)

        base = a0.float() @ b.float().T
        coeff = float(world_size * (world_size + 1) // 2)
        ref = base * coeff
        if not verify_output(out.float(), ref, atol=0.2, rtol=0.2):
            raise AssertionError(f"rank={rank} gemm_allreduce output mismatch")
    finally:
        try:
            if "ar" in locals() and hasattr(ar, "close"):
                ar.close()
        except Exception:
            pass
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_gemm_allreduce_flydsl_integration_dist2():
    import importlib.util

    if str(os.environ.get("RUN_FLYDSL_GEMM_AR_INTEGRATION", "0")).lower() not in {"1", "true", "yes"}:
        pytest.skip("set RUN_FLYDSL_GEMM_AR_INTEGRATION=1 to run distributed integration test")
    if importlib.util.find_spec("custom_all_reduce") is None:
        pytest.skip("custom_all_reduce.py not importable in current environment")

    world_size = 2
    port = _get_free_port()
    mp.spawn(
        _dist_gemm_allreduce_worker,
        args=(world_size, port, 32, 64, 128),
        nprocs=world_size,
        join=True,
    )

