#!/usr/bin/env python3
"""Batched WMMA GEMM (BMM) tests for gfx1250.

Kernel: C[b,m,n] = A[b,m,k] @ B[b,k,n]
V4 use case: einsum("sgd,grd->sgr", o, wo_a)
  B=G=16, M=T, K=D=4096, N=R=1024
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

import flydsl  # noqa: E402,F401

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.bmm_gfx1250 import compile_bmm_gfx1250
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _pad_tensor(tensor: torch.Tensor, target_shape: tuple, fill_value: float = 0.0) -> torch.Tensor:
    if tensor.shape == target_shape:
        return tensor
    padded = torch.full(target_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, s) for s in tensor.shape)
    padded[slices] = tensor
    return padded


def _bmm_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference: C[b,m,n] = A[b,m,k] @ B[b,k,n] in fp32, returned as bf16."""
    return torch.bmm(a.float(), b.float()).bfloat16()


def run_bmm_test(
    B, M, N, K,
    tile_m, tile_n, tile_k,
    in_dtype="bf16",
    out_dtype=None,
    num_buffers=3,
    m_warp=2, n_warp=4,
    l2_prefetch_distance=2,
    use_tdm_store=True,
    waves_per_eu=None,
    atol=3e-2, rtol=3e-2,
    bench=False, bench_iters=100,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"BMM kernel requires gfx1250, got {arch}")

    mpad = _align_up(M, tile_m)
    if N % tile_n != 0:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n}")
    if K % tile_k != 0:
        pytest.skip(f"K={K} must be divisible by tile_k={tile_k}")
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        pytest.skip(f"{num_buffers}-buffer requires num_k_tiles >= {num_buffers}, got {num_k_tiles}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    _eff_out = out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[_eff_out]

    torch.manual_seed(42)
    a = torch.randn((B, M, K), dtype=torch_dtype, device="cpu")
    b = torch.randn((B, K, N), dtype=torch_dtype, device="cpu")
    ref = _bmm_reference(a, b)

    a_pad = _pad_tensor(a, (B, mpad, K)).cuda()
    b_gpu = b.cuda()
    c_pad = torch.zeros((B, mpad, N), dtype=_out_torch, device="cuda")

    print(
        f"BMM B={B} M={M}(pad={mpad}) K={K} N={N} "
        f"tile={tile_m}x{tile_n}x{tile_k} bufs={num_buffers} "
        f"dtype={in_dtype} out={_eff_out} tdm_store={use_tdm_store}"
    )

    launch_fn = compile_bmm_gfx1250(
        B=B, M=mpad, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        in_dtype=in_dtype, out_dtype=out_dtype,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
    )

    a_flat = a_pad.contiguous().view(-1)
    b_flat = b_gpu.contiguous().view(-1)
    c_flat = c_pad.view(-1)

    launch_fn(c_flat, a_flat, b_flat, mpad, torch.cuda.current_stream())
    torch.cuda.synchronize()

    c_out = c_pad[:, :M, :].cpu().float()
    assert verify_output(c_out, ref.float(), rtol=rtol, atol=atol), (
        f"Precision check FAILED: max_diff={( c_out - ref.float()).abs().max():.4f}"
    )
    print("  PASSED")

    if bench:
        for _ in range(10):
            launch_fn(c_flat, a_flat, b_flat, mpad, torch.cuda.current_stream())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            launch_fn(c_flat, a_flat, b_flat, mpad, torch.cuda.current_stream())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / bench_iters * 1e6
        flops = 2 * B * M * N * K
        tflops = flops / (us * 1e-6) / 1e12
        mem_bytes = (B * M * K * 2 + B * K * N * 2 + B * M * N * 2)  # bf16
        bw_tbs = mem_bytes / (us * 1e-6) / 1e12
        print(f"  bench: {us:.2f} µs  {tflops:.2f} TFLOPS  {bw_tbs:.2f} TB/s")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_dtype", ["bf16"])
@pytest.mark.parametrize("M", [1, 32, 64, 128, 256])
def test_bmm_v4_decode_prefill(in_dtype, M):
    """V4 shapes: B=16, K=4096, N=1024, T=M."""
    run_bmm_test(
        B=16, M=M, N=1024, K=4096,
        tile_m=64, tile_n=128, tile_k=128,
        in_dtype=in_dtype,
        num_buffers=2,
        m_warp=2, n_warp=4,
        use_tdm_store=True,
    )


@pytest.mark.parametrize("in_dtype", ["bf16"])
@pytest.mark.parametrize("M", [512, 1024, 4096])
def test_bmm_v4_large(in_dtype, M):
    """V4 shapes, larger M (prefill regime)."""
    run_bmm_test(
        B=16, M=M, N=1024, K=4096,
        tile_m=128, tile_n=128, tile_k=128,
        in_dtype=in_dtype,
        num_buffers=3,
        m_warp=2, n_warp=4,
        use_tdm_store=True,
    )


@pytest.mark.parametrize(
    "B, M, N, K, tile_m, tile_n, tile_k",
    [
        (4,  64,  128, 256,  64, 128, 128),
        (8,  128, 128, 512,  64, 128, 128),
        (16, 256, 128, 512, 128, 128, 128),
    ],
)
def test_bmm_small_shapes(B, M, N, K, tile_m, tile_n, tile_k):
    """Smaller shapes for quick smoke test."""
    run_bmm_test(
        B=B, M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype="bf16",
        num_buffers=2,
        m_warp=2, n_warp=4,
        use_tdm_store=True,
    )


@pytest.mark.parametrize("use_tdm_store", [True, False])
def test_bmm_epilogue_variants(use_tdm_store):
    """Test both TDM store and buffer_store epilogue."""
    run_bmm_test(
        B=16, M=128, N=128, K=256,
        tile_m=64, tile_n=128, tile_k=128,
        in_dtype="bf16",
        num_buffers=2,
        m_warp=2, n_warp=4,
        use_tdm_store=use_tdm_store,
    )


@pytest.mark.parametrize("num_buffers", [2, 3])
def test_bmm_pipeline_depths(num_buffers):
    """Test double and triple buffer pipelining."""
    run_bmm_test(
        B=16, M=256, N=128, K=512,
        tile_m=128, tile_n=128, tile_k=128,
        in_dtype="bf16",
        num_buffers=num_buffers,
        m_warp=2, n_warp=4,
        use_tdm_store=True,
    )


# ---------------------------------------------------------------------------
# CLI for benchmark mode
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="BMM gfx1250 benchmark")
    parser.add_argument("-B", type=int, default=16)
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num-buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--l2-prefetch-distance", type=int, default=2)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--bench", action="store_true", default=False)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Sweep T over V4 decode/prefill range")
    return parser


def _sweep_v4(args):
    """Sweep T over [1, 8, 32, 64, 128, 256, 512, 1024, 4096, 8192] for V4 config."""
    t_list = [1, 8, 32, 64, 128, 256, 512, 1024, 4096, 8192]
    for T in t_list:
        tile_m = 64 if T <= 64 else 128
        num_bufs = 2 if T <= 64 else 3
        run_bmm_test(
            B=16, M=T, N=1024, K=4096,
            tile_m=tile_m, tile_n=128, tile_k=128,
            in_dtype="bf16", num_buffers=num_bufs,
            m_warp=2, n_warp=4, use_tdm_store=True,
            bench=True, bench_iters=args.bench_iters,
        )


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.sweep:
        _sweep_v4(args)
    else:
        run_bmm_test(
            B=args.B, M=args.M, N=args.N, K=args.K,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            m_warp=args.m_warp, n_warp=args.n_warp,
            in_dtype=args.dtype,
            num_buffers=args.num_buffers,
            l2_prefetch_distance=args.l2_prefetch_distance,
            use_tdm_store=not args.no_tdm_store,
            bench=args.bench, bench_iters=args.bench_iters,
        )
