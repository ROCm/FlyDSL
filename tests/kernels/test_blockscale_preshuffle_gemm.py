#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Blockscale Preshuffle GEMM Test (FP8 A8W8, per-block scales)."""

import os
import sys
import logging
import flydsl.compiler as flyc

import torch
import torch.nn.functional as F
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch as get_hip_arch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.blockscale_preshuffle_gemm import compile_blockscale_preshuffle_gemm
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

BLOCK_SHAPE = (128, 128)  # (block_n, block_k)

try:
    import aiter
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight

    HAS_AITER = True
except Exception:
    HAS_AITER = False

def select_tile_config(M: int, N: int, K: int, scale_block_k: int = 128):
    """Auto-select tile config for blockscale GEMM benchmarks."""
    candidates = [
        (16, 64, 256), (16, 128, 256),
        (32, 64, 128), (32, 64, 256), (32, 128, 128), (32, 128, 256),
        (64, 64, 128), (64, 64, 256), (64, 128, 128), (64, 128, 256), (64, 256, 128),
    ]
    def _valid(tm, tn, tk):
        return (N % tn == 0 and K % tk == 0 and tk % scale_block_k == 0
                and tm * tk // 256 >= 16)
    valid = [(tm, tn, tk) for tm, tn, tk in candidates if _valid(tm, tn, tk)]
    if not valid:
        return (64, 128, 128)
    def _score(tm, tn, tk):
        s = 0
        total_blocks = ((M + tm - 1) // tm) * (N // tn)
        s += 15 if total_blocks >= 256 else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if M <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        else:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        s += 6 if tk == 128 else 3
        return s
    return max(valid, key=lambda t: _score(*t))


def run_torch_blockscale(x, weight, x_scale, w_scale, block_shape=BLOCK_SHAPE,
                         dtype=torch.bfloat16):
    """Torch reference for blockscale GEMM."""
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x_f32 = x.to(x_scale.dtype).view(m, k // block_shape_k, block_shape_k) * x_scale.unsqueeze(-1)
    x_f32 = x_f32.view(m, k)

    w_scale_expanded = (
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    w_scale_expanded = w_scale_expanded[:n, :k]
    weight_f32 = weight.to(w_scale_expanded.dtype) * w_scale_expanded

    out = F.linear(x_f32.to(torch.float32), weight_f32.to(torch.float32))
    return out.to(dtype)

@pytest.mark.parametrize(
    "M, N, K",
    [
        # Original shapes
        # (128, 1536, 7168),
        # (256, 3072, 1536),
        # (512, 4608, 7168),
        # (1024, 7168, 2048),
        # (2048, 7168, 2304),
        # (4096, 1536, 7168),
        # DeepSeek V3 shapes x different M (including M < tile_m and M % tile_m != 0)
        pytest.param(1, 7168, 2304, id="DS-7168x2304-M1"),
        pytest.param(7, 7168, 2304, id="DS-7168x2304-M7"),
        pytest.param(16, 7168, 2304, id="DS-7168x2304-M16"),
        pytest.param(33, 7168, 2304, id="DS-7168x2304-M33"),
        pytest.param(64, 7168, 2304, id="DS-7168x2304-M64"),
        pytest.param(256, 7168, 2304, id="DS-7168x2304-M256", marks=pytest.mark.large_shape),
        pytest.param(1024, 7168, 2304, id="DS-7168x2304-M1024", marks=pytest.mark.large_shape),
        pytest.param(1, 2112, 7168, id="DS-2112x7168-M1"),
        pytest.param(7, 2112, 7168, id="DS-2112x7168-M7"),
        pytest.param(16, 2112, 7168, id="DS-2112x7168-M16"),
        pytest.param(33, 2112, 7168, id="DS-2112x7168-M33"),
        pytest.param(64, 2112, 7168, id="DS-2112x7168-M64"),
        pytest.param(256, 2112, 7168, id="DS-2112x7168-M256", marks=pytest.mark.large_shape),
        pytest.param(1024, 2112, 7168, id="DS-2112x7168-M1024", marks=pytest.mark.large_shape),
        pytest.param(1, 3072, 1536, id="DS-3072x1536-M1"),
        pytest.param(7, 3072, 1536, id="DS-3072x1536-M7"),
        pytest.param(16, 3072, 1536, id="DS-3072x1536-M16"),
        pytest.param(33, 3072, 1536, id="DS-3072x1536-M33"),
        pytest.param(64, 3072, 1536, id="DS-3072x1536-M64"),
        pytest.param(256, 3072, 1536, id="DS-3072x1536-M256", marks=pytest.mark.large_shape),
        pytest.param(1024, 3072, 1536, id="DS-3072x1536-M1024", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
    pytest.param(True, id="graph"),
])
def test_blockscale_preshuffle_gemm(
    M, N, K, out_dtype, test_graph,
    *,
    bench_iters=20,
    bench_warmup=3,
    use_async_copy=False,
):

    block_shape_n, block_shape_k = BLOCK_SHAPE
    scale_k = (K + block_shape_k - 1) // block_shape_k
    scale_n = (N + block_shape_n - 1) // block_shape_n

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    tile_m, tile_n, tile_k = select_tile_config(M, N, K, scale_block_k=block_shape_k)

    print("=" * 80)
    print(
        f"Blockscale GEMM Test: M={M}, N={N}, K={K}, "
        f"tile=({tile_m}x{tile_n}x{tile_k}), out={out_dtype}, async={use_async_copy}"
    )
    print("=" * 80)

    exe = compile_blockscale_preshuffle_gemm(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=block_shape_k,
        out_dtype=out_dtype,
        use_async_copy=bool(use_async_copy),
    )
    print(f"  Compiled OK")

    device = torch.device("cuda")

    x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)
    weight = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)

    c_ref = run_torch_blockscale(x, weight, x_scale, w_scale, dtype=torch.float32)

    b_shuffled = shuffle_weight(weight, layout=(16, 16))

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(-1)

    w_scale_flat = w_scale.contiguous().view(-1)

    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)

    compiled_exe = flyc.compile(exe, c_out, x, b_shuffled, x_scale_t, w_scale_flat,
                               M, N, torch.cuda.current_stream())

    def launch_kernel(c, a, b, sa, sb):
        compiled_exe(c, a, b, sa, sb, M, N, torch.cuda.current_stream())

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        c_out, x, b_shuffled, x_scale_t, w_scale_flat,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()
    c_out_f32 = c_out.to(torch.float32)

    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=0.01)

    flops = 2 * M * N * K
    elem_bytes = 1  # fp8
    bytes_moved = (M * K * elem_bytes) + (N * K * elem_bytes) + (M * N * 2) + (M * scale_k + scale_n * scale_k) * 4
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"  Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s"
    )

    if HAS_AITER:
        try:
            def launch_aiter(a, b, sa, sb):
                return aiter.gemm_a8w8_blockscale_bpreshuffle(
                    a, b, sa, sb, torch_out_dtype
                )

            x_scale_t_2d = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
            b_aiter = aiter_shuffle_weight(weight, layout=(16, 16))
            c_aiter, us_aiter = run_perftest(
                launch_aiter, x, b_aiter, x_scale_t_2d, w_scale,
            )
            c_aiter_f32 = c_aiter.to(torch.float32)
            verify_output(c_aiter_f32, c_ref, rtol=1e-2, atol=0.01)
            tflops_aiter = flops / (us_aiter / 1e6) / 1e12
            print(
                f"  Aiter: {us_aiter:.1f} us, {tflops_aiter:.2f} TFLOPS, "
                f"speedup: {us_aiter/us:.2f}x"
            )
        except (Exception, SystemExit) as e:
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            print(f"  Aiter comparison skipped: {msg}")

    assert passed, "Kernel output verification failed"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blockscale Preshuffle GEMM benchmark")
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=1536)
    parser.add_argument("-K", type=int, default=7168)
    parser.add_argument("--tile_m", type=int, default=None)
    parser.add_argument("--tile_n", type=int, default=None)
    parser.add_argument("--tile_k", type=int, default=None)
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--use_async_copy", action="store_true", default=False)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    if args.tile_m is None or args.tile_n is None or args.tile_k is None:
        auto_tm, auto_tn, auto_tk = select_tile_config(args.M, args.N, args.K)
        tile_m = args.tile_m or auto_tm
        tile_n = args.tile_n or auto_tn
        tile_k = args.tile_k or auto_tk
    else:
        tile_m, tile_n, tile_k = args.tile_m, args.tile_n, args.tile_k
    print(f"Using tile config: {tile_m}x{tile_n}x{tile_k}")
    test_blockscale_preshuffle_gemm(
        M=args.M, N=args.N, K=args.K,
        out_dtype=args.out_dtype,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
        use_async_copy=args.use_async_copy,
    )
