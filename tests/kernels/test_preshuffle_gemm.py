#!/usr/bin/env python3
"""MFMA FP8/INT8/BF16 GEMM Test with B preshuffle — @flyc.kernel API.

Kernel implementation lives in `kernels/preshuffle_gemm_flyc.py`.
This file is the correctness + perf harness.
"""

import os
import sys
import logging

import torch
import torch.nn.functional as F
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from kernels.preshuffle_gemm_flyc import compile_preshuffle_gemm_a8 as compile_preshuffle_gemm_a8_flyc
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant, shuffle_weight
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)


try:
    import aiter
    HAS_AITER = True
except ImportError:
    HAS_AITER = False

arch = str(get_rocm_arch())
if "gfx95" in arch:
    DTYPE_FP8 = torch.float8_e4m3fn
else:
    DTYPE_FP8 = torch.float8_e4m3fnuz

DEFAULT_LDS_STAGE = 2
DEFAULT_BENCH_ITERS = 100
DEFAULT_BENCH_WARMUP = 10
DEFAULT_RUN_AITER_BENCH = True


def run_torch(a, b, scale_a, scale_b, bias=None, dtype=torch.float32):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    if bias is not None:
        c = c + bias
    return c.to(dtype)


@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (16, 5120, 8192, 16, 64, 512),
        (5120, 5120, 8320, 64, 256, 128),
        (9728, 8192, 8320, 128, 128, 128),
        (5133, 5120, 8320, 64, 256, 128),
    ]
)
def test_mfma_a8_flyc_preshuffle(
    in_dtype,
    M, N, K,
    tile_m, tile_n, tile_k,
    *,
    lds_stage: int = DEFAULT_LDS_STAGE,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
):
    """Preshuffle GEMM using the @flyc.kernel / @flyc.jit API."""
    print("=" * 80)
    print(
        f"[flyc] MFMA {in_dtype.upper()} GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})"
    )
    print("=" * 80)

    lds_stage = int(lds_stage)
    if lds_stage not in (1, 2):
        raise ValueError(f"lds_stage must be 1 or 2, got {lds_stage!r}")

    launch_fn = compile_preshuffle_gemm_a8_flyc(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype,
        lds_stage=lds_stage,
    )
    print(f"✓ Kernel prepared (lds_stage={lds_stage})")

    size_c = M * N
    size_a = M * K
    if in_dtype == "int4":
        size_b = (N * K) // 2
        elem_bytes = 1
    elif in_dtype in ("fp16", "bf16"):
        size_b = (N * K) * 2
        elem_bytes = 2
    else:
        size_b = (N * K)
        elem_bytes = 1

    device = torch.device("cuda")
    torch.manual_seed(42)
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)

    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4

    if in_dtype in ("fp16", "bf16"):
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q = a_fp32.to(torch_dtype)
        b_q = b_fp32_t.to(torch_dtype)
        scale_a = None
        scale_b = None
    else:
        quant_dtype = torch.int8 if is_int8 else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)
        if is_int4:
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.int8, dtypeMax=7)
        else:
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8):
        flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
        assert flat.numel() % 8 == 0
        u = (flat & 0xF).to(torch.uint8).view(-1, 8)
        out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
        out[:, 0] = u[:, 0] | (u[:, 4] << 4)
        out[:, 1] = u[:, 1] | (u[:, 5] << 4)
        out[:, 2] = u[:, 2] | (u[:, 6] << 4)
        out[:, 3] = u[:, 3] | (u[:, 7] << 4)
        return out.view(-1).to(torch.int8)

    b_packed = None
    if is_int4:
        b_packed = _pack_shuffled_int8_to_packed_int4_no_perm(b_shuffled)

    c_ref = run_torch(a_q, b_q, scale_a, scale_b, bias=None, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    b_input = b_packed if is_int4 else b_shuffled
    if scale_a is None:
        sa_flat = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        sa_flat = scale_a.contiguous().view(-1)
    if scale_b is None:
        sb_flat = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        sb_flat = scale_b.contiguous().view(-1)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def launch_kernel(c, a, b, sa, sb):
        launch_fn(
            c.contiguous().view(-1),
            _as_i8(a.contiguous().view(-1)),
            _as_i8(b.contiguous().view(-1)),
            sa.contiguous().view(-1) if sa.numel() > 0 else sa,
            sb.contiguous().view(-1) if sb.numel() > 0 else sb,
            M, N,
        )

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        c_out_raw,
        a_q,
        b_input,
        sa_flat,
        sb_flat,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = (size_a * elem_bytes) + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preshuffle GEMM benchmark")
    parser.add_argument("--in_dtype", type=str, default="fp8",
                        choices=["fp8", "int8", "int4", "fp16", "bf16"])
    parser.add_argument("-M", type=int, default=16)
    parser.add_argument("-N", type=int, default=10240)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=16)
    parser.add_argument("--tile_n", type=int, default=64)
    parser.add_argument("--tile_k", type=int, default=256)
    parser.add_argument("--lds_stage", type=int, default=DEFAULT_LDS_STAGE, choices=[1, 2])
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--flyc", action="store_true", default=True)

    args = parser.parse_args()
    torch.set_default_device("cuda")
    test_mfma_a8_flyc_preshuffle(
        args.in_dtype,
        M=args.M, N=args.N, K=args.K,
        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        lds_stage=args.lds_stage,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
    )
