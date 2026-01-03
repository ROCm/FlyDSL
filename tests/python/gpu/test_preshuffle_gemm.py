#!/usr/bin/env python3
"""MFMA FP8/INT8 GEMM Test using flir with B preshuffle.

NOTE:
- Kernel implementation (IR builder) lives in `samples/preshuffle_gemm.py`
  (same structure as `samples/moe_gemm_2stage.py`).
- This file is only the correctness + perf harness.
"""

import os
import logging

import torch
import torch.nn.functional as F
import pytest

from samples.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant, shuffle_weight


logging.basicConfig(level=logging.INFO)


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


# Aiter imports (optional)
try:
    import aiter

    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False


RUN_AITER_BENCH = os.environ.get("COMPARE_AITER_CK", "1") == "1"


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """Torch reference implementation (from aiter project).

    Dequantize 8-bit inputs (FP8/INT8) and compute FP32 matmul.
    """
    x = x.to(torch.float32) * x_scale
    weight = weight.to(torch.float32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)


@pytest.mark.parametrize("in_dtype", ["fp8", "int8"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k", [(1024, 7168, 2048, 128, 128, 128)]
)
def test_mfma_a8_flir_preshuffle(in_dtype, M, N, K, tile_m, tile_n, tile_k):
    print("=" * 80)
    print(
        f"MFMA {in_dtype.upper()} GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]"
    )
    print("=" * 80)

    # Select LDS staging via env var:
    #   - FLIR_PRESHUFFLE_GEMM_LDS_STAGE=2 : ping-pong LDS (2 buffers)  [default]
    #   - FLIR_PRESHUFFLE_GEMM_LDS_STAGE=1 : single LDS buffer (CK intrawave bpreshuffle v1 spirit)
    lds_stage = int(os.environ.get("FLIR_PRESHUFFLE_GEMM_LDS_STAGE", '2'))

    if lds_stage not in (1, 2):
        raise ValueError(
            f"FLIR_PRESHUFFLE_GEMM_LDS_STAGE must be 1 or 2, got {lds_stage!r}"
        )
    exe = compile_preshuffle_gemm_a8(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        lds_stage=lds_stage,
    )
    print(f"✓ Compiled (lds_stage={lds_stage})")

    size_c = M * N
    size_a = M * K
    size_b = N * K

    device = torch.device("cuda")

    torch.manual_seed(42)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)  # (N, K)

    is_int8 = in_dtype == "int8"
    quant_dtype = torch.int8 if is_int8 else torch.float8_e4m3fnuz
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)  # (M, K)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)  # (N, K)

    # When using fixed-width global loads (16B chunks), some threads can over-read.
    # Pad the underlying storage so the over-read stays in-bounds.
    PAD_ELEMS = 64  # bytes for 8-bit; generous guard for safety
    a_flat = a_q.contiguous().view(-1)
    a_storage = torch.empty(a_flat.numel() + PAD_ELEMS, device=device, dtype=a_q.dtype)
    a_storage[: a_flat.numel()] = a_flat
    a_q = a_storage[: a_flat.numel()].view(M, K)

    b_flat = b_q.contiguous().view(-1)
    b_storage = torch.empty(b_flat.numel() + PAD_ELEMS, device=device, dtype=b_q.dtype)
    b_storage[: b_flat.numel()] = b_flat
    b_q = b_storage[: b_flat.numel()].view(N, K)

    # Preshuffle B to CK/aiter layout.
    b_shuffled = shuffle_weight(b_q)

    # Reference (dequant + matmul).
    c_ref = run_torch(a_q, b_q, scale_a, scale_b, bias=None, dtype=torch.float32)

    # Run kernel (f16 output, in-kernel scaling).
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    def launch_kernel(c, a, b, sa, sb):
        exe(c, a, b, sa, sb, M, N, K)

    # `run_perftest` requires num_iters > 1.
    bench_iters = max(2, int(os.environ.get("pyflir_BENCH_ITERS", "20")))
    bench_warmup = int(os.environ.get("pyflir_BENCH_WARMUP", "3"))
    _, us = run_perftest(
        launch_kernel,
        c_out_raw,
        a_q,
        b_shuffled,
        scale_a,
        scale_b,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    if HAS_AITER and RUN_AITER_BENCH:
        print("-" * 40)
        print("Running Aiter Benchmark...")
        try:
            def launch_aiter(a, b, sa, sb):
                return aiter.gemm_a8w8_bpreshuffle(a, b, sa, sb, None, torch.float16)

            c_aiter, us1 = run_perftest(launch_aiter, a_q, b_shuffled, scale_a, scale_b)
            verify_output(c_aiter.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

            tflops_aiter = flops / (us1 / 1e6) / 1e12
            bw_aiter = bytes_moved / 1e9 / (us1 / 1e6)
            print(
                f"Aiter Throughput: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s"
            )
            print(
                f"Speedup vs Aiter: {tflops / tflops_aiter:.2f}x, Tflops {tflops:.1f} vs {tflops_aiter:.1f}"
            )
            print("-" * 40)
        except Exception as e:
            # Best-effort only: aiter can be importable but fail to load its JIT .so deps.
            print(f"Skipping Aiter benchmark (not runnable here): {e}")
            print("-" * 40)
    elif HAS_AITER and not RUN_AITER_BENCH:
        print("-" * 40)
        print("Skipping Aiter benchmark (set pyflir_RUN_AITER_BENCH=1 to enable)")
        print("-" * 40)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    print("Running Tiling Tests...")
    test_mfma_a8_flir_preshuffle("fp8", 5120, 5120, 8320, tile_m=64, tile_n=256, tile_k=128)
    test_mfma_a8_flir_preshuffle("int8", 5120, 5120, 8320, tile_m=64, tile_n=256, tile_k=128)
    test_mfma_a8_flir_preshuffle("fp8", 16, 5120, 8192, tile_m=16, tile_n=64, tile_k=512)

