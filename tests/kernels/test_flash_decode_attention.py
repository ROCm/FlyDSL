#!/usr/bin/env python3
"""
Flash Decode Attention Test

Verifies the single-query (decode-phase) attention kernel:
  O = softmax(Q @ K^T / sqrt(d)) @ V

where Q has a single token per (batch, head).

Grid:  (batch_size * num_heads, 1, 1)
Block: (64, 1, 1)  -- single AMD wave64

Model shape references (decode phase, KV-cache heads):
  LLaMA 3.1 8B  :  8 KV heads, head_dim=128
  LLaMA 3.1 70B :  8 KV heads, head_dim=128
  DeepSeek V3   : compressed KV via MLA, head_dim=128
  Mixtral 8x7B  :  8 KV heads, head_dim=128
"""

import sys
import os
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    print_perf_table,
)

import flydsl
from kernels.flash_decode_attention import (
    build_flash_decode_attention_module,
    KERNEL_NAME,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100

DTYPE_MAP = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}

ATOL_MAP = {
    "f32": 1e-4,
    "f16": 2e-2,
    "bf16": 3e-2,
}


def _sdpa_ref_us(Q_ref, K_ref, V_ref, warmup=10, iters=100):
    """Benchmark PyTorch scaled_dot_product_attention as baseline."""
    def fn():
        F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=False)
    return bench_gpu_us_torch(fn, warmup=warmup, iters=iters)


def run_test(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_str: str = "f16",
    do_compare: bool = False,
):
    total_heads = batch_size * num_heads
    torch_dtype = DTYPE_MAP[dtype_str]
    atol = ATOL_MAP[dtype_str]

    shape_tag = f"B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}"
    print(
        f"\nTesting Flash Decode Attention "
        f"(B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}, dtype={dtype_str})"
    )

    try:
        m = build_flash_decode_attention_module(seq_len, head_dim, dtype_str)
        exe = flydsl.compile(m)
    except Exception as e:
        print(f"[FAIL] Compile failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

    torch.manual_seed(42)

    Q_ref = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32)
    K_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    V_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    expected = F.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref, is_causal=False
    )
    expected = expected.squeeze(2).reshape(total_heads, head_dim).to(torch.float32)

    Q_dev = Q_ref.squeeze(2).reshape(total_heads, head_dim).to(torch_dtype).contiguous()
    K_dev = K_ref.reshape(total_heads, seq_len, head_dim).to(torch_dtype).contiguous()
    V_dev = V_ref.reshape(total_heads, seq_len, head_dim).to(torch_dtype).contiguous()
    O_dev = torch.empty(total_heads, head_dim, device="cuda", dtype=torch_dtype)

    print("  Launching kernel...")

    def kernel_launch():
        exe(Q_dev, K_dev, V_dev, O_dev, total_heads)

    kernel_launch()
    torch.cuda.synchronize()

    flir_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = flir_gpu_us / 1000.0

    elem_bytes = 4 if dtype_str == "f32" else 2
    kv_bytes = 2 * total_heads * seq_len * head_dim * elem_bytes
    bandwidth_gbs = kv_bytes / (flir_gpu_us / 1e6) / 1e9
    print(f"  Kernel avg time: {avg_ms:.4f} ms  (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"  Bandwidth (KV read): {bandwidth_gbs:.2f} GB/s")

    sdpa_us = None
    if do_compare:
        Q_sdpa = Q_ref.to(torch_dtype)
        K_sdpa = K_ref.to(torch_dtype)
        V_sdpa = V_ref.to(torch_dtype)
        sdpa_us = _sdpa_ref_us(Q_sdpa, K_sdpa, V_sdpa, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
        print(f"  PyTorch SDPA avg time: {sdpa_us / 1000.0:.4f} ms")

    output_f32 = O_dev.to(torch.float32)
    error = (output_f32 - expected).abs().max().item()
    print(f"  Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("  PASSED")
        ok = True
    else:
        print("  FAILED")
        print("  Expected (first 8):", expected[0, :8])
        print("  Got      (first 8):", output_f32[0, :8])
        ok = False

    perf_row = PerfRow(
        op="flash_decode_attn",
        shape=shape_tag,
        dtype=dtype_str,
        flir_gpu_us=flir_gpu_us,
        aiter_gpu_us=sdpa_us,
    )
    return ok, perf_row


# ---------------------------------------------------------------------------
# CI test configs  (kept small for fast CI turnaround)
# ---------------------------------------------------------------------------
CI_CONFIGS = [
    # (batch, heads, seq_len, head_dim, dtype)
    (1, 1, 64, 128, "f32"),
    (2, 4, 128, 128, "f16"),
    (1, 2, 64, 128, "bf16"),
]

# ---------------------------------------------------------------------------
# Real model shapes  (decode phase with moderate seq_len for compile-time
# unrolled loop; production seq_len would need scf.for dynamic loop)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    # LLaMA 3.1 8B  : 8 KV heads, head_dim=128
    (1, 8, 256, 128, "f16"),
    (8, 8, 256, 128, "f16"),
    (32, 8, 128, 128, "bf16"),
    # LLaMA 3.1 70B : 8 KV heads, head_dim=128
    (1, 8, 256, 128, "bf16"),
    (4, 8, 256, 128, "f16"),
    # Mixtral / DeepSeek-like : larger head count
    (1, 32, 128, 128, "f16"),
    (1, 32, 256, 128, "bf16"),
]


def test_flash_decode_attention():
    """Pytest entry point -- CI correctness + optional model shapes."""
    shapes_env = os.environ.get("FLYDSL_FLASH_ATTN_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            b, h, s, d, dt = [x.strip() for x in p.split(",")]
            configs.append((int(b), int(h), int(s), int(d), dt))
    else:
        run_models = os.environ.get("FLYDSL_FLASH_ATTN_MODELS", "0") == "1"
        configs = CI_CONFIGS + (MODEL_CONFIGS if run_models else [])

    do_compare = os.environ.get("FLYDSL_COMPARE_SDPA", "0") == "1"

    print("=" * 80)
    print("Running Flash Decode Attention Tests")
    print(f"  Configs: {len(configs)}  compare_sdpa={do_compare}")
    print("=" * 80)

    failures = 0
    perf_rows = []
    for batch, heads, seq_len, head_dim, dtype in configs:
        ok, row = run_test(batch, heads, seq_len, head_dim, dtype, do_compare=do_compare)
        if not ok:
            failures += 1
        if row is not None:
            perf_rows.append(row)

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)

    if do_compare and perf_rows:
        print_perf_table(perf_rows)

    assert failures == 0, f"{failures} test(s) failed"


if __name__ == "__main__":
    test_flash_decode_attention()
