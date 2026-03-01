#!/usr/bin/env python3
"""
Flash Decode Attention Test

Verifies the single-query (decode-phase) attention kernel:
  O = softmax(Q @ K^T / sqrt(d)) @ V

where Q has a single token per (batch, head).

Grid:  (batch_size * num_heads, 1, 1)
Block: (64, 1, 1)  -- single AMD wave64
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

import flydsl
from kernels.flash_decode_attention import (
    build_flash_decode_attention_module,
    KERNEL_NAME,
)

WARMUP_ITERS = 5
BENCH_ITERS = 20

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


def run_test(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_str: str = "f16",
):
    total_heads = batch_size * num_heads
    torch_dtype = DTYPE_MAP[dtype_str]
    atol = ATOL_MAP[dtype_str]

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
        return False

    torch.manual_seed(42)

    Q_ref = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32)
    K_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    V_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    # PyTorch reference (always in fp32 for stability).
    expected = F.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref, is_causal=False
    )  # [B, H, 1, D]
    expected = expected.squeeze(2).reshape(total_heads, head_dim).to(torch.float32)

    # Prepare device tensors in target dtype, flattened to [total_heads, ...].
    Q_dev = Q_ref.squeeze(2).reshape(total_heads, head_dim).to(torch_dtype).contiguous()
    K_dev = K_ref.reshape(total_heads, seq_len, head_dim).to(torch_dtype).contiguous()
    V_dev = V_ref.reshape(total_heads, seq_len, head_dim).to(torch_dtype).contiguous()
    O_dev = torch.empty(total_heads, head_dim, device="cuda", dtype=torch_dtype)

    print("  Launching kernel...")

    def kernel_launch():
        exe(Q_dev, K_dev, V_dev, O_dev, total_heads)

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()),
        num_iters=BENCH_ITERS,
        num_warmup=WARMUP_ITERS,
    )
    torch.cuda.synchronize()
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype_str == "f32" else 2
    kv_bytes = 2 * total_heads * seq_len * head_dim * elem_bytes
    bandwidth_gbs = kv_bytes / (avg_us / 1e6) / 1e9
    print(f"  Kernel avg time: {avg_ms:.4f} ms  (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"  Bandwidth (KV read): {bandwidth_gbs:.2f} GB/s")

    output_f32 = O_dev.to(torch.float32)
    error = (output_f32 - expected).abs().max().item()
    print(f"  Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        print("  Expected (first 8):", expected[0, :8])
        print("  Got      (first 8):", output_f32[0, :8])
        return False


def test_flash_decode_attention():
    """Pytest entry point -- small configs for CI."""
    configs = [
        # (batch, heads, seq_len, head_dim, dtype)
        (1, 1, 64, 128, "f32"),
        (2, 4, 128, 128, "f16"),
        (1, 2, 64, 128, "bf16"),
    ]

    shapes_env = os.environ.get("FLYDSL_FLASH_ATTN_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            b, h, s, d, dt = [x.strip() for x in p.split(",")]
            configs.append((int(b), int(h), int(s), int(d), dt))

    print("=" * 80)
    print("Running Flash Decode Attention Tests")
    print("=" * 80)

    failures = 0
    for batch, heads, seq_len, head_dim, dtype in configs:
        if not run_test(batch, heads, seq_len, head_dim, dtype):
            failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)

    assert failures == 0, f"{failures} test(s) failed"


if __name__ == "__main__":
    test_flash_decode_attention()
