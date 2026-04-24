#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
TopK Gating Softmax Operator Test

Fused softmax + top-K expert selection for Mixture-of-Experts gating.
Validates:
  - topk_weights match torch.softmax -> torch.topk reference
  - topk_indices match the reference top-K expert indices
  - token_expert_indices follow the k * num_tokens + token_row convention
  - Optional renormalization (selected weights sum to 1.0)
"""

import os

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

from tests.test_common import run_perftest
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    print_perf_table,
)

from kernels.topk_gating_softmax_kernel import (
    build_topk_gating_softmax_module,
    KERNEL_NAME,
)

WARMUP_ITERS = 10
BENCH_ITERS = 100


def _torch_dtype(dtype_str):
    return {"f32": DTYPE_FP32, "f16": DTYPE_FP16, "bf16": DTYPE_BF16}[dtype_str]


def run_test(num_tokens, num_experts, topk, dtype_str, renormalize=True):
    print(
        f"\nTesting TopK Gating Softmax: "
        f"tokens={num_tokens}, experts={num_experts}, topk={topk}, "
        f"dtype={dtype_str}, renorm={renormalize}"
    )

    try:
        launch_fn = build_topk_gating_softmax_module(
            num_experts=num_experts,
            topk=topk,
            dtype_str=dtype_str,
            renormalize=renormalize,
        )
    except Exception as e:
        print(f"Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

    torch.manual_seed(42)
    torch_dtype = _torch_dtype(dtype_str)
    gating_fp32 = (
        torch.rand((num_tokens, num_experts), device="cuda", dtype=DTYPE_FP32) * 4.0
    ) - 2.0
    # Quantize to the kernel's input dtype FIRST so the reference sees the
    # exact bytes the kernel sees. Otherwise the K-th expert can flip at
    # bf16/f16 precision boundaries (both vLLM and FlyDSL pick a different
    # near-tie expert than fp32-softmax does).
    gating_dev = gating_fp32.to(torch_dtype).contiguous()
    gating_for_ref = gating_dev.to(DTYPE_FP32)

    # --- PyTorch reference ---
    probs_ref = torch.softmax(gating_for_ref, dim=1)
    ref_weights, ref_indices = torch.topk(probs_ref, topk, dim=1)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=1, keepdim=True).clamp(min=1e-20)
    ref_weights = ref_weights.to(DTYPE_FP32)
    ref_indices = ref_indices.to(torch.int32)

    # token_expert_indices reference: k * num_tokens + row
    ref_tei = torch.zeros_like(ref_indices)
    for k in range(topk):
        ref_tei[:, k] = k * num_tokens + torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    # --- Device tensors ---
    topk_weights_dev = torch.empty(
        (num_tokens, topk), device="cuda", dtype=DTYPE_FP32
    )
    topk_indices_dev = torch.empty(
        (num_tokens, topk), device="cuda", dtype=torch.int32
    )
    token_expert_indices_dev = torch.empty(
        (num_tokens, topk), device="cuda", dtype=torch.int32
    )

    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(
            gating_dev,
            topk_weights_dev,
            topk_indices_dev,
            token_expert_indices_dev,
            num_tokens,
            stream=stream,
        )

    kernel_launch()
    torch.cuda.synchronize()

    _, avg_us = run_perftest(
        lambda: (kernel_launch(), torch.cuda.synchronize()),
        num_iters=BENCH_ITERS,
        num_warmup=WARMUP_ITERS,
    )
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(
            kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS
        )

    avg_ms = avg_us / 1000.0
    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL topk_gating_softmax gpu: {flydsl_gpu_us:.1f} us")

    # --- Verification ---
    atol_weight = 2e-2 if dtype_str in ("bf16", "f16") else 1e-5
    passed = True

    # 1. Check topk_indices: the selected expert sets should match
    #    (order within top-K may differ with tied values, so compare as sets per row)
    got_indices = topk_indices_dev.cpu()
    exp_indices = ref_indices.cpu()

    indices_match = 0
    for row in range(num_tokens):
        got_set = set(got_indices[row].tolist())
        exp_set = set(exp_indices[row].tolist())
        if got_set == exp_set:
            indices_match += 1
    indices_pct = 100.0 * indices_match / num_tokens
    print(f"  Indices match: {indices_match}/{num_tokens} rows ({indices_pct:.1f}%)")
    if indices_pct < 99.0:
        print("  FAILED: too many index mismatches")
        passed = False

    # 2. Check topk_weights: for matching rows, compare sorted weights
    got_weights = topk_weights_dev.cpu().to(DTYPE_FP32)
    exp_weights = ref_weights.cpu().to(DTYPE_FP32)

    got_sorted, _ = got_weights.sort(dim=1, descending=True)
    exp_sorted, _ = exp_weights.sort(dim=1, descending=True)
    weight_err = (got_sorted - exp_sorted).abs().max().item()
    print(f"  Max weight error (sorted): {weight_err:.2e} (atol={atol_weight})")
    if weight_err > atol_weight:
        print("  FAILED: weight error too large")
        passed = False

    # 3. Check token_expert_indices
    got_tei = token_expert_indices_dev.cpu()
    tei_match = (got_tei == ref_tei.cpu()).all().item()
    print(f"  token_expert_indices correct: {tei_match}")
    if not tei_match:
        print("  FAILED: token_expert_indices mismatch")
        print(f"    Expected first row: {ref_tei[0].tolist()}")
        print(f"    Got first row:      {got_tei[0].tolist()}")
        passed = False

    # 4. Check renormalization (weights sum to ~1.0)
    if renormalize:
        row_sums = got_weights.sum(dim=1)
        max_sum_err = (row_sums - 1.0).abs().max().item()
        print(f"  Max renorm sum error: {max_sum_err:.2e}")
        if max_sum_err > atol_weight:
            print("  FAILED: renormalized weights don't sum to 1")
            passed = False

    if passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return passed, flydsl_gpu_us


def test_all():
    print("=" * 80)
    print("Running TopK Gating Softmax Tests")
    print("=" * 80)

    shapes_env = os.environ.get("ROCDSL_TOPK_GATING_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            toks, exps, k, dt = [x.strip() for x in p.split(",")]
            configs.append((int(toks), int(exps), int(k), dt))
    else:
        configs = [
            (1024, 128, 6, "bf16"),
            (512, 64, 2, "bf16"),
            (256, 8, 2, "f32"),
            (128, 128, 6, "f16"),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows = []

    failures = 0
    for num_tokens, num_experts, topk, dtype_str in configs:
        ok, flydsl_gpu_us = run_test(
            num_tokens, num_experts, topk, dtype_str, renormalize=True
        )
        if not ok:
            failures += 1

        if do_compare:
            perf_rows.append(
                PerfRow(
                    op="topk_gating_softmax",
                    shape=f"{num_tokens}x{num_experts}xk{topk}",
                    dtype=dtype_str,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=None,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_all()
