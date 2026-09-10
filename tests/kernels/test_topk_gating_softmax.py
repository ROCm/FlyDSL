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

from kernels.moe.topk_gating_softmax_kernel import (
    build_topk_gating_softmax_module,
)
from tests.kernels.benchmark_common import (
    PerfRow,
    bench_gpu_us_torch,
    print_perf_table,
)
from tests.test_common import run_perftest

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
    gating_fp32 = (torch.rand((num_tokens, num_experts), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0
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
    topk_weights_dev = torch.empty((num_tokens, topk), device="cuda", dtype=DTYPE_FP32)
    topk_indices_dev = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices_dev = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

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
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)

    avg_ms = avg_us / 1000.0
    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL topk_gating_softmax gpu: {flydsl_gpu_us:.1f} us")

    # --- Verification ---
    atol_weight = 2e-2 if dtype_str in ("bf16", "f16") else 1e-5
    passed = True

    # 1. Check topk_indices: every kernel-selected expert must be a valid
    #    top-K choice. We compare against the K-th largest reference
    #    probability rather than torch.topk's specific index set, since
    #    torch.topk and the kernel may legitimately disagree on which
    #    experts to take when several share the boundary probability
    #    (a common bf16/f16 quantization artifact).
    got_indices = topk_indices_dev.cpu()
    exp_indices = ref_indices.cpu()

    probs_ref_cpu = probs_ref.cpu()
    # K-th largest reference probability per row (the top-K threshold).
    kth_threshold, _ = torch.topk(probs_ref_cpu, topk, dim=1)
    kth_threshold = kth_threshold[:, -1]
    # Probability tolerance: bf16/f16 representable gap between adjacent values
    # near the typical softmax magnitude is ~1e-3; f32 is much tighter.
    prob_tol = 1e-3 if dtype_str in ("bf16", "f16") else 1e-6

    indices_match = 0  # strict set equality with torch.topk (informational)
    indices_valid = 0  # every selected expert is at-or-above the K-th threshold
    for row in range(num_tokens):
        got_list = got_indices[row].tolist()
        if set(got_list) == set(exp_indices[row].tolist()):
            indices_match += 1
        if len(set(got_list)) == topk:
            row_thr = kth_threshold[row].item() - prob_tol
            if all(probs_ref_cpu[row, idx].item() >= row_thr for idx in got_list):
                indices_valid += 1
    indices_pct = 100.0 * indices_match / num_tokens
    valid_pct = 100.0 * indices_valid / num_tokens
    print(
        f"  Indices match torch.topk: {indices_match}/{num_tokens} rows "
        f"({indices_pct:.1f}%; ties at the K-th boundary may diverge)"
    )
    print(f"  Indices valid (>= K-th prob): {indices_valid}/{num_tokens} rows " f"({valid_pct:.1f}%)")
    if valid_pct < 100.0:
        print("  FAILED: kernel selected experts below the top-K threshold")
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


def _dispatch_contract_input(capacity, num_experts, dtype):
    """Exactly representable logits with ties across both 8- and 16-value lanes."""
    rows = torch.arange(capacity, dtype=torch.int64)[:, None]
    cols = torch.arange(num_experts, dtype=torch.int64)[None, :]
    logits = ((rows * 17 + cols * 7) % 13 - 6).to(torch.float32)
    logits[0::6] = 0  # Every expert ties: the first K indices must win.
    logits[1::6] = -4
    boundary_experts = [i for i in (7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255) if i < num_experts]
    logits[1::6, boundary_experts] = 4
    # The kernel selects after f32 softmax. All but the final expert underflow
    # to zero, so the remaining winners must be the lowest unused indices.
    logits[2::6] = cols.to(torch.float32) - 1000
    logits[2::6, -1] = 0
    # All eight winners can belong to the final lane, including lane 63.
    logits[4::6] = -4
    logits[4::6, -8:] = torch.arange(1, 9, dtype=torch.float32)
    # Distinct f32 logits round to equal softmax probabilities. Selecting
    # logits instead would incorrectly reverse the lowest-index tie order.
    logits[5::6] = cols.to(torch.float32) * 1e-10
    return logits.to(dtype)


_DISPATCH_CONTRACT_CONFIGS = [
    (num_experts, topk, dtype_str, (e_idx + k_idx + d_idx) % 2 == 0)
    for e_idx, num_experts in enumerate((128, 256))
    for k_idx, topk in enumerate((4, 6, 8))
    for d_idx, dtype_str in enumerate(("bf16", "f16", "f32"))
]


@pytest.mark.parametrize("num_experts,topk,dtype_str,renormalize", _DISPATCH_CONTRACT_CONFIGS)
def test_topk_runtime_dispatch_contract(num_experts, topk, dtype_str, renormalize):
    """Keep the public launcher correct across its runtime batch boundary.

    Reuse one launcher and identically shaped tensors while changing the token
    count. This catches stale specialization of the runtime scalar, including
    switching back to the short-batch path. No performance loop is involved.
    """
    capacity = 2049
    quantized_input = _dispatch_contract_input(capacity, num_experts, _torch_dtype(dtype_str))
    # Stable sorting expresses the kernel's explicit lowest-index tie rule.
    # The discrete inputs avoid ambiguous approximate-exp boundary rankings.
    probabilities = torch.softmax(quantized_input.float(), dim=-1)
    expected_indices = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :topk]
    expected_weights = probabilities.gather(1, expected_indices)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    expected_indices = expected_indices.to(torch.int32)

    gating = quantized_input.to(device="cuda")
    frozen_input = gating.clone()
    weights = torch.empty((capacity, topk), dtype=torch.float32, device="cuda")
    indices = torch.empty((capacity, topk), dtype=torch.int32, device="cuda")
    tei = torch.empty((capacity, topk), dtype=torch.int32, device="cuda")
    launch_fn = build_topk_gating_softmax_module(num_experts, topk, dtype_str, renormalize)

    def launch(num_tokens):
        launch_fn(gating, weights, indices, tei, num_tokens, stream=torch.cuda.current_stream())

    def poison_outputs():
        weights.fill_(float("nan"))
        indices.fill_(-1)
        tei.fill_(-1)

    def check_outputs(num_tokens):
        torch.cuda.synchronize()
        assert torch.equal(gating, frozen_input), "gating input was modified"
        got_weights, got_indices, got_tei = (tensor.cpu() for tensor in (weights, indices, tei))
        torch.testing.assert_close(got_weights[:num_tokens], expected_weights[:num_tokens], atol=2e-6, rtol=2e-5)
        assert torch.equal(got_indices[:num_tokens], expected_indices[:num_tokens]), "expert order/ties differ"
        expected_tei = (
            torch.arange(topk, dtype=torch.int32)[None, :] * num_tokens
            + torch.arange(num_tokens, dtype=torch.int32)[:, None]
        )
        assert torch.equal(got_tei[:num_tokens], expected_tei)
        # The dynamic token count can be smaller than the allocated tensor.
        assert bool(torch.isnan(got_weights[num_tokens:]).all()), "wrote weights beyond num_tokens"
        assert bool((got_indices[num_tokens:] == -1).all()), "wrote indices beyond num_tokens"
        assert bool((got_tei[num_tokens:] == -1).all()), "wrote TEI beyond num_tokens"

    for num_tokens in (1, 17, 2048, 2049, 2048):
        poison_outputs()
        launch(num_tokens)
        check_outputs(num_tokens)

    # torch.cuda.graph selects a capture stream. A launch on a stale/default
    # stream produces an empty or invalid graph; poisoning after capture makes
    # this observable instead of accepting the output of the eager warmup.
    for num_tokens in (2048, 2049):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(num_tokens)
        poison_outputs()
        graph.replay()
        check_outputs(num_tokens)


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
        ok, flydsl_gpu_us = run_test(num_tokens, num_experts, topk, dtype_str, renormalize=True)
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
