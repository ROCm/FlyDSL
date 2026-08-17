#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Softmax backward correctness and performance tests.

Covers dx = y * (dy - sum(dy * y)) across:
  - the vectorized path (N % tile_cols == 0), both register-buffered and
    reload variants either side of MAX_RESIDENT_TILES;
  - the generic masked scalar path (arbitrary N, including N < BLOCK_THREADS);
  - f32 / f16 / bf16.
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

from kernels.norm.softmax_bwd_kernel import (  # noqa: E402
    build_softmax_bwd_module,
    softmax_bwd_uses_register_buffering,
)
from tests.test_common import run_perftest  # noqa: E402

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

_TORCH_DTYPE = {"f32": DTYPE_FP32, "f16": DTYPE_FP16, "bf16": DTYPE_BF16}

# Relative to max|ref|. bf16 carries 8 mantissa bits, so one ulp is ~3.9e-3.
_REL_TOL = {"f32": 1e-5, "f16": 2e-3, "bf16": 1e-2}

WARMUP_ITERS = 10
BENCH_ITERS = 100

# (M, N, dtype) -> which code path it exercises.
_BWD_CONFIGS = (
    (32, 128, "f16"),  # generic, N < BLOCK_THREADS: only half the lanes valid
    (64, 256, "f32"),  # generic, launch-bound small case from the issue
    (16, 512, "bf16"),  # generic, bf16 scalar convert
    (128, 2000, "f32"),  # generic tail: 2000 % 256 = 208, partial last iteration
    (4096, 4097, "bf16"),  # generic tail, large M
    (1, 8192, "bf16"),  # vectorized, M=1 grid edge
    (64, 1024, "f32"),  # vectorized buffered, num_tiles=1 (minimum fast path)
    (1024, 2048, "bf16"),  # vectorized buffered, num_tiles=1
    (1024, 4096, "f16"),  # vectorized buffered, num_tiles=2
    (1024, 8192, "bf16"),  # vectorized buffered, num_tiles=4
    (512, 8192, "f32"),  # vectorized buffered, num_tiles=8
    (256, 16384, "f32"),  # vectorized buffered, 16 tiles -- at the cap
    (512, 16384, "bf16"),  # vectorized buffered, 8 tiles -- at the cap
    (128, 32768, "f32"),  # vectorized reload -- past the cap
    (256, 32768, "bf16"),  # vectorized reload -- past the cap
)


def _get_bwd_configs():
    """Config list, overridable via ROCDSL_SOFTMAX_BWD_SHAPES="M,N,dtype;..."."""
    shapes_env = os.environ.get("ROCDSL_SOFTMAX_BWD_SHAPES", "").strip()
    if not shapes_env:
        return list(_BWD_CONFIGS)
    configs = []
    for part in shapes_env.split(";"):
        p = part.strip()
        if p:
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    return configs


def _make_inputs(M, N, dtype_str, seed=42):
    """Probabilities from a real softmax plus nontrivial upstream gradients."""
    torch.manual_seed(seed)
    torch_dtype = _TORCH_DTYPE[dtype_str]
    logits = (torch.rand((M, N), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0
    y = torch.softmax(logits, dim=1).to(torch_dtype).contiguous()
    dy = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32).to(torch_dtype).contiguous()
    return y, dy


def _reference_softmax_bwd(y, dy):
    """fp32 reference from the already-quantized inputs.

    Computed from y/dy as the kernel sees them, so the comparison isolates
    kernel error from input quantization.
    """
    y32 = y.to(DTYPE_FP32)
    dy32 = dy.to(DTYPE_FP32)
    return y32 * (dy32 - (dy32 * y32).sum(dim=1, keepdim=True))


def run_bwd_test(M, N, dtype_str, verbose=True):
    """Build, launch and compare one config. Returns (ok, gpu_us)."""
    if verbose:
        buffered = softmax_bwd_uses_register_buffering(N, dtype_str)
        print(f"\nSoftmax bwd: M={M}, N={N}, dtype={dtype_str}, register_buffered={buffered}")

    try:
        launch_fn = build_softmax_bwd_module(N, dtype_str)
    except Exception as e:  # noqa: BLE001 - report, do not abort the sweep
        print(f"[FAIL] Compile failed for (M={M}, N={N}, {dtype_str}): {type(e).__name__}: {e}")
        return False, None

    y, dy = _make_inputs(M, N, dtype_str)
    dx = torch.empty_like(y)
    expected = _reference_softmax_bwd(y, dy)

    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(dy, y, dx, M, stream=stream)

    kernel_launch()
    torch.cuda.synchronize()

    res = dx.to(DTYPE_FP32)
    scale = expected.abs().max().item()
    max_err = (res - expected).abs().max().item()
    rel_err = max_err / scale if scale > 0 else max_err
    tol = _REL_TOL[dtype_str]

    # Rows of y sum to 1, so each row of dx must sum to ~0. Normalize by the row
    # L1 norm: this is a cancelling sum of N terms, so a single element's
    # magnitude is the wrong scale (the residual grows like sqrt(N) * ulp).
    l1 = res.abs().sum(dim=1)
    row_sum_err = (res.sum(dim=1).abs() / l1.clamp_min(1e-30)).max().item()

    ok = rel_err < tol
    if verbose:
        print(f"  Max rel error: {rel_err:.2e} (tol={tol:.0e})   row-sum/L1: {row_sum_err:.2e}")

    gpu_us = None
    if verbose:
        _, avg_us = run_perftest(
            lambda: (kernel_launch(), torch.cuda.synchronize()),
            num_iters=BENCH_ITERS,
            num_warmup=WARMUP_ITERS,
        )
        torch.cuda.synchronize()
        gpu_us = avg_us
        elem_bytes = 4 if dtype_str == "f32" else 2
        total_bytes = 3 * M * N * elem_bytes  # read y + read dy, write dx
        print(f"Kernel avg time: {avg_us / 1000.0:.4f} ms")
        print(f"Bandwidth: {total_bytes / (avg_us / 1e6) / 1e9:.2f} GB/s")

    print("  Passed" if ok else "  Failed")
    return ok, gpu_us


@pytest.mark.parametrize("M,N,dtype", _BWD_CONFIGS)
def test_softmax_bwd(M, N, dtype):
    ok, _ = run_bwd_test(M, N, dtype, verbose=False)
    assert ok, f"softmax backward mismatch for M={M}, N={N}, dtype={dtype}"


@pytest.mark.parametrize("M,N,dtype", ((1024, 4096, "f32"), (128, 2000, "f32")))
def test_softmax_bwd_matches_autograd(M, N, dtype):
    """Cross-check the closed form against autograd, on one aligned and one tail shape.

    The closed-form reference would happily reproduce a sign error; autograd
    validates the formula itself.
    """
    torch.manual_seed(7)
    logits = ((torch.rand((M, N), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0).requires_grad_(True)
    y = torch.softmax(logits, dim=1)
    dy = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    (expected,) = torch.autograd.grad(y, [logits], grad_outputs=dy)

    launch_fn = build_softmax_bwd_module(N, dtype)
    y_in = y.detach().contiguous()
    dx = torch.empty_like(y_in)
    launch_fn(dy.contiguous(), y_in, dx, M, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(dx, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("M,N,dtype", ((1024, 8192, "bf16"), (64, 1024, "f32"), (128, 2000, "f32")))
def test_softmax_bwd_row_sum_invariant(M, N, dtype):
    """sum(dx) over a row must vanish, because rows of y sum to 1.

    dy is all-positive here on purpose: then dot = sum(y*dy) > 0 and the row sum
    of an uncorrected dx = y*dy would equal dot, i.e. a ratio of exactly 1.0
    against the L1 norm. That makes an omitted dot correction unmissable rather
    than a marginal tolerance call.
    """
    torch.manual_seed(11)
    torch_dtype = _TORCH_DTYPE[dtype]
    logits = (torch.rand((M, N), device="cuda", dtype=DTYPE_FP32) * 4.0) - 2.0
    y = torch.softmax(logits, dim=1).to(torch_dtype).contiguous()
    dy = (torch.rand((M, N), device="cuda", dtype=DTYPE_FP32) + 0.5).to(torch_dtype).contiguous()

    launch_fn = build_softmax_bwd_module(N, dtype)
    dx = torch.empty_like(y)
    launch_fn(dy, y, dx, M, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    res = dx.to(DTYPE_FP32)
    ratio = (res.sum(dim=1).abs() / res.abs().sum(dim=1).clamp_min(1e-30)).max().item()
    assert ratio < 1e-2, f"row sums do not vanish (max |sum|/L1 = {ratio:.3e}); dot correction likely wrong"


@pytest.mark.parametrize("M,N,dtype", ((1024, 8192, "bf16"), (128, 2000, "f32")))
def test_softmax_bwd_is_deterministic(M, N, dtype):
    """Repeat determinism on one aligned and one predicated case."""
    launch_fn = build_softmax_bwd_module(N, dtype)
    y, dy = _make_inputs(M, N, dtype)
    stream = torch.cuda.current_stream()

    first = torch.empty_like(y)
    launch_fn(dy, y, first, M, stream=stream)
    torch.cuda.synchronize()
    first = first.clone()

    for _ in range(3):
        again = torch.empty_like(y)
        launch_fn(dy, y, again, M, stream=stream)
        torch.cuda.synchronize()
        assert torch.equal(first, again), "softmax backward is not run-to-run deterministic"


def test_register_buffering_threshold():
    """Pin the dispatch threshold so retuning MAX_RESIDENT_COLS is deliberate.

    The cap is on elements held per thread (N / BLOCK_THREADS), so it lands on
    the same N for every dtype.
    """
    assert softmax_bwd_uses_register_buffering(1024, "f32") is True
    assert softmax_bwd_uses_register_buffering(8192, "f32") is True
    assert softmax_bwd_uses_register_buffering(16384, "f32") is True  # at the cap
    assert softmax_bwd_uses_register_buffering(32768, "f32") is False  # past it
    assert softmax_bwd_uses_register_buffering(16384, "bf16") is True  # at the cap
    assert softmax_bwd_uses_register_buffering(32768, "bf16") is False  # past it
    # Not on the vectorized path at all.
    assert softmax_bwd_uses_register_buffering(2000, "f32") is False  # not a multiple
    assert softmax_bwd_uses_register_buffering(512, "bf16") is False  # below tile_cols


@pytest.mark.large_shape
def test_softmax_bwd_large_shape():
    ok, _ = run_bwd_test(32768, 8192, "bf16")
    assert ok


def test_all():
    """Script entry point: run the whole sweep and report."""
    failures = 0
    for M, N, dtype in _get_bwd_configs():
        ok, _ = run_bwd_test(M, N, dtype)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TESTS FAILED")
    print("=" * 80)
    # Ensure a non-zero exit code on failure for shell wrappers.
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_all()
