#!/usr/bin/env python3
"""RDNA4 GEMM correctness tests (gfx120x, wave32).

Kernel implementations:
  kernels/rdna_f16_gemm.py          — BF16/F16 GEMM with LDS
  kernels/rdna_fp8_preshuffle_gemm.py — FP8 GEMM with B preshuffle
"""

import logging
import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module as _create_wmma_gemm_module_gfx11  # noqa: E402
from kernels.gemm.rdna3_f16_gemm_autotune import pick_tile  # noqa: E402
from kernels.gemm.rdna_f16_gemm import create_wmma_gemm_module as _create_wmma_gemm_module_gfx12  # noqa: E402
from kernels.gemm.rdna_fp8_preshuffle_gemm import (  # noqa: E402
    compile_fp8_gemm,
    fp8_quantize_per_channel,
    fp8_quantize_per_token,
    preshuffle_b_fp8,
)
from tests.test_common import run_perftest, verify_output  # noqa: E402

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna4():
    if not ARCH.startswith("gfx120"):
        pytest.skip(f"RDNA4 GEMM requires gfx120x, got {ARCH}")


def _requires_rdna_wmma():
    """gfx11* (RDNA3/RDNA3.5) or gfx120* (RDNA4) — anything with f16/bf16 WMMA."""
    if not (ARCH.startswith("gfx11") or ARCH.startswith("gfx120")):
        pytest.skip(f"RDNA WMMA GEMM requires gfx11* or gfx120*, got {ARCH}")


def _requires_rdna3():
    """Only rdna3_f16_gemm picks its tile from the shape; gfx12 still uses a fixed one."""
    if not ARCH.startswith("gfx11"):
        pytest.skip(f"gfx11-only behaviour, got {ARCH}")


def create_wmma_gemm_module(*args, **kwargs):
    """Pick the kernel variant matching the current arch.

    gfx11 uses the legacy v16-operand WMMA ABI; gfx12 uses v8 — different
    enough that the LDS-load and accumulator-store math differ. The two
    kernels share the same call signature.
    """
    if ARCH.startswith("gfx11"):
        return _create_wmma_gemm_module_gfx11(*args, **kwargs)
    return _create_wmma_gemm_module_gfx12(*args, **kwargs)


# ── BF16/F16 GEMM ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(256, 256, 256, id="256x256x256"),
        pytest.param(256, 256, 512, id="256x256x512"),
        pytest.param(512, 512, 512, id="512x512x512", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        ("bf16", "bf16"),
        ("f16", "bf16"),
        ("f16", "f16"),
        ("bf16", "f16"),
    ],
)
def test_f16_gemm_correctness(M, N, K, in_dtype, out_dtype):
    """Test BF16/F16 GEMM correctness for various shapes and dtypes."""
    _requires_rdna_wmma()

    in_torch = torch.bfloat16 if in_dtype == "bf16" else torch.float16
    out_torch = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    torch.manual_seed(42)

    launch_fn, BLOCK_M, BLOCK_N, BLOCK_K = create_wmma_gemm_module(M, N, K, in_dtype=in_dtype, out_dtype=out_dtype)

    A = torch.randn(M, K, dtype=in_torch, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=in_torch, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=out_torch, device="cuda")

    launch_fn(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.05, rtol=0.05)


def test_f16_gemm_stochastic_rounding():
    """BF16 GEMM with the stochastic-rounding epilogue: bounded, seed-varying, reproducible.

    Unbiasedness of the rounding itself is proven in tests/unit/test_stochastic_rounding.py;
    here we only check the GEMM wiring.
    """
    _requires_rdna_wmma()

    M = N = K = 256
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    C_ref = A.float() @ B_T.float().T
    stream = torch.cuda.current_stream()

    rn_gemm, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="bf16", rounding="rn")
    rs_gemm, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="bf16", rounding="rs")

    def run(launch_fn, *seed):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        launch_fn(C, A, B_T, stream, *seed)
        torch.cuda.synchronize()
        return C.float()

    C_rn = run(rn_gemm)  # default 4-arg launch: seed is unused for round-to-nearest
    C_rs1, C_rs2 = run(rs_gemm, 1), run(rs_gemm, 2)

    # SR stays within about an ULP of round-to-nearest against the f32 reference
    assert (C_rs1 - C_ref).abs().max() < 3 * (C_rn - C_ref).abs().max() + 5e-3
    # the seed is a runtime argument: one compiled kernel, different seeds vary
    # the rounding, and a repeated seed is reproducible
    assert not torch.equal(C_rs1, C_rs2)
    assert torch.equal(C_rs1, run(rs_gemm, 1))


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(256, 256, 256, id="256x256x256"),
    ],
)
def test_f16_gemm_f32_output(M, N, K):
    """Test BF16 GEMM with f32 output accumulation."""
    _requires_rdna_wmma()

    torch.manual_seed(42)
    launch_fn, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="f32")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(384, 384, 2048, id="384x384x2048"),
        pytest.param(1152, 1152, 1024, id="1152x1152x1024"),
        pytest.param(2560, 2560, 1024, id="2560x2560x1024", marks=pytest.mark.large_shape),
    ],
)
def test_f16_gemm_grid_m_not_a_multiple_of_the_group_width(M, N, K):
    """Shapes whose M-tile count is not a multiple of the L2 grouping cap.

    The grid swizzle derives bid_m from a fixed group width, so before the width
    was snapped down to a divisor of grid_m the last group addressed tiles past
    the end of the grid. This is reachable at the default 128x128 tile, not only
    at narrower ones: measured on gfx1100, 1152, 1280 and 1664 square came back
    wrong by roughly 400x the bf16 rounding floor, while 1536 and 2560 square
    faulted the GPU. Which of the two you get depends on whether the address past
    the grid happens to be mapped, so the silent wrong answer is the common case.

    At the default 128x128 tile these give grid_m of 3, 9 and 20, none of them a
    multiple of the group width of 8.
    """
    _requires_rdna3()
    torch.manual_seed(42)

    launch_fn, BLOCK_M, _, _ = _create_wmma_gemm_module_gfx11(M, N, K, in_dtype="bf16", out_dtype="bf16")
    grid_m = M // BLOCK_M
    assert grid_m % 8, f"grid_m={grid_m} divides the default group width; shape no longer covers the fault"

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    launch_fn(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(256, 256, 4096, id="256x256x4096"),
        pytest.param(1024, 1024, 1024, id="1024x1024x1024"),
    ],
)
def test_f16_gemm_autotuned_matches_the_heuristic_path(M, N, K):
    """The autotune wrapper, left unconfigured, is a pass-through.

    It resolves through the tuner once and then calls the built module directly:
    the tuner re-derives its cache key on every call, which costs more host time
    than these shapes take on the GPU. So this checks both halves — the same
    tile as ``pick_tile``, and that the second call does not build again.
    """
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    signature = (M, N, K, "bf16", "bf16", "rn")
    gemm_autotune._resolved.pop(signature, None)

    gemm_autotune.rdna3_gemm_autotuned(C, A, B_T)
    torch.cuda.synchronize()
    assert verify_output(C.float(), A.float() @ B_T.float().T, atol=0.05, rtol=0.05)

    resolved = gemm_autotune._resolved[signature]
    expected_tile = pick_tile(M, N, K)
    assert resolved is gemm_autotune._build(M, N, K, "bf16", "bf16", "rn", *expected_tile)

    C.zero_()
    gemm_autotune.rdna3_gemm_autotuned(C, A, B_T)
    torch.cuda.synchronize()
    assert verify_output(C.float(), A.float() @ B_T.float().T, atol=0.05, rtol=0.05)
    assert gemm_autotune._resolved[signature] is resolved


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(1024, 1024, 1024, id="1k"),
        pytest.param(2048, 2048, 2048, id="2k", marks=pytest.mark.large_shape),
    ],
)
def test_f16_gemm_benchmark(M, N, K):
    """Benchmark BF16 GEMM throughput."""
    _requires_rdna_wmma()

    torch.manual_seed(42)
    launch_fn, _, _, _ = create_wmma_gemm_module(M, N, K, in_dtype="bf16", out_dtype="bf16")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.01
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.01
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def run_kernel():
        launch_fn(C, A, B_T, torch.cuda.current_stream())

    _, avg_us = run_perftest(run_kernel, num_iters=20, num_warmup=3)

    flops = 2 * M * N * K
    tflops = flops / (avg_us / 1e6) / 1e12
    logging.getLogger("flydsl").info(f"[f16_gemm] {M}x{N}x{K} bf16: {avg_us:.1f} us, {tflops:.2f} TFLOPS")

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C.float(), C_ref, atol=0.1, rtol=0.1, msg=f"{M}x{N}x{K}")


# ── FP8 Preshuffle GEMM ──────────────────────────────────────────────────────


def _run_fp8_gemm(M, N, K, tile_m=32, tile_n=None, tile_k=32):
    """Helper: quantize (per-token/per-channel), preshuffle B, compile, launch."""
    launch_fn = compile_fp8_gemm(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    A_f32 = torch.randn(M, K, device="cuda") * 0.1
    B_f32 = torch.randn(K, N, device="cuda") * 0.1

    A_fp8, scale_a = fp8_quantize_per_token(A_f32)
    B_fp8, scale_b = fp8_quantize_per_channel(B_f32)

    B_shuf = preshuffle_b_fp8(B_fp8)

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    sa = scale_a.to(device="cuda", dtype=torch.float32).contiguous()
    sb = scale_b.to(device="cuda", dtype=torch.float32).contiguous()

    A_f32_view = A_fp8.view(torch.float32).contiguous()
    B_shuf_f32 = B_shuf.view(torch.float32).contiguous()

    launch_fn(C, A_f32_view, B_shuf_f32, sa, sb, torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A_fp8.float() * scale_a.unsqueeze(1)) @ (B_fp8.float() * scale_b.unsqueeze(0))
    return C, C_ref


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(32, 128, 128, id="32x128x128"),
        pytest.param(32, 128, 256, id="32x128x256"),
        pytest.param(32, 256, 256, id="32x256x256"),
    ],
)
def test_fp8_gemm_correctness(M, N, K):
    """Test FP8 preshuffle GEMM correctness."""
    _requires_rdna4()
    torch.manual_seed(42)

    C, C_ref = _run_fp8_gemm(M, N, K)
    assert verify_output(C.float(), C_ref.float(), atol=0.5, rtol=0.1)


def test_fp8_preshuffle_b():
    """Test preshuffle_b_fp8 produces correct layout."""
    _requires_rdna4()

    K, N = 64, 32
    B = torch.arange(K * N, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn).reshape(K, N)
    B_shuf = preshuffle_b_fp8(B)
    assert B_shuf.shape == (N // 16, K // 16, 2, 16, 8), f"Wrong shape: {B_shuf.shape}"


def test_fp8_quantize():
    """Test fp8_quantize_per_token roundtrip."""
    _requires_rdna4()

    x = torch.randn(64, 64, device="cuda")
    x_fp8, scale = fp8_quantize_per_token(x)

    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (64,)
    assert (scale > 0).all()

    x_roundtrip = x_fp8.float() * scale.unsqueeze(1)
    rel_err = ((x - x_roundtrip).abs() / (x.abs() + 1e-6)).mean().item()
    assert rel_err < 0.2, f"Mean relative roundtrip error too large: {rel_err}"
