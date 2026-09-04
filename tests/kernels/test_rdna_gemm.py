#!/usr/bin/env python3
"""RDNA4 GEMM correctness tests (gfx120x, wave32).

Kernel implementations:
  kernels/rdna_f16_gemm.py          — BF16/F16 GEMM with LDS
  kernels/rdna_fp8_preshuffle_gemm.py — FP8 GEMM with B preshuffle
"""

import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module as _create_wmma_gemm_module_gfx11  # noqa: E402
from kernels.gemm.rdna3_f16_gemm_autotune import (  # noqa: E402
    _TILE_FIELDS,
    NUM_CU,
    TILE_32x32x64,
    TILE_32x64x64,
    TILE_64x64x64,
    TILE_128x64x32,
    TILE_128x128x32,
    TILE_256x256x32,
    _default_config,
    _ladder_for,
    _tile_workgroups,
    pick_tile,
)
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

    strides = (A.stride(0), B_T.stride(0), C.stride(0))
    arch = gemm_autotune._device_arch(A.device.index)
    signature = gemm_autotune._signature(M, N, K, arch, "bf16", "bf16", "rn", *strides)
    gemm_autotune._resolved.pop(signature, None)

    gemm_autotune.rdna3_gemm_autotuned(C, A, B_T)
    torch.cuda.synchronize()
    assert verify_output(C.float(), A.float() @ B_T.float().T, atol=0.05, rtol=0.05)

    resolved = gemm_autotune._resolved[signature]
    expected_tile = pick_tile(M, N, K, arch=arch)
    assert resolved is gemm_autotune._build(M, N, K, arch, "bf16", "bf16", "rn", *expected_tile, *strides)

    C.zero_()
    gemm_autotune.rdna3_gemm_autotuned(C, A, B_T)
    torch.cuda.synchronize()
    assert verify_output(C.float(), A.float() @ B_T.float().T, atol=0.05, rtol=0.05)
    assert gemm_autotune._resolved[signature] is resolved


def test_f16_gemm_autotuned_keys_the_cache_on_the_row_strides():
    """A padded and a tight operand of one shape must not share a built module.

    The strides are compile-time arguments, so the module built for a padded slice
    reads a tight one at the wrong pitch. Both orders are exercised because only
    the second call of each pair can be served from the cache.
    """
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    M = N = K = 512
    pad = 64
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    B_T = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ref = A.float() @ B_T.float().T

    A_pad = torch.zeros(M, K + pad, dtype=torch.bfloat16, device="cuda")[:, :K]
    B_T_pad = torch.zeros(N, K + pad, dtype=torch.bfloat16, device="cuda")[:, :K]
    A_pad.copy_(A)
    B_T_pad.copy_(B_T)
    assert (A_pad.stride(0), B_T_pad.stride(0)) == (K + pad, K + pad)

    for a, b in ((A_pad, B_T_pad), (A, B_T), (A_pad, B_T_pad)):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        gemm_autotune.rdna3_gemm_autotuned(C, a, b)
        torch.cuda.synchronize()
        assert verify_output(C.float(), ref, atol=0.05, rtol=0.05)


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


# ── RDNA3 host-side tile selection ───────────────────────────────────────────
# pick_tile runs before any kernel is built, so these are plain integer checks.
# They cover the two properties that make selection safe to leave on by default
# and the measured expectations behind it.

# Every shape the heuristic was timed against the whole ladder on, with the tile
# it is expected to pick, so a heuristic edit has to state which shapes it moves.
MEASURED_TILES_GFX1100 = [
    pytest.param((256, 256, 4096), TILE_32x32x64, id="256x256x4096"),
    pytest.param((256, 256, 1024), TILE_64x64x64, id="256x256x1024"),
    pytest.param((384, 384, 2048), TILE_64x64x64, id="384x384x2048"),
    pytest.param((512, 512, 1024), TILE_64x64x64, id="512x512x1024"),
    pytest.param((512, 512, 4096), TILE_64x64x64, id="512x512x4096"),
    pytest.param((256, 1024, 4096), TILE_64x64x64, id="256x1024x4096"),
    pytest.param((768, 768, 2048), TILE_64x64x64, id="768x768x2048"),
    pytest.param((1024, 1024, 512), TILE_128x64x32, id="1024x1024x512"),
    pytest.param((1024, 1024, 1024), TILE_128x64x32, id="1024x1024x1024"),
    pytest.param((1024, 1024, 4096), TILE_64x64x64, id="1024x1024x4096"),
    # 128x128x32 is 8.1% faster here, the worst case the heuristic accepts.
    pytest.param((1152, 1152, 1024), TILE_64x64x64, id="1152x1152x1024"),
    pytest.param((1536, 1536, 1024), TILE_64x64x64, id="1536x1536x1024"),
    pytest.param((1792, 1792, 1024), TILE_64x64x64, id="1792x1792x1024"),
    pytest.param((2048, 2048, 512), TILE_128x128x32, id="2048x2048x512"),
    pytest.param((2048, 2048, 2048), TILE_128x128x32, id="2048x2048x2048"),
    pytest.param((3072, 3072, 1024), TILE_128x128x32, id="3072x3072x1024"),
    pytest.param((4096, 4096, 4096), TILE_128x128x32, id="4096x4096x4096"),
]

# Representative gfx1151 measured shapes.
MEASURED_TILES_GFX1151 = [
    pytest.param((256, 256, 256), TILE_32x64x64, id="256x256x256"),
    pytest.param((256, 256, 4096), TILE_32x64x64, id="256x256x4096"),
    pytest.param((640, 640, 640), TILE_64x64x64, id="640x640x640"),
    pytest.param((768, 768, 768), TILE_128x128x32, id="768x768x768"),
    pytest.param((1024, 1024, 1024), TILE_64x64x64, id="1024x1024x1024"),
    pytest.param((1152, 1152, 1152), TILE_64x64x64, id="1152x1152x1152"),
    pytest.param((1280, 1280, 1280), TILE_128x128x32, id="1280x1280x1280"),
    pytest.param((1536, 1536, 1536), TILE_256x256x32, id="1536x1536x1536"),
    pytest.param((2048, 2048, 2048), TILE_128x128x32, id="2048x2048x2048"),
    pytest.param((8192, 8192, 8192), TILE_128x128x32, id="8192x8192x8192"),
    pytest.param((1024, 2048, 2048), TILE_256x256x32, id="1024x2048x2048"),
    pytest.param((2048, 1024, 2048), TILE_256x256x32, id="2048x1024x2048"),
    pytest.param((512, 4096, 4096), TILE_64x64x64, id="512x4096x4096"),
    pytest.param((4096, 512, 4096), TILE_64x64x64, id="4096x512x4096"),
]

# Square and skewed both ways, K on both sides of the ladder split, sizes from
# "cannot fill the machine" to "fills it easily". Deliberately not the measured
# set, so the properties below are checked where the heuristic extrapolates.
SELECTION_SHAPES = [
    (256, 256, 512),
    (256, 256, 4096),
    (256, 2048, 1024),
    (512, 512, 2048),
    (512, 2048, 4096),
    (768, 768, 1024),
    (1024, 256, 4096),
    (1024, 1024, 1024),
    (1536, 512, 2048),
    (1536, 1536, 4096),
    (2048, 1024, 512),
    (2048, 2048, 2048),
    (3072, 3072, 1024),
    (4096, 512, 1024),
    (4096, 2048, 4096),
    (4096, 4096, 4096),
]

_shape_id = lambda shape: "x".join(map(str, shape))  # noqa: E731


@pytest.mark.parametrize("shape, expected", MEASURED_TILES_GFX1100)
def test_pick_tile_matches_the_gfx1100_measured_shapes(shape, expected):
    _requires_rdna3()
    assert pick_tile(*shape, arch="gfx1100") == expected


@pytest.mark.parametrize("shape, expected", MEASURED_TILES_GFX1151)
def test_pick_tile_matches_the_gfx1151_measured_shapes(shape, expected):
    _requires_rdna3()
    assert pick_tile(*shape, arch="gfx1151") == expected


@pytest.mark.parametrize(
    "device_index, arch, expected",
    [
        pytest.param(0, "gfx1100", TILE_64x64x64, id="gfx1100"),
        pytest.param(1, "gfx1151:sramecc-:xnack+", TILE_128x128x32, id="gfx1151"),
    ],
)
def test_default_config_uses_the_input_device_arch(monkeypatch, device_index, arch, expected):
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    device = torch.device("cuda", device_index)
    seen = []

    def get_device_properties(requested):
        seen.append(requested)
        return SimpleNamespace(gcnArchName=arch)

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    gemm_autotune._device_arch.cache_clear()
    try:
        config = gemm_autotune._default_config(A=SimpleNamespace(device=device), M=768, N=768, K=768)

        assert seen == [device_index]
        assert tuple(config.kwargs[field] for field in _TILE_FIELDS) == expected
    finally:
        gemm_autotune._device_arch.cache_clear()


def test_device_arch_is_cached_by_device_index(monkeypatch):
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    seen = []

    def get_device_properties(device_index):
        seen.append(device_index)
        return SimpleNamespace(gcnArchName=("gfx1100", "gfx1151:sramecc-:xnack+")[device_index])

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    gemm_autotune._device_arch.cache_clear()

    try:
        assert gemm_autotune._device_arch(0) == "gfx1100"
        assert gemm_autotune._device_arch(0) == "gfx1100"
        assert gemm_autotune._device_arch(1) == "gfx1151"
        assert seen == [0, 1]
    finally:
        gemm_autotune._device_arch.cache_clear()


def test_build_cache_is_arch_specific(monkeypatch):
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    seen = []

    def create_module(*_args, arch=None, **_kwargs):
        seen.append(arch)
        return object(), None, None, None

    monkeypatch.setattr(gemm_autotune, "create_wmma_gemm_module", create_module)
    gemm_autotune._build.cache_clear()
    args = (256, 256, 256, "bf16", "bf16", "rn", *TILE_64x64x64, 256, 256, 256)

    try:
        gfx1100 = gemm_autotune._build(args[0], args[1], args[2], "gfx1100", *args[3:])
        gfx1151 = gemm_autotune._build(args[0], args[1], args[2], "gfx1151", *args[3:])

        assert gfx1100 is gemm_autotune._build(args[0], args[1], args[2], "gfx1100", *args[3:])
        assert gfx1100 is not gfx1151
        assert seen == ["gfx1100", "gfx1151"]
    finally:
        gemm_autotune._build.cache_clear()


@pytest.mark.parametrize("shape", SELECTION_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("arch", ("gfx1100", "gfx1151"))
def test_picked_tile_is_buildable(shape, arch):
    """create_wmma_gemm_module asserts the rules _tile_workgroups screens for.

    Returning a tile the shape cannot use is not a slow kernel, it is an
    assertion failure at build time, so this is the property that has to hold
    even where the heuristic is extrapolating.
    """
    _requires_rdna3()
    if all(_tile_workgroups(*shape, cfg) is None for cfg in _ladder_for(shape[2])):
        pytest.skip("no tile in the ladder fits this shape")
    assert _tile_workgroups(*shape, pick_tile(*shape, arch=arch)) is not None


@pytest.mark.parametrize("shape", SELECTION_SHAPES, ids=_shape_id)
def test_deep_grids_keep_the_default_tile(shape):
    """A shape whose 128x128x32 grid is deep enough must not be moved off it.

    This is what makes selection safe to enable by default: the large shapes take
    the same path as before it existed, so they cannot regress. Merely covering
    the machine is not the bar -- 128x128x32 lost 19% at 1792x1792x1024 and 37%
    at 1664x1664x1024, both past one workgroup per CU -- so the measured 2.5x is.
    """
    _requires_rdna3()
    workgroups = _tile_workgroups(*shape, TILE_128x128x32)
    if workgroups is None or workgroups < 2.5 * NUM_CU:
        pytest.skip("128x128x32's grid is not deep enough for this shape")
    assert pick_tile(*shape, arch="gfx1100") == TILE_128x128x32


@pytest.mark.parametrize("shape", SELECTION_SHAPES, ids=_shape_id)
def test_untuned_config_is_the_heuristic_tile(shape):
    """An untuned call must be indistinguishable from calling the kernel directly.

    The wrapper exposes tile selection through the shared autotuner, so the
    tuner's default has to be exactly what pick_tile would have returned.
    """
    _requires_rdna3()
    from kernels.gemm import rdna3_f16_gemm_autotune as gemm_autotune

    M, N, K = shape
    A = torch.empty(0, device="cuda")
    arch = gemm_autotune._device_arch(A.device.index)
    config = _default_config(A=A, M=M, N=N, K=K)
    assert tuple(config.kwargs[field] for field in _TILE_FIELDS) == pick_tile(*shape, arch=arch)
