#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MFMA preshuffle GEMM tests (layout-API kernels).

Kernel implementations live in `kernels/preshuffle_gemm.py` (fp8/int8/fp16/bf16)
and `kernels/mxfp4_preshuffle.py` (MXFP4). This file is the correctness +
perf harness.
"""

import logging
import os
import sys
from itertools import combinations, product

import pytest
import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.mxfp4_preshuffle import launch_gemm, launch_gemm_fused, prefetch_depth  # noqa: E402
from kernels.gemm.preshuffle_gemm import compile_preshuffle_gemm  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402
from tests.utils import pertoken_quant, shuffle_weight  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _ptr(t):
    """Raw data_ptr as an fx.Pointer kernel arg for the batched launch_gemm."""
    return flyc.from_c_void_p(fx.Uint8, t.contiguous().data_ptr())


def _mxfp4_launcher(
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    out_dtype,
    a_dtype,
    waves_per_eu=0,
    b_dtype="fp4",
    xcd_swizzle=0,
    k_batch=1,
    blockscale="none",
    a_scale_layout="mfma_tile",
    b_scale_layout="mfma_tile",
    prefetch=1,
    single_row=False,
    b_nt=False,
    b_scale_32x32=False,
):
    """Adapt the batched launch_gemm to the (c, a, b, sa, sb, bias, M, N, stream) call shape
    the tests use. This closure fixes every Constexpr, so it reuses one compiled handle:
    a direct @flyc.jit call would retrace launch_gemm on every launch."""
    compiled = None

    def _launch(c, a, b, sa, sb, bias, M, N_, stream):
        nonlocal compiled
        args = (
            _ptr(c),
            _ptr(a),
            _ptr(b),
            _ptr(sa),
            _ptr(sb),
            M,
            N_,
            stream,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            a_dtype,
            out_dtype,
            b_dtype,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            int(waves_per_eu or 0),
            xcd_swizzle,
            k_batch,
            blockscale,
            a_scale_layout,
            b_scale_layout,
            prefetch,
            single_row,
            b_nt,
            b_scale_32x32,
        )
        if compiled is None:
            compiled = flyc.compile(launch_gemm, *args)  # Compiles and launches once.
        else:
            compiled(*args)

    return _launch


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

try:
    import aiter

    HAS_AITER = True
except Exception:
    HAS_AITER = False

ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

DEFAULT_LDS_STAGE = 2
DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3
DEFAULT_RUN_AITER_BENCH = True


def _run_correctness_or_benchmark(
    launch,
    *args,
    bench_iters: int,
    bench_warmup: int,
    test_graph: bool = False,
):
    """Run once under pytest; retain timed iterations for the CLI benchmark."""
    if int(bench_iters) > 0:
        _, us = run_perftest(
            launch,
            *args,
            num_iters=max(2, int(bench_iters)),
            num_warmup=int(bench_warmup),
            testGraph=test_graph,
        )
        return us
    if test_graph:
        # Keep graph capture/replay coverage without turning a correctness case
        # into a performance loop. run_perftest needs two samples for timing.
        run_perftest(launch, *args, num_iters=2, num_warmup=0, testGraph=True)
    else:
        launch(*args)
    torch.cuda.synchronize()
    return None


def _pairwise_projection(cases):
    return {
        (left_index, left, right_index, right)
        for case in cases
        for left_index, right_index in combinations(range(len(case)), 2)
        for left, right in [(case[left_index], case[right_index])]
    }


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


_A8_DTYPES = ["fp8", "int8", "fp16", "bf16"]
_A8_SHAPES = [
    (16, 1024, 1024, 16, 32, 512),
    (16, 1024, 1024, 16, 32, 1024),
    (33, 1024, 2048, 32, 32, 512),
    (33, 1024, 2048, 32, 32, 1024),
    (16, 5120, 8192, 16, 64, 512),
    (33, 1024, 2048, 32, 64, 512),
    (5120, 5120, 8320, 64, 256, 128),
    (5120, 2048, 8320, 128, 128, 128),
    (9728, 8192, 8320, 128, 128, 128),
    (5133, 5120, 8320, 64, 256, 128),
]

# The old dtype x shape x copy-mode x launch-mode Cartesian product ran 96
# expensive GEMMs. Dtype x shape supplies the lower bound, so use that matrix
# bound and distribute sync/async plus eager/graph to retain every pairwise
# interaction. Shapes 6+ keep the original large_shape classification.
_A8_CASE_VALUES = [
    (in_dtype, shape, bool((dtype_index + shape_index) % 2), bool((dtype_index // 2 + shape_index) % 2))
    for dtype_index, in_dtype in enumerate(_A8_DTYPES)
    for shape_index, shape in enumerate(_A8_SHAPES)
]
_A8_CASES = [
    pytest.param(
        in_dtype,
        *shape,
        use_async_copy,
        test_graph,
        marks=pytest.mark.large_shape if shape in _A8_SHAPES[6:] else (),
        id=(
            f"{in_dtype}-M{shape[0]}-tile{shape[3]}x{shape[4]}x{shape[5]}-"
            f"{'async' if use_async_copy else 'sync'}-{'graph' if test_graph else 'eager'}"
        ),
    )
    for in_dtype, shape, use_async_copy, test_graph in _A8_CASE_VALUES
]


@pytest.mark.parametrize(
    "in_dtype,M,N,K,tile_m,tile_n,tile_k,use_async_copy,test_graph",
    _A8_CASES,
)
def test_mfma_a8_flyc_preshuffle(
    in_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    *,
    use_async_copy,
    test_graph,
    out_dtype: str = "bf16",
    bench_iters: int = 0,
    bench_warmup: int = 0,
    run_aiter_bench: bool = False,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
):
    """Preshuffle GEMM using the layout-API v2 kernel (fp8/int8/fp16/bf16)."""
    if use_async_copy and get_rocm_arch() != "gfx950":
        # The layout-API kernel builds the buffer_load_lds source descriptor from a
        # global (.p8.p1) make.buffer.rsrc. LLVM's CDNA3 (gfx942) backend cannot
        # legalize that into buffer_load_lds ("Do not know how to expand this
        # operator's operand!"), while CDNA4 (gfx950) handles it. Restrict async
        # copy to gfx950 until the gfx942 codegen path is supported.
        pytest.skip(f"async copy (buffer_load_lds) is only supported on gfx950, not {get_rocm_arch()}")
    if tile_n == 32 and get_rocm_arch() != "gfx950":
        pytest.skip(f"tile_n=32 two-wave path requires gfx950, not {get_rocm_arch()}")
    print("=" * 80)
    print(f"[flyc] MFMA {in_dtype.upper()} GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})")
    print("=" * 80)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    _wpe = int(waves_per_eu) if waves_per_eu else 0
    _wpe = None if _wpe <= 0 else _wpe
    launch_fn = compile_preshuffle_gemm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        waves_per_eu=_wpe,
        use_async_copy=bool(use_async_copy),
        xcd_swizzle=int(xcd_swizzle),
    )
    print(f"✓ Kernel prepared (async_copy={use_async_copy}, waves_per_eu={_wpe}, xcd_swizzle={int(xcd_swizzle)})")

    size_c = M * N
    size_a = M * K
    if in_dtype in ("fp16", "bf16"):
        size_b = (N * K) * 2
        elem_bytes = 2
    else:
        size_b = N * K
        elem_bytes = 1

    device = torch.device("cuda")
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)

    is_int8 = in_dtype == "int8"

    if in_dtype in ("fp16", "bf16"):
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q = a_fp32.to(torch_dtype)
        b_q = b_fp32_t.to(torch_dtype)
        scale_a = None
        scale_b = None
    else:
        quant_dtype = torch.int8 if is_int8 else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)
        b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    c_ref = run_torch(a_q, b_q, scale_a, scale_b, bias=None, dtype=torch.float32)
    # Allocate guard rows past M and fill with a sentinel (outputs are >= 0, so a
    # negative sentinel never collides). For ragged M, blocks covering rows
    # [M, grid_rows) must have their C stores dropped by the buffer descriptor;
    # the guard rows must stay sentinel after the run (OOB check below).
    grid_rows = ((M + tile_m - 1) // tile_m) * tile_m
    oob_sentinel = torch.tensor(-8192.0, dtype=torch_out_dtype, device=device)
    c_alloc = torch.full((grid_rows, N), oob_sentinel, dtype=torch_out_dtype, device=device)
    c_out_raw = c_alloc[:M]

    b_input = b_shuffled
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

    # Dummy bias tensor (epilogue="none" → unused by the kernel).
    _dummy_bias = torch.empty(0, dtype=torch_out_dtype, device=a_q.device)

    def _gemm_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _as_i8(a.contiguous().view(-1)),
            _as_i8(b.contiguous().view(-1)),
            sa.contiguous().view(-1) if sa.numel() > 0 else sa,
            sb.contiguous().view(-1) if sb.numel() > 0 else sb,
            _dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(launch_fn, *_gemm_args(c_out_raw, a_q, b_input, sa_flat, sb_flat))

    def launch_kernel(c, a, b, sa, sb):
        compiled_fn(*_gemm_args(c, a, b, sa, sb))

    us = _run_correctness_or_benchmark(
        launch_kernel,
        c_out_raw,
        a_q,
        b_input,
        sa_flat,
        sb_flat,
        bench_iters=bench_iters,
        bench_warmup=bench_warmup,
        test_graph=test_graph,
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    guard = c_alloc[M:]
    n_bad = int((guard != oob_sentinel).sum().item())
    assert n_bad == 0, f"C store wrote {n_bad} element(s) past row M={M} (missing num_records bound)"

    if HAS_AITER and bool(run_aiter_bench) and (in_dtype in ("fp8", "int8")):
        print("-" * 40)
        print("Running Aiter Benchmark...")
        try:

            def launch_aiter(a, b, sa, sb):
                return aiter.gemm_a8w8_bpreshuffle(a, b, sa, sb, None, torch_out_dtype)

            c_aiter, us1 = run_perftest(
                launch_aiter,
                a_q,
                b_shuffled,
                scale_a,
                scale_b,
                testGraph=test_graph,
            )
            c_aiter_f32 = c_aiter.to(torch.float32)
            verify_output(c_aiter_f32, c_ref, rtol=0.1, atol=0.1)

            bytes_moved_a = (size_a * elem_bytes) + size_b + size_c * 2 + (M + N) * 4
            flops_a = 2 * M * N * K
            tflops_aiter = flops_a / (us1 / 1e6) / 1e12
            bw_aiter = bytes_moved_a / 1e9 / (us1 / 1e6)
            print(f"Aiter Throughput: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s")
            print("-" * 40)
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            print(f"Skipping Aiter benchmark (not runnable here): {msg}")
            print("-" * 40)

    if us is not None:
        bytes_moved = (size_a * elem_bytes) + size_b + size_c * 2 + (M + N) * 4
        flops = 2 * M * N * K
        tflops = flops / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "fp16", "bf16"])
def test_v2_preshuffle_c_store_oob(in_dtype):
    """v2 layout-API kernel must not load A / store C past row M for ragged M.

    M=33 with tile_m=32 makes the last block span rows [32, 64); rows [33, 64)
    are out of bounds and must be dropped by the A/C buffer descriptors. The
    harness fills guard rows past M with a sentinel and asserts none were
    overwritten (in addition to verifying the in-range output).
    """
    if get_rocm_arch() not in ("gfx942", "gfx950"):
        pytest.skip(f"v2 preshuffle GEMM requires gfx942/gfx950, got {get_rocm_arch()}")
    test_mfma_a8_flyc_preshuffle(
        in_dtype,
        M=33,
        N=1024,
        K=2048,
        tile_m=32,
        tile_n=64,
        tile_k=512,
        use_async_copy=False,
        test_graph=False,
        run_aiter_bench=False,
    )


@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("a_dtype", ["fp4"])
@pytest.mark.parametrize("b_dtype", ["fp4"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (64, 8192, 8192, 64, 128, 128),
        (32, 8192, 8192, 32, 128, 256),
        pytest.param(128, 8192, 8192, 64, 128, 256, marks=pytest.mark.large_shape),
        pytest.param(1024, 8192, 8192, 64, 256, 256, marks=pytest.mark.large_shape),
        pytest.param(5133, 8192, 8192, 64, 256, 256, marks=pytest.mark.large_shape),
    ],
)
def test_mfma_w4_flyc_preshuffle(
    a_dtype,
    b_dtype,
    out_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    *,
    bench_iters: int = 0,
    bench_warmup: int = 0,
    waves_per_eu: int = 0,
):
    """FP4 (MXFP4) preshuffle GEMM (layout-API v2) — gfx950 only."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"FP4 GEMM requires gfx950, got {get_rocm_arch()}")
    print("=" * 80)
    print(f"MFMA MXFP4 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})")
    print("=" * 80)

    _wpe = int(waves_per_eu) if waves_per_eu else 0
    _wpe = None if _wpe <= 0 else _wpe
    launch_fn = _mxfp4_launcher(N, K, tile_m, tile_n, tile_k, out_dtype, "fp4", _wpe)
    print(f"✓ Compiled (waves_per_eu={_wpe})")

    device = torch.device("cuda")
    M_align_32 = (M + 31) // 32 * 32
    N_align_32 = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

    a_fp32_padded = torch.zeros(M_align_32, K, device=device, dtype=torch.float32)
    b_fp32_padded = torch.zeros(N_align_32, K, device=device, dtype=torch.float32)
    a_fp32_padded[:M] = a_fp32[:M]
    b_fp32_padded[:N] = b_fp32[:N]

    a_q, scale_a_orig, _ = gemm_common_utils.per_1x32_f4_quant(a_fp32_padded)
    a_q = a_q[:M]
    scale_a = gemm_common_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    b_q, scale_b, _ = gemm_common_utils.per_1x32_f4_quant(b_fp32_padded)
    b_q = b_q[:N]

    def run_torch_w4(x, w, x_scales, w_scales, dtype):
        x_f32 = gemm_common_utils.mxfp4_to_f32(x)
        w_f32 = gemm_common_utils.mxfp4_to_f32(w)
        x_scales_f32 = gemm_common_utils.e8m0_to_f32(x_scales[: x.shape[0]].repeat_interleave(32, dim=1))
        w_scales_f32 = gemm_common_utils.e8m0_to_f32(w_scales[: w.shape[0]].repeat_interleave(32, dim=1))
        return torch.mm(x_f32 * x_scales_f32, (w_f32 * w_scales_f32).T).to(dtype)

    c_ref = run_torch_w4(a_q, b_q, scale_a_orig, scale_b, torch.float32)

    b_shuffled = gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuffled = gemm_common_utils.shuffle_scale_w4(scale_b, 1, False)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)

    def _to_bytes(t):
        if t.dtype == torch.uint8 or t.dtype == torch.int8:
            return t
        return t.view(torch.uint8)

    # Create a dummy bias tensor (unused when epilogue="none")
    _dummy_bias_w4 = torch.empty(0, dtype=torch.bfloat16, device=a_q.device)

    def _w4_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _to_bytes(a).contiguous().view(-1),
            _to_bytes(b).contiguous().view(-1),
            _to_bytes(sa).contiguous().view(-1),
            _to_bytes(sb).contiguous().view(-1),
            _dummy_bias_w4,
            M,
            N,
            torch.cuda.current_stream(),
        )

    def launch_kernel(c, a, b, sa, sb):
        launch_fn(*_w4_args(c, a, b, sa, sb))

    us = _run_correctness_or_benchmark(
        launch_kernel,
        c_out,
        a_q,
        b_shuffled,
        scale_a,
        scale_b_shuffled,
        bench_iters=bench_iters,
        bench_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
    c_out_f32 = c_out.to(torch.float32)

    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    if us is not None:
        size_a = (M * K) // 2
        size_b = (N * K) // 2
        size_c = M * N
        bytes_moved = size_a + size_b + size_c * 2 + (M + N) * (K // 32)
        flops = 2 * M * N * K
        tflops = flops / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


# ── W4A6: MXFP6 (E2M3) A × MXFP4 (E2M1) B ─────────────────────────────────


@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    [
        (64, 8192, 8192, 64, 128, 128),
        (32, 8192, 8192, 32, 128, 256),
        pytest.param(128, 8192, 8192, 64, 128, 256, marks=pytest.mark.large_shape),
        pytest.param(1024, 8192, 8192, 64, 256, 256, marks=pytest.mark.large_shape),
        pytest.param(256, 4096, 14336, 128, 256, 256, marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mfma_a6w4_preshuffle(
    out_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    *,
    bench_iters: int = 0,
    bench_warmup: int = 0,
    waves_per_eu: int = 0,
):
    """W4A6: MXFP6 (E2M3) A × MXFP4 (E2M1) B preshuffle GEMM — gfx950 only."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"FP6/FP4 GEMM requires gfx950, got {get_rocm_arch()}")

    print("=" * 80)
    print(f"MFMA W4A6 (MXFP6 A × MXFP4 B) GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})")
    print("=" * 80)

    _wpe = int(waves_per_eu) if waves_per_eu else 0
    _wpe = None if _wpe <= 0 else _wpe
    launch_fn = _mxfp4_launcher(N, K, tile_m, tile_n, tile_k, out_dtype, "fp6", _wpe)
    print(f"✓ Compiled (waves_per_eu={_wpe})")

    device = torch.device("cuda")
    M_align_32 = (M + 31) // 32 * 32
    N_align_32 = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
    a_fp32_padded = torch.zeros(M_align_32, K, device=device, dtype=torch.float32)
    b_fp32_padded = torch.zeros(N_align_32, K, device=device, dtype=torch.float32)
    a_fp32_padded[:M] = a_fp32
    b_fp32_padded[:N] = b_fp32

    # A: MXFP6 E2M3, FP8-padded (1 byte/code).
    a_pad, scale_a_orig, a_unpacked = gemm_common_utils.per_1x32_f6_quant(a_fp32_padded)
    a_codes = a_pad[:M]
    scale_a = gemm_common_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    # B: MXFP4 E2M1, identical to test_mfma_w4_flyc_preshuffle.
    b_q, scale_b, _ = gemm_common_utils.per_1x32_f4_quant(b_fp32_padded)
    b_q = b_q[:N]
    b_shuffled = gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuffled = gemm_common_utils.shuffle_scale_w4(scale_b, 1, False)

    # Reference: dequant(A) @ dequant(B).T in fp32.
    a_deq = gemm_common_utils.fp6_e2m3_to_f32(a_unpacked) * gemm_common_utils.e8m0_to_f32(
        scale_a_orig[:M].repeat_interleave(32, dim=1)
    )
    b_deq = gemm_common_utils.mxfp4_to_f32(b_q) * gemm_common_utils.e8m0_to_f32(
        scale_b[:N].repeat_interleave(32, dim=1)
    )
    c_ref = torch.mm(a_deq, b_deq.T).to(torch.float32)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)
    _dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _to_bytes(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    def _a6w4_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _to_bytes(a).contiguous().view(-1),
            _to_bytes(b).contiguous().view(-1),
            _to_bytes(sa).contiguous().view(-1),
            _to_bytes(sb).contiguous().view(-1),
            _dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    def launch_kernel(c, a, b, sa, sb):
        launch_fn(*_a6w4_args(c, a, b, sa, sb))

    us = _run_correctness_or_benchmark(
        launch_kernel,
        c_out,
        a_codes,
        b_shuffled,
        scale_a,
        scale_b_shuffled,
        bench_iters=bench_iters,
        bench_warmup=bench_warmup,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    # A: 1 byte/code (FP8-padded); B: 0.5 byte/code (MXFP4).
    if us is not None:
        bytes_moved = M * K + (N * K) // 2 + M * N * 2 + (M + N) * (K // 32)
        tflops = (2 * M * N * K) / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(f"[flyc] W4A6 Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


# ── A8W8: MXFP8 (E4M3) A × MXFP8 (E4M3) B ─────────────────────────────────


_A8W8_OUT_DTYPES = ["bf16", "fp16"]
_A8W8_SHAPES = [
    (64, 8192, 8192, 64, 128, 128),
    (32, 8192, 8192, 32, 128, 256),
    (128, 8192, 8192, 64, 128, 256),
    (1024, 8192, 8192, 64, 256, 256),
]
_A8W8_XCD_SWIZZLES = [0, 4]
_A8W8_CASE_VALUES = [
    (out_dtype, shape, _A8W8_XCD_SWIZZLES[(out_index + shape_index) % 2])
    for out_index, out_dtype in enumerate(_A8W8_OUT_DTYPES)
    for shape_index, shape in enumerate(_A8W8_SHAPES)
]
_A8W8_CASES = [
    pytest.param(
        out_dtype,
        *shape,
        xcd_swizzle,
        marks=pytest.mark.large_shape if shape in _A8W8_SHAPES[2:] else (),
        id=f"{out_dtype}-M{shape[0]}-tile{shape[3]}x{shape[4]}x{shape[5]}-xcd{xcd_swizzle}",
    )
    for out_dtype, shape, xcd_swizzle in _A8W8_CASE_VALUES
]


def test_preshuffle_case_lists_preserve_pairwise_coverage():
    """Prevent matrix edits from silently dropping a two-factor interaction."""
    a8_expected = _pairwise_projection(product(_A8_DTYPES, _A8_SHAPES, [False, True], [False, True]))
    a8_missing = a8_expected - _pairwise_projection(_A8_CASE_VALUES)
    assert not a8_missing, f"A8 matrix is missing {sorted(a8_missing, key=repr)}"

    a8w8_expected = _pairwise_projection(product(_A8W8_OUT_DTYPES, _A8W8_SHAPES, _A8W8_XCD_SWIZZLES))
    a8w8_missing = a8w8_expected - _pairwise_projection(_A8W8_CASE_VALUES)
    assert not a8w8_missing, f"A8W8 matrix is missing {sorted(a8w8_missing, key=repr)}"


@pytest.mark.parametrize(
    "out_dtype,M,N,K,tile_m,tile_n,tile_k,xcd_swizzle",
    _A8W8_CASES,
)
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mfma_a8w8_preshuffle(
    out_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    xcd_swizzle,
    *,
    bench_iters: int = 0,
    bench_warmup: int = 0,
    waves_per_eu: int = 0,
    a_scale_layout: str = "mfma_tile",
    b_scale_layout: str = "mfma_tile",
    prefetch: int = 1,
    b_nt: bool = False,
):
    """A8W8: MXFP8 (E4M3) A × MXFP8 (E4M3) B preshuffle GEMM — gfx950 only.

    A "row" scale layout passes that operand's unshuffled [rows, K // 32] e8m0 scale."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"FP8 MX GEMM requires gfx950, got {get_rocm_arch()}")

    print("=" * 80)
    print(f"MFMA A8W8 (MXFP8 A × MXFP8 B) GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})")
    print("=" * 80)

    _wpe = int(waves_per_eu) if waves_per_eu else 0
    _wpe = None if _wpe <= 0 else _wpe
    launch_fn = _mxfp4_launcher(
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        out_dtype,
        "fp8",
        _wpe,
        b_dtype="fp8",
        xcd_swizzle=xcd_swizzle,
        a_scale_layout=a_scale_layout,
        b_scale_layout=b_scale_layout,
        prefetch=prefetch,
        b_nt=b_nt,
    )
    print(f"✓ Compiled (waves_per_eu={_wpe})")

    device = torch.device("cuda")
    M_align_32 = (M + 31) // 32 * 32
    N_align_32 = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
    a_fp32_padded = torch.zeros(M_align_32, K, device=device, dtype=torch.float32)
    b_fp32_padded = torch.zeros(N_align_32, K, device=device, dtype=torch.float32)
    a_fp32_padded[:M] = a_fp32
    b_fp32_padded[:N] = b_fp32

    # A: MXFP8 E4M3, 1 byte/code, row-major (not preshuffled). B: MXFP8, preshuffled 16x16.
    a_q, scale_a_orig = gemm_common_utils.per_1x32_f8_quant(a_fp32_padded)
    a_codes = a_q[:M]
    if a_scale_layout == "row":
        scale_a = scale_a_orig[:M].contiguous()
    else:
        scale_a = gemm_common_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    w_q, scale_b_orig = gemm_common_utils.per_1x32_f8_quant(b_fp32_padded)
    w_q = w_q[:N]
    b_shuffled = gemm_common_utils.shuffle_weight_w4(w_q, 16, False, False)
    if b_scale_layout == "row":
        scale_b_shuffled = scale_b_orig[:N].contiguous()
    else:
        scale_b_shuffled = gemm_common_utils.shuffle_scale_w4(scale_b_orig, 1, False)

    # Reference: dequant(A) @ dequant(B).T in fp32 (torch decodes fp8 natively).
    a_deq = a_codes.float() * gemm_common_utils.e8m0_to_f32(scale_a_orig[:M].repeat_interleave(32, dim=1))
    b_deq = w_q.float() * gemm_common_utils.e8m0_to_f32(scale_b_orig[:N].repeat_interleave(32, dim=1))
    c_ref = torch.mm(a_deq, b_deq.T).to(torch.float32)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)
    _dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _to_bytes(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    def _a8w8_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _to_bytes(a).contiguous().view(-1),
            _to_bytes(b).contiguous().view(-1),
            _to_bytes(sa).contiguous().view(-1),
            _to_bytes(sb).contiguous().view(-1),
            _dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    def launch_kernel(c, a, b, sa, sb):
        launch_fn(*_a8w8_args(c, a, b, sa, sb))

    us = _run_correctness_or_benchmark(
        launch_kernel,
        c_out,
        a_codes,
        b_shuffled,
        scale_a,
        scale_b_shuffled,
        bench_iters=bench_iters,
        bench_warmup=bench_warmup,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    # A and B: 1 byte/code; scales: (M+N) * (K//32) bytes.
    if us is not None:
        bytes_moved = M * K + N * K + M * N * 2 + (M + N) * (K // 32)
        tflops = (2 * M * N * K) / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(f"[flyc] A8W8 Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


_ROW = dict(a_scale_layout="row", b_scale_layout="row")
# Non-default a8w8 paths, one row per path: (M, N, K, tile_m, tile_n, tile_k, launch kwargs).
_A8W8_VARIANTS = [
    # Row-major [rows, K // 32] scales staged through LDS.
    ("row", (32, 1024, 2048, 32, 128, 256), _ROW),
    ("row-raggedM-tk128", (100, 1024, 2048, 64, 128, 128), _ROW),  # 128-K tiles share a scale word
    ("rowA-k5tiles", (256, 2048, 1280, 64, 256, 256), dict(a_scale_layout="row")),  # not a multiple of the chunk
    ("rowB-tk128", (100, 1024, 2048, 64, 128, 128), dict(b_scale_layout="row")),
    ("row-tm16-m1", (1, 1024, 2048, 16, 64, 256), _ROW),  # a 16-row tile (M <= 16), one accumulator
    ("rowA-tm16-m9", (9, 1024, 2048, 16, 128, 256), dict(a_scale_layout="row")),
    ("row-tk512", (64, 1024, 4096, 64, 128, 512), _ROW),  # two scale words per row pair
    # Prefetch pipeline: B and the row scales loaded prefetch - 1 K tiles ahead.
    ("pf2", (32, 1024, 2048, 32, 128, 256), dict(_ROW, prefetch=2)),
    ("pf3-raggedM", (100, 1024, 2048, 64, 128, 128), dict(_ROW, prefetch=3)),
    ("pf3-k5tiles", (256, 2048, 1280, 64, 128, 256), dict(_ROW, prefetch=3)),  # steady loop, remainder, peel
    ("pf4-k2tiles", (64, 512, 512, 32, 64, 256), dict(_ROW, prefetch=4)),  # fewer K tiles than the depth
    ("pf3-tk512", (32, 1024, 4096, 32, 64, 512), dict(_ROW, prefetch=3)),
    # b_nt: a single M tile streams B non-temporal.
    ("pf2-bnt", (32, 1024, 2048, 32, 128, 256), dict(_ROW, prefetch=2, b_nt=True)),
    ("pf3-bnt-tm16", (16, 1024, 2048, 16, 64, 256), dict(_ROW, prefetch=3, b_nt=True)),
    # The depth prefetch_depth picks (per tile height, capped by vmcnt and LDS) builds and runs.
    ("pfauto-tm32", (32, 7168, 2048, 32, 128, 256), dict(_ROW, prefetch=prefetch_depth(32, 128, 256, 7168, 2048))),
    ("pfauto-tm64", (128, 5120, 5120, 64, 256, 256), dict(_ROW, prefetch=prefetch_depth(64, 256, 256, 5120, 5120))),
    ("pfauto-tm192", (192, 2048, 4096, 192, 128, 128), dict(_ROW, prefetch=prefetch_depth(192, 128, 128, 2048, 4096))),
]


@pytest.mark.parametrize("shape,kwargs", [pytest.param(sh, kw, id=name) for name, sh, kw in _A8W8_VARIANTS])
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mfma_a8w8_preshuffle_variants(shape, kwargs):
    """A8W8 non-default launch paths (bf16 out, no XCD swizzle)."""
    test_mfma_a8w8_preshuffle("bf16", *shape, 0, **kwargs)


def _a8w8_row_operands(M, N, K):
    """fp8 A, preshuffled fp8 B and row-layout e8m0 scales, plus the fp32 reference."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    a_q, sa = gemm_common_utils.per_1x32_f8_quant(torch.randn(M, K, device=device))
    w_q, sb = gemm_common_utils.per_1x32_f8_quant(torch.randn(N, K, device=device))
    a_deq = a_q.float() * gemm_common_utils.e8m0_to_f32(sa.repeat_interleave(32, dim=1))
    b_deq = w_q.float() * gemm_common_utils.e8m0_to_f32(sb.repeat_interleave(32, dim=1))
    b = gemm_common_utils.shuffle_weight_w4(w_q, 16, False, False)
    return a_q, b, sa.contiguous(), sb.contiguous(), torch.mm(a_deq, b_deq.T)


def _a8w8_blockscale_operands(M, N, K):
    """fp8 A (1x128 scales, per-block jitter), preshuffled fp8 B (128x128 scales) in the
    shuffled blockscale scale order, plus the fp32 reference."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    scale_k = K // SCALE_BLOCK_128
    a_fp32 = torch.randn(M, K, device=device)
    a_fp32 = (a_fp32.view(M, scale_k, SCALE_BLOCK_128) * _blk_jitter((M, scale_k, 1), device)).view(M, K)
    a_q, scale_a = gemm_common_utils.per_block_f8_quant(a_fp32, 1, SCALE_BLOCK_128)
    b_q, scale_b = gemm_common_utils.per_block_f8_quant(
        torch.randn(N, K, device=device), SCALE_BLOCK_128, SCALE_BLOCK_128
    )
    a_deq = a_q.float().view(M, scale_k, SCALE_BLOCK_128) * gemm_common_utils.e8m0_to_f32(scale_a).unsqueeze(-1)
    b_sc_rows = gemm_common_utils.e8m0_to_f32(scale_b).repeat_interleave(SCALE_BLOCK_128, dim=0)
    b_deq = b_q.float().view(N, scale_k, SCALE_BLOCK_128) * b_sc_rows.unsqueeze(-1)
    c_ref = torch.mm(a_deq.view(M, K), b_deq.view(N, K).T)
    b = gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa = gemm_common_utils.shuffle_scale_blockscale_a(scale_a, K)
    sb = gemm_common_utils.shuffle_scale_blockscale_b(scale_b, N, K)
    return a_q, b, sa.view(torch.uint8), sb.view(torch.uint8), c_ref


# launch_gemm_fused paths: (M, N, K, tile_m, tile_n, tile_k, k_batch, launch kwargs).
_A8W8_FUSED_VARIANTS = [
    ("sk4", (32, 1024, 4096, 32, 128, 256, 4), {}),
    ("sk2-raggedM", (100, 512, 2048, 64, 64, 128, 2), {}),  # the last tile reduces fewer rows
    ("sk8-pf3", (64, 2048, 8192, 32, 128, 256, 8), dict(prefetch=3)),
    ("sk2-10tiles", (32, 640, 2048, 32, 64, 256, 2), {}),
    # l2_reduce: a tile's splits 8 pids apart on one XCD.
    ("sk8-pf3-l2", (64, 2048, 8192, 32, 128, 256, 8), dict(prefetch=3, l2_reduce=True)),
    ("sk2-10tiles-l2", (32, 640, 2048, 32, 64, 256, 2), dict(l2_reduce=True)),  # a partial last group of 8
    # 16-row bf16 blockscale tile: bf16 partials; single_row stores only row 0 and wave 0 reduces.
    ("bs-tm16-m1-sk4-single", (1, 1024, 4096, 16, 128, 256, 4), dict(blockscale="ab", single_row=True)),
    ("bs-tm16-m7-sk2", (7, 1024, 4096, 16, 64, 256, 2), dict(blockscale="ab")),  # every wave reduces
]


@pytest.mark.parametrize("shape,kwargs", [pytest.param(sh, kw, id=name) for name, sh, kw in _A8W8_FUSED_VARIANTS])
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mfma_a8w8_preshuffle_fused_split_k(shape, kwargs):
    """fused_split_k: the last block of each output tile reduces the fp32 partials in-kernel."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"FP8 MX GEMM requires gfx950, got {get_rocm_arch()}")
    M, N, K, tile_m, tile_n, tile_k, k_batch = shape
    blockscale = kwargs.get("blockscale", "none")
    if blockscale == "none":
        a, b, sa, sb, c_ref = _a8w8_row_operands(M, N, K)
        scale_layout, ws_dtype = "row", torch.float32
    else:
        a, b, sa, sb, c_ref = _a8w8_blockscale_operands(M, N, K)
        scale_layout, ws_dtype = "mfma_tile", torch.bfloat16
    ws = torch.empty((k_batch, M, N), dtype=ws_dtype, device="cuda")
    out = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    n_tiles = -(-M // tile_m) * (N // tile_n)
    sem = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")
    args = (
        _ptr(ws),
        _ptr(a.view(torch.uint8)),
        _ptr(b.view(torch.uint8)),
        _ptr(sa),
        _ptr(sb),
        _ptr(out),
        _ptr(sem),
        M,
        N,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        "fp8",
        "bf16",
        "fp8",
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        k_batch,
        blockscale,
        scale_layout,
        scale_layout,
        kwargs.get("prefetch", 1),
        kwargs.get("l2_reduce", False),
        kwargs.get("single_row", False),
    )
    exe = flyc.compile(launch_gemm_fused, *args)  # compiles and launches once
    torch.cuda.synchronize()
    assert verify_output(out.float(), c_ref, rtol=0.1, atol=0.1)
    assert not sem.any(), "the reducing block must re-zero its tile counter"
    # A second launch runs on the counters the first one left.
    out.zero_()
    exe(*args)
    torch.cuda.synchronize()
    assert verify_output(out.float(), c_ref, rtol=0.1, atol=0.1)
    assert not sem.any()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preshuffle GEMM benchmark")
    parser.add_argument("--in_dtype", type=str, default="fp8", choices=["fp8", "int8", "fp16", "bf16", "fp4"])
    parser.add_argument(
        "--out_dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Output dtype (default: bf16)."
    )
    parser.add_argument("-M", type=int, default=16)
    parser.add_argument("-N", type=int, default=10240)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=16)
    parser.add_argument("--tile_n", type=int, default=64)
    parser.add_argument("--tile_k", type=int, default=256)
    parser.add_argument("--lds_stage", type=int, default=DEFAULT_LDS_STAGE, choices=[1, 2])
    parser.add_argument("--dsrd_preload", type=int, default=2)
    parser.add_argument("--dvmem_preload", type=int, default=2)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--flyc", action="store_true", default=True)
    parser.add_argument("--use_async_copy", action="store_true", default=False)
    parser.add_argument("--use_cshuffle_epilog", action="store_true", default=False)
    parser.add_argument("--waves_per_eu", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--xcd_swizzle", type=int, default=0, help="XCD L2-rasterization group size (0 = off).")
    parser.add_argument("--run_aiter_bench", action="store_true", default=DEFAULT_RUN_AITER_BENCH)
    parser.add_argument("--no_aiter_bench", action="store_false", dest="run_aiter_bench")
    parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
    parser.add_argument(
        "--wfp4", action="store_true", default=False, help="Run weight-fp4 (MXFP4) preshuffle GEMM test."
    )
    args = parser.parse_args()
    torch.set_default_device("cuda")
    try:
        if not args.wfp4:
            if args.in_dtype == "fp4":
                raise ValueError("--in_dtype fp4 requires --wfp4")
            test_mfma_a8_flyc_preshuffle(
                args.in_dtype,
                M=args.M,
                N=args.N,
                K=args.K,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                tile_k=args.tile_k,
                out_dtype=args.out_dtype,
                use_async_copy=bool(args.use_async_copy),
                test_graph=bool(args.test_graph),
                bench_iters=args.num_iters,
                bench_warmup=args.num_warmup,
                run_aiter_bench=bool(args.run_aiter_bench),
                waves_per_eu=int(args.waves_per_eu),
                xcd_swizzle=int(args.xcd_swizzle),
            )
        else:
            test_mfma_w4_flyc_preshuffle(
                "fp4",
                "fp4",
                args.out_dtype,
                M=args.M,
                N=args.N,
                K=args.K,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                tile_k=args.tile_k,
                bench_iters=args.num_iters,
                bench_warmup=args.num_warmup,
                waves_per_eu=int(args.waves_per_eu),
                use_async_copy=bool(args.use_async_copy),
            )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")


# ── CUDAGraph Capture Test ────────────────────────────────────────────────


@pytest.mark.parametrize("in_dtype", ["bf16", "fp8"])
def test_cudagraph_capture_preshuffle(in_dtype):
    """Verify FlyDSL preshuffle GEMM kernels are captured by CUDAGraph.

    This test ensures that passing torch.cuda.current_stream() correctly
    routes the kernel launch to the capture stream during graph recording.
    Without proper stream handling, CUDAGraph replay produces all-zeros.
    """
    device = "cuda:0"
    M, N, K = 1, 8192, 8192
    tile_m, tile_n, tile_k = 16, 64, 256

    arch = str(get_rocm_arch())
    if not arch.startswith("gfx94") and not arch.startswith("gfx95"):
        pytest.skip(f"Unsupported arch: {arch}")

    # Prepare data
    a_raw = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b_raw = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    if in_dtype == "fp8":
        a_q, scale_a = pertoken_quant(a_raw, quant_dtype=DTYPE_FP8)
        b_q, scale_b = pertoken_quant(b_raw, quant_dtype=DTYPE_FP8)
        a_q = a_q.view(torch.int8)
        b_input = shuffle_weight(b_q.view(torch.int8), layout=(16, 16)).contiguous().view(-1)
        sa_flat = scale_a.contiguous().view(-1)
        sb_flat = scale_b.contiguous().view(-1)
    else:
        a_q = a_raw
        b_input = shuffle_weight(b_raw.contiguous(), layout=(16, 16)).contiguous().view(-1)
        sa_flat = torch.empty(0, dtype=torch.float32, device=device)
        sb_flat = torch.empty(0, dtype=torch.float32, device=device)

    c_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    _dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    # Compile kernel
    launch_fn = compile_preshuffle_gemm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
    )

    def _args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            a.contiguous().view(-1) if "int" not in str(a.dtype) else a.contiguous().view(-1),
            b,
            sa.contiguous().view(-1) if sa.numel() > 0 else sa,
            sb.contiguous().view(-1) if sb.numel() > 0 else sb,
            _dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(launch_fn, *_args(c_out, a_q, b_input, sa_flat, sb_flat))

    # Warmup
    compiled_fn(*_args(c_out, a_q, b_input, sa_flat, sb_flat))
    torch.cuda.synchronize()

    # ── Regular execution (reference) ──
    c_out.zero_()
    compiled_fn(*_args(c_out, a_q, b_input, sa_flat, sb_flat))
    torch.cuda.synchronize()
    ref = c_out.clone()
    assert ref.abs().max().item() > 0, "Regular execution produced all zeros"

    # ── CUDAGraph capture ──
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    # Warmup on capture stream
    with torch.cuda.stream(s):
        compiled_fn(*_args(c_out, a_q, b_input, sa_flat, sb_flat))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Record
    c_out.zero_()
    with torch.cuda.graph(g, stream=s):
        compiled_fn(*_args(c_out, a_q, b_input, sa_flat, sb_flat))
    torch.cuda.synchronize()

    # ── Replay ──
    c_out.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_result = c_out.clone()

    # ── Verify ──
    max_diff = (ref - graph_result).abs().max().item()
    assert (
        graph_result.abs().max().item() > 0
    ), f"CUDAGraph replay produced all zeros — kernel was NOT captured! ref max={ref.abs().max().item():.4f}"
    assert torch.allclose(ref, graph_result, atol=1e-2), (
        f"CUDAGraph result mismatch: max_diff={max_diff:.6f}, "
        f"ref max={ref.abs().max().item():.4f}, graph max={graph_result.abs().max().item():.4f}"
    )
    print(f"✓ CUDAGraph capture verified ({in_dtype}): max_diff={max_diff:.6f}")


# ── Fused epilogue correctness test ─────────────────────────────────────────


@pytest.mark.parametrize("epilogue", ["bias", "bias_relu", "bias_silu", "bias_gelu"])
def test_fused_epilogue_correctness(epilogue):
    """Verify fused epilogue (bias + activation) matches a torch reference.

    The previous test suite only exercised epilogue='none' with a dummy bias
    tensor, so a regression in body_row's fused bias/activation path would
    not have been caught. This test runs each of the four epilogue modes
    end-to-end and compares against a torch reference.
    """

    arch = str(get_rocm_arch())
    if not arch.startswith("gfx94") and not arch.startswith("gfx95"):
        pytest.skip(f"Unsupported arch: {arch}")

    device = "cuda:0"
    M, N, K = 16, 5120, 8192
    tile_m, tile_n, tile_k = 16, 64, 512
    in_dtype = "bf16"
    out_dtype = "bf16"
    torch_out_dtype = torch.bfloat16

    torch.manual_seed(0)
    a_raw = torch.randn(M, K, dtype=torch_out_dtype, device=device)
    b_raw = torch.randn(N, K, dtype=torch_out_dtype, device=device)
    bias = torch.randn(N, dtype=torch_out_dtype, device=device)

    # Torch reference: GEMM + bias + activation
    a_f32 = a_raw.to(torch.float32)
    b_f32 = b_raw.to(torch.float32)
    ref_f32 = a_f32 @ b_f32.T + bias.to(torch.float32)
    if epilogue == "bias_relu":
        ref_f32 = F.relu(ref_f32)
    elif epilogue == "bias_silu":
        ref_f32 = F.silu(ref_f32)
    elif epilogue == "bias_gelu":
        ref_f32 = F.gelu(ref_f32, approximate="tanh")
    ref = ref_f32.to(torch_out_dtype)

    # FlyDSL kernel
    b_input = shuffle_weight(b_raw.contiguous(), layout=(16, 16)).contiguous().view(-1)
    sa_flat = torch.empty(0, dtype=torch.float32, device=device)
    sb_flat = torch.empty(0, dtype=torch.float32, device=device)
    c_out = torch.zeros(M, N, dtype=torch_out_dtype, device=device)

    launch_fn = compile_preshuffle_gemm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        epilogue=epilogue,
    )

    def _args(c, a, b, sa, sb, bs):
        return (
            c.contiguous().view(-1),
            a.contiguous().view(-1),
            b,
            sa.contiguous().view(-1) if sa.numel() > 0 else sa,
            sb.contiguous().view(-1) if sb.numel() > 0 else sb,
            bs,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(launch_fn, *_args(c_out, a_raw, b_input, sa_flat, sb_flat, bias))
    compiled_fn(*_args(c_out, a_raw, b_input, sa_flat, sb_flat, bias))
    torch.cuda.synchronize()

    # bf16 has ~7 bits mantissa; for K=8192 reduction the per-element
    # error is bounded by ~K * eps_bf16 ~ 8192 * 2^-7 ~= 64 ULP. We use
    # rtol=0.05 (5%) and atol=2.0 (covers small-magnitude outputs).
    assert not torch.isnan(
        c_out
    ).any(), f"Epilogue {epilogue}: kernel produced NaN(s) (count={int(torch.isnan(c_out).sum().item())})"
    assert not torch.isinf(c_out).any(), f"Epilogue {epilogue}: kernel produced Inf(s)"
    atol = 2.0
    rtol = 0.05
    diff = (c_out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    rel = (diff / (ref.to(torch.float32).abs() + 1e-3)).max().item()
    assert torch.allclose(c_out, ref, atol=atol, rtol=rtol), (
        f"Epilogue {epilogue} mismatch: max_abs_diff={max_diff:.4f} max_rel={rel:.4f}, "
        f"ref max={ref.abs().max().item():.4f}, out max={c_out.abs().max().item():.4f}"
    )
    print(
        f"✓ Fused epilogue {epilogue} correctness verified: "
        f"max_abs_diff={max_diff:.4f}, max_rel={rel:.4f}, ref_max={ref.abs().max().item():.2f}"
    )


@pytest.mark.parametrize("tile_m", [48, 80, 112, 144])
def test_preshuffle_rejects_partial_a_tile(tile_m):
    """Reject tile_m values that leave a partial 16B A load per thread.

    Each of the 256 threads copies tile_m * tile_k * elem_bytes / 256 bytes of the A
    tile in 16B chunks. When that is not a multiple of 16 the chunk count truncates and
    the tail of the A tile is never fetched, which used to yield wrong results with no
    diagnostic. For bf16 with tile_k=64 the product must be a multiple of 4096.
    """
    if get_rocm_arch() not in ("gfx942", "gfx950"):
        pytest.skip(f"v2 preshuffle GEMM requires gfx942/gfx950, got {get_rocm_arch()}")
    with pytest.raises(ValueError, match=r"must be a multiple of"):
        compile_preshuffle_gemm(N=1024, K=2048, tile_m=tile_m, tile_n=256, tile_k=64, in_dtype="bf16", out_dtype="bf16")


@pytest.mark.parametrize("tile_m", [64, 96, 128, 160])
def test_preshuffle_accepts_whole_a_tile(tile_m):
    """tile_m values that divide the A tile evenly across threads must still compile."""
    if get_rocm_arch() not in ("gfx942", "gfx950"):
        pytest.skip(f"v2 preshuffle GEMM requires gfx942/gfx950, got {get_rocm_arch()}")
    compile_preshuffle_gemm(N=1024, K=2048, tile_m=tile_m, tile_n=256, tile_k=64, in_dtype="bf16", out_dtype="bf16")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"N": 48, "K": 512, "tile_m": 16, "tile_n": 32, "tile_k": 512}, "N must be divisible"),
        ({"N": 64, "K": 512, "tile_m": 20, "tile_n": 32, "tile_k": 512}, "tile_m must be"),
        ({"N": 64, "K": 512, "tile_m": 16, "tile_n": 32, "tile_k": 32}, "MMA step"),
    ],
)
def test_preshuffle_rejects_truncated_tile_geometry(kwargs, match):
    """Reject tile geometries that would silently leave rows, columns, or K work undone."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"tile_n=32 two-wave path requires gfx950, got {get_rocm_arch()}")
    with pytest.raises(ValueError, match=match):
        compile_preshuffle_gemm(in_dtype="int8", out_dtype="bf16", **kwargs)


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tile_k", [64, 128])
def test_preshuffle_async_copy_2byte_dtypes(in_dtype, tile_k):
    """The gmem->LDS DMA must match the sync path for 2-byte input dtypes.

    tile_k is parametrized because the DMA's LDS swizzle width derives from
    ``tile_k * elem_bytes``, so tile_k=128 exercises a different swizzle than the
    tile_k=64 case that the fixed Swizzle<3,3,3> happened to match.
    """
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"async copy (buffer_load_lds) requires gfx950, got {get_rocm_arch()}")
    test_mfma_a8_flyc_preshuffle(
        in_dtype,
        M=512,
        N=1024,
        K=2048,
        tile_m=128,
        tile_n=256,
        tile_k=tile_k,
        use_async_copy=True,
        test_graph=False,
        run_aiter_bench=False,
    )


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3), (-1, 0), (0, -1), 4])
def test_preshuffle_preload_validation(bad):
    """preload is a public knob, so reject malformed values instead of emitting bad hints."""
    if get_rocm_arch() not in ("gfx942", "gfx950"):
        pytest.skip(f"v2 preshuffle GEMM requires gfx942/gfx950, got {get_rocm_arch()}")
    with pytest.raises((ValueError, TypeError)):
        compile_preshuffle_gemm(
            N=1024, K=2048, tile_m=128, tile_n=256, tile_k=64, in_dtype="bf16", out_dtype="bf16", preload=bad
        )


# ── A8W8 blockscale: coarse E8M0 grid (A per 1x128, B per 128x128) ──────────

SCALE_BLOCK_128 = 128


def _blk_jitter(shape, device):
    """Random power-of-2 magnitude per scale block, so block E8M0 scales actually differ."""
    return torch.exp2(torch.randint(-2, 3, shape, device=device, dtype=torch.float32))


# Coarse E8M0 scale paths: (M, N, K, tile_m, tile_n, tile_k), A block, B block, scale layout, launch kwargs.
# "mfma_tile" takes the coarse scales in the kernel's compact blockscale order (1x32 ones
# shuffle_scale_w4'd); "row" takes them as the quantizer emits them (a 1x128 A column-major).
_A8W8_COARSE_CASES = [
    ("ab", (64, 8192, 8192, 64, 128, 128), "1x128", "128x128", "mfma_tile", {}),
    ("a", (64, 8192, 8192, 64, 128, 128), "1x128", "1x32", "mfma_tile", {}),
    ("b", (64, 8192, 8192, 64, 128, 128), "1x32", "128x128", "mfma_tile", {}),
    ("ab-tm16", (13, 1024, 2048, 16, 128, 256), "1x128", "128x128", "mfma_tile", {}),  # M <= 16
    ("ab-tm16-m1-single", (1, 1024, 2048, 16, 64, 256), "1x128", "128x128", "mfma_tile", dict(single_row=True)),
    ("row-a", (100, 1024, 2048, 64, 256, 256), "1x128", "1x32", "row", {}),
    ("row-b32", (32, 512, 2048, 32, 128, 128), "1x32", "32x32", "row", {}),
    ("row-b128", (100, 1024, 2048, 64, 256, 256), "1x32", "128x128", "row", {}),
    ("row-ab-tm16-tk512", (9, 1024, 4096, 16, 128, 512), "1x128", "128x128", "row", {}),
    # The 32x32 B rides the 1x32 panels, so the prefetch pipeline takes it too.
    ("row-b32-pf3", (100, 1024, 2048, 64, 256, 256), "1x32", "32x32", "row", dict(prefetch=3)),
    ("row-b32-tm16-tk512-pf3", (9, 1024, 4096, 16, 128, 512), "1x32", "32x32", "row", dict(prefetch=3)),
]


@pytest.mark.parametrize("out_dtype", ["bf16"])
@pytest.mark.parametrize(
    "shape,a_block,b_block,scale_layout,kwargs",
    [pytest.param(sh, ab, bb, lay, kw, id=name) for name, sh, ab, bb, lay, kw in _A8W8_COARSE_CASES],
)
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mfma_a8w8_preshuffle_blockscale(
    out_dtype,
    shape,
    a_block,
    b_block,
    scale_layout,
    kwargs,
    *,
    bench_iters: int = 0,
    bench_warmup: int = 0,
    waves_per_eu: int = 0,
):
    """A8W8 with coarse E8M0 scales driven from randn fp32 operands: A per 1x128 and/or B per
    128x128 (blockscale "a" / "b" / "ab") or 32x32, the other operand per 1x32."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"FP8 MX GEMM requires gfx950, got {get_rocm_arch()}")
    M, N, K, tile_m, tile_n, tile_k = shape
    # blockscale is a8w8-only and needs N%128==0 (aiter host enforces the same).
    assert N % SCALE_BLOCK_128 == 0 and K % SCALE_BLOCK_128 == 0
    blockscale = {(True, True): "ab", (True, False): "a", (False, True): "b", (False, False): "none"}[
        (a_block == "1x128", b_block == "128x128")
    ]

    print("=" * 80)
    print(f"MFMA A8W8 {a_block} A / {b_block} B {scale_layout} scale GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k})")
    print("=" * 80)

    _wpe = int(waves_per_eu) if waves_per_eu else 0
    _wpe = None if _wpe <= 0 else _wpe
    launch_fn = _mxfp4_launcher(
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        out_dtype,
        "fp8",
        _wpe,
        b_dtype="fp8",
        blockscale=blockscale,
        a_scale_layout=scale_layout,
        b_scale_layout=scale_layout,
        b_scale_32x32=b_block == "32x32",
        **kwargs,
    )
    print(f"✓ Compiled (waves_per_eu={_wpe})")

    device = torch.device("cuda")
    torch.manual_seed(0)
    scale_k = K // SCALE_BLOCK_128

    def _quant(x, block):
        """fp8 codes, the scale in the kernel's layout and the dequant of x on a (bm x bk)
        grid. Per-block magnitude jitter: plain randn makes every block max nearly
        identical, so a mis-indexed scale would read a value that happens to be right."""
        bm, bk = {"1x32": (1, 32), "1x128": (1, 128), "32x32": (32, 32), "128x128": (128, 128)}[block]
        rows = x.shape[0]
        # A shuffle_scale_w4'd 1x32 scale needs rows padded to 32.
        pad = 32 if (scale_layout, block) == ("mfma_tile", "1x32") else bm
        padded = torch.zeros((rows + pad - 1) // pad * pad, K, device=device)
        padded[:rows] = x
        jitter = _blk_jitter((padded.shape[0] // bm, K // bk), device)
        padded = padded * jitter.repeat_interleave(bm, 0).repeat_interleave(bk, 1)
        q, sc = gemm_common_utils.per_block_f8_quant(padded, bm, bk)
        sc_f = gemm_common_utils.e8m0_to_f32(sc).repeat_interleave(bm, 0).repeat_interleave(bk, 1)
        return q[:rows], sc, q[:rows].float() * sc_f[:rows]

    a_q, sa, a_deq = _quant(torch.randn(M, K, device=device), a_block)
    b_q, sb, b_deq = _quant(torch.randn(N, K, device=device), b_block)
    c_ref = torch.mm(a_deq, b_deq.T)

    # gfx950 packing: B preshuffled 16x16.
    b_shuffled = gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False)
    if scale_layout == "row":
        scale_a_shuffled = sa.T.contiguous() if a_block == "1x128" else sa  # 1x128: column-major [K // 128, M]
        scale_b_shuffled = sb
    else:
        if a_block == "1x128":
            scale_a_shuffled = gemm_common_utils.shuffle_scale_blockscale_a(sa, K)
        else:
            scale_a_shuffled = gemm_common_utils.shuffle_scale_w4(sa, 1, False)
        if b_block == "128x128":
            scale_b_shuffled = gemm_common_utils.shuffle_scale_blockscale_b(sb, N, K)
        else:
            scale_b_shuffled = gemm_common_utils.shuffle_scale_w4(sb, 1, False)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)
    _dummy_bias = torch.empty(0, dtype=torch_out_dtype, device=device)

    def launch_kernel(c, a, b, sa, sb):
        launch_fn(
            c.contiguous().view(-1),
            a.view(torch.uint8).contiguous().view(-1),
            b.view(torch.uint8).contiguous().view(-1),
            sa.view(torch.uint8).contiguous().view(-1),
            sb.view(torch.uint8).contiguous().view(-1),
            _dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    us = _run_correctness_or_benchmark(
        launch_kernel,
        c_out,
        a_q,
        b_shuffled,
        scale_a_shuffled,
        scale_b_shuffled,
        bench_iters=bench_iters,
        bench_warmup=bench_warmup,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    # A and B: 1 byte/code; scales: M + N//128 rows of K//128 bytes.
    if us is not None:
        bytes_moved = M * K + N * K + M * N * 2 + (M + N // SCALE_BLOCK_128) * scale_k
        tflops = (2 * M * N * K) / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(f"[flyc] A8W8 blockscale Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")
