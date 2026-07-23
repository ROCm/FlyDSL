#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GEMM kernel tests: A8W4 mxscale, A8W8 ptpc, A8W8 blockscale for gfx1250."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc  # noqa: E402,I001
import flydsl.expr as fx  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.gemm_a8w4_mxscale_gfx1250 import launch_gemm_a8w4_mxscale  # noqa: E402
from kernels.gemm.gemm_a8w8_blockscale_gfx1250 import launch_gemm_a8w8_bsc_col  # noqa: E402
from kernels.gemm.gemm_a8w8_ptpc_gfx1250 import launch_gemm_a8w8_ptpc  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_DT = {"bf16": torch.bfloat16, "f16": torch.float16}
SCALE_BLOCK_32 = 32
SCALE_BLOCK_128 = 128


def _require_gpu():
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def _random_fp8_bytes(rows: int, cols: int) -> torch.Tensor:
    """Finite FP8 E4M3 bytes (avoids the 0x7F/0xFF NaN encodings)."""
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8)


def _with_strided_a(a: torch.Tensor, K: int, lda: int) -> torch.Tensor:
    """Return A backed by runtime lda when lda exceeds logical K."""
    if lda == K:
        return a
    M = a.shape[0]
    out = torch.zeros(M, lda, dtype=a.dtype, device=a.device)
    out[:, :K] = a
    return out


def _bench_us(launch, output: torch.Tensor, *, warmup: int = 5, iters: int = 20) -> float:
    """Median per-launch latency (us) via hipGraph capture/replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.synchronize()
    if output.abs().max().item() == 0:
        raise RuntimeError("hipGraph replay produced an all-zero output")

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            start.record()
            graph.replay()
            end.record()
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends))
    return samples[len(samples) // 2]


def _tflops(M: int, N: int, K: int, us: float) -> float:
    return 2.0 * M * N * K / (us * 1e-6) / 1e12


def _assert_case(build_fn, launch_fn, M, N, K, *rest, **kwargs):
    """Build inputs, compile+run once, and assert against the reference.

    Returns (c_gpu, make_args, compiled) so callers (e.g. the perf CLI) can
    replay the same compiled kernel without rebuilding it.
    """
    c_gpu, make_args, ref, (rtol, atol) = build_fn(M, N, K, *rest, **kwargs)
    compiled = flyc.compile(launch_fn, *make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    c_out = c_gpu[:M, :N].float().cpu()
    torch.testing.assert_close(c_out, ref.float(), rtol=rtol, atol=atol)
    return c_gpu, make_args, compiled


def _preshuffle_scale_32x4(scale: torch.Tensor) -> torch.Tensor:
    """[R, K] uint8 E8M0 -> [ceil(R/32), K] 32-row x 4-K-group preshuffled layout."""
    rows, k_scale = scale.shape
    row_blocks = (rows + 31) // 32
    if row_blocks * 32 != rows:
        padded = torch.zeros((row_blocks * 32, k_scale), dtype=scale.dtype, device=scale.device)
        padded[:rows] = scale
        scale = padded
    x = scale.view(row_blocks, 32, k_scale // 4, 4).permute(0, 2, 1, 3).contiguous()
    return x.reshape(row_blocks, -1)


def _e8m0_exp_range(scale: torch.Tensor) -> tuple[int, int]:
    s = scale.view(torch.uint8).to(torch.int16)
    return int(s.min().item()) - 127, int(s.max().item()) - 127


def _a8w4_tolerances(a_scale: torch.Tensor, b_scale: torch.Tensor, K: int) -> tuple[float, float]:
    """Scale-range-aware tolerance for mixed FP8xFP4 WMMA-scale GEMM (bf16/f16 output)."""
    _, a_max_exp = _e8m0_exp_range(a_scale)
    _, b_max_exp = _e8m0_exp_range(b_scale)
    peak_prod_exp = max(0, a_max_exp) + max(0, b_max_exp)
    rtol = min(5e-2, 1e-2 + 3e-3 * peak_prod_exp)
    atol = max(5e-2, K * (0.6 + 1.5 * peak_prod_exp))
    return rtol, atol


def _reference_a8w4(a, b, a_scale, b_scale, M, N, K):
    a_f32 = gemm_common_utils.fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K]
    b_f32 = gemm_common_utils.mxfp4_to_f32(b.view(torch.uint8))[:N, :K]
    a_sc = gemm_common_utils.e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:M, :K]
    b_sc = gemm_common_utils.e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:N, :K]
    return torch.matmul(a_f32 * a_sc, (b_f32 * b_sc).T)


def _build_a8w4_case(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    out_dtype="bf16",
    *,
    lda_extra=0,
    ldc_extra=0,
    cluster_m=1,
    cluster_n=1,
):
    torch.manual_seed(0)
    a = _random_fp8_bytes(M, K)
    b = gemm_common_utils.random_fp4_packed(N, K)
    # f16 output overflows (~65504 max) with the default E8M0 exponent range at this K;
    # pin scales to 1.0 so f16 accumulation stays in range like the other dtypes.
    scale_exp = {"low_exp": 127, "high_exp": 127} if out_dtype == "f16" else {}
    a_scale = gemm_common_utils.random_e8m0(M, K // SCALE_BLOCK_32, **scale_exp)
    b_scale = gemm_common_utils.random_e8m0(N, K // SCALE_BLOCK_32, **scale_exp)
    ref = _reference_a8w4(a, b, a_scale, b_scale, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda).cuda()
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K // 2).cuda()
    as_gpu = _preshuffle_scale_32x4(a_scale).cuda()
    bs_gpu = _preshuffle_scale_32x4(b_scale).cuda()
    c_gpu = torch.zeros(M, ldc, dtype=_DT[out_dtype], device="cuda")
    out_is_f16 = 0 if out_dtype == "bf16" else 1

    def make_args(stream):
        return (
            c_gpu,
            flyc.from_c_void_p(fx.Int8, a_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, b_gpu.data_ptr(), assumed_align=16),
            as_gpu,
            bs_gpu,
            M,
            stream,
            N,
            K,
            lda,
            ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            out_is_f16,
            num_buffers,
            cluster_m,
            cluster_n,
        )

    return c_gpu, make_args, ref, _a8w4_tolerances(a_scale, b_scale, K)


def _run_a8w4_mxscale_case(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype="bf16", **kwargs):
    _require_gpu()
    if K % SCALE_BLOCK_32:
        pytest.skip(f"K={K} must be divisible by {SCALE_BLOCK_32}")
    if N % tile_n:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n}")
    if K % tile_k or (K // tile_k) < num_buffers:
        pytest.skip(f"K={K} incompatible with tile_k={tile_k}, num_buffers={num_buffers}")

    _assert_case(
        _build_a8w4_case,
        launch_gemm_a8w4_mxscale,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype,
        **kwargs,
    )


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra)
_A8W4_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "bf16", 0, 0),
    (128, 512, 1024, 128, 256, 256, 2, 2, 2, "bf16", 0, 0),
    (256, 256, 512, 256, 256, 256, 2, 2, 2, "bf16", 0, 0),
    (1024, 1024, 1024, 128, 256, 128, 2, 4, 3, "bf16", 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "f16", 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "bf16", 64, 96),
]


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra", _A8W4_CASES
)
def test_a8w4_mxscale_gemm(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra
):
    _run_a8w4_mxscale_case(
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype,
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


@pytest.mark.parametrize("M", [1, 5, 33, 63, 100, 129, 200])
def test_a8w4_mxscale_gemm_ragged_m(M):
    _run_a8w4_mxscale_case(M, 256, 512, 128, 256, 128, 2, 2, 2)


@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
def test_a8w4_mxscale_gemm_cluster(cluster_m, cluster_n):
    _run_a8w4_mxscale_case(129, 512, 512, 128, 256, 128, 2, 2, 2, cluster_m=cluster_m, cluster_n=cluster_n)


def _reference_ptpc(a, b, sa, sb, M, N, K):
    a_f32 = gemm_common_utils.fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K]
    b_f32 = gemm_common_utils.fp8_e4m3_to_f32(b.view(torch.uint8))[:N, :K]
    raw = torch.matmul(a_f32, b_f32.T)
    return raw * sa[:M].view(M, 1) * sb[:N].view(1, N)


def _build_a8w8_ptpc_case(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    out_dtype="bf16",
    *,
    cluster_m=1,
    cluster_n=1,
    lda_extra=0,
    ldc_extra=0,
    scale_scale=1.0,
):
    torch.manual_seed(0)
    a = _random_fp8_bytes(M, K)
    b = _random_fp8_bytes(N, K)
    sa = (scale_scale * (0.5 + torch.rand(M, dtype=torch.float32))).contiguous()
    sb = (scale_scale * (0.5 + torch.rand(N, dtype=torch.float32))).contiguous()
    ref = _reference_ptpc(a, b, sa, sb, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda).cuda().contiguous()
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K).cuda().contiguous()
    sa_gpu, sb_gpu = sa.cuda().contiguous(), sb.cuda().contiguous()
    c_gpu = torch.zeros(M, ldc, dtype=_DT[out_dtype], device="cuda")
    out_is_f16 = 1 if out_dtype == "f16" else 0
    peak = float(ref.float().abs().max())

    def make_args(stream):
        return (
            c_gpu,
            flyc.from_c_void_p(fx.Uint8, a_gpu.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, b_gpu.data_ptr()),
            sa_gpu,
            sb_gpu,
            M,
            stream,
            N,
            K,
            lda,
            ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            out_is_f16,
            num_buffers,
            cluster_m,
            cluster_n,
        )

    return c_gpu, make_args, ref, (2e-2, max(5e-2, 2e-2 * peak))


def _run_a8w8_ptpc_case(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype="bf16", **kwargs):
    _require_gpu()
    if N % tile_n:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n} (no silent pad)")
    if K % tile_k:
        pytest.skip(f"K={K} must be divisible by tile_k={tile_k} (no silent pad)")
    if num_buffers > 1 and (K // tile_k) < num_buffers:
        pytest.skip(f"{num_buffers}-buf requires num_k_tiles >= {num_buffers}")

    _assert_case(
        _build_a8w8_ptpc_case,
        launch_gemm_a8w8_ptpc,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype,
        **kwargs,
    )


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, scale_scale, lda_extra, ldc_extra)
_PTPC_CASES = [
    (256, 256, 512, 256, 256, 128, 2, 2, 4, "bf16", 1.0, 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 4, "bf16", 1.0, 0, 0),
    (128, 128, 1024, 128, 128, 256, 2, 2, 3, "bf16", 1.0, 0, 0),
    (64, 64, 512, 64, 64, 128, 2, 2, 2, "bf16", 1.0, 0, 0),
    (128, 96, 512, 128, 96, 128, 2, 2, 2, "bf16", 1.0, 0, 0),
    (128, 128, 512, 128, 128, 128, 1, 2, 2, "bf16", 1.0, 0, 0),
    (256, 256, 512, 256, 256, 128, 2, 2, 4, "f16", 0.02, 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 4, "bf16", 1.0, 128, 256),
]


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, scale_scale, lda_extra, ldc_extra",
    _PTPC_CASES,
)
def test_a8w8_ptpc_gemm(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, scale_scale, lda_extra, ldc_extra
):
    _run_a8w8_ptpc_case(
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype,
        scale_scale=scale_scale,
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


@pytest.mark.parametrize("M", [1, 16, 65, 127, 129, 255, 257, 500, 2048])
def test_a8w8_ptpc_gemm_ragged_m(M):
    _run_a8w8_ptpc_case(M, 256, 512, 128, 128, 128, 2, 2, 4, "bf16")


@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("M", [1, 65, 129, 384])
def test_a8w8_ptpc_gemm_ragged_m_cluster(M, cluster_m, cluster_n):
    _run_a8w8_ptpc_case(M, 256, 512, 128, 128, 128, 2, 2, 4, "bf16", cluster_m=cluster_m, cluster_n=cluster_n)


def test_a8w8_ptpc_gemm_min_waves():
    """m_warp * n_warp < 2 must raise (need at least one A and one B loader wave)."""
    with pytest.raises(ValueError):
        _run_a8w8_ptpc_case(128, 256, 512, 128, 256, 128, 1, 1, 4, "bf16")


def _reference_blockscale(a, b, a_scale, b_scale, M, N, K):
    scale_k = K // SCALE_BLOCK_128
    a_f32 = gemm_common_utils.fp8_e4m3_to_f32(a.view(torch.uint8))[:M, :K].clone()
    b_f32 = gemm_common_utils.fp8_e4m3_to_f32(b.view(torch.uint8))[:N, :K].clone()
    a_sc = gemm_common_utils.e8m0_to_f32(a_scale.view(torch.uint8))[:M, :scale_k]
    b_sc = gemm_common_utils.e8m0_to_f32(b_scale.view(torch.uint8))[: N // SCALE_BLOCK_128, :scale_k]
    a_f32.view(M, scale_k, SCALE_BLOCK_128).mul_(a_sc.unsqueeze(-1))
    b_sc_rows = b_sc.repeat_interleave(SCALE_BLOCK_128, dim=0)[:N]
    b_f32.view(N, scale_k, SCALE_BLOCK_128).mul_(b_sc_rows.unsqueeze(-1))
    return torch.matmul(a_f32, b_f32.T)


def _build_a8w8_blockscale_case(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    *,
    lda_extra=0,
    ldc_extra=0,
    cluster_m=1,
    cluster_n=1,
):
    torch.manual_seed(0)
    a = _random_fp8_bytes(M, K)
    b = _random_fp8_bytes(N, K)
    scale_k = K // SCALE_BLOCK_128
    a_scale = gemm_common_utils.random_e8m0(M, scale_k, low_exp=126, high_exp=129)
    b_scale = gemm_common_utils.random_e8m0(N // SCALE_BLOCK_128, scale_k, low_exp=126, high_exp=129)
    ref = _reference_blockscale(a, b, a_scale, b_scale, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda).cuda()
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K).cuda()
    as_gpu = a_scale.T.contiguous().cuda()  # [scale_k, M], row stride == M
    bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(M, ldc, dtype=torch.bfloat16, device="cuda")

    def make_args(stream):
        return (
            c_gpu,
            flyc.from_c_void_p(fx.Int8, a_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, b_gpu.data_ptr(), assumed_align=16),
            as_gpu,
            bs_gpu,
            M,
            stream,
            N,
            K,
            M,  # stride_ascale_k
            lda,
            ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            0,  # bf16 output
            num_buffers,
            cluster_m,
            cluster_n,
        )

    return c_gpu, make_args, ref, (1e-2, 5e-2)


def _run_a8w8_blockscale_case(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, **kwargs):
    _require_gpu()
    if m_warp * n_warp < 4:
        pytest.skip("blockscale requires m_warp * n_warp >= 4 (A/B/scaleA/scaleB loader waves)")
    if K % SCALE_BLOCK_128 or N % SCALE_BLOCK_128:
        pytest.skip(f"N={N}, K={K} must both be divisible by {SCALE_BLOCK_128}")
    if N % tile_n:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n}")
    if K % tile_k or (K // tile_k) < num_buffers:
        pytest.skip(f"K={K} incompatible with tile_k={tile_k}, num_buffers={num_buffers}")

    _assert_case(
        _build_a8w8_blockscale_case,
        launch_gemm_a8w8_bsc_col,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        **kwargs,
    )


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra)
_BLOCKSCALE_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2, 0, 0),
    (256, 256, 512, 256, 256, 128, 2, 2, 4, 0, 0),
    (1024, 1024, 1024, 128, 256, 128, 2, 4, 3, 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, 128, 192),
]


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra", _BLOCKSCALE_CASES
)
def test_a8w8_blockscale_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra):
    _run_a8w8_blockscale_case(
        M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra=lda_extra, ldc_extra=ldc_extra
    )


@pytest.mark.parametrize("M", [1, 5, 33, 63, 100, 129, 200])
def test_a8w8_blockscale_gemm_ragged_m(M):
    _run_a8w8_blockscale_case(M, 256, 512, 128, 256, 128, 2, 2, 2)


@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
def test_a8w8_blockscale_gemm_cluster(cluster_m, cluster_n):
    _run_a8w8_blockscale_case(129, 512, 512, 128, 256, 128, 2, 2, 2, cluster_m=cluster_m, cluster_n=cluster_n)


_MODES = {
    "mxscale_a8w4": (_build_a8w4_case, launch_gemm_a8w4_mxscale, True),
    "ptpc_a8w8": (_build_a8w8_ptpc_case, launch_gemm_a8w8_ptpc, True),
    "blockscale_a8w8": (_build_a8w8_blockscale_case, launch_gemm_a8w8_bsc_col, False),
}


def _parse_csv_ints(value: str, n: int, name: str) -> list[int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != n:
        raise SystemExit(f"-{name} needs {n} comma-separated ints, got {value!r}")
    return parts


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Manual correctness/perf run for the gfx1250 GEMM kernels")
    parser.add_argument("-mode", choices=sorted(_MODES), required=True)
    parser.add_argument("-mnk", required=True, help="M,N,K")
    parser.add_argument("-tiles", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", default="1,1", help="cluster_m,cluster_n")
    parser.add_argument("-out-dtype", default="bf16", choices=["bf16", "f16"])
    parser.add_argument("-bench", action="store_true", help="also measure perf (warmup=10, iters=100)")
    args = parser.parse_args()

    M, N, K = _parse_csv_ints(args.mnk, 3, "mnk")
    tile_m, tile_n, tile_k = _parse_csv_ints(args.tiles, 3, "tiles")
    m_warp, n_warp = _parse_csv_ints(args.warps, 2, "warps")
    cluster_m, cluster_n = _parse_csv_ints(args.cluster, 2, "cluster")

    build_fn, launch_fn, supports_out_dtype = _MODES[args.mode]
    kwargs = {"cluster_m": cluster_m, "cluster_n": cluster_n}
    if supports_out_dtype:
        kwargs["out_dtype"] = args.out_dtype
    c_gpu, make_args, compiled = _assert_case(
        build_fn, launch_fn, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, args.nb, **kwargs
    )
    print(f"PASSED correctness: mode={args.mode} M={M} N={N} K={K}")

    if args.bench:

        def launch():
            compiled(*make_args(torch.cuda.current_stream()))

        us = _bench_us(launch, c_gpu, warmup=10, iters=100)
        print(f"perf: mode={args.mode} M={M} N={N} K={K} {us:.3f}us ({_tflops(M, N, K, us):.2f} TFLOPS)")


if __name__ == "__main__":
    _main()
