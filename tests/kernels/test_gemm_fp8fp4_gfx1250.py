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
from kernels.gemm.gemm_a8w8_gfx1250 import launch_gemm_a8w8  # noqa: E402
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


def _const_code(val: float, *, fp4: bool) -> int:
    """The uint8 code (FP4 E2M1 nibble / FP8 E4M3 byte) that decodes exactly to `val`."""
    codes = torch.arange(16 if fp4 else 126, dtype=torch.uint8).view(1, -1)
    vals = gemm_common_utils.mxfp4_to_f32(codes)[0, ::2] if fp4 else gemm_common_utils.fp8_e4m3_to_f32(codes)[0]
    match = (vals == val).nonzero()
    if not len(match):
        raise ValueError(f"{val} is not exactly representable in {'fp4' if fp4 else 'fp8'}")
    return int(match[0, 0])


def _fp8_bytes(rows: int, cols: int, const_val: float | None = None) -> torch.Tensor:
    """Finite FP8 E4M3 bytes (avoids the 0x7F/0xFF NaN encodings), or a constant fill."""
    if const_val is None:
        return torch.randint(0, 126, (rows, cols), dtype=torch.uint8)
    return torch.full((rows, cols), _const_code(const_val, fp4=False), dtype=torch.uint8)


def _fp4_bytes(rows: int, K: int, const_val: float | None = None) -> torch.Tensor:
    """Packed FP4 E2M1 bytes [rows, K//2], random or a constant fill."""
    if const_val is None:
        return gemm_common_utils.random_fp4_packed(rows, K)
    code = _const_code(const_val, fp4=True)
    return torch.full((rows, K // 2), code | (code << 4), dtype=torch.uint8)


def _with_strided_a(a: torch.Tensor, K: int, lda: int) -> torch.Tensor:
    """Return A backed by runtime lda when lda exceeds logical K."""
    if lda == K:
        return a
    M = a.shape[0]
    out = torch.zeros(M, lda, dtype=a.dtype, device=a.device)
    out[:, :K] = a
    return out


def _bench_us(launch, output: torch.Tensor, *, warmup: int = 10, iters: int = 100) -> float:
    """Median per-launch latency (us) from saturated back-to-back throughput."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    output.zero_()
    launch()
    torch.cuda.synchronize()
    if output.abs().max().item() == 0:
        raise RuntimeError("the launch produced an all-zero output; it is not running")

    rounds, batch = 10, max(1, iters // 10)
    samples = []
    for _ in range(rounds):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(batch):
            launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / batch)
    return sorted(samples)[len(samples) // 2]


def _tflops(M: int, N: int, K: int, us: float) -> float:
    return 2.0 * M * N * K / (us * 1e-6) / 1e12


def _run_and_check(build_fn, launch_fn, M, N, K, *rest, **kwargs):
    """Build inputs, compile+run once, and compare against the reference."""
    c_gpu, make_args, ref, (rtol, atol) = build_fn(M, N, K, *rest, **kwargs)
    compiled = flyc.compile(launch_fn, *make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    error = None
    try:
        torch.testing.assert_close(c_gpu[:M, :N].float(), ref.float(), rtol=rtol, atol=atol)
    except AssertionError as exc:
        error = exc
    return c_gpu, make_args, compiled, error


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
    const_val=None,
):
    torch.manual_seed(0)
    a = _fp8_bytes(M, K, const_val)
    b = _fp4_bytes(N, K, const_val)
    # f16 output overflows (~65504 max) with the default E8M0 exponent range at this K;
    # pin scales to 1.0 so f16 accumulation stays in range like the other dtypes.
    scale_exp = {"low_exp": 127, "high_exp": 127} if out_dtype == "f16" else {}
    a_scale = gemm_common_utils.random_e8m0(M, K // SCALE_BLOCK_32, **scale_exp)
    b_scale = gemm_common_utils.random_e8m0(N, K // SCALE_BLOCK_32, **scale_exp)
    a, b = a.cuda(), b.cuda()
    a_scale, b_scale = a_scale.cuda(), b_scale.cuda()
    ref = _reference_a8w4(a, b, a_scale, b_scale, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda)
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K // 2)
    as_gpu = _preshuffle_scale_32x4(a_scale)
    bs_gpu = _preshuffle_scale_32x4(b_scale)
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


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra)
_A8W4_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "bf16", 0, 0),
    (128, 512, 1024, 128, 256, 256, 2, 2, 2, "bf16", 0, 0),
    (256, 256, 512, 256, 256, 256, 2, 2, 2, "bf16", 0, 0),
    (1024, 1024, 1024, 128, 256, 128, 2, 2, 3, "bf16", 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "f16", 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, "bf16", 64, 96),
]


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra", _A8W4_CASES
)
def test_a8w4_mxscale_gemm(
    M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, out_dtype, lda_extra, ldc_extra
):
    _run_case(
        "mxscale_a8w4",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype=out_dtype,
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


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
    const_val=None,
):
    torch.manual_seed(0)
    a = _fp8_bytes(M, K, const_val)
    b = _fp8_bytes(N, K, const_val)
    sa = (scale_scale * (0.5 + torch.rand(M, dtype=torch.float32))).contiguous()
    sb = (scale_scale * (0.5 + torch.rand(N, dtype=torch.float32))).contiguous()
    a, b = a.cuda(), b.cuda()
    sa_gpu, sb_gpu = sa.cuda().contiguous(), sb.cuda().contiguous()
    ref = _reference_ptpc(a, b, sa_gpu, sb_gpu, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda).contiguous()
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K).contiguous()
    c_gpu = torch.zeros(M, ldc, dtype=_DT[out_dtype], device="cuda")
    out_is_f16 = 1 if out_dtype == "f16" else 0
    peak = float(ref.float().abs().max())

    def make_args(stream):
        return (
            flyc.from_c_void_p(fx.Uint8, c_gpu.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, a_gpu.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, b_gpu.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, sa_gpu.data_ptr()),
            flyc.from_c_void_p(fx.Uint8, sb_gpu.data_ptr()),
            M,
            stream,
            N,
            K,
            0,
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
            False,
        )

    return c_gpu, make_args, ref, (2e-2, max(5e-2, 2e-2 * peak))


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
    _run_case(
        "ptpc_a8w8",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        out_dtype=out_dtype,
        scale_scale=scale_scale,
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


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
    const_val=None,
):
    torch.manual_seed(0)
    a = _fp8_bytes(M, K, const_val)
    b = _fp8_bytes(N, K, const_val)
    scale_k = K // SCALE_BLOCK_128
    a_scale = gemm_common_utils.random_e8m0(M, scale_k, low_exp=126, high_exp=129)
    b_scale = gemm_common_utils.random_e8m0(N // SCALE_BLOCK_128, scale_k, low_exp=126, high_exp=129)
    a, b = a.cuda(), b.cuda()
    a_scale, b_scale = a_scale.cuda(), b_scale.cuda()
    ref = _reference_blockscale(a, b, a_scale, b_scale, M, N, K)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K, lda)
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K)
    as_gpu = a_scale.T.contiguous()  # [scale_k, M], row stride == M
    bs_gpu = b_scale
    c_gpu = torch.zeros(M, ldc, dtype=torch.bfloat16, device="cuda")

    def make_args(stream):
        return (
            flyc.from_c_void_p(fx.Int8, c_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, a_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, b_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, as_gpu.data_ptr(), assumed_align=16),
            flyc.from_c_void_p(fx.Int8, bs_gpu.data_ptr(), assumed_align=16),
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
            True,
        )

    return c_gpu, make_args, ref, (1e-2, 5e-2)


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra)
_BLOCKSCALE_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2, 0, 0),
    (256, 256, 512, 256, 256, 128, 2, 2, 4, 0, 0),
    (1024, 1024, 1024, 128, 256, 128, 2, 2, 3, 0, 0),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, 128, 192),
]


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra", _BLOCKSCALE_CASES
)
def test_a8w8_blockscale_gemm(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, lda_extra, ldc_extra):
    _run_case(
        "blockscale_a8w8",
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        lda_extra=lda_extra,
        ldc_extra=ldc_extra,
    )


_MODES = {
    "mxscale_a8w4": dict(
        build=_build_a8w4_case,
        launch=launch_gemm_a8w4_mxscale,
        supports_out_dtype=True,
        # A fp8 + B fp4 (2/byte) + E8M0 scales for A and B + C
        bytes_moved=lambda M, N, K: M * K + N * K // 2 + (M + N) * (K // SCALE_BLOCK_32) + M * N * 2,
        checks=lambda N, K, tile_n, tile_k, num_buffers: [
            (K % SCALE_BLOCK_32 != 0, f"K={K} must be divisible by {SCALE_BLOCK_32}"),
            (N % tile_n != 0, f"N={N} must be divisible by tile_n={tile_n}"),
            (K % tile_k != 0 or (K // tile_k) < num_buffers, f"K={K} incompatible with tile_k={tile_k}"),
        ],
        # N gives 2 tile_n-wide blocks so cluster_n=2 has a real 2nd block to span.
        smoke=dict(N=512, K=512, tile=(128, 256, 128), warps=(2, 2), num_buffers=2),
    ),
    "ptpc_a8w8": dict(
        build=_build_a8w8_ptpc_case,
        launch=launch_gemm_a8w8,
        supports_out_dtype=True,
        # A + B fp8 + one f32 scale per row/column + C
        bytes_moved=lambda M, N, K: M * K + N * K + (M + N) * 4 + M * N * 2,
        checks=lambda N, K, tile_n, tile_k, num_buffers: [
            (N % tile_n != 0, f"N={N} must be divisible by tile_n={tile_n} (no silent pad)"),
            (K % tile_k != 0, f"K={K} must be divisible by tile_k={tile_k} (no silent pad)"),
            (num_buffers > 1 and (K // tile_k) < num_buffers, f"{num_buffers}-buf requires more K-tiles"),
        ],
        smoke=dict(N=256, K=512, tile=(128, 128, 128), warps=(2, 2), num_buffers=4),
    ),
    "blockscale_a8w8": dict(
        build=_build_a8w8_blockscale_case,
        launch=launch_gemm_a8w8,
        supports_out_dtype=False,
        # A + B fp8 + 128-blocked E8M0 scales (per A row, per B 128-row block) + C
        bytes_moved=lambda M, N, K: M * K + N * K + (M + N // SCALE_BLOCK_128) * (K // SCALE_BLOCK_128) + M * N * 2,
        checks=lambda N, K, tile_n, tile_k, num_buffers: [
            (K % SCALE_BLOCK_128 != 0 or N % SCALE_BLOCK_128 != 0, f"N={N}, K={K} must both be divisible by 128"),
            (N % tile_n != 0, f"N={N} must be divisible by tile_n={tile_n}"),
            (K % tile_k != 0 or (K // tile_k) < num_buffers, f"K={K} incompatible with tile_k={tile_k}"),
        ],
        # N gives 2 tile_n-wide blocks so cluster_n=2 has a real 2nd block to span.
        smoke=dict(N=512, K=512, tile=(128, 256, 128), warps=(2, 2), num_buffers=2),
    ),
}


def _run_case(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, **kwargs):
    _require_gpu()
    cfg = _MODES[mode]
    for bad, msg in cfg["checks"](N, K, tile_n, tile_k, num_buffers):
        if bad:
            pytest.skip(msg)
    error = _run_and_check(
        cfg["build"], cfg["launch"], M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, **kwargs
    )[3]
    if error is not None:
        raise error


def _run_smoke(mode, M, **kwargs):
    """Run `mode`'s fixed small tile/warp/buffer config at a given M (ragged-M / cluster sweeps)."""
    smoke = _MODES[mode]["smoke"]
    _run_case(mode, M, smoke["N"], smoke["K"], *smoke["tile"], *smoke["warps"], smoke["num_buffers"], **kwargs)


_RAGGED_M_VALUES = [1, 2, 5, 15, 16, 17, 33, 63, 65, 100, 127, 128, 129, 191, 200, 255, 256, 257, 384, 500, 1000, 2048]


@pytest.mark.parametrize("M", _RAGGED_M_VALUES)
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_gemm_ragged_m(mode, M):
    _run_smoke(mode, M)


@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("M", [1, 65, 129, 384])
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_gemm_cluster(mode, M, cluster_m, cluster_n):
    _run_smoke(mode, M, cluster_m=cluster_m, cluster_n=cluster_n)


def _parse_csv_ints(value: str, n: int, name: str) -> list[int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != n:
        raise SystemExit(f"-{name} needs {n} comma-separated ints, got {value!r}")
    return parts


def _parse_init_mode(value: str) -> float | None:
    """'random' -> None (random fill); 'const,<float>' -> that constant A/B fill value."""
    if value == "random":
        return None
    kind, _, num = value.partition(",")
    if kind != "const" or not num:
        raise SystemExit(f"--init-mode expects 'random' or 'const,<float>', got {value!r}")
    return float(num)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Fixed-width plain-text table: first column left-aligned, the rest right-aligned."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def _main():
    import argparse
    import itertools

    parser = argparse.ArgumentParser(description="Manual correctness/perf run for the gfx1250 GEMM kernels")
    parser.add_argument("-mode", choices=sorted(_MODES), required=True)
    parser.add_argument("-mnk", nargs="+", required=True, metavar="M,N,K", help="one or more shapes")
    parser.add_argument("-tiles", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", default="1,1", help="cluster_m,cluster_n")
    parser.add_argument("-out-dtype", default="bf16", choices=["bf16", "f16"])
    parser.add_argument("-bench", action="store_true", help="also measure perf (warmup=10, iters=100)")
    parser.add_argument(
        "--init-mode",
        nargs="+",
        default=["random", "const,0.5"],
        metavar="MODE",
        help="A/B fill(s) to run: 'random' and/or 'const,<float>' (default: both)",
    )
    args = parser.parse_args()

    shapes = [_parse_csv_ints(v, 3, "mnk") for v in args.mnk]
    tile_m, tile_n, tile_k = _parse_csv_ints(args.tiles, 3, "tiles")
    m_warp, n_warp = _parse_csv_ints(args.warps, 2, "warps")
    cluster_m, cluster_n = _parse_csv_ints(args.cluster, 2, "cluster")

    cfg = _MODES[args.mode]
    out_dtype = args.out_dtype if cfg["supports_out_dtype"] else "bf16"

    rows = []
    for (M, N, K), init in itertools.product(shapes, args.init_mode):
        kwargs = {"cluster_m": cluster_m, "cluster_n": cluster_n, "const_val": _parse_init_mode(init)}
        if cfg["supports_out_dtype"]:
            kwargs["out_dtype"] = args.out_dtype
        c_gpu, make_args, compiled, error = _run_and_check(
            cfg["build"], cfg["launch"], M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, args.nb, **kwargs
        )
        perf = ["-", "-", "-"]
        if args.bench:
            us = _bench_us(lambda: compiled(*make_args(torch.cuda.current_stream())), c_gpu, warmup=10, iters=100)
            moved = cfg["bytes_moved"](M, N, K)
            perf = [f"{us:.3f}", f"{_tflops(M, N, K, us):.2f}", f"{moved / (us * 1e-6) / 1e12:.3f}"]
        rows.append([args.mode, str(M), str(N), str(K), out_dtype, init, *perf, "PASS" if error is None else "FAIL"])
        if error is not None:
            print(f"\n{M}x{N}x{K} {init}: {error}\n")

    print(f"\ntiles={tile_m},{tile_n},{tile_k} warps={m_warp},{n_warp} nb={args.nb} cluster={cluster_m},{cluster_n}")
    _print_table(["mode", "M", "N", "K", "out", "init_mode", "latency us", "TFLOPS", "BW TB/s", "result"], rows)
    if any(r[-1] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
