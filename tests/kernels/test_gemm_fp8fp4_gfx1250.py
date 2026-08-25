#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GEMM kernel tests for gfx1250: A8W4 / A8W8 mxscale, A8W8 ptpc, A8W8 blockscale,
and the hand-scheduled 256-wide A8W4 / A4W4 / A8W8 kernels."""

import functools
import os
import sys
from typing import NamedTuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc  # noqa: E402,I001
import flydsl.expr as fx  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.gemm_a4w4_256x256_gfx1250 import launch_gemm_a4w4_256x256  # noqa: E402
from kernels.gemm.gemm_a8w4_256x256_gfx1250 import launch_gemm_a8w4_256x256  # noqa: E402
from kernels.gemm.gemm_a8w4_mxscale_gfx1250 import launch_gemm_a8w4_mxscale  # noqa: E402
from kernels.gemm.gemm_a8w8_256x256_gfx1250 import launch_gemm_a8w8_256x256  # noqa: E402
from kernels.gemm.gemm_a8w8_gfx1250 import launch_gemm_a8w8  # noqa: E402
from kernels.gemm.gemm_a8w8_splitk_reduce_gfx1250 import compile_gemm_a8w8_splitk_reduce  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_DT = {"bf16": torch.bfloat16, "f16": torch.float16}
_DT_NAME = {v: k for k, v in _DT.items()}
SCALE_BLOCK_32 = 32
SCALE_BLOCK_128 = 128


def _require_gpu():
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def _i8(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _u8(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


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
    c_gpu, make_args, ref, (rtol, atol), epilogue = build_fn(M, N, K, *rest, **kwargs)
    compiled = flyc.compile(launch_fn, *make_args(torch.cuda.current_stream()))
    if epilogue is not None:
        epilogue()  # split-K reduce
    torch.cuda.synchronize()
    error = None
    try:
        torch.testing.assert_close(c_gpu[:M, :N].float(), ref.float(), rtol=rtol, atol=atol)
    except AssertionError as exc:
        error = exc
    if error is None and c_gpu.shape[0] > M:  # NaN guard rows below M (see c_guard_rows)
        clobbered = int((~torch.isnan(c_gpu[M:].float())).sum())
        if clobbered:
            error = AssertionError(f"M={M}: {clobbered} elements written at/after row {M}; the store clamp failed")
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


def _reference_mx32(a_f32, b_f32, a_scale, b_scale, M, N, K):
    """Reference for 32-element-block MX scales, from already-decoded operands."""
    a_sc = gemm_common_utils.e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:M, :K]
    b_sc = gemm_common_utils.e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:N, :K]
    return torch.matmul(a_f32[:M, :K] * a_sc, (b_f32[:N, :K] * b_sc).T)


def _reference_ptpc(a_f32, b_f32, sa, sb, M, N, K):
    raw = torch.matmul(a_f32[:M, :K], b_f32[:N, :K].T)
    return raw * sa[:M].view(M, 1) * sb[:N].view(1, N)


def _reference_blockscale(a_f32, b_f32, a_scale, b_scale, M, N, K):
    scale_k = K // SCALE_BLOCK_128
    a_f32 = a_f32[:M, :K].clone()
    b_f32 = b_f32[:N, :K].clone()
    a_sc = gemm_common_utils.e8m0_to_f32(a_scale.view(torch.uint8))[:M, :scale_k]
    b_sc = gemm_common_utils.e8m0_to_f32(b_scale.view(torch.uint8))[: N // SCALE_BLOCK_128, :scale_k]
    a_f32.view(M, scale_k, SCALE_BLOCK_128).mul_(a_sc.unsqueeze(-1))
    b_sc_rows = b_sc.repeat_interleave(SCALE_BLOCK_128, dim=0)[:N]
    b_f32.view(N, scale_k, SCALE_BLOCK_128).mul_(b_sc_rows.unsqueeze(-1))
    return torch.matmul(a_f32, b_f32.T)


# The profiles gemm_a8w8_256x256_gfx1250 is hand-scheduled for.
_A8W8_256X256_TILES = [
    (256, 256, 128, 2, 2, 4),
    (256, 256, 128, 2, 2, 2),
    (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 3),
]
# 128x128 leaves 4 WMMA slots per quadrant, room for block128's scale seeds but not mx32's.
_A8W8_256X256_TILES_MX128 = _A8W8_256X256_TILES + [(128, 128, 128, 2, 2, 4)]


def _splitk_reduce(partials: torch.Tensor, out: torch.Tensor, N: int, split_k: int) -> None:
    """Fold the GEMM's split-K partials into `out` with the production reduce epilogue."""
    rows, ldc = out.shape
    dense = ldc == N  # a contiguous C is one long run; a padded one is one run per row
    flyc.compile(  # compiles and runs
        compile_gemm_a8w8_splitk_reduce(split_k=split_k, out_dtype_str=_DT_NAME[out.dtype]),
        _u8(partials),
        _u8(out),
        rows * ldc if dense else N,
        1 if dense else rows,
        ldc,
        rows * ldc * out.element_size(),
        torch.cuda.current_stream(),
    )


class _Kind(NamedTuple):
    """What a mode's inputs look like: operand dtypes, scale layout, argument style."""

    fp4_act: bool = False
    fp4_w: bool = False
    scale: object = SCALE_BLOCK_32  # E8M0 block size, or "ptpc" for f32 per-row/column scales
    tensor_args: bool = False  # C and the scales go in as fx.Tensor rather than as pointers


_A8W4_MX = _Kind(fp4_w=True, tensor_args=True)
_A8W4 = _Kind(fp4_w=True)
_A4W4 = _Kind(fp4_act=True, fp4_w=True)
_MX32 = _Kind()
_MX128 = _Kind(scale=SCALE_BLOCK_128)
_PTPC = _Kind(scale="ptpc")


def _build_case(
    kind,
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
    scale_scale=1.0,
    const_val=None,
    c_guard_rows=0,
    split_k=1,
):
    """Build one case: inputs, the launch-argument factory, the reference and its tolerance.

    Every kernel here takes the same argument order; `kind` picks the operand dtypes and
    the scale layout. Only the A8W8 launchers take the extra stride_ascale_k argument and
    the trailing is_mxscale / block_size pair.
    """
    a8w8 = not kind.fp4_w
    torch.manual_seed(0)
    a = _fp4_bytes(M, K, const_val) if kind.fp4_act else _fp8_bytes(M, K, const_val)
    b = _fp4_bytes(N, K, const_val) if kind.fp4_w else _fp8_bytes(N, K, const_val)
    a, b = a.cuda(), b.cuda()
    a_f32 = gemm_common_utils.mxfp4_to_f32(a) if kind.fp4_act else gemm_common_utils.fp8_e4m3_to_f32(a)
    b_f32 = gemm_common_utils.mxfp4_to_f32(b) if kind.fp4_w else gemm_common_utils.fp8_e4m3_to_f32(b)

    if kind.scale == SCALE_BLOCK_32:
        # f16 output overflows (~65504 max) with the default E8M0 exponent range, so pin
        # the scales to 1.0 there; FP8 operands need the narrower range even for bf16.
        if out_dtype == "f16":
            exp = dict(low_exp=127, high_exp=127)
        else:
            exp = {} if kind.fp4_w else dict(low_exp=126, high_exp=129)
        a_scale = gemm_common_utils.random_e8m0(M, K // SCALE_BLOCK_32, **exp).cuda()
        b_scale = gemm_common_utils.random_e8m0(N, K // SCALE_BLOCK_32, **exp).cuda()
        ref = _reference_mx32(a_f32, b_f32, a_scale, b_scale, M, N, K)
        as_gpu, bs_gpu = _preshuffle_scale_32x4(a_scale), _preshuffle_scale_32x4(b_scale)
        stride_ascale_k = 0
        tol = _a8w4_tolerances(a_scale, b_scale, K) if kind.fp4_w else (1e-2, 5e-2)
    elif kind.scale == SCALE_BLOCK_128:
        scale_k = K // SCALE_BLOCK_128
        a_scale = gemm_common_utils.random_e8m0(M, scale_k, low_exp=126, high_exp=129).cuda()
        b_scale = gemm_common_utils.random_e8m0(N // SCALE_BLOCK_128, scale_k, low_exp=126, high_exp=129).cuda()
        ref = _reference_blockscale(a_f32, b_f32, a_scale, b_scale, M, N, K)
        as_gpu, bs_gpu = a_scale.T.contiguous(), b_scale  # A-scale is [K/128, M], row stride M
        stride_ascale_k = M
        tol = (1e-2, 5e-2)
    else:
        as_gpu = (scale_scale * (0.5 + torch.rand(M, dtype=torch.float32))).cuda().contiguous()
        bs_gpu = (scale_scale * (0.5 + torch.rand(N, dtype=torch.float32))).cuda().contiguous()
        ref = _reference_ptpc(a_f32, b_f32, as_gpu, bs_gpu, M, N, K)
        stride_ascale_k = 0
        tol = (2e-2, max(5e-2, 2e-2 * float(ref.abs().max())))
    if split_k > 1:  # the partials round-trip through bf16/f16 before the reduce
        tol = (tol[0], tol[1] + split_k * float(ref.abs().max()) * 2**-8)

    lda, ldc = K + lda_extra, N + ldc_extra
    a_gpu = _with_strided_a(a, K // 2 if kind.fp4_act else K, lda // 2 if kind.fp4_act else lda)
    b_gpu = gemm_common_utils.preshuffle_b_16x16(b, N, K // 2 if kind.fp4_w else K)
    # Guard rows are filled with NaN and must stay NaN: nothing may be stored at/after row M.
    fill = float("nan") if c_guard_rows else 0.0
    c_gpu = torch.full((M + c_guard_rows, ldc), fill, dtype=_DT[out_dtype], device="cuda")
    partials = torch.full((split_k, *c_gpu.shape), fill, dtype=c_gpu.dtype, device="cuda") if split_k > 1 else None
    dev = [c_gpu if partials is None else partials, a_gpu, b_gpu, as_gpu, bs_gpu]
    wrap = _u8 if kind.scale == "ptpc" else _i8
    mx_args = (kind.scale != "ptpc", SCALE_BLOCK_128 if kind.scale == "ptpc" else kind.scale) if a8w8 else ()

    def make_args(stream):
        ptrs = [dev[0], wrap(dev[1]), wrap(dev[2]), dev[3], dev[4]] if kind.tensor_args else [wrap(t) for t in dev]
        return (
            *ptrs,
            M,
            stream,
            N,
            K,
            *((stride_ascale_k,) if a8w8 else ()),
            lda,
            ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            1 if out_dtype == "f16" else 0,
            num_buffers,
            cluster_m,
            cluster_n,
            *mx_args,  # is_mxscale, block_size
            split_k,
        )

    epilogue = None if partials is None else (lambda: _splitk_reduce(partials, c_gpu, N, split_k))
    return c_gpu, make_args, ref, tol, epilogue


def _bytes_moved(kind, M: int, N: int, K: int) -> int:
    """Logical A / B / scale bytes read plus C written by one launch."""
    a_bytes = M * K // (2 if kind.fp4_act else 1)
    b_bytes = N * K // (2 if kind.fp4_w else 1)
    if kind.scale == SCALE_BLOCK_32:
        scale_bytes = (M + N) * (K // SCALE_BLOCK_32)
    elif kind.scale == SCALE_BLOCK_128:
        scale_bytes = (M + N // SCALE_BLOCK_128) * (K // SCALE_BLOCK_128)
    else:
        scale_bytes = (M + N) * 4
    return a_bytes + b_bytes + scale_bytes + M * N * 2


def _shape_checks(kind, N, K, tile_cfg):
    """Shapes a kernel cannot serve; none of them silently pads."""
    _, tile_n, tile_k, _, _, num_buffers = tile_cfg
    checks = [
        (N % tile_n != 0, f"N={N} must be divisible by tile_n={tile_n}"),
        (K % tile_k != 0, f"K={K} must be divisible by tile_k={tile_k}"),
        (K // tile_k < num_buffers, f"K={K} yields fewer K-tiles than {num_buffers} buffers"),
    ]
    if kind.scale == SCALE_BLOCK_32:
        checks.append((K % SCALE_BLOCK_32 != 0, f"K={K} must be divisible by {SCALE_BLOCK_32}"))
    elif kind.scale == SCALE_BLOCK_128:
        both = f"N={N}, K={K} must both be divisible by {SCALE_BLOCK_128}"
        checks.append((N % SCALE_BLOCK_128 != 0 or K % SCALE_BLOCK_128 != 0, both))
    return checks


def _mode(kind, launch, **rest):
    """One _MODES entry; `kind` drives the builder, the byte count and the shape checks."""
    build = functools.partial(_build_case, kind)
    return dict(kind=kind, build=build, bytes_moved=functools.partial(_bytes_moved, kind), launch=launch, **rest)


_MODES = {
    "mxscale_a8w4": _mode(
        _A8W4_MX,
        launch_gemm_a8w4_mxscale,
        supports_out_dtype=True,
        # N gives 2 tile_n-wide blocks so cluster_n=2 has a real 2nd block to span.
        smoke=dict(N=512, K=512, tile=(128, 256, 128), warps=(2, 2), num_buffers=2),
    ),
    "ptpc_a8w8": _mode(
        _PTPC,
        launch_gemm_a8w8,
        supports_out_dtype=True,
        smoke=dict(N=256, K=512, tile=(128, 128, 128), warps=(2, 2), num_buffers=4),
    ),
    "blockscale_a8w8": _mode(
        _MX128,
        launch_gemm_a8w8,
        supports_out_dtype=False,
        smoke=dict(N=512, K=512, tile=(128, 256, 128), warps=(2, 2), num_buffers=2),
    ),
    "mxscale_a8w8": _mode(
        _MX32,
        launch_gemm_a8w8,
        supports_out_dtype=False,
        smoke=dict(N=512, K=512, tile=(128, 256, 128), warps=(2, 2), num_buffers=2),
    ),
    "mxscale_a8w4_256x256": _mode(
        _A8W4,
        launch_gemm_a8w4_256x256,
        supports_out_dtype=True,
        f16_kw={},  # the builder pins the scales to 1.0 for f16
        profile=dict(
            tiles=[(256, 256, 128, 2, 2, 3)],
            cluster=(4, 4),
            cluster_ok=lambda cm, cn: (cm, cn) == (4, 4),
            k_mult=256,  # one TDM covers 2 K-tiles
            k_min=512,
        ),
        smoke=dict(N=1024, K=512, tile=(256, 256, 128), warps=(2, 2), num_buffers=3),
        shapes=[(1024, 1024, 1024), (2048, 1024, 1280), (1024, 2048, 1536)],
    ),
    "mxscale_a4w4_256x256": _mode(
        _A4W4,
        launch_gemm_a4w4_256x256,
        supports_out_dtype=True,
        f16_kw={},
        profile=dict(
            tiles=[(256, 256, 256, 2, 2, 4)],
            cluster=(4, 4),
            cluster_ok=lambda cm, cn: (cm, cn) == (4, 4),
            k_mult=1024,  # K must cover whole 4-K-tile revolutions
            k_min=1024,
        ),
        smoke=dict(N=1024, K=1024, tile=(256, 256, 256), warps=(2, 2), num_buffers=4),
        shapes=[(1024, 1024, 1024), (2048, 1024, 3072), (1000, 2048, 4096)],
    ),
    "mxscale_a8w8_256x256": _mode(
        _MX32,
        launch_gemm_a8w8_256x256,
        supports_out_dtype=True,
        f16_kw=dict(const_val=0.25),  # keep the f16 accumulation in range at these K
        profile=dict(
            tiles=_A8W8_256X256_TILES,
            cluster=(4, 4),
            cluster_ok=lambda cm, cn: 1 < cm * cn <= 16,
            k_mult=SCALE_BLOCK_128,
            k_min=512,
        ),
        smoke=dict(N=1024, K=1024, tile=(256, 256, 128), warps=(2, 2), num_buffers=4),
        shapes=[(1024, 1024, 1024), (2048, 1024, 1152), (1024, 2048, 1280), (1000, 1024, 1408)],
    ),
    "blockscale_a8w8_256x256": _mode(
        _MX128,
        launch_gemm_a8w8_256x256,
        supports_out_dtype=True,
        f16_kw=dict(const_val=0.25),
        profile=dict(
            tiles=_A8W8_256X256_TILES_MX128,
            cluster=(4, 4),
            cluster_ok=lambda cm, cn: 1 < cm * cn <= 16,
            k_mult=SCALE_BLOCK_128,
            k_min=512,
        ),
        smoke=dict(N=1024, K=1024, tile=(256, 256, 128), warps=(2, 2), num_buffers=4),
        shapes=[(1024, 1024, 1024), (2048, 1024, 1152), (1024, 2048, 1280), (1000, 1024, 1408)],
    ),
}


_PROFILE_MODES = [m for m in sorted(_MODES) if "profile" in _MODES[m]]


def _tuned(mode):
    """The mode's own (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)."""
    smoke = _MODES[mode]["smoke"]
    return (*smoke["tile"], *smoke["warps"], smoke["num_buffers"])


def _profile_checks(profile, N, K, tile_cfg, cluster, split_k):
    """A hand-scheduled kernel only accepts the tilings and clusters it was scheduled for."""
    k_step = profile["k_mult"] * split_k
    return [
        (tile_cfg not in profile["tiles"], f"hand-scheduled for {profile['tiles']}, not {tile_cfg}"),
        (not profile["cluster_ok"](*cluster), f"not tuned for a {cluster[0]}x{cluster[1]} cluster"),
        (N % (tile_cfg[1] * cluster[1]) != 0, f"N={N} must cover whole clusters of {tile_cfg[1]}-wide tiles"),
        (K % k_step != 0, f"K={K} must be a multiple of {k_step} for a {split_k}-way split"),
        (K // split_k < profile["k_min"], f"K={K} leaves under {profile['k_min']} per split"),
    ]


def _run_case(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, **kwargs):
    _require_gpu()
    cfg = _MODES[mode]
    tile_cfg = (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
    checks = _shape_checks(cfg["kind"], N, K, tile_cfg)
    profile = cfg.get("profile")
    if profile is not None:  # sweeps that do not parametrize the cluster get the tuned one
        kwargs.setdefault("cluster_m", profile["cluster"][0])
        kwargs.setdefault("cluster_n", profile["cluster"][1])
        cluster = (kwargs["cluster_m"], kwargs["cluster_n"])
        checks += _profile_checks(profile, N, K, tile_cfg, cluster, kwargs.get("split_k", 1))
    for bad, msg in checks:
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
    K = smoke["K"] * kwargs.get("split_k", 1)  # each split gets the smoke shape's own K
    _run_case(mode, M, smoke["N"], K, *smoke["tile"], *smoke["warps"], smoke["num_buffers"], **kwargs)


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, extra kwargs)
_MX_A8W8_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2, {}),
    (256, 256, 512, 256, 256, 128, 2, 2, 4, {}),
    (1024, 1024, 1024, 128, 256, 128, 2, 2, 3, {}),
    (128, 256, 512, 128, 256, 128, 2, 2, 2, dict(lda_extra=128, ldc_extra=192)),
]
_CASES = {
    "mxscale_a8w4": [
        (128, 256, 512, 128, 256, 128, 2, 2, 2, {}),
        (128, 512, 1024, 128, 256, 256, 2, 2, 2, {}),
        (256, 256, 512, 256, 256, 256, 2, 2, 2, {}),
        (1024, 1024, 1024, 128, 256, 128, 2, 2, 3, {}),
        (128, 256, 512, 128, 256, 128, 2, 2, 2, dict(out_dtype="f16")),
        (128, 256, 512, 128, 256, 128, 2, 2, 2, dict(lda_extra=64, ldc_extra=96)),
    ],
    "ptpc_a8w8": [
        (256, 256, 512, 256, 256, 128, 2, 2, 4, {}),
        (128, 256, 512, 128, 256, 128, 2, 2, 4, {}),
        (128, 128, 1024, 128, 128, 256, 2, 2, 3, {}),
        (64, 64, 512, 64, 64, 128, 2, 2, 2, {}),
        (128, 96, 512, 128, 96, 128, 2, 2, 2, {}),  # tile_n not a multiple of 128
        (128, 128, 512, 128, 128, 128, 1, 2, 2, {}),  # 2-wave workgroup
        (256, 256, 512, 256, 256, 128, 2, 2, 4, dict(out_dtype="f16", scale_scale=0.02)),
        (128, 256, 512, 128, 256, 128, 2, 2, 4, dict(lda_extra=128, ldc_extra=256)),
    ],
    "mxscale_a8w8": _MX_A8W8_CASES,
    "blockscale_a8w8": _MX_A8W8_CASES,
}
_CASE_PARAMS = [(mode, case) for mode, cases in _CASES.items() for case in cases]


def _case_id(mode, case):
    extra = "-".join(f"{k}{v}" for k, v in case[-1].items())
    return f"{mode}-{case[0]}x{case[1]}x{case[2]}-t{case[3]}x{case[4]}x{case[5]}nb{case[8]}" + (
        f"-{extra}" if extra else ""
    )


@pytest.mark.parametrize("mode, case", _CASE_PARAMS, ids=[_case_id(m, c) for m, c in _CASE_PARAMS])
def test_gemm_shapes(mode, case):
    *tile_cfg, kwargs = case
    _run_case(mode, *tile_cfg, **kwargs)


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


@pytest.mark.parametrize("split_k", [1, 2, 4])
@pytest.mark.parametrize("mode, M, N, K", [(m, *mnk) for m in _PROFILE_MODES for mnk in _MODES[m]["shapes"]])
def test_256x256_shapes(mode, M, N, K, split_k):
    """Shapes per hand-scheduled kernel, with K picked to hit each K-tile remainder."""
    _run_case(mode, M, N, K, *_tuned(mode), split_k=split_k)


_TUNED_PROFILES = [
    (mode, tile)
    for mode in ("mxscale_a8w8_256x256", "blockscale_a8w8_256x256")
    for tile in _MODES[mode]["profile"]["tiles"]
]


@pytest.mark.parametrize(
    "mode, tile", _TUNED_PROFILES, ids=[f"{m}-{'x'.join(map(str, t))}" for m, t in _TUNED_PROFILES]
)
def test_a8w8_256x256_tuned_profiles(mode, tile):
    """Every tiling gemm_a8w8_256x256 claims to support is a separate hand-schedule."""
    _run_case(mode, 1024, 1024, 1024, *tile)


# split_k > 1 with ldc > N also covers the reduce epilogue's padded-C path (one run per row).
@pytest.mark.parametrize("lda_extra, ldc_extra, split_k", [(128, 192, 1), (64, 96, 1), (128, 192, 2)])
@pytest.mark.parametrize("mode", _PROFILE_MODES)
def test_256x256_strided_lda_ldc(mode, lda_extra, ldc_extra, split_k):
    _run_smoke(mode, 1024, lda_extra=lda_extra, ldc_extra=ldc_extra, split_k=split_k)


@pytest.mark.parametrize("mode", [m for m in sorted(_MODES) if _MODES[m].get("f16_kw") is not None])
def test_256x256_f16_out(mode):
    _run_smoke(mode, 1024, out_dtype="f16", **_MODES[mode]["f16_kw"])


_M_GUARD_ROWS = 5 * 256  # more than the worst-case grid padding, at any M


@pytest.mark.parametrize("M", [255, 513, 769, 1025])
@pytest.mark.parametrize("mode", _PROFILE_MODES)
def test_256x256_ragged_m_no_oob_store(mode, M):
    """A grid padded up to whole clusters must still clamp its stores to M rows."""
    _run_smoke(mode, M, c_guard_rows=_M_GUARD_ROWS)


@pytest.mark.parametrize("mode", _PROFILE_MODES)
def test_256x256_back_to_back_determinism(mode):
    """Consecutive launches must not race the previous launch's cluster tail."""
    _require_gpu()
    cfg = _MODES[mode]
    smoke = cfg["smoke"]
    c_gpu, make_args, compiled, error = _run_and_check(
        cfg["build"],
        cfg["launch"],
        1024,
        smoke["N"],
        smoke["K"],
        *_tuned(mode),
        cluster_m=cfg["profile"]["cluster"][0],
        cluster_n=cfg["profile"]["cluster"][1],
    )
    if error is not None:
        raise error
    golden = c_gpu.clone()
    stream = torch.cuda.current_stream()
    for _ in range(32):
        compiled(*make_args(stream))
    torch.cuda.synchronize()
    assert torch.equal(c_gpu, golden), "back-to-back launches drifted from the synchronized result"


def _parse_csv_ints(value: str, n: int, name: str) -> list[int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != n:
        raise SystemExit(f"-{name} needs {n} comma-separated ints, got {value!r}")
    return parts


def _per_run(groups: list, n_modes: int, n_shapes: int, name: str) -> list:
    """Spread one group over every run, one group per mode, or one per (mode, shape) run."""
    if len(groups) == 1:
        groups = groups * n_modes
    if len(groups) == n_modes:
        groups = [g for g in groups for _ in range(n_shapes)]
    if len(groups) != n_modes * n_shapes:
        raise SystemExit(f"-{name} needs 1, {n_modes} (per mode) or {n_modes * n_shapes} (per run) values")
    return groups


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

    parser = argparse.ArgumentParser(
        description="Manual correctness/perf run for the gfx1250 GEMM kernels. Every mode runs every shape; "
        "-tiles/-warps/-nb/-cluster take one value, one per mode, or one per (mode, shape) run.",
    )
    parser.add_argument("-mode", nargs="+", choices=sorted(_MODES), required=True)
    parser.add_argument("-mnk", nargs="+", required=True, metavar="M,N,K", help="one or more shapes")
    parser.add_argument("-tiles", nargs="+", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", nargs="+", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", nargs="+", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", nargs="+", help="cluster_m,cluster_n; default: each mode's own")
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
    runs = [(mode, mnk) for mode in args.mode for mnk in shapes]
    spread = functools.partial(_per_run, n_modes=len(args.mode), n_shapes=len(shapes))
    tiles = spread([_parse_csv_ints(v, 3, "tiles") for v in args.tiles], name="tiles")
    warps = spread([_parse_csv_ints(v, 2, "warps") for v in args.warps], name="warps")
    tile_cfgs = [(*t, *w, nb) for t, w, nb in zip(tiles, warps, spread(args.nb, name="nb"))]
    # The hand-scheduled kernels reject a 1x1 cluster outright, so default to their own.
    clusters = (
        spread([_parse_csv_ints(v, 2, "cluster") for v in args.cluster], name="cluster")
        if args.cluster
        else [_MODES[m].get("profile", {}).get("cluster", (1, 1)) for m, _ in runs]
    )

    rows = []
    for (mode, (M, N, K)), tile_cfg, (cluster_m, cluster_n) in zip(runs, tile_cfgs, clusters):
        cfg = _MODES[mode]
        out_dtype = args.out_dtype if cfg["supports_out_dtype"] else "bf16"
        config = "t{},{},{} w{},{} nb{} c{},{}".format(*tile_cfg, cluster_m, cluster_n)
        for init in args.init_mode:
            kwargs = {"cluster_m": cluster_m, "cluster_n": cluster_n, "const_val": _parse_init_mode(init)}
            if cfg["supports_out_dtype"]:
                kwargs["out_dtype"] = args.out_dtype
            c_gpu, make_args, compiled, error = _run_and_check(
                cfg["build"], cfg["launch"], M, N, K, *tile_cfg, **kwargs
            )
            perf = ["-", "-", "-"]
            if args.bench:
                us = _bench_us(lambda: compiled(*make_args(torch.cuda.current_stream())), c_gpu, warmup=10, iters=100)
                moved = cfg["bytes_moved"](M, N, K)
                perf = [f"{us:.3f}", f"{_tflops(M, N, K, us):.2f}", f"{moved / (us * 1e-6) / 1e12:.3f}"]
            result = "PASS" if error is None else "FAIL"
            rows.append([mode, str(M), str(N), str(K), config, out_dtype, init, *perf, result])
            if error is not None:
                print(f"\n{mode} {M}x{N}x{K} {init}: {error}\n")

    print()
    headers = ["mode", "M", "N", "K", "config", "out", "init_mode", "latency us", "TFLOPS", "BW TB/s", "result"]
    _print_table(headers, rows)
    if any(r[-1] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
