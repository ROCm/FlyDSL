#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 preshuffled-B GEMM tests."""

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
from kernels.gemm.gemm_a4w4_256x256_gfx1250 import launch_gemm_a4w4_256x256  # noqa: E402
from kernels.gemm.gemm_a8w4_256x256_gfx1250 import launch_gemm_a8w4_256x256  # noqa: E402
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


def _u8(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _i8(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _random_fp8_bytes(rows: int, cols: int) -> torch.Tensor:
    """Finite FP8 E4M3 bytes (avoids the 0x7F/0xFF NaN encodings)."""
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8)


def _with_strided_a(a: torch.Tensor, K: int, lda: int) -> torch.Tensor:
    """Return A backed by runtime lda when lda exceeds logical K."""
    if lda == K:
        return a
    out = torch.zeros(a.shape[0], lda, dtype=a.dtype, device=a.device)
    out[:, :K] = a
    return out


def _preshuffle_scale_32x4(scale: torch.Tensor) -> torch.Tensor:
    """[R, K/32] uint8 E8M0 -> [ceil(R/32), K] 32-row x 4-K-group preshuffled layout."""
    rows, k_scale = scale.shape
    row_blocks = (rows + 31) // 32
    if row_blocks * 32 != rows:
        padded = torch.zeros((row_blocks * 32, k_scale), dtype=scale.dtype, device=scale.device)
        padded[:rows] = scale
        scale = padded
    x = scale.view(row_blocks, 32, k_scale // 4, 4).permute(0, 2, 1, 3).contiguous()
    return x.reshape(row_blocks, -1)


def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    return gemm_common_utils.e8m0_to_f32(scale.view(torch.uint8))


def _fp4_tolerances(a_scale: torch.Tensor, b_scale: torch.Tensor, K: int) -> tuple[float, float]:
    """Scale-range-aware tolerance for FP4 weights, whose quantization error needs an
    absolute floor that a purely relative tolerance would not give."""
    exps = [int(s.view(torch.uint8).max().item()) - 127 for s in (a_scale, b_scale)]
    peak_prod_exp = sum(max(0, e) for e in exps)
    return min(5e-2, 1e-2 + 3e-3 * peak_prod_exp), max(5e-2, K * (0.6 + 1.5 * peak_prod_exp))


def _scales_mx32(a_f32, b_f32, M, N, K, fp4_w, scale_exp, _scale_scale):
    a_s = gemm_common_utils.random_e8m0(M, K // SCALE_BLOCK_32, low_exp=scale_exp[0], high_exp=scale_exp[1])
    b_s = gemm_common_utils.random_e8m0(N, K // SCALE_BLOCK_32, low_exp=scale_exp[0], high_exp=scale_exp[1])
    a_sc = _decode_e8m0(a_s).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:M, :K]
    b_sc = _decode_e8m0(b_s).repeat_interleave(SCALE_BLOCK_32, dim=-1)[:N, :K]
    ref = torch.matmul(a_f32[:M, :K] * a_sc, (b_f32[:N, :K] * b_sc).T)
    tol = _fp4_tolerances(a_s, b_s, K) if fp4_w else (1e-2, 5e-2)
    return _preshuffle_scale_32x4(a_s), _preshuffle_scale_32x4(b_s), 0, ref, tol


def _scales_mx128(a_f32, b_f32, M, N, K, _fp4_w, scale_exp, _scale_scale):
    sk = K // SCALE_BLOCK_128
    a_s = gemm_common_utils.random_e8m0(M, sk, low_exp=scale_exp[0], high_exp=scale_exp[1])
    b_s = gemm_common_utils.random_e8m0(N // SCALE_BLOCK_128, sk, low_exp=scale_exp[0], high_exp=scale_exp[1])
    a = a_f32[:M, :K].clone().view(M, sk, SCALE_BLOCK_128) * _decode_e8m0(a_s).unsqueeze(-1)
    b_sc = _decode_e8m0(b_s).repeat_interleave(SCALE_BLOCK_128, dim=0)[:N]
    b = b_f32[:N, :K].clone().view(N, sk, SCALE_BLOCK_128) * b_sc.unsqueeze(-1)
    ref = torch.matmul(a.view(M, K), b.view(N, K).T)
    return a_s.T.contiguous(), b_s, M, ref, (1e-2, 5e-2)  # A-scale is [K/128, M], row stride M


def _scales_ptpc(a_f32, b_f32, M, N, K, _fp4_w, _scale_exp, scale_scale):
    a_s = (scale_scale * (0.5 + torch.rand(M, dtype=torch.float32))).contiguous()
    b_s = (scale_scale * (0.5 + torch.rand(N, dtype=torch.float32))).contiguous()
    ref = torch.matmul(a_f32[:M, :K], b_f32[:N, :K].T) * a_s.view(M, 1) * b_s.view(1, N)
    return a_s, b_s, 0, ref, (2e-2, max(5e-2, 2e-2 * float(ref.abs().max())))


_SCALES = {"mx32": _scales_mx32, "mx128": _scales_mx128, "ptpc": _scales_ptpc}

_A8W4_256_PROFILE = (256, 256, 128, 2, 2, 4, 4, 4)
_A4W4_256_PROFILE = (256, 256, 256, 2, 2, 4, 1, 2)

_MODES = {
    "a8w4_mx32": dict(
        launch=launch_gemm_a8w4_mxscale, fp4_w=True, scale="mx32", a8w8=False, tail=(), wrap=_i8,
        scale_exp=(127, 132), f16_kw=dict(scale_exp=(127, 127)),
    ),
    "a8w4_256x256": dict(
        launch=launch_gemm_a8w4_256x256, fp4_w=True, scale="mx32", a8w8=False, tail=(), wrap=_i8,
        scale_exp=(127, 132), f16_kw=dict(scale_exp=(127, 127)),
        profile=_A8W4_256_PROFILE, smoke=(1024, 512, 256, 256, 128, 2, 2), k_pair=True,
    ),
    "a4w4_256x256": dict(
        launch=launch_gemm_a4w4_256x256, fp4_w=True, scale="mx32", a8w8=False, tail=(), wrap=_i8,
        scale_exp=(127, 132), f16_kw=dict(scale_exp=(127, 127)), fp4_act=True,
        profile=_A4W4_256_PROFILE, smoke=(512, 1024, 256, 256, 256, 2, 2),
    ),
    "a8w8_mx32": dict(
        launch=launch_gemm_a8w8, fp4_w=False, scale="mx32", a8w8=True, tail=(True, 32), wrap=_i8,
        scale_exp=(126, 129), f16_kw=None,
    ),
    "a8w8_mx128": dict(
        launch=launch_gemm_a8w8, fp4_w=False, scale="mx128", a8w8=True, tail=(True, 128), wrap=_i8,
        scale_exp=(126, 129), f16_kw=None,
    ),
    "a8w8_ptpc": dict(
        launch=launch_gemm_a8w8, fp4_w=False, scale="ptpc", a8w8=True, tail=(False, 128), wrap=_u8,
        scale_exp=(126, 129), f16_kw=dict(scale_scale=0.02),
    ),
}  # fmt: skip


def _skip_reason(spec, M, N, K, tile_cfg, cluster):
    """None when the mode supports this shape/tile combination, else why it cannot."""
    tile_m, tile_n, tile_k, _m_warp, _n_warp, num_buffers = tile_cfg
    profile = spec.get("profile")
    if profile is not None:
        if (*tile_cfg, *cluster) != profile:
            return f"this kernel hand-schedules one profile, {profile}"
        if N % (tile_n * cluster[1]) or (-(-M // tile_m)) % cluster[0]:
            return f"the {cluster[0]}x{cluster[1]} cluster needs whole clusters of {tile_m}x{tile_n} tiles"
        if spec.get("k_pair") and K % (tile_k * 2):
            return f"K={K} must divide {tile_k * 2}: one TDM covers two K-tiles"
    if N % tile_n or K % tile_k:
        return f"N={N} / K={K} must divide tile_n={tile_n} / tile_k={tile_k} (the kernel does not pad)"
    if K // tile_k < num_buffers:
        return f"K={K} yields fewer than {num_buffers} K-tiles at tile_k={tile_k}"
    if spec["scale"] == "mx128" and (N % SCALE_BLOCK_128 or K % SCALE_BLOCK_128):
        return f"mx128 needs N={N} and K={K} divisible by {SCALE_BLOCK_128}"
    if spec["scale"] == "mx32":
        if K % SCALE_BLOCK_128 or tile_n % SCALE_BLOCK_32:
            return f"mx32 needs K={K} divisible by 128 and tile_n={tile_n} by {SCALE_BLOCK_32}"
        # The 32x4 shuffle needs whole 32-row supers; only a8w8 handles a 16-row tile.
        if tile_m % SCALE_BLOCK_32 and (spec["fp4_w"] or tile_m != 16):
            return f"mx32 needs tile_m={tile_m} divisible by {SCALE_BLOCK_32}" + ("" if spec["fp4_w"] else " or == 16")
    return None


def _build_case(
    mode,
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
    out_dtype="bf16",
    lda_extra=0,
    ldc_extra=0,
    cluster_m=1,
    cluster_n=1,
    scale_exp=None,
    scale_scale=1.0,
):
    spec = _MODES[mode]
    torch.manual_seed(0)
    if spec.get("fp4_act"):
        a = gemm_common_utils.random_fp4_packed(M, K)  # [M, K//2], two E2M1 nibbles per byte
        a_f32, a_cols = gemm_common_utils.mxfp4_to_f32(a), K // 2
    else:
        a = _random_fp8_bytes(M, K)
        a_f32, a_cols = gemm_common_utils.fp8_e4m3_to_f32(a), K
    if spec["fp4_w"]:
        b = gemm_common_utils.random_fp4_packed(N, K)  # [N, K//2], two E2M1 nibbles per byte
        b_f32, b_cols = gemm_common_utils.mxfp4_to_f32(b), K // 2
    else:
        b = _random_fp8_bytes(N, K)
        b_f32, b_cols = gemm_common_utils.fp8_e4m3_to_f32(b), K
    a_s, b_s, ask, ref, tol = _SCALES[spec["scale"]](
        a_f32, b_f32, M, N, K, spec["fp4_w"], scale_exp or spec["scale_exp"], scale_scale
    )

    lda, ldc = K + lda_extra, N + ldc_extra
    dev = [
        torch.zeros(M, ldc, dtype=_DT[out_dtype], device="cuda"),
        _with_strided_a(a, a_cols, lda if a_cols == K else lda // 2).cuda(),
        gemm_common_utils.preshuffle_b_16x16(b, N, b_cols).cuda(),
        a_s.cuda(),
        b_s.cuda(),
    ]
    c_gpu = dev[0]

    def make_args(stream):
        w = spec["wrap"]
        ptr_args = spec["a8w8"] or spec.get("profile") is not None
        ptrs = [w(t) for t in dev] if ptr_args else [dev[0], w(dev[1]), w(dev[2]), dev[3], dev[4]]
        return (
            *ptrs,
            M,
            stream,
            N,
            K,
            *((ask,) if spec["a8w8"] else ()),
            lda,
            ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            int(out_dtype == "f16"),
            num_buffers,
            cluster_m,
            cluster_n,
            *spec["tail"],
        )

    return c_gpu, make_args, ref, tol


def _assert_case(mode, M, N, K, *tile_cfg, **kwargs):
    """Build inputs, compile+run once, assert against the reference.

    Returns (c_gpu, make_args, compiled) so the perf CLI can replay the same
    compiled kernel without rebuilding it.
    """
    c_gpu, make_args, ref, (rtol, atol) = _build_case(mode, M, N, K, *tile_cfg, **kwargs)
    compiled = flyc.compile(_MODES[mode]["launch"], *make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    torch.testing.assert_close(c_gpu[:M, :N].float().cpu(), ref.float(), rtol=rtol, atol=atol)
    return c_gpu, make_args, compiled


def _run_case(mode, M, N, K, *tile_cfg, **kwargs):
    _require_gpu()
    profile = _MODES[mode].get("profile")
    if profile is not None:  # sweeps that do not parametrize the cluster get the tuned one
        kwargs.setdefault("cluster_m", profile[6])
        kwargs.setdefault("cluster_n", profile[7])
    cluster = (kwargs.get("cluster_m", 1), kwargs.get("cluster_n", 1))
    reason = _skip_reason(_MODES[mode], M, N, K, tile_cfg, cluster)
    if reason:
        pytest.skip(reason)
    _assert_case(mode, M, N, K, *tile_cfg, **kwargs)


_MODE_IDS = sorted(_MODES)
# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
_CASES = [
    (128, 256, 512, 128, 256, 128, 2, 2, 2),
    (128, 512, 1024, 128, 256, 256, 2, 2, 2),
    (256, 256, 512, 256, 256, 256, 2, 2, 2),
    (256, 256, 512, 256, 256, 128, 2, 2, 4),
    (1024, 1024, 1024, 128, 256, 128, 2, 2, 3),
    (128, 128, 1024, 128, 128, 256, 2, 2, 3),
    (64, 64, 512, 64, 64, 128, 2, 2, 2),
    (128, 96, 512, 128, 96, 128, 2, 2, 2),  # tile_n not a multiple of 128
    (128, 128, 512, 128, 128, 128, 1, 2, 2),  # 2-wave workgroup
    (64, 64, 1024, 16, 32, 512, 1, 2, 2),  # tile_m below one 32-row scale super
    (128, 128, 1024, 32, 32, 512, 2, 2, 2),
    (256, 512, 512, 256, 256, 128, 2, 2, 4),
    (257, 512, 4608, 256, 256, 128, 2, 2, 4),
    (256, 512, 640, 256, 256, 128, 2, 2, 4),  # 5 K-tiles -> 1 active stage in rev 1
    (256, 512, 768, 256, 256, 128, 2, 2, 4),  # 6 K-tiles -> 2 active stages
    (257, 512, 896, 256, 256, 128, 2, 2, 4),  # 7 K-tiles -> 3 active, ragged M
    (256, 512, 4736, 256, 256, 128, 2, 2, 4),  # 37 K-tiles, long run
    (12288, 512, 512, 256, 256, 128, 2, 2, 4),  # 48 M-tiles -> grid split 16 x 3
    (16384, 512, 512, 256, 256, 128, 2, 2, 4),  # 64 M-tiles -> grid split 32 x 2
    (8448, 512, 512, 256, 256, 128, 2, 2, 4),  # 33 M-tiles -> no exact split, flat grid
]
_SMOKE = (512, 512, 128, 256, 128, 2, 2)  # N, K, tile_m, tile_n, tile_k, m_warp, n_warp
_SMOKE_BUFFERS = [2, 4]
_RAGGED_M_VALUES = [1, 2, 5, 15, 16, 17, 33, 63, 65, 100, 127, 128, 129, 191, 200, 255, 256, 257, 384, 500, 1000, 2048]


def _run_smoke(mode, M, num_buffers=None, **kwargs):
    spec = _MODES[mode]
    N, K, *tile_cfg = spec.get("smoke", _SMOKE)
    profile = spec.get("profile")
    _run_case(mode, M, N, K, *tile_cfg, num_buffers or (profile[5] if profile else 2), **kwargs)


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers", _CASES)
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_shapes(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    _run_case(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)


@pytest.mark.parametrize("lda_extra, ldc_extra", [(128, 192), (128, 256), (64, 96)])
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_strided_lda_ldc(mode, lda_extra, ldc_extra):
    _run_smoke(mode, 128, lda_extra=lda_extra, ldc_extra=ldc_extra)


@pytest.mark.parametrize("mode", [m for m in _MODE_IDS if _MODES[m]["f16_kw"]])
def test_gemm_f16_out(mode):
    _run_smoke(mode, 128, out_dtype="f16", **_MODES[mode]["f16_kw"])


@pytest.mark.parametrize("num_buffers", _SMOKE_BUFFERS)
@pytest.mark.parametrize("M", _RAGGED_M_VALUES)
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_ragged_m(mode, M, num_buffers):
    _run_smoke(mode, M, num_buffers=num_buffers)


@pytest.mark.parametrize("num_buffers", _SMOKE_BUFFERS)
@pytest.mark.parametrize("cluster_m, cluster_n", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("M", [1, 65, 129, 384])
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_cluster(mode, M, cluster_m, cluster_n, num_buffers):
    _run_smoke(mode, M, num_buffers=num_buffers, cluster_m=cluster_m, cluster_n=cluster_n)


@pytest.mark.parametrize("knob", range(len(_A8W4_256_PROFILE)))
def test_a8w4_256x256_rejects_other_profiles(knob):
    _require_gpu()
    cfg = list(_A8W4_256_PROFILE)
    cfg[knob] = cfg[knob] * 2
    _, make_args, _, _ = _build_case("a8w4_256x256", 256, 512, 512, *cfg[:6], cluster_m=cfg[6], cluster_n=cfg[7])
    with pytest.raises(AssertionError, match="only the tuned"):
        flyc.compile(launch_gemm_a8w4_256x256, *make_args(torch.cuda.current_stream()))


# K/256 mod 3 selects the drain length, so 1024 / 1280 / 1536 walk all three tails.
@pytest.mark.parametrize("M, N, K", [(1024, 1024, 1024), (2048, 1024, 1280), (1024, 2048, 1536)])
def test_a8w4_256x256_cluster4x4(M, N, K):
    """The 4x4 cluster multicasts A across four workgroups and B across four more."""
    _require_gpu()
    _run_case("a8w4_256x256", M, N, K, *_A8W4_256_PROFILE[:6], cluster_m=4, cluster_n=4)


@pytest.mark.parametrize("mode, K", [("a8w4_256x256", 4608), ("a4w4_256x256", 4096)])
def test_256x256_back_to_back_determinism(mode, K):
    """A drifting result across relaunches means a pipeline race, not quantization noise."""
    _require_gpu()
    profile = _MODES[mode]["profile"]
    c_gpu, make_args, compiled = _assert_case(
        mode, 1024, 256 * profile[7], K, *profile[:6], cluster_m=profile[6], cluster_n=profile[7]
    )
    golden = c_gpu.clone()
    stream = torch.cuda.current_stream()
    for _ in range(32):
        compiled(*make_args(stream))
    torch.cuda.synchronize()
    assert torch.equal(c_gpu, golden), "back-to-back launches drifted from the synchronized result"


def _bench_us(launch, output: torch.Tensor, *, warmup: int = 10, iters: int = 100) -> float:
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


def _main():
    import argparse

    def ints(n, name):
        def parse(value):
            parts = [int(x) for x in value.split(",")]
            if len(parts) != n:
                raise argparse.ArgumentTypeError(f"{name} needs {n} comma-separated ints, got {value!r}")
            return parts

        return parse

    parser = argparse.ArgumentParser(description="Manual correctness/perf run for the gfx1250 GEMM kernels")
    parser.add_argument("-mode", choices=_MODE_IDS, required=True)
    parser.add_argument("-mnk", type=ints(3, "-mnk"), required=True, help="M,N,K")
    parser.add_argument("-tiles", type=ints(3, "-tiles"), required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", type=ints(2, "-warps"), required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", type=ints(2, "-cluster"), default=[1, 1], help="cluster_m,cluster_n")
    parser.add_argument("-out-dtype", default="bf16", choices=sorted(_DT))
    parser.add_argument("-bench", action="store_true", help="also measure perf")
    args = parser.parse_args()

    M, N, K = args.mnk
    kwargs = dict(zip(("cluster_m", "cluster_n"), args.cluster))
    if args.out_dtype == "f16":
        kwargs.update(out_dtype="f16", **(_MODES[args.mode]["f16_kw"] or {}))
    c_gpu, make_args, compiled = _assert_case(args.mode, M, N, K, *args.tiles, *args.warps, args.nb, **kwargs)
    print(f"PASSED correctness: mode={args.mode} M={M} N={N} K={K}")

    if args.bench:
        us = _bench_us(lambda: compiled(*make_args(torch.cuda.current_stream())), c_gpu)
        print(
            f"perf: mode={args.mode} M={M} N={N} K={K} {us:.3f}us ({2.0 * M * N * K / (us * 1e-6) / 1e12:.2f} TFLOPS)"
        )


if __name__ == "__main__":
    _main()
