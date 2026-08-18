#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 preshuffled-B GEMM tests."""

import os
import random
import sys
import time

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


def _new_seed():
    seed = int(os.environ["FLYDSL_TEST_SEED"]) if os.environ.get("FLYDSL_TEST_SEED") else random.randrange(1 << 31)
    torch.manual_seed(seed)
    return seed


def _random_fp8_bytes(rows: int, cols: int) -> torch.Tensor:
    """Finite FP8 E4M3 bytes (avoids the 0x7F/0xFF NaN encodings)."""
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8)


def _make_quant_input(rows: int, K: int, fp4: bool, const_val: float | None):
    if const_val is None:
        q = gemm_common_utils.random_fp4_packed(rows, K) if fp4 else _random_fp8_bytes(rows, K)
    else:
        codes = torch.arange(16 if fp4 else 126, dtype=torch.uint8).view(1, -1)
        vals = gemm_common_utils.mxfp4_to_f32(codes)[0, ::2] if fp4 else gemm_common_utils.fp8_e4m3_to_f32(codes)[0]
        match = (vals == const_val).nonzero()
        if not len(match):
            raise ValueError(f"{const_val} is not exactly representable")
        code = int(match[0, 0])
        q = (
            torch.full((rows, K // 2), code | (code << 4), dtype=torch.uint8)
            if fp4
            else torch.full((rows, K), code, dtype=torch.uint8)
        )
    return (q, gemm_common_utils.mxfp4_to_f32(q), K // 2) if fp4 else (q, gemm_common_utils.fp8_e4m3_to_f32(q), K)


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

_SMOKE_TILE = (128, 256, 128, 2, 2)  # tile_m, tile_n, tile_k, m_warp, n_warp for the flexible kernels


def _spec(
    launch,
    *,
    fp4_w=False,
    fp4_act=False,
    scale="mx32",
    profile=None,
    cluster=(1, 1),
    smoke=(512, 512),
    k_pair=1,
    k_whole_rev=False,
    f16=True,
    tensor_args=False,
):
    if f16 is True:
        f16 = dict(scale_scale=0.02) if scale == "ptpc" else dict(scale_exp=(127, 127)) if fp4_w else None
    return dict(
        launch=launch,
        fp4_w=fp4_w,
        fp4_act=fp4_act,
        scale=scale,
        profile=profile,
        cluster=cluster,
        smoke=smoke,
        k_pair=k_pair,
        k_whole_rev=k_whole_rev,
        tensor_args=tensor_args,
        f16_kw=f16 or None,
        a8w8=not fp4_w,  # only the a8w8 launchers take stride_ascale_k
        tail=() if fp4_w else (scale != "ptpc", 32 if scale == "mx32" else 128),  # is_mxscale, block_size
        wrap=_u8 if scale == "ptpc" else _i8,
        scale_exp=(127, 132) if fp4_w else (126, 129),
        tiles=profile[:5] if profile else _SMOKE_TILE,
        num_buffers=profile[5] if profile else 2,
    )


_MODES = {
    "a8w4_mx32": _spec(launch_gemm_a8w4_mxscale, fp4_w=True, tensor_args=True),
    "a8w4_256x256": _spec(
        launch_gemm_a8w4_256x256,
        fp4_w=True,
        cluster=(4, 4),
        k_pair=2,
        profile=(256, 256, 128, 2, 2, 3),
        smoke=(1024, 512),
    ),
    "a4w4_256x256": _spec(
        launch_gemm_a4w4_256x256,
        fp4_w=True,
        fp4_act=True,
        cluster=(4, 4),
        k_whole_rev=True,
        profile=(256, 256, 256, 2, 2, 4),
        smoke=(1024, 1024),
    ),
    "a8w8_mx32": _spec(launch_gemm_a8w8, f16=False),
    "a8w8_mx128": _spec(launch_gemm_a8w8, scale="mx128", f16=False),
    "a8w8_ptpc": _spec(launch_gemm_a8w8, scale="ptpc"),
}
for _tm, _nb in ((256, 4), (256, 2), (128, 4), (128, 3)):
    for _sc in ("mx32", "mx128"):
        _MODES[f"a8w8_{_tm}x256" + ("" if _nb == 4 else f"_nb{_nb}") + ("" if _sc == "mx32" else "_mx128")] = _spec(
            launch_gemm_a8w8_256x256,
            scale=_sc,
            cluster=(4, 4),
            profile=(_tm, 256, 128, 2, 2, _nb),
            smoke=(1024, 1024),
            k_pair=1 if _nb == 4 else 2,
            f16=dict(const_val=0.25),
        )


def _bytes_moved(mode: str, M: int, N: int, K: int) -> int:
    """Logical A/B/scales read plus C written by one GEMM launch."""
    spec = _MODES[mode]
    a_bytes = M * K // (2 if spec["fp4_act"] else 1)
    b_bytes = N * K // (2 if spec["fp4_w"] else 1)
    if spec["scale"] == "mx32":
        scale_bytes = (M + N) * (K // SCALE_BLOCK_32)
    elif spec["scale"] == "mx128":
        scale_bytes = M * (K // SCALE_BLOCK_128) + (N // SCALE_BLOCK_128) * (K // SCALE_BLOCK_128)
    else:
        scale_bytes = (M + N) * 4
    return a_bytes + b_bytes + scale_bytes + M * N * 2


def _skip_reason(spec, N, K, tile_cfg, cluster):
    """None when the mode supports this shape/tile combination, else why it cannot."""
    tile_m, tile_n, tile_k, _m_warp, _n_warp, num_buffers = tile_cfg
    profile = spec["profile"]
    if profile is not None:
        if tile_cfg != profile:
            return f"this kernel hand-schedules one profile, {profile}"
        if cluster != spec["cluster"]:
            return f"this kernel is tuned for a {spec['cluster']} cluster"
        if N % (tile_n * cluster[1]):
            return f"the {cluster[0]}x{cluster[1]} cluster needs N whole clusters of {tile_n}-wide tiles"
        if spec["k_pair"] > 1 and K % (tile_k * spec["k_pair"]):
            return f"K={K} must divide {tile_k * spec['k_pair']}: one TDM covers {spec['k_pair']} K-tiles"
        if spec["k_whole_rev"] and (K // tile_k) % num_buffers:
            return f"K={K} must cover whole {num_buffers}-K-tile revolutions"
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
    cluster=(1, 1),
    scale_exp=None,
    scale_scale=1.0,
    c_guard_rows=0,
    const_val=None,
):
    spec = _MODES[mode]
    seed = _new_seed()
    a, a_f32, a_cols = _make_quant_input(M, K, spec["fp4_act"], const_val)
    b, b_f32, b_cols = _make_quant_input(N, K, spec["fp4_w"], const_val)
    a_s, b_s, ask, ref, tol = _SCALES[spec["scale"]](
        a_f32, b_f32, M, N, K, spec["fp4_w"], scale_exp or spec["scale_exp"], scale_scale
    )

    lda, ldc = K + lda_extra, N + ldc_extra
    c_gpu = torch.full(
        (M + c_guard_rows, ldc), float("nan") if c_guard_rows else 0.0, dtype=_DT[out_dtype], device="cuda"
    )  # noqa: E501
    dev = [
        c_gpu,
        _with_strided_a(a, a_cols, lda if a_cols == K else lda // 2).cuda(),
        gemm_common_utils.preshuffle_b_16x16(b, N, b_cols).cuda(),
        a_s.cuda(),
        b_s.cuda(),
    ]

    def make_args(stream):
        w = spec["wrap"]
        ptrs = [dev[0], w(dev[1]), w(dev[2]), dev[3], dev[4]] if spec["tensor_args"] else [w(t) for t in dev]
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
            *cluster,
            *spec["tail"],
        )

    return c_gpu, make_args, ref, tol, seed


def _assert_case(mode, M, N, K, *tile_cfg, **kwargs):
    """Compile+run once and return replay handles."""
    guard_rows = kwargs.get("c_guard_rows", 0)
    c_gpu, make_args, ref, (rtol, atol), seed = _build_case(mode, M, N, K, *tile_cfg, **kwargs)
    compiled = flyc.compile(_MODES[mode]["launch"], *make_args(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    replay = f"FLYDSL_TEST_SEED={seed} reproduces this input"
    out = c_gpu[:M, :N].float()
    if guard_rows:
        assert not torch.isnan(out).any(), f"OOB load reached the accumulator (NaN in the real output); {replay}"
    torch.testing.assert_close(out.cpu(), ref.float(), rtol=rtol, atol=atol, msg=lambda m: f"{m}\n{replay}")
    if guard_rows:
        clobbered = int((~torch.isnan(c_gpu[M:].float())).sum())
        assert (
            clobbered == 0
        ), f"M={M}: {clobbered} elements written at/after row {M} (store OOB clamp failed); {replay}"
    return c_gpu, make_args, compiled


def _run_case(mode, M, N, K, *tile_cfg, **kwargs):
    _require_gpu()
    spec = _MODES[mode]
    if spec["profile"] is not None:  # sweeps that do not parametrize buffers/cluster get the tuned ones
        kwargs.setdefault("cluster", spec["cluster"])
        tile_cfg = (*tile_cfg[:5], spec["num_buffers"])
    reason = _skip_reason(spec, N, K, tile_cfg, kwargs.get("cluster", (1, 1)))
    if reason:
        pytest.skip(reason)
    _assert_case(mode, M, N, K, *tile_cfg, **kwargs)


def _run_smoke(mode, M, num_buffers=None, **kwargs):
    """Run the mode's own (N, K) smoke shape at the mode's own tiling."""
    spec = _MODES[mode]
    N, K = spec["smoke"]
    _run_case(mode, M, N, K, *spec["tiles"], num_buffers or spec["num_buffers"], **kwargs)


_MODE_IDS = sorted(_MODES)
_PROFILE_MODES = [m for m in _MODE_IDS if _MODES[m]["profile"]]

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
_SMOKE_BUFFERS = [2, 4]
_RAGGED_M_VALUES = [1, 2, 5, 15, 16, 17, 33, 63, 65, 100, 127, 128, 129, 191, 200, 255, 256, 257, 384, 500, 1000, 2048]


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers", _CASES)
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_shapes(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers):
    _run_case(mode, M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)


def _whole_clusters_m(mode, M):
    """Round M up so it fills whole clusters; for tests where the exact M is incidental."""
    spec = _MODES[mode]
    if spec["profile"] is None:
        return M
    tile_m, cluster_m = spec["profile"][0], spec["cluster"][0]
    gx = -(-M // tile_m)
    return tile_m * (gx + (-gx % cluster_m))


@pytest.mark.parametrize("lda_extra, ldc_extra", [(128, 192), (128, 256), (64, 96)])
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_strided_lda_ldc(mode, lda_extra, ldc_extra):
    _run_smoke(mode, _whole_clusters_m(mode, 128), lda_extra=lda_extra, ldc_extra=ldc_extra)


@pytest.mark.parametrize("mode", [m for m in _MODE_IDS if _MODES[m]["f16_kw"]])
def test_gemm_f16_out(mode):
    _run_smoke(mode, _whole_clusters_m(mode, 128), out_dtype="f16", **_MODES[mode]["f16_kw"])


@pytest.mark.parametrize("num_buffers", _SMOKE_BUFFERS)
@pytest.mark.parametrize("M", _RAGGED_M_VALUES)
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_ragged_m(mode, M, num_buffers):
    _run_smoke(mode, M, num_buffers=num_buffers)


@pytest.mark.parametrize("num_buffers", _SMOKE_BUFFERS)
@pytest.mark.parametrize("cluster", [(2, 1), (1, 2), (2, 2)], ids=["2x1", "1x2", "2x2"])
@pytest.mark.parametrize("M", [1, 65, 129, 384])
@pytest.mark.parametrize("mode", _MODE_IDS)
def test_gemm_cluster(mode, M, cluster, num_buffers):
    _run_smoke(mode, M, num_buffers=num_buffers, cluster=cluster)


@pytest.mark.parametrize("knob", [*range(6), "cluster"])
@pytest.mark.parametrize("mode", _PROFILE_MODES)
def test_profile_rejects_untuned_config(mode, knob):
    """A hand-scheduled kernel must reject any tiling/cluster it was not scheduled for."""
    _require_gpu()
    spec = _MODES[mode]
    cfg, cluster = list(spec["profile"]), spec["cluster"]
    if knob == "cluster":
        cluster = (5, 5)
    else:
        cfg[knob] = cfg[knob] * 2 + 1  # do not land on another valid profile of the same kernel
    _, make_args, _, _, _ = _build_case(mode, 256, 512, 512, *cfg, cluster=cluster)
    with pytest.raises(AssertionError, match="only the tuned|cluster"):
        flyc.compile(spec["launch"], *make_args(torch.cuda.current_stream()))


_CLUSTER4X4_SHAPES = {  # (M, N, K) per hand-scheduled kernel; K picked to hit each K-tile remainder
    "a8w8_256x256": [(1024, 1024, 1024), (2048, 1024, 1152), (1024, 2048, 1280), (1000, 1024, 1408)],
    "a8w8_256x256_kpair": [(1024, 1024, 1024), (2048, 1024, 1280), (1024, 2048, 1536), (1000, 1024, 1792)],
    # The 128-row tile leans on small M, where a 256-row tile pads whole cluster rows away.
    "a8w8_128x256": [(128, 1024, 1024), (192, 1024, 1024), (384, 1024, 1024), (1024, 2048, 1280), (1000, 1024, 1792)],
    "a8w4_256x256": [(1024, 1024, 1024), (2048, 1024, 1280), (1024, 2048, 1536)],
    "a4w4_256x256": [(1024, 1024, 1024), (2048, 1024, 3072), (1000, 2048, 4096)],
}


def _cluster4x4_shapes(mode):
    if mode.startswith("a8w8_128x256"):
        return _CLUSTER4X4_SHAPES["a8w8_128x256"]
    if mode.startswith("a8w8_256x256"):
        return _CLUSTER4X4_SHAPES["a8w8_256x256" + ("_kpair" if _MODES[mode]["k_pair"] > 1 else "")]
    return _CLUSTER4X4_SHAPES[mode]


@pytest.mark.parametrize("mode, M, N, K", [(m, *mnk) for m in _PROFILE_MODES for mnk in _cluster4x4_shapes(m)])
def test_256x256_cluster4x4(mode, M, N, K):
    _run_case(mode, M, N, K, *_MODES[mode]["profile"])


_RAGGED_M_GUARD_ROWS = 5 * 256  # > worst case (gx padded up 3 tiles), for any M
_M_GUARD_SWEEP = (255, 512, 513, 769, 1025, 12289)
_GUARD_CASES = [(m, *_MODES[m]["smoke"], _MODES[m]["profile"], _MODES[m]["cluster"]) for m in _PROFILE_MODES] + [
    ("a8w8_mx128", 1536, 1024, tile, (1, 1))
    for tile in [
        (256, 256, 128, 2, 2, 4),
        (128, 192, 128, 2, 2, 2),
        (128, 128, 256, 2, 2, 2),
        (64, 64, 128, 2, 2, 2),
        (32, 32, 512, 2, 2, 2),
    ]
]


@pytest.mark.parametrize("M", _M_GUARD_SWEEP)
@pytest.mark.parametrize(
    "mode, N, K, cfg, cluster", _GUARD_CASES, ids=[f"{m}-t{c[0]}x{c[1]}x{c[2]}" for m, _, _, c, _ in _GUARD_CASES]
)
def test_ragged_m_no_oob_store(mode, N, K, cfg, cluster, M):
    _run_case(mode, M, N, K, *cfg, cluster=cluster, c_guard_rows=_RAGGED_M_GUARD_ROWS)


def _determinism_k(mode):
    if _MODES[mode]["k_whole_rev"]:
        return 4096
    return 4608 if _MODES[mode]["k_pair"] > 1 else 4736


@pytest.mark.parametrize("mode, K", [(m, _determinism_k(m)) for m in _PROFILE_MODES])
def test_256x256_back_to_back_determinism(mode, K):
    _require_gpu()
    spec = _MODES[mode]
    c_gpu, make_args, compiled = _assert_case(
        mode, 1024, 256 * spec["cluster"][1], K, *spec["profile"], cluster=spec["cluster"]
    )
    golden = c_gpu.clone()
    stream = torch.cuda.current_stream()
    for _ in range(32):
        compiled(*make_args(stream))
    torch.cuda.synchronize()
    assert torch.equal(c_gpu, golden), "back-to-back launches drifted from the synchronized result"


def _bench_us(launch, output: torch.Tensor, *, warmup: int = 10, iters: int = 100, gap_us: float = 100.0) -> float:
    """Median per-launch latency (us), each launch timed in isolation."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    output.zero_()
    launch()
    torch.cuda.synchronize()
    if output.abs().max().item() == 0:
        raise RuntimeError("the launch produced an all-zero output; it is not running")

    saturated = gap_us <= 0
    # saturated mode: one event pair around a batch of back-to-back launches
    rounds, batch = (10, max(1, iters // 10)) if saturated else (iters, 1)

    samples = []
    for _ in range(rounds):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(batch):
            launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / batch)
        if not saturated:
            time.sleep(gap_us * 1e-6)
    return sorted(samples)[len(samples) // 2]


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
    parser.add_argument("-tiles", type=ints(3, "-tiles"), help="tile_m,tile_n,tile_k; default: the mode's own")
    parser.add_argument("-warps", type=ints(2, "-warps"), help="m_warp,n_warp; default: the mode's own")
    parser.add_argument("-nb", type=int, help="num_buffers; default: the mode's own")
    parser.add_argument("-cluster", type=ints(2, "-cluster"), help="cluster_m,cluster_n; default: the mode's own")
    parser.add_argument("-out-dtype", default="bf16", choices=sorted(_DT))
    parser.add_argument("-bench", action="store_true", help="also measure perf")
    parser.add_argument("-const", type=float, default=None, help="fill A/B with a representable constant")
    parser.add_argument(
        "-bench-gap-us",
        type=float,
        default=0.0,
        help="host idle after each isolated timed launch; default 0 measures saturated back-to-back throughput",
    )
    args = parser.parse_args()

    M, N, K = args.mnk
    spec = _MODES[args.mode]
    tile_cfg = (
        *(args.tiles or spec["tiles"][:3]),
        *(args.warps or spec["tiles"][3:]),
        args.nb or spec["num_buffers"],
    )
    kwargs = dict(cluster=tuple(args.cluster or spec["cluster"]))
    if args.const is not None:
        kwargs["const_val"] = args.const
    if args.out_dtype == "f16":
        kwargs.update(out_dtype="f16", **(spec["f16_kw"] or {}))
    c_gpu, make_args, compiled = _assert_case(args.mode, M, N, K, *tile_cfg, **kwargs)
    print(f"PASSED correctness: mode={args.mode} M={M} N={N} K={K} tiles={tile_cfg} cluster={kwargs['cluster']}")

    if args.bench:
        us = _bench_us(lambda: compiled(*make_args(torch.cuda.current_stream())), c_gpu, gap_us=args.bench_gap_us)
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        tbps = _bytes_moved(args.mode, M, N, K) / (us * 1e-6) / 1e12
        print(f"perf: mode={args.mode} M={M} N={N} K={K} {us:.3f}us " f"({tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s)")


if __name__ == "__main__":
    _main()
