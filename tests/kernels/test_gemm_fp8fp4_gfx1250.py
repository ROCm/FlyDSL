#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 FP8/FP4 GEMM correctness, layout and pipeline regressions.

Run with pytest, or use the CLI below for correctness and throughput measurements.
"""

import functools
import os
import sys
from dataclasses import dataclass
from typing import Callable

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402,I001
import flydsl.expr as fx  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.gemm_a4w4_256x256_gfx1250 import launch_gemm_a4w4_256x256  # noqa: E402
from kernels.gemm.gemm_a4w4_gfx1250 import launch_gemm_a4w4  # noqa: E402
from kernels.gemm.gemm_a8w4_256x256_gfx1250 import KPAIR_BY_BUFFERS, launch_gemm_a8w4_256x256  # noqa: E402
from kernels.gemm.gemm_a8w4_gfx1250 import launch_gemm_a8w4  # noqa: E402
from kernels.gemm.gemm_a8w8_256x256_gfx1250 import a8w8_kpair, launch_gemm_a8w8_256x256  # noqa: E402
from kernels.gemm.gemm_a8w8_gfx1250 import launch_gemm_a8w8  # noqa: E402
from kernels.gemm.gemm_a8w8_splitk_reduce_gfx1250 import compile_gemm_a8w8_splitk_reduce  # noqa: E402
from tests.kernels.utils import gemm_common_utils as utils  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
_DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


@dataclass(frozen=True)
class _Mode:
    launch: object
    tiles: tuple
    shape: tuple[int, int, int]
    scale: int | str = 32
    fp4_act: bool = False
    fp4_w: bool = False
    supports_out_dtype: bool = True

    @property
    def scheduled(self):
        return self.launch in (launch_gemm_a8w4_256x256, launch_gemm_a4w4_256x256, launch_gemm_a8w8_256x256)

    @property
    def compute(self):
        return self.launch in (launch_gemm_a4w4_256x256, launch_gemm_a8w8_256x256)

    @property
    def cluster(self):
        return (4, 4) if self.scheduled else (1, 1)


_A8W8_TILES = (
    (256, 256, 128, 2, 2, 4),
    (256, 256, 128, 2, 2, 2),
    (128, 128, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 4),
    (128, 256, 128, 2, 2, 3),
)
_A8W4_TILES = tuple((m, n, 128, 2, 2, nb) for m in (256, 128) for n in (256, 128) for nb in (3, 4))
_A4W4_TILES = tuple((m, n, k, 2, 2, 4) for m in (256, 128) for n in (256, 128) for k in (256, 128))
_MX_TILES = ((128, 256, 128, 2, 2, 2), (256, 256, 128, 2, 2, 4), (128, 256, 128, 2, 2, 3))
_MODES = {
    "a4w4": _Mode(
        launch_gemm_a4w4,
        ((32, 128, 256, 1, 4, 3), (16, 128, 128, 1, 4, 2), (128, 256, 256, 2, 2, 3), (128, 512, 256, 2, 2, 3)),
        (65, 137, 800),
        fp4_act=True,
        fp4_w=True,
    ),
    "mxscale_a8w4": _Mode(
        launch_gemm_a8w4,
        (_MX_TILES[0], (128, 256, 256, 2, 2, 2), (256, 256, 256, 2, 2, 2), _MX_TILES[2]),
        (1024, 512, 512),
        fp4_w=True,
    ),
    "ptpc_a8w8": _Mode(
        launch_gemm_a8w8,
        (
            (128, 128, 128, 2, 2, 4),
            (256, 256, 128, 2, 2, 4),
            (128, 128, 256, 2, 2, 3),
            (64, 64, 128, 2, 2, 2),
            (128, 96, 128, 2, 2, 2),
            (128, 128, 128, 1, 2, 2),
        ),
        (1024, 256, 512),
        scale="ptpc",
    ),
    "blockscale_a8w8": _Mode(launch_gemm_a8w8, _MX_TILES, (1024, 512, 512), scale=128, supports_out_dtype=False),
    "mxscale_a8w8": _Mode(launch_gemm_a8w8, _MX_TILES, (1024, 512, 512), supports_out_dtype=False),
    "mxscale_a8w4_256x256": _Mode(launch_gemm_a8w4_256x256, _A8W4_TILES, (1024, 1024, 512), fp4_w=True),
    "mxscale_a4w4_256x256": _Mode(launch_gemm_a4w4_256x256, _A4W4_TILES, (1024, 1024, 1024), fp4_act=True, fp4_w=True),
    "mxscale_a8w8_256x256": _Mode(launch_gemm_a8w8_256x256, _A8W8_TILES, (1024, 1024, 1024)),
    "blockscale_a8w8_256x256": _Mode(launch_gemm_a8w8_256x256, _A8W8_TILES, (1024, 1024, 1024), scale=128),
}


@pytest.fixture(scope="module", autouse=True)
def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def _pointer(tensor, unsigned=False):
    return flyc.from_c_void_p(fx.Uint8 if unsigned else fx.Int8, tensor.data_ptr(), assumed_align=16)


def _operand(rows, K, fp4, const_val, out_dtype, signed=False):
    decode = utils.mxfp4_to_f32 if fp4 else utils.fp8_e4m3_to_f32
    cols = K // 2 if fp4 else K
    if const_val is None:
        # Use a moderate range for f16 and cancellation cases; other bf16 cases
        # exercise the full FP8 exponent range with the original tolerances.
        data = (
            utils.random_fp4_packed(rows, K, device="cuda")
            if fp4
            else torch.randint(
                0, 56 if signed else (64 if out_dtype == "f16" else 126), (rows, cols), dtype=torch.uint8, device="cuda"
            )
        )
        if signed and not fp4:
            data |= torch.randint(0, 2, data.shape, dtype=torch.uint8, device="cuda") << 7
    else:
        codes = torch.arange(16 if fp4 else 256, dtype=torch.uint8).view(1, -1)
        values = decode(codes)[0, ::2] if fp4 else decode(codes)[0]
        matches = (values == const_val).nonzero()
        if not len(matches):
            raise ValueError(f"{const_val} is not exactly representable in {'fp4' if fp4 else 'fp8'}")
        code = int(matches[0, 0])
        data = torch.full((rows, cols), code | (code << 4) if fp4 else code, dtype=torch.uint8, device="cuda")
    return data, decode(data)


def _preshuffle_scale(scale):
    """Pad and pack E8M0 scales in 32-row x 4-K-group order."""
    rows, cols = scale.shape
    blocks = (rows + 31) // 32
    scale = torch.nn.functional.pad(scale, (0, 0, 0, blocks * 32 - rows))
    return scale.view(blocks, 32, cols // 4, 4).permute(0, 2, 1, 3).contiguous().view(blocks, -1)


def _scales_and_reference(cfg, a, b, out_dtype, const_val, scale_scale):
    M, K = a.shape
    N = b.shape[0]
    if cfg.scale == "ptpc":

        def scale(rows):
            values = (
                torch.full((rows,), scale_scale, device="cuda")
                if const_val is not None
                else (scale_scale * (0.5 + torch.rand(rows, device="cuda")))
            )
            return values.float().contiguous()

        sa, sb = scale(M), scale(N)
        ref = (a @ b.T) * sa[:, None] * sb[None, :]
        return sa, sb, ref, (2e-2, max(5e-2, 2e-2 * float(ref.abs().max()))), 0

    block = cfg.scale
    low, high = (
        (127, 127) if out_dtype == "f16" and block == 32 else ((127, 132) if cfg.fp4_w and block == 32 else (126, 129))
    )

    if cfg.launch is launch_gemm_a4w4:
        low, high = 124, 129

    def scale(rows):
        groups = max(1, (K + block - 1) // block) if cfg.launch is launch_gemm_a4w4 else K // block
        if const_val is None:
            return utils.random_e8m0(rows, groups, low_exp=low, high_exp=high, device="cuda")
        return torch.full((rows, groups), min(max(127, low), high), dtype=torch.uint8, device="cuda")

    sa, sb = scale(M), scale(N if block == 32 else N // block)
    a_scale = utils.e8m0_to_f32(sa).repeat_interleave(block, dim=1)[:, :K]
    b_scale = utils.e8m0_to_f32(sb).repeat_interleave(block, dim=1)[:, :K]
    if block == 128:
        b_scale = b_scale.repeat_interleave(block, dim=0)
    ref = (a * a_scale) @ (b * b_scale).T
    tol = (1e-2, 5e-2)
    if cfg.launch is launch_gemm_a4w4:
        ref, tol = ref.to(_DT[out_dtype]).float(), (0, 0)
    if cfg.fp4_w and out_dtype != "f16" and cfg.launch is not launch_gemm_a4w4:
        peak_exp = max(0, int(sa.max()) - 127) + max(0, int(sb.max()) - 127)
        tol = (min(5e-2, 1e-2 + 3e-3 * peak_exp), max(5e-2, K * (0.6 + 1.5 * peak_exp)))
    if block == 32:
        sa = torch.nn.functional.pad(sa, (0, (-sa.shape[1]) % 4), value=127)
        sb = torch.nn.functional.pad(sb, (0, (-sb.shape[1]) % 4), value=127)
        return _preshuffle_scale(sa), _preshuffle_scale(sb), ref, tol, 0
    return sa.T.contiguous(), sb, ref, tol, M


@dataclass
class _Run:
    output: torch.Tensor
    reference: torch.Tensor
    tolerance: tuple[float, float]
    launch: Callable[[], None]
    buffers: tuple  # Keep tensor allocations alive while launch arguments hold raw pointers.

    def check(self, repeats=0):
        torch.cuda.synchronize()
        M, N = self.reference.shape
        actual = self.output[:M, :N]
        assert torch.isfinite(self.reference).all(), "reference overflowed; adjust the test inputs"
        torch.testing.assert_close(actual.float(), self.reference, rtol=self.tolerance[0], atol=self.tolerance[1])
        if repeats:
            golden = actual.clone()
            for _ in range(repeats):
                self.launch()
            torch.cuda.synchronize()
            assert torch.equal(actual, golden), "back-to-back output drifted across launches"
        assert torch.isnan(self.output[M:]).all(), f"store wrote past M={M}"
        assert torch.isnan(self.output[:M, N:]).all(), f"store wrote into ldc padding past N={N}"


def _execute(
    cfg,
    shape,
    tile,
    *,
    out_dtype="bf16",
    lda_extra=0,
    ldc_extra=0,
    cluster_m=1,
    cluster_n=1,
    scale_scale=1.0,
    const_val=None,
    c_guard_rows=0,
    split_k=1,
    b_preshuffle=True,
    a_preshuffle=False,
    persistent_n_tiles=1,
    fused_splitk=False,
    bounded_m=None,
    signed=False,
):
    """Prepare operands and reference, then compile GEMM and its optional FP32 reducer once."""
    M, N, K = shape
    tm, tn, tk, mw, nw, nb = tile
    general_a4w4 = cfg.launch is launch_gemm_a4w4
    if general_a4w4:
        if not b_preshuffle:
            raise ValueError("general A4W4 requires 16x16-byte B preshuffle")
        fused_splitk = split_k > 1
    torch.manual_seed(0)
    packed_k = max(32, (K + 31) // 32 * 32) if cfg.launch is launch_gemm_a4w4 else K
    a, a_f32 = _operand(M, packed_k, cfg.fp4_act, const_val, out_dtype, signed)
    b, b_f32 = _operand(N, packed_k, cfg.fp4_w, const_val, out_dtype, signed)
    if cfg.launch is launch_gemm_a4w4:
        for operand in (a, b):
            operand[:, (K + 1) // 2 :] = 0
            if K % 2:
                operand[:, K // 2] &= 0x0F
        a_f32, b_f32 = a_f32[:, :K], b_f32[:, :K]
    sa, sb, ref, tol, stride_sa = _scales_and_reference(cfg, a_f32, b_f32, out_dtype, const_val, scale_scale)
    if split_k > 1 and not cfg.scheduled and cfg.launch is not launch_gemm_a4w4:
        tol = (tol[0], tol[1] + split_k * float(ref.abs().max()) * 2**-8)
    lda, ldc = packed_k + lda_extra, N + ldc_extra
    a = torch.nn.functional.pad(a, (0, lda_extra // (2 if cfg.fp4_act else 1)))
    if a_preshuffle:
        a = utils.preshuffle_a_2x128(a)
    if b_preshuffle or cfg.launch not in (launch_gemm_a8w4, launch_gemm_a4w4):
        if cfg.launch is launch_gemm_a4w4:
            b = torch.nn.functional.pad(b, (0, 0, 0, (-N) % 16))
        b = utils.preshuffle_b_16x16(b, b.shape[0], packed_k // (2 if cfg.fp4_w else 1))
    out = torch.full((M + c_guard_rows, ldc), float("nan"), dtype=_DT[out_dtype], device="cuda")
    m_tiles = (M + tm - 1) // tm
    if fused_splitk:
        padded_m = ((m_tiles + cluster_m - 1) // cluster_m) * cluster_m
        scratch_rows = tm // 2 if split_k == 2 else tm
        padded_n = ((N + tn * cluster_n - 1) // (tn * cluster_n)) * cluster_n
        partials = torch.empty(padded_m * padded_n * split_k * scratch_rows * tn, dtype=torch.float32, device="cuda")
    elif split_k > 1:
        partials = torch.empty(
            (split_k, M, ldc),
            dtype=torch.float32 if cfg.scheduled or cfg.launch is launch_gemm_a4w4 else out.dtype,
            device="cuda",
        )
    else:
        partials = out
    ptr = functools.partial(_pointer, unsigned=cfg.scale == "ptpc")
    tensor_args = cfg.launch is launch_gemm_a8w4
    buffers = (partials, ptr(a), ptr(b), sa, sb) if tensor_args else tuple(ptr(t) for t in (partials, a, b, sa, sb))
    bounded_m = (M % tm != 0 or m_tiles % cluster_m != 0) if bounded_m is None else bounded_m
    tail = () if general_a4w4 else (b_preshuffle,) if tensor_args else (a_preshuffle,)
    if cfg.compute:
        tail += (() if cfg.fp4_act else (persistent_n_tiles,)) + (fused_splitk, bounded_m)
    args = (
        *buffers,
        M,
        torch.cuda.current_stream(),
        N,
        K,
        *((stride_sa,) if not cfg.fp4_w else ()),
        lda,
        ldc,
        *((ptr(out),) if cfg.compute or general_a4w4 else ()),
        tm,
        tn,
        tk,
        mw,
        nw,
        int(out_dtype == "f16"),
        nb,
        cluster_m,
        cluster_n,
        *((cfg.scale != "ptpc", 128 if cfg.scale == "ptpc" else cfg.scale) if not cfg.fp4_w else ()),
        split_k,
        *tail,
    )
    compiled = flyc.compile(cfg.launch, *args)  # Also executes the first launch.
    reduce, reduce_args = None, ()
    if split_k > 1 and not fused_splitk:
        dtype_name = next(name for name, dtype in _DT.items() if dtype == partials.dtype)
        reduce_args = (
            _pointer(partials, True),
            _pointer(out, True),
            M * N if ldc == N else N,
            1 if ldc == N else M,
            ldc,
            M * ldc * partials.element_size(),
            torch.cuda.current_stream(),
        )
        reduce = flyc.compile(
            compile_gemm_a8w8_splitk_reduce(split_k=split_k, out_dtype_str=out_dtype, partial_dtype_str=dtype_name),
            *reduce_args,
        )

    def launch():
        compiled(*args)
        if reduce is not None:
            reduce(*reduce_args)

    return _Run(out, ref, tol, launch, (a, b, sa, sb, partials))


def _unsupported(cfg, shape, tile, options):
    """Return the first unsupported configuration, shared by pytest and the CLI."""
    M, N, K = shape
    tm, tn, tk, _, _, nb = tile
    cm, cn = options.get("cluster_m", 1), options.get("cluster_n", 1)
    sk, ps = options.get("split_k", 1), options.get("persistent_n_tiles", 1)
    general_a4w4 = cfg.launch is launch_gemm_a4w4
    if min(M, N, *tile, cm, cn, sk, ps) <= 0 or (K < 0 if general_a4w4 else K <= 0):
        return "shape, tiles, cluster and split counts must be positive"
    step, minimum = (32, 32) if cfg.launch is launch_gemm_a4w4 else (tk, tk * nb)
    if cfg.scheduled:
        if cfg.launch is launch_gemm_a8w4_256x256:
            step = tk * KPAIR_BY_BUFFERS.get(nb, 1)
            minimum = 512 if (tm, tn) == (256, 256) else step
        elif cfg.launch is launch_gemm_a8w8_256x256:
            step, minimum = tk * a8w8_kpair(tn, nb), 512
        else:
            minimum = tk
    packed_k = max(32, (K + 31) // 32 * 32) if general_a4w4 else K
    lda_bytes = (packed_k + options.get("lda_extra", 0)) // (2 if cfg.fp4_act else 1)
    checks = [
        (not (cm < 16 and cn < 16 and cm * cn <= 16), "cluster dimensions must be below 16 with at most 16 workgroups"),
        (cfg.scheduled and tile not in cfg.tiles, f"unsupported tile profile {tile}"),
        (cfg.scheduled and cm * cn == 1, "hand-scheduled kernels require a cluster of 2..16 workgroups"),
        (general_a4w4 and not options.get("b_preshuffle", True), "general A4W4 requires 16x16-byte B preshuffle"),
        (
            general_a4w4
            and (
                sk not in (1, 2, 4, 8)
                or cm * cn * sk > 16
                or tm % sk
                or (sk > 1 and tm // sk < tile[3] * tile[4] * 128 // tn)
            ),
            "general A4W4 split-K uses 2/4/8 splits in one cluster with complete epilogue row stripes",
        ),
        (
            cfg.launch is not launch_gemm_a4w4 and N % (tn * cn) != 0,
            f"N={N} must cover whole clusters of {tn}-wide tiles",
        ),
        (
            not general_a4w4 and (K % (step * sk) != 0 or K // sk < minimum),
            f"K={K} needs multiples of {step} and at least {minimum} per split",
        ),
        (
            not general_a4w4 and cfg.scale != "ptpc" and K % cfg.scale != 0,
            f"K must be divisible by the scale block size {cfg.scale}",
        ),
        (cfg.scale == 128 and N % 128 != 0, "blockscale N must be divisible by 128"),
        (
            options.get("a_preshuffle")
            and (
                cfg.launch in (launch_gemm_a8w4, launch_gemm_a4w4)
                or tm % 2
                or lda_bytes % 128
                or (cfg.fp4_act and tk != 256)
            ),
            "A preshuffle needs a supported launcher, even tile_m, whole 128-byte rows and FP4 tile_k=256",
        ),
        (
            ps > 1 and (cfg.launch is not launch_gemm_a8w8_256x256 or sk != 1 or (N // tn) % (ps * cn)),
            "persistent N tiles need the A8W8 compute kernel, split_k=1 and whole clusters",
        ),
        (
            options.get("fused_splitk")
            and (not (cfg.compute or general_a4w4) or sk not in (2, 4, 8) or tm % sk or cm * cn * sk > 16),
            "fused split-K needs 2/4/8 splits dividing tile_m in one cluster of at most 16 workgroups",
        ),
    ]
    return next((message for bad, message in checks if bad), None)


# Cases describe behavior; the fixture and one test body handle every kernel.
def _case(mode, label, shape=None, tile=None, **options):
    return pytest.param(mode, shape, tile, options, id=f"{mode}-{label}")


def _cases():
    for mode, cfg in _MODES.items():
        _, N, K = cfg.shape
        # WMMA rows, 32-row scale blocks, tile edges and partially populated clusters.
        if cfg.launch is launch_gemm_a4w4:
            for shape in ((1, 1, 1), (17, 33, 31), (33, 129, 257), (65, 257, 544)):
                yield _case(mode, f"bounds-{shape}", shape, repeats=16)
            for tile in cfg.tiles:
                yield _case(mode, f"layout-{tile}", (65, 137, 800), tile, repeats=16)
            for k in (0, 1, 128, 224, 256, 288, 512, 768, 1024, 1536):
                yield _case(mode, f"drain-{k}", (33, 65, k))
            for sk in (2, 4, 8):
                yield _case(mode, f"fused-split-{sk}", (33, 129, 1057), split_k=sk, repeats=32)
                yield _case(mode, f"fused-f16-{sk}", (17, 65, 257), split_k=sk, out_dtype="f16", ldc_extra=32)
            yield _case(mode, "fused-padded-cluster", (65, 137, 1057), split_k=4, cluster_m=2, cluster_n=2)
            yield _case(mode, "fused-zero-k", (17, 65, 0), split_k=8)
            yield _case(mode, "fused-short-k-nb2", (17, 65, 1), (16, 128, 128, 1, 4, 2), split_k=4)
            yield _case(mode, "stride-f16", out_dtype="f16", lda_extra=64, ldc_extra=96)
            yield _case(mode, "constant", const_val=0.5)
            yield _case(mode, "cluster-padded", (65, 137, 1057), cluster_m=2, cluster_n=2, repeats=32)
            yield _case(
                mode,
                "large-tile-general",
                (512, 512, 1536),
                (256, 256, 256, 2, 2, 4),
                cluster_m=2,
                cluster_n=2,
                repeats=32,
            )
            continue
        for M in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513, 769, 1025, 2048):
            yield _case(mode, f"m-tail-{M}", (M, N, K))
        for M in (1, 129, 384):
            for cm, cn in ((2, 1), (1, 2), (2, 2)):
                yield _case(mode, f"cluster-{cm}x{cn}-m{M}", (M, N, K), cluster_m=cm, cluster_n=cn)
        for tile in cfg.tiles:
            # Odd M also exercises row-pair padding in the A preshuffle.
            supports_apre = cfg.launch is not launch_gemm_a8w4 and (not cfg.fp4_act or tile[2] == 256)
            for apre in (False, True) if supports_apre else (False,):
                shape = (777, tile[1] * 4, max(K, tile[2] * tile[5]))
                yield _case(
                    mode,
                    f"tile-{'x'.join(map(str, tile))}-apre{int(apre)}",
                    shape,
                    tile,
                    a_preshuffle=apre,
                    repeats=32 if cfg.scheduled and not apre else 0,
                )
        for lda, ldc, sk in ((128, 192, 1), (64, 96, 1), (128, 192, 2)):
            yield _case(
                mode, f"stride-a{lda}-c{ldc}-sk{sk}", (1024, N, K * sk), lda_extra=lda, ldc_extra=ldc, split_k=sk
            )
        if cfg.supports_out_dtype:
            yield _case(mode, "f16", out_dtype="f16", scale_scale=0.02 if cfg.scale == "ptpc" else 1.0)
        yield _case(mode, "constant", const_val=0.5)
        if not cfg.fp4_act:  # FP4 random inputs already include both signs.
            yield _case(mode, "signed-operands", signed=True)
        if not cfg.scheduled:
            if cfg.launch is launch_gemm_a8w4:
                yield _case(mode, "b-row-major", b_preshuffle=False)
            continue
        # Every split has enough K work; no collected case silently skips.
        for sk in (2, 3, 4):
            yield _case(mode, f"split-{sk}-m-tail", (1000, N, K * sk), split_k=sk)
        # cm=4 actually factors the M grid into z; cm=3 must leave it intact.
        for cm in (4, 3):
            yield _case(mode, f"m-grid-cm{cm}", (12000, 256, 1024), cluster_m=cm, cluster_n=1)
        yield _case(mode, "m-grid-split", (12001, 256, 2048), cluster_m=4, cluster_n=1, split_k=2)
        if cfg.fp4_w:
            for cm, cn in ((4, 2), (2, 4), (8, 1), (1, 8), (3, 1)):
                yield _case(
                    mode, f"multicast-{cm}x{cn}", (777, 2048, 1024), cluster_m=cm, cluster_n=cn, c_guard_rows=2048
                )
        if cfg.launch is launch_gemm_a8w4_256x256:
            yield _case(mode, "short-k", (1024, 1024, 128), (128, 128, 128, 2, 2, 4), cluster_m=2, cluster_n=2)
            yield _case(
                mode, "tile-split-2", (513, 512, 1024), (128, 256, 128, 2, 2, 3), split_k=2, cluster_m=2, cluster_n=2
            )
            for k in (1280, 1536):
                yield _case(mode, f"k-drain-{k}", (2048, 1024, k))
        if cfg.fp4_act:
            yield _case(mode, "short-k", (1024, 1024, 128), (128, 128, 128, 2, 2, 4), cluster_m=2, cluster_n=2)
            for k in (256, 512, 768, 1280, 1536, 1792, 2304):
                yield _case(mode, f"k-drain-{k}", (512, 1024, k), cluster_m=2, cluster_n=4, a_preshuffle=k == 1536)
            yield _case(mode, "k-drain-1536-raw", (512, 1024, 1536), cluster_m=2, cluster_n=4)
            yield _case(
                mode, "tile-split-2", (777, 512, 1024), (128, 128, 128, 2, 2, 4), split_k=2, cluster_m=2, cluster_n=2
            )
        if cfg.launch is launch_gemm_a8w8_256x256:
            for k in (1152, 1280, 1408):
                yield _case(mode, f"k-drain-{k}", (1000, 1024, k))
            # Two K tiles per slot: whole and partial final revolutions of slot pairs.
            for k in (1280, 1536):
                yield _case(mode, f"k-drain-{k}-kpair2", (1000, 1024, k), (128, 256, 128, 2, 2, 3))
            for ps in (2, 4):
                yield _case(
                    mode, f"persistent-n-{ps}", (777, 2048, 1024), persistent_n_tiles=ps, a_preshuffle=True, cluster_n=2
                )
        if cfg.compute:
            yield _case(
                mode,
                "fused-large-m",
                (12001, 256, 2048),
                split_k=2,
                cluster_m=2,
                cluster_n=1,
                fused_splitk=True,
            )
            for M, N, K, sk, cm, cn in (
                (1024, 1024, 2048, 2, 2, 2),
                (1001, 512, 2048, 2, 2, 1),
                (769, 512, 4096, 4, 2, 1),
                (513, 512, 8192, 8, 1, 2),
                (256, 512, 2048, 2, 4, 1),
                (257, 512, 2048, 2, 4, 1),
            ):
                yield _case(
                    mode,
                    f"fused-{sk}-m{M}",
                    (M, N, K),
                    split_k=sk,
                    cluster_m=cm,
                    cluster_n=cn,
                    fused_splitk=True,
                    repeats=32 if M == 1024 else 0,
                )
            for fused, sk in ((True, 2), (False, 4)):
                yield _case(
                    mode,
                    f"split-{sk}-strided-f16",
                    (1001, 1024, 1024 * sk),
                    split_k=sk,
                    cluster_m=2,
                    cluster_n=1,
                    fused_splitk=fused,
                    out_dtype="f16",
                    lda_extra=128,
                    ldc_extra=64,
                    const_val=0.5,
                )
            if cfg.fp4_act:
                yield _case(
                    mode,
                    "fused-8-cluster-m",
                    (513, 256, 8192),
                    split_k=8,
                    cluster_m=2,
                    cluster_n=1,
                    fused_splitk=True,
                )
                yield _case(
                    mode,
                    "fused-tail-k3",
                    (513, 512, 1536),
                    split_k=2,
                    cluster_m=2,
                    cluster_n=1,
                    fused_splitk=True,
                    a_preshuffle=True,
                )


@pytest.fixture
def gemm(mode, shape, tile, options):
    cfg = _MODES[mode]
    shape, tile = shape or cfg.shape, tile or cfg.tiles[0]
    options = {"cluster_m": cfg.cluster[0], "cluster_n": cfg.cluster[1], **options}
    options.pop("repeats", None)
    options.setdefault("c_guard_rows", tile[0] * options["cluster_m"])
    reason = _unsupported(cfg, shape, tile, options)
    assert reason is None, f"invalid test case: {reason}"
    return _execute(cfg, shape, tile, **options)


@pytest.mark.parametrize("mode,shape,tile,options", tuple(_cases()))
def test_gemm(gemm, options):
    gemm.check(repeats=options.get("repeats", 0))


@pytest.mark.parametrize("K", (256, 512, 768, 1536, 16384))
def test_a4w4_k_is_runtime_in_jit_cache(K):
    """Short prologues, drain remainders and long K share a compiled launcher."""
    launch = launch_gemm_a4w4_256x256
    launch._ensure_sig()
    ptr = flyc.from_c_void_p(fx.Int8, 0, assumed_align=16)
    args = [ptr] * 5 + [1536, fx.Stream(0), 24576, 1536, 16384, 24576, ptr, 256, 256, 256, 2, 2, 0, 4, 2, 4]

    def key(k):
        args[8] = k
        bound = launch._sig.bind(*args)
        bound.apply_defaults()
        return launch._resolve_and_make_cache_key(bound.arguments)

    assert key(K) == key(1536)


def _bench_us(launch, output, *, warmup=10, iters=100):
    """Median per-launch latency (us) from saturated back-to-back throughput."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    output.zero_()
    launch()
    torch.cuda.synchronize()
    if output.abs().max().item() == 0:
        raise RuntimeError("the launch produced an all-zero output; it is not running")
    samples = []
    batch = max(1, iters // 10)
    for _ in range(10):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(batch):
            launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / batch)
    return sorted(samples)[len(samples) // 2]


def _bytes_moved(cfg, M, N, K):
    a_bytes = M * K // (2 if cfg.fp4_act else 1)
    b_bytes = N * K // (2 if cfg.fp4_w else 1)
    scale_bytes = (M + N) * 4 if cfg.scale == "ptpc" else (M + (N if cfg.scale == 32 else N // 128)) * (K // cfg.scale)
    return a_bytes + b_bytes + scale_bytes + M * N * 2


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
    """'random' -> None (random fill); 'const,<float>' -> that constant A/B fill value (scales go constant too)."""
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
        "-tiles/-warps/-nb/-cluster/-ps/-sk take one value, one per mode, or one per (mode, shape) run.",
    )
    parser.add_argument("-mode", nargs="+", choices=sorted(_MODES), required=True)
    parser.add_argument("-mnk", nargs="+", required=True, metavar="M,N,K", help="one or more shapes")
    parser.add_argument("-tiles", nargs="+", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", nargs="+", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", "-n", nargs="+", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", nargs="+", help="cluster_m,cluster_n; default: each mode's own")
    parser.add_argument("-out-dtype", default="bf16", choices=["bf16", "f16"])
    parser.add_argument("-bench", action="store_true", help="also measure perf (warmup=10, iters=100)")
    parser.add_argument("-sk", "--split-k", nargs="+", type=int, default=[1], help="K splits per run (default 1)")
    parser.add_argument("--fused-splitk", action="store_true", help="reduce K splits within each hardware cluster")
    parser.add_argument(
        "-no-b-preshuffle",
        dest="b_preshuffle",
        action="store_false",
        help="disable the 16x16 B preshuffle for mxscale_a8w4 (default: enabled)",
    )
    parser.add_argument(
        "-ps",
        nargs="+",
        type=int,
        default=[1],
        help="persistent_n_tiles: N-tiles walked per workgroup (a8w8 256x256 only, default 1)",
    )
    parser.add_argument(
        "--apre",
        nargs="+",
        type=int,
        choices=[0, 1],
        default=[0, 1],
        metavar="0|1",
        help="A-preshuffle variant(s) to run: 0 (row-major) and/or 1 (2x128-tiled A); default both",
    )
    parser.add_argument(
        "--init-mode",
        nargs="+",
        default=["random", "const,0.5"],
        metavar="MODE",
        help="A/B/scale fill(s) to run: 'random' and/or 'const,<float>' (default: both)",
    )
    args = parser.parse_args()

    shapes = [_parse_csv_ints(v, 3, "mnk") for v in args.mnk]
    runs = [(mode, mnk) for mode in args.mode for mnk in shapes]
    spread = functools.partial(_per_run, n_modes=len(args.mode), n_shapes=len(shapes))
    tiles = spread([_parse_csv_ints(v, 3, "tiles") for v in args.tiles], name="tiles")
    warps = spread([_parse_csv_ints(v, 2, "warps") for v in args.warps], name="warps")
    tile_cfgs = [(*t, *w, nb) for t, w, nb in zip(tiles, warps, spread(args.nb, name="nb"))]
    ps_tiles = spread(args.ps, name="ps")
    split_ks = spread(args.split_k, name="sk")
    # The hand-scheduled kernels reject a 1x1 cluster outright, so default to their own.
    clusters = (
        spread([_parse_csv_ints(v, 2, "cluster") for v in args.cluster], name="cluster")
        if args.cluster
        else [_MODES[m].cluster for m, _ in runs]
    )

    rows = []
    for (mode, (M, N, K)), tile_cfg, (cluster_m, cluster_n), ps, split_k in zip(
        runs, tile_cfgs, clusters, ps_tiles, split_ks
    ):
        cfg = _MODES[mode]
        out_dtype = args.out_dtype if cfg.supports_out_dtype else "bf16"
        config = "t{},{},{} w{},{} nb{} c{},{}".format(*tile_cfg, cluster_m, cluster_n)
        if ps > 1:
            config += f" ps{ps}"
        if split_k > 1:
            config += f" sk{split_k}" + (" fused" if args.fused_splitk else "")
        if mode in ("mxscale_a8w4", "a4w4"):
            config += f" b={'preshuffle' if args.b_preshuffle else 'row-major+pad'}"
        for apre in args.apre:
            kwargs = {
                "cluster_m": cluster_m,
                "cluster_n": cluster_n,
                "a_preshuffle": bool(apre),
                "persistent_n_tiles": ps,
                "split_k": split_k,
                "fused_splitk": args.fused_splitk,
            }
            if mode in ("mxscale_a8w4", "a4w4"):
                kwargs["b_preshuffle"] = args.b_preshuffle
            if cfg.supports_out_dtype:
                kwargs["out_dtype"] = args.out_dtype
            skip = _unsupported(cfg, (M, N, K), tile_cfg, kwargs)
            if skip is not None:
                rows.append([mode, str(M), str(N), str(K), config, out_dtype, str(apre), "-", "-", "-", "-", "SKIP"])
                print(f"\n{mode} {M}x{N}x{K} apre={apre}: skipped, {skip}\n")
                continue
            for init in args.init_mode:
                kwargs["const_val"] = _parse_init_mode(init)
                run = _execute(cfg, (M, N, K), tile_cfg, **kwargs)
                error = None
                try:
                    run.check()
                except AssertionError as exc:
                    error = exc
                perf = ["-", "-", "-"]
                if args.bench:
                    us = _bench_us(run.launch, run.output, warmup=10, iters=100)
                    moved = _bytes_moved(cfg, M, N, K)
                    perf = [f"{us:.3f}", f"{2.0 * M * N * K / us / 1e6:.2f}", f"{moved / us / 1e6:.3f}"]
                result = "PASS" if error is None else "FAIL"
                rows.append([mode, str(M), str(N), str(K), config, out_dtype, str(apre), init, *perf, result])
                if error is not None:
                    print(f"\n{mode} {M}x{N}x{K} apre={apre} {init}: {error}\n")

    print()
    headers = ["mode", "M", "N", "K", "config", "out", "apre", "init_mode", "latency us", "TFLOPS", "BW TB/s", "result"]
    _print_table(headers, rows)
    if any(r[-1] == "FAIL" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
