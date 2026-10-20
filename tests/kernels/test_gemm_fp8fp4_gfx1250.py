#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 FP8/FP4 GEMM correctness, layout and pipeline regressions.

Run the representative correctness cases with pytest, or use the CLI for
individual configurations and throughput measurements.
"""

import copy
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
from kernels.gemm.gemm_common_gfx1250 import B_SCALE_BLOCKS  # noqa: E402
from tests.kernels.utils import gemm_common_utils as utils  # noqa: E402
from tests.test_common import run_perftest  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
_DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


@dataclass(frozen=True)
class _Mode:
    launch: object
    tiles: tuple
    shape: tuple[int, int, int]
    b_scale: str = "1x32"  # "ptpc", or the kernels' b_scale_block: "1x32" / "32x32" / "128x128"
    fp4_act: bool = False
    fp4_w: bool = False
    supports_out_dtype: bool = True

    @property
    def ptpc(self):
        return self.b_scale == "ptpc"

    @property
    def b_block(self):
        """(N rows, K) of one weight-scale block; the A scale is one per token per K block."""
        return B_SCALE_BLOCKS[self.b_scale]

    @property
    def large_tile(self):
        return self.launch in (launch_gemm_a8w4_256x256, launch_gemm_a4w4_256x256, launch_gemm_a8w8_256x256)

    @property
    def compute(self):
        return self.launch in (launch_gemm_a4w4_256x256, launch_gemm_a8w8_256x256)

    @property
    def cluster(self):
        return (4, 4) if self.large_tile else (1, 1)


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
_DECODE_TILES = ((16, 256, 256, 1, 4, 4), (64, 256, 256, 2, 2, 3))
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
        b_scale="ptpc",
    ),
    "blockscale_a8w8": _Mode(
        launch_gemm_a8w8, _MX_TILES, (1024, 512, 512), b_scale="128x128", supports_out_dtype=False
    ),
    "mxscale_a8w8": _Mode(launch_gemm_a8w8, _MX_TILES, (1024, 512, 512), supports_out_dtype=False),
    "mxscale_a8w4_256x256": _Mode(launch_gemm_a8w4_256x256, _A8W4_TILES, (1024, 1024, 512), fp4_w=True),
    "mxscale_a4w4_256x256": _Mode(launch_gemm_a4w4_256x256, _A4W4_TILES, (1024, 1024, 1024), fp4_act=True, fp4_w=True),
    "mxscale_a8w8_256x256": _Mode(launch_gemm_a8w8_256x256, _A8W8_TILES, (1024, 1024, 1024)),
    "b32x32_a8w8": _Mode(
        launch_gemm_a8w8, _MX_TILES + _DECODE_TILES, (1024, 512, 512), b_scale="32x32", supports_out_dtype=False
    ),
    "b32x32_a8w8_256x256": _Mode(launch_gemm_a8w8_256x256, _A8W8_TILES, (1024, 1024, 1024), b_scale="32x32"),
    "blockscale_a8w8_256x256": _Mode(launch_gemm_a8w8_256x256, _A8W8_TILES, (1024, 1024, 1024), b_scale="128x128"),
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
    scale = torch.nn.functional.pad(scale, (0, (-cols) % 4, 0, blocks * 32 - rows), value=127)
    return scale.view(blocks, 32, -1, 4).permute(0, 2, 1, 3).contiguous().view(blocks, -1)


def _scales_and_reference(cfg, a, b, out_dtype, const_val, scale_scale, a_scale_preshuffled=True):
    M, K = a.shape
    N = b.shape[0]
    if cfg.ptpc:

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

    b_rows, block = cfg.b_block
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

    sa, sb = scale(M), scale(N // b_rows)
    a_scale = utils.e8m0_to_f32(sa).repeat_interleave(block, dim=1)[:, :K]
    b_scale = utils.e8m0_to_f32(sb).repeat_interleave(block, dim=1)[:, :K]
    if b_rows > 1:
        b_scale = b_scale.repeat_interleave(b_rows, dim=0)
    ref = (a * a_scale) @ (b * b_scale).T
    tol = (1e-2, 5e-2)
    if cfg.launch is launch_gemm_a4w4:
        ref, tol = ref.to(_DT[out_dtype]).float(), (0, 0)
    if cfg.fp4_w and out_dtype != "f16" and cfg.launch is not launch_gemm_a4w4:
        peak_exp = max(0, int(sa.max()) - 127) + max(0, int(sb.max()) - 127)
        tol = (min(5e-2, 1e-2 + 3e-3 * peak_exp), max(5e-2, K * (0.6 + 1.5 * peak_exp)))
    if block == 32:
        if a_scale_preshuffled:
            sa = _preshuffle_scale(sa)
        if b_rows == 32:  # the compact [N/32, K/32] array goes in as it is
            return sa, sb.view(torch.uint8).contiguous(), ref, tol, 0
        return sa, _preshuffle_scale(sb), ref, tol, 0
    return sa.T.contiguous(), sb, ref, tol, M


@dataclass
class _Run:
    output: torch.Tensor
    reference: torch.Tensor
    tolerance: tuple[float, float]
    launch: Callable[[], None]
    buffers: tuple  # Keep tensor allocations alive while launch arguments hold raw pointers.
    bind: Callable | None = None

    def check_output(self, output):
        torch.cuda.synchronize()
        M, N = self.reference.shape
        actual = output[:M, :N]
        assert torch.isfinite(self.reference).all(), "reference overflowed; adjust the test inputs"
        torch.testing.assert_close(actual.float(), self.reference, rtol=self.tolerance[0], atol=self.tolerance[1])
        assert torch.isnan(output[M:]).all(), f"store wrote past M={M}"
        assert torch.isnan(output[:M, N:]).all(), f"store wrote into ldc padding past N={N}"

    def check(self, repeats=0):
        self.check_output(self.output)
        if repeats:
            M, N = self.reference.shape
            actual = self.output[:M, :N]
            golden = actual.clone()
            for _ in range(repeats):
                self.launch()
            torch.cuda.synchronize()
            assert torch.equal(actual, golden), "back-to-back output drifted across launches"
            self.check_output(self.output)


def _a8w8_bounds(shape, tile, cluster_n, split_k, persistent_n_tiles):
    _, N, K = shape
    _, tn, tk, _, _, nb = tile
    return K % (tk * a8w8_kpair(tn, nb) * split_k) != 0, N % (tn * cluster_n * persistent_n_tiles) != 0


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
    a_scale_preshuffled=None,
    signed=False,
    seed=0,
):
    """Prepare operands and reference, then compile GEMM and its optional FP32 reducer once."""
    reason = _unsupported(cfg, shape, tile, locals())
    if reason is not None:
        raise ValueError(reason)
    M, N, K = shape
    tm, tn, tk, mw, nw, nb = tile
    general_a4w4 = cfg.launch is launch_gemm_a4w4
    if general_a4w4:
        fused_splitk = split_k > 1
    torch.manual_seed(seed)
    packed_k = max(32, (K + 31) // 32 * 32) if cfg.launch is launch_gemm_a4w4 else K
    a, a_f32 = _operand(M, packed_k, cfg.fp4_act, const_val, out_dtype, signed)
    b, b_f32 = _operand(N, packed_k, cfg.fp4_w, const_val, out_dtype, signed)
    if cfg.launch is launch_gemm_a4w4:
        for operand in (a, b):
            operand[:, (K + 1) // 2 :] = 0
            if K % 2:
                operand[:, K // 2] &= 0x0F
        a_f32, b_f32 = a_f32[:, :K], b_f32[:, :K]
    sa, sb, ref, tol, stride_sa = _scales_and_reference(
        cfg, a_f32, b_f32, out_dtype, const_val, scale_scale, a_scale_preshuffled is not False
    )
    if split_k > 1 and cfg.launch is launch_gemm_a8w4:
        tol = (tol[0], tol[1] + split_k * float(ref.abs().max()) * 2**-8)
    lda, ldc = packed_k + lda_extra, N + ldc_extra
    a = torch.nn.functional.pad(a, (0, lda_extra // (2 if cfg.fp4_act else 1)))
    if a_preshuffle:
        a = utils.preshuffle_a_2x128(a)
    if b_preshuffle or cfg.launch not in (launch_gemm_a8w4, launch_gemm_a4w4, launch_gemm_a8w8):
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
            dtype=torch.float32 if cfg.large_tile or cfg.launch in (launch_gemm_a4w4, launch_gemm_a8w8) else out.dtype,
            device="cuda",
        )
    else:
        partials = out
    ptr = functools.partial(_pointer, unsigned=cfg.ptpc)
    tensor_args = cfg.launch is launch_gemm_a8w4
    bounded_m = (M % tm != 0 or m_tiles % cluster_m != 0) if bounded_m is None else bounded_m
    if general_a4w4 or tensor_args:
        tail = (b_preshuffle,)
    elif cfg.launch is launch_gemm_a8w8:
        tail = (a_preshuffle, b_preshuffle, a_scale_preshuffled is not False, K % (tk * split_k) != 0)
    else:
        tail = (a_preshuffle,)
    if cfg.compute:
        tail += (() if cfg.fp4_act else (persistent_n_tiles,)) + (fused_splitk, bounded_m)
    if cfg.launch is launch_gemm_a8w8_256x256:
        k_tail, bounded_n = _a8w8_bounds(shape, tile, cluster_n, split_k, persistent_n_tiles)
        tail += (a_scale_preshuffled is not False, k_tail, bounded_n)

    def make_args(out, a, b, sa, sb, partials):
        buffers = (partials, ptr(a), ptr(b), sa, sb) if tensor_args else tuple(ptr(t) for t in (partials, a, b, sa, sb))
        return (
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
            *((not cfg.ptpc, "1x32" if cfg.ptpc else cfg.b_scale) if not cfg.fp4_w else ()),
            split_k,
            *tail,
        )

    args = make_args(out, a, b, sa, sb, partials)
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

    def bind(out, a, b, sa, sb, partials):
        args = make_args(out, a, b, sa, sb, partials)
        bound_reduce_args = (_pointer(partials, True), _pointer(out, True), *reduce_args[2:]) if reduce else ()

        def launch():
            compiled(*args)
            if reduce is not None:
                reduce(*bound_reduce_args)

        return launch

    return _Run(out, ref, tol, bind(out, a, b, sa, sb, partials), (a, b, sa, sb, partials), bind)


def _unsupported(cfg, shape, tile, options):
    """Return the first unsupported configuration, shared by pytest and the CLI."""
    M, N, K = shape
    tm, tn, tk, _, _, nb = tile
    cm, cn = options.get("cluster_m", 1), options.get("cluster_n", 1)
    sk, ps = options.get("split_k", 1), options.get("persistent_n_tiles", 1)
    general_a4w4 = cfg.launch is launch_gemm_a4w4
    compute_a8w8 = cfg.launch is launch_gemm_a8w8_256x256
    general_a8w8 = cfg.launch is launch_gemm_a8w8
    a8w8 = compute_a8w8 or general_a8w8
    a_scale_preshuffled = options.get("a_scale_preshuffled")
    b_rows, k_block = (1, 1) if cfg.ptpc else cfg.b_block  # ptpc has no scale block
    if min(M, N, *tile, cm, cn, sk, ps) <= 0 or (K < 0 if general_a4w4 else K <= 0):
        return "shape, tiles, cluster and split counts must be positive"
    k_tail, bounded_n = _a8w8_bounds(shape, tile, cn, sk, ps) if compute_a8w8 else (False, False)
    if general_a8w8:
        k_tail = K % (tk * sk) != 0
    step, minimum = (32, 32) if cfg.launch is launch_gemm_a4w4 else (tk, tk * nb)
    if general_a8w8 and k_tail:
        minimum = (nb - 1) * tk + 1
    if cfg.large_tile:
        if cfg.launch is launch_gemm_a8w4_256x256:
            step = tk * KPAIR_BY_BUFFERS.get(nb, 1)
            minimum = 512 if (tm, tn) == (256, 256) else step
        elif cfg.launch is launch_gemm_a8w8_256x256:
            step, minimum = tk * a8w8_kpair(tn, nb), 512
        else:
            minimum = tk
    ck = sk if general_a4w4 or options.get("fused_splitk") else 1
    packed_k = max(32, (K + 31) // 32 * 32) if general_a4w4 else K
    lda_bytes = (packed_k + options.get("lda_extra", 0)) // (2 if cfg.fp4_act else 1)
    checks = [
        (
            cfg.launch is launch_gemm_a8w8 and not options.get("b_preshuffle", True) and tk < 256,
            "row-major FP8 B requires tile_k >= 256",
        ),
        (
            not (cm < 16 and cn < 16 and ck < 16 and cm * cn * ck <= 16),
            "cluster dimensions must be below 16 with at most 16 workgroups",
        ),
        (((M + tm - 1) // tm) % cm != 0, "M grid must not pad clusters with completely empty tiles"),
        (
            ((N + tn - 1) // tn) % (cn * ps) != 0,
            "N grid must not pad clusters/persistent tiles with completely empty tiles",
        ),
        (cfg.large_tile and tile not in cfg.tiles, f"unsupported tile profile {tile}"),
        (
            cfg.large_tile and not compute_a8w8 and cm * cn == 1,
            "these large-tile kernels require a cluster of 2..16 workgroups",
        ),
        (compute_a8w8 and N % 16 != 0, "preshuffled FP8 B requires N divisible by 16"),
        (
            (k_tail or bounded_n) and sk != 1,
            "A8W8 tails require split_k=1",
        ),
        (
            general_a8w8
            and k_tail
            and (cfg.b_scale != "32x32" or options.get("a_preshuffle") or a_scale_preshuffled is not False),
            "general A8W8 K tails require a 32x32 B scale and row-major A/scales",
        ),
        (
            a_scale_preshuffled is not None and not a8w8,
            "A-scale layout selection requires an A8W8 kernel",
        ),
        (
            a8w8 and k_block == 32 and K % 128 != 0 and (a_scale_preshuffled is not False or b_rows != 32),
            "preshuffled MX32 scales require K divisible by 128",
        ),
        (
            a8w8 and k_block != 32 and a_scale_preshuffled,
            "A-scale preshuffle requires MX32",
        ),
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
            cfg.launch is not launch_gemm_a4w4 and not bounded_n and N % (tn * cn) != 0,
            f"N={N} must cover whole clusters of {tn}-wide tiles",
        ),
        (
            not general_a4w4 and ((not k_tail and K % (step * sk) != 0) or K // sk < minimum),
            f"K={K} needs multiples of {step} and at least {minimum} per split",
        ),
        (
            not general_a4w4 and not cfg.ptpc and K % k_block != 0,
            f"K must be divisible by the scale block size {k_block}",
        ),
        (
            N % b_rows != 0,
            f"N must be divisible by the weight-scale block rows {b_rows}",
        ),
        (
            options.get("a_preshuffle")
            and (
                cfg.launch in (launch_gemm_a8w4, launch_gemm_a4w4)
                or tm % 2
                or lda_bytes % 128
                or (compute_a8w8 and K % 128)
                or (cfg.fp4_act and tk != 256)
            ),
            "A preshuffle needs a supported launcher, even tile_m, whole 128-byte rows and FP4 tile_k=256",
        ),
        (
            ps > 1 and (not compute_a8w8 or sk != 1 or (not bounded_n and (N // tn) % (ps * cn))),
            "persistent N tiles need the A8W8 compute kernel, split_k=1 and whole clusters",
        ),
        (
            options.get("fused_splitk")
            and (not (cfg.compute or general_a4w4) or sk not in (2, 4, 8) or tm % sk or cm * cn * sk > 16),
            "fused split-K needs 2/4/8 splits dividing tile_m in one cluster of at most 16 workgroups",
        ),
    ]
    return next((message for bad, message in checks if bad), None)


# Each format gets one representative case; special cases cover distinct paths.
def _case(mode, label, shape, tile=None, **options):
    cfg = _MODES[mode]
    tile = tile or cfg.tiles[0]
    options = dict(cluster_m=1, cluster_n=2, lda_extra=128, ldc_extra=32) | options
    options.setdefault("c_guard_rows", tile[0] * options["cluster_m"])
    reason = _unsupported(cfg, shape, tile, options)
    assert reason is None, f"invalid test case {mode}-{label}: {reason}"
    return pytest.param(mode, shape, tile, options, id=f"{mode}-{label}")


def _cases():
    for mode, cfg in _MODES.items():
        tile = min(cfg.tiles, key=lambda t: t[0] * t[1] * t[2]) if cfg.large_tile else cfg.tiles[0]
        tm, tn, tk, _, _, nb = tile
        shape = (tm + 1, tn * 2, max(cfg.shape[2], tk * nb))
        options = dict(signed=not cfg.fp4_act)
        if not cfg.ptpc and cfg.launch in (launch_gemm_a8w8, launch_gemm_a8w8_256x256) and cfg.b_block[1] == 32:
            options["a_scale_preshuffled"] = False
        if mode == "a4w4":
            shape = (33, 137, 800)
            options["split_k"] = 2
        elif mode == "ptpc_a8w8":
            options.update(out_dtype="f16", const_val=0.5, scale_scale=0.02, signed=False)
        elif mode in ("mxscale_a8w8", "blockscale_a8w8", "blockscale_a8w8_256x256"):
            options["a_preshuffle"] = True
        elif mode in ("mxscale_a8w4_256x256", "mxscale_a4w4_256x256", "mxscale_a8w8_256x256"):
            shape = (tm + 1, tn * 2, shape[2] * 2)
            options.update(split_k=2, fused_splitk=cfg.compute)
        elif mode == "b32x32_a8w8":
            shape, tile = (65, 256, 544), (64, 128, 256, 2, 2, 3)
            options["b_preshuffle"] = False
        elif mode == "b32x32_a8w8_256x256":
            shape = (tm + 1, tn * 4 - 32, 640)
            options.update(persistent_n_tiles=2, out_dtype="f16")
        yield _case(mode, "ragged-strided", shape, tile, **options)

    yield _case("a4w4", "zero-k", (33, 137, 0))
    yield _case("mxscale_a8w4", "split2-row-major-b", (129, 512, 1024), split_k=2, b_preshuffle=False)
    for mode in ("mxscale_a8w4_256x256", "mxscale_a4w4_256x256", "mxscale_a8w8_256x256"):
        yield _case(mode, "256x256-drain", (257, 512, 1536), bounded_m=True)


@pytest.fixture
def gemm(mode, shape, tile, options):
    return _execute(_MODES[mode], shape, tile, **options)


@pytest.mark.parametrize("mode,shape,tile,options", tuple(_cases()))
def test_gemm(gemm):
    gemm.check(repeats=3)


@pytest.mark.parametrize("mode,split_k", (("mxscale_a8w8", 1), ("mxscale_a8w4", 2), ("b32x32_a8w8_256x256", 2)))
def test_rebind_gemm_buffers(mode, split_k):
    """Rotated inputs, scales, output and split-K scratch must all use the new allocations."""
    cfg = _MODES[mode]
    tile = _A8W8_TILES[2] if cfg.large_tile else cfg.tiles[0]
    shape = (tile[0] + 1, tile[1] * 2, cfg.shape[2] * split_k)
    options = dict(
        cluster_n=2,
        split_k=split_k,
        c_guard_rows=tile[0],
        lda_extra=128,
        ldc_extra=32,
        a_scale_preshuffled=False if cfg.launch is not launch_gemm_a8w4 else None,
        a_preshuffle=mode == "mxscale_a8w8",
        b_preshuffle=mode != "mxscale_a8w4",
    )
    assert _unsupported(cfg, shape, tile, options) is None
    run = _execute(cfg, shape, tile, **options)
    run.check()
    original = (run.output, *run.buffers)
    tensors = copy.deepcopy(original)
    assert all(old.data_ptr() != new.data_ptr() for old, new in zip(original, tensors))
    assert (tensors[0] is tensors[-1]) == (split_k == 1)
    for tensor in original:
        tensor.fill_(float("nan") if tensor.is_floating_point() else 0)
    tensors[0].fill_(float("nan"))
    tensors[-1].fill_(float("nan"))
    launch = run.bind(*tensors)
    for _ in range(3):
        launch()
    run.check_output(tensors[0])
    assert torch.isnan(run.output).all(), "rotated launch wrote to the original output"
    assert torch.isnan(run.buffers[-1]).all(), "rotated launch wrote to the original scratch"


def test_benchmark_rejects_stale_output(monkeypatch):
    """A skipped launch must fail even if C was copied from a correct result."""

    def measure_once(launch, *tensors):
        launch(*tensors)  # The default helper profiles memory before copying C.
        tensors = copy.deepcopy(tensors)
        launch(*tensors)
        return None, 0.0

    monkeypatch.setattr(sys.modules[__name__], "run_perftest", measure_once)
    case = next(case for case in _cases() if case.values[0] == "mxscale_a8w8")
    mode, shape, tile, options = case.values
    run = _execute(_MODES[mode], shape, tile, **options)
    run.check()
    run.bind = lambda output, *buffers: run.launch if output is run.output else lambda: None
    with pytest.raises(AssertionError):
        _bench_us(run)


def test_a8w8_runtime_shape(monkeypatch):
    """Runtime shapes and scale reallocations reuse one compiled launcher."""
    compile_original = flyc.compile
    compiled = cache_key = None

    def compile_once(launcher, *args):
        nonlocal compiled, cache_key
        launcher._ensure_sig()
        bound = launcher._sig.bind(*args)
        bound.apply_defaults()
        key = launcher._resolve_and_make_cache_key(bound.arguments)
        if compiled is None:
            cache_key = key
            compiled = compile_original(launcher, *args)
        else:
            assert key == cache_key, "changing runtime dimensions created a new JIT specialization"
            compiled(*args)
        return compiled

    def integer_operand(rows, K, fp4, const_val, out_dtype, signed=False):
        # Exact long-K sums isolate bounds/pipeline bugs from FP32 cancellation rounding.
        values = torch.randint(-2, 3, (rows, K), device="cuda").float()
        return values.to(torch.float8_e4m3fn).view(torch.uint8), values

    monkeypatch.setattr(flyc, "compile", compile_once)
    monkeypatch.setattr(sys.modules[__name__], "_operand", integer_operand)
    cfg, tile = _MODES["b32x32_a8w8"], _MX_TILES[0]
    # Cross the compact-scale preload limit in both directions.
    for shape in ((65, 512, 768), (129, 1024, 4352), (65, 512, 65536), (513, 1024, 65792), (65, 512, 768)):
        _execute(cfg, shape, tile, c_guard_rows=tile[0], ldc_extra=32, signed=True).check(repeats=3)


@pytest.mark.parametrize("b_scale,preload", (("1x32", False), ("32x32", False), ("128x128", False), ("128x128", True)))
def test_a8w8_general_batched(b_scale, preload):
    M, N, K, batch = 65, 512, 512, 3
    block = B_SCALE_BLOCKS[b_scale][1]
    a = torch.randint(-2, 3, (M, batch, K), device="cuda").float()
    b = torch.randint(-2, 3, (batch, N, K), device="cuda").float()
    sa = utils.random_e8m0(M * batch, K // block, low_exp=126, high_exp=129, device="cuda").view(M, batch, -1)
    sb_rows = N // B_SCALE_BLOCKS[b_scale][0]
    sb = utils.random_e8m0(batch * sb_rows, K // block, low_exp=126, high_exp=129, device="cuda").view(
        batch, sb_rows, -1
    )
    scaled_a = a * utils.e8m0_to_f32(sa).repeat_interleave(block, dim=-1)
    scaled_b = b * utils.e8m0_to_f32(sb).repeat_interleave(block, dim=-1).repeat_interleave(N // sb_rows, dim=1)
    ref = torch.einsum("mbk,bnk->mbn", scaled_a, scaled_b)
    a = a.to(torch.float8_e4m3fn).view(torch.uint8)
    b = torch.stack([utils.preshuffle_b_16x16(x, N, K) for x in b.to(torch.float8_e4m3fn).view(torch.uint8)])
    if b_scale == "1x32":
        sb = torch.stack([_preshuffle_scale(x) for x in sb])
    out = torch.full((M + 128, batch, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    args = (
        *map(_pointer, (out, a, b, sa, sb)),
        M,
        torch.cuda.current_stream(),
        N,
        K,
        batch * (K // block),
        batch * K,
        batch * N,
    )
    args += (128, 256, 128, 2, 2, 0, 2, 1, 1, True, b_scale)
    compiled = flyc.compile(
        launch_gemm_a8w8,
        *args,
        batched=True,
        a_scale_preshuffled=False,
        preload_ks=K // block if preload else 0,
        batch=batch,
    )
    for _ in range(8):
        compiled(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:M].float(), ref, rtol=1e-2, atol=5e-2)
    assert torch.isnan(out[M:]).all()


@pytest.mark.parametrize("shape", ((65, 288, 2048), (65, 1024, 1280)))
def test_a8w8_tails_reject_split_k(shape):
    cfg, tile = _MODES["b32x32_a8w8_256x256"], _A8W8_TILES[1]
    assert _unsupported(cfg, shape, tile, {"split_k": 2}) == "A8W8 tails require split_k=1"
    with pytest.raises(ValueError, match="A8W8 tails require split_k=1"):
        _execute(cfg, shape, tile, split_k=2)


def test_a8w8_row_major_rejects_small_k():
    cfg = _MODES["mxscale_a8w8"]
    shape, tile = (64, 256, 512), (64, 128, 128, 2, 2, 2)
    assert _unsupported(cfg, shape, tile, {"b_preshuffle": False}) == "row-major FP8 B requires tile_k >= 256"
    with pytest.raises(ValueError, match="row-major FP8 B requires tile_k >= 256"):
        _execute(cfg, shape, tile, b_preshuffle=False)


def test_a4w4_k_is_runtime_in_jit_cache():
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

    assert all(key(K) == key(1536) for K in (256, 512, 768, 16384))


def _bench_us(run):
    """Use the shared timing defaults and validate every buffer set outside timing."""
    tensors = (run.output, *run.buffers)
    launches = {run.output.data_ptr(): (run.launch, tensors)}
    run.output.fill_(float("nan"))

    def launch(*tensors):
        key = tensors[0].data_ptr()
        if key not in launches:
            launches[key] = (run.bind(*tensors), tensors)
        launches[key][0]()

    _, us = run_perftest(launch, *tensors)
    for bound_launch, buffers in launches.values():
        run.check_output(buffers[0])
        # A copied C must not hide a skipped launch.
        buffers[0].fill_(float("nan"))
        bound_launch()
        run.check_output(buffers[0])
    return us


def _bytes_moved(cfg, M, N, K):
    a_bytes = M * K // (2 if cfg.fp4_act else 1)
    b_bytes = N * K // (2 if cfg.fp4_w else 1)
    scale_bytes = (M + N) * 4 if cfg.ptpc else (M + N // cfg.b_block[0]) * (K // cfg.b_block[1])
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
    parser.add_argument(
        "-mode", nargs="+", choices=sorted((*_MODES, "a8w8")), required=True, help="a8w8 aliases mxscale_a8w8"
    )
    parser.add_argument("-mnk", nargs="+", required=True, metavar="M,N,K", help="one or more shapes")
    parser.add_argument("-tiles", nargs="+", required=True, help="tile_m,tile_n,tile_k")
    parser.add_argument("-warps", nargs="+", required=True, help="m_warp,n_warp")
    parser.add_argument("-nb", "-n", nargs="+", type=int, required=True, help="num_buffers")
    parser.add_argument("-cluster", nargs="+", help="cluster_m,cluster_n; default: each mode's own")
    parser.add_argument("-out-dtype", default="bf16", choices=["bf16", "f16"])
    parser.add_argument(
        "-bench",
        action="store_true",
        help="measure perf via run_perftest (3 warmup launches, 20 timed launches)",
    )
    parser.add_argument("-sk", "--split-k", nargs="+", type=int, default=[1], help="K splits per run (default 1)")
    parser.add_argument("--fused-splitk", action="store_true", help="reduce K splits within each hardware cluster")
    parser.add_argument("--a-scale-row-major", action="store_true", help="use raw A scales in either A8W8 kernel")
    parser.add_argument(
        "-no-b-preshuffle",
        dest="b_preshuffle",
        action="store_false",
        help="disable B preshuffle where row-major B is supported (default: enabled)",
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
    args.mode = ["mxscale_a8w8" if mode == "a8w8" else mode for mode in args.mode]

    shapes = [_parse_csv_ints(v, 3, "mnk") for v in args.mnk]
    runs = [(mode, mnk) for mode in args.mode for mnk in shapes]
    spread = functools.partial(_per_run, n_modes=len(args.mode), n_shapes=len(shapes))
    tiles = spread([_parse_csv_ints(v, 3, "tiles") for v in args.tiles], name="tiles")
    warps = spread([_parse_csv_ints(v, 2, "warps") for v in args.warps], name="warps")
    tile_cfgs = [(*t, *w, nb) for t, w, nb in zip(tiles, warps, spread(args.nb, name="nb"))]
    ps_tiles = spread(args.ps, name="ps")
    split_ks = spread(args.split_k, name="sk")
    # Default to each mode's usual cluster configuration.
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
        if args.a_scale_row_major:
            config += " ascale=row-major"
        if split_k > 1:
            config += f" sk{split_k}" + (" fused" if args.fused_splitk else "")
        if cfg.launch is launch_gemm_a8w8 or mode in ("mxscale_a8w4", "a4w4"):
            config += f" b={'preshuffle' if args.b_preshuffle else 'row-major+pad'}"
        for apre in args.apre:
            kwargs = {
                "cluster_m": cluster_m,
                "cluster_n": cluster_n,
                "a_preshuffle": bool(apre),
                "persistent_n_tiles": ps,
                "split_k": split_k,
                "fused_splitk": args.fused_splitk,
                "a_scale_preshuffled": False if args.a_scale_row_major else None,
            }
            if cfg.launch is launch_gemm_a8w8 or mode in ("mxscale_a8w4", "a4w4"):
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
                if args.bench and error is None:
                    us = _bench_us(run)
                    moved = _bytes_moved(cfg, M, N, K)
                    perf = [f"{us:.3f}", f"{2.0 * M * N * K / us / 1e6:.2f}", f"{moved / us / 1e6:.3f}"]
                    try:
                        run.check()
                    except AssertionError as exc:
                        error = exc
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
