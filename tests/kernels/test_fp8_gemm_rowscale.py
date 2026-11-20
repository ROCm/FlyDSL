#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 row-scale GEMM correctness + perf harness.

Kernel implementations:
- 4 Wave variant --> ``kernels/fp8_gemm_4wave.py``.
- 8 Wave variant --> ``kernels/fp8_gemm_8wave.py``.
"""

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.fp8_gemm_4wave import compile_fp8_gemm_4w  # noqa: E402
from kernels.gemm.fp8_gemm_8wave import compile_fp8_gemm_8w  # noqa: E402
from kernels.gemm.fp8_gemm_utils import preshuffle_b  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402
from tests.utils import pertoken_quant  # noqa: E402

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
    b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    return torch.mm(a_f32, b_f32.T).to(dtype)


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _signed_inputs(M, N, K, device, seed=0):
    """Zero-mean inputs whose rows of A and of B_T differ in magnitude by up to 64x."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    row_mag = torch.exp2(torch.rand(M, 1, generator=g, device=device) * 6 - 3)
    col_mag = torch.exp2(torch.rand(N, 1, generator=g, device=device) * 6 - 3)
    a = torch.randn(M, K, generator=g, device=device) * row_mag
    b = torch.randn(N, K, generator=g, device=device) * col_mag
    return a, b


def _assert_close_to_fp64_product(c_out, a_q, b_q, scale_a, scale_b):
    """Check every output against an fp64 product of the same FP8 operands and scales.

    The reference uses the kernel's own e4m3 values and scales, so FP8 rounding cancels. What is
    left is one bf16 rounding of the output (relative 2**-8) and fp32 accumulation, which stays
    below 2**-16 of sum |a * b| for K <= 16384. The tolerance doubles the first and adds the second.
    """
    a = a_q.to(torch.float32).to(torch.float64) * scale_a.view(-1, 1).to(torch.float64)
    b = b_q.to(torch.float32).to(torch.float64) * scale_b.view(-1, 1).to(torch.float64)
    ref = a @ b.T
    tol = 2.0**-7 * ref.abs() + 2.0**-16 * (a.abs() @ b.abs().T)
    err = (c_out.to(torch.float64) - ref).abs()
    bad = ~(err <= tol)
    worst = float((err / tol).max())
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} outputs out of tolerance, worst err/tol {worst:.3g}"


def _bench_torch_scaled_mm(M, N, K, a_q, b_q, scale_a, scale_b, c_ref, num_warmups, num_iters):
    """Run torch._scaled_mm with the same inputs / perf harness as flydsl."""
    b_t = b_q.t()
    sa_v = scale_a.view(M, 1).to(torch.float32).contiguous()
    sb_v = scale_b.view(1, N).to(torch.float32).contiguous()
    out = torch.empty((M, N), dtype=OUT_DTYPE, device="cuda")

    def _launch(_out):
        torch._scaled_mm(a_q, b_t, scale_a=sa_v, scale_b=sb_v, out_dtype=OUT_DTYPE, out=_out)

    _, us = run_perftest(_launch, out, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    ok = verify_output(out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    return us, ok


def _bench_fp8_gemm(
    M: int,
    N: int,
    K: int,
    *,
    use_8w: bool,
    tile_m: int,
    tile_n: int,
    disable_xcd_remap: bool = False,
    num_warmups: int = 2,
    num_iters: int = 10,
    vs_torch: bool = False,
    b_preshuffled: bool = False,
    static_weight_scale: bool = True,
    signed_inputs: bool = False,
):
    """Run + verify a single (M, N, K, tile) configuration. Returns TFLOPS.

    ``signed_inputs`` uses ``_signed_inputs`` and checks every output with
    ``_assert_close_to_fp64_product``; unwritten outputs stay NaN and fail it.
    """
    if "gfx95" not in ARCH:
        pytest.skip("FP8 row-scale GEMMs requires CDNA4 (gfx95*)")

    device = torch.device("cuda")
    if signed_inputs:
        a_fp32, b_fp32_t = _signed_inputs(M, N, K, device)
        c_out_raw = torch.full((M, N), float("nan"), dtype=OUT_DTYPE, device=device)
    else:
        a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
        b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
        c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    c_ref = _run_torch(a_q, b_q, scale_a, scale_b)

    b_kernel = preshuffle_b(b_q) if b_preshuffled else b_q

    if use_8w:
        launch_fn = compile_fp8_gemm_8w(
            K=K,
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            b_preshuffled=b_preshuffled,
        )
        print(
            f"\n[fp8_gemm_8wave] M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n} "
            f"preshuffle_b={b_preshuffled} static_weight_scale={static_weight_scale}"
        )
    else:
        launch_fn = compile_fp8_gemm_4w(
            K=K,
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            use_xcd_remap=not disable_xcd_remap,
            b_preshuffled=b_preshuffled,
        )
        print(
            f"\n[fp8_gemm_4wave] M={M} N={N} K={K} "
            f"BLOCK_M={tile_m} BLOCK_N={tile_n} xcd_remap={not disable_xcd_remap} "
            f"preshuffle_b={b_preshuffled} static_weight_scale={static_weight_scale}"
        )

    def _args(c, a, b, sa, sb):
        b_flat = _as_i8(b).contiguous().view(-1)
        sa_flat = sa.contiguous().view(-1)
        sb_flat = sb.contiguous().view(-1)
        if static_weight_scale:
            b_flat = flyc.from_torch_tensor(b_flat)
            sa_flat = flyc.from_torch_tensor(sa_flat)
            sb_flat = flyc.from_torch_tensor(sb_flat)
        return (
            _as_i8(a).contiguous().view(-1),
            b_flat,
            c.contiguous().view(-1),
            sa_flat,
            sb_flat,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw, a_q, b_kernel, scale_a, scale_b))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out_raw,
        a_q,
        b_kernel,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    if signed_inputs:
        _assert_close_to_fp64_product(c_out_raw, a_q, b_q, scale_a, scale_b)
    else:
        assert verify_output(c_out_raw.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    bytes_moved = M * K + N * K + M * N * 2 + (M + N) * 4
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    if vs_torch:
        try:
            us_t, ok_t = _bench_torch_scaled_mm(
                M,
                N,
                K,
                a_q,
                b_q,
                scale_a,
                scale_b,
                c_ref,
                num_warmups,
                num_iters,
            )
            tflops_t = flops / (us_t / 1e6) / 1e12
            print(
                f"[torch._scaled_mm] Throughput: {us_t:.1f} us, "
                f"{tflops_t:.2f} TFLOPS, flydsl/torch={us_t / us * 100:.1f}%, ok={ok_t}"
            )
        except Exception as e:
            print(f"[torch._scaled_mm] failed: {e}")

    return tflops


_FP8_GEMM_SHAPES = (
    (512, 2112, 7168),
    (5120, 5120, 8320),
    (8192, 8192, 8192),
    (9728, 8192, 8320),
    (16384, 16384, 16384),
)


def _fp8_gemm_case(use_8w, shape_index, preshuffle_b):
    m, n, k = _FP8_GEMM_SHAPES[shape_index]
    tile_m, tile_n = ((128, 256) if use_8w else (64, 64)) if shape_index == 0 else (256, 256)
    layout = "preshuffle" if preshuffle_b else "rowmajor"
    wave = "8w" if use_8w else "4w"
    marks = pytest.mark.large_shape if shape_index >= 2 else ()
    return pytest.param(
        use_8w,
        shape_index,
        m,
        n,
        k,
        tile_m,
        tile_n,
        preshuffle_b,
        marks=marks,
        id=f"{wave}-{m}x{n}x{k}-{layout}",
    )


# Pairwise covering array over wave count, shape, and B layout.  Each shape is
# still exercised by both kernels and both layouts, while avoiding the redundant
# three-factor Cartesian product (20 correctness launches -> 10).
_FP8_GEMM_CASES = tuple(
    _fp8_gemm_case(use_8w, shape_index, bool(shape_index % 2) ^ use_8w)
    for shape_index in range(len(_FP8_GEMM_SHAPES))
    for use_8w in (False, True)
)
_FP8_GEMM_CASE_VALUES = tuple(
    (use_8w, shape_index, bool(shape_index % 2) ^ use_8w)
    for shape_index in range(len(_FP8_GEMM_SHAPES))
    for use_8w in (False, True)
)


@pytest.mark.parametrize(
    "use_8w,shape_index,M,N,K,tile_m,tile_n,preshuffle_b",
    _FP8_GEMM_CASES,
)
def test_fp8_gemm_rowscale(use_8w, shape_index, M, N, K, tile_m, tile_n, preshuffle_b):
    _bench_fp8_gemm(
        M=M,
        N=N,
        K=K,
        use_8w=use_8w,
        tile_m=tile_m,
        tile_n=tile_n,
        b_preshuffled=preshuffle_b,
    )


# M <= SMALL_M_MAX sends the 8-wave GEMM to its skinny-M kernel: both row tiles (16 rows up to
# M = 16, 32 rows above) full and partial, a large N, the first M past the threshold, partial last N
# tiles (N % 32 = 8 and 24), a K shorter than the prefetch depth, and dynamic B and scale tensors.
_FP8_GEMM_8W_SMALL_M_CASES = (
    # (M, N, K, static_weight_scale)
    (1, 2112, 7168, True),
    (17, 7168, 2048, True),
    (32, 8192, 8192, True),
    (16, 40960, 5120, True),
    (33, 8192, 8192, True),
    (9, 2120, 2048, True),
    (24, 2136, 2048, True),
    (5, 4096, 256, True),
    (16, 4096, 2048, False),
)


@pytest.mark.parametrize(
    "M,N,K,static_weight_scale",
    _FP8_GEMM_8W_SMALL_M_CASES,
    ids=[f"{m}x{n}x{k}" + ("" if static else "-dynamic") for m, n, k, static in _FP8_GEMM_8W_SMALL_M_CASES],
)
def test_fp8_gemm_8w_small_m(M, N, K, static_weight_scale):
    _bench_fp8_gemm(
        M=M,
        N=N,
        K=K,
        use_8w=True,
        tile_m=256,
        tile_n=256,
        static_weight_scale=static_weight_scale,
        signed_inputs=True,
    )


def test_fp8_gemm_case_list_preserves_pairwise_coverage():
    def _projection(cases):
        return {
            (left_axis, left, right_axis, right)
            for case in cases
            for left_axis in range(len(case))
            for right_axis in range(left_axis + 1, len(case))
            for left, right in [(case[left_axis], case[right_axis])]
        }

    exhaustive = (
        (use_8w, shape_index, preshuffle_b)
        for use_8w in (False, True)
        for shape_index in range(len(_FP8_GEMM_SHAPES))
        for preshuffle_b in (False, True)
    )
    missing = _projection(exhaustive) - _projection(_FP8_GEMM_CASE_VALUES)
    assert not missing, f"FP8 row-scale matrix is missing pairwise interactions: {sorted(missing, key=repr)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP8 row-scale GEMM benchmark")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=256)
    parser.add_argument("--tile_n", type=int, default=256)
    parser.add_argument("--disable_xcd_remap", action="store_true", default=False)
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Benchmark iterations.",
    )
    parser.add_argument(
        "--num_warmups",
        type=int,
        default=10,
        help="Warmup iterations.",
    )
    parser.add_argument(
        "--vs_torch",
        action="store_true",
        default=False,
        help="Also run torch._scaled_mm with the same input + harness for perf comparison.",
    )
    parser.add_argument(
        "--preshuffle_b",
        action="store_true",
        default=False,
        help="Use preshuffled B layout.",
    )
    parser.add_argument(
        "--wave_8",
        action="store_true",
        default=False,
        help="Use 8-Wave Ping-Pong variant.",
    )
    parser.add_argument(
        "--dynamic_weight_scale",
        action="store_true",
        default=False,
        help="Use dynamic tensor arguments for weight and scales instead of static DLPack adaptors.",
    )
    args = parser.parse_args()

    torch.set_default_device("cuda")

    try:
        _bench_fp8_gemm(
            M=args.M,
            N=args.N,
            K=args.K,
            use_8w=args.wave_8,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            disable_xcd_remap=args.disable_xcd_remap,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
            vs_torch=args.vs_torch,
            b_preshuffled=args.preshuffle_b,
            static_weight_scale=not args.dynamic_weight_scale,
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
