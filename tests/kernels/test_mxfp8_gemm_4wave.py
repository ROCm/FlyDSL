#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""MXFP8 4-wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/gemm/mxfp8_gemm_4wave.py``.
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
from kernels.gemm.mxfp8_gemm_4wave import compile_mxfp8_gemm_4w  # noqa: E402
from tests.test_common import run_perftest  # noqa: E402

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.float16
MX_BLOCK_K = 32
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _make_mxfp8_inputs(M: int, N: int, K: int, device: torch.device):
    """Create MXFP8 inputs and raw per-K32 E8M0 scale bytes."""
    assert K % MX_BLOCK_K == 0

    a = (torch.randn(M, K, device=device) * 2).clamp(-6, 6).to(FP8_DTYPE)
    b = (torch.randn(N, K, device=device) * 2).clamp(-6, 6).to(FP8_DTYPE)
    a_scale_raw = torch.randint(
        125,
        130,
        (M, K // MX_BLOCK_K),
        dtype=torch.uint8,
        device=device,
    )
    b_scale_raw = torch.randint(
        125,
        130,
        (N, K // MX_BLOCK_K),
        dtype=torch.uint8,
        device=device,
    )
    return a, b, a_scale_raw, b_scale_raw


def _dequantize_mxfp8(q: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales_u8.float() - 127.0).repeat_interleave(
        MX_BLOCK_K,
        dim=1,
    )
    return q.float() * scale


def _pack_scale_words(scales_u8: torch.Tensor) -> torch.Tensor:
    """Pack raw ``[rows, K/32]`` E8M0 bytes as ``[K/128, rows]`` int32."""
    assert scales_u8.dtype == torch.uint8
    rows, k32_tiles = scales_u8.shape
    assert k32_tiles % 4 == 0
    assert rows % 64 == 0

    s32 = scales_u8.contiguous().view(rows, k32_tiles // 4, 4).to(torch.int32)
    iteration_major = (
        (s32[:, :, 0] | (s32[:, :, 1] << 8) | (s32[:, :, 2] << 16) | (s32[:, :, 3] << 24)).transpose(0, 1).contiguous()
    )

    row = torch.arange(rows, device=scales_u8.device, dtype=torch.int64)
    row_in_16 = row % 16
    k32_in_word = (row // 16) % 4
    tile_64 = row // 64

    packed = torch.zeros_like(iteration_major)
    for byte_index in range(4):
        source_row = tile_64 * 64 + byte_index * 16 + row_in_16
        source_word = iteration_major[:, source_row]
        source_byte = (source_word >> (k32_in_word * 8).view(1, rows)) & 0xFF
        packed |= source_byte << (byte_index * 8)

    return packed.contiguous()


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if "float8" in str(tensor.dtype) else tensor


def _verify_mxfp8_output(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Apply the HipKittens MXFP8 absolute-error criterion."""
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = (actual_f32 - expected_f32).abs()

    tolerance = expected_f32.abs().max() * 1.0e-3
    assert torch.isfinite(actual_f32).all(), "MXFP8 kernel produced non-finite output"
    assert torch.all(difference <= tolerance), (
        f"MXFP8 mismatch: max_abs={difference.max().item():.6g}, "
        f"mean_abs={difference.mean().item():.6g}, "
        f"atol={tolerance.item():.6g}"
    )


def _bench_mxfp8_gemm_4wave(
    M: int,
    N: int,
    K: int,
    *,
    tile_m: int = 256,
    tile_n: int = 256,
    disable_xcd_remap: bool = False,
    num_warmups: int = 2,
    num_iters: int = 10,
    static_weight_scale: bool = True,
):
    """Run + verify one MXFP8 GEMM configuration. Returns TFLOPS."""
    if "gfx95" not in ARCH:
        pytest.skip("MXFP8 GEMM requires CDNA4 (gfx95*)")

    if M % tile_m != 0 or N % tile_n != 0 or K % 128 != 0 or K < 512:
        pytest.skip("MXFP8 4-wave kernel requires M/N divisible by the selected tile and K >= 512 divisible by 128")

    device = torch.device("cuda")
    a, b, a_scale_raw, b_scale_raw = _make_mxfp8_inputs(M, N, K, device)
    a_scale = _pack_scale_words(a_scale_raw)
    b_scale = _pack_scale_words(b_scale_raw)
    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    a_dequantized = _dequantize_mxfp8(a, a_scale_raw)
    b_dequantized = _dequantize_mxfp8(b, b_scale_raw)
    c_ref = (a_dequantized @ b_dequantized.T).to(OUT_DTYPE)

    launch_fn = compile_mxfp8_gemm_4w(
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        use_xcd_remap=not disable_xcd_remap,
    )

    print(
        f"\n[mxfp8_gemm_4wave] M={M} N={N} K={K} "
        f"BLOCK_M={tile_m} BLOCK_N={tile_n} "
        f"xcd_remap={not disable_xcd_remap} "
        f"static_weight_scale={static_weight_scale}"
    )

    def _args(c, a_input, b_input, a_scale_input, b_scale_input):
        b_flat = _as_u8(b_input).contiguous().view(-1)
        a_scale_flat = a_scale_input.contiguous().view(-1)
        b_scale_flat = b_scale_input.contiguous().view(-1)

        if static_weight_scale:
            b_flat = flyc.from_torch_tensor(b_flat)
            a_scale_flat = flyc.from_torch_tensor(a_scale_flat)
            b_scale_flat = flyc.from_torch_tensor(b_scale_flat)

        return (
            _as_u8(a_input).contiguous().view(-1),
            b_flat,
            c.contiguous().view(-1),
            a_scale_flat,
            b_scale_flat,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(
        launch_fn,
        *_args(c_out, a, b, a_scale, b_scale),
    )

    def _launch(c, a_input, b_input, a_scale_input, b_scale_input):
        compiled(*_args(c, a_input, b_input, a_scale_input, b_scale_input))

    num_iters = max(2, int(num_iters))
    _, microseconds = run_perftest(
        _launch,
        c_out,
        a,
        b,
        a_scale,
        b_scale,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    _verify_mxfp8_output(c_out, c_ref)

    flops = 2 * M * N * K
    bytes_moved = (
        M * K
        + N * K
        + M * N * torch.tensor([], dtype=OUT_DTYPE).element_size()
        + M * (K // MX_BLOCK_K)
        + N * (K // MX_BLOCK_K)
    )
    tflops = flops / (microseconds / 1.0e6) / 1.0e12
    tbps = bytes_moved / 1.0e12 / (microseconds / 1.0e6)
    print(f"[flyc] Throughput: {microseconds:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")
    return tflops


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    [
        pytest.param(512, 512, 512, 256, 256, id="512x512x512"),
        pytest.param(5120, 5120, 8320, 256, 256, id="5120x5120x8320"),
        pytest.param(
            8192,
            8192,
            8192,
            256,
            256,
            marks=pytest.mark.large_shape,
            id="8192x8192x8192",
        ),
        pytest.param(
            9728,
            8192,
            8320,
            256,
            256,
            marks=pytest.mark.large_shape,
            id="9728x8192x8320",
        ),
        pytest.param(
            16384,
            16384,
            16384,
            256,
            256,
            marks=pytest.mark.large_shape,
            id="16384x16384x16384",
        ),
    ],
)
def test_mxfp8_gemm_4wave(M, N, K, tile_m, tile_n):
    _bench_mxfp8_gemm_4wave(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MXFP8 4-wave GEMM benchmark")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=256)
    parser.add_argument("--tile_n", type=int, default=256)
    parser.add_argument(
        "--disable_xcd_remap",
        action="store_true",
        default=False,
    )
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
        "--dynamic_weight_scale",
        action="store_true",
        default=False,
        help=("Use dynamic tensor arguments for weight and scales instead of static DLPack adaptors."),
    )
    args = parser.parse_args()

    torch.set_default_device("cuda")

    try:
        _bench_mxfp8_gemm_4wave(
            M=args.M,
            N=args.N,
            K=args.K,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            disable_xcd_remap=args.disable_xcd_remap,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
            static_weight_scale=not args.dynamic_weight_scale,
        )
    except pytest.skip.Exception as error:
        print(f"Skipped: {error}")
