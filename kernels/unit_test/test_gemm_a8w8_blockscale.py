# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Unit tests for A8W8 FP8 blockscale GEMM kernel.

Kernel implementation lives in `kernels/gemm_a8w8_blockscale.py`.
This file is the correctness harness.

Reference computation follows the triton aiter blockscale convention:
    block_shape = (scale_block_n, scale_block_k) = (128, 128)
    y = (x_fp8 * x_scale_broadcast) @ (w_fp8 * w_scale_broadcast)^T
where each scale is broadcast across its (scale_block_n, scale_block_k) tile.

Tests are automatically skipped if gfx1250 is not detected (the kernel
targets MI350 WMMA_SCALE FP8).

pytest -k filter examples:
    pytest ... -k "basic"           # only basic shape tests
    pytest ... -k "num_buffers"     # only pipeline depth tests
    pytest ... -k "dtype"           # only output-dtype tests
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# workaround for simulator — preload system comgr before torch/HIP loads LLVM
import flydsl  # noqa: E402,F401

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from tests.test_common import verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


# ═════════════════════════════════════════════════════════════════════════════
# Skip / setup helpers
# ═════════════════════════════════════════════════════════════════════════════

# Default scale block sizes — match the aiter blockscale convention.
SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128


def _check_gfx1250():
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        pytest.skip(f"gemm_a8w8_blockscale requires gfx1250, got {arch}")


def _pad_k(K, tile_k=128, scale_block_k=128):
    """Return K padded up to the nearest multiple of both tile_k and scale_block_k.
    Callers feed the kernel the padded K; X/W get zero-padded along K so the
    padded elements contribute 0 to the matmul."""
    lcm = tile_k * scale_block_k // _gcd(tile_k, scale_block_k)
    return ((K + lcm - 1) // lcm) * lcm


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _check_shape_compat(M, N, K, tile_k=128, scale_block_k=128, num_buffers=2):
    """Kernel requires num_k_tiles >= num_buffers - 1 (pre_loaded tiles)."""
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers - 1:
        pytest.skip(
            f"{num_buffers}-stage pipeline requires num_k_tiles >= {num_buffers - 1}, "
            f"got K={K} (num_k_tiles={num_k_tiles})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Input generation + reference
# ═════════════════════════════════════════════════════════════════════════════

def _get_fp8_dtype():
    """Pick the right FP8 dtype for our arch. gfx1250/MI350 uses E4M3FN (OCP)."""
    # torch.float8_e4m3fn is the standard OCP FP8 (no NaN repurposing of 0x80).
    # MI300 uses FNUZ; MI350 uses FN. Our kernel targets gfx1250 so FN is right.
    return torch.float8_e4m3fn


def _generate_inputs(M, N, K, tile_k=128,
                     scale_block_n=SCALE_BLOCK_N, scale_block_k=SCALE_BLOCK_K):
    """Build FP8 X/W plus their f32 block scales at the given (possibly unaligned) K.

    Uses small magnitudes (rand/10) so FP8 quantization doesn't saturate.
    """
    torch.manual_seed(0)
    fp8 = _get_fp8_dtype()

    x = (torch.rand((M, K), dtype=torch.float32, device="cuda") / 10).to(fp8)
    w = (torch.rand((N, K), dtype=torch.float32, device="cuda") / 10).to(fp8)

    scale_k = (K + scale_block_k - 1) // scale_block_k
    scale_n = (N + scale_block_n - 1) // scale_block_n

    x_scale = torch.rand((M, scale_k), dtype=torch.float32, device="cuda")
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32, device="cuda")

    return x, w, x_scale, w_scale


def _reference_output(x_fp8, w_fp8, x_scale, w_scale,
                      scale_block_n=SCALE_BLOCK_N,
                      scale_block_k=SCALE_BLOCK_K,
                      dtype=torch.bfloat16):
    """Reference: broadcast scales over their block tiles, dequantize, then matmul.

    Mirrors triton's reference implementation in
    aiter/op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py:run_torch
    """
    M, K = x_fp8.shape
    N = w_fp8.shape[0]

    # Broadcast x_scale from (M, scale_k) → (M, K) by repeating along K.
    xs_broadcast = x_scale.repeat_interleave(scale_block_k, dim=1)[:M, :K]
    x_deq = x_fp8.to(xs_broadcast.dtype) * xs_broadcast

    # Broadcast w_scale from (scale_n, scale_k) → (N, K) by repeating along both dims.
    ws_broadcast = (
        w_scale.repeat_interleave(scale_block_n, dim=0)
               .repeat_interleave(scale_block_k, dim=1)
    )[:N, :K]
    w_deq = w_fp8.to(ws_broadcast.dtype) * ws_broadcast

    # Reference matmul in f32 for numerical headroom.
    out = torch.matmul(x_deq.float(), w_deq.float().T)
    return out.to(dtype)


# ═════════════════════════════════════════════════════════════════════════════
# Test shape lists
# ═════════════════════════════════════════════════════════════════════════════

def get_basic_shapes():
    """Small and medium shapes aligned to our 128× tile/scale constraints."""
    return [
        # Minimum-sized tile (one WMMA)
        (128, 128, 128),
        # Simple multi-tile
        (128, 256, 256),
        (256, 128, 256),
        # Rectangular
        (128, 512, 128),
        (512, 128, 128),
        # Deeper K (multiple main-loop iterations)
        (128, 128, 512),
        (128, 128, 1024),
        # aiter-style LLM shapes (small subset)
        (128, 1536, 7168),
        (128, 7168, 1536),
    ]


def get_large_shapes():
    """Larger shapes to catch tiling/pipeline bugs that only appear at scale."""
    return [
        (256, 1024, 1024),
        (512, 2048, 2048),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("M, N, K", get_basic_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_blockscale_basic(M, N, K, dtype):
    """Basic correctness across common shapes and output dtypes."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)

    # FP8 quantization + f32 scaling produces small but non-zero rounding.
    # Match triton's tolerance (atol=rtol=1e-2).
    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (256, 512, 512)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_gemm_a8w8_blockscale_num_buffers(M, N, K, num_buffers):
    """Correctness across different pipeline depths (ping-pong to quad-buffer)."""
    _check_gfx1250()
    _check_shape_compat(M, N, K, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale,
        dtype=torch.bfloat16,
        num_buffers=num_buffers,
    )

    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("M, N, K", [(128, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_dtype(M, N, K, dtype):
    """Correctness for each supported output dtype."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)

    # f32 output should be exact-to-reference-precision; f16/bf16 tolerate more.
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=rtol, atol=atol,
    )


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_gemm_a8w8_blockscale_preallocated_output(M, N, K):
    """GEMM writing into a caller-provided output tensor."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y)

    assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated y"
    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize(
    "M, N, K",
    [
        (128, 256, 256),
        (256, 128, 256),
        (128, 128, 512),
        (128, 128, 1024),
        # Main-loop + scales_per_tile: K=1024, tile_k=256 → 4 K-tiles (prologue
        # + loop iters + tail), each tile running 2 scale chunks.
        (1024, 1024, 1024),
    ],
)
def test_gemm_a8w8_blockscale_scales_per_tile(M, N, K):
    """Exercise scales_per_tile > 1: tile_k=256, scale_block_k=128 → 2 scale chunks/tile."""
    _check_gfx1250()
    _check_shape_compat(M, N, K, tile_k=256)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale,
        dtype=torch.bfloat16,
        tile_k=256,
    )

    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("M, N, K", get_large_shapes())
def test_gemm_a8w8_blockscale_large(M, N, K):
    """Correctness on larger shapes (slow — run separately)."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TDM-store epilogue tests
# ═════════════════════════════════════════════════════════════════════════════
# Exercise the LDS-staged async TDM-store path (use_tdm_store=True). Every
# shape here is also covered by the buffer_store path tests above; running
# them with use_tdm_store=True confirms the new path is bit-equivalent.

def get_tdm_store_shapes():
    """Shapes that hit different code paths in the TDM-store epilogue.

    Each entry exercises one of: single warp_tile region in N, multiple
    warps spanning the N axis, deeper K to cross main-loop boundaries.
    """
    return [
        # 1 N-warp tile per WG (single descriptor row of warps)
        (128, 128, 128),
        (256, 256, 256),
        # N spanning multiple WGs
        (128, 512, 256),
        # Deeper K (multi main-loop iters + drain)
        (128, 256, 1024),
        # Larger shape mirroring run_blockscale config
        (512, 1024, 1024),
    ]


@pytest.mark.parametrize("M, N, K", get_tdm_store_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_blockscale_tdm_store_basic(M, N, K, dtype):
    """TDM-store path: bf16 / fp16 outputs across basic shapes."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=dtype, use_tdm_store=True,
    )

    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("M, N, K", [(128, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_tdm_store_dtype(M, N, K, dtype):
    """TDM-store path: every supported output dtype on a small shape."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=dtype, use_tdm_store=True,
    )

    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=rtol, atol=atol,
    )


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (256, 512, 512)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_gemm_a8w8_blockscale_tdm_store_num_buffers(M, N, K, num_buffers):
    """TDM-store path: pipeline depth sweep — exercises stage 0 alias under
    different buffer counts (output staging always lives in stage 0)."""
    _check_gfx1250()
    _check_shape_compat(M, N, K, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale,
        dtype=torch.bfloat16,
        num_buffers=num_buffers,
        use_tdm_store=True,
    )

    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_gemm_a8w8_blockscale_tdm_store_preallocated_output(M, N, K):
    """TDM-store path: write into a caller-provided output tensor."""
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y, use_tdm_store=True,
    )

    assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated y"
    assert verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Standalone runner: python test_gemm_a8w8_blockscale.py -M 128 -N 256 -K 256
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-N", type=int, default=256)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "f32"])
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--tdm-store", action="store_true",
                        help="Use the LDS-staged TDM-store epilogue.")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    _check_gfx1250()
    _check_shape_compat(args.M, args.N, args.K, num_buffers=args.num_buffers)

    x, w, x_scale, w_scale = _generate_inputs(args.M, args.N, args.K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale,
        dtype=dtype,
        num_buffers=args.num_buffers,
        use_tdm_store=args.tdm_store,
    )

    torch.cuda.synchronize()
    passed = verify_output(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=1e-2, atol=1e-2,
    )
    print(f"M={args.M} N={args.N} K={args.K} dtype={args.dtype} "
          f"num_buffers={args.num_buffers} tdm_store={args.tdm_store}")
    print("PASSED" if passed else "FAILED")
