#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device correctness tests for the gfx11 INT8 WMMA GEMM."""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401 -- preload comgr before torch/HIP loads LLVM
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx11"):
    pytest.skip(f"RDNA3 integer WMMA requires gfx11*, got {_ARCH}", allow_module_level=True)

from kernels.gemm.rdna3_int8_gemm import create_wmma_int8_gemm_module  # noqa: E402
from kernels.gemm.rdna3_int8_gemm_autotune import (  # noqa: E402
    TILE_32x32x64,
    TILE_32x64x64,
    TILE_64x128x64,
    TILE_128x128x64,
    persistent_workgroups,
    pick_split_k,
    pick_tile,
    rdna3_int8_gemm_autotuned,
)

_TORCH_IN = {"int8": torch.int8, "uint8": torch.uint8}
_RANGE_IN = {"int8": (-128, 128), "uint8": (0, 256)}


def _operands(M, N, K, in_dtype, seed=2026):
    """Create contiguous inputs with dtype extremes in the first rows."""
    torch.manual_seed(seed)
    low, high = _RANGE_IN[in_dtype]
    dtype = _TORCH_IN[in_dtype]
    A = torch.randint(low, high, (M, K), dtype=dtype)
    B_T = torch.randint(low, high, (N, K), dtype=dtype)
    A[0].fill_(low)
    A[1 % M].fill_(high - 1)
    A[2 % M].zero_()
    B_T[0].fill_(high - 1)
    B_T[1].fill_(low)
    B_T[2].zero_()
    return A.contiguous(), B_T.contiguous()


def _int_reference(A, B_T):
    return A.cpu().to(torch.int64) @ B_T.cpu().to(torch.int64).T


@pytest.mark.parametrize(
    "M,N,K",
    [
        (128, 128, 128),
        (256, 256, 256),
        (256, 128, 512),
        pytest.param(512, 512, 512, marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("in_dtype", ["int8", "uint8"])
def test_int8_gemm_exact(M, N, K, in_dtype):
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(M, N, K, in_dtype=in_dtype, out_dtype="i32")

    A, B_T = _operands(M, N, K, in_dtype)
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = _int_reference(A, B_T)
    assert ref.abs().max() < 2**31, "reference overflows i32; pick a smaller K"
    torch.testing.assert_close(C.cpu(), ref.to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("M", [1, 16, 17, 65, 130])
def test_int8_gemm_partial_m(M):
    N, K = 64, 128
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype="i32",
        reg_m=1,
        reg_n=2,
        reg_k=4,
        waves_m=1,
        waves_n=2,
    )

    A, B_T = _operands(M, N, K, "int8")
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", ["bf16", "f16", "f32"])
def test_int8_gemm_dequant(out_dtype):
    M = N = K = 256
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype=out_dtype,
        scale_mode="row_col",
    )

    A, B_T = _operands(M, N, K, "int8")
    torch.manual_seed(7)
    scale_a = (torch.rand(M, dtype=torch.float32) * 0.01 + 0.001).cuda()
    scale_b = (torch.rand(N, dtype=torch.float32) * 0.01 + 0.001).cuda()
    out_torch = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[out_dtype]
    C = torch.zeros(M, N, dtype=out_torch, device="cuda")

    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream(), scale_a, scale_b)
    torch.cuda.synchronize()

    ref = _int_reference(A, B_T).float().cuda() * scale_a.unsqueeze(1) * scale_b.unsqueeze(0)
    rtol, atol = {"bf16": (4e-3, 0.0), "f16": (1e-3, 2**-24), "f32": (0.0, 0.0)}[out_dtype]
    torch.testing.assert_close(C.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "tile",
    [
        (4, 4, 2, 2, 2),  # 128x128x32
        (4, 4, 2, 4, 4),  # 256x256x32
        (2, 2, 4, 2, 2),  # 64x64x64
        (2, 2, 4, 1, 1),  # 32x32x64
    ],
)
@pytest.mark.parametrize("lds_layout", ["pad", "kblock"])
def test_int8_gemm_tiles(tile, lds_layout):
    M = N = K = 256
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype="i32",
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        lds_layout=lds_layout,
        sched_hint=True,
    )

    A, B_T = _operands(M, N, K, "int8")
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


def test_int8_gemm_persistent():
    M = N = K = 256
    _, BLOCK_M, BLOCK_N, _ = create_wmma_int8_gemm_module(M, N, K, in_dtype="int8", out_dtype="i32")
    num_tiles = -(-M // BLOCK_M) * (N // BLOCK_N)
    persistent_wgs = max(1, num_tiles // 2)
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype="i32",
        persistent_wgs=persistent_wgs,
    )

    A, B_T = _operands(M, N, K, "int8")
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


def test_int8_gemm_rejects_too_many_persistent_workgroups():
    with pytest.raises(ValueError, match="between 0"):
        create_wmma_int8_gemm_module(256, 256, 256, persistent_wgs=9)


@pytest.mark.parametrize("out_dtype", ["bf16", "f16", "f32"])
def test_int8_gemm_unscaled_float_out(out_dtype):
    M = N = K = 256
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(M, N, K, in_dtype="int8", out_dtype=out_dtype)

    A, B_T = _operands(M, N, K, "int8")
    out_torch = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[out_dtype]
    C = torch.zeros(M, N, dtype=out_torch, device="cuda")

    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = _int_reference(A, B_T).to(out_torch)
    torch.testing.assert_close(C.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("M,N,K,split_k", [(256, 256, 512, 2), (64, 256, 1024, 4), (130, 128, 512, 2)])
def test_int8_gemm_split_k_is_exact(M, N, K, split_k):
    """Integer adds are associative, so a split K has to match tile for tile."""
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype="i32",
        split_k=split_k,
    )

    A, B_T = _operands(M, N, K, "int8")
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")
    launch_fn(C, A.to("cuda"), B_T.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


def test_int8_gemm_split_k_rejects_unsupported_epilogues():
    with pytest.raises(ValueError, match="out_dtype='i32'"):
        create_wmma_int8_gemm_module(256, 256, 512, in_dtype="int8", out_dtype="bf16", split_k=2)
    with pytest.raises(ValueError, match="must divide"):
        create_wmma_int8_gemm_module(256, 256, 512, in_dtype="int8", out_dtype="i32", split_k=3)
    with pytest.raises(ValueError, match="under 2 K-tiles"):
        create_wmma_int8_gemm_module(256, 256, 512, in_dtype="int8", out_dtype="i32", split_k=8)


def test_int8_autotune_splits_k_only_when_the_grid_is_short():
    # 32 output tiles on a 48-processor part leave room for several K slices.
    assert pick_split_k(64, 4096, 4096, TILE_64x128x64, num_cu=48) == 4
    # 128 tiles already oversubscribe it, and C is large besides.
    assert pick_split_k(1024, 1024, 1024, TILE_64x128x64, num_cu=48) == 1
    assert pick_split_k(4096, 4096, 4096, TILE_64x128x64, num_cu=48) == 1


def test_int8_autotune_avoids_tiles_taller_than_m():
    # A 64-row tile on a 32-row problem multiplies 32 rows of padding.
    assert pick_tile(32, 4096, 4096, num_cu=48) == TILE_32x64x64
    # M is a whole number of 64-row tiles here, so the wide tile stays.
    assert pick_tile(64, 4096, 4096, num_cu=48) == TILE_64x128x64


def test_int8_autotune_narrows_the_tile_when_the_grid_is_short():
    # 64x128 leaves 512^3 with 32 workgroups for 48 processors; 32x64 gives 128.
    assert pick_tile(512, 512, 512, num_cu=48) == TILE_32x64x64
    assert pick_tile(1024, 1024, 1024, num_cu=48) == TILE_64x128x64


def test_int8_autotune_counts_split_k_towards_filling_the_grid():
    # 32x64 reaches only 64 workgroups alone; split_k=4 carries it over the line.
    assert pick_tile(32, 4096, 4096, num_cu=48, splittable=True) == TILE_32x64x64
    assert pick_tile(32, 4096, 4096, num_cu=48, splittable=False) == TILE_32x32x64


def test_int8_autotune_uses_device_cu_count():
    shape = (6144, 6144, 1024)
    assert pick_tile(*shape, num_cu=96) == TILE_64x128x64
    assert pick_tile(*shape, num_cu=40) == TILE_128x128x64
    assert persistent_workgroups(8192, 8192, 1024, TILE_64x128x64, num_cu=96) == 384
    assert persistent_workgroups(8192, 8192, 1024, TILE_64x128x64, num_cu=40) == 160


def test_int8_gemm_autotuned_default():
    M = N = K = 256
    A, B_T = _operands(M, N, K, "int8")
    A_gpu, B_gpu = A.cuda(), B_T.cuda()
    C = torch.full((M, N), -1, dtype=torch.int32, device="cuda")

    rdna3_int8_gemm_autotuned(C, A_gpu, B_gpu)
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


def test_int8_gemm_strided():
    M = N = K = 256
    pad = 16
    launch_fn, _, _, _ = create_wmma_int8_gemm_module(
        M,
        N,
        K,
        in_dtype="int8",
        out_dtype="i32",
        lda=K + pad,
        ldb=K + pad,
        ldc=N + pad,
    )

    A, B_T = _operands(M, N, K, "int8")
    A_pad = torch.zeros(M, K + pad, dtype=torch.int8)
    B_pad = torch.zeros(N, K + pad, dtype=torch.int8)
    A_pad[:, :K] = A
    B_pad[:, :K] = B_T
    C = torch.full((M, N + pad), -1, dtype=torch.int32, device="cuda")

    launch_fn(C, A_pad.to("cuda"), B_pad.to("cuda"), torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C[:, :N].cpu(), _int_reference(A, B_T).to(torch.int32), atol=0, rtol=0)


def test_int8_gemm_rejects_oversized_lds():
    with pytest.raises(ValueError, match="LDS"):
        create_wmma_int8_gemm_module(
            1024,
            1024,
            1024,
            in_dtype="int8",
            out_dtype="i32",
            reg_m=4,
            reg_k=8,
        )


def test_int8_gemm_default_tile_follows_shape():
    _, small_bm, _, _ = create_wmma_int8_gemm_module(1024, 1024, 1024, in_dtype="int8", out_dtype="i32")
    _, large_bm, _, _ = create_wmma_int8_gemm_module(8192, 8192, 8192, in_dtype="int8", out_dtype="i32")
    assert small_bm == 64
    assert large_bm == 128

    _, forced_bm, _, _ = create_wmma_int8_gemm_module(
        1024,
        1024,
        1024,
        in_dtype="int8",
        out_dtype="i32",
        reg_m=4,
    )
    assert forced_bm == 128


def test_int8_gemm_rejects_misaligned_ld():
    with pytest.raises(ValueError, match="multiples of 16"):
        create_wmma_int8_gemm_module(256, 256, 256, in_dtype="int8", out_dtype="i32", lda=264)


def test_int8_gemm_rejects_i32_with_scales():
    with pytest.raises(ValueError, match="out_dtype cannot be 'i32'"):
        create_wmma_int8_gemm_module(
            256,
            256,
            256,
            in_dtype="int8",
            out_dtype="i32",
            scale_mode="row_col",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
