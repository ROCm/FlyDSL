#!/usr/bin/env python3
"""RDNA 3 / 3.5 WMMA GEMM correctness tests (gfx1100 / gfx1150, wave32).

Exercises the atom-based path through ``MmaOpRDNA3_WMMAType`` introduced
for the gfx1150 (Strix Point) PoC. RDNA 4 (gfx1201) and gfx1250 keep
their existing ``MmaOpGFX1250_WMMAType`` path and are not exercised here
(they have wider K-shapes and FP8 support that this atom does not cover).
"""

import logging
import os
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.ir import Context

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna3_f16_gemm import create_rdna3_wmma_gemm_module
from tests.test_common import verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna3():
    if not (ARCH.startswith("gfx115") or ARCH.startswith("gfx110")):
        pytest.skip(f"RDNA3 / 3.5 WMMA tests require gfx110x or gfx115x, got {ARCH}")


def _create_rdna3_wmma_iu8_i32_module():
    """Build a minimal single-tile IU8->I32 RDNA3 WMMA kernel (16x16x16)."""
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    THREADS_PER_BLOCK = 32

    @flyc.kernel
    def gemm_iu8_i32_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        tid = fx.thread_idx.x

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA = fx.zipped_divide(A, (WMMA_M, WMMA_K))
        bB = fx.zipped_divide(B, (WMMA_N, WMMA_K))
        bC = fx.zipped_divide(C, (WMMA_M, WMMA_N))
        bA = fx.slice(bA, (None, (0, 0)))
        bB = fx.slice(bB, (None, (0, 0)))
        bC = fx.slice(bC, (None, (0, 0)))

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA_RDNA3(WMMA_M, WMMA_N, WMMA_K, fx.Int8, fx.Int32)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_ab = fx.make_copy_atom(
            fx.rocdl.BufferCopy(fx.Int8.width), fx.Int8
        )
        copy_atom_c = fx.make_copy_atom(
            fx.rocdl.BufferCopy(fx.Int32.width), fx.Int32
        )
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        copy_src_A = thr_copy_A.partition_S(bA)
        copy_src_B = thr_copy_B.partition_S(bB)
        copy_dst_C = thr_copy_C.partition_S(bC)

        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_frag_A = thr_copy_A.retile(frag_A)
        copy_frag_B = thr_copy_B.retile(frag_B)
        copy_frag_C = thr_copy_C.retile(frag_C)

        frag_C.fill(0)
        fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
        fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_iu8_i32_kernel(A, B, C).launch(
            grid=(1, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    return launch_gemm


# ── Single-warp WMMA correctness ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 16, id="16x16x16"),
        pytest.param(32, 16, 16, id="32x16x16"),
        pytest.param(16, 32, 16, id="16x32x16"),
        pytest.param(16, 16, 32, id="16x16x32"),
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
    ],
)
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_rdna3_wmma_f32_acc(M, N, K, dtype):
    """RDNA 3 / 3.5 WMMA with F32 accumulator, F16 / BF16 inputs."""
    _requires_rdna3()

    torch.manual_seed(42)
    torch_dtype = torch.float16 if dtype == "f16" else torch.bfloat16

    A = (torch.randn(M, K, dtype=torch_dtype, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch_dtype, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype=dtype, out_dtype="f32"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C, C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 16, id="16x16x16"),
        pytest.param(64, 64, 64, id="64x64x64"),
    ],
)
def test_rdna3_wmma_f16_acc(M, N, K):
    """RDNA 3 / 3.5 WMMA with F16 accumulator (same-precision)."""
    _requires_rdna3()

    torch.manual_seed(42)

    A = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype="f16", out_dtype="f16"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A.float() @ B_T.float().T).to(torch.float16)
    assert verify_output(C, C_ref, atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 16, id="16x16x16"),
        pytest.param(16, 16, 32, id="16x16x32"),
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
    ],
)
def test_rdna3_wmma_bf16_acc(M, N, K):
    """RDNA 3 / 3.5 WMMA with BF16 accumulator (same-precision).

    Exercises the multi-K-tile accumulation path for the bf16-acc lowering,
    which promotes to f32 internally and truncates to bf16 on output.
    """
    _requires_rdna3()

    torch.manual_seed(42)

    A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype="bf16", out_dtype="bf16"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A.float() @ B_T.float().T).to(torch.bfloat16)
    assert verify_output(C, C_ref, atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 32, id="16x16x32"),
        pytest.param(128, 128, 128, id="128x128x128"),
    ],
)
def test_rdna3_wmma_f16_acc_multi_k(M, N, K):
    """f16-acc with multi-K-tile accumulation (stresses the f32 fallback path)."""
    _requires_rdna3()

    torch.manual_seed(42)

    A = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype="f16", out_dtype="f16"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A.float() @ B_T.float().T).to(torch.float16)
    assert verify_output(C, C_ref, atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(32, 16, 16, id="32x16x16"),
        pytest.param(16, 32, 16, id="16x32x16"),
        pytest.param(64, 64, 64, id="64x64x64"),
    ],
)
def test_rdna3_wmma_iu8_i32_multi_tile(M, N, K):
    """IU8 -> I32 with multi-tile / multi-K to validate the iu8 lowering at scale."""
    _requires_rdna3()

    torch.manual_seed(42)
    A = torch.randint(0, 8, (M, K), dtype=torch.int8, device="cuda")
    B_T = torch.randint(0, 8, (N, K), dtype=torch.int8, device="cuda")
    C = torch.zeros((M, N), dtype=torch.int32, device="cuda")

    @flyc.kernel
    def gemm_iu8_i32_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        WMMA_M = 16
        WMMA_N = 16
        WMMA_K = 16
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA_all = fx.zipped_divide(A, (WMMA_M, WMMA_K))
        bB_all = fx.zipped_divide(B, (WMMA_N, WMMA_K))
        bC = fx.zipped_divide(C, (WMMA_M, WMMA_N))
        bC = fx.slice(bC, (None, (bid_m, bid_n)))

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA_RDNA3(WMMA_M, WMMA_N, WMMA_K, fx.Int8, fx.Int32)
        )
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_ab = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Int8.width), fx.Int8)
        copy_atom_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Int32.width), fx.Int32)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        copy_dst_C = thr_copy_C.partition_S(bC)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_frag_C = thr_copy_C.retile(frag_C)
        frag_C.fill(0)

        for k_tile in fx.range_constexpr(K // WMMA_K):
            bA = fx.slice(bA_all, (None, (bid_m, k_tile)))
            bB = fx.slice(bB_all, (None, (bid_n, k_tile)))
            copy_src_A = thr_copy_A.partition_S(bA)
            copy_src_B = thr_copy_B.partition_S(bB)
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            copy_frag_A = thr_copy_A.retile(frag_A)
            copy_frag_B = thr_copy_B.retile(frag_B)
            fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
            fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        gemm_iu8_i32_kernel(A, B, C).launch(
            grid=(M // 16, N // 16, 1), block=(32, 1, 1), stream=stream,
        )

    launch(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    C_ref = (A.cpu().to(torch.int32) @ B_T.cpu().to(torch.int32).T).to("cuda")
    assert torch.equal(C, C_ref)


def test_rdna3_wmma_iu8_i32_acc():
    """RDNA3 WMMA IU8->I32 runtime correctness on a single 16x16x16 tile."""
    _requires_rdna3()

    torch.manual_seed(42)
    A = torch.randint(0, 8, (16, 16), dtype=torch.int8, device="cuda")
    B_T = torch.randint(0, 8, (16, 16), dtype=torch.int8, device="cuda")
    C = torch.zeros((16, 16), dtype=torch.int32, device="cuda")

    launch_fn = _create_rdna3_wmma_iu8_i32_module()
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (
        A.cpu().to(torch.int32) @ B_T.cpu().to(torch.int32).T
    ).to(device="cuda", dtype=torch.int32)
    assert torch.equal(C, C_ref)


def test_rdna3_wmma_iu4_i32_module_contract():
    """IU4->I32 should be accepted by the RDNA3 WMMA atom builder.

    Runtime execution is covered by MLIR conversion tests because host-side
    tensor frameworks do not expose a native int4 tensor dtype.
    """
    _requires_rdna3()
    with Context():
        mma_ty = fx.rocdl.WMMA_RDNA3(16, 16, 16, fx.Int4, fx.Int32)
        mma_atom = fx.make_mma_atom(mma_ty)
        assert mma_atom is not None
