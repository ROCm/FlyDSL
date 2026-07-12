#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for the gfx1250 N-D TDM copy atom and the MX-scaled
WMMA MMA atom introduced in PR #830.

These exercise the new high-level atom APIs end-to-end on hardware:
  * ``fx.rocdl.make_tdm_atom`` + ``fx.copy_atom_call`` (Global<->LDS TDM DMA)
  * ``fx.rocdl.WMMAScale`` + ``fx.make_mma_atom`` + ``fx.gemm`` (E8M0 block scale)

!!! UNVALIDATED SCAFFOLD !!!
These were authored on a non-gfx1250 host (MI300 / gfx942) and could NOT be run
there — the atoms lower to gfx1250-only intrinsics. They are gated to skip on any
non-gfx1250 GPU. They follow the documented atom APIs and the existing kernel /
tiled-MMA idioms, but the hardware-specific fragment layouts and TDM async
fencing have not been confirmed on real gfx1250 silicon. A gfx1250 owner must run
and, if needed, correct these before treating them as regression coverage.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if "gfx1250" not in _ARCH:
    pytest.skip(
        f"gfx1250 TDM / MX-scale atoms require gfx1250, got {_ARCH}",
        allow_module_level=True,
    )

WAVE_SIZE = 32


# =============================================================================
# TDM copy atom: Global -> LDS -> Global whole-tile round-trip (identity copy).
# =============================================================================


def _compile_tdm_roundtrip(M: int, N: int, num_warps: int):
    from flydsl.expr.rocdl import tdm_ops

    @flyc.kernel
    def tdm_roundtrip_kernel(A: fx.Tensor, C: fx.Tensor):
        # 2D tile views over the flat global tensors (row-major (M, N):(N, 1)).
        a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, N), (N, 1)))
        c2d = fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1)))

        # One contiguous LDS tile of the same shape.
        lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, M * N]).peek()
        lds2d = fx.make_view(lds.ptr, fx.make_layout((M, N), (N, 1)))

        # Global -> LDS. tensor_extents = full tile (no OOB clamp); the innermost
        # stride is assumed 1 so only the outer stride (N) is carried, and it
        # falls back to the static layout stride here (passed None).
        load_atom = fx.rocdl.make_tdm_atom(a2d, [M, N], num_warps=num_warps)
        fx.copy_atom_call(load_atom, a2d, lds2d)
        tdm_ops.tensor_wait(0)
        fx.barrier()

        # LDS -> Global.
        store_atom = fx.rocdl.make_tdm_atom(c2d, [M, N], num_warps=num_warps)
        fx.copy_atom_call(store_atom, lds2d, c2d)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(A: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        tdm_roundtrip_kernel(A, C).launch(grid=(1, 1, 1), block=(num_warps * WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("M,N", [(128, 64)])
def test_tdm_roundtrip_identity(M, N, num_warps):
    """A whole-tile Global->LDS->Global TDM copy must reproduce the input."""
    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    c = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    launch = _compile_tdm_roundtrip(M, N, num_warps)
    launch(a, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(c, a, atol=0, rtol=0)


# =============================================================================
# MX-scaled WMMA atom: 16x16x128 fp8 with E8M0 block scale == 1.0.
# =============================================================================


def _e8m0_ones_i32() -> int:
    """block-32 scale operand (i32): four E8M0 bytes, each exponent 127 == 1.0."""
    return 0x7F7F7F7F


def _compile_wmma_scale_fp8(M: int, N: int, K: int):
    f8 = fx.Float8E4M3FN
    f32 = fx.Float32
    scale_ones = _e8m0_ones_i32()

    @flyc.kernel
    def wmma_scale_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x

        # CDNA buffer copy loads/stores through a buffer resource, so wrap the
        # global tensors in a buffer-resource view (make_buffer_tensor). This is
        # legal on gfx1250 too (buffer_load / buffer_store); only the TDM DMA
        # path needs a raw VA instead.
        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, K), (K, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, K), (K, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(M, N, K, f8, f8, f32))
        # E8M0 block scales of 1.0 on both operands (atom state).
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(scale_ones))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(scale_ones))

        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(f8.width), f8)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(f32.width), f32)
        tiled_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_a, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_a, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))

        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_scale_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


def test_wmma_scale_fp8_identity_scale():
    """16x16x128 fp8 scaled WMMA with unit E8M0 scales == plain A @ B.T."""
    M, N, K = 16, 16, 128
    torch.manual_seed(0)
    # Small integer-ish values so fp8 e4m3 rounding does not dominate the check.
    a = (torch.randn(M, K, device="cuda") * 4).round().clamp(-8, 8)
    b = (torch.randn(N, K, device="cuda") * 4).round().clamp(-8, 8)
    a8 = a.to(torch.float8_e4m3fn)
    b8 = b.to(torch.float8_e4m3fn)
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch = _compile_wmma_scale_fp8(M, N, K)
    launch(a8, b8, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = a8.to(torch.float32) @ b8.to(torch.float32).T
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
