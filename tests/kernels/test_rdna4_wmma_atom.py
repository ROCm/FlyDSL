#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for the RDNA4 (gfx120x) WMMA MMA atom.

RDNA4 keeps the gfx11 16x16x16 WMMA shapes but uses the gfx1250 v8-operand
register ABI, so it gets its own ``gfx120x.wmma`` atom
(``lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp``). These tests pin the fragment
layouts by running a single-wave 16x16x16 GEMM through
``fx.make_mma_atom`` / ``fx.make_tiled_copy_{A,B,C}`` / ``fx.gemm`` and
comparing against torch.
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
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 WMMA atom requires gfx120x, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = K = 16


def _compile_single_wmma(elem_cls):
    """One wave, one atom: C[16,16] = A[16,16] @ B[16,16].T."""
    f32 = fx.Float32

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x

        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, K), (K, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, K), (K, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, elem_cls, f32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy(elem_cls.width), elem_cls)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(f32.width), f32)
        thr_copy_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

        fx.copy(copy_ab, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_ab, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))

        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "elem_cls, torch_dtype",
    [
        (fx.BFloat16, torch.bfloat16),
        (fx.Float16, torch.float16),
    ],
    ids=["bf16", "f16"],
)
def test_single_wmma_atom(elem_cls, torch_dtype):
    """A single gfx120x.wmma atom call must match A @ B.T exactly for integers.

    Integer-valued inputs keep the result exactly representable, so any
    mismatch is a fragment-layout bug rather than rounding.
    """
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 4).round().to(torch_dtype)
    b = (torch.randn(N, K, device="cuda") * 4).round().to(torch_dtype)
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch = _compile_single_wmma(elem_cls)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = a.float() @ b.float().T
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
