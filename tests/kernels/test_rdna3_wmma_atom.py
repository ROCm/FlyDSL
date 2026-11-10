#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for the RDNA3 (gfx11*) WMMA MMA atom.

The gfx11 counterpart of ``test_rdna4_wmma_atom.py``. RDNA3 keeps the same
16x16x16 shapes as RDNA4 but uses the legacy v16-operand register ABI, modelled
by the ``gfx11.wmma`` atom (``lib/Dialect/FlyROCDL/GFX11/MmaAtom.cpp``):

  * A/B carry 16 elements per lane, and lanes 16-31 hold a *replica* of what
    lanes 0-15 hold. The atom expresses that as stride 0 on the ``lane/16`` axis
    of the thread layout, so the A/B partition is deliberately not injective.
  * C/D interleaves rows between the two lane halves (``row = 2*val + lane/16``)
    rather than giving each half a contiguous run of rows.

Both of those are exactly the places where a hand-written kernel gets the lane
math wrong, and both are what ``make_tiled_copy_{A,B,C}`` has to reproduce for
gfx11 kernels to be written against the layout API instead of raw intrinsics.
``kernels/gemm/rdna3_f16_gemm.py`` still spells the math out by hand; these
tests are what has to pass before it can stop doing that.

The integer atom is covered separately in ``test_rdna3_integer_wmma_atom.py``,
which builds its fragments by hand and so does not exercise the tiled copies.
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
from flydsl.expr import range_constexpr  # noqa: E402
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
    pytest.skip(f"RDNA3 WMMA atom requires gfx11*, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
WMMA_M = WMMA_N = WMMA_K = 16

# Unlike RDNA4, gfx11 WMMA does not return a bit-exact f32 accumulator even for
# integer-valued fp16/bf16 inputs: results land within about one f32 ULP of the
# exact sum. The residue is in the instruction, not in the fragment layouts —
# assembling the fragments by hand and going through make_tiled_copy produce
# bit-identical output. So this tolerance has to be loose enough to absorb one
# ULP but far tighter than any layout error, which mixes in whole other rows or
# columns and is wrong by O(1) or more.
_ATOL = 1e-3
_RTOL = 1e-6


def _compile_single_wmma(elem_cls):
    """One wave, one atom: C[16,16] = A[16,16] @ B[16,16].T."""
    f32 = fx.Float32

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x

        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((WMMA_M, WMMA_K), (WMMA_K, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((WMMA_N, WMMA_K), (WMMA_K, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((WMMA_M, WMMA_N), (WMMA_N, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, f32))
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


def _compile_tiled_wmma(elem_cls, tile_m, tile_n, tile_k, waves_m, waves_n):
    """A wave grid with atom repeats, the geometry a real GEMM block tile uses."""
    f32 = fx.Float32
    threads = waves_m * waves_n * WAVE_SIZE
    k_iters = tile_k // WMMA_K

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x

        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((tile_m, tile_k), (tile_k, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((tile_n, tile_k), (tile_k, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((tile_m, tile_n), (tile_n, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, f32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)))
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
        for ki in range_constexpr(k_iters):
            fx.gemm(tiled_mma, frag_C, frag_A[None, None, ki], frag_B[None, None, ki], frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def _exact_operands(m, n, k, torch_dtype):
    """Small integer inputs, so the f32 result is exact and any mismatch is layout."""
    torch.manual_seed(0)
    a = (torch.randn(m, k, device="cuda") * 4).round().to(torch_dtype)
    b = (torch.randn(n, k, device="cuda") * 4).round().to(torch_dtype)
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    return a, b, c


@pytest.mark.parametrize(
    "elem_cls, torch_dtype",
    [
        (fx.BFloat16, torch.bfloat16),
        (fx.Float16, torch.float16),
    ],
    ids=["bf16", "f16"],
)
def test_single_wmma_atom(elem_cls, torch_dtype):
    """A single gfx11.wmma atom call must match A @ B.T exactly.

    This is the narrowest check that ``make_tiled_copy_A`` can partition an
    operand whose thread layout replicates across the two lane halves.
    """
    a, b, c = _exact_operands(WMMA_M, WMMA_N, WMMA_K, torch_dtype)

    launch = _compile_single_wmma(elem_cls)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(c, a.float() @ b.float().T, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize(
    "tile_m, tile_n, tile_k, waves_m, waves_n",
    [
        pytest.param(32, 32, 16, 2, 2, id="32x32x16-2x2waves"),
        pytest.param(64, 64, 32, 2, 2, id="64x64x32-2x2waves-2x2x2repeats"),
        pytest.param(32, 32, 32, 1, 1, id="32x32x32-1wave-2x2x2repeats"),
    ],
)
def test_tiled_wmma_gemm(tile_m, tile_n, tile_k, waves_m, waves_n):
    """Atom repeats across a wave grid, the geometry a block tile actually uses.

    A single atom can be right while the tiled partition is wrong, because the
    repeat and wave axes are what stack on top of the replicated lane axis.
    """
    a, b, c = _exact_operands(tile_m, tile_n, tile_k, torch.bfloat16)

    launch = _compile_tiled_wmma(fx.BFloat16, tile_m, tile_n, tile_k, waves_m, waves_n)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(c, a.float() @ b.float().T, atol=_ATOL, rtol=_RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
