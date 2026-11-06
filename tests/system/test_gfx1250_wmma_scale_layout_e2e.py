#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Hardware check for the gfx1250 block-scaled WMMA A/B register layout.

These go through ``make_tiled_mma`` / ``make_fragment_A`` / ``make_tiled_copy_A``,
so the atom's thread-value layout decides where every element is fetched from.
That is the part a layout bug breaks, and the part no existing device test
covers: the shipping gfx1250 GEMM kernels fill their fragments with hand-written
LDS reads and never consult these layouts.

The case that matters is ``block_size=16`` with scales that differ per block.
With a wrong K hand-over width, every value still lands inside the right 32-wide
block -- which is why block-32 keeps passing -- but lands in the wrong 16-wide
one, so the scales are applied to the wrong elements.

  block32_uniform   control: passes under any plausible layout
  block32_varying   control: a shared K permutation cancels out here
  block16_uniform   control: no scale differences to misplace
  block16_varying   the real check
"""

import pytest

import flydsl  # noqa: F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if "gfx1250" not in _ARCH:
    pytest.skip(f"MX-scaled WMMA atoms require gfx1250, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = 16
K = 128


def _e8m0_bytes(exponents):
    """Pack E8M0 exponents (one per K block, per row) into the atom's scale word.

    Byte j of the word holds the scale for K block j of that lane's row; the
    word is i32 for block-32 (4 blocks) and i64 for block-16 (8 blocks).
    """
    word = 0
    for j, e in enumerate(exponents):
        word |= (int(e) & 0xFF) << (8 * j)
    return word


def _compile(block_size, scale_a_word, scale_b_word):
    f8, f32 = fx.Float8E4M3FN, fx.Float32
    scale_ty = fx.Int64 if block_size == 16 else fx.Int32

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, K), (K, 1)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, K), (K, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(M, N, K, f8, f8, f32, block_size=block_size))
        mma_atom = fx.atom_set_value(mma_atom, "scale_a", scale_ty(scale_a_word))
        mma_atom = fx.atom_set_value(mma_atom, "scale_b", scale_ty(scale_b_word))

        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(f8.width), f8)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(f32.width), f32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_a, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_a, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("block_size", [32, 16], ids=["block32", "block16"])
@pytest.mark.parametrize("varying", [False, True], ids=["uniform", "varying"])
def test_scaled_wmma_matches_reference(block_size, varying):
    """D must equal (A*2^sa) @ (B*2^sb).T with the scales the atom carries."""
    nblk = K // block_size
    torch.manual_seed(1323)

    a = (torch.randn(M, K) * 3).round().clamp(-8, 8)
    b = (torch.randn(N, K) * 3).round().clamp(-8, 8)
    a8 = a.to(torch.float8_e4m3fn).cuda()
    b8 = b.to(torch.float8_e4m3fn).cuda()

    # One exponent per K block. Every row shares the word, which is what a single
    # scale operand can express; that is enough to separate the blocks.
    if varying:
        exps_a = [127 + (j % 3) - 1 for j in range(nblk)]
        exps_b = [127 - (j % 3) + 1 for j in range(nblk)]
    else:
        exps_a = [127] * nblk
        exps_b = [127] * nblk

    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    launch = _compile(block_size, _e8m0_bytes(exps_a), _e8m0_bytes(exps_b))
    launch(a8, b8, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    sa = torch.tensor([2.0 ** (e - 127) for e in exps_a]).repeat_interleave(block_size)
    sb = torch.tensor([2.0 ** (e - 127) for e in exps_b]).repeat_interleave(block_size)
    ref = (a8.float().cpu() * sa) @ (b8.float().cpu() * sb).T

    torch.testing.assert_close(c.cpu(), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
