# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit register placement with ``flydsl.experimental.register_allocation``.

``add_kernel`` places values by indexing a register class (``reg.rocdl.VGPR32[n](value)``);
``mma_kernel`` places MMA fragments with ``register_place``. The API are described in
docs/experimental/register_allocation.md.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.experimental import register_allocation as reg

BLOCK_SIZE = 64
BLOCKS = 4
TILE_M, TILE_N, TILE_K = 64, 64, 8


@flyc.kernel
def add_kernel(
    X: fx.Tensor,
    Y: fx.Tensor,
    Out: fx.Tensor,
    bias: fx.Int32,
):
    X = fx.rocdl.make_buffer_tensor(X)
    Y = fx.rocdl.make_buffer_tensor(Y)
    Out = fx.rocdl.make_buffer_tensor(Out)

    # Uniform across the block -> SGPRs.
    base = reg.rocdl.SGPR32[40](fx.Int32(fx.block_idx.x) * BLOCK_SIZE)
    bias = reg.rocdl.SGPR32[41](bias)
    index = base + fx.Int32(fx.thread_idx.x)  # not placed: LLVM chooses freely

    # One value per lane -> VGPRs.
    x = reg.rocdl.VGPR32[32](X[index])
    y = reg.rocdl.VGPR32[33](Y[index])
    Out[index] = reg.rocdl.VGPR32[34](x + y + bias)


# add_kernel on gfx942 (abridged):
#
#   s_load_dword       s41, s[0:1], 0x2c             ; bias -> s41
#   s_lshl_b32         s40, s2, 6                    ; base -> s40
#   v_add_lshl_u32     v0, s40, v0, 2                ; index * 4 (not placed, LLVM's choice)
#   buffer_load_dword  v32, v0, s[4:7], 0 offen      ; x -> v32
#   buffer_load_dword  v33, v0, s[8:11], 0 offen     ; y -> v33
#   v_add3_u32         v34, v32, s41, v33            ; x + y + bias -> v34
#   buffer_store_dword v34, v0, s[12:15], 0 offen
#
# Every placed value is written straight into its register; placement adds no moves. Fixed
# numbers set a floor on the register count: the kernel uses 35 VGPRs and 42 SGPRs.


@flyc.kernel
def mma_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.slice(fx.zipped_divide(fx.rocdl.make_buffer_tensor(A), (TILE_M, TILE_K)), (None, bid))
    B = fx.slice(fx.zipped_divide(fx.rocdl.make_buffer_tensor(B), (TILE_N, TILE_K)), (None, bid))
    C = fx.slice(fx.zipped_divide(fx.rocdl.make_buffer_tensor(C), (TILE_M, TILE_N)), (None, bid))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
    thr_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(tid)

    # The MMA API creates the fragments, so they are placed rather than allocated.
    frag_A = thr_mma.make_fragment_A(A)  # 4 x f32 per lane
    frag_B = thr_mma.make_fragment_B(B)  # 4 x f32 per lane
    frag_C = thr_mma.make_fragment_C(C)  # 16 x f32 per lane

    # Request A/B in AGPRs, which MFMA reads directly, and C in VGPRs.
    reg.register_place(frag_A, reg.rocdl.AGPR32, start_idx=0)  # a0-a3
    reg.register_place(frag_B, reg.rocdl.AGPR32, start_idx=4)  # a4-a7
    reg.register_place(frag_C, reg.rocdl.VGPR32, start_idx=64)  # v64-v79

    fx.copy(copy_atom, thr_copy_A.partition_S(A), thr_copy_A.retile(frag_A), pred=None)
    fx.copy(copy_atom, thr_copy_B.partition_S(B), thr_copy_B.retile(frag_B), pred=None)
    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(C), pred=None)


# mma_kernel on gfx942 (abridged); "!" marks where the result falls short of the request:
#
#   buffer_load_dword      a0, v1, s[8:11], 0 offen      ; frag_A[0] -> a0
#   buffer_load_dword      a4, v3, s[12:15], 0 offen     ; frag_B[0] -> a4
#   ...                                                  ; the rest of frag_A/B -> a1-a3, a5-a7
#   v_mov_b64_e32          v[64:65], s[16:17]            ; ! frag_C.fill(0): zeros moved into
#   ...                                                  ;   v64-v79, not folded into the MFMAs
#   v_mfma_f32_16x16x4_f32 v[64:67], a0, a4, v[64:67]    ; first MFMA accumulates in place
#   v_mfma_f32_16x16x4_f32 v[8:11], a1, a5, v[64:67]     ; ! the last one writes v8-v11, not v64-v67
#   ...
#   v_mov_b64_e32          v[16:17], v[8:9]              ; ! the result is copied around ...
#   v_mov_b64_e32          v[2:3], v[16:17]
#   v_mov_b32_e32          v64, v2                       ; ! ... and back into v64, element by element
#   buffer_store_dword     v64, v1, s[4:7], 0 offen
#
# frag_A/B land in a0-a7 and frag_C starts in v64-v79, as requested. The shortfalls all come from
# frag_C. The zero accumulator is materialized in registers. The copy to C reads frag_C one element
# at a time and each of those reads is placed on its own, so the last MFMAs write other registers
# and every element is copied back into v64-v79 before it is stored ("Placing part of a placed
# value"). The kernel uses 80 VGPRs and 8 AGPRs.


@flyc.jit
def add(X: fx.Tensor, Y: fx.Tensor, Out: fx.Tensor, bias: fx.Int32):
    add_kernel(X, Y, Out, bias).launch(grid=(BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@flyc.jit
def mma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    mma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1))


if __name__ == "__main__":
    x = torch.arange(BLOCKS * BLOCK_SIZE, dtype=torch.int32, device="cuda")
    y = 3 - 2 * x
    out = torch.empty_like(x)
    add(x, y, out, fx.Int32(7))

    a = torch.randn(TILE_M, TILE_K, dtype=torch.float32, device="cuda")
    b = torch.randn(TILE_N, TILE_K, dtype=torch.float32, device="cuda")
    c = torch.empty(TILE_M, TILE_N, dtype=torch.float32, device="cuda")
    mma(a, b, c)

    torch.cuda.synchronize()
    print("Result correct:", torch.equal(out.cpu(), (x + y + 7).cpu()))
    print("Result correct:", torch.allclose(c.cpu(), (a @ b.T).cpu(), atol=1e-5, rtol=1e-5))
