# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit register placement with ``fx.experimental.register_allocation``.

``add_kernel`` allocates placed register memory with ``RegisterAllocator``;
``mma_kernel`` places MMA fragments with ``register_place``. The API and its
limitations are described in docs/experimental/register_allocation.md.

The comments after each kernel show its gfx942 ISA, including where the hints are
not met. To inspect it yourself, run with ``FLYDSL_DUMP_IR=1`` and
``FLYDSL_RUNTIME_ENABLE_CACHE=0`` and open ``<kernel>/*_final_isa.s`` under
``FLYDSL_DUMP_DIR`` (default ``~/.flydsl/debug``).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

fx_regalloc = fx.experimental.register_allocation

BLOCK_SIZE = 64
BLOCKS = 4
TILE_M, TILE_N, TILE_K = 64, 64, 8


# One struct per register class: each allocation is placed in a single class.
@fx.struct
class BlockRegisters:  # uniform across the block -> SGPRs
    base: fx.Int32
    bias: fx.Int32


@fx.struct
class LaneRegisters:  # one value per lane -> VGPRs
    x: fx.Int32
    y: fx.Int32
    result: fx.Int32


@flyc.kernel
def add_kernel(
    X: fx.Tensor,
    Y: fx.Tensor,
    Out: fx.Tensor,
    bias: fx.Int32,
):
    sgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.SGPR32, start_idx=40)
    vgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=32)
    block = sgprs.allocate(BlockRegisters)  # s40-s41
    lane = vgprs.allocate(LaneRegisters)  # v32-v34

    X = fx.rocdl.make_buffer_tensor(X)
    Y = fx.rocdl.make_buffer_tensor(Y)
    Out = fx.rocdl.make_buffer_tensor(Out)

    block.base.poke(fx.Int32(fx.block_idx.x) * BLOCK_SIZE)
    block.bias.poke(bias)
    index = block.base.peek() + fx.Int32(fx.thread_idx.x)  # not placed: LLVM chooses freely

    lane.x.poke(X[index])
    lane.y.poke(Y[index])
    lane.result.poke(lane.x.peek() + lane.y.peek() + block.bias.peek())
    Out[index] = lane.result.peek()


# add_kernel on gfx942 (abridged):
#
#   s_load_dword       s41, s[0:1], 0x2c             ; block.bias  = bias
#   s_lshl_b32         s40, s2, 6                    ; block.base  = block_idx.x * 64
#   v_add_lshl_u32     v0, s40, v0, 2                ; index * 4   (not placed, LLVM's choice)
#   buffer_load_dword  v32, v0, s[4:7], 0 offen      ; lane.x
#   buffer_load_dword  v33, v0, s[8:11], 0 offen     ; lane.y
#   v_add3_u32         v34, v32, s41, v33            ; lane.result
#   buffer_store_dword v34, v0, s[12:15], 0 offen
#
# Each placed value is written straight into its register; placement adds no moves.
# Fixed numbers set a floor on the register count: the kernel uses 35 VGPRs and 42 SGPRs.


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
    fx_regalloc.register_place(frag_A, fx_regalloc.rocdl.AGPR32, start_idx=0)  # a0-a3
    fx_regalloc.register_place(frag_B, fx_regalloc.rocdl.AGPR32, start_idx=4)  # a4-a7
    fx_regalloc.register_place(frag_C, fx_regalloc.rocdl.VGPR32, start_idx=64)  # v64-v79

    fx.copy(copy_atom, thr_copy_A.partition_S(A), thr_copy_A.retile(frag_A), pred=None)
    fx.copy(copy_atom, thr_copy_B.partition_S(B), thr_copy_B.retile(frag_B), pred=None)
    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(C), pred=None)


# mma_kernel on gfx942 (abridged):
#
#   buffer_load_dword      a0, v1, s[8:11], 0 offen      ; frag_A[0] -> a0, as requested
#   buffer_load_dword      v4, v5, s[12:15], 0 offen     ; frag_B[0] -> v4, not a4
#   ...
#   v_mov_b64_e32          v[76:77], s[16:17]            ; frag_C.fill(0) kept in registers
#   ...
#   v_mfma_f32_16x16x4_f32 v[64:67], a0, v4, v[76:79]    ; first products in v64-v79
#   v_mfma_f32_16x16x4_f32 v[68:71], a0, v6, v[76:79]
#   ...
#   v_mfma_f32_16x16x4_f32 v[12:15], v1, v5, v[64:67]    ; final frag_C outside v64-v79
#   v_mov_b64_e32          v[16:17], v[14:15]
#   ...
#   buffer_store_dword     v14, v1, s[4:7], 0 offen
#
# The hints are only partly met. The AGPR requests (except a0) are given the VGPRs with the
# same numbers, the zero accumulator is materialized instead of folded into MFMA, and the
# element-wise reads of frag_C override the hint of the whole accumulator, so the final
# results land elsewhere and need copies. The kernel uses 80 VGPRs and 1 AGPR.


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
