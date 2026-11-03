# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit register placement with the experimental register_allocation feature.

Experimental features are imported by their subpackage rather than reached through
``fx``, so every use is visibly experimental:

    from flydsl.experimental import register_allocation as fx_regalloc

The feature has two entry points:

* ``fx_regalloc.RegisterAllocator`` allocates new register memory in a register class,
  with the same ``allocate(T)`` / ``Storage[T]`` interface as ``SharedAllocator``.
* ``fx_regalloc.register_place`` places register memory created elsewhere, such as
  MMA fragments.

Register classes are symbolic LLVM classes; ``fx_regalloc.rocdl`` provides the AMDGPU
ones (``SGPR32``, ``VGPR32``, ``AGPR32``), numbered in 32-bit registers. Both entry
points take ``start_idx``, the number of the first register. Placement is enforced
during codegen: a request that the generated code cannot honor fails compilation
instead of being dropped.

The comments after each kernel show its gfx942 ISA. To inspect it yourself, run with
``FLYDSL_DUMP_IR=1`` and
``FLYDSL_RUNTIME_ENABLE_CACHE=0`` and open ``<kernel>/*_final_isa.s`` under
``FLYDSL_DUMP_DIR`` (default ``~/.flydsl/debug``).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.experimental import register_allocation as fx_regalloc

BLOCK_SIZE = 64
BLOCKS = 4
TILE_M, TILE_N, TILE_K = 64, 64, 8


# One struct per register class: an allocation is a single block of register memory,
# so values that belong in different classes go into different structs.
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
    # start_idx is the first register number. Each allocator hands out consecutive
    # numbers, padded to whole registers.
    sgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.SGPR32, start_idx=40)
    vgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=32)
    block = sgprs.allocate(BlockRegisters)  # s40-s41
    lane = vgprs.allocate(LaneRegisters)  # v32-v34

    X = fx.rocdl.make_buffer_tensor(X)
    Y = fx.rocdl.make_buffer_tensor(Y)
    Out = fx.rocdl.make_buffer_tensor(Out)

    # Fields are read and written through peek()/poke(), like any other Storage.
    block.base.poke(fx.Int32(fx.block_idx.x) * BLOCK_SIZE)
    block.bias.poke(bias)
    index = block.base.peek() + fx.Int32(fx.thread_idx.x)  # not placed: LLVM chooses freely

    lane.x.poke(X[index])
    lane.y.poke(Y[index])
    # The instruction defining a fixed value must write exactly its registers. An integer
    # multiply-add here would become v_mad_u64_u32, whose 64-bit result and addend do not
    # fit one 32-bit register, so the placement would be rejected at compile time.
    lane.result.poke(lane.x.peek() + lane.y.peek() + block.bias.peek())
    Out[index] = lane.result.peek()


# add_kernel on gfx942 (abridged):
#
#   s_load_dword       s41, s[4:5], 0x2c             ; block.bias  = bias
#   s_lshl_b32         s40, s8, 6                    ; block.base  = block_idx.x * 64
#   v_or_b32_e32       v0, s40, v0                   ; index       (not placed, LLVM's choice)
#   buffer_load_dword  v32, v0, s[0:3], 0 offen      ; lane.x
#   buffer_load_dword  v33, v0, s[12:15], 0 offen    ; lane.y
#   v_add3_u32         v34, v33, v32, s41            ; lane.result
#   buffer_store_dword v34, v0, s[16:19], 0 offen
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

    # Fragments are register memory that the MMA API creates, so they are placed with
    # register_place rather than allocated. Call it on the fragment itself (not a view
    # or slice of it), in the block that creates it; it covers the whole fragment.
    frag_A = thr_mma.make_fragment_A(A)  # 4 x f32 per lane
    frag_B = thr_mma.make_fragment_B(B)  # 4 x f32 per lane
    frag_C = thr_mma.make_fragment_C(C)  # 16 x f32 per lane

    # MFMA reads A/B from AGPRs and accumulates C in VGPRs, so no copies are needed.
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
#   buffer_load_dword      a0, v1, s[12:15], 0 offen     ; frag_A -> a0-a3
#   buffer_load_dword      a4, v3, s[16:19], 0 offen     ; frag_B -> a4-a7
#   ...
#   v_mfma_f32_16x16x4_f32 v[64:67], a0, a4, 0           ; frag_C -> v64-v79
#   v_mfma_f32_16x16x4_f32 v[68:71], a0, a6, 0
#   ...
#   v_mfma_f32_16x16x4_f32 v[64:67], a1, a5, v[64:67]
#   ...
#   buffer_store_dword     v64, v1, s[0:3], 0 offen
#
# The loads write A/B straight into AGPRs, MFMA reads them there and accumulates in the
# placed VGPRs; no v_accvgpr_read/write moves appear. The kernel uses 80 VGPRs and 8 AGPRs.


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
