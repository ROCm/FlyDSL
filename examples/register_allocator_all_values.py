# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""RegisterAllocator experiment covering every named runtime numeric value.

The single-element case places every named runtime numeric value at an exact
register number. The indexed case processes full 64-element blocks and computes
    out[i] = (((x[i] + y[i]) ^ mask) - i) ^ bias.
The indexed case supports automatic numbering; fixed numbering deliberately
exercises the current rejection of tuples widened by LLVM coalescing.
Each struct describes Numeric fields and is allocated as a whole by one
register allocator. Plain mode uses RegisterAllocator() without placement
constraints.
Tensor handles, ABI pointers, immediates and compiler-generated temporaries are
outside the register-memory placement interface. No instruction encodings or
optimizer switches are changed by this experiment.
"""

import argparse
import resource

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_SIZE = 64


@fx.struct
class IndexedScalarRegisters:
    block: fx.Int32
    base: fx.Int32
    mask: fx.Int32
    bias: fx.Int32


@fx.struct
class IndexedVectorRegisters:
    lane: fx.Int32
    index: fx.Int32
    x: fx.Int32
    y: fx.Int32
    total: fx.Int32
    scrambled: fx.Int32
    delta: fx.Int32
    result: fx.Int32


@flyc.kernel
def all_values_kernel(
    X: fx.Tensor,
    Y: fx.Tensor,
    Out: fx.Tensor,
    Mask: fx.Int32,
    Bias: fx.Int32,
    mode: fx.Constexpr[str],
):
    if fx.const_expr(mode == "plain"):
        scalar = fx.RegisterAllocator()
        vector = fx.RegisterAllocator()
    else:
        scalar = fx.RegisterAllocator(fx.rocdl.SGPR, start_offset=None if mode == "auto" else 40)
        vector = fx.RegisterAllocator(fx.rocdl.VGPR, start_offset=None if mode == "auto" else 32)

    sreg = scalar.allocate(IndexedScalarRegisters)
    vreg = vector.allocate(IndexedVectorRegisters)

    sreg.block.poke(fx.block_idx.x)
    sreg.base.poke(sreg.block.peek() * BLOCK_SIZE)
    sreg.mask.poke(Mask)
    sreg.bias.poke(Bias)

    vreg.lane.poke(fx.thread_idx.x)
    vreg.index.poke(sreg.base.peek() + vreg.lane.peek())
    vreg.x.poke(X[vreg.index.peek()])
    vreg.y.poke(Y[vreg.index.peek()])
    vreg.total.poke(vreg.x.peek() + vreg.y.peek())
    vreg.scrambled.poke(vreg.total.peek() ^ sreg.mask.peek())
    vreg.delta.poke(vreg.scrambled.peek() - vreg.index.peek())
    vreg.result.poke(vreg.delta.peek() ^ sreg.bias.peek())
    Out[vreg.index.peek()] = vreg.result.peek()


@fx.struct
class SingleScalarRegisters:
    block: fx.Int32
    mask: fx.Int32
    bias: fx.Int32


@fx.struct
class SingleVectorRegisters:
    x: fx.Int32
    y: fx.Int32
    total: fx.Int32
    scrambled: fx.Int32
    delta: fx.Int32
    result: fx.Int32


@flyc.kernel
def single_value_kernel(X: fx.Tensor, Y: fx.Tensor, Out: fx.Tensor, mode: fx.Constexpr[str]):
    if fx.const_expr(mode == "plain"):
        scalar = fx.RegisterAllocator()
        vector = fx.RegisterAllocator()
    else:
        scalar = fx.RegisterAllocator(fx.rocdl.SGPR, start_offset=None if mode == "auto" else 40)
        vector = fx.RegisterAllocator(fx.rocdl.VGPR, start_offset=None if mode == "auto" else 32)

    sreg = scalar.allocate(SingleScalarRegisters)
    vreg = vector.allocate(SingleVectorRegisters)

    sreg.block.poke(fx.Int32(fx.block_idx.x))
    sreg.mask.poke(sreg.block.peek() + 0x1357)
    sreg.bias.poke(sreg.block.peek() ^ 0x2468)

    vreg.x.poke(X[0])
    vreg.y.poke(Y[0])

    vreg.total.poke(vreg.x.peek() + vreg.y.peek())
    vreg.scrambled.poke(vreg.total.peek() ^ sreg.mask.peek())
    vreg.delta.poke(vreg.scrambled.peek() - vreg.x.peek())
    vreg.result.poke(vreg.delta.peek() ^ sreg.bias.peek())
    Out[0] = vreg.result.peek()


@flyc.jit
def launch_single(X: fx.Tensor, Y: fx.Tensor, Out: fx.Tensor, mode: fx.Constexpr[str]):
    single_value_kernel(X, Y, Out, mode).launch(grid=(1, 1, 1), block=(1, 1, 1))


@flyc.jit
def launch(
    X: fx.Tensor,
    Y: fx.Tensor,
    Out: fx.Tensor,
    mask: fx.Int32,
    bias: fx.Int32,
    blocks: fx.Constexpr[int],
    mode: fx.Constexpr[str],
):
    all_values_kernel(X, Y, Out, mask, bias, mode).launch(grid=(blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


def main():
    from flydsl.utils import env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "auto", "plain"], default="fixed")
    parser.add_argument("--case", choices=["single", "indexed"], default="single")
    parser.add_argument("--blocks", type=int, default=7)
    args = parser.parse_args()
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    device = "cpu" if env.compile.compile_only else "cuda"
    count = 1 if args.case == "single" else args.blocks * BLOCK_SIZE
    index = torch.arange(count, dtype=torch.int32, device=device)
    x = index * 17 - 1000
    y = index * -9 + 31
    out = torch.empty_like(x)
    mask, bias = 0x1357, 0x2468
    if args.case == "single":
        launch_single(flyc.from_torch_tensor(x), flyc.from_torch_tensor(y), flyc.from_torch_tensor(out), args.mode)
    else:
        launch(
            flyc.from_torch_tensor(x),
            flyc.from_torch_tensor(y),
            flyc.from_torch_tensor(out),
            fx.Int32(mask),
            fx.Int32(bias),
            args.blocks,
            args.mode,
        )
    if env.compile.compile_only:
        print(f"Compiled {args.mode} mode")
        return
    torch.cuda.synchronize()
    expected = (((x + y) ^ mask) - (x if args.case == "single" else index)) ^ bias
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    print(f"{args.mode} {args.case}: {out.numel()} values correct")
    print("First 8 outputs:", out[:8].tolist())


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    main()
