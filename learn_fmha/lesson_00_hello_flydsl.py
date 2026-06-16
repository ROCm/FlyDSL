# SPDX-License-Identifier: Apache-2.0
"""Lesson 00 — Hello FlyDSL: the kernel + launch model.

Goal: the absolute minimum — write a device kernel, launch it, get data back. We add
1.0 to every element of a vector. No matrix math yet; this is purely "how do I express
a kernel and run it" so the later lessons can focus on the GPU ideas.

If you know CK Tile:
  - `@flyc.kernel`  ~= a `__global__` kernel (the device function).
  - `@flyc.jit`     ~= the host-side launcher; it JIT-compiles on first call and caches.
  - `.launch(grid=, block=)` ~= the `<<<grid, block>>>` triple-chevron / kernel_launch.
  - `fx.thread_idx.x` / `fx.block_idx.x` ~= threadIdx / blockIdx.
  - `fx.Int32` / `fx.Index` are typed scalars in the DSL's expression IR (you build an
    expression graph that gets lowered to MLIR, not Python that runs on device).

Two FlyDSL-specific facts to remember:
  1. NEVER name the @flyc.jit wrapper `launch` — it shadows the `.launch()` method and
     infinite-recurses. We always call it `run_*`.
  2. We write in the IDIOMATIC layout-algebra style from the start (see
     examples/01-vectorAdd.py). Instead of a raw buffer resource + index arithmetic, we:
       a. wrap the global tensor in a buffer descriptor: `fx.rocdl.make_buffer_tensor`,
       b. carve it into per-block then per-thread tiles with `logical_divide` + `slice`,
       c. move data with a typed COPY ATOM (`make_copy_atom` + `copy_atom_call`) instead
          of hand-written `buffer_load`/`buffer_store`.
     The layout describes WHICH element a thread owns; the copy atom describes HOW the
     bytes move. Later lessons add the MMA atom (the matrix-instruction analogue of the
     copy atom). This is Recipe A ("elementwise / 1-D") from the layout-algebra skill.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_00_hello_flydsl.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

N = 256


@flyc.kernel(known_block_size=[N, 1, 1])  # tells the compiler the exact block size (helps codegen)
def add_one_kernel(X: fx.Tensor, Y: fx.Tensor):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # 1. wrap the globals in AMD buffer descriptors (the layout-algebra entry point).
    X = fx.rocdl.make_buffer_tensor(X)
    Y = fx.rocdl.make_buffer_tensor(Y)

    # 2. split each vector into block tiles, pick this block's tile, then expose one
    #    element per thread. The two-stage logical_divide keeps the coordinate rank lined
    #    up so the trailing `slice(..., (None, tid))` selects exactly this lane's element.
    tX = fx.logical_divide(X, fx.make_layout(N, 1))
    tY = fx.logical_divide(Y, fx.make_layout(N, 1))
    tX = fx.slice(tX, (None, bid))
    tY = fx.slice(tY, (None, bid))
    tX = fx.logical_divide(tX, fx.make_layout(1, 1))
    tY = fx.logical_divide(tY, fx.make_layout(1, 1))

    # 3. a 32-bit buffer copy atom moves one f32 between global and a register tensor.
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    rX = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

    fx.copy_atom_call(copy_atom, fx.slice(tX, (None, tid)), rX)  # load element `tid`
    y = fx.Float32(fx.memref_load_vec(rX)) + fx.Float32(1.0)     # add 1.0 in registers
    fx.memref_store_vec(y, rX)
    fx.copy_atom_call(copy_atom, rX, fx.slice(tY, (None, tid)))  # store it back


@flyc.jit
def run_add_one(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    add_one_kernel(X, Y).launch(grid=(1, 1, 1), block=(N, 1, 1), stream=stream)


if __name__ == "__main__":
    X = torch.arange(N, dtype=torch.float32).cuda()
    Y = torch.zeros(N, dtype=torch.float32).cuda()
    run_add_one(X, Y, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    err = (Y - (X + 1.0)).abs().max().item()
    print(f"max abs err = {err:.6f}  ->  {'PASS' if err < 1e-6 else 'FAIL'}")
