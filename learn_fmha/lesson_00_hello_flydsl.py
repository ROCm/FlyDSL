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
  2. There are TWO buffer styles. The high-level layout/copy-atom API (see
     examples/01-vectorAdd.py: make_buffer_tensor + copy_atom) hides memory movement.
     This tutorial uses the LOW-LEVEL raw style (create_buffer_resource + explicit
     buffer_load/store by index) because every optimization we teach is ABOUT explicit
     memory movement — the high-level API would hide exactly what we want to see.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_00_hello_flydsl.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

N = 256


@flyc.kernel(known_block_size=[N, 1, 1])  # tells the compiler the exact block size (helps codegen)
def add_one_kernel(X: fx.Tensor, Y: fx.Tensor):
    tid = fx.Int32(fx.thread_idx.x)
    rX = fx.buffer_ops.create_buffer_resource(X)
    rY = fx.buffer_ops.create_buffer_resource(Y)
    # load element `tid`, add 1.0, store it back. buffer_load/store index by ELEMENT, not byte.
    x = fx.buffer_ops.buffer_load(rX, tid, vec_width=1, dtype=fx.Float32)
    y = fx.Float32(x) + fx.Float32(1.0)
    fx.buffer_ops.buffer_store(y.ir_value(), rY, tid.ir_value())


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
