# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 15 - 2D convolution as an implicit GEMM (out = conv2d(x, w)).

GOAL: see convolution *as a GEMM*. With x: (N, C, H, W) and w: (K, C, R, S),
stride 1 and no padding, the output o: (N, K, Ho, Wo) satisfies

    o[n, k, ho, wo] = sum_{c, r, s} x[n, c, ho+r, wo+s] * w[k, c, r, s].

The right-hand side is a dot product over a flattened reduction axis of length
C*R*S - exactly one entry of a GEMM whose

    M      = K                (output channels)          <- weight rows
    N_gemm = N * Ho * Wo      (output pixels)             <- im2col columns
    K_gemm = C * R * S        (patch elements)            <- shared reduction

Instead of materializing the im2col matrix, gather each patch element on the fly
(the "implicit" GEMM): one thread owns one output pixel (k, ho, wo), streams the
C*R*S patch, and accumulates the dot product in a register.

CONCEPTS: im2col / implicit GEMM, gather-on-the-fly reduction, coord decode.
READ: FlyDSL for Dummies - Chapter 7; reference kernels/conv/.
HINT: decode the flat thread id gid -> (k, ho, wo) with fx.idx2crd on the output
layout (K, Ho, Wo):(Ho*Wo, Wo, 1). Make row-major views of X/W/O, then use a
range_constexpr triple loop over (c, r, s) accumulating x[c,ho+r,wo+s]*w[k,c,r,s].
Grid tiles K*Ho*Wo exactly (stride=1/pad=0), so every access is in bounds.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr


def build(N: int, C: int, H: int, W: int, K: int, R: int, S: int):
    assert N == 1, "this teaching kernel handles a single image (N=1)"
    Ho = H - R + 1
    Wo = W - S + 1
    total = K * Ho * Wo
    block = 256
    while total % block != 0:
        block //= 2
    grid = total // block

    @flyc.kernel
    def conv_kernel(X: fx.Tensor, Wt: fx.Tensor, O: fx.Tensor):
        gid = fx.block_idx.x * block + fx.thread_idx.x

        # ==== YOUR CODE HERE ====
        # 1. Decode gid -> (k, ho, wo):
        #      o_layout = fx.make_layout((K, Ho, Wo), (Ho*Wo, Wo, 1))
        #      crd = fx.idx2crd(gid, o_layout); k/ho/wo = fx.get(crd, 0/1/2)
        # 2. Row-major views (N=1 folds away for X and O):
        #      Xv over (C, H, W):(H*W, W, 1); Wv over (K, C, R, S):(C*R*S, R*S, S, 1); Ov over o_layout.
        #      Use fx.make_view(fx.get_iter(T), layout) on the RAW tensor T (X/Wt/O).
        #      Do NOT wrap T in make_buffer_tensor: UniversalCopy needs a plain pointer.
        # 3. copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32); acc = fx.Float32(0.0).
        # 4. Gather one scalar at coord (..., last): a fully-concrete slice collapses to an
        #      empty coord tuple (unsupported), so keep the last mode as None, re-tile to 1,
        #      then index it:
        #        row = fx.slice(view, (..., None)); row1 = fx.logical_divide(row, fx.make_layout(1, 1))
        #        r = fx.make_rmem_tensor(1, fx.Float32); fx.copy_atom_call(copy, fx.slice(row1, (None, last)), r)
        #        val = fx.Vector(fx.memref_load_vec(r))[0]
        # 5. Triple range_constexpr loop over (c, r, s): acc += gather(Xv, (c, ho+r, wo+s)) * gather(Wv, (k, c, r, s)).
        # 6. Store acc to Ov at (k, ho, wo) the same way (row-slice + logical_divide + copy_atom_call).
        raise NotImplementedError("Fill in conv_kernel")

    @flyc.jit
    def conv(X: fx.Tensor, Wt: fx.Tensor, O: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv_kernel(X, Wt, O).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return conv


def run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n, c, h, ww = x.shape
    k, _, r, s = w.shape
    ho, wo = h - r + 1, ww - s + 1
    o = torch.zeros(n, k, ho, wo, dtype=torch.float32, device=x.device)
    build(n, c, h, ww, k, r, s)(x, w, o, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return o


if __name__ == "__main__":
    from puzzles import common

    x = torch.randn(1, 8, 16, 16, dtype=torch.float32).cuda()
    w = torch.randn(16, 8, 3, 3, dtype=torch.float32).cuda()
    common.check(run(x, w), common.ref_conv2d(x, w), name="puzzle15 conv2d", atol=5e-2, rtol=5e-2)
