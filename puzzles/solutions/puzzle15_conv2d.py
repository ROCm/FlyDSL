# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Puzzle 15 (solution) - 2D convolution as an implicit GEMM (out = conv2d(x, w)).

Convolution is a GEMM in disguise. With x: (N, C, H, W) and w: (K, C, R, S),
stride 1 and no padding, the output o: (N, K, Ho, Wo) satisfies

    o[n, k, ho, wo] = sum_{c, r, s} x[n, c, ho+r, wo+s] * w[k, c, r, s].

Read the right-hand side as a dot product over a flattened reduction axis of
length C*R*S: that is exactly one entry of a GEMM whose

    M      = K                (output channels)          <- weight rows
    N_gemm = N * Ho * Wo      (output pixels)             <- im2col columns
    K_gemm = C * R * S        (patch elements)            <- shared reduction

Rather than materialize the im2col matrix, this kernel gathers each patch element
on the fly (the "implicit" GEMM): one thread owns one output pixel (k, ho, wo),
streams the C*R*S patch, and accumulates the dot product in a register. Because
stride=1/padding=0 and the grid tiles K*Ho*Wo exactly, every gather is in bounds.

Concepts: im2col / implicit GEMM, gather-on-the-fly reduction, coord decode.
Book: Chapter 7; reference kernels/conv/.
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

        # Decode the flat output id into (k, ho, wo) via the output layout.
        o_layout = fx.make_layout((K, Ho, Wo), (Ho * Wo, Wo, 1))
        crd = fx.idx2crd(gid, o_layout)
        k = fx.get(crd, 0)
        ho = fx.get(crd, 1)
        wo = fx.get(crd, 2)

        # Logical row-major views (N=1 folds away for X and O). Build them on the
        # raw tensor's iterator: UniversalCopy needs a plain pointer, so do NOT wrap
        # the tensor in make_buffer_tensor here.
        Xv = fx.make_view(fx.get_iter(X), fx.make_layout((C, H, W), (H * W, W, 1)))
        Wv = fx.make_view(fx.get_iter(Wt), fx.make_layout((K, C, R, S), (C * R * S, R * S, S, 1)))
        Ov = fx.make_view(fx.get_iter(O), o_layout)

        copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        # Gather one scalar at `coord`. A fully-concrete slice would collapse to an
        # empty coordinate tuple (unsupported), so keep the last mode as `None`,
        # then re-tile it to 1 element and index it with (None, last).
        def gather(view, coord):
            *lead, last = coord
            row = fx.slice(view, tuple(lead) + (None,))
            row1 = fx.logical_divide(row, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy, fx.slice(row1, (None, last)), r)
            return fx.Vector(fx.memref_load_vec(r))[0]

        # Implicit-GEMM dot product over the flattened patch (C*R*S).
        acc = fx.Float32(0.0)
        for c in range_constexpr(C):
            for r in range_constexpr(R):
                for s in range_constexpr(S):
                    acc = acc + gather(Xv, (c, ho + r, wo + s)) * gather(Wv, (k, c, r, s))

        o_row = fx.slice(Ov, (k, ho, None))
        o_row1 = fx.logical_divide(o_row, fx.make_layout(1, 1))
        rO = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([acc], fx.Float32), rO)
        fx.copy_atom_call(copy, rO, fx.slice(o_row1, (None, wo)))

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
