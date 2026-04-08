#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch


def test_control_flow_kernel_snippet_compiles_without_error(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for control-flow compile coverage test")

    @flyc.kernel
    def vecAbsKernel(
        A: fx.Tensor,
        C: fx.Tensor,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        print_debug: fx.Constexpr[bool] = True,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        if print_debug and bid == 0 and tid <= 2:
            fx.printf("[kernel] bid={}, tid={}", bid, tid)

    @flyc.jit
    def vecAbs(
        A: fx.Tensor,
        C,
        n: fx.Int32,
        const_n: fx.Constexpr[int],
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile_elems = block_dim * vec_width
        grid_x = (n + tile_elems - 1) // tile_elems
        vecAbsKernel(A, C, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    monkeypatch.setenv("FLYDSL_COMPILE_ONLY", "1")
    threads = 64
    vec = 4
    size = threads * vec
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    t_a = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=vec)
    vecAbs(t_a, c, size, size, threads, vec)
