#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx


def test_control_flow_kernel_snippet_compiles_without_error():
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

    assert vecAbsKernel is not None
