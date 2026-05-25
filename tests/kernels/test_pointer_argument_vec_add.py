#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Verify raw global pointer arguments in JIT launchers."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU pointer argument test.", allow_module_level=True)


@flyc.kernel
def pointer_vec_add_kernel(
    A: fx.Pointer,
    B: fx.Pointer,
    C: fx.Pointer,
    n: fx.Int32,
):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        a_val = fx.ptr_load(A + idx)
        b_val = fx.ptr_load(B + idx)
        fx.ptr_store(a_val + b_val, C + idx)


@flyc.jit
def pointer_vec_add(
    A: fx.Pointer,
    B: fx.Pointer,
    C: fx.Pointer,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 256
    grid_x = (n + block_dim - 1) // block_dim
    pointer_vec_add_kernel(A, B, C, n).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


def test_pointer_argument_vector_add():
    size = 4099
    a_dev = torch.randn(size, device="cuda", dtype=torch.float32)
    b_dev = torch.randn(size, device="cuda", dtype=torch.float32)
    c_dev = torch.empty_like(a_dev)
    stream = torch.cuda.Stream()

    pointer_vec_add(
        flyc.from_c_void_p(fx.Float32, a_dev.data_ptr()),
        flyc.from_c_void_p(fx.Float32, b_dev.data_ptr()),
        flyc.from_c_void_p(fx.Float32, c_dev.data_ptr()),
        size,
        stream=stream,
    )
    torch.cuda.synchronize()

    error = (c_dev - (a_dev + b_dev)).abs().max().item()
    assert error < 1e-5
