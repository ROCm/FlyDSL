#!/usr/bin/env python3
"""Shared memory matmul - correctness smoke test.

This used to execute at import time; it is now a proper pytest test and is
skipped when HIP isn't available.
"""

import time

import numpy as np
import pytest
import _mlir.extras.types as T

import pyflir
import torch

from pyflir.dialects.ext import arith, flir
from pyflir.runtime.device import get_rocm_arch
from pyflir.utils import SmemAllocator

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


M, N, K = 256, 256, 256
TILE_SIZE = 16


def test_matmul_shared_working():
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _MatmulShared(flir.MlirModule):
        GPU_MODULE_NAME = "matmul_shared"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            # Allocate enough space for TILE_SIZE x TILE_SIZE float32 elements.
            _state["s_a_decl"] = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
            _state["s_b_decl"] = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
            allocator.finalize()

        @flir.kernel
        def matmul_shared(
            self: flir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            gpu = flir.gpu_ext

            # Get references to shared memory using Allocator
            base_ptr = allocator.get_base()
            As = _state["s_a_decl"](base_ptr)
            Bs = _state["s_b_decl"](base_ptr)

            tile_c = arith.index(TILE_SIZE)
            row = (flir.block_idx("y") * tile_c + flir.thread_idx("y"))
            col = (flir.block_idx("x") * tile_c + flir.thread_idx("x"))

            tx = flir.thread_idx("x")
            ty = flir.thread_idx("y")

            zero = arith.index(0)
            one = arith.index(1)
            zero_f = arith.f32(0.0)

            acc = zero_f
            num_tiles = arith.index(K // TILE_SIZE)

            # Flir Layout definition for LDS tiles
            tile_shape = flir.make_shape(tile_c, tile_c)
            tile_stride = flir.make_stride(tile_c, one)
            tile_layout = flir.make_layout(tile_shape, tile_stride)

            def get_tile_idx(y, x):
                coord = flir.make_coord(y, x)
                idx_val = flir.crd2idx(coord, tile_layout)
                return idx_val.value if hasattr(idx_val, "value") else idx_val

            for t in range(num_tiles):
                k_base = (t * tile_c)

                a_col = (k_base + tx)
                a_val = flir.memref.load(A, [row.value, a_col.value])
                As.store(a_val.value, [get_tile_idx(ty.value, tx.value)])

                b_row = (k_base + ty)
                b_val = flir.memref.load(B, [b_row.value, col.value])
                Bs.store(b_val.value, [get_tile_idx(ty.value, tx.value)])

                gpu.barrier()

                for k_local in range(tile_c):
                    a_smem = As.load([get_tile_idx(ty.value, k_local)])
                    b_smem = Bs.load([get_tile_idx(k_local, tx.value)])
                    acc = (acc + a_smem * b_smem)

                gpu.barrier()

            out_v = acc.value if hasattr(acc, "value") else acc
            flir.memref.store(out_v, C, [row.value, col.value])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            c1 = arith.index(1).value
            tile = arith.index(TILE_SIZE).value
            grid = arith.index((M + TILE_SIZE - 1) // TILE_SIZE).value
            flir.gpu_ext.LaunchFuncOp(
                ["matmul_shared", "matmul_shared"],
                grid_size=(grid, grid, c1),
                block_size=(tile, tile, c1),
                kernel_operands=[A, B, C],
            )

    m = _MatmulShared()
    exe = pyflir.compile(m)

    np.random.seed(42)
    a_host = np.random.randn(M, K).astype(np.float32) * 0.01
    b_host = np.random.randn(K, N).astype(np.float32) * 0.01
    expected = a_host @ b_host

    A = torch.tensor(a_host, device="cuda", dtype=torch.float32)
    B = torch.tensor(b_host, device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    exe(A, B, C)
    torch.cuda.synchronize()

    # quick perf-ish loop (keep small for test runtime)
    start_time = time.time()
    for _ in range(3):
        exe(A, B, C)
    torch.cuda.synchronize()
    _ = (time.time() - start_time)

    c_host = C.cpu().numpy()

    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)
    assert rel_error < 5e-2



