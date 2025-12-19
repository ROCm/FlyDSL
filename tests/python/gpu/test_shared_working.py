#!/usr/bin/env python3
"""Shared memory matmul - correctness smoke test.

This used to execute at import time; it is now a proper pytest test and is
skipped when HIP isn't available.
"""

import ctypes
import time

import numpy as np
import pytest
import _mlir.extras.types as T
from _mlir import ir

from rocdsl.dialects.ext import arith, rocir
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco

try:
    from hip import hip
except ImportError:
    pytest.skip("HIP module not found. Skipping GPU tests.", allow_module_level=True)


M, N, K = 256, 256, 256
TILE_SIZE = 16


def test_matmul_shared_working():
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _MatmulShared(rocir.MlirModule):
        GPU_MODULE_NAME = "matmul_shared"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            # Allocate enough space for TILE_SIZE x TILE_SIZE float32 elements.
            _state["s_a_decl"] = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
            _state["s_b_decl"] = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
            allocator.finalize()

        @rocir.kernel
        def matmul_shared(
            self: rocir.T.i64,
            A: lambda: T.memref(M, K, T.f32()),
            B: lambda: T.memref(K, N, T.f32()),
            C: lambda: T.memref(M, N, T.f32()),
        ):
            gpu = rocir.gpu_ext
            scf = rocir.scf

            # Get references to shared memory using Allocator
            base_ptr = allocator.get_base()
            As = _state["s_a_decl"](base_ptr)
            Bs = _state["s_b_decl"](base_ptr)

            tile_c = arith.index(TILE_SIZE)
            row = (rocir.block_idx("y") * tile_c + rocir.thread_idx("y"))
            col = (rocir.block_idx("x") * tile_c + rocir.thread_idx("x"))

            tx = rocir.thread_idx("x")
            ty = rocir.thread_idx("y")

            zero = arith.index(0)
            one = arith.index(1)
            zero_f = arith.f32(0.0)

            acc = zero_f
            num_tiles = arith.index(K // TILE_SIZE)

            # Rocir Layout definition for LDS tiles
            tile_shape = rocir.make_shape(tile_c, tile_c)
            tile_stride = rocir.make_stride(tile_c, one)
            tile_layout = rocir.make_layout(tile_shape, tile_stride)

            def get_tile_idx(y, x):
                coord = rocir.make_coord(y, x)
                idx_val = rocir.crd2idx(coord, tile_layout)
                return idx_val.value if hasattr(idx_val, "value") else idx_val

            for_tiles = scf.ForOp(zero.value, num_tiles.value, one.value, [acc.value])
            with ir.InsertionPoint(for_tiles.body):
                t = for_tiles.induction_variable
                acc_val = for_tiles.inner_iter_args[0]
                k_base = (t * tile_c)

                a_col = (k_base + tx)
                a_val = rocir.memref.load(A, [row.value, a_col.value])
                As.store(a_val.value, [get_tile_idx(ty.value, tx.value)])

                b_row = (k_base + ty)
                b_val = rocir.memref.load(B, [b_row.value, col.value])
                Bs.store(b_val.value, [get_tile_idx(ty.value, tx.value)])

                gpu.barrier()

                for_k = scf.ForOp(zero.value, tile_c.value, one.value, [acc_val.value])
                with ir.InsertionPoint(for_k.body):
                    k_local = for_k.induction_variable
                    acc_k = for_k.inner_iter_args[0]

                    a_smem = As.load([get_tile_idx(ty.value, k_local)])
                    b_smem = Bs.load([get_tile_idx(k_local, tx.value)])
                    new_acc = (acc_k + a_smem * b_smem)

                    scf.yield_([new_acc.value])

                gpu.barrier()
                out = for_k.results[0]
                out_v = out.value if hasattr(out, "value") else out
                scf.yield_([out_v])

            out = for_tiles.results[0]
            out_v = out.value if hasattr(out, "value") else out
            rocir.memref.store(out_v, C, [row.value, col.value])

    m = _MatmulShared()
    hsaco = compile_to_hsaco(m.module, kernel_name="matmul_shared")

    np.random.seed(42)
    a_host = np.random.randn(M, K).astype(np.float32) * 0.01
    b_host = np.random.randn(K, N).astype(np.float32) * 0.01
    c_host = np.zeros((M, N), dtype=np.float32)
    expected = a_host @ b_host

    d_a = hip_check(hip.hipMalloc(M * K * 4))
    d_b = hip_check(hip.hipMalloc(K * N * 4))
    d_c = hip_check(hip.hipMalloc(M * N * 4))

    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * K * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, K * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul_shared"))

    grid_x = grid_y = (M + TILE_SIZE - 1) // TILE_SIZE
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())

    # quick perf-ish loop (keep small for test runtime)
    start_time = time.time()
    for _ in range(3):
        hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    _ = (time.time() - start_time)

    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))

    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)
    assert rel_error < 5e-2



