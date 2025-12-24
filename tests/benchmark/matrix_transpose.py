#!/usr/bin/env python3
"""Matrix Transpose Benchmark - GPU kernel with Shared Memory + Vec2 Vectorization"""

import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyflir
from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import gpu, flir, arith
from pyflir.dialects.ext.python_control_flow import lower_range_for_loops, range_constexpr
from pyflir.runtime.device import get_rocm_arch
from _mlir import ir
from _mlir.dialects import memref, vector
from _mlir.ir import F32Type, IntegerType
import _mlir.extras.types as T
import numpy as np
import torch

from pyflir.utils import SmemAllocator


def benchmark_matrix_transpose_arith(TILE_SIZE=4, BLOCK_TILE=32):
    """Benchmark matrix transpose kernel performance (Arith MLIR Implementation)

    Updated: each thread processes multiple vec4 chunks. Threads are assigned
    contiguous vec4 chunks across the block.
    """
    VEC_SIZE = 4  # vec4
    if TILE_SIZE % VEC_SIZE != 0:
        TILE_SIZE = ((TILE_SIZE + VEC_SIZE - 1) // VEC_SIZE) * VEC_SIZE
        print(f"Adjusted TILE_SIZE to {TILE_SIZE} (multiple of vec4)")

    ITERS = TILE_SIZE // VEC_SIZE
    M, N = 4096, 4096

    # Configuration
    PAD = 2
    BLOCK_X = BLOCK_TILE // TILE_SIZE
    BLOCK_Y = BLOCK_TILE

    BLOCK_THREADS = BLOCK_X * BLOCK_Y
    MAX_WG = 256
    if BLOCK_THREADS > MAX_WG:
        raise ValueError(
            f"Workgroup threads {BLOCK_THREADS} exceed max {MAX_WG}. "
            f"Reduce BLOCK_TILE or use smaller BLOCK_TILE for fixed TILE_SIZE=8."
        )
    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    SMEM_BYTES = SMEM_SIZE * 4
    MAX_LDS = 65536  # 64KB typical
    if SMEM_BYTES > MAX_LDS:
        raise ValueError(
            f"LDS requirement {SMEM_BYTES} bytes exceeds limit {MAX_LDS} bytes. "
            f"Reduce BLOCK_TILE or TILE_SIZE."
        )

    print(
        f"Config: Block={BLOCK_X}x{BLOCK_Y}, Iters/thread={ITERS}, Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}"
    )

    # Compile kernel
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    def _matrix_transpose_arith_impl(A, B):
        base_ptr = allocator.get_base()
        smem = _state["tile_smem_decl"](base_ptr).get()

        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")

        # Constants (Keep as ArithValue wrappers)
        m_c = arith.index(M)
        n_c = arith.index(N)
        block_tile_c = arith.index(BLOCK_TILE)
        smem_stride_c = arith.index(BLOCK_TILE + PAD)
        block_threads_c = arith.index(BLOCK_THREADS)
        vec_width_c = arith.index(VEC_SIZE)

        vec_type = T.vector(VEC_SIZE, T.f32())

        # Linear thread id
        tid = ty * arith.index(BLOCK_X) + tx

        # Phase 1: Global A -> Smem (Coalesced)
        for i in range_constexpr(ITERS):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c
            row_off = tile_linear // block_tile_c
            col_off = tile_linear % block_tile_c

            global_row = by * block_tile_c + row_off
            global_col = bx * block_tile_c + col_off

            row_valid = global_row < m_c
            col_valid = global_col < n_c
            col_end_valid = (global_col + vec_width_c) <= n_c
            valid_load = row_valid & col_valid & col_end_valid
            if valid_load:
                vals = []
                for t in range_constexpr(VEC_SIZE):
                    t_c = arith.index(t)
                    g_idx = global_row * n_c + global_col + t_c
                    val_op = memref.load(
                        A, [g_idx.value if hasattr(g_idx, "value") else g_idx]
                    )
                    val = (
                        val_op.value
                        if hasattr(val_op, "value")
                        else (val_op.result if hasattr(val_op, "result") else val_op)
                    )
                    vals.append(val)

                vec_val = vector.from_elements(vec_type, vals)
                s_idx = row_off * smem_stride_c + col_off
                vector.store(
                    vec_val, smem, [s_idx.value if hasattr(s_idx, "value") else s_idx]
                )

        gpu.barrier()

        # Phase 2: Smem -> Global B (Transpose)
        for i in range_constexpr(ITERS):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c
            row_off = tile_linear // block_tile_c
            col_off = tile_linear % block_tile_c

            base_row_b = bx * block_tile_c + row_off
            base_col_b = by * block_tile_c + col_off

            row_valid = base_row_b < n_c
            col_valid = base_col_b < m_c
            col_end_valid = (base_col_b + vec_width_c) <= m_c
            valid_store = row_valid & col_valid & col_end_valid
            if valid_store:
                vals = []
                for t in range_constexpr(VEC_SIZE):
                    t_c = arith.index(t)
                    s_idx = (col_off + t_c) * smem_stride_c + row_off
                    val_op = memref.load(
                        smem, [s_idx.value if hasattr(s_idx, "value") else s_idx]
                    )
                    val = (
                        val_op.value
                        if hasattr(val_op, "value")
                        else (val_op.result if hasattr(val_op, "result") else val_op)
                    )
                    vals.append(val)

                vec_val = vector.from_elements(vec_type, vals)
                g_idx_b = base_row_b * m_c + base_col_b
                vector.store(
                    vec_val,
                    B,
                    [g_idx_b.value if hasattr(g_idx_b, "value") else g_idx_b],
                )

    _matrix_transpose_arith_impl = lower_range_for_loops(_matrix_transpose_arith_impl)

    class _TransposeArith(flir.MlirModule):
        GPU_MODULE_NAME = "transpose_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["tile_smem_decl"] = allocator.allocate_array(T.f32(), SMEM_SIZE)
            allocator.finalize()

        # Use flat 1D memrefs as kernel parameters
        @flir.kernel
        def matrix_transpose(
            self,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            _matrix_transpose_arith_impl(A, B)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            c1 = arith.index(1).value
            gx = arith.index((N + BLOCK_TILE - 1) // BLOCK_TILE).value
            gy = arith.index((M + BLOCK_TILE - 1) // BLOCK_TILE).value
            bx = arith.index(BLOCK_X).value
            by = arith.index(BLOCK_Y).value
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "matrix_transpose"],
                grid_size=(gx, gy, c1),
                block_size=(bx, by, c1),
                kernel_operands=[A, B],
            )

    m = _TransposeArith()
    run_pipeline(m.module, Pipeline().canonicalize().cse())
    exe = pyflir.compile(m)
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")

    # Allocate device memory
    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    a_host = a_host_2d.flatten("C")
    a_dev = torch.tensor(a_host, device="cuda", dtype=torch.float32)
    b_dev = torch.empty((N * M,), device="cuda", dtype=torch.float32)

    # Grid: each block processes BLOCK_TILE x BLOCK_TILE
    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE

    print(f"Grid: ({grid_x}, {grid_y}), Block: ({BLOCK_X}, {BLOCK_Y})")
    print(f"Total threads: {grid_x * grid_y * BLOCK_X * BLOCK_Y:,}")

    # Define kernel launch function
    def launch_kernel():
        exe(a_dev, b_dev)
        torch.cuda.synchronize()

    # Run benchmark
    from tests.test_common import run_perftest

    _, avg_us = run_perftest(launch_kernel, num_iters=20, num_warmup=2)

    # Verify correctness
    b_host = b_dev.cpu().numpy()
    b_result_2d = b_host.reshape(N, M, order="C")
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))

    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")

    # Calculate metrics
    total_bytes = 2 * M * N * 4  # Read + Write
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000

    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": M * N,
        "total_bytes": total_bytes,
    }

    print(f"\n  Performance:")
    print(f"  Average Time: {avg_ms:.3f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")

    return error < 1e-5, results


def benchmark_matrix_transpose_buffer_load(TILE_SIZE=4, BLOCK_TILE=32):
    """Benchmark matrix transpose using Buffer Load (AMD CDNA3 optimized)."""
    from tests.test_common import run_perftest as _run_perftest_import

    VEC_WIDTH = 4
    if TILE_SIZE % VEC_WIDTH != 0:
        TILE_SIZE = ((TILE_SIZE + VEC_WIDTH - 1) // VEC_WIDTH) * VEC_WIDTH
        print(f"Adjusted TILE_SIZE to {TILE_SIZE} (multiple of vec4)")

    VEC_PER_THREAD = TILE_SIZE // VEC_WIDTH
    M, N = 4096, 4096
    PAD = 2

    THREADS_PER_BLOCK_X = BLOCK_TILE // TILE_SIZE
    THREADS_PER_BLOCK_Y = BLOCK_TILE

    BLOCK_THREADS = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y
    MAX_WG = 256
    if BLOCK_THREADS > MAX_WG:
        raise ValueError(f"Workgroup threads {BLOCK_THREADS} exceed max {MAX_WG}.")

    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    SMEM_BYTES = SMEM_SIZE * 4
    MAX_LDS = 65536
    if SMEM_BYTES > MAX_LDS:
        raise ValueError(
            f"LDS requirement {SMEM_BYTES} bytes exceeds limit {MAX_LDS} bytes."
        )

    print(
        f"Config: Block={THREADS_PER_BLOCK_X}x{THREADS_PER_BLOCK_Y}, "
        + f"Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}, Buffer Load Enabled"
    )

    # Import buffer operations
    from pyflir.dialects.ext import buffer_ops

    gpu_arch = get_rocm_arch()
    _state = {}

    def _matrix_transpose_buffer_load_impl(A, B):
        smem = memref.get_global(_state["smem_type"], "tile_smem_buffer_load")

        tx = flir.thread_idx("x")
        ty = flir.thread_idx("y")
        bx = flir.block_idx("x")
        by = flir.block_idx("y")

        # Constants
        m_c = arith.index(M)
        n_c = arith.index(N)
        block_tile_c = arith.index(BLOCK_TILE)
        block_threads_c = arith.index(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
        vec_width_c = arith.index(VEC_WIDTH)
        smem_stride_c = arith.index(BLOCK_TILE + PAD)

        tid = ty * arith.index(THREADS_PER_BLOCK_X) + tx

        a_rsrc = buffer_ops.create_buffer_resource(A)
        b_rsrc = buffer_ops.create_buffer_resource(B)

        vec_type = T.vector(VEC_WIDTH, T.f32())

        # Phase 1: Global -> Shared via buffer load
        for i in range(VEC_PER_THREAD):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c

            row_off = tile_linear // block_tile_c
            col_off = tile_linear % block_tile_c

            global_row = by * block_tile_c + row_off
            global_col = bx * block_tile_c + col_off

            row_valid = global_row < m_c
            col_valid = global_col < n_c
            col_end_valid = (global_col + vec_width_c) <= n_c
            valid_load = row_valid & col_valid & col_end_valid
            mask_val = valid_load.value if hasattr(valid_load, "value") else valid_load

            vec_val = buffer_ops.buffer_load_2d(
                a_rsrc, global_row, global_col, n_c, vec_width=VEC_WIDTH, mask=mask_val
            )

            s_idx = row_off * smem_stride_c + col_off
            s_idx_val = s_idx.value if hasattr(s_idx, "value") else s_idx
            vector.store(vec_val, smem, [s_idx_val])

        gpu.barrier()

        # Phase 2: Shared -> Global via buffer store (transposed)
        for i in range(VEC_PER_THREAD):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c

            row_off = tile_linear // block_tile_c
            col_off = tile_linear % block_tile_c

            base_row_b = bx * block_tile_c + row_off
            base_col_b = by * block_tile_c + col_off

            vals = []
            for t in range(VEC_WIDTH):
                t_c = arith.index(t)
                s_idx = (col_off + t_c) * smem_stride_c + row_off
                s_idx_val = s_idx.value if hasattr(s_idx, "value") else s_idx
                val_op = memref.load(smem, [s_idx_val])
                val = (
                    val_op.value
                    if hasattr(val_op, "value")
                    else (val_op.result if hasattr(val_op, "result") else val_op)
                )
                vals.append(val)

            vec_val = vector.from_elements(vec_type, vals)

            row_valid = base_row_b < n_c
            col_valid = base_col_b < m_c
            col_end_valid = (base_col_b + vec_width_c) <= m_c
            valid_store = row_valid & col_valid & col_end_valid
            mask_val = valid_store.value if hasattr(valid_store, "value") else valid_store

            buffer_ops.buffer_store_2d(
                vec_val, b_rsrc, base_row_b, base_col_b, m_c, mask=mask_val
            )

    class _TransposeBufferLoad(flir.MlirModule):
        GPU_MODULE_NAME = "transpose_kernels_buffer_load"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # Shared memory definition
            _state["smem_type"] = T.memref(SMEM_SIZE, T.f32(), memory_space=gpu.lds_space())
            memref.global_(sym_name="tile_smem_buffer_load", type_=_state["smem_type"], alignment=16)

        @flir.kernel
        def matrix_transpose_buffer_load(
            self,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            _matrix_transpose_buffer_load_impl(A, B)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            c1 = arith.index(1).value
            gx = arith.index((N + BLOCK_TILE - 1) // BLOCK_TILE).value
            gy = arith.index((M + BLOCK_TILE - 1) // BLOCK_TILE).value
            bx = arith.index(THREADS_PER_BLOCK_X).value
            by = arith.index(THREADS_PER_BLOCK_Y).value
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "matrix_transpose_buffer_load"],
                grid_size=(gx, gy, c1),
                block_size=(bx, by, c1),
                kernel_operands=[A, B],
            )

    m = _TransposeBufferLoad()

    print("  Running optimization pipeline...")
    # Try simpler pipeline for buffer ops (avoid aggressive canonicalization)
    try:
        optimized = run_pipeline(m.module, Pipeline().cse())
    except Exception as e:
        print(
            f"  Warning: Pipeline with CSE failed ({e}), trying without optimization..."
        )
        optimized = m.module

    exe = pyflir.compile(optimized)
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")

    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    a_host = a_host_2d.flatten("C")
    a_dev = torch.tensor(a_host, device="cuda", dtype=torch.float32)
    b_dev = torch.empty((N * M,), device="cuda", dtype=torch.float32)

    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE

    print(
        f"Grid: ({grid_x}, {grid_y}), Block: ({THREADS_PER_BLOCK_X}, {THREADS_PER_BLOCK_Y})"
    )
    print(
        f"Total threads: {grid_x * grid_y * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y:,}"
    )

    def launch_kernel():
        exe(a_dev, b_dev)
        torch.cuda.synchronize()

    # Run benchmark
    _, avg_us = _run_perftest_import(launch_kernel, num_iters=20, num_warmup=2)

    b_host = b_dev.cpu().numpy()
    b_result_2d = b_host.reshape(N, M, order="C")
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))

    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")

    # Calculate metrics
    total_bytes = 2 * M * N * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000

    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": M * N,
        "total_bytes": total_bytes,
    }

    print(f"\n  Performance:")
    print(f"  Average Time: {avg_ms:.3f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")

    return error < 1e-5, results


def benchmark_matrix_transpose_flir(TILE_SIZE=4, BLOCK_TILE=32):
    """Benchmark matrix transpose using Flir Layout Algebra."""
    from tests.test_common import run_perftest

    VEC_WIDTH = 4  # vec4 for float32
    if TILE_SIZE % VEC_WIDTH != 0:
        TILE_SIZE = ((TILE_SIZE + VEC_WIDTH - 1) // VEC_WIDTH) * VEC_WIDTH
        print(f"Adjusted TILE_SIZE to {TILE_SIZE} (multiple of vec4)")

    VEC_PER_THREAD = TILE_SIZE // VEC_WIDTH
    M, N = 4096, 4096
    PAD = 2

    THREADS_PER_BLOCK_X = BLOCK_TILE // TILE_SIZE
    THREADS_PER_BLOCK_Y = BLOCK_TILE

    BLOCK_THREADS = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y
    MAX_WG = 256
    if BLOCK_THREADS > MAX_WG:
        raise ValueError(f"Workgroup threads {BLOCK_THREADS} exceed max {MAX_WG}.")

    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    SMEM_BYTES = SMEM_SIZE * 4
    MAX_LDS = 65536
    if SMEM_BYTES > MAX_LDS:
        raise ValueError(
            f"LDS requirement {SMEM_BYTES} bytes exceeds limit {MAX_LDS} bytes."
        )

    print(
        f"Config: Block={THREADS_PER_BLOCK_X}x{THREADS_PER_BLOCK_Y}, "
        + f"Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}"
    )

    gpu_arch = get_rocm_arch()
    _state = {}

    def _matrix_transpose_flir_impl(A, B):
        # Helpers to keep the kernel code readable.
        def v(x):
            """Unwrap ArithValue (or similar wrappers) to an MLIR Value."""
            return x.value if hasattr(x, "value") else x

        smem = memref.get_global(_state["smem_type"], "tile_smem_flir")

        tensor_A = flir.TensorView(A, shape=(M * N,))
        tensor_smem = flir.TensorView(smem, shape=(BLOCK_TILE * (BLOCK_TILE + PAD),))

        tx = flir.thread_idx("x")
        ty = flir.thread_idx("y")
        bx = flir.block_idx("x")
        by = flir.block_idx("y")

        m_c = arith.index(M)
        n_c = arith.index(N)
        block_tile_c = arith.index(BLOCK_TILE)
        smem_stride_c = arith.index(BLOCK_TILE + PAD)
        block_threads_c = arith.index(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
        vec_width_c = arith.index(VEC_WIDTH)

        tid = ty * arith.index(THREADS_PER_BLOCK_X) + tx

        vec_type = T.vector(VEC_WIDTH, T.f32())

        # Layouts for index mapping (express address math via flir ops where possible).
        one_c = arith.index(1)
        i0 = arith.index(0)
        i1 = arith.index(1)

        tile_layout = flir.make_layout(
            (block_tile_c, block_tile_c),
            stride=(block_tile_c, one_c),  # row-major
        )
        a_layout = flir.make_layout((m_c, n_c), stride=(n_c, one_c))
        b_layout = flir.make_layout((n_c, m_c), stride=(m_c, one_c))
        smem_layout = flir.make_layout(
            (block_tile_c, smem_stride_c), stride=(smem_stride_c, one_c)
        )

        for i in range_constexpr(VEC_PER_THREAD):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c
            # (row_off, col_off) = idx2crd(tile_linear, tile_layout)
            tile_coord = flir.idx2crd(tile_linear, tile_layout)
            row_off = flir.get(tile_coord, i0)
            col_off = flir.get(tile_coord, i1)

            global_row = by * block_tile_c + row_off
            global_col = bx * block_tile_c + col_off

            row_valid = global_row < m_c
            col_valid = global_col < n_c
            col_end_valid = (global_col + vec_width_c) <= n_c
            valid_load = row_valid & col_valid & col_end_valid
            if valid_load:
                vals = []
                for t in range_constexpr(VEC_WIDTH):
                    t_c = arith.index(t)
                    a_coord = flir.make_coord(global_row, global_col + t_c)
                    g_idx = flir.crd2idx(a_coord, a_layout)
                    val = tensor_A[g_idx]
                    vals.append(val)

                vec_val = vector.from_elements(vec_type, vals)
                s_coord = flir.make_coord(row_off, col_off)
                s_idx = flir.crd2idx(s_coord, smem_layout)
                vector.store(vec_val, smem, [v(s_idx)])

        gpu.barrier()

        for i in range_constexpr(VEC_PER_THREAD):
            i_c = arith.index(i)
            vec_index = tid + i_c * block_threads_c
            tile_linear = vec_index * vec_width_c
            tile_coord = flir.idx2crd(tile_linear, tile_layout)
            row_off = flir.get(tile_coord, i0)
            col_off = flir.get(tile_coord, i1)

            base_row_b = bx * block_tile_c + row_off
            base_col_b = by * block_tile_c + col_off

            row_valid = base_row_b < n_c
            col_valid = base_col_b < m_c
            col_end_valid = (base_col_b + vec_width_c) <= m_c
            valid_store = row_valid & col_valid & col_end_valid
            if valid_store:
                vals = []
                for t in range_constexpr(VEC_WIDTH):
                    t_c = arith.index(t)
                    s_coord = flir.make_coord(col_off + t_c, row_off)
                    s_idx = flir.crd2idx(s_coord, smem_layout)
                    val = tensor_smem[s_idx]
                    vals.append(val)

                vec_val = vector.from_elements(vec_type, vals)
                b_coord = flir.make_coord(base_row_b, base_col_b)
                g_idx_b = flir.crd2idx(b_coord, b_layout)
                vector.store(vec_val, B, [v(g_idx_b)])

    _matrix_transpose_flir_impl = lower_range_for_loops(_matrix_transpose_flir_impl)

    class _TransposeFlir(flir.MlirModule):
        GPU_MODULE_NAME = "transpose_kernels_flir"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            # Shared memory definition
            _state["smem_type"] = T.memref(SMEM_SIZE, T.f32(), memory_space=gpu.lds_space())
            memref.global_(sym_name="tile_smem_flir", type_=_state["smem_type"], alignment=16)

        @flir.kernel
        def matrix_transpose_flir(
            self,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            _matrix_transpose_flir_impl(A, B)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M * N, T.f32()),
            B: lambda: T.memref(N * M, T.f32()),
        ):
            c1 = arith.index(1).value
            gx = arith.index((N + BLOCK_TILE - 1) // BLOCK_TILE).value
            gy = arith.index((M + BLOCK_TILE - 1) // BLOCK_TILE).value
            bx = arith.index(THREADS_PER_BLOCK_X).value
            by = arith.index(THREADS_PER_BLOCK_Y).value
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "matrix_transpose_flir"],
                grid_size=(gx, gy, c1),
                block_size=(bx, by, c1),
                kernel_operands=[A, B],
            )

    m = _TransposeFlir()

    print("  Running optimization pipeline...")
    optimized = run_pipeline(m.module, Pipeline().canonicalize().cse())

    exe = pyflir.compile(optimized)
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")

    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    a_host = a_host_2d.flatten("C")
    a_dev = torch.tensor(a_host, device="cuda", dtype=torch.float32)
    b_dev = torch.empty((N * M,), device="cuda", dtype=torch.float32)

    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE

    print(
        f"Grid: ({grid_x}, {grid_y}), Block: ({THREADS_PER_BLOCK_X}, {THREADS_PER_BLOCK_Y})"
    )
    print(
        f"Total threads: {grid_x * grid_y * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y:,}"
    )

    def launch_kernel():
        exe(a_dev, b_dev)
        torch.cuda.synchronize()

    # Run benchmark
    _, avg_us = run_perftest(launch_kernel, num_iters=20, num_warmup=2)
    
    b_host = b_dev.cpu().numpy()
    b_result_2d = b_host.reshape(N, M, order="C")
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))

    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")

    # Calculate metrics
    total_bytes = 2 * M * N * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000

    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": M * N,
        "total_bytes": total_bytes,
    }

    print(f"\n  Performance:")
    print(f"  Average Time: {avg_ms:.3f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")

    return error < 1e-5, results


def test_benchmark_matrix_transpose():
    """Pytest wrapper for matrix transpose benchmark."""
    result, _ = benchmark_matrix_transpose_arith(TILE_SIZE=4, BLOCK_TILE=32)
    assert result, "Matrix transpose benchmark failed correctness check"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Matrix Transpose Benchmark - Compare Arith vs Flir"
    )
    parser.add_argument(
        "--tile-size", type=int, default=4, help="Elements per thread (default: 4)"
    )
    parser.add_argument(
        "--block-tile", type=int, default=32, help="Block tile size (default: 32)"
    )
    args = parser.parse_args()

    if args.tile_size <= 0 or args.block_tile <= 0:
        print("Error: tile-size and block-tile must be positive.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Matrix Transpose Comparison")
    print(f"GPU: {get_rocm_arch()}")

    results_arith = None
    results_flir = None
    results_buffer_load = None

    print("\n" + "=" * 80)
    print("RUNNING: Arith Implementation")
    try:
        success, results_arith = benchmark_matrix_transpose_arith(
            TILE_SIZE=args.tile_size, BLOCK_TILE=args.block_tile
        )
        if not success:
            print("Arith implementation failed correctness check")
    except Exception as e:
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("RUNNING: Flir Layout API Implementation")
    try:
        success, results_flir = benchmark_matrix_transpose_flir(
            TILE_SIZE=args.tile_size, BLOCK_TILE=args.block_tile
        )
        if not success:
            print("Flir implementation failed correctness check")
    except Exception as e:
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("RUNNING: Buffer Load Implementation (AMD CDNA3)")
    try:
        print("  Importing buffer_ops...")
        from pyflir.dialects.ext import buffer_ops

        print("  ✓ buffer_ops imported")

        success, results_buffer_load = benchmark_matrix_transpose_buffer_load(
            TILE_SIZE=args.tile_size, BLOCK_TILE=args.block_tile
        )
        if not success:
            print("Buffer Load implementation failed correctness check")
    except Exception as e:
        print(f"  ✗ Error in Buffer Load implementation: {e}")
        import traceback

        traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Implementation':<25} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'Speedup':<10}"
    )
    print("-" * 80)
    
    def _metric(res, name: str, default=None):
        """Support either dict-style or object-style benchmark result containers."""
        if res is None:
            return default
        if isinstance(res, dict):
            return res.get(name, default)
        return getattr(res, name, default)

    if results_arith:
        arith_bw = _metric(results_arith, "bandwidth_gbs")
        arith_ms = _metric(results_arith, "avg_ms")
        print(f"{'Arith':<25} {arith_ms:<15.3f} {arith_bw:<20.2f} {'1.00x':<10}")
        
        if results_flir:
            flir_bw = _metric(results_flir, "bandwidth_gbs")
            flir_ms = _metric(results_flir, "avg_ms")
            speedup = flir_bw / arith_bw
            print(f"{'Flir Layout API':<25} {flir_ms:<15.3f} {flir_bw:<20.2f} {f'{speedup:.2f}x':<10}")
        
        if results_buffer_load:
            buffer_bw = _metric(results_buffer_load, "bandwidth_gbs")
            buffer_ms = _metric(results_buffer_load, "avg_ms")
            speedup = buffer_bw / arith_bw
            print(f"{'Buffer Load (CDNA3)':<25} {buffer_ms:<15.3f} {buffer_bw:<20.2f} {f'{speedup:.2f}x':<10}")
    
    print("="*80)
    print("✓ BENCHMARK COMPLETED")
