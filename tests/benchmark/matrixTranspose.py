#!/usr/bin/env python3
"""Matrix Transpose Benchmark - GPU kernel with Shared Memory + Vec2 Vectorization"""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu, rocir, arith, scf
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import memref, vector
from mlir.ir import F32Type, InsertionPoint, IntegerType
from mlir.dialects import arith as std_arith
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

# Import benchmark utilities from shared tests/utils.py
from utils import BenchmarkResults, perftest, compile_to_hsaco


def benchmark_matrix_transpose_arith(TILE_SIZE=4, BLOCK_TILE=32):
    """Benchmark matrix transpose kernel performance (Arith MLIR Implementation)"""
    assert TILE_SIZE % 2 == 0, "TILE_SIZE must be multiple of 2 for vec2"
        
    M, N = 4096, 4096
    VEC_SIZE = 2  # vec2
    
    # Configuration
    # BLOCK_TILE and TILE_SIZE provided by user
    PAD = 2          # Shared memory padding to avoid bank conflicts
    
    # Thread block dimensions
    # Each thread handles TILE_SIZE columns, so we need BLOCK_TILE/TILE_SIZE threads in X
    BLOCK_X = BLOCK_TILE // TILE_SIZE
    BLOCK_Y = BLOCK_TILE
    ITERS = TILE_SIZE // VEC_SIZE  # Number of vec2 operations per thread
    # Workgroup size check (metadata currently emits max_flat_workgroup_size=256)
    BLOCK_THREADS = BLOCK_X * BLOCK_Y
    MAX_WG = 256
    if BLOCK_THREADS > MAX_WG:
        # Suggest a larger TILE_SIZE (even divisor) that fits
        suggestion = None
        for d in range(TILE_SIZE + (TILE_SIZE % 2), BLOCK_TILE + 1, 2):
            if BLOCK_TILE % d == 0:
                threads = (BLOCK_TILE * BLOCK_TILE) // d
                if threads <= MAX_WG:
                    suggestion = d
                    break
        msg = f"Workgroup threads {BLOCK_THREADS} exceed max {MAX_WG}. Reduce BLOCK_TILE or increase TILE_SIZE."
        if suggestion:
            msg += f" For this BLOCK_TILE={BLOCK_TILE}, try TILE_SIZE={suggestion} (threads={(BLOCK_TILE*BLOCK_TILE)//suggestion})."
        raise ValueError(msg)
    # LDS usage check (bytes)
    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    SMEM_BYTES = SMEM_SIZE * 4
    MAX_LDS = 65536  # 64KB typical
    if SMEM_BYTES > MAX_LDS:
        raise ValueError(f"LDS requirement {SMEM_BYTES} bytes exceeds limit {MAX_LDS} bytes. "
                         f"Reduce BLOCK_TILE or TILE_SIZE.")
    
    print(f"Config: Block={BLOCK_X}x{BLOCK_Y}, Iters/thread={ITERS}, Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}")
    
    # Compile kernel
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    # Shared memory: 1D layout for vectorized access
    smem_type = T.memref(SMEM_SIZE, T.f32(), memory_space=gpu.lds_space())
    memref.global_(sym_name="tile_smem", type_=smem_type, alignment=16)
    
    # Use flat 1D memrefs as kernel parameters
    @gpu.func(emit=True)
    def matrixTranspose(A: T.memref(M * N, T.f32()), B: T.memref(N * M, T.f32())):
        smem = memref.get_global(smem_type, "tile_smem")
        
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        # Constants
        m_c = arith.index(M)._value
        n_c = arith.index(N)._value
        block_tile_c = arith.index(BLOCK_TILE)._value
        smem_stride_c = arith.index(BLOCK_TILE + PAD)._value
        tile_size_c = arith.index(TILE_SIZE)._value
        two_c = arith.index(2)._value
        
        vec2_type = T.vector(VEC_SIZE, T.f32())
        
        # Load from global memory A using vec2
        # Thread (tx, ty) loads from A: row = ty, col_chunk = tx
        row_a = (by * block_tile_c + ty)._value
        row_a_valid = (row_a < m_c)._value
        
        for i in range(ITERS):
            i_c = arith.index(i)._value
            # Col index in A: bx*BLOCK_TILE + tx*TILE_SIZE + i*VEC_SIZE
            col_a_offset = (tx * tile_size_c + i_c * two_c)._value
            col_a = (bx * block_tile_c + col_a_offset)._value
            
            col_a_valid = (col_a < n_c)._value
            col_a_end_valid = ((col_a + two_c) <= n_c)._value
            valid_load = (row_a_valid & col_a_valid & col_a_end_valid)._value
            
            with ir.InsertionPoint(scf.IfOp(valid_load.value).then_block):
                # Load vec2 from A
                g_idx = (row_a * n_c + col_a)._value
                vec_val = vector.load(vec2_type, A, 
                                     [g_idx.value if hasattr(g_idx, "value") else g_idx])
                
                # Store to smem[ty][col_a_offset]
                s_idx = (ty * smem_stride_c + col_a_offset)._value
                vector.store(vec_val, smem, [s_idx.value if hasattr(s_idx, "value") else s_idx])
                scf.yield_([])
        
        gpu.barrier()
        
        # Phase 2: Store to global memory B using vec2        
        threads_per_row_b = BLOCK_TILE // VEC_SIZE
        rows_per_iter_val = (BLOCK_X * BLOCK_Y) // threads_per_row_b
        num_phase2_iters = BLOCK_TILE // rows_per_iter_val
        
        # Flatten thread ID: tid = ty * BLOCK_X + tx
        tid = (ty * arith.index(BLOCK_X)._value + tx)._value
        
        threads_per_row_c = arith.index(threads_per_row_b)._value
        write_row_base = (tid // threads_per_row_c)._value
        write_col = ((tid % threads_per_row_c) * two_c)._value
        
        rows_per_iter_c = arith.index(rows_per_iter_val)._value
        
        for k in range(num_phase2_iters):
            k_c = arith.index(k)._value
            curr_row_local = (write_row_base + k_c * rows_per_iter_c)._value
            
            # Global row index in B: bx*BLOCK_TILE + curr_row_local
            row_b = (bx * block_tile_c + curr_row_local)._value
            
            # Global col index in B: by*BLOCK_TILE + write_col
            col_b = (by * block_tile_c + write_col)._value
            
            row_b_valid = (row_b < n_c)._value
            col_b_valid = (col_b < m_c)._value
            col_b_end_valid = ((col_b + two_c) <= m_c)._value
            valid_store = (row_b_valid & col_b_valid & col_b_end_valid)._value
            
            with ir.InsertionPoint(scf.IfOp(valid_store.value).then_block):
                # Read from smem[write_col][curr_row_local] (transposed)
                s_idx_0 = (write_col * smem_stride_c + curr_row_local)._value
                val_0 = memref.load(smem, [s_idx_0.value if hasattr(s_idx_0, "value") else s_idx_0])
                
                one_c = arith.index(1)._value
                s_idx_1 = ((write_col + one_c) * smem_stride_c + curr_row_local)._value
                val_1 = memref.load(smem, [s_idx_1.value if hasattr(s_idx_1, "value") else s_idx_1])
                
                # Form vec2
                vec_out = vector.from_elements(vec2_type, 
                                              [val_0.value if hasattr(val_0, "value") else val_0,
                                               val_1.value if hasattr(val_1, "value") else val_1])
                
                # Store vec2 to B
                g_idx_b = (row_b * m_c + col_b)._value
                vector.store(vec_out, B, [g_idx_b.value if hasattr(g_idx_b, "value") else g_idx_b])
                scf.yield_([])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module, kernel_name="matrixTranspose")
    print(f"\nCompiled to HSACO: {len(hsaco)} bytes")
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")
    
    # Allocate device memory
    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    # Flatten to row-major 1D for kernel
    a_host = a_host_2d.flatten('C')  # C order = row-major
    b_host = np.zeros(N * M, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_b = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matrixTranspose"))
    
    # Grid: each block processes BLOCK_TILE x BLOCK_TILE
    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE
    
    print(f"Grid: ({grid_x}, {grid_y}), Block: ({BLOCK_X}, {BLOCK_Y})")
    print(f"Total threads: {grid_x * grid_y * BLOCK_X * BLOCK_Y:,}")
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Define kernel launch function
    def launch_kernel():
        hip_check(hip.hipModuleLaunchKernel(
            kernel_func,
            grid_x, grid_y, 1,  # grid dimensions
            BLOCK_X, BLOCK_Y, 1,  # block dimensions
            0,  # shared memory bytes (static allocation via memref.global_)
            None,  # stream
            args,
            None
        ))
        hip_check(hip.hipDeviceSynchronize())
    
    @perftest
    def run_benchmark():
        return {
            "launch": launch_kernel,
            "size": M * N,
            "total_bytes": 2 * M * N * 4,  # Read + Write
        }
    
    # Run benchmark
    results = run_benchmark()
    
    # Verify correctness
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    # Reshape back to 2D for comparison
    b_result_2d = b_host.reshape(N, M, order='C')  # row-major
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    # Print benchmark results
    print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5, results


def benchmark_matrix_transpose_rocir(TILE_SIZE=4, BLOCK_TILE=32):
    """Benchmark matrix transpose using Rocir Layout Algebra."""
    VEC_WIDTH = 2  # vec2 for float32
    assert TILE_SIZE % 2 == 0, "TILE_SIZE must be divisible by VEC_WIDTH (2)"

    M, N = 4096, 4096
    PAD = 2
    
    # Block dimensions
    THREADS_PER_BLOCK_X = BLOCK_TILE // TILE_SIZE
    THREADS_PER_BLOCK_Y = BLOCK_TILE
    
    # Workgroup size check (metadata currently emits max_flat_workgroup_size=256)
    BLOCK_THREADS = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y
    MAX_WG = 256
    if BLOCK_THREADS > MAX_WG:
        suggestion = None
        for d in range(TILE_SIZE + (TILE_SIZE % 2), BLOCK_TILE + 1, 2):
            if BLOCK_TILE % d == 0:
                threads = (BLOCK_TILE * BLOCK_TILE) // d
                if threads <= MAX_WG:
                    suggestion = d
                    break
        msg = f"Workgroup threads {BLOCK_THREADS} exceed max {MAX_WG}. Reduce BLOCK_TILE or increase TILE_SIZE."
        if suggestion:
            msg += f" For this BLOCK_TILE={BLOCK_TILE}, try TILE_SIZE={suggestion} (threads={(BLOCK_TILE*BLOCK_TILE)//suggestion})."
        raise ValueError(msg)
    
    # LDS usage check (bytes)
    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    SMEM_BYTES = SMEM_SIZE * 4
    MAX_LDS = 65536  # 64KB typical
    if SMEM_BYTES > MAX_LDS:
        raise ValueError(f"LDS requirement {SMEM_BYTES} bytes exceeds limit {MAX_LDS} bytes. "
                         f"Reduce BLOCK_TILE or TILE_SIZE.")
    
    print(f"Config: Block={THREADS_PER_BLOCK_X}x{THREADS_PER_BLOCK_Y}, " +
          f"Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}")
    
    # Create kernel using rocir API
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels_rocir", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    # Shared memory definition
    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    smem_type = T.memref(SMEM_SIZE, T.f32(), memory_space=gpu.lds_space())
    memref.global_(sym_name="tile_smem_rocir", type_=smem_type, alignment=16)
    
    @gpu.func(emit=True)
    def matrixTransposeRocir(
        A: T.memref(M * N, T.f32()),
        B: T.memref(N * M, T.f32())
    ):
        smem = memref.get_global(smem_type, "tile_smem_rocir")
        
        # Helper: reinterpret 1D memref as 2D with given static shape/strides
        def cast_1d_to_2d(source, shape, strides, use_layout=False):
            layout = memref.StridedLayoutAttr.get(offset=0, strides=strides) if use_layout else None
            out_type = T.memref(
                shape[0], shape[1], T.f32(),
                memory_space=source.type.memory_space,
                layout=layout,
            )
            return memref.reinterpret_cast(
                out_type,
                source,
                offsets=[], sizes=[], strides=[],
                static_offsets=[0],
                static_sizes=shape,
                static_strides=strides,
            )

        # Rocir indices
        tx = rocir.thread_idx("x")
        ty = rocir.thread_idx("y")
        bx = rocir.block_idx("x")
        by = rocir.block_idx("y")
        
        # Define Thread Layout
        thr_layout = rocir.make_ordered_layout(
            (THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_X),
            order=(1, 0) 
        )
        
        # Define Value Layout
        val_layout = rocir.make_ordered_layout(
            (1, TILE_SIZE),
            order=(1, 0)
        )
        
        # Copy Atoms (Vectorized)
        copy_atom = rocir.make_copy_atom(T.f32(), vector_size=VEC_WIDTH)
        
        # Tiled Copy Definition (A -> Smem)
        tiled_copy_A = rocir.make_tiled_copy_tv(
            copy_atom,
            thr_layout,
            val_layout,
            thr_shape=(THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_X),
            val_shape=(1, TILE_SIZE) 
        )
        
        # Phase 1: Global A -> Smem
        # Cast A (1D) to 2D view for Layout Algebra
        memref_A_2d = cast_1d_to_2d(A, [M, N], [N, 1])
        tensor_A = rocir.make_tensor(memref_A_2d, shape=(M, N), strides=(N, 1))
        
        smem_stride = BLOCK_TILE + PAD
        # Smem 2D view
        memref_Smem_2d = cast_1d_to_2d(smem, [BLOCK_TILE, BLOCK_TILE], [smem_stride, 1], use_layout=True)
        tensor_Smem = rocir.make_tensor(memref_Smem_2d, shape=(BLOCK_TILE, BLOCK_TILE), strides=(smem_stride, 1))
        
        tile_shape = (BLOCK_TILE, BLOCK_TILE)
        gA = rocir.zipped_divide(tensor_A, tile_shape)
        
        blk_coord = (by, bx)
        tile_A = gA[blk_coord]
        
        thr_copy_A_slice = tiled_copy_A.get_slice((ty * THREADS_PER_BLOCK_X + tx))
        
        thr_tile_A = thr_copy_A_slice.partition_S(tile_A)
        thr_tile_Smem = thr_copy_A_slice.partition_S(tensor_Smem)
        
        frg_A = rocir.make_fragment_like(thr_tile_A, T.f32())
        
        rocir.copy(tiled_copy_A, thr_tile_A, frg_A)
        rocir.copy(tiled_copy_A, frg_A, thr_tile_Smem)
        
        gpu.barrier()
        
        # Phase 2: Smem -> Global B (Transpose)
        memref_B_2d = cast_1d_to_2d(B, [N, M], [M, 1])
        tensor_B = rocir.make_tensor(memref_B_2d, shape=(N, M), strides=(M, 1))
        
        # Transposed Smem view (swap strides)
        memref_Smem_T_2d = cast_1d_to_2d(smem, [BLOCK_TILE, BLOCK_TILE], [1, smem_stride], use_layout=True)
        tensor_Smem_T = rocir.make_tensor(memref_Smem_T_2d, shape=(BLOCK_TILE, BLOCK_TILE), strides=(1, smem_stride))
        
        tiled_copy_B = tiled_copy_A 
        
        gB = rocir.zipped_divide(tensor_B, tile_shape)
        
        blk_coord_B = (bx, by)
        tile_B = gB[blk_coord_B]
        
        thr_copy_B_slice = tiled_copy_B.get_slice((ty * THREADS_PER_BLOCK_X + tx))
        
        thr_tile_B = thr_copy_B_slice.partition_S(tile_B)
        thr_tile_Smem_T = thr_copy_B_slice.partition_S(tensor_Smem_T)
        
        frg_B = rocir.make_fragment_like(thr_tile_B, T.f32())
        
        rocir.copy(tiled_copy_B, thr_tile_Smem_T, frg_B)
        rocir.copy(tiled_copy_B, frg_B, thr_tile_B)
    
    ip.__exit__(None, None, None)
    
    # Compile with optimization pipeline
    print("  Running optimization pipeline...")
    optimized = run_pipeline(ctx.module, Pipeline().canonicalize().cse())
    
    hsaco = compile_to_hsaco(optimized, kernel_name="matrixTransposeRocir")
    print(f"\nCompiled to HSACO: {len(hsaco)} bytes")
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")
    
    # Allocate device memory
    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    a_host = a_host_2d.flatten('C')
    b_host = np.zeros(N * M, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_b = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matrixTransposeRocir"))
    
    # Grid configuration
    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE
    
    print(f"Grid: ({grid_x}, {grid_y}), Block: ({THREADS_PER_BLOCK_X}, {THREADS_PER_BLOCK_Y})")
    print(f"Total threads: {grid_x * grid_y * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y:,}")
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    def launch_kernel():
        hip_check(hip.hipModuleLaunchKernel(
            kernel_func,
            grid_x, grid_y, 1,
            THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1,
            0, None, args, None
        ))
        hip_check(hip.hipDeviceSynchronize())
    
    @perftest
    def run_benchmark():
        return {
            "launch": launch_kernel,
            "size": M * N,
            "total_bytes": 2 * M * N * 4,
        }
    
    # Run benchmark
    results = run_benchmark()
    
    # Verify correctness
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    b_result_2d = b_host.reshape(N, M, order='C')
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5, results

# Pytest test function
def test_benchmark_matrix_transpose():
    """Pytest wrapper for matrix transpose benchmark."""
    # Test with TILE_SIZE=4, BLOCK_TILE=32 (optimal configuration)
    result, _ = benchmark_matrix_transpose_arith(TILE_SIZE=4, BLOCK_TILE=32)
    assert result, "Matrix transpose benchmark failed correctness check"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Matrix Transpose Benchmark - Compare Arith vs Rocir')
    parser.add_argument('--tile-size', type=int, default=4,
                       help='Elements per thread (default: 4)')
    parser.add_argument('--block-tile', type=int, default=32,
                       help='Block tile size (default: 32)')
    args = parser.parse_args()

    # Basic config validation to avoid cryptic failures
    if args.tile_size <= 0 or args.block_tile <= 0:
        print("Error: tile-size and block-tile must be positive.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Matrix Transpose Comparison")
    print(f"GPU: {get_hip_arch()}")
    
    results_arith = None
    results_rocir = None
    
    print("\n" + "="*80)
    print("RUNNING: Arith Implementation")
    try:
        success, results_arith = benchmark_matrix_transpose_arith(
            TILE_SIZE=args.tile_size, 
            BLOCK_TILE=args.block_tile
        )
        if not success:
            print("Arith implementation failed correctness check")
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("RUNNING: Rocir Layout API Implementation")
    try:
        success, results_rocir = benchmark_matrix_transpose_rocir(
            TILE_SIZE=args.tile_size,
            BLOCK_TILE=args.block_tile
        )
        if not success:
            print("Rocir implementation failed correctness check")
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    # Compare results
    if results_arith and results_rocir:
        arith_bw = results_arith.bandwidth_gbs
        rocir_bw = results_rocir.bandwidth_gbs
        speedup = rocir_bw / arith_bw
        
        print(f"{'Arith':<20} {results_arith.avg_ms:<15.3f} {arith_bw:<20.2f} {'1.00x':<10}")
        print(f"{'Rocir Layout API':<20} {results_rocir.avg_ms:<15.3f} {rocir_bw:<20.2f} {f'{speedup:.2f}x':<10}")
    
    print("✓ BENCHMARK COMPLETED")