#!/usr/bin/env python3
"""
Benchmark: Per-Token Quantization Kernel

Usage Examples:
    # Single test with default size (4096x8192)
    python per_token_quant_benchmark_3.py
    
    # Single test with custom size
    python per_token_quant_benchmark_3.py -m 2048 -n 4096
    
    # Multi-size testing (same N - only compiles ONCE! 🚀)
    python per_token_quant_benchmark_3.py --multi-test "2048x8192,4096x8192,8192x8192"
    # Output: ✓ All tests use N=8192 - will compile once and reuse!
    
    # Multi-size testing (different N - compiles per N value)
    python per_token_quant_benchmark_3.py --multi-test "2048x4096,4096x8192"
    # Output: ⚠ 2 different N values detected: [4096, 8192]
    
    # Complex scenario (5 tests, 2 N values = 2 compilations)
    python per_token_quant_benchmark_3.py --multi-test "1024x4096,2048x4096,4096x4096,2048x8192,4096x8192"
    # Compiles N=4096 once → runs 3 tests
    # Compiles N=8192 once → runs 2 tests
"""

import sys
import os
import argparse


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ctypes
import numpy as np

try:
    import torch
    import aiter
    from aiter.ops.quant import per_token_quant_hip

    HAS_AITER = True
except ImportError:
    HAS_AITER = False

from hip import hip
from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import arith, gpu, rocir, block_reduce_ops
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from mlir import ir
from mlir.dialects import (
    scf,
    vector,
    memref,
)
import mlir.extras.types as T
from utils import compile_to_hsaco
from test_common import run_perftest


class scf_if:
    """Helper to restore scf.if_ like syntax using scf.IfOp context manager."""

    def __init__(self, cond):
        self.op = scf.IfOp(cond)

    def __enter__(self):
        self.ip = ir.InsertionPoint(self.op.then_block)
        self.ip.__enter__()
        return self.op

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if not self.op.then_block.operations or not isinstance(
                self.op.then_block.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])
        self.ip.__exit__(exc_type, exc_val, exc_tb)


class KernelCompilationCache:
    
    def __init__(self):
        self.cache = {}
    
    def get_or_compile(self, N, compile_fn):
        if N in self.cache:
            return self.cache[N] + (True,)
        
        result = compile_fn()
        self.cache[N] = result
        return result + (False,)
    
    def clear(self):
        self.cache.clear()


def compile_kernel_for_n(N, gpu_arch=None):
    if gpu_arch is None:
        gpu_arch = get_hip_arch()
    
    NUM_WARPS = 4
    BLOCK_SIZE = 64 * NUM_WARPS
    VEC_WIDTH = 32
    ELEMS_PER_BLOCK_ITER = BLOCK_SIZE * VEC_WIDTH
    
    assert N % VEC_WIDTH == 0, f"N must be multiple of {VEC_WIDTH}"
    ITERS = (N + ELEMS_PER_BLOCK_ITER - 1) // ELEMS_PER_BLOCK_ITER
    
    M_compile = 16384
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    allocator = SmemAllocator(ctx, arch=gpu_arch)
    f32_type = ir.F32Type.get()
    red_buffer_decl = allocator.allocate_array(f32_type, 64)

    @gpu.module("quant_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">'])
    def gpu_mod():
        allocator.finalize()

    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()

    red_type = T.memref(NUM_WARPS, T.f32(), memory_space=lds_space())
    memref.global_(sym_name="red_buffer", type_=red_type, alignment=16)

    @gpu.func(emit=True)
    def quant_kernel(
        input: T.memref(M_compile * N, T.f16()),
        output: T.memref(M_compile * N, T.i8()),
        scales: T.memref(M_compile, T.f32()),
    ):
        tid_x = rocir.thread_idx("x")
        bid_x = rocir.block_idx("x")
        tid_linear = tid_x

        thr_layout = rocir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
        val_layout = rocir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))

        copy_atom_load = rocir.make_copy_atom(T.f16(), vector_size=VEC_WIDTH)
        copy_atom_store = rocir.make_copy_atom(T.i8(), vector_size=VEC_WIDTH)

        tiled_copy_input = rocir.make_tiled_copy_tv(
            copy_atom_load,
            thr_layout,
            val_layout,
            thr_shape=(1, BLOCK_SIZE),
            val_shape=(1, VEC_WIDTH),
        )
        tiled_copy_output = rocir.make_tiled_copy_tv(
            copy_atom_store,
            thr_layout,
            val_layout,
            thr_shape=(1, BLOCK_SIZE),
            val_shape=(1, VEC_WIDTH),
        )

        tensor_input = rocir.make_tensor(input, shape=(M_compile, N), strides=(N, 1))
        tensor_output = rocir.make_tensor(output, shape=(M_compile, N), strides=(N, 1))
        tensor_scales = rocir.make_tensor(scales, shape=(M_compile,), strides=(1,))

        tile_shape = (1, ELEMS_PER_BLOCK_ITER)

        gInput = rocir.zipped_divide(tensor_input, tile_shape)
        gOutput = rocir.zipped_divide(tensor_output, tile_shape)

        thr_copy_input = tiled_copy_input.get_slice(tid_linear)
        thr_copy_output = tiled_copy_output.get_slice(tid_linear)

        base_ptr = allocator.get_base()
        red_val = red_buffer_decl(base_ptr).get()

        f_0 = arith.f32(0.0)
        f_1 = arith.f32(1.0)
        f_127 = arith.f32(127.0)
        c_0 = arith.index(0)

        local_max = f_0
        cached_vecs = []

        vec_type_f16 = T.vector(VEC_WIDTH, T.f16())
        vec_type_f32 = T.vector(VEC_WIDTH, T.f32())

        c_n = arith.index(N)

        for i in range(ITERS):
            c_chunk_offset = arith.index(i * ELEMS_PER_BLOCK_ITER)
            thread_offset = tid_linear * arith.index(VEC_WIDTH)
            col_base = c_chunk_offset + thread_offset

            is_valid = col_base < c_n

            blk_coord = (bid_x, arith.index(i))
            blkInput = gInput[blk_coord]
            thrInput = thr_copy_input.partition_S(blkInput)

            frgInput = rocir.make_fragment_like(thrInput, T.f16())

            zero_vec = arith.constant_vector(0.0, vec_type_f16).value
            frg_memref = frgInput.memref if hasattr(frgInput, "memref") else frgInput
            vector.store(zero_vec, frg_memref, [c_0.value, c_0.value])

            rocir.copy(tiled_copy_input, thrInput, frgInput, pred=is_valid)

            vec_val_f16 = vector.load(vec_type_f16, frg_memref, [c_0.value, c_0.value])
            vec_val_f32 = arith.extf(vec_type_f32, vec_val_f16)

            vec_abs_f32 = arith.absf(vec_val_f32)
            chunk_max = arith.reduce(vec_abs_f32, "max")
            local_max = arith.maximum(local_max, chunk_max)

            cached_vecs.append(vec_val_f32)

        reduced_max = block_reduce_ops.block_reduce_max(
            local_max, red_val, tid_linear, num_warps=NUM_WARPS, warp_size=64
        )

        scale = reduced_max / f_127
        is_zero = scale == 0
        final_scale = arith.select(is_zero, f_1, scale)

        c_0_idx = arith.index(0)
        is_thread_0 = tid_linear == c_0_idx

        with scf_if(is_thread_0.value):
            tensor_scales[bid_x] = final_scale

        vec_scale = vector.BroadcastOp(vec_type_f32, final_scale.value).result
        vec_f1 = vector.BroadcastOp(vec_type_f32, f_1.value).result
        vec_f1_arith = arith.ArithValue(vec_f1)
        vec_scale_arith = arith.ArithValue(vec_scale)
        vec_inv_scale = (vec_f1_arith / vec_scale_arith)

        for i in range(ITERS):
            c_chunk_offset = arith.index(i * ELEMS_PER_BLOCK_ITER)
            c_vec_width = arith.index(VEC_WIDTH)
            thread_offset = tid_linear * c_vec_width
            col_base = c_chunk_offset + thread_offset

            is_valid = col_base < c_n

            vec_val = cached_vecs[i]
            vec_scaled = vec_val * vec_inv_scale
            vec_i8_type = T.vector(VEC_WIDTH, T.i8())
            vec_quant = arith.fptosi(vec_i8_type, vec_scaled)

            blk_coord = (bid_x, arith.index(i))
            blkOutput = gOutput[blk_coord]
            thrOutput = thr_copy_output.partition_D(blkOutput)

            frgOutput = rocir.make_fragment_like(thrOutput, T.i8())
            frg_out_memref = (
                frgOutput.memref if hasattr(frgOutput, "memref") else frgOutput
            )

            vector.store(vec_quant.value, frg_out_memref, [c_0.value, c_0.value])
            rocir.copy(tiled_copy_output, frgOutput, thrOutput, pred=is_valid.value)

    ip.__exit__(None, None, None)

    hsaco = compile_to_hsaco(ctx.module)
    
    config = {
        'N': N,
        'NUM_WARPS': NUM_WARPS,
        'BLOCK_SIZE': BLOCK_SIZE,
        'VEC_WIDTH': VEC_WIDTH,
        'ELEMS_PER_BLOCK_ITER': ELEMS_PER_BLOCK_ITER,
        'ITERS': ITERS,
    }
    
    return hsaco, config


def benchmark_per_token_quant(M=4096, N=8192, hsaco=None, config=None):
    """Run Per-Token Quantization benchmark.
    
    Args:
        M: Number of tokens
        N: Hidden dimension
        hsaco: Pre-compiled HSACO bytes (optional)
        config: Config dict from compile_kernel_for_n (optional)
    
    Returns:
        bool: True if correctness check passed
    """
    print("\n" + "=" * 80)
    print(f"Benchmark: Per-Token Quantization Performance (RocDSL) [M={M}, N={N}]")
    print("=" * 80)

    # Compile if not provided
    if hsaco is None or config is None:
        gpu_arch = get_hip_arch()
        print(f"Detected HIP Arch: {gpu_arch}")
        print("Compiling MLIR module...")
        hsaco, config = compile_kernel_for_n(N, gpu_arch)
        print(f"Compiled to HSACO: {len(hsaco)} bytes")
    else:
        print(f"Using pre-compiled kernel ({len(hsaco)} bytes)")
    
    # Extract config
    NUM_WARPS = config['NUM_WARPS']
    BLOCK_SIZE = config['BLOCK_SIZE']
    VEC_WIDTH = config['VEC_WIDTH']
    ELEMS_PER_BLOCK_ITER = config['ELEMS_PER_BLOCK_ITER']
    ITERS = config['ITERS']

    total_elements = M * N
    total_bytes_rw = (M * N * 2) * 1 + (M * N * 1) + (M * 4)

    print(f"Configuration:")
    print(f"  - Shape: [{M}, {N}]")
    print(f"  - Block Size: {BLOCK_SIZE}")
    print(f"  - Total Elements: {total_elements/1e6:.2f}M")
    print(f"  - Loops per Block: {ITERS}")
    print(f"  - Est. Memory Traffic: {total_bytes_rw/1e9:.2f} GB per call")

    # Prepare test data
    np.random.seed(42)
    input_data_fp16 = np.random.uniform(-5.0, 5.0, size=(M, N)).astype(np.float16)
    input_data = input_data_fp16.astype(np.float32)
    dtypeMax = 127.0
    per_token_amax = np.max(np.abs(input_data), axis=1)
    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1.0
    scale_expanded = per_token_scale[:, np.newaxis]
    output_ref = (input_data / scale_expanded).astype(np.int8)

    input_size_bytes = M * N * 2
    output_size_bytes = M * N * 1
    scales_size_bytes = M * 4

    d_input = hip_check(hip.hipMalloc(input_size_bytes))
    d_output = hip_check(hip.hipMalloc(output_size_bytes))
    d_scales = hip_check(hip.hipMalloc(scales_size_bytes))

    hip_check(
        hip.hipMemcpy(
            d_input,
            input_data_fp16.ctypes.data,
            input_size_bytes,
            hip.hipMemcpyKind.hipMemcpyHostToDevice,
        )
    )

    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"quant_kernel"))

    arg_ptrs = [
        ctypes.c_void_p(int(d_input)),
        ctypes.c_void_p(int(d_output)),
        ctypes.c_void_p(int(d_scales)),
    ]
    args_array = (ctypes.c_void_p * len(arg_ptrs))(
        *[ctypes.addressof(p) for p in arg_ptrs]
    )

    def hip_kernel_launch():
        hip.hipModuleLaunchKernel(
            kernel_func,
            M,
            1,
            1,
            BLOCK_SIZE,
            1,
            1,
            sharedMemBytes=0,
            stream=None,
            kernelParams=args_array,
            extra=None,
        )
        hip.hipDeviceSynchronize()

    print("Running benchmark...")
    _, avg_us = run_perftest(hip_kernel_launch, num_iters=20, num_warmup=2)
    
    # Calculate metrics
    bandwidth_gbs = total_bytes_rw / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000
    
    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": total_elements,
        "total_bytes": total_bytes_rw,
    }

    output_host = np.zeros((M, N), dtype=np.int8)
    scales_host = np.zeros(M, dtype=np.float32)

    hip_check(
        hip.hipMemcpy(
            output_host.ctypes.data,
            d_output,
            output_size_bytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost,
        )
    )
    hip_check(
        hip.hipMemcpy(
            scales_host.ctypes.data,
            d_scales,
            scales_size_bytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost,
        )
    )

    scale_diff = np.max(np.abs(scales_host - per_token_scale))
    output_diff = np.max(
        np.abs(output_host.astype(np.float32) - output_ref.astype(np.float32))
    )

    print(f"\nRocDSL Kernel Results:")
    print(f"  Max Scale Diff:  {scale_diff:.2e}")
    print(f"  Max Output Diff: {output_diff:.2e}")
    # Standardized bandwidth line so run_tests.sh can pick it up (like matrixTranspose).
    print(f"\nBandwidth: {bandwidth_gbs:.2f} GB/s")
    print(f"  {results}")

    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipFree(d_scales))
    hip_check(hip.hipModuleUnload(hip_module))

    if HAS_AITER:
        print("\n" + "=" * 80)
        print("Benchmarking Reference Implementation (aiter)")
        print("=" * 80)

        input_torch = torch.from_numpy(input_data_fp16).cuda()

        def launch_aiter():
            per_token_quant_hip(input_torch)
            torch.cuda.synchronize()

        _, aiter_avg_us = run_perftest(launch_aiter, num_iters=20, num_warmup=2)
        
        aiter_bandwidth_gbs = total_bytes_rw / (aiter_avg_us / 1e6) / 1e9
        aiter_avg_ms = aiter_avg_us / 1000
        
        aiter_results = {
            "avg_ms": aiter_avg_ms,
            "avg_us": aiter_avg_us,
            "bandwidth_gbs": aiter_bandwidth_gbs,
            "size": total_elements,
            "total_bytes": total_bytes_rw,
        }

        output_torch, scale_torch = per_token_quant_hip(input_torch)
        torch.cuda.synchronize()

        output_ref_torch = output_torch.cpu().numpy()
        scale_ref_torch = scale_torch.squeeze().cpu().numpy()

        scale_diff_ref = np.max(np.abs(scale_ref_torch - per_token_scale))
        output_diff_ref = np.max(
            np.abs(output_ref_torch.astype(np.float32) - output_ref.astype(np.float32))
        )

        print(f"\n  Reference Correctness Check:")
        print(f"  Max Scale Diff:  {scale_diff_ref:.2e}")
        print(f"  Max Output Diff: {output_diff_ref:.2e}")

        rocdsl_time = results["avg_ms"]
        aiter_time = aiter_results["avg_ms"]
        speedup = aiter_time / rocdsl_time

        print(f"\n" + "=" * 80)
        print(f"Performance Comparison:")
        print(
            f"  RocDSL:     {rocdsl_time:7.3f} ms  ({results['bandwidth_gbs']:8.2f} GB/s)"
        )
        print(
            f"  Reference:  {aiter_time:7.3f} ms  ({aiter_results['bandwidth_gbs']:8.2f} GB/s)"
        )
        # Avoid printing another "Bandwidth:" line here; run_tests.sh greps for "Bandwidth:".
        print(f"  Speedup:    {speedup:7.2f}x")
        print("=" * 80)

    return output_diff <= 1.0


def test_benchmark_per_token_quant():
    """Pytest wrapper for per-token quantization benchmark."""
    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Per-Token Quantization")
    print(f"GPU: {get_hip_arch()}")
    print("=" * 80)
    assert (
        benchmark_per_token_quant()
    ), "Per-token quantization benchmark failed correctness check"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Per-Token Quantization")
    parser.add_argument(
        "-m", "--tokens", type=int, default=4096, help="Number of tokens (M)"
    )
    parser.add_argument(
        "-n", "--hidden", type=int, default=8192, help="Hidden dimension (N)"
    )
    parser.add_argument(
        "--multi-test", type=str, default=None,
        help="Run multiple tests with different sizes. Format: 'MxN,MxN,...' (e.g., '2048x4096,4096x8192')"
    )
    args = parser.parse_args()

    if args.multi_test:
        test_configs = []
        for size_str in args.multi_test.split(','):
            m_str, n_str = size_str.strip().split('x')
            test_configs.append((int(m_str), int(n_str)))

        print(f"\n{'='*80}")
        print(f"Per-Token Quantization Multi-Size Benchmark (v3 with caching)")
        print(f"Test Configurations: {len(test_configs)}")
        for i, (m, n) in enumerate(test_configs, 1):
            print(f"  {i}. M={m}, N={n}")
        print(f"{'='*80}")

        from collections import defaultdict
        tests_by_n = defaultdict(list)
        for m, n in test_configs:
            tests_by_n[n].append((m, n))
        
        unique_ns = sorted(tests_by_n.keys())
        if len(unique_ns) == 1:
            print(f"\n✓ All tests use N={unique_ns[0]} - will compile once and reuse!")
        else:
            print(f"\n⚠ {len(unique_ns)} different N values detected: {unique_ns}")
            print(f"  Will compile once per N value (total {len(unique_ns)} compilations)")
        
        gpu_arch = get_hip_arch()
        results = []
        cache = KernelCompilationCache()
        
        for n_val in unique_ns:
            n_tests = tests_by_n[n_val]
            
            print(f"\n{'='*80}")
            print(f"Processing N={n_val} group ({len(n_tests)} test(s))")
            print(f"{'='*80}")
            
            # Compile kernel for this N (only once)
            print(f"Compiling kernel for N={n_val}...")
            hsaco, config, was_cached = cache.get_or_compile(
                n_val,
                lambda: compile_kernel_for_n(n_val, gpu_arch)
            )
            
            if was_cached:
                print(f"✓ Using cached kernel ({len(hsaco)} bytes)")
            else:
                print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
            
            # Run all tests with this N using the compiled kernel
            for M, N in n_tests:
                print(f"\n{'='*80}")
                print(f"Test {len(results)+1}/{len(test_configs)}: M={M}, N={N}")
                print(f"{'='*80}")
                
                success = benchmark_per_token_quant(M=M, N=N, hsaco=hsaco, config=config)
                
                results.append({
                    'M': M,
                    'N': N,
                    'success': success,
                })
                
                if not success:
                    print(f"✗ Failed for M={M}, N={N}")

        # Summary
        print(f"\n{'='*80}")
        print(f"Summary of All Tests")
        print(f"{'='*80}")
        print(f"{'M':>6} {'N':>6} {'Status':>8}")
        print(f"{'-'*80}")
        for r in results:
            status = "✓ Pass" if r['success'] else "✗ Fail"
            print(f"{r['M']:6} {r['N']:6} {status:>8}")
        
        all_passed = all(r['success'] for r in results)
        print(f"\n{'='*80}")
        if all_passed:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")
        print(f"{'='*80}\n")

        sys.exit(0 if all_passed else 1)
    else:
        # Single test mode
        success = benchmark_per_token_quant(M=args.tokens, N=args.hidden)
        if not success:
            sys.exit(1)
