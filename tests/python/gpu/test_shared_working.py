#!/usr/bin/env python3
"""Shared memory matmul - FIXED VERSION"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.compiler.rocir_opt_helper import apply_rocir_coord_lowering
from rocdsl.dialects.ext import gpu, scf
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import arith, memref
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes
import time

def compile_to_hsaco(mlir_module):
    lowered_module = apply_rocir_coord_lowering(mlir_module)
    lowered = run_pipeline(
        lowered_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip="gfx942")
        .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
        .gpu_to_llvm()
        .lower_to_llvm()
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)

M, N, K = 256, 256, 256
TILE_SIZE = 16

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("matmul_shared", [f'#rocdl.target<chip = "{get_hip_arch()}", abi = "500">'])
def gpu_mod():
    pass

ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
ip.__enter__()

# KEY FIX: Use memref.global_ with lds_space()
tile_type = T.memref(TILE_SIZE, TILE_SIZE, T.f32(), memory_space=lds_space())

memref.global_(sym_name="A_shared_tile", type_=tile_type, alignment=16)
memref.global_(sym_name="B_shared_tile", type_=tile_type, alignment=16)

@gpu.func(emit=True)
def matmul_shared(A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())):
    # Get references to shared memory
    As = memref.get_global(tile_type, "A_shared_tile")
    Bs = memref.get_global(tile_type, "B_shared_tile")
    
    row = arith.addi(arith.muli(gpu.block_id("y"), arith.constant(T.index(), TILE_SIZE)), gpu.thread_id("y"))
    col = arith.addi(arith.muli(gpu.block_id("x"), arith.constant(T.index(), TILE_SIZE)), gpu.thread_id("x"))
    
    tx = gpu.thread_id("x")
    ty = gpu.thread_id("y")
    
    zero = arith.constant(T.index(), 0)
    one = arith.constant(T.index(), 1)
    tile_c = arith.constant(T.index(), TILE_SIZE)
    k_c = arith.constant(T.index(), K)
    zero_f = arith.constant(T.f32(), 0.0)
    
    acc = zero_f
    num_tiles = arith.constant(T.index(), K // TILE_SIZE)
    
    for_tiles = scf.ForOp(zero, num_tiles, one, [acc])
    with ir.InsertionPoint(for_tiles.body):
        t = for_tiles.induction_variable
        acc_val = for_tiles.inner_iter_args[0]
        k_base = arith.muli(t, tile_c)
        
        a_col = arith.addi(k_base, tx)
        a_val = memref.load(A, [row, a_col])
        memref.store(a_val, As, [ty, tx])
        
        b_row = arith.addi(k_base, ty)
        b_val = memref.load(B, [b_row, col])
        memref.store(b_val, Bs, [ty, tx])
        
        gpu.barrier()
        
        for_k = scf.ForOp(zero, tile_c, one, [acc_val])
        with ir.InsertionPoint(for_k.body):
            k_local = for_k.induction_variable
            acc_k = for_k.inner_iter_args[0]
            
            a_smem = memref.load(As, [ty, k_local])
            b_smem = memref.load(Bs, [k_local, tx])
            new_acc = arith.addf(acc_k, arith.mulf(a_smem, b_smem))
            
            scf.yield_([new_acc])
        
        gpu.barrier()
        scf.yield_([for_k.results[0]])
    
    memref.store(for_tiles.results[0], C, [row, col])

ip.__exit__(None, None, None)

print("="*80)
print("Compiling shared memory matmul...")
hsaco = compile_to_hsaco(ctx.module)
print(f"✓ HSACO: {len(hsaco)} bytes")

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

start_time = time.time()
for _ in range(10):
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())
avg_time_ms = (time.time() - start_time) * 100

hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

error = np.max(np.abs(c_host - expected))
rel_error = error / (np.max(np.abs(expected)) + 1e-8)
gflops = (2.0 * M * N * K * 1e-9) / (avg_time_ms * 1e-3)

print(f"\n{'='*80}")
print(f"Results (256x256):")
print(f"  Time: {avg_time_ms:.3f} ms")
print(f"  Performance: {gflops:.1f} GFLOPS")
print(f"  Max error: {error:.2e}")
print(f"  Relative error: {rel_error:.2e}")
print(f"  Shared memory: {2 * TILE_SIZE * TILE_SIZE * 4} bytes/block")
print("="*80)

if rel_error < 1e-3:
    print("\n✅ SHARED MEMORY OPTIMIZATION WORKING!")
    print("   Solution: memref.global_ with lds_space() + memref.get_global()")
else:
    print(f"\n❌ Error: {rel_error:.2e}")
    print(expected[:5,:5])
    print(c_host[:5,:5])
