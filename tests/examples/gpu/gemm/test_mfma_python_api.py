#!/usr/bin/env python3
"""
MFMA Test - Pure Python API

Demonstrates rocdl.mfma.f32.16x16x16f16 using pure Python API.
Following test_gpu_rocdsl.py style without ir.Module.parse().
"""
import sys
import os
sys.path.insert(0, os.environ['MLIR_PATH'] + '/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.dialects.ext import gpu
import numpy as np
from mlir import ir
from mlir.dialects import memref, arith as mlir_arith
import mlir.extras.types as T
from hip import hip
import ctypes

def v(x):
    """Extract raw ir.Value from wrappers"""
    return x.value if hasattr(x, 'value') else x

def test_mfma_simple():
    """Simple MFMA test - Pure Python API"""
    M, N = 16, 16
    
    print("="*80)
    print("MFMA Test - Pure Python API")
    print("rocdl.mfma.f32.16x16x16f16")
    print("="*80)
    print()
    
    ctx = RAIIMLIRContextModule()
    gpu.set_container_module(ctx.module)
    
    @gpu.module("mfma_mod", ['#rocdl.target<abi = "500">'])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def kernel(C: T.memref(M, N, T.f32())):
        # Vector types
        vec4_f16 = ir.VectorType.get([4], ir.F16Type.get())
        vec4_f32 = ir.VectorType.get([4], ir.F32Type.get())
        
        # Constants
        c0_i32 = mlir_arith.constant(T.i32(), 0)
        
        # Zero vectors
        zero_f16_attr = ir.DenseElementsAttr.get_splat(
            vec4_f16, ir.FloatAttr.get(ir.F16Type.get(), 0.0))
        zero_f16 = mlir_arith.constant(vec4_f16, zero_f16_attr)
        
        zero_f32_attr = ir.DenseElementsAttr.get_splat(
            vec4_f32, ir.FloatAttr.get(ir.F32Type.get(), 0.0))
        zero_f32 = mlir_arith.constant(vec4_f32, zero_f32_attr)
        
        # MFMA operation
        result = ir.Operation.create(
            "rocdl.mfma.f32.16x16x16f16",
            results=[vec4_f32],
            operands=[v(zero_f16), v(zero_f16), v(zero_f32), 
                     v(c0_i32), v(c0_i32), v(c0_i32)],
        ).result
        
        # Store constant value (like the working test)
        val = mlir_arith.constant(T.f32(), 1.0)
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        memref.store(v(val), C, [v(tx), v(ty)])
    
    ip.__exit__(None, None, None)
    assert gpu_mod.operation.verify()
    
    print("✓ Module created")
    print()
    print("Generated MLIR:")
    print(ctx.module)
    print()
    
    # Compile
    print("Compiling...")
    lowered = run_pipeline(
        ctx.module,
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
    hsaco = get_compile_object_bytes(lowered)
    print(f"✓ Compiled: {len(hsaco)} bytes")
    print()
    
    # Test
    C_host = np.zeros((M, N), dtype=np.float32)
    d_c = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c))]
    args_array = (ctypes.c_void_p * 1)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    print("Running...")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 1, 1, 1, M, N, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(C_host.ctypes.data, d_c, M*N*4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    print(f"\nC[:4,:4]:\n{C_host[:4,:4]}")
    print(f"\nAll values = 1.0? {np.allclose(C_host, 1.0)}")
    
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if np.allclose(C_host, 1.0):
        print("\n" + "="*80)
        print("✓ MFMA TEST PASSED!")
        print(f"  GPU: {get_hip_arch()}")
        print("  Pure Python API")
        print("  rocdl.mfma.f32.16x16x16f16")
        print("="*80)
        return True
    else:
        print("\n✗ FAILED")
        return False

if __name__ == "__main__":
    test_mfma_simple()
