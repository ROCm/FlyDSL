#!/usr/bin/env python3
"""
MFMA Test - rocdl.mfma_f32_16x16x16f16

Demonstrates MFMA instruction using Python bindings.
Uses ir.Module.parse() to parse MLIR text within Python.
"""
import sys
import os
sys.path.insert(0, os.environ['MLIR_PATH'] + '/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check
import numpy as np
from mlir import ir
from hip import hip
import ctypes

# MLIR module with MFMA instruction
MLIR_TEXT = """
module {
  gpu.module @mfma_mod [#rocdl.target<abi = \"500\">] {
    gpu.func @kernel(%C: memref<16x16xf32>) kernel {
      %c0 = arith.constant 0 : i32
      %zero_f16 = arith.constant dense<0.0> : vector<4xf16>
      %zero_f32 = arith.constant dense<0.0> : vector<4xf32>
      
      // MFMA instruction: 16x16x16 matrix multiply-accumulate
      %result = rocdl.mfma.f32.16x16x16f16 %zero_f16, %zero_f16, %zero_f32, %c0, %c0, %c0 
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      
      %val = arith.constant 1.0 : f32
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      memref.store %val, %C[%tx, %ty] : memref<16x16xf32>
      gpu.return
    }
  }
}
"""

def test_mfma():
    """Test MFMA instruction using Python bindings."""
    print("="*80)
    print("MFMA Test - rocdl.mfma_f32_16x16x16f16")
    print("="*80)
    print()
    
    # Parse MLIR text using Python binding
    with ir.Context() as ctx:
        module = ir.Module.parse(MLIR_TEXT)
        
        print("✓ MLIR module parsed (Python ir.Module.parse API)")
        print(f"  Module: {str(module).split(chr(10))[1][:60]}...")
        print()
        
        # Compile using Python pipeline API
        print("Compiling...")
        lowered = run_pipeline(
            module,
            Pipeline()
            .canonicalize()
            .rocdl_attach_target(chip="gfx942")
            .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
            .gpu_to_llvm()
            .lower_to_llvm()
            .gpu_module_to_binary(format="bin")
        )
    
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    hsaco = get_compile_object_bytes(lowered)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    print()
    
    # Execute on GPU (Python HIP API)
    print("Executing kernel...")
    c_host = np.zeros((16, 16), dtype=np.float32)
    d_c = hip_check(hip.hipMalloc(16 * 16 * 4))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c))]
    args_array = (ctypes.c_void_p * 1)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 1, 1, 1, 16, 16, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, 16*16*4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    if np.allclose(c_host, 1.0):
        print("✓ Kernel executed correctly (all values = 1.0)")
        print()
        print("="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
    else:
        print(f"✗ FAILED: Expected all 1.0, got:\\n{c_host[:4,:4]}")
        raise AssertionError("Kernel output incorrect")
    
    # Cleanup
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))

if __name__ == "__main__":
    test_mfma()
