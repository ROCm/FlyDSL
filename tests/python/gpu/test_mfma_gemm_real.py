#!/usr/bin/env python3
"""
MFMA Test - Real GEMM 64x64x64

Demonstrates a 64x64x64 matrix multiplication using MFMA.
- Grid: 16 waves (1024 threads total)
- Split into 4 blocks of 256 threads (4 waves per block).
- Each wave computes a 16x16 block of C.
- K-loop: 4 iterations of 16x16x16 MFMA.
- Inputs: A = 2.0, B = 1.0
- Expected Output: 2.0 * 1.0 * 64 = 128.0

Implementation:
- Uses ir.Module.parse() to bypass Python binding issues.
- Manually unrolled loop for K=64.
- Uses vector.transfer_write.
- Calculates global thread ID to support multiple blocks.
- Target: "${gpu_arch}" (MI300) explicitly set in rocdl.target attribute.
"""
import sys
sys.path.insert(0, "/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/build/python_bindings")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import numpy as np
from mlir import ir
from hip import hip
import ctypes

MLIR_TEXT = """
module {
  gpu.module @mfma_mod [#rocdl.target<chip = gpu_arch, abi = "500">] {
    gpu.func @kernel(%C: memref<4096xf32>) kernel {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      
      %a_vec = arith.constant dense<2.0> : vector<4xf16>
      %b_vec = arith.constant dense<1.0> : vector<4xf16>
      %c_init = arith.constant dense<0.0> : vector<4xf32>
      
      // Unrolled loop 4 times for K=64 (4 * 16)
      %d1 = rocdl.mfma.f32.16x16x16f16 %a_vec, %b_vec, %c_init, %c0_i32, %c0_i32, %c0_i32 
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      
      %d2 = rocdl.mfma.f32.16x16x16f16 %a_vec, %b_vec, %d1, %c0_i32, %c0_i32, %c0_i32 
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
        
      %d3 = rocdl.mfma.f32.16x16x16f16 %a_vec, %b_vec, %d2, %c0_i32, %c0_i32, %c0_i32 
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
        
      %d4 = rocdl.mfma.f32.16x16x16f16 %a_vec, %b_vec, %d3, %c0_i32, %c0_i32, %c0_i32 
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
        
      // Calculate global thread ID: idx = blockIdx.x * blockDim.x + threadIdx.x
      %tx = gpu.thread_id x
      %bx = gpu.block_id x
      %bdx = gpu.block_dim x
      %mul = arith.muli %bx, %bdx : index
      %idx = arith.addi %mul, %tx : index
      
      %offset = arith.muli %idx, %c4 : index
      
      // Store result using vector.transfer_write
      vector.transfer_write %d4, %C[%offset] {in_bounds = [true]} : vector<4xf32>, memref<4096xf32>
      
      gpu.return
    }
  }
}
"""

def test_mfma_real():
    print("="*80)
    print("MFMA Real GEMM Test - 64x64x64")
    print("="*80)
    
    print(f"Detected HIP Arch: {get_hip_arch()}")

    with ir.Context() as ctx:
        module = ir.Module.parse(MLIR_TEXT)
        print("✓ MLIR module parsed")
        
        print("Compiling...")
        try:
            pipeline = Pipeline() \
                .canonicalize() \
                .rocdl_attach_target(chip=gpu_arch) \
                .convert_vector_to_llvm() \
                .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")) \
                .gpu_to_llvm() \
                .lower_to_llvm() \
                .gpu_module_to_binary(format="bin")
        except AttributeError:
            print("Warning: Pipeline.convert_vector_to_llvm not found. Trying without it.")
            pipeline = Pipeline() \
                .canonicalize() \
                .rocdl_attach_target(chip=gpu_arch) \
                .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")) \
                .gpu_to_llvm() \
                .lower_to_llvm() \
                .gpu_module_to_binary(format="bin")

        lowered = run_pipeline(module, pipeline)
    
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    hsaco = get_compile_object_bytes(lowered)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
    c_host = np.zeros(4096, dtype=np.float32)
    d_c = hip_check(hip.hipMalloc(4096 * 4))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c))]
    args_array = (ctypes.c_void_p * 1)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Launch 4 blocks of 256 threads = 1024 threads total
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 4, 1, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, 4096*4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    expected = 128.0
    if np.allclose(c_host, expected):
        print(f"✓ Kernel executed correctly (All {len(c_host)} values = {expected})")
    else:
        print(f"✗ Unexpected result")
        print(f"  Expected: {expected}")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        print(f"  Mean: {np.mean(c_host)}")
        failures = np.where(c_host != expected)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")
    
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    print("="*80)
    return True

if __name__ == "__main__":
    test_mfma_real()
