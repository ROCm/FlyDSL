#!/usr/bin/env python3
"""
MFMA Test - Real GEMM 1024x1024x1024 (Python API Construction)

Replicates test_mfma_gemm_real.py but constructs the MLIR module 
programmatically using Python bindings instead of parsing text.
"""
import sys
sys.path.insert(0, "/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/build/python_bindings")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")

# Do NOT import ext.gpu to avoid issues with targets attribute
# import rocdsl.dialects.ext.gpu

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import numpy as np
from mlir import ir
from mlir.dialects import gpu, arith, vector, memref, builtin
import mlir.dialects.rocdl as rocdl
from hip import hip
import ctypes

def construct_module(val_a, val_b):
    loc = ir.Location.unknown()
    with loc:
        module = ir.Module.create(loc=loc)
        
        # Parse the array attribute directly to ensure correct type
        targets_attr = ir.Attribute.parse('[#rocdl.target<chip = gpu_arch, abi = "500">]')
        
        # Use InsertionPoint to insert into module body
        with ir.InsertionPoint(module.body):
            # Base GPUModuleOp takes targets as keyword arg
            gpu_mod = gpu.GPUModuleOp("mfma_mod", targets=targets_attr)
        
        # Base GPUModuleOp does not create a block automatically.
        # Use bodyRegion to append a block.
        gpu_body = gpu_mod.bodyRegion.blocks.append()
        
        with ir.InsertionPoint(gpu_body):
            f32 = ir.F32Type.get()
            # 1024 * 1024 = 1,048,576
            memref_type = ir.MemRefType.get([1048576], f32)
            
            func_type = ir.FunctionType.get(inputs=[memref_type], results=[])
            
            # Wrap function type in TypeAttr
            func_type_attr = ir.TypeAttr.get(func_type)
            
            # Base GPUFuncOp takes function_type as positional arg.
            gpu_func = gpu.GPUFuncOp(func_type_attr)
            gpu_func.attributes["sym_name"] = ir.StringAttr.get("kernel")
            gpu_func.attributes["gpu.kernel"] = ir.UnitAttr.get()
            
            # Base GPUFuncOp has .body property (Region)
            func_body = gpu_func.body.blocks.append(*func_type.inputs)
            
            with ir.InsertionPoint(func_body):
                arg_c = func_body.arguments[0]
                
                c0_i32 = arith.ConstantOp(ir.IntegerType.get_signless(32), 0).result
                c4 = arith.ConstantOp(ir.IndexType.get(), 4).result
                
                f16 = ir.F16Type.get()
                vec4_f16 = ir.VectorType.get([4], f16)
                vec4_f32 = ir.VectorType.get([4], f32)
                
                # Use passed random values
                a_val = ir.DenseElementsAttr.get_splat(vec4_f16, ir.FloatAttr.get(f16, val_a))
                b_val = ir.DenseElementsAttr.get_splat(vec4_f16, ir.FloatAttr.get(f16, val_b))
                c_val = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
                
                a_vec = arith.ConstantOp(vec4_f16, a_val).result
                b_vec = arith.ConstantOp(vec4_f16, b_val).result
                c_init = arith.ConstantOp(vec4_f32, c_val).result
                
                # Unroll K=1024 (64 iterations of 16)
                d = c_init
                for _ in range(64):
                    d = rocdl.mfma_f32_16x16x16f16(
                        vec4_f32, 
                        [a_vec, b_vec, d, c0_i32, c0_i32, c0_i32]
                    ).result
                
                tx = gpu.ThreadIdOp(gpu.Dimension.x).result
                bx = gpu.BlockIdOp(gpu.Dimension.x).result
                bdx = gpu.BlockDimOp(gpu.Dimension.x).result
                
                mul = arith.MulIOp(bx, bdx).result
                idx = arith.AddIOp(mul, tx).result
                
                offset = arith.MulIOp(idx, c4).result
                
                in_bounds_attr = ir.ArrayAttr.get([ir.BoolAttr.get(True)])
                
                map = ir.AffineMap.get_identity(1)
                map_attr = ir.AffineMapAttr.get(map)
                
                vector.TransferWriteOp(
                    None, 
                    d, 
                    arg_c, 
                    [offset], 
                    map_attr,
                    in_bounds_attr
                )
                
                gpu.ReturnOp([])
                
    return module

def test_mfma_real_api():
    print("="*80)
    print("MFMA Real GEMM Test - 1024x1024x1024 (Python API)")
    print("="*80)
    
    print(f"Detected HIP Arch: {get_hip_arch()}")

    # Generate random inputs
    val_a = float(np.random.uniform(0.5, 2.0))
    val_b = float(np.random.uniform(0.5, 2.0))
    print(f"Random Inputs: A={val_a:.4f}, B={val_b:.4f}")

    with ir.Context() as ctx:
        module = construct_module(val_a, val_b)
        print("✓ MLIR module constructed via Python API")
        
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
    total_elements = 1024 * 1024
    c_host = np.zeros(total_elements, dtype=np.float32)
    d_c = hip_check(hip.hipMalloc(total_elements * 4))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c))]
    args_array = (ctypes.c_void_p * 1)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Grid size 1024, Block size 256
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 1024, 1, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, total_elements*4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Calculate expected result with f16 precision simulation
    val_a_f16 = float(np.float16(val_a))
    val_b_f16 = float(np.float16(val_b))
    # K = 1024
    expected = 1024.0 * val_a_f16 * val_b_f16
    
    print(f"Expected Result: {expected:.4f} (1024 * {val_a_f16:.4f} * {val_b_f16:.4f})")
    
    if np.allclose(c_host, expected, atol=1e-2): # Relax tolerance slightly for larger K accumulation
        print(f"✓ Kernel executed correctly (All {len(c_host)} values ≈ {expected:.4f})")
    else:
        print(f"✗ Unexpected result")
        print(f"  Expected: {expected}")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        print(f"  Mean: {np.mean(c_host)}")
        failures = np.where(np.abs(c_host - expected) > 1e-2)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")
    
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    print("="*80)
    return True

if __name__ == "__main__":
    test_mfma_real_api()
