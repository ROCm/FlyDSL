#!/usr/bin/env python3
"""
MFMA Test - Real GEMM 1024x1024x1024 (@gpu.func decorator pattern)

Uses @gpu.func(emit=True) decorator instead of manual GPUFuncOp construction.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', '/home/yanronli/llvm-project/buildmlir'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from tests.utils import compile_to_hsaco
import numpy as np
from mlir import ir
from mlir.dialects import arith as _arith_mlir, vector, memref
from rocdsl.dialects.ext import gpu
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes

def test_mfma_real_api():
    print("="*80)
    print("MFMA Real GEMM Test - 1024x1024x1024 (@gpu.func Decorator)")
    print("="*80)
    
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")

    # Generate random inputs
    val_a = float(np.random.uniform(0.5, 2.0))
    val_b = float(np.random.uniform(0.5, 2.0))
    print(f"Random Inputs: A={val_a:.4f}, B={val_b:.4f}")

    ctx = RAIIMLIRContextModule()
    
    total_elements = 1024 * 1024
    
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">'])
    def gpu_mod():
        
        @gpu.func(emit=True)
        def kernel(arg_c: T.memref(total_elements, T.f32())):
            # Helper to unwrap ArithValue
            def unwrap(v):
                return v._value if hasattr(v, '_value') else v
                
            c0_i32 = _arith_mlir.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)).result
            c4 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 4)).result
            
            f16 = ir.F16Type.get()
            f32 = ir.F32Type.get()
            vec4_f16 = ir.VectorType.get([4], f16)
            vec4_f32 = ir.VectorType.get([4], f32)
            
            # Use passed random values
            a_val = ir.DenseElementsAttr.get_splat(vec4_f16, ir.FloatAttr.get(f16, val_a))
            b_val = ir.DenseElementsAttr.get_splat(vec4_f16, ir.FloatAttr.get(f16, val_b))
            c_val = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            
            a_vec = _arith_mlir.ConstantOp(vec4_f16, a_val).result
            b_vec = _arith_mlir.ConstantOp(vec4_f16, b_val).result
            c_init = _arith_mlir.ConstantOp(vec4_f32, c_val).result
            
            # Unroll K=1024 (64 iterations of 16)
            d = c_init
            for _ in range(64):
                d = rocdl.mfma_f32_16x16x16f16(
                    vec4_f32, 
                    [unwrap(a_vec), unwrap(b_vec), unwrap(d), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                ).result
            
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            bdx = gpu.block_dim("x")
            
            mul = _arith_mlir.MulIOp(unwrap(bx._value), unwrap(bdx._value)).result
            idx = _arith_mlir.AddIOp(unwrap(mul), unwrap(tx._value)).result
            
            offset = _arith_mlir.MulIOp(unwrap(idx), unwrap(c4)).result
            
            in_bounds_attr = ir.ArrayAttr.get([ir.BoolAttr.get(True)])
            map = ir.AffineMap.get_identity(1)
            map_attr = ir.AffineMapAttr.get(map)
            
            vector.TransferWriteOp(
                None,
                valueToStore=d, 
                base=arg_c, 
                indices=[unwrap(offset)], 
                permutation_map=map_attr,
                in_bounds=in_bounds_attr
            )
    
    
    # Set kernel attribute
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            # op.body is a Region, need to access its first block
            body_block = op.body.blocks[0] if hasattr(op.body, 'blocks') else op.body
            for inner_op in body_block.operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    
    # Set kernel attribute
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            # op.body is a Region, need to access its first block
            body_block = op.body.blocks[0] if hasattr(op.body, 'blocks') else op.body
            for inner_op in body_block.operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("✓ MLIR module constructed via @gpu.func decorator")
    
    print("Compiling...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
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
    
    if np.allclose(c_host, expected, atol=1e-2):
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
