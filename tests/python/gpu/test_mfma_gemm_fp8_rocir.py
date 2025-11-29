import sys

import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', '/home/yanronli/llvm-project/buildmlir'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.runtime.fp8_util import to_byte
import numpy as np
from mlir import ir
from mlir.dialects import gpu, arith, vector, memref, builtin, scf
import mlir.dialects.rocdl as rocdl
from hip import hip
import ctypes

def unwrap(v):
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    return v


def construct_module():
    import sys
    print("DEBUG: construct_module called", file=sys.stderr)
    loc = ir.Location.unknown()
    ctx = loc.context
    ctx.allow_unregistered_dialects = True
    
    # Register Rocir dialect
    try:
        import rocdsl.dialects.ext.rocir
        import _rocirPassesExt
        _rocirPassesExt.register_dialect(ctx)
        print("Successfully registered Rocir dialect via _rocirPassesExt", file=sys.stderr)
        
        try:
            d = ctx.get_or_load_dialect("rocir")
            print(f"Dialect rocir loaded: {d}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load dialect rocir: {e}", file=sys.stderr)
            
    except ImportError as e:
        print(f"Warning: Could not import _rocirPassesExt: {e}", file=sys.stderr)
    except AttributeError:
        print("Warning: _rocirPassesExt does not have register_dialect", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error registering dialect: {e}", file=sys.stderr)

    # Register other dialects
    ctx.load_all_available_dialects()
    # memref.register_dialect(ctx)
    # gpu.register_dialect(ctx)
    # scf.register_dialect(ctx)
    # vector.register_dialect(ctx)
    # rocdl.register_dialect(ctx)

    with loc:
        module = ir.Module.create(loc=loc)
        targets_attr = ir.Attribute.parse('[#rocdl.target<chip = "gfx942", abi = "500", features = "+sramecc,+xnack">]')
        
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp("mfma_mod")
            gpu_mod.attributes["targets"] = targets_attr
        
        gpu_body = gpu_mod.bodyRegion.blocks.append()
        with ir.InsertionPoint(gpu_body):
            f8 = ir.Float8E4M3FNType.get()
            f32 = ir.F32Type.get()
            
            # 1024x1024x1280
            M, N, K = 1024, 1024, 1280
            size_c = M * N
            size_a = M * K
            size_b = N * K # Transposed B (NxK)
            
            memref_type_c = ir.MemRefType.get([size_c], f32)
            memref_type_a = ir.MemRefType.get([size_a], f8)
            memref_type_b = ir.MemRefType.get([size_b], f8)
            
            # LDS Globals (Tile size 32x128)
            lds_mem_type = ir.MemRefType.get([4096], f8, memory_space=ir.Attribute.parse("3"))
            
            lds_a_global = memref.GlobalOp(sym_name="lds_a", type_=lds_mem_type, initial_value=ir.UnitAttr.get(), sym_visibility=ir.StringAttr.get("private"))
            lds_b_global = memref.GlobalOp(sym_name="lds_b", type_=lds_mem_type, initial_value=ir.UnitAttr.get(), sym_visibility=ir.StringAttr.get("private"))
            
            func_type = ir.FunctionType.get(inputs=[memref_type_c, memref_type_a, memref_type_b], results=[])
            func_type_attr = ir.TypeAttr.get(func_type)
            
            gpu_func = gpu.GPUFuncOp(func_type_attr)
            gpu_func.attributes["sym_name"] = ir.StringAttr.get("kernel")
            gpu_func.attributes["gpu.kernel"] = ir.UnitAttr.get()
            
            # Block size 256 (4 waves)
            gpu_func.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get("256,256")
            gpu_func.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])

            func_body = gpu_func.body.blocks.append(*func_type.inputs)
            
            with ir.InsertionPoint(func_body):
                arg_c = func_body.arguments[0]
                arg_a = func_body.arguments[1]
                arg_b = func_body.arguments[2]
                
                c0 = arith.ConstantOp(ir.IndexType.get(), 0).result
                c1 = arith.ConstantOp(ir.IndexType.get(), 1).result
                c_m = arith.ConstantOp(ir.IndexType.get(), 1024).result
                c_n = arith.ConstantOp(ir.IndexType.get(), 1024).result
                c_k = arith.ConstantOp(ir.IndexType.get(), 1280).result
                c128 = arith.ConstantOp(ir.IndexType.get(), 128).result
                c32 = arith.ConstantOp(ir.IndexType.get(), 32).result
                c16 = arith.ConstantOp(ir.IndexType.get(), 16).result
                import sys; print(f'DEBUG: c16: {c16}, type: {type(c16)}', file=sys.stderr, flush=True)
                c8 = arith.ConstantOp(ir.IndexType.get(), 8).result
                c4 = arith.ConstantOp(ir.IndexType.get(), 4).result
                c2 = arith.ConstantOp(ir.IndexType.get(), 2).result
                c64 = arith.ConstantOp(ir.IndexType.get(), 64).result
                c0_i32 = arith.ConstantOp(ir.IntegerType.get_signless(32), 0).result
                identity_map = ir.AffineMap.get_identity(1)
                
                # Define Layouts using Rocir
                # Layout A: (M, K) with stride (K, 1)
                shape_a = rocir.make_shape(c_m, c_k)
                stride_a = rocir.make_stride(c_k, c1)
                layout_a = rocir.make_layout(shape_a, stride_a)
                
                # Layout B (Transposed): (N, K) with stride (K, 1)
                shape_b = rocir.make_shape(c_n, c_k)
                stride_b = rocir.make_stride(c_k, c1)
                layout_b = rocir.make_layout(shape_b, stride_b)
                
                # Layout C: (M, N) with stride (N, 1)
                shape_c = rocir.make_shape(c_m, c_n)
                stride_c = rocir.make_stride(c_n, c1)
                layout_c = rocir.make_layout(shape_c, stride_c)
                
                # LDS Layout: 32x128 (Row Major)
                # Used for storing tiles of A and B
                shape_lds = rocir.make_shape(c32, c128)
                stride_lds = rocir.make_stride(c128, c1)
                layout_lds = rocir.make_layout(shape_lds, stride_lds)

                tx = gpu.ThreadIdOp(gpu.Dimension.x).result
                import sys; print(f'DEBUG: tx: {tx}, type: {type(tx)}', file=sys.stderr, flush=True)
                bx = gpu.BlockIdOp(gpu.Dimension.x).result
                by = gpu.BlockIdOp(gpu.Dimension.y).result
                
                lds_a = memref.GetGlobalOp(lds_mem_type, ir.FlatSymbolRefAttr.get("lds_a")).result
                lds_b = memref.GetGlobalOp(lds_mem_type, ir.FlatSymbolRefAttr.get("lds_b")).result
                
                # Accumulator Init
                vec4_f32 = ir.VectorType.get([4], f32)
                zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
                acc_init = arith.ConstantOp(vec4_f32, zero_attr).result
                
                # Global Load Indices
                tx_16 = tx * c16
                
                row_a_local = tx_16 // c128
                col_a_local = tx_16 % c128
                
                bx_32 = bx * c32
                row_a_global = bx_32 + row_a_local
                
                # For B (Transposed)
                row_b_local = tx_16 // c128
                col_b_local = tx_16 % c128
                
                by_32 = by * c32
                row_b_global = by_32 + row_b_local
                
                # LDS Write Index (Linear)
                # We can use crd2idx with LDS layout to verify, but tx_16 is already linear 
                # relative to the tile start if we view the tile as 1D.
                # But let's stick to the original logic for LDS write index as it's just a vector store
                lds_write_idx = tx_16
                
                vec16_f8 = ir.VectorType.get([16], f8)
                pad_f8 = arith.ConstantOp(f8, ir.FloatAttr.get(f8, 0.0)).result
                
                # Pre-calculate LDS read indices
                wave_id = tx // c64
                lane_id = tx % c64
                
                wave_row = wave_id // c2
                wave_col = wave_id % c2
                
                # A Index: (wave_row * 16 + lane_id % 16) * 128 + (ki + (lane_id // 16) * 8)
                lane_mod_16 = lane_id % c16
                lane_div_16 = lane_id // c16
                
                row_a_lds_base = wave_row * c16
                row_a_lds = row_a_lds_base + lane_mod_16
                
                col_offset_base = lane_div_16 * c8
                
                # B Index: (wave_col * 16 + lane_id % 16) * 128 + (ki + (lane_id // 16) * 8)
                row_b_lds_base = wave_col * c16
                row_b_lds = row_b_lds_base + lane_mod_16
                
                # Main Loop K
                loop = scf.ForOp(unwrap(c0), unwrap(c_k), unwrap(c128), iter_args=[unwrap(acc_init)])
                with ir.InsertionPoint(loop.body):
                    k = loop.induction_variable
                    current_acc = loop.inner_iter_args[0]
                    
                    # Load A using Rocir
                    col_a_global_k = k + col_a_local
                    coord_a = rocir.make_coord(row_a_global, col_a_global_k)
                    idx_a = rocir.crd2idx(coord_a, layout_a)
                    
                    vec_a = vector.TransferReadOp(vec16_f8, arg_a, [unwrap(idx_a)], identity_map, unwrap(pad_f8), [True]).result
                    vector.StoreOp(vec_a, lds_a, [unwrap(lds_write_idx)])
                    
                    # Load B (Transposed) using Rocir
                    col_b_global_k = k + col_b_local
                    coord_b = rocir.make_coord(row_b_global, col_b_global_k)
                    idx_b = rocir.crd2idx(coord_b, layout_b)
                    
                    vec_b = vector.TransferReadOp(vec16_f8, arg_b, [unwrap(idx_b)], identity_map, unwrap(pad_f8), [True]).result
                    vector.StoreOp(vec_b, lds_b, [unwrap(lds_write_idx)])
                    
                    gpu.BarrierOp()
                    
                    # Inner Loop
                    inner_loop = scf.ForOp(unwrap(c0), unwrap(c128), unwrap(c32), iter_args=[unwrap(current_acc)])
                    with ir.InsertionPoint(inner_loop.body):
                        ki = inner_loop.induction_variable
                        curr_acc_inner = inner_loop.inner_iter_args[0]
                        
                        # Calculate LDS indices using Rocir
                        col_lds = ki + col_offset_base
                        
                        # A LDS Index
                        coord_a_lds = rocir.make_coord(row_a_lds, col_lds)
                        idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                        
                        # B LDS Index
                        coord_b_lds = rocir.make_coord(row_b_lds, col_lds)
                        idx_b_mfma = rocir.crd2idx(coord_b_lds, layout_lds)
                        
                        vec8_f8 = ir.VectorType.get([8], f8)
                        vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                        vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                        
                        vec_a_load = vector.LoadOp(vec8_f8, lds_a, [unwrap(idx_a_mfma)]).result
                        vec_b_load = vector.LoadOp(vec8_f8, lds_b, [unwrap(idx_b_mfma)]).result
                        
                        a_bytes = arith.BitcastOp(vec8_i8, vec_a_load).result
                        b_bytes = arith.BitcastOp(vec8_i8, vec_b_load).result
                        
                        a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                        b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                        
                        a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                        b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                        
                        new_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            vec4_f32, [unwrap(a_pack), unwrap(b_pack), curr_acc_inner, unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                        ).result
                        
                        scf.YieldOp([new_acc])
                        
                    gpu.BarrierOp()
                    scf.YieldOp([inner_loop.results[0]])
                
                final_acc = loop.results[0]
                
                # Store Result using Rocir
                lane_div_16 = lane_id // c16
                lane_rem_16 = lane_id % c16
                
                row_wave_base = wave_row * c16
                col_wave_base = wave_col * c16
                
                bx_32 = bx * c32
                by_32 = by * c32
                
                row_base_g = bx_32 + row_wave_base
                col_base_g = by_32 + col_wave_base
                
                for i in range(4):
                    val = vector.ExtractOp(final_acc, [], [i]).result
                    
                    # Row offset = (lane_div_16 * 4) + i
                    c_i = arith.ConstantOp(ir.IndexType.get(), i).result
                    row_offset_base = lane_div_16 * c4
                    row_offset = row_offset_base + c_i
                    
                    # Col offset = lane_rem_16
                    col_offset = lane_rem_16
                    
                    # Global indices
                    row_g = row_base_g + row_offset
                    col_g = col_base_g + col_offset
                    
                    # Use Rocir for C store index
                    coord_c = rocir.make_coord(row_g, col_g)
                    idx = rocir.crd2idx(coord_c, layout_c)
                    
                    memref.StoreOp(unwrap(val), arg_c, [unwrap(idx)])
                
                gpu.ReturnOp([])
                
    return module

def test_mfma_fp8_rocir():
    print("="*80)
    print("MFMA Real FP8 GEMM Test (Rocir) - 1024x1024x1280")
    print("="*80)
    
    print(f"Detected HIP Arch: {get_hip_arch()}")

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        
        # Try to ensure rocir dialect is registered
        try:
            import rocir
            # Some bindings expose a register_dialect method
            if hasattr(rocir, "register_dialect"):
                rocir.register_dialect(ctx)
            else:
                # Force loading by accessing the dialect class if available
                if hasattr(rocir, "_Dialect"):
                    d = rocir._Dialect(ctx)
        except Exception as e:
            print(f"Warning: Failed to register rocir dialect: {e}")

        module = construct_module()
        print("✓ MLIR module constructed")
        
        print("Compiling...")
        pipeline = Pipeline() \
            .add_pass("rocir-coord-lowering") \
            .canonicalize() \
            .rocdl_attach_target(chip="gfx942") \
            .convert_vector_to_llvm() \
            .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP", chipset="gfx942")) \
            .gpu_to_llvm() \
            .lower_to_llvm() \
            .gpu_module_to_binary(format="bin")
            
        lowered = run_pipeline(module, pipeline)
    
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    hsaco = get_compile_object_bytes(lowered)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
    
    M, N, K = 1024, 1024, 1280
    size_c = M * N
    size_a = M * K
    size_b = N * K
    
    # Random inputs (0, 1, 2)
    a_host = np.random.randint(-16, 16, size=(M, K)).astype(np.float32)
    b_host = np.random.randint(-16, 16, size=(K, N)).astype(np.float32)
    
    # Transpose B for the kernel (NxK)
    b_host_T = np.ascontiguousarray(b_host.T)
    
    a_bytes = np.array([to_byte(x) for x in a_host.flatten()], dtype=np.uint8)
    b_bytes = np.array([to_byte(x) for x in b_host_T.flatten()], dtype=np.uint8)
    
    c_host = np.zeros(size_c, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(size_a))
    d_b = hip_check(hip.hipMalloc(size_b))
    d_c = hip_check(hip.hipMalloc(size_c * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_bytes.ctypes.data, size_a, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_bytes.ctypes.data, size_b, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c)), ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args_array = (ctypes.c_void_p * 3)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Grid: 32x32 blocks. Block: 256 threads.
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 32, 32, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, size_c * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verification with np.matmul
    print("Computing expected result with np.matmul...")
    expected_matrix = np.matmul(a_host, b_host)
    expected = expected_matrix.flatten()
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    print("="*80)
    print(f"Max Absolute Difference: {np.max(np.abs(c_host - expected))}")
    
    if np.allclose(c_host, expected, atol=1.0):
        print(f"✓ Kernel executed correctly (Matches np.matmul)")
        return True
    else:
        print(f"✗ Unexpected result")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        failures = np.where(np.abs(c_host - expected) > 1.0)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: Expected {expected[failures[0]]}, Got {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")
        raise ValueError("Kernel result does not match expected values")
        return False
    
    

if __name__ == "__main__":
    test_mfma_fp8_rocir()
