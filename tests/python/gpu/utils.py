"""Shared utilities for GPU testing and compilation."""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.compiler.rocir_opt_helper import apply_rocir_coord_lowering
from rocdsl.runtime.hip_util import get_hip_arch


def compile_to_hsaco(mlir_module):
    """
    Compile MLIR module to HSACO binary for AMD GPUs.
    
    Pipeline:
    1. Apply rocir coordinate lowering (rocir ops -> arithmetic)
    2. Canonicalize and CSE
    3. Attach ROCDL target for current GPU architecture
    4. Convert GPU dialect to ROCDL
    5. Lower to LLVM
    6. Generate binary
    
    Args:
        mlir_module: MLIR module containing GPU kernels
        
    Returns:
        bytes: HSACO binary object
    """
    # Apply rocir coordinate lowering first
    lowered_module = apply_rocir_coord_lowering(mlir_module)
    
    # Get the current GPU architecture
    gpu_arch = get_hip_arch()
    
    # Then run the main GPU compilation pipeline
    lowered = run_pipeline(
        lowered_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip=gpu_arch)
        
        # Lower control flow and standard ops globally first
        # This ensures gpu.module content (gpu.func body) is lowered to LLVM-compatible ops
        # BEFORE we convert the function itself to llvm.func.
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
        .convert_arith_to_llvm()
        .convert_math_to_llvm()
        .convert_vector_to_llvm(force_32bit_vector_indices=True)
        .convert_index_to_llvm()
        .reconcile_unrealized_casts()
        
        # Now convert GPU dialect to ROCDL
        # This handles gpu.func -> llvm.func and memref -> llvm pointers
        .Gpu(Pipeline()
            .convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")
            .reconcile_unrealized_casts()
        )
        
        # Finally serialize
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)
