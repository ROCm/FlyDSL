"""Shared utilities for GPU testing, compilation, and benchmarking."""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.compiler.rocir_opt_helper import apply_rocir_coord_lowering
from rocdsl.runtime.hip_util import get_hip_arch
from hip import hip
from mlir import ir
import numpy as np
from functools import wraps
import os
import tempfile


def compile_to_hsaco(mlir_module, kernel_name="kernel"):
    """
    Compile MLIR module to HSACO binary for AMD GPUs.
    
    Pipeline:
    1. Apply rocir coordinate lowering (rocir ops -> arithmetic)
    2. Canonicalize and CSE
    3. Attach ROCDL target for current GPU architecture
    4. Convert GPU dialect to ROCDL
    5. Lower to LLVM
    6. Generate binary
    
    Environment Variables:
    - ROCDSL_DUMP_IR=1: Enable IR dumping at each compilation stage
    - ROCDSL_DUMP_DIR=/path/to/dir: Directory to save IR files (default: /tmp/rocdsl_dump)
    - ROCDSL_ENABLE_IR_PRINTING=1: Print IR to console during compilation
    
    Args:
        mlir_module: MLIR module containing GPU kernels
        kernel_name: Name prefix for dumped files (default: "kernel")
        
    Returns:
        bytes: HSACO binary object
    """
    # Check environment variables for IR dumping
    dump_ir = os.environ.get('ROCDSL_DUMP_IR', '0') == '1'
    dump_dir = os.environ.get('ROCDSL_DUMP_DIR', '/tmp/rocdsl_dump')
    enable_ir_printing = os.environ.get('ROCDSL_ENABLE_IR_PRINTING', '0') == '1'
    
    # Create dump directory if needed
    if dump_ir:
        os.makedirs(dump_dir, exist_ok=True)
        print(f"  📁 IR dump directory: {dump_dir}")
    
    def dump_stage(module, stage_name):
        """Helper function to dump IR at a specific stage."""
        if dump_ir:
            ir_str = str(module)
            filename = os.path.join(dump_dir, f"{kernel_name}_{stage_name}.mlir")
            with open(filename, 'w') as f:
                f.write(ir_str)
            print(f"  📝 Dumped {stage_name}: {filename}")
            
            if enable_ir_printing:
                print(f"\n{'='*80}")
                print(f"IR Stage: {stage_name}")
                print(f"{'='*80}")
                print(ir_str)
                print(f"{'='*80}\n")
    
    # Dump initial IR (with rocir ops)
    dump_stage(mlir_module, "01_initial")
    
    # Apply rocir coordinate lowering first
    lowered_module = apply_rocir_coord_lowering(mlir_module)
    dump_stage(lowered_module, "02_rocir_lowered")
    
    # Get the current GPU architecture
    gpu_arch = get_hip_arch()
    
    # Build pipeline step by step for intermediate dumps
    # Stage 1: Canonicalize
    canonicalized = run_pipeline(
        lowered_module,
        Pipeline().canonicalize()
    )
    dump_stage(canonicalized, "03_canonicalized")
    
    # Stage 2: CSE
    cse_result = run_pipeline(
        canonicalized,
        Pipeline().cse()
    )
    dump_stage(cse_result, "04_cse")
    
    # Stage 3: Attach ROCDL target
    with_target = run_pipeline(
        cse_result,
        Pipeline().rocdl_attach_target(chip=gpu_arch)
    )
    dump_stage(with_target, "05_with_target")
    
    # Stage 4: Convert SCF to CF before GPU lowering (structured control flow to control flow)
    cf_converted = run_pipeline(
        with_target,
        Pipeline().convert_scf_to_cf()
    )
    dump_stage(cf_converted, "06_scf_to_cf")
    
    # Stage 5: Convert GPU to ROCDL
    rocdl_converted = run_pipeline(
        cf_converted,
        Pipeline().Gpu(
            Pipeline()
            .convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")
        )
    )
    dump_stage(rocdl_converted, "07_rocdl")
    
    # Stage 6: GPU to LLVM  
    llvm_converted = run_pipeline(
        rocdl_converted,
        Pipeline().gpu_to_llvm()
    )
    dump_stage(llvm_converted, "08_llvm_gpu")
    
    # Stage 7: Lower to LLVM (includes CF to LLVM conversion)
    llvm_lowered = run_pipeline(
        llvm_converted,
        Pipeline().lower_to_llvm()
    )
    dump_stage(llvm_lowered, "09_llvm_final")
    
    # Stage 8: Generate binary (ISA/assembly)
    # For assembly dump, we need to use a different format first
    if dump_ir:
        try:
            # Try to get assembly output
            asm_module = run_pipeline(
                ir.Module.parse(str(llvm_lowered), context=llvm_lowered.context),
                Pipeline().gpu_module_to_binary(format="isa")
            )
            from rocdsl.dialects.ext.gpu import get_compile_object_bytes
            asm_bytes = get_compile_object_bytes(asm_module)
            asm_filename = os.path.join(dump_dir, f"{kernel_name}_10_assembly.s")
            with open(asm_filename, 'wb') as f:
                f.write(asm_bytes)
            print(f"  📝 Dumped assembly: {asm_filename}")
        except Exception as e:
            print(f"  ⚠️  Could not dump assembly: {e}")
    
    # Final binary generation
    lowered = run_pipeline(
        llvm_lowered,
        Pipeline().gpu_module_to_binary(format="bin")
    )
    dump_stage(lowered, "11_binary_module")
    
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    hsaco_bytes = get_compile_object_bytes(lowered)
    
    # Save HSACO binary if dumping
    if dump_ir:
        hsaco_filename = os.path.join(dump_dir, f"{kernel_name}_12_final.hsaco")
        with open(hsaco_filename, 'wb') as f:
            f.write(hsaco_bytes)
        print(f"  📝 Dumped HSACO binary: {hsaco_filename}")
    
    return hsaco_bytes


class BenchmarkResults:
    """Container for benchmark timing results."""
    
    def __init__(self, times_ms, size, dtype_bytes=4, total_bytes=None):
        """
        Initialize benchmark results.
        
        Args:
            times_ms: List of execution times in milliseconds
            size: Number of elements processed (or total bytes if used that way)
            dtype_bytes: Size of each element in bytes (default: 4 for float32)
            total_bytes: Total bytes transferred (overrides calculation from size)
        """
        self.times_ms = times_ms
        self.size = size
        self.dtype_bytes = dtype_bytes
        self._total_bytes = total_bytes
        
    @property
    def avg_ms(self):
        """Average execution time in milliseconds."""
        return np.mean(self.times_ms)
    
    @property
    def min_ms(self):
        """Minimum execution time in milliseconds."""
        return np.min(self.times_ms)
    
    @property
    def max_ms(self):
        """Maximum execution time in milliseconds."""
        return np.max(self.times_ms)
    
    @property
    def std_ms(self):
        """Standard deviation of execution times in milliseconds."""
        return np.std(self.times_ms)
    
    @property
    def bandwidth_gbs(self):
        """Calculate achieved bandwidth in GB/s."""
        if self._total_bytes is not None:
            total_bytes = self._total_bytes
        else:
            # For vector operations like A + B = C, we have 3 memory operations
            total_bytes = 3 * self.size * self.dtype_bytes
            
        return (total_bytes / 1e9) / (self.avg_ms / 1000)
    
    def __str__(self):
        """Format benchmark results as a string."""
        result = "  Benchmark Results:\n"
        result += f"  Average Time:  {self.avg_ms:.3f} ms\n"
        result += f"  Min Time:      {self.min_ms:.3f} ms\n"
        result += f"  Max Time:      {self.max_ms:.3f} ms\n"
        result += f"  Std Dev:       {self.std_ms:.3f} ms\n"
        result += f"  Bandwidth:     {self.bandwidth_gbs:.2f} GB/s"
        return result


def perftest(func):
    """
    Decorator for benchmarking GPU kernels.
    
    The decorated function should return a tuple containing:
        (kernel_func, args, grid_dims, block_dims, size)
    
    The decorator will:
    1. Run warmup iterations
    2. Run benchmark iterations with timing
    3. Return BenchmarkResults object
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get kernel configuration from the decorated function
        # func returns: (kernel_func, args, grid_dims, block_dims, size_or_bytes)
        # We can't change the signature easily, so we check if size seems to be total bytes
        # or if we can adapt.
        # For now, just assume existing behavior unless total_bytes kwarg is added to perftest?
        # No, perftest decorator calls func().
        
        ret = func(*args, **kwargs)
        if len(ret) == 5:
            kernel_func, kernel_args, grid_dims, block_dims, size = ret
            total_bytes = None
        elif len(ret) == 6:
            kernel_func, kernel_args, grid_dims, block_dims, size, total_bytes = ret
        else:
            raise ValueError("Decorated function must return 5 or 6 values")
        
        # Warmup iterations
        warmup_iters = 5
        print(f"\n  Running {warmup_iters} warmup iterations...")
        
        if callable(kernel_func):
             for i in range(warmup_iters):
                kernel_func(*kernel_args)
        else:
            for i in range(warmup_iters):
                hip.hipModuleLaunchKernel(
                    kernel_func,
                    *grid_dims,
                    *block_dims,
                    sharedMemBytes=0,
                    stream=None,
                    kernelParams=kernel_args,
                    extra=None
                )
        hip.hipDeviceSynchronize()
        
        # Benchmark iterations
        num_iters = 100
        print(f"  Running {num_iters} benchmark iterations...")
        
        times_ms = []
        for i in range(num_iters):
            # Create events for timing
            start_event = hip.hipEventCreate()[1]
            stop_event = hip.hipEventCreate()[1]
            
            # Record start event
            hip.hipEventRecord(start_event, None)
            
            # Launch kernel
            if callable(kernel_func):
                kernel_func(*kernel_args)
            else:
                hip.hipModuleLaunchKernel(
                    kernel_func,
                    *grid_dims,
                    *block_dims,
                    sharedMemBytes=0,
                    stream=None,
                    kernelParams=kernel_args,
                    extra=None
                )
            
            # Record stop event
            hip.hipEventRecord(stop_event, None)
            
            # Wait for completion
            hip.hipEventSynchronize(stop_event)
            
            # Get elapsed time
            elapsed_ms = hip.hipEventElapsedTime(start_event, stop_event)[1]
            times_ms.append(elapsed_ms)
            
            # Cleanup events
            hip.hipEventDestroy(start_event)
            hip.hipEventDestroy(stop_event)
        
        return BenchmarkResults(times_ms, size, total_bytes=total_bytes)
    
    return wrapper

