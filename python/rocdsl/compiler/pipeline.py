"""Pass pipeline builder for RocDSL compiler transformations.

This module provides a fluent API for constructing MLIR pass pipelines
for Rocir dialect lowering and optimization.
"""

from typing import Optional, Union, List
from mlir.ir import Module as ir_Module, Context
from mlir.passmanager import PassManager

from rocdsl.compiler.context import ensure_rocir_python_extensions


class RocDSLCompilerError(Exception):
    """Exception raised when a pass pipeline fails."""
    pass


class Pipeline:
    """Fluent API for building MLIR pass pipelines.
    
    Example:
        >>> pipeline = (Pipeline()
        ...     .rocir_coord_lowering()
        ...     .rocir_to_standard()
        ...     .Func(Pipeline().canonicalize().cse())
        ...     .symbol_dce())
        >>> result = pipeline.run(module)
    """
    
    def __init__(self, passes: Optional[List[str]] = None):
        """Initialize a new pipeline.

        Args:
            passes: Optional list of pass strings to start with.
        """
        self._passes: List[str] = passes if passes is not None else []
        self._parent_op: Optional[str] = None
    
    def add_pass(self, pass_name: str, **options) -> "Pipeline":
        """Add a pass with optional arguments.
        
        Args:
            pass_name: Name of the pass to add.
            **options: Pass-specific options as keyword arguments.
        
        Returns:
            Self for method chaining.
        """
        if options:
            # Use space separation for MLIR pass options
            opts = " ".join(f"{k}={v}" for k, v in options.items())
            self._passes.append(f"{pass_name}{{{opts}}}")
        else:
            self._passes.append(pass_name)
        return self

    def nest(self, op_name: str, nested_pipeline: "Pipeline") -> "Pipeline":
        """Add a nested pipeline for a specific operation type.
        
        Args:
            op_name: Operation name (e.g., 'func.func', 'gpu.module').
            nested_pipeline: Pipeline to nest.
        
        Returns:
            Self for method chaining.
        """
        nested_str = ",".join(nested_pipeline._passes)
        self._passes.append(f"{op_name}({nested_str})")
        return self

    def Func(self, nested_pipeline: "Pipeline") -> "Pipeline":
        """Add a nested pipeline for func.func operations."""
        return self.nest("func.func", nested_pipeline)
    
    def Gpu(self, nested_pipeline: "Pipeline") -> "Pipeline":
        """Add a nested pipeline for gpu.module operations."""
        return self.nest("gpu.module", nested_pipeline)
    
    def Module(self, nested_pipeline: "Pipeline") -> "Pipeline":
        """Add a nested pipeline for builtin.module operations."""
        return self.nest("builtin.module", nested_pipeline)
    
    # ========================================================================
    # Rocir-specific passes
    # ========================================================================
    
    def rocir_coord_lowering(self) -> "Pipeline":
        """Lower Rocir coordinate operations to arithmetic."""
        return self.add_pass("rocir-coord-lowering")
    
    def rocir_to_standard(self) -> "Pipeline":
        """Lower Rocir dialect operations to standard dialects."""
        return self.add_pass("rocir-to-standard")
    
    def cute_to_rocm(self) -> "Pipeline":
        """Lower Rocir layout dialect to the ROCm dialect."""
        return self.add_pass("cute-to-rocm")
    
    def cute_layout_canonicalize(self) -> "Pipeline":
        """Canonicalize Rocir layout operations."""
        return self.add_pass("cute-layout-canonicalize")
    
    def cute_tensor_partition(self) -> "Pipeline":
        """Partition tensors for parallel execution."""
        return self.add_pass("cute-tensor-partition")
    
    def cute_nvgpu_to_nvgpu(self, target_arch: Optional[str] = None, enable_tma: Optional[bool] = None) -> "Pipeline":
        """Transform NVGPU operations.
        
        Args:
            target_arch: Target architecture (e.g., "sm_80", "sm_90")
            enable_tma: Enable Tensor Memory Accelerator
        """
        options = {}
        if target_arch is not None:
            options["target-arch"] = target_arch
        if enable_tma is not None:
            options["enable-tma"] = 1 if enable_tma else 0
        return self.add_pass("cute-nvgpu-to-nvgpu", **options)
    
    def cute_nvgpu_mma_lowering(self) -> "Pipeline":
        """Lower NVGPU MMA operations."""
        return self.add_pass("cute-nvgpu-mma-lowering")
    
    def cute_nvgpu_copy_lowering(self) -> "Pipeline":
        """Lower NVGPU copy operations."""
        return self.add_pass("cute-nvgpu-copy-lowering")
    
    def cute_layout_fusion(self) -> "Pipeline":
        """Fuse layout operations."""
        return self.add_pass("cute-layout-fusion")
    
    def cute_vectorization(self) -> "Pipeline":
        """Vectorize memory operations."""
        return self.add_pass("cute-vectorization")
    
    def cute_memory_coalescing(self) -> "Pipeline":
        """Coalesce memory accesses."""
        return self.add_pass("cute-memory-coalescing")
    
    def cute_smem_swizzling(self) -> "Pipeline":
        """Apply shared memory swizzling."""
        return self.add_pass("cute-smem-swizzling")
    
    def cute_async_pipeline(self, pipeline_depth: Optional[int] = None) -> "Pipeline":
        """Apply async pipelining.

        Args:
            pipeline_depth: Optional depth for the async pipeline.
        """
        if pipeline_depth is not None:
            return self.add_pass("cute-async-pipeline", pipeline_depth=pipeline_depth)
        return self.add_pass("cute-async-pipeline")
    
    def cute_warp_specialization(self, enable_producer_consumer: Optional[bool] = None) -> "Pipeline":
        """Apply warp-level specialization.

        Args:
            enable_producer_consumer: Enable producer-consumer specialization
        """
        options = {}
        if enable_producer_consumer is not None:
            options["enable-producer-consumer"] = 1 if enable_producer_consumer else 0
        return self.add_pass("cute-warp-specialization", **options)
    
    def cute_layout_analysis(self, print_analysis: bool = False) -> "Pipeline":
        """Run layout analysis.
        
        Args:
            print_analysis: Whether to print analysis results.
        """
        if print_analysis:
            return self.add_pass("cute-layout-analysis", print_analysis=1)
        return self.add_pass("cute-layout-analysis")
    
    def cute_atom_validation(self) -> "Pipeline":
        """Validate layout atom configurations."""
        return self.add_pass("cute-atom-validation")
    
    # ========================================================================
    # Standard MLIR optimization passes
    # ========================================================================
    
    def canonicalize(self) -> "Pipeline":
        """Canonicalize operations."""
        return self.add_pass("canonicalize")
    
    def cse(self) -> "Pipeline":
        """Common subexpression elimination."""
        return self.add_pass("cse")
    
    def inline(self) -> "Pipeline":
        """Inline function calls."""
        return self.add_pass("inline")
    
    def symbol_dce(self) -> "Pipeline":
        """Dead code elimination for symbols."""
        return self.add_pass("symbol-dce")
    
    def loop_invariant_code_motion(self) -> "Pipeline":
        """Hoist loop-invariant code."""
        return self.add_pass("loop-invariant-code-motion")
    
    def sccp(self) -> "Pipeline":
        """Sparse conditional constant propagation."""
        return self.add_pass("sccp")
    
    # ========================================================================
    # Standard MLIR dialect conversion passes
    # ========================================================================
    
    def convert_scf_to_cf(self) -> "Pipeline":
        """Convert SCF to ControlFlow dialect."""
        return self.add_pass("convert-scf-to-cf")
    
    def convert_arith_to_llvm(self) -> "Pipeline":
        """Convert Arith dialect to LLVM dialect."""
        return self.add_pass("convert-arith-to-llvm")
    
    def convert_func_to_llvm(self) -> "Pipeline":
        """Convert Func dialect to LLVM dialect."""
        return self.add_pass("convert-func-to-llvm")
    
    def convert_memref_to_llvm(self) -> "Pipeline":
        """Convert MemRef dialect to LLVM dialect."""
        return self.add_pass("convert-memref-to-llvm")
    
    def convert_gpu_to_rocdl(self, use_bare_ptr_memref_call_conv: Optional[bool] = None, runtime: Optional[str] = None) -> "Pipeline":
        """Convert GPU dialect to ROCDL dialect.

        Args:
            use_bare_ptr_memref_call_conv: Use bare pointer calling convention for memrefs
            runtime: Runtime to target ("HIP", "OpenCL")
        """
        options = {}
        if use_bare_ptr_memref_call_conv is not None:
            options["use-bare-ptr-memref-call-conv"] = 1 if use_bare_ptr_memref_call_conv else 0
        if runtime is not None:
            options["runtime"] = runtime
        return self.add_pass("convert-gpu-to-rocdl", **options)
    
    def convert_gpu_to_nvvm(self) -> "Pipeline":
        """Convert GPU dialect to NVVM dialect."""
        return self.add_pass("convert-gpu-to-nvvm")
    
    def reconcile_unrealized_casts(self) -> "Pipeline":
        """Reconcile unrealized conversion casts."""
        return self.add_pass("reconcile-unrealized-casts")
    
    def gpu_to_rocdl(self) -> "Pipeline":
        """Convert GPU dialect operations to ROCDL dialect operations."""
        # This pass MUST be run on the gpu.module, not the builtin.module
        return self.nest("gpu.module", Pipeline().add_pass("convert-gpu-to-rocdl"))

    def memref_to_llvm(self) -> "Pipeline":
        """Convert memref ops/types to LLVM."""
        return self.add_pass("convert-memref-to-llvm")
    
    def gpu_to_llvm(self, use_bare_ptr_memref_call_conv: Optional[bool] = None) -> "Pipeline":
        """Convert GPU-related types/ops to LLVM."""
        options = {}
        if use_bare_ptr_memref_call_conv is not None:
            options["use-bare-ptr-memref-call-conv"] = 1 if use_bare_ptr_memref_call_conv else 0
        return self.add_pass("gpu-to-llvm", **options)
    
    def rocdl_to_binary(self) -> "Pipeline":
        """Convert ROCDL module to binary (HSACO)."""
        # This wraps the gpu-module-to-binary pass for AMD
        return self.add_pass("gpu-module-to-binary", format="bin")
    
    def lower_to_llvm(self) -> "Pipeline":
        """Lower to LLVM dialect (alias for convert-func-to-llvm)."""
        return self.add_pass("convert-func-to-llvm")
    
    def rocdl_attach_target(self, chip: Optional[str] = None, features: Optional[str] = None) -> "Pipeline":
        """Attach ROCDL target for AMD GPU compilation.

        Args:
            chip: Target chip architecture (e.g., "gfx908", "gfx90a", "gfx942")
            features: Additional target features
        """
        options = {}
        if chip is not None:
            options["chip"] = chip
        if features is not None:
            options["features"] = features
        return self.add_pass("rocdl-attach-target", **options)
    
    def gpu_module_to_binary(self, format: Optional[str] = None, toolkit_path: Optional[str] = None) -> "Pipeline":
        """Compile GPU module to binary.
        
        Args:
            format: Output format ("bin", "fatbin", "isa")
            toolkit_path: Path to GPU toolkit
        """
        options = {}
        if format is not None:
            options["format"] = format
        if toolkit_path is not None:
            options["toolkit-path"] = toolkit_path
        return self.add_pass("gpu-module-to-binary", **options)
    
    # ========================================================================
    # Pipeline composition
    # ========================================================================
    
    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine two pipelines using + operator.
        
        Returns:
            New pipeline with passes from both pipelines.
        """
        return Pipeline(self._passes + other._passes)
    
    def __iadd__(self, other: "Pipeline") -> "Pipeline":
        """Append another pipeline using += operator.
        
        Returns:
            Self with passes from other pipeline appended.
        """
        self._passes.extend(other._passes)
        return self

    def __str__(self) -> str:
        """Get the pipeline string representation."""
        if not self._passes:
            return "builtin.module()"
        
        passes_str = ",".join(self._passes)
        return f"builtin.module({passes_str})"
    
    def __repr__(self) -> str:
        """Get detailed representation."""
        return f"Pipeline({self._passes!r})"
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    def run(self, module: ir_Module, context: Optional[Context] = None) -> ir_Module:
        """Execute the pipeline on a module.
        
        Args:
            module: MLIR module to transform.
            context: Optional MLIR context (uses module's context if not provided).
        
        Returns:
            The transformed module (same instance, modified in-place).
        
        Raises:
            RocDSLCompilerError: If the pipeline fails.
        """
        ctx = context or module.context
        ensure_rocir_python_extensions(ctx)
        
        pipeline_str = str(self)
        
        try:
            pm = PassManager.parse(pipeline_str, context=ctx)
            with ctx:
                pm.run(module.operation)
        except RuntimeError as exc:
            raise RocDSLCompilerError(
                f"Pipeline failed: {exc}\nPipeline: {pipeline_str}"
            ) from exc
        
        return module
    
    def to_string(self) -> str:
        """Get the pipeline string (alias for __str__)."""
        return str(self)
    
    # ========================================================================
    # Convenience/Recipe Methods
    # ========================================================================
    
    def lower_cute_nvgpu_to_nvgpu(self, target_arch: Optional[str] = None, enable_pipeline: Optional[bool] = None) -> "Pipeline":
        """Convenience method for NVGPU lowering pipeline.

        Args:
            target_arch: Target architecture
            enable_pipeline: Enable async pipeline
        """
        return self.cute_nvgpu_to_nvgpu(target_arch=target_arch).cute_nvgpu_mma_lowering().cute_nvgpu_copy_lowering()


# ========================================================================
# Convenience functions
# ========================================================================

def run_pipeline(module: ir_Module, pipeline: Union[Pipeline, str]) -> ir_Module:
    """Run a pipeline on a module.
    
        Args:
        module: MLIR module to transform.
        pipeline: Pipeline object or pipeline string.
    
    Returns:
        The transformed module.
    
    Raises:
        RocDSLCompilerError: If the pipeline fails.
    """
    if isinstance(pipeline, str):
        ctx = module.context
        ensure_rocir_python_extensions(ctx)
        
        try:
            pm = PassManager.parse(pipeline, context=ctx)
            with ctx:
                pm.run(module.operation)
        except RuntimeError as exc:
            raise RocDSLCompilerError(
                f"Pipeline failed: {exc}\nPipeline: {pipeline}"
            ) from exc
    else:
        pipeline.run(module)
    
    return module


def lower_rocir_to_standard(module: ir_Module) -> ir_Module:
    """Convenience function to lower Rocir to standard dialects.

        Args:
        module: MLIR module containing Rocir operations.
    
    Returns:
        The transformed module.
    """
    pipeline = Pipeline().rocir_to_standard()
    return pipeline.run(module)


def apply_rocir_coord_lowering(module: ir_Module) -> ir_Module:
    """Apply the rocir-coord-lowering pass (backward compatibility).
    
        Args:
        module: MLIR module containing Rocir coordinate operations.
    
    Returns:
        The transformed module.
    """
    pipeline = Pipeline().rocir_coord_lowering()
    return pipeline.run(module)
