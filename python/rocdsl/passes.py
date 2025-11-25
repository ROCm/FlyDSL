"""Pass management and lowering utilities for RocDSL.

Provides Pythonic interface for MLIR transformation passes, following
mlir-python-extras patterns.
"""

# Import cute-opt runner for cute-specific passes
try:
    from .cute_opt_runner import run_cute_opt
    HAS_CUTE_OPT = True
except ImportError:
    HAS_CUTE_OPT = False

import logging
import sys
import tempfile
from contextlib import contextmanager
from io import StringIO
from typing import List, Optional, Union

from mlir.ir import Module, StringAttr
from mlir.passmanager import PassManager

logger = logging.getLogger(__name__)


class RocDSLCompilerError(Exception):
    """Exception raised when pass pipeline execution fails."""
    pass


def get_module_name(module: Module) -> str:
    """Extract module name from debug attributes."""
    if "debug_module_name" not in module.operation.attributes:
        return "UnnamedModule"
    return StringAttr(module.operation.attributes["debug_module_name"]).value


def run_pipeline(
    module: Module,
    pipeline: Union[str, "Pipeline"],
    description: Optional[str] = None,
    enable_ir_printing: bool = False,
    print_pipeline: bool = False,
    verify: bool = True,
) -> Module:
    """Run transformation pipeline on module.
    
    Args:
        module: MLIR module to transform
        pipeline: Pass pipeline string or Pipeline object
        description: Human-readable description for error messages
        enable_ir_printing: Print IR after each pass
        print_pipeline: Print the pipeline structure
        verify: Run verification after each pass
        
    Returns:
        Transformed module
        
    Raises:
        RocDSLCompilerError: If pipeline execution fails
        
    Example:
        >>> module = Module.parse(mlir_code)
        >>> lowered = run_pipeline(module, "cute-to-standard")
        >>> print(lowered)
    """
    # Re-parse to get fresh module
    module = Module.parse(module.operation.get_asm(enable_debug_info=True), context=module.context)
    
    if isinstance(pipeline, Pipeline):
        pipeline = str(pipeline)
    
    module_name = get_module_name(module)

    # Check if this is a cute-specific pass that needs cute-opt
    cute_passes = ["cute-to-standard", "cute-to-rocm", "cute-nvgpu-to-nvgpu"]
    needs_cute_opt = any(cp in pipeline for cp in cute_passes)

    if needs_cute_opt and HAS_CUTE_OPT:
        # Use cute-opt for cute-specific passes
        for cute_pass in cute_passes:
            if cute_pass in pipeline:
                try:
                    return run_cute_opt(module, cute_pass)
                except Exception as e:
                    raise RocDSLCompilerError(f"cute-opt execution failed: {e}") from e

    
    try:
        # Capture stderr for error reporting
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        with module.context:
            # Save IR for error report
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10,
                enable_debug_info=True,
            )
            
            # Create and configure pass manager
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(verify)
            
            if print_pipeline:
                print(pm)
            
            if enable_ir_printing:
                pm.enable_ir_printing()
            
            # Run the pipeline
            pm.run(module.operation)
            
    except Exception as e:
        print(e, file=sys.stderr)
        
        # Write failing IR to temp file for debugging
        import os
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"{module_name} compilation"
        
        message = f"""\
{description} failed with diagnostics:

{'=' * 80}
{sys.stderr.getvalue().strip()}
{'=' * 80}

Reproducer:
$ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
        """
        
        raise RocDSLCompilerError(message.strip())
    
    finally:
        sys.stderr = original_stderr
    
    return module


class Pipeline:
    """Fluent API for building MLIR pass pipelines.
    
    Example:
        >>> pipeline = (Pipeline()
        ...     .cute_to_standard()
        ...     .Func(Pipeline().canonicalize())
        ...     .lower_to_llvm())
        >>> print(pipeline)
        builtin.module(cute-to-standard,func.func(canonicalize),...)
    """
    
    def __init__(self, passes: Optional[List[str]] = None):
        self._passes = passes or []
    
    def Nested(self, context: str, pipeline: "Pipeline") -> "Pipeline":
        """Nest a pipeline within a dialect context."""
        inner = pipeline.materialize(module=False)
        self._passes.append(f"{context}({inner})")
        return self
    
    def Func(self, pipeline: "Pipeline") -> "Pipeline":
        """Nest pipeline in func.func context."""
        return self.Nested("func.func", pipeline)
    
    def Gpu(self, pipeline: "Pipeline") -> "Pipeline":
        """Nest pipeline in gpu.module context."""
        return self.Nested("gpu.module", pipeline)
    
    def materialize(self, module: bool = True) -> str:
        """Convert pipeline to string representation."""
        pipeline_str = ",".join(self._passes)
        if module:
            pipeline_str = f"builtin.module({pipeline_str})"
        logger.debug(f"Pipeline: {pipeline_str}")
        return pipeline_str
    
    def __str__(self) -> str:
        return self.materialize()
    
    def __iadd__(self, other: "Pipeline") -> "Pipeline":
        """Extend pipeline in-place."""
        self._passes.extend(other._passes)
        return self
    
    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine pipelines."""
        return Pipeline(self._passes + other._passes)
    
    def add_pass(self, pass_name: str, **kwargs) -> "Pipeline":
        """Add a pass with optional parameters.
        
        Args:
            pass_name: Pass name (underscores converted to hyphens)
            **kwargs: Pass options (bool converted to 0/1)
            
        Example:
            >>> pipeline.add_pass("cute-to-standard")
            >>> pipeline.add_pass("cute-nvgpu-to-nvgpu", target_arch="sm_90")
        """
        # Convert Python conventions to MLIR
        kwargs = {
            k.replace("_", "-"): (int(v) if isinstance(v, bool) else v)
            for k, v in kwargs.items()
            if v is not None
        }
        
        if kwargs:
            args_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            pass_str = f"{pass_name}{{{args_str}}}"
        else:
            pass_str = pass_name
        
        self._passes.append(pass_str)
        return self
    
    # =========================================================================
    # RocDSL/CuTe Passes
    # =========================================================================
    
    def cute_to_standard(self) -> "Pipeline":
        """Lower CuTe IR to standard MLIR dialects (scf, arith, memref)."""
        return self.add_pass("cute-to-standard")
    
    def cute_layout_canonicalize(self) -> "Pipeline":
        """Canonicalize and simplify Layout operations."""
        return self.add_pass("cute-layout-canonicalize")
    
    def cute_tensor_partition(self) -> "Pipeline":
        """Materialize tensor partitioning into explicit indexing."""
        return self.add_pass("cute-tensor-partition")
    
    def cute_nvgpu_to_nvgpu(
        self,
        target_arch: str = "sm_90",
        enable_tma: bool = True
    ) -> "Pipeline":
        """Lower cute_nvgpu_ir to MLIR nvgpu dialect.
        
        Args:
            target_arch: Target GPU (sm_80, sm_90, sm_100)
            enable_tma: Enable TMA operations (SM90+)
        """
        return self.add_pass(
            "cute-nvgpu-to-nvgpu",
            target_arch=target_arch,
            enable_tma=enable_tma
        )
    
    def cute_nvgpu_mma_lowering(self) -> "Pipeline":
        """Lower TiledMma to architecture-specific MMA instructions."""
        return self.add_pass("cute-nvgpu-mma-lowering")
    
    def cute_nvgpu_copy_lowering(self) -> "Pipeline":
        """Lower TiledCopy to architecture-specific copy instructions."""
        return self.add_pass("cute-nvgpu-copy-lowering")
    
    def cute_nvgpu_tma_materialize(self) -> "Pipeline":
        """Materialize TMA descriptor creation and initialization."""
        return self.add_pass("cute-nvgpu-tma-materialize")
    
    def cute_to_rocm(self) -> "Pipeline":
        """Lower CuTe IR to ROCm-specific dialects."""
        return self.add_pass("cute-to-rocm")
    
    # =========================================================================
    # Optimization Passes
    # =========================================================================
    
    def cute_layout_fusion(self) -> "Pipeline":
        """Fuse adjacent Layout transformations."""
        return self.add_pass("cute-layout-fusion")
    
    def cute_vectorization(self) -> "Pipeline":
        """Vectorize tensor copy operations based on Layout."""
        return self.add_pass("cute-vectorization")
    
    def cute_memory_coalescing(self) -> "Pipeline":
        """Optimize memory access patterns for coalescing."""
        return self.add_pass("cute-memory-coalescing")
    
    def cute_smem_swizzling(self) -> "Pipeline":
        """Apply shared memory swizzling to avoid bank conflicts."""
        return self.add_pass("cute-smem-swizzling")
    
    def cute_async_pipeline(
        self,
        pipeline_depth: int = 2,
        warp_specialization: bool = False
    ) -> "Pipeline":
        """Insert async copy pipeline for overlapping compute and data transfer.
        
        Args:
            pipeline_depth: Number of pipeline stages (2-8)
            warp_specialization: Use warp specialization (SM90+)
        """
        return self.add_pass(
            "cute-async-pipeline",
            pipeline_depth=pipeline_depth,
            warp_specialization=warp_specialization
        )
    
    def cute_warp_specialization(
        self,
        num_producer_warps: int = 1
    ) -> "Pipeline":
        """Apply warp specialization for Hopper producer-consumer pattern.
        
        Args:
            num_producer_warps: Number of producer warps (typically 1 for TMA)
        """
        return self.add_pass(
            "cute-warp-specialization",
            num_producer_warps=num_producer_warps
        )
    
    # =========================================================================
    # Analysis Passes
    # =========================================================================
    
    def cute_layout_analysis(self, print_analysis: bool = False) -> "Pipeline":
        """Analyze Layout properties for optimization decisions."""
        return self.add_pass("cute-layout-analysis", print_analysis=print_analysis)
    
    def cute_atom_validation(self) -> "Pipeline":
        """Validate MmaAtom/CopyAtom configurations."""
        return self.add_pass("cute-atom-validation")
    
    # =========================================================================
    # Standard MLIR Passes
    # =========================================================================
    
    def canonicalize(self) -> "Pipeline":
        """Apply canonicalization patterns."""
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
    
    def sccp(self) -> "Pipeline":
        """Sparse conditional constant propagation."""
        return self.add_pass("sccp")
    
    def loop_invariant_code_motion(self) -> "Pipeline":
        """Hoist loop-invariant code."""
        return self.add_pass("loop-invariant-code-motion")
    
    def lower_affine(self) -> "Pipeline":
        """Lower affine dialect to standard."""
        return self.add_pass("lower-affine")
    
    def convert_scf_to_cf(self) -> "Pipeline":
        """Convert SCF to control flow."""
        return self.add_pass("convert-scf-to-cf")
    
    def convert_arith_to_llvm(self) -> "Pipeline":
        """Convert arith dialect to LLVM."""
        return self.add_pass("convert-arith-to-llvm")
    
    def convert_func_to_llvm(self) -> "Pipeline":
        """Convert func dialect to LLVM."""
        return self.add_pass("convert-func-to-llvm")
    
    def convert_cf_to_llvm(self) -> "Pipeline":
        """Convert control flow to LLVM."""
        return self.add_pass("convert-cf-to-llvm")
    
    def convert_memref_to_llvm(self) -> "Pipeline":
        """Convert memref to LLVM."""
        return self.add_pass("convert-memref-to-llvm")
    
    def reconcile_unrealized_casts(self) -> "Pipeline":
        """Remove unrealized conversion casts."""
        return self.add_pass("reconcile-unrealized-casts")
    
    # =========================================================================
    # Pipeline Recipes
    # =========================================================================
    
    def lower_cute_to_standard(self) -> "Pipeline":
        """Complete pipeline: CuTe → Standard dialects.
        
        Returns optimized IR using scf, arith, memref.
        """
        return (
            self
            .cute_to_standard()
            .Func(Pipeline()
                  .cute_layout_canonicalize()
                  .canonicalize()
                  .cse())
        )
    
    def lower_cute_nvgpu_to_nvgpu(
        self,
        target_arch: str = "sm_90",
        enable_pipeline: bool = True
    ) -> "Pipeline":
        """Complete pipeline: cute_nvgpu_ir → nvgpu.
        
        Args:
            target_arch: Target GPU architecture
            enable_pipeline: Add async pipeline transformation
        """
        pipeline = (
            self.Gpu(Pipeline()
                     .cute_nvgpu_to_nvgpu(target_arch=target_arch)
                     .cute_nvgpu_mma_lowering()
                     .cute_nvgpu_copy_lowering())
        )
        
        if enable_pipeline and target_arch >= "sm_80":
            pipeline = pipeline.Gpu(Pipeline().cute_async_pipeline())
        
        return pipeline
    
    def optimize_cute_layouts(self) -> "Pipeline":
        """Apply all layout optimization passes."""
        return (
            self
            .Func(Pipeline()
                  .cute_layout_fusion()
                  .cute_vectorization())
            .Gpu(Pipeline()
                 .cute_memory_coalescing()
                 .cute_smem_swizzling())
        )
    
    def lower_to_llvm(self) -> "Pipeline":
        """Lower to LLVM dialect (standard MLIR pipeline)."""
        return (
            self
            .convert_scf_to_cf()
            .Func(Pipeline()
                  .convert_arith_to_llvm()
                  .convert_func_to_llvm())
            .convert_cf_to_llvm()
            .convert_memref_to_llvm()
            .reconcile_unrealized_casts()
        )


# Convenience functions
def lower_cute_to_standard(module: Module) -> Module:
    """One-step lowering: CuTe → Standard dialects."""
    pipeline = Pipeline().lower_cute_to_standard()
    return run_pipeline(module, pipeline, "CuTe to Standard")


def lower_cute_to_nvgpu(module: Module, target_arch: str = "sm_90") -> Module:
    """One-step lowering: cute_nvgpu_ir → nvgpu."""
    pipeline = Pipeline().lower_cute_nvgpu_to_nvgpu(target_arch=target_arch)
    return run_pipeline(module, pipeline, f"CuTe NVGPU to NVGPU ({target_arch})")


def optimize_layouts(module: Module) -> Module:
    """Apply layout optimizations."""
    pipeline = Pipeline().optimize_cute_layouts()
    return run_pipeline(module, pipeline, "Layout Optimization")


__all__ = [
    "Pipeline",
    "run_pipeline",
    "RocDSLCompilerError",
    "lower_cute_to_standard",
    "lower_cute_to_nvgpu",
    "optimize_layouts",
]
