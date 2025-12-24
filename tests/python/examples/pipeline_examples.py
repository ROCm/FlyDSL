"""Examples demonstrating the Pipeline API for FLIR compiler.

This file shows various ways to use the Pipeline class to build
and execute transformation passes on MLIR modules.
"""

from pyflir.compiler.pipeline import Pipeline
from pyflir.compiler.context import RAIIMLIRContextModule


# ============================================================================
# Example 1: Simple single pass
# ============================================================================

def example_single_pass():
    """Apply a single pass using Pipeline."""
    # Old way (still works for backward compatibility):
    # from pyflir.compiler.flir_opt_helper import apply_flir_coord_lowering
    # module = apply_flir_coord_lowering(module)
    
    # New way with Pipeline:
    ctx = RAIIMLIRContextModule()
    # ... build your MLIR module here ...
    
    pipeline = Pipeline().flir_to_standard()
    result = pipeline.run(ctx.module)
    
    print(f"Pipeline: {pipeline}")
    # Output: builtin.module(flir-to-standard)


# ============================================================================
# Example 2: Sequential passes
# ============================================================================

def example_sequential_passes():
    """Chain multiple passes together."""
    ctx = RAIIMLIRContextModule()
    
    # Build a pipeline with multiple passes
    pipeline = (Pipeline()
                .flir_to_standard()
                .canonicalize()
                .cse())
    
    result = pipeline.run(ctx.module)
    
    print(f"Pipeline: {pipeline}")
    # Output: builtin.module(flir-to-standard,canonicalize,cse)


# ============================================================================
# Example 3: Nested passes (function-level)
# ============================================================================

def example_nested_passes():
    """Apply passes at different nesting levels."""
    ctx = RAIIMLIRContextModule()
    
    # Apply some passes at module level, others at function level
    pipeline = (Pipeline()
                .flir_to_standard()
                .Func(Pipeline()
                      .canonicalize()
                      .cse()
                      .loop_invariant_code_motion())
                .symbol_dce())
    
    result = pipeline.run(ctx.module)
    
    print(f"Pipeline: {pipeline}")
    # Output: builtin.module(flir-to-standard,func.func(canonicalize,cse,loop-invariant-code-motion),symbol-dce)


# ============================================================================
# Example 4: Complex compilation pipeline
# ============================================================================

def example_full_lowering_pipeline():
    """Example of a complete lowering pipeline for GPU compilation."""
    ctx = RAIIMLIRContextModule()
    
    pipeline = (Pipeline()
        # Stage 1: Lower high-level Flir operations
        .flir_coord_lowering()
        .flir_to_standard()
        
        # Stage 2: Function-level optimizations
        .Func(Pipeline()
              .canonicalize()
              .cse()
              .loop_invariant_code_motion())
        
        # Stage 3: GPU-specific transformations (ROCDL lowering)
        .Gpu(Pipeline())
        
        # Stage 4: Standard dialect lowering
        .convert_scf_to_cf()
        .Func(Pipeline()
              .convert_arith_to_llvm()
              .convert_func_to_llvm())
        .convert_memref_to_llvm()
        .convert_gpu_to_rocdl()
        .reconcile_unrealized_casts()
    )
    
    result = pipeline.run(ctx.module)
    
    print(f"Full pipeline:\n{pipeline}")


# ============================================================================
# Example 5: Pipeline composition with operators
# ============================================================================

def example_pipeline_composition():
    """Combine pipelines using + and += operators."""
    # Create reusable pipeline components
    lowering = Pipeline().flir_coord_lowering().flir_to_standard()
    
    optimization = Pipeline().canonicalize().cse()
    
    # Combine with + operator (creates new pipeline)
    basic_pipeline = lowering + optimization
    
    # Or use += to extend existing pipeline
    full_pipeline = Pipeline().flir_to_standard()
    full_pipeline += Pipeline().canonicalize()
    full_pipeline += Pipeline().cse()
    
    print(f"Basic pipeline: {basic_pipeline}")
    print(f"Full pipeline: {full_pipeline}")


# ============================================================================
# Example 6: Pass with options
# ============================================================================

def example_pass_with_options():
    """Use passes that take configuration options."""
    ctx = RAIIMLIRContextModule()
    
    pipeline = (Pipeline()
                .flir_to_standard()
                .add_pass("canonicalize", max_iterations=3))
    
    result = pipeline.run(ctx.module)
    
    print(f"Pipeline with options: {pipeline}")
    # Output: builtin.module(flir-to-standard,<async-pipeline pass>,<layout-analysis pass>)


# ============================================================================
# Example 7: Error handling
# ============================================================================

def example_error_handling():
    """Demonstrate error handling with pipelines."""
    from pyflir.compiler.pipeline import FLIRCompilerError
    
    ctx = RAIIMLIRContextModule()
    
    pipeline = Pipeline().flir_to_standard()
    
    try:
        result = pipeline.run(ctx.module)
    except FLIRCompilerError as e:
        print(f"Pipeline failed: {e}")
        # This exception includes the full pipeline string for debugging


# ============================================================================
# Example 8: Using string pipelines directly
# ============================================================================

def example_string_pipeline():
    """Run a pipeline from a string (for advanced users)."""
    from pyflir.compiler.pipeline import run_pipeline
    
    ctx = RAIIMLIRContextModule()
    
    # Can pass a Pipeline object or a string
    result = run_pipeline(
        ctx.module,
        "builtin.module(flir-to-standard,canonicalize)"
    )
    
    # Or use Pipeline object
    result = run_pipeline(
        ctx.module,
        Pipeline().flir_to_standard().canonicalize()
    )


# ============================================================================
# Example 9: Inspect pipeline before running
# ============================================================================

def example_pipeline_inspection():
    """Inspect a pipeline before executing it."""
    pipeline = (Pipeline()
                .flir_to_standard()
                .Func(Pipeline().canonicalize())
                .cse())
    
    # Get the pipeline string
    pipeline_str = str(pipeline)
    print(f"Pipeline string: {pipeline_str}")
    
    # Access internal passes list
    print(f"Number of passes: {len(pipeline._passes)}")
    print(f"Pass list: {pipeline._passes}")


# ============================================================================
# Example 10: Convenience functions
# ============================================================================

def example_convenience_functions():
    """Use high-level convenience functions."""
    from pyflir.compiler.pipeline import (
        lower_flir_to_standard,
        apply_flir_coord_lowering
    )
    
    ctx = RAIIMLIRContextModule()
    
    # Quick lowering with predefined pipelines
    result = lower_flir_to_standard(ctx.module)
    
    # Or for coordinate lowering only
    result = apply_flir_coord_lowering(ctx.module)


if __name__ == "__main__":
    print("=== FLIR Pipeline API Examples ===\n")
    
    examples = [
        ("Single Pass", example_single_pass),
        ("Sequential Passes", example_sequential_passes),
        ("Nested Passes", example_nested_passes),
        ("Full Lowering Pipeline", example_full_lowering_pipeline),
        ("Pipeline Composition", example_pipeline_composition),
        ("Pass with Options", example_pass_with_options),
        ("Pipeline Inspection", example_pipeline_inspection),
    ]
    
    for name, example_fn in examples:
        print(f"\n{'='*60}")
        print(f"Example: {name}")
        print('='*60)
        try:
            example_fn()
        except Exception as e:
            print(f"Note: {e}")

