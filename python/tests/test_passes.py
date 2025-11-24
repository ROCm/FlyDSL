"""Test pass pipelines and lowering transformations."""

import pytest
from mlir.ir import Context, Module

try:
    from rocdsl.passes import (
        Pipeline, run_pipeline, RocDSLCompilerError,
        lower_cute_to_standard
    )
except ImportError:
    pytest.skip("RocDSL passes not available", allow_module_level=True)


def test_pipeline_construction():
    """Test Pipeline fluent API construction."""
    pipeline = (Pipeline()
                .cute_to_standard()
                .Func(Pipeline().canonicalize().cse())
                .symbol_dce())
    
    pipeline_str = str(pipeline)
    
    # Should contain all passes
    assert "cute-to-standard" in pipeline_str
    assert "func.func(canonicalize,cse)" in pipeline_str
    assert "symbol-dce" in pipeline_str
    assert pipeline_str.startswith("builtin.module(")


def test_pipeline_add_pass_with_options():
    """Test adding passes with options."""
    pipeline = Pipeline().cute_nvgpu_to_nvgpu(
        target_arch="sm_90",
        enable_tma=True
    )
    
    pipeline_str = str(pipeline)
    
    # Should format options correctly
    assert "cute-nvgpu-to-nvgpu" in pipeline_str
    assert "target-arch=sm_90" in pipeline_str
    assert "enable-tma=1" in pipeline_str


def test_pipeline_composition():
    """Test pipeline addition operators."""
    p1 = Pipeline().cute_to_standard()
    p2 = Pipeline().canonicalize()
    
    # Test += operator
    p1 += p2
    assert "cute-to-standard" in str(p1)
    assert "canonicalize" in str(p1)
    
    # Test + operator
    p3 = Pipeline().cse()
    p4 = Pipeline().inline()
    p5 = p3 + p4
    assert "cse" in str(p5)
    assert "inline" in str(p5)


def test_nested_pipelines():
    """Test nested pipeline contexts."""
    pipeline = (Pipeline()
                .Func(Pipeline()
                      .canonicalize()
                      .cse())
                .Gpu(Pipeline()
                     .cute_memory_coalescing()))
    
    pipeline_str = str(pipeline)
    
    assert "func.func(canonicalize,cse)" in pipeline_str
    assert "gpu.module(cute-memory-coalescing)" in pipeline_str


def test_pipeline_recipes():
    """Test pre-built pipeline recipes."""
    # CuTe to Standard recipe
    p1 = Pipeline().lower_cute_to_standard()
    s1 = str(p1)
    assert "cute-to-standard" in s1
    assert "canonicalize" in s1
    
    # Optimization recipe
    p2 = Pipeline().optimize_cute_layouts()
    s2 = str(p2)
    assert "cute-layout-fusion" in s2
    assert "cute-vectorization" in s2
    
    # LLVM lowering recipe  
    p3 = Pipeline().lower_to_llvm()
    s3 = str(p3)
    assert "convert-scf-to-cf" in s3
    assert "convert-func-to-llvm" in s3


def test_run_pipeline_basic():
    """Test running a simple pipeline on valid IR."""
    ctx = Context()
    
    # Create simple func module
    mlir_code = """
    module {
        func.func @test() {
            return
        }
    }
    """
    
    with ctx:
        module = Module.parse(mlir_code)
        
        # Run canonicalize pipeline
        pipeline = Pipeline().Func(Pipeline().canonicalize())
        result = run_pipeline(module, pipeline)
        
        # Should return a module
        assert isinstance(result, Module)
        assert "func.func" in str(result)


def test_run_pipeline_with_string():
    """Test running pipeline from string."""
    ctx = Context()
    
    mlir_code = """
    module {
        func.func @test() {
            return
        }
    }
    """
    
    with ctx:
        module = Module.parse(mlir_code)
        
        # Run with pipeline string
        result = run_pipeline(module, "builtin.module(func.func(canonicalize))")
        
        assert isinstance(result, Module)


def test_cute_pass_methods():
    """Test all CuTe-specific pass methods are callable."""
    pipeline = Pipeline()
    
    # Lowering passes
    pipeline.cute_to_standard()
    pipeline.cute_layout_canonicalize()
    pipeline.cute_tensor_partition()
    pipeline.cute_nvgpu_to_nvgpu()
    pipeline.cute_nvgpu_mma_lowering()
    pipeline.cute_nvgpu_copy_lowering()
    pipeline.cute_to_rocm()
    
    # Optimization passes
    pipeline.cute_layout_fusion()
    pipeline.cute_vectorization()
    pipeline.cute_memory_coalescing()
    pipeline.cute_smem_swizzling()
    pipeline.cute_async_pipeline()
    pipeline.cute_warp_specialization()
    
    # Analysis passes
    pipeline.cute_layout_analysis()
    pipeline.cute_atom_validation()
    
    # Should have accumulated many passes
    assert len(pipeline._passes) > 10


def test_standard_mlir_passes():
    """Test standard MLIR pass methods."""
    pipeline = (Pipeline()
                .canonicalize()
                .cse()
                .inline()
                .symbol_dce()
                .sccp()
                .loop_invariant_code_motion()
                .lower_affine()
                .convert_scf_to_cf()
                .convert_arith_to_llvm())
    
    pipeline_str = str(pipeline)
    
    assert "canonicalize" in pipeline_str
    assert "cse" in pipeline_str
    assert "inline" in pipeline_str
    assert "convert-scf-to-cf" in pipeline_str


def test_async_pipeline_options():
    """Test async pipeline with options."""
    pipeline = Pipeline().cute_async_pipeline(
        pipeline_depth=4,
        warp_specialization=True
    )
    
    pipeline_str = str(pipeline)
    
    assert "cute-async-pipeline" in pipeline_str
    assert "pipeline-depth=4" in pipeline_str
    assert "warp-specialization=1" in pipeline_str


def test_warp_specialization_options():
    """Test warp specialization with options."""
    pipeline = Pipeline().cute_warp_specialization(
        num_producer_warps=2
    )
    
    pipeline_str = str(pipeline)
    
    assert "cute-warp-specialization" in pipeline_str
    assert "num-producer-warps=2" in pipeline_str


def test_layout_analysis_options():
    """Test layout analysis with print option."""
    pipeline = Pipeline().cute_layout_analysis(print_analysis=True)
    
    pipeline_str = str(pipeline)
    
    assert "cute-layout-analysis" in pipeline_str
    assert "print-analysis=1" in pipeline_str


def test_complex_pipeline():
    """Test building a complex realistic pipeline."""
    pipeline = (
        Pipeline()
        # Initial CuTe lowering
        .cute_to_standard()
        
        # Function-level optimizations
        .Func(Pipeline()
              .cute_layout_canonicalize()
              .cute_layout_fusion()
              .canonicalize()
              .cse()
              .loop_invariant_code_motion())
        
        # GPU-specific optimizations
        .Gpu(Pipeline()
             .cute_memory_coalescing()
             .cute_smem_swizzling()
             .cute_async_pipeline(pipeline_depth=3))
        
        # Standard MLIR lowering
        .convert_scf_to_cf()
        .Func(Pipeline()
              .convert_arith_to_llvm()
              .convert_func_to_llvm())
        .convert_memref_to_llvm()
        .reconcile_unrealized_casts()
    )
    
    pipeline_str = str(pipeline)
    
    # Verify all stages present
    assert "cute-to-standard" in pipeline_str
    assert "func.func(" in pipeline_str
    assert "gpu.module(" in pipeline_str
    assert "convert-scf-to-cf" in pipeline_str
    assert "convert-memref-to-llvm" in pipeline_str


def test_nvgpu_lowering_pipeline():
    """Test NVGPU lowering pipeline recipe."""
    pipeline = Pipeline().lower_cute_nvgpu_to_nvgpu(
        target_arch="sm_80",
        enable_pipeline=True
    )
    
    pipeline_str = str(pipeline)
    
    assert "gpu.module(" in pipeline_str
    assert "cute-nvgpu-to-nvgpu" in pipeline_str
    assert "target-arch=sm_80" in pipeline_str
    assert "cute-nvgpu-mma-lowering" in pipeline_str
    assert "cute-async-pipeline" in pipeline_str


def test_materialize_with_without_module():
    """Test pipeline materialization with/without module wrapper."""
    pipeline = Pipeline().canonicalize().cse()
    
    # With module wrapper (default)
    with_module = pipeline.materialize(module=True)
    assert with_module.startswith("builtin.module(")
    assert "canonicalize,cse" in with_module
    
    # Without module wrapper
    without_module = pipeline.materialize(module=False)
    assert not without_module.startswith("builtin.module(")
    assert without_module == "canonicalize,cse"


def test_underscore_to_hyphen_conversion():
    """Test that underscores in pass names convert to hyphens."""
    pipeline = Pipeline()
    
    # Python method names use underscores
    pipeline.loop_invariant_code_motion()
    pipeline.convert_scf_to_cf()
    
    pipeline_str = str(pipeline)
    
    # MLIR pass names use hyphens
    assert "loop-invariant-code-motion" in pipeline_str
    assert "convert-scf-to-cf" in pipeline_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
