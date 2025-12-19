"""Test pass pipelines and lowering transformations."""

import pytest
from _mlir.ir import Context, Module

try:
    from rocdsl.passes import (
        Pipeline, run_pipeline, RocDSLCompilerError,
        lower_rocir_to_standard
    )
except ImportError:
    pytest.skip("RocDSL passes not available", allow_module_level=True)


def test_pipeline_construction():
    """Test Pipeline fluent API construction."""
    pipeline = (Pipeline()
                .rocir_to_standard()
                .Func(Pipeline().canonicalize().cse())
                .symbol_dce())
    
    pipeline_str = str(pipeline)
    
    # Should contain all passes
    assert "rocir-to-standard" in pipeline_str
    assert "func.func(canonicalize,cse)" in pipeline_str
    assert "symbol-dce" in pipeline_str
    assert pipeline_str.startswith("builtin.module(")


def test_pipeline_add_pass_with_options():
    """Test adding passes with options."""
    pipeline = Pipeline().add_pass("canonicalize", max_iterations=10, top_down=True)
    
    pipeline_str = str(pipeline)
    
    # Should format options correctly
    assert "canonicalize" in pipeline_str
    assert "max_iterations=10" in pipeline_str
    assert "top_down=True" in pipeline_str


def test_pipeline_composition():
    """Test pipeline addition operators."""
    p1 = Pipeline().rocir_to_standard()
    p2 = Pipeline().canonicalize()
    
    # Test += operator
    p1 += p2
    assert "rocir-to-standard" in str(p1)
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
                     .canonicalize()))
    
    pipeline_str = str(pipeline)
    
    assert "func.func(canonicalize,cse)" in pipeline_str
    assert "gpu.module(canonicalize)" in pipeline_str


def test_pipeline_recipes():
    """Test pre-built pipeline recipes."""
    # Simple composed recipe (API-level check)
    p = Pipeline().rocir_to_standard().canonicalize().cse()
    s = str(p)
    assert "rocir-to-standard" in s
    assert "canonicalize" in s
    assert "cse" in s


def test_run_pipeline_basic():
    """Test running a simple pipeline on valid IR."""
    ctx = Context()
    ctx.load_all_available_dialects()
    ctx.allow_unregistered_dialects = True
    from rocdsl.compiler.context import ensure_rocir_python_extensions
    ensure_rocir_python_extensions(ctx)
    
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
    ctx.load_all_available_dialects()
    ctx.allow_unregistered_dialects = True
    from rocdsl.compiler.context import ensure_rocir_python_extensions
    ensure_rocir_python_extensions(ctx)
    
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


def test_rocir_pass_methods():
    """Test Rocir-specific pass methods are callable.

    NOTE: This test should only reference APIs that exist in `rocdsl.compiler.pipeline.Pipeline`.
    """
    pipeline = Pipeline()
    
    # Rocir lowering passes
    pipeline.rocir_coord_lowering()
    pipeline.rocir_to_standard()
    pipeline.convert_gpu_to_rocdl()
    
    assert len(pipeline._passes) >= 3


def test_standard_mlir_passes():
    """Test standard MLIR pass methods."""
    pipeline = (Pipeline()
                .canonicalize()
                .cse()
                .inline()
                .symbol_dce()
                .sccp()
                .loop_invariant_code_motion()
                .convert_scf_to_cf()
                .convert_arith_to_llvm()
                .convert_func_to_llvm()
                .convert_memref_to_llvm()
                .reconcile_unrealized_casts())
    
    pipeline_str = str(pipeline)
    
    assert "canonicalize" in pipeline_str
    assert "cse" in pipeline_str
    assert "inline" in pipeline_str
    assert "convert-scf-to-cf" in pipeline_str
    assert "convert-arith-to-llvm" in pipeline_str
    assert "convert-func-to-llvm" in pipeline_str
    assert "convert-memref-to-llvm" in pipeline_str


def test_complex_pipeline():
    """Test building a non-trivial composed pipeline."""
    pipeline = (
        Pipeline()
        .rocir_to_standard()
        
        .Func(Pipeline()
              .canonicalize()
              .cse()
              .loop_invariant_code_motion())
        .convert_scf_to_cf()
        .Func(Pipeline()
              .convert_arith_to_llvm()
              .convert_func_to_llvm())
        .convert_memref_to_llvm()
        .reconcile_unrealized_casts()
    )
    
    pipeline_str = str(pipeline)
    
    # Verify all stages present
    assert "rocir-to-standard" in pipeline_str
    assert "func.func(" in pipeline_str
    assert "convert-scf-to-cf" in pipeline_str
    assert "convert-memref-to-llvm" in pipeline_str
    assert "convert-func-to-llvm" in pipeline_str
    assert "reconcile-unrealized-casts" in pipeline_str

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
