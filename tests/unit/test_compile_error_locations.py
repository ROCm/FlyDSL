# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile errors and pass diagnostics point at the user's source, also without debug info."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith
from flydsl.compiler.diagnostics import DSLCompileError, SourceFrame
from flydsl.compiler.jit_function import _run_pipeline


@pytest.mark.l1a_compile_no_target_dialect
def test_pass_pipeline_error_keeps_the_source_chain():
    # PassManager.run captures its errors into the MLIRError it raises, not into a context handler.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse("func.func @f(%a: i32, %b: i64) { return }")
        block = module.body.operations[0].regions[0].blocks[0]
        loc = ir.Location.callsite(ir.Location.file("a.py", 7, 4), [ir.Location.file("b.py", 3, 1)])
        with ir.InsertionPoint.at_block_begin(block):
            ir.Operation.create(
                "arith.addi", results=[ir.IntegerType.get_signless(32)], operands=list(block.arguments), loc=loc
            )
        with pytest.raises(DSLCompileError) as raised:
            _run_pipeline(module, ["symbol-dce"], verifier=True, print_after_all=False)

    assert raised.value.diagnostics[0].chain == [SourceFrame("a.py", 7, 4, 4), SourceFrame("b.py", 3, 1, 1)]


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_pass_pipeline_ir_keeps_user_locations_without_debug_info(tmp_path, monkeypatch):
    # Pass diagnostics can only point at user code if the IR the passes run on still has its source
    # locations, also when debug info is off.
    for name, value in {
        "ARCH": "gfx942",
        "COMPILE_ONLY": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        "FLYDSL_DUMP_IR": "1",
        "FLYDSL_DUMP_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", raising=False)

    @flyc.kernel
    def double(x: fx.Tensor):
        tile = fx.slice(fx.zipped_divide(x, (1, 1)), (None, 0))
        frag = fx.make_fragment_like(tile)
        atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        fx.copy(atom, tile, frag)
        frag.store(frag.load() * 2.0)
        fx.copy(atom, frag, tile)

    @flyc.jit
    def launch(x: fx.Tensor):
        double(x).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(torch.empty(1, 64))
    first_pass_ir = sorted(tmp_path.glob("*/01_*.mlir"))[0].read_text()
    assert f'loc("{__file__}"' in first_pass_ir


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_compile_error_points_at_the_kernel(monkeypatch):
    for name, value in {"ARCH": "gfx942", "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FLYDSL_DUMP_IR", raising=False)

    @flyc.kernel
    def leftover_cast(x: fx.Tensor):
        # A cast no pass removes, so translation to LLVM IR fails on it.
        one = arith.ConstantOp(ir.IntegerType.get_signless(32), 1).result
        cast = ir.Operation.create("builtin.unrealized_conversion_cast", results=[ir.F32Type.get()], operands=[one])
        ir.Operation.create("gpu.printf", operands=list(cast.results), attributes={"format": ir.StringAttr.get("%f")})

    @flyc.jit
    def launch(x: fx.Tensor):
        leftover_cast(x).launch(grid=(1, 1, 1), block=(64, 1, 1))

    with pytest.raises(DSLCompileError) as raised:
        launch(torch.empty(1, 64))

    first = raised.value.diagnostics[0]
    assert "unrealized_conversion_cast" in first.message
    assert first.chain[0].filename == __file__
