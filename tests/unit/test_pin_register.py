# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for fly.pin_register: op verification, inline-asm lowering, and Python API."""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def test_pin_register_lowers_to_inline_asm():
    """fly.pin_register on i32 lowers to llvm.inline_asm with ={v68},{v68}."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i32) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 68 : i64} : i32
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        asm = str(module)
        assert "llvm.inline_asm" in asm
        assert "={v68},{v68}" in asm


def test_pin_register_vector_lowers_to_tuple_constraint():
    """fly.pin_register on vector<4xi32> lowers to ={v[68:71]},{v[68:71]}."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: vector<4xi32>) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 68 : i64} : vector<4xi32>
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        asm = str(module)
        assert "llvm.inline_asm" in asm
        assert "={v[68:71]},{v[68:71]}" in asm


def test_pin_register_sgpr():
    """SGPR pin lowers to ={s16},{s16}."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i32) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "SGPR_32">, index = 16 : i64} : i32
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        assert "={s16},{s16}" in str(module)


def test_pin_register_agpr():
    """AGPR pin lowers to ={a0},{a0}."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i32) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "AGPR_32">, index = 0 : i64} : i32
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        assert "={a0},{a0}" in str(module)


def test_pin_register_rejects_negative_index():
    """Verifier rejects negative index."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i32) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = -1 : i64} : i32
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        with pytest.raises(ir.MLIRError, match="index must be nonnegative"):
            ir.Module.parse(source)


def test_pin_register_rejects_non_32bit_type():
    """Verifier rejects types whose total width is not a multiple of 32."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i16) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, index = 0 : i64} : i16
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        with pytest.raises(ir.MLIRError, match="total bit width must be a multiple of 32"):
            ir.Module.parse(source)


def test_pin_register_rejects_wrong_target():
    """Lowering rejects non-amdgcn targets."""
    source = """module {
      gpu.module @m {
        gpu.func @test(%x: i32) kernel {
          %pinned = fly.pin_register %x {regClass = #fly.register_class<"nvptx64", "Int32Regs">, index = 0 : i64} : i32
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match="pin_register requires amdgcn target"):
            PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)


def test_python_pin_vgpr_emits_fly_pin_register():
    """Python pin_vgpr API emits fly.pin_register op."""
    import flydsl.expr as fx

    source = """module {
      func.func @test(%x: i32) -> i32 {
        return %x : i32
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        func = module.body.operations[0]
        block = func.regions[0].blocks[0]
        with ir.InsertionPoint.at_block_begin(block):
            x = block.arguments[0]
            fx.rocdl.pin_vgpr(x, reg=68)
        text = str(module)
        assert "fly.pin_register" in text
        assert 'regClass = #fly.register_class<"amdgcn", "VGPR_32">' in text
        assert "index = 68" in text
