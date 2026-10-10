# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl._mlir.passmanager import PassManager

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.mark.parametrize("setter", ["field", "dict", "primitive"])
@pytest.mark.parametrize("runtime", [False, True])
def test_tiled_copy_set_value_preserves_type_and_copy_state(setter, runtime):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            tiled = fx.make_tiled_copy(atom, fx.make_layout((64, 1), (1, 64)), (64, 1))
            src_type = ir.Type.parse("!fly.memref<f32, #fly_rocdl.buffer_desc, (1,2):(1,1)>")
            dst_type = ir.Type.parse("!fly.memref<f32, register, (1,2):(1,1)>")
            function = func.FuncOp("copy_with_state", ([tiled.type, fx.Int32.ir_type, src_type, dst_type], []))
            with ir.InsertionPoint(function.add_entry_block()):
                original, offset, src, dst = function.arguments
                value = fx.Int32(offset) if runtime else 7
                if setter == "field":
                    updated = original.set_value("soffset", value)
                elif setter == "dict":
                    updated = original.set_value({"soffset": value})
                else:
                    updated = fx.atom_set_value(original, "soffset", value)
                assert isinstance(updated, fx.TiledCopy)
                assert updated.type == original.type
                assert updated.tile_mn.type == original.tile_mn.type
                assert updated.layout_tv_tiled.type == original.layout_tv_tiled.type
                assert updated.owner.operands[0] == original
                fx.copy(updated, src, dst)
                func.ReturnOp([])
        assert module.operation.verify()
        PassManager.parse(
            "builtin.module(fly-rewrite-func-signature,fly-canonicalize,fly-layout-lowering,"
            "canonicalize,convert-fly-to-rocdl,canonicalize)"
        ).run(module.operation)
        function = next(op for op in module.body.operations if op.operation.name == "func.func")
        loads = [op for op in function.body.blocks[0].operations if op.name == "rocdl.raw.ptr.buffer.load"]
        assert len(loads) == 2
        for load in loads:
            byte_offset = load.operands[2]
            if runtime:
                assert byte_offset.owner.name == "arith.muli"
                assert function.arguments[1] in byte_offset.owner.operands
            else:
                assert ir.IntegerAttr(byte_offset.owner.attributes["value"]).value == 28


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("runtime", [False, True])
def test_tiled_copy_tdm_boundary_check_with_coord_tensor(direction, runtime):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        atom_type = ir.Type.parse(
            f"!fly.copy_atom<!fly_rocdl.cdna5.tensor_{direction}"
            "<shape = [128, 64], elem = f16, tensor2tdm = (1E0,1E1)>, 16>"
        )
        coord_type = ir.Type.parse("!fly.coord_tensor<(128,384), (128,64):(1E1,1E0)>")
        lds_type = ir.Type.parse("!fly.memref<f16, shared, (128,64):(64,1)>")
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("tdm_with_state", ([atom_type, coord_type, lds_type, fx.Int32.ir_type], []))
            with ir.InsertionPoint(function.add_entry_block()):
                atom, coord, lds, flag = function.arguments
                tiled = fx.make_tiled_copy(atom, fx.make_layout((1, 8192), (1, 1)), (128, 64))
                updated = tiled.set_value("boundary_check", fx.make_int_tuple((1, fx.Int32(flag) if runtime else 0)))
                assert isinstance(updated, fx.TiledCopy)
                assert updated.type == tiled.type
                src, dst = (coord, lds) if direction == "load" else (lds, coord)
                fx.copy(updated, src, dst)
                func.ReturnOp([])
        assert module.operation.verify()
        PassManager.parse("builtin.module(fly-rocdl-expand-ops)").run(module.operation)
        setter = next(op for op in function.body.blocks[0].operations if op.operation.name == "fly.atom.set_value")
        assert ir.StringAttr(setter.attributes["field"]).value == "boundary_check_axes"
        assert setter.result.type == tiled.type
        PassManager.parse(
            "builtin.module(fly-rewrite-func-signature,fly-canonicalize,fly-layout-lowering,"
            "canonicalize,fly-convert-atom-call-to-ssa-form,convert-fly-to-rocdl,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        text = str(module)
        assert f"rocdl.tensor.{direction}" in text
        assert ("arith.select" in text) == runtime
