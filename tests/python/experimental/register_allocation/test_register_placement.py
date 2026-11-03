# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""register_place and register class descriptors: what the frontend accepts and emits."""

import pytest

from flydsl._mlir import ir
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = pytest.mark.l1a_compile_no_target_dialect


def test_python_register_place_emits_a_separate_op():
    import flydsl.expr as fx
    from flydsl.experimental import register_allocation as fx_regalloc

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(8, fx.Float16)
            allocation_before = str(tensor.owner)
            assert fx_regalloc.register_place(tensor, fx_regalloc.rocdl.AGPR32, start_idx=32) is None
            assert str(tensor.owner) == allocation_before
            attrs = ir.DictAttr.get({"allocSize": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 8)})
            ptr = fx.make_ptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Register), [], dict_attrs=attrs)
            pointer_before = str(ptr.owner)
            fx_regalloc.register_place(ptr, fx_regalloc.rocdl.VGPR32, start_idx=64)
            assert str(ptr.owner) == pointer_before
            with pytest.raises(ValueError, match="register-memory"):
                fx_regalloc.register_place(fx.Int32(3), fx_regalloc.rocdl.VGPR32, start_idx=0)
            # startIdx is an i32 attribute; a wider index must not wrap to another register.
            with pytest.raises(ValueError, match="int32"):
                fx_regalloc.register_place(tensor, fx_regalloc.rocdl.AGPR32, start_idx=2**32 + 32)
        assert module.operation.verify()
        assert str(module).count("fly.register_place") == 2


@pytest.mark.parametrize(
    "descriptor", [("amdgcn", "SGPR_32", 32), ("nvptx64", "Int32Regs", 32), ("x86_64", "GR32", 32)]
)
def test_register_class_representation_is_target_independent(descriptor):
    import flydsl.expr as fx
    from flydsl.experimental import register_allocation as fx_regalloc

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(4, fx.Int32)
            fx_regalloc.register_place(tensor, fx_regalloc.RegisterClass(*descriptor), start_idx=4)
        assert module.operation.verify()
        assert str(ir.Module.parse(str(module))) == str(module)
    assert fx_regalloc.rocdl.SGPR32 == fx_regalloc.RegisterClass("amdgcn", "SGPR_32", 32)
