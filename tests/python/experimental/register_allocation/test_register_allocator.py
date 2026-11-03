# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import re

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import construct_from_ir_values, dsl_align_of, dsl_size_of, extract_to_ir_values
from flydsl.experimental import register_allocation as fx_regalloc
from tests.language.conftest import frontend_only_jit  # noqa: F401  (autouse: L1a cases skip backend compile)
from tests.language.lang_utils import launch_ir, source_ir

pytestmark = pytest.mark.l1a_compile_no_target_dialect


@fx.struct
class RegisterPayload:
    tag: fx.Int8
    scalar: fx.Float32
    vector: fx.Align[fx.Vector[fx.Float16, (2, 4)], 16]


@fx.union
class RegisterOverlay:
    word: fx.Int32
    halves: fx.Array[fx.Int16, 2]


def test_register_api_is_reached_only_through_its_subpackage():
    assert fx.experimental.register_allocation is fx_regalloc
    for name in ("RegisterAllocator", "RegisterClass", "register_place"):
        assert getattr(fx_regalloc, name) is getattr(fx_regalloc.register, name)
        assert not hasattr(fx.experimental, name)
        assert not hasattr(fx, name)
    assert not hasattr(fx.experimental, "rocdl")
    for name in ("AGPR32", "VGPR32", "SGPR32"):
        assert not hasattr(fx.rocdl, name)


def test_allocator_uses_absolute_alignment_and_whole_registers():
    captured = {}

    def body():
        regs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=65)
        regs.allocate(fx.Int8)
        regs.allocate(fx.Align[fx.Float32, 16])
        regs.allocate(RegisterPayload)
        regs.allocate(RegisterOverlay)
        regs.allocate(3)
        captured["bytes"] = regs.allocated_bytes

    text = source_ir(body)
    assert re.findall(r"startIdx = (\d+) : i32", text) == ["65", "68", "72", "80", "81"]
    assert captured == {"bytes": 68}
    assert dsl_size_of(RegisterPayload) == 32
    assert dsl_align_of(RegisterPayload) == 16


def test_scalar_vector_and_composite_storage_access():
    @flyc.kernel
    def kernel():
        regs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 32)
        scalar = regs.allocate(fx.Int32)
        scalar.poke(fx.Int32(fx.thread_idx.x))
        scalar.poke(scalar.peek() + 1)
        payload = regs.allocate(RegisterPayload)
        payload.tag.poke(1)
        payload.scalar.poke(2.0)
        payload.vector.poke(fx.Vector.filled((2, 4), 3.0, fx.Float16))
        vec = payload.vector.peek()
        assert vec.shape == (2, 4)
        assert vec.dtype is fx.Float16
        value = payload.peek()
        payload.poke(value)
        assert value.vector.shape == (2, 4)
        overlay = regs.allocate(RegisterOverlay)
        overlay.word.poke(scalar.peek())
        overlay.halves.peek()[1] = fx.Int16(4)

    text = launch_ir(kernel)
    assert "fly.register_place" in text
    assert "fly.recast_iter" in text
    assert "vector<8xf16>" in text


@pytest.mark.parametrize("alignment", [0, -1, 3, True, 1.5])
def test_invalid_alignment(alignment):
    def body():
        fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 32).allocate(fx.Int32, alignment=alignment)

    with pytest.raises(Exception, match="alignment must be a positive power of two"):
        source_ir(body)


@pytest.mark.parametrize("offset", [None, -1, True, 1.5])
def test_invalid_start(offset):
    with pytest.raises(ValueError, match="start_idx"):
        fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, offset)


def test_storage_roundtrip_and_loop():
    @flyc.kernel
    def kernel(n: fx.Int32):
        buf = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 32).allocate(RegisterPayload)
        rebuilt = construct_from_ir_values(type(buf), buf, extract_to_ir_values(buf))
        rebuilt.scalar.poke(1.0)
        for _ in range(0, n):
            buf.scalar.poke(buf.scalar.peek() + 1.0)

    assert "scf.for" in launch_ir(kernel, fx.Int32(3))


def test_custom_storable_and_aligned_composite():
    class Word:
        @classmethod
        def __dsl_size_of__(cls):
            return 4

        @classmethod
        def __dsl_align_of__(cls):
            return 8

        @classmethod
        def __peek_from_ptr__(cls, ptr):
            return fx.Int32.__peek_from_ptr__(ptr)

        @classmethod
        def __poke_into_ptr__(cls, ptr, value):
            fx.Int32.__poke_into_ptr__(ptr, value)

    def body():
        regs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 65)
        word = regs.allocate(Word)
        word.poke(42)
        assert isinstance(word.peek(), fx.Int32)
        overlay = regs.allocate(RegisterOverlay, alignment=16)
        overlay.word.poke(word.peek())
        assert isinstance(overlay.halves.peek()[1], fx.Int16)

    assert re.findall(r"startIdx = (\d+) : i32", source_ir(body)) == ["66", "68"]


def test_fixed_allocator_rounds_to_byte_alignment():
    def body():
        regs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 65)
        regs.allocate(fx.Int32, alignment=16)
        regs.allocate(fx.Align[fx.Int32, 16])
        regs.allocate(fx.Int64)

    text = source_ir(body)
    assert re.findall(r"startIdx = (\d+) : i32", text) == ["68", "72", "74"]


def test_allocator_requires_register_class():
    with pytest.raises(TypeError):
        fx_regalloc.RegisterAllocator()
    with pytest.raises(TypeError, match="register_class must be a RegisterClass"):
        fx_regalloc.RegisterAllocator(None, start_idx=32)


@fx.struct
class MixedRegisterRefs:
    scalar: fx.Storage[fx.Int32]
    vector: fx.Storage[fx.Vector[fx.Float32, 4]]
    accumulator: fx.Storage[fx.Float32]
    scratch: fx.Storage[fx.Int16]


@flyc.jit
def update_register_refs(refs: MixedRegisterRefs):
    refs.scalar.poke(refs.scalar.peek() + 1)
    return refs


def test_struct_groups_independent_storage_across_classes_and_jit_loop():
    @flyc.kernel
    def kernel(n: fx.Int32):
        refs = MixedRegisterRefs(
            scalar=fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.SGPR32, 40).allocate(fx.Int32),
            vector=fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 32).allocate(fx.Vector[fx.Float32, 4]),
            accumulator=fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.AGPR32, 64).allocate(fx.Float32),
            scratch=fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, 36).allocate(fx.Int16),
        )
        refs = construct_from_ir_values(type(refs), refs, extract_to_ir_values(refs))
        refs.scalar.poke(fx.Int32(fx.block_idx.x))
        refs.vector.poke(fx.Vector.filled(4, 2.0, fx.Float32))
        refs.accumulator.poke(refs.vector.peek()[0])
        refs.scratch.poke(3)
        for _ in range(0, n):
            refs = update_register_refs(refs)
        refs.scratch.poke(fx.Int16(refs.scalar.peek()))

    text = launch_ir(kernel, fx.Int32(3))
    assert text.count("fly.register_place") == 4
    for name in ("SGPR_32", "VGPR_32", "AGPR_32"):
        assert name in text
    assert "scf.for" in text
    assert "vector<4xf32>" in text
