# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import re

import pytest
from lang_utils import launch_ir, source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import construct_from_ir_values, dsl_align_of, dsl_size_of, extract_to_ir_values

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


def test_allocator_uses_absolute_alignment_and_whole_registers():
    captured = {}

    def body():
        regs = fx.RegisterAllocator(fx.rocdl.VGPR, start_offset=65)
        regs.allocate(fx.Int8)
        regs.allocate(fx.Align[fx.Float32, 16])
        regs.allocate(RegisterPayload)
        regs.allocate(RegisterOverlay)
        regs.allocate(3)
        captured["bytes"] = regs.allocated_bytes
        captured["registers"] = regs.allocated_registers

    text = source_ir(body)
    assert re.findall(r"start = (\d+) : i64", text) == ["65", "68", "72", "80", "81"]
    assert captured == {"bytes": 68, "registers": 17}
    assert dsl_size_of(RegisterPayload) == 32
    assert dsl_align_of(RegisterPayload) == 16


def test_scalar_vector_and_composite_storage_access():
    @flyc.kernel
    def kernel():
        regs = fx.RegisterAllocator(fx.rocdl.VGPR, 32)
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
    assert "fly.set_register" in text
    assert "fly.recast_iter" in text
    assert "vector<8xf16>" in text


def test_vector_storable_also_works_with_shared_allocator():
    @flyc.kernel
    def kernel():
        buf = fx.SharedAllocator().allocate(fx.Vector[fx.Float32, (2, 4)])
        buf.poke(fx.Vector.filled((2, 4), 2.0, fx.Float32))
        assert buf.peek().shape == (2, 4)

    assert "vector<8xf32>" in launch_ir(kernel)


@pytest.mark.parametrize("alignment", [0, -1, 3, True, 1.5])
def test_invalid_alignment(alignment):
    def body():
        fx.RegisterAllocator(fx.rocdl.VGPR).allocate(fx.Int32, alignment=alignment)

    with pytest.raises(Exception, match="alignment must be a positive power of two"):
        source_ir(body)


@pytest.mark.parametrize("offset", [-1, True, 1.5, 2**63])
def test_invalid_start(offset):
    with pytest.raises(ValueError, match="start_offset"):
        fx.RegisterAllocator(fx.rocdl.VGPR, offset)


def test_unknown_class_and_range():
    with pytest.raises(ValueError, match="unknown LLVM register class"):
        fx.RegisterAllocator(fx.RegisterClass("amdgcn", "DOES_NOT_EXIST"))

    def body():
        fx.RegisterAllocator(fx.rocdl.VGPR, fx.rocdl.VGPR.member_count).allocate(fx.Int32)

    with pytest.raises(Exception, match="exceeds the LLVM register class"):
        source_ir(body)


@pytest.mark.parametrize("dtype,shape", [(fx.Boolean, 4), (fx.Int32, 0), (fx.Int32, (2, -1))])
def test_invalid_vector_storage_type(dtype, shape):
    with pytest.raises((TypeError, ValueError)):
        fx.Vector[dtype, shape]


def test_vector_storage_checks_logical_shape():
    def body():
        ptr = fx.RegisterAllocator(fx.rocdl.VGPR).allocate(fx.Vector[fx.Float32, (2, 4)])
        ptr.poke(fx.Vector.filled(8, 1.0, fx.Float32))

    with pytest.raises(Exception, match="storage shape/dtype"):
        source_ir(body)


@pytest.mark.parametrize("allocator_kind", ["register", "shared_static", "shared_dynamic"])
def test_storage_roundtrip_and_loop(allocator_kind):
    @flyc.kernel
    def kernel(n: fx.Int32):
        if fx.const_expr(allocator_kind == "register"):
            allocator = fx.RegisterAllocator(fx.rocdl.VGPR, 32)
        else:
            allocator = fx.SharedAllocator(static=allocator_kind == "shared_static")
        buf = allocator.allocate(RegisterPayload)
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
        regs = fx.RegisterAllocator(fx.rocdl.VGPR, 65)
        word = regs.allocate(Word)
        word.poke(42)
        assert isinstance(word.peek(), fx.Int32)
        overlay = regs.allocate(fx.Align[RegisterOverlay, 16])
        overlay.word.poke(word.peek())
        assert isinstance(overlay.halves.peek()[1], fx.Int16)

    assert re.findall(r"start = (\d+) : i64", source_ir(body)) == ["66", "68"]
