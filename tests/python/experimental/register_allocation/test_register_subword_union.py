# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register storage narrower than one 32-bit register, and unions.

Placement binds whole 32-bit words. A field narrower than a register keeps its
byte offset inside the covering word: an F16 at byte 0 occupies bits [15:0] and
one at byte 2 occupies bits [31:16]. Placement is enforced only while real code
still consumes the placed word. When LLVM forwards each sub-word value directly
to its readers, the packed word is used only by the placement marker and
compilation is rejected. UNSUPPORTED records those patterns so that a change in
behavior is visible; they compile without placement.
"""

import os
import re
import subprocess
import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.experimental import register_allocation as fx_regalloc

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@fx.struct
class HalfPair:
    lo: fx.Float16
    hi: fx.Float16


@fx.union
class HalfWord:
    pair: HalfPair
    word: fx.Int32


@fx.struct
class ShortPair:
    lo: fx.Uint16
    hi: fx.Uint16


@fx.union
class ShortWord:
    pair: ShortPair
    word: fx.Int32


@fx.struct
class Bytes4:
    b0: fx.Uint8
    b1: fx.Uint8
    b2: fx.Uint8
    b3: fx.Uint8


@fx.union
class ByteWord:
    bytes: Bytes4
    word: fx.Int32


@fx.struct
class Mixed:
    a: fx.Int8
    b: fx.Int8
    c: fx.Int16


@fx.union
class MixedWord:
    parts: Mixed
    word: fx.Int32


@fx.union
class Halves:
    word: fx.Int32
    halves: fx.Array[fx.Uint16, 2]


@fx.union
class FloatInt:
    f: fx.Float32
    i: fx.Int32


@fx.union
class Wide:
    d: fx.Int64
    w: fx.Array[fx.Int32, 2]


@fx.struct
class Tagged:
    tag: fx.Int32
    u: FloatInt


@fx.struct
class HalfVector:
    v: fx.Vector[fx.Float16, 2]


# Case -> registers written by the fixed placement (VGPR origin 64, SGPR origin 40).
SUPPORTED = {
    "f16_pair_word": {"v64"},  # write lo/hi F16 fields, read the union word
    "f16_vector": {"v64"},  # Vector[F16, 2] packed arithmetic
    "f16_pair_loop": {"v64"},  # loop-carried word, upper half updated
    "i8x4_word": {"v64"},  # write four bytes, read the union word
    "i8_i16_word": {"v64"},  # Int8, Int8, Int16 fields packed into one word
    "union_partial_update": {"v64"},  # write word, overwrite the low half
    "union_word_to_halves": {"v64"},  # write word, read F16 halves separately
    "union_word_to_bytes": {"v64"},  # write word, read bytes separately
    "union_float_int": {"v64"},  # Float32/Int32 reinterpretation
    "union_int64_halves": {"v64", "v65"},  # Int64 over Array[Int32, 2]
    "union_in_struct": {"v64", "v65"},  # union field after an Int32 field
    "union_sgpr": {"s40"},  # uniform union in SGPR storage
    "u16_pair_sgpr_word": {"s40"},  # uniform 16-bit fields packed in an SGPR
    "f16_scalar_arith": {"v64"},  # one F16 per register, used in arithmetic
    "f16_scalar_update": {"v64"},  # one F16 read and rewritten several times
    "f16_scalar_loop": {"v64"},  # one F16 updated in a loop
    "i8_scalar_arith": {"v64"},  # one Int8 per register, used in arithmetic
    "i16_scalar_arith": {"v64"},  # one Int16 per register, used in arithmetic
}

# Each sub-word value is written and read on its own. Real code uses the
# unpacked values; the placed word survives only in the placement marker.
UNSUPPORTED = [
    "f16_pair_separate",
    "u16_pair_separate",
    "i8x4_separate",
    "f16_array3_separate",
    # A lone F16 in the upper half: arithmetic reads the value unshifted, so
    # the placed word is a shifted copy that nothing else uses.
    "f16_hi_only",
]


def _allocator(bank="VGPR"):
    # "plain" allocates like "fixed"; run_case drops its placement declarations.
    if bank == "SGPR":
        return fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.SGPR32, start_idx=40)
    return fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=64)


@flyc.kernel
def subword_kernel(X: fx.Tensor, Y: fx.Tensor, Out: fx.Tensor, case: fx.Constexpr[str], mode: fx.Constexpr[str]):
    tid = fx.thread_idx.x
    x = X[tid]
    y = Y[tid]
    if case == "f16_pair_word":
        p = _allocator().allocate(HalfWord)
        p.pair.lo.poke(x.to(fx.Float16))
        p.pair.hi.poke(y.to(fx.Float16))
        Out[tid] = p.word.peek()
    elif case == "f16_vector":
        p = _allocator().allocate(HalfVector)
        p.v.poke(fx.Vector.from_elements([x.to(fx.Float16), y.to(fx.Float16)], fx.Float16))
        v = p.v.peek()
        Out[tid] = (v * v + v).reduce(fx.ReductionOp.ADD).to(fx.Int32)
    elif case == "f16_pair_loop":
        pair = _allocator().allocate(fx.Array[fx.Float16, 2]).peek()
        pair[0] = x.to(fx.Float16)
        pair[1] = y.to(fx.Float16)
        for _ in range(0, x & 3):
            pair[1] = pair[1] + pair[0]
        Out[tid] = (pair[0].to(fx.Float32) + pair[1].to(fx.Float32) * 2.0).to(fx.Int32)
    elif case == "i8x4_word":
        p = _allocator().allocate(ByteWord)
        p.bytes.b0.poke(x.to(fx.Uint8))
        p.bytes.b1.poke(y.to(fx.Uint8))
        p.bytes.b2.poke((x + 1).to(fx.Uint8))
        p.bytes.b3.poke((y + 2).to(fx.Uint8))
        Out[tid] = p.word.peek()
    elif case == "i8_i16_word":
        p = _allocator().allocate(MixedWord)
        p.parts.a.poke(x.to(fx.Int8))
        p.parts.b.poke(y.to(fx.Int8))
        p.parts.c.poke((x * y).to(fx.Int16))
        Out[tid] = p.word.peek()
    elif case == "union_partial_update":
        p = _allocator().allocate(Halves)
        p.word.poke(x * 65536 + y)
        p.halves.peek()[0] = (y + 5).to(fx.Uint16)
        Out[tid] = p.word.peek()
    elif case == "union_word_to_halves":
        p = _allocator().allocate(HalfWord)
        p.word.poke(x * 65536 + y)
        Out[tid] = p.pair.lo.peek().bitcast(fx.Uint16).to(fx.Int32) + p.pair.hi.peek().bitcast(fx.Uint16).to(fx.Int32)
    elif case == "union_word_to_bytes":
        p = _allocator().allocate(ByteWord)
        p.word.poke(x * 16777216 + y * 65536 + x * 256 + y)
        b = p.bytes
        Out[tid] = (
            b.b0.peek().to(fx.Int32)
            + b.b1.peek().to(fx.Int32) * 2
            + b.b2.peek().to(fx.Int32) * 4
            + b.b3.peek().to(fx.Int32) * 8
        )
    elif case == "union_float_int":
        p = _allocator().allocate(FloatInt)
        p.f.poke(x.to(fx.Float32) * 0.5)
        Out[tid] = p.i.peek()
    elif case == "union_int64_halves":
        p = _allocator().allocate(Wide)
        p.d.poke(x.to(fx.Int64) * 4294967296 + y.to(fx.Int64))
        p.w.peek()[1] = x + 100
        Out[tid] = (p.d.peek() >> 32).to(fx.Int32) + (p.d.peek() & 0xFFFF).to(fx.Int32)
    elif case == "union_in_struct":
        t = _allocator().allocate(Tagged)
        t.tag.poke(x)
        t.u.f.poke(y.to(fx.Float32))
        Out[tid] = t.tag.peek() + t.u.i.peek()
    elif case == "union_sgpr":
        p = _allocator("SGPR").allocate(Halves)
        p.word.poke(fx.Int32(fx.block_idx.x) * 65536 + 0x1234)
        p.halves.peek()[0] = fx.Uint16(0x55)
        Out[tid] = p.word.peek() + x
    elif case == "u16_pair_sgpr_word":
        p = _allocator("SGPR").allocate(ShortWord)
        p.pair.lo.poke((fx.Int32(fx.block_idx.x) + 3).to(fx.Uint16))
        p.pair.hi.poke(fx.Uint16(7))
        Out[tid] = p.word.peek() + x
    elif case == "f16_pair_separate":
        p = _allocator().allocate(HalfPair)
        p.lo.poke(x.to(fx.Float16))
        p.hi.poke(y.to(fx.Float16))
        Out[tid] = (p.lo.peek().to(fx.Float32) * 2.0 + p.hi.peek().to(fx.Float32)).to(fx.Int32)
    elif case == "u16_pair_separate":
        p = _allocator().allocate(ShortWord)
        p.pair.lo.poke(x.to(fx.Uint16))
        p.pair.hi.poke(y.to(fx.Uint16))
        Out[tid] = p.pair.lo.peek().to(fx.Int32) + p.pair.hi.peek().to(fx.Int32) * 2
    elif case == "i8x4_separate":
        b = _allocator().allocate(Bytes4)
        b.b0.poke(x.to(fx.Uint8))
        b.b1.poke(y.to(fx.Uint8))
        b.b2.poke((x + 1).to(fx.Uint8))
        b.b3.poke((y + 2).to(fx.Uint8))
        Out[tid] = (
            b.b0.peek().to(fx.Int32)
            + b.b1.peek().to(fx.Int32) * 2
            + b.b2.peek().to(fx.Int32) * 4
            + b.b3.peek().to(fx.Int32) * 8
        )
    elif case == "f16_array3_separate":
        arr = _allocator().allocate(fx.Array[fx.Float16, 3]).peek()
        arr[0] = x.to(fx.Float16)
        arr[1] = y.to(fx.Float16)
        arr[2] = (x + y).to(fx.Float16)
        Out[tid] = (arr[0].to(fx.Float32) + arr[1].to(fx.Float32) * 2.0 + arr[2].to(fx.Float32) * 4.0).to(fx.Int32)
    elif case == "f16_scalar_arith":
        a = _allocator().allocate(fx.Float16)
        a.poke(x.to(fx.Float16))
        Out[tid] = (a.peek().to(fx.Float32) * 2.0).to(fx.Int32)
    elif case == "f16_scalar_update":
        a = _allocator().allocate(fx.Float16)
        a.poke(x.to(fx.Float16))
        a.poke(a.peek() * a.peek() + y.to(fx.Float16))
        a.poke(a.peek() - fx.Float16(1.0))
        Out[tid] = a.peek().to(fx.Int32)
    elif case == "f16_scalar_loop":
        a = _allocator().allocate(fx.Float16)
        a.poke(x.to(fx.Float16))
        for _ in range(0, x & 3):
            a.poke(a.peek() + y.to(fx.Float16))
        Out[tid] = (a.peek().to(fx.Float32) * 2.0).to(fx.Int32)
    elif case == "f16_hi_only":
        p = _allocator().allocate(HalfPair)
        p.hi.poke(x.to(fx.Float16))
        Out[tid] = (p.hi.peek().to(fx.Float32) * 2.0).to(fx.Int32)
    elif case == "i8_scalar_arith":
        a = _allocator().allocate(fx.Int8)
        a.poke(x.to(fx.Int8))
        Out[tid] = a.peek().to(fx.Int32) * 3 + y
    elif case == "i16_scalar_arith":
        a = _allocator().allocate(fx.Int16)
        a.poke((x * y).to(fx.Int16))
        Out[tid] = a.peek().to(fx.Int32) + x


@flyc.jit
def launch_subword(X: fx.Tensor, Y: fx.Tensor, Out: fx.Tensor, case: fx.Constexpr[str], mode: fx.Constexpr[str]):
    subword_kernel(X, Y, Out, case, mode).launch(grid=(1, 1, 1), block=(64, 1, 1))


def _expected(case, x, y):
    import torch

    def f16_bits(t):
        return t.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF

    def f16(t):
        return t.to(torch.float16).to(torch.float32)

    if case == "f16_pair_word":
        return f16_bits(x) | (f16_bits(y) << 16)
    if case == "f16_vector":
        a, b = x.to(torch.float16), y.to(torch.float16)
        return ((a * a + a) + (b * b + b)).to(torch.int32)
    if case == "f16_pair_loop":
        a, b = x.to(torch.float16), y.to(torch.float16).clone()
        for i in range(x.numel()):
            for _ in range(int(x[i]) & 3):
                b[i] = b[i] + a[i]
        return (a.float() + b.float() * 2).to(torch.int32)
    if case == "i8x4_word":
        return (x & 255) | ((y & 255) << 8) | (((x + 1) & 255) << 16) | (((y + 2) & 255) << 24)
    if case == "i8_i16_word":
        return (x & 255) | ((y & 255) << 8) | (((x * y) & 0xFFFF) << 16)
    if case == "union_partial_update":
        return ((x * 65536 + y) & ~0xFFFF) | ((y + 5) & 0xFFFF)
    if case == "union_word_to_halves":
        return y + x
    if case == "union_word_to_bytes":
        return y + x * 2 + y * 4 + x * 8
    if case == "union_float_int":
        return (x.to(torch.float32) * 0.5).view(torch.int32)
    if case == "union_int64_halves":
        return (x + 100) + (y & 0xFFFF)
    if case == "union_in_struct":
        return x + y.to(torch.float32).view(torch.int32)
    if case == "union_sgpr":
        return x + 0x55
    if case == "u16_pair_sgpr_word":
        return x + (3 | (7 << 16))
    if case == "f16_pair_separate":
        return (f16(x) * 2 + f16(y)).to(torch.int32)
    if case == "u16_pair_separate":
        return x + y * 2
    if case == "i8x4_separate":
        return x + y * 2 + (x + 1) * 4 + (y + 2) * 8
    if case == "f16_array3_separate":
        return (f16(x) + f16(y) * 2 + f16(x + y) * 4).to(torch.int32)
    if case == "f16_scalar_arith":
        return (f16(x) * 2).to(torch.int32)
    if case == "f16_scalar_update":
        return x * x + y - 1
    if case == "f16_scalar_loop":
        return (x + (x & 3) * y) * 2
    if case == "f16_hi_only":
        return x * 2
    if case == "i8_scalar_arith":
        return x * 3 + y
    if case == "i16_scalar_arith":
        return x * y + x
    raise KeyError(case)


def run_case(case, mode, *, run=False):
    import contextlib
    from unittest import mock

    import torch

    device = "cuda" if run else "cpu"
    tid = torch.arange(64, dtype=torch.int32)
    x, y = tid % 13 + 1, tid % 7 + 3
    out = torch.zeros(64, dtype=torch.int32, device=device)
    # "plain" compiles the same program as ordinary register memory, without placement.
    no_placement = (
        mock.patch("flydsl.experimental.register_allocation.register.register_place", lambda *args, **kwargs: None)
        if mode == "plain"
        else contextlib.nullcontext()
    )
    with no_placement:
        launch_subword(
            flyc.from_torch_tensor(x.to(device)),
            flyc.from_torch_tensor(y.to(device)),
            flyc.from_torch_tensor(out),
            case,
            mode,
        )
    if run:
        torch.cuda.synchronize()
        expected = _expected(case, x, y).to(torch.int32)
        assert torch.equal(out.cpu(), expected), (case, mode, out.cpu(), expected)


def _compile(tmp_path, case, mode, dump=False):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH="gfx942",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    if dump:
        env.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path / "ir"))
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from tests.python.experimental.register_allocation.test_register_subword_union import run_case
run_case({case!r}, {mode!r})
"""
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)


def _written_registers(isa):
    written = set()
    for line in isa.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2 or not re.fullmatch(r"[a-z][a-z0-9_]*", parts[0]):
            continue
        dst = re.fullmatch(r"([sv])(?:\[(\d+):(\d+)\]|(\d+))", parts[1].split(",")[0].strip())
        if dst:
            first, last = (int(dst[2]), int(dst[3])) if dst[2] else (int(dst[4]), int(dst[4]))
            written.update(f"{dst[1]}{r}" for r in range(first, last + 1))
    return written


@pytest.mark.parametrize("case", sorted(SUPPORTED))
def test_supported_fixed_placement_writes_requested_registers(tmp_path, case):
    result = _compile(tmp_path, case, "fixed", dump=True)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / "ir").rglob("*_final_isa.s"))
    assert len(files) == 1
    isa = files[0].read_text()
    assert SUPPORTED[case] <= _written_registers(isa), isa


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_unsupported_subword_patterns_are_rejected(tmp_path, case):
    result = _compile(tmp_path, case, "fixed")
    assert result.returncode != 0, f"{case} now compiles with placement; move it to SUPPORTED"
    assert "placement cannot be enforced" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_unsupported_subword_patterns_compile_without_placement(tmp_path, case):
    result = _compile(tmp_path, case, "plain")
    assert result.returncode == 0, result.stdout + result.stderr
