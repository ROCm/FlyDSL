# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile the allocator through the DSL frontend and inspect final AMDGPU ISA."""

import os
import re
import subprocess
import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@fx.struct
class PackedPair:
    lo: fx.Uint16
    hi: fx.Uint16


@fx.union
class PackedWord:
    pair: PackedPair
    word: fx.Int32


@fx.struct
class Payload:
    scalar: fx.Int32
    vector: fx.Align[fx.Vector[fx.Int32, (2, 2)], 16]
    packed: PackedWord


@fx.union
class Overlay:
    word: fx.Int32
    halves: fx.Array[fx.Uint16, 2]


SCALAR_TYPES = {
    "int8": fx.Int8,
    "int16": fx.Int16,
    "int64": fx.Int64,
    "float16": fx.Float16,
    "float32": fx.Float32,
    "float64": fx.Float64,
}


@flyc.kernel
def allocator_kernel(
    In: fx.Tensor,
    Out: fx.Tensor,
    mode: fx.Constexpr[str],
    bank: fx.Constexpr[str],
    automatic: fx.Constexpr[bool],
    register_alignment: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    reg_class = fx.rocdl.AGPR if bank == "AGPR" else fx.rocdl.VGPR
    regs = fx.RegisterAllocator(
        reg_class, start_offset=None if automatic else 65, register_alignment=register_alignment
    )
    if mode == "scalar":
        scalar = regs.allocate(fx.Align[fx.Int32, 16])
        scalar.poke(In[tid])
        Out[tid] = scalar.peek() + 7
    elif mode == "vector":
        buf = regs.allocate(fx.Align[fx.Vector[fx.Int32, (2, 2)], 16])
        ptr = fx.add_offset(fx.get_iter(In), tid * 4)
        buf.poke(fx.Vector(fx.ptr_load(ptr, fx.Vector.make_type((2, 2), fx.Int32)), (2, 2), fx.Int32))
        Out[tid] = buf.peek().reduce(fx.ReductionOp.ADD)
    elif mode == "sgpr":
        scalar = fx.RegisterAllocator(
            fx.rocdl.SGPR, start_offset=None if automatic else 40, register_alignment=register_alignment
        ).allocate(fx.Int32)
        scalar.poke(fx.Int32(fx.block_idx.x))
        Out[tid] = scalar.peek() + 7
    elif mode == "union":
        buf = regs.allocate(fx.Align[Overlay, 16])
        buf.word.poke(In[tid])
        buf.halves.peek()[1] = In[tid + 64].to(fx.Uint16)
        Out[tid] = buf.word.peek()
    elif mode == "loop":
        buf = regs.allocate(fx.Align[fx.Int32, 16])
        buf.poke(In[tid])
        for i in range(0, In[tid + 64] & 3):
            buf.poke(buf.peek() + i + 1)
        Out[tid] = buf.peek()
    elif mode == "branch":
        buf = regs.allocate(fx.Align[fx.Int32, 16])
        buf.poke(In[tid])
        if (In[tid + 64] & 1) != 0:
            buf.poke(buf.peek() + 7)
        else:
            buf.poke(buf.peek() - 11)
        Out[tid] = buf.peek()
    elif mode == "fma":
        buf = regs.allocate(fx.Align[fx.Float32, 16])
        buf.poke((In[tid] & 31).to(fx.Float32))
        for i in range(0, In[tid + 64] & 3):
            buf.poke(fx.Float32(fx.fma((In[tid + 128] & 7).to(fx.Float32), fx.Int32(i + 1).to(fx.Float32), buf.peek())))
        Out[tid] = buf.peek().to(fx.Int32)
    elif mode.split("_")[0] in SCALAR_TYPES:
        dtype = SCALAR_TYPES[mode.split("_")[0]]
        buf = regs.allocate(fx.Align[dtype, 16])
        buf.poke((In[tid] & 31).to(dtype))
        if mode == "float16":
            Out[tid] = buf.peek().bitcast(fx.Uint16).to(fx.Int32)
        elif mode == "float64":
            Out[tid] = (buf.peek() * buf.peek()).to(fx.Int32)
        else:
            Out[tid] = buf.peek().to(fx.Int32)
    else:
        payload = regs.allocate(Payload)
        payload.scalar.poke(In[tid])
        ptr = fx.add_offset(fx.get_iter(In), 256 + tid * 4)
        payload.vector.poke(fx.Vector(fx.ptr_load(ptr, fx.Vector.make_type((2, 2), fx.Int32)), (2, 2), fx.Int32))
        payload.packed.pair.lo.poke(In[512 + tid].to(fx.Uint16))
        payload.packed.pair.hi.poke(In[768 + tid].to(fx.Uint16))
        packed = (
            payload.packed.pair.lo.peek().to(fx.Int32) + payload.packed.pair.hi.peek().to(fx.Int32)
            if mode == "packed_fields"
            else payload.packed.word.peek()
        )
        Out[tid] = payload.scalar.peek() + payload.vector.peek().reduce(fx.ReductionOp.ADD) + packed


@flyc.jit
def launch_allocator(
    In: fx.Tensor,
    Out: fx.Tensor,
    mode: fx.Constexpr[str],
    bank: fx.Constexpr[str] = "VGPR",
    automatic: fx.Constexpr[bool] = False,
    register_alignment: fx.Constexpr[int] = 1,
):
    allocator_kernel(In, Out, mode, bank, automatic, register_alignment).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.parametrize(
    "mode,registers",
    [
        ("scalar", {68}),
        ("vector", set(range(68, 72))),
        ("packed", {68, 72, 73, 74, 75, 76}),
        ("sgpr", {40}),
        ("union", {68}),
        ("loop", {68}),
        ("branch", {68}),
        ("fma", {68}),
    ]
    + [(mode, {68}) for mode in SCALAR_TYPES],
)
def test_allocator_final_isa(tmp_path, mode, registers):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH="gfx942",
        FLYDSL_DUMP_IR="1",
        FLYDSL_DUMP_DIR=str(tmp_path / "ir"),
        FLYDSL_REGISTER_DUMP_DIR=str(tmp_path / "mir"),
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    result = subprocess.run([sys.executable, __file__, mode], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / "ir").rglob("*_final_isa.s"))
    assert len(files) == 1
    isa = files[0].read_text()
    bank = "s" if mode == "sgpr" else "v"
    used = set()
    for first, last, scalar in re.findall(rf"\b{bank}(?:\[(\d+):(\d+)\]|(\d+)\b)", isa):
        used.update(range(int(first), int(last) + 1) if first else [int(scalar)])
    assert registers <= used, isa
    if mode == "packed":
        # Both half-word writes must survive optimization. Merely mentioning
        # v76 does not establish that the upper half was preserved.
        assert "offset:2048" in isa and "offset:3072" in isa
    mir = "\n".join(p.read_text() for p in (tmp_path / "mir").glob("*.registers.txt"))
    assert mir and "STACKMAP" not in mir
    for register in registers:
        assert f"${bank}gpr{register}" in mir or f"_{bank}gpr{register}" in mir


@pytest.mark.parametrize("mode", ["packed_fields", "float16_roundtrip", "float64_roundtrip"])
def test_allocator_rejects_values_only_used_by_carriers(tmp_path, mode):
    # Keep the earlier arithmetic/roundtrip cases as regressions: LLVM can
    # bypass the packed or converted value in real code, leaving only its
    # marker use. Success must not mean that a dead fixed copy disappears.
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH="gfx942",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from tests.unit.test_register_allocator_codegen import run_allocator
run_allocator({mode!r})
"""
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "placement cannot be enforced" in result.stderr, result.stdout + result.stderr
    script = (
        "import flydsl.expr.register as registers\nregisters.set_register = lambda *args, **kwargs: None\n" + script
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["loop", "branch", "fma", "packed"])
def test_allocator_rejects_agpr_writeback(tmp_path, mode):
    # Allocator syntax must not weaken the same direct-definition rule applied
    # to set_register, including across loops, branches and packed updates.
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH="gfx942",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from tests.unit.test_register_allocator_codegen import run_allocator
run_allocator({mode!r}, bank="AGPR")
"""
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "automatic write-back copies are disabled" in result.stderr, result.stdout + result.stderr
    script = (
        "import flydsl.expr.register as registers\nregisters.set_register = lambda *args, **kwargs: None\n" + script
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def run_allocator(mode, *, run=False, bank="VGPR", automatic=False, register_alignment=1):
    import torch

    device = "cuda" if run else "cpu"
    source = torch.arange(1024, dtype=torch.int32, device=device) * 7919 - 50000
    out = torch.empty(64, dtype=torch.int32, device=device)
    launch_allocator(
        flyc.from_torch_tensor(source), flyc.from_torch_tensor(out), mode, bank, automatic, register_alignment
    )
    if run:
        torch.cuda.synchronize()
        if mode == "scalar":
            expected = source[:64] + 7
        elif mode == "vector":
            expected = source[:256].reshape(64, 4).sum(dim=1)
        elif mode == "sgpr":
            expected = torch.full_like(out, 7)
        elif mode == "union":
            expected = (source[:64] & 65535) | ((source[64:128] & 65535) << 16)
        elif mode == "loop":
            n = source[64:128] & 3
            expected = source[:64] + n * (n + 1) // 2
        elif mode == "branch":
            expected = source[:64] + torch.where((source[64:128] & 1) != 0, 7, -11)
        elif mode == "fma":
            n = source[64:128] & 3
            expected = (source[:64] & 31) + (source[128:192] & 7) * n * (n + 1) // 2
        elif mode == "float16":
            expected = (source[:64] & 31).to(torch.float16).view(torch.int16).to(torch.int32) & 65535
        elif mode == "float64":
            expected = (source[:64] & 31) ** 2
        elif mode in SCALAR_TYPES:
            expected = source[:64] & 31
        else:
            expected = (
                source[:64]
                + source[256:512].reshape(64, 4).sum(dim=1)
                + ((source[512:576] & 65535) | ((source[768:832] & 65535) << 16))
            ).to(torch.int32)
        assert torch.equal(out, expected), (out, expected)
        print(f"{bank} {mode}: numeric verification passed", flush=True)


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run_allocator(sys.argv[1], run="--run" in sys.argv)
