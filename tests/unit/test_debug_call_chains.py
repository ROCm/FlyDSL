# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Op locations name the user's call chain: the standard library is not a user frame, and with debug info every
frame of the chain stays one inlined level of the line table without changing the generated code."""

import re
from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

SOURCE = Path(__file__).read_text().splitlines()


def line_of(marker: str) -> int:
    return next(i for i, text in enumerate(SOURCE, 1) if text.rstrip().endswith(f"# {marker}"))


def _scale(frag):
    frag.store(frag.load() * 2.0)  # SCALE


def _outer(frag):
    _scale(frag)  # OUTER_CALL


def compile_kernel(monkeypatch, dump_dir: Path, debug: bool) -> Path:
    """Compile a kernel that calls _outer from two lines (it calls _scale), without a GPU; returns the dump
    directory of its module."""
    for name, value in {
        "ARCH": "gfx942",
        "COMPILE_ONLY": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        "FLYDSL_DUMP_IR": "1",
        "FLYDSL_DUMP_DIR": str(dump_dir),
    }.items():
        monkeypatch.setenv(name, value)
    if debug:
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    else:
        monkeypatch.delenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", raising=False)

    @flyc.kernel
    def twice(x: fx.Tensor):
        tile = fx.slice(fx.zipped_divide(x, (1, 1)), (None, 0))
        frag = fx.make_fragment_like(tile)
        atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        fx.copy(atom, tile, frag)
        _outer(frag)  # CALL_A
        _outer(frag)  # CALL_B
        fx.copy(atom, frag, tile)

    @flyc.jit
    def launch(x: fx.Tensor):
        twice(x).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(torch.empty(1, 64))
    (module,) = {p.parent for p in dump_dir.glob("*/*_final_isa.s")}
    return module


def inlined_chains(llvm_ir: str, line: int) -> set[tuple[int, ...]]:
    """The line chains (innermost first, through inlinedAt) of every DILocation on ``line``."""
    locs = dict(re.findall(r"^(!\d+) = !DILocation\((.*)\)$", llvm_ir, re.M))

    def chain(ref: str) -> tuple[int, ...]:
        out = []
        while ref in locs:
            out.append(int(re.search(r"line: (\d+)", locs[ref])[1]))
            nxt = re.search(r"inlinedAt: (!\d+)", locs[ref])
            if not nxt:
                break
            ref = nxt[1]
        return tuple(out)

    return {c for ref in locs if (c := chain(ref)) and c[0] == line}


def instructions(asm: str) -> list[str]:
    """The instruction lines of an assembly file, without debug sections, directives and labels."""
    out, in_debug = [], False
    for text in asm.splitlines():
        s = text.split(";", 1)[0].split("//", 1)[0].strip()
        if s.startswith(".section"):
            in_debug = ".debug_" in s
        if in_debug or not s or s.startswith(".") or s.endswith(":"):
            continue
        out.append(s)
    return out


def test_arithmetic_ops_point_at_the_users_line_not_at_functools(tmp_path, monkeypatch):
    # ArithValue's operators are functools.partialmethods: their wrapper frame is not the user's.
    module = compile_kernel(monkeypatch, tmp_path, debug=False)
    origin = next(module.glob("00_*.mlir")).read_text()
    assert "functools.py" not in origin
    assert re.search(rf'loc\("{re.escape(__file__)}":{line_of("SCALE")}:', origin)


def test_debug_info_keeps_every_frame_of_the_call_chain(tmp_path, monkeypatch):
    module = compile_kernel(monkeypatch, tmp_path, debug=True)
    llvm_ir = next(module.glob("*_llvm_ir.ll")).read_text()
    scale, outer, call_a, call_b = (line_of(m) for m in ("SCALE", "OUTER_CALL", "CALL_A", "CALL_B"))
    # each call of _outer keeps its own line, and _outer's call of _scale stays between them
    assert {(scale, outer, call_a), (scale, outer, call_b)} <= inlined_chains(llvm_ir, scale)
    assert re.search(r'!DISubprogram\(name: "_outer"', llvm_ir) and re.search(r'!DISubprogram\(name: "_scale"', llvm_ir)


def test_debug_info_does_not_change_the_generated_code(tmp_path, monkeypatch):
    plain = compile_kernel(monkeypatch, tmp_path / "plain", debug=False)
    debug = compile_kernel(monkeypatch, tmp_path / "debug", debug=True)
    plain_asm = next(plain.glob("*_final_isa.s")).read_text()
    debug_asm = next(debug.glob("*_final_isa.s")).read_text()
    has_lines = re.compile(r"^\s*\.loc\s", re.M)
    assert has_lines.search(debug_asm) and not has_lines.search(plain_asm)
    assert instructions(debug_asm) == instructions(plain_asm)
