# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Op locations name the user's call chain: the standard library is not a user frame, and with debug info every
frame of the chain stays one inlined level of the line table without changing the generated code."""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_function import _create_mlir_context
from flydsl.expr import meta

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

SOURCE = Path(__file__).read_text().splitlines()


def line_of(marker: str) -> int:
    return next(i for i, text in enumerate(SOURCE, 1) if text.rstrip().endswith(f"# {marker}"))


def _scale(frag):
    frag.store(frag.load() * 2.0)  # SCALE


def _outer(frag):
    _scale(frag)  # OUTER_CALL


def set_compile_env(monkeypatch, debug: bool, **env: str) -> None:
    """Compile for gfx942 without a GPU, with or without debug info."""
    for name, value in {"ARCH": "gfx942", "COMPILE_ONLY": "1", **env}.items():
        monkeypatch.setenv(name, value)
    if debug:
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    else:
        monkeypatch.delenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", raising=False)


def compile_kernel(monkeypatch, dump_dir: Path, debug: bool) -> Path:
    """Compile make_launch's kernel; returns the dump directory of its module."""
    set_compile_env(
        monkeypatch, debug, FLYDSL_RUNTIME_ENABLE_CACHE="0", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(dump_dir)
    )
    make_launch()(torch.empty(1, 64))
    (module,) = {p.parent for p in dump_dir.glob("*/*_final_isa.s")}
    return module


def make_launch():
    """The launch of a kernel that calls _outer from two lines (it calls _scale)."""

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

    return launch


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


def test_each_thread_keeps_its_own_debug_scopes():
    # Concurrent compiles build scopes in their own contexts. A cache shared between threads lets one thread
    # put another context's scope into its locations, and the process crashes once that context is freed.
    # Run one after the other, the threads show whether one thread's compile replaces the other's scopes.
    contexts = {}

    def scale_scope(thread: str):
        return meta._frame_scope(_scale.__code__, contexts.setdefault(thread, _create_mlir_context()))

    with ThreadPoolExecutor(1) as a, ThreadPoolExecutor(1) as b:
        first = a.submit(scale_scope, "a").result()
        b.submit(scale_scope, "b").result()
        assert a.submit(scale_scope, "a").result() == first, "another thread's compile replaced this thread's scopes"


def test_a_cached_compile_at_another_loc_depth_is_not_served(tmp_path, monkeypatch):
    # The depth decides which frames of a chain the line table keeps.
    set_compile_env(
        monkeypatch,
        True,
        FLYDSL_RUNTIME_ENABLE_CACHE="1",
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path),
        FLYDSL_DUMP_IR="0",
    )
    for depth in ("1", "8"):
        monkeypatch.setenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", depth)
        make_launch()(torch.empty(1, 64))
    assert len(list(tmp_path.glob("*/*.pkl"))) == 2, "the depth-8 compile was served the depth-1 artifact"


GRID_SIZE_PROBE = """
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


def _grid_col():
    return fx.grid_dim.x


@flyc.kernel
def grid_kernel(x: fx.Tensor):
    tile = fx.slice(fx.zipped_divide(x, (1, 1)), (None, _grid_col()))
    frag = fx.make_fragment_like(tile)
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    fx.copy(atom, tile, frag)
    fx.copy(atom, frag, tile)


@flyc.jit
def launch(x: fx.Tensor):
    grid_kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1))


launch(torch.empty(1, 64))
"""


def test_debug_info_compiles_a_helper_that_reads_the_grid_size(tmp_path):
    # Reading the grid size declares __ockl_get_num_groups under the helper's call chain. If the chain's scope
    # were the helper's DISubprogram itself, the declaration would take it as its own and LLVM would reject the
    # module, then crash the process; so the compile runs in a child.
    probe = tmp_path / "probe.py"
    probe.write_text(GRID_SIZE_PROBE)
    dump_dir = tmp_path / "dump"
    probe_env = {
        **os.environ,
        "ARCH": "gfx942",
        "COMPILE_ONLY": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        "FLYDSL_DEBUG_ENABLE_DEBUG_INFO": "1",
        "FLYDSL_DUMP_IR": "1",
        "FLYDSL_DUMP_DIR": str(dump_dir),
    }
    result = subprocess.run([sys.executable, str(probe)], env=probe_env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    llvm_ir = next(dump_dir.glob("*/*_llvm_ir.ll")).read_text()
    assert re.search(r"^declare .*@__ockl_get_num_groups\(", llvm_ir, re.M), "the kernel no longer declares a function"
