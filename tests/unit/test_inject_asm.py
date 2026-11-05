# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""tools/inject_asm/inject_asm.py against real cache entries, compiled without a GPU."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("inject_asm", REPO_ROOT / "tools" / "inject_asm" / "inject_asm.py")
inject_asm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inject_asm)

_CLANG = inject_asm._rocm_path() / "llvm" / "bin" / "clang"
requires_clang = pytest.mark.skipif(not _CLANG.is_file(), reason=f"ROCm clang not found at {_CLANG}")


# _DRIVER compiles these in a child process, importing them from this file: the
# AST rewriter reads kernels back with inspect.getsource.
@flyc.kernel
def tiny_kernel():
    pass


@flyc.jit
def single():
    tiny_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))


@flyc.jit
def sized(block: fx.Constexpr[int]):
    # Each value is its own cache entry, all launching the same kernel symbol.
    tiny_kernel().launch(grid=(1, 1, 1), block=(block, 1, 1))


_DRIVER = "import sys\nfrom tests.unit import test_inject_asm as k\nfor c in sys.argv[1:]:\n    eval(c, vars(k))\n"

# A hand-written stand-in for an edited dump of tiny_kernel.
ISA = """\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
\t.amdhsa_code_object_version 6
\t.text
\t.globl\ttiny_kernel_0
\t.p2align\t8
\t.type\ttiny_kernel_0,@function
tiny_kernel_0:
\ts_endpgm
.Lfunc_end0:
\t.size\ttiny_kernel_0, .Lfunc_end0-tiny_kernel_0

\t.amdhsa_kernel tiny_kernel_0
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 0
\t\t.amdhsa_next_free_vgpr 4
\t\t.amdhsa_next_free_sgpr 1
\t\t.amdhsa_accum_offset 4
\t.end_amdhsa_kernel
"""


@pytest.fixture
def compile_env(monkeypatch):
    """Compile for gfx950 without a GPU: COMPILE_ONLY still writes the cache entry."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO_ROOT), *sys.path, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)


def _refused(*argv) -> str:
    with pytest.raises(SystemExit) as excinfo:
        inject_asm.main(argv)
    return str(excinfo.value.code)


@pytest.mark.parametrize("relative", [".flydsl/cache", ".flydsl", ".flydsl/cache/single_0123"])
def test_refuses_anything_overlapping_the_default_cache(tmp_path, monkeypatch, relative):
    """A patched entry in the default cache would be served to every later ordinary run.

    Entries are found recursively, so a parent of the default cache reaches it as
    surely as the cache itself, and a subdirectory is part of it.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    msg = _refused("inject", tmp_path / "home" / relative, tmp_path / "edited.s")
    assert "overlaps the default cache" in msg


@requires_clang
def test_inject_patches_and_checks(tmp_path, compile_env):
    cache, asm = tmp_path / "cache", tmp_path / "edited.s"
    env = {**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(cache)}
    subprocess.run([sys.executable, "-c", _DRIVER, "single()"], env=env, check=True, timeout=600)
    (entry,) = cache.rglob("*.pkl")
    compiled = entry.read_bytes()
    asm.write_text(ISA)

    inject_asm.main(["inject", cache, asm])
    assert entry.read_bytes() != compiled
    assert not asm.with_suffix(".hsaco").exists(), "saving the code object is opt-in"

    # --check passes only if the entry now holds what the .s assembles to, and
    # writes nothing: the cache is written by atomic replace, so any write shows up
    # as a new inode even when the bytes come out the same.
    patched = (entry.read_bytes(), entry.stat().st_ino)
    inject_asm.main(["inject", cache, asm, "--check"])
    asm.write_text(ISA.replace("\ts_endpgm\n", "\ts_nop 0\n\ts_endpgm\n"))
    assert "differs" in _refused("inject", cache, asm, "--check")
    assert (entry.read_bytes(), entry.stat().st_ino) == patched

    inject_asm.main(["inject", cache, asm, "--save-hsaco"])
    assert asm.with_suffix(".hsaco").read_bytes() == inject_asm.assemble(asm, "gfx950")


@requires_clang
def test_prepare_sets_up_a_workspace(tmp_path, monkeypatch, capsys, compile_env):
    # Exported knobs must not leak into prepare's runs: with FLYDSL_DUMP_IR the
    # caching run would write no entry, and RUN_ONLY rejects the dump run outright.
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    # --clang must only assemble, leaving the link to the toolkit ld.lld as in the
    # compiler: a clang that links by itself uses the lld beside it, which can lay
    # out the same code differently.
    marker, clang = tmp_path / "clang-was-called", tmp_path / "clang"
    clang.write_text(
        f'#!/bin/sh\ncase " $* " in *" -c "*) ;; *) echo "asked to link" >&2; exit 1;; esac\n'
        f'touch "{marker}"\nexec "{_CLANG}" "$@"\n'
    )
    clang.chmod(0o755)
    ws = tmp_path / "ws"
    # Relative on purpose: Path("./clang") prints as "clang", which subprocess would look up on PATH.
    monkeypatch.chdir(tmp_path)

    # --clang ahead of the command: the command's own flags must not be read as ours, and ours not as the command.
    inject_asm.main(["prepare", ws, "--clang", "./clang", "--", sys.executable, "-c", _DRIVER, "single()"])

    original, edited = ws / "original.s", ws / "edited.s"
    assert inject_asm.kernel_names_in_isa(original.read_text()) == {"tiny_kernel_0"}
    assert edited.read_text() == original.read_text()
    assert stat.S_IMODE(original.stat().st_mode) == 0o444
    assert json.loads((ws / "prepare.json").read_text())["clang"] == str(clang.resolve())
    assert marker.exists()
    assert "check passed" in capsys.readouterr().out


@requires_clang
def test_specializations_are_refused(tmp_path, compile_env):
    """Two constexpr values launch the same symbol: the dump keeps one, the cache holds both.

    Patching one of the entries by guess would, half the time, leave the run
    executing the compiler's code under RUN_ONLY with no error, so neither prepare
    nor inject picks one.
    """
    ws = tmp_path / "ws"
    msg = _refused("prepare", ws, "--", sys.executable, "-c", _DRIVER, "sized(64)", "sized(128)")
    assert "prepare needs exactly one" in msg

    entries = sorted((ws / "cache").rglob("*.pkl"))
    before = [e.read_bytes() for e in entries]
    (dumped,) = (ws / "dump").rglob("*_final_isa.s")
    assert "matches 2 entries" in _refused("inject", ws / "cache", dumped)
    assert [e.read_bytes() for e in entries] == before
