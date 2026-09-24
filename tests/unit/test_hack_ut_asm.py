# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import os
from pathlib import Path

import pytest

from flydsl.compiler import hack_asm as hack_asm_mod
from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.diagnostics import DSLCompileError
from flydsl.compiler.hack_asm import _assemble_isa_to_hsaco, _infer_kernel_names_from_s

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

ELF_MAGIC = b"\x7fELF"

MINIMAL_ISA = """\t.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
\t.amdhsa_code_object_version 6
\t.text
\t.globl\ttiny_kernel
\t.p2align\t8
\t.type\ttiny_kernel,@function
tiny_kernel:
\ts_endpgm
.Lfunc_end0:
\t.size\ttiny_kernel, .Lfunc_end0-tiny_kernel

\t.amdhsa_kernel tiny_kernel
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 0
\t\t.amdhsa_next_free_vgpr 4
\t\t.amdhsa_next_free_sgpr 1
\t\t.amdhsa_accum_offset 4
\t.end_amdhsa_kernel
"""


_CLANG = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "llvm" / "bin" / "clang"
requires_clang = pytest.mark.skipif(not _CLANG.is_file(), reason=f"ROCm clang not found at {_CLANG}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_infer_kernel_names_from_s():
    names = _infer_kernel_names_from_s(MINIMAL_ISA)
    assert names == {"tiny_kernel"}


def test_infer_kernel_names_from_s_ignores_unrelated_directives():
    assert _infer_kernel_names_from_s("\t.text\n\t.p2align 8\n") == set()


def test_isa_assemble_arch_matches_pipeline():
    backend = RocmBackend(RocmBackend.make_target("gfx950"))
    assert backend.isa_assemble_arch() == "gfx950"


@requires_clang
def test_assemble_isa_to_hsaco_roundtrip(tmp_path):
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    obj = _assemble_isa_to_hsaco(src, "gfx950", _digest(src))
    assert obj.startswith(ELF_MAGIC)
    assert b"tiny_kernel" in obj


@requires_clang
def test_assemble_isa_to_hsaco_reports_assembler_diagnostics(tmp_path):
    src = tmp_path / "bad.s"
    src.write_text("this is not valid asm\n", encoding="utf-8")
    with pytest.raises(DSLCompileError) as excinfo:
        _assemble_isa_to_hsaco(src, "gfx950", _digest(src))
    # The clang stderr must reach the user, not be swallowed into a generic message.
    assert "invalid instruction" in str(excinfo.value)


@requires_clang
def test_assemble_isa_to_hsaco_rejects_arch_mismatch(tmp_path):
    """A .s dumped for another arch must be refused, not silently assembled."""
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    with pytest.raises(DSLCompileError) as excinfo:
        _assemble_isa_to_hsaco(src, "gfx942", _digest(src))
    assert "target id" in str(excinfo.value)


# --- Scoping: a .s applies only to the kernels it names -----------------------------


@pytest.fixture
def clean_hack_state(monkeypatch):
    """Isolate the process-level match tracking from other tests."""
    log = hack_asm_mod._MatchLog()
    log.hooked = True  # suppress the atexit registration
    monkeypatch.setattr(hack_asm_mod, "_MATCH_LOG", log)
    monkeypatch.setattr(hack_asm_mod, "_ASSEMBLE_CACHE", {})


def test_unmatched_module_is_skipped_not_fatal(tmp_path, monkeypatch, clean_hack_state, capsys):
    """A kernel the .s does not name keeps compiler codegen instead of aborting the run.

    This is what lets a benchmark compile the hand-edited kernel alongside untouched
    ones; the previous behaviour raised and took the whole process down.
    """
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    monkeypatch.setenv("FLYDSL_HACK_UT_ASM", str(src))

    applied = hack_asm_mod.substitute_hacked_asm(
        None,  # never reached: the name check returns before touching the module
        arch="gfx950",
        func_name="some_other_kernel",
        module_kernel_names=["some_other_kernel"],
    )
    assert applied is False
    assert "keeps compiler codegen" in capsys.readouterr().out
    assert hack_asm_mod._MATCH_LOG.skipped == {"some_other_kernel"}
    assert hack_asm_mod._MATCH_LOG.matched == set()


def test_report_unmatched_warns_when_nothing_matched(clean_hack_state, capsys):
    """A .s that matched no kernel at all must say so: those are the compiler's numbers."""
    hack_asm_mod._MATCH_LOG.skipped.update({"kernel_a", "kernel_b"})
    hack_asm_mod._MATCH_LOG.report()
    out = capsys.readouterr().out
    assert "never matched a compiled kernel" in out
    assert "NOT the hand-edit's numbers" in out


def test_report_unmatched_silent_when_something_matched(clean_hack_state, capsys):
    """Skipping unrelated kernels is normal and must not warn."""
    hack_asm_mod._MATCH_LOG.matched.add("tiny_kernel")
    hack_asm_mod._MATCH_LOG.skipped.add("unrelated_kernel")
    hack_asm_mod._MATCH_LOG.report()
    assert capsys.readouterr().out == ""


def test_report_unmatched_silent_when_never_used(clean_hack_state, capsys):
    """No kernels seen at all (var set, nothing compiled) is not a warning."""
    hack_asm_mod._MATCH_LOG.report()
    assert capsys.readouterr().out == ""


@requires_clang
def test_assemble_is_cached_across_kernels(tmp_path, clean_hack_state):
    """N kernels must not pay N clang invocations for the same .s."""
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")

    first = _assemble_isa_to_hsaco(src, "gfx950", _digest(src))
    assert len(hack_asm_mod._ASSEMBLE_CACHE) == 1
    second = _assemble_isa_to_hsaco(src, "gfx950", _digest(src))
    assert second is first  # same object, not merely equal: the cache was hit
    assert len(hack_asm_mod._ASSEMBLE_CACHE) == 1


@requires_clang
def test_assemble_cache_follows_edits(tmp_path, clean_hack_state):
    """Editing the .s within one process must not serve the stale object."""
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    first = _assemble_isa_to_hsaco(src, "gfx950", _digest(src))

    # An equal-length edit: the file keeps its size, and on a coarse-granularity
    # filesystem it can keep its mtime too.  Keying the cache on content rather than
    # on stat is what makes this invalidate.
    src.write_text(MINIMAL_ISA.replace("group_segment_fixed_size 0", "group_segment_fixed_size 8"), encoding="utf-8")
    second = _assemble_isa_to_hsaco(src, "gfx950", _digest(src))
    assert second != first


def test_partial_s_is_refused(tmp_path, monkeypatch, clean_hack_state):
    """A .s covering only some of the module's kernels must not replace the binary.

    The whole gpu.binary is swapped, so the kernels the .s omits would vanish from the
    code object while the host still launches them by name.
    """
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    monkeypatch.setenv("FLYDSL_HACK_UT_ASM", str(src))

    with pytest.raises(DSLCompileError) as excinfo:
        hack_asm_mod.substitute_hacked_asm(
            None,  # never reached: the coverage check runs before the module is touched
            arch="gfx950",
            func_name="both",
            module_kernel_names=["tiny_kernel", "other_kernel"],
        )
    msg = str(excinfo.value)
    assert "other_kernel" in msg
    assert "would drop those kernels" in msg


def test_module_without_kernels_is_refused(tmp_path, monkeypatch, clean_hack_state):
    """An empty kernel list must raise, not skip: skipping leaves nothing to warn about."""
    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    monkeypatch.setenv("FLYDSL_HACK_UT_ASM", str(src))

    with pytest.raises(DSLCompileError) as excinfo:
        hack_asm_mod.substitute_hacked_asm(
            None,
            arch="gfx950",
            func_name="empty",
            module_kernel_names=[],
        )
    assert "no kernel symbols were found" in str(excinfo.value)
    # The silent path this guards: an empty skip records nothing, so the exit warning
    # would have had nothing to fire on.
    assert hack_asm_mod._MATCH_LOG.skipped == set()


# --- The override must reach the cache key itself ----------------------------------


def test_cache_stamp_expands_user(tmp_path, monkeypatch):
    """A ~ path must be stat'ed, not stamped as unknown.

    substitute_hacked_asm expanduser()s the path, so a stamp taken from the literal
    string would fail its stat and hand back the same (path, None, None) before and
    after every edit -- the artifact would never be invalidated.
    """
    from flydsl.compiler.jit_function import _hack_asm_cache_stamp

    src = tmp_path / "tiny.s"
    src.write_text(MINIMAL_ISA, encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FLYDSL_HACK_UT_ASM", "~/tiny.s")

    stamp = _hack_asm_cache_stamp()
    assert stamp is not None
    assert stamp[1] is not None and stamp[2] is not None, f"path was not stat'ed: {stamp}"

    # Size, not content: this stamp is stat-based by design (it runs on every call),
    # so an equal-length edit inside one mtime tick is knowingly out of its reach --
    # the content hash in hack_asm.py is what covers that case.
    src.write_text(MINIMAL_ISA + "\ts_nop 0\n", encoding="utf-8")
    assert _hack_asm_cache_stamp() != stamp, "an edit must move the stamp"


def test_cache_stamp_is_none_without_override(monkeypatch):
    monkeypatch.delenv("FLYDSL_HACK_UT_ASM", raising=False)
    from flydsl.compiler.jit_function import _hack_asm_cache_stamp

    assert _hack_asm_cache_stamp() is None


def test_save_switch_defaults_off(monkeypatch):
    """Writing the code object to disk is opt-in."""
    from flydsl.utils import env

    monkeypatch.delenv("FLYDSL_HACK_UT_ASM_SAVE", raising=False)
    assert env.debug.hack_ut_asm_save is False
    monkeypatch.setenv("FLYDSL_HACK_UT_ASM_SAVE", "1")
    assert env.debug.hack_ut_asm_save is True
