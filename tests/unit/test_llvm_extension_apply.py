# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LLVM patch series: build_llvm.sh's apply step and scripts/llvm_extension.sh.

The apply step is sliced verbatim from build_llvm.sh and run against a fixture
repository, so no LLVM, network or CMake is needed.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_LLVM = _REPO_ROOT / "scripts" / "build_llvm.sh"
_STATE = ".flydsl-extension-in-progress"
_PADDED = "one\n" + "".join(f"pad{i}\n" for i in range(12)) + "three\n"  # hunks that do not overlap

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(shutil.which("git") is None, reason="git is required"),
]


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _checkout(tmp_path, files):
    """An llvm-project stand-in whose single commit is the pin."""
    work = tmp_path / "llvm-project"
    for name, text in files.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(text)
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "pin")
    return work, _git(work, "rev-parse", "HEAD").stdout.strip()


def _patch(work, out, before, after, file="src/target.c"):
    """Write the diff turning `before` into `after` in `file`, then revert it."""
    f = work / file
    f.write_text(f.read_text().replace(before, after))
    out.write_text(_git(work, "diff").stdout)
    _git(work, "checkout", "--force", "--", ".")


# ---- build_llvm.sh apply step ------------------------------------------------


def _patch_section():
    """build_llvm.sh from the in-progress guard to the end of the apply loop."""
    lines = _BUILD_LLVM.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(f"if [ -f {_STATE} ]")]
    ends = [i for i, line in enumerate(lines) if line.startswith("LLVM_COMMIT_RESOLVED=")]
    # A moved or duplicated marker must fail loudly, not test some other region.
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], "slice markers moved in build_llvm.sh"
    section = "\n".join(lines[starts[0] : ends[0]])
    for expected in ("git checkout --force", "git apply --index", "LLVM_EXTENSIONS"):
        assert expected in section, f"sliced section is missing {expected!r}"
    return section


def _build_fixture(tmp_path, text="one\ntwo\nthree\n"):
    work, pin = _checkout(tmp_path, {"src/target.c": text})
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    return work, pin, ext_dir


def _run_apply(work, pin, ext_dir, extensions=(), required=(), no_ext="0"):
    def array(name, items):
        return f"{name}=(\n" + "".join(f"    {i}\n" for i in items) + ")"

    runner = ext_dir.parent / "run.sh"
    runner.write_text(
        "\n".join(
            [
                "set -e",
                f'LLVM_REF="{pin}"',
                f'LLVM_EXT_DIR="{ext_dir}"',
                f'LLVM_NO_EXT="{no_ext}"',
                "LLVM_FETCH_ARGS=(--depth 1)",
                array("REQUIRED_PATCHES", required),
                array("LLVM_EXTENSIONS", extensions),
                f'cd "{work}"',
                _patch_section(),
            ]
        )
    )
    return subprocess.run(["bash", str(runner)], capture_output=True, text=True, cwd=work)


def _target(work):
    return (work / "src" / "target.c").read_text()


def test_series_replays_on_a_warm_checkout(tmp_path):
    """B rewrites A's line, so A no longer reverse-applies once both are on: a
    per-patch "already applied?" probe fails here. Replaying from the pin works,
    and must leave build outputs inside the checkout alone."""
    work, pin, ext_dir = _build_fixture(tmp_path)
    _patch(work, ext_dir / "a.patch", "two", "two-A")
    _git(work, "apply", str(ext_dir / "a.patch"))
    _git(work, "commit", "-qam", "wip")  # B is authored on top of A
    _patch(work, ext_dir / "b.patch", "two-A", "two-A-then-B")
    _git(work, "reset", "-q", "--hard", pin)
    (work / "build-flydsl").mkdir()
    (work / "build-flydsl" / "obj.o").write_text("obj")
    (work / "mlir_install").mkdir()
    (work / "mlir_install" / "lib.so").write_text("lib")

    for _ in range(3):
        result = _run_apply(work, pin, ext_dir, ["a.patch", "b.patch"])
        assert result.returncode == 0, result.stderr
    assert _target(work) == "one\ntwo-A-then-B\nthree\n"
    assert (work / "build-flydsl" / "obj.o").exists()
    assert (work / "mlir_install" / "lib.so").exists()


def test_new_file_patch_replays_on_a_warm_checkout(tmp_path):
    """Without --index the file stays untracked and the next apply says "already exists"."""
    work, pin, ext_dir = _build_fixture(tmp_path)
    (work / "src" / "new.c").write_text("new\n")
    _git(work, "add", "-N", "src/new.c")
    (ext_dir / "new.patch").write_text(_git(work, "diff").stdout)
    _git(work, "reset", "-q")
    (work / "src" / "new.c").unlink()

    for _ in range(2):
        result = _run_apply(work, pin, ext_dir, ["new.patch"])
        assert result.returncode == 0, result.stderr


def test_authoring_leftovers_do_not_wedge_the_build(tmp_path):
    """An interrupted llvm_extension.sh leaves a baseline commit and edits to a
    patched file; a plain checkout refuses to move HEAD and aborts every run."""
    work, pin, ext_dir = _build_fixture(tmp_path)
    _patch(work, ext_dir / "a.patch", "two", "TWO")
    _git(work, "apply", str(ext_dir / "a.patch"))
    _git(work, "commit", "-qam", "flydsl: patch baseline (throwaway)")
    (work / "src" / "target.c").write_text(_target(work) + "// WIP\n")

    for _ in range(2):
        result = _run_apply(work, pin, ext_dir, ["a.patch"])
        assert result.returncode == 0, result.stderr
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == pin


def test_build_refuses_while_a_patch_is_being_authored(tmp_path):
    work, pin, ext_dir = _build_fixture(tmp_path)
    (work / _STATE).write_text("wip.patch\n")
    (work / "src" / "target.c").write_text("author edits\n")

    result = _run_apply(work, pin, ext_dir)

    assert result.returncode == 1
    assert "being authored" in result.stderr
    assert _target(work) == "author edits\n"
    assert f'{_STATE}"' in (_REPO_ROOT / "scripts" / "llvm_extension.sh").read_text()


@pytest.mark.parametrize(
    "extensions, stray, no_ext, message",
    [
        (["a.patch"], "forgotten.patch", "0", "listed in neither array: forgotten.patch"),
        (["a.patch"], "forgotten.patch", "1", "listed in neither array: forgotten.patch"),
        (["a.patch", "ghost.patch"], None, "0", "does not exist: ghost.patch"),
    ],
    ids=["unlisted", "unlisted-with-no-ext", "missing"],
)
def test_tree_mistakes_fail_before_anything_is_applied(tmp_path, extensions, stray, no_ext, message):
    work, pin, ext_dir = _build_fixture(tmp_path)
    _patch(work, ext_dir / "a.patch", "two", "TWO")
    if stray:
        (ext_dir / stray).write_text("junk")

    result = _run_apply(work, pin, ext_dir, extensions, no_ext=no_ext)

    assert result.returncode == 1
    assert message in result.stderr
    assert "TWO" not in _target(work)


def test_stale_patch_names_it_and_the_recovery(tmp_path):
    work, _, ext_dir = _build_fixture(tmp_path)
    _patch(work, ext_dir / "stale.patch", "two", "TWO")
    (work / "src" / "target.c").write_text("upstream\nmoved\n")
    _git(work, "commit", "-qam", "moved")
    moved = _git(work, "rev-parse", "HEAD").stdout.strip()

    result = _run_apply(work, moved, ext_dir, ["stale.patch"])

    assert result.returncode == 1
    assert "LLVM extension failed to apply: stale.patch" in result.stderr
    assert f"LLVM pin: {moved}" in result.stderr
    assert "llvm_extension.sh --rebase" in result.stderr


@pytest.mark.parametrize("no_ext, with_ext", [("0", True), ("1", False)])
def test_no_ext_drops_extensions_but_keeps_required(tmp_path, no_ext, with_ext):
    """Dropping the required patch too made the control arm a build without lld."""
    work, pin, ext_dir = _build_fixture(tmp_path, _PADDED)
    _patch(work, ext_dir / "req.patch", "one", "ONE")
    _patch(work, ext_dir / "ext.patch", "three", "THREE")

    result = _run_apply(work, pin, ext_dir, ["ext.patch"], ["req.patch"], no_ext)

    assert result.returncode == 0, result.stderr
    assert "ONE" in _target(work)
    assert ("THREE" in _target(work)) == with_ext


def test_env_knobs_reach_the_internal_variables():
    """These assignments sit above the slice the other tests run, so a typo in
    an env name would silently fall back to the default."""
    text = _BUILD_LLVM.read_text()

    assert 'LLVM_NO_EXT="${FLYDSL_LLVM_NO_EXT:-0}"' in text
    assert 'LLVM_REF="${FLYDSL_LLVM_REF:-' in text
    assert 'LLVM_REMOTE="${FLYDSL_LLVM_REMOTE:-' in text


# ---- build_llvm.sh --list-patches ----------------------------------------------


def _list_patches(script=_BUILD_LLVM):
    out = subprocess.run(["bash", str(script), "--list-patches"], check=True, capture_output=True, text=True).stdout
    return [tuple(line.split(None, 1)) for line in out.splitlines()]


def _with_arrays(text, required, extensions):
    """build_llvm.sh `text` with both array assignments replaced."""
    for name, value in (("REQUIRED_PATCHES", required), ("LLVM_EXTENSIONS", extensions)):
        text, n = re.subn(rf"^{name}=\(\n.*?^\)\n", f"{name}={value}\n", text, count=1, flags=re.M | re.S)
        assert n == 1, f"{name} not found in build_llvm.sh"
    return text


def test_repo_patches_are_listed_exactly_once():
    on_disk = {p.name for p in (_REPO_ROOT / "thirdparty" / "llvm-extensions").glob("*.patch")}
    listed = [name for _, name in _list_patches()]

    assert on_disk == set(listed)
    assert len(listed) == len(set(listed))


def test_list_patches_reads_the_arrays_as_bash_does(tmp_path):
    """A sed reader ran a one-line array to end of file and eval'd the whole script."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "thirdparty").mkdir()
    (tmp_path / "thirdparty" / "llvm-build-info.json").write_text('{"upstream":{"llvm_hash":"%s"}}' % ("0" * 40))
    script = tmp_path / "scripts" / "build_llvm.sh"
    script.write_text(_with_arrays(_BUILD_LLVM.read_text(), '("req.patch")  # quoted', "(a.patch b.patch)"))

    assert _list_patches(script) == [("required", "req.patch"), ("extension", "a.patch"), ("extension", "b.patch")]


# ---- scripts/llvm_extension.sh -------------------------------------------------


def _authoring(tmp_path, required=()):
    """An llvm-project checkout plus a FlyDSL tree holding the real scripts."""
    work, pin = _checkout(tmp_path, {"a.c": "x\n"})
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "thirdparty" / "llvm-extensions").mkdir(parents=True)
    shutil.copy(_REPO_ROOT / "scripts" / "llvm_extension.sh", repo / "scripts")
    body = "(\n" + "".join(f"    {r}\n" for r in required) + ")"
    (repo / "scripts" / "build_llvm.sh").write_text(_with_arrays(_BUILD_LLVM.read_text(), body, "(\n)"))
    (repo / "thirdparty" / "llvm-build-info.json").write_text('{"upstream":{"llvm_hash":"%s"}}' % pin)
    return work, pin, repo


def _author(repo, work, *args, **env):
    clean = {k: v for k, v in os.environ.items() if k not in ("FLYDSL_LLVM_REF", "LLVM_REF", "LLVM_COMMIT")}
    return subprocess.run(
        ["bash", "scripts/llvm_extension.sh", *args],
        cwd=repo,
        env={**clean, "LLVM_SRC_DIR": str(work), **env},
        capture_output=True,
        text=True,
    )


def test_authoring_resets_to_the_base_before_replaying(tmp_path):
    """`git checkout -- .` leaves HEAD alone, so a leftover commit became the base."""
    work, pin, repo = _authoring(tmp_path)
    (work / "a.c").write_text("x\ny\n")
    _git(work, "commit", "-qam", "leftover")

    result = _author(repo, work, "feat")

    assert result.returncode == 0, result.stderr
    assert _git(work, "rev-parse", "HEAD~1").stdout.strip() == pin
    assert (work / "a.c").read_text() == "x\n"


def test_finish_captures_new_files_and_keeps_build_outputs(tmp_path):
    work, pin, repo = _authoring(tmp_path)
    (work / "mlir_install").mkdir()
    (work / "mlir_install" / "lib.so").write_text("lib")
    (work / "mlir_install.tgz").write_text("tgz")
    assert _author(repo, work, "feat").returncode == 0
    (work / "a.c").write_text("x\ny\n")
    (work / "new.c").write_text("new\n")

    result = _author(repo, work, "--finish")

    assert result.returncode == 0, result.stderr
    patch = (repo / "thirdparty" / "llvm-extensions" / "feat.patch").read_text()
    assert "b/new.c" in patch and "+y" in patch
    assert "mlir_install" not in patch and _STATE not in patch
    assert (work / "mlir_install" / "lib.so").exists() and (work / "mlir_install.tgz").exists()
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == pin


def test_rebase_reaches_a_stale_required_patch(tmp_path):
    work, _, repo = _authoring(tmp_path, required=["req.patch"])
    (repo / "thirdparty" / "llvm-extensions" / "req.patch").write_text(
        "--- a/a.c\n+++ b/a.c\n@@ -1 +1 @@\n-gone\n+replaced\n"
    )

    result = _author(repo, work, "--rebase", "req.patch")

    assert result.returncode == 0, result.stderr
    assert (work / _STATE).read_text().strip() == "req.patch"


def test_authoring_uses_the_ref_the_build_uses(tmp_path):
    work, _, repo = _authoring(tmp_path)
    (work / "a.c").write_text("x\nfork\n")
    _git(work, "commit", "-qam", "fork")
    fork = _git(work, "rev-parse", "HEAD").stdout.strip()

    missing = _author(repo, work, "feat", FLYDSL_LLVM_REF="0" * 40)
    assert missing.returncode == 1
    assert "is not in" in missing.stderr

    result = _author(repo, work, "feat", FLYDSL_LLVM_REF=fork)
    assert result.returncode == 0, result.stderr
    assert _git(work, "rev-parse", "HEAD~1").stdout.strip() == fork
