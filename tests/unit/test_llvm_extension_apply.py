# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Guardrails for the LLVM extension series applied by scripts/build_llvm.sh.

Each case here is a failure mode the mechanism actually has. The headline is
test_ordered_series_replays_on_a_warm_checkout: a per-patch
``git apply --reverse --check`` probe cannot detect "already applied" for a
series, because once a later extension rewrites the same lines the earlier one
no longer reverse-applies. That made every build after the first abort. The fix
is to replay from a forcibly-reset tree, and these tests pin that behavior.

The apply section is driven directly against a fixture repository: no LLVM, no
network, no CMake.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_LLVM = _REPO_ROOT / "scripts" / "build_llvm.sh"

pytestmark.append(pytest.mark.skipif(shutil.which("git") is None, reason="git is required"))


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _patch_section():
    """The checkout-and-apply region of build_llvm.sh, verbatim.

    Taken from the real script so the tests cannot drift from what CI runs.
    """
    lines = _BUILD_LLVM.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('if [[ "$LLVM_REF" =~')]
    ends = [i for i, line in enumerate(lines) if line.startswith("LLVM_COMMIT_RESOLVED=")]
    # A duplicated or reordered marker would silently slice a different region,
    # leaving these tests green while exercising code the build never runs.
    assert len(starts) == 1, f"expected one start marker, found {len(starts)}"
    assert len(ends) == 1, f"expected one end marker, found {len(ends)}"
    assert starts[0] < ends[0], "markers are out of order in build_llvm.sh"
    section = "\n".join(lines[starts[0] : ends[0]])
    # The slice must contain the behavior under test, or the tests prove nothing.
    for expected in ("git checkout --force", "git apply", "LLVM_EXTENSIONS"):
        assert expected in section, f"sliced section is missing {expected!r}"
    return section


def _required_array():
    """The REQUIRED_PATCHES array declaration from build_llvm.sh."""
    lines = _BUILD_LLVM.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("REQUIRED_PATCHES=("))
    end = next(i for i, line in enumerate(lines[start:], start) if line == ")")
    return "\n".join(lines[start : end + 1])


def _extensions_array():
    """The LLVM_EXTENSIONS array declaration from build_llvm.sh."""
    lines = _BUILD_LLVM.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("LLVM_EXTENSIONS=("))
    end = next(i for i, line in enumerate(lines[start:], start) if line == ")")
    return "\n".join(lines[start : end + 1])


def _make_fixture(tmp_path, file_lines):
    """A shallow-cloned checkout at a pin, mirroring how build_llvm.sh works."""
    upstream = tmp_path / "upstream"
    (upstream / "src").mkdir(parents=True)
    (upstream / "src" / "target.c").write_text("\n".join(file_lines) + "\n")
    _git(upstream, "init", "-q")
    _git(upstream, "config", "user.email", "t@t")
    _git(upstream, "config", "user.name", "t")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-qm", "base")
    pin = _git(upstream, "rev-parse", "HEAD").stdout.strip()

    work = tmp_path / "llvm-project"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "remote", "add", "origin", str(upstream))
    _git(work, "fetch", "-q", "--depth", "1", "origin", pin)
    _git(work, "checkout", "-q", pin)
    return work, pin


def _run_apply(tmp_path, work, pin, extensions, ext_dir, no_ext="0", profile="extended", required=()):
    """Run the real apply section with the given patch arrays and profile.

    `profile` defaults to "extended" so the existing cases keep exercising the
    extension path; the real script defaults to "baseline".
    """
    array = "LLVM_EXTENSIONS=(\n" + "\n".join(f"    {e}" for e in extensions) + "\n)"
    req = "REQUIRED_PATCHES=(\n" + "\n".join(f"    {r}" for r in required) + "\n)"
    script = "\n".join(
        [
            "set -e",
            f'LLVM_REF="{pin}"',
            f'LLVM_EXT_DIR="{ext_dir}"',
            f'LLVM_NO_EXT="{no_ext}"',
            f'FLYDSL_LLVM_PROFILE="{profile}"',
            "LLVM_FETCH_ARGS=(--depth 1)",
            req,
            array,
            f'cd "{work}"',
            _patch_section(),
            'echo "APPLY-OK"',
        ]
    )
    runner = tmp_path / "run.sh"
    runner.write_text(script)
    return subprocess.run(["bash", str(runner)], capture_output=True, text=True, cwd=str(work))


def _write_extension(path, work, before, after):
    """Author an extension as a diff of the current tree, then revert the tree."""
    target = work / "src" / "target.c"
    target.write_text(target.read_text().replace(before, after))
    diff = _git(work, "diff").stdout
    path.write_text(diff)
    _git(work, "checkout", "--force", "--", ".")


def test_single_extension_applies_and_is_idempotent(tmp_path):
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")

    for _ in range(3):
        result = _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir)
        assert result.returncode == 0, result.stderr
        assert "TWO" in (work / "src" / "target.c").read_text()

    # Applied once, not accumulated across runs.
    assert (work / "src" / "target.c").read_text().count("TWO") == 1


def test_ordered_series_replays_on_a_warm_checkout(tmp_path):
    """The regression: a stacked series must survive repeated runs.

    Patch B rewrites the line patch A produced, so A alone no longer
    reverse-applies once both are on. Detecting "already applied" per patch
    therefore fails; replaying from a reset tree is what makes this work.
    """
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()

    _write_extension(ext_dir / "a.patch", work, "two", "two-A")
    # B is authored on top of A, via a throwaway commit, so its diff is
    # incremental rather than cumulative -- see scripts/llvm_extension.sh.
    _git(work, "apply", str(ext_dir / "a.patch"))
    _git(work, "add", "-A")
    _git(work, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "wip")
    _write_extension(ext_dir / "b.patch", work, "two-A", "two-A-then-B")
    _git(work, "reset", "-q", "--hard", pin)

    for _ in range(3):
        result = _run_apply(tmp_path, work, pin, ["a.patch", "b.patch"], ext_dir)
        assert result.returncode == 0, result.stderr
        assert "two-A-then-B" in (work / "src" / "target.c").read_text()


def test_reset_preserves_build_outputs(tmp_path):
    """build-flydsl/ and mlir_install/ live inside the checkout.

    A `git clean -fd` style reset would delete them and turn every incremental
    LLVM build into a full one. Only tracked files may be reverted.
    """
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")

    (work / "build-flydsl").mkdir()
    (work / "build-flydsl" / "obj.o").write_text("expensive")
    (work / "mlir_install").mkdir()
    (work / "mlir_install" / "lib.so").write_text("installed")

    assert _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir).returncode == 0

    assert (work / "build-flydsl" / "obj.o").read_text() == "expensive"
    assert (work / "mlir_install" / "lib.so").read_text() == "installed"


def test_authoring_commit_and_dirty_tree_do_not_wedge_the_build(tmp_path):
    """llvm_extension.sh may be interrupted, leaving a commit and a dirty tree.

    A plain `git checkout <pin>` refuses to move HEAD when a modified file
    differs between the two trees, and under `set -e` the build then aborts on
    every subsequent run with no way to recover. The checkout must be forced.
    """
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")

    # An interrupted authoring session: baseline commit plus in-progress edits
    # to the very file the patch touches.
    _git(work, "apply", str(ext_dir / "a.patch"))
    _git(work, "add", "-A")
    _git(
        work,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-qm",
        "flydsl: patch baseline (throwaway)",
    )
    target = work / "src" / "target.c"
    target.write_text(target.read_text() + "// author WIP\n")

    for _ in range(2):
        result = _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir)
        assert result.returncode == 0, result.stderr
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == pin


def test_extension_file_absent_from_the_array_is_rejected(tmp_path):
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")
    (ext_dir / "forgotten.patch").write_text("unused")

    result = _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir)

    assert result.returncode == 1
    assert "listed in neither array: forgotten.patch" in result.stderr


def test_array_entry_without_a_file_fails_before_touching_the_tree(tmp_path):
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")

    result = _run_apply(tmp_path, work, pin, ["a.patch", "ghost.patch"], ext_dir)

    assert result.returncode == 1
    assert "does not exist: ghost.patch" in result.stderr
    # Validation runs first, so no patch was applied.
    assert "TWO" not in (work / "src" / "target.c").read_text()


def test_stale_extension_names_it_and_the_recovery(tmp_path):
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "stale.patch", work, "two", "TWO")
    # Upstream moved: the context the patch expects is gone.
    _git(work, "checkout", "--force", "--", ".")
    target = work / "src" / "target.c"
    target.write_text("completely\ndifferent\ncontent\n")
    _git(work, "add", "-A")
    _git(work, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "moved")
    moved = _git(work, "rev-parse", "HEAD").stdout.strip()

    result = _run_apply(tmp_path, work, moved, ["stale.patch"], ext_dir)

    assert result.returncode == 1
    # The headline is what distinguishes a failed apply from the earlier
    # validation errors, which also name the file.
    assert "LLVM extension failed to apply: stale.patch" in result.stderr
    assert f"LLVM pin: {moved}" in result.stderr
    assert "llvm_extension.sh --rebase" in result.stderr


def test_empty_array_is_a_no_op(tmp_path):
    """Deleting the last patch must not break the build.

    Safe today because build_llvm.sh uses `set -e` without `-u`; this test
    fails rather than CI if that is ever hardened to `set -euo pipefail`.
    """
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()

    result = _run_apply(tmp_path, work, pin, [], ext_dir)

    assert result.returncode == 0, result.stderr
    assert "LLVM Extensions: 0" in result.stdout


def test_no_ext_builds_stock_upstream(tmp_path):
    """The control arm: same pin, extensions the only variable.

    Without this a regression cannot be attributed -- rebuilding at a different
    pin to get a clean tree confounds the extension with upstream's own changes.
    """
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")

    result = _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir, no_ext="1")

    assert result.returncode == 0, result.stderr
    assert "none (FLYDSL_LLVM_NO_EXT=1)" in result.stdout
    assert (work / "src" / "target.c").read_text() == "one\ntwo\nthree\n"


def test_no_ext_env_var_reaches_the_apply_logic():
    """The env name users type must be the one the script reads.

    The other tests drive the apply section directly, supplying LLVM_NO_EXT
    themselves -- so the FLYDSL_LLVM_NO_EXT -> LLVM_NO_EXT assignment lives
    outside everything they cover, and a typo there would silently give a build
    WITH the extensions to someone who asked for one without.
    """
    text = _BUILD_LLVM.read_text()

    assert 'LLVM_NO_EXT="${FLYDSL_LLVM_NO_EXT:-0}"' in text
    # The apply logic must branch on the internal name, not re-read the env.
    assert '"${LLVM_NO_EXT}" == "1"' in text


def test_no_ext_does_not_suppress_the_unlisted_file_check(tmp_path):
    """Building without extensions is a choice; an unlisted file is a tree mistake."""
    work, pin = _make_fixture(tmp_path, ["one", "two", "three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "a.patch", work, "two", "TWO")
    (ext_dir / "unlisted.patch").write_text("junk")

    result = _run_apply(tmp_path, work, pin, ["a.patch"], ext_dir, no_ext="1")

    assert result.returncode == 1
    assert "listed in neither array: unlisted.patch" in result.stderr


def test_authoring_resets_to_the_pin_before_replaying(tmp_path):
    """prepare_tree must move HEAD, not just restore the working tree.

    `git checkout -- .` restores paths from the index and leaves HEAD alone, so
    a checkout left on a previous baseline commit would become the parent of the
    next one and every patch authored here would diff against the wrong base --
    while the script printed "Tree is at the pin".
    """
    script = _REPO_ROOT / "scripts" / "llvm_extension.sh"
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "thirdparty" / "llvm-extensions").mkdir(parents=True)
    shutil.copy(script, repo / "scripts" / "llvm_extension.sh")
    (repo / "scripts" / "build_llvm.sh").write_text(
        "#!/usr/bin/env bash\nREQUIRED_PATCHES=(\n)\nLLVM_EXTENSIONS=(\n)\n"
    )

    work = tmp_path / "llvm-project"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    (work / "a.c").write_text("x\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "pin")
    pin = _git(work, "rev-parse", "HEAD").stdout.strip()
    # HEAD drifts off the pin, as a leftover authoring session would leave it.
    (work / "a.c").write_text("x\ny\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "leftover")
    (repo / "thirdparty" / "llvm-build-info.json").write_text(
        '{"baseline":{"llvm_hash":"%s"},"extended":{"llvm_hash":"%s"}}\n' % (pin, pin)
    )

    result = subprocess.run(
        ["bash", "scripts/llvm_extension.sh", "feat"],
        cwd=repo,
        env={**os.environ, "LLVM_SRC_DIR": str(work)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # The baseline commit's parent is the pin, and the tree holds the pin's content.
    assert _git(work, "rev-parse", "HEAD~1").stdout.strip() == pin
    assert (work / "a.c").read_text() == "x\n"


def test_baseline_profile_applies_only_required_patches(tmp_path):
    """The default build carries the required patches but no extensions."""
    work, pin = _make_fixture(tmp_path, ["one"] + [f"pad{i}" for i in range(12)] + ["three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "req.patch", work, "one", "ONE")
    _write_extension(ext_dir / "ext.patch", work, "three", "THREE")

    result = _run_apply(tmp_path, work, pin, ["ext.patch"], ext_dir, profile="baseline", required=["req.patch"])

    assert result.returncode == 0, result.stderr
    text = (work / "src" / "target.c").read_text()
    assert "ONE" in text
    assert "THREE" not in text
    assert "none (profile=baseline)" in result.stdout


def test_extended_profile_applies_required_patches_too(tmp_path):
    """Required patches are not an extension: both profiles carry them."""
    work, pin = _make_fixture(tmp_path, ["one"] + [f"pad{i}" for i in range(12)] + ["three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "req.patch", work, "one", "ONE")
    _write_extension(ext_dir / "ext.patch", work, "three", "THREE")

    result = _run_apply(tmp_path, work, pin, ["ext.patch"], ext_dir, profile="extended", required=["req.patch"])

    assert result.returncode == 0, result.stderr
    text = (work / "src" / "target.c").read_text()
    assert "ONE" in text
    assert "THREE" in text


def test_no_ext_keeps_required_patches(tmp_path):
    """Turning the extensions off must not drop what the build needs to work."""
    work, pin = _make_fixture(tmp_path, ["one"] + [f"pad{i}" for i in range(12)] + ["three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "req.patch", work, "one", "ONE")
    _write_extension(ext_dir / "ext.patch", work, "three", "THREE")

    result = _run_apply(
        tmp_path,
        work,
        pin,
        ["ext.patch"],
        ext_dir,
        no_ext="1",
        profile="extended",
        required=["req.patch"],
    )

    assert result.returncode == 0, result.stderr
    text = (work / "src" / "target.c").read_text()
    assert "ONE" in text
    assert "THREE" not in text


def test_a_stale_extension_does_not_block_the_baseline_profile(tmp_path):
    """The reason the pins are separate.

    When upstream moves under an extension, `baseline` must still build so the
    pin can be bumped and tested while the extension is rebased.
    """
    work, pin = _make_fixture(tmp_path, ["one"] + [f"pad{i}" for i in range(12)] + ["three"])
    ext_dir = tmp_path / "patches"
    ext_dir.mkdir()
    _write_extension(ext_dir / "req.patch", work, "one", "ONE")
    # An extension whose context no longer exists at this pin.
    (ext_dir / "stale.patch").write_text(
        "--- a/src/target.c\n+++ b/src/target.c\n@@ -1,3 +1,3 @@\n" " gone\n-vanished\n+replaced\n context\n"
    )

    baseline = _run_apply(tmp_path, work, pin, ["stale.patch"], ext_dir, profile="baseline", required=["req.patch"])
    extended = _run_apply(tmp_path, work, pin, ["stale.patch"], ext_dir, profile="extended", required=["req.patch"])

    assert baseline.returncode == 0, baseline.stderr
    assert extended.returncode == 1
    assert "stale.patch" in extended.stderr


def test_profile_must_be_one_of_the_two(tmp_path):
    text = _BUILD_LLVM.read_text()

    assert 'FLYDSL_LLVM_PROFILE="${FLYDSL_LLVM_PROFILE:-baseline}"' in text
    assert "baseline|extended)" in text


def _array_entries(text):
    return {line.strip() for line in text.splitlines()[1:-1] if line.strip() and not line.strip().startswith("#")}


def test_repo_patches_are_listed_in_one_of_the_arrays():
    """Every file on disk must appear in exactly one array, and vice versa."""
    ext_dir = _REPO_ROOT / "thirdparty" / "llvm-extensions"
    on_disk = {p.name for p in ext_dir.glob("*.patch")}
    required = _array_entries(_required_array())
    extensions = _array_entries(_extensions_array())

    assert on_disk == required | extensions
    # A patch in both arrays would be applied twice, and the second apply fails.
    assert not (required & extensions)
