#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Print a ROCm root that aiter can pass to -L... -lamdhip64.

The PyPI/nightly wheel-test image (ROCm 7.14 PyTorch) is TheRock pip-ROCm:
there is no /opt/rocm, ``hipcc`` lives in the venv, and ``_rocm_sdk_core``
only ships the versioned soname ``libamdhip64.so.N``. The unversioned
``libamdhip64.so`` that ``ld`` needs lives in ``rocm-sdk-devel`` after
``rocm-sdk init``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

AMD_ROCM_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
LINK_ROOT = Path("/tmp/flydsl_rocm_home")


def _module_root(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(spec.submodule_search_locations[0])


def _soname_version(path: Path) -> tuple[int, ...]:
    """Numeric soname suffix, e.g. libamdhip64.so.7.2 -> (7, 2)."""
    suffix = path.name[len("libamdhip64.so") :].lstrip(".")
    parts: list[int] = []
    for piece in suffix.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _hip64_libs(root: Path) -> list[Path]:
    hits: list[Path] = []
    for sub in ("lib", "lib64"):
        hits.extend(Path(root, sub).glob("libamdhip64.so"))
        hits.extend(Path(root, sub).glob("libamdhip64.so.*"))
    # Highest ABI major first; sonames must be ordered numerically, not as strings
    # (".so.10" sorts before ".so.7" lexicographically).
    return sorted(hits, key=_soname_version, reverse=True)


def _unversioned(root: Path) -> Path | None:
    for sub in ("lib", "lib64"):
        cand = root / sub / "libamdhip64.so"
        if cand.exists():
            return cand
    return None


def _rocm_sdk_root() -> Path | None:
    exe = shutil.which("rocm-sdk")
    if not exe:
        return None
    proc = subprocess.run([exe, "path", "--root"], check=False, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    return Path(out) if out else None


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if env:
        roots.append(Path(env))
    for name in ("_rocm_sdk_devel", "_rocm_sdk_core", "_rocm_sdk_libraries"):
        loc = _module_root(name)
        if loc is not None:
            roots.append(loc)
    sdk = _rocm_sdk_root()
    if sdk is not None:
        roots.append(sdk)
    roots.extend(Path(p) for p in ("/opt/rocm", "/opt/rocm-7.14.0", "/opt/rocm-7.14"))
    roots.extend(sorted(Path("/opt").glob("rocm*")))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _pick_root() -> Path | None:
    with_unversioned: list[Path] = []
    with_versioned: list[Path] = []
    for root in _candidate_roots():
        if _unversioned(root) is not None:
            with_unversioned.append(root)
        elif _hip64_libs(root):
            with_versioned.append(root)
    if with_unversioned:
        return with_unversioned[0]
    if with_versioned:
        return with_versioned[0]
    return None


def _ensure_unversioned(root: Path) -> Path:
    if _unversioned(root) is not None:
        return root
    libs = _hip64_libs(root)
    if not libs:
        raise FileNotFoundError(f"no libamdhip64 under {root}")
    versioned = libs[0]
    in_place = versioned.parent / "libamdhip64.so"
    try:
        # A dangling link (left by an earlier `rocm-sdk init` against a since-removed
        # version) is invisible to _unversioned's exists() check but still occupies
        # the name, so symlink_to would raise FileExistsError.
        if in_place.is_symlink() and not in_place.exists():
            in_place.unlink()
        in_place.symlink_to(versioned.name)
        print(f"created {in_place} -> {versioned.name}", file=sys.stderr)
        return root
    except OSError as exc:
        print(f"could not symlink {in_place}: {exc}", file=sys.stderr)

    if LINK_ROOT.exists():
        shutil.rmtree(LINK_ROOT)
    LINK_ROOT.mkdir(parents=True)
    # "lib" is deliberately excluded: aliasing it onto root/lib would put the
    # recovery symlink back in the directory whose unwritability got us here.
    for sub in ("lib64", "include", "bin", "hip", "llvm", "share"):
        src = root / sub
        if src.exists():
            (LINK_ROOT / sub).symlink_to(src)
    libdir = LINK_ROOT / "lib"
    libdir.mkdir()
    src_lib = root / "lib"
    if src_lib.is_dir():
        for entry in src_lib.iterdir():
            (libdir / entry.name).symlink_to(entry)
    link = libdir / "libamdhip64.so"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(versioned.resolve())
    print(f"wrapper ROCM_PATH={LINK_ROOT} libamdhip64.so -> {versioned}", file=sys.stderr)
    return LINK_ROOT


def _pkg_version(*names: str) -> str | None:
    import importlib.metadata as metadata

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _pip_install(spec: str, extra: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", spec, *extra]
    print(" ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode == 0


def install_devel() -> None:
    if _module_root("_rocm_sdk_devel") is not None:
        print("_rocm_sdk_devel already present", file=sys.stderr)
        return
    ver = _pkg_version("rocm-sdk-core", "rocm")
    specs = [f"rocm[devel]=={ver}", f"rocm-sdk-devel=={ver}"] if ver else ["rocm[devel]", "rocm-sdk-devel"]
    extras = [
        [],
        ["--extra-index-url", AMD_ROCM_INDEX],
    ]
    for spec in specs:
        for extra in extras:
            if _pip_install(spec, extra):
                return
    print("warning: could not pip install rocm-sdk-devel", file=sys.stderr)


def rocm_sdk_init() -> None:
    exe = shutil.which("rocm-sdk")
    if not exe:
        return
    print("running rocm-sdk init", file=sys.stderr)
    subprocess.run([exe, "init"], check=False)


def dump_debug() -> None:
    print("failed to locate libamdhip64.so*", file=sys.stderr)
    for cmd in (
        ["bash", "-lc", "command -v hipcc; command -v rocm-sdk; ls -ld /opt/rocm /opt/rocm-* 2>/dev/null || true"],
        ["bash", "-lc", "ls -l /opt/rocm*/lib/libamdhip64.so* 2>/dev/null || true"],
    ):
        subprocess.run(cmd, check=False)
    for name in ("_rocm_sdk_devel", "_rocm_sdk_core", "_rocm_sdk_libraries"):
        loc = _module_root(name)
        print(f"{name}={loc}", file=sys.stderr)
        if loc is not None:
            for p in _hip64_libs(loc):
                print(f"  {p}", file=sys.stderr)
    sdk = _rocm_sdk_root()
    print(f"rocm-sdk path --root={sdk}", file=sys.stderr)
    for pattern in (
        "/opt/venv/**/libamdhip64.so*",
        "/usr/**/libamdhip64.so*",
        "/opt/rocm*/**/libamdhip64.so*",
    ):
        hits = glob.glob(pattern, recursive=True)
        for hit in hits[:20]:
            print(hit, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensure-devel",
        action="store_true",
        help="pip install matching rocm-sdk-devel and run rocm-sdk init",
    )
    args = parser.parse_args()
    if args.ensure_devel:
        install_devel()
        rocm_sdk_init()
    root = _pick_root()
    if root is None:
        dump_debug()
        return 1
    root = _ensure_unversioned(root)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
