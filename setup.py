from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

from setuptools import Distribution, find_namespace_packages, find_packages, setup


class BinaryDistribution(Distribution):
    """Component to force a binary distribution."""

    def has_ext_modules(self):
        return True


try:
    # Optional: when building a wheel, mark it as non-pure so it gets a platform tag.
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # type: ignore

    class bdist_wheel(_bdist_wheel):  # type: ignore
        def finalize_options(self):
            super().finalize_options()
            # `_mlir` ships CPython extension modules and shared libraries.
            self.root_is_pure = False

        def run(self):
            # Default behavior:
            # - `bdist_wheel`: behave like a "release" build (strip binaries) unless overridden.
            # - editable/dev installs: keep fast path (no wheel packaging steps).
            os.environ.setdefault("FLIR_BUILD_WHEEL", "1")

            # When invoked from `flir/build.sh`, that script already handles stripping/auditwheel.
            in_build_sh = os.environ.get("FLIR_IN_BUILD_SH") == "1"
            do_release_wheel = os.environ.get("FLIR_BUILD_WHEEL", "0") == "1"

            if do_release_wheel and not in_build_sh:
                _strip_embedded_shared_libs()

            # Track wheel outputs so we can locate the wheel we just built.
            dist_dir = Path(getattr(self, "dist_dir", "dist"))
            before = {p.resolve() for p in dist_dir.glob("*.whl")}

            super().run()

            if do_release_wheel and not in_build_sh:
                # Try to auto-repair to a manylinux tag when tooling is available.
                after = {p.resolve() for p in dist_dir.glob("*.whl")}
                built = sorted(after - before, key=lambda p: p.stat().st_mtime)
                wheel_path = built[-1] if built else _latest_wheel_in(dist_dir)
                if wheel_path is not None:
                    _auditwheel_repair_in_place(wheel_path, dist_dir)

except Exception:  # pragma: no cover
    bdist_wheel = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent
# IMPORTANT: setuptools editable builds require *relative* paths in setup()
# arguments (not absolute paths).
PY_SRC_REL = Path("flydsl") / "src"
DEFAULT_OUT_DIR_REL = Path(".flir")
DEFAULT_BUILD_DIR_REL = DEFAULT_OUT_DIR_REL / "build"

# Keep build artifacts under a single output directory.
# - flir/build.sh defaults to: .flir/build
# - You can override with:
#   FLIR_OUT_DIR=.flir          (relative to repo root)
#   FLIR_BUILD_DIR=.flir/build  (relative to repo root)
_out_dir_env = os.environ.get("FLIR_OUT_DIR") or os.environ.get("FLIR_OUT_DIR")
_build_dir_env = os.environ.get("FLIR_BUILD_DIR") or os.environ.get("FLIR_BUILD_DIR")
if _build_dir_env:
    BUILD_DIR_REL = Path(_build_dir_env)
elif _out_dir_env:
    BUILD_DIR_REL = Path(_out_dir_env) / "build"
else:
    BUILD_DIR_REL = DEFAULT_BUILD_DIR_REL

if BUILD_DIR_REL.is_absolute():
    raise RuntimeError(
        "FLIR_BUILD_DIR must be a repo-relative path for packaging "
        f"(got absolute: {BUILD_DIR_REL})."
    )

EMBEDDED_MLIR_ROOT_REL = BUILD_DIR_REL / "python_packages" / "flydsl"
EMBEDDED__MLIR_REL = EMBEDDED_MLIR_ROOT_REL / "_mlir"

PY_SRC = REPO_ROOT / PY_SRC_REL
EMBEDDED_MLIR_ROOT = REPO_ROOT / EMBEDDED_MLIR_ROOT_REL
EMBEDDED__MLIR = REPO_ROOT / EMBEDDED__MLIR_REL


def _read_version() -> str:
    init_py = (PY_SRC / "flydsl" / "__init__.py").read_text(encoding="utf-8")
    for line in init_py.splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.1.0"
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def _load_requirements() -> list[str]:
    # Keep Python requirements alongside the flydsl source root.
    req = REPO_ROOT / "flydsl" / "requirements.txt"
    if not req.exists():
        return []
    out: list[str] = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        out.append(line)
    return out


def _assert_embedded_mlir_exists() -> None:
    # Skip build if we're already running inside build.sh to avoid recursion.
    if os.environ.get("FLIR_IN_BUILD_SH") == "1":
        return

    # For runtime, FLIR expects the embedded MLIR runtime under `_mlir/`.
    # This is built by the repo build (CMake) and staged under build/python_packages/flydsl.
    # Default to build if missing, but don't force rebuild by default to allow
    # packaging from a pre-built state (e.g. when building a wheel from an sdist).
    rebuild_mode = (os.environ.get("FLIR_REBUILD") or "auto").strip().lower()
    # Semantics:
    # - auto (default): build iff embedded runtime is missing
    # - 1/true/yes:     always run ./flir/build.sh before installing
    # - 0/false/no:     never run ./flir/build.sh (error if missing)
    force_rebuild = rebuild_mode in {"1", "true", "yes"}
    never_rebuild = rebuild_mode in {"0", "false", "no"}

    need_build = force_rebuild or (not EMBEDDED__MLIR.exists() and not never_rebuild)
    if need_build:
        try:
            env = dict(os.environ)
            # Ensure build.sh writes artifacts where setup.py expects them.
            env.setdefault("FLIR_BUILD_DIR", str(BUILD_DIR_REL))
            # `setup.py` may be invoked with FLIR_BUILD_WHEEL=1 (e.g. bdist_wheel),
            # but when we call build.sh from here we only want to build the embedded
            # runtime, not run the "release wheel" packaging branch.
            env.pop("FLIR_BUILD_WHEEL", None)
            subprocess.run(["bash", "flir/build.sh"], cwd=str(REPO_ROOT), check=True, env=env)
        except Exception as e:
            raise RuntimeError(
                "Failed to build embedded MLIR runtime via `./flir/build.sh`.\n"
                f"Original error: {e}\n"
            ) from e

    if not EMBEDDED__MLIR.exists():
        raise RuntimeError(
            "Embedded MLIR python runtime not found at "
            f"{EMBEDDED__MLIR}.\n\n"
            "Build it first (e.g. `./flir/build.sh`), or run the CMake build that "
            "produces `build/python_packages/flydsl/_mlir`.\n\n"
            "Controls:\n"
            "  - FLIR_REBUILD=auto (default): build iff missing\n"
            "  - FLIR_REBUILD=1:              always rebuild\n"
            "  - FLIR_REBUILD=0:              never rebuild (error if missing)\n"
        )


_assert_embedded_mlir_exists()

IS_WHEEL_BUILD = any(a in {"bdist_wheel", "sdist"} for a in os.sys.argv[1:])

def _strip_embedded_shared_libs() -> None:
    """Reduce wheel size by stripping debug symbols from embedded shared libraries.

    Mirrors the behavior in `flir/build.sh` for wheel builds, but is safe to run
    directly from `python setup.py bdist_wheel`.
    """
    strip_bin = shutil.which("strip")
    if not strip_bin:
        print("Warning: strip not found; skipping binary stripping.")
        return

    # Drop the non-versioned CAPI .so which is typically a symlink/copy of the versioned lib.
    capi = EMBEDDED__MLIR / "_mlir_libs" / "libFlirPythonCAPI.so"
    try:
        capi.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: failed to remove {capi}: {e}")

    # Strip shared libraries (extensions + runtime deps).
    so_files: list[Path] = [
        p
        for p in EMBEDDED__MLIR.rglob("*.so*")
        if p.is_file() and not p.is_symlink()
    ]
    if not so_files:
        return

    # Run `strip` in chunks to avoid command-line length limits.
    def _chunks(items: list[str], n: int = 200):
        for i in range(0, len(items), n):
            yield items[i : i + n]

    print(f"Stripping {len(so_files)} shared libraries under {EMBEDDED__MLIR} ...")
    for chunk in _chunks([str(p) for p in so_files]):
        subprocess.run([strip_bin, "--strip-unneeded", *chunk], check=False)


def _latest_wheel_in(dist_dir: Path) -> Path | None:
    wheels = [p for p in dist_dir.glob("*.whl") if p.is_file()]
    if not wheels:
        return None
    return max(wheels, key=lambda p: p.stat().st_mtime)


def _auditwheel_repair_in_place(wheel_path: Path, dist_dir: Path) -> None:
    """Run `auditwheel repair` to produce a manylinux-tagged wheel when possible.

    Note:
      - This does NOT guarantee a `manylinux_2_28` tag; the best achievable policy
        depends on your binary's ELF deps and glibc symbol versions.
      - For a guaranteed manylinux_2_28 wheel, build inside a manylinux_2_28 image
        (see tools/build_manylinux_2_28.sh).
    """
    if sys.platform != "linux":
        return

    auditwheel = shutil.which("auditwheel")
    if not auditwheel:
        print("Warning: auditwheel not found; skipping manylinux repair. (Install with `pip install auditwheel`.)")
        return

    # auditwheel uses `patchelf` to rewrite RPATH; if missing, the repair will fail.
    if not shutil.which("patchelf"):
        print("Warning: patchelf not found; skipping manylinux repair. (Install with `apt-get install patchelf`.)")
        return

    wheelhouse = dist_dir / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)

    cmd = [
        auditwheel,
        "repair",
        str(wheel_path),
        "-w",
        str(wheelhouse),
        # IMPORTANT: do not bundle ROCm user-space runtime libraries into the wheel.
        "--exclude",
        "libamdhip64.so.*",
        "--exclude",
        "libhsa-runtime64.so.*",
        "--exclude",
        "libdrm_amdgpu.so.*",
    ]

    print(f"Running auditwheel repair on {wheel_path.name} ...")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Warning: auditwheel repair failed; leaving original wheel. ({e})")
        shutil.rmtree(wheelhouse, ignore_errors=True)
        return

    repaired = sorted([p for p in wheelhouse.glob("*.whl") if p.is_file()], key=lambda p: p.stat().st_mtime)
    if not repaired:
        print("Warning: auditwheel repair produced no wheels; leaving original wheel.")
        shutil.rmtree(wheelhouse, ignore_errors=True)
        return

    # Replace original wheel with repaired wheel(s).
    try:
        wheel_path.unlink()
    except Exception:
        pass

    for p in repaired:
        target = dist_dir / p.name
        if target.exists():
            target.unlink()
        p.replace(target)

    shutil.rmtree(wheelhouse, ignore_errors=True)

def _ensure_python_embedded_mlir_package() -> None:
    """Make `_mlir` importable for editable installs.

    pip's PEP660 editable install mode does not reliably honor multi-root
    `package_dir` mappings. To keep `pip install -e .` and `setup.py develop`
    working, we create a `_mlir` package entry under `flydsl/src/_mlir` by
    symlinking to the embedded runtime produced by the CMake build.
    """
    dst = PY_SRC / "_mlir"
    # `Path.exists()` follows symlinks; for a broken symlink it returns False.
    # We want to repair broken/outdated symlinks automatically.
    target = Path("..") / EMBEDDED__MLIR_REL

    if dst.is_symlink():
        try:
            current_target = Path(os.readlink(dst))
        except Exception:
            current_target = None
        # If it's already correct and resolves, keep it.
        if current_target == target and dst.exists():
            return
        # Otherwise replace it (covers broken links and moved build dirs).
        dst.unlink()

    if dst.exists():
        # A real directory/file exists here; don't overwrite silently.
        return

    if os.path.lexists(dst):
        # Path exists but is neither a working directory nor a symlink we can manage.
        raise RuntimeError(f"{dst} exists but is not a usable symlink/directory; please remove it and retry.")
    # Prefer a relative symlink so the repo remains relocatable.
    try:
        dst.symlink_to(target, target_is_directory=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create symlink {dst} -> {target}.\n"
            "Either create it manually, or install with PYTHONPATH pointing at "
            "`build/python_packages/flydsl`.\n"
            f"Original error: {e}"
        ) from e


if not IS_WHEEL_BUILD:
    _ensure_python_embedded_mlir_package()
    # Editable/dev installs: single-root under `flydsl/src/` (includes `_mlir` via symlink).
    all_packages = sorted(
        set(find_packages(where=str(PY_SRC_REL)))
        | set(find_namespace_packages(where=str(PY_SRC_REL), include=["_mlir*"]))
    )
    package_dir = {
        "": str(PY_SRC_REL),
    }
else:
    # Wheel/sdist builds: take `_mlir` from the embedded build output directly,
    # so the wheel can include the CPython extension modules.
    py_packages = find_packages(where=str(PY_SRC_REL))
    embedded_packages = find_namespace_packages(where=str(EMBEDDED_MLIR_ROOT_REL), include=["_mlir*"])
    all_packages = sorted(set(py_packages + embedded_packages))
    package_dir = {
        "": str(PY_SRC_REL),
        "_mlir": str(EMBEDDED__MLIR_REL),
    }

setup(
    name="flydsl",
    version=_read_version(),
    description="FLIR - ROCm Domain Specific Language for layout algebra (Python + embedded MLIR runtime)",
    long_description=(REPO_ROOT / "README.md").read_text(encoding="utf-8") if (REPO_ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=all_packages,
    package_dir=package_dir,
    include_package_data=True,
    # Ensure embedded shared libs are packaged in wheels.
    # IMPORTANT: we must include versioned shared libraries (e.g. *.so.22.0git),
    # otherwise the wheel will miss required runtime deps and be unusable.
    package_data={
        "_mlir": [
            # Python extension modules
            "_mlir_libs/_*.so",
            # CAPI library
            "_mlir_libs/libFlirPythonCAPI.so.*",
            # JIT runtime: include all versioned shared libraries.
            "_mlir_libs/lib/*.so",
            "_mlir_libs/lib/*.so.*",
            "*.pyi",
        ],
    },
    install_requires=_load_requirements(),
    cmdclass=({"bdist_wheel": bdist_wheel} if bdist_wheel is not None else {}),
    distclass=BinaryDistribution,
)


