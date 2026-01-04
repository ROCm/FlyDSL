from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup

try:
    # Optional: when building a wheel, mark it as non-pure so it gets a platform tag.
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # type: ignore

    class bdist_wheel(_bdist_wheel):  # type: ignore
        def finalize_options(self):
            super().finalize_options()
            # `_mlir` ships CPython extension modules and shared libraries.
            self.root_is_pure = False

except Exception:  # pragma: no cover
    bdist_wheel = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent
# IMPORTANT: setuptools editable builds require *relative* paths in setup()
# arguments (not absolute paths).
PY_SRC_REL = Path("pyflir") / "src"
DEFAULT_OUT_DIR_REL = Path(".flir")
DEFAULT_BUILD_DIR_REL = DEFAULT_OUT_DIR_REL / "build"

# Keep build artifacts under a single output directory.
# - build.sh defaults to: .flir/build
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

EMBEDDED_MLIR_ROOT_REL = BUILD_DIR_REL / "python_packages" / "pyflir"
EMBEDDED__MLIR_REL = EMBEDDED_MLIR_ROOT_REL / "_mlir"

PY_SRC = REPO_ROOT / PY_SRC_REL
EMBEDDED_MLIR_ROOT = REPO_ROOT / EMBEDDED_MLIR_ROOT_REL
EMBEDDED__MLIR = REPO_ROOT / EMBEDDED__MLIR_REL


def _read_version() -> str:
    init_py = (PY_SRC / "pyflir" / "__init__.py").read_text(encoding="utf-8")
    for line in init_py.splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.1.0"
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def _load_requirements() -> list[str]:
    # Keep Python requirements alongside the pyflir source root.
    req = REPO_ROOT / "pyflir" / "requirements.txt"
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
    # For runtime, FLIR expects the embedded MLIR runtime under `_mlir/`.
    # This is built by the repo build (CMake) and staged under build/python_packages/pyflir.
    # Default to ALWAYS rebuilding unless the user opts out.
    rebuild_mode = (os.environ.get("FLIR_REBUILD") or os.environ.get("FLIR_REBUILD") or "1").strip().lower()
    # Semantics:
    # - 1 (default):    always run ./build.sh before installing
    # - auto:           run ./build.sh iff embedded runtime is missing
    # - 1/true/yes:     always run ./build.sh before installing
    # - 0/false/no:     never run ./build.sh (error if missing)
    force_rebuild = rebuild_mode in {"1", "true", "yes"}
    never_rebuild = rebuild_mode in {"0", "false", "no"}

    need_build = force_rebuild or (not EMBEDDED__MLIR.exists() and not never_rebuild)
    if need_build:
        try:
            env = dict(os.environ)
            # Ensure build.sh writes artifacts where setup.py expects them.
            env.setdefault("FLIR_BUILD_DIR", str(BUILD_DIR_REL))
            subprocess.run(["bash", "./build.sh"], cwd=str(REPO_ROOT), check=True, env=env)
        except Exception as e:
            raise RuntimeError(
                "Failed to build embedded MLIR runtime via `./build.sh`.\n"
                f"Original error: {e}\n"
            ) from e

    if not EMBEDDED__MLIR.exists():
        raise RuntimeError(
            "Embedded MLIR python runtime not found at "
            f"{EMBEDDED__MLIR}.\n\n"
            "Build it first (e.g. `./build.sh`), or run the CMake build that "
            "produces `build/python_packages/pyflir/_mlir`.\n\n"
            "Controls:\n"
            "  - FLIR_REBUILD=auto (default): build iff missing\n"
            "  - FLIR_REBUILD=1:              always rebuild\n"
            "  - FLIR_REBUILD=0:              never rebuild (error if missing)\n"
        )


_assert_embedded_mlir_exists()

IS_WHEEL_BUILD = any(a in {"bdist_wheel", "sdist"} for a in os.sys.argv[1:])

def _ensure_python_embedded_mlir_package() -> None:
    """Make `_mlir` importable for editable installs.

    pip's PEP660 editable install mode does not reliably honor multi-root
    `package_dir` mappings. To keep `pip install -e .` and `setup.py develop`
    working, we create a `_mlir` package entry under `pyflir/src/_mlir` by
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
            "`build/python_packages/pyflir`.\n"
            f"Original error: {e}"
        ) from e


if not IS_WHEEL_BUILD:
    _ensure_python_embedded_mlir_package()
    # Editable/dev installs: single-root under `pyflir/src/` (includes `_mlir` via symlink).
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
    name="pyflir",
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
    package_data={
        # Note: we also stage MLIR ExecutionEngine runtime libs under
        # `_mlir/_mlir_libs/lib/`. We intentionally only package the FLIR-owned
        # thin runtime (`libflir_jit_runtime.so*`) to avoid shipping upstream
        # `libmlir_*` runtime shared libraries (which can export large LLVM/MLIR
        # symbol surfaces and increase collision risk when multiple LLVM/MLIR
        # copies are present in one process).
        "": [
            "*.so",
            "*.so.*",
            "*.dylib",
            "*.dll",
            "*.pyi",
            "lib/libflir_jit_runtime.so",
            "lib/libflir_jit_runtime.so.*",
            "lib/libflir_jit_runtime.dylib",
            "lib/libflir_jit_runtime.dll",
        ],
    },
    install_requires=_load_requirements(),
    cmdclass=({"bdist_wheel": bdist_wheel} if bdist_wheel is not None else {}),
)


