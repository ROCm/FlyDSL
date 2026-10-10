# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_rocm_backend(monkeypatch):
    """Load the real backend module without requiring built MLIR bindings."""
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "python" / "flydsl"
    prefix = "_flydsl_rocm_resolver_test"

    packages = {
        prefix: package_root,
        f"{prefix}.compiler": package_root / "compiler",
        f"{prefix}.compiler.backends": package_root / "compiler" / "backends",
        f"{prefix}.runtime": package_root / "runtime",
        f"{prefix}.utils": package_root / "utils",
    }
    for name, path in packages.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    runtime_device = types.ModuleType(f"{prefix}.runtime.device")
    runtime_device.get_rocm_arch = lambda: "gfx000"
    runtime_device.get_warp_size = lambda _arch: 32
    monkeypatch.setitem(sys.modules, runtime_device.__name__, runtime_device)

    env = types.ModuleType(f"{prefix}.utils.env")
    monkeypatch.setitem(sys.modules, env.__name__, env)

    base_name = f"{prefix}.compiler.backends.base"
    base_spec = importlib.util.spec_from_file_location(
        base_name, package_root / "compiler" / "backends" / "base.py"
    )
    base = importlib.util.module_from_spec(base_spec)
    monkeypatch.setitem(sys.modules, base_name, base)
    base_spec.loader.exec_module(base)

    module_name = f"{prefix}.compiler.backends.rocm"
    spec = importlib.util.spec_from_file_location(
        module_name, package_root / "compiler" / "backends" / "rocm.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rocm(monkeypatch):
    return _load_rocm_backend(monkeypatch)


def _make_toolkit(root: Path) -> Path:
    (root / "llvm" / "bin").mkdir(parents=True)
    (root / "llvm" / "bin" / "ld.lld.exe").touch()
    (root / "amdgcn" / "bitcode").mkdir(parents=True)
    return root


def _clear_toolkit_env(monkeypatch):
    for name in ("FLYDSL_ROCM_TOOLKIT", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)


def test_direct_sdk_from_rocm_path(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    sdk = _make_toolkit(tmp_path / "sdk")
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setattr(rocm.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unexpected junction"))

    assert rocm._windows_rocm_toolkit() == sdk.resolve().as_posix()


def test_explicit_toolkit_overrides_discovery_and_requires_layout(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    explicit = _make_toolkit(tmp_path / "explicit")
    monkeypatch.setenv("FLYDSL_ROCM_TOOLKIT", str(explicit))
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "ignored"))

    assert rocm._windows_rocm_toolkit() == explicit.resolve().as_posix()

    monkeypatch.setenv("FLYDSL_ROCM_TOOLKIT", str(tmp_path / "incomplete"))
    with pytest.raises(RuntimeError, match="Invalid FLYDSL_ROCM_TOOLKIT"):
        rocm._windows_rocm_toolkit()


def test_split_sdk_stages_junctions_and_reuses_them(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    sdk = tmp_path / "sdk"
    llvm = sdk / "lib" / "llvm"
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / "ld.lld.exe").touch()
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    junctions = {}
    real_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        target = junctions.get(path)
        if target is not None:
            return real_resolve(target, *args, **kwargs)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(rocm, "_is_junction", lambda path: path in junctions)
    commands = []

    def mklink(command, **kwargs):
        commands.append(command)
        link, target = Path(command[-2]), Path(command[-1])
        junctions[link] = target
        link.mkdir()
        if link.name == "llvm":
            (link / "bin").mkdir()
            (link / "bin" / "ld.lld.exe").touch()
        else:
            (link / "bitcode").mkdir()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(rocm.subprocess, "run", mklink)
    expected_stage = (
        tmp_path
        / "local"
        / "FlyDSL"
        / "rocm-toolkit"
        / hashlib.sha256(str(llvm).encode()).hexdigest()[:12]
    )

    assert rocm._windows_rocm_toolkit() == expected_stage.as_posix()
    assert len(commands) == 2
    assert all(command[:5] == ["cmd.exe", "/d", "/c", "mklink", "/J"] for command in commands)
    assert Path(commands[0][-1]) == llvm
    assert Path(commands[1][-1]) == llvm / "amdgcn"

    assert rocm._windows_rocm_toolkit() == expected_stage.as_posix()
    assert len(commands) == 2


def test_split_sdk_refuses_unexpected_existing_path(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    sdk = tmp_path / "sdk"
    llvm = sdk / "lib" / "llvm"
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / "ld.lld.exe").touch()
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    local = tmp_path / "local"
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    stage = local / "FlyDSL" / "rocm-toolkit" / hashlib.sha256(str(llvm).encode()).hexdigest()[:12]
    (stage / "llvm").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Refusing to use unexpected ROCm toolkit junction"):
        rocm._windows_rocm_toolkit()


def test_split_sdk_reports_junction_creation_failure(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    sdk = tmp_path / "sdk"
    llvm = sdk / "lib" / "llvm"
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / "ld.lld.exe").touch()
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(rocm, "_is_junction", lambda _path: False)
    monkeypatch.setattr(
        rocm.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="Access is denied."),
    )

    with pytest.raises(RuntimeError, match="Could not create ROCm toolkit junction.*Access is denied"):
        rocm._windows_rocm_toolkit()


def test_split_sdk_reports_incomplete_staging(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    sdk = tmp_path / "sdk"
    llvm = sdk / "lib" / "llvm"
    (llvm / "bin").mkdir(parents=True)
    (llvm / "bin" / "ld.lld.exe").touch()
    (llvm / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(sdk))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    junctions = {}
    real_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        target = junctions.get(path)
        if target is not None:
            return real_resolve(target, *args, **kwargs)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(rocm, "_is_junction", lambda path: path in junctions)

    def mklink(command, **kwargs):
        link, target = Path(command[-2]), Path(command[-1])
        junctions[link] = target
        link.mkdir()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(rocm.subprocess, "run", mklink)
    with pytest.raises(RuntimeError, match="Incomplete staged ROCm toolkit"):
        rocm._windows_rocm_toolkit()


def test_missing_sdk_has_actionable_error(rocm, monkeypatch, tmp_path):
    _clear_toolkit_env(monkeypatch)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "also-missing"))
    monkeypatch.setattr(rocm.sys, "path", [str(tmp_path / "empty-python-path")])

    with pytest.raises(RuntimeError, match="Could not find ld.lld.exe and AMDGPU bitcode"):
        rocm._windows_rocm_toolkit()
