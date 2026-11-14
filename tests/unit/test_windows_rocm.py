# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Windows ROCm plumbing that runs on any OS: ROCDL toolkit staging, arch detection errors,
and the JIT cache lock (fcntl on POSIX, msvcrt on Windows)."""

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.jit_function import FileLock
from flydsl.runtime import device
from flydsl.runtime.device_runtime import rocm as rw

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _fresh_toolkit_cache():
    rw._rocm_toolkit_root.cache_clear()
    yield
    rw._rocm_toolkit_root.cache_clear()


def _write(path, data=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_toolkit_is_staged_from_real_files(tmp_path, monkeypatch):
    llvm = tmp_path / "sdk" / "lib" / "llvm"
    _write(llvm / "bin" / "ld.lld.exe", b"lld")
    _write(llvm / "amdgcn" / "bitcode" / "ocml.bc")
    stage = rw._stage_toolkit(llvm, tmp_path / "stage")
    lld = stage / "llvm" / "bin" / "ld.lld.exe"
    for path in (lld, lld.parent, stage / "amdgcn" / "bitcode", stage / "amdgcn" / "bitcode" / "ocml.bc"):
        assert path.exists() and not path.is_symlink() and not getattr(os.path, "isjunction", lambda _: False)(path)

    def reinstall(data):
        (llvm / "bin" / "ld.lld.exe").unlink()
        _write(llvm / "bin" / "ld.lld.exe", data)
        rw._stage_toolkit(llvm, stage)
        assert lld.read_bytes() == data
        assert sorted(p.name for p in stage.rglob("*") if p.is_file()) == ["ld.lld.exe", "ocml.bc"]

    publish = rw._publish

    def racing_publish(src, tmp, dst):
        rival = dst.with_name("rival.tmp")  # another process publishes first, then cleans up
        publish(src, rival, dst)
        rival.unlink(missing_ok=True)
        publish(src, tmp, dst)

    def no_hard_links(*_):
        raise OSError("no hard links on this volume")

    monkeypatch.setattr(rw, "_publish", racing_publish)
    reinstall(b"lld 2")
    monkeypatch.setattr(os, "link", no_hard_links)
    reinstall(b"lld v3")  # sizes differ: two writes may share an mtime tick
    monkeypatch.setattr(shutil, "copy2", lambda *_: pytest.fail("staged an up-to-date file again"))
    rw._stage_toolkit(llvm, stage)


def test_toolkit_root_uses_a_ready_rocm_path_or_says_what_to_install(tmp_path, monkeypatch):
    _write(tmp_path / "llvm" / "bin" / "ld.lld.exe")
    (tmp_path / "amdgcn" / "bitcode").mkdir(parents=True)
    monkeypatch.setattr(rw, "_therock_root", lambda: None)
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert rw._rocm_toolkit_root() == tmp_path

    monkeypatch.delenv("ROCM_PATH")
    rw._rocm_toolkit_root.cache_clear()
    with pytest.raises(RuntimeError, match=r"rocm\[libraries\].*ROCM_PATH"):
        rw._rocm_toolkit_root()


def test_detect_target_without_an_arch_names_flydsl_gpu_arch(monkeypatch):
    monkeypatch.setattr("flydsl.compiler.backends.rocm.get_rocm_arch", lambda: "")
    with pytest.raises(RuntimeError, match="FLYDSL_GPU_ARCH"):
        RocmBackend.detect_target()


# Another process tries the lock for 0.5 s: exit 0 if it got it, 3 if it timed out.
_TRY_LOCK = """
import sys
from flydsl.compiler.jit_function import FileLock
try:
    with FileLock(sys.argv[1], timeout=0.5):
        pass
except RuntimeError:
    sys.exit(3)
"""


def _another_process_gets_lock(path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    # A lock that never gives up keeps the child running past timeout=30: TimeoutExpired.
    proc = subprocess.run([sys.executable, "-c", _TRY_LOCK, path], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode in (0, 3), proc.stderr
    return proc.returncode == 0


def test_jit_cache_lock_keeps_other_processes_out_until_released(tmp_path):
    path = str(tmp_path / "kernel.lock")
    with FileLock(path):
        assert not _another_process_gets_lock(path)
    assert _another_process_gets_lock(path)


class _FakeHip:
    def __init__(self, count, status=0):
        self.count, self.status = count, status

    def hipGetDevice(self, ref):
        return self.status

    def hipGetDeviceCount(self, ref):
        ref._obj.value = self.count
        return self.status


@pytest.mark.parametrize("hip, expected", [(_FakeHip(2), 2), (_FakeHip(0, status=100), 0), (None, 0)])
def test_windows_device_count_comes_from_hip(monkeypatch, hip, expected):
    # Windows has no rocm_agent_enumerator; status 100 is hipErrorNoDevice, None is no HIP at all.
    monkeypatch.setattr(device, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(rw, "_HIP_LIB", hip)
    monkeypatch.setattr(rw, "_HIP_LIB_TRIED", True)
    device.get_rocm_device_count.cache_clear()
    try:
        assert device.get_rocm_device_count() == expected
    finally:
        device.get_rocm_device_count.cache_clear()
