# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Invocation-device target resolution and device-bound JIT state."""

from __future__ import annotations

import threading

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.runtime.device_runtime as dr
from flydsl.compiler import jit_function
from flydsl.compiler.backends import _make_backend, get_backend
from flydsl.compiler.jit_argument import _dlpack_device
from flydsl.compiler.jit_executor import CompiledArtifact
from flydsl.expr.buffer_ops import _get_buffer_flags
from flydsl.runtime.device import get_rocm_arch
from flydsl.runtime.device_runtime import Device, DeviceRuntime

pytestmark = [pytest.mark.l0_backend_agnostic]


class _FakeRuntime(DeviceRuntime):
    kind = "rocm"

    def __init__(self, archs=("gfx942",), current=0):
        self.archs = tuple(archs)
        self.current = current
        self.set_calls = []

    def device_count(self) -> int:
        return len(self.archs)

    def current_device_id(self) -> int:
        return self.current

    def set_device_id(self, device_id: int) -> None:
        self.set_calls.append(device_id)
        self.current = device_id

    def device_arch(self, device_id: int) -> str:
        return self.archs[device_id]


class _DeviceArg:
    def __init__(self, device_id):
        self.device = Device(kind="rocm", index=device_id)

    def __get_ir_types__(self):
        return []

    def __cache_signature__(self):
        return (type(self),)

    def __c_abi_spec__(self):
        return []

    def __flydsl_device__(self):
        return self.device


class _DeviceStream:
    cuda_stream = 1234

    def __init__(self, device_id):
        self.device = torch.device("cuda", device_id)


@flyc.jit
def _device_launch(x):
    pass


@flyc.jit
def _device_less_launch():
    pass


@flyc.jit
def _stream_launch(x, stream: fx.Stream):
    pass


@flyc.jit
def _nested_launch():
    pass


@flyc.jit
def _parent_launch():
    _nested_launch()


@pytest.fixture(autouse=True)
def _target_env(monkeypatch):
    old_runtime = dr._instance
    for name in ("ARCH", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION", "COMPILE_ONLY"):
        monkeypatch.delenv(name, raising=False)
    _make_backend.cache_clear()
    if hasattr(_device_less_launch, "_cache_managers"):
        _device_less_launch._cache_managers.clear()
    _nested_launch._cache_managers.clear()
    _parent_launch._cache_managers.clear()
    yield
    dr._instance = old_runtime
    _make_backend.cache_clear()


def _key_and_invocation(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._build_full_cache_key(bound.arguments, return_invocation=True)


def test_target_override_precedence(monkeypatch):
    dr._instance = _FakeRuntime(("gfx1100",))
    device = Device(kind="rocm", index=0)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.4.2")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("ARCH", "gfx1201")

    assert get_backend(device=device).target.arch == "gfx1201"
    assert get_rocm_arch() == "gfx1201"

    monkeypatch.delenv("ARCH")
    assert get_backend(device=device).target.arch == "gfx950"
    assert get_rocm_arch() == "gfx950"

    monkeypatch.delenv("FLYDSL_GPU_ARCH")
    assert get_backend(device=device).target.arch == "gfx942"
    assert get_rocm_arch() == "gfx942"


def test_tensor_device_selects_target_and_cache_partition():
    dr._instance = _FakeRuntime(("gfx942", "gfx942", "gfx950"))

    key0, invocation0 = _key_and_invocation(_device_launch, _DeviceArg(0))
    key1, invocation1 = _key_and_invocation(_device_launch, _DeviceArg(1))
    key2, invocation2 = _key_and_invocation(_device_launch, _DeviceArg(2))

    assert invocation0.target.arch == invocation1.target.arch == "gfx942"
    assert invocation2.target.arch == "gfx950"
    assert key0 == key1
    assert key0 != key2
    assert invocation0.device != invocation1.device


def test_mixed_device_arguments_fail_before_compilation():
    @flyc.jit
    def launch(x, y):
        pass

    dr._instance = _FakeRuntime(("gfx942", "gfx950"))

    with pytest.raises(ValueError, match="same device"):
        _key_and_invocation(launch, _DeviceArg(0), _DeviceArg(1))


def test_stream_device_participates_in_target_and_mismatch_validation():
    dr._instance = _FakeRuntime(("gfx942", "gfx950"))

    with pytest.raises(ValueError, match="same device"):
        _key_and_invocation(_stream_launch, _DeviceArg(0), _DeviceStream(1))

    _, invocation = _key_and_invocation(_stream_launch, _DeviceArg(1), _DeviceStream(1))
    assert invocation.device == Device(kind="rocm", index=1)
    assert invocation.target.arch == "gfx950"


def test_buffer_flags_follow_canonical_override_precedence(monkeypatch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")
    monkeypatch.setenv("ARCH", "gfx1201")

    assert _get_buffer_flags() == 0x21027000

    monkeypatch.delenv("ARCH")
    assert _get_buffer_flags() == 0x00027000


def test_generic_dlpack_cuda_is_not_reclassified_as_rocm():
    assert _dlpack_device(2, 1) == Device(kind="cuda", index=1)
    assert _dlpack_device(10, 1) == Device(kind="rocm", index=1)


def test_cache_managers_are_isolated_by_target_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        jit_function,
        "_jit_function_cache_key",
        lambda _func, owner_cls=None, target=None: target.arch,
    )
    targets = [get_backend(arch=arch).target for arch in ("gfx942", "gfx950")]
    barrier = threading.Barrier(len(targets))
    results = {}

    def resolve(target):
        barrier.wait()
        results[target.arch] = _device_less_launch._ensure_cache_manager(target=target)

    threads = [threading.Thread(target=resolve, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results["gfx942"][0] == "gfx942"
    assert results["gfx950"][0] == "gfx950"
    assert results["gfx942"][1] is not results["gfx950"][1]
    assert results["gfx942"][1].cache_dir.name.endswith("_gfx942")
    assert results["gfx950"][1].cache_dir.name.endswith("_gfx950")


def test_nested_jit_cache_manager_inherits_invocation_target(monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    target = get_backend(arch="gfx950").target

    _parent_launch._ensure_cache_manager(target=target)

    nested_targets = {slot[1] for slot in _nested_launch._cache_managers}
    assert nested_targets == {target}


def test_compile_only_without_device_requires_explicit_target(monkeypatch):
    dr._instance = _FakeRuntime(("gfx942",))
    monkeypatch.setenv("COMPILE_ONLY", "1")

    with pytest.raises(RuntimeError, match="explicit target"):
        _key_and_invocation(_device_less_launch)

    monkeypatch.setenv("ARCH", "gfx950")
    _, invocation = _key_and_invocation(_device_less_launch)
    assert invocation.target.arch == "gfx950"
    assert invocation.device is None


def test_detection_failure_does_not_fall_back_to_gfx942():
    class _BrokenRuntime(_FakeRuntime):
        def device_arch(self, device_id):
            raise RuntimeError("HIP target query failed")

    dr._instance = _BrokenRuntime()

    with pytest.raises(RuntimeError, match="HIP target query failed"):
        get_rocm_arch()


def test_device_guard_restores_the_callers_device():
    runtime = _FakeRuntime(("gfx942", "gfx950"), current=0)

    with runtime.device_guard(1):
        assert runtime.current_device_id() == 1

    assert runtime.current_device_id() == 0
    assert runtime.set_calls == [1, 0]


def test_compiled_artifact_loads_device_bound_state_once_per_device(monkeypatch):
    artifact = CompiledArtifact(compiled_module="module", func_name="launch")
    loads = []

    def load(device):
        loads.append(device)
        return type("_Executable", (), {"func_exe": object()})()

    monkeypatch.setattr(artifact, "_load_executable", load)
    device0 = Device(kind="rocm", index=0)
    device1 = Device(kind="rocm", index=1)

    func0 = artifact._get_func_exe(device0)
    assert artifact._get_func_exe(device0) is func0
    assert artifact._get_func_exe(device1) is not func0
    assert loads == [device0, device1]
