# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Backend-agnostic checks for the CUDA/NVVM compile backend.

These exercise the pure-Python surface (registry, target detection, pipeline
shape, address-space mapping) so the NVVM backend keeps regression coverage on
CI runners that have no NVIDIA GPU and no ``nvvm`` in ``FLYDSL_BACKENDS``.
Anything needing the FlyNVVM bindings or a device belongs elsewhere.
"""

import pytest

from flydsl.compiler.backends import get_backend
from flydsl.compiler.backends.cuda import CudaBackend
from flydsl.runtime.device_runtime.cuda import get_cuda_arch

pytestmark = pytest.mark.l0_backend_agnostic


def test_backend_is_discovered_under_the_cuda_name():
    backend = get_backend("cuda", arch="sm_80")
    assert isinstance(backend, CudaBackend)
    assert backend.target.backend == "cuda"
    assert backend.target.arch == "sm_80"
    assert backend.target.warp_size == 32


def test_supports_target_is_exclusive():
    assert CudaBackend.supports_target(CudaBackend.make_target("sm_90"))
    assert not CudaBackend.supports_target(get_backend("rocm", arch="gfx942").target)


@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
def test_pipeline_carries_the_nvvm_stages_and_chip(arch):
    fragments = CudaBackend(CudaBackend.make_target(arch)).pipeline_fragments(compile_hints={})
    joined = "\n".join(fragments)
    assert "convert-fly-to-nvvm" in joined
    assert "convert-gpu-to-nvvm" in joined
    assert f"chip={arch}" in joined
    # The target attribute is attached by nvvm-attach-target, never twice.
    assert "nvvm-attach-target" in joined
    assert CudaBackend(CudaBackend.make_target(arch)).gpu_module_targets() == []
    # The device binary must be the last fragment.
    assert fragments[-1].startswith("gpu-module-to-binary")


def test_external_binary_split_matches_the_full_pipeline():
    backend = CudaBackend(CudaBackend.make_target("sm_90"))
    pre, binary = backend.external_binary_pipeline_fragments(compile_hints={})
    assert [*pre, binary] == backend.pipeline_fragments(compile_hints={})


def test_fast_fp_math_hint_reaches_the_nvvm_target():
    backend = CudaBackend(CudaBackend.make_target("sm_90"))
    assert "fast=true" in "\n".join(backend.pipeline_fragments(compile_hints={"fast_fp_math": True}))
    assert "fast=false" in "\n".join(backend.pipeline_fragments(compile_hints={}))


def test_llvm_address_space_matches_the_nvvm_conversion():
    # Must stay in sync with mapToLLVMAddressSpace in FlyToNVVM.cpp. NVVM has no
    # dedicated register address space, so Register lowers to generic (0).
    from flydsl._mlir.dialects.fly import AddressSpace

    assert CudaBackend.llvm_address_space(AddressSpace.Generic) == 0
    assert CudaBackend.llvm_address_space(AddressSpace.Global) == 1
    assert CudaBackend.llvm_address_space(AddressSpace.Shared) == 3
    assert CudaBackend.llvm_address_space(AddressSpace.Register) == 0


def test_jit_runtime_libs_name_the_cuda_wrapper():
    backend = CudaBackend(CudaBackend.make_target("sm_90"))
    assert "libfly_cuda_runtime.so" in backend.jit_runtime_lib_basenames()
    assert "libfly_cuda_runtime.so" in backend.native_lib_patterns()


@pytest.mark.parametrize("value", ["sm_80", "sm_90", "sm_100"])
def test_get_cuda_arch_honors_the_arch_override(monkeypatch, value):
    monkeypatch.setenv("ARCH", value)
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    assert get_cuda_arch() == value


def test_get_cuda_arch_ignores_a_non_sm_override(monkeypatch):
    """A gfx arch left over from a ROCm session must not leak into sm_* land.

    Detection falls through to the driver, which reports ``sm_XX`` when a device
    is visible and ``None`` when there is none (as on a ROCm CI runner) -- never
    the gfx value.
    """
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    arch = get_cuda_arch()
    assert arch is None or arch.startswith("sm_")


def test_cuda_runtime_kind_is_registered():
    from flydsl.runtime.device_runtime import COMPILE_BACKEND_TO_RUNTIME_KIND, CudaDeviceRuntime

    assert COMPILE_BACKEND_TO_RUNTIME_KIND["cuda"] == "cuda"
    assert CudaDeviceRuntime.kind == "cuda"
