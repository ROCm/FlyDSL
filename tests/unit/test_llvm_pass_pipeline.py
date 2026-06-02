# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Custom LLVM pass pipeline JIT plumbing (no kernel execution / external toolchain needed)."""

import flydsl.compiler as flyc
from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.external_llvm import llvm_opt_fingerprint
from flydsl.compiler.jit_function import _effective_llvm_pass_config


def test_recodegen_fragments_use_opt_level_zero_by_default():
    backend = RocmBackend(RocmBackend.detect_target())
    arch = backend.target.arch
    attach, binary = backend.llvm_recodegen_fragments(compile_hints={})
    assert attach.startswith("rocdl-attach-target{")
    assert "O=0" in attach
    assert f"chip={arch}" in attach
    assert binary.startswith("gpu-module-to-binary{")
    assert "format=fatbin" in binary


def test_recodegen_fragments_opt_level_override():
    backend = RocmBackend(RocmBackend.detect_target())
    attach, _ = backend.llvm_recodegen_fragments(compile_hints={}, opt_level=3)
    assert "O=3" in attach


def test_jit_decorator_records_llvm_pass_hints():
    @flyc.jit(llvm_pass_pipeline="default<O3>,my-pass", llvm_pass_plugins=["/tmp/libMy.so"])
    def f():  # pragma: no cover - never executed
        pass

    assert f.compile_hints["llvm_pass_pipeline"] == "default<O3>,my-pass"
    assert f.compile_hints["llvm_pass_plugins"] == ["/tmp/libMy.so"]


def test_jit_decorator_without_llvm_pass_has_no_hints():
    @flyc.jit
    def f():  # pragma: no cover - never executed
        pass

    assert "llvm_pass_pipeline" not in f.compile_hints


def test_effective_config_prefers_hints_over_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_PASS_PIPELINE", "default<O0>")
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_PASS_PLUGINS", "/env/a.so:/env/b.so")

    # hints win
    pipe, plugins = _effective_llvm_pass_config({"llvm_pass_pipeline": "default<O3>", "llvm_pass_plugins": ["/h.so"]})
    assert pipe == "default<O3>"
    assert plugins == ["/h.so"]

    # env fallback when hints absent
    pipe, plugins = _effective_llvm_pass_config({})
    assert pipe == "default<O0>"
    assert plugins == ["/env/a.so", "/env/b.so"]


def test_fingerprint_changes_with_pipeline_and_plugins(tmp_path):
    assert llvm_opt_fingerprint("default<O0>") != llvm_opt_fingerprint("default<O3>")

    so = tmp_path / "libP.so"
    so.write_bytes(b"v1")
    fp1 = llvm_opt_fingerprint("default<O0>", [str(so)])
    so.write_bytes(b"v2-changed")
    fp2 = llvm_opt_fingerprint("default<O0>", [str(so)])
    assert fp1 != fp2  # plugin content edit invalidates
    assert str(so) in fp1


def test_fingerprint_tolerates_missing_plugin():
    fp = llvm_opt_fingerprint("default<O0>", ["/does/not/exist.so"])
    assert "<missing>" in fp
