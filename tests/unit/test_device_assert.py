# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device assertion API and AMDGPU lowering, without launching a failing GPU."""

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager

pytestmark = [pytest.mark.l0_backend_agnostic]


def _module(build, types="i1"):
    with ir.Context() as ctx, ir.Location.file("assert_test.py", 17, 4):
        ctx.load_all_available_dialects()
        args = ", ".join(f"%arg{i}: {ty}" for i, ty in enumerate(types.split(";")))
        module = ir.Module.parse(
            f"module {{ gpu.module @kernels {{ gpu.func @test({args}) kernel {{ gpu.return }} }} }}"
        )
        fn = module.body.operations[0].regions[0].blocks[0].operations[0]
        block = fn.regions[0].blocks[0]
        with ir.InsertionPoint(block.operations[0]):
            build(*block.arguments)
        module.operation.verify()
        return str(module)


def test_debug_gating_and_diagnostic(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "0")
    disabled = _module(lambda cond: fx.device_assert(cond, "disabled"))
    assert "printf" not in disabled and "trap" not in disabled and "scf.if" not in disabled

    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    text = _module(lambda cond: fx.device_assert(fx.Boolean(cond), "100% valid {}"))
    assert "scf.if" in text and "llvm.intr.trap" in text
    assert "100%% valid {}" in text
    assert "test_device_assert.py:" in text
    assert "gpu.block_id" in text and "gpu.thread_id" in text


def test_vector_condition(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "true")
    text = _module(lambda cond: fx.device_assert(cond, "vector"), "vector<4xi1>")
    assert "vector.reduction <or>" in text
    assert "arith.xori" in text


def test_failure_branch_semantics(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    for condition in (True, False):
        text = _module(lambda _: fx.device_assert(condition))
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = ir.Module.parse(text)
            PassManager.parse("builtin.module(canonicalize)").run(module.operation)
            assert ("llvm.intr.trap" in str(module)) == (not condition)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_compile_to_binary(monkeypatch, arch):
    """Exercise the actual JIT and code generator without needing a device."""
    import flydsl.compiler as flyc

    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")

    @flyc.kernel
    def check(limit: fx.Int32):
        fx.device_assert(fx.thread_idx.x < limit, "thread out of range")

    @flyc.jit
    def launch(limit: fx.Int32):
        check(limit).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(64)
    artifact = next(iter(launch._mem_cache.values()))
    assert "thread out of range" in artifact.source_ir
    assert "gpu.binary" in artifact.ir

    # Changing debug mode must select another specialization in the same JIT.
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "0")
    launch(64)
    assert len(launch._mem_cache) == 2
    disabled = [value for value in launch._mem_cache.values() if value is not artifact][0]
    assert "llvm.intr.trap" not in disabled.source_ir
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    launch(64)
    assert len(launch._mem_cache) == 2
