# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device assertion API and AMDGPU lowering, without launching a failing GPU."""

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _cache_invalidating_env_values

pytestmark = [pytest.mark.l0_backend_agnostic]


def _module(build, types="i1", *, host=False):
    with ir.Context() as ctx, ir.Location.file("assert_test.py", 17, 4):
        ctx.load_all_available_dialects()
        args = ", ".join(f"%arg{i}: {ty}" for i, ty in enumerate(types.split(";")))
        if host:
            module = ir.Module.parse(f"module {{ func.func @test({args}) {{ return }} }}")
            fn = module.body.operations[0]
        else:
            module = ir.Module.parse(
                f"module {{ gpu.module @kernels {{ gpu.func @test({args}) kernel {{ gpu.return }} }} }}"
            )
            fn = module.body.operations[0].regions[0].blocks[0].operations[0]
        block = fn.regions[0].blocks[0]
        with ir.InsertionPoint(block.operations[0]):
            build(*block.arguments)
        module.operation.verify()
        return str(module)


def test_disabled_assert_has_no_ir_or_location_capture(monkeypatch):
    monkeypatch.delenv("FLYDSL_DEBUG_DEVICE_ASSERT", raising=False)
    # A disabled check can also be called outside any MLIR context.
    fx.device_assert(False, "disabled")
    text = _module(lambda cond: fx.device_assert(cond, "disabled"))
    assert "printf" not in text and "trap" not in text and "scf.if" not in text


@pytest.mark.parametrize("debug_info", ["0", "1"])
def test_assertions_are_independent_of_debug_info(monkeypatch, debug_info):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", debug_info)
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "0")
    disabled = _module(lambda cond: fx.device_assert(cond, "independent"))
    assert "llvm.intr.trap" not in disabled
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    enabled = _module(lambda cond: fx.device_assert(cond, "independent"))
    assert "llvm.intr.trap" in enabled


def test_enabled_scalar_and_source_diagnostic(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    text = _module(lambda cond: fx.device_assert(fx.Boolean(cond), "100% valid {}"))
    assert "scf.if" in text and "llvm.intr.trap" in text
    assert "100%% valid {}" in text
    assert "test_device_assert.py:" in text
    assert "gpu.block_id" in text and "gpu.thread_id" in text
    assert "gpu.barrier" not in text


@pytest.mark.parametrize("mask_type", ["i1", "vector<4xi1>"])
def test_vector_condition_and_mask(monkeypatch, mask_type):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "true")
    text = _module(lambda cond, mask: fx.device_assert(cond, "vector", mask=mask), f"vector<4xi1>;{mask_type}")
    assert "vector.reduction <and>" in text
    assert "arith.ori" in text and "arith.xori" in text
    assert text.count("llvm.intr.trap") == 1


@pytest.mark.parametrize("condition", [True, False])
def test_python_boolean_condition(monkeypatch, condition):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    _module(lambda _: fx.device_assert(condition))


@pytest.mark.parametrize(
    "types,match",
    [
        ("i32", "boolean"),
        ("vector<4xf32>", "boolean"),
        ("vector<4xi1>;vector<2xi1>", "shape"),
        ("i1;vector<4xi1>", "shape"),
        ("i1;i32", "boolean"),
    ],
)
def test_invalid_conditions_and_masks(monkeypatch, types, match):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    with pytest.raises(TypeError, match=match):
        _module(lambda cond, *mask: fx.device_assert(cond, mask=mask[0] if mask else None), types)


def test_message_validation(monkeypatch):
    monkeypatch.delenv("FLYDSL_DEBUG_DEVICE_ASSERT", raising=False)
    with pytest.raises(TypeError, match="string"):
        fx.device_assert(True, 1)
    with pytest.raises(ValueError, match="NUL"):
        fx.device_assert(True, "bad\0message")


def test_reject_launcher(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    with pytest.raises(ValueError, match="GPU kernel"):
        _module(lambda cond: fx.device_assert(cond), host=True)


def test_debug_mode_invalidates_cache_key(monkeypatch):
    monkeypatch.delenv("FLYDSL_DEBUG_DEVICE_ASSERT", raising=False)
    disabled = _cache_invalidating_env_values()
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    assert _cache_invalidating_env_values() != disabled
    assert dict(_cache_invalidating_env_values())["FLYDSL_DEBUG_DEVICE_ASSERT"] == "1"
    monkeypatch.delenv("FLYDSL_DEBUG_DEVICE_ASSERT")
    assert _cache_invalidating_env_values() == disabled


@pytest.mark.parametrize(
    "condition,mask,fails",
    [
        (True, True, False),
        (False, True, True),
        (False, False, False),
    ],
)
def test_failure_branch_semantics(monkeypatch, condition, mask, fails):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    text = _module(lambda _: fx.device_assert(condition, mask=mask))
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(text)
        PassManager.parse("builtin.module(convert-vector-to-llvm,convert-arith-to-llvm,canonicalize)").run(
            module.operation
        )
        assert ("llvm.intr.trap" in str(module)) == fails


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("types", ["i1;i1", "vector<4xi1>;vector<4xi1>"])
def test_rocdl_lowering(monkeypatch, arch, types):
    monkeypatch.setenv("FLYDSL_DEBUG_DEVICE_ASSERT", "1")
    text = _module(lambda cond, mask: fx.device_assert(cond, "lowering", mask=mask), types)
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(text)
        PassManager.parse(
            "builtin.module(gpu.module(convert-scf-to-cf,"
            f"convert-gpu-to-rocdl{{chipset={arch} runtime=HIP use-bare-ptr-memref-call-conv=true}}),"
            "convert-vector-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts)"
        ).run(module.operation)
        module.operation.verify()
        lowered = str(module)
        assert "llvm.intr.trap" in lowered
        assert "__ockl_printf" in lowered
        assert "gpu.printf" not in lowered


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
