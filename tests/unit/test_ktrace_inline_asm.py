# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the gfx942/gfx950 timestamp and hardware-id wrappers.

These wrappers back in-kernel wave tracing, so the properties that matter are
(a) they reach the intended instruction -- via LLVM intrinsics, not inline asm, so the
scheduler is not blocked by an opaque block -- and (b) two identical event sites are not
coalesced: a tracing primitive that gets CSE'd silently merges two events into one.

The tests drive arch selection through ``ARCH`` and stop before codegen, so they
exercise gfx950 behaviour without gfx950 hardware.
"""

import re

import pytest

import flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function
from flydsl.expr.rocdl import inline_asm

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _frontend_only(monkeypatch, arch):
    """Trace and lower to IR for *arch* without touching a device."""
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", arch)  # the compile target; the wrappers gate on this
    # Pin the host arch too, but only so get_rocm_arch() cannot fall through to
    # rocm_agent_enumerator (a 300s-timeout subprocess). The gate must not depend on it --
    # see test_gate_follows_compile_target_not_host_arch.
    monkeypatch.setenv("FLYDSL_GPU_ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-ktrace-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def _kernel_ir(launch_fn):
    assert launch_fn._mem_cache, "expected at least one cached compilation"
    return next(iter(launch_fn._mem_cache.values())).source_ir


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_wrappers_call_llvm_intrinsics_not_inline_asm(monkeypatch, arch):
    """The timestamp and hwreg reads go through llvm.call_intrinsic.

    Not a style preference: an asm block is opaque to the scheduler, so its lgkm wait has
    to be written into the string and no independent work can move across it. The
    intrinsic lets the backend place the wait at the first real use instead.
    """
    _frontend_only(monkeypatch, arch)

    @flyc.kernel
    def probe():
        inline_asm.s_memrealtime()
        inline_asm.s_getreg_hw_id()
        inline_asm.s_getreg_xcc_id()

    @flyc.jit
    def launch():
        probe().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    ir_text = _kernel_ir(launch)

    assert 'llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"' in ir_text
    # One intrinsic serves both registers; the hwreg selector is its immarg operand.
    assert ir_text.count('llvm.call_intrinsic "llvm.amdgcn.s.getreg"') == 2

    # The immarg encoding is id | (size-1)<<11 (offset is 0 for both). Getting it wrong
    # reads a different register and yields plausible-looking garbage, so pin both values
    # rather than just checking that some constant is present.
    assert f"arith.constant {4 | (31 << 11)} : i32" in ir_text  # HW_ID, whole 32-bit register
    assert f"arith.constant {20 | (3 << 11)} : i32" in ir_text  # XCC_ID[3:0]

    # No inline asm should remain on this path at all.
    assert "llvm.inline_asm" not in ir_text

    # @dsl_loc_tracing must point each op at ITS OWN call line. Dropping the decorator
    # still leaves a loc() -- the op inherits the enclosing kernel's location instead --
    # so presence proves nothing. Resolve each loc through the #loc table and require the
    # three lines to be the three consecutive call sites in `probe` above.
    intrinsic_lines = [ln for ln in ir_text.splitlines() if "llvm.call_intrinsic" in ln]
    assert len(intrinsic_lines) == 3
    loc_table = dict(re.findall(r"^(#loc\d*) = loc\((.*)\)$", ir_text, re.M))
    resolved = [loc_table.get(ln.rsplit("loc(", 1)[1].rstrip(")"), "") for ln in intrinsic_lines]
    lines = [int(m.group(1)) for r in resolved if (m := re.search(r'":(\d+):\d+$', r))]
    assert len(lines) == 3, f"intrinsic ops lack resolvable locations: {resolved}"
    assert (
        lines == sorted(lines) and lines[-1] - lines[0] == 2
    ), f"event sites do not map to their own call lines, @dsl_loc_tracing missing: {resolved}"


def test_identical_timestamp_sites_survive_cse(monkeypatch):
    """Two identical reads must survive as two reads.

    This is what makes the wrappers usable for tracing: CSE merging them would
    silently turn two events into one.

    Both intrinsics ARE declared ``IntrNoMem`` (readnone) in IntrinsicsAMDGPU.td -- what
    keeps them distinct is ``IntrHasSideEffects`` at the LLVM level, and at the MLIR level
    that ``llvm.call_intrinsic`` carries no ``Pure``/``NoMemoryEffect`` interface for CSE
    to key on. Neither is a property of this module, so the pass is run for real here
    rather than assumed from the intrinsic's attributes.

    The assertion runs a real ``cse`` pass rather than inspecting the traced IR --
    counting ops before any pass runs would pass even with side effects disabled,
    so it would not be testing anything.
    """
    _frontend_only(monkeypatch, "gfx950")

    @flyc.kernel
    def probe():
        inline_asm.s_memrealtime()
        inline_asm.s_memrealtime()

    @flyc.jit
    def launch():
        probe().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    traced = _kernel_ir(launch)
    assert (
        traced.count('llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"') == 2
    ), "expected two traced reads before optimization"

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(traced)
        PassManager.parse("builtin.module(cse,canonicalize)").run(module.operation)
        optimized = str(module)

    assert (
        optimized.count('llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"') == 2
    ), "cse coalesced two distinct event sites"


@pytest.mark.parametrize(
    "wrapper",
    [inline_asm.s_memrealtime, inline_asm.s_getreg_hw_id, inline_asm.s_getreg_xcc_id],
    ids=["s_memrealtime", "s_getreg_hw_id", "s_getreg_xcc_id"],
)
def test_unsupported_arch_raises_with_a_useful_message(monkeypatch, wrapper):
    """Fail loudly on RDNA rather than emitting an instruction that does not exist.

    HW_REG_XCC_ID is gfx94x/gfx95x-only and s_memrealtime is absent on RDNA, so a
    silent fallback would produce a kernel that does not assemble.
    """
    _frontend_only(monkeypatch, "gfx1201")

    @flyc.kernel
    def probe():
        wrapper()

    @flyc.jit
    def launch():
        probe().launch(grid=(1, 1, 1), block=(32, 1, 1))

    # @dsl_loc_tracing builds a source Location before the wrapper body runs, so the
    # gate is only reachable inside a tracing context -- i.e. exactly how a kernel
    # author would hit it.
    with pytest.raises(ValueError) as excinfo:
        launch()

    message = str(excinfo.value)
    # Naming the supported set is not enough: the offending arch is the one piece of
    # information a user cannot get anywhere else.
    assert "gfx1201" in message, f"error does not name the unsupported arch: {message!r}"
    assert "gfx942" in message and "gfx950" in message


@pytest.mark.parametrize(
    "wrapper",
    [inline_asm.s_memrealtime, inline_asm.s_getreg_hw_id, inline_asm.s_getreg_xcc_id],
    ids=["s_memrealtime", "s_getreg_hw_id", "s_getreg_xcc_id"],
)
def test_gate_follows_compile_target_not_host_arch(monkeypatch, wrapper):
    """A CDNA host must not license an RDNA cross-compile.

    ``ARCH`` overrides the compile target while ``get_rocm_arch()`` reports the host, so
    a gate reading the host would pass this through and emit asm for the wrong target.
    That is not a hypothetical invocation -- ``ARCH=gfx950 COMPILE_ONLY=1`` is documented
    in .claude/skills/isa-resource-diff/SKILL.md.

    It fails quietly rather than loudly: on RDNA ``s_memrealtime`` does not assemble, but
    ``s_getreg`` does -- hwreg(4) and hwreg(20, 0, 4) silently retarget to
    WAVE_STATE_PRIV / WAVE_SCRATCH_BASE_LO, producing wrong data with no error.
    """
    _frontend_only(monkeypatch, "gfx1201")  # compile target = RDNA4
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")  # ...on a CDNA4 host

    @flyc.kernel
    def probe():
        wrapper()

    @flyc.jit
    def launch():
        probe().launch(grid=(1, 1, 1), block=(32, 1, 1))

    with pytest.raises(ValueError) as excinfo:
        launch()
    assert "gfx1201" in str(excinfo.value)


@pytest.mark.parametrize(
    "reg_id, size",
    [(64, 32), (-1, 32), (4, 0), (4, 33)],
    ids=["id_too_large", "id_negative", "size_zero", "size_too_large"],
)
def test_hwreg_immarg_rejects_out_of_range(reg_id, size):
    """Out-of-range arguments must raise, not wrap into a valid selector.

    Masking would keep the encoding well-formed: id 64 truncates to 4 (HW_ID) and size 0
    becomes a 32-bit field, both of which assemble and read a register the caller never
    asked for. A wrong register returns plausible numbers, so nothing downstream notices.
    """
    with pytest.raises(ValueError):
        inline_asm._hwreg_immarg(reg_id, size)
