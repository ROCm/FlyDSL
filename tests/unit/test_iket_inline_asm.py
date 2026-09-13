# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the gfx942/gfx950 timestamp and hardware-id inline asm wrappers.

These wrappers back in-kernel event tracing, so the properties that matter are
(a) they lower to the intended ISA, and (b) two identical event sites are not
coalesced -- a tracing primitive that gets CSE'd silently merges two events into
one.

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
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-iket-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def _kernel_ir(launch_fn):
    assert launch_fn._mem_cache, "expected at least one cached compilation"
    return next(iter(launch_fn._mem_cache.values())).source_ir


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_wrappers_emit_expected_inline_asm(monkeypatch, arch):
    """Each wrapper emits its instruction; s_memrealtime carries its lgkm wait."""
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

    # Assert each asm string as a unit, not its pieces. s_memrealtime is an SMEM op whose
    # result register is invalid until the wait, so ORDER is the property that matters --
    # two independent substring checks pass even with the wait placed before the read.
    # "\\0A" is how MLIR escapes the newline inside the asm string.
    assert "s_memrealtime $0\\0As_waitcnt lgkmcnt(0)" in ir_text
    assert "s_getreg_b32 $0, hwreg(4)" in ir_text  # HW_REG_HW_ID, whole register
    assert "s_getreg_b32 $0, hwreg(20, 0, 4)" in ir_text  # XCC_ID[3:0], extracted by hardware

    # All three are SCALAR instructions: "=v" would allocate a VGPR and emit a kernel that
    # does not assemble ("s_memrealtime v[0:1]" -> invalid operand for instruction).
    assert '"=v"' not in ir_text
    assert ir_text.count('"=s"') == 3

    # @dsl_loc_tracing must point each op at ITS OWN call line. Dropping the decorator
    # still leaves a loc() -- the op inherits the enclosing kernel's location instead --
    # so presence proves nothing. Resolve each loc through the #loc table and require the
    # three lines to be the three consecutive call sites in `probe` above.
    asm_lines = [ln for ln in ir_text.splitlines() if "llvm.inline_asm" in ln]
    assert len(asm_lines) == 3
    loc_table = dict(re.findall(r"^(#loc\d*) = loc\((.*)\)$", ir_text, re.M))
    resolved = [loc_table.get(ln.rsplit("loc(", 1)[1].rstrip(")"), "") for ln in asm_lines]
    lines = [int(m.group(1)) for r in resolved if (m := re.search(r'":(\d+):\d+$', r))]
    assert len(lines) == 3, f"inline asm ops lack resolvable locations: {resolved}"
    assert (
        lines == sorted(lines) and lines[-1] - lines[0] == 2
    ), f"event sites do not map to their own call lines, @dsl_loc_tracing missing: {resolved}"


def test_identical_timestamp_sites_survive_cse(monkeypatch):
    """Two identical reads must survive as two reads.

    This is what makes the wrappers usable for tracing: CSE merging them would
    silently turn two events into one. ``has_side_effects=True`` prevents it.

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
    assert traced.count("s_memrealtime") == 2, "expected two traced reads before optimization"

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(traced)
        PassManager.parse("builtin.module(cse,canonicalize)").run(module.operation)
        optimized = str(module)

    assert optimized.count("s_memrealtime") == 2, "cse coalesced two distinct event sites"


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
