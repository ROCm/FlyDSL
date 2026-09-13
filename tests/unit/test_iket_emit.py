# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device-side emission and host-side reconstruction for in-kernel event tracing.

The properties worth pinning are the ones whose failure is silent: instrumentation
surviving into a disabled build, the per-event atomic creeping back onto a mainloop's
hot path, and a loop-carried token failing at the loop boundary rather than where it
was written.
"""

import re

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function
from flydsl.expr import iket
from flydsl.utils import iket_trace

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _frontend_only(monkeypatch, *, enable: bool, arch: str = "gfx950", **iket_env):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("FLYDSL_GPU_ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    if enable:
        monkeypatch.setenv("FLYDSL_IKET_ENABLE", "1")
    else:
        monkeypatch.delenv("FLYDSL_IKET_ENABLE", raising=False)
    for key, value in iket_env.items():
        monkeypatch.setenv(f"FLYDSL_IKET_{key.upper()}", str(value))
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-iket-emit-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def _ir(launch_fn):
    launch_fn()
    assert launch_fn._mem_cache, "expected a cached compilation"
    return next(iter(launch_fn._mem_cache.values())).source_ir


def _canonicalized(text):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(text)
        PassManager.parse("builtin.module(canonicalize)").run(module.operation)
        return str(module)


def _normalize(text):
    """Erase names and locations so two kernels can be compared structurally."""
    text = re.sub(r"loc\([^)]*\)|#loc\d* = .*", "", text)
    # Both the kernel symbol and the host launch wrapper are named after the Python
    # function, so erase them: only the structure is being compared.
    return re.sub(r"@?(launch_)?(annotated|plain)(_\d+)?", "K", text)


def test_three_events_emit_one_atomic_not_three(monkeypatch):
    """The slot atomic belongs in the prologue, once per wave.

    Claiming a slot per *event* would put an agent-scope atomic on a mainloop's hot path,
    which costs more than everything else in the instrumentation combined.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        iket.mark("a")
        iket.mark("b")
        iket.mark("c")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _ir(launch)
    assert text.count("s_memrealtime") == 3, "each event needs its own timestamp"
    assert text.count("atomicrmw") == 1, "slot claim must be hoisted to the prologue"
    assert text.count("llvm.mlir.global") == 2  # __iket_cursor, __iket_bufptr


def test_disabled_build_is_identical_to_uninstrumented(monkeypatch):
    """An annotated kernel compiled with iket off must match one with no annotations.

    This is the whole justification for the feature being safe to leave in tree.
    """
    _frontend_only(monkeypatch, enable=False)

    @flyc.kernel
    def annotated():
        iket.mark("start")
        iket.range_push("body")
        iket.range_pop()
        token = iket.range_start("tok")
        iket.range_end(token)

    @flyc.kernel
    def plain():
        pass

    @flyc.jit
    def launch_annotated():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    @flyc.jit
    def launch_plain():
        plain().launch(grid=(1, 1, 1), block=(64, 1, 1))

    annotated_ir = _ir(launch_annotated)
    for artifact in ("s_memrealtime", "atomicrmw", "__iket"):
        assert artifact not in annotated_ir, f"{artifact} leaked into a disabled build"

    # The disabled path still materialises two dead constants so the token can be
    # loop-carried; canonicalize removes them.
    assert _normalize(_canonicalized(annotated_ir)) == _normalize(_canonicalized(_ir(launch_plain)))


@pytest.mark.parametrize("enable", [True, False], ids=["on", "off"])
def test_sentinel_token_survives_scf_for(monkeypatch, enable):
    """The cross-iteration idiom must compile both ways.

    With iket off the token still becomes an scf.for carried variable -- the tracer
    registers `tok` by syntax -- so its two values have to be real SSA constants. A bare
    Python bool raises AttributeError at the loop boundary, before range_end runs.
    """
    _frontend_only(monkeypatch, enable=enable)

    @flyc.kernel
    def annotated():
        token = iket.sentinel_token("k_tile")
        for _ in fx.range(0, 8, 1):
            iket.range_end(token)
            token = iket.range_start("k_tile")
        iket.range_end(token)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    match = re.search(r"scf\.for[^\n]*", _ir(launch))
    assert match, "expected an scf.for"
    assert "iter_args" in match.group(0), "token was not loop-carried"


def test_prologue_is_emitted_at_kernel_entry(monkeypatch):
    """Prologue values must dominate every event, including events inside a loop.

    Emitting at the first event site instead puts them inside whatever region that site
    is in, and the failure surfaces as "operand does not dominate this use" far from the
    line that caused it.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        # The first event site is inside the loop; the prologue must not follow it there.
        for _ in fx.range(0, 4, 1):
            iket.mark("in_loop")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _ir(launch)
    body = text.split("scf.for", 1)[1]
    assert "atomicrmw" not in body, "prologue was emitted inside the loop body"
    assert text.count("atomicrmw") == 1


def test_block_filter_guards_the_store(monkeypatch):
    """FLYDSL_IKET_BLOCKS filters on the device, so a skipped wave writes nothing."""
    _frontend_only(monkeypatch, enable=True, blocks="0,0,0")

    @flyc.kernel
    def annotated():
        iket.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(4, 1, 1), block=(64, 1, 1))

    text = _ir(launch)
    assert "block_id" in text or "block_dim" in text or "gpu.block_id" in text
    assert text.count("arith.andi") >= 1, "filter is not combined with the leader predicate"


def test_payload_rejects_a_token(monkeypatch):
    """A RangeToken passed as a payload must raise, not produce a malformed record.

    Routing payloads through a multi-value-tolerant helper would silently write a list.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        token = iket.range_start("r")
        iket.mark("bad", token)
        iket.range_end(token)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    with pytest.raises(Exception, match="payload|single value|RangeToken"):
        launch()


# -- host-side reconstruction ------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_ranges_pair_by_start_slot_not_by_name():
    """N iterations of one range share an event_id, so only the slot can pair them."""
    names = {"k_tile": 1}
    wave = {"hw_id": 0x1234, "xcc_id": 0, "wave": 4, "simd": 3, "cu": 2, "sh": 0, "se": 1, "payload": 0}
    records = [
        {**wave, "ts": 100, "event_id": 1, "kind": iket.KIND_RANGE_START, "slot": 0, "start_slot": 0},
        {**wave, "ts": 150, "event_id": 1, "kind": iket.KIND_RANGE_END, "slot": 1, "start_slot": 0},
        {**wave, "ts": 200, "event_id": 1, "kind": iket.KIND_RANGE_START, "slot": 2, "start_slot": 0},
        {**wave, "ts": 260, "event_id": 1, "kind": iket.KIND_RANGE_END, "slot": 3, "start_slot": 2},
    ]

    summary = iket_trace.summarize(records, names)
    assert summary["phases"]["k_tile"]["count"] == 2
    # 50 + 60 ticks at 10 ns each: pairing by name would have merged the two ranges.
    assert summary["phases"]["k_tile"]["total_ns"] == pytest.approx(1100.0)


@pytest.mark.l0_backend_agnostic
def test_records_are_grouped_by_wave_before_ordering():
    """Cursor order is global atomic order, not per-wave order."""
    names = {"phase": 1}
    common = {"payload": 0, "event_id": 1, "simd": 0, "cu": 0, "sh": 0, "se": 0}
    records = [
        {
            **common,
            "ts": 100,
            "hw_id": 1,
            "xcc_id": 0,
            "wave": 1,
            "kind": iket.KIND_RANGE_PUSH,
            "slot": 0,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 110,
            "hw_id": 2,
            "xcc_id": 1,
            "wave": 2,
            "kind": iket.KIND_RANGE_PUSH,
            "slot": 1,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 140,
            "hw_id": 1,
            "xcc_id": 0,
            "wave": 1,
            "kind": iket.KIND_RANGE_POP,
            "slot": 2,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 180,
            "hw_id": 2,
            "xcc_id": 1,
            "wave": 2,
            "kind": iket.KIND_RANGE_POP,
            "slot": 3,
            "start_slot": 0,
        },
    ]

    summary = iket_trace.summarize(records, names)
    assert summary["waves"] == 2
    # 40 + 70 ticks; interleaving the two waves would pair the wrong push with the wrong pop.
    assert summary["phases"]["phase"]["total_ns"] == pytest.approx(1100.0)


@pytest.mark.l0_backend_agnostic
def test_chrome_trace_names_each_wave_by_hardware_location():
    names = {"phase": 1}
    records = [
        {
            "ts": 100,
            "payload": 7,
            "hw_id": 0,
            "xcc_id": 3,
            "event_id": 1,
            "kind": iket.KIND_MARK,
            "slot": 0,
            "start_slot": 0,
            "wave": 5,
            "simd": 2,
            "cu": 9,
            "sh": 1,
            "se": 2,
        }
    ]
    doc = iket_trace.build_chrome_trace(records, names, kernel="gemm")
    labels = [e["args"]["name"] for e in doc["traceEvents"] if e.get("name") == "thread_name"]
    assert labels == ["XCD3/SE2/SH1/CU9/SIMD2/wave5"]
    instant = [e for e in doc["traceEvents"] if e.get("ph") == "i"]
    assert instant and instant[0]["name"] == "phase" and instant[0]["args"]["payload"] == 7
