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
    # No device globals at all: the buffer arrives as a kernel argument and its first
    # record slot holds the cursor, so nothing needs host-side symbol binding.
    assert text.count("llvm.mlir.global") == 0


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


def test_traced_kernel_stays_disk_cacheable(monkeypatch):
    """Tracing must not register a post_load_processor.

    Doing so sets ``extern_linked`` in the jit layer, which disables the disk cache for
    that kernel -- so every profiling run would recompile from scratch. Passing the trace
    buffer as an implicit kernel argument avoids the callback entirely.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        iket.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    assert not getattr(
        launch, "_extern_linkage_keys", None
    ), "tracing registered a post_load_processor, which disables the disk cache"


def test_trace_buffer_is_a_trailing_kernel_argument(monkeypatch):
    """The buffer is an operand of gpu.launch_func, not an async dependency.

    The stream travels through async_dependencies; putting a pointer there is a type
    error rather than a kernel argument.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        iket.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _ir(launch)
    assert "fly.ptr<i8, global>" in text, "trace buffer parameter is missing"
    launch_line = next(ln for ln in text.splitlines() if "gpu.launch_func" in ln)
    assert "fly.ptr" in launch_line, "buffer is not passed as a launch operand"


def test_buffer_pointer_is_real_when_the_kernel_can_launch(monkeypatch):
    """A launchable traced kernel must get a real device address, not a null pointer.

    The kernel dereferences this pointer, so a null reaches the GPU as a segfault at
    launch with no Python-level error -- the failure mode this test exists to prevent.
    Under COMPILE_ONLY nothing launches, so allocation is skipped there instead.
    """
    from flydsl.compiler.jit_function import _ensure_iket_buffer_arg

    _frontend_only(monkeypatch, enable=True)  # sets COMPILE_ONLY=1

    args = []
    assert _ensure_iket_buffer_arg(args) is True
    assert len(args) == 1
    # COMPILE_ONLY: no allocation, so no device is touched.
    assert not args[0].pointer  # c_void_p(None) is falsy

    # Without COMPILE_ONLY the helper must allocate rather than pass a null through.
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    allocated = {}

    class _FakeBuffer:
        device_ptr = 0xDEADBEEF

    from flydsl.expr import iket_emit

    def _fake_ensure():
        allocated["called"] = True
        return _FakeBuffer()

    monkeypatch.setattr(iket_emit, "ensure_buffer", _fake_ensure)
    args2 = []
    assert _ensure_iket_buffer_arg(args2) is True
    assert allocated.get("called"), "launchable kernel did not allocate a trace buffer"
    assert args2[0].pointer.value == 0xDEADBEEF
