# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Frontend annotation, its expansion, and host-side reconstruction for wave tracing.

The properties worth pinning are the ones whose failure is silent: instrumentation
surviving into a disabled build, the contended atomic creeping back out of the
prologue onto a mainloop's hot path, and a loop-carried token failing at the loop
boundary rather than where it was written.

Tests that assert on the record-writing shape run convert-fly-ktrace-to-rocdl first,
via _expanded(); the frontend alone only produces fly_ktrace ops.
"""

import re
import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function
from flydsl.expr import ktrace
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils import ktrace_trace

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

# Most tests here drive the compile target through ARCH and stop before codegen, so they
# run anywhere. The two subprocess probes below cannot: they launch for real, on the
# HOST arch, and ktrace is gated to CDNA3/CDNA4 -- on RDNA the gate correctly refuses to
# emit s_memrealtime, which does not exist there.
_arch = get_rocm_arch() or ""
_skip_not_cdna = pytest.mark.skipif(
    not _arch.startswith(("gfx942", "gfx950")),
    reason=f"ktrace needs gfx942/gfx950 to launch, got {_arch}",
)


def _frontend_only(monkeypatch, *, enable: bool, arch: str = "gfx950", **ktrace_env):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("FLYDSL_GPU_ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    if enable:
        monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "1")
    else:
        monkeypatch.delenv("FLYDSL_KTRACE_ENABLE", raising=False)
    for key, value in ktrace_env.items():
        monkeypatch.setenv(f"FLYDSL_KTRACE_{key.upper()}", str(value))
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-ktrace-emit-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def _ir(launch_fn):
    launch_fn()
    assert launch_fn._mem_cache, "expected a cached compilation"
    return next(iter(launch_fn._mem_cache.values())).source_ir


def _expanded(launch_fn):
    """Frontend IR with the ktrace annotations expanded into their record form.

    The frontend emits `fly_ktrace.*` annotations; the slot claim, leader guard,
    bounds checks and stores are produced by convert-fly-ktrace-to-rocdl. Tests that
    assert on that shape have to run it, which is also what the real pipeline does --
    see _ktrace_tracing_enabled in compiler/backends/rocm.py.
    """
    text = _ir(launch_fn)
    with ir.Context():
        module = ir.Module.parse(text)
        PassManager.parse("builtin.module(convert-fly-ktrace-to-rocdl,reconcile-unrealized-casts)").run(
            module.operation
        )
        return str(module)


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


def test_slot_reservation_atomic_is_hoisted_to_the_prologue(monkeypatch):
    """The *contended* atomic belongs in the prologue, once per wave.

    Claiming the wave's slot range per event would put an agent-scope atomic that every
    wave contends on into a mainloop's hot path. The per-event counter is a separate,
    uncontended atomic on the wave's own reserved word.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.mark("a")
        ktrace.mark("b")
        ktrace.mark("c")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert text.count('llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"') == 3, "each event needs its own timestamp"
    # One contended reservation in the prologue, plus one uncontended counter bump per
    # event -- the latter is what makes a call site inside a loop record every iteration.
    assert text.count("atomicrmw") == 4
    # No device globals at all: the buffer arrives as a kernel argument and its first
    # record slot holds the cursor, so nothing needs host-side symbol binding.
    assert text.count("llvm.mlir.global") == 0


def test_disabled_build_is_identical_to_uninstrumented(monkeypatch):
    """An annotated kernel compiled with ktrace off must match one with no annotations.

    This is the whole justification for the feature being safe to leave in tree.
    """
    _frontend_only(monkeypatch, enable=False)

    @flyc.kernel
    def annotated():
        ktrace.mark("start")
        ktrace.range_push("body")
        ktrace.range_pop()
        token = ktrace.range_start("tok")
        ktrace.range_end(token)
        # sentinel_token has to be here: it is the only entry point that must return a
        # value rather than returning early, so it is the one that can leak a
        # fly_ktrace op into a disabled build. It did.
        ktrace.range_end(ktrace.sentinel_token("seed"))

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
    for artifact in ("s.memrealtime", "s_memrealtime", "atomicrmw", "__ktrace", "fly_ktrace"):
        assert artifact not in annotated_ir, f"{artifact} leaked into a disabled build"

    # The disabled path still materialises two dead constants so the token can be
    # loop-carried; canonicalize removes them.
    assert _normalize(_canonicalized(annotated_ir)) == _normalize(_canonicalized(_ir(launch_plain)))


@pytest.mark.parametrize("enable", [True, False], ids=["on", "off"])
def test_sentinel_token_survives_scf_for(monkeypatch, enable):
    """The cross-iteration idiom must compile both ways.

    With ktrace off the token still becomes an scf.for carried variable -- the tracer
    registers `tok` by syntax -- so its two values have to be real SSA constants. A bare
    Python bool raises AttributeError at the loop boundary, before range_end runs.
    """
    _frontend_only(monkeypatch, enable=enable)

    @flyc.kernel
    def annotated():
        token = ktrace.sentinel_token("k_tile")
        for _ in fx.range(0, 8, 1):
            ktrace.range_end(token)
            token = ktrace.range_start("k_tile")
        ktrace.range_end(token)

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
            ktrace.mark("in_loop")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    head = text.split("scf.for", 1)[0]
    # The wave's slot reservation must precede the loop; the per-event counter bump
    # legitimately lives inside it.
    assert "atomicrmw" in head, "prologue reservation was not emitted before the loop"


def test_block_filter_guards_the_store(monkeypatch):
    """FLYDSL_KTRACE_BLOCKS filters on the device, so a skipped wave writes nothing."""
    _frontend_only(monkeypatch, enable=True, blocks="0,0,0")

    @flyc.kernel
    def annotated():
        ktrace.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(4, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert "block_id" in text or "block_dim" in text or "gpu.block_id" in text
    assert text.count("arith.andi") >= 1, "filter is not combined with the leader predicate"


def test_payload_rejects_a_token(monkeypatch):
    """A RangeToken passed as a payload must raise, not produce a malformed record.

    Routing payloads through a multi-value-tolerant helper would silently write a list.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        token = ktrace.range_start("r")
        ktrace.mark("bad", token)
        ktrace.range_end(token)

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
        {**wave, "ts": 100, "event_id": 1, "kind": ktrace.KIND_RANGE_START, "slot": 0, "start_slot": 0},
        {**wave, "ts": 150, "event_id": 1, "kind": ktrace.KIND_RANGE_END, "slot": 1, "start_slot": 0},
        {**wave, "ts": 200, "event_id": 1, "kind": ktrace.KIND_RANGE_START, "slot": 2, "start_slot": 0},
        {**wave, "ts": 260, "event_id": 1, "kind": ktrace.KIND_RANGE_END, "slot": 3, "start_slot": 2},
    ]

    summary = ktrace_trace.summarize(records, names)
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
            "kind": ktrace.KIND_RANGE_PUSH,
            "slot": 0,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 110,
            "hw_id": 2,
            "xcc_id": 1,
            "wave": 2,
            "kind": ktrace.KIND_RANGE_PUSH,
            "slot": 1,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 140,
            "hw_id": 1,
            "xcc_id": 0,
            "wave": 1,
            "kind": ktrace.KIND_RANGE_POP,
            "slot": 2,
            "start_slot": 0,
        },
        {
            **common,
            "ts": 180,
            "hw_id": 2,
            "xcc_id": 1,
            "wave": 2,
            "kind": ktrace.KIND_RANGE_POP,
            "slot": 3,
            "start_slot": 0,
        },
    ]

    summary = ktrace_trace.summarize(records, names)
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
            "kind": ktrace.KIND_MARK,
            "slot": 0,
            "start_slot": 0,
            "wave": 5,
            "simd": 2,
            "cu": 9,
            "sh": 1,
            "se": 2,
        }
    ]
    doc = ktrace_trace.build_chrome_trace(records, names, kernel="gemm")
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
        ktrace.mark("a")

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
        ktrace.mark("a")

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
    from flydsl.compiler.jit_function import _ensure_ktrace_buffer_arg

    _frontend_only(monkeypatch, enable=True)  # sets COMPILE_ONLY=1

    args = []
    assert _ensure_ktrace_buffer_arg(args) is True
    assert len(args) == 1
    # COMPILE_ONLY: no allocation, so no device is touched.
    assert not args[0].pointer  # c_void_p(None) is falsy

    # Without COMPILE_ONLY the helper must allocate rather than pass a null through.
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    allocated = {}

    class _FakeBuffer:
        device_ptr = 0xDEADBEEF

    from flydsl.expr.rocdl import ktrace_emit

    def _fake_ensure():
        allocated["called"] = True
        return _FakeBuffer()

    monkeypatch.setattr(ktrace_emit, "ensure_buffer", _fake_ensure)
    args2 = []
    assert _ensure_ktrace_buffer_arg(args2) is True
    assert allocated.get("called"), "launchable kernel did not allocate a trace buffer"
    assert args2[0].pointer.value == 0xDEADBEEF


def test_loop_call_site_records_every_iteration(monkeypatch):
    """A call site inside a loop must write a distinct slot per iteration.

    The loop body is traced once, so a compile-time slot index makes every iteration
    overwrite the same record and only the last survives -- silently, and precisely for
    the per-k-tile annotation this feature exists to support. The slot index therefore
    has to come from a runtime counter.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        for _ in fx.range(0, 8, 1):
            ktrace.mark("per_iter")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    body = _expanded(launch).split("scf.for", 1)[1]
    # The slot must be derived inside the loop from an atomic, not from a constant
    # folded in at trace time.
    assert "atomicrmw" in body, "slot index is compile-time; loop iterations will collide"


def test_decode_skips_each_waves_counter_word():
    """The wave's first reserved slot holds its event counter, not a record.

    The counter is a small integer that lands in the ``ts`` field, so a decoder that only
    drops zero-timestamp slots reports it as a record with an absurd timestamp. Values
    here are taken from a real gfx950 buffer dump.
    """
    import struct

    from flydsl.utils import ktrace_runtime

    raw = struct.pack("<QQIIII", 8, 0, 0, 0, 0, 0)  # slot 1: counter, 8 events recorded
    for i in range(3):  # slots 2..4: real records
        raw += struct.pack("<QQIIII", 395099147398924 + i * 32, i, 1110450432, 0, 1, 0)

    recs, _ = ktrace_runtime.decode_records(raw)

    assert [r["slot"] for r in recs] == [2, 3, 4]
    assert [r["payload"] for r in recs] == [0, 1, 2]


def test_range_start_yields_the_slot_its_record_occupies(monkeypatch):
    """A token pairs on the record's own slot, the same index the host decodes.

    Records are written one slot past the counter's value, to skip the wave's counter
    word. Yielding the unoffset value made every END carry a start_slot one below its
    START -- 0 of 8 token ranges paired on real hardware, while push/pop ranges, which do
    not use the returned slot at all, still looked perfectly fine.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        tok = ktrace.range_start("span")
        ktrace.range_end(tok)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    # The guarded region yields an i32 out of both the leader guard and the bounds guard,
    # so the value that reaches range_end is the stored record's slot.
    assert text.count("scf.yield") >= 4, "slot must be yielded out of both guards"
    assert "scf.if" in text


def test_reading_a_trace_re_arms_the_buffer(monkeypatch, tmp_path):
    """``collect``/``summary`` must consume the trace, not accumulate it.

    The trace buffer is process-wide and its cursor only ever counts up, so without a
    reset every launch's claims stack on top of the last. A repeated trace -- a benchmark
    loop, an autotune sweep -- then fails with "trace overflowed" even though no single
    launch came near the capacity, and the error points at the buffer size rather than the
    missing reset.

    The reset must clear the records too, not just the cursor: decode drops a record on a
    zero timestamp, so slots left by a longer previous launch would otherwise be re-read
    as records of the next one once the cursor rewinds past them.
    """
    from flydsl.expr.rocdl import ktrace_emit

    class _FakeBuffer:
        def __init__(self):
            self.reads = 0
            self.zeroed = 0

        def read_records(self):
            self.reads += 1
            return [
                {
                    "ts": 100,
                    "payload": 0,
                    "hw_id": 0,
                    "xcc_id": 0,
                    "event_id": 1,
                    "kind": ktrace.KIND_RANGE_PUSH,
                    "start_slot": 0,
                    "slot": 2,
                    "wave": 0,
                    "simd": 0,
                    "cu": 0,
                    "sh": 0,
                    "se": 0,
                }
            ]

        def claimed_slots(self):
            # A fresh launch's worth of records on the first look, then empty -- the
            # cursor a re-armed buffer reports, which is what keeps the host cache valid.
            return 0 if self.zeroed else 8

        def reset(self):
            self.zeroed += 1

    buf = _FakeBuffer()
    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", None)
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: buf)

    assert ktrace.summary() is not None
    assert buf.zeroed == 1, "summary() left the buffer armed; the next launch accumulates"

    # The second call is served from the host cache, so it describes the SAME launch
    # instead of finding the buffer summary() just zeroed.
    assert ktrace.collect("k", out_dir=str(tmp_path)) is not None
    assert buf.reads == 1, "the device was re-read; the two calls describe different launches"


def test_reset_survives_an_overflow(monkeypatch):
    """An overflow must still re-arm the buffer.

    Leaving a full buffer behind makes every later call raise the same error, so the
    launch that actually overflowed is hidden behind the ones that merely inherited it.
    """
    from flydsl.expr.rocdl import ktrace_emit

    class _OverflowingBuffer:
        def __init__(self):
            self.zeroed = 0

        def read_records(self):
            raise RuntimeError("trace overflowed: waves claimed 999 slots")

        def claimed_slots(self):
            return 999

        def reset(self):
            self.zeroed += 1

    buf = _OverflowingBuffer()
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: buf)

    with pytest.raises(RuntimeError, match="trace overflowed"):
        ktrace.summary()
    assert buf.zeroed == 1, "an overflow left the buffer full; later calls inherit the error"


# The name table is host-side state, but the ids it maps are baked into the compiled
# binary. A disk-cache hit replays that binary WITHOUT re-running the kernel body, so any
# table rebuilt by tracing is empty exactly when it is needed. Two real processes are
# required: within one process the in-memory state hides the bug.
_CACHE_HIT_PROBE = """
import os, sys, json
os.environ["FLYDSL_KTRACE_ENABLE"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
os.environ["HOME"] = sys.argv[1]

import flydsl.compiler as flyc
from flydsl.expr import ktrace
from flydsl.expr.rocdl import ktrace_emit

@flyc.kernel
def annotated():
    ktrace.range_push("mainloop")
    ktrace.mark("tick")
    ktrace.range_pop()

@flyc.jit
def launch():
    annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

launch()
print("NAMES=" + json.dumps(ktrace_emit.name_table()))
"""


@pytest.mark.l2_device
@_skip_not_cdna
def test_name_table_survives_a_disk_cache_hit(tmp_path):
    """A second process must decode records against the ids its cached binary writes.

    Regression: the table was rebuilt only by tracing the kernel body, so on a cache hit
    ``name_table()`` returned ``{}``. Nothing raised -- ``summarize`` drops every record
    whose id is unknown, so a full trace reported ``phases: {}``, and the Chrome Trace
    renamed each event to ``eventN``. Reproduced only across processes.
    """
    import json
    import subprocess

    probe = tmp_path / "probe.py"
    probe.write_text(_CACHE_HIT_PROBE)
    home = tmp_path / "home"
    home.mkdir()

    def _run(tag):
        proc = subprocess.run(
            [sys.executable, str(probe), str(home)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, f"{tag} probe failed:\n{proc.stdout}\n{proc.stderr}"
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("NAMES=")]
        assert line, f"{tag} probe printed no name table:\n{proc.stdout}"
        return json.loads(line[-1][len("NAMES=") :])

    cold = _run("cold")
    assert cold, "the compiling run produced no name table at all -- probe is not tracing"

    warm = _run("warm")
    assert warm == cold, (
        "a disk-cache hit lost the name table: the cached binary still writes the ids "
        f"assigned at compile time ({cold}), so decoding against {warm} mislabels or "
        "silently drops every record"
    )


# The compile path decides whether the kernel gets a trailing trace-buffer parameter while
# a `ktrace` compile hint is in scope; the host packs its ABI slots after that scope has
# closed. If the second decision re-derives the answer it misses the hint, packs one slot
# fewer than the kernel declares, and the kernel dereferences whatever follows the array.
_HINT_ABI_PROBE = """
import os, sys
os.environ.pop("FLYDSL_KTRACE_ENABLE", None)
os.environ["HOME"] = sys.argv[1]
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import flydsl.compiler as flyc
from flydsl.expr import ktrace

@flyc.kernel
def annotated():
    ktrace.range_push("mainloop")
    ktrace.mark("tick")
    ktrace.range_pop()

@flyc.jit
def launch():
    annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

launch.compile_hints = {"ktrace": True}
launch()
summary = ktrace.summary()
print("PHASES=" + ",".join(sorted((summary or {}).get("phases", {}))))
"""


@pytest.mark.l2_device
@_skip_not_cdna
def test_compile_hint_packs_the_buffer_slot_it_compiled(tmp_path):
    """A ``ktrace`` compile hint must reach the ABI packing, not just the annotation.

    Regression: ``_ensure_ktrace_buffer_arg`` runs inside the ``compile_hints`` scope and
    added the trailing parameter, while ``_build_call_state`` runs after that scope closed,
    saw only the unset env var, and packed one slot fewer. The kernel then read its trace
    buffer pointer from past the end of the packed array -- this probe core-dumped before
    the fix, so it runs in a subprocess to keep a crash from taking the suite with it.
    """
    import subprocess

    probe = tmp_path / "probe.py"
    probe.write_text(_HINT_ABI_PROBE)
    home = tmp_path / "home"
    home.mkdir()

    proc = subprocess.run(
        [sys.executable, str(probe), str(home)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "the hint-enabled launch did not exit cleanly (a negative return code is the "
        f"ABI mismatch faulting on device):\nrc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("PHASES=")]
    assert line, f"probe printed no summary:\n{proc.stdout}\n{proc.stderr}"
    assert line[-1] == "PHASES=mainloop,tick", (
        "the hint compiled an instrumented kernel but the host did not collect its " f"phases: {line[-1]}"
    )


def test_suppressed_store_yields_an_unpairable_slot(monkeypatch):
    """A range whose START was suppressed must not pair with a real record.

    Regression: the else-branch yielded the unoffset ``slot``, which is exactly the
    absolute slot the PREVIOUS event wrote (a store at local-1 lands at base+local). A
    range_end carrying that id popped an unrelated open range and charged a fabricated
    duration to the wrong phase -- silently, since both values are legal slot numbers.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        token = ktrace.range_start("r")
        ktrace.range_end(token)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    # -1 : i32 is UNPAIRED_SLOT; it decodes as 0xFFFFFFFF through the unsigned start_slot
    # field, a value no real slot can reach.
    assert "-1 : i32" in text, "the suppressed-store branch no longer yields the sentinel"


def test_store_bounds_use_unsigned_compares(monkeypatch):
    """The slot guards must be unsigned, or they invert once the cursor passes 2^31.

    Regression: both guards used signed ``slt`` on counters fed by an unsaturated i32
    atomic add. Once total claims cross 2^31 the slot is negative as a signed i32, ``slt``
    becomes unconditionally true, and the guard flips from "in bounds" to "always pass" --
    writing at a negative byte offset, the exact out-of-bounds store it exists to prevent.
    The host already reads the cursor as c_uint32, so unsigned is the shared reading.
    """
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert re.search(r"arith\.cmpi\s+ult", text), "store bounds are not unsigned"
    # The leader test is an equality compare and the filter uses its own predicate; no
    # signed less-than belongs anywhere in the emitted instrumentation.
    assert not re.search(r"arith\.cmpi\s+slt", text), "a signed bounds compare survived"


def test_range_end_without_a_start_is_dropped():
    """The host must drop an unpairable range_end, not pair or render it."""
    import struct

    from flydsl.utils import ktrace_runtime

    # Build the record through the real decoder rather than by hand, so this test cannot
    # drift from the on-device layout. Slot 0 is the wave counter word, which the decoder
    # skips, so the record goes in slot 1.
    packed = (ktrace.KIND_RANGE_END << 24) | 1
    raw = bytes(ktrace.RECORD_BYTES) + struct.pack("<QQIIII", 100, 0, 0, 0, packed, ktrace.UNPAIRED_SLOT)
    records, _ = ktrace_runtime.decode_records(raw)
    assert len(records) == 1, f"expected one decoded record, got {records}"
    assert records[0]["start_slot"] == ktrace.UNPAIRED_SLOT
    names = {"r": 1}

    assert ktrace_trace.summarize(records, names)["phases"] == {}, "an unpairable end was accumulated"

    events = ktrace_trace.build_chrome_trace(records, names, kernel="k")["traceEvents"]
    assert not [e for e in events if e.get("ph") == "e"], "a dangling range-end reached the trace"


def test_buffer_bytes_invalidates_the_cache(monkeypatch):
    """Every ktrace setting compiled INTO the kernel must be part of the cache key.

    Regression: FLYDSL_KTRACE_BUFFER_BYTES is emitted as the store-bounds constant
    (``_buffer_capacity_slots``), but was missing from the list. Two runs at different
    buffer sizes produced the same cache key, so the second was served the first's binary
    and kept its capacity. Shrinking the buffer then let the device write past the new
    allocation -- the guard still passed, because it checked the old, larger bound.
    """
    baked = {
        # env var -> the value it is compiled into
        "FLYDSL_KTRACE_ENABLE": "tracing on/off changes the whole kernel body",
        "FLYDSL_KTRACE_BLOCKS": "block-filter predicate",
        "FLYDSL_KTRACE_BUFFER_BYTES": "store-bounds capacity constant",
    }
    missing = [var for var in baked if var not in jit_function._CACHE_INVALIDATING_ENV_VARS]
    assert not missing, (
        "these ktrace settings are compiled into the kernel but do not invalidate the "
        f"JIT cache, so a stale binary can be served with the wrong constant: {missing}"
    )


def test_buffer_bytes_reaches_the_emitted_capacity(monkeypatch):
    """The guard constant must track the configured buffer size.

    Pins the link the test above relies on: if the capacity stopped being compiled in,
    the cache-key requirement would silently become unnecessary rather than wrong.
    """
    from flydsl.expr.rocdl import ktrace_emit

    _frontend_only(monkeypatch, enable=True, buffer_bytes=4 << 20)
    small = ktrace_emit._buffer_capacity_slots()
    _frontend_only(monkeypatch, enable=True, buffer_bytes=256 << 20)
    large = ktrace_emit._buffer_capacity_slots()
    assert large > small, "buffer_bytes no longer determines the emitted capacity"


def test_payload_zero_is_recorded(monkeypatch):
    """Payload 0 must reach the trace: it is the first loop index, not "no payload".

    Regression: a truthiness test dropped it, so iteration 0 of every annotated mainloop
    rendered with no payload while iterations 1..N-1 had one -- indistinguishable from a
    genuinely absent payload.
    """
    rec = {
        "ts": 100,
        "payload": 0,
        "hw_id": 0,
        "xcc_id": 0,
        "event_id": 1,
        "kind": ktrace.KIND_RANGE_PUSH,
        "start_slot": 0,
        "slot": 1,
        "wave": 0,
        "simd": 0,
        "cu": 0,
        "sh": 0,
        "se": 0,
    }
    events = ktrace_trace.build_chrome_trace([rec], {"k_tile": 1}, kernel="k")["traceEvents"]
    begins = [e for e in events if e.get("ph") == "B"]
    assert begins and begins[0]["args"] == {"payload": 0}, "a zero payload was dropped from the trace"


def test_collect_and_summary_agree_on_an_empty_trace(monkeypatch):
    """Both must report "nothing traced" the same way.

    Regression: collect() tested `not records` and summary() tested `records is None`, so
    a launch that recorded nothing (every wave excluded by FLYDSL_KTRACE_BLOCKS) made
    summary() return {"phases": {}} -- reading as "ran, no phases" -- while collect()
    correctly returned None.
    """
    from flydsl.expr.rocdl import ktrace_emit

    class _EmptyBuffer:
        def claimed_slots(self):
            return 0

        def read_records(self):
            return []

        def reset(self):
            pass

    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", None)
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: _EmptyBuffer())

    assert ktrace.summary() is None, "summary() reported phases for a trace with no records"
    assert ktrace.collect("k") is None


def test_events_per_wave_matches_the_pass_constant():
    """The Python constant and the C++ pass's kEventsPerWave must agree.

    They are two declarations of one number: the emitter bakes it into the per-wave
    bound and the decoder uses it to locate counter words, so a mismatch makes the
    decoder skip real records and read counter words as timestamps. It was an env var
    read independently in three places, which is exactly how that drift happened.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[2] / "lib/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.cpp"
    if not src.exists():
        pytest.skip("C++ pass not present in this checkout")
    m = re.search(r"kEventsPerWave\s*=\s*(\d+)", src.read_text())
    assert m, "kEventsPerWave not found in the pass source"
    assert int(m.group(1)) == ktrace.EVENTS_PER_WAVE, (
        f"pass has kEventsPerWave={m.group(1)} but Python has " f"EVENTS_PER_WAVE={ktrace.EVENTS_PER_WAVE}"
    )


def test_a_launch_invalidates_cached_records(monkeypatch):
    """Records cached from one launch must not be served to the next.

    Regression: the cache was validated against the device cursor, which cannot tell
    "nothing launched since the read" from "a launch that recorded nothing" -- a launch
    whose every wave is excluded by FLYDSL_KTRACE_BLOCKS claims no slots and leaves the
    cursor at zero. summary() then reported the PREVIOUS kernel's phase totals as if they
    described the new launch, with nothing signalling the swap.
    """
    from flydsl.expr.rocdl import ktrace_emit

    reads = []

    class _Buffer:
        def claimed_slots(self):
            return 0

        def read_records(self):
            reads.append(1)
            return []  # this launch recorded nothing

        def reset(self):
            pass

    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", [{"stale": True}])
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: _Buffer())
    monkeypatch.setattr(jit_function, "_invalidate_ktrace_records", jit_function._invalidate_ktrace_records)

    # Simulate the jit layer's per-launch hook, then read.
    jit_function._invalidate_ktrace_records()
    assert ktrace.summary() is None, "a new launch was served the previous launch's records"
    assert reads, "the device was never re-read after the launch"


def test_every_replay_path_takes_its_abi_from_the_artifact():
    """No replay path may re-derive the tracing decision from the ambient environment.

    Regression: the compile-lock branch -- taken when another process compiled the kernel
    while this one waited -- restored the name table but left the ABI flag unset, so
    _build_call_state fell back to reading the env. That read happens after the
    compile_hints scope has closed, so a {"ktrace": True} hint is invisible there: the
    binary declares the trailing buffer parameter and the host packs one slot fewer,
    leaving the kernel to dereference past the end of the packed array.

    Asserted structurally rather than by racing two processes: every site that produces a
    CallState from an artifact it did not just compile must pass ktrace_enabled.
    """
    import inspect

    source = inspect.getsource(jit_function.JitFunction.__call__)
    calls = [ln for ln in source.splitlines() if "_build_call_state(" in ln]
    assert calls, "no _build_call_state call sites found -- test is stale"

    # Each call must be given the decision explicitly; the default (None) means
    # "re-derive from the env", which is only safe where no artifact exists.
    for marker in ("ktrace_enabled=_ktrace_compiled", 'ktrace_enabled=getattr(cached_func, "_ktrace_traced", None)'):
        assert marker in source, f"a replay path does not pass its artifact's ABI flag: {marker}"

    # And the lock-hit branch must set the flag from the artifact it adopted.
    assert '_ktrace_compiled = getattr(compiled_func, "_ktrace_traced", None)' in source, (
        "the compile-lock cache-hit branch leaves _ktrace_compiled unset, so "
        "_build_call_state falls back to an env read that cannot see a ktrace compile hint"
    )


def test_colliding_event_ids_are_rejected(monkeypatch):
    """Two names on one id must raise, not silently drop one.

    Regression: merge_names used dict.update, so tracing a fresh kernel (epilogue -> 1)
    and then hitting the disk cache for one whose binary baked mainloop -> 1 left both
    names claiming id 1. The trace writer inverts the table, so one name vanished and its
    records rendered under the other's -- a plausible timeline for the wrong phase.
    Neither id can be renumbered, since each is already compiled into a binary.
    """
    from flydsl.expr.rocdl import ktrace_emit

    monkeypatch.setattr(ktrace_emit, "_NAMES", {})
    monkeypatch.setattr(ktrace_emit, "_ID_OWNER", {})

    # A fresh kernel's table, as the pass published it.
    ktrace_emit.merge_names({"epilogue": 1})
    with pytest.raises(RuntimeError, match="claimed by both"):
        ktrace_emit.merge_names({"mainloop": 1})


def test_merging_the_same_binding_twice_is_fine(monkeypatch):
    """Re-merging an artifact's own table must not be mistaken for a collision."""
    from flydsl.expr.rocdl import ktrace_emit

    monkeypatch.setattr(ktrace_emit, "_NAMES", {})
    monkeypatch.setattr(ktrace_emit, "_ID_OWNER", {})

    ktrace_emit.merge_names({"mainloop": 1, "tick": 2})
    ktrace_emit.merge_names({"mainloop": 1, "tick": 2})  # same artifact, second launch
    assert ktrace_emit.name_table() == {"mainloop": 1, "tick": 2}


def test_a_failing_reset_does_not_mask_the_overflow(monkeypatch):
    """The overflow diagnosis must survive a failing buffer re-arm.

    Regression: buf.reset() ran unguarded in a finally, so a hipMemset failure replaced
    the actionable "claimed N slots, buffer holds M, re-run with BUFFER_BYTES>=..." with a
    bare HIP error code. The two are correlated -- an overflow means the device just wrote
    out of bounds, exactly when the memset is likely to fail -- so the swap happens
    precisely when the diagnosis matters most.
    """
    from flydsl.expr.rocdl import ktrace_emit

    class _Buffer:
        def read_records(self):
            raise RuntimeError("trace overflowed: waves claimed 999 slots, buffer holds 127")

        def reset(self):
            raise RuntimeError("hipMemset(trace buffer) failed with HIP error 4")

    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", None)
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: _Buffer())

    with pytest.raises(RuntimeError, match="trace overflowed"):
        ktrace.summary()


def test_pass_assigned_names_reach_the_decoder(monkeypatch):
    """The pass's ids must land in the emitter's table, not just on the artifact.

    Regression: _capture_ktrace_names preferred the pass's fly_ktrace.event_names and
    returned early, but collect() and summary() decode against ktrace_emit.name_table().
    Under the dialect path the frontend never allocates, so that table stayed empty on a
    fresh compile and a full trace decoded against {} -- every record's id unknown, so
    summarize drops it and the timeline renders empty. Nothing raises.
    """
    from flydsl.expr.rocdl import ktrace_emit

    monkeypatch.setattr(ktrace_emit, "_NAMES", {})
    monkeypatch.setattr(ktrace_emit, "_ID_OWNER", {})

    class _Artifact:
        pass

    artifact = _Artifact()
    with ir.Context():
        module = ir.Module.parse(
            "module attributes {fly_ktrace.event_names = {mainloop = 1 : i32, epilogue = 2 : i32}} {}"
        )
        jit_function._capture_ktrace_names(artifact, True, module)

    expected = {"mainloop": 1, "epilogue": 2}
    assert artifact._ktrace_names == expected
    assert ktrace_emit.name_table() == expected, "the decoder's table did not get the pass's ids"


def test_tracing_a_non_cdna_target_fails_with_a_ktrace_diagnostic():
    """Asking to trace on RDNA must say so, not fail deep in LLVM translation.

    The frontend emits annotations whatever the target -- it does not know the arch --
    so the refusal is the pass's job. Gating the pass out on non-CDNA instead would
    leave the ops in the module to reach LLVM translation and fail there with
    "cannot be converted to LLVM IR", which names neither ktrace nor the reason.
    """
    source = """
    module attributes {gpu.container_module} {
      gpu.module @kernels [#rocdl.target<chip = "gfx1201">] {
        gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
          fly_ktrace.mark "x"
          gpu.return
        }
      }
    }
    """
    with ir.Context():
        module = ir.Module.parse(source)
        with pytest.raises(Exception) as excinfo:
            PassManager.parse("builtin.module(convert-fly-ktrace-to-rocdl)").run(module.operation)
    message = str(excinfo.value)
    assert "ktrace" in message, f"the diagnostic does not mention ktrace: {message[:200]}"
    assert "gfx1201" in message, f"the diagnostic does not name the target: {message[:200]}"
