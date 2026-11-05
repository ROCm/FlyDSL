# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


import gzip
import os
import re
import sys
import time
from unittest import mock

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function
from flydsl.compiler.backends import _rocm_ktrace_trace as ktrace_trace
from flydsl.expr import ktrace, ktrace_common
from flydsl.expr.rocdl import ktrace_format
from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

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
    text = re.sub(r"loc\([^)]*\)|#loc\d* = .*", "", text)
    return re.sub(r"@?(launch_)?(annotated|plain)(_\d+)?", "K", text)


def test_slot_reservation_atomic_is_hoisted_to_the_prologue(monkeypatch):
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
    assert text.count("atomicrmw") == 2, "only the launch count and the wave's claim may be atomic, not an event"
    assert text.rindex("atomicrmw") < text.index("s.memrealtime"), "an atomic was emitted after the prologue"
    assert text.count("llvm.mlir.global") == 0


def test_disabled_build_is_identical_to_uninstrumented(monkeypatch):
    _frontend_only(monkeypatch, enable=False)

    @flyc.kernel
    def annotated():
        ktrace.mark("start")
        with ktrace.range("scoped"):
            ktrace.mark("inside")
        ktrace.range_push("body")
        ktrace.range_pop()
        token = ktrace.range_start("tok")
        ktrace.range_stop(token)
        ktrace.range_stop(ktrace.inactive_token("seed"))

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

    assert _normalize(_canonicalized(annotated_ir)) == _normalize(_canonicalized(_ir(launch_plain)))


@pytest.mark.parametrize("enable", [True, False], ids=["on", "off"])
def test_inactive_token_survives_scf_for(monkeypatch, enable):
    _frontend_only(monkeypatch, enable=enable)

    @flyc.kernel
    def annotated():
        token = ktrace.inactive_token("k_tile")
        for _ in fx.range(0, 8, 1):
            ktrace.range_stop(token)
            token = ktrace.range_start("k_tile")
        ktrace.range_stop(token)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    match = re.search(r"scf\.for[^\n]*", _ir(launch))
    assert match, "expected an scf.for"
    assert "iter_args" in match.group(0), "token was not loop-carried"


def test_prologue_is_emitted_at_kernel_entry(monkeypatch):
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        for _ in fx.range(0, 4, 1):
            ktrace.mark("in_loop")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    head = text.split("scf.for", 1)[0]
    assert "atomicrmw" in head, "prologue reservation was not emitted before the loop"


def test_block_filter_guards_the_store(monkeypatch):
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
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        token = ktrace.range_start("r")
        ktrace.mark("bad", token)
        ktrace.range_stop(token)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    with pytest.raises(Exception, match="payload|single value|RangeToken"):
        launch()


@pytest.mark.l0_backend_agnostic
def test_ranges_pair_by_start_slot_not_by_name():
    names = {"k_tile": 1}
    wave = {"hw_id": 0x1234, "xcc_id": 0, "wave": 4, "simd": 3, "cu": 2, "sh": 0, "se": 1, "payload": 0}
    records = [
        {**wave, "ts": 100, "event_id": 1, "kind": ktrace.KIND_RANGE_START, "slot": 0, "start_slot": 0},
        {**wave, "ts": 150, "event_id": 1, "kind": ktrace.KIND_RANGE_STOP, "slot": 1, "start_slot": 0},
        {**wave, "ts": 200, "event_id": 1, "kind": ktrace.KIND_RANGE_START, "slot": 2, "start_slot": 0},
        {**wave, "ts": 260, "event_id": 1, "kind": ktrace.KIND_RANGE_STOP, "slot": 3, "start_slot": 2},
    ]

    summary = ktrace_trace.summarize(records, names)
    assert summary["phases"]["k_tile"]["count"] == 2
    assert summary["phases"]["k_tile"]["total_ns"] == pytest.approx(1100.0)


@pytest.mark.l0_backend_agnostic
def test_records_are_grouped_by_wave_before_ordering():
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
    assert summary["phases"]["phase"]["total_ns"] == pytest.approx(1100.0)


@pytest.mark.l0_backend_agnostic
def test_timeline_nests_each_wave_under_its_cu_and_workgroup():
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
            "block": (4, 1, 0),
            "wave_id": 3,
        }
    ]
    timeline = ktrace_trace.build_timeline(records, names, kernel="gemm")
    tracks = {track["uuid"]: track for track in timeline["tracks"]}
    wave = timeline["instants"][0]["track"]
    path = []
    while wave is not None:
        path.insert(0, tracks[wave]["name"])
        wave = tracks[wave]["parent"]
    assert path[0] == "gemm" and path[1].startswith("launch 1")
    assert path[2:] == ["XCD3/SE2/SH1/CU9", "WG(4, 1, 0)", "Wave3"]
    assert timeline["instants"][0]["name"] == "phase" and timeline["instants"][0]["payload"] == 7


def _wave_record(ts, kind, slot, start_slot=0, payload=None, payload_type=None):
    return {
        **dict.fromkeys(("hw_id", "xcc_id", "wave", "simd", "cu", "sh", "se", "wave_id"), 0),
        "ts": ts,
        "payload": payload,
        "payload_type": payload_type,
        "event_id": 1,
        "kind": kind,
        "slot": slot,
        "start_slot": start_slot,
        "block": (0, 0, 0),
    }


@pytest.mark.l0_backend_agnostic
def test_overlapping_token_ranges_never_share_a_lane():
    # B starts in the same tick A stops, but before A's stop record: they overlap.
    records = [
        _wave_record(100, ktrace.KIND_RANGE_START, 1),
        _wave_record(110, ktrace.KIND_RANGE_START, 2),
        _wave_record(110, ktrace.KIND_RANGE_STOP, 3, start_slot=1),
        _wave_record(200, ktrace.KIND_RANGE_STOP, 4, start_slot=2),
        _wave_record(300, ktrace.KIND_RANGE_START, 5),
        _wave_record(310, ktrace.KIND_RANGE_STOP, 6, start_slot=5),
    ]
    timeline = ktrace_trace.build_timeline(records, {"r": 1})
    lanes = {t["uuid"]: t["name"] for t in timeline["tracks"] if t["name"].startswith("ranges")}
    ranges = [(lanes[sl["track"]], sl["start"], sl["end"]) for sl in timeline["slices"] if sl["track"] in lanes]
    assert sorted(ranges) == [("ranges 0", 0, 100), ("ranges 0", 2000, 2100), ("ranges 1", 100, 1000)]


@pytest.mark.l0_backend_agnostic
def test_unclosed_ranges_are_labelled():
    records = [
        _wave_record(100, ktrace.KIND_RANGE_PUSH, 1),
        _wave_record(110, ktrace.KIND_RANGE_START, 2),
        _wave_record(120, ktrace.KIND_MARK, 3),
    ]
    timeline = ktrace_trace.build_timeline(records, {"r": 1})
    names = sorted(sl["name"] for sl in timeline["slices"])
    assert names == ["Wave Span", "r (unclosed)", "r (unclosed)"]


@pytest.mark.l0_backend_agnostic
def test_payload_annotation_keeps_non_finite_and_unsigned_values():
    assert b"nan" in ktrace_trace._payload_annotation(float("nan"))
    assert b"-inf" in ktrace_trace._payload_annotation(float("-inf"))
    # uint64 payloads use the unsigned field even below 2^63 (field 3, varint wire type).
    assert ktrace_trace._payload_annotation(5, "uint64").endswith(bytes([3 << 3, 5]))
    assert ktrace_trace._payload_annotation(5, "int64").endswith(bytes([4 << 3, 5]))


@pytest.mark.l0_backend_agnostic
def test_write_trace_emits_gzipped_perfetto_without_overwriting(tmp_path):
    records = [
        {
            **dict.fromkeys(("hw_id", "xcc_id", "slot", "start_slot", "wave", "simd", "cu", "sh", "se", "wave_id"), 0),
            "ts": 100,
            "payload": None,
            "event_id": 1,
            "kind": ktrace.KIND_MARK,
            "block": (0, 0, 0),
        }
    ]
    first = ktrace_trace.write_trace(records, {"a": 1}, kernel="k", out_dir=str(tmp_path))
    second = ktrace_trace.write_trace(records, {"a": 1}, kernel="k", out_dir=str(tmp_path))
    assert [os.path.basename(p) for p in (first, second)] == ["k_ktrace.pftrace.gz", "k_ktrace.1.pftrace.gz"]
    assert sorted(os.listdir(tmp_path)) == ["k_ktrace.1.pftrace.gz", "k_ktrace.pftrace.gz"]
    timeline = ktrace_trace.build_timeline(records, {"a": 1}, kernel="k")
    with gzip.open(first, "rb") as fh:
        assert fh.read() == ktrace_trace.encode_pftrace(timeline)


def test_traced_kernel_stays_disk_cacheable(monkeypatch):
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    artifact = launch._last_compiled[1]
    assert not artifact._post_load_processors
    assert not artifact._link_libs
    assert not artifact._uses_explicit_module


def test_trace_buffer_is_a_trailing_kernel_argument(monkeypatch):
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
    from flydsl.compiler.jit_function import _ensure_ktrace_buffer_arg

    _frontend_only(monkeypatch, enable=True)

    args = []
    assert _ensure_ktrace_buffer_arg(args) is True
    assert len(args) == 1
    assert not args[0].pointer

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
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        for _ in fx.range(0, 8, 1):
            ktrace.mark("per_iter")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    body = _expanded(launch).split("scf.for", 1)[1]
    assert re.search(
        r"llvm\.store .*!llvm\.ptr<5>", body
    ), "the loop does not advance the event count; iterations collide"


def test_decode_skips_each_waves_counter_word():
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    raw = struct.pack("<QQIIII", 8, 0, 0, 0, 0, 0)
    for i in range(3):
        raw += struct.pack("<QQIIII", 395099147398924 + i * 32, i, 1110450432, 0, (1 << 31) | 1, 0)

    recs, _ = ktrace_buffer.decode_records(raw)

    assert [r["slot"] for r in recs] == [2, 3, 4]
    assert [r["payload"] for r in recs] == [0, 1, 2]


def test_optional_payload_preserves_absent_and_explicit_zero():
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    raw = bytes(ktrace_format.RECORD_BYTES)
    raw += struct.pack("<QQIIII", 100, 0, 0, 0, 1, 0)
    raw += struct.pack("<QQIIII", 110, 0, 0, 0, (1 << 31) | 1, 0)
    records, _ = ktrace_buffer.decode_records(raw)
    assert [record["payload"] for record in records] == [None, 0]
    instants = ktrace_trace.build_timeline(records, {"mark": 1})["instants"]
    assert [instant["payload"] for instant in instants] == [None, 0]


def test_dispatch_ids_separate_repeated_kernel_launches():
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    def counter(queue_id, dispatch_id):
        return struct.pack("<IIQQII", 2, (5 << 8) | 3, dispatch_id, queue_id, 7, 6)

    def event(ts, kind):
        return struct.pack("<QQIIII", ts, 0, 0, 0, (kind << 24) | 1, 0)

    raw = counter(11, 101) + event(100, ktrace.KIND_RANGE_PUSH) + event(140, ktrace.KIND_RANGE_POP)
    raw += bytes((ktrace_format.EVENTS_PER_WAVE - 3) * ktrace_format.RECORD_BYTES)
    raw += counter(12, 101) + event(200, ktrace.KIND_RANGE_PUSH) + event(270, ktrace.KIND_RANGE_POP)
    records, overflowed = ktrace_buffer.decode_records(raw)

    assert not overflowed
    assert [(r["queue_id"], r["dispatch_id"]) for r in records] == [(11, 101)] * 2 + [(12, 101)] * 2
    assert all(r["block"] == (7, 6, 5) and r["wave_id"] == 3 for r in records)
    groups = ktrace_trace.group_launches(records)
    assert [identity for identity, _ in groups] == [(11, 101), (12, 101)]
    assert [ktrace_trace.summarize(group, {"phase": 1})["phases"]["phase"]["total_ns"] for _, group in groups] == [
        400.0,
        700.0,
    ]
    tracks = ktrace_trace.build_timeline(records, {"phase": 1})["tracks"]
    assert len([track for track in tracks if track["name"].startswith("launch ")]) == 2


def test_events_add_no_branches(monkeypatch):
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.mark("start", fx.Int32(7))
        with ktrace.range("scoped"):
            stage = ktrace.inactive_token("k_tile")
            for _ in fx.range(0, 8, 1):
                ktrace.range_stop(stage)
                stage = ktrace.range_start("k_tile")
            ktrace.range_stop(stage)

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert text.count("scf.if") == 2, "only the launch count and the claim may branch; an event added a branch"
    assert text.count('"s_buffer_store_dword') == 2 + 2 * 6 + 1, "each event is two straight-line scalar stores"
    assert text.count("llvm.amdgcn.s.dcache.wb") == 1, "the wave does not write its scalar stores back"


def test_reading_a_trace_re_arms_the_buffer(monkeypatch, tmp_path):
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

        def launch_count(self):
            return 0 if self.zeroed else 1

        def reset(self):
            self.zeroed += 1

    buf = _FakeBuffer()
    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", None)
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: buf)

    assert ktrace.summary() is not None
    assert buf.zeroed == 1, "summary() left the buffer armed; the next launch accumulates"

    assert ktrace.collect("k", out_dir=str(tmp_path)) is not None
    assert buf.reads == 1, "the device was re-read; the two calls describe different launches"


def test_reset_survives_an_overflow(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit

    class _OverflowingBuffer:
        def __init__(self):
            self.zeroed = 0

        def launch_count(self):
            return 1

        def read_records(self):
            raise RuntimeError("trace overflowed: waves claimed 999 slots")

        def reset(self):
            self.zeroed += 1

    buf = _OverflowingBuffer()
    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", [{"stale": True}])
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: buf)

    with pytest.raises(RuntimeError, match="trace overflowed"):
        ktrace.summary()
    assert buf.zeroed == 1, "an overflow left the buffer full; later calls inherit the error"


_CACHE_HIT_PROBE = """
import os, sys, json
os.environ["FLYDSL_KTRACE_ENABLE"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = sys.argv[1]

import flydsl.compiler as flyc
from flydsl.expr import ktrace, ktrace_common

@flyc.kernel
def annotated():
    ktrace.range_push("mainloop")
    ktrace.mark("tick")
    ktrace.range_pop()

@flyc.jit
def launch():
    annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

launch()
print("NAMES=" + json.dumps(ktrace_common.name_table()))
"""


@pytest.mark.l2_device
@_skip_not_cdna
def test_name_table_survives_a_disk_cache_hit(tmp_path):
    import json
    import subprocess

    probe = tmp_path / "probe.py"
    probe.write_text(_CACHE_HIT_PROBE)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def _run(tag):
        proc = subprocess.run(
            [sys.executable, str(probe), str(cache_dir)],
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


_HINT_ABI_PROBE = """
import os, sys
os.environ.pop("FLYDSL_KTRACE_ENABLE", None)
os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = sys.argv[1]
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
    import subprocess

    probe = tmp_path / "probe.py"
    probe.write_text(_HINT_ABI_PROBE)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    proc = subprocess.run(
        [sys.executable, str(probe), str(cache_dir)],
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
    assert (
        line[-1] == "PHASES=mainloop,tick"
    ), f"the hint compiled an instrumented kernel but the host did not collect its phases: {line[-1]}"


@pytest.mark.l2_device
@_skip_not_cdna
def test_capture_switches_off_on_off_without_recompiling_off(monkeypatch):
    monkeypatch.delenv("FLYDSL_KTRACE_ENABLE", raising=False)

    @flyc.kernel
    def annotated():
        ktrace.mark("tick")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    off = launch._last_compiled[1]
    assert not off._ktrace_traced

    with ktrace.capture("toggle", blocks=(0, 0, 0)) as trace:
        launch()
        traced = launch._last_compiled[1]
        assert traced is not off and traced._ktrace_traced

    assert "tick" in trace.summary()["phases"]
    trace.clear()
    launch()
    assert launch._last_compiled[1] is off
    assert len(launch._mem_cache) == 2
    assert trace.read().summary() is None


@pytest.mark.l2_device
@_skip_not_cdna
def test_environment_switches_off_on_off_without_recompiling_off(monkeypatch):
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")

    @flyc.kernel
    def annotated():
        ktrace.mark("tick")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    off = launch._last_compiled[1]

    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "1")
    launch()
    assert "tick" in ktrace.summary()["phases"]
    assert launch._last_compiled[1]._ktrace_traced

    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")
    snapshot = ktrace.capture("toggle")
    snapshot.clear()
    launch()
    assert launch._last_compiled[1] is off
    assert len(launch._mem_cache) == 2
    assert snapshot.read().summary() is None


def test_inactive_token_stop_is_dropped_without_a_branch(monkeypatch):
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.range_stop(ktrace.inactive_token("r"))

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert re.search(r"arith\.cmpi ne, %\S+, %\S+ : i32", text), "the stop does not test its token for the sentinel"
    assert "llvm.mlir.constant(8192 : i32)" in text, "an inactive stop is not sent past the wave's region"
    assert text.count("scf.if") == 2, "the inactive-token check became a branch"


def test_store_bounds_use_unsigned_compares(monkeypatch):
    _frontend_only(monkeypatch, enable=True)

    @flyc.kernel
    def annotated():
        ktrace.mark("a")

    @flyc.jit
    def launch():
        annotated().launch(grid=(1, 1, 1), block=(64, 1, 1))

    text = _expanded(launch)
    assert re.search(r"arith\.minui", text), "the per-wave event index is not clamped unsigned"
    assert not re.search(r"arith\.cmpi\s+s[lg][te]", text), "a signed bounds compare survived"


def test_range_stop_without_a_start_is_dropped():
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    packed = (ktrace.KIND_RANGE_STOP << 24) | 1
    raw = bytes(ktrace_format.RECORD_BYTES) + struct.pack("<QQIIII", 100, 0, 0, 0, packed, ktrace_format.UNPAIRED_SLOT)
    records, _ = ktrace_buffer.decode_records(raw)
    assert len(records) == 1, f"expected one decoded record, got {records}"
    assert records[0]["start_slot"] == ktrace_format.UNPAIRED_SLOT
    names = {"r": 1}

    assert ktrace_trace.summarize(records, names)["phases"] == {}, "an unpairable end was accumulated"

    slices = ktrace_trace.build_timeline(records, names, kernel="k")["slices"]
    assert [s["name"] for s in slices] == ["Wave Span"], "a dangling range-stop reached the trace"


def test_buffer_bytes_invalidates_the_cache(monkeypatch):
    baked = {
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
    from flydsl.expr.rocdl import ktrace_emit

    _frontend_only(monkeypatch, enable=True, buffer_bytes=4 << 20)
    small = ktrace_emit._buffer_capacity_slots()
    _frontend_only(monkeypatch, enable=True, buffer_bytes=256 << 20)
    large = ktrace_emit._buffer_capacity_slots()
    assert large > small, "buffer_bytes no longer determines the emitted capacity"


def test_payload_zero_is_recorded(monkeypatch):
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
    slices = ktrace_trace.build_timeline([rec], {"k_tile": 1}, kernel="k")["slices"]
    ranges = [s for s in slices if s["name"] == "k_tile (unclosed)"]
    assert ranges and ranges[0]["payload"] == 0, "a zero payload was dropped from the trace"


def test_collect_and_summary_agree_on_an_empty_trace(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit

    class _EmptyBuffer:
        def launch_count(self):
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
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[2] / "lib/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.cpp"
    if not src.exists():
        pytest.skip("C++ pass not present in this checkout")
    m = re.search(r"kEventsPerWave\s*=\s*(\d+)", src.read_text())
    assert m, "kEventsPerWave not found in the pass source"
    assert (
        int(m.group(1)) == ktrace_format.EVENTS_PER_WAVE
    ), f"pass has kEventsPerWave={m.group(1)} but Python has EVENTS_PER_WAVE={ktrace_format.EVENTS_PER_WAVE}"


def test_a_traced_launch_invalidates_cached_records(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit

    reads = []

    class _Buffer:
        def read_records(self):
            reads.append(1)
            return []

        def reset(self):
            pass

    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", [{"stale": True}])
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: _Buffer())

    from flydsl.compiler.jit_executor import CallState

    class _CallState(CallState):
        def _make_dispatch(self):
            return lambda _args: None

    _CallState([], func_exe=None, predispatch=lambda: None)(())
    assert ktrace.summary() is None, "a new launch was served the previous launch's records"
    assert reads, "the device was never re-read after the launch"


def test_untraced_launch_keeps_the_last_trace_snapshot(monkeypatch):
    from flydsl.compiler.jit_executor import CallState

    cached = [{"event": "last traced launch"}]
    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", cached)

    class _CallState(CallState):
        def _make_dispatch(self):
            return lambda _args: None

    with (
        mock.patch.object(
            ktrace, "invalidate_records", side_effect=AssertionError("untraced launch invalidated records")
        ),
        mock.patch.object(ktrace, "collection_lock", side_effect=AssertionError("untraced launch took trace lock")),
    ):
        _CallState([], func_exe=None)(())
    assert ktrace._CACHED_RECORDS is cached


def test_colliding_event_ids_are_rejected(monkeypatch):

    monkeypatch.setattr(ktrace_common, "_NAMES", {})
    monkeypatch.setattr(ktrace_common, "_ID_OWNER", {})

    ktrace_common.merge_names({"epilogue": 1})
    with pytest.raises(RuntimeError, match="claimed by both"):
        ktrace_common.merge_names({"mainloop": 1})


def test_merging_the_same_binding_twice_is_fine(monkeypatch):

    monkeypatch.setattr(ktrace_common, "_NAMES", {})
    monkeypatch.setattr(ktrace_common, "_ID_OWNER", {})

    ktrace_common.merge_names({"mainloop": 1, "tick": 2})
    ktrace_common.merge_names({"mainloop": 1, "tick": 2})
    assert ktrace_common.name_table() == {"mainloop": 1, "tick": 2}


def test_a_failing_reset_does_not_mask_the_overflow(monkeypatch):
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

    monkeypatch.setattr(ktrace_common, "_NAMES", {})
    monkeypatch.setattr(ktrace_common, "_ID_OWNER", {})

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
    assert ktrace_common.name_table() == expected, "the decoder's table did not get the pass's ids"


def test_tracing_a_non_cdna_target_fails_with_a_ktrace_diagnostic():
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


def test_event_ids_stay_unique_across_modules(monkeypatch):
    monkeypatch.setattr(ktrace_common, "_NAMES", {})
    monkeypatch.setattr(ktrace_common, "_ID_OWNER", {})

    def expand(source, known):
        with ir.Context():
            module = ir.Module.parse(source)
            if known:
                i32 = ir.IntegerType.get_signless(32)
                module.operation.attributes["fly_ktrace.known_event_names"] = ir.DictAttr.get(
                    {n: ir.IntegerAttr.get(i32, i) for n, i in known.items()}
                )
            PassManager.parse("builtin.module(convert-fly-ktrace-to-rocdl)").run(module.operation)
            return jit_function._ktrace_names_from_module(module)

    body = """
    module attributes {gpu.container_module} {
      gpu.module @kernels [#rocdl.target<chip = "gfx950">] {
        gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {
          fly_ktrace.mark "%A%"
          fly_ktrace.mark "%B%"
          gpu.return
        }
      }
    }
    """
    first = expand(body.replace("%A%", "shared").replace("%B%", "only_in_first"), {})
    second = expand(body.replace("%A%", "shared").replace("%B%", "only_in_second"), first)

    assert first["shared"] == second["shared"], "a name common to both modules was renumbered"
    assert set(first.values()).isdisjoint(
        {second["only_in_second"]}
    ), "the second module reused an id the first had already handed out"


def test_compiled_capacity_tracks_the_allocated_buffer(monkeypatch):
    from flydsl.compiler import jit_function
    from flydsl.expr.rocdl import ktrace_emit

    class _Buffer:
        capacity_slots = 511

    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: _Buffer())
    monkeypatch.setattr(ktrace_emit, "_buffer_capacity_slots", lambda: 4095)

    with ir.Context():
        module = ir.Module.parse("module {}")
        jit_function._attach_ktrace_config(module, True)
        baked = int(str(module.operation.attributes["fly_ktrace.capacity_slots"]).split(":")[0])

    assert baked == 511, f"compiled bound {baked} does not match the allocation (511)"


def test_a_trace_covers_every_launch_since_the_last_read(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit

    reads = []

    class _Buffer:
        def __init__(self):
            self.launches = 0

        def read_records(self):
            return [
                {
                    "ts": i + 1,
                    "payload": 0,
                    "hw_id": 0,
                    "xcc_id": 0,
                    "event_id": 1,
                    "kind": ktrace.KIND_MARK,
                    "start_slot": 0,
                    "slot": i + 1,
                    "wave": 0,
                    "simd": 0,
                    "cu": 0,
                    "sh": 0,
                    "se": 0,
                }
                for i in range(self.launches)
            ]

        def reset(self):
            reads.append(self.launches)
            self.launches = 0

    buf = _Buffer()
    monkeypatch.setattr(ktrace_emit, "current_buffer", lambda: buf)
    monkeypatch.setattr(ktrace, "_CACHED_RECORDS", None)

    buf.launches = 3
    records = ktrace._take_records()

    assert len(records) == 3, "a read must return every launch since the previous read"
    assert reads == [3], "the buffer is re-armed by the read, not by each launch"


def test_buffer_bytes_is_capped_where_the_offset_would_wrap():
    from flydsl.utils.env import KtraceEnvManager

    option = KtraceEnvManager.__dict__["buffer_bytes"]
    two_gib = 1 << 31
    assert option.validator(two_gib), "2 GiB is the largest offset that stays positive"
    assert not option.validator(two_gib + 1), "past 2 GiB the byte offset wraps negative"


def test_index_payload_is_accepted(monkeypatch):
    from flydsl._mlir.dialects import arith as _arith

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index_value = _arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 42)).result
            converted = ktrace_common.payload_operand(index_value)[0]
            assert str(converted.type) == "i64", f"index payload became {converted.type}, not i64"


def test_perfetto_range_closes_when_the_token_was_rebound():
    from flydsl.compiler.backends import _rocm_ktrace_trace as ktrace_trace

    def record(kind, event_id, slot, start_slot, ts):
        return {
            "ts": ts,
            "payload": 0,
            "hw_id": 0,
            "xcc_id": 0,
            "event_id": event_id,
            "kind": kind,
            "start_slot": start_slot,
            "slot": slot,
            "wave": 0,
            "simd": 0,
            "cu": 0,
            "sh": 0,
            "se": 0,
        }

    records = [
        record(ktrace.KIND_RANGE_START, 1, slot=5, start_slot=0, ts=100),
        record(ktrace.KIND_RANGE_STOP, 2, slot=6, start_slot=5, ts=200),
    ]
    slices = ktrace_trace.build_timeline(records, {"a": 1, "seed": 2})["slices"]
    ranges = [(s["name"], s["end"] - s["start"]) for s in slices if s["name"] != "Wave Span"]
    assert ranges == [("a", 1000)], f"the rebound token did not close its range: {ranges}"


def test_reading_a_buffer_from_another_device_is_refused(monkeypatch):
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    buffer = ktrace_buffer.TraceBuffer(device_ptr=0x1000, capacity_slots=511, device_id=0)

    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 1)
    with pytest.raises(RuntimeError, match="allocated on device 0"):
        buffer._require_own_device()

    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 0)
    buffer._require_own_device()


def test_ensure_buffer_refuses_a_foreign_device(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    buffer = ktrace_buffer.TraceBuffer(device_ptr=0x1000, capacity_slots=511, device_id=0)
    monkeypatch.setattr(ktrace_emit, "_BUFFER", buffer)

    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 0)
    assert ktrace_emit.ensure_buffer() is buffer, "the owning device must still be served"

    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 1)
    with pytest.raises(RuntimeError, match="allocated on device 0"):
        ktrace_emit.ensure_buffer()


def test_cached_binary_must_match_the_live_buffer(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    monkeypatch.setattr(
        ktrace_emit, "_BUFFER", ktrace_buffer.TraceBuffer(device_ptr=0x1000, capacity_slots=511, device_id=-1)
    )

    class _Artifact:
        _ktrace_traced = True
        _ktrace_buffer_config = 2047

    with pytest.raises(RuntimeError, match="compiled to bound its stores at 2047 slots"):
        ktrace.require_compatible_buffer(_Artifact())

    _Artifact._ktrace_buffer_config = 511
    ktrace.require_compatible_buffer(_Artifact())


def test_hot_dispatch_rechecks_the_buffer_device(monkeypatch):
    from flydsl.compiler.jit_executor import CallState
    from flydsl.expr.rocdl import ktrace_emit
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    monkeypatch.setattr(
        ktrace_emit, "_BUFFER", ktrace_buffer.TraceBuffer(device_ptr=0x1000, capacity_slots=511, device_id=0)
    )
    state = CallState([], func_exe=None, presets={}, predispatch=ktrace_emit.require_dispatchable_buffer)

    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 1)
    with pytest.raises(RuntimeError, match="allocated on device 0"):
        state(())


def test_capacity_check_allocates_rather_than_skipping(monkeypatch):
    from flydsl.expr.rocdl import ktrace_emit
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    monkeypatch.setattr(ktrace_emit, "_BUFFER", None)
    monkeypatch.setattr(
        ktrace_buffer,
        "allocate",
        lambda *a, **k: ktrace_buffer.TraceBuffer(device_ptr=0x2000, capacity_slots=511, device_id=0),
    )
    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 0)

    class _Artifact:
        _ktrace_traced = True
        _ktrace_buffer_config = 2047

    with pytest.raises(RuntimeError, match="compiled to bound its stores at 2047 slots"):
        ktrace.require_compatible_buffer(_Artifact())


def test_concurrent_first_launches_allocate_one_buffer(monkeypatch):
    import threading
    import time

    from flydsl.expr.rocdl import ktrace_emit
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    allocated = []

    def slow_allocate(*a, **k):
        time.sleep(0.05)
        buf = ktrace_buffer.TraceBuffer(device_ptr=0x3000 + len(allocated), capacity_slots=511, device_id=0)
        allocated.append(buf)
        return buf

    monkeypatch.setattr(ktrace_emit, "_BUFFER", None)
    monkeypatch.setattr(ktrace_buffer, "allocate", slow_allocate)
    monkeypatch.setattr(ktrace_buffer, "_current_device", lambda: 0)

    handed_out = []
    threads = [threading.Thread(target=lambda: handed_out.append(ktrace_emit.ensure_buffer())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(allocated) == 1, f"{len(allocated)} buffers allocated; all but one leak"
    assert len({id(b) for b in handed_out}) == 1, "threads were handed different buffers"


def test_expr_exposes_ktrace_lazily():
    import flydsl.expr as fx

    module = fx.ktrace
    assert module.__name__ == "flydsl.expr.ktrace"
    assert callable(module.range_push)
    assert not hasattr(module, "__all__"), "an __all__ would make ktrace a stable API"


def test_importing_the_runtime_does_not_pull_in_expr():
    import subprocess
    import sys

    probe = "import sys; import flydsl.runtime; print('flydsl.expr' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "importing flydsl.runtime now pulls in flydsl.expr"


@pytest.mark.parametrize("buffer_bytes", [64 << 10, 64 << 20], ids=["one_partition", "partitioned"])
def test_overflow_message_suggests_a_size_that_actually_works(buffer_bytes):
    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    buf = ktrace_buffer.TraceBuffer(
        device_ptr=0x1000, capacity_slots=buffer_bytes // ktrace_format.RECORD_BYTES - 1, device_id=-1
    )
    layout = buf.layout
    busiest = layout.capacity(0) + 1
    claimed = [busiest] + [1] * (layout.partitions - 1)
    with pytest.raises(RuntimeError, match="trace overflowed") as excinfo:
        with mock.patch.object(type(buf), "claimed_waves", lambda self: claimed):
            buf.read_records()

    suggested = int(re.search(r"FLYDSL_KTRACE_BUFFER_BYTES>=(\d+)", str(excinfo.value)).group(1))
    grown = ktrace_format.layout(suggested // ktrace_format.RECORD_BYTES - 1)
    # A larger buffer only splits partitions, so each must hold the busiest one's waves.
    assert grown.partitions >= layout.partitions
    assert grown.capacity(grown.partitions - 1) >= busiest, (
        f"suggested {suggested} bytes holds {grown.capacity(grown.partitions - 1)} waves per partition for "
        f"{busiest} -- the caller would hit the same error again"
    )


def test_layout_partitions_grow_with_the_buffer():
    previous = 0
    for shift in range(12, 32):
        total = (1 << shift) // ktrace_format.RECORD_BYTES
        layout = ktrace_format.layout(total - 1)
        assert layout.partitions & (layout.partitions - 1) == 0 and layout.partitions <= ktrace_format.MAX_PARTITIONS
        assert layout.partitions >= previous, "a larger buffer must never merge partitions"
        assert layout.header_slots == ktrace_format.CURSOR_STRIDE_SLOTS * (layout.partitions + 1)
        assert layout.header_slots + layout.wave_slots == total, "the regions do not end at the buffer's end"
        assert sum(layout.capacity(p) for p in range(layout.partitions)) == layout.regions
        assert layout.partitions == 1 or layout.capacity(layout.partitions - 1) >= ktrace_format.MIN_PARTITION_WAVES
        previous = layout.partitions
    assert ktrace_format.layout((64 << 20) // ktrace_format.RECORD_BYTES - 1).partitions == 64


def test_pass_layout_matches_the_decoder():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "lib/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.cpp"
    if not src.exists():
        pytest.skip("C++ pass not present in this checkout")
    text = src.read_text()
    for constant, expected in (
        ("kCursorStrideSlots", ktrace_format.CURSOR_STRIDE_SLOTS),
        ("kLaunchCountByte", ktrace_format.LAUNCH_COUNT_BYTE),
    ):
        m = re.search(rf"{constant}\s*=\s*(\d+)", text)
        assert m and int(m.group(1)) == expected, f"pass {constant} does not match the decoder's {expected}"

    capacity = (64 << 20) // ktrace_format.RECORD_BYTES - 1
    layout = ktrace_format.layout(capacity)
    source = f"""
    module attributes {{gpu.container_module, fly_ktrace.capacity_slots = {capacity} : i32,
                        fly_ktrace.partitions = {layout.partitions} : i32}} {{
      gpu.module @kernels [#rocdl.target<chip = "gfx950">] {{
        gpu.func @probe(%buf: !fly.ptr<i8, global>) kernel {{
          fly_ktrace.mark "x"
          gpu.return
        }}
      }}
    }}
    """
    with ir.Context():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-ktrace-to-rocdl)").run(module.operation)
        expanded = str(module)
    for value in (layout.header_slots, layout.wave_slots, layout.regions, layout.partitions):
        assert f"llvm.mlir.constant({value} : i32)" in expanded, f"the pass does not use the decoder's {value}"


def test_decode_flags_a_wave_that_ran_past_a_short_region():
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    header = struct.pack("<IIQQII", 9, 0, 0, 0, 0, 0)
    events = b"".join(struct.pack("<QQIIII", 100 + i, 0, 0, 0, 1, 0) for i in range(5))
    records, overflowed = ktrace_buffer.decode_records(header + events, first_slot=40)
    assert [r["slot"] for r in records] == [41, 42, 43, 44, 45]
    assert overflowed == [(40, 9, 5)], "a wave that issued 9 events into 5 slots was not flagged"


def test_read_records_decodes_each_partition(monkeypatch):
    import struct

    from flydsl.runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer

    capacity = (64 << 20) // ktrace_format.RECORD_BYTES - 1
    buf = ktrace_buffer.TraceBuffer(device_ptr=0, capacity_slots=capacity, device_id=-1)
    layout = buf.layout
    memory = bytearray((capacity + 1) * ktrace_format.RECORD_BYTES)

    def wave(partition, index, *events):
        start = layout.header_slots + (index * layout.partitions + partition) * ktrace_format.EVENTS_PER_WAVE
        struct.pack_into("<IIQQII", memory, start * ktrace_format.RECORD_BYTES, len(events), 0, 5, 9, partition, 0)
        for i, (kind, start_slot) in enumerate(events):
            struct.pack_into(
                "<QQIIII",
                memory,
                (start + 1 + i) * ktrace_format.RECORD_BYTES,
                100 + i,
                0,
                0,
                0,
                (kind << 24) | 1,
                start_slot,
            )
        return start

    first = wave(3, 0, (ktrace.KIND_MARK, 0))
    second = wave(3, 1, (ktrace.KIND_RANGE_START, 0))
    struct.pack_into(
        "<QQIIII",
        memory,
        (second + 2) * ktrace_format.RECORD_BYTES,
        300,
        0,
        0,
        0,
        (ktrace.KIND_RANGE_STOP << 24) | 1,
        second + 1,
    )
    struct.pack_into("<I", memory, second * ktrace_format.RECORD_BYTES, 2)
    last = wave(layout.partitions - 1, 0, (ktrace.KIND_MARK, 0))
    for partition, waves in ((3, 2), (layout.partitions - 1, 1)):
        struct.pack_into("<I", memory, layout.cursor_byte(partition), waves)

    monkeypatch.setattr(type(buf), "_read", lambda self, offset, nbytes, what: bytes(memory[offset : offset + nbytes]))
    records = buf.read_records()

    assert sorted(r["slot"] for r in records) == sorted([first + 1, second + 1, second + 2, last + 1])
    assert {r["block"][0] for r in records} == {3, layout.partitions - 1}
    phases = ktrace_trace.summarize(records, {"x": 1})["phases"]
    assert phases["x"]["count"] == 3, "the token range in partition 3 did not pair on its start slot"


def test_a_rejected_name_merge_leaves_the_table_untouched():

    ktrace_common._NAMES.clear()
    ktrace_common._ID_OWNER.clear()
    ktrace_common.merge_names({"phase": 1})
    before = ktrace_common.name_table()

    with pytest.raises(RuntimeError, match="claimed by both"):
        ktrace_common.merge_names({"second": 2, "zzz": 1})

    assert ktrace_common.name_table() == before, "a rejected merge rebound a name anyway"
    assert ktrace_common._ID_OWNER == {1: "phase"}, "a rejected merge left a stale id owner"


def test_a_cached_name_cannot_rebind_its_id():

    ktrace_common._NAMES.clear()
    ktrace_common._ID_OWNER.clear()
    ktrace_common.merge_names({"phase": 1})
    with pytest.raises(RuntimeError, match="has ids 1 and 2"):
        ktrace_common.merge_names({"phase": 2})
    assert ktrace_common.name_table() == {"phase": 1}
    assert ktrace_common._ID_OWNER == {1: "phase"}


def test_a_traced_launch_cannot_land_inside_a_trace_read():
    import threading

    from flydsl.compiler.jit_executor import CallState
    from flydsl.expr import ktrace as _kt
    from flydsl.expr.rocdl import ktrace_emit

    device = {"records": [{"ev": 1}]}
    reader_in_read = threading.Event()

    class _Buffer:
        capacity_slots = 511
        device_ptr = 0x1000
        device_id = -1

        def read_records(self):
            snapshot = list(device["records"])
            reader_in_read.set()
            time.sleep(0.15)
            return snapshot

        def launch_count(self):
            return 0

        def reset(self):
            device["records"] = []

        def _require_own_device(self):
            pass

    with mock.patch.object(ktrace_emit, "_BUFFER", _Buffer()), mock.patch.object(_kt, "_CACHED_RECORDS", None):
        collected = {}

        def launch():
            device["records"].append({"ev": 2})

        class _TracedCallState(CallState):
            def _make_dispatch(self):
                return lambda _args: launch()

        state = _TracedCallState([], func_exe=None, presets={}, predispatch=lambda: None)

        def reader():
            collected["records"] = _kt._take_records()

        def launcher():
            reader_in_read.wait()
            state(())

        threads = [threading.Thread(target=reader), threading.Thread(target=launcher)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        after = _kt._take_records()

    seen = [r for batch in (collected["records"] or [], after or []) for r in batch]
    assert {"ev": 2} in seen, "the launch's records were wiped by a concurrent trace read"


def test_a_trace_read_cannot_land_inside_a_traced_launch():
    import threading

    from flydsl.compiler.jit_executor import CallState
    from flydsl.expr import ktrace as _kt
    from flydsl.expr.rocdl import ktrace_emit

    device = {"records": []}
    launch_started = threading.Event()

    class _Buffer:
        capacity_slots = 511
        device_ptr = 0x1000
        device_id = -1

        def read_records(self):
            return list(device["records"])

        def launch_count(self):
            return 0

        def reset(self):
            device["records"] = []

        def _require_own_device(self):
            pass

    class _TracedCallState(CallState):
        def _make_dispatch(self):
            def _dispatch(_args):
                launch_started.set()
                time.sleep(0.15)
                device["records"].append({"ev": 9})

            return _dispatch

    with mock.patch.object(ktrace_emit, "_BUFFER", _Buffer()), mock.patch.object(_kt, "_CACHED_RECORDS", None):
        state = _TracedCallState([], func_exe=None, presets={}, predispatch=lambda: None)

        def launcher():
            state(())

        collected = {}

        def reader():
            launch_started.wait()
            collected["records"] = _kt._take_records()

        threads = [threading.Thread(target=launcher), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        after = _kt._take_records()

    seen = [r for batch in (collected.get("records") or [], after or []) for r in batch]
    assert {"ev": 9} in seen, "a concurrent trace read wiped records mid-launch"
