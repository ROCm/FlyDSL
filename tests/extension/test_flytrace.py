# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import json

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension import flytrace
from flydsl.extension._flytrace_backend import get_trace_backend
from flydsl.extension._flytrace_export import export_perfetto_per_kernel, perfetto_events
from flydsl.runtime.device import get_rocm_arch


@flyc.kernel(known_block_size=[64, 1, 1])
def _dynamic_trace_kernel(iterations: fx.Int32):
    flytrace.range_push("kernel")
    flytrace.boundary("loop")
    i = fx.Int32(0)
    while i < iterations:
        flytrace.boundary("iteration", i)
        if i % fx.Int32(2) == fx.Int32(0):
            flytrace.mark("even", i)
        i = i + fx.Int32(1)
    flytrace.boundary("done")
    flytrace.end()
    token = flytrace.range_start("finalize", iterations)
    flytrace.range_end(token, iterations)
    flytrace.range_pop()


@flyc.jit
def _dynamic_trace_launch(iterations: fx.Int32, grid: fx.Int32, stream: fx.Stream):
    _dynamic_trace_kernel(iterations).launch(grid=(grid, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[64, 1, 1])
def _static_trace_kernel():
    flytrace.mark("event")


@flyc.jit
def _runtime_grid_static_trace_launch(grid: fx.Int32, stream: fx.Stream):
    _static_trace_kernel().launch(grid=(grid, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[32, 2, 1])
def _two_dimensional_trace_kernel():
    flytrace.mark("two_dimensional")


@flyc.jit
def _two_dimensional_trace_launch(stream: fx.Stream):
    _two_dimensional_trace_kernel().launch(grid=(1, 1, 1), block=(32, 2, 1), stream=stream)


@flyc.kernel(known_block_size=[32, 1, 1])
def _partial_wave_trace_kernel():
    flytrace.mark("partial_wave")


@flyc.jit
def _partial_wave_trace_launch(stream: fx.Stream):
    _partial_wave_trace_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[64, 1, 1])
def _second_static_trace_kernel():
    flytrace.mark("second_event")


@flyc.jit
def _multi_kernel_trace_launch(stream: fx.Stream):
    _static_trace_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    _second_static_trace_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def test_capture_option_validation():
    assert flytrace.capture(block=(1, 2, 3)).options.version == "flytrace-v4"
    assert flytrace.capture(block=(1, 2, 3)).options.blocks == ((1, 2, 3),)
    assert flytrace.capture(block=[(1, 0, 0), (2, 0, 0)]).options[2] == ((1, 0, 0), (2, 0, 0))
    with pytest.raises(ValueError, match="duplicates"):
        flytrace.capture(block=[(1, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="mode"):
        flytrace.capture(mode="unknown")
    with pytest.raises(ValueError, match="max_events"):
        flytrace.capture(max_events=0)
    with pytest.raises(TypeError, match="iterable"):
        flytrace.capture(exclude="event")
    with pytest.raises(TypeError, match="bool"):
        flytrace.capture(hardware=1)
    with pytest.raises(TypeError, match="per_kernel"):
        flytrace.capture(per_kernel=1)


def test_backend_errors_are_reported_at_the_trace_boundary():
    with pytest.raises(ValueError, match="no flytrace backend"):
        get_trace_backend("unregistered-test-target")
    with pytest.raises(TypeError, match="name must be a string"):
        get_trace_backend(1)

    class DummyBackend(flytrace.TraceBackend):
        name = "unit-test-target"
        clock_hz = 1

        def lower_kernel(self, func, ctx, grid, block, stream):
            pass

        def current_device(self):
            return 0

        def synchronize(self, device):
            pass

        def allocate_buffer(self, words, device):
            return [0] * words

        def buffer_pointer(self, buffer):
            return 0

        def buffer_words(self, buffer):
            return buffer

        def decode(self, spec, words):
            return []

        def raw_metadata(self, specs):
            return {"backend": self.name, "clock_hz": self.clock_hz}

    flytrace.register_backend("unit-test-target", DummyBackend)
    assert isinstance(get_trace_backend("unit-test-target"), DummyBackend)


def test_capture_replaces_repeated_launches_and_enforces_total_allocation(monkeypatch):
    import flydsl.extension._flytrace as implementation

    class RecordingBackend(flytrace.TraceBackend):
        name = "recording-test-target"
        clock_hz = 1

        def __init__(self):
            self.allocations = 0
            self.clears = 0
            self.graph_capturing = False

        def lower_kernel(self, func, ctx, grid, block, stream):
            pass

        def current_device(self):
            return 0

        def synchronize(self, device):
            pass

        def is_current_stream_capturing(self):
            return self.graph_capturing

        def allocate_buffer(self, words, device):
            self.allocations += 1
            return {"words": words, "launch": None, "pointer": self.allocations}

        def clear_buffer(self, buffer):
            self.clears += 1
            buffer["launch"] = None

        def buffer_pointer(self, buffer):
            return buffer["pointer"]

        def buffer_words(self, buffer):
            return [buffer["launch"]]

        def decode(self, spec, words):
            return [(spec["name"], words[0])]

        def raw_metadata(self, specs):
            return {"backend": self.name, "clock_hz": self.clock_hz}

    backend = RecordingBackend()
    monkeypatch.setattr(implementation, "get_trace_backend", lambda: backend)
    spec = {"backend": backend.name, "name": "same-compiled-launch", "words": 1}
    cap = implementation.capture()
    with cap:
        assert cap._buffer({"words": 0}) == 0
        empty_call = implementation.TraceCallState(lambda args: args, {"options": cap.options, "words": 0})
        assert empty_call(("unannotated",)) == ("unannotated",)
        first = cap._buffer(spec)
        cap._records[id(spec)][1]["launch"] = 1
        second = cap._buffer(spec)
        cap._records[id(spec)][1]["launch"] = 2
        backend.graph_capturing = True
        graph_pointer = cap._buffer(spec)
        backend.graph_capturing = False
    assert first != second
    assert graph_pointer == second
    assert backend.allocations == 2
    assert cap.decode() == [("same-compiled-launch", 2)]
    with cap:
        third = cap._buffer(spec)
        cap._records[id(spec)][1]["launch"] = 3
    assert third == second
    assert backend.allocations == 2
    assert backend.clears == 1
    assert cap.decode() == [("same-compiled-launch", 3)]

    oversized = implementation.capture()
    with oversized:
        limit = {"backend": backend.name, "name": "limit", "words": 128 * 1024**2}
        oversized._buffer(limit)
        oversized._buffer(limit)
        with pytest.raises(ValueError, match="512 MiB"):
            oversized._buffer(spec)

    unprepared = implementation.capture()
    with unprepared:
        backend.graph_capturing = True
        with pytest.raises(RuntimeError, match="run this traced specialization once"):
            unprepared._buffer(spec)
        backend.graph_capturing = False

    backend.graph_capturing = True
    with pytest.raises(RuntimeError, match="enter flytrace.capture.*before device graph capture"):
        with implementation.capture():
            pass
    backend.graph_capturing = False


def test_generic_export_supports_both_range_models():
    events = [
        {"name": "outer", "kind": "push", "payload": None, "tick": 100, "ordinal": 0},
        {"name": "load", "kind": "range_start", "range_id": 7, "payload": 3, "tick": 110, "ordinal": 1},
        {"name": "ready", "kind": "mark", "payload": None, "tick": 120, "ordinal": 2},
        {"name": "load", "kind": "range_end", "range_id": 7, "payload": 4, "tick": 130, "ordinal": 3},
        {"name": "__pop", "kind": "pop", "payload": None, "tick": 140, "ordinal": 4},
    ]
    waves = [
        {
            "kernel": "generic_operator",
            "block": (0, 0, 0),
            "wave": 0,
            "epoch": 90,
            "end_tick": 150,
            "events": events,
            "attempted_events": len(events),
            "overflow": False,
        }
    ]
    trace = perfetto_events(waves, 100_000_000)
    slices = {event["name"]: event for event in trace if event.get("ph") == "X"}
    assert set(slices) == {"load", "outer"}
    assert slices["load"]["args"] == {"payload": 3, "end_payload": 4}


def test_generic_export_pairs_crossing_same_name_ranges_by_token():
    events = [
        {"name": "tile", "kind": "range_start", "range_id": 10, "payload": 0, "tick": 100},
        {"name": "tile", "kind": "range_start", "range_id": 11, "payload": 1, "tick": 110},
        {"name": "tile", "kind": "range_end", "range_id": 10, "payload": 2, "tick": 130},
        {"name": "tile", "kind": "range_end", "range_id": 11, "payload": 3, "tick": 150},
    ]
    waves = [
        {
            "kernel": "token_ranges",
            "block": (0, 0, 0),
            "wave": 0,
            "epoch": 90,
            "end_tick": 160,
            "events": events,
            "attempted_events": len(events),
            "overflow": False,
        }
    ]
    slices = [event for event in perfetto_events(waves, 100_000_000) if event.get("ph") == "X"]
    assert [(event["args"]["payload"], event["args"]["end_payload"], event["dur"]) for event in slices] == [
        (0, 2, 0.3),
        (1, 3, 0.4),
    ]


def test_per_kernel_export_uses_one_capture_wide_origin(tmp_path):
    def wave(kernel, epoch, tick, end_tick):
        return {
            "kernel": kernel,
            "block": (0, 0, 0),
            "wave": 0,
            "epoch": epoch,
            "end_tick": end_tick,
            "events": [{"name": "event", "kind": "mark", "payload": None, "tick": tick}],
            "attempted_events": 1,
            "overflow": False,
        }

    manifest = export_perfetto_per_kernel(
        [
            {"recording": 0, "kernel": "first/kernel", "waves": [wave("first/kernel", 90, 100, 110)]},
            {"recording": 0, "kernel": "second kernel", "waves": [wave("second kernel", 190, 200, 210)]},
        ],
        tmp_path,
        100_000_000,
    )
    assert manifest["kernels"] == 2
    assert manifest["waves"] == 2
    assert [entry["file"] for entry in manifest["files"]] == [
        "000-first_kernel.json",
        "001-second_kernel.json",
    ]
    timestamps = []
    for entry in manifest["files"]:
        trace = json.loads((tmp_path / entry["file"]).read_text())
        timestamps.append(next(event["ts"] for event in trace["traceEvents"] if event.get("ph") == "i"))
    assert timestamps == [0.1, 1.1]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_static_all_grid_rejects_runtime_launch_dimensions():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with pytest.raises(ValueError, match="static.*block=None.*static launch dimensions"):
        with flytrace.capture(block=None, mode="static", max_blocks=4):
            flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(2), torch.cuda.current_stream())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_capture_profiles_kernels_with_every_event_excluded(tmp_path):
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with flytrace.capture(mode="static", exclude=("event",)) as cap:
        flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(1), torch.cuda.current_stream())
    waves = cap.decode()
    assert len(waves) == 1
    assert waves[0]["events"] == []
    path = tmp_path / "kernel-envelope.json"
    assert cap.export(path) == {"waves": 1, "records": 0, "dropped": 0}
    slices = [event for event in json.loads(path.read_text())["traceEvents"] if event.get("ph") == "X"]
    assert len(slices) == 1
    assert slices[0]["cat"] == "flytrace.kernel"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_static_runtime_grid_accepts_selected_blocks():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    selected = [(0, 0, 0), (1, 0, 0)]
    with flytrace.capture(block=selected, mode="static") as cap:
        flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(2), torch.cuda.current_stream())
    waves = cap.decode()
    assert {tuple(wave["block"]) for wave in waves} == set(selected)
    assert [[event["name"] for event in wave["events"]] for wave in waves] == [["event"], ["event"]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_trace_supports_multidimensional_blocks():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with flytrace.capture(mode="static") as cap:
        flyc.compile(_two_dimensional_trace_launch, torch.cuda.current_stream())
    waves = cap.decode()
    assert len(waves) == 1
    assert waves[0]["block"] == (0, 0, 0)
    assert [event["name"] for event in waves[0]["events"]] == ["two_dimensional"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_trace_supports_partial_waves():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with flytrace.capture(mode="static") as cap:
        flyc.compile(_partial_wave_trace_launch, torch.cuda.current_stream())
    waves = cap.decode()
    assert len(waves) == 1
    assert [event["name"] for event in waves[0]["events"]] == ["partial_wave"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_capture_exports_multiple_kernels_combined_or_separately(tmp_path):
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    directory = tmp_path / "per-kernel"
    with flytrace.capture(directory, mode="static", per_kernel=True) as cap:
        flyc.compile(_multi_kernel_trace_launch, torch.cuda.current_stream())

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kernels"] == 2
    assert manifest["waves"] == 2
    assert manifest["records"] == 2
    assert {entry["records"] for entry in manifest["files"]} == {1}
    assert all((directory / entry["file"]).is_file() for entry in manifest["files"])

    combined = tmp_path / "combined.json"
    assert cap.export(combined, per_kernel=False) == {"waves": 2, "records": 2, "dropped": 0}
    names = {
        event["args"]["name"].split(" Block ", 1)[0]
        for event in json.loads(combined.read_text())["traceEvents"]
        if event.get("name") == "thread_name"
    }
    assert len(names) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_capture_supports_prepared_multi_kernel_cuda_graph_replay():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    stream = torch.cuda.Stream()
    with flytrace.capture(mode="static") as cap:
        compiled = flyc.compile(_multi_kernel_trace_launch, stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compiled(stream)

        # Clear the eager preparation result so only graph replay can make the
        # following decode succeed through the captured stable buffer address.
        for _, buffer in cap._records.values():
            buffer.zero_()
        torch.cuda.synchronize()
        graph.replay()
        assert cap.decode()[0]["epoch"] > 0

        # An eager launch after graph construction must reuse rather than free
        # the graph-owned address; a later replay must remain valid.
        compiled(stream)
        stream.synchronize()
        for _, buffer in cap._records.values():
            buffer.zero_()
        torch.cuda.synchronize()
        graph.replay()
        waves = cap.decode()

    assert len(waves) == 2
    assert all(wave["epoch"] > 0 for wave in waves)
    assert [[event["name"] for event in wave["events"]] for wave in waves] == [
        ["event"],
        ["second_event"],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_dynamic_kernel_envelope_supports_cuda_graph_replay():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    stream = torch.cuda.Stream()
    with flytrace.capture(
        block=None,
        exclude=("event",),
        mode="dynamic",
        max_blocks=4,
        max_events=32,
    ) as cap:
        compiled = flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(1), stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compiled(fx.Int32(1), stream)

        for _, buffer in cap._records.values():
            buffer.zero_()
        torch.cuda.synchronize()
        graph.replay()
        waves = cap.decode()

    assert len(waves) == 1
    assert waves[0]["epoch"] > 0
    assert waves[0]["events"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_dynamic_loop_runtime_grid_and_nondefault_stream(tmp_path):
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    stream = torch.cuda.Stream()
    with flytrace.capture(block=None, mode="auto", max_blocks=4, max_events=32) as cap:
        flyc.compile(_dynamic_trace_launch, fx.Int32(4), fx.Int32(2), stream)

    waves = cap.decode()
    assert len(waves) == 2
    assert {tuple(wave["block"]) for wave in waves} == {(0, 0, 0), (1, 0, 0)}
    for wave in waves:
        assert wave["overflow"] is False
        assert [event["name"] for event in wave["events"]] == [
            "kernel",
            "loop",
            "iteration",
            "even",
            "iteration",
            "iteration",
            "even",
            "iteration",
            "done",
            "__end",
            "finalize",
            "finalize",
            "__pop",
        ]
        assert [event["payload"] for event in wave["events"] if event["name"] == "iteration"] == [0, 1, 2, 3]

    path = tmp_path / "dynamic.json"
    stats = cap.export(path)
    assert stats == {"waves": 2, "records": 26, "dropped": 0}
    assert json.loads(path.read_text())["traceEvents"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_dynamic_overflow_is_reported(tmp_path):
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with flytrace.capture(mode="dynamic", max_events=4) as cap:
        flyc.compile(_dynamic_trace_launch, fx.Int32(4), fx.Int32(1), torch.cuda.current_stream())
    wave = cap.decode()[0]
    assert wave["overflow"] is True
    assert wave["attempted_events"] == 13
    stats = cap.export(tmp_path / "overflow.json")
    assert stats == {"waves": 1, "records": 4, "dropped": 9}
