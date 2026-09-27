# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import json

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension import flytrace
from flydsl.runtime.device import get_rocm_arch


@flyc.kernel(known_block_size=[64, 1, 1])
def _dynamic_trace_kernel(iterations: fx.Int32):
    flytrace.push("kernel")
    flytrace.boundary("loop")
    i = fx.Int32(0)
    while i < iterations:
        flytrace.boundary("iteration", i)
        if i % fx.Int32(2) == fx.Int32(0):
            flytrace.mark("even", i)
        i = i + fx.Int32(1)
    flytrace.boundary("done")
    flytrace.end()
    flytrace.pop()


@flyc.jit
def _dynamic_trace_launch(iterations: fx.Int32, grid: fx.Int32, stream: fx.Stream):
    _dynamic_trace_kernel(iterations).launch(grid=(grid, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[64, 1, 1])
def _static_trace_kernel():
    flytrace.mark("event")


@flyc.jit
def _runtime_grid_static_trace_launch(grid: fx.Int32, stream: fx.Stream):
    _static_trace_kernel().launch(grid=(grid, 1, 1), block=(64, 1, 1), stream=stream)


def test_capture_option_validation():
    assert flytrace.capture(block=(1, 2, 3)).options[2] == ((1, 2, 3),)
    assert flytrace.capture(block=[(1, 0, 0), (2, 0, 0)]).options[2] == ((1, 0, 0), (2, 0, 0))
    with pytest.raises(ValueError, match="duplicates"):
        flytrace.capture(block=[(1, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="mode"):
        flytrace.capture(mode="unknown")
    with pytest.raises(ValueError, match="max_events"):
        flytrace.capture(max_events=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_static_all_grid_rejects_runtime_launch_dimensions():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with pytest.raises(ValueError, match="static.*block=None.*static launch dimensions"):
        with flytrace.capture(block=None, mode="static", max_blocks=4):
            flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(2), torch.cuda.current_stream())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_static_runtime_grid_accepts_selected_ctas():
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    selected = [(0, 0, 0), (1, 0, 0)]
    with flytrace.capture(block=selected, mode="static") as cap:
        flyc.compile(_runtime_grid_static_trace_launch, fx.Int32(2), torch.cuda.current_stream())
    waves = cap.decode()
    assert {tuple(wave["block"]) for wave in waves} == set(selected)
    assert [[event["name"] for event in wave["events"]] for wave in waves] == [["event"], ["event"]]


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
            "__pop",
        ]
        assert [event["payload"] for event in wave["events"] if event["name"] == "iteration"] == [0, 1, 2, 3]

    path = tmp_path / "dynamic.json"
    stats = cap.export(path)
    assert stats == {"waves": 2, "records": 22, "dropped": 0}
    assert json.loads(path.read_text())["traceEvents"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an AMD GPU")
def test_dynamic_overflow_is_reported(tmp_path):
    if get_rocm_arch().split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("flytrace recorder currently targets gfx942/gfx950")
    with flytrace.capture(mode="dynamic", max_events=4) as cap:
        flyc.compile(_dynamic_trace_launch, fx.Int32(4), fx.Int32(1), torch.cuda.current_stream())
    wave = cap.decode()[0]
    assert wave["overflow"] is True
    assert wave["attempted_events"] == 11
    stats = cap.export(tmp_path / "overflow.json")
    assert stats == {"waves": 1, "records": 4, "dropped": 7}
