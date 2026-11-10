# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import _rocm_ktrace_trace as ktrace_trace
from flydsl.expr import ktrace

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.fixture(autouse=True)
def _requires_cdna(rocm_arch):
    """Skip device tests when no launch-capable CDNA target is available."""
    if not rocm_arch.startswith(("gfx942", "gfx950")):
        pytest.skip(f"ktrace requires gfx942 or gfx950, got {rocm_arch}")


@flyc.kernel
def _add_kernel(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor):
    i = fx.thread_idx.x
    with ktrace.range("add"):
        out[i] = a[i] + b[i]


@flyc.jit
def _add(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _add_kernel(a, b, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def _accumulate_kernel(a: fx.Tensor, out: fx.Tensor):
    row = fx.block_idx.x
    col = fx.thread_idx.x
    with ktrace.range("accumulate"):
        acc = fx.Float32(0)
        for k in fx.range(0, 4, 1):
            acc = acc + a[row, k * 64 + col]
        out[row, col] = acc


@flyc.jit
def _accumulate(a: fx.Tensor, out: fx.Tensor):
    _accumulate_kernel(a, out).launch(grid=(2, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _matmul_kernel(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor):
    i = fx.thread_idx.x
    row = i // 8
    col = i % 8
    with ktrace.range("matmul"):
        with ktrace.range("gemm.prologue"):
            acc = fx.Float32(0)
        with ktrace.range("gemm.mainloop"):
            for k in fx.range(0, 8, 1):
                acc = acc + a[row, k] * b[k, col]
        with ktrace.range("gemm.epilogue"):
            out[row, col] = acc


@flyc.kernel
def _moe_kernel(x: fx.Tensor, weights: fx.Tensor, expert_ids: fx.Tensor, out: fx.Tensor):
    i = fx.thread_idx.x
    token = i // 8
    col = i % 8
    with ktrace.range("moe.prologue"):
        expert = expert_ids[token]
        acc = fx.Float32(0)
    with ktrace.range("moe.mainloop"):
        for k in fx.range(0, 8, 1):
            acc = acc + x[token, k] * weights[expert, k, col]
    with ktrace.range("moe.epilogue"):
        out[token, col] = acc


@flyc.jit
def _moe(
    x: fx.Tensor,
    weights: fx.Tensor,
    expert_ids: fx.Tensor,
    out: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    _moe_kernel(x, weights, expert_ids, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def _attention_kernel(q: fx.Tensor, k: fx.Tensor, v: fx.Tensor, out: fx.Tensor, causal: fx.Int32):
    i = fx.thread_idx.x
    row = i // 8
    col = i % 8
    with ktrace.range("attn.prologue"):
        query = q[row]
        numer = fx.Float32(0)
        denom = fx.Float32(0)
        key_range = ktrace.inactive_token("attn.key")
    with ktrace.range("attn.mainloop"):
        for key in fx.range(0, 8, 1):
            ktrace.range_stop(key_range)
            key_range = ktrace.range_start("attn.key", key)
            weight = fx.exp(query * k[key])
            if causal != 0 and key > row:
                weight = fx.Float32(0)
            numer = numer + weight * v[key, col]
            denom = denom + weight
        ktrace.range_stop(key_range)
    with ktrace.range("attn.epilogue"):
        out[row, col] = numer / denom


@flyc.jit
def _attention(
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    out: fx.Tensor,
    causal: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    _attention_kernel(q, k, v, out, causal).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _matmul(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _matmul_kernel(a, b, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def _grid_kernel():
    with ktrace.range("body"):
        ktrace.mark("tick")


@flyc.jit
def _grid(blocks: fx.Constexpr[int]):
    _grid_kernel().launch(grid=(blocks, 1, 1), block=(256, 1, 1))


@flyc.jit
def _grid_rows(cols: fx.Constexpr[int], rows: fx.Constexpr[int]):
    _grid_kernel().launch(grid=(cols, rows, 1), block=(256, 1, 1))


@flyc.kernel
def _loop_kernel():
    for i in fx.range(0, 8, 1):
        ktrace.mark("iter", i)


@flyc.jit
def _loop():
    _loop_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _divergent_loop_kernel(out: fx.Tensor):
    i = fx.thread_idx.x
    for _ in fx.range(0, i % 4 + 1, 1):
        with ktrace.range("lap"):
            out[i] = out[i] + fx.Float32(1)
    ktrace.mark("after")


@flyc.jit
def _divergent_loop(out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _divergent_loop_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def _divergent_kernel():
    lane = fx.thread_idx.x % fx.Int32(64)
    if lane >= fx.Int32(32):
        ktrace.mark("upper")
    ktrace.mark("after")


@flyc.jit
def _divergent():
    _divergent_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _both_sides_kernel():
    lane = fx.thread_idx.x % fx.Int32(64)
    if lane >= fx.Int32(32):
        ktrace.mark("upper")
    else:
        ktrace.mark("lower")
    ktrace.mark("after")


@flyc.jit
def _both_sides():
    _both_sides_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _loop_in_branch_kernel():
    lane = fx.thread_idx.x % fx.Int32(64)
    if lane >= fx.Int32(32):
        for i in fx.range(0, 3, 1):
            ktrace.mark("upper", i)
    else:
        ktrace.mark("lower")
    ktrace.mark("after")


@flyc.jit
def _loop_in_branch():
    _loop_in_branch_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _disjoint_ifs_kernel():
    lane = fx.thread_idx.x % fx.Int32(64)
    if lane >= fx.Int32(32):
        ktrace.mark("upper")
    if lane < fx.Int32(32):
        ktrace.mark("lower")
    ktrace.mark("after")


@flyc.jit
def _disjoint_ifs():
    _disjoint_ifs_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _branch_in_loop_kernel():
    lane = fx.thread_idx.x % fx.Int32(64)
    for i in fx.range(0, 3, 1):
        if lane >= fx.Int32(32):
            ktrace.mark("upper", i)
    if lane < fx.Int32(32):
        ktrace.mark("lower")


@flyc.jit
def _branch_in_loop():
    _branch_in_loop_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _pointer_if_kernel(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor):
    i = fx.thread_idx.x
    p = fx.get_iter(a)
    if i % fx.Int32(64) >= fx.Int32(32):
        ktrace.mark("upper")
        p = fx.get_iter(a) + i
    else:
        ktrace.mark("lower")
        p = fx.get_iter(b) + i
    out[i] = p[0]
    ktrace.mark("after")


@flyc.jit
def _pointer_if(a: fx.Tensor, b: fx.Tensor, out: fx.Tensor):
    _pointer_if_kernel(a, b, out).launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _uneven_trips_kernel(lower_trips: fx.Int32, upper_trips: fx.Int32):
    lane = fx.thread_idx.x % fx.Int32(64)
    trips = lower_trips + (upper_trips - lower_trips) * (lane // fx.Int32(32))
    stage = ktrace.inactive_token("stage")
    for _ in fx.range(0, trips, 1):
        ktrace.range_stop(stage)
        stage = ktrace.range_start("stage")
    ktrace.range_stop(stage)


@flyc.jit
def _uneven_trips(lower_trips: fx.Int32, upper_trips: fx.Int32):
    _uneven_trips_kernel(lower_trips, upper_trips).launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _inactive_kernel():
    ktrace.range_stop(ktrace.inactive_token("never"))
    ktrace.mark("only")


@flyc.jit
def _inactive():
    _inactive_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


@flyc.kernel
def _long_kernel():
    for _ in fx.range(0, 300, 1):
        ktrace.mark("spin")


@flyc.jit
def _long():
    _long_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_capture_three_kernel_patterns(tmp_path):
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    added = torch.empty_like(a)
    tiled = torch.randn(2, 256, device="cuda")
    accumulated = torch.empty(2, 64, device="cuda")
    lhs = torch.randn(8, 8, device="cuda")
    rhs = torch.randn(8, 8, device="cuda")
    product = torch.empty_like(lhs)

    _add(a, b, added)
    with ktrace.capture("three_kernels", blocks=(0, 0, 0), out_dir=tmp_path) as trace:
        _add(a, b, added)
        _accumulate(tiled, accumulated)
        _matmul(lhs, rhs, product)

    torch.testing.assert_close(added, a + b)
    torch.testing.assert_close(accumulated, tiled.view(2, 4, 64).sum(dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(product, lhs @ rhs, atol=1e-5, rtol=1e-5)
    stats = trace.summary()
    assert stats is not None
    assert {"add", "accumulate", "matmul", "gemm.prologue", "gemm.mainloop", "gemm.epilogue"} <= stats["phases"].keys()
    assert trace.export() is not None


@pytest.mark.parametrize("causal", [0, 1])
def test_moe_and_attention_control_flow(causal):
    x = torch.randn(8, 8, device="cuda")
    weights = torch.randn(2, 8, 8, device="cuda")
    expert_ids = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    moe_out = torch.empty_like(x)
    q = torch.randn(8, device="cuda")
    k = torch.randn(8, device="cuda")
    v = torch.randn(8, 8, device="cuda")
    attn_out = torch.empty_like(v)

    with ktrace.capture("moe_attention", blocks=(0, 0, 0)) as trace:
        _moe(x, weights, expert_ids, moe_out)
        _attention(q, k, v, attn_out, fx.Int32(causal))

    reference_moe = torch.einsum("tk,tkc->tc", x, weights[expert_ids.long()])
    scores = q[:, None] * k[None, :]
    if causal:
        scores = scores.masked_fill(torch.ones(8, 8, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    reference_attn = scores.softmax(dim=1) @ v
    torch.testing.assert_close(moe_out, reference_moe, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(attn_out, reference_attn, atol=1e-5, rtol=1e-5)
    stats = trace.summary()
    assert stats is not None
    phases = stats["phases"]
    assert {
        "moe.prologue",
        "moe.mainloop",
        "moe.epilogue",
        "attn.prologue",
        "attn.mainloop",
        "attn.key",
        "attn.epilogue",
    } <= phases.keys()
    assert phases["attn.key"]["count"] == 8


def test_graph_replay_snapshots_accumulate_until_read():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    out = torch.empty_like(a)

    with ktrace.capture("graph_add", blocks=(0, 0, 0)) as trace:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _add(a, b, out, fx.Stream(stream))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _add(a, b, out, fx.Stream(stream))
        trace.clear()
        graph.replay()
        first = trace.summary()
        assert first is not None and first["phases"]["add"]["count"] == 1
        graph.replay()

    second = trace.summary()
    assert second is not None and second["phases"]["add"]["count"] == 2, "a mid-capture summary dropped records"
    graph.replay()
    trace.read()
    third = trace.summary()
    assert third is not None and third["phases"]["add"]["count"] == 1
    assert ktrace.summary() is not None
    torch.testing.assert_close(out, a + b)


def test_filtered_graph_replay_drops_cached_records():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    out = torch.empty_like(a)

    with ktrace.capture("filtered", blocks=(99, 0, 0)):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _add(a, b, out, fx.Stream(stream))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _add(a, b, out, fx.Stream(stream))

    with ktrace.capture("unfiltered", blocks=(0, 0, 0)) as trace:
        _add(a, b, out)
    assert trace.summary() is not None

    graph.replay()
    assert ktrace.summary() is None
    torch.testing.assert_close(out, a + b)


def test_capture_snapshot_stays_with_its_window():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    out = torch.empty_like(a)
    tiled = torch.randn(2, 256, device="cuda")
    accumulated = torch.empty(2, 64, device="cuda")

    with ktrace.capture("first") as first:
        _add(a, b, out)
        first.read()
        assert set(first.summary()["phases"]) == {"add"}

    with ktrace.capture("second") as second:
        _accumulate(tiled, accumulated)

    assert set(first.summary()["phases"]) == {"add"}
    assert set(second.summary()["phases"]) == {"accumulate"}


def test_another_capture_cannot_consume_active_window():
    other = ktrace.capture("other")
    with ktrace.capture("owner"):
        with pytest.raises(RuntimeError, match="another capture's active trace buffer"):
            other.clear()
        with pytest.raises(RuntimeError, match="another capture's active trace buffer"):
            other.read()


def test_multiple_eager_layers_export_separate_traces(tmp_path):
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    outputs = [torch.empty_like(a) for _ in range(3)]

    with ktrace.capture("layers", blocks=(0, 0, 0), out_dir=tmp_path) as trace:
        for output in outputs:
            _add(a, b, output)

    for output in outputs:
        torch.testing.assert_close(output, a + b)
    launches = trace.launches()
    assert len(launches) == 3
    assert len({launch["dispatch_id"] for launch in launches}) == 3
    assert all(launch["phases"]["add"]["count"] == 1 for launch in launches)
    paths = trace.export_launches(names=["layer_0", "layer_1", "layer_2"])
    assert len(paths) == 3
    assert all(path.startswith(str(tmp_path)) for path in paths)
    assert [Path(path).name for path in paths] == [f"layer_{i}_ktrace.pftrace.gz" for i in range(3)]
    assert all(record["payload"] is None for record in trace._snapshot())
    with pytest.raises(ValueError, match="one string per recorded launch"):
        trace.export_launches(names=["only_one"])


def test_graph_replay_of_multiple_layers_keeps_each_launch(tmp_path):
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    outputs = [torch.empty_like(a) for _ in range(3)]

    with ktrace.capture("graph_layers", blocks=(0, 0, 0), out_dir=tmp_path) as trace:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        fly_stream = fx.Stream(stream)
        with torch.cuda.stream(stream):
            _add(a, b, outputs[0], fly_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for output in outputs:
                    _add(a, b, output, fly_stream)
        trace.clear()
        graph.replay()
        graph.replay()

    for output in outputs:
        torch.testing.assert_close(output, a + b)
    launches = trace.launches()
    assert len(launches) == 6
    assert len({launch["dispatch_id"] for launch in launches}) == 6
    assert all(launch["phases"]["add"]["count"] == 1 for launch in launches)
    assert len(trace.export_launches()) == 6
    assert trace.export().endswith(".pftrace.gz")
    timeline = ktrace_trace.build_timeline(trace._snapshot(), trace._names)
    assert len([track for track in timeline["tracks"] if track["name"].startswith("launch ")]) == 6


def test_graph_replay_with_gemm_moe_and_attention():
    lhs = torch.randn(8, 8, device="cuda")
    rhs = torch.randn(8, 8, device="cuda")
    product = torch.empty_like(lhs)
    x = torch.randn(8, 8, device="cuda")
    weights = torch.randn(2, 8, 8, device="cuda")
    expert_ids = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    moe_out = torch.empty_like(x)
    q = torch.randn(8, device="cuda")
    k = torch.randn(8, device="cuda")
    v = torch.randn(8, 8, device="cuda")
    attn_out = torch.empty_like(v)

    with ktrace.capture("graph_mix", blocks=(0, 0, 0)) as trace:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        fly_stream = fx.Stream(stream)
        with torch.cuda.stream(stream):
            _matmul(lhs, rhs, product, fly_stream)
            _moe(x, weights, expert_ids, moe_out, fly_stream)
            _attention(q, k, v, attn_out, fx.Int32(1), fly_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _matmul(lhs, rhs, product, fly_stream)
                _moe(x, weights, expert_ids, moe_out, fly_stream)
                _attention(q, k, v, attn_out, fx.Int32(1), fly_stream)
        trace.clear()
        graph.replay()

    torch.testing.assert_close(product, lhs @ rhs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(moe_out, torch.einsum("tk,tkc->tc", x, weights[expert_ids.long()]), atol=1e-5, rtol=1e-5)
    scores = (q[:, None] * k[None, :]).masked_fill(
        torch.ones(8, 8, device="cuda", dtype=torch.bool).triu(1), float("-inf")
    )
    torch.testing.assert_close(attn_out, scores.softmax(dim=1) @ v, atol=1e-5, rtol=1e-5)
    phases = trace.summary()["phases"]
    assert {
        "gemm.prologue",
        "gemm.mainloop",
        "gemm.epilogue",
        "moe.prologue",
        "moe.mainloop",
        "moe.epilogue",
        "attn.prologue",
        "attn.mainloop",
        "attn.key",
        "attn.epilogue",
    } <= phases.keys()


def test_every_wave_of_a_grid_is_recorded():
    with ktrace.capture("grid") as trace:
        _grid(1024)
    records = trace._snapshot()
    waves = {(r["block"], r["wave_id"]) for r in records}
    assert len(waves) == 4096, f"{4096 - len(waves)} of 4096 waves left no records"
    assert len(records) == 3 * 4096
    assert len({r["slot"] for r in records}) == len(records), "two records share a slot"
    assert trace.summary()["phases"]["body"]["count"] == 4096


@pytest.mark.parametrize("two_d", [False, True], ids=["1-D", "2-D"])
def test_one_launch_fills_every_partition(two_d):
    from flydsl.expr.rocdl import ktrace_emit

    with ktrace.capture("warmup"):
        _grid(1)
    layout = ktrace_emit.current_buffer().layout
    per_partition = min(layout.capacity(p) for p in range(layout.partitions))
    blocks = per_partition * layout.partitions // 4
    # A row of partitions // 4 four-wave blocks gives every partition one wave.
    cols, rows = (layout.partitions // 4, per_partition) if two_d else (blocks, 1)
    with ktrace.capture("full") as trace:
        _grid_rows(cols, rows)
    waves = {(r["block"], r["wave_id"]) for r in trace._snapshot()}
    assert len(waves) == 4 * cols * rows, f"{4 * cols * rows - len(waves)} waves left no records"


def test_loop_iterations_take_distinct_slots():
    with ktrace.capture("loop") as trace:
        _loop()
    records = sorted(trace._snapshot(), key=lambda r: r["slot"])
    assert [r["payload"] for r in records] == list(range(8))
    assert len({r["slot"] for r in records}) == 8


def test_divergent_loop_records_every_iteration():
    out = torch.zeros(64, device="cuda")
    with ktrace.capture("divergent_loop") as trace:
        _divergent_loop(out)
    phases = trace.summary()["phases"]
    assert phases["lap"]["count"] == 4, "the wave runs the loop as often as its longest lane"
    assert phases["after"]["count"] == 1, "the event after the loop must not reuse a loop slot"
    torch.testing.assert_close(out, (torch.arange(64, device="cuda") % 4 + 1).float())


def test_a_lane_divergent_mark_keeps_its_slot():
    with ktrace.capture("divergent") as trace:
        _divergent()
    records = sorted(trace._snapshot(), key=lambda r: r["slot"])
    names = {v: k for k, v in trace._names.items()}
    assert [names[r["event_id"]] for r in records] == ["upper", "after"], "the mark after the branch took its slot"


@pytest.mark.parametrize(
    "launch, expected",
    [
        (_both_sides, ["upper", "lower", "after"]),
        (_loop_in_branch, ["upper", "upper", "upper", "lower", "after"]),
        (_disjoint_ifs, ["upper", "lower", "after"]),
        (_branch_in_loop, ["upper", "upper", "upper", "lower"]),
    ],
    ids=["both_sides", "loop_in_branch", "disjoint_ifs", "branch_in_loop"],
)
def test_lanes_that_record_apart_keep_their_records(launch, expected):
    with ktrace.capture("apart") as trace:
        launch()
    records = sorted(trace._snapshot(), key=lambda r: r["slot"])
    names = {v: k for k, v in trace._names.items()}
    assert [names[r["event_id"]] for r in records] == expected, "one group of lanes overwrote the other's records"


def test_pointer_valued_branches_keep_both_records():
    a = torch.full((64,), 1.0, device="cuda")
    b = torch.full((64,), 2.0, device="cuda")
    out = torch.zeros(64, device="cuda")
    with ktrace.capture("pointer_if") as trace:
        _pointer_if(a, b, out)
    records = sorted(trace._snapshot(), key=lambda r: r["slot"])
    names = {v: k for k, v in trace._names.items()}
    order = [names[r["event_id"]] for r in records]
    assert order == ["upper", "lower", "after"], "one branch overwrote the other's record"
    torch.testing.assert_close(out, torch.where(torch.arange(64, device="cuda") >= 32, a, b))


@pytest.mark.parametrize("lower_trips, upper_trips", [(0, 1), (1, 0), (1, 3), (3, 1)])
def test_lanes_that_leave_a_loop_apart_close_every_range_once(lower_trips, upper_trips):
    with ktrace.capture("uneven") as trace:
        _uneven_trips(lower_trips, upper_trips)
    records = trace._snapshot()
    starts = sorted(r["slot"] for r in records if r["kind"] == ktrace.KIND_RANGE_START)
    stops = sorted(r["start_slot"] for r in records if r["kind"] == ktrace.KIND_RANGE_STOP)
    assert len(starts) == max(lower_trips, upper_trips)
    assert stops == starts, "a range was closed twice or left open"


def test_an_inactive_token_stop_takes_no_slot():
    from flydsl.expr.rocdl import ktrace_emit, ktrace_format

    with ktrace.capture("inactive") as trace:
        _inactive()
    records = trace._snapshot()
    assert [r["kind"] for r in records] == [ktrace.KIND_MARK]
    layout = ktrace_emit.current_buffer().layout
    offset = (records[0]["slot"] - layout.header_slots) % ktrace_format.EVENTS_PER_WAVE
    assert offset == 1, "the stop of an inactive token used a slot"


def test_repeated_small_launches_spread_over_partitions():
    from flydsl.expr.rocdl import ktrace_emit, ktrace_format

    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    out = torch.empty_like(a)
    with ktrace.capture("repeated") as trace:
        for _ in range(64):
            _add(a, b, out)
    layout = ktrace_emit.current_buffer().layout
    regions = {(r["slot"] - layout.header_slots) // ktrace_format.EVENTS_PER_WAVE for r in trace._snapshot()}
    partitions = {region % layout.partitions for region in regions}
    assert len(partitions) >= 16, f"64 one-wave launches used only {len(partitions)} of {layout.partitions} partitions"


def test_a_wave_past_its_event_limit_is_reported():
    with pytest.raises(RuntimeError, match="trace truncated: 1 wave"):
        with ktrace.capture("long"):
            _long()
