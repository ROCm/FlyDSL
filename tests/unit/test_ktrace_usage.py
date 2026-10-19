# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import _rocm_ktrace_trace as ktrace_trace
from flydsl.expr import ktrace
from flydsl.runtime.device import get_rocm_arch

pytestmark = [
    pytest.mark.l1b_target_dialect,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not (get_rocm_arch() or "").startswith(("gfx942", "gfx950")),
        reason="ktrace requires gfx942 or gfx950",
    ),
]


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
