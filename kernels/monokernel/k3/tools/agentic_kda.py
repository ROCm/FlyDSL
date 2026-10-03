# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Benchmark Kimi-K3 TP8/DCP1 DSpark q8 Agentic KDA.

The harness compares the current native split (one KDA/AttnRes launch followed
by one eight-row staged MoE tail per request) with the complete one-launch
layer.  Every timed HIP graph contains 32 layer invocations.  Measurements use
eight graph warmups, at least fifty repeats, and the slowest TP rank for each
repeat before taking the median.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import socket
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

INVOCATIONS_PER_GRAPH = 32
WARMUPS = 8
MIN_REPEATS = 50
SUPPORTED_BATCHES = (1, 2, 4)
QUERY_LEN = 8

STARTING_POINTS_US = {
    1: {"baseline": 152.4895653128624, "full": 340.8207595348358},
    2: {"baseline": 261.4358067512512, "full": 603.6175489425659},
    4: {"baseline": 488.148957490921, "full": 1130.292534828186},
}

_STAGE_LABELS = (
    "pre_and_input",
    "recurrence_wait",
    "output_projection",
    "post_attn_res",
    "dense_projections",
    "expert_up",
    "expert_down",
    "routed_norm",
    "tail",
)


@dataclass(frozen=True)
class AgenticKdaShape:
    """Reference metadata contract for one graph-stable q8 bucket."""

    batch: int

    def __post_init__(self) -> None:
        if self.batch not in SUPPORTED_BATCHES:
            raise ValueError(f"batch must be one of {SUPPORTED_BATCHES}, got {self.batch}")

    @property
    def rows(self) -> int:
        return self.batch * QUERY_LEN

    @property
    def snapshot_shape(self) -> tuple[int, int]:
        return (self.batch, QUERY_LEN)

    def _validate_accepted(self, accepted: torch.Tensor) -> None:
        if (
            accepted.shape != (self.batch,)
            or accepted.dtype != torch.int32
            or not accepted.is_contiguous()
        ):
            raise ValueError(f"accepted must be contiguous int32 [{self.batch}]")
        if bool(torch.any((accepted < 1) | (accepted > QUERY_LEN))):
            raise ValueError("accepted counts must be in [1, 8]")

    def transition_slots(
        self,
        snapshots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            snapshots.shape != self.snapshot_shape
            or snapshots.dtype != torch.int32
            or not snapshots.is_contiguous()
        ):
            raise ValueError(
                "snapshots must be contiguous int32 " f"{list(self.snapshot_shape)}"
            )
        self._validate_accepted(accepted)
        columns = torch.arange(
            QUERY_LEN,
            dtype=torch.int64,
            device=snapshots.device,
        ).expand(self.batch, QUERY_LEN).clone()
        columns[:, 0] = accepted.to(torch.int64) - 1
        columns[:, 1:] -= 1
        return snapshots.gather(1, columns), snapshots

    def conv_window_plan(
        self,
        accepted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q8 convolution read rows and the rolled-back 10-row window."""

        self._validate_accepted(accepted)
        device = accepted.device
        count = accepted.to(torch.int64).view(-1, 1, 1)
        token = torch.arange(QUERY_LEN, dtype=torch.int64, device=device).view(
            1, QUERY_LEN, 1
        )
        history = torch.arange(3, dtype=torch.int64, device=device).view(1, 1, 3)
        logical = count - 1 + token + history
        first_draft = count + 2
        reads = torch.where(
            logical < first_draft,
            logical,
            10 + logical - first_draft,
        )
        committed = count.view(-1, 1) + torch.arange(
            2, dtype=torch.int64, device=device
        )
        draft = 10 + torch.arange(QUERY_LEN, dtype=torch.int64, device=device)
        final = torch.cat((committed, draft.expand(self.batch, QUERY_LEN)), dim=1)
        return reads, final


@dataclass
class _Fixture:
    prefix: torch.Tensor
    blocks: torch.Tensor
    snapshots: torch.Tensor
    accepted: torch.Tensor
    conv: torch.Tensor
    recurrent: torch.Tensor
    output: torch.Tensor


def critical_rank_medians(
    rank_samples: Sequence[Sequence[float]],
    *,
    layers: int,
) -> tuple[float, ...]:
    """Convert rank-major graph times into per-layer critical-rank samples."""

    if layers <= 0:
        raise ValueError("layers must be positive")
    if not rank_samples:
        raise ValueError("rank_samples must not be empty")
    repeat_count = len(rank_samples[0])
    if repeat_count == 0 or any(len(samples) != repeat_count for samples in rank_samples):
        raise ValueError("all ranks must provide the same nonzero repeat count")
    return tuple(
        max(float(rank_samples[rank][repeat]) for rank in range(len(rank_samples)))
        / layers
        for repeat in range(repeat_count)
    )


def critical_rank_kernel_profile(
    rank_profiles: Sequence[Sequence[dict]],
    *,
    layers: int,
) -> dict:
    """Select the median replay's exact critical-rank kernel/stage profile."""

    if not rank_profiles or layers <= 0:
        raise ValueError("rank profiles and layers must be positive")
    repeats = len(rank_profiles[0])
    if repeats == 0 or any(len(profiles) != repeats for profiles in rank_profiles):
        raise ValueError("all ranks must provide the same nonzero repeat count")
    critical_repeats = [
        max(
            (profiles[repeat] for profiles in rank_profiles),
            key=lambda profile: profile["graph_us"],
        )
        for repeat in range(repeats)
    ]
    critical = sorted(
        critical_repeats,
        key=lambda profile: profile["graph_us"],
    )[len(critical_repeats) // 2]
    kernels = [
        {
            "name": name,
            "calls_per_layer": values["calls"] / layers,
            "total_us_per_layer": values["total_us"] / layers,
            "mean_us": values["total_us"] / values["calls"],
        }
        for name, values in critical["kernels"].items()
        if values["calls"]
    ]
    kernels.sort(key=lambda kernel: kernel["total_us_per_layer"], reverse=True)
    return {
        "critical_rank": critical["rank"],
        "median_repeat": critical["repeat"],
        "profiled_graph_us_per_layer": critical["graph_us"] / layers,
        "kernels": kernels,
        "stages": critical["stages"],
    }


def graph_epoch_plan(layers: int) -> tuple[tuple[int, ...], int]:
    """Return the common per-layer tags and decode-step advance count."""

    if not 0 < layers <= 128:
        raise ValueError("layers must be in [1, 128]")
    return tuple(range(layers)), 1


def parity_metrics(
    *,
    expected_output: torch.Tensor,
    actual_output: torch.Tensor,
    expected_state: torch.Tensor,
    actual_state: torch.Tensor,
    expected_conv: torch.Tensor,
    actual_conv: torch.Tensor,
    rank_equal: bool,
) -> dict[str, float | bool]:
    """Compute parity facts from dedicated, untimed fixtures."""

    return {
        "output_max_abs": float(
            (expected_output.float() - actual_output.float()).abs().max()
        ),
        "state_max_abs": float(
            (expected_state.float() - actual_state.float()).abs().max()
        ),
        "conv_exact": bool(torch.equal(expected_conv, actual_conv)),
        "state_exact": bool(torch.equal(expected_state, actual_state)),
        "output_nonzero": bool(torch.count_nonzero(actual_output)),
        "rank_equal": bool(rank_equal),
    }


def q8_recurrence_reference(
    recurrent_state: torch.Tensor,
    snapshot_slots: torch.Tensor,
    accepted: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent ordered q8 delta-rule recurrence with FP16 publication."""

    batch, query_len, heads, width = query.shape
    if query_len != QUERY_LEN or snapshot_slots.shape != (batch, QUERY_LEN):
        raise ValueError("q8 recurrence requires snapshot slots [B,8]")
    persisted = recurrent_state.clone()
    output = torch.empty_like(query, dtype=torch.float32)
    carries = torch.empty(
        batch,
        heads,
        width,
        width,
        dtype=torch.float32,
        device=query.device,
    )
    q_scale = width**-0.5
    for request in range(batch):
        input_slot = int(snapshot_slots[request, int(accepted[request]) - 1])
        state = recurrent_state[input_slot].float()
        for token in range(QUERY_LEN):
            q = query[request, token].float()
            k = key[request, token].float()
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6)
            q = q * q_scale
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
            dt = torch.sigmoid(
                torch.exp(a_log.float())[:, None]
                * (gate[request, token].float() + dt_bias.float())
            )
            decay = torch.exp(-5.0 * dt)
            decayed = state * decay[:, None, :]
            state_key = torch.einsum("hvk,hk->hv", decayed, k)
            state_query = torch.einsum("hvk,hk->hv", decayed, q)
            new_value = (
                value[request, token].float() - state_key
            ) * torch.sigmoid(beta[request, token].float())[:, None]
            state = decayed + new_value[:, :, None] * k[:, None, :]
            output[request, token] = state_query + new_value * (k * q).sum(-1)[:, None]
            persisted[int(snapshot_slots[request, token])].copy_(state.to(torch.float16))
        carries[request].copy_(state)
    return output, persisted, carries


def q8_conv_reference(
    conv_state: torch.Tensor,
    snapshot_slots: torch.Tensor,
    accepted: torch.Tensor,
    drafts: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent causal width-4 convolution and 10-row rollback."""

    batch, query_len, _ = drafts.shape
    output = torch.empty_like(drafts, dtype=torch.float32)
    persisted = conv_state.clone()
    for request in range(batch):
        count = int(accepted[request])
        slot = int(snapshot_slots[request, 0])
        old = conv_state[slot].clone()
        for token in range(query_len):
            history = torch.cat(
                (old[count - 1 : count + 2], drafts[request, :token])
            )
            values = torch.cat(
                (history[-3:], drafts[request, token : token + 1])
            )
            convolved = (values.float() * weight.float().T).sum(0)
            output[request, token] = convolved * torch.sigmoid(convolved)
        persisted[slot, :2].copy_(old[count : count + 2])
        persisted[slot, 2:].copy_(drafts[request])
    return output, persisted


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=SUPPORTED_BATCHES)
    parser.add_argument("--layers", type=int, default=INVOCATIONS_PER_GRAPH)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--rank-skew-ms", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output")
    return parser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _slot_table(shape: AgenticKdaShape, slots: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(shape.rows, dtype=torch.int32, device=device) * 7 + 3
    ).remainder(slots).view(shape.snapshot_shape)


def _fixture(shape: AgenticKdaShape, device: torch.device, seed: int) -> _Fixture:
    from kernels.monokernel.config import KIMI_K3_CONFIG

    config = KIMI_K3_CONFIG
    slots = {1: 13, 2: 23, 4: 37}[shape.batch]
    snapshots = _slot_table(shape, slots, device)
    accepted = torch.tensor(
        {1: (2,), 2: (2, 7), 4: (2, 7, 1, 5)}[shape.batch],
        dtype=torch.int32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    prefix = torch.randn(
        shape.rows,
        config.hidden,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    blocks = torch.randn(
        shape.rows,
        1,
        config.hidden,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    conv = torch.linspace(
        -0.125,
        0.125,
        slots * 10 * 3 * config.local_heads * config.v_dim,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16).view(
        slots,
        10,
        3 * config.local_heads * config.v_dim,
    )
    recurrent = torch.empty(
        slots,
        config.local_heads,
        config.v_dim,
        config.v_dim,
        dtype=torch.float16,
        device=device,
    )
    for slot in range(slots):
        recurrent[slot].copy_(
            torch.linspace(
                0.75 + slot / 1024,
                1.25 + slot / 1024,
                recurrent[slot].numel(),
                dtype=torch.float32,
                device=device,
            ).view_as(recurrent[slot])
        )
    return _Fixture(
        prefix,
        blocks,
        snapshots,
        accepted,
        conv,
        recurrent,
        torch.empty_like(prefix),
    )


def _capture_layer_graph(call, advance, layers: int) -> torch.cuda.CUDAGraph:
    """Capture layer tags 0..N-1 and exactly one decode-step advance."""

    epoch_layers, advances = graph_epoch_plan(layers)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for layer in epoch_layers:
            call(layer)
        for _ in range(advances):
            advance()
    graph.instantiate()
    torch.cuda.synchronize()
    return graph


def _hip_graph_counts(graph: torch.cuda.CUDAGraph) -> dict[str, int]:
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGraphGetNodes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    hip.hipGraphGetNodes.restype = ctypes.c_int
    hip.hipGraphNodeGetType.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    hip.hipGraphNodeGetType.restype = ctypes.c_int
    counts: dict[str, int] = {}
    count = ctypes.c_size_t()
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    if hip.hipGraphGetNodes(handle, None, ctypes.byref(count)):
        raise RuntimeError("hipGraphGetNodes(count) failed")
    nodes = (ctypes.c_void_p * count.value)()
    if hip.hipGraphGetNodes(handle, nodes, ctypes.byref(count)):
        raise RuntimeError("hipGraphGetNodes(nodes) failed")
    names = {
        0: "kernel",
        1: "memcpy",
        2: "memset",
        3: "host",
        4: "child_graph",
        5: "empty",
        6: "wait_event",
        7: "event_record",
    }
    for node in nodes:
        node_type = ctypes.c_int()
        if hip.hipGraphNodeGetType(node, ctypes.byref(node_type)):
            raise RuntimeError("hipGraphNodeGetType failed")
        name = names.get(node_type.value, f"type_{node_type.value}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    layers: int,
    warmups: int,
    repeats: int,
    rank_skew_ms: float = 0,
) -> tuple[float, list[float]]:
    for _ in range(warmups):
        dist.barrier()
        if rank_skew_ms and dist.get_rank() == 0:
            time.sleep(rank_skew_ms / 1000.0)
        graph.replay()
    torch.cuda.synchronize()
    local = []
    for _ in range(repeats):
        dist.barrier()
        if rank_skew_ms and dist.get_rank() == 0:
            time.sleep(rank_skew_ms / 1000.0)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        local.append(start.elapsed_time(end) * 1000.0)
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    rank_samples = tuple(tuple(samples or ()) for samples in gathered)
    critical = list(critical_rank_medians(rank_samples, layers=layers))
    return statistics.median(critical), critical


def _profile_full(
    graph: torch.cuda.CUDAGraph,
    timeline: torch.Tensor,
    *,
    repeats: int,
    labels: Sequence[str] = _STAGE_LABELS,
) -> dict[str, float]:
    local_runs = []
    for _ in range(repeats):
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        ticks = timeline.cpu().tolist()[: len(labels) + 1]
        local_runs.append(
            {
                label: (ticks[index + 1] - ticks[index]) / 100.0
                for index, label in enumerate(labels)
            }
        )
    gathered: list[list[dict[str, float]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_runs)
    critical_runs = []
    for repeat in range(repeats):
        rank_runs = [runs[repeat] for runs in gathered if runs is not None]
        critical_runs.append(max(rank_runs, key=lambda run: sum(run.values())))
    critical_runs.sort(key=lambda run: sum(run.values()))
    return critical_runs[len(critical_runs) // 2]


def _profile_staged_graph(
    graph: torch.cuda.CUDAGraph,
    timeline: torch.Tensor,
    *,
    rank: int,
    layers: int,
    repeats: int,
    labels: Sequence[str],
) -> dict:
    """Profile each replay and return its exact median critical-rank record."""

    local_runs = []
    marker_prefix = "staged_graph_repeat_"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as profiler:
        for repeat in range(repeats):
            dist.barrier()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function(f"{marker_prefix}{repeat}"):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
            ticks = timeline.cpu().tolist()[: len(labels) + 1]
            local_runs.append(
                {
                    "rank": rank,
                    "repeat": repeat,
                    "graph_us": start.elapsed_time(end) * 1000.0,
                    "kernels": {},
                    "stages": {
                        label: (ticks[index + 1] - ticks[index]) / 100.0
                        for index, label in enumerate(labels)
                    },
                }
            )

    device_events = [
        event
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    repeat_ranges = {
        int(event.name.removeprefix(marker_prefix)): event.time_range
        for event in device_events
        if event.name.startswith(marker_prefix)
    }
    if len(repeat_ranges) != repeats:
        raise RuntimeError(
            f"profiler recorded {len(repeat_ranges)} of {repeats} repeat markers"
        )
    for run in local_runs:
        repeat_range = repeat_ranges[run["repeat"]]
        for event in device_events:
            if event.name.startswith(marker_prefix):
                continue
            if (
                event.time_range.start < repeat_range.start
                or event.time_range.end > repeat_range.end
            ):
                continue
            values = run["kernels"].setdefault(
                event.name,
                {"calls": 0, "total_us": 0.0},
            )
            values["calls"] += 1
            values["total_us"] += event.self_device_time_total
    gathered: list[list[dict] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_runs)
    return critical_rank_kernel_profile(
        [profiles for profiles in gathered if profiles is not None],
        layers=layers,
    )


def _check_observable_full_path(layer, shape: AgenticKdaShape) -> None:
    from kernels.monokernel.k3.kernel import monokernel_layout

    layout = monokernel_layout(
        shape.rows,
        fuse_attn_res=True,
        fuse_moe=True,
        mtp=True,
    )
    scratch = layer.attention.monokernel_scratch
    regions = (
        ("router", shape.rows * 896 * 4),
        ("shared_mid", shape.rows * 768 * 4),
        ("routed", shape.rows * 3584 * 2),
    )
    for name, size in regions:
        region = scratch[layout[name] : layout[name] + size]
        if int(torch.count_nonzero(region)) == 0:
            raise AssertionError(f"{name} path was not observable")


def _rank_equal(output: torch.Tensor) -> bool:
    """Compare CPU checksums through the control group; never send CUDA to Gloo."""

    checksum = torch.stack(
        (
            output.float().sum(),
            output.float().square().sum(),
            output.float().abs().sum(),
        )
    ).cpu()
    gathered: list[torch.Tensor | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, checksum)
    return all(
        peer is not None
        and torch.allclose(peer, checksum, atol=1e-2, rtol=1e-5)
        for peer in gathered
    )


def _deterministic_weights(device, rank):
    import torch

    from kernels.monokernel.config import (
        KIMI_K3_CONFIG,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )
    from kernels.monokernel.weights import LayerWeights

    config = KIMI_K3_CONFIG
    hidden = config.hidden
    routed = config.routed_hidden
    shared = config.shared_inter
    assert routed is not None and shared is not None
    projection = config.local_heads * config.v_dim
    fused = 4 * projection + config.local_heads + config.v_dim
    shard = hidden // 8

    def bf16(*shape, value=0):
        return torch.full(shape, value, dtype=torch.bfloat16, device=device)

    tensors = {
        "w_r": bf16(config.n_experts, hidden),
        "bias": torch.zeros(config.n_experts, dtype=torch.float32, device=device),
        "w_latent_down": bf16(routed, hidden),
        "g_latent": bf16(routed, value=1),
        "w_latent_up": bf16(shard, routed),
        "w_shared_ug": bf16(2 * shared, hidden),
        "w_shared_dn": bf16(hidden, shared),
        "w_kda_in": bf16(fused, hidden),
        "w_kda_fb": bf16(projection, config.v_dim),
        "w_kda_conv": bf16(3 * projection, 4),
        "kda_a_log": torch.zeros(
            config.local_heads,
            dtype=torch.float32,
            device=device,
        ),
        "kda_dt_bias": bf16(
            config.local_heads,
            config.v_dim,
            value=-10.375,
        ),
        "g_kda_out": bf16(config.v_dim, value=1),
        "w_kda_o": bf16(hidden, projection),
    }
    for name in (
        "g_self_res",
        "w_self_res",
        "g_in",
        "g_mlp_res",
        "w_mlp_res",
        "g_post",
    ):
        tensors[name] = bf16(hidden, value=1)

    rows = torch.arange(fused, device=device)
    tensors["w_kda_in"][rows, rows.remainder(16)] = (
        0.015625 + rows.remainder(7).to(torch.float32) / 1024
    ).to(torch.bfloat16)
    tensors["w_kda_conv"][:, 0] = 0.0625
    tensors["w_kda_conv"][:, 1] = 0.125
    tensors["w_kda_conv"][:, 2] = 0.25
    tensors["w_kda_conv"][:, 3] = 0.5
    gate_rows = torch.arange(projection, device=device)
    tensors["w_kda_fb"][gate_rows, gate_rows.remainder(config.v_dim)] = 0.03125
    output_rows = torch.arange(hidden, device=device)
    rank_scale = (rank + 1) / 1024
    tensors["w_kda_o"][
        output_rows, output_rows.remainder(projection)
    ] = rank_scale
    tensors["w_r"][0, :16] = torch.linspace(
        -0.125, 0.125, 16, dtype=torch.bfloat16, device=device
    )
    latent_rows = torch.arange(routed, device=device)
    tensors["w_latent_down"][latent_rows, latent_rows.remainder(16)] = 0.03125
    shared_rows = torch.arange(2 * shared, device=device)
    tensors["w_shared_ug"][shared_rows, shared_rows.remainder(16)] = 0.03125
    tensors["w_shared_dn"][
        output_rows, output_rows.remainder(shared)
    ] = rank_scale
    shard_rows = torch.arange(shard, device=device)
    tensors["w_latent_up"][
        shard_rows, shard_rows.remainder(routed)
    ] = rank_scale

    experts = config.n_experts
    ug_rows = experts * 2 * config.inter
    dn_rows = experts * routed
    tensors["w_ug"] = torch.zeros(
        ug_rows * routed // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["w_dn"] = torch.zeros(
        dn_rows * config.inter // 2,
        dtype=torch.uint8,
        device=device,
    )
    tensors["w_ug"].is_shuffled = True
    tensors["w_dn"].is_shuffled = True

    def scale(rows, width):
        return torch.zeros(
            math.ceil(rows / 256) * 256,
            math.ceil((width // 32) / 8) * 8,
            dtype=torch.uint8,
            device=device,
        )

    tensors["s_ug"] = scale(ug_rows, routed)
    tensors["s_dn"] = scale(dn_rows, config.inter)
    tensors["w_ug"].fill_(0x11)
    tensors["w_dn"].fill_(0x11)
    tensors["s_ug"].fill_(120)
    tensors["s_dn"].fill_(120)
    return LayerWeights(
        heads=config.local_heads,
        t=tensors,
        config=config,
        rank=rank,
        npes=8,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


def _projected_input(prefix: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    rows = torch.arange(weights["w_kda_in"].shape[0], device=prefix.device)
    columns = rows.remainder(16)
    return (
        prefix[:, columns].float()
        * weights["w_kda_in"][rows, columns].float().unsqueeze(0)
    ).to(torch.bfloat16)


def _independent_kda_oracle(
    fixture: _Fixture,
    initial_conv: torch.Tensor,
    initial_recurrent: torch.Tensor,
    projected: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cache and recurrent snapshots without using either KDA kernel."""

    from kernels.monokernel.config import KIMI_K3_CONFIG

    config = KIMI_K3_CONFIG
    projection = config.local_heads * config.v_dim
    batch = fixture.snapshots.shape[0]
    drafts = projected[:, : 3 * projection].view(batch, QUERY_LEN, 3 * projection)
    convolved, expected_conv = q8_conv_reference(
        initial_conv,
        fixture.snapshots,
        fixture.accepted,
        drafts,
        weights["w_kda_conv"],
    )
    qkv = convolved.to(torch.bfloat16).view(
        batch, QUERY_LEN, 3, config.local_heads, config.v_dim
    )
    f_a = projected[
        :,
        4 * projection
        + config.local_heads : 4 * projection
        + config.local_heads
        + config.v_dim,
    ]
    gate = (
        f_a.float() @ weights["w_kda_fb"].float().T
    ).to(torch.bfloat16).view(
        batch, QUERY_LEN, config.local_heads, config.v_dim
    )
    beta = projected[
        :, 4 * projection : 4 * projection + config.local_heads
    ].view(batch, QUERY_LEN, config.local_heads)
    _, expected_recurrent, _ = q8_recurrence_reference(
        initial_recurrent,
        fixture.snapshots,
        fixture.accepted,
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        gate,
        beta,
        weights["kda_a_log"],
        weights["kda_dt_bias"],
    )
    return expected_conv, expected_recurrent


def _benchmark_batch(rank: int, args, shape: AgenticKdaShape) -> dict | None:
    from kernels.monokernel.config import ConvStateLayout
    from kernels.monokernel.k3.op import KimiK3MonoKernel, KimiK3StagedAgenticOp

    device = torch.device("cuda", rank)
    weights = _deterministic_weights(device, rank)
    full = KimiK3MonoKernel(
        weights,
        shape.rows,
        layer_idx=0,
        rank=rank,
        npes=8,
        group=dist.group.WORLD,
        reduce_group=dist.group.WORLD,
        mtp=True,
        agentic_batch_size=shape.batch,
        conv_state_layout=ConvStateLayout.TIME_MAJOR,
        state_dtype=torch.float16,
    )
    packed = full.packed_artifacts()
    staged = KimiK3StagedAgenticOp(
        weights,
        batch_size=shape.batch,
        layer_idx=0,
        rank=rank,
        npes=8,
        group=dist.group.WORLD,
        reduce_group=dist.group.WORLD,
        packed_artifacts=packed,
    )

    def full_call(
        fixture: _Fixture,
        epoch_layer: int,
        *,
        advance: bool,
    ) -> None:
        full.forward(
            fixture.prefix,
            fixture.blocks,
            fixture.snapshots,
            fixture.conv,
            fixture.recurrent,
            x_out=fixture.output,
            num_accepted_tokens=fixture.accepted,
            epoch_layer=epoch_layer,
            advance=advance,
        )

    def staged_call(
        fixture: _Fixture,
        epoch_layer: int,
        *,
        advance: bool,
    ) -> None:
        staged.forward(
            fixture.prefix,
            fixture.blocks,
            fixture.snapshots,
            fixture.conv,
            fixture.recurrent,
            x_out=fixture.output,
            num_accepted_tokens=fixture.accepted,
            epoch_layer=epoch_layer,
            advance=advance,
        )

    # Dedicated eager parity fixtures are consumed before profiling or timing.
    full_parity = _fixture(shape, device, args.seed + 99)
    staged_parity = _fixture(shape, device, args.seed + 99)
    initial_conv = full_parity.conv.clone()
    initial_recurrent = full_parity.recurrent.clone()
    full_call(full_parity, 0, advance=True)
    staged_call(staged_parity, 0, advance=True)
    torch.cuda.synchronize()
    expected_conv, expected_recurrent = _independent_kda_oracle(
        full_parity,
        initial_conv,
        initial_recurrent,
        _projected_input(full.pre_attn, weights.t),
        weights.t,
    )
    for name, fixture in (("full", full_parity), ("staged", staged_parity)):
        if not torch.equal(fixture.conv, expected_conv):
            raise AssertionError(f"{name} convolution cache differs from oracle")
        torch.testing.assert_close(
            fixture.recurrent,
            expected_recurrent,
            atol=2e-3,
            rtol=2e-3,
        )
    torch.testing.assert_close(
        full_parity.output,
        staged_parity.output,
        atol=8e-2,
        rtol=8e-2,
    )
    if not torch.count_nonzero(full_parity.output):
        raise AssertionError("full output must be nonzero")
    eager_rank_equal = _rank_equal(full_parity.output) and _rank_equal(
        staged_parity.output
    )
    if not eager_rank_equal:
        raise AssertionError("eager output checksums differ across TP ranks")
    _check_observable_full_path(full, shape)

    # Dedicated graph parity fixtures use the identical layer/epoch protocol.
    full_graph_parity = _fixture(shape, device, args.seed + 199)
    staged_graph_parity = _fixture(shape, device, args.seed + 199)
    full_parity_graph = _capture_layer_graph(
        lambda layer: full_call(full_graph_parity, layer, advance=False),
        full.advance_step,
        args.layers,
    )
    staged_parity_graph = _capture_layer_graph(
        lambda layer: staged_call(staged_graph_parity, layer, advance=False),
        staged.advance_step,
        args.layers,
    )
    for graph in (full_parity_graph, staged_parity_graph):
        for _ in range(2):
            dist.barrier()
            if args.rank_skew_ms and rank == 0:
                time.sleep(args.rank_skew_ms / 1000.0)
            graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        full_graph_parity.output,
        staged_graph_parity.output,
        atol=8e-2,
        rtol=8e-2,
    )
    torch.testing.assert_close(
        full_graph_parity.recurrent,
        staged_graph_parity.recurrent,
        atol=2e-3,
        rtol=2e-3,
    )
    if not torch.equal(full_graph_parity.conv, staged_graph_parity.conv):
        raise AssertionError("graph convolution caches differ")
    graph_rank_equal = _rank_equal(full_graph_parity.output) and _rank_equal(
        staged_graph_parity.output
    )
    if not graph_rank_equal:
        raise AssertionError("graph output checksums differ across TP ranks")
    del full_parity_graph, staged_parity_graph

    # Profiling has its own state/cache and cannot perturb parity or timing.
    staged_profile = None
    full_profile = None
    if args.profile:
        full_profile_fixture = _fixture(shape, device, args.seed + 299)
        staged_profile_fixture = _fixture(shape, device, args.seed + 299)
        staged_call(staged_profile_fixture, 0, advance=True)
        torch.cuda.synchronize()
        for tail in staged.tails:
            tail.start_stage_profile()
        staged_call(staged_profile_fixture, 1, advance=True)
        tail_profiles = [tail.finish_stage_profile() for tail in staged.tails]
        staged_profile = {
            name: statistics.median(profile[name] for profile in tail_profiles)
            for name in tail_profiles[0]
        }
        full_profile_graph = _capture_layer_graph(
            lambda layer: full_call(full_profile_fixture, layer, advance=False),
            full.advance_step,
            args.layers,
        )
        full_profile = _profile_full(
            full_profile_graph,
            full.attention.monokernel_timeline,
            repeats=args.profile_repeats,
        )
        del full_profile_graph

    staged_kernel_profile = None
    staged_front_stage_profile = None

    # Timing fixtures are fresh and used only by their corresponding graph.
    full_timing = _fixture(shape, device, args.seed + 399)
    staged_timing = _fixture(shape, device, args.seed + 399)
    full_graph = _capture_layer_graph(
        lambda layer: full_call(full_timing, layer, advance=False),
        full.advance_step,
        args.layers,
    )
    staged_graph = _capture_layer_graph(
        lambda layer: staged_call(staged_timing, layer, advance=False),
        staged.advance_step,
        args.layers,
    )
    full_counts = _hip_graph_counts(full_graph)
    staged_counts = _hip_graph_counts(staged_graph)
    full_us, _ = _time_graph(
        full_graph,
        layers=args.layers,
        warmups=args.warmups,
        repeats=args.repeats,
        rank_skew_ms=args.rank_skew_ms,
    )
    staged_us, _ = _time_graph(
        staged_graph,
        layers=args.layers,
        warmups=args.warmups,
        repeats=args.repeats,
        rank_skew_ms=args.rank_skew_ms,
    )
    if args.kernel_profile:
        staged_kernel_fixture = _fixture(shape, device, args.seed + 349)
        staged_kernel_graph = _capture_layer_graph(
            lambda layer: staged_call(
                staged_kernel_fixture,
                layer,
                advance=False,
            ),
            staged.advance_step,
            args.layers,
        )
        staged_kernel_profile = _profile_staged_graph(
            staged_kernel_graph,
            staged.front.attention.monokernel_timeline,
            rank=rank,
            layers=args.layers,
            repeats=args.profile_repeats,
            labels=_STAGE_LABELS[:4],
        )
        staged_front_stage_profile = staged_kernel_profile["stages"]
        del staged_kernel_graph
    eager_parity = parity_metrics(
        expected_output=full_parity.output,
        actual_output=staged_parity.output,
        expected_state=expected_recurrent,
        actual_state=staged_parity.recurrent,
        expected_conv=expected_conv,
        actual_conv=staged_parity.conv,
        rank_equal=eager_rank_equal,
    )
    graph_parity = parity_metrics(
        expected_output=full_graph_parity.output,
        actual_output=staged_graph_parity.output,
        expected_state=full_graph_parity.recurrent,
        actual_state=staged_graph_parity.recurrent,
        expected_conv=full_graph_parity.conv,
        actual_conv=staged_graph_parity.conv,
        rank_equal=graph_rank_equal,
    )
    result = None
    if rank == 0:
        result = {
            "model": "kimi_k3",
            "attention_family": "kda",
            "batch": shape.batch,
            "rows": shape.rows,
            "query_len": QUERY_LEN,
            "tp": 8,
            "dcp": 1,
            "layers": args.layers,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "rank_skew_ms": args.rank_skew_ms,
            "critical_rank_median_us": {
                "staged": staged_us,
                "full": full_us,
            },
            "speedup_staged_over_full": staged_us / full_us,
            "launches_per_layer": {
                "staged": staged_counts.get("kernel", 0) / args.layers,
                "full": full_counts.get("kernel", 0) / args.layers,
            },
            "graph_node_counts": {
                "staged": staged_counts,
                "full": full_counts,
            },
            "parity": {
                "eager": eager_parity,
                "graph": graph_parity,
                "full_oracle_state_max_abs": float(
                    (
                        full_parity.recurrent.float()
                        - expected_recurrent.float()
                    )
                    .abs()
                    .max()
                ),
            },
            "starting_point_us": STARTING_POINTS_US[shape.batch],
        }
        if full_profile is not None:
            result["full_stage_profile_us"] = full_profile
            result["full_stage_profile_sum_us"] = sum(full_profile.values())
        if staged_profile is not None:
            result["staged_tail_profile_us"] = staged_profile
            result["staged_tail_profile_sum_us"] = sum(staged_profile.values())
        if staged_kernel_profile is not None:
            result["staged_kernel_profile"] = staged_kernel_profile
            result["staged_front_stage_profile_us"] = staged_front_stage_profile
        print(json.dumps(result, sort_keys=True), flush=True)

    del full_graph, staged_graph
    staged.close()
    full.close()
    dist.barrier()
    return result


def _worker(rank: int, args, port: int, results) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
    )
    try:
        payload = []
        for batch in args.batches:
            result = _benchmark_batch(rank, args, AgenticKdaShape(batch))
            if result is not None:
                payload.append(result)
        if rank == 0:
            results.extend(payload)
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = make_argument_parser()
    args = parser.parse_args()
    args.batches = tuple(args.batches)
    if any(batch not in SUPPORTED_BATCHES for batch in args.batches):
        parser.error(f"--batches must be selected from {SUPPORTED_BATCHES}")
    if args.layers != INVOCATIONS_PER_GRAPH:
        parser.error(f"--layers must be exactly {INVOCATIONS_PER_GRAPH}")
    if args.warmups < WARMUPS:
        parser.error(f"--warmups must be at least {WARMUPS}")
    if args.repeats < MIN_REPEATS:
        parser.error(f"--repeats must be at least {MIN_REPEATS}")
    if args.profile_repeats < MIN_REPEATS:
        parser.error(f"--profile-repeats must be at least {MIN_REPEATS}")
    if 0 < args.rank_skew_ms < 25:
        parser.error("--rank-skew-ms must be zero or at least 25")

    manager = mp.Manager()
    results = manager.list()
    mp.spawn(
        _worker,
        args=(args, _free_port(), results),
        nprocs=8,
        join=True,
    )
    payload = list(results)
    print("BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True), flush=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
