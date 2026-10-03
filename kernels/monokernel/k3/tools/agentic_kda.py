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
import socket
import statistics
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


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=SUPPORTED_BATCHES)
    parser.add_argument("--layers", type=int, default=INVOCATIONS_PER_GRAPH)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=MIN_REPEATS)
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


def _capture(call, layers: int) -> torch.cuda.CUDAGraph:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(layers):
            call()
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
) -> tuple[float, list[float]]:
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    local = []
    for _ in range(repeats):
        dist.barrier()
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
) -> dict[str, float]:
    local_runs = []
    for _ in range(repeats):
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        ticks = timeline.cpu().tolist()[: len(_STAGE_LABELS) + 1]
        local_runs.append(
            {
                label: (ticks[index + 1] - ticks[index]) / 100.0
                for index, label in enumerate(_STAGE_LABELS)
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


def _check_observable_full_path(layer, shape: AgenticKdaShape) -> None:
    from kernels.monokernel.k3.kernel import monokernel_layout

    layout = monokernel_layout(
        shape.rows,
        fuse_attn_res=True,
        fuse_moe=True,
        mtp=True,
        agentic_batch_size=shape.batch,
        state_dtype=torch.float16,
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


def _check_rank_equal(output: torch.Tensor) -> None:
    checksum = torch.stack(
        (
            output.float().sum(),
            output.float().square().sum(),
            output.float().abs().sum(),
        )
    )
    gathered = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    for peer in gathered[1:]:
        torch.testing.assert_close(peer, gathered[0], atol=1e-2, rtol=1e-5)


def _benchmark_batch(rank: int, args, shape: AgenticKdaShape) -> dict | None:
    from kernels.monokernel.config import (
        KIMI_K3_CONFIG,
        ConvStateLayout,
        MoeMode,
    )
    from kernels.monokernel.k3.op import KimiK3MonoKernel
    from kernels.monokernel.k3.staged import _KimiK3KdaStagedPath
    from kernels.monokernel.reference import make_weights

    device = torch.device("cuda", rank)
    weights = make_weights(
        rank,
        heads=KIMI_K3_CONFIG.local_heads,
        device=device,
        seed=args.seed,
        moe_mode=MoeMode.A16W4,
        model_config=KIMI_K3_CONFIG,
        npes=8,
        attention_family="kda",
    )
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
    front = _KimiK3KdaStagedPath(
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
        packed_artifacts=packed,
        monokernel_only=True,
    )
    front.attention.configure_monokernel(0, fuse_moe=False)
    tails = []
    for _ in range(shape.batch):
        tail = _KimiK3KdaStagedPath(
            weights,
            QUERY_LEN,
            layer_idx=0,
            rank=rank,
            npes=8,
            group=dist.group.WORLD,
            reduce_group=dist.group.WORLD,
            mtp=True,
            agentic_batch_size=1,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            state_dtype=torch.float16,
            packed_artifacts=packed,
        )
        tail.attention.step = front.attention.step
        tails.append(tail)

    full_fixture = _fixture(shape, device, args.seed + 99)
    staged_fixture = _fixture(shape, device, args.seed + 99)

    def full_call() -> None:
        full.forward(
            full_fixture.prefix,
            full_fixture.blocks,
            full_fixture.snapshots,
            full_fixture.conv,
            full_fixture.recurrent,
            x_out=full_fixture.output,
            num_accepted_tokens=full_fixture.accepted,
            epoch_layer=0,
        )

    def staged_call() -> None:
        front.attention.forward(
            staged_fixture.prefix,
            staged_fixture.snapshots,
            staged_fixture.conv,
            staged_fixture.recurrent,
            x_out=front.attention_delta,
            num_accepted_tokens=staged_fixture.accepted,
            block_residual=staged_fixture.blocks,
            pre_updated=front.pre_updated,
            pre_output=front.pre_attn,
            updated_prefix=front.updated_prefix,
            moe_input=front.moe_input,
            quantized_moe_input=front.latent_projection.activation,
            quantized_moe_scale=front.latent_projection.activation_scale,
            layer=0,
            advance=False,
        )
        for request, tail in enumerate(tails):
            chunk = slice(request * QUERY_LEN, (request + 1) * QUERY_LEN)
            tail._moe(
                front.moe_input[chunk],
                0,
                front.updated_prefix[chunk],
                staged_fixture.output[chunk],
            )
        front.advance_step()

    full_call()
    staged_call()
    torch.cuda.synchronize()
    if not torch.equal(full_fixture.conv, staged_fixture.conv):
        raise AssertionError("full/staged convolution cache differs")
    torch.testing.assert_close(
        full_fixture.recurrent,
        staged_fixture.recurrent,
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        full_fixture.output,
        staged_fixture.output,
        atol=8e-2,
        rtol=8e-2,
    )
    if int(torch.count_nonzero(full_fixture.output)) == 0:
        raise AssertionError("full output must be nonzero")
    _check_rank_equal(full_fixture.output)
    _check_rank_equal(staged_fixture.output)
    _check_observable_full_path(full, shape)

    full_graph = _capture(full_call, args.layers)
    staged_graph = _capture(staged_call, args.layers)
    full_counts = _hip_graph_counts(full_graph)
    staged_counts = _hip_graph_counts(staged_graph)
    full_us, _ = _time_graph(
        full_graph,
        layers=args.layers,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    staged_us, _ = _time_graph(
        staged_graph,
        layers=args.layers,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    stage_profile = (
        _profile_full(
            full_graph,
            full.attention.monokernel_timeline,
            repeats=args.profile_repeats,
        )
        if args.profile
        else None
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
                "output_max_abs": float(
                    (full_fixture.output.float() - staged_fixture.output.float())
                    .abs()
                    .max()
                ),
                "state_max_abs": float(
                    (
                        full_fixture.recurrent.float()
                        - staged_fixture.recurrent.float()
                    )
                    .abs()
                    .max()
                ),
                "conv_exact": True,
                "output_nonzero": True,
                "rank_equal": True,
            },
            "starting_point_us": STARTING_POINTS_US[shape.batch],
        }
        if stage_profile is not None:
            result["full_stage_profile_us"] = stage_profile
            result["full_stage_profile_sum_us"] = sum(stage_profile.values())
        print(json.dumps(result, sort_keys=True), flush=True)

    del full_graph, staged_graph
    for tail in tails:
        tail.close()
    front.close()
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
