#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""M13 MegaMoEV2 INT8 smooth-quant correctness and performance tests.

Run the full EP8 matrix directly:

    MORI_SHMEM_HEAP_SIZE=16G PYTHONPATH=/path/to/aiter:. \
      torchrun --standalone --nproc_per_node=8 \
      tests/kernels/test_mega_moe_int8.py --mode a8w4smooth --bs-list 32,128

``MEGAMOE_INT8_MEASURE_PERF`` and ``MEGAMOE_INT8_SKIP_ACC`` provide
environment equivalents for the matching command-line switches.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

try:
    import mori.shmem as ms

    from kernels.mega_moe import MegaMoEV2
    from tests.utils import shuffle_weight

    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    ms = MegaMoEV2 = shuffle_weight = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODEL_DIM = 3584
INTER_DIM = 1280
EXPERTS = 384
TOPK = 8
WORLD = 8
EXPERTS_PER_RANK = EXPERTS // WORLD
WEIGHT_SCALE = 1.0e-3
REL_L2_LIMIT = 0.01
STAGE1_REL_L2_LIMIT = 0.001
GRAPH_REPLAY_REL_L2_LIMIT = 1.0e-6
LEGACY_TARGET_US = 212.0


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in {"", "0", "false", "no"}


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != WORLD:
        raise RuntimeError(f"M13 test requires WORLD_SIZE={WORLD}, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world,
        device_id=device,
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _cleanup_dist():
    try:
        ms.shmem_finalize()
    finally:
        dist.destroy_process_group()


def _all_max(device, value: float) -> float:
    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def _all_min_bool(device, value: bool) -> bool:
    result = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    return bool(result.item())


def _dequant_lqq(u4, scale_u8, zero_u8):
    scale = scale_u8.repeat_interleave(64, dim=2).to(torch.int32)
    zero = zero_u8.repeat_interleave(64, dim=2).to(torch.int32)
    unsigned = (u4.to(torch.int32) * scale + zero).clamp(0, 255).to(torch.uint8)
    return unsigned.bitwise_xor(0x80).view(torch.int8).contiguous()


def _build_weights(mode: str, rank: int, device, *, keep_reference: bool):
    """Build one local 48-expert shard in the public constructor layout."""
    generator = torch.Generator(device=device).manual_seed(1701 + rank)
    w1_scale = torch.full(
        (EXPERTS_PER_RANK, 2 * INTER_DIM),
        WEIGHT_SCALE,
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.full(
        (EXPERTS_PER_RANK, MODEL_DIM),
        WEIGHT_SCALE,
        dtype=torch.float32,
        device=device,
    )
    common = dict(w1_scale=w1_scale, w2_scale=w2_scale)

    if mode == "w8a8smooth":
        w1_ref = torch.randint(
            -8,
            9,
            (EXPERTS_PER_RANK, 2 * INTER_DIM, MODEL_DIM),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        w2_ref = torch.randint(
            -8,
            9,
            (EXPERTS_PER_RANK, MODEL_DIM, INTER_DIM),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        common.update(
            w1=shuffle_weight(w1_ref).contiguous().view(-1),
            w2=shuffle_weight(w2_ref).contiguous().view(-1),
        )
        if not keep_reference:
            w1_ref = w2_ref = None
        return common, (w1_ref, w2_ref)

    if mode != "a8w4smooth":
        raise ValueError(f"unsupported INT8 mode {mode!r}")
    w1_u4 = torch.randint(
        0,
        16,
        (EXPERTS_PER_RANK, 2 * INTER_DIM, MODEL_DIM),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_u4 = torch.randint(
        0,
        16,
        (EXPERTS_PER_RANK, MODEL_DIM, INTER_DIM),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w1_qscale = torch.randint(
        1,
        3,
        (EXPERTS_PER_RANK, 2 * INTER_DIM, MODEL_DIM // 64),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_qscale = torch.randint(
        1,
        3,
        (EXPERTS_PER_RANK, MODEL_DIM, INTER_DIM // 64),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w1_qzero = torch.randint(
        0,
        16,
        w1_qscale.shape,
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_qzero = torch.randint(
        0,
        16,
        w2_qscale.shape,
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w1_ref = _dequant_lqq(w1_u4, w1_qscale, w1_qzero) if keep_reference else None
    w2_ref = _dequant_lqq(w2_u4, w2_qscale, w2_qzero) if keep_reference else None
    common.update(
        w1=w1_u4,
        w2=w2_u4,
        w1_lqq_scale=w1_qscale,
        w1_lqq_zero=w1_qzero,
        w2_lqq_scale=w2_qscale,
        w2_lqq_zero=w2_qzero,
        weight_format="aiter_lqq",
    )
    return common, (w1_ref, w2_ref)


def _make_inputs(tokens: int, rank: int, device):
    generator = torch.Generator(device=device).manual_seed(9109 + rank)
    x = torch.randn(
        (tokens, MODEL_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scores = torch.rand(
        (tokens, EXPERTS),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    weights = torch.rand(
        (tokens, TOPK),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).add_(0.05)
    weights.div_(weights.sum(dim=-1, keepdim=True))
    assert bool((weights > 0).all()) and int(torch.unique(ids).numel()) >= TOPK
    return x.contiguous(), ids, weights.contiguous()


def _gather_routes(x, ids, weights):
    world = dist.get_world_size()
    gathered_x = [torch.empty_like(x) for _ in range(world)]
    gathered_ids = [torch.empty_like(ids) for _ in range(world)]
    gathered_weights = [torch.empty_like(weights) for _ in range(world)]
    dist.all_gather(gathered_x, x)
    dist.all_gather(gathered_ids, ids)
    dist.all_gather(gathered_weights, weights)
    return torch.cat(gathered_x), torch.cat(gathered_ids), torch.cat(gathered_weights)


def _torch_aiter_oracle(
    x,
    ids,
    weights,
    fc1,
    fc2,
    w1_ref,
    w2_ref,
    w1_scale,
    w2_scale,
    rank,
    actual_moe,
):
    """Distributed torch GEMM oracle with AITER's exact smooth quantizers."""
    from aiter.ops.quant import smooth_per_token_scaled_quant

    actual_stage1 = actual_moe._int8_stage1_output
    x_all, ids_all, weights_all = _gather_routes(x, ids, weights)
    total_tokens = x_all.shape[0]
    q1 = torch.zeros((total_tokens, TOPK, MODEL_DIM), dtype=torch.int8, device=x.device)
    s1 = torch.zeros((total_tokens, TOPK, 1), dtype=torch.float32, device=x.device)
    smooth_per_token_scaled_quant(
        q1,
        x_all[:, None, :].expand(-1, TOPK, -1),
        s1,
        fc1,
        ids_all,
        smooth_scale_map_hash=None,
        enable_ps=True,
    )

    flat_ids = ids_all.view(-1).long()
    flat_q1 = q1.view(-1, MODEL_DIM)
    flat_s1 = s1.view(-1, 1)
    a2 = torch.zeros((total_tokens * TOPK, INTER_DIM), dtype=torch.float16, device=x.device)
    expert_begin = rank * EXPERTS_PER_RANK
    expert_end = expert_begin + EXPERTS_PER_RANK
    for expert in range(expert_begin, expert_end):
        rows = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        local_expert = expert - expert_begin
        gemm1 = (flat_q1[rows].float() * flat_s1[rows]) @ w1_ref[local_expert].float().T
        gemm1.mul_(w1_scale[local_expert])
        gate, up = gemm1.chunk(2, dim=-1)
        a2[rows] = (torch.nn.functional.silu(gate) * up).to(torch.float16)
    local_rows = (flat_ids >= expert_begin) & (flat_ids < expert_end)
    stage1_reference = a2[local_rows].float()
    stage1_actual = actual_stage1.a2[: total_tokens * TOPK][local_rows].float()
    stage1_rel_l2 = float(
        (
            torch.linalg.vector_norm(stage1_actual - stage1_reference)
            / torch.linalg.vector_norm(stage1_reference).clamp_min(1e-12)
        ).item()
    )
    dist.all_reduce(a2, op=dist.ReduceOp.SUM)

    q2 = torch.zeros_like(a2, dtype=torch.int8)
    s2 = torch.zeros((total_tokens * TOPK, 1), dtype=torch.float32, device=x.device)
    smooth_per_token_scaled_quant(
        q2.view(total_tokens, TOPK, INTER_DIM),
        a2.view(total_tokens, TOPK, INTER_DIM),
        s2.view(total_tokens, TOPK, 1),
        fc2,
        ids_all,
        smooth_scale_map_hash=None,
        enable_ps=True,
    )

    oracle = torch.zeros((total_tokens, MODEL_DIM), dtype=torch.float32, device=x.device)
    flat_weights = weights_all.view(-1)
    flat_tokens = torch.arange(total_tokens, device=x.device)[:, None].expand(-1, TOPK).reshape(-1)
    for expert in range(expert_begin, expert_end):
        rows = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        local_expert = expert - expert_begin
        gemm2 = (q2[rows].float() * s2[rows]) @ w2_ref[local_expert].float().T
        gemm2.mul_(w2_scale[local_expert])
        gemm2.mul_(flat_weights[rows, None])
        oracle.index_add_(0, flat_tokens[rows], gemm2)
    dist.all_reduce(oracle, op=dist.ReduceOp.SUM)
    start = rank * x.shape[0]
    return oracle[start : start + x.shape[0]], stage1_rel_l2


def _capture_graph(body):
    body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        body()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    return graph


def _time_graph(graph, iterations: int, device):
    ms.shmem_barrier_all()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return _all_max(device, begin.elapsed_time(end) * 1000.0 / iterations)


def _run_bucket(
    mode,
    tokens,
    iterations,
    measure_perf,
    skip_acc,
    rank,
    device,
    weights,
    references,
    fc1,
    fc2,
    trace_dir=None,
    eager_trace=False,
):
    x, ids, routing_weights = _make_inputs(tokens, rank, device)
    moe = MegaMoEV2(
        rank=rank,
        world_size=WORLD,
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=TOPK,
        quant=mode,
        max_tok_per_rank=tokens,
        fc1_smooth_scale=fc1,
        fc2_smooth_scale=fc2,
        **weights,
    )
    holder = {}

    def e2e_body():
        holder["output"] = moe.forward(x, routing_weights, ids)

    if eager_trace:
        for _ in range(10):
            e2e_body()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        e2e_graph = None
    else:
        e2e_graph = _capture_graph(e2e_body)

    if e2e_graph is None:
        replay_rel_l2_all = []
        replay_rel_l2_max = 0.0
        replay_stable = True
    else:
        replay_reference = holder["output"][:tokens].float().clone()
        local_replay_rel_l2 = 0.0
        for _ in range(3):
            e2e_graph.replay()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            replay_output = holder["output"][:tokens].float()
            replay_rel_l2 = float(
                (
                    torch.linalg.vector_norm(replay_output - replay_reference)
                    / torch.linalg.vector_norm(replay_reference).clamp_min(1e-12)
                ).item()
            )
            local_replay_rel_l2 = max(local_replay_rel_l2, replay_rel_l2)
        local_replay_rel = torch.tensor(local_replay_rel_l2, dtype=torch.float64, device=device)
        gathered_replay_rel = [torch.empty_like(local_replay_rel) for _ in range(WORLD)]
        dist.all_gather(gathered_replay_rel, local_replay_rel)
        replay_rel_l2_all = [float(value.item()) for value in gathered_replay_rel]
        replay_rel_l2_max = max(replay_rel_l2_all)
        replay_stable = replay_rel_l2_max < GRAPH_REPLAY_REL_L2_LIMIT

    output = holder["output"][:tokens].float()
    finite = bool(torch.isfinite(output).all())
    rel_l2 = -1.0
    if not skip_acc:
        oracle, stage1_rel_l2 = _torch_aiter_oracle(
            x,
            ids,
            routing_weights,
            fc1,
            fc2,
            references[0],
            references[1],
            weights["w1_scale"],
            weights["w2_scale"],
            rank,
            moe,
        )
        rel_l2 = float(
            (torch.linalg.vector_norm(output - oracle) / torch.linalg.vector_norm(oracle).clamp_min(1e-12)).item()
        )
        finite = finite and bool(torch.isfinite(oracle).all()) and math.isfinite(rel_l2)
    if skip_acc:
        rel_l2_all = []
        rel_l2_max = -1.0
        stage1_rel_l2_max = -1.0
    else:
        local_rel = torch.tensor(rel_l2, dtype=torch.float64, device=device)
        gathered_rel = [torch.empty_like(local_rel) for _ in range(WORLD)]
        dist.all_gather(gathered_rel, local_rel)
        rel_l2_all = [float(value.item()) for value in gathered_rel]
        rel_l2_max = max(rel_l2_all)
        local_stage1_rel = torch.tensor(stage1_rel_l2, dtype=torch.float64, device=device)
        gathered_stage1_rel = [torch.empty_like(local_stage1_rel) for _ in range(WORLD)]
        dist.all_gather(gathered_stage1_rel, local_stage1_rel)
        stage1_rel_l2_all = [float(value.item()) for value in gathered_stage1_rel]
        stage1_rel_l2_max = max(stage1_rel_l2_all)
    finite_all = _all_min_bool(device, finite)

    stage1_us = stage2_us = e2e_us = -1.0
    if measure_perf and not eager_trace:
        stage1_holder = {}

        def stage1_body():
            front_q, front_scale = moe._run_int8_front_quant(x, ids)
            stage1_holder["output"] = moe._run_int8_stage1(front_q, front_scale, routing_weights, ids)

        stage1_graph = _capture_graph(stage1_body)
        stage1_us = _time_graph(stage1_graph, iterations, device)
        stage1_body()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        def stage2_body():
            stage1_output = stage1_holder["output"]
            requant_q, requant_scale = moe._run_int8_requant(stage1_output, ids)
            holder["stage2"] = moe._run_int8_stage2(
                requant_q,
                requant_scale,
                stage1_output,
                tokens,
                None,
                True,
            )

        stage2_graph = _capture_graph(stage2_body)
        stage2_us = _time_graph(stage2_graph, iterations, device)
        e2e_us = _time_graph(e2e_graph, iterations, device)

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(max(20, iterations // 2)):
                e2e_body() if eager_trace else e2e_graph.replay()
            torch.cuda.synchronize()
        prof.export_chrome_trace(
            os.path.join(
                trace_dir,
                f"mega_moe_int8_{mode}_bs{tokens}_rank{rank}.json",
            )
        )

    passed = finite_all and replay_stable and (
        skip_acc
        or (
            rel_l2_max < REL_L2_LIMIT
            and stage1_rel_l2_max < STAGE1_REL_L2_LIMIT
        )
    )
    if rank == 0:
        accuracy = "skip" if skip_acc else f"{rel_l2_max:.4e}"
        perf = "skip"
        if measure_perf:
            delta = (e2e_us / LEGACY_TARGET_US - 1.0) * 100.0
            perf = (
                f"stage1={stage1_us:.1f}us stage2={stage2_us:.1f}us "
                f"e2e={e2e_us:.1f}us vs {LEGACY_TARGET_US:.0f}us={delta:+.1f}%"
            )
        print(
            f"[M13-INT8] mode={mode} bs={tokens} relL2max={accuracy} "
            f"stage1RelL2max={'skip' if skip_acc else f'{stage1_rel_l2_max:.4e}'} "
            f"finite={finite_all} graphReplayRelL2max={replay_rel_l2_max:.4e} "
            f"graph_replay={'PASS' if replay_stable else 'FAIL'} {perf} "
            f"=> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        if not skip_acc:
            print(
                "  8-rank relL2: " + " ".join(f"r{index}={value:.4e}" for index, value in enumerate(rel_l2_all)),
                flush=True,
            )
            print(
                "  8-rank stage1 relL2: "
                + " ".join(f"r{index}={value:.4e}" for index, value in enumerate(stage1_rel_l2_all)),
                flush=True,
            )
        if replay_rel_l2_all:
            print(
                "  8-rank graph replay relL2: "
                + " ".join(f"r{index}={value:.4e}" for index, value in enumerate(replay_rel_l2_all)),
                flush=True,
            )
    del e2e_graph
    torch.cuda.empty_cache()
    dist.barrier()
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("a8w4smooth", "w8a8smooth"),
        default=os.environ.get("MEGAMOE_INT8_QUANT", "a8w4smooth"),
    )
    parser.add_argument(
        "--bs-list",
        default=os.environ.get("MEGAMOE_INT8_BS", "128"),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("MEGAMOE_INT8_ITERS", "50")),
    )
    parser.add_argument(
        "--measure-perf",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MEGAMOE_INT8_MEASURE_PERF", True),
    )
    parser.add_argument(
        "--skip-acc",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("MEGAMOE_INT8_SKIP_ACC", False),
    )
    parser.add_argument(
        "--trace-dir",
        default=os.environ.get("MEGAMOE_INT8_TRACE_DIR", ""),
    )
    parser.add_argument("--eager-trace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rank, _, device = _setup_dist()
    torch.manual_seed(1234)
    fc1 = 0.75 + 0.5 * torch.rand((EXPERTS, MODEL_DIM), dtype=torch.float32, device=device)
    fc2 = 0.75 + 0.5 * torch.rand((EXPERTS, INTER_DIM), dtype=torch.float32, device=device)
    weights, references = _build_weights(args.mode, rank, device, keep_reference=not args.skip_acc)
    failures = 0
    for tokens in (int(value) for value in args.bs_list.split(",") if value.strip()):
        if tokens <= 0 or tokens & (tokens - 1):
            raise ValueError(f"batch size must be a positive power of two, got {tokens}")
        failures += not _run_bucket(
            args.mode,
            tokens,
            max(1, args.iters),
            args.measure_perf,
            args.skip_acc,
            rank,
            device,
            weights,
            references,
            fc1,
            fc2,
            args.trace_dir or None,
            args.eager_trace,
        )
    global_failures = _all_max(device, failures)
    _cleanup_dist()
    if args.strict and global_failures:
        raise SystemExit(1)


def _physical_gpu_count() -> int:
    env = {key: value for key, value in os.environ.items() if key != "HIP_VISIBLE_DEVICES"}
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:  # noqa: BLE001
        return 0


def _require_gfx95(required_gpus: int):
    if _IMPORT_ERROR:
        pytest.skip(f"MegaMoE INT8 dependencies unavailable: {_IMPORT_ERROR}")
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = str(get_rocm_arch() or "")
    except Exception:  # noqa: BLE001
        arch = ""
    if not arch.startswith("gfx95"):
        pytest.skip(f"MegaMoE INT8 requires gfx95x, found {arch or 'unknown'}")
    if _physical_gpu_count() < required_gpus:
        pytest.skip(f"requires at least {required_gpus} physical GPUs")


def _run_subprocess(command, *, timeout=2400, extra_env=None):
    env = {key: value for key, value in os.environ.items() if key != "HIP_VISIBLE_DEVICES"}
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))]
        + [path for path in sys.path if path]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    for line in result.stdout.splitlines():
        if "[M13-INT8]" in line or "correctness=PASS" in line or "e2e=PASS" in line:
            print(line)
    assert result.returncode == 0, (
        f"command failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-3000:]}"
    )


@pytest.mark.multi_gpu
@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_m13_int8_8gpu_e2e(mode):
    _require_gfx95(WORLD)
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={WORLD}",
            os.path.abspath(__file__),
            "--mode",
            mode,
            "--bs-list",
            os.environ.get("MEGAMOE_INT8_PYTEST_BS", "32,128,256"),
            "--iters",
            os.environ.get("MEGAMOE_INT8_PYTEST_ITERS", "20"),
            "--measure-perf",
            "--strict",
        ]
    )


@pytest.mark.multi_gpu
def test_m13_native_a8w4_8gpu_acceptance():
    """Gate the production MXFP8/E8M0 decode path over its full contract."""
    _require_gfx95(WORLD)
    from tests.kernels.test_mega_moe_v2 import _run_mega_8gpu

    # Native decode requires tokens=MTPR, so every point is a distinct
    # production operator configuration.  Isolate those configurations in
    # separate torchrun processes: this also gives CI a bounded timeout and a
    # complete result for each shape instead of losing an entire matrix behind
    # one buffered subprocess.
    for tokens in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        _run_mega_8gpu(
            network="m13",
            quant="a8w4",
            bs_list=str(tokens),
            iters=int(os.environ.get("MEGAMOE_M13_A8W4_CI_ITERS", "20")),
            measure_perf=True,
            skip_acc=False,
            timeout=300,
        )


@pytest.mark.multi_gpu
@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_int8_stage1_distributed_smoke(mode):
    _require_gfx95(WORLD)
    smoke = os.path.join(os.path.dirname(__file__), "mega_moe_int8_stage1_smoke.py")
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={WORLD}",
            smoke,
        ],
        extra_env={"MEGAMOE_INT8_QUANT": mode},
    )


@pytest.mark.parametrize("mode", ["a8w4smooth", "w8a8smooth"])
def test_int8_stage2_correctness_smoke(mode):
    _require_gfx95(1)
    smoke = os.path.join(os.path.dirname(__file__), "mega_moe_int8_stage2_smoke.py")
    _run_subprocess(
        [sys.executable, smoke, "--mode", mode, "--iterations", "10"],
        timeout=600,
    )


if __name__ == "__main__":
    main()
