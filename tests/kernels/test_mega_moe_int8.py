#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""M13 MegaMoEV2 INT8 smooth-quant correctness and performance tests.

Run the EP8 W8A8 prefill case (SP8 32K means 4096 tokens per rank):

    MORI_SHMEM_HEAP_SIZE=16G PYTHONPATH=/path/to/aiter:. \
      torchrun --standalone --nproc_per_node=8 \
      tests/kernels/test_mega_moe_int8.py --mode w8a8smooth \
      --dispatch-quant mxfp4 --bs-list 4096

``MEGAMOE_INT8_MEASURE_PERF`` and ``MEGAMOE_INT8_SKIP_ACC`` provide
environment equivalents for the matching command-line switches.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest
import torch
import torch.distributed as dist

try:
    import mori.shmem as ms

    from kernels.mega_moe import MegaMoEV2
    from kernels.mega_moe.quant import per_1x32_mx_quant
    from tests.kernels.mega_moe_mxfp4_smooth_oracle import decode_mxfp4_e8m0
    from tests.utils import shuffle_weight

    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    ms = MegaMoEV2 = shuffle_weight = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODEL_DIM = 3584
INTER_DIM = 1280
TOPK = 8
WORLD = int(os.environ.get("MEGAMOE_INT8_WORLD", "8"))
EXPERTS_PER_RANK = 48
EXPERTS = EXPERTS_PER_RANK * WORLD
WEIGHT_SCALE = 1.0e-3
REL_L2_LIMIT = 0.01
STAGE1_REL_L2_LIMIT = 0.001
GRAPH_REPLAY_REL_L2_LIMIT = 1.0e-6
GRAPH_REPLAY_CHECKS = int(os.environ.get("MEGAMOE_GRAPH_REPLAY_CHECKS", "20"))
LEGACY_TARGET_US = 212.0
MXFP4_PREFILL_MTPR = 32768
MXFP4_PREFILL_PERF_TOL = 0.07
MXFP4_PREFILL_E2E_US = {
    1: 191.8,
    4: 298.5,
    8: 325.0,
    16: 379.3,
    32: 430.2,
    64: 426.0,
    128: 464.7,
    256: 534.1,
    512: 651.0,
    4096: 1915.1,
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in {"", "0", "false", "no"}


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != WORLD:
        raise RuntimeError(
            f"M13 test requires WORLD_SIZE={WORLD}, got {world}; "
            "set MEGAMOE_INT8_WORLD to the torchrun process count"
        )
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


def _make_inputs(tokens: int, rank: int, device, hot_fraction: float = 0.0):
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
    if not 0.0 <= hot_fraction <= 1.0:
        raise ValueError(f"hot_fraction must be in [0, 1], got {hot_fraction}")
    hot_rows = int(tokens * hot_fraction)
    if hot_rows:
        # Force one common hot expert while retaining seven unique random
        # routes.  This models a cross-rank prefill gating long tail.
        scores[:hot_rows, 0] = 2.0
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
    dispatch_quant=None,
):
    """Distributed torch GEMM oracle with AITER's exact smooth quantizers."""
    from aiter.ops.quant import smooth_per_token_scaled_quant

    actual_stage1 = actual_moe._int8_stage1_output
    x_all, ids_all, weights_all = _gather_routes(x, ids, weights)
    if dispatch_quant == "mxfp4":
        fp4, mx_scale = per_1x32_mx_quant(x_all, quant_mode="fp4")
        x_all = decode_mxfp4_e8m0(fp4, mx_scale).to(torch.bfloat16)
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
    # Stage1 stores by the 24-bit global source encoding, whose rank stride is
    # MTPR rather than the live token count.  Map the dense oracle rows into
    # that sparse ATOM layout when MTPR > tokens_per_rank.
    dense_tokens = torch.arange(total_tokens, device=x.device)
    source_ranks = torch.div(dense_tokens, x.shape[0], rounding_mode="floor")
    source_lids = dense_tokens - source_ranks * x.shape[0]
    sparse_tokens = source_ranks * actual_moe.mtpr + source_lids
    sparse_rows = (
        sparse_tokens[:, None] * TOPK
        + torch.arange(TOPK, device=x.device)[None, :]
    ).reshape(-1)
    stage1_actual = actual_stage1.a2[sparse_rows[local_rows]].float()
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
    dispatch_quant=None,
    stage1_override=None,
    stage2_override=None,
    route_hot_fraction=0.0,
    debug_replay_stages=False,
    max_tok_per_rank=None,
):
    mtpr = tokens if max_tok_per_rank is None else int(max_tok_per_rank)
    if mtpr < tokens or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr must be a power of two >= tokens, got tokens={tokens}, mtpr={mtpr}")
    x, ids, routing_weights = _make_inputs(tokens, rank, device, route_hot_fraction)
    moe = MegaMoEV2(
        rank=rank,
        world_size=WORLD,
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=TOPK,
        quant=mode,
        max_tok_per_rank=mtpr,
        fc1_smooth_scale=fc1,
        fc2_smooth_scale=fc2,
        dispatch_quant=dispatch_quant,
        **weights,
    )
    if os.environ.get("MEGAMOE_DEBUG_SKIP_STAGE1_GEMM", "0") == "1":
        moe._int8_a2.zero_()
    if stage1_override or stage2_override:
        tuned = moe._select_config(tokens)
        if stage1_override:
            tuned = replace(
                tuned,
                stage1=replace(tuned.stage1, **stage1_override),
            )
        if stage2_override:
            tuned = replace(
                tuned,
                stage2=replace(tuned.stage2, **stage2_override),
            )

        def select_tuned_config(_tokens):
            # MegaMoEV2._select_config also publishes the selected Stage2
            # configuration for the INT8 pipeline.  Preserve that side effect
            # so test-only Stage2 overrides reach _run_int8_stage2.
            moe._active_config = tuned
            return tuned

        moe._select_config = select_tuned_config

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

    if _env_flag("MEGAMOE_DEBUG_PRINT_NUM_VALID", False):
        torch.cuda.synchronize()
        local_num_valid = [int(value) for value in moe._s1_op.num_valid.tolist()]
        gathered_num_valid = [None for _ in range(WORLD)] if rank == 0 else None
        dist.gather_object(local_num_valid, gathered_num_valid, dst=0)
        if rank == 0:
            print(
                f"[M13-INT8-DEBUG] num_valid_by_rank={gathered_num_valid} "
                f"sort_block_m={moe._select_config(tokens).stage1.sort_block_m}",
                flush=True,
            )

    if e2e_graph is None:
        replay_rel_l2_all = []
        replay_rel_l2_max = 0.0
        replay_stable = True
    else:
        replay_reference = holder["output"][:tokens].float().clone()
        e2e_debug_reference = None
        if debug_replay_stages:
            ref_meta = moe._int8_sorted_tokens.clone()
            ref_token = ref_meta.bitwise_and(0x00FFFFFF).long()
            ref_slot = ref_meta.bitwise_right_shift(24).long()
            ref_valid = (ref_token < moe.max_recv) & (ref_slot < TOPK)
            ref_sources = torch.sort(ref_meta[ref_valid]).values
            ref_rows = ref_token[ref_valid] * TOPK + ref_slot[ref_valid]
            p2p_rows = tokens * TOPK
            e2e_debug_reference = {
                "sources": ref_sources,
                "rows": ref_rows,
                "a2": moe._int8_a2.view(-1, INTER_DIM)[ref_rows].clone(),
                "q": moe._int8_requant_q.view(-1, INTER_DIM)[ref_rows].clone(),
                "scale": moe._int8_requant_scale.view(-1)[ref_rows].clone(),
                "p2p": moe.comb_op.shmem_comb_inp_tok.view(torch.bfloat16)
                .view(-1, MODEL_DIM)[:p2p_rows]
                .clone(),
            }
        local_replay_rel_l2 = 0.0
        for replay_index in range(GRAPH_REPLAY_CHECKS):
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
            if e2e_debug_reference is not None and replay_rel_l2 >= GRAPH_REPLAY_REL_L2_LIMIT:
                current_meta = moe._int8_sorted_tokens
                current_token = current_meta.bitwise_and(0x00FFFFFF)
                current_slot = current_meta.bitwise_right_shift(24)
                current_valid = (current_token < moe.max_recv) & (current_slot < TOPK)
                current_sources = torch.sort(current_meta[current_valid]).values
                ref_rows = e2e_debug_reference["rows"]
                current_a2 = moe._int8_a2.view(-1, INTER_DIM)[ref_rows]
                current_q = moe._int8_requant_q.view(-1, INTER_DIM)[ref_rows]
                current_scale = moe._int8_requant_scale.view(-1)[ref_rows]
                current_p2p = (
                    moe.comb_op.shmem_comb_inp_tok.view(torch.bfloat16)
                    .view(-1, MODEL_DIM)[: tokens * TOPK]
                )
                source_count_delta = abs(
                    int(current_sources.numel())
                    - int(e2e_debug_reference["sources"].numel())
                )
                source_mismatch = source_count_delta
                if source_count_delta == 0:
                    source_mismatch += int(
                        torch.count_nonzero(
                            current_sources != e2e_debug_reference["sources"]
                        ).item()
                    )
                a2_mismatch = int(
                    torch.count_nonzero(current_a2 != e2e_debug_reference["a2"]).item()
                )
                q_mismatch = int(
                    torch.count_nonzero(current_q != e2e_debug_reference["q"]).item()
                )
                scale_max_abs = float(
                    (current_scale - e2e_debug_reference["scale"]).abs().max().item()
                )
                p2p_mismatch = int(
                    torch.count_nonzero(current_p2p != e2e_debug_reference["p2p"]).item()
                )
                p2p_row_mismatch = torch.count_nonzero(
                    current_p2p != e2e_debug_reference["p2p"], dim=1
                )
                bad_rows = torch.nonzero(p2p_row_mismatch, as_tuple=False).flatten()
                p2p_detail = ""
                if bad_rows.numel():
                    bad_row = int(bad_rows[0].item())
                    cur_row = current_p2p[bad_row].float()
                    ref_row = e2e_debug_reference["p2p"][bad_row].float()
                    p2p_detail = (
                        f" badRows={bad_rows[:4].cpu().tolist()} badRow={bad_row} "
                        f"rowMismatch={int(p2p_row_mismatch[bad_row].item())} "
                        f"curZero={int(torch.count_nonzero(cur_row == 0).item())} "
                        f"refZero={int(torch.count_nonzero(ref_row == 0).item())} "
                        f"curNorm={float(torch.linalg.vector_norm(cur_row).item()):.4e} "
                        f"refNorm={float(torch.linalg.vector_norm(ref_row).item()):.4e} "
                        f"curHead={cur_row[:8].cpu().tolist()} "
                        f"refHead={ref_row[:8].cpu().tolist()}"
                    )
                print(
                    f"[M13-INT8-E2E-DEBUG] rank={rank} replay={replay_index} "
                    f"outputRelL2={replay_rel_l2:.4e} metaMismatch={source_mismatch} "
                    f"a2Mismatch={a2_mismatch} qMismatch={q_mismatch} "
                    f"scaleMaxAbs={scale_max_abs:.4e} p2pMismatch={p2p_mismatch}"
                    f"{p2p_detail}",
                    flush=True,
                )
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
            dispatch_quant,
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
    unfused_gemm1_us = unfused_serial_us = -1.0
    stage1_replay_stable = True
    stage2_replay_stable = True
    if measure_perf and not eager_trace:
        stage1_holder = {}

        def stage1_body():
            if dispatch_quant == "mxfp4":
                front_q, front_scale = moe._run_mxfp4_front_quant(x)
            elif moe._s1_smoothquant_mode == "bf16_route":
                front_q, front_scale = x, None
            else:
                front_q, front_scale = moe._run_int8_front_quant(x, ids)
            stage1_holder["output"] = moe._run_int8_stage1(
                front_q,
                front_scale,
                routing_weights,
                ids,
                mxfp4_transport=dispatch_quant == "mxfp4",
            )

        stage1_graph = _capture_graph(stage1_body)
        if hasattr(moe, "_int8_stage1_q"):
            stage1_input_q = moe._int8_stage1_q
            stage1_input_scale = moe._int8_stage1_scale
        else:
            # Standalone SmoothQuant dispatches route-major INT8 rows directly
            # into the fixed-slot transport buffers; no converted MX buffer is
            # allocated in this mode.
            stage1_input_q = moe._s1_rx
            stage1_input_scale = moe._s1_scale_i32.view(torch.float32)
        unfused_gemm1_graph = unfused_serial_graph = None
        if _env_flag("MEGAMOE_BENCH_UNFUSED_STAGE1", False):
            if os.environ.get("MEGAMOE_DEBUG_SKIP_STAGE1_GEMM", "0") != "1":
                raise RuntimeError(
                    "MEGAMOE_BENCH_UNFUSED_STAGE1 requires "
                    "MEGAMOE_DEBUG_SKIP_STAGE1_GEMM=1 so Stage1 measures only "
                    "front-quant/dispatch/dequant/SmoothQuant"
                )
            from flydsl import expr as fx
            from kernels.mega_moe.gemm1 import int8_gemm1_kernel

            selected_stage1 = moe._select_config(tokens).stage1
            qscale = (
                moe._int8_w1_lqq_scale
                if moe._int8_w1_lqq_scale is not None
                else moe._int8_w1_scale
            )
            qzero = (
                moe._int8_w1_lqq_zero
                if moe._int8_w1_lqq_zero is not None
                else moe._int8_w1_scale
            )
            def unfused_gemm1_body():
                int8_gemm1_kernel(
                    moe._int8_a2,
                    stage1_input_q,
                    moe._int8_w1,
                    stage1_input_scale,
                    moe._int8_w1_scale,
                    moe._s1_op.tile_row_base,
                    moe._s1_op.sorted_expert_ids,
                    moe._s1_op.num_valid,
                    moe._s1_op.srcmap_em,
                    moe._s1_op.wts_em,
                    moe._int8_sorted_tokens,
                    moe._int8_sorted_experts,
                    moe._int8_sorted_weights,
                    qscale,
                    qzero,
                    fx.Stream(torch.cuda.current_stream()),
                    model_dim=MODEL_DIM,
                    inter_dim=INTER_DIM,
                    expert_offset=rank * EXPERTS_PER_RANK,
                    atom_tokens=WORLD * mtpr,
                    topk=TOPK,
                    packed_int4=mode == "a8w4smooth",
                    sort_block_m=selected_stage1.sort_block_m,
                    tile_n=selected_stage1.tile_n,
                    tile_k=selected_stage1.tile_k,
                    num_waves=selected_stage1.num_waves,
                    swizzle_a=selected_stage1.swizzle_a,
                    async_a_copy=selected_stage1.async_a_copy,
                    waves_per_eu_hint=selected_stage1.waves_per_eu_hint,
                    b_cache_modifier=selected_stage1.b_nt,
                    swiglu_limit=moe.swiglu_limit,
                    num_cu=moe._s1_num_cu,
                )

            def unfused_serial_body():
                stage1_body()
                unfused_gemm1_body()

            # Compile and capture the exact shared GEMM body separately, then
            # capture the serialized unfused chain to include launch ordering.
            unfused_gemm1_graph = _capture_graph(unfused_gemm1_body)
            unfused_serial_graph = _capture_graph(unfused_serial_body)
        if debug_replay_stages:
            stage1_body()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            ref_stage1 = stage1_holder["output"]
            ref_q, ref_scale = moe._run_int8_requant(ref_stage1, ids)
            torch.cuda.synchronize()
            ref_tokens = ref_stage1.sorted_token_ids.clone()
            ref_token = ref_tokens.bitwise_and(0x00FFFFFF).long()
            ref_slot = ref_tokens.bitwise_right_shift(24).long()
            ref_valid = (ref_token < moe.max_recv) & (ref_slot < TOPK)
            ref_sources = torch.sort(ref_tokens[ref_valid]).values
            ref_rows = ref_token[ref_valid] * TOPK + ref_slot[ref_valid]
            ref_valid_positions = torch.nonzero(ref_valid, as_tuple=False).flatten()
            ref_transport_q = (
                moe._s1_rx.view(torch.uint8)
                .view(moe._s1_nvm, -1)[ref_valid_positions]
                .clone()
            )
            ref_transport_scale = (
                moe._s1_scale_i32.view(torch.uint8)
                .view(moe._s1_nvm, -1)[ref_valid_positions]
                .clone()
            )
            ref_stage1_input_q = stage1_input_q[ref_valid_positions].clone()
            ref_stage1_input_scale = stage1_input_scale[ref_valid_positions].clone()
            ref_a2 = ref_stage1.a2.view(-1, INTER_DIM)[ref_rows].clone()
            ref_q = ref_q.view(-1, INTER_DIM)[ref_rows].clone()
            ref_scale = ref_scale.view(-1)[ref_rows].clone()
            a2_mismatch = q_mismatch = meta_mismatch = 0
            scale_delta = 0.0
            for replay_index in range(int(os.environ.get("MEGAMOE_STAGE1_REPLAY_CHECKS", "20"))):
                stage1_graph.replay()
                torch.cuda.synchronize()
                ms.shmem_barrier_all()
                current = stage1_holder["output"]
                current_q, current_scale = moe._run_int8_requant(current, ids)
                torch.cuda.synchronize()
                current_a2 = current.a2.view(-1, INTER_DIM)[ref_rows]
                current_a2_row_mismatch = torch.count_nonzero(
                    current_a2 != ref_a2, dim=1
                )
                current_a2_mismatch = int(current_a2_row_mismatch.sum().item())
                a2_mismatch = max(a2_mismatch, current_a2_mismatch)
                q_mismatch = max(
                    q_mismatch,
                    int(
                        torch.count_nonzero(
                            current_q.view(-1, INTER_DIM)[ref_rows] != ref_q
                        ).item()
                    ),
                )
                current_token = current.sorted_token_ids.bitwise_and(0x00FFFFFF)
                current_slot = current.sorted_token_ids.bitwise_right_shift(24)
                current_valid = (current_token < moe.max_recv) & (current_slot < TOPK)
                current_rows = (
                    current_token[current_valid].long() * TOPK
                    + current_slot[current_valid].long()
                )
                if current_a2_mismatch:
                    bad_selection = int(
                        torch.nonzero(current_a2_row_mismatch, as_tuple=False)[0].item()
                    )
                    bad_route_row = int(ref_rows[bad_selection].item())
                    current_match = torch.nonzero(
                        current_rows == bad_route_row, as_tuple=False
                    ).flatten()
                    input_detail = " currentRouteMissing"
                    if current_match.numel():
                        current_valid_positions = torch.nonzero(
                            current_valid, as_tuple=False
                        ).flatten()
                        current_position = int(
                            current_valid_positions[int(current_match[0].item())].item()
                        )
                        current_input_q = stage1_input_q[current_position]
                        ref_input_q = ref_stage1_input_q[bad_selection]
                        current_transport_q = (
                            moe._s1_rx.view(torch.uint8)
                            .view(moe._s1_nvm, -1)[current_position]
                        )
                        current_transport_scale = (
                            moe._s1_scale_i32.view(torch.uint8)
                            .view(moe._s1_nvm, -1)[current_position]
                        )
                        input_detail = (
                            f" inputMismatch={int(torch.count_nonzero(current_input_q != ref_input_q).item())} "
                            f"inputScaleAbs={float((stage1_input_scale[current_position] - ref_stage1_input_scale[bad_selection]).abs().item()):.4e} "
                            f"transportMismatch={int(torch.count_nonzero(current_transport_q != ref_transport_q[bad_selection]).item())} "
                            f"transportScaleMismatch={int(torch.count_nonzero(current_transport_scale != ref_transport_scale[bad_selection]).item())}"
                        )
                    print(
                        f"[M13-INT8-STAGE1-ROW] rank={rank} replay={replay_index} "
                        f"routeRow={bad_route_row} a2Mismatch={current_a2_mismatch}"
                        f"{input_detail}",
                        flush=True,
                    )
                current_sources = torch.sort(current.sorted_token_ids[current_valid]).values
                meta_mismatch = max(
                    meta_mismatch,
                    abs(int(current_sources.numel()) - int(ref_sources.numel()))
                    + int(torch.count_nonzero(current_sources != ref_sources).item()),
                )
                scale_delta = max(
                    scale_delta,
                    float((current_scale.view(-1)[ref_rows] - ref_scale).abs().max().item()),
                )
            q_mismatch = int(_all_max(device, q_mismatch))
            a2_mismatch = int(_all_max(device, a2_mismatch))
            meta_mismatch = int(_all_max(device, meta_mismatch))
            scale_delta = _all_max(device, scale_delta)
            stage1_replay_stable = (
                a2_mismatch == 0
                and q_mismatch == 0
                and meta_mismatch == 0
                and scale_delta == 0.0
            )
            if rank == 0:
                print(
                    f"[M13-INT8-DEBUG] stage1 a2_mismatch={a2_mismatch} "
                    f"q_mismatch={q_mismatch} "
                    f"meta_mismatch={meta_mismatch} scale_max_abs={scale_delta:.4e}",
                    flush=True,
                )
        stage1_us = _time_graph(stage1_graph, iterations, device)
        if unfused_gemm1_graph is not None:
            unfused_gemm1_us = _time_graph(
                unfused_gemm1_graph, iterations, device
            )
            unfused_serial_us = _time_graph(
                unfused_serial_graph, iterations, device
            )
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
        if debug_replay_stages:
            stage2_body()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            ref_stage2 = holder["stage2"][:tokens].float().clone()
            stage2_rel = 0.0
            for _ in range(int(os.environ.get("MEGAMOE_STAGE2_REPLAY_CHECKS", "20"))):
                stage2_graph.replay()
                torch.cuda.synchronize()
                ms.shmem_barrier_all()
                current = holder["stage2"][:tokens].float()
                stage2_rel = max(
                    stage2_rel,
                    float(
                        (torch.linalg.vector_norm(current - ref_stage2)
                         / torch.linalg.vector_norm(ref_stage2).clamp_min(1e-12)).item()
                    ),
                )
            stage2_rel = _all_max(device, stage2_rel)
            stage2_replay_stable = stage2_rel < GRAPH_REPLAY_REL_L2_LIMIT
            if rank == 0:
                print(f"[M13-INT8-DEBUG] stage2 relL2max={stage2_rel:.4e}", flush=True)
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

    perf_ok = True
    perf_limit_us = -1.0
    if measure_perf and dispatch_quant == "mxfp4":
        baseline_us = MXFP4_PREFILL_E2E_US.get(tokens)
        if baseline_us is None:
            raise ValueError(f"no MXFP4 prefill performance baseline for tokens={tokens}")
        perf_limit_us = baseline_us * (1.0 + MXFP4_PREFILL_PERF_TOL)
        perf_ok = e2e_us <= perf_limit_us

    passed = (
        finite_all
        and replay_stable
        and stage1_replay_stable
        and stage2_replay_stable
        and perf_ok
        and (
            skip_acc
            or (
                rel_l2_max < REL_L2_LIMIT
                and stage1_rel_l2_max < STAGE1_REL_L2_LIMIT
            )
        )
    )
    if rank == 0:
        accuracy = "skip" if skip_acc else f"{rel_l2_max:.4e}"
        perf = "skip"
        stage_replay = ""
        if measure_perf and debug_replay_stages:
            stage_replay = (
                f" stage_replay=s1:{'PASS' if stage1_replay_stable else 'FAIL'},"
                f"s2:{'PASS' if stage2_replay_stable else 'FAIL'}"
            )
        if measure_perf:
            delta = (e2e_us / LEGACY_TARGET_US - 1.0) * 100.0
            perf = (
                f"stage1={stage1_us:.1f}us stage2={stage2_us:.1f}us "
                f"e2e={e2e_us:.1f}us vs {LEGACY_TARGET_US:.0f}us={delta:+.1f}%"
            )
            if unfused_gemm1_us >= 0:
                perf += (
                    f" prologue={stage1_us:.1f}us"
                    f" standaloneGemm1={unfused_gemm1_us:.1f}us"
                    f" serialStage1={unfused_serial_us:.1f}us"
                )
        print(
            f"[M13-INT8] mode={mode} dispatch={dispatch_quant or 'int8'} "
            f"bs={tokens} relL2max={accuracy} "
            f"stage1RelL2max={'skip' if skip_acc else f'{stage1_rel_l2_max:.4e}'} "
            f"finite={finite_all} graphReplayRelL2max={replay_rel_l2_max:.4e} "
            f"graph_replay={'PASS' if replay_stable else 'FAIL'}{stage_replay} {perf} "
            f"perf_gate={'PASS' if perf_ok else 'FAIL'}"
            f"{'' if perf_limit_us < 0 else f' limit={perf_limit_us:.1f}us'} "
            f"=> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        if not skip_acc:
            print(
                f"  {WORLD}-rank relL2: "
                + " ".join(f"r{index}={value:.4e}" for index, value in enumerate(rel_l2_all)),
                flush=True,
            )
            print(
                f"  {WORLD}-rank stage1 relL2: "
                + " ".join(f"r{index}={value:.4e}" for index, value in enumerate(stage1_rel_l2_all)),
                flush=True,
            )
        if replay_rel_l2_all:
            print(
                f"  {WORLD}-rank graph replay relL2: "
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
    parser.add_argument(
        "--dispatch-quant",
        choices=("int8", "mxfp4"),
        default=os.environ.get("MEGAMOE_DISPATCH_QUANT", "int8"),
    )
    parser.add_argument(
        "--mtpr",
        type=int,
        default=int(os.environ.get("MEGAMOE_MTPR", "0")),
        help="max tokens per rank; 0 uses the current --bs-list value",
    )
    parser.add_argument(
        "--stage1-override",
        default=os.environ.get("MEGAMOE_STAGE1_OVERRIDE", ""),
        help="test-only Stage1Config overrides, for example tile_n=256,grid_mult=3",
    )
    parser.add_argument(
        "--stage2-override",
        default=os.environ.get("MEGAMOE_STAGE2_OVERRIDE", ""),
        help="test-only Stage2Config overrides, for example persist_cu=224,skew_cu=64",
    )
    parser.add_argument(
        "--hot-route-fraction",
        type=float,
        default=float(os.environ.get("MEGAMOE_HOT_ROUTE_FRACTION", "0")),
        help="fraction of local tokens forced to include global expert 0",
    )
    parser.add_argument("--debug-replay-stages", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.mode == "a8w4smooth" and args.dispatch_quant == "mxfp4":
        parser.error(
            "a8w4smooth decode supports only standalone AITER SmoothQuant; "
            "MXFP4 transport is w8a8smooth-only"
        )

    def parse_override(spec, config_type):
        parsed = {}
        if not spec:
            return parsed

        field_names = {field.name for field in fields(config_type)}
        for item in spec.split(","):
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name not in field_names:
                raise ValueError(f"unknown {config_type.__name__} field {name!r}")
            if value.lower() in ("true", "false"):
                parsed[name] = value.lower() == "true"
            else:
                parsed[name] = int(value)
        return parsed

    from kernels.mega_moe.mega_moe_config import Stage1Config, Stage2Config

    stage1_override = parse_override(args.stage1_override, Stage1Config)
    stage2_override = parse_override(args.stage2_override, Stage2Config)

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
            None if args.dispatch_quant == "int8" else args.dispatch_quant,
            stage1_override,
            stage2_override,
            args.hot_route_fraction,
            args.debug_replay_stages,
            args.mtpr or tokens,
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
def test_m13_w8a8smooth_mxfp4_prefill_8gpu_acceptance():
    """Gate M13 MXFP4-communication prefill correctness and latency."""
    _require_gfx95(WORLD)
    for tokens in MXFP4_PREFILL_E2E_US:
        _run_subprocess(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={WORLD}",
                os.path.abspath(__file__),
                "--mode",
                "w8a8smooth",
                "--dispatch-quant",
                "mxfp4",
                "--mtpr",
                str(MXFP4_PREFILL_MTPR),
                "--bs-list",
                str(tokens),
                "--iters",
                os.environ.get("MEGAMOE_M13_MXFP4_CI_ITERS", "20"),
                "--measure-perf",
                "--strict",
            ],
            timeout=300,
            extra_env={"MORI_SHMEM_HEAP_SIZE": "64G"},
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
