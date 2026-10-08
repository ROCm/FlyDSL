# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Validate and benchmark the Kimi-K3 TP8 decode MonoKernel."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig
from kernels.kimi_k3_monokernel.kda import KimiK3KdaAttention  # noqa: E402
from kernels.kimi_k3_monokernel.kernel import monokernel_layout  # noqa: E402
from kernels.kimi_k3_monokernel.op import KimiK3MonoKernel  # noqa: E402
from kernels.kimi_k3_monokernel.staged import _KimiK3KdaStagedPath  # noqa: E402
from kernels.kimi_k3_monokernel.routed_pipeline import SOFT_FIELDS  # noqa: E402
from kernels.kimi_k3_monokernel.tools.tail_check import check_routed_pipeline, check_tail_reduction  # noqa: E402
from kernels.kimi_k3_monokernel.torch_fusions import situ  # noqa: E402
from kernels.monokernel.config import (  # noqa: E402
    EPS,
    KIMI_K3_CONFIG,
    MAX_LAYERS_PER_STEP,
    MoeMode,
)
from kernels.monokernel.formats import (  # noqa: E402
    dequantize_mxfp8,
    quant_dequant_mxfp8,
    quantize_mxfp8,
)
from kernels.monokernel.reference import (  # noqa: E402
    golden_kimi_k3_kda_attention,
    golden_kimi_k3_kda_layer,
    golden_kimi_k3_moe,
    make_weights,
    route,
)
from kernels.monokernel.weights import LayerWeights  # noqa: E402


def _allreduce_reference(value: torch.Tensor, world_size: int) -> torch.Tensor:
    parts = [torch.empty_like(value.cpu()) for _ in range(world_size)]
    dist.all_gather(parts, value.cpu().contiguous())
    total = parts[0].float()
    for part in parts[1:]:
        total.add_(part.float())
    return total.to(torch.bfloat16).to(value.device)


def _relative_l2(got: torch.Tensor, expected: torch.Tensor) -> float:
    delta = got.float() - expected.float()
    return float(delta.norm() / expected.float().norm().clamp_min(1e-12))


def _worker(rank: int, args, port: int, results) -> None:
    if args.dump_ir_dir:
        rank_dump_dir = Path(args.dump_ir_dir) / f"rank-{rank}"
        rank_dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(rank_dump_dir)
        os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.npes,
    )
    reduce_group = dist.new_group(ranks=list(range(args.npes)), backend="nccl")
    config = KIMI_K3_CONFIG
    weights = make_weights(
        rank,
        heads=config.local_heads,
        device=device,
        seed=args.seed,
        moe_mode=MoeMode.A16W4,
        model_config=config,
        attention_only=args.attention_only,
        npes=args.npes,
        attention_family="kda",
    )
    if args.attention_only:
        layer = KimiK3KdaAttention(
            weights,
            args.samples,
            rank=rank,
            npes=args.npes,
            group=dist.group.WORLD,
            reduce_group=reduce_group,
            reduce_backend=args.reduce_backend,
            mtp=args.mtp,
            seq_len=args.seq,
            compile_config=KimiK3CompileConfig(path=args.compile_path, input_mfma=args.input_mfma, input_schedule=args.input_schedule, gate_schedule=args.gate_schedule, route_publication=args.route_publication, output_prefetch_units=None if args.output_prefetch_units == 'auto' else int(args.output_prefetch_units)),
        )
    elif args.staged:
        layer = _KimiK3KdaStagedPath(
            weights,
            args.samples,
            layer_idx=args.layer_idx,
            rank=rank,
            npes=args.npes,
            group=dist.group.WORLD,
            reduce_group=reduce_group,
            fuse_attn_res=not args.eager_attn_res,
            fuse_router=not args.eager_router,
            fuse_shared_experts=not args.eager_shared_experts,
            reduce_backend=args.reduce_backend,
            mtp=args.mtp,
            seq_len=args.seq,
            compile_config=KimiK3CompileConfig(path=args.compile_path, input_mfma=args.input_mfma, input_schedule=args.input_schedule, gate_schedule=args.gate_schedule, route_publication=args.route_publication, output_prefetch_units=None if args.output_prefetch_units == 'auto' else int(args.output_prefetch_units)),
            routed_pipeline=args.routed_pipeline,
            routed_producers=args.routed_producers,
            routed_prefetch=args.routed_prefetch,
            routed_grid=args.routed_grid,
            routed_sample_group=args.routed_sample_group,
            routed_down_tile=args.routed_down_tile,
            routed_tp_mode=args.routed_tp_mode,
            routed_up_order=args.routed_up_order,
            routed_soft_profile=args.routed_soft_profile,
            routed_route_prefetch=args.routed_route_prefetch,
        )
    else:
        layer = KimiK3MonoKernel(
            weights,
            args.samples,
            layer_idx=args.layer_idx,
            rank=rank,
            npes=args.npes,
            group=dist.group.WORLD,
            reduce_group=reduce_group,
            mtp=args.mtp,
            seq_len=args.seq,
            compile_config=KimiK3CompileConfig(path=args.compile_path, input_mfma=args.input_mfma, input_schedule=args.input_schedule, gate_schedule=args.gate_schedule, route_publication=args.route_publication, output_prefetch_units=None if args.output_prefetch_units == 'auto' else int(args.output_prefetch_units)),
        )

    generator = torch.Generator(device=device).manual_seed(args.seed + 99)
    prefix = torch.randn(
        args.samples,
        config.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    num_blocks = args.layer_idx // config.attn_res_block_size + 1
    blocks0 = torch.randn(
        args.samples,
        num_blocks,
        config.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    slot_stride = 7 if args.full_replay_check else args.seq + 3
    slots = args.batch * slot_stride
    state_count = args.batch * (args.seq + 1) if args.mtp else args.samples
    state_indices = torch.tensor([b * slot_stride + t for b in range(args.batch) for t in range(args.seq + 1)], device=device, dtype=torch.int32) if args.mtp else torch.arange(state_count, device=device,dtype=torch.int32)
    if args.samples > 1 and not args.mtp:
        state_indices.copy_(torch.roll(state_indices, 1) + 1)
    if args.negative_slot and args.mtp:
        raise ValueError("--negative-slot is only supported for independent decode samples")
    if args.negative_slot:
        state_indices[-1] = -1
    conv_state0 = torch.randn(
        slots,
        3 * config.local_heads * config.v_dim,
        3,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    recurrent_state0 = (
        torch.randn(
            slots,
            config.local_heads,
            config.v_dim,
            config.v_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    conv_state = conv_state0.clone()
    recurrent_state = recurrent_state0.clone()
    blocks = blocks0.clone()
    output = torch.empty_like(prefix)

    def run_layer(*, epoch_layer: int = 0, advance: bool = True):
        if args.attention_only:
            return layer.forward(
                prefix,
                state_indices,
                conv_state,
                recurrent_state,
                x_out=output,
                layer=epoch_layer,
                advance=advance,
            )
        return layer.forward(
            prefix,
            blocks,
            state_indices,
            conv_state,
            recurrent_state,
            x_out=output,
            epoch_layer=epoch_layer,
            advance=advance,
        )

    run_layer()
    torch.cuda.synchronize()

    peers = [torch.empty_like(output.cpu()) for _ in range(args.npes)]
    dist.all_gather(peers, output.cpu().contiguous())
    rank_equal = all(torch.equal(peers[0], peer) for peer in peers[1:])
    result = {
        "attention_family": "kda",
        "attention_only": args.attention_only,
        "launch_mode": "attention-only" if args.attention_only else ("staged" if args.staged else "single"),
        "layer_idx": args.layer_idx,
        "reduce_backend": args.reduce_backend,
        "mtp": args.mtp,
        "batch": args.batch,
            "compile_path": args.compile_path,
        "seq": args.seq,
        "routed_pipeline": args.routed_pipeline,
        "routed_tp_mode": args.routed_tp_mode if args.routed_pipeline else None,
        "benchmark_scope": "moe" if args.moe_only_bench else "layer",
        "instrumented": bool(args.routed_soft_profile),
        "rank_equal": rank_equal,
        "finite": bool(torch.isfinite(output).all()),
    }
    if args.negative_slot and args.attention_only:
        result["negative_output_zero"] = bool(torch.count_nonzero(output[-1]) == 0)
    if args.routed_pipeline:
        pipeline = layer.routed_pipeline
        result["pipeline_config"] = {
            "grid": pipeline.grid,
            "producers": pipeline.producers,
            "prefetch": pipeline.prefetch,
            "down_tile": pipeline.down_tile,
            "up_order": pipeline.up_order,
            "route_prefetch": pipeline.route_prefetch,
            "tp_mode": pipeline.tp_mode,
        }

    if args.check:
        if args.mtp:
            slot = int(state_indices[0])
            result["incoming_conv_unchanged"] = torch.equal(conv_state[slot], conv_state0[slot])
            result["incoming_recurrent_unchanged"] = torch.equal(recurrent_state[slot], recurrent_state0[slot])
        reference_tensors = weights.t.copy()
        if not args.attention_only:
            quantized_names = ["w_latent_down", "w_shared_ug"]
            if layer.fused_tail is not None:
                quantized_names += ["w_shared_dn", "w_latent_up"]
            for name in quantized_names:
                quantized, scale = quantize_mxfp8(reference_tensors[name])
                reference_tensors[name] = dequantize_mxfp8(quantized, scale).to(torch.bfloat16)
        reference_weights = LayerWeights(
            weights.heads,
            reference_tensors,
            weights.config,
            weights.rank,
            weights.npes,
        )
        reference_conv = conv_state0.clone()
        reference_recurrent = recurrent_state0.clone()
        if args.attention_only:
            reference = golden_kimi_k3_kda_attention(
                reference_weights,
                prefix,
                state_indices,
                reference_conv,
                reference_recurrent,
                lambda value: _allreduce_reference(value, args.npes),
                mtp=args.mtp,
                seq_len=args.seq,
            )
            result.update(
                attention_rel_l2=_relative_l2(output, reference["output"]),
                conv_state_rel_l2=_relative_l2(conv_state, reference_conv),
                recurrent_state_rel_l2=_relative_l2(recurrent_state, reference_recurrent),
            )
        else:
            reference = golden_kimi_k3_kda_layer(
                reference_weights,
                prefix,
                blocks0.clone(),
                state_indices,
                reference_conv,
                reference_recurrent,
                lambda value: _allreduce_reference(value, args.npes),
                layer_idx=args.layer_idx,
                mtp=args.mtp,
                seq_len=args.seq,
            )
            projection_states = quant_dequant_mxfp8(layer.moe_input).to(torch.bfloat16)
            expected_activation, expected_activation_scale = quantize_mxfp8(layer.moe_input)
            actual_activation = layer.latent_projection.activation
            actual_scale = layer.latent_projection.activation_scale
            expected_padded_scale = torch.zeros(
                layer.latent_projection.padded_rows,
                config.hidden // 32,
                dtype=torch.uint8,
                device=device,
            )
            expected_padded_scale[: args.samples] = expected_activation_scale
            expected_packed_scale = (
                expected_padded_scale.reshape(
                    layer.latent_projection.padded_rows // 32,
                    2,
                    16,
                    (config.hidden // 32) // 8,
                    2,
                    4,
                )
                .permute(0, 3, 5, 2, 4, 1)
                .contiguous()
                .view(-1)
            )
            moe_reference = golden_kimi_k3_moe(
                reference_weights,
                layer.moe_input.clone(),
                lambda value: _allreduce_reference(value, args.npes),
                projection_states=projection_states,
            )
            output_reference = (layer.updated_prefix.float() + moe_reference["moe_delta"].float()).to(torch.bfloat16)
            if layer.attention.fuse_moe:
                shared_gu_reference = (projection_states.float() @ reference_weights.t["w_shared_ug"].float().T).to(
                    torch.bfloat16
                )
            else:
                shared_gu_reference = layer.shared_gu
            shared_mid_reference = situ(
                shared_gu_reference,
                config.situ_beta,
                config.situ_linear_beta,
            )
            if layer.attention.fuse_moe:
                scratch_layout = monokernel_layout(
                    args.samples,
                    fuse_attn_res=True,
                    fuse_moe=True,
                )
                scratch = layer.attention.monokernel_scratch

                def tagged_f32(name: str, count: int) -> torch.Tensor:
                    offset = scratch_layout[name]
                    return (
                        scratch[offset : offset + count * 8].view(torch.int32).view(count, 2)[:, 0].view(torch.float32)
                    )

                def tagged_bf16(name: str, count: int) -> torch.Tensor:
                    offset = scratch_layout[name]
                    packed = (
                        scratch[offset : offset + (count // 2) * 8]
                        .view(torch.int32)
                        .view(count // 2, 2)[:, 0]
                        .contiguous()
                    )
                    return packed.view(torch.bfloat16).view(count)

                def raw_bf16(name: str, count: int) -> torch.Tensor:
                    offset = scratch_layout[name]
                    return scratch[offset : offset + count * 2].view(torch.bfloat16)

                fused_ids = (
                    tagged_f32("selection_id", args.samples * config.top_k)
                    .view(torch.int32)
                    .view(args.samples, config.top_k)
                )
                fused_weights = tagged_f32("selection_weight", args.samples * config.top_k).view(
                    args.samples, config.top_k
                )
                fused_mxfp8 = actual_activation[: args.samples].view(-1)
                fused_latent = raw_bf16("latent", args.samples * config.routed_hidden).view(
                    args.samples, config.routed_hidden
                )
                fused_routed = raw_bf16("routed", args.samples * config.routed_hidden).view(
                    args.samples, config.routed_hidden
                )
                fused_routed_inv = tagged_f32("routed_inv", args.samples).view(args.samples, 1)
                fused_routed_norm = (fused_routed.float() * fused_routed_inv * layer.t["g_latent"].float()).to(
                    torch.bfloat16
                )
                fused_mid = raw_bf16("expert_mid", args.samples * config.top_k * config.inter).view(
                    args.samples, config.top_k, config.inter
                )
                fused_shared_mid = tagged_bf16("shared_mid", args.samples * config.shared_inter).view(
                    args.samples, config.shared_inter
                )
                routed_reduced_reference = moe_reference["routed_reduced"]
            else:
                fused_ids = layer.topk_ids
                fused_weights = layer.topk_weights
                fused_mxfp8 = actual_activation[: args.samples].view(-1)
                fused_latent = layer.latent
                fused_routed = layer.routed_reduced
                fused_routed_norm = layer.latent_norm
                fused_mid = (
                    layer.routed_pipeline.mid[..., 0]
                    .contiguous()
                    .view(torch.bfloat16)
                    .reshape(args.samples, config.top_k, config.inter)
                    if args.routed_pipeline
                    else moe_reference["mid"]
                )
                fused_shared_mid = layer.shared_mid
                routed_reduced_reference = _allreduce_reference(layer.routed_partial, args.npes)
            routed_norm_reference = (
                routed_reduced_reference.float()
                * torch.rsqrt(routed_reduced_reference.float().square().mean(-1, keepdim=True) + EPS)
                * layer.t["g_latent"].float()
            ).to(torch.bfloat16)
            result.update(
                pre_attn_rel_l2=_relative_l2(layer.pre_attn, reference["pre_attn"]),
                attention_rel_l2=_relative_l2(layer.attention_delta, reference["attention_delta"]),
                conv_state_rel_l2=_relative_l2(conv_state, reference_conv),
                recurrent_state_rel_l2=_relative_l2(recurrent_state, reference_recurrent),
                moe_input_rel_l2=_relative_l2(layer.moe_input, reference["moe_input"]),
                mxfp8_activation_equal=bool(
                    torch.equal(
                        actual_activation[: args.samples],
                        expected_activation.view(torch.uint8),
                    )
                ),
                mxfp8_scale_equal=bool(torch.equal(actual_scale, expected_packed_scale)),
                mxfp8_mailbox_mismatches=int(
                    torch.count_nonzero(fused_mxfp8 != actual_activation[: args.samples].view(-1))
                ),
                latent_mxfp8_rel_l2=_relative_l2(fused_latent, moe_reference["latent"]),
                selection_weight_rel_l2=_relative_l2(fused_weights, moe_reference["prob"]),
                expert_mid_rel_l2=_relative_l2(fused_mid, moe_reference["mid"]),
                routed_rel_l2=_relative_l2(fused_routed, moe_reference["routed_reduced"]),
                routed_reduce_rel_l2=_relative_l2(fused_routed, routed_reduced_reference),
                routed_norm_rel_l2=_relative_l2(fused_routed_norm, routed_norm_reference),
                shared_mid_rel_l2=_relative_l2(fused_shared_mid, shared_mid_reference),
                selection_equal=bool(torch.equal(fused_ids, moe_reference["sel"])),
                unique_routed_experts=int(fused_ids.unique().numel()),
                output_rel_l2=_relative_l2(output, output_reference),
            )

    def check_staged_tail():
        checks = check_tail_reduction(layer, output, args.npes)
        expected_scores = torch.sigmoid((layer.moe_input @ layer.t["w_r"].T).float())
        expected_ids = torch.stack([route(row, layer.t["bias"], config)[0] for row in expected_scores])
        checks["selection_equal"] = torch.equal(layer.topk_ids, expected_ids.to(torch.int32))
        if args.mtp:
            slot = int(state_indices[0])
            checks["incoming_conv_unchanged"] = torch.equal(conv_state[slot], conv_state0[slot])
            checks["incoming_recurrent_unchanged"] = torch.equal(recurrent_state[slot], recurrent_state0[slot])
        if args.routed_pipeline:
            checks.update(check_routed_pipeline(layer, args.npes))
        return checks

    if args.check and args.staged and not args.attention_only and layer.fused_tail is not None:
        result.update(check_staged_tail())
        if args.layer_idx % config.attn_res_block_size != 0:
            result["attnres_delta_equal"] = torch.equal(
                layer.updated_prefix, (prefix.float() + layer.attention_delta.float()).to(torch.bfloat16)
            )

    if args.profile and not args.attention_only and not layer.attention.fuse_moe:
        blocks.copy_(blocks0)
        conv_state.copy_(conv_state0)
        recurrent_state.copy_(recurrent_state0)
        torch.cuda.synchronize()
        dist.barrier()
        layer.start_stage_profile()
        for _ in range(args.profile_repeats):
            run_layer()
        local_profile = layer.finish_stage_profile()
        gathered_profiles = [None] * args.npes
        dist.all_gather_object(gathered_profiles, local_profile)
        result["stage_profile_us"] = {
            name: max(profile[name] for profile in gathered_profiles) for name in local_profile
        }
    elif (
        args.profile
        and (layer.monokernel_timeline if args.attention_only else layer.attention.monokernel_timeline) is not None
    ):
        timeline = layer.monokernel_timeline if args.attention_only else layer.attention.monokernel_timeline
        labels = ["pre_and_input", "recurrence_wait", "output_projection"]
        if not args.attention_only:
            labels += [
                "post_attn_res",
                "dense_projections",
                "expert_up",
                "expert_down",
                "routed_norm",
                "tail",
            ]
        profile_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(profile_graph):
            run_layer(advance=False)
            layer.advance_step()
        samples_by_stage = {label: [] for label in labels}
        last_ticks = []
        for _ in range(args.profile_repeats):
            torch.cuda.synchronize()
            dist.barrier()
            profile_graph.replay()
            torch.cuda.synchronize()
            last_ticks = timeline.cpu().tolist()[: len(labels) + 1]
            for index, label in enumerate(labels):
                samples_by_stage[label].append((last_ticks[index + 1] - last_ticks[index]) / 100.0)
        local_runs = [
            {label: samples_by_stage[label][index] for label in labels} for index in range(args.profile_repeats)
        ]
        gathered_runs = [None] * args.npes
        dist.all_gather_object(gathered_runs, local_runs)
        critical_runs = []
        for index in range(args.profile_repeats):
            critical_runs.append(
                max(
                    (rank_runs[index] for rank_runs in gathered_runs),
                    key=lambda run: sum(run.values()),
                )
            )
        critical_runs.sort(key=lambda run: sum(run.values()))
        result["stage_profile_us"] = critical_runs[len(critical_runs) // 2]
        result["stage_profile_sum_us"] = sum(result["stage_profile_us"].values())
        result["last_timeline_ticks"] = last_ticks

    if args.bench:
        for _ in range(2):
            blocks.copy_(blocks0)
            conv_state.copy_(conv_state0)
            recurrent_state.copy_(recurrent_state0)
            run_layer()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for epoch in range(args.layers):
                if args.moe_only_bench:
                    layer._moe(layer.moe_input, epoch, layer.updated_prefix, output)
                else:
                    run_layer(epoch_layer=epoch, advance=False)
            layer.advance_step()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0 / args.layers)
        gathered = [None] * args.npes
        dist.all_gather_object(gathered, times)
        critical = [max(values) for values in zip(*gathered)]
        result.update(
            median_us=statistics.median(critical),
            min_us=min(critical),
            max_us=max(critical),
            layers=args.layers,
            repeats=args.repeats,
            critical_times_us=critical,
        )
        if args.check and args.staged and not args.attention_only and layer.fused_tail is not None:
            result.update({f"replay_{key}": value for key, value in check_staged_tail().items()})
        if args.kernel_profile:
            dist.barrier()
            if rank == 0:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    graph.replay()
                    torch.cuda.synchronize()
                events = [event for event in profiler.key_averages() if event.self_device_time_total > 0]
                events.sort(key=lambda event: event.self_device_time_total, reverse=True)
                result["kernel_profile"] = [
                    {
                        "name": event.key,
                        "calls": event.count,
                        "total_us": event.self_device_time_total,
                        "mean_us": event.self_device_time_total / event.count,
                    }
                    for event in events
                ]
            else:
                graph.replay()
                torch.cuda.synchronize()
            dist.barrier()

    if args.full_replay_check:
        from full_replay import run as check_full_replay
        result.update(check_full_replay(layer, reference_weights, prefix, blocks, state_indices,
                                       conv_state, recurrent_state, output, args))

    if args.routed_soft_profile:
        soft_dir = Path(args.soft_output_dir)
        soft_dir.mkdir(parents=True, exist_ok=True)
        (soft_dir / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "mode": args.routed_soft_profile,
                    "fields": SOFT_FIELDS,
                    "counter": "s_memrealtime",
                    "ticks_per_us": 100,
                    "last_graph_layer": args.layers - 1,
                    "scope": result["benchmark_scope"],
                    "pipeline_config": result["pipeline_config"],
                    "durations": layer.routed_pipeline.soft_durations.cpu().tolist(),
                }
            )
            + "\n"
        )

    results[rank] = result
    all_results = [None] * args.npes
    dist.all_gather_object(all_results, result)
    if rank == 0:
        payload = {
            "model": "kimi_k3",
            "npes": args.npes,
            "samples": args.samples,
            "batch": args.batch,
            "compile_path": args.compile_path,
            "seq": args.seq,
            **result,
            "all_ranks": all_results,
        }
        print(json.dumps(payload), flush=True)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n")
    layer.close()
    dist.barrier()
    dist.destroy_process_group(reduce_group)
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npes", type=int, choices=(8,), default=8)
    parser.add_argument("--samples", type=int, choices=range(1, 33), default=None)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--staged", action="store_true", help="benchmark the fastest staged KDA + MoE path")
    parser.add_argument("--routed-pipeline", action="store_true")
    parser.add_argument("--routed-producers", type=int)
    parser.add_argument("--routed-prefetch", type=int)
    parser.add_argument("--routed-grid", type=int, choices=(256, 512, 768), default=256)
    parser.add_argument(
        "--routed-route-prefetch",
        action="store_true",
        help="reuse metadata across three K128 chunks; requires S4 and grid/producers 512/256 or 768/544",
    )
    parser.add_argument("--routed-sample-group", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--routed-down-tile", type=int, choices=(0, 16, 32, 64), default=0)
    parser.add_argument("--routed-tp-mode", choices=("complete", "reduce_scatter"), default="complete")
    parser.add_argument("--routed-up-order", choices=("sample", "interleaved"), default="sample")
    parser.add_argument("--routed-soft-profile", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--soft-output-dir")
    parser.add_argument(
        "--moe-only-bench", action="store_true", help="time router through tail after the same KDA warmup"
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="treat samples as one ordered MTP group and state_indices as an S+1 snapshot chain",
    )
    parser.add_argument("--negative-slot", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--eager-attn-res", action="store_true")
    parser.add_argument("--eager-router", action="store_true")
    parser.add_argument("--eager-shared-experts", action="store_true")
    parser.add_argument("--reduce-backend", choices=("symmetric", "nccl"), default="symmetric")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument("--dump-ir-dir")
    parser.add_argument("--output")
    parser.add_argument("--full-replay-check", action="store_true")
    parser.add_argument("--compile-path", choices=("auto", "small_batch", "general"), default="auto")
    parser.add_argument("--input-mfma", choices=("auto", "f32", "bf16_k16"), default="auto")
    parser.add_argument("--input-schedule", choices=("auto", "cta", "flat"), default="auto")
    parser.add_argument('--gate-schedule', choices=('auto', 'serial', 'overlap'), default='auto')
    parser.add_argument('--route-publication', choices=('auto', 'stream', 'wave'), default='auto')
    parser.add_argument("--batch", type=int, choices=range(1, 9))
    parser.add_argument("--seq", type=int, choices=(1,2,3,4))
    parser.add_argument('--output-prefetch-units', choices=('auto', '0', '1', '6'), default='auto')
    args = parser.parse_args()
    explicit_shape = args.batch is not None or args.seq is not None
    if explicit_shape:
        args.batch = args.batch or 1
        args.seq = args.seq or 1
        if args.samples is not None and args.samples != args.batch * args.seq:
            parser.error("samples must equal batch * seq")
        args.samples = args.batch * args.seq
        args.mtp = args.mtp or args.seq > 1
    else:
        args.samples = args.samples or 1
        args.batch = 1 if args.mtp else args.samples
        args.seq = args.samples if args.mtp else 1
    if not 1 <= args.batch <= 8 or not 1 <= args.seq <= 4:
        parser.error("batch must be in [1, 8] and seq in [1, 4]")
    if args.full_replay_check and not (args.check and not args.staged and not args.attention_only):
        parser.error("Full replay check requires the complete MonoKernel with --check")
    if args.routed_route_prefetch and not args.routed_pipeline:
        parser.error("--routed-route-prefetch requires --routed-pipeline")
    if args.routed_soft_profile and (not args.routed_pipeline or not args.bench or not args.soft_output_dir):
        parser.error("soft profiling requires --routed-pipeline --bench --soft-output-dir")
    if (args.routed_pipeline or args.moe_only_bench) and (not args.staged or args.attention_only):
        parser.error("routed pipeline and MoE-only timing require --staged full-layer mode")
    if args.moe_only_bench and not args.bench:
        parser.error("--moe-only-bench requires --bench")
    if args.routed_pipeline and (args.reduce_backend != "symmetric" or args.eager_shared_experts):
        parser.error("routed pipeline requires the symmetric fused tail")
    if args.layer_idx == 0 and not args.attention_only:
        parser.error("layer 0 uses the out-of-scope dense FFN; choose a KDA MoE layer")
    if not args.staged and not args.attention_only:
        if args.reduce_backend != "symmetric":
            parser.error("the MonoKernel requires --reduce-backend symmetric")
        if args.eager_attn_res or args.eager_router or args.eager_shared_experts:
            parser.error("eager component switches apply only to --staged")
    if not args.check and not args.bench and not args.profile:
        args.check = True
    if not 1 <= args.layers <= MAX_LAYERS_PER_STEP:
        parser.error(f"--layers must be in [1, {MAX_LAYERS_PER_STEP}]")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(args, port, results), nprocs=args.npes)
    ok = all(result["rank_equal"] and result["finite"] for result in results.values())
    if args.full_replay_check:
        ok = ok and len(results) == args.npes and all(result["full_replay_correct"] for result in results.values())
    if args.check:
        for result in results.values():
            ok = ok and all(
                value
                for key, value in result.items()
                if key.endswith(("_equal", "_unchanged")) and isinstance(value, bool)
            )
            for prefix in ("", "replay_"):
                if args.routed_pipeline:
                    if prefix == "replay_" and not args.bench:
                        continue
                    ok = ok and result[f"{prefix}pipeline_mid_rel_l2"] < 2e-3
                    ok = ok and result[f"{prefix}pipeline_output_rel_l2"] < 1e-2
            if args.mtp:
                ok = ok and result["attention_rel_l2"] < 2e-3
                ok = ok and result["conv_state_rel_l2"] < 5e-4
                ok = ok and result["recurrent_state_rel_l2"] < 5e-4
                if not args.attention_only:
                    ok = ok and result["output_rel_l2"] < 1e-2
        ok = ok and all(
            result["attention_rel_l2"] < 0.02
            and result["conv_state_rel_l2"] < 0.001
            and result["recurrent_state_rel_l2"] < 0.02
            and (
                args.attention_only
                or (
                    result["selection_equal"]
                    and result["latent_mxfp8_rel_l2"] < 0.01
                    and result["routed_rel_l2"] < 0.02
                    and result["routed_reduce_rel_l2"] < 0.01
                    and result["routed_norm_rel_l2"] < 0.01
                    and result["output_rel_l2"] < 0.08
                )
            )
            for result in results.values()
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
