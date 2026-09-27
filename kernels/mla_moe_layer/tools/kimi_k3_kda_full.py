# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and graph-latency harness for Kimi-K3 TP8 KDA decode."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernels.common.mx_formats import (  # noqa: E402
    dequantize_mxfp8,
    quant_dequant_mxfp8,
    quantize_mxfp8,
)
from kernels.mla_moe_layer.config import (  # noqa: E402
    EPS,
    KIMI_K3_CONFIG,
    MAX_LAYERS_PER_STEP,
    MoeMode,
)
from kernels.mla_moe_layer.kda import KimiK3KdaAttention  # noqa: E402
from kernels.mla_moe_layer.kimi_k3 import KimiK3KdaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.reference import (  # noqa: E402
    LayerWeights,
    golden_kimi_k3_kda_attention,
    golden_kimi_k3_kda_layer,
    golden_kimi_k3_moe,
    make_weights,
)
from kernels.mla_moe_layer.torch_fusions import situ  # noqa: E402


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
    layer_type = KimiK3KdaAttention if args.attention_only else KimiK3KdaMoeLayer
    if args.attention_only:
        layer = layer_type(
            weights,
            args.samples,
            rank=rank,
            npes=args.npes,
            group=dist.group.WORLD,
            reduce_group=reduce_group,
            reduce_backend=args.reduce_backend,
        )
    else:
        layer = layer_type(
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
    slots = args.samples + 3
    state_indices = torch.arange(args.samples, device=device, dtype=torch.int32)
    if args.samples > 1:
        state_indices.copy_(torch.roll(state_indices, 1) + 1)
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
        "layer_idx": args.layer_idx,
        "reduce_backend": args.reduce_backend,
        "rank_equal": rank_equal,
        "finite": bool(torch.isfinite(output).all()),
    }
    if args.negative_slot and args.attention_only:
        result["negative_output_zero"] = bool(torch.count_nonzero(output[-1]) == 0)

    if args.check:
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
            )
            moe_reference = golden_kimi_k3_moe(
                reference_weights,
                layer.moe_input.clone(),
                lambda value: _allreduce_reference(value, args.npes),
                projection_states=quant_dequant_mxfp8(layer.moe_input).to(torch.bfloat16),
            )
            output_reference = (layer.updated_prefix.float() + moe_reference["moe_delta"].float()).to(torch.bfloat16)
            routed_reduced_reference = _allreduce_reference(layer.routed_partial, args.npes)
            routed_norm_reference = (
                routed_reduced_reference.float()
                * torch.rsqrt(routed_reduced_reference.float().square().mean(-1, keepdim=True) + EPS)
                * layer.t["g_latent"].float()
            ).to(torch.bfloat16)
            shared_mid_reference = situ(layer.shared_gu, config.situ_beta, config.situ_linear_beta)
            result.update(
                pre_attn_rel_l2=_relative_l2(layer.pre_attn, reference["pre_attn"]),
                attention_rel_l2=_relative_l2(layer.attention_delta, reference["attention_delta"]),
                conv_state_rel_l2=_relative_l2(conv_state, reference_conv),
                recurrent_state_rel_l2=_relative_l2(recurrent_state, reference_recurrent),
                moe_input_rel_l2=_relative_l2(layer.moe_input, reference["moe_input"]),
                latent_mxfp8_rel_l2=_relative_l2(layer.latent, moe_reference["latent"]),
                routed_rel_l2=_relative_l2(layer.routed_partial, moe_reference["routed_partial"]),
                routed_reduce_rel_l2=_relative_l2(layer.routed_reduced, routed_reduced_reference),
                routed_norm_rel_l2=_relative_l2(layer.latent_norm, routed_norm_reference),
                shared_mid_rel_l2=_relative_l2(layer.shared_mid, shared_mid_reference),
                selection_equal=bool(torch.equal(layer.topk_ids, moe_reference["sel"])),
                output_rel_l2=_relative_l2(output, output_reference),
                e2e_output_rel_l2=_relative_l2(output, reference["x_out"]),
            )

    if args.profile and not args.attention_only:
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
        )
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

    results[rank] = result
    if rank == 0:
        payload = {
            "model": "kimi_k3",
            "npes": args.npes,
            "samples": args.samples,
            **result,
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
    parser.add_argument("--samples", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention-only", action="store_true")
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
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.layer_idx == 0 and not args.attention_only:
        parser.error("layer 0 uses the out-of-scope dense FFN; choose a KDA MoE layer")
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
    if args.check:
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
                    and result["routed_reduce_rel_l2"] < 0.001
                    and result["routed_norm_rel_l2"] < 0.001
                    and result["output_rel_l2"] < 0.08
                )
            )
            for result in results.values()
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
