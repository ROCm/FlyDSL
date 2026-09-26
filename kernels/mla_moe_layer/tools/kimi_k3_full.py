# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness and graph-latency harness for the full Kimi-K3 TP8 layer."""

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

from kernels.mla_moe_layer.config import KIMI_K3_CONFIG, MAX_LAYERS_PER_STEP, MoeMode  # noqa: E402
from kernels.mla_moe_layer.kimi_k3 import KimiK3MlaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.reference import (  # noqa: E402
    golden_kimi_k3_layer,
    golden_kimi_k3_moe,
    make_weights,
    rope_table,
)


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
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.npes)
    reduce_group = dist.new_group(ranks=list(range(args.npes)), backend="nccl")
    config = KIMI_K3_CONFIG
    weights = make_weights(
        rank,
        heads=config.local_heads,
        device=device,
        seed=args.seed,
        moe_mode=MoeMode.A16W4,
        model_config=config,
        attention_only=False,
        npes=args.npes,
    )
    layer = KimiK3MlaMoeLayer(
        weights,
        args.samples,
        layer_idx=args.layer_idx,
        rank=rank,
        npes=args.npes,
        group=dist.group.WORLD,
        reduce_group=reduce_group,
        topk=args.cache_topk,
        fuse_attn_res=not args.eager_attn_res,
        fuse_router=not args.eager_router,
        fuse_shared_experts=not args.eager_shared_experts,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 99)
    prefix = torch.randn(args.samples, config.hidden, generator=generator, device=device).bfloat16()
    num_blocks = args.layer_idx // config.attn_res_block_size + 1
    blocks0 = torch.randn(
        args.samples,
        num_blocks,
        config.hidden,
        generator=generator,
        device=device,
    ).bfloat16()
    kv0 = torch.randn(args.max_seq, config.kv_lora, generator=generator, device=device).bfloat16()
    pe0 = torch.randn(args.max_seq, config.pe_dim, generator=generator, device=device).bfloat16()
    indices = torch.stack(
        [
            torch.randperm(max(args.pos + sample + 1, args.cache_topk), generator=generator, device=device)[
                : args.cache_topk
            ]
            .sort()
            .values
            for sample in range(args.samples)
        ]
    ).int()
    if args.pos + 1 > args.cache_topk:
        indices[:, -1] = torch.arange(args.pos, args.pos + args.samples, device=device, dtype=torch.int32)
    cos, sin = rope_table(args.max_seq, device=device, model_config=config)
    pos = torch.tensor([args.pos], dtype=torch.int32, device=device)
    output = torch.empty_like(prefix)

    blocks = blocks0.clone()
    kv, pe = kv0.clone(), pe0.clone()
    layer.forward(prefix, blocks, pos, kv, pe, indices, cos, sin, x_out=output)
    torch.cuda.synchronize()

    peers = [torch.empty_like(output.cpu()) for _ in range(args.npes)]
    dist.all_gather(peers, output.cpu().contiguous())
    rank_equal = all(torch.equal(peers[0], peer) for peer in peers[1:])

    stage_tensors = {
        "pre_attn": layer.pre_attn,
        "attention_delta": layer.attention_delta,
        "moe_input": layer.moe_input,
        "router_scores": layer.router_scores,
        "latent": layer.latent,
        "routed_partial": layer.routed_partial,
        "routed_reduced": layer.routed_reduced,
        "latent_norm": layer.latent_norm,
        "shared_mid": layer.shared_mid,
        "shared_partial": layer.shared_partial,
        "tail": layer.tail,
        "final_partial": layer.final_partial,
        "moe_delta": layer.moe_delta,
        "output": output,
    }
    stage_health = {
        name: {
            "finite": bool(torch.isfinite(value).all()),
            "absmax": float(torch.nan_to_num(value.float()).abs().max()),
        }
        for name, value in stage_tensors.items()
    }
    result = {
        "rank_equal": rank_equal,
        "finite": bool(torch.isfinite(output).all()),
        "stage_health": stage_health,
    }
    if args.check:
        ref_blocks = blocks0.clone()
        ref_kv, ref_pe = kv0.clone(), pe0.clone()
        reference = golden_kimi_k3_layer(
            weights,
            prefix,
            ref_blocks,
            args.pos,
            ref_kv,
            ref_pe,
            indices,
            cos,
            sin,
            lambda value: _allreduce_reference(value, args.npes),
            layer_idx=args.layer_idx,
            topk=args.cache_topk,
        )
        # Validate MoE from the implementation's own post-attention state.  A
        # small legal MLA accumulation difference can otherwise flip a near-tied
        # synthetic router decision and obscure whether the MoE path is correct.
        moe_reference = golden_kimi_k3_moe(
            weights,
            layer.moe_input.clone(),
            lambda value: _allreduce_reference(value, args.npes),
        )
        output_reference = (layer.updated_prefix.float() + moe_reference["moe_delta"].float()).to(torch.bfloat16)
        result.update(
            pre_attn_rel_l2=_relative_l2(layer.pre_attn, reference["pre_attn"]),
            attention_rel_l2=_relative_l2(layer.attention_delta, reference["attention_delta"]),
            moe_input_rel_l2=_relative_l2(layer.moe_input, reference["moe_input"]),
            router_rel_l2=_relative_l2(layer.router_scores, reference["scores"]),
            output_rel_l2=_relative_l2(output, output_reference),
            routed_rel_l2=_relative_l2(layer.routed_partial, moe_reference["routed_partial"]),
            selection_equal=bool(torch.equal(layer.topk_ids, moe_reference["sel"])),
            selection_mismatches=int((layer.topk_ids != moe_reference["sel"]).sum()),
            e2e_output_rel_l2=_relative_l2(output, reference["x_out"]),
            e2e_selection_equal=bool(torch.equal(layer.topk_ids, reference["sel"])),
            e2e_selection_mismatches=int((layer.topk_ids != reference["sel"]).sum()),
            kv_rel_l2=_relative_l2(
                kv[args.pos : args.pos + args.samples],
                ref_kv[args.pos : args.pos + args.samples],
            ),
        )

    if args.profile:
        blocks.copy_(blocks0)
        kv.copy_(kv0)
        pe.copy_(pe0)
        torch.cuda.synchronize()
        dist.barrier()
        layer.start_stage_profile()
        for _ in range(args.profile_repeats):
            layer.forward(prefix, blocks, pos, kv, pe, indices, cos, sin, x_out=output)
        local_profile = layer.finish_stage_profile()
        gathered_profiles = [None] * args.npes
        dist.all_gather_object(gathered_profiles, local_profile)
        result["stage_profile_us"] = {
            name: max(profile[name] for profile in gathered_profiles) for name in local_profile
        }

    if args.bench:
        for _ in range(2):
            blocks.copy_(blocks0)
            layer.forward(prefix, blocks, pos, kv, pe, indices, cos, sin, x_out=output)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with layer.capture():
            with torch.cuda.graph(graph):
                for epoch in range(args.layers):
                    layer.forward(
                        prefix,
                        blocks,
                        pos,
                        kv,
                        pe,
                        indices,
                        cos,
                        sin,
                        x_out=output,
                        epoch_layer=epoch,
                        advance=False,
                    )
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

    results[rank] = result
    if rank == 0:
        payload = {"model": "kimi_k3", "npes": args.npes, "samples": args.samples, **result}
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
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--pos", type=int, default=3000)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--cache-topk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--profile", action="store_true", help="report median eager GPU stage timings")
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--eager-attn-res", action="store_true", help="use the unfused Torch AttnRes path")
    parser.add_argument("--eager-router", action="store_true", help="use the unfused Torch router path")
    parser.add_argument(
        "--eager-shared-experts",
        action="store_true",
        help="use the unfused Torch shared-expert path",
    )
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", help="write rank-0 JSON results to this path")
    args = parser.parse_args()
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
            result["selection_equal"]
            and result["routed_rel_l2"] < 0.02
            and result["output_rel_l2"] < 0.08
            and result["kv_rel_l2"] < 0.03
            for result in results.values()
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
