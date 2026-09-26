# SPDX-License-Identifier: Apache-2.0
"""Benchmark ATOM's native GLM-5.1 MXFP4 decoder-layer operator chain.

The FlyDSL mono-kernel consumes preselected sparse-KV indices, so this driver
does the same: it keeps ATOM's native RMSNorm, MLA projections/attention,
router, MXFP4 fused-MoE, and TP collectives, while bypassing only the GLM
indexer that produces those indices.  No ATOM source is modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_CONFIG = Path(__file__).with_name("atom_glm51_mxfp4_config")


def _source_hash(atom_root: Path, aiter_root: Path) -> str:
    digest = hashlib.sha256()
    sources = (
        atom_root / "atom/models/deepseek_v2.py",
        atom_root / "atom/model_ops/attention_mla.py",
        atom_root / "atom/model_ops/moe.py",
        aiter_root / "aiter/ops/attention.py",
    )
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _trimmed_mean(values: list[float], trim_each_tail: int) -> float:
    ordered = sorted(values)
    if trim_each_tail:
        ordered = ordered[trim_each_tail:-trim_each_tail]
    return statistics.mean(ordered)


def _init_weights(layer: torch.nn.Module, rank: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if parameter.dtype == torch.float4_e2m1fn_x2:
                parameter.view(torch.uint8).zero_()
            elif parameter.dtype == torch.uint8:
                # UE8M0 exponent 127 represents scale 1.0.
                parameter.fill_(127)
            elif name.endswith("layernorm.weight") or name.endswith("_norm.weight"):
                parameter.fill_(1)
            elif name.endswith("e_score_correction_bias"):
                parameter.zero_()
            elif parameter.dtype.is_floating_point:
                parameter.normal_(mean=0.0, std=0.01, generator=generator)
            else:
                parameter.zero_()


def _postprocess_weights(layer: torch.nn.Module) -> None:
    from atom.model_ops.base_config import QuantizeMethodBase
    from atom.model_ops.moe import FusedMoEMethodBase

    for module in layer.modules():
        process = getattr(module, "process_weights_after_loading", None)
        if process is not None:
            process()
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)


def _allocate_mla_work(samples: int):
    import aiter

    metadata_info = aiter.get_mla_metadata_info_v1(
        samples,
        1,
        16,
        torch.bfloat16,
        torch.bfloat16,
        is_sparse=True,
        fast_mode=True,
    )
    return [torch.empty(size, dtype=dtype, device="cuda") for size, dtype in metadata_info]


def _make_forward_context(config, layer, samples: int, pos: int, generator):
    import aiter
    from atom.config import KVCacheTensor
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
        set_kv_cache_data,
    )

    topk = config.hf_config.index_topk
    cache_tokens = max(4096, pos + samples)
    kv_cache = torch.randn(
        cache_tokens,
        1,
        config.hf_config.kv_lora_rank + config.hf_config.qk_rope_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(pos, pos + samples, dtype=torch.int64, device="cuda")
    sparse_indices = torch.stack(
        [
            torch.randperm(
                max(pos + sample + 1, topk),
                dtype=torch.int64,
                device="cuda",
                generator=generator,
            )[:topk]
            .sort()
            .values.to(torch.int32)
            for sample in range(samples)
        ]
    )
    if pos >= topk:
        sparse_indices[:, -1] = positions.to(torch.int32)

    cu_seqlens_q = torch.arange(samples + 1, dtype=torch.int32, device="cuda")
    sparse_kv_indptr = torch.arange(0, (samples + 1) * topk, topk, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(samples, dtype=torch.int32, device="cuda")
    context_lens = positions.to(torch.int32) + 1
    work = _allocate_mla_work(samples)
    aiter.get_mla_metadata_v1(
        cu_seqlens_q,
        sparse_kv_indptr,
        kv_last_page_lens,
        16,
        1,
        True,
        work[0],
        work[2],
        work[1],
        work[3],
        work[4],
        work[5],
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=16,
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
    )
    metadata = AttentionMetaData(
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=int(context_lens.max().item()),
        slot_mapping=positions,
        context_lens=context_lens,
        kv_indptr=sparse_kv_indptr,
        kv_indices=sparse_indices.flatten(),
        kv_last_page_lens=kv_last_page_lens,
        sparse_kv_indptr=sparse_kv_indptr,
        work_meta_data=work[0],
        work_indptr=work[1],
        work_info_set=work[2],
        reduce_indptr=work[3],
        reduce_final_map=work[4],
        reduce_partial_map=work[5],
    )
    metadata.dtype_q = torch.bfloat16
    layer.self_attn.mla_attn.impl.sparse_kv_indices_buffer = sparse_indices.flatten()
    set_kv_cache_data(
        {
            "layer_3": KVCacheTensor(
                layer_num=3,
                k_cache=kv_cache,
                v_cache=None,
                k_scale=None,
                v_scale=None,
            )
        },
        config,
    )
    set_forward_context(
        attn_metadata=metadata,
        atom_config=config,
        context=Context(
            positions=positions,
            is_prefill=False,
            batch_size=samples,
            graph_bs=samples,
        ),
        num_tokens=samples,
        in_hipgraph=True,
    )
    return positions, kv_cache, sparse_indices, work


def _worker(rank: int, args, port: int) -> None:
    sys.path[:0] = [str(args.atom_root), str(args.aiter_root)]

    from aiter.dist.parallel_state import (
        destroy_model_parallel,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from atom.config import Config, set_current_atom_config
    from atom.models.deepseek_v2 import DeepseekV2DecoderLayer

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_method = f"tcp://127.0.0.1:{port}"
    init_distributed_environment(
        world_size=args.npes,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=args.npes)

    config = Config(
        model=str(args.config),
        max_model_len=max(4096, args.pos + max(args.samples)),
        max_num_seqs=max(args.samples),
        max_num_batched_tokens=max(args.samples),
        tensor_parallel_size=args.npes,
        kv_cache_dtype="bf16",
        enforce_eager=True,
    )
    config.quant_config.remap_layer_name(
        config.hf_config,
        quant_exclude_name_mapping={"indexers_proj": "indexer.weights_proj"},
    )
    set_current_atom_config(config)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with torch.device(device):
            layer = DeepseekV2DecoderLayer(
                config.hf_config,
                "model.layers.3",
                cache_config="bf16",
                quant_config=config.quant_config,
                layer_num=3,
                alt_stream=torch.cuda.Stream(device=device),
                use_indexer_wk_weights_proj_fusion=False,
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    # The FlyDSL kernel receives top-2048 indices, so remove only ATOM's
    # producer for those indices while retaining its sparse MLA consumer.
    layer.self_attn.indexer = None
    layer.self_attn.mla_attn.indexer = None
    gc.collect()
    torch.cuda.empty_cache()
    _init_weights(layer, rank)
    if rank == 0:
        experts = layer.mlp.experts
        logical_weight_shapes = {
            "w13_weight": list(experts.w13_weight.shape),
            "w13_weight_scale": list(experts.w13_weight_scale.shape),
            "w2_weight": list(experts.w2_weight.shape),
            "w2_weight_scale": list(experts.w2_weight_scale.shape),
        }
    else:
        logical_weight_shapes = None
    _postprocess_weights(layer)
    layer.eval()

    if rank == 0:
        experts = layer.mlp.experts
        runtime_weight_shapes = {
            "w13_weight": list(experts.w13_weight.shape),
            "w13_weight_scale": list(experts.w13_weight_scale.shape),
            "w2_weight": list(experts.w2_weight.shape),
            "w2_weight_scale": list(experts.w2_weight_scale.shape),
        }
    else:
        runtime_weight_shapes = None

    for samples in args.samples:
        generator = torch.Generator(device=device).manual_seed(args.seed + rank * 17 + samples)
        hidden = torch.randn(samples, 6144, dtype=torch.bfloat16, device=device, generator=generator)
        residual = torch.randn(samples, 6144, dtype=torch.bfloat16, device=device, generator=generator)
        positions, kv_cache, sparse_indices, work = _make_forward_context(config, layer, samples, args.pos, generator)

        for _ in range(3):
            output, residual_output = layer(positions, hidden, residual)
        torch.cuda.synchronize()
        dist.barrier()

        with graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                for _ in range(args.layers):
                    output, residual_output = layer(positions, hidden, residual)
        torch.cuda.synchronize()

        times = []
        for repetition in range(args.repeats + 2):
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all() or not torch.isfinite(residual_output).all():
                raise AssertionError("ATOM layer produced a non-finite output")
            if repetition >= 2:
                times.append(start.elapsed_time(end) * 1000 / args.layers)

        per_rank = [None] * args.npes
        dist.all_gather_object(per_rank, times)
        if rank == 0:
            critical = [max(repetition) for repetition in zip(*per_rank)]
            result = {
                "benchmark_version": 2,
                "backend": "atom-native",
                "scope": "decoder-layer-with-preselected-sparse-indices",
                "atom_commit": args.atom_commit,
                "aiter_commit": args.aiter_commit,
                "npes": args.npes,
                "samples": samples,
                "pos": args.pos,
                "sparse_topk": config.hf_config.index_topk,
                "kv_cache_dtype": "bf16",
                "attention_weight_dtype": "bf16",
                "expert_weight_dtype": "mxfp4",
                "layers_per_graph": args.layers,
                "measured_repeats": args.repeats,
                "trim_each_tail": args.trim,
                "trimmed_mean_us": _trimmed_mean(critical, args.trim),
                "median_us": statistics.median(critical),
                "min_us": min(critical),
                "max_us": max(critical),
                "rank_times_us": per_rank,
                "logical_weight_shapes": logical_weight_shapes,
                "runtime_weight_shapes": runtime_weight_shapes,
                "source_sha256": _source_hash(args.atom_root, args.aiter_root),
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__,
            }
            encoded = json.dumps(result)
            print(encoded, flush=True)
            if args.output:
                with args.output.open("a") as output_file:
                    output_file.write(encoded + "\n")
        dist.barrier()
        del graph, output, residual_output, hidden, residual, positions, kv_cache, sparse_indices, work
        gc.collect()
        torch.cuda.empty_cache()

    destroy_model_parallel()
    dist.destroy_process_group()


def _git_commit(path: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-root", type=Path, default=Path("/root/ATOM"))
    parser.add_argument("--aiter-root", type=Path, default=Path("/root/aiter"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--npes", choices=(8,), type=int, default=8)
    parser.add_argument("--samples", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument("--pos", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.atom_root = args.atom_root.resolve()
    args.aiter_root = args.aiter_root.resolve()
    args.config = args.config.resolve()
    args.atom_commit = _git_commit(args.atom_root)
    args.aiter_commit = _git_commit(args.aiter_root)
    if args.pos < 2048:
        parser.error("--pos must be at least 2048 for the sparse top-2048 decode comparison")
    if args.layers < 1 or args.repeats < 1:
        parser.error("--layers and --repeats must be positive")
    if args.repeats <= 2 * args.trim:
        parser.error("--repeats must be greater than twice --trim")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(args, port), nprocs=args.npes)
