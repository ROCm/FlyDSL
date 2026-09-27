# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Benchmark ATOM's original Kimi-K3 TP8 decoder layer with synthetic weights.

This harness intentionally imports the implementation from a separate ATOM
checkout.  It needs no model checkpoint: parameter shapes and quantization are
the production Kimi-K3 shapes, while deterministic dummy values are installed
through ATOM's own dummy-weight and post-load paths.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_imports(args) -> None:
    for path in (args.deps_root, args.flydsl_python, args.aiter_root, args.atom_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _build_atom_config(
    samples: int,
    context_len: int,
    *,
    attention_family: str,
    layer_idx: int,
    dense_ffn: bool,
):
    from aiter import QuantType, dtypes
    from atom.config import CompilationConfig, CUDAGraphMode, ParallelConfig, QuantizationConfig
    from atom.models.kimi_k3 import _normalize_kimi_config
    from atom.quant_spec import LayerQuantConfig
    from transformers import PretrainedConfig

    hf_config = PretrainedConfig(
        hidden_size=7168,
        intermediate_size=18432,
        hidden_act="situ",
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_experts=896,
        num_experts_per_token=16,
        routed_expert_hidden_size=3584,
        moe_intermediate_size=3072,
        num_shared_experts=2,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        latent_moe_use_norm=True,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        rms_norm_eps=1.0e-5,
        attn_res_block_size=12,
        num_hidden_layers=93,
        max_position_embeddings=max(4096, context_len + 1),
        rope_theta=50000.0,
        rope_parameters={"rope_theta": 50000.0, "rope_type": "default"},
        linear_attn_config={
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [layer_idx + 1] if attention_family == "kda" else [],
            "full_attn_layers": [layer_idx + 1] if attention_family == "mla" else [],
        },
        model_type="kimi_linear",
    )
    hf_config.torch_dtype = torch.bfloat16
    hf_config.dtype = torch.bfloat16
    hf_config.first_k_dense_replace = 1 if dense_ffn else 0
    hf_config.quantization_config = {
        "quant_method": "quark",
        "global_quant_config": {},
        "layer_quant_config": {
            "*block_sparse_moe.experts": {
                "weight": {"dtype": "fp4", "qscheme": "per_group"},
                "input_tensors": {"is_dynamic": True},
            }
        },
        "exclude": [],
    }
    _normalize_kimi_config(hf_config)

    quant_config = QuantizationConfig(hf_config)
    # Keep this explicit in case a future parser changes the spelling of the
    # synthetic pattern above.  The benchmark contract is BF16 everywhere
    # except the routed experts, which use dynamic-activation MXFP4 weights.
    quant_config.global_spec = LayerQuantConfig.no_quant(torch.bfloat16)
    quant_config.layer_pattern_specs = [
        (
            "*block_sparse_moe.experts",
            LayerQuantConfig(
                quant_type=QuantType.per_1x32,
                quant_dtype=dtypes.fp4x2,
                is_dynamic=True,
                quant_method="quark",
            ),
        )
    ]

    compilation_config = CompilationConfig(
        level=0,
        use_cudagraph=True,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    parallel_config = ParallelConfig()
    return SimpleNamespace(
        hf_config=hf_config,
        quant_config=quant_config,
        compilation_config=compilation_config,
        parallel_config=parallel_config,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        kv_cache_dtype="bf16",
        kv_cache_block_size=1,
        max_model_len=max(4096, context_len + 1),
        max_num_batched_tokens=max(8, samples),
        attn_prefill_chunk_size=0,
        torch_dtype=torch.bfloat16,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        moe_ep_flatten_tp_across_dp=False,
        enable_tbo=False,
        moe_backend="standard",
        eplb_enable=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        kv_transfer_config={},
    )


def _post_load(layer) -> None:
    from atom.model_ops.base_config import QuantizeMethodBase
    from atom.model_ops.moe import FusedMoEMethodBase

    for module in layer.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if hook is not None:
            hook()
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)


def _prewarm_fused_moe(layer, samples: int, rank: int, npes: int, device: torch.device) -> None:
    """Compile the large FlyDSL MoE kernels one rank at a time.

    First-use compilation on all eight ranks at once can exhaust lld resources.
    The production launch itself is unchanged; this only serializes benchmark
    setup so each worker has its process-local executable before TP collectives
    begin.
    """

    moe = getattr(layer, "block_sparse_moe", None)
    if moe is None:
        return
    for compile_rank in range(npes):
        if rank == compile_rank:
            routed_input = torch.zeros(
                samples,
                moe.moe_hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            router_logits = torch.zeros(
                samples,
                moe.num_experts,
                dtype=torch.bfloat16,
                device=device,
            )
            moe.experts(routed_input, router_logits)
            torch.cuda.synchronize()
        dist.barrier()


def _build_metadata(
    samples: int,
    context_len: int,
    device: torch.device,
    *,
    attention_family: str,
):
    from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    from atom.utils.forward_context import AttentionMetaData, AttnState

    padded_heads = 16
    cu_seqlens_q = torch.arange(samples + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(samples + 1, dtype=torch.int32, device=device) * context_len
    kv_indices = torch.arange(samples * context_len, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(samples, dtype=torch.int32, device=device)
    # The fused Q/K RoPE + cache writer consumes 64-bit physical slot ids.
    slot_mapping = (kv_indptr[1:] - 1).to(torch.int64)

    metadata_info = get_mla_metadata_info_v1(
        samples,
        1,
        padded_heads,
        dtypes.bf16,
        dtypes.bf16,
        is_sparse=False,
        fast_mode=True,
        max_split_per_batch=-1,
    )
    buffers = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in metadata_info]
    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = buffers
    get_mla_metadata_v1(
        cu_seqlens_q,
        kv_indptr,
        kv_last_page_lens,
        padded_heads,
        1,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=-1,
    )
    metadata = AttentionMetaData(
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=context_len,
        slot_mapping=slot_mapping,
        context_lens=torch.full((samples,), context_len, dtype=torch.int32, device=device),
        block_tables=kv_indices.view(samples, context_len),
        state=AttnState.DECODE,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )
    metadata.dtype_q = dtypes.bf16
    if attention_family == "kda":
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata

        state_indices = torch.arange(samples, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(samples + 1, dtype=torch.int32, device=device)
        kda_metadata = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=samples,
            num_decode_tokens=samples,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=samples,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=state_indices,
            non_spec_state_indices_in_tensor=state_indices,
        )
        metadata.kda_metadata = kda_metadata
        metadata.gdn_metadata = kda_metadata
    return metadata


def _worker(rank: int, args, port: int, results) -> None:
    _configure_imports(args)
    os.environ.setdefault("AITER_REUSE_IDENTICAL_COMM_GROUPS", "1")
    os.environ.setdefault("ATOM_MLA_PAGE_SIZE", "1")
    os.environ.setdefault("ATOM_USE_TRITON_MLA", "0")
    os.environ.setdefault("ATOM_USE_TRITON_MXFP4_BMM", "0")

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from atom.config import CUDAGraphMode, KVCacheTensor, set_current_atom_config
    from atom.model_loader.loader import initialize_dummy_weights
    from atom.models.kimi_k3 import KimiDecoderLayer, KimiKDAAttention
    from atom.utils.forward_context import (
        Context,
        get_forward_context,
        set_forward_context,
        set_kv_cache_data,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(
        world_size=args.npes,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=args.npes)

    atom_config = _build_atom_config(
        args.samples,
        args.context_len,
        attention_family=args.attention_family,
        layer_idx=args.layer_idx,
        dense_ffn=args.dense_ffn,
    )
    set_current_atom_config(atom_config)
    # ATOM's ModelRunner constructs parameters under the configured default
    # dtype.  Mirror that here so RMSNorm weights and BF16 activations satisfy
    # the fused Q/K norm kernel's same-dtype contract.
    torch.set_default_dtype(atom_config.torch_dtype)
    layer = None
    with torch.device(device):
        if args.attention_only:
            layer = KimiKDAAttention(
                atom_config,
                quant_config=atom_config.quant_config,
                prefix=f"model.layers.{args.layer_idx}.self_attn",
            )
        else:
            layer = KimiDecoderLayer(
                atom_config,
                prefix=f"model.layers.{args.layer_idx}",
                layer_num=args.layer_idx,
                alt_stream=torch.cuda.Stream(device=device),
            )
    assert layer is not None
    initialize_dummy_weights(layer, "xavier")
    _post_load(layer)
    if not args.attention_only:
        _prewarm_fused_moe(layer, args.samples, rank, args.npes, device)

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    prefix_sum = torch.randn(
        args.samples,
        atom_config.hf_config.hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    block_residual = torch.randn(
        args.samples,
        (args.layer_idx + atom_config.hf_config.attn_res_block_size - 1) // atom_config.hf_config.attn_res_block_size,
        atom_config.hf_config.hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    positions = torch.full(
        (args.samples,),
        args.context_len - 1,
        dtype=torch.int64,
        device=device,
    )
    metadata = _build_metadata(
        args.samples,
        args.context_len,
        device,
        attention_family=args.attention_family,
    )
    if args.attention_family == "kda":
        local_heads = atom_config.hf_config.linear_num_key_heads // args.npes
        head_dim = atom_config.hf_config.linear_key_head_dim
        local_proj = local_heads * head_dim
        conv_state = torch.randn(
            args.samples,
            atom_config.hf_config.linear_conv_kernel_dim - 1,
            3 * local_proj,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        recurrent_state = torch.randn(
            args.samples,
            local_heads,
            head_dim,
            head_dim,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        layer_cache = KVCacheTensor(
            layer_num=args.layer_idx,
            k_cache=conv_state,
            v_cache=recurrent_state,
            per_request_state=True,
        )
    else:
        kv_cache = torch.zeros(
            args.samples * args.context_len,
            1,
            atom_config.hf_config.kv_lora_rank + atom_config.hf_config.qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        layer_cache = KVCacheTensor(layer_num=args.layer_idx, k_cache=kv_cache)
    set_kv_cache_data({f"layer_{args.layer_idx}": layer_cache}, config=atom_config)
    set_forward_context(
        metadata,
        atom_config,
        Context(positions, is_prefill=False, batch_size=args.samples, graph_bs=args.samples),
        num_tokens=args.samples,
        in_hipgraph=True,
    )
    get_forward_context().cudagraph_runtime_mode = CUDAGraphMode.FULL

    def run_layer():
        if args.attention_only:
            return layer(prefix_sum)
        return layer(positions, prefix_sum, block_residual)

    for _ in range(2):
        outputs = run_layer()
    torch.cuda.synchronize()
    dist.barrier()

    with graph_capture() as capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_context.stream):
            for _ in range(args.layers):
                outputs = run_layer()

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

    kernel_profile = None
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
            kernel_profile = [
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

    if args.attention_only:
        finite = bool(torch.isfinite(outputs).all())
    else:
        prefix_out, routed, shared, block_out = outputs
        finite = all(
            value is None or bool(torch.isfinite(value).all()) for value in (prefix_out, routed, shared, block_out)
        )
    result = {
        "implementation": "ATOM KimiDecoderLayer",
        "attention_only": args.attention_only,
        "atom_commit": args.atom_commit,
        "attention_family": args.attention_family,
        "layer_idx": args.layer_idx,
        "ffn": "dense" if args.dense_ffn else "latent_moe",
        "npes": args.npes,
        "samples": args.samples,
        "context_len": args.context_len,
        "finite": finite,
        "median_us": statistics.median(critical),
        "min_us": min(critical),
        "max_us": max(critical),
        "layers": args.layers,
        "repeats": args.repeats,
    }
    if kernel_profile is not None:
        result["kernel_profile"] = kernel_profile
    if rank == 0:
        print(json.dumps(result), flush=True)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n")
    results[rank] = result

    del graph, layer_cache
    torch.cuda.synchronize()
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-root", default="../ATOM-k3-baseline")
    parser.add_argument("--aiter-root", default="../aiter")
    parser.add_argument("--flydsl-python", default="../FlyDSL/build-fly/python_packages")
    parser.add_argument("--deps-root", default="/tmp/atom-k3-deps")
    parser.add_argument("--atom-commit", default="3cea04f45")
    parser.add_argument("--npes", type=int, choices=(8,), default=8)
    parser.add_argument("--samples", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--attention-family", choices=("mla", "kda"), default="mla")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--dense-ffn", action="store_true")
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--context-len", type=int, default=3001)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--kernel-profile", action="store_true", help="record one rank-0 HIP-graph kernel profile")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.dense_ffn and args.layer_idx != 0:
        parser.error("--dense-ffn is only valid for --layer-idx 0")
    if args.attention_only and args.attention_family != "kda":
        parser.error("--attention-only currently supports only --attention-family kda")

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(args, _free_port(), results), nprocs=args.npes, join=True)
    if len(results) != args.npes:
        raise RuntimeError(f"expected {args.npes} rank results, got {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
