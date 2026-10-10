# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Measure ATOM's external AITER indexer core for GLM-5.2 MTP4 C1/C2.

The default includes Q/K RoPE, FP8 quantization and K-cache update, paged
scoring, and stable top-2048. ``--projections`` also measures the upstream
RMSNorm/QKV and index Q/K/W projection path on synthetic weights. This is a
proxy: it uses AITER CK for FP8 projections and PyTorch BF16 linear for K/W,
because the installed AITER's tuned FlyDSL GEMMs do not compile with this
FlyDSL runtime. Selected-index conversion to physical attention KV slots
remains excluded.
Use the same TP4 row counts, 3k context, 32 index heads, and 16-token pages
as the FlyDSL fused microbenchmark.
"""

import argparse
import statistics

import aiter
import torch
from aiter import dtypes
from aiter.ops.cache import indexer_k_quant_and_cache
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_ck
from aiter.ops.quant import per_token_quant_hip
from aiter.ops.rmsnorm import rmsnorm2d_fwd
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.topk import top_k_per_row_decode
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, replays):
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / replays)
    return statistics.median(samples)


def bench(requests, replays, projections):
    dev = "cuda"
    width, max_seq, page, heads, dim = 5, 4096, 16, 32, 128
    S = requests * width
    generator = torch.Generator(device=dev).manual_seed(1244)
    pages_per_request = max_seq // page
    block_tables = torch.arange(requests * pages_per_request, dtype=torch.int32, device=dev).view(
        requests, pages_per_request
    )
    kv_cache = torch.zeros((requests * pages_per_request, page, dim + 4), dtype=dtypes.fp8, device=dev)
    old_k = torch.randn((requests * max_seq, dim), dtype=torch.bfloat16, device=dev, generator=generator)
    old_slots = torch.arange(requests * max_seq, dtype=torch.int64, device=dev)
    indexer_k_quant_and_cache(old_k, kv_cache, old_slots, dim, "ue8m0", preshuffle=True)
    q = torch.randn((S, heads, dim), dtype=torch.bfloat16, device=dev, generator=generator)
    k = torch.randn((S, dim), dtype=torch.bfloat16, device=dev, generator=generator)
    weights = torch.randn((S, heads), dtype=torch.bfloat16, device=dev, generator=generator)
    q_fp8 = torch.empty((S, heads, dim), dtype=dtypes.fp8, device=dev)
    weights_out = torch.empty((S, heads), dtype=torch.float32, device=dev)
    norm_weight = torch.ones(dim, dtype=torch.float32, device=dev)
    norm_bias = torch.zeros(dim, dtype=torch.float32, device=dev)
    positions = torch.tensor([3000 + i % width for i in range(S)], dtype=torch.int64, device=dev)
    slots = torch.tensor([3000 + i % width + (i // width) * max_seq for i in range(S)], dtype=torch.int64, device=dev)
    angles = torch.arange(max_seq, device=dev, dtype=torch.float32)[:, None] * (
        10000.0 ** (-torch.arange(dim // 4, device=dev, dtype=torch.float32)[None, :] / (dim // 4))
    )
    cos, sin = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
    context_lens = torch.full((requests,), 3005, dtype=torch.int32, device=dev)
    logits = torch.empty((S, max_seq), dtype=torch.float32, device=dev)
    indices = torch.empty((S, 2048), dtype=torch.int32, device=dev)

    def qk(q_input=q, k_input=k, weights_input=weights):
        aiter.indexer_qk_rope_quant_and_cache(
            q_input,
            q_fp8,
            weights_input,
            weights_out,
            k_input,
            kv_cache,
            slots,
            norm_weight,
            norm_bias,
            positions,
            cos,
            sin,
            1e-6,
            dim,
            "ue8m0",
            dim**-0.5 * heads**-0.5,
            preshuffle=True,
            is_neox=False,
        )

    def score():
        deepgemm_fp8_paged_mqa_logits(
            q_fp8.view(requests, width, heads, dim),
            kv_cache.unsqueeze(-2),
            weights_out,
            logits,
            context_lens,
            block_tables,
            max_seq,
            Preshuffle=True,
            KVBlockSize=page,
            ChunkK=256,
            WavePerEU=2,
        )

    def select():
        top_k_per_row_decode(logits, width, context_lens, indices, S, max_seq, 1, k=2048, stable=True)

    stages = {"qk/cache": qk, "paged score": score, "stable topk": select}
    graphs = {name: capture(fn) for name, fn in stages.items()}

    def complete():
        qk()
        score()
        select()

    graphs["core total"] = capture(complete)
    results = {name: time_graph(graph, replays) for name, graph in graphs.items()}

    if projections:
        hidden = torch.randn((S, 6144), dtype=torch.bfloat16, device=dev, generator=generator)
        g_hidden = torch.ones(6144, dtype=torch.bfloat16, device=dev)
        g_q = torch.ones(2048, dtype=torch.bfloat16, device=dev)

        def fp8_weight(rows, cols):
            logical = (torch.randn((rows, cols), device=dev, generator=generator) * 0.05).to(dtypes.fp8)
            return shuffle_weight(logical), torch.ones((rows, 1), dtype=torch.float32, device=dev)

        w_qkv, s_qkv = fp8_weight(2624, 6144)
        w_index_q, s_index_q = fp8_weight(4096, 2048)
        w_index_kw = (torch.randn((160, 6144), device=dev, generator=generator) * 0.01).to(torch.bfloat16)

        def project():
            normalized = rmsnorm2d_fwd(hidden, g_hidden, 1e-6)
            qkv_input, qkv_scale = per_token_quant_hip(normalized, quant_dtype=dtypes.fp8)
            qkv = torch.empty((S, 2624), dtype=torch.bfloat16, device=dev)
            gemm_a8w8_bpreshuffle_ck(qkv_input, w_qkv, qkv_scale, s_qkv, qkv)
            qr = rmsnorm2d_fwd(qkv[:, :2048], g_q, 1e-6)
            iq_input, iq_scale = per_token_quant_hip(qr, quant_dtype=dtypes.fp8)
            iq = torch.empty((S, 4096), dtype=torch.bfloat16, device=dev)
            gemm_a8w8_bpreshuffle_ck(iq_input, w_index_q, iq_scale, s_index_q, iq)
            kw = torch.nn.functional.linear(normalized, w_index_kw)
            return iq.view(S, heads, dim), kw[:, :dim], kw[:, dim:]

        project_graph = capture(project)
        results["projection path"] = time_graph(project_graph, replays)

        def full():
            iq, ik, iw = project()
            qk(iq, ik, iw)
            score()
            select()

        full_graph = capture(full)
        results["projections + core"] = time_graph(full_graph, replays)
    torch.cuda.synchronize()
    valid = bool(((indices >= 0) & (indices < 3005)).all().item())
    path = "external indexer proxy" if projections else "external AITER core"
    print(f"C{requests} MTP4 S={S} {path}: {results}; valid indices={valid}", flush=True)
    if not valid:
        raise RuntimeError("AITER external core selected invalid indices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--replays", type=int, default=128)
    parser.add_argument("--projections", action="store_true")
    args = parser.parse_args()
    for request_count in args.requests:
        bench(request_count, args.replays, args.projections)
