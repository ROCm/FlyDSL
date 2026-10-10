# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Synthetic TP4 GLM-5.2 MTP4 C1/C2 kernel timing.

Run on four MI355X GPUs with a current FlyDSL runtime::

    python tests/kernels/bench_glm5_tp4_native.py --samples 5 10 --native 1
    python tests/kernels/bench_glm5_tp4_native.py --samples 5 10 --native 0

S=5 and S=10 correspond to C1 and C2 with five tokens per MTP4 request.
The default uses zero-filled attention and expert weights; --fused-indexer
adds synthetic index weights and a paged BF16 index cache. This measures a
single GLM layer without checkpoint prefill, scheduler, or serving overhead.
"""

import argparse
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.monokernel.config import (
    AttentionWeight,
    KvCacheLayout,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_tp_config,
)
from kernels.monokernel.glm.op import Glm5MonoKernel, prepare_glm5_weights
from kernels.monokernel.glm.reference import indexer_golden
from kernels.monokernel.packing import pack_ptpc_fp8
from kernels.monokernel.reference import attention_mats, scale_shape
from kernels.monokernel.weights import LayerWeights


def make_weights(rank, npes=4):
    dev = torch.device("cuda", rank)
    cfg = glm5_tp_config(npes)
    experts = cfg.n_experts + cfg.num_shared_experts
    fp8 = torch.float8_e4m3fnuz
    f8 = lambda *shape: torch.zeros(shape, dtype=fp8, device=dev)
    b16 = lambda *shape: torch.zeros(shape, dtype=torch.bfloat16, device=dev)
    u8 = lambda *shape: torch.zeros(shape, dtype=torch.uint8, device=dev)
    t = {
        "g_in": torch.ones(cfg.hidden, dtype=torch.bfloat16, device=dev),
        "g_q": torch.ones(cfg.q_lora, dtype=torch.bfloat16, device=dev),
        "g_kv": torch.ones(cfg.kv_lora, dtype=torch.bfloat16, device=dev),
        "g_post": torch.ones(cfg.hidden, dtype=torch.bfloat16, device=dev),
        "w_qkv_a": f8(cfg.qkv_a_rows, cfg.hidden),
        "s_qkv_a": torch.ones(cfg.qkv_a_rows, device=dev),
        "w_q_b": f8(cfg.local_heads * (cfg.nope_dim + cfg.pe_dim), cfg.q_lora),
        "s_q_b": torch.ones(cfg.local_heads * (cfg.nope_dim + cfg.pe_dim), device=dev),
        "w_uk": f8(cfg.local_heads * cfg.kv_lora, cfg.nope_dim),
        "s_uk": torch.ones(1, device=dev),
        "w_uv": f8(cfg.local_heads * cfg.v_dim, cfg.kv_lora),
        "s_uv": torch.ones(1, device=dev),
        "w_o": f8(cfg.hidden, cfg.local_heads * cfg.v_dim),
        "s_o": torch.ones(cfg.hidden, device=dev),
        "w_r": b16(cfg.n_experts, cfg.hidden),
        "bias": torch.zeros(cfg.n_experts, device=dev),
        "w_ug": u8(experts, 2 * cfg.inter, cfg.hidden // 2),
        "s_ug": u8(experts * 2 * cfg.inter, cfg.hidden // 32),
        "w_dn": u8(experts, cfg.hidden, cfg.inter // 2),
        "s_dn": u8(experts * cfg.hidden, cfg.inter // 32),
    }
    t["w_ug"].is_shuffled = True
    t["w_dn"].is_shuffled = True
    t["s_ug"].fill_(127)
    t["s_dn"].fill_(127)
    return LayerWeights(
        cfg.local_heads,
        t,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=experts,
    )


def worker(
    rank,
    port,
    samples,
    native,
    replays,
    fused_indexer,
    debug_progress,
    split_bf16_kv,
    check_golden,
    shuffled_pages,
    check_attention,
    npes,
    indexer_cp,
    check_cp,
    position,
    max_seq,
    request_pos_stride,
    indexer_ties,
):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    W = make_weights(rank, npes)
    attention_weight = AttentionWeight.FP8_BLOCK128 if split_bf16_kv else AttentionWeight.FP8_PTPC
    cache_layout = KvCacheLayout.SPLIT if split_bf16_kv else KvCacheLayout.ATOM
    cache_dtype = "bf16" if split_bf16_kv else "fp8"
    if split_bf16_kv:
        for name, (rows, width, bk) in attention_mats(W.config.local_heads, W.config).items():
            W.t[f"s_{name}"] = torch.ones(scale_shape(rows, width, bk), device=dev)
    if fused_indexer:
        generator = torch.Generator(device=dev).manual_seed(2435)
        logical_qkv = (torch.randn(W.t["w_qkv_a"].shape, device=dev, generator=generator) * 0.05).to(
            torch.float8_e4m3fnuz
        )
        W.t["w_qkv_a"].copy_(
            logical_qkv
            if split_bf16_kv
            else pack_ptpc_fp8(logical_qkv).view(torch.float8_e4m3fnuz).view_as(W.t["w_qkv_a"])
        )
        W.t["s_qkv_a"].fill_(W.config.hidden**-0.5)
        W.t.update(
            w_index_k=(torch.randn((128, W.config.hidden), device=dev, generator=generator) * 0.05).to(
                torch.float8_e4m3fn
            ),
            s_index_k=torch.full((1, W.config.hidden // 128), W.config.hidden**-0.5, device=dev),
            w_index_w=(torch.randn((32, W.config.hidden), device=dev, generator=generator) / W.config.hidden**0.5).to(
                torch.bfloat16
            ),
            w_index_q=(torch.randn((4096, W.config.q_lora), device=dev, generator=generator) * 0.05).to(
                torch.float8_e4m3fn
            ),
            s_index_q=torch.full((32, W.config.q_lora // 128), W.config.q_lora**-0.5, device=dev),
            g_index_k=torch.ones(128, device=dev),
            b_index_k=torch.zeros(128, device=dev),
        )
        if check_attention:
            for name in ("q_b", "uk", "uv", "o"):
                logical = (torch.randn(W.t[f"w_{name}"].shape, device=dev, generator=generator) * 0.02).to(
                    torch.float8_e4m3fnuz
                )
                W.t[f"w_{name}"].copy_(
                    pack_ptpc_fp8(logical).view(torch.float8_e4m3fnuz).view_as(W.t[f"w_{name}"])
                    if name in ("q_b", "o")
                    else logical
                )
    if indexer_ties:
        W.t["w_index_w"].zero_()
    prepared = prepare_glm5_weights(W, attention_weight)
    for S in samples:
        if rank == 0:
            print(f"starting TP{npes} S={S} native_fp4_mfma={native} fused_indexer={fused_indexer}", flush=True)
        op = Glm5MonoKernel(
            W,
            S,
            rank=rank,
            npes=npes,
            topk=2048,
            launches_per_step=16,
            attention_weight=attention_weight,
            kv_cache_layout=cache_layout,
            kv_cache_dtype=cache_dtype,
            prepared_weights=prepared,
            native_fp4_mfma=native,
            rope_dtype="bf16",
            index_request_width=5 if S >= 5 else 1,
            index_max_seq=max_seq,
            with_indexer=fused_indexer,
            timeline=debug_progress,
            indexer_cp=indexer_cp,
        )
        h = torch.randn(
            (S, W.config.hidden),
            dtype=torch.bfloat16,
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(524),
        )
        x = torch.empty_like(h)
        cur_pos = torch.tensor([position], dtype=torch.int32, device=dev)
        positions = torch.tensor(
            [position + i % 5 + (i // 5) * request_pos_stride for i in range(S)], dtype=torch.int64, device=dev
        )
        slots = torch.tensor(
            [position + i % 5 + ((i // 5) * max_seq if fused_indexer else (500 if i >= 5 else 0)) for i in range(S)],
            dtype=torch.int64,
            device=dev,
        )
        indptr = torch.arange(S + 1, dtype=torch.int32, device=dev) * 2048
        indices = torch.arange(2048, dtype=torch.int32, device=dev).repeat(S)
        kv = torch.zeros(
            (max_seq * (S // 5 if fused_indexer and S >= 5 else 1), 512 if split_bf16_kv else 576),
            dtype=torch.bfloat16 if split_bf16_kv else torch.float8_e4m3fnuz,
            device=dev,
        )
        if check_attention:
            kv.copy_((torch.randn(kv.shape, device=dev, generator=generator) * 0.05).to(kv.dtype))
        pe = torch.zeros((max_seq, 64), dtype=torch.bfloat16, device=dev) if split_bf16_kv else kv
        angles = torch.arange(max_seq, device=dev, dtype=torch.float32)[:, None] * (
            10000.0 ** (-torch.arange(32, device=dev, dtype=torch.float32)[None, :] / 32)
        )
        cos = angles.cos().to(torch.bfloat16)
        sin = angles.sin().to(torch.bfloat16)
        index_cache = (
            torch.randn(
                (max_seq * (S // 5 if S >= 5 else 1), 128),
                device=dev,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=dev).manual_seed(912),
            )
            if fused_indexer
            else None
        )
        index_cache_before = index_cache.clone() if check_golden and fused_indexer else None
        block_tables = (
            torch.arange(index_cache.shape[0] // 16, dtype=torch.int32, device=dev).view(-1, max_seq // 16)
            if fused_indexer
            else None
        )
        if shuffled_pages and fused_indexer:
            permutation = torch.randperm(max_seq // 16, device=dev, generator=generator)
            block_tables = block_tables[:, permutation].contiguous()
        if fused_indexer:
            request_ids = torch.arange(S, device=dev) // (5 if S >= 5 else 1)
            logical_positions = positions.to(torch.int64)
            slots = block_tables[request_ids, logical_positions // 16].to(torch.int64) * 16 + logical_positions % 16

        def launch(layer, advance):
            return op.forward(
                h,
                cur_pos,
                kv,
                pe,
                indices,
                cos,
                sin,
                x_out=x,
                layer=layer,
                advance=advance,
                positions=positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
                index_cache=index_cache,
                block_tables=block_tables,
            )

        launch(0, not debug_progress)
        torch.cuda.synchronize()
        if debug_progress:
            op.advance_step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer in range(16):
                launch(layer, False)
            op.advance_step()
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        if debug_progress and rank == 0:
            print(op.timeline_report(), flush=True)
        if fused_indexer:
            data = op.intermediates()
            selected = data["indices"]
            if debug_progress and rank == 0:
                stats = {}
                for name, value in data.items():
                    if name in ("sel", "indices"):
                        continue
                    floats = value.float()
                    valid = floats[torch.isfinite(floats)]
                    stats[name] = (
                        int(torch.isnan(floats).sum()),
                        int(torch.isinf(floats).sum()),
                        float(valid.abs().max()) if valid.numel() else None,
                    )
                print(
                    "fused diagnostics (nan, inf, max abs finite):",
                    stats,
                    "index range:",
                    selected.min().item(),
                    selected.max().item(),
                    flush=True,
                )
            if not bool(((selected >= 0) & (selected <= positions[:, None])).all().item()):
                raise RuntimeError(f"fused indexer selected an invalid token for TP{npes} S={S}")
            if check_golden:
                overlaps = []
                for start in range(0, S, 5 if S >= 5 else 1):
                    width = min(5, S - start)
                    physical_pages = block_tables[start // 5].to(torch.int64)
                    logical_cache = index_cache_before.view(-1, 16, 128)[physical_pages].reshape(max_seq, 128).clone()
                    ref, _, _, _, _ = indexer_golden(
                        W,
                        h[start : start + width],
                        data["q_a"][start : start + width],
                        position + (start // 5) * request_pos_stride,
                        logical_cache,
                        cos,
                        sin,
                        topk=2048,
                    )
                    for row in range(width):
                        got = torch.sort(selected[start + row]).values
                        expected = torch.sort(ref[row]).values
                        overlaps.append(torch.isin(got, expected).float().mean().item())
                if rank == 0:
                    print(
                        f"TP{npes} S={S} fused indexer golden top-2048 overlap: "
                        f"min={min(overlaps):.5f} rows={overlaps}",
                        flush=True,
                    )
                if min(overlaps) < 0.99:
                    raise RuntimeError(f"fused indexer top-k overlap below 99% for TP{npes} S={S}")
            if check_attention:
                request_ids = torch.arange(S, device=dev) // (5 if S >= 5 else 1)
                physical_indices = (
                    block_tables[request_ids[:, None], selected.long() // 16].to(torch.int32) * 16 + selected % 16
                ).flatten()
                comparison = Glm5MonoKernel(
                    W,
                    S,
                    rank=rank,
                    npes=npes,
                    topk=2048,
                    launches_per_step=1,
                    attention_weight=attention_weight,
                    kv_cache_layout=cache_layout,
                    kv_cache_dtype=cache_dtype,
                    prepared_weights=prepared,
                    native_fp4_mfma=native,
                    rope_dtype="bf16",
                    index_request_width=5 if S >= 5 else 1,
                    index_max_seq=max_seq,
                    with_indexer=False,
                )
                x_comparison = comparison.forward(
                    h,
                    cur_pos,
                    kv,
                    pe,
                    physical_indices,
                    cos,
                    sin,
                    positions=positions,
                    slot_mapping=slots,
                    sparse_kv_indptr=indptr,
                )
                torch.cuda.synchronize()
                max_diff = (x.float() - x_comparison.float()).abs().max().item()
                attention_signal = data["o"].float().abs().max().item()
                if rank == 0:
                    print(
                        f"TP{npes} S={S} fused vs selected-index unfused attention: max output diff={max_diff:.6f}, "
                        f"nonzero attention max={attention_signal:.6f}",
                        flush=True,
                    )
                if max_diff > 0.05 or attention_signal < 1e-4:
                    raise RuntimeError(f"fused attention physical-cache comparison failed for TP{npes} S={S}")
                comparison.close()
        if check_cp:
            control = Glm5MonoKernel(
                W,
                S,
                rank=rank,
                npes=npes,
                topk=2048,
                launches_per_step=1,
                attention_weight=attention_weight,
                kv_cache_layout=cache_layout,
                kv_cache_dtype=cache_dtype,
                prepared_weights=prepared,
                native_fp4_mfma=native,
                with_indexer=True,
                indexer_cp=not indexer_cp,
                rope_dtype="bf16",
                index_request_width=5 if S >= 5 else 1,
                index_max_seq=max_seq,
            )
            kv_control, index_control = kv.clone(), index_cache.clone()
            expected = control.forward(
                h,
                cur_pos,
                kv_control,
                kv_control,
                indices,
                cos,
                sin,
                positions=positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
                index_cache=index_control,
                block_tables=block_tables,
            )
            torch.cuda.synchronize()
            chosen = op.intermediates()["indices"].clone()
            # Compare FP8 storage bytes: the cache is consumed as E4M3FN by
            # the kernel, while the PR fixture uses an FNUZ-typed container.
            # FN negative zero (0x80) appears as NaN through an FNUZ view.
            same = (
                torch.equal(x.view(torch.uint8), expected.view(torch.uint8))
                and torch.equal(kv.view(torch.uint8), kv_control.view(torch.uint8))
                and torch.equal(index_cache.view(torch.uint8), index_control.view(torch.uint8))
                and torch.equal(chosen, control.intermediates()["indices"])
            )
            if not same:
                control_data = control.intermediates()
                actual_data = op.intermediates()
                differences = {}
                for label, got, ref in [
                    ("x", x, expected),
                    ("kv_bytes", kv.view(torch.uint8), kv_control.view(torch.uint8)),
                    ("index_cache", index_cache, index_control),
                ] + [(name, actual_data[name], control_data[name]) for name in actual_data]:
                    differences[label] = {
                        "count": int((got != ref).sum()),
                        "max": float((got.float() - ref.float()).abs().max()),
                    }
                print(f"TP{npes} rank={rank} S={S} CP differences: {differences}", flush=True)
                raise RuntimeError(f"TP{npes} S={S} CP on/off bitwise mismatch")
            # Cross the old 8-bit epoch boundary and alternate mailbox slots.
            for _ in range(306):
                graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(x.view(torch.uint8), expected.view(torch.uint8)) or not torch.equal(
                chosen, op.intermediates()["indices"]
            ):
                raise RuntimeError(f"TP{npes} S={S} CP graph replay mismatch")
            if rank == 0:
                print(f"TP{npes} S={S} CP control/replay: bitwise_equal=True (306 graphs x 16 layers)", flush=True)
            control.close()
        if not torch.isfinite(x).all().item():
            raise RuntimeError(f"nonfinite output for TP{npes} S={S}, native={native}")
        elapsed = []
        for _ in range(3):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            stop.record()
            stop.synchronize()
            elapsed.append(start.elapsed_time(stop) * 1000 / (replays * 16))
        gathered = [None] * npes
        dist.all_gather_object(gathered, statistics.median(elapsed))
        if rank == 0:
            print(
                f"TP{npes} S={S} native={native} fused_indexer={fused_indexer} indexer_cp={indexer_cp}: "
                f"median-rank={statistics.median(gathered):.3f} us/layer; "
                f"max-rank={max(gathered):.3f} us/layer; ranks={gathered}",
                flush=True,
            )
        op.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npes", type=int, choices=(4, 8), default=4)
    parser.add_argument("--indexer-cp", action="store_true")
    parser.add_argument("--check-cp", action="store_true")
    parser.add_argument("--pos", type=int, default=3000)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--request-pos-stride", type=int, default=0)
    parser.add_argument("--indexer-ties", action="store_true")
    parser.add_argument("--samples", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--native", type=int, default=1)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--fused-indexer", action="store_true")
    parser.add_argument("--debug-progress", action="store_true")
    parser.add_argument("--split-bf16-kv", action="store_true")
    parser.add_argument("--check-golden", action="store_true")
    parser.add_argument("--shuffled-pages", action="store_true")
    parser.add_argument("--check-attention", action="store_true")
    args = parser.parse_args()
    if (args.indexer_cp or args.check_cp or args.indexer_ties) and not args.fused_indexer:
        parser.error("CP and tie checks require --fused-indexer")
    if args.max_seq <= 0 or args.max_seq % 64 or args.pos < 0 or args.request_pos_stride < 0:
        parser.error("capacity must be positive and aligned to 64; positions must be nonnegative")
    if args.pos + 5 + (max(args.samples) // 5 - 1) * args.request_pos_stride > args.max_seq:
        parser.error("request positions must fit max-seq")
    if args.fused_indexer and any(sample not in (1, 5, 10) for sample in args.samples):
        parser.error("fused indexer supports TP4 MTP4 C1/C2: --samples 5 or 10 (or 1 for debug)")
    if args.fused_indexer and args.split_bf16_kv:
        parser.error("TP4 fused indexer requires ATOM paged FP8 KV cache")
    if not args.fused_indexer and (args.check_golden or args.shuffled_pages or args.check_attention):
        parser.error("indexer correctness options require --fused-indexer")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(
        worker,
        args=(
            port,
            args.samples,
            bool(args.native),
            args.replays,
            args.fused_indexer,
            args.debug_progress,
            args.split_bf16_kv,
            args.check_golden,
            args.shuffled_pages,
            args.check_attention,
            args.npes,
            args.indexer_cp,
            args.check_cp,
            args.pos,
            args.max_seq,
            args.request_pos_stride,
            args.indexer_ties,
        ),
        nprocs=args.npes,
    )
