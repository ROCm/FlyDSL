# SPDX-License-Identifier: Apache-2.0
"""Graph latency of the shared/reuse GLM-5 shard with FlyDSL or native TileRT.

Both backends use the same generated weights and inputs. TileRT's released
whole-layer kernel supports one or eight peers. Every rank keeps eight heads
and intermediate size 256, so two/four-GPU FlyDSL runs are communication-scale
measurements rather than full-model TP2/TP4 configurations.
"""

from __future__ import annotations

import argparse
import hashlib
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

from kernels.mla_moe_layer.config import KV_LORA, MAX_LAYERS_PER_STEP, PE_DIM, MoeMode  # noqa: E402
from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.native_baseline import make_native_glm5_baseline  # noqa: E402
from kernels.mla_moe_layer.reference import make_weights, rope_table  # noqa: E402
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer  # noqa: E402


def _source_hash() -> str:
    digest = hashlib.sha256()
    sources = [
        ROOT / "kernels/common/mx_formats.py",
        *(ROOT / "kernels/mla_moe_layer" / name for name in ("config.py", "packing.py", "runtime.py")),
        ROOT / "kernels/mla_moe_layer/shared_reuse_moe_kernel.py",
        ROOT / "kernels/mla_moe_layer/layer.py",
    ]
    for source in sources:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _worker(rank, args, port):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.npes)
    weights = make_weights(rank, heads=8, device=device, seed=args.seed, moe_mode=args.moe_mode)
    native = make_native_glm5_baseline(weights, device, args.moe_mode) if args.backend == "tilert" else None
    cos, sin = rope_table(4096, device=device)

    for samples in args.samples:
        generator = torch.Generator(device=device).manual_seed(args.seed + 99)
        kv = torch.randn(4096, KV_LORA, generator=generator, device=device).bfloat16()
        pe = torch.randn(4096, PE_DIM, generator=generator, device=device).bfloat16()
        indices = torch.stack(
            [
                torch.randperm(max(args.pos + sample + 1, 2048), generator=generator, device=device)[:2048]
                .sort()
                .values
                for sample in range(samples)
            ]
        ).int()
        if args.pos >= 2048:
            indices[:, -1] = torch.arange(args.pos, args.pos + samples, device=device)
        hidden = torch.randn(samples, 6144, generator=generator, device=device).bfloat16()
        output = torch.empty_like(hidden)
        pos = torch.tensor([args.pos], dtype=torch.int32, device=device)

        if native is None:
            layer = SharedReuseMlaMoeLayer(
                weights,
                samples,
                rank=rank,
                npes=args.npes,
                timeline=args.trace,
                moe_mode=args.moe_mode,
            )

            def run(epoch):
                layer.forward(hidden, pos, kv, pe, indices, cos, sin, x_out=output, layer=epoch, advance=False)

            def advance():
                layer.advance_step()

            def reset():
                pass

        else:
            from tilert.models.glm_5_2_rocm.ops import unprojo_allreduce

            attn_peers = SymmetricPeerBuffer(unprojo_allreduce.sym_bytes(samples), rank, args.npes)
            ffn_peers = SymmetricPeerBuffer(unprojo_allreduce.sym_bytes(samples), rank, args.npes)
            partials = native.alloc_partials(samples)
            unprojected = torch.empty_like(hidden)
            freqs = torch.stack(
                [cos[args.pos : args.pos + samples], sin[args.pos : args.pos + samples]], dim=-1
            ).reshape(samples, 64)
            exchange = native.mla.exchange(samples)
            reset_tensors = [
                exchange.q_pairs,
                exchange.kv_pairs,
                exchange.pe_pairs,
                exchange.m1_pairs,
                native.mla.legs.qlines,
                native.mla.legs.kvnew_pairs,
                native.mla.legs.penew_pairs,
                native.mla.tail.sen_a,
                native.mla.tail.sen_b,
                native.mla.tail.sen_proj,
                native.moe.score_lines,
                native.moe.flags,
                native.moe.mid_pairs,
                native.legs.hlines,
            ]

            def run(epoch):
                native.forward(
                    hidden,
                    pos,
                    freqs,
                    pe[None],
                    kv[None],
                    indices,
                    partials,
                    weights.t["bias"],
                    residual=hidden,
                    sym_attn=attn_peers.addresses,
                    sym_ffn=ffn_peers.addresses,
                    mype=rank,
                    npes=args.npes,
                    tag=epoch + 1,
                    sen_tag=epoch + 1,
                    ffn_flag=epoch + 1,
                    unproj_o=unprojected,
                    x_out=output,
                    reuse_selection=1,
                )

            def advance():
                pass

            def reset():
                for tensor in reset_tensors:
                    tensor.zero_()
                attn_peers.storage.zero_()
                ffn_peers.storage.zero_()

        for epoch in range(5):
            run(epoch)
            advance()
        reset()
        torch.cuda.synchronize()
        dist.barrier()
        run(0)
        torch.cuda.synchronize()
        reference = output.clone()
        if args.dump_outputs:
            target = Path(args.dump_outputs)
            target.mkdir(parents=True, exist_ok=True)
            torch.save(
                output.cpu(),
                target / f"{args.backend}-{args.moe_mode}-tp{args.npes}-s{samples}-r{rank}.pt",
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for epoch in range(args.layers):
                run(epoch)
            advance()

        times = []
        for repetition in range(args.repeats + 2):
            reset()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            if not torch.isfinite(output).all():
                raise AssertionError("kernel produced a non-finite output")
            if repetition >= 2:
                times.append(start.elapsed_time(end) * 1000 / args.layers)

        per_rank = [None] * args.npes
        dist.all_gather_object(per_rank, times)
        if args.trace:
            trace_dir = Path(args.trace_dir) / f"{args.moe_mode}-s{samples}" / f"rank{rank}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(stages=layer.stages, ticks=layer.timeline.cpu()), trace_dir / "trace.pt")
            if rank == 0:
                print(layer.timeline_report(), flush=True)
        if rank == 0:
            critical = [max(batch) for batch in zip(*per_rank)]
            result = dict(
                benchmark_version=3,
                backend=args.backend,
                moe_mode=args.moe_mode,
                instrumented=args.trace,
                npes=args.npes,
                samples=samples,
                pos=args.pos,
                heads_per_rank=8,
                inter_per_rank=256,
                seed=args.seed,
                layers=args.layers,
                median_us=statistics.median(critical),
                min_us=min(critical),
                max_us=max(critical),
                rank_times_us=per_rank,
                kernel_sha256=_source_hash(),
                torch=torch.__version__,
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            print(json.dumps(result), flush=True)
            if args.output:
                with Path(args.output).open("a") as output_file:
                    output_file.write(json.dumps(result) + "\n")
        dist.barrier()
        if native is None:
            layer.close()
        else:
            attn_peers.close()
            ffn_peers.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flydsl", "tilert"), required=True)
    parser.add_argument("--moe-mode", choices=tuple(mode.value for mode in MoeMode), default=MoeMode.W8A8.value)
    parser.add_argument("--npes", choices=(1, 2, 4, 8), type=int, required=True)
    parser.add_argument("--samples", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4])
    parser.add_argument("--pos", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", type=int, default=MAX_LAYERS_PER_STEP)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output")
    parser.add_argument("--dump-outputs")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-dir", default="/root/glm5-perf-results/traces")
    args = parser.parse_args()
    if args.backend == "tilert" and args.npes not in (1, 8):
        parser.error("TileRT's released whole-layer kernel only supports 1 or 8 peers")
    if args.backend == "tilert" and args.moe_mode not in (MoeMode.W8A8.value, MoeMode.W8A16.value):
        parser.error("the TileRT comparison adapter supports only w8a8 and w8a16 expert weights")
    if args.backend == "tilert" and any(samples == 8 for samples in args.samples):
        parser.error("TileRT's released whole-layer kernel only supports sample counts 1, 2, and 4")
    if args.trace and args.backend != "flydsl":
        parser.error("--trace is available for the FlyDSL backend")
    if not 1 <= args.layers <= MAX_LAYERS_PER_STEP:
        parser.error(f"layers must be in [1, {MAX_LAYERS_PER_STEP}]")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(args, port), nprocs=args.npes)
