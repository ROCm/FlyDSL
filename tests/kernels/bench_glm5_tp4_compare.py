# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Paired synthetic TP4 graph timing against ATOM PR 2435.

Point PYTHONPATH at the ATOM PR 2435 checkout, the FlyDSL runtime, and AITER,
then run this script from the FlyDSL checkout. Both backends share the exact
same input tensors, zero-filled weights, and graph timing boundaries. The
comparison excludes the external indexer and all serving overhead.
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

from atom.model_ops.monokernel.config import AttentionWeight as AtomAttentionWeight
from atom.model_ops.monokernel.config import KvCacheLayout as AtomKvCacheLayout
from atom.model_ops.monokernel.config import Mxfp4ScaleLayout as AtomScaleLayout
from atom.model_ops.monokernel.config import Mxfp4WeightLayout as AtomWeightLayout
from atom.model_ops.monokernel.config import glm5_tp_config as atom_glm5_tp_config
from atom.model_ops.monokernel.glm.op import Glm5MonoKernel as AtomOp
from atom.model_ops.monokernel.glm.op import prepare_glm5_weights as atom_prepare
from atom.model_ops.monokernel.weights import LayerWeights as AtomWeights
from bench_glm5_tp4_native import make_weights

from kernels.monokernel.config import AttentionWeight as FlyAttentionWeight
from kernels.monokernel.config import KvCacheLayout as FlyKvCacheLayout
from kernels.monokernel.glm.tp4_op import Glm5TP4MonoKernel as FlyOp
from kernels.monokernel.glm.tp4_op import prepare_glm5_weights as fly_prepare


def worker(rank, port, samples, replays, reverse, reverse_construct):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
    wf = make_weights(rank)
    wa = AtomWeights(
        wf.heads,
        wf.t,
        atom_glm5_tp_config(4),
        rank,
        4,
        mxfp4_weight_layout=AtomWeightLayout.ATOM,
        mxfp4_scale_layout=AtomScaleLayout.ATOM,
        physical_experts=wf.physical_experts,
    )
    pa = atom_prepare(wa, AtomAttentionWeight.FP8_PTPC)
    pf = fly_prepare(wf, FlyAttentionWeight.FP8_PTPC)
    for S in samples:
        make_atom = lambda: AtomOp(
            wa,
            S,
            rank=rank,
            npes=4,
            topk=2048,
            launches_per_step=16,
            attention_weight=AtomAttentionWeight.FP8_PTPC,
            kv_cache_layout=AtomKvCacheLayout.ATOM,
            kv_cache_dtype="fp8",
            prepared_weights=pa,
            native_fp4_mfma=True,
        )
        make_fly = lambda: FlyOp(
            wf,
            S,
            rank=rank,
            npes=4,
            topk=2048,
            launches_per_step=16,
            attention_weight=FlyAttentionWeight.FP8_PTPC,
            kv_cache_layout=FlyKvCacheLayout.ATOM,
            kv_cache_dtype="fp8",
            prepared_weights=pf,
            native_fp4_mfma=True,
        )
        makers = (
            (("FlyDSL", make_fly), ("ATOM", make_atom))
            if reverse_construct
            else (("ATOM", make_atom), ("FlyDSL", make_fly))
        )
        ops = {name: make() for name, make in makers}
        h = torch.randn(S, wa.config.hidden, dtype=torch.bfloat16, device=dev)
        cur_pos = torch.tensor([3000], dtype=torch.int32, device=dev)
        positions = torch.tensor([3000 + i % 5 for i in range(S)], dtype=torch.int64, device=dev)
        slots = torch.tensor([3000 + i % 5 + (i // 5) * 4096 for i in range(S)], dtype=torch.int64, device=dev)
        indptr = torch.arange(S + 1, dtype=torch.int32, device=dev) * 2048
        indices = torch.arange(2048, dtype=torch.int32, device=dev).repeat(S, 1)
        indices += (torch.arange(S, dtype=torch.int32, device=dev) // 5 * 4096)[:, None]
        indices = indices.flatten()
        angles = torch.arange(4096, device=dev, dtype=torch.float32)[:, None] * (
            10000.0 ** (-torch.arange(32, device=dev, dtype=torch.float32)[None, :] / 32)
        )
        cos = angles.cos().to(torch.bfloat16)
        sin = angles.sin().to(torch.bfloat16)
        outputs = {name: torch.empty_like(h) for name in ops}
        caches = {name: torch.zeros((4096 * max(1, S // 5), 576), dtype=torch.float8_e4m3fnuz, device=dev) for name in ops}
        graphs = {}
        for name, op in ops.items():

            def launch(layer, advance, op=op, name=name):
                return op.forward(
                    h,
                    cur_pos,
                    caches[name],
                    caches[name],
                    indices,
                    cos,
                    sin,
                    x_out=outputs[name],
                    layer=layer,
                    advance=advance,
                    positions=positions,
                    slot_mapping=slots,
                    sparse_kv_indptr=indptr,
                )

            launch(0, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for layer in range(16):
                    launch(layer, False)
                op.advance_step()
            graphs[name] = graph
        for name in ops:
            for _ in range(20):
                graphs[name].replay()
        torch.cuda.synchronize()
        finite = all(torch.isfinite(outputs[name]).all().item() for name in ops)
        same = torch.equal(outputs["ATOM"], outputs["FlyDSL"])
        records = {name: [] for name in ops}
        order = ("FlyDSL", "ATOM", "ATOM", "FlyDSL") if reverse else ("ATOM", "FlyDSL", "FlyDSL", "ATOM")
        for _ in range(3):
            for name in order:
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graphs[name].replay()
                stop.record()
                stop.synchronize()
                records[name].append(start.elapsed_time(stop) * 1000 / (16 * replays))
        local = {name: statistics.median(values) for name, values in records.items()}
        gathered = [None] * 4
        dist.all_gather_object(gathered, (local, finite, same))
        if rank == 0:
            atom = statistics.median(row[0]["ATOM"] for row in gathered)
            fly = statistics.median(row[0]["FlyDSL"] for row in gathered)
            print(
                f"TP4 S={S} native FP4: ATOM={atom:.3f} us/layer FlyDSL={fly:.3f} us/layer Fly/ATOM={fly/atom:.4f}; finite={all(row[1] for row in gathered)} equal={all(row[2] for row in gathered)}",
                flush=True,
            )
            print(f"  rank medians: {gathered}", flush=True)
        for op in ops.values():
            op.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--reverse-construct", action="store_true")
    args = parser.parse_args()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(worker, args=(port, args.samples, args.replays, args.reverse, args.reverse_construct), nprocs=4)
