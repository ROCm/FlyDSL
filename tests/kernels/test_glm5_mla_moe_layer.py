# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 shared/reuse MLA+MoE layer monokernel vs the torch golden.

Single GPU (TP1 view of one shard, the peer reduce is a 1-rank loopback)::

    python3 tests/kernels/test_glm5_mla_moe_layer.py --npes 1

TP8, one process per GPU::

    python3 tests/kernels/test_glm5_mla_moe_layer.py --npes 8
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.mla_moe_layer.reference import (  # noqa: E402
    KV_LORA,
    PE_DIM,
    golden_layer,
    golden_moe,
    make_weights,
    rope_table,
)

MAX_SEQ = 4096
TOL = {  # name -> (atol, rtol) on the fp32/bf16 intermediates
    "q_a": (2e-3, 2e-3),
    "kv_a": (2e-3, 2e-3),
    "q_nope": (5e-3, 5e-3),
    "q_pe": (5e-3, 5e-3),
    "q_lat": (1e-2, 1e-2),
    "o": (1e-2, 1e-2),
    "a": (3e-2, 2e-2),
    "scores": (1e-3, 1e-3),
    "prob": (1e-5, 1e-5),
    "kv": (2e-2, 1e-2),
    "x_out": (1.6e-2, 8e-3),  # 1 bf16 ulp
    "mid": (1e-3, 1e-3),  # FP8 x FP8 MFMA accumulation (~1e-4 abs), far below one E4M3 step
}
# End to end, single FP8 rounding flips propagate, so judge by relative L2.
FP8_FLIPS = ("xq",)
REL_L2 = {
    "x_out_e2e": 5e-2,  # bf16 activation roundings at 5 MFMA inputs compound across ranks
}


def _check(name, got, ref, report):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    if name in FP8_FLIPS:
        # a 1-ulp f32 difference upstream may move a value across an FP8 rounding
        # midpoint: allow a few elements to differ by at most one E4M3 step (<= 1/8 rel)
        flips = err > 1e-5 * (1 + ref.abs())
        too_far = err > ref.abs() / 8 + 1e-4  # + subnormal step (block scale * 2^-9)
        report.append((name, err.max().item(), ref.abs().max().item(), f"flips={int(flips.sum())}"))
        return not bool(too_far.any()) and int(flips.sum()) <= max(4, ref.numel() // 1000)
    if name in REL_L2:
        rel = (err.norm() / ref.norm()).item()
        report.append((name, err.max().item(), ref.abs().max().item(), f"rel_l2={rel:.2e}"))
        return rel <= REL_L2[name]
    atol, rtol = TOL[name]
    bad = err > atol + rtol * ref.abs()
    report.append((name, err.max().item(), ref.abs().max().item(), f"bad={int(bad.sum())}"))
    return not bool(bad.any())


def run_rank(rank, npes, S, cur_pos, iters, group=None, seed=1234):
    from kernels.mla_moe_layer.op import Glm5MlaMoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    topk = 2048
    W = make_weights(rank, heads=8, device=dev, seed=seed)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 99)  # same inputs on every rank
    kv0 = (torch.randn(MAX_SEQ, KV_LORA, generator=gen, device=dev)).to(torch.bfloat16)
    pe0 = (torch.randn(MAX_SEQ, PE_DIM, generator=gen, device=dev)).to(torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(max(cur_pos + s + 1, topk), generator=gen, device=dev)[:topk].sort().values for s in range(S)]
    ).to(torch.int32)
    if cur_pos + 1 > topk:  # the reused selection always holds the newest token
        indices[:, -1] = torch.arange(cur_pos, cur_pos + S, device=dev, dtype=torch.int32)
    op = Glm5MlaMoeLayer(W, S, rank=rank, npes=npes, group=group, topk=topk)

    if npes == 1:
        allreduce = lambda x: x  # noqa: E731
    else:
        import torch.distributed as dist

        def allreduce(x):
            parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
            dist.all_gather(parts, x.cpu().contiguous(), group=group)
            return sum(parts[1:], parts[0]).to(x.device)

    ok = True
    for it in range(iters):
        h = (torch.randn(S, 6144, generator=gen, device=dev)).to(torch.bfloat16)
        kv, pe = kv0.clone(), pe0.clone()
        pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
        out = op.forward(h, pos_t, kv, pe, indices, cos, sin)
        torch.cuda.synchronize()
        got = op.intermediates()
        got["x_out"] = out
        kv_ref, pe_ref = kv0.clone(), pe0.clone()
        ref = golden_layer(W, h, cur_pos, kv_ref, pe_ref, indices, cos, sin, allreduce, topk=topk)
        report = []
        # attention half vs the full golden
        for name in ("q_a", "kv_a", "q_nope", "q_pe", "q_lat", "o", "a"):
            ok &= _check(name, got[name], ref[name], report)
        # MoE half vs the golden fed the kernel's own post-attention state, so a
        # 1-ulp difference in ``a`` cannot flip FP8 roundings downstream
        moe = golden_moe(W, got["a"].clone(), allreduce)
        for name in ("scores", "prob", "xq"):
            ok &= _check(name, got[name], moe[name], report)
        # up/gate + SiLU from the kernel's own FP8 activation
        ug = golden_moe(W, got["a"].clone(), allreduce, xq=got["xq"].clone())
        ok &= _check("mid", got["mid"], ug["mid"], report)
        # down + route weighting + TP reduce + residual from the kernel's own mid
        down = golden_moe(W, got["a"].clone(), allreduce, got["mid"].clone(), got["sel"].clone(), got["prob"].clone())
        ok &= _check("x_out", got["x_out"], down["x_out"], report)
        ok &= torch.equal(got["sel"], moe["sel"])
        # fully independent golden: only comparable when a 1-ulp difference in ``a``
        # did not flip a near-tied routing decision
        if torch.equal(ref["sel"], moe["sel"]):
            ok &= _check("x_out_e2e", got["x_out"], ref["x_out"], report)
        else:
            report.append(("x_out_e2e", 0.0, 0.0, "skipped: golden routing flipped on a 1-ulp difference in a"))
        rows = slice(cur_pos, cur_pos + S)
        ok &= _check("kv", kv[rows], kv_ref[rows], report)
        if rank == 0 or not ok:
            lines = [f"[rank {rank} iter {it}] ok={ok} sel_ok={torch.equal(got['sel'], moe['sel'])}"]
            lines += [f"   {n:9s} max_err={e:.3e} ref_max={m:.3e} {nb}" for n, e, m, nb in report]
            print("\n".join(lines), flush=True)
    return ok


def bench_rank(rank, npes, S, cur_pos, iters=320, group=None, seed=1234):
    """HIP-graph replay of 16 layer launches per step; returns us per layer."""
    from kernels.mla_moe_layer.op import Glm5MlaMoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    W = make_weights(rank, heads=8, device=dev, seed=seed)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    kv = torch.randn(MAX_SEQ, KV_LORA, device=dev).to(torch.bfloat16)
    pe = torch.randn(MAX_SEQ, PE_DIM, device=dev).to(torch.bfloat16)
    indices = torch.stack([torch.randperm(max(cur_pos + s + 1, 2048), device=dev)[:2048] for s in range(S)]).int()
    op = Glm5MlaMoeLayer(W, S, rank=rank, npes=npes, group=group)
    h = torch.randn(S, 6144, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
    for _ in range(10):
        op.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x)
    torch.cuda.synchronize()
    import torch.distributed as dist

    if npes > 1:
        dist.barrier()
    if os.environ.get("MLA_MOE_TIMELINE"):
        top = Glm5MlaMoeLayer(W, S, rank=rank, npes=npes, group=group, timeline=True)
        for _ in range(3):
            top.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
    # a HIP graph of LAYERS layer launches + one step bump, like a decode step
    LAYERS = 16
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(LAYERS):
            op.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x, layer=layer, advance=False)
        op.advance_step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // LAYERS):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (iters // LAYERS * LAYERS)


def _worker(rank, npes, S, cur_pos, iters, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29541", rank=rank, world_size=npes)
    if iters < 0:
        us = bench_rank(rank, npes, S, cur_pos)
        results[rank] = us
        print(f"[rank {rank}] {us:.1f} us/layer", flush=True)
    else:
        results[rank] = run_rank(rank, npes, S, cur_pos, iters, group=None)
    dist.barrier()
    dist.destroy_process_group()


def run(npes, S, cur_pos, iters):
    if npes == 1:
        return run_rank(0, 1, S, cur_pos, iters)
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    results = mgr.dict()
    mp.spawn(_worker, args=(npes, S, cur_pos, iters, results), nprocs=npes)
    return all(results[r] for r in range(npes))


@pytest.mark.parametrize("S,cur_pos", [(1, 100), (1, 3000), (2, 3000)])
def test_layer_single_gpu(S, cur_pos):
    assert run(1, S, cur_pos, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_layer_tp8():
    assert run(8, 1, 3000, 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=1)
    ap.add_argument("-S", type=int, default=1)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true", help="time back-to-back launches")
    a = ap.parse_args()
    if a.bench:
        if a.npes == 1:
            print(f"{bench_rank(0, 1, a.S, a.pos):.1f} us/layer")
        else:
            run(a.npes, a.S, a.pos, -1)
        sys.exit(0)
    print("PASS" if run(a.npes, a.S, a.pos, a.iters) else "FAIL")
