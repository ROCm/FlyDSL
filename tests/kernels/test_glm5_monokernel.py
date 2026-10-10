# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 indexed decode MonoKernel vs the torch golden.

Single GPU (TP1 view of one shard, the peer reduce is a 1-rank loopback)::

    python3 tests/kernels/test_glm5_monokernel.py --npes 1

TP8, one process per GPU::

    python3 tests/kernels/test_glm5_monokernel.py --npes 8
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"GLM-5 MonoKernel requires gfx950, got {_ARCH}", allow_module_level=True)

from kernels.monokernel.config import GLM5_CONFIG, KV_LORA, PE_DIM, glm5_tp_config  # noqa: E402
from kernels.monokernel.glm.layout import INDEX_DIM  # noqa: E402
from kernels.monokernel.glm.reference import golden_layer, golden_moe, indexer_golden, make_weights  # noqa: E402
from kernels.monokernel.reference import rope, rope_table  # noqa: E402

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
    "x_out": (2e-2, 8e-3),  # BF16 peer partials can add ~2e-2 absolute error under cancellation
    "mid": (1e-3, 1e-3),  # FP8 x FP8 MFMA accumulation (~1e-4 abs), far below one E4M3 step
    "index_q": (2e-2, 2e-2),
    "index_w": (2e-3, 2e-3),
    "index_k": (2e-2, 2e-2),
}
# End to end, single FP8 rounding flips propagate, so judge by relative L2.
FP8_FLIPS = ("xq",)
REL_L2 = {"x_out_e2e": 5e-2}  # bf16 activation roundings at 5 MFMA inputs compound across ranks


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


def run_rank(
    rank,
    npes,
    S,
    cur_pos,
    iters,
    group=None,
    seed=1234,
    with_indexer=False,
    expert_mxfp4=False,
    max_seq=MAX_SEQ,
    indexer_ties=False,
    model_tp=None,
    vary_expert_scales=False,
):
    from kernels.monokernel.glm import Glm5MonoKernel

    if cur_pos < 0 or cur_pos + S > max_seq:
        raise ValueError("positions must fit max_seq")
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    topk = 2048
    config = GLM5_CONFIG if model_tp is None else glm5_tp_config(model_tp)
    if model_tp is not None and model_tp != npes:
        raise ValueError("model TP geometry must match the number of ranks")
    W = make_weights(
        rank,
        heads=config.local_heads,
        device=dev,
        seed=seed,
        with_indexer=with_indexer,
        expert_mxfp4=expert_mxfp4,
        config=config,
        vary_expert_scales=vary_expert_scales,
    )
    if indexer_ties:
        W.t["w_index_w"].zero_()
    cos, sin = rope_table(max_seq, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 99)  # same inputs on every rank
    kv0 = (torch.randn(max_seq, KV_LORA, generator=gen, device=dev)).to(torch.bfloat16)
    pe0 = (torch.randn(max_seq, PE_DIM, generator=gen, device=dev)).to(torch.bfloat16)
    if with_indexer:
        indices = torch.zeros(S, topk, dtype=torch.int32, device=dev)
        index0 = torch.randn(max_seq, INDEX_DIM, generator=gen, device=dev).to(torch.bfloat16)
    else:
        indices = torch.stack(
            [
                torch.randperm(max(cur_pos + s + 1, topk), generator=gen, device=dev)[:topk].sort().values
                for s in range(S)
            ]
        ).to(torch.int32)
        if cur_pos + 1 > topk:  # the reused selection always holds the newest token
            indices[:, -1] = torch.arange(cur_pos, cur_pos + S, device=dev, dtype=torch.int32)
        index0 = None
    op = Glm5MonoKernel(
        W, S, rank=rank, npes=npes, group=group, topk=topk, with_indexer=with_indexer, index_max_seq=max_seq
    )
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
        index_cache = index0.clone() if with_indexer else None
        pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
        out = op.forward(h, pos_t, kv, pe, indices, cos, sin, index_cache=index_cache)
        torch.cuda.synchronize()
        got = op.intermediates()
        got["x_out"] = out
        kv_ref, pe_ref = kv0.clone(), pe0.clone()
        report = []
        ref_indices = indices
        if with_indexer:
            index_ref = index0.clone()
            selected, index_q, index_w, _, _ = indexer_golden(W, h, got["q_a"], cur_pos, index_ref, cos, sin, topk=topk)
            got_index_q = got["index_q"].clone()
            for s in range(S):
                pos = cur_pos + s
                got_index_q[s, :, :PE_DIM] = rope(got_index_q[s, :, :PE_DIM], cos[pos], sin[pos])
            got_index_q = got_index_q.to(torch.bfloat16).float()
            ok &= _check("index_q", got_index_q, index_q, report)
            ok &= _check("index_w", got["index_w"], index_w, report)
            rows = slice(cur_pos, cur_pos + S)
            ok &= _check("index_k", index_cache[rows], index_ref[rows], report)
            selected_ok = all(torch.equal(got["indices"][s].sort().values, selected[s].sort().values) for s in range(S))
            ok &= selected_ok
            report.append(("index_sel", 0.0, 0.0, f"set_equal={selected_ok}"))
            # Preserve the kernel's ascending-token output order so split-attention
            # rounding is compared independently of the selector's order.
            ref_indices = got["indices"]
        ref = golden_layer(W, h, cur_pos, kv_ref, pe_ref, ref_indices, cos, sin, allreduce, topk=topk)
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
    op.close()
    return ok


def bench_rank(
    rank,
    npes,
    S,
    cur_pos,
    iters=320,
    group=None,
    seed=1234,
    with_indexer=False,
    expert_mxfp4=False,
    max_seq=MAX_SEQ,
    model_tp=None,
):
    """HIP-graph replay of ``GLM5_LAUNCHES_PER_STEP`` layers; returns us per layer."""
    from kernels.monokernel.glm import Glm5MonoKernel

    if cur_pos < 0 or cur_pos + S > max_seq:
        raise ValueError("positions must fit max_seq")
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    config = GLM5_CONFIG if model_tp is None else glm5_tp_config(model_tp)
    if model_tp is not None and model_tp != npes:
        raise ValueError("model TP geometry must match the number of ranks")
    W = make_weights(
        rank,
        heads=config.local_heads,
        device=dev,
        seed=seed,
        with_indexer=with_indexer,
        expert_mxfp4=expert_mxfp4,
        config=config,
    )
    gen = torch.Generator(device=dev).manual_seed(seed + 99)
    cos, sin = rope_table(max_seq, device=dev)
    kv = torch.randn(max_seq, KV_LORA, generator=gen, device=dev).to(torch.bfloat16)
    pe = torch.randn(max_seq, PE_DIM, generator=gen, device=dev).to(torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(max(cur_pos + s + 1, 2048), generator=gen, device=dev)[:2048] for s in range(S)]
    ).int()
    index_cache = (
        torch.randn(max_seq, INDEX_DIM, generator=gen, device=dev).to(torch.bfloat16) if with_indexer else None
    )
    launches_per_step = int(os.environ.get("GLM5_LAUNCHES_PER_STEP", "16"))
    op = Glm5MonoKernel(
        W,
        S,
        rank=rank,
        npes=npes,
        group=group,
        launches_per_step=launches_per_step,
        with_indexer=with_indexer,
        index_max_seq=max_seq,
    )
    h = torch.randn(S, 6144, generator=gen, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
    for _ in range(10):
        op.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x, index_cache=index_cache)
    torch.cuda.synchronize()
    import torch.distributed as dist

    if npes > 1:
        dist.barrier()
    if os.environ.get("MLA_MOE_TIMELINE"):
        top = Glm5MonoKernel(
            W, S, rank=rank, npes=npes, group=group, with_indexer=with_indexer, index_max_seq=max_seq, timeline=True
        )
        for _ in range(3):
            top.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x, index_cache=index_cache)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
        top.close()
    # a HIP graph of LAYERS layer launches + one step bump, like a decode step
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(launches_per_step):
            op.forward(
                h, pos_t, kv, pe, indices, cos, sin, x_out=x, layer=layer, advance=False, index_cache=index_cache
            )
        op.advance_step()
    # Short decode graphs otherwise spend a material fraction of the timed
    # window ramping clocks from the idle state.  Scale warmup with the run so
    # isolated and already-busy-machine measurements remain comparable.
    for _ in range(max(3, min(50, iters // 64))):
        graph.replay()
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // launches_per_step):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) * 1e3 / (iters // launches_per_step * launches_per_step)
    op.close()
    return elapsed


def _worker(
    rank,
    npes,
    S,
    cur_pos,
    iters,
    with_indexer,
    expert_mxfp4,
    max_seq,
    indexer_ties,
    model_tp,
    vary_expert_scales,
    results,
):
    import torch.distributed as dist

    master_port = os.environ.get("GLM5_MASTER_PORT", "29541")
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{master_port}", rank=rank, world_size=npes)
    if iters < 0:
        us = bench_rank(
            rank,
            npes,
            S,
            cur_pos,
            iters=-iters,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            max_seq=max_seq,
            model_tp=model_tp,
        )
        results[rank] = us
        print(f"[rank {rank}] {us:.1f} us/layer", flush=True)
    else:
        results[rank] = run_rank(
            rank,
            npes,
            S,
            cur_pos,
            iters,
            group=None,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            max_seq=max_seq,
            model_tp=model_tp,
            vary_expert_scales=vary_expert_scales,
            indexer_ties=indexer_ties,
        )
    dist.barrier()
    dist.destroy_process_group()


def run(
    npes,
    S,
    cur_pos,
    iters,
    with_indexer=False,
    expert_mxfp4=False,
    max_seq=MAX_SEQ,
    indexer_ties=False,
    model_tp=None,
    vary_expert_scales=False,
):
    if npes == 1:
        return run_rank(
            0,
            1,
            S,
            cur_pos,
            iters,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            max_seq=max_seq,
            model_tp=model_tp,
            vary_expert_scales=vary_expert_scales,
            indexer_ties=indexer_ties,
        )
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    results = mgr.dict()
    mp.spawn(
        _worker,
        args=(
            npes,
            S,
            cur_pos,
            iters,
            with_indexer,
            expert_mxfp4,
            max_seq,
            indexer_ties,
            model_tp,
            vary_expert_scales,
            results,
        ),
        nprocs=npes,
    )
    return all(results[r] for r in range(npes))


@pytest.mark.multi_gpu
@pytest.mark.parametrize("npes", [4, 8])
@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_layer_model_shards_mxfp4(npes, S):
    if torch.cuda.device_count() < npes:
        pytest.skip(f"needs {npes} GPUs")
    assert run(npes, S, 3000, 1, with_indexer=True, expert_mxfp4=True, model_tp=npes, vary_expert_scales=True)


@pytest.mark.multi_gpu
@pytest.mark.parametrize("npes", [4, 8])
@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_layer_model_shards_fp8(npes, S):
    if torch.cuda.device_count() < npes:
        pytest.skip(f"needs {npes} GPUs")
    assert run(npes, S, 3000, 1, with_indexer=True, model_tp=npes)


@pytest.mark.parametrize("S,cur_pos", [(1, 100), (1, 3000), (2, 3000)])
def test_layer_single_gpu(S, cur_pos):
    assert run(1, S, cur_pos, 2)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_layer_tp8():
    assert run(8, 1, 3000, 3)


@pytest.mark.parametrize(
    "S,cur_pos,expert_mxfp4", [(1, 0, False), (1, 2047, False), (1, 3000, False), (8, 0, True), (8, 3000, True)]
)
def test_layer_single_gpu_indexer(S, cur_pos, expert_mxfp4):
    assert run(1, S, cur_pos, 1, with_indexer=True, expert_mxfp4=expert_mxfp4)


def test_reference_default_dtype():
    """ATOM's BF16 default must not lower golden accumulators or probabilities."""
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        dev = torch.device("cuda", 0)
        W = make_weights(0, device=dev)
        gen = torch.Generator(device=dev).manual_seed(42)
        h = torch.randn(1, 6144, generator=gen, device=dev).bfloat16()
        kv = torch.randn(MAX_SEQ, KV_LORA, generator=gen, device=dev).bfloat16()
        pe = torch.randn(MAX_SEQ, PE_DIM, generator=gen, device=dev).bfloat16()
        indices = torch.arange(2048, dtype=torch.int32, device=dev)[None]
        cos, sin = rope_table(MAX_SEQ, device=dev)

        def allreduce(value):
            assert value.dtype == torch.float32
            return value

        outputs = []
        for dtype in (torch.float32, torch.bfloat16):
            torch.set_default_dtype(dtype)
            outputs.append(golden_layer(W, h, 100, kv.clone(), pe.clone(), indices, cos, sin, allreduce))
            assert outputs[-1]["prob"].dtype == torch.float32
        for name, value in outputs[0].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, outputs[1][name]), name
    finally:
        torch.set_default_dtype(original)


def test_router_preserves_close_scores():
    """A higher expert score must win even when its lead is below 256 FP32 ULPs."""
    from kernels.monokernel.glm import Glm5MonoKernel

    dev = torch.device("cuda", 0)
    W = make_weights(0, device=dev)
    W.t["w_r"].zero_()  # All uncorrected sigmoid scores are exactly 0.5.
    W.t["bias"].fill_(-10)
    W.t["bias"][:7] = 8  # Equal scores: lower expert IDs break the exact tie.
    W.t["bias"][92] = 7
    W.t["bias"][226] = 7 + 9.5367431640625e-5
    h = torch.ones(1, 6144, dtype=torch.bfloat16, device=dev)
    kv = torch.zeros(MAX_SEQ, KV_LORA, dtype=torch.bfloat16, device=dev)
    pe = torch.zeros(MAX_SEQ, PE_DIM, dtype=torch.bfloat16, device=dev)
    indices = torch.zeros(1, 2048, dtype=torch.int32, device=dev)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    with Glm5MonoKernel(W, 1) as op:
        op.forward(h, torch.zeros(1, dtype=torch.int32, device=dev), kv, pe, indices, cos, sin)
        torch.cuda.synchronize()
        assert op.intermediates()["sel"].tolist() == [[256, 0, 1, 2, 3, 4, 5, 6, 226]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=1)
    ap.add_argument("--model-tp", type=int, choices=(4, 8), help="use real GLM TP4/TP8 shard geometry")
    ap.add_argument("-S", type=int, default=1)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true", help="time back-to-back launches")
    ap.add_argument("--bench-iters", type=int, default=320, help="layer launches timed by --bench")
    ap.add_argument("--indexer", action="store_true", help="fuse indexer projection, scoring, and top-k")
    ap.add_argument("--mxfp4", action="store_true", help="use native MXFP4 expert weights")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ, help="KV/index cache capacity; multiple of 64")
    a = ap.parse_args()
    if a.bench:
        if a.npes == 1:
            print(
                f"{bench_rank(0, 1, a.S, a.pos, iters=a.bench_iters, with_indexer=a.indexer, expert_mxfp4=a.mxfp4, max_seq=a.max_seq, model_tp=a.model_tp):.1f} us/layer"
            )
        else:
            run(
                a.npes,
                a.S,
                a.pos,
                -a.bench_iters,
                with_indexer=a.indexer,
                expert_mxfp4=a.mxfp4,
                max_seq=a.max_seq,
                model_tp=a.model_tp,
            )
        sys.exit(0)
    ok = run(
        a.npes,
        a.S,
        a.pos,
        a.iters,
        with_indexer=a.indexer,
        expert_mxfp4=a.mxfp4,
        max_seq=a.max_seq,
        model_tp=a.model_tp,
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
