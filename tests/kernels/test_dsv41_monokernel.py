# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""DeepSeek-V4.1 gfx1250 mono decode (``kernels/monokernel/dsv41``) against the torch
golden (``dsv41.reference``), and its per-layer time.

* host: the K1 / K2 ABIs and the weight shuffle (no device);
* K1 on one GPU (one TP4 rank, no peers);
* TP4, token parallel, one process a GPU over a HIP VMM peer buffer:
  - accuracy: K1 (seam outputs on the rank's tokens, q and the window row on
    every token) and K2 (seam outputs, ``normed``, the router's top-k and the
    reduced MoE ``out``, the MoE teacher-forced on the kernel's own rows and route);
  - performance: one CUDA graph of a decode step (clears, ``step_begin``, then
    ``LAYERS`` layers of K1 + K2, every layer its own weights), each kernel's
    device time from torch.profiler, held to a budget. The step replays a served
    one's routing (``data/dsv41_route_s48.npz``: an ATOM conc-8 DSpark verify step,
    8 requests x 6 tokens; each layer's top-k ids / weights and key counts): K2
    built with ``forced_route`` still runs its router and waits for the logits,
    then publishes the recorded picks.

    python3 -m pytest tests/kernels/test_dsv41_monokernel.py -s
    python3 tests/kernels/test_dsv41_monokernel.py --perf      # the TP4 timing alone

Budgets (us a layer, S = 48): ``DSV41_K1_US_MAX`` / ``DSV41_K2_US_MAX``, 29 / 112 by
default: the measured 25.1 / 97 on MI455X (2026-11-04) + 15 %. No indexer in the timed step.
"""

import argparse
import json
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.monokernel.dsv41 import op  # noqa: E402
from kernels.monokernel.dsv41 import reference as ref  # noqa: E402
from kernels.monokernel.dsv41.attention import KMETA  # noqa: E402
from kernels.monokernel.dsv41.attn_pre import scratch_bytes as k1_scratch_bytes  # noqa: E402
from kernels.monokernel.dsv41.dims import HC, HEAD_DIM, HIDDEN, Dims  # noqa: E402
from kernels.monokernel.dsv41.layer_post import LayerPostBuild  # noqa: E402
from kernels.monokernel.dsv41.layer_post import scratch_layout as k2_layout  # noqa: E402
from kernels.monokernel.dsv41.layout import peer_bytes  # noqa: E402
from kernels.monokernel.dsv41.moe import FROUTE_LAYERS, ROUTE_COPIES  # noqa: E402
from kernels.monokernel.dsv41.moe_shape import ROUTER_PARTS, TOPK  # noqa: E402
from kernels.monokernel.dsv41.packing import (  # noqa: E402
    pack_dense_fp8,
    pack_moe_fp4,
    shuffle_gfx1250,
    unshuffle_gfx1250,
    validate_dense_fp8,
)
from kernels.monokernel.execution_gfx1250 import BLOCKS  # noqa: E402
from kernels.monokernel.sync_gfx1250 import DIAG_WORDS  # noqa: E402

TP, S = 4, 48
LAYERS = 8  # the timed step's layers (each its own weights: none in GL2 from the one before)
ROUTE_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dsv41_route_s48.npz")
SLOTS = 1024  # the accuracy case's window ring
QK_SCALE = torch.tensor(HEAD_DIM**-0.5 * 1.4426950408889634, dtype=torch.float32)
K1_REL_L2 = {"res_out": 0.0, "pre_out": 2e-3, "post_out": 2e-3, "comb_out": 5e-3, "ring": 2e-2, "q_out": 2e-2}
K2_REL_L2 = {"res_out": 1e-2, "pre_out": 5e-3, "post_out": 5e-3, "comb_out": 1e-2, "normed": 2e-2, "logits": 1e-2,
             "out": 3e-2}  # fmt: skip
K1_US_MAX = float(os.environ.get("DSV41_K1_US_MAX", "29"))
K2_US_MAX = float(os.environ.get("DSV41_K2_US_MAX", "112"))


def _gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    from flydsl.runtime.device import get_rocm_arch

    return str(get_rocm_arch() or "") == "gfx1250"


def _rel_l2(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _pairs(scratch, layout, name):
    """A scratch region's (value, tag) pairs -> values (int32)."""
    off, n = layout[name]
    return scratch[off : off + n].view(torch.int32).view(-1, 2)[:, 0]


# ---------------------------------------------------------------------------- host


def test_abi_pack_by_name():
    vals = {n: i for i, n in enumerate(op.K2_ABI.names)}
    assert op.K2_ABI.pack(vals) == list(range(len(op.K2_ABI.names)))
    assert op.K2_ABI.names[-1] == "nxt"
    for abi in (op.K1_ABI, op.K2_ABI):
        assert len(set(abi.names)) == len(abi.names)
    with pytest.raises(TypeError, match="missing"):
        op.K2_ABI.pack({n: 0 for n in op.K2_ABI.names[:-1]})
    with pytest.raises(TypeError, match="unexpected"):
        op.K2_ABI.pack({**vals, "bogus": 0})


@pytest.mark.parametrize("n,k", [(16, 32), (2048, 1280), (5120, 576)])
def test_shuffle_round_trip(n, k):
    w = torch.randint(0, 255, (n, k), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)
    assert torch.equal(unshuffle_gfx1250(shuffle_gfx1250(w), n, k), w)


# ---------------------------------------------------------------------------- K1


def _k1(t, tp, rank=0, peer=(0, 0), tok_par=False) -> dict:
    """One rank's K1 -> {output: rel_l2}; ``tok_par``: the seam outputs of the rank's
    own tokens only (q and the window rows: every token, through the peers)."""
    dev = "cuda"
    w = ref.random_layer(tp, device=dev)
    x = ref.random_inputs(t, device=dev)
    want = ref.attn_pre({k: v.cpu() for k, v in w.items()}, {k: v.cpu() for k, v in x.items()}, tp)
    pk = {n: pack_dense_fp8(w[n], w[s]) for n, s in (("wqkv", "wqkv_s"), ("wqb", "wqb_s"), ("iq_w", "iq_ws"))}
    for n, p in pk.items():
        validate_dense_fp8(p, w[n], w["iq_ws" if n == "iq_w" else n + "_s"])
    out = {
        "res_out": torch.empty(t, HC, HIDDEN, dtype=torch.bfloat16, device=dev),
        "post_out": torch.empty(t, HC, device=dev),
        "comb_out": torch.empty(t, HC, HC, device=dev),
        "pre_out": torch.empty(t, HC, device=dev),
        "q_out": torch.empty(t, Dims(tp).heads * 512, dtype=torch.bfloat16, device=dev),
    }
    ring = torch.zeros(t, 512, dtype=torch.bfloat16, device=dev)
    ring_rows = torch.arange(t, dtype=torch.int32, device=dev)
    pos = torch.stack([x["pos"].int(), torch.zeros_like(x["pos"]).int()], 1).contiguous()
    scratch = torch.zeros(k1_scratch_bytes(t), dtype=torch.uint8, device=dev)
    dummy = torch.zeros(16, dtype=torch.int32, device=dev)
    io = {
        "normed": torch.zeros(t, HIDDEN, dtype=torch.bfloat16, device=dev),
        "qr": torch.zeros(t, 1280, dtype=torch.uint8, device=dev),
        "qrs": torch.zeros(t, 40, dtype=torch.uint8, device=dev),
        "iq": torch.zeros(t, 32, 128, dtype=torch.uint8, device=dev),
        "iqs": torch.zeros(t, 32, device=dev),
        "iw": torch.zeros(t, 32, device=dev),
    }
    wd = {n: (p.weight.to(dev), p.scale.to(dev)) for n, p in pk.items()}
    p = lambda v: v.data_ptr()  # noqa: E731
    op.build_k1(t, tp, tok_par=tok_par)(
        p(x["res_in"]), p(x["pend"]), p(x["post_in"]), p(x["comb_in"]), p(x["pre_in"]),
        p(w["hc_fn"]), p(w["hc_scale"]), p(w["hc_base"]), p(w["attn_w"]),
        p(wd["wqkv"][0]), p(wd["wqkv"][1]), p(w["qn_w"]), p(w["kvn_w"]),
        p(wd["wqb"][0]), p(wd["wqb"][1]), p(x["cos"]), p(x["sin"]), p(pos),
        p(ring), p(ring_rows), 0,
        p(out["res_out"]), p(out["post_out"]), p(out["comb_out"]), p(out["pre_out"]), p(out["q_out"]),
        p(io["normed"]), p(io["qr"]), p(io["qrs"]), p(dummy),
        p(wd["iq_w"][0]), p(wd["iq_w"][1]), p(w["iw_w"]),
        p(io["iq"]), p(io["iqs"]), p(io["iw"]), p(scratch), rank, 0, 0, *peer,
    )  # fmt: skip
    torch.cuda.synchronize()
    got = {k: v.cpu() for k, v in out.items()} | {"ring": ring.cpu()}
    n_own = t // tp if tok_par else t
    own = slice(rank * n_own, rank * n_own + n_own)
    seam = ("res_out", "pre_out", "post_out", "comb_out")
    return {k: _rel_l2(got[k][own] if k in seam else got[k], want[k][own] if k in seam else want[k]) for k in K1_REL_L2}


def _over(errs, budget):
    return {k: e for k, e in errs.items() if not e <= budget[k]}


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not _gfx1250(), reason="dsv41 mono kernels need gfx1250")
@pytest.mark.parametrize("tokens", [1, 16])
def test_k1_single_gpu(tokens):
    errs = _k1(tokens, TP)
    print(f"K1 S={tokens}: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items()))
    assert not _over(errs, K1_REL_L2), f"S={tokens}: over budget {_over(errs, K1_REL_L2)}"


# ---------------------------------------------------------------------------- K2


def _k2_case(s, tp, rank, seed=0):
    """A rank's K2 inputs: the tokens' seam rows and the KV pool shared by the TP
    group (one seed), the rank's own attention / MoE weights, query heads and sinks;
    the golden attention over each token's window."""
    d = Dims(tp)
    g = torch.Generator().manual_seed(seed)
    x = ref.random_inputs(s, seed=seed + 1)
    counts = torch.randint(1, SLOTS + 1, (s,), generator=g)
    counts[0] = SLOTS
    pool = torch.randn(s * SLOTS, HEAD_DIM, generator=g).to(torch.bfloat16)
    first = torch.randint(0, SLOTS, (s,), generator=g)
    gr = torch.Generator().manual_seed(seed + 1000 * (rank + 1))
    w = ref.random_post_layer(tp, seed=seed + 2 + 1000 * rank)
    w |= {k: v for k, v in ref.random_post_layer(tp, seed=seed + 2).items() if k.startswith("ffn_")}  # replicated
    q = (torch.randn(s, d.heads, HEAD_DIM, generator=gr) * 0.1).to(torch.bfloat16)
    sink = torch.randn(d.heads, generator=gr)
    kmeta = torch.zeros(s, KMETA, dtype=torch.int32)
    kmeta[:, 0], kmeta[:, 1], kmeta[:, 3] = first.int(), counts.int(), (torch.arange(s) * SLOTS).int()
    rows = [pool[t * SLOTS + (first[t] + torch.arange(int(counts[t]))) % SLOTS].float() for t in range(s)]
    attn = torch.stack([ref.split_attention(q[t].float(), rows[t], sink, QK_SCALE) for t in range(s)])
    wm = ref.random_moe_layer(tp, seed=3 + 1000 * rank, device="cuda", router_seed=3)
    return {"x": x, "w": w, "q": q, "sink": sink, "pool": pool, "kmeta": kmeta, "attn": attn, "wm": wm}


def _k2(s, tp, rank, buf, gather) -> dict:
    """One rank's token-parallel K2 (bounded waits) -> {output: rel_l2}, plus
    ``route`` (1.0: the kernel's top-k is its own logits' golden top-k) and
    ``waits`` (CTAs that gave up a wait)."""
    dev = "cuda"
    c = _k2_case(s, tp, rank)
    w, x, wm = c["w"], c["x"], c["wm"]
    opts = {"tok_par": True}
    layout = k2_layout(LayerPostBuild(tokens=s, tp=tp, **opts))
    diag_off = -(-max(o + n for o, n in layout.values()) // 256) * 256
    scratch = torch.zeros(diag_off + BLOCKS * (DIAG_WORDS + 1) * 4, dtype=torch.uint8, device=dev)
    n_own = s // tp
    own = slice(rank * n_own, rank * n_own + n_own)
    keep = []

    def dv(t):
        keep.append(t.to(dev).contiguous())
        return keep[-1]

    pk = {n: pack_dense_fp8(w[n], w[n + "_s"]) for n in ("woa", "wob")}
    sgu, sw2 = pack_dense_fp8(wm["sgu"], wm["sgu_s"]), pack_dense_fp8(wm["sw2"], wm["sw2_s"])
    w13, w2 = pack_moe_fp4(wm["w13"], wm["w13_s"], gugu=True), pack_moe_fp4(wm["w2"], wm["w2_s"], gugu=False)
    xd = {k: dv(v) for k, v in x.items()}
    wd = {k: dv(v) for k, v in w.items()}
    pos = dv(torch.stack([x["pos"].int(), torch.zeros_like(x["pos"]).int()], 1))
    o = {
        "res_out": torch.empty(s, HC, HIDDEN, dtype=torch.bfloat16, device=dev),
        "post_out": torch.empty(s, HC, device=dev), "comb_out": torch.empty(s, HC, HC, device=dev),
        "pre_out": torch.empty(s, HC, device=dev),
        "normed": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
        "out": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
    }  # fmt: skip
    p = lambda v: v.data_ptr()  # noqa: E731
    launch = op.build_k2(s, tp, diag_off=diag_off, **opts)
    dist.barrier()
    launch(
        p(dv(c["q"])), p(dv(c["pool"])), p(dv(c["kmeta"])), 0, SLOTS, p(dv(c["sink"])),
        int(QK_SCALE.view(torch.int32)), p(pos), p(xd["cos"]), p(xd["sin"]),
        p(dv(pk["woa"].weight)), p(dv(pk["woa"].scale)), p(dv(pk["wob"].weight)), p(dv(pk["wob"].scale)),
        p(xd["res_in"]), p(xd["post_in"]), p(xd["comb_in"]), p(xd["pre_in"]),
        p(wd["ffn_hc_fn"]), p(wd["ffn_hc_scale"]), p(wd["ffn_hc_base"]), p(wd["ffn_w"]),
        p(o["res_out"]), p(o["post_out"]), p(o["comb_out"]), p(o["pre_out"]), p(o["normed"]),
        p(wm["gate_w"]), p(wm["bias"]), p(w13.weight), p(w13.scale), p(w2.weight), p(w2.scale),
        p(dv(sgu.weight)), p(dv(sgu.scale)), p(dv(sw2.weight)), p(dv(sw2.scale)), p(o["out"]),
        p(scratch), buf.local_address, p(buf.addresses), rank, 0, 0, 0, 0, 0, 0, 0,
    )  # fmt: skip
    torch.cuda.synchronize()
    dist.barrier()
    sc = scratch.cpu()
    rec = sc[diag_off : diag_off + BLOCKS * DIAG_WORDS * 4].view(torch.int32).view(BLOCKS, DIAG_WORDS)
    waits = int(rec[:, 0].ne(0).sum())
    got = {k: v.cpu() for k, v in o.items()}
    # K2a: the golden with every rank's golden wo_b partial
    parts = torch.stack([v.float() for v in gather(ref.attn_post(w, x, c["attn"], tp, rank, torch.zeros(tp, s, HIDDEN))["partial"])])
    want = ref.attn_post(w, x, c["attn"], tp, rank, parts)
    errs = {k: _rel_l2(got[k][own], want[k].reshape(got[k].shape)[own]) for k in ("res_out", "pre_out", "post_out", "comb_out")}
    errs["normed"] = _rel_l2(got["normed"], want["normed"])
    # K2b on the kernel's own rows: its logits, its top-k, the reduced MoE out
    e = wm["gate_w"].shape[0]
    lp = _pairs(sc, layout, "logit").view(torch.float32).view(s, e, ROUTER_PARTS)
    logits = ref.bf16(lp.sum(-1))
    xk = o["normed"].float()
    errs["logits"] = _rel_l2(logits, ref.bf16((xk @ wm["gate_w"].float().T).cpu()))
    route = _pairs(sc, layout, "route").view(ROUTE_COPIES, s, 2, TOPK)[0]
    ids, wts = route[:, 0].long(), route[:, 1].view(torch.float32)
    errs["route"] = float(torch.equal(ids, ref.route_topk(logits, wm["bias"].cpu(), TOPK)[0]))
    full = torch.stack([v.float() for v in gather(ref.moe_partial(wm, xk, ids, wts.to(dev), tp).cpu())])
    errs["out"] = _rel_l2(got["out"][own], ref.bf16(full.sum(0))[own])
    errs["waits"] = waits
    return errs


def _accuracy_rank(rank, tp, s, port, results):
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=tp)

    def gather(own):
        parts = [None] * tp
        dist.all_gather_object(parts, own.cpu())
        return parts

    from kernels.monokernel.dsv41.peer import VmmPeerBuffer

    try:
        with VmmPeerBuffer(peer_bytes(s, tp), rank=rank, npes=tp) as buf:
            dist.barrier()  # every rank's buffer zeroed before any push
            k1 = _k1(s, tp, rank, peer=(buf.local_address, buf.addresses.data_ptr()), tok_par=True)
            dist.barrier()
        with VmmPeerBuffer(peer_bytes(s, tp), rank=rank, npes=tp) as buf:
            dist.barrier()
            k2 = _k2(s, tp, rank, buf, gather)
            dist.barrier()
        results[rank] = {"k1": k1, "k2": k2}
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------- perf


def _perf_rank(rank, tp, s, layers, port, results, warmup=5, replays=10):
    """One rank of a captured decode step of ``layers`` layers -> each kernel's mean
    device time a call and the step's replay wall time (us)."""
    import torch.profiler as tprof

    from kernels.monokernel.dsv41.peer import VmmPeerBuffer, memset_async
    from kernels.monokernel.dsv41.layer_post import scratch_bytes as k2_scratch_bytes
    from kernels.monokernel.dsv41.step import barrier_offset, build_step_begin, symmetric_bytes

    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=tp)
    dev, bf = "cuda", torch.bfloat16
    keep = []

    def dv(t):
        keep.append(t.to(dev).contiguous())
        return keep[-1]

    try:
        w1 = ref.random_layer(tp, device=dev)
        w2a = ref.random_post_layer(tp, seed=2 + 1000 * rank)
        wm = ref.random_moe_layer(tp, seed=3 + 1000 * rank, device=dev, router_seed=3)
        k1p = {n: pack_dense_fp8(w1[n], w1[sc]) for n, sc in (("wqkv", "wqkv_s"), ("wqb", "wqb_s"), ("iq_w", "iq_ws"))}
        k2p = {n: pack_dense_fp8(w2a[n], w2a[n + "_s"]) for n in ("woa", "wob")}
        sgu, sw2 = pack_dense_fp8(wm["sgu"], wm["sgu_s"]), pack_dense_fp8(wm["sw2"], wm["sw2_s"])
        w13, w2 = pack_moe_fp4(wm["w13"], wm["w13_s"], gugu=True), pack_moe_fp4(wm["w2"], wm["w2_s"], gugu=False)
        base = ({k: dv(v) for k, v in (w1 | w2a).items()},
                {n: dv(t) for n, pk in (k1p | k2p).items() for n, t in ((n, pk.weight), (n + "_s", pk.scale))},
                [dv(t) for t in (sgu.weight, sgu.scale, sw2.weight, sw2.scale)],
                [wm["gate_w"], wm["bias"], w13.weight, w13.scale, w2.weight, w2.scale])  # fmt: skip
        # every layer its own copy: nothing a layer reads is in GL2 from the one before
        lw = [base] + [tuple({k: v.clone() for k, v in m.items()} if isinstance(m, dict) else [v.clone() for v in m]
                             for m in base) for _ in range(layers - 1)]  # fmt: skip
        x = ref.random_inputs(s)
        cos, sin = dv(x["cos"]), dv(x["sin"])
        pos = dv(torch.stack([x["pos"].int(), torch.zeros_like(x["pos"]).int()], 1))
        g = torch.Generator().manual_seed(7)
        # the served step: a request's tokens share its KV rows; each layer its own key
        # counts and (``forced_route``) its recorded top-k
        import numpy as np

        rec = np.load(ROUTE_NPZ)
        assert rec["topkid"].shape[1] == s and layers <= rec["topkid"].shape[0]
        keys = torch.from_numpy(rec["total_keys"].astype("int32"))
        slots = int(keys.max())
        pool = dv(torch.randn(s * slots, HEAD_DIM, generator=g).to(bf))
        kmeta_l = []
        for li in range(layers):
            km = torch.zeros(s, KMETA, dtype=torch.int32)
            km[:, 1], km[:, 3] = keys[li], torch.from_numpy(rec["batch_ids"].astype("int32")) * slots
            kmeta_l.append(dv(km))
        fr = torch.zeros(FROUTE_LAYERS, s, TOPK, 2, dtype=torch.int32)
        fr[:layers, ..., 0] = torch.from_numpy(rec["topkid"][:layers].astype("int32"))
        fr[:layers, ..., 1] = torch.from_numpy(rec["route_weights"][:layers].astype("float32")).view(torch.int32)
        froute = dv(fr.view(torch.uint8).flatten())
        sink = dv(torch.randn(Dims(tp).heads, generator=g))

        def acts():
            return {"res": dv(torch.randn(s, HC, HIDDEN, generator=g).to(bf)), "post": dv(torch.rand(s, HC, generator=g) * 2),
                    "comb": dv(torch.full((s, HC, HC), 1.0 / HC)), "pre": dv(torch.rand(s, HC, generator=g) + 0.1)}  # fmt: skip

        state, mid = [acts() for _ in range(layers + 1)], [acts() for _ in range(layers)]
        pend = [dv(torch.randn(s, HIDDEN, generator=g).to(bf)) for _ in range(layers + 1)]
        q = [dv(torch.zeros(s, Dims(tp).heads * HEAD_DIM, dtype=bf)) for _ in range(layers)]
        normed = [dv(torch.zeros(s, HIDDEN, dtype=bf)) for _ in range(layers)]
        ring, ring_rows = dv(torch.zeros(s, 512, dtype=bf)), dv(torch.arange(s, dtype=torch.int32))
        io = {"qr": dv(torch.zeros(s, 1280, dtype=torch.uint8)), "qrs": dv(torch.zeros(s, 40, dtype=torch.uint8)),
              "aux": dv(torch.zeros(16, dtype=torch.int32)), "n1": dv(torch.zeros(s, HIDDEN, dtype=bf)),
              "iq": dv(torch.zeros(s, 32, 128, dtype=torch.uint8)), "iqs": dv(torch.zeros(s, 32)), "iw": dv(torch.zeros(s, 32))}  # fmt: skip
        scr1 = dv(torch.zeros(k1_scratch_bytes(s), dtype=torch.uint8))
        k2key = {"tok_par": True, "forced_route": True}
        scr2 = dv(torch.zeros(k2_scratch_bytes(LayerPostBuild(tokens=s, tp=tp, **k2key)), dtype=torch.uint8))
        froute_off = k2_layout(LayerPostBuild(tokens=s, tp=tp, **k2key))["froute"][0]
        epoch = dv(torch.zeros(4, dtype=torch.int32))
        buf = VmmPeerBuffer(symmetric_bytes(s, tp), rank=rank, npes=tp)
        sym, peers, bar = buf.local_address, buf.addresses, barrier_offset(s, tp)
        k1, k2, begin = op.build_k1(s, tp, tok_par=True), op.build_k2(s, tp, **k2key), build_step_begin(tp)
        p = lambda t: t.data_ptr()  # noqa: E731

        def step(stream):
            scr1.zero_()
            scr2.zero_()
            scr2[froute_off : froute_off + froute.numel()].copy_(froute)
            memset_async(sym, bar, stream)
            begin(p(epoch), sym, p(peers), bar, rank, stream=stream)
            for li in range(layers):
                a, m, z = state[li], mid[li], state[li + 1]
                lwd, lpk, lsh, lmw = lw[li]
                k1(
                    p(a["res"]), p(pend[li]), p(a["post"]), p(a["comb"]), p(a["pre"]),
                    p(lwd["hc_fn"]), p(lwd["hc_scale"]), p(lwd["hc_base"]), p(lwd["attn_w"]),
                    p(lpk["wqkv"]), p(lpk["wqkv_s"]), p(lwd["qn_w"]), p(lwd["kvn_w"]),
                    p(lpk["wqb"]), p(lpk["wqb_s"]), p(cos), p(sin), p(pos), p(ring), p(ring_rows), 0,
                    p(m["res"]), p(m["post"]), p(m["comb"]), p(m["pre"]), p(q[li]),
                    p(io["n1"]), p(io["qr"]), p(io["qrs"]), p(io["aux"]),
                    p(lpk["iq_w"]), p(lpk["iq_w_s"]), p(lwd["iw_w"]),
                    p(io["iq"]), p(io["iqs"]), p(io["iw"]), p(scr1), rank, li, 0, sym, p(peers), stream=stream,
                )  # fmt: skip
                k2(
                    p(q[li]), p(pool), p(kmeta_l[li]), 0, slots, p(sink), int(QK_SCALE.view(torch.int32)), p(pos),
                    p(cos), p(sin), p(lpk["woa"]), p(lpk["woa_s"]), p(lpk["wob"]), p(lpk["wob_s"]),
                    p(m["res"]), p(m["post"]), p(m["comb"]), p(m["pre"]),
                    p(lwd["ffn_hc_fn"]), p(lwd["ffn_hc_scale"]), p(lwd["ffn_hc_base"]), p(lwd["ffn_w"]),
                    p(z["res"]), p(z["post"]), p(z["comb"]), p(z["pre"]), p(normed[li]),
                    *(p(t) for t in lmw), *(p(t) for t in lsh), p(pend[li + 1]),
                    p(scr2), sym, p(peers), rank, li, 0, 0, 0, 0, 0, 0, stream=stream,
                )  # fmt: skip

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step(stream)  # eager once: builds every kernel
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                step(stream)
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(replays):
            graph.replay()
        t1.record()
        torch.cuda.synchronize()
        dist.barrier()
        with tprof.profile(activities=[tprof.ProfilerActivity.CUDA]) as prof:
            for _ in range(replays):
                graph.replay()
            torch.cuda.synchronize()
        dist.barrier()

        def dev_us(e):
            return float(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0) or 0)

        k = {}
        for e in prof.key_averages():
            for stem, name in (("attn_pre", "k1_us"), ("layer_post", "k2_us")):
                if stem in e.key and dev_us(e) > 0:
                    k[name] = dev_us(e) / e.count
        results[rank] = k | {"step_us": t0.elapsed_time(t1) * 1e3 / replays}
        graph = None
        buf.close()
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------- TP4


def _spawn(fn, *args):
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(fn, args=(TP, *args, port, results), nprocs=TP)
    return [results[r] for r in range(TP)]


def run_accuracy(s=S):
    per = _spawn(_accuracy_rank, s)
    bad = []
    for r, res in enumerate(per):
        print(f"rank {r} K1: " + ", ".join(f"{k} {v:.2e}" for k, v in res["k1"].items()))
        print(f"rank {r} K2: " + ", ".join(f"{k} {v:.2e}" for k, v in res["k2"].items()))
        bad += [f"rank {r} K1 {k}" for k in _over(res["k1"], K1_REL_L2)]
        bad += [f"rank {r} K2 {k}" for k in _over({k: res["k2"][k] for k in K2_REL_L2}, K2_REL_L2)]
        if res["k2"]["route"] != 1.0:
            bad.append(f"rank {r} K2 route ids differ")
        if res["k2"]["waits"]:
            bad.append(f"rank {r} K2: {res['k2']['waits']} CTAs gave up a wait")
    return bad


def run_perf(s=S, layers=LAYERS):
    per = _spawn(_perf_rank, s, layers)
    k1 = sum(r["k1_us"] for r in per) / TP
    k2 = sum(r["k2_us"] for r in per) / TP
    step = max(r["step_us"] for r in per)
    res = {"tokens": s, "tp": TP, "layers": layers, "k1_us": round(k1, 1), "k2_us": round(k2, 1),
           "layer_wall_us": round(step / layers, 1), "k1_us_max": K1_US_MAX, "k2_us_max": K2_US_MAX}  # fmt: skip
    print(json.dumps(res), flush=True)
    return res


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.multi_gpu
@pytest.mark.skipif(not _gfx1250() or torch.cuda.device_count() < TP, reason=f"needs {TP} gfx1250 GPUs")
def test_layer_tp4_accuracy():
    bad = run_accuracy()
    assert not bad, bad


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.multi_gpu
@pytest.mark.skipif(not _gfx1250() or torch.cuda.device_count() < TP, reason=f"needs {TP} gfx1250 GPUs")
def test_layer_tp4_perf():
    res = run_perf()
    assert res["k1_us"] <= K1_US_MAX and res["k2_us"] <= K2_US_MAX, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true", help="the TP4 timing alone")
    ap.add_argument("--accuracy", action="store_true", help="the TP4 accuracy alone")
    ap.add_argument("--layers", type=int, default=LAYERS)
    a = ap.parse_args()
    if a.perf or not a.accuracy:
        run_perf(layers=a.layers)
    if a.accuracy or not a.perf:
        bad = run_accuracy()
        print("PASS" if not bad else f"FAIL {bad}")
