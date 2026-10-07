# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check :mod:`kernels.monokernel.dsv4.reference` against DeepSeek's own reference.

The oracle is DeepSeek-V4's unmodified ``inference/model.py`` with a pure-torch stand-in
for its tilelang kernels; point ``DSV4_ORACLE_DIR`` at it (``model.py``, ``kernel.py``,
``fast_hadamard_transform.py``) or the tests skip.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from kernels.monokernel.dsv4.config import MoeMode, moe_format
from kernels.monokernel.dsv4.reference import (
    V4Config,
    bf,
    compress_step,
    contiguous_pool,
    dequant,
    expert_matrix,
    fp4_row_bytes,
    fp8_mats,
    golden_layer,
    indexer_step,
    make_weights,
    pack_fp4,
    qkv_a_matrix,
    qkv_a_split,
    quant_dequant_fp4,
    rmsnorm,
    rope_table,
    unpack_fp4,
)

ORACLE_DIR = os.environ.get("DSV4_ORACLE_DIR", "")
# top-k margin below which the two sides may legitimately route to different experts
NEAR_TIE = 0.05

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _oracle():
    if not ORACLE_DIR or not os.path.isfile(os.path.join(ORACLE_DIR, "model.py")):
        pytest.skip(f"DeepSeek-V4 oracle not found (DSV4_ORACLE_DIR={ORACLE_DIR!r})")
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    import model as oracle_model

    # only Transformer.__init__ sets this global; the submodules built here need it too
    oracle_model.scale_fmt = "ue8m0"
    return oracle_model


def _cfg(hc_mult=4, compress_ratio=0, max_seq=1024):
    # small but structurally faithful: nope_dim a multiple of 64, V4's 64 rope lanes
    return V4Config(
        heads=8,
        hidden=256,
        q_lora=128,
        head_dim=128,
        rope_dim=64,
        o_groups=2,
        o_lora=64,
        n_experts=8,
        top_k=2,
        inter=128,  # must be a multiple of the 128 FP8 block
        window=32,
        hc_mult=hc_mult,
        compress_ratio=compress_ratio,
        max_seq=max_seq,
    )


def _oracle_modules(om, cfg, device):
    args = om.ModelArgs(
        max_batch_size=1,
        max_seq_len=256,
        dtype="bf16",
        scale_fmt="ue8m0",  # the checkpoint's power-of-two scales, as the golden and the kernel use
        scale_dtype="fp32",
        vocab_size=32,
        dim=cfg.hidden,
        moe_inter_dim=cfg.inter,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=cfg.heads,
        n_routed_experts=cfg.n_experts,
        n_shared_experts=1,
        n_activated_experts=cfg.top_k,
        score_func="sqrtsoftplus",
        route_scale=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_dim,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora,
        window_size=cfg.window,
        compress_ratios=(0,),
        norm_eps=cfg.eps,
        rope_theta=cfg.rope_theta,
        original_seq_len=0,
    )
    with torch.device(device):
        attn = om.Attention(0, args)
        moe = om.MoE(0, args)
    return attn, moe


@torch.no_grad()
def _load_oracle_weights(attn, moe, W, cfg, weight_fmt):
    t = W.t
    dq = {n: dequant(t[f"w_{n}"], t[f"s_{n}"], bk) for n, (_, _, bk) in fp8_mats(cfg).items()}
    bf16 = torch.bfloat16

    attn.wq_a.weight.copy_(dq["qkv_a"][: cfg.q_lora].to(bf16))
    attn.wkv.weight.copy_(dq["qkv_a"][cfg.q_lora : cfg.q_lora + cfg.head_dim].to(bf16))
    attn.wq_b.weight.copy_(dq["q_b"].to(bf16))
    attn.wo_a.weight.copy_(dq["o_a"].to(bf16))
    attn.wo_b.weight.copy_(dq["o_b"].to(bf16))
    attn.q_norm.weight.copy_(t["g_q"].float())
    attn.kv_norm.weight.copy_(t["g_kv"].float())
    attn.attn_sink.copy_(t["attn_sink"].float())

    moe.gate.weight.copy_(t["w_r"].to(bf16))
    moe.gate.bias.copy_(t["bias"].float())
    for e in range(cfg.n_experts + 1):
        ug = expert_matrix(t, "ug", e, cfg, weight_fmt)
        dn = expert_matrix(t, "dn", e, cfg, weight_fmt)
        target = moe.shared_experts if e == cfg.shared_expert else moe.experts[e]
        target.w1.weight.copy_(ug[: cfg.inter].to(bf16))
        target.w3.weight.copy_(ug[cfg.inter :].to(bf16))
        target.w2.weight.copy_(dn.to(bf16))


@torch.no_grad()
def _oracle_step(om, attn, moe, h, pos, g_in, g_post, eps):
    """One layer of the oracle, with a plain residual (no hyper-connections)."""
    # the oracle builds index tensors with a bare torch.arange
    with torch.device(h.device):
        x = bf(rmsnorm(h, g_in, eps)).to(torch.bfloat16)
        o = attn(x.unsqueeze(0), pos).squeeze(0)
        a = (h.float() + o.float()).to(torch.bfloat16)
        x2 = bf(rmsnorm(a, g_post, eps)).to(torch.bfloat16)
        ids = torch.zeros(1, x2.shape[0], dtype=torch.long, device=h.device)
        y = moe(x2.unsqueeze(0), ids).squeeze(0)
    return a, (a.float() + y.float()).to(torch.bfloat16)


@pytest.mark.parametrize("steps", [6])
def test_v4_layer_matches_deepseek_reference(steps):
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)  # this case drives attn/ffn directly, around mHC

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    attn, moe = _oracle_modules(om, cfg, device)
    _load_oracle_weights(attn, moe, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(256, theta=cfg.rope_theta, device=device)
    kv_cache = torch.zeros(cfg.window, cfg.head_dim, dtype=torch.bfloat16, device=device)

    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(W, h, [pos], kv_cache, dest, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
        a_ref, out_ref = _oracle_step(om, attn, moe, h, pos, W.t["g_in"], W.t["g_post"], cfg.eps)

        da = (res["a"].float() - a_ref.float()).abs().max().item()
        do = (res["x_out"].float() - out_ref.float()).abs().max().item()
        scale = out_ref.float().abs().max().item()
        assert da < 3e-2 * max(scale, 1.0), f"pos={pos} attention half diverges: {da}"
        assert do < 5e-2 * max(scale, 1.0), f"pos={pos} layer output diverges: {do}"


# DeepSeek's whole Block: also covers hc_pre / hc_post / Sinkhorn and the [S, hc, d] residual.


def _oracle_args(om, cfg):
    return om.ModelArgs(
        max_batch_size=1,
        max_seq_len=256,
        dtype="bf16",
        scale_fmt="ue8m0",
        scale_dtype="fp32",
        vocab_size=32,
        dim=cfg.hidden,
        moe_inter_dim=cfg.inter,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=cfg.heads,
        n_routed_experts=cfg.n_experts,
        n_shared_experts=1,
        n_activated_experts=cfg.top_k,
        score_func="sqrtsoftplus",
        route_scale=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_dim,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora,
        window_size=cfg.window,
        compress_ratios=(cfg.compress_ratio,),
        norm_eps=cfg.eps,
        rope_theta=cfg.rope_theta,
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=0,  # YaRN off: it is a host-side rope table, not kernel work
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
    )


@torch.no_grad()
def _load_block_weights(block, W, cfg, weight_fmt):
    t = W.t
    _load_oracle_weights(block.attn, block.ffn, W, cfg, weight_fmt)
    if cfg.compress_ratio:
        # the compressors' projections live in our fused qkv_a; split them back out
        dq = qkv_a_matrix(t)
        cut = qkv_a_split(cfg)
        c = block.attn.compressor
        c.wkv.weight.copy_(dq[slice(*cut["c_kv"])].float())
        c.wgate.weight.copy_(dq[slice(*cut["c_gate"])].float())
        c.ape.copy_(t["ape"].float())
        c.norm.weight.copy_(t["g_ckv"].float())
        if cfg.indexed:
            ix = block.attn.indexer
            ix.compressor.wkv.weight.copy_(dq[slice(*cut["i_kv"])].float())
            ix.compressor.wgate.weight.copy_(dq[slice(*cut["i_gate"])].float())
            ix.compressor.ape.copy_(t["i_ape"].float())
            ix.compressor.norm.weight.copy_(t["g_ickv"].float())
            ix.wq_b.weight.copy_(dequant(t["w_i_q_b"], t["s_i_q_b"], 128))
            ix.weights_proj.weight.copy_(t["i_w"])
            # the model builds it under a bf16 default dtype; these tests default to fp32
            ix.kv_cache = ix.kv_cache.to(torch.bfloat16)
            ix.compressor.kv_cache = None  # re-bound from ix.kv_cache on first use
    block.attn_norm.weight.copy_(t["g_in"].float())
    block.ffn_norm.weight.copy_(t["g_post"].float())
    for side in ("attn", "ffn"):
        # ours is row-padded to the MFMA group; the oracle's is exactly hc_mix
        getattr(block, f"hc_{side}_fn").copy_(t[f"hc_{side}_fn"][: cfg.hc_mix].float())
        getattr(block, f"hc_{side}_base").copy_(t[f"hc_{side}_base"].float())
        getattr(block, f"hc_{side}_scale").copy_(t[f"hc_{side}_scale"].float())


@pytest.mark.parametrize("steps", [5])
def test_v4_block_with_hyper_connections_matches_deepseek(steps):
    """The golden's full layer, hyper-connections included, against DeepSeek's Block."""
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg()
    assert cfg.hc_mult > 1, "this case is about the hyper-connection path"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    with torch.device(device):
        block = om.Block(0, _oracle_args(om, cfg))
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(256, theta=cfg.rope_theta, device=device)
    kv_cache = torch.zeros(cfg.window, cfg.head_dim, dtype=torch.bfloat16, device=device)

    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(W, h, [pos], kv_cache, dest, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)

        assert res["x_out"].shape == h.shape, f"layer must preserve [S, hc, hidden], got {res['x_out'].shape}"
        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} block output diverges: {d} (|ref| {scale})"


@pytest.mark.parametrize("steps", [40])
def test_v4_hca_compressor_matches_deepseek(steps):
    """The HCA layer (compressed attention) against DeepSeek's Block, across several compression boundaries."""
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    ratio = 16  # stands in for V4's 128; same code path, far fewer steps to cross
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    cfg.compress_ratio = ratio
    cfg.validate()

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    with torch.device(device):
        block = om.Block(0, _oracle_args(om, cfg))
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    kv_cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(1, ratio, cfg.head_dim, device=device)
    score_state = torch.zeros(1, ratio, cfg.head_dim, device=device)

    compressed_seen = 0
    near_ties = 0
    # the oracle's expert set: a near-tie on score + bias can tip either way
    cap = {}
    block.ffn.gate.register_forward_hook(lambda m, i, o: cap.__setitem__("gate", o))
    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(
            W,
            h,
            [pos],
            kv_cache,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            kv_state=kv_state,
            score_state=score_state,
            cos_c=cos,
            sin_c=sin,
        )
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)
        if (pos + 1) % ratio == 0:
            compressed_seen += 1

        # scores agree to ~0.017 abs, so a top-k margin under NEAR_TIE may route differently: skip
        sc = res["scores"][0].float() + W.t["bias"].float()
        top = torch.topk(sc, cfg.top_k + 1).values
        o_experts = set(cap["gate"][1].reshape(-1).tolist())
        g_experts = set(res["sel"].reshape(-1).tolist()) - {cfg.n_experts}  # less the shared slot
        if (top[cfg.top_k - 1] - top[cfg.top_k]).item() < NEAR_TIE or o_experts != g_experts:
            near_ties += 1
            continue

        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} diverges: {d} (|ref| {scale})"

    assert compressed_seen >= 2, "the run must cross at least two compression boundaries"
    assert near_ties < steps // 4, f"too many near-ties to have tested much: {near_ties}/{steps}"


@pytest.mark.parametrize("ratio", [4, 8])
def test_v4_compressor_matches_deepseek_directly(ratio):
    """The compressor alone vs DeepSeek's Compressor; ratio 4 is CSA's overlapping (2*ratio-token) form."""
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    cfg.compress_ratio = ratio
    assert cfg.overlap == (ratio == 4), "overlap is tied to ratio 4"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=11, moe_mode=MoeMode.W8A16)
    t = W.t
    args = _oracle_args(om, cfg)
    with torch.device(device):
        comp = om.Compressor(args, ratio, cfg.head_dim)
        comp.kv_cache = torch.zeros(1, cfg.n_compressed, cfg.head_dim, device=device)
        comp.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.compress_rope_theta, 1.0, 32, 1)
    dq = qkv_a_matrix(t)
    coff = cfg.c_coff
    # by name: at ratio 4 the fused GEMV also carries the indexer's compressor
    cut = qkv_a_split(cfg)
    with torch.no_grad():
        comp.wkv.weight.copy_(dq[slice(*cut["c_kv"])].float())
        comp.wgate.weight.copy_(dq[slice(*cut["c_gate"])].float())
        comp.ape.copy_(t["ape"].float())
        comp.norm.weight.copy_(t["g_ckv"].float())

    cos, sin = rope_table(512, theta=cfg.compress_rope_theta, device=device)
    cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(cfg.c_rows, coff * cfg.head_dim, device=device)
    # -inf: overlap rows unwritten before the first emit must drop out of the softmax
    score_state = torch.full((cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device)

    emitted = 0
    for pos in range(6 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        proj = x.float() @ dq.float().T
        ours = compress_step(
            proj[0, slice(*cut["c_kv"])],
            proj[0, slice(*cut["c_gate"])],
            pos,
            cfg,
            t,
            kv_state,
            score_state,
            cache,
            cos,
            sin,
            dest_row=cfg.window + pos // ratio,
        )
        with torch.device(device):
            theirs = comp(x.unsqueeze(0), pos)
        if (pos + 1) % ratio:
            assert ours is None and theirs is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        d = (ours.float() - theirs.reshape(-1).float()).abs().max().item()
        scale = max(theirs.float().abs().max().item(), 1e-6)
        assert d < 2e-2 * scale, f"ratio={ratio} pos={pos} compressed entry differs: {d / scale:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_v4_indexer_compressor_matches_deepseek():
    """The indexer's compressor (Hadamard rotation + FP4) vs DeepSeek's `Compressor(..., rotate=True)`."""
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    ihd, coff = 128, cfg.c_coff  # index_head_dim; V4 keeps rope_head_dim at 64
    assert cfg.overlap, "ratio 4 is the overlapping form"

    args = _oracle_args(om, cfg)
    with torch.device(device):
        comp = om.Compressor(args, ratio, ihd, True)
        n_comp = cfg.max_seq // ratio
        comp.kv_cache = torch.zeros(1, n_comp, ihd, device=device)
        comp.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.compress_rope_theta, 1.0, 32, 1)

    gen = torch.Generator(device=device).manual_seed(5)
    wkv = torch.randn(coff * ihd, cfg.hidden, generator=gen, device=device) / cfg.hidden**0.5
    wgate = torch.randn(coff * ihd, cfg.hidden, generator=gen, device=device) / cfg.hidden**0.5
    ape = 0.5 * torch.randn(ratio, coff * ihd, generator=gen, device=device)
    gamma = (1 + 0.1 * torch.randn(ihd, generator=gen, device=device)).to(torch.bfloat16)
    with torch.no_grad():
        comp.wkv.weight.copy_(wkv.float())
        comp.wgate.weight.copy_(wgate.float())
        comp.ape.copy_(ape.float())
        comp.norm.weight.copy_(gamma.float())

    cos, sin = rope_table(512, theta=cfg.compress_rope_theta, device=device)
    cache = torch.zeros(n_comp, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    kv_state = torch.zeros(cfg.c_rows, coff * ihd, device=device)
    score_state = torch.full((cfg.c_rows, coff * ihd), float("-inf"), device=device)

    emitted = 0
    for pos in range(6 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        ours = compress_step(
            (x.float() @ wkv.float().T)[0],
            (x.float() @ wgate.float().T)[0],
            pos,
            cfg,
            None,
            kv_state,
            score_state,
            cache,
            cos,
            sin,
            head_dim=ihd,
            ape=ape,
            gamma=gamma,
            rotate=True,
        )
        with torch.device(device):
            theirs = comp(x.unsqueeze(0), pos)
        if (pos + 1) % ratio:
            assert ours is None and theirs is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        d = (ours.float() - theirs.reshape(-1).float()).abs().max().item()
        scale = max(theirs.float().abs().max().item(), 1e-6)
        assert d < 2e-2 * scale, f"pos={pos} indexer compressed entry differs: {d / scale:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_mxfp8_ceil_scale_never_clips():
    """Ceil-rounded E8M0 scales never clip a block max (nearest rounding does); error <= one E4M3 half-ulp."""
    from kernels.monokernel.dsv4.reference import quant_dequant_mxfp8
    from kernels.monokernel.formats import quant_dequant_mxfp8 as nearest_mxfp8

    torch.manual_seed(0)
    x = torch.randn(64, 1024) * torch.exp2(torch.randint(-8, 8, (64, 1)).float())
    x[:, ::32] *= 20.0  # an outlier per block, as real activations have
    q = quant_dequant_mxfp8(x)
    blocks = x.reshape(64, -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    err = (q.reshape(64, -1, 32) - blocks).abs()
    assert (err <= 2**-4 * blocks.abs() + amax * 2**-17).all(), "ceil-scaled MXFP8 clipped or over-rounded"
    assert (q.abs().reshape(64, -1, 32).amax(-1, keepdim=True) >= amax * (1 - 2**-4)).all(), "a block max clipped"
    # the nearest-rounded scale does clip on this input
    assert (nearest_mxfp8(x) - x).norm() > 2 * (q - x).norm()


def test_v4_indexer_cache_packs_fp4_losslessly():
    """The FP4 indexer cache round-trips exactly to ``quant_dequant_fp4`` and has the byte layout the kernel decodes."""
    torch.manual_seed(0)
    n = 128
    x = torch.randn(64, n) * torch.logspace(-20, 12, 64, base=2.0)[:, None]
    x[5] = 0  # a zero row: the clamp's smallest scale
    x[6, 32:64] = 0  # one zero block among live ones
    p = pack_fp4(x)
    assert p.dtype == torch.uint8 and p.shape == (64, fp4_row_bytes(n)) == (64, 68)
    assert torch.equal(unpack_fp4(p), quant_dequant_fp4(x))

    # element i in nibble i % 2 of byte i // 2, sign in bit 3; one e8m0 per 32
    y = torch.zeros(n)
    y[0], y[1], y[2], y[33] = 6.0, -0.5, -6.0, 3.0  # block 0 scale 1, block 1 scale 0.5
    q = pack_fp4(y)
    assert q[0].item() == 0x7 | (0x9 << 4), f"byte 0 {q[0].item():#x}"
    assert q[1].item() == 0xF, f"byte 1 {q[1].item():#x}"
    assert q[16].item() == 0x7 << 4, f"byte 16 {q[16].item():#x}"
    assert q[64:].tolist() == [127, 126, 1, 1], q[64:].tolist()


def test_v4_indexer_matches_deepseek():
    """The lightning indexer vs DeepSeek's Indexer: the same selected set (single rank, allreduce is identity)."""
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    # small k so the top-k discards; V4-Pro's 1024 would select every entry here
    cfg.index_topk = 4
    assert cfg.indexed, "ratio 4 is the indexed form"
    ih, ihd = cfg.index_heads, cfg.index_head_dim

    W = make_weights(rank=0, cfg=cfg, device=device, seed=13, moe_mode=MoeMode.W8A16)
    t = W.t
    args = _oracle_args(om, cfg)
    args.index_n_heads, args.index_head_dim, args.index_topk = ih, ihd, cfg.index_topk
    with torch.device(device):
        idxr = om.Indexer(args, ratio)
        # bf16, as under the model's default dtype
        idxr.kv_cache = torch.zeros(1, cfg.n_compressed, ihd, dtype=torch.bfloat16, device=device)
        idxr.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.rope_base, 1.0, 32, 1)

    dq = qkv_a_matrix(t)
    cut = qkv_a_split(cfg)
    with torch.no_grad():
        idxr.compressor.wkv.weight.copy_(dq[slice(*cut["i_kv"])].float())
        idxr.compressor.wgate.weight.copy_(dq[slice(*cut["i_gate"])].float())
        idxr.compressor.ape.copy_(t["i_ape"].float())
        idxr.compressor.norm.weight.copy_(t["g_ickv"].float())
        idxr.wq_b.weight.copy_(dequant(t["w_i_q_b"], t["s_i_q_b"], 128))
        idxr.weights_proj.weight.copy_(t["i_w"])

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    i_cache = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    i_state = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=device)
    i_score = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=device)

    checked = discriminated = 0
    for pos in range(8 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        # q_norm returns bf16 in the model, and both wq_b's consume it as such
        q_a_n = rmsnorm(0.5 * torch.randn(1, cfg.q_lora, device=device), t["g_q"], cfg.eps).to(torch.bfloat16)
        proj = x.float() @ dq.float().T
        ours = indexer_step(
            x[0],
            q_a_n[0],
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_state,
            i_score,
            i_cache,
            cos,
            sin,
        )
        with torch.device(device):
            # offset 0: indexer_step returns entry indices; the plane offset is the caller's
            theirs = idxr(x.unsqueeze(0), q_a_n.unsqueeze(0), pos, 0)
        n = (pos + 1) // ratio
        if not n:
            assert int((ours >= 0).sum()) == 0, f"pos={pos}: nothing compressed yet"
            continue
        checked += 1
        a = set(ours[ours >= 0].tolist())
        b = set(theirs.reshape(-1).tolist())
        assert a == b, f"pos={pos} selected {sorted(a)} vs {sorted(b)}"
        if n > cfg.index_topk:
            discriminated += 1
            assert len(a) == cfg.index_topk, f"pos={pos} picked {len(a)}, want {cfg.index_topk}"

    assert checked >= 6, f"expected several scored steps, got {checked}"
    # otherwise every step selected everything and the scores were never tested
    assert discriminated >= 3, f"top-k never had to discard anything ({discriminated})"


@pytest.mark.parametrize("steps", [40])
def test_v4_csa_layer_matches_deepseek(steps):
    """A whole CSA layer (overlapping compression + indexer) vs DeepSeek's Block, with a small index_topk."""
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    cfg.index_topk = 4
    cfg.validate()
    assert cfg.indexed and cfg.overlap, "ratio 4 is the indexed, overlapping form"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    args = _oracle_args(om, cfg)
    args.index_n_heads = cfg.index_heads
    args.index_head_dim = cfg.index_head_dim
    args.index_topk = cfg.index_topk
    with torch.device(device):
        block = om.Block(0, args)
    assert block.attn.indexer is not None, "the oracle must have built an indexer"
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    ihd, coff = cfg.index_head_dim, cfg.c_coff
    kv_cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(1, cfg.c_rows, coff * cfg.head_dim, device=device)
    score_state = torch.full((1, cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device)
    i_cache = torch.zeros(1, cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    i_state = torch.zeros(1, cfg.c_rows, coff * ihd, device=device)
    i_score = torch.full((1, cfg.c_rows, coff * ihd), float("-inf"), device=device)

    # a near-tie in expert routing or the indexer top-k can flip a pick and move the output
    # by tens of percent; such a step is a tie, not a mismatch
    cap = {}
    block.ffn.gate.register_forward_hook(lambda m, i, o: cap.__setitem__("gate", o))
    block.attn.indexer.register_forward_hook(lambda m, i, o: cap.__setitem__("picks", o))
    selected, near_ties = 0, 0
    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        cap.clear()
        res = golden_layer(
            W,
            h,
            [pos],
            kv_cache,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            kv_state=kv_state,
            score_state=score_state,
            cos_c=cos,
            sin_c=sin,
            i_state=i_state,
            i_score_state=i_score,
            i_cache=i_cache,
        )
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)
        if (pos + 1) // ratio > cfg.index_topk:
            selected += 1

        top = res["scores"].reshape(-1).sort(descending=True).values
        o_experts = set(cap["gate"][1].reshape(-1).tolist())
        g_experts = set(res["sel"].reshape(-1).tolist()) - {cfg.n_experts}  # less the shared slot
        o_picks = (
            {int(x) - cfg.window for x in cap["picks"].reshape(-1).tolist() if x >= 0} if "picks" in cap else set()
        )
        g_picks = {int(x) for x in res["picks"].reshape(-1).tolist() if x >= 0}
        if (top[cfg.top_k - 1] - top[cfg.top_k]).item() < NEAR_TIE or o_experts != g_experts or o_picks != g_picks:
            near_ties += 1
            continue
        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} diverges: {d} (|ref| {scale})"

    assert selected >= 5, f"the indexer never had to discard anything ({selected})"
    assert near_ties < steps // 4, f"too many near-ties to have tested much: {near_ties}/{steps}"


@pytest.mark.parametrize("ratio", [0, 4])
def test_v4_golden_batches_independent_sequences(ratio):
    """Batched sequences equal their own S=1 runs (catches a stage reading sample 0's state for all)."""
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    if ratio:
        cfg.index_topk = 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    ihd, coff, S = cfg.index_head_dim, cfg.c_coff, 2

    def state(n):
        # ONE plane; sample s owns rows [s * cache_rows, (s+1) * cache_rows)
        d = dict(kv_cache=torch.zeros(n * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device))
        if not ratio:
            return d
        d |= dict(
            kv_state=torch.zeros(n, cfg.c_rows, coff * cfg.head_dim, device=device),
            score_state=torch.full((n, cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device),
            cos_c=cos,
            sin_c=sin,
        )
        if cfg.indexed:
            d |= dict(
                i_state=torch.zeros(n, cfg.c_rows, coff * ihd, device=device),
                i_score_state=torch.full((n, cfg.c_rows, coff * ihd), float("-inf"), device=device),
                i_cache=torch.zeros(n, cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device),
            )
        return d

    def run(st, h, pos):
        n = h.shape[0]
        kw = dict(st)
        idx, dest = contiguous_pool([pos] * n, cfg, device)
        return golden_layer(
            W,
            h,
            [pos] * n,
            kw.pop("kv_cache"),
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            **kw,
        )

    batched, alone = state(S), [state(1) for _ in range(S)]
    steps = 4 * cfg.window // 3
    for pos in range(steps):
        h = (0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        got = run(batched, h, pos)["x_out"]
        for s in range(S):
            want = run(alone[s], h[s : s + 1], pos)["x_out"]
            # norm-relative: batch 2 and batch 1 may take GEMMs that round a few bf16 ulps apart;
            # reading another sample's state is a ~0.65 difference
            rel = ((got[s].float() - want[0].float()).norm() / want[0].float().norm()).item()
            assert rel < 2e-2, f"pos={pos} sample {s}: batched differs from its own run (rel {rel:.3e})"
    # the run has to be deep enough that the compressor and its state actually ran
    if ratio:
        assert steps > 2 * ratio, "too shallow to have crossed a compression boundary"
