#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness tests for the gfx11 (RDNA3) bf16 MoE in ``kernels/moe/rdna3_moe.py``.

The routing kernel is checked against a python loop, each GEMM stage against torch per routed slot (with
sentinel rows past the output to catch a padded row that stores), the bounded grid against the exact one,
the direct, token_down and one-wave g2r paths against the routed ones, and the layer against
``torch_moe_gemm1/2`` at four TP=8 model shapes with hidden cut to 256. Inputs are unit scale and outputs are
compared by cosine and max error, so an all-zero output fails. The tile tests check that the tile height grows
with the token count and that every shape entry of the tile table is what its shape builds.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (os.path.join(_REPO_ROOT, "build-fly", "python_packages"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

_ARCH = str(get_rocm_arch())
if _ARCH.split(":")[0] != "gfx1100":
    pytest.skip(f"the RDNA3 MoE is tuned for gfx1100 only, got {_ARCH}", allow_module_level=True)

import kernels.moe.rdna3_moe as moe  # noqa: E402
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2  # noqa: E402

TILE_HEIGHTS = (16, 32, 64, 128)
# (experts, topk, hidden, inter) of four TP=8 shards, a fused shared expert counted in experts and topk.
SHAPES = {
    "qwen38-flash-next-tp8": (513, 11, 2560, 80),
    "qwen3-30b-a3b-tp8": (128, 8, 2048, 96),
    "glm45-air-tp8": (129, 9, 4096, 176),
    "mixtral-8x7b-tp8": (8, 2, 4096, 1792),
}
SMALL_HIDDEN = 256
TOKENS = (1, 8, 16, 32, 64, 128, 512, 760, 2048, 8192)
_ROUTING_CASES = [(1, 2, 2), (7, 8, 3), (64, 16, 4), (600, 8, 4), (760, 513, 11), (8192, 8, 2), (2048, 129, 9)]
# The first generic tile of each height, (reg_m, reg_n, reg_k, waves_m, waves_n).
_FIRST_TILE = {
    "gate_up": {16: (1, 2, 4, 1, 2), 32: (2, 1, 4, 1, 2), 64: (2, 1, 4, 2, 2), 128: (4, 1, 2, 2, 2)},
    "down": {16: (1, 2, 2, 1, 2), 32: (2, 2, 2, 1, 4), 64: (2, 2, 2, 2, 4), 128: (2, 2, 2, 4, 2)},
}
_NO_WEIGHTS = torch.empty(0, dtype=torch.float32, device="cuda")


def _stream():
    return torch.cuda.current_stream()


def _rand(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16).contiguous()


def _close(got, ref, rel=0.02):
    got, ref = got.float().flatten(), ref.float().flatten()
    assert torch.nn.functional.cosine_similarity(got, ref, dim=0).item() >= 0.999
    max_abs = (got - ref).abs().max().item()
    assert max_abs <= rel * ref.abs().max().item(), f"max_abs {max_abs} against max |ref| {ref.abs().max().item()}"


def _route(tokens, experts, topk, seed=0):
    """Distinct experts per token, and their f32 weights."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights, ids = torch.topk(torch.rand(tokens, experts, generator=g, device="cuda"), k=topk, dim=1)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


def _host_routing(topk_ids, experts, tile_m, tokens):
    """The routing by a python loop: rows grouped by expert, each expert padded up to ``tile_m``, then the rows of
    out-of-range experts as expert ``experts``."""
    topk = topk_ids.shape[1]
    sentinel = (topk << 24) | tokens
    sorted_ids, expert_ids = [], []
    for e in range(experts + 1):
        hit = topk_ids == e if e < experts else (topk_ids < 0) | (topk_ids >= experts)
        token, slot = torch.where(hit)
        rows = ((slot.to(torch.int32) << 24) | token.to(torch.int32)).tolist()
        blocks = -(-len(rows) // tile_m)
        sorted_ids += rows + [sentinel] * (blocks * tile_m - len(rows))
        expert_ids += [e] * blocks
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        len(expert_ids),
    )


def _canonical(sorted_ids, expert_ids, tile_m):
    """``sorted_ids`` sorted inside each expert: the kernel hands out slots with an atomic, so only that is fixed."""
    labels = expert_ids.repeat_interleave(tile_m)
    out = sorted_ids.clone()
    for e in torch.unique(labels).tolist():
        mask = labels == e
        out[mask] = torch.sort(sorted_ids[mask]).values
    return out


# ── Routing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tile_m", [16, 64])
@pytest.mark.parametrize("tokens, experts, topk", _ROUTING_CASES)
def test_routing_matches_host_reference(tokens, experts, topk, tile_m):
    """513 experts take three scan chunks with a carry; 7 tokens leave a ragged last row walk; the last two take
    the wide workgroup. The second seed routes into the buffers the first one left."""
    for seed in (0, 1):
        ids, _ = _route(tokens, experts, topk, seed=seed)
        r = moe.build_routing_device(ids, experts=experts, tile_m=tile_m, exact=True)
        sorted_ids, expert_ids, num_blocks = _host_routing(ids, experts, tile_m, tokens)
        assert r.exact and r.num_blocks == num_blocks
        assert num_blocks <= moe.max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
        assert torch.equal(r.expert_ids, expert_ids)
        assert torch.equal(_canonical(r.sorted_ids, r.expert_ids, tile_m), _canonical(sorted_ids, expert_ids, tile_m))


@pytest.mark.parametrize("tokens, experts, topk", [(7, 8, 3), (2048, 129, 9)])
def test_routing_puts_out_of_range_ids_last(tokens, experts, topk):
    ids, _ = _route(tokens, experts, topk)
    ids[:, 0] = experts
    ids[::3, 1] = -1
    r = moe.build_routing_device(ids, experts=experts, tile_m=16, exact=True, reuse=False)
    sorted_ids, expert_ids, num_blocks = _host_routing(ids, experts, 16, tokens)
    assert r.num_blocks == num_blocks and int(r.expert_ids[-1]) == experts
    assert num_blocks <= moe.max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=16)
    assert torch.equal(r.expert_ids, expert_ids)
    assert torch.equal(_canonical(r.sorted_ids, r.expert_ids, 16), _canonical(sorted_ids, expert_ids, 16))


def test_routing_workspace_is_one_per_tile_height():
    """Every prefill length shares one power-of-two workspace per tile height, grown only when outgrown."""
    experts, topk, tile_m = 8, 2, 64
    moe._WORKSPACES.clear()
    for tokens in (300, 100, 700, 650):
        ids, _ = _route(tokens, experts, topk)
        r = moe.build_routing_device(ids, experts=experts, tile_m=tile_m)
        assert r.sorted_ids.numel() == r.num_blocks * tile_m
    keys = [k for k in moe._WORKSPACES if k[0] == tile_m]
    assert len(keys) == 1
    bound = moe.max_blocks_for(tokens=700, topk=topk, experts=experts, tile_m=tile_m)
    assert moe._WORKSPACES[keys[0]].expert_ids.numel() == 1 << (bound - 1).bit_length()


def test_routing_block_for():
    assert moe._routing_block_for(moe._ROWS_FOR_WIDE_ROUTING - 1) == 256 == moe._routing_block_for(128 * 11)
    assert moe._routing_block_for(moe._ROWS_FOR_WIDE_ROUTING) != 256


def test_routing_bound_mode_keeps_count_on_device():
    tokens, experts, topk, tile_m = 32, 8, 2, 16
    ids, _ = _route(tokens, experts, topk)
    exact = moe.build_routing_device(ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    bound = moe.build_routing_device(ids, experts=experts, tile_m=tile_m)
    torch.cuda.synchronize()
    assert bound.exact is False
    assert bound.num_blocks == moe.max_blocks_for(tokens=tokens, topk=topk, experts=experts, tile_m=tile_m)
    assert int(bound.num_blocks_device[0]) == exact.num_blocks
    assert torch.equal(bound.expert_ids[: exact.num_blocks], exact.expert_ids)


def test_unrouted_expert_gets_no_tile():
    ids = torch.zeros(16, 1, dtype=torch.int32, device="cuda")
    r = moe.build_routing_device(ids, experts=8, tile_m=16, exact=True, reuse=False)
    assert r.num_blocks == 1 and r.expert_ids.tolist() == [0]
    assert torch.equal(torch.sort(r.sorted_ids).values, torch.arange(16, dtype=torch.int32, device="cuda"))


# ── The two GEMM stages ──────────────────────────────────────────────────────


def _gemm(stage, tile_m, *, k_dim, n_out, experts, topk, cfg=None, **opts):
    reg_m, reg_n, reg_k, waves_m, waves_n = cfg or _FIRST_TILE[stage][tile_m]
    launch, block_m, _, _ = moe.create_grouped_gemm_module(
        k_dim=k_dim,
        n_out=n_out,
        experts=experts,
        stage=stage,
        topk=topk,
        doweight=stage == "down",
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        **opts,
    )
    assert block_m == tile_m
    return launch


def _stage(stage, tile_m, *, tokens, k_dim, n_out, experts, topk, cfg=None, **opts):
    """One stage on the host routing, into an output with 64 sentinel rows of 7.0 behind it."""
    launch = _gemm(stage, tile_m, k_dim=k_dim, n_out=n_out, experts=experts, topk=topk, cfg=cfg, **opts)
    ids, weights = _route(tokens, experts, topk)
    a = _rand(tokens, k_dim, seed=1) if stage == "gate_up" else _rand(tokens, topk, k_dim, seed=1)
    w = _rand(experts, (2 if stage == "gate_up" else 1) * n_out, k_dim, seed=2)
    sorted_ids, expert_ids, num_blocks = _host_routing(ids, experts, tile_m, tokens)
    backing = torch.full((tokens * topk + 64, n_out), 7.0, dtype=torch.bfloat16, device="cuda")
    out = backing[: tokens * topk].view(tokens, topk, n_out)
    w_in = moe.wmma_order(w) if stage == "gate_up" else w
    launch(out, a, w_in, sorted_ids, expert_ids, weights, tokens, num_blocks, _stream())
    torch.cuda.synchronize()
    return out, backing[tokens * topk :], (a, w, ids, weights)


@pytest.mark.parametrize("tile_m", TILE_HEIGHTS)
@pytest.mark.parametrize("opts", [{}, {"g2r": True}])
def test_gate_up_matches_torch(tile_m, opts):
    cfg = (tile_m // 16, 1, 4, 1, 1) if opts.get("g2r") else None
    kw = dict(tokens=128, k_dim=256, n_out=128, experts=4, topk=2, cfg=cfg, **opts)
    out, extra, (x, w1, ids, weights) = _stage("gate_up", tile_m, **kw)
    ref = torch_moe_gemm1(x, w1, None, None, ids, weights, 128, False)
    _close(out.reshape(ref.shape), ref)
    assert torch.equal(extra, torch.full_like(extra, 7.0)), "a padded routing row stored past the output"


@pytest.mark.parametrize("tile_m", TILE_HEIGHTS)
def test_down_matches_torch_per_slot(tile_m):
    out, extra, (a2, w2, ids, weights) = _stage("down", tile_m, tokens=128, k_dim=256, n_out=256, experts=4, topk=2)
    ref = torch.einsum("tsk,tsnk->tsn", a2.float(), w2[ids.long()].float()) * weights[..., None]
    _close(out, ref)
    assert torch.equal(extra, torch.full_like(extra, 7.0)), "a padded routing row stored past the output"


@pytest.mark.parametrize("tile_m", [16, 64])
def test_bounded_grid_matches_exact_grid(tile_m):
    tokens, hidden, inter, experts, topk = 64, 256, 128, 8, 2
    ids, _ = _route(tokens, experts, topk)
    x, w1 = _rand(tokens, hidden, seed=1), _rand(experts, 2 * inter, hidden, seed=2)
    exact = moe.build_routing_device(ids, experts=experts, tile_m=tile_m, exact=True, reuse=False)
    bound = moe.build_routing_device(ids, experts=experts, tile_m=tile_m, reuse=False)
    assert bound.num_blocks > exact.num_blocks
    outs = []
    for r in (exact, bound):
        launch = _gemm(
            "gate_up", tile_m, k_dim=hidden, n_out=inter, experts=experts, topk=topk, bounded_blocks=not r.exact
        )
        out = torch.zeros(tokens, topk, inter, dtype=torch.bfloat16, device="cuda")
        nb_dev = None if r.exact else r.num_blocks_device
        launch(out, x, w1, r.sorted_ids, r.expert_ids, _NO_WEIGHTS, tokens, r.num_blocks, _stream(), nb_dev)
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("tokens", [37, 760])
def test_g2r_tile_matches_the_lds_tile(tokens):
    """A one-wave tile loading straight to registers, over a K tail of 16 (inter 176), is the LDS tile's answer."""
    kw = dict(tokens=tokens, k_dim=176, n_out=256, experts=129, topk=9, cfg=(1, 2, 2, 1, 1))
    want, _, _ = _stage("down", 16, **kw)
    got, _, _ = _stage("down", 16, **kw, g2r=True)
    assert torch.equal(got, want)


def test_direct_gate_up_matches_the_routed_tile():
    """At one token each slot is its own tile, so ``direct`` reads tiles off ``topk_ids`` with no routing."""
    experts, topk, hidden, inter = 16, 4, 256, 128
    ids, _ = _route(1, experts, topk)
    x, w1 = _rand(1, hidden, seed=1), _rand(experts, 2 * inter, hidden, seed=2)
    kw = dict(model_dim=hidden, inter_dim=inter, experts=experts, topk=topk, tile_m=16, rows_per_expert=topk / experts)
    routed, *_ = moe.compile_moe_gemm1(**kw)
    direct, *_ = moe.compile_moe_gemm1(**kw, direct=True)
    assert direct.direct and not routed.direct
    r = moe.build_routing_device(ids, experts=experts, tile_m=16, exact=True, reuse=False)
    want = torch.empty(1, topk, inter, dtype=torch.bfloat16, device="cuda")
    got = torch.full_like(want, float("nan"))
    routed(want, x, w1, r.sorted_ids, r.expert_ids, _NO_WEIGHTS, 1, r.num_blocks, _stream())
    direct(got, x, w1, ids, ids, _NO_WEIGHTS, 1, topk, _stream())
    torch.cuda.synchronize()
    assert torch.equal(got, want)


@pytest.mark.parametrize("tokens", [1, 3])
def test_token_down_sums_the_weighted_slots(tokens):
    """inter 48 is six 128-bit chunks over 16 lanes, so some lanes idle in every step."""
    experts, topk, hidden, inter = 4, 3, 256, 48
    ids, weights = _route(tokens, experts, topk)
    a2, w2 = _rand(tokens, topk, inter, seed=1), _rand(experts, hidden, inter, seed=2)
    ref = torch.einsum("tsk,tsnk->tn", a2.float(), w2[ids.long()].float() * weights[..., None, None])
    down = moe.compile_token_down(model_dim=hidden, inter_dim=inter, experts=experts, topk=topk)
    got = torch.full((tokens, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    down(got, a2, w2, ids, weights, tokens, _stream())
    torch.cuda.synchronize()
    _close(got, ref)


# ── The layer ────────────────────────────────────────────────────────────────


def _layer_inputs(experts, topk, hidden, inter, tokens):
    """``w1`` comes back in ``wmma_order``, the layout the layer takes."""
    ids, weights = _route(tokens, experts, topk)
    x, w1, w2 = (
        _rand(tokens, hidden, seed=1),
        moe.wmma_order(_rand(experts, 2 * inter, hidden, seed=2)),
        _rand(experts, hidden, inter, seed=3),
    )
    return x, w1, w2, ids, weights


def _torch_layer(x, w1, w2, ids, weights):
    e, n, k = w1.shape
    w1 = w1.view(e, n // 16, k // 8, 16, 8).transpose(2, 3).reshape(e, n, k)
    a2 = torch_moe_gemm1(x, w1, None, None, ids, weights, w2.shape[2], False)
    return torch_moe_gemm2(a2.to(x.dtype), w2, None, None, ids, weights, w2.shape[1], doweight_stage2=True)


@pytest.mark.parametrize("tokens", [8, 2048])
@pytest.mark.parametrize("shape_id", list(SHAPES))
def test_layer_matches_torch(shape_id, tokens):
    """Heights from the real hidden size, so the eight points cover 16/32/64/128 and the direct path."""
    experts, topk, hidden, inter = SHAPES[shape_id]
    tile_m = moe.tile_m_for(tokens=tokens, topk=topk, experts=experts, inter=inter, hidden=hidden)
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, tokens)
    got = moe.moe_forward(x, w1, w2, ids, weights, tile_m=tile_m)
    torch.cuda.synchronize()
    _close(got, _torch_layer(x, w1, w2, ids, weights))


def test_layer_default_tile_is_the_rule():
    experts, topk, _, inter = SHAPES["mixtral-8x7b-tp8"]
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, 760)
    tile_m = moe.tile_m_for(tokens=760, topk=topk, experts=experts, inter=inter, hidden=SMALL_HIDDEN)
    want = moe.moe_forward(x, w1, w2, ids, weights, tile_m=tile_m).clone()
    got = moe.moe_forward(x, w1, w2, ids, weights, tile_m=None)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


def test_layer_takes_int64_ids_and_rejects_bad_inputs():
    """``torch.topk`` gives int64 ids, and 4 rows an expert takes the routed path, which must convert them too."""
    experts, topk, _, inter = SHAPES["qwen3-30b-a3b-tp8"]
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, 64)
    want = moe.moe_forward(x, w1, w2, ids, weights).clone()
    got = moe.moe_forward(x, w1, w2, ids.long(), weights)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    strided = torch.empty(x.shape[1], x.shape[0], dtype=x.dtype, device="cuda").t()
    fewer = moe.build_routing_device(ids, experts=experts - 1, tile_m=16, reuse=False)
    for call, match in (
        (lambda: moe.moe_forward(x, w1, w2, ids, weights, routing=fewer), "the routing was built for"),
        (lambda: moe.moe_forward(strided, w1, w2, ids, weights), "contiguous"),
        (lambda: moe.moe_forward(x, w1.half(), w2, ids, weights), "must all be bf16 or all f16"),
        (lambda: moe.moe_forward(x, w1, w2, ids, weights, tile_m=48), "tile_m must be one of"),
        (lambda: moe.build_routing_device(ids.long(), experts=experts, tile_m=16), "int32"),
    ):
        with pytest.raises(ValueError, match=match):
            call()


def test_one_token_layer_launches_no_routing_and_no_reduce(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("a one-token layer has no routing or reduce launch")

    for name in ("build_routing_device", "moe_reduce"):
        monkeypatch.setattr(moe, name, _refuse)
    experts, topk, _, inter = SHAPES["qwen3-30b-a3b-tp8"]
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, 1)
    got = moe.moe_forward(x, w1, w2, ids, weights)
    torch.cuda.synchronize()
    _close(got, _torch_layer(x, w1, w2, ids, weights))


@pytest.mark.parametrize("tokens", [1, 16, 64])
def test_out_of_range_expert_contributes_nothing(tokens):
    """1 token is direct + token_down, 16 routed + token_down, 64 routed + gemm2 + reduce; the caching allocator
    hands the layer's buffers memory filled with NaN, so a row left unwritten shows."""
    experts, topk, _, inter = SHAPES["qwen3-30b-a3b-tp8"]
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, tokens)
    ids[:, 0] = experts
    ids[::2, 1] = -1
    ids[1::3, 2] = 1 << 20
    keep = (ids >= 0) & (ids < experts)
    torch.full((64 << 20,), float("nan"), device="cuda")
    got = moe.moe_forward(x, w1, w2, ids, weights)
    torch.cuda.synchronize()
    _close(got, _torch_layer(x, w1, w2, torch.where(keep, ids, 0), weights * keep))


def test_layer_checks_out_tokens_and_offsets():
    experts, topk, _, inter = SHAPES["qwen3-30b-a3b-tp8"]
    x, w1, w2, ids, weights = _layer_inputs(experts, topk, SMALL_HIDDEN, inter, 8)
    empty = moe.moe_forward(x[:0], w1, w2, ids[:0], weights[:0])
    assert tuple(empty.shape) == (0, SMALL_HIDDEN) and empty.dtype == x.dtype
    for bad in (torch.empty(8, SMALL_HIDDEN + 1, dtype=x.dtype, device="cuda"), torch.empty_like(x, dtype=torch.half)):
        with pytest.raises(ValueError, match="out must be"):
            moe.moe_forward(x, w1, w2, ids, weights, out=bad)
    # 2**31 bytes of y at bf16: tokens * topk * hidden = 2**30.
    tokens, hidden = (1 << 30) // (topk * 4096), 4096
    big_x = torch.empty(tokens, hidden, dtype=x.dtype, device="cuda")
    big_w1 = torch.empty(experts, 2 * inter, hidden, dtype=x.dtype, device="cuda")
    big_w2 = torch.empty(experts, hidden, inter, dtype=x.dtype, device="cuda")
    big_ids = torch.zeros(tokens, topk, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="2 GiB"):
        moe.moe_forward(big_x, big_w1, big_w2, big_ids, torch.zeros(tokens, topk, device="cuda"))


def test_gating_then_layer_matches_torch():
    experts, topk, tokens, hidden, inter = 8, 2, 32, SMALL_HIDDEN, 128
    x, w1, w2, _, _ = _layer_inputs(experts, topk, hidden, inter, tokens)
    # Distinct logits per token, so the top-K has no ties for the two sides to break differently.
    g = torch.Generator().manual_seed(0)
    logits = torch.stack([torch.randperm(experts, generator=g) for _ in range(tokens)]).to("cuda", torch.bfloat16) * 0.5
    weights, ids = torch.topk(torch.softmax(logits.float(), dim=1), topk, dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True)
    got = moe.moe_forward_from_logits(x, w1, w2, logits, topk=topk)
    torch.cuda.synchronize()
    _close(got, _torch_layer(x, w1, w2, ids.to(torch.int32), weights))


# ── Tile choice ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("wide", [False, True])
def test_tile_m_thresholds(wide):
    """One row an expert below each threshold is the shorter tile, at the threshold the taller one."""
    dims = {"inter": 1792, "hidden": 4096} if wide else {}
    t = moe._WIDE_EXPERT_ROWS if wide else moe._NARROW_EXPERT_ROWS
    crossings = [(t[0], 16, 32), (t[1], 32, 64), (t[2], 64, 128)]
    for i, (rows, lower, upper) in enumerate(crossings):
        if rows >= 1e9:
            assert moe.tile_m_for(tokens=8192, topk=2, experts=8, **dims) != upper
            continue
        rows = int(-(-rows // 1))
        below = lower
        for rows_prev, lower_prev, _ in reversed(crossings[:i]):
            if rows_prev >= rows:
                below = lower_prev
        assert moe.tile_m_for(tokens=rows - 1, topk=1, experts=1, **dims) == below, (wide, rows)
        if i + 1 < len(crossings) and crossings[i + 1][0] <= rows:
            continue
        assert moe.tile_m_for(tokens=rows, topk=1, experts=1, **dims) == upper, (wide, rows)


@pytest.mark.parametrize("shape_id", list(SHAPES))
def test_tile_m_grows_with_tokens(shape_id):
    experts, topk, hidden, inter = SHAPES[shape_id]
    heights = [
        moe.tile_m_for(tokens=t, topk=topk, experts=experts, inter=inter, hidden=hidden) for t in range(1, 9000, 7)
    ]
    assert set(heights) <= set(moe.SUPPORTED_TILE_M) and heights == sorted(heights)


_SHAPE_ENTRIES = [k for k in moe._TILES if len(k) >= 4]
# rows per expert that land in each bucket, as a multiple of the tile height
_BUCKET_ROWS = {"partial": 0.5, "few": 2.0, "many": 8.0}


@pytest.mark.parametrize("key", _SHAPE_ENTRIES, ids=lambda k: "-".join(map(str, k)))
def test_shape_entries_build_as_listed(key):
    """A shape entry the builder rejects would fall through to a generic tile without a word."""
    stage, tile_m, k_dim, n_out = key[:4]
    rows = _BUCKET_ROWS[key[4]] * tile_m if len(key) == 5 else None
    reg_m, reg_n, reg_k, waves_m, waves_n, *flags = moe._TILES[key][0]
    launch, bm, bn, bk = moe.compile_grouped_gemm(
        k_dim=k_dim,
        n_out=n_out,
        experts=8,
        stage=stage,
        topk=2,
        doweight=stage == "down",
        tile_m=tile_m,
        rows_per_expert=rows,
    )
    assert (bm, bn, bk) == (16 * reg_m * waves_m, 16 * reg_n * waves_n, 16 * reg_k)
    assert launch.g2r == ("g2r" in flags)
    assert launch.m_major == ((stage, k_dim, n_out) in moe._M_MAJOR)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"stage": "down", "tile_m": 48}, "tile_m must be one of"),
        ({"stage": "gate_up", "doweight": True}, "doweight is only implemented"),
        ({"stage": "linear"}, "stage must be one of"),
    ],
)
def test_rejects_bad_requests(kw, match):
    assert moe.SUPPORTED_TILE_M == TILE_HEIGHTS
    with pytest.raises(ValueError, match=match):
        moe.compile_grouped_gemm(**{"k_dim": 256, "n_out": 128, "experts": 4, "topk": 2, **kw})
