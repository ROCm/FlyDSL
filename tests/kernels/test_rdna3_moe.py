#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness tests for the gfx11 (RDNA3) bf16 MoE in ``kernels/moe/rdna3_moe.py``.

The routing kernel is checked against a python loop, each GEMM stage against torch per routed slot (with
sentinel rows past the output to catch a padded row that stores), the bounded grid against the exact one,
the direct, token_down and one-wave g2r paths against the routed ones, and the layer against
``torch_moe_gemm1/2`` at four TP=8 model shapes with hidden cut to 256. The tile tests pin the tile-height
rule and the measured tiles, so a table edit cannot change a kernel without a test noticing.
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
if not _ARCH.startswith("gfx11"):
    pytest.skip(f"RDNA3 MoE requires gfx11*, got {_ARCH}", allow_module_level=True)

import kernels.moe.rdna3_moe as moe  # noqa: E402
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2  # noqa: E402
from tests.test_common import verify_output  # noqa: E402

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
    return (torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()


def _route(tokens, experts, topk, seed=0):
    """Distinct experts per token, and their f32 weights."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights, ids = torch.topk(torch.rand(tokens, experts, generator=g, device="cuda"), k=topk, dim=1)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


def _host_routing(topk_ids, experts, tile_m, tokens):
    """The routing by a python loop: rows grouped by expert, each expert padded up to ``tile_m``."""
    topk = topk_ids.shape[1]
    sentinel = (topk << 24) | tokens
    sorted_ids, expert_ids = [], []
    for e in range(experts):
        token, slot = torch.where(topk_ids == e)
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
        assert torch.equal(r.expert_ids, expert_ids)
        assert torch.equal(_canonical(r.sorted_ids, r.expert_ids, tile_m), _canonical(sorted_ids, expert_ids, tile_m))


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
    tail_e, tail_s = bound.expert_ids[exact.num_blocks :], bound.sorted_ids[exact.num_blocks * tile_m :]
    assert tail_e.numel() > 0
    assert torch.equal(tail_e, torch.zeros_like(tail_e))
    assert torch.equal(tail_s, torch.full_like(tail_s, (topk << 24) | tokens))


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
    w_in = moe.wmma_order(w) if opts.get("b_wmma") else w
    launch(out, a, w_in, sorted_ids, expert_ids, weights, tokens, num_blocks, _stream())
    torch.cuda.synchronize()
    return out, backing[tokens * topk :], (a, w, ids, weights)


@pytest.mark.parametrize("tile_m", TILE_HEIGHTS)
@pytest.mark.parametrize("opts", [{}, {"b_wmma": True}, {"b_wmma": True, "g2r": True}])
def test_gate_up_matches_torch(tile_m, opts):
    cfg = (tile_m // 16, 1, 4, 1, 1) if opts.get("g2r") else None
    kw = dict(tokens=128, k_dim=256, n_out=128, experts=4, topk=2, cfg=cfg, **opts)
    out, extra, (x, w1, ids, weights) = _stage("gate_up", tile_m, **kw)
    ref = torch_moe_gemm1(x, w1, None, None, ids, weights, 128, False)
    assert verify_output(out.float().reshape(ref.shape), ref, atol=0.05, rtol=0.05)
    assert torch.equal(extra, torch.full_like(extra, 7.0)), "a padded routing row stored past the output"


@pytest.mark.parametrize("tile_m", TILE_HEIGHTS)
def test_down_matches_torch_per_slot(tile_m):
    out, extra, (a2, w2, ids, weights) = _stage("down", tile_m, tokens=128, k_dim=256, n_out=256, experts=4, topk=2)
    ref = torch.einsum("tsk,tsnk->tsn", a2.float(), w2[ids.long()].float()) * weights[..., None]
    assert verify_output(out.float(), ref, atol=0.05, rtol=0.05)
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
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


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
    assert verify_output(got.float(), _torch_layer(x, w1, w2, ids, weights), atol=0.05, rtol=0.05)


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
    for call, match in (
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
    assert verify_output(got.float(), _torch_layer(x, w1, w2, ids, weights), atol=0.05, rtol=0.05)


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
    ref = _torch_layer(x, w1, w2, ids.to(torch.int32), weights)
    assert verify_output(got.float(), ref, atol=0.05, rtol=0.05)


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


_RULE_POINTS = {
    "qwen38-flash-next-tp8": (16, 16, 16, 16, 16, 16, 16, 16, 16, 64),
    "qwen3-30b-a3b-tp8": (16, 16, 16, 16, 16, 16, 16, 16, 64, 64),
    "glm45-air-tp8": (16, 16, 16, 16, 16, 16, 16, 64, 64, 64),
    "mixtral-8x7b-tp8": (16, 32, 32, 32, 32, 32, 64, 64, 128, 128),
}


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("shape_id", list(SHAPES))
def test_tile_m_at_benchmark_points(shape_id, tokens):
    experts, topk, hidden, inter = SHAPES[shape_id]
    want = _RULE_POINTS[shape_id][TOKENS.index(tokens)]
    assert moe.tile_m_for(tokens=tokens, topk=topk, experts=experts, inter=inter, hidden=hidden) == want


# (stage, k_dim, n_out, tile_m, rows_per_expert) -> (BLOCK_M, BLOCK_N, BLOCK_K, g2r)
_MEASURED = {
    ("gate_up", 4096, 1792, 16, None): (16, 16, 64, True),
    ("gate_up", 4096, 1792, 32, None): (32, 16, 128, True),
    ("gate_up", 4096, 1792, 128, None): (128, 128, 32, False),
    ("down", 1792, 4096, 16, None): (16, 32, 64, False),
    ("down", 1792, 4096, 32, None): (32, 16, 128, False),
    ("down", 1792, 4096, 128, None): (128, 128, 32, False),
    ("down", 1792, 4096, 64, 760 * 2 / 8): (64, 128, 32, False),
    ("gate_up", 4096, 176, 16, None): (16, 16, 128, False),
    ("gate_up", 4096, 176, 64, None): (64, 16, 64, False),
    ("down", 176, 4096, 16, None): (16, 32, 32, False),
    ("down", 176, 4096, 64, None): (64, 128, 16, False),
    ("gate_up", 2048, 96, 16, None): (16, 16, 128, False),
    ("gate_up", 2048, 96, 16, 128 * 8 / 128): (16, 16, 64, True),
    ("gate_up", 2560, 80, 16, None): (16, 16, 128, False),
    ("down", 80, 2560, 16, None): (16, 64, 16, False),
    ("gate_up", 2048, 768, 16, None): (16, 64, 64, False),
    ("gate_up", 2048, 768, 64, None): (64, 32, 64, False),
    ("down", 768, 2048, 16, None): (16, 64, 32, False),
    ("down", 768, 2048, 64, None): (64, 128, 32, False),
    # A bucket entry beats the generic tile, a shape entry beats the bucket entry.
    ("down", 768, 2048, 64, 128.0): (64, 128, 64, False),
    ("down", 176, 4096, 64, 128.0): (64, 128, 16, False),
    # Shape-and-bucket entries: GLM gate_up overhanging inter 176, one-wave g2r down tiles.
    ("gate_up", 4096, 176, 64, 2048 * 9 / 129): (64, 32, 64, False),
    ("gate_up", 4096, 176, 64, 8192 * 9 / 129): (64, 64, 32, False),
    ("down", 80, 2560, 16, 760 * 11 / 513): (16, 32, 16, True),
    ("down", 96, 2048, 16, 512 * 8 / 128): (16, 32, 32, True),
}


@pytest.mark.parametrize("key", list(_MEASURED), ids=lambda k: "-".join(map(str, k[:4])) + f"-r{k[4]}")
def test_measured_tiles_are_kept(key):
    stage, k_dim, n_out, tile_m, rows_per_expert = key
    launch, bm, bn, bk = moe.compile_grouped_gemm(
        k_dim=k_dim,
        n_out=n_out,
        experts=8,
        stage=stage,
        topk=2,
        doweight=stage == "down",
        tile_m=tile_m,
        rows_per_expert=rows_per_expert,
    )
    assert (bm, bn, bk, launch.g2r) == _MEASURED[key]
    assert launch.m_major is ((stage, k_dim, n_out) == ("gate_up", 4096, 1792))


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
