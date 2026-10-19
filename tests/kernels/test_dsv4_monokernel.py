# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the fused DeepSeek-V4 layer kernel against its torch golden, stage by stage.

Covers the sliding-window, HCA and CSA layers of :mod:`kernels.monokernel.dsv4.kernel`.
The reduced shard keeps ``head_dim`` 512 and ``head_dim - rope_dim`` a multiple of 64,
which the kernel's mappings depend on. The TP8 cases run under ``-m multi_gpu``.
"""

from __future__ import annotations

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.monokernel.dsv4.config import COMPRESS_CSA, COMPRESS_HCA, MoeMode
from kernels.monokernel.dsv4.reference import (
    V4Config,
    bf,
    contiguous_pool,
    decode_kv_fp8,
    dequant,
    encode_kv_fp8,
    fp4_pool_rows,
    fp4_pool_store,
    fp4_row_bytes,
    golden_layer,
    golden_moe,
    make_weights,
    qkv_a_matrix,
    qkv_a_split,
    rmsnorm,
    rope_table,
    unpack_fp4,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"DeepSeek-V4 MonoKernel requires gfx950, got {_ARCH}", allow_module_level=True)

# Each bar is ~2-4x the worst relative error over 12 draws x {w8a8, a8w4} x {hc 1, 4} x {tp 1, 8};
# only `mid` and `x_out` scale with configuration (see _tol).
STAGE_TOL = {
    "q_a": 1e-3,  # worst seen 1e-4
    "kv": 1e-3,  # worst seen 1e-4
    "q": 0.010,  # worst seen 0.0039
    "o": 0.015,  # relative L2 (L2_STAGES); worst seen 0.0036
    "o_lora": 0.015,  # worst seen 0.0052
    "a": 0.015,  # worst seen 0.0065
    "scores": 0.012,  # worst seen 0.0052
    "mid": 0.050,  # worst seen 0.0236 at hc=1/tp1, 0.109 at hc=4/tp8/a8w4
}
SCALES_WITH_CONFIG = ("mid",)
# relative L2: a sharply peaked softmax head turns a 0.2% query error into ~2% on one element
L2_STAGES = ("o",)
OUT_REL_L2 = 0.050  # worst seen 0.0278 at hc=1/tp1, 0.0914 at hc=4/tp8/a8w4
EXACT_STAGES = ("q_a", "kv", "q", "o", "o_lora", "a", "scores")


def _tol(base, hc_mult, npes):
    """Double the bar for mHC mixing and for multi-rank bf16 partial sums, as measured."""
    return base * (2 if hc_mult > 1 else 1) * (2 if npes > 1 else 1)


def _rel(a, b):
    """Max error relative to the reference's largest magnitude."""
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / max(b.norm().item(), 1e-6)).item()


def _cfg(hc_mult=1, compress_ratio=0, max_seq=None, index_topk=None):
    cfg = V4Config(
        heads=8,
        hidden=1024,
        q_lora=512,
        head_dim=512,
        rope_dim=64,
        o_groups=2,
        o_lora=128,
        n_experts=128,
        top_k=6,
        inter=128,
        window=128,
        hc_mult=hc_mult,
    )
    if compress_ratio:
        cfg.compress_ratio = compress_ratio
    if max_seq is not None:
        cfg.max_seq = max_seq
    if index_topk is not None:
        cfg.index_topk = index_topk
    return cfg


def _kernel(cfg, mode=MoeMode.W8A8, S=1, **kw):
    """The weights (seed 3) and a single-rank layer over them."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    W = make_weights(rank=0, cfg=cfg, device="cuda", seed=3, moe_mode=mode)
    return W, Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode, **kw)


def _h(cfg, S=1):
    shape = (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden)
    return (0.5 * torch.randn(*shape, device="cuda")).bfloat16()


def _state(cfg, S=1, width=None):
    """A fresh compressor state, -inf scores like the layer's own so unwritten rows drop out of the softmax."""
    w = cfg.c_coff * (cfg.head_dim if width is None else width)
    return torch.zeros(S, cfg.c_rows, w, device="cuda"), torch.full((S, cfg.c_rows, w), float("-inf"), device="cuda")


def _golden(W, h, pos, kv, dest, idx, cos, sin, mode, state=None, allreduce=lambda z: z, **kw):
    """golden_layer; ``state`` = (kv_state, score_state) for a compressing layer."""
    if state is not None:
        kw |= dict(kv_state=state[0], score_state=state[1], cos_c=cos, sin_c=sin)
    return golden_layer(W, h, pos, kv, dest, idx, cos, sin, allreduce, moe_mode=mode, **kw)


def _own_routing_rel(W, got, out, mode, allreduce=lambda z: z):
    """x_out against the golden MoE run on the kernel's own routing (immune to near-tie flips)."""
    own = golden_moe(W, got["a"], allreduce, mid=got["mid"], sel=got["sel"], prob=got["prob"], moe_mode=mode)
    return _rel_l2(out, own["x_out"])


def _step(layer, cfg, pos, kv, cos, sin, h=None):
    """One single-sample decode step at ``pos``; returns (h, out)."""
    h = _h(cfg) if h is None else h
    idx, dest = contiguous_pool([pos], cfg, "cuda")
    out = layer.forward(h, torch.tensor([pos], dtype=torch.int32, device="cuda"), kv, dest, idx, cos, sin)
    return h, out


def _icache_rows(layer, s, n):
    """Sample ``s``'s first ``n`` indexer-cache entries from the paged FP4 pool, as pack_fp4 rows."""
    return fp4_pool_rows(layer.i_cache[s], layer.i_cache_s[s], layer.block_tables[s], n)


def _routing_flipped(got, ref, W, cfg, S):
    """Did the two sides pick different experts? Asserts any set change is a near-tie, not a selection bug."""
    if got["sel"].tolist() == ref["sel"].tolist():
        return False
    for s in range(S):
        a, b = got["sel"][s].tolist(), ref["sel"][s].tolist()
        if set(a) == set(b):
            continue  # same experts, slot order swapped; still rebased since `mid` is per slot
        key = got["scores"][s].float() + W.t["bias"].float()
        if "tid2eid" not in W.t:
            # the selection itself is checked exactly, on the kernel's own scores
            own = set(key.topk(cfg.top_k).indices.tolist())
            assert set(a) - {cfg.shared_expert} == own, f"sample {s} did not pick its own top-{cfg.top_k}"
        sc = key.sort(descending=True).values
        margin = (sc[cfg.top_k - 1] - sc[cfg.top_k]).item()
        # a flip only if the cut is within score noise, which outgrows 1e-4 in a chained stack
        noise = 2 * (got["scores"][s].float() - ref["scores"][s].float()).abs().max().item()
        assert margin < max(
            1e-4, noise
        ), f"sample {s} chose a different expert SET on a {margin:.3e} margin (score noise {noise:.1e})"
    return True


def _rebase_on_own_routing(got, ref, W, moe_mode):
    """Recompute the golden's MoE half from the kernel's routing (sel/prob only, so `mid` stays under test)."""
    return dict(ref, **golden_moe(W, got["a"], lambda z: z, sel=got["sel"], prob=got["prob"], moe_mode=moe_mode))


def _compare_stages(got, ref, cfg, npes):
    """Per-stage relative error against the golden."""
    for name, base in STAGE_TOL.items():
        tol = _tol(base, cfg.hc_mult, npes) if name in SCALES_WITH_CONFIG else base
        rel = _rel_l2(got[name], ref[name]) if name in L2_STAGES else _rel(got[name], ref[name])
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"


def _check_topk(layer, cfg, n, what=""):
    """The indexer picked exactly min(index_topk, n) distinct slots, the top of its own scores; returns them."""
    k = min(cfg.index_topk, n)
    got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
    sel = got[got >= 0].tolist()
    assert len(sel) == k, f"{what}wrote {len(sel)} slots, want {k}"
    assert len(set(sel)) == k, f"{what}{k - len(set(sel))} picks collided on a slot"
    # judged on the kernel's own scores: golden FP4 ties can move a borderline entry
    sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
    srt = sc.sort(descending=True).values
    margin = (srt[k - 1] - srt[k]).item() if n > k else 1.0
    want = set((cfg.window + sc.topk(k).indices).tolist()) if k else set()
    if set(sel) != want and margin > 1e-6:
        raise AssertionError(f"{what}picked {len(set(sel) - want)} entries the scores do not rank")
    return sel


def _indexer_q(cfg, t, q_an, pos, cos, sin):
    """The golden indexer query: projection, RoPE on the last rope_dim lanes, Hadamard, FP4."""
    from kernels.monokernel.dsv4.reference import hadamard, quant_dequant_fp4, rope

    ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
    q = (q_an @ dequant(t["w_i_q_b"], t["s_i_q_b"], 128).T).view(ih, ihd)
    q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
    return quant_dequant_fp4(bf(hadamard(bf(q))) if cfg.indexer_hadamard else bf(q))


def _layer_vs_golden(cfg, S, mode):
    """One launch of S sequences against the golden, stage by stage and end to end; returns (got, out)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    cfg.validate()
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    h = _h(cfg, S)
    pos = cfg.window  # ring already wrapped once
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = _golden(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, mode)
    if _routing_flipped(got, ref, W, cfg, S):
        ref = _rebase_on_own_routing(got, ref, W, mode)
    _compare_stages(got, ref, cfg, 1)
    assert out.shape == h.shape, f"the layer must preserve its input shape, got {out.shape}"
    # end to end by relative L2: one upstream rounding flip moves an element far, hc_post spreads it
    rel_l2, out_tol = _rel_l2(out, ref["x_out"]), _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol} (rel_max {_rel(out, ref['x_out']):.5f})"
    return got, ref


@pytest.mark.parametrize("moe_mode", [MoeMode.A8W4, MoeMode.W8A8])
@pytest.mark.parametrize("hc_mult", [1, 4])
@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_dsv4_layer_matches_golden(S, moe_mode, hc_mult):
    """Every stage vs the golden; S > 1 is independent sequences, hc_mult=4 the hyper-connection stream."""
    torch.manual_seed(0)
    got, ref = _layer_vs_golden(_cfg(hc_mult), S, moe_mode)
    # relative L2: a lone FP8 rounding flip is one element, a clipped block max (E8M0 rounded down) ~5%
    xq_rel = _rel_l2(got["xq"], ref["xq"])
    assert xq_rel < 0.01, f"quantized expert input diverged: rel_l2 {xq_rel:.5f}"


@pytest.mark.large_shape
@pytest.mark.parametrize("S,hc_mult", [(1, 1), (8, 1), (1, 4), (8, 4)])
def test_dsv4_layer_matches_golden_at_real_dims(S, hc_mult):
    """V4-Pro TP8 dims: mappings only real sizes exercise (384 expert ids, two ug tiles per CTA at S=8)."""
    torch.manual_seed(0)
    cfg = V4Config(hc_mult=hc_mult)  # defaults are DeepSeek-V4-Pro at TP8
    got, _ = _layer_vs_golden(cfg, S, MoeMode.A8W4)
    # an id above 255 needs more than an 8-bit key id field
    assert max(got["sel"].reshape(-1).tolist()[1:]) > 255 or cfg.n_experts <= 256


def test_dsv4_csa_shape_is_the_selected_one():
    """CSA's gather is sized for index_topk selected slots, not the whole compressed half."""
    from kernels.monokernel.dsv4.config import validate_shard

    validate_shard(1, 16, 0, 8, compress_ratio=COMPRESS_CSA)

    csa = V4Config(hc_mult=1, compress_ratio=COMPRESS_CSA, max_seq=4096)
    assert csa.indexed and csa.overlap and csa.c_coff == 2
    assert csa.n_index == min(csa.index_topk, csa.n_compressed)
    assert csa.n_keys == csa.window + csa.n_index  # already a multiple of KEY_BLOCK

    # a long enough sequence is where the cap actually bites
    far = V4Config(hc_mult=1, compress_ratio=COMPRESS_CSA, max_seq=4096 * 16)
    assert far.n_compressed > far.index_topk
    assert far.n_index == far.index_topk, "the gather must stay bounded by index_topk"
    assert far.cache_rows > far.n_keys, "the cache still holds every compressed entry"


def test_dsv4_compress_schedule_is_the_checkpoints():
    """The per-layer compress-ratio schedule is V4-Pro's."""
    from kernels.monokernel.dsv4.config import compress_ratios

    r = compress_ratios()
    assert len(r) == 62, f"61 layers plus one MTP entry, got {len(r)}"
    main, mtp = r[:61], r[61:]
    assert main.count(COMPRESS_HCA) == 31 and main.count(COMPRESS_CSA) == 30, f"31 HCA + 30 CSA, got {main}"
    assert 0 not in main, "V4-Pro has no sliding-window-only main layer"
    assert mtp == (0,), "the MTP block is the ratio-0 entry"
    assert main[0] == main[1] == COMPRESS_HCA, "layers 0 and 1 are the one break in the alternation"
    for i in range(2, 61):
        want = COMPRESS_CSA if i % 2 == 0 else COMPRESS_HCA
        assert main[i] == want, f"layer {i} should be {want}, schedule says {main[i]}"


def test_dsv4_layout_sizes_moe_mailboxes_from_the_build_dims():
    """scores / sel / prob / mid follow the build's n_experts, top_k and inter, not the module defaults."""
    from kernels.monokernel.dsv4.kernel.plan import layout, stage_tasks

    S, ne, k, inter = 2, 1024, 8, 768
    sc, _ = layout(S, 16, 1, n_experts=ne, top_k=k, inter=inter)
    names = list(sc)
    size = {n: sc[names[i + 1]] - sc[n] for i, n in enumerate(names[:-1])}
    assert size["scores"] >= S * ne * 8
    assert size["sel"] >= S * (1 + k) * 8 and size["prob"] >= S * (1 + k) * 8
    assert size["mid"] >= S * (1 + k) * inter * 8
    assert dict(stage_tasks(S, 16, n_experts=ne))["router"] * 8 * 2 >= S * ne  # 8 experts, <= 2 samples a task


def test_dsv4_rejects_unsafe_inputs():
    """Inputs that would fault or hang are refused up front: int64 index tensors (read as int32), a mis-sized
    x_out, a bf16 router bias (read past its end), and a head_dim the PV MFMA grouping cannot fill."""
    from kernels.monokernel.dsv4.kernel import build_dsv4_kernel
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    cfg = _cfg()
    W, layer = _kernel(cfg, MoeMode.A8W4)
    cos, sin = rope_table(4096, theta=cfg.rope_base, device="cuda")
    kv = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device="cuda")
    idx, dest = contiguous_pool([cfg.window], cfg, "cuda")
    cur = torch.tensor([cfg.window], dtype=torch.int32, device="cuda")
    h = torch.zeros(1, cfg.hidden, dtype=torch.bfloat16, device="cuda")
    for bad in (dict(indices=idx.long()), dict(dest_rows=dest.long()), dict(cur_pos=cur.long())):
        args = {"cur_pos": cur, "dest_rows": dest, "indices": idx, **bad}
        with pytest.raises(ValueError, match="contiguous int32"):
            layer.forward(h, args["cur_pos"], kv, args["dest_rows"], args["indices"], cos, sin)
    with pytest.raises(ValueError, match="x_out"):
        layer.forward(h, cur, kv, dest, idx, cos, sin, x_out=torch.empty(1, cfg.hidden // 2, device="cuda"))
    layer.close()
    W.t["bias"] = W.t["bias"].bfloat16()
    with pytest.raises(ValueError, match="router bias must be float32"):
        Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=MoeMode.A8W4)
    with pytest.raises(AssertionError, match="head_dim"):
        build_dsv4_kernel(S=1, heads=8, npes=1, head_dim=128)


def test_dsv4_bounded_poll_flags_instead_of_hanging():
    """Timeout 0 flags the launch and still terminates; the default timeout never fires on a healthy launch."""
    from kernels.monokernel.dsv4.kernel.plan import POLL_TIMEOUT_US
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel, Dsv4Variant

    torch.manual_seed(0)
    cfg, dev = _cfg(), "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=MoeMode.A8W4)
    h = _h(cfg)
    cur = torch.tensor([cfg.window], dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool([cfg.window], cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    kv = (0.3 * torch.randn(cfg.window, cfg.head_dim, device=dev)).bfloat16()
    for timeout, expect in [(POLL_TIMEOUT_US, False), (0, True)]:
        variant = Dsv4Variant(cfg, 1, rank=0, npes=1, moe_mode=MoeMode.A8W4, poll_timeout_us=timeout)
        layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=MoeMode.A8W4, variant=variant)
        for _ in range(2):  # a second launch must still terminate after a flagged one
            layer.forward(h, cur, kv.clone(), dest, idx, cos, sin)
            torch.cuda.synchronize()
        assert variant.hang_detected() == expect, f"timeout {timeout}: hang flag {variant.hang.item()}"
        variant.close()


def _i32(x):
    """``x`` wrapped to int32, as the kernel's tag arithmetic wraps."""
    return (x + 2**31) % 2**32 - 2**31


@pytest.mark.parametrize("s0", [5, 2**25 - 3, 2**31 - 3])
def test_dsv4_step_advance_scrubs_stale_mailboxes(s0):
    """Over one scrub period the step advance zeroes only stale-tagged pairs, across the tag and int32 wraps."""
    from kernels.monokernel.dsv4.kernel import scrub_period
    from kernels.monokernel.dsv4.kernel.plan import LAYER_SLOTS
    from kernels.monokernel.dsv4.op import Dsv4Variant

    variant = Dsv4Variant(_cfg(), 1, rank=0, npes=1, moe_mode=MoeMode.A8W4)
    bufs = [variant.scratch[: variant.scr_pairs * 8], variant.sym_storage[: variant.sym_pairs * 8]]
    pairs = [b.view(torch.int32).view(-1, 2) for b in bufs]
    period = scrub_period(variant.scr_pairs + variant.sym_pairs)
    # old: tagged before s0; new: the last step done and a peer one launch ahead
    old = [_i32((s0 - 1) * LAYER_SLOTS + 1), _i32(s0 * LAYER_SLOTS), _i32((s0 - 7) * LAYER_SLOTS + 4)]
    new = [_i32((s0 + period - 1) * LAYER_SLOTS + 1), _i32((s0 + period) * LAYER_SLOTS + 9)]
    kinds = torch.tensor(old + new + [0], dtype=torch.int32, device="cuda")
    for pr in pairs:
        pr[:, 0] = torch.arange(pr.shape[0], dtype=torch.int32, device="cuda") + 1
        pr[:, 1] = kinds[torch.arange(pr.shape[0], device="cuda") % len(kinds)]
    want = [pr.clone() for pr in pairs]
    for w in want:
        w[(w[:, 1].unsqueeze(1) == kinds[: len(old)]).any(1)] = 0
    variant.step.fill_(_i32(s0))
    for _ in range(period):
        variant.advance_step()
    torch.cuda.synchronize()
    assert variant.step.item() == _i32(s0 + period)
    for name, pr, w in zip(["scratch", "sym"], pairs, want):
        bad = (pr != w).any(1).nonzero()
        assert bad.numel() == 0, f"{name}: {bad.numel()} pairs wrong, first {pr[bad[0, 0]].tolist()}"
    variant.close()


def test_dsv4_layer_output_is_the_same_across_the_tag_wrap():
    """Launches whose tags straddle the 2**32 wrap give bit-identical outputs to steps 0 and 1."""
    from kernels.monokernel.dsv4.kernel.plan import LAYER_SLOTS
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    cfg, dev = _cfg(), "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=MoeMode.A8W4)
    h = _h(cfg)
    idx, dest = contiguous_pool([cfg.window], cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    kv0 = (0.3 * torch.randn(cfg.window, cfg.head_dim, device=dev)).bfloat16()
    outs = []
    for s0 in [0, 2**32 // LAYER_SLOTS - 1]:
        layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=MoeMode.A8W4)
        layer.variant.step.fill_(_i32(s0))
        kv = kv0.clone()
        got = []
        for p in range(2):
            cur = torch.tensor([cfg.window + p], dtype=torch.int32, device=dev)
            got.append(layer.forward(h, cur, kv, dest, idx, cos, sin).clone())
        torch.cuda.synchronize()
        outs.append(got)
        layer.variant.close()
    for p in range(2):
        assert torch.equal(outs[0][p], outs[1][p]), f"step {p}: output differs across the tag wrap"


# Multi-rank TP: ranks sum peer partials in rank order, so they must agree bit-identically on routing.

TP_SEED = 1234


def run_rank(rank, npes, iters=2, moe_mode=MoeMode.A8W4, hc_mult=1, compress_ratio=0):
    import torch.distributed as dist

    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    # not a module global: mp.spawn re-imports this module in each child
    cfg = _cfg(hc_mult, compress_ratio, 256 if compress_ratio else None)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=rank, npes=npes, moe_mode=moe_mode)
    cos, sin = rope_table(4096, theta=cfg.rope_base, device=dev)
    gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical inputs everywhere
    # the compressor carries state, so its run is a real sequential decode; otherwise re-seed from kv0
    kv0 = torch.randn(cfg.window, cfg.head_dim, generator=gen, device=dev).to(torch.bfloat16)
    pos = cfg.window
    idx, dest = contiguous_pool([pos], cfg, dev)
    if compress_ratio:
        pos = 0
        kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        state = _state(cfg)

    def allreduce(x):
        # model peer_reduce: bf16-rounded partials summed in rank order
        if npes == 1:
            return x
        x = x.to(torch.bfloat16).float()
        parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
        dist.all_gather(parts, x.cpu().contiguous())
        return sum(parts[1:], parts[0]).to(x.device)

    def same_everywhere(t):
        peers = [torch.empty_like(t.cpu()) for _ in range(npes)]
        dist.all_gather(peers, t.cpu().contiguous())
        for other in peers[1:]:
            torch.testing.assert_close(other, peers[0], atol=0, rtol=0)

    ok, boundaries = True, 0
    for it in range(iters):
        hshape = (1, cfg.hidden) if cfg.hc_mult == 1 else (1, cfg.hc_mult, cfg.hidden)
        h = torch.randn(*hshape, generator=gen, device=dev).to(torch.bfloat16)
        if compress_ratio:
            pos = it
            idx, dest = contiguous_pool([pos], cfg, dev)
            boundaries += (pos + 1) % compress_ratio == 0
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        out = layer.forward(h, cur, kv_k if compress_ratio else kv0.clone(), dest, idx, cos, sin)
        torch.cuda.synchronize()
        got = layer.intermediates()
        if npes > 1:
            same_everywhere(out)
            same_everywhere(got["sel"])

        kv_ref = kv_r if compress_ratio else kv0.clone()
        ref = _golden(W, h, [pos], kv_ref, dest, idx, cos, sin, moe_mode, state if compress_ratio else None, allreduce)
        if compress_ratio and (pos + 1) % compress_ratio == 0:
            slot = cfg.window + pos // compress_ratio
            c_rel = _rel(kv_k[slot], kv_r[slot])
            if c_rel >= 2e-2:
                print(f"rank {rank}: compressed row at {slot} rel {c_rel:.5f}", flush=True)
                ok = False
        # a near-tie can flip routing; then judge the expert math against the kernel's own routing
        flipped = got["sel"].tolist() != ref["sel"].tolist()
        for name, base in STAGE_TOL.items():
            if flipped and name in SCALES_WITH_CONFIG:
                continue
            tol = _tol(base, cfg.hc_mult, npes) if name in SCALES_WITH_CONFIG else base
            rel = _rel(got[name], ref[name])
            if rel >= tol:
                print(f"rank {rank}: stage {name} rel_max {rel:.5f} rel_l2 {_rel_l2(got[name], ref[name]):.5f}")
                ok = False
        if flipped:
            print(f"rank {rank}: routing flipped on a near-tie (expected; judging by own routing)", flush=True)
        out_tol = _tol(OUT_REL_L2, cfg.hc_mult, npes)
        d_rel = _own_routing_rel(W, got, out, moe_mode, allreduce)
        if d_rel >= out_tol:
            print(f"rank {rank}: x_out vs own-routing golden rel_l2 {d_rel:.5f}", flush=True)
            ok = False
        rel_l2 = _rel_l2(out, ref["x_out"])
        if rank == 0:  # pytest captures this; it is what makes a near-miss legible
            print(f"rank {rank}: x_out rel_max {_rel(out, ref['x_out']):.5f}  rel_l2 {rel_l2:.5f}", flush=True)
        if rel_l2 >= out_tol and not flipped:  # end to end only when routing agrees
            ok = False
    layer.close()
    if compress_ratio and boundaries < 2:
        print(f"rank {rank}: only crossed {boundaries} compression boundaries", flush=True)
        ok = False
    return ok


def _free_port():
    """A fresh port: a fixed one collides with a socket still in TIME_WAIT on back-to-back runs."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, fn, npes, port, results, kw):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = fn(rank, npes, **kw)
    finally:
        dist.destroy_process_group()


def _spawn8(fn, **kw):
    """``fn(rank, 8, **kw)`` on eight ranks; True if every rank returned True."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_worker, args=(fn, 8, _free_port(), results, kw), nprocs=8)
    return all(results[r] for r in range(8))


@pytest.mark.multi_gpu
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_dsv4_layer_tp8(hc_mult):
    """Both residual widths; the tolerances are calibrated per configuration."""
    assert _spawn8(run_rank, hc_mult=hc_mult)


@pytest.mark.multi_gpu
def test_dsv4_hca_layer_tp8():
    """HCA across ranks over two compression boundaries; the replicated compressor must agree on every rank."""
    assert _spawn8(run_rank, iters=20, compress_ratio=8)


def _hca_decode(cfg, mode, steps, stage_names, stage_at):
    """Decode ``steps`` positions through an HCA layer against the golden: each compressed row at its
    boundary, ``stage_names`` at the positions ``stage_at(pos)`` picks, and x_out on the kernel's own routing
    every step (a long run eventually flips a near-tied expert). Returns (boundaries, stage checks, top id)."""
    torch.manual_seed(0)
    cfg.validate()
    ratio, dev = cfg.compress_ratio, "cuda"
    W, layer = _kernel(cfg, mode)
    # one rope table per layer (cfg.rope_base): window q/kv and compressed rows share it
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    state = _state(cfg)
    boundaries, checked, top_id = 0, 0, 0
    for pos in range(steps):
        h, out = _step(layer, cfg, pos, kv_k, cos, sin)
        torch.cuda.synchronize()
        idx, dest = contiguous_pool([pos], cfg, dev)
        ref = _golden(W, h, [pos], kv_r, dest, idx, cos, sin, mode, state)
        if (pos + 1) % ratio == 0:
            boundaries += 1
            slot = cfg.window + pos // ratio
            # not bit-exact: online softmax on hardware exp2 vs a batch softmax, rounded to bf16
            rel = _rel(kv_k[slot], kv_r[slot])
            assert rel < 2e-2, f"compressed row at slot {slot} differs by rel {rel:.5f}"
        got = layer.intermediates()
        if stage_at(pos):
            checked += 1
            same_route = got["sel"].tolist() == ref["sel"].tolist()
            for name in stage_names:
                if name == "mid" and not same_route:  # skip only `mid` on a routing flip
                    continue
                base = STAGE_TOL[name]
                tol = _tol(base, cfg.hc_mult, 1) if name in SCALES_WITH_CONFIG else base
                rel = _rel(got[name], ref[name])
                assert rel < tol, f"pos={pos} stage {name} diverged: rel {rel:.5f} >= {tol}"
        top_id = max(top_id, *got["sel"].reshape(-1).tolist()[1:])
        rel_l2, tol = _own_routing_rel(W, got, out, mode), _tol(OUT_REL_L2, cfg.hc_mult, 1)
        assert rel_l2 < tol, f"pos={pos} x_out rel_l2 {rel_l2:.5f} >= {tol}"
    layer.close()
    return boundaries, checked, top_id


@pytest.mark.parametrize("moe_mode", [MoeMode.W8A8, MoeMode.A8W4])
def test_dsv4_hca_layer_matches_golden(moe_mode):
    """The HCA layer over several compression boundaries; ratio 16 is the same code path as V4's 128."""
    ratio = 16
    boundaries, _, _ = _hca_decode(_cfg(1, ratio, 512), moe_mode, 3 * ratio + 2, ("o",), lambda pos: True)
    assert boundaries >= 3, "must cross several compression boundaries"


@pytest.mark.large_shape
def test_dsv4_hca_layer_at_real_dims():
    """HCA at V4-Pro's real dims: ratio 128 pooling, the full ``ape`` table and the real ``n_keys``."""
    cfg = V4Config(hc_mult=1)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.compress_ratio = COMPRESS_HCA
    assert cfg.n_keys > cfg.window, "the gather must reach past the window"
    ratio = cfg.compress_ratio

    # two boundaries: the first anchors at pos 0, where rope cannot tell the compressor's base apart
    def near(pos):
        return any(abs(pos - (b * ratio - 1)) <= 2 for b in (1, 2))

    boundaries, checked, top_id = _hca_decode(cfg, MoeMode.W8A8, 2 * ratio + 4, tuple(STAGE_TOL), near)
    assert boundaries == 2 and checked == 10
    # an id above 255 needs more than an 8-bit key id field
    assert top_id > 255 or cfg.n_experts <= 256


@pytest.mark.parametrize(
    "S,max_seq,positions,why",
    [
        (2, 4096, (300, 3000), "two samples far apart: a wrong live-split count moves `o`"),
        (8, 65536, tuple(30000 + 3700 * i for i in range(8)), "past one grid round a task folds several tiles"),
    ],
    ids=["dead_splits", "folded_tiles"],
)
def test_dsv4_hca_attends_each_samples_live_keys(S, max_seq, positions, why):
    """Each sample's split / merge covers exactly its own live 64-key tiles (``why``); checks `o` per sample."""
    torch.manual_seed(0)
    ratio, mode = 16, MoeMode.W8A8
    cfg = _cfg(1, ratio, max_seq)
    cfg.validate()
    dev = "cuda"
    W, layer = _kernel(cfg, mode, S)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    pos = [p + ratio // 2 for p in positions]  # off the compression boundaries
    live = [(cfg.window + (p + 1) // ratio + 63) // 64 for p in pos]
    if S == 2:
        assert live[0] < live[1] < cfg.n_keys // 64, f"live splits {live}: the shape no longer has dead ones"
    else:
        assert S * max(live) > 256, f"live tiles {live}: one round of the grid, so no task folds tiles"
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    h = _h(cfg, S)
    idx, dest = contiguous_pool(pos, cfg, dev)
    layer.forward(h, torch.tensor(pos, dtype=torch.int32, device=dev), kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = _golden(W, h, pos, kv0.clone(), dest, idx, cos, sin, mode, _state(cfg, S))
    for s_ in range(S):
        rel = _rel(got["o"][s_], ref["o"][s_])
        assert rel < STAGE_TOL["o"], f"sample {s_} (pos {pos[s_]}, {live[s_]} live tiles): o rel {rel:.5f}"
    layer.close()


@pytest.mark.parametrize(
    "which,indexer_hadamard",
    [("kv", True), ("index", True), ("index", False)],  # ATOM's indexer rotates neither side
)
def test_dsv4_csa_compressors_in_kernel(which, indexer_hadamard):
    """CSA's overlapping KV compressor, and the indexer's (Hadamard + FP4), each row vs ``compress_step``."""
    from kernels.monokernel.dsv4.reference import compress_step

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, 256)
    cfg.indexer_hadamard = indexer_hadamard
    assert cfg.overlap and cfg.c_coff == 2 and cfg.indexed, "ratio 4 is the overlapping, indexed form"
    ihd, dev = cfg.index_head_dim, "cuda"
    W, layer = _kernel(cfg)
    t = W.t
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    if which == "kv":
        ks, ss = (x[0] for x in _state(cfg))
        cache_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        kw = dict(dest_row=0)
    else:
        ks, ss = (x[0] for x in _state(cfg, width=ihd))
        cache_r = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
        kw = dict(head_dim=ihd, ape=t["i_ape"], gamma=t["g_ickv"], rotate=True)
    dq, cut = qkv_a_matrix(t), qkv_a_split(cfg)
    emitted = 0
    for pos in range(5 * ratio):
        h, _ = _step(layer, cfg, pos, kv_k, cos, sin)
        torch.cuda.synchronize()
        proj = bf(rmsnorm(h.float(), t["g_in"], cfg.eps)) @ dq.float().T
        if which == "kv":
            kw["dest_row"] = cfg.window + pos // ratio
        c_kv, c_gate = ("c_kv", "c_gate") if which == "kv" else ("i_kv", "i_gate")
        ref = compress_step(
            proj[0, slice(*cut[c_kv])], proj[0, slice(*cut[c_gate])], pos, cfg, t, ks, ss, cache_r, cos, sin, **kw
        )
        if (pos + 1) % ratio:
            assert ref is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        if which == "kv":
            slot = cfg.window + pos // ratio
            a, b, bar = kv_k[slot].float(), cache_r[slot].float(), 5e-3
        else:
            slot = pos // ratio
            a, b, bar = unpack_fp4(_icache_rows(layer, 0, slot + 1)[slot]), unpack_fp4(cache_r[slot]), 1e-2
        # a last-bit fp32 difference can push one element across an FP8 / FP4 code: bound the count and the
        # bulk, not the max; a wrong overlap or rotation moves most of the row
        n_diff = int((a != b).sum())
        assert n_diff <= 4, f"pos={pos}: {n_diff} elements differ, not a quantization tie"
        assert _rel_l2(a, b) < bar, f"pos={pos} compressed row rel_l2 {_rel_l2(a, b):.5f}"
    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


@pytest.mark.parametrize("indexer_hadamard", [True, False])  # ATOM's indexer rotates neither side
def test_dsv4_indexer_query_in_kernel(indexer_hadamard):
    """The indexer's query path in the kernel: projection, RoPE, Hadamard, FP4 (no per-head RMS)."""
    torch.manual_seed(0)
    cfg = _cfg(1, COMPRESS_CSA, 256)
    cfg.indexer_hadamard = indexer_hadamard
    ih, ihd, dev = cfg.index_heads, cfg.index_head_dim, "cuda"
    W, layer = _kernel(cfg)
    t = W.t
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq_qkv = qkv_a_matrix(t)
    for pos in range(3):
        h, _ = _step(layer, cfg, pos, kv_k, cos, sin)
        torch.cuda.synchronize()
        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        q_an = bf(rmsnorm((x @ dq_qkv.float().T)[:, : cfg.q_lora], t["g_q"], cfg.eps))
        ref = _indexer_q(cfg, t, q_an, pos, cos, sin)
        got = layer.debug("i_q", (1, ih, ihd))[0]
        # one FP4 step is ~0.5 here: bound the rate of differing elements, a broken rotation moves most
        n_diff = int((got != ref).sum())
        assert n_diff <= ih * ihd // 50, f"pos={pos}: {n_diff}/{ih * ihd} elements differ"
        assert _rel_l2(got, ref) < 6e-2, f"pos={pos} indexer query rel_l2 {_rel_l2(got, ref):.5f}"


@pytest.mark.parametrize("indexer_hadamard", [True, False])  # ATOM's indexer rotates neither side
def test_dsv4_indexer_scoring_in_kernel(indexer_hadamard):
    """The indexer score sum_h relu(q[h] . k[c]) * w[h] per written entry; unwritten entries score NEG."""
    from kernels.monokernel.dsv4.reference import compress_step

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    # up to index_topk live entries nothing is scored, so a small k scores from the third entry on
    cfg = _cfg(1, ratio, 256, index_topk=2)
    cfg.indexer_hadamard = indexer_hadamard
    ih, ihd, dev = cfg.index_heads, cfg.index_head_dim, "cuda"
    W, layer = _kernel(cfg)
    t = W.t
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks, i_ss = (x[0] for x in _state(cfg, width=ihd))
    i_ref = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
    dq_qkv, cut = qkv_a_matrix(t), qkv_a_split(cfg)
    scale = ihd**-0.5 * cfg.index_heads**-0.5
    scored = 0
    for pos in range(4 * ratio):
        h, _ = _step(layer, cfg, pos, kv_k, cos, sin)
        torch.cuda.synchronize()
        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        kw = dict(head_dim=ihd, ape=t["i_ape"], gamma=t["g_ickv"], rotate=True)
        compress_step(
            proj[0, slice(*cut["i_kv"])], proj[0, slice(*cut["i_gate"])], pos, cfg, t, i_ks, i_ss, i_ref, cos, sin, **kw
        )
        q = _indexer_q(cfg, t, bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps)), pos, cos, sin)
        w = bf(x @ t["i_w"].float().T)[0] * scale
        n = (pos + 1) // ratio
        got = layer.debug("i_score", (1, cfg.n_compressed))[0]
        if n <= cfg.index_topk:  # every live entry kept, nothing scored
            continue
        scored += 1
        ref = (torch.einsum("hd,td->ht", q, unpack_fp4(i_ref[:n])).relu() * w.view(ih, 1)).sum(0)
        assert _rel_l2(got[:n], ref) < 2e-2, f"pos={pos} score rel_l2 {_rel_l2(got[:n], ref):.5f}"
        assert bool((got[n:] < 0).all()), f"pos={pos}: unwritten entries are scorable"
    assert scored >= 3, f"expected several scored steps, got {scored}"


def _indexer_score_rank(rank, npes):
    """One rank of the replicated indexer's scoring; see the test below."""
    import torch.distributed as dist

    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, 256, index_topk=2)
    ih, ihd, mode = cfg.index_heads, cfg.index_head_dim, MoeMode.W8A8
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=rank, npes=npes, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq_qkv = qkv_a_matrix(t)
    scale = ihd**-0.5 * cfg.index_heads**-0.5
    gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical everywhere
    ok, scored = True, 0
    for pos in range(4 * ratio):  # entries 3 and 4 pass index_topk and are scored
        h = torch.randn(1, cfg.hidden, generator=gen, device=dev).to(torch.bfloat16)
        _step(layer, cfg, pos, kv_k, cos, sin, h)
        torch.cuda.synchronize()
        got = layer.debug("i_score", (1, cfg.n_compressed))[0]
        # bit-identical across ranks, or the top-k would pick different keys per rank
        peers = [torch.empty_like(got.cpu()) for _ in range(npes)]
        dist.all_gather(peers, got.cpu().contiguous())
        for other in peers[1:]:
            torch.testing.assert_close(other, peers[0], atol=0, rtol=0)
        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        q_an = bf(rmsnorm((x @ dq_qkv.float().T)[:, : cfg.q_lora], t["g_q"], cfg.eps))
        q_ref, w_ref = _indexer_q(cfg, t, q_an, pos, cos, sin), bf(x @ t["i_w"].float().T)[0] * scale
        # score from the kernel's own q, w and key cache (each checked on its own bar),
        # so only the dot product and head sum are under test
        q, w = layer.debug("i_q", (1, ih, ihd))[0], layer.debug("i_wp", (1, ih))[0]
        dq_n, dq_l2, dw = int((q != q_ref).sum()), _rel_l2(q, q_ref), _rel(w, w_ref)
        if dq_n > ih * ihd // 50 or dq_l2 >= 6e-2 or dw >= 1e-2:
            print(f"rank {rank} pos={pos} q differs on {dq_n} (l2 {dq_l2:.5f}), w rel {dw:.5f}", flush=True)
            ok = False
        n = (pos + 1) // ratio
        if n <= cfg.index_topk:  # every live entry kept, nothing scored
            continue
        scored += 1
        ref = (torch.einsum("hd,td->ht", q, unpack_fp4(_icache_rows(layer, 0, n))).relu() * w.view(ih, 1)).sum(0)
        if _rel_l2(got[:n], ref) >= 1e-3:
            print(f"rank {rank} pos={pos} score rel_l2 {_rel_l2(got[:n], ref):.5f}", flush=True)
            ok = False
    layer.close()
    return ok and scored >= 2


@pytest.mark.multi_gpu
def test_dsv4_indexer_scores_match_on_every_rank_tp8():
    """Every rank scores with all index heads: the scores match the golden and are bit-identical across ranks."""
    assert _spawn8(_indexer_score_rank)


def test_dsv4_indexer_scores_the_new_entry_past_the_first_tile():
    """The entry written this launch is scored from the mailbox even past the first SCORE_TILE (poisoned cache)."""
    from kernels.monokernel.dsv4.kernel.plan import SCORE_TILE

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, 4096, index_topk=64)
    ih, ihd, dev = cfg.index_heads, cfg.index_head_dim, "cuda"
    assert cfg.n_compressed > SCORE_TILE + 4, "the shape must reach the second score tile"
    W, layer = _kernel(cfg)
    layer.i_cache.copy_(torch.randint(0, 256, layer.i_cache.shape, device=dev))
    layer.i_cache_s.copy_(torch.randint(0, 256, layer.i_cache_s.shape, device=dev))
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    # the last entry of tile 0 as a control, then the first four of tile 1
    checks = [ratio * (SCORE_TILE + j) - 1 for j in range(5)]
    for pos in range(checks[-1] + 1):
        _step(layer, cfg, pos, kv_k, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        new = pos // ratio
        q, w = layer.debug("i_q", (1, ih, ihd))[0], layer.debug("i_wp", (1, ih))[0]
        got = layer.debug("i_score", (1, cfg.n_compressed))[0][new].item()
        ref = ((q @ unpack_fp4(_icache_rows(layer, 0, new + 1)[new])).relu() * w).sum().item()
        assert abs(got - ref) <= 1e-3 * max(abs(ref), 1e-3), f"pos={pos} entry {new}: score {got} vs {ref}"
    layer.close()


def test_dsv4_indexer_topk_in_kernel():
    """The indexer's selected set is exactly the top-k of its scores (index_topk reduced so it discards)."""
    from kernels.monokernel.dsv4.reference import indexer_step

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, 256, index_topk=4)
    ihd, dev = cfg.index_head_dim, "cuda"
    W, layer = _kernel(cfg)
    t = W.t
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks, i_ss = (x[0] for x in _state(cfg, width=ihd))
    i_ref = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
    dq_qkv, cut = qkv_a_matrix(t), qkv_a_split(cfg)
    chose, discarded = 0, 0
    for pos in range(8 * ratio):
        h, _ = _step(layer, cfg, pos, kv_k, cos, sin)
        torch.cuda.synchronize()
        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
        ref = indexer_step(
            x[0], q_an[0], proj[0, slice(*cut["i_kv"])], proj[0, slice(*cut["i_gate"])], pos, cfg, t,
            i_ks, i_ss, i_ref, cos, sin,
        )  # fmt: skip
        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        _check_topk(layer, cfg, n, f"pos={pos}: ")
        if not k:
            continue
        chose += 1
        discarded += n > cfg.index_topk
        assert len(set(ref[ref >= 0].tolist())) == k
    assert chose >= 6 and discarded >= 3, f"chose {chose}, discarded on {discarded}"


@pytest.mark.parametrize(
    "max_seq,index_topk,checks,min_reach,why",
    [
        # picks reaching past candidate 256 span five of the eight waves: the cross-wave compaction scan
        (2048, 200, (400, 700, 1100), 256, "the compaction's cross-wave scan"),
        # 1500 live candidates: rounds 0, 1 and part of 2 of the radix's strided per-thread walk
        (8192, 300, (2500, 4000, 5999), 2 * 512, "several candidates per thread"),
    ],
    ids=["cross_wave_scan", "candidates_per_thread"],
)
def test_dsv4_indexer_topk_over_a_decode(max_seq, index_topk, checks, min_reach, why):
    """The top-k stays exact over a real decode at shapes that reach ``why``."""
    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, max_seq, index_topk=index_topk)
    dev = "cuda"
    _, layer = _kernel(cfg)
    cos, sin = rope_table(max(4096, max_seq), theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    reach = 0
    for pos in range(checks[-1] + 1):
        _step(layer, cfg, pos, kv_k, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        sel = _check_topk(layer, cfg, (pos + 1) // ratio, f"pos={pos}: ")
        reach = max(reach, max(s - cfg.window for s in sel))
    # the point of the shape: without this the path is never read
    assert reach >= min_reach, f"picks stopped at candidate {reach}, so {why} is untested"
    layer.close()


@pytest.mark.parametrize(
    "max_seq,index_topk,n_live",
    [
        (65536, 300, 3000),  # candidates in one CTA part
        (65536, 300, 6000),  # split across parts, one of them empty
        (1 << 20, None, None),  # a full 1M context: eight register-held trips per thread, all live
    ],
)
def test_dsv4_indexer_topk_over_a_filled_cache(max_seq, index_topk, n_live):
    """The top-k over an FP4 key cache filled directly: across several CTA parts and at a full 1M context."""
    from kernels.monokernel.dsv4.kernel.plan import THREADS, n_topk_parts
    from kernels.monokernel.dsv4.reference import pack_fp4

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(1, ratio, max_seq, index_topk=index_topk)
    n_all, ihd, dev = cfg.n_compressed, cfg.index_head_dim, "cuda"
    parts = n_topk_parts(cfg.max_seq, ratio, ihd)
    if n_live is None:
        trips = n_all // (THREADS * 4 * parts)
        assert trips >= 8, f"only {trips} trips a thread; the point is several"
    else:
        assert parts == 4 and n_all // parts == 4096, "shape no longer splits as the cases assume"
    _, layer = _kernel(cfg)
    rows = pack_fp4(torch.randn(n_all, ihd, device=dev))
    fp4_pool_store(layer.i_cache[0], layer.i_cache_s[0], layer.block_tables[0], rows)
    cos, sin = rope_table(max_seq, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    pos = cfg.max_seq - 1 if n_live is None else n_live * ratio - 1
    _step(layer, cfg, pos, kv_k, cos, sin)
    torch.cuda.synchronize()
    sel = _check_topk(layer, cfg, (pos + 1) // ratio)
    if n_live is not None and n_live > n_all // parts:
        reach = max(s_ - cfg.window for s_ in sel)
        assert reach >= 2 * THREADS * 4, f"picks stopped at candidate {reach}, so part 2 was not tested"
    layer.close()


@pytest.mark.parametrize("case", ["group_scales", "poisoned_rows"])
def test_dsv4_fp8_kv_read(case):
    """The fp8 KV read: each 64-wide group takes its own scale (group g scaled 4**g, so a wrong one is >= 4x
    off); rows ATOM never wrote (all 0xFF) or wrote with a NaN RoPE half are left out, as if the key were -1."""
    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg()
    cfg.kv_fp8 = True
    cfg.validate()
    W, layer = _kernel(cfg, mode, S)
    if case == "group_scales":
        nope = cfg.head_dim - cfg.rope_dim
        gain = torch.ones(cfg.head_dim, device=dev)
        gain[:nope] = 4.0 ** (torch.arange(nope, device=dev) // 64 - 3).float()
        kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev) * gain).bfloat16()
    else:
        kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    nope_u8, rope = encode_kv_fp8(kv0)
    h = _h(cfg, S)
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    idx_ref = idx.clone()
    if case == "poisoned_rows":
        unwritten, nan_rope = int(idx[0, 3]), int(idx[0, 10])
        nope_u8[unwritten] = 0xFF
        rope[unwritten].view(torch.int16).fill_(-1)
        rope[nan_rope] = float("nan")
        idx_ref[0, 3] = idx_ref[0, 10] = -1
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    layer.forward(h, cur, (nope_u8, rope), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    kv_ref = torch.nan_to_num(decode_kv_fp8(nope_u8, rope)).bfloat16()  # the same values, as a bf16 plane
    ref = _golden(W, h, [pos] * S, kv_ref, dest, idx_ref, cos, sin, mode)
    assert not torch.isnan(got["o"]).any(), "a poisoned row reached the attention output"
    rel = _rel_l2(got["o"], ref["o"])
    assert rel < 2e-2, f"attention output off the golden by rel_l2 {rel:.4f}"
    layer.close()


@pytest.mark.parametrize("S", [1, 2])
def test_dsv4_hash_routing_takes_the_table(S):
    """Hash-routed layers pick exactly tid2eid[token] per sample, not the scored top-k."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode = "cuda", MoeMode.W8A8
    cfg = _cfg()
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    vocab = 1000
    tid2eid = [torch.randperm(cfg.n_experts, device=dev)[: cfg.top_k] for _ in range(vocab)]
    W.t["tid2eid"] = torch.stack(tid2eid).to(torch.int32)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    tokens = torch.randint(0, vocab, (S,), dtype=torch.int32, device=dev)
    h = _h(cfg, S)
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin, tokens=tokens)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = _golden(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, mode, tokens=tokens)
    for s in range(S):
        want = set(W.t["tid2eid"][tokens[s].long()].tolist())
        picked = set(got["sel"][s].tolist()[1:])  # slot 0 is the shared expert
        assert picked == want, f"sample {s}: routed to {sorted(picked)}, table says {sorted(want)}"
    assert _rel_l2(out, ref["x_out"]) < _tol(OUT_REL_L2, cfg.hc_mult, 1), "x_out off the golden"
    layer.close()


@pytest.mark.parametrize("ratio", [0, COMPRESS_CSA])
def test_dsv4_batching_is_independent_sequences(ratio):
    """S=2 equals two S=1 runs exactly: catches a stage reading sample 0's state or position for all."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 2
    cfg = _cfg(4, ratio, 256, index_topk=4 if ratio else None)
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    # deep enough that the top-k discards, or a wrong sample's scores would change nothing
    steps = 8 * ratio if ratio else 3
    if ratio:
        assert (steps - 1) // ratio > cfg.index_topk, "the top-k must discard, or the scores are untested"
    hs = [(0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=dev)).bfloat16() for _ in range(steps)]
    # staggered starts, not a multiple of the ratio: equal positions would hide a wrong-sample position
    OFFSETS = (0, 3)
    assert len(OFFSETS) == S and (not ratio or OFFSETS[1] % ratio), "offsets must stagger the boundary"

    def run(rows):
        n = len(rows)
        layer = Dsv4MonoKernel(W, samples=n, rank=0, npes=1, moe_mode=mode)
        kv = torch.zeros(n * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        seq = []
        for step in range(steps):
            ps = [OFFSETS[r] + step for r in rows]
            idx, dest = contiguous_pool(ps, cfg, dev)
            h = torch.cat([hs[step][r : r + 1] for r in rows])
            out = layer.forward(h, torch.tensor(ps, dtype=torch.int32, device=dev), kv, dest, idx, cos, sin).clone()
            torch.cuda.synchronize()
            seq.append((out, {k: v.clone() for k, v in layer.intermediates().items()}))
        layer.close()
        return seq

    batched = run(list(range(S)))
    for s in range(S):
        alone = run([s])
        for pos in range(steps):
            for name in EXACT_STAGES:
                d = (batched[pos][1][name][s].float() - alone[pos][1][name][0].float()).abs().max().item()
                assert d == 0.0, f"pos={pos} sample {s}: stage {name} moved by {d:.3e} when batched"
            assert batched[pos][1]["sel"][s].tolist() == alone[pos][1]["sel"][0].tolist()
            # x_out is not bit-exact: batching regroups `down`'s f32 summation tree
            rel = _rel_l2(batched[pos][0][s], alone[pos][0][0])
            assert rel < 1e-3, f"pos={pos} sample {s}: x_out rel {rel:.3e} when batched"


def _mtp_pool(positions, seqs, cfg, k, dev):
    """``(indices, dest_rows, n_seq, rows_per_seq)`` for an MTP pool with ATOM's ``window + k`` ring rows."""
    ring = cfg.window + k
    tot = ring + cfg.n_compressed
    rows, d0, d1 = [], [], []
    for p, q in zip(positions, seqs):
        base = q * tot
        n = min(p + 1, cfg.window)
        r = [base + pp % ring for pp in range(p + 1 - n, p + 1)] + [-1] * (cfg.window - n)
        if cfg.compress_ratio:
            nc = 0 if cfg.indexed else (p + 1) // cfg.compress_ratio
            r += [base + ring + i for i in range(nc)] + [-1] * (cfg.n_index - nc)
        rows.append(r + [-1] * (cfg.n_keys - len(r)))
        d0.append(base + p % ring)
        d1.append(base + ring)
    idx = torch.tensor(rows, dtype=torch.int32, device=dev)
    return idx, torch.tensor([d0, d1], dtype=torch.int32, device=dev), max(seqs) + 1, tot


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("ratio", [COMPRESS_CSA, 16])
def test_dsv4_mtp_verify_step_is_sequential_decode(ratio, rollback):
    """An MTP verify launch (K + 1 tokens per sequence) equals sequential decode, with random draft rejection."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, K = "cuda", MoeMode.W8A8, 3
    TOK, NSEQ = K + 1, 2
    cfg = _cfg(1, ratio, 512, index_topk=4 if ratio == COMPRESS_CSA else None)
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    OFFSETS = (0, 3)
    n_acc = 8 * ratio if ratio == COMPRESS_CSA else 3 * ratio + 2
    if ratio == COMPRESS_CSA:
        assert (n_acc - 1) // ratio > cfg.index_topk, "the top-k must discard"
    h_cache = {}

    def h_acc(r, pos):
        if (r, pos) not in h_cache:
            h_cache[(r, pos)] = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        return h_cache[(r, pos)]

    # reference: one launch per accepted position, both sequences per launch
    ref = {}
    lay = Dsv4MonoKernel(W, samples=NSEQ, rank=0, npes=1, moe_mode=mode)
    kv = None
    for k in range(n_acc):
        ps = [OFFSETS[r] + k for r in range(NSEQ)]
        idx, dest, nseq, tot = _mtp_pool(ps, list(range(NSEQ)), cfg, K, dev)
        kv = kv if kv is not None else torch.zeros(nseq * tot, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        h = torch.cat([h_acc(r, ps[r]) for r in range(NSEQ)])
        out = lay.forward(h, torch.tensor(ps, dtype=torch.int32, device=dev), kv, dest, idx, cos, sin).clone()
        torch.cuda.synchronize()
        got = lay.intermediates()
        for r in range(NSEQ):
            ref[(r, ps[r])] = (out[r], {n: got[n][r].clone() for n in EXACT_STAGES + ("sel",)})
    lay.close()

    # MTP: K + 1 tokens per sequence per launch
    gen = torch.Generator().manual_seed(1)
    lay = Dsv4MonoKernel(W, samples=NSEQ * TOK, rank=0, npes=1, moe_mode=mode, tokens_per_seq=TOK)
    kv = torch.zeros(NSEQ * tot, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    nxt = list(OFFSETS)
    checked = 0
    while min(nxt[r] - OFFSETS[r] for r in range(NSEQ)) < n_acc - TOK:
        acc = [
            int(torch.randint(1, TOK + 1, (1,), generator=gen, device="cpu")) if rollback else TOK for _ in range(NSEQ)
        ]
        ps = [nxt[r] + j for r in range(NSEQ) for j in range(TOK)]
        sq = [r for r in range(NSEQ) for _ in range(TOK)]
        hs = []
        for r in range(NSEQ):
            for j in range(TOK):
                ok = j < acc[r]
                hs.append(h_acc(r, nxt[r] + j) if ok else (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16())
        idx, dest, _, _ = _mtp_pool(ps, sq, cfg, K, dev)
        cur = torch.tensor(ps, dtype=torch.int32, device=dev)
        out = lay.forward(torch.cat(hs), cur, kv, dest, idx, cos, sin).clone()
        torch.cuda.synchronize()
        got = lay.intermediates()
        for r in range(NSEQ):
            for j in range(acc[r]):
                i, pos = r * TOK + j, nxt[r] + j
                if (r, pos) not in ref:  # past the reference's history (the other sequence lagged)
                    continue
                r_out, r_st = ref[(r, pos)]
                for name in EXACT_STAGES:
                    d = (got[name][i].float() - r_st[name].float()).abs().max().item()
                    assert d == 0.0, f"seq {r} pos {pos} (token {j}): stage {name} moved by {d:.3e} in the MTP launch"
                assert got["sel"][i].tolist() == r_st["sel"].tolist(), f"seq {r} pos {pos}: routing differs"
                rel = _rel_l2(out[i], r_out)
                assert rel < 1e-3, f"seq {r} pos {pos}: x_out rel {rel:.3e}"
                checked += 1
            nxt[r] += acc[r]
    lay.close()
    assert checked >= n_acc, f"only {checked} tokens compared"


@pytest.mark.parametrize("ratio", [COMPRESS_CSA, COMPRESS_HCA])
def test_dsv4_paged_blocks_are_pure_addressing(ratio):
    """ATOM-style scattered paging of compressed entries is bit-identical to a contiguous pool."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 2
    cfg = _cfg(1, ratio, 1024)
    if cfg.indexed:
        cfg.index_topk = 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    lay_a = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    lay_b = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    k_pb, nb = lay_a.k_pb, lay_a.block_tables.shape[1]
    env = 2 * k_pb
    phys = torch.randperm(2 * nb * S, device=dev)[: S * nb].to(torch.int32).view(S, nb)
    comp_base = S * cfg.cache_rows
    kv_a = torch.zeros(S * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_b = torch.zeros(comp_base + 2 * nb * S * env, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    if cfg.indexed:
        lay_b.st_ic = 0
        lay_b.i_cache = torch.zeros(2 * nb * S, lay_a.i_cache.shape[-1], dtype=torch.uint8, device=dev)
        lay_b.i_cache_s = torch.zeros(2 * nb * S, lay_a.i_cache_s.shape[-1], dtype=torch.uint8, device=dev)

    def paged(row, s):
        # clamped: torch.where evaluates this for window and -1 rows too (a device fault otherwise)
        e = (row - s * cfg.cache_rows - cfg.window).clamp(min=0)
        return comp_base + phys[s, e // k_pb] * env + e % k_pb

    steps = 3 * k_pb * ratio
    for pos in range(steps):
        h = _h(cfg, S)
        cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos] * S, cfg, dev)
        idx_b, dest_b = idx.clone(), dest.clone()
        dest_b[1] = comp_base
        for s in range(S):
            row = idx[s]
            comp = (row >= s * cfg.cache_rows + cfg.window) & (row >= 0)
            idx_b[s] = torch.where(comp, paged(row, s), row)
        a = lay_a.forward(h, cur, kv_a, dest, idx, cos, sin)
        b = lay_b.forward(h, cur, kv_b, dest_b, idx_b, cos, sin, block_tables=phys, env_rows=env)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"pos={pos}: paging changed the output by {(a.float() - b.float()).abs().max():.3e}"
    n = steps // ratio
    for s in range(S):
        rows_a = kv_a[s * cfg.cache_rows + cfg.window : s * cfg.cache_rows + cfg.window + n]
        rows_b = kv_b[paged(torch.arange(n, device=dev) + s * cfg.cache_rows + cfg.window, s)]
        assert torch.equal(rows_a, rows_b), f"sample {s}: compressed KV rows differ at their paged addresses"
        if cfg.indexed:
            ia = fp4_pool_rows(lay_a.i_cache[s], lay_a.i_cache_s[s], lay_a.block_tables[s], n)
            ib = fp4_pool_rows(lay_b.i_cache, lay_b.i_cache_s, phys[s], n)
            assert torch.equal(ia, ib), f"sample {s}: indexer entries differ at their paged addresses"
    lay_a.close()
    lay_b.close()


@pytest.mark.large_shape
def test_dsv4_rows_and_state_past_4gb():
    """KV rows and compressor state 4.4 GB into their pools (as in ATOM's) are bit-identical: no 32-bit wrap."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg(1, COMPRESS_HCA, 512)
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    lay_a = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    lay_b = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    far_rows = (4400 << 20) // (cfg.head_dim * 2)  # 4.4 GB of bf16 rows
    kv_a = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_b = torch.zeros(far_rows + cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    width = cfg.c_coff * cfg.head_dim
    st = (4400 << 20) // 4  # f32 elements: slot 1 starts 4.4 GB in
    kst = torch.zeros(st + cfg.c_rows * width, device=dev)
    sst = torch.zeros_like(kst)
    sst[st:] = float("-inf")
    far = dict(kv_state=kst, score_state=sst, st_kv=st, state_slots=torch.ones(S, dtype=torch.int32, device=dev))
    for pos in range(2 * cfg.compress_ratio + 3):
        h = _h(cfg, S)
        cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos] * S, cfg, dev)
        a = lay_a.forward(h, cur, kv_a, dest, idx, cos, sin)
        far_idx = torch.where(idx >= 0, idx + far_rows, idx)
        b = lay_b.forward(h, cur, kv_b, dest + far_rows, far_idx, cos, sin, state=far)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"pos={pos}: rows / state past 4 GB changed the output"
    assert torch.equal(kv_a, kv_b[far_rows:]), "the rows past 4 GB do not hold what the small pool does"
    lay_a.close()
    lay_b.close()


def test_dsv4_state_slots_place_the_rolling_state():
    """The compressor state lives where `state_slots` says (non-identity slots give identical answers)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S, ratio = "cuda", MoeMode.W8A8, 2, COMPRESS_CSA
    cfg = _cfg(4, ratio, 256, index_topk=4)
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    steps, coff, ihd = 6 * ratio, cfg.c_coff, cfg.index_head_dim
    hs = [(0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=dev)).bfloat16() for _ in range(steps)]

    def run(slots, pool):
        layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
        if pool != S:  # repoint at a wider pool, as a runtime's allocator would
            layer.state_slots = torch.tensor(slots, dtype=torch.int32, device=dev)
            # poison every slot, then initialise only the assigned ones, so a wrong slot reads noise
            layer.kv_state = torch.randn(pool, cfg.c_rows, coff * cfg.head_dim, device=dev)
            layer.score_state = torch.randn(pool, cfg.c_rows, coff * cfg.head_dim, device=dev)
            layer.i_kv_state = torch.randn(pool, cfg.c_rows, coff * ihd, device=dev)
            layer.i_score_state = torch.randn(pool, cfg.c_rows, coff * ihd, device=dev)
            layer.i_cache = torch.randint(0, 256, (pool, *layer.i_cache.shape[1:]), device=dev).byte()
            layer.i_cache_s = torch.randint(0, 256, (pool, *layer.i_cache_s.shape[1:]), device=dev).byte()
            for sl in slots:
                layer.kv_state[sl] = 0
                layer.score_state[sl] = float("-inf")
                layer.i_kv_state[sl] = 0
                layer.i_score_state[sl] = float("-inf")
                layer.i_cache[sl] = 0
                layer.i_cache_s[sl] = 0
        kv = torch.zeros(S * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        outs = []
        for pos in range(steps):
            cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool([pos] * S, cfg, dev)
            outs.append(layer.forward(hs[pos], cur, kv, dest, idx, cos, sin).clone())
        torch.cuda.synchronize()
        layer.close()
        return outs

    packed, scattered = run([0, 1], S), run([3, 1], S + 3)
    for pos in range(steps):
        d = (packed[pos].float() - scattered[pos].float()).abs().max().item()
        assert d == 0.0, f"pos={pos}: moving the state to other slots changed the answer by {d:.3e}"


def test_dsv4_split_merge_spans_the_block():
    """More than 64 key splits takes the block-wide merge path; overrunning it is silently wrong, not a crash."""
    from kernels.monokernel.dsv4.kernel.plan import SPLIT_KEYS

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg(1, COMPRESS_HCA, 655360)
    cfg.validate()
    splits = cfg.n_keys // SPLIT_KEYS
    assert splits > 64, f"this shape must exercise the wide path, got {splits} splits"
    W, layer = _kernel(cfg, mode, S)
    h = _h(cfg, S)
    pos = cfg.max_seq - 1  # every compressed entry live
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_theta, device=dev)
    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    ref = _golden(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, mode, _state(cfg, S))
    assert torch.isfinite(out.float()).all(), "the wide merge produced non-finite output"
    rel_l2 = _rel_l2(out, ref["x_out"])
    assert rel_l2 < _tol(OUT_REL_L2, cfg.hc_mult, 1), f"x_out diverged: rel_l2 {rel_l2:.5f}"


@pytest.mark.parametrize("kv_fp8", [False, True])
def test_dsv4_alternating_stack_matches_golden(kv_fp8):
    """V4-Pro's [HCA, HCA, CSA, HCA] prefix: layers of one variant share a kernel, not weights or state."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel, Dsv4Variant

    torch.manual_seed(0)
    dev, mode, S, n_layers = "cuda", MoeMode.W8A8, 1, 4
    base = _cfg(4, max_seq=256)
    cfgs = []
    for i in range(n_layers):
        c = base.for_layer(i)
        c.kv_fp8 = kv_fp8
        if c.indexed:
            c.index_topk = 4
        c.validate()
        cfgs.append(c)
    assert {c.compress_ratio for c in cfgs} == {COMPRESS_HCA, COMPRESS_CSA}, "the prefix must alternate"

    Ws = [make_weights(rank=0, cfg=c, device=dev, seed=100 + i, moe_mode=mode) for i, c in enumerate(cfgs)]
    variants, layers = {}, []
    for i, c in enumerate(cfgs):
        if c.compress_ratio not in variants:
            variants[c.compress_ratio] = Dsv4Variant(c, S, rank=0, npes=1, moe_mode=mode)
        layers.append(Dsv4MonoKernel(Ws[i], S, rank=0, npes=1, moe_mode=mode, variant=variants[c.compress_ratio]))
    assert len(variants) == 2, "three HCA layers must share one compiled kernel"

    tables = {c.rope_base: rope_table(2048, theta=c.rope_base, device=dev) for c in cfgs}
    # separate caches, or the golden would gather rows the kernel wrote
    kvs_k = [torch.zeros(c.cache_rows, c.head_dim, dtype=torch.bfloat16, device=dev) for c in cfgs]
    if kv_fp8:
        kvs_k = [encode_kv_fp8(k) for k in kvs_k]
    kvs_r = [torch.zeros(c.cache_rows, c.head_dim, dtype=torch.bfloat16, device=dev) for c in cfgs]

    def fresh_states():
        st_all = []
        for c in cfgs:
            st = dict(zip(("kv_state", "score_state"), _state(c, S)))
            if c.indexed:
                ihd = c.index_head_dim
                st |= dict(zip(("i_state", "i_score_state"), _state(c, S, width=ihd)))
                st["i_cache"] = torch.zeros(S, c.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
            st_all.append(st)
        return st_all

    k_states, states = fresh_states(), fresh_states()
    for lay, st in zip(layers, k_states):
        for k, v in st.items():
            setattr(lay, {"i_state": "i_kv_state"}.get(k, k), v)

    # each layer vs the golden fed this layer's kernel input: one routing flip would dominate a chained run
    for pos in range(3 * COMPRESS_CSA):
        h = (0.5 * torch.randn(S, base.hc_mult, base.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
        hk = h
        for i, (c, lay) in enumerate(zip(cfgs, layers)):
            cos, sin = tables[c.rope_base]
            idx, dest = contiguous_pool([pos] * S, c, dev)
            h_in = hk
            hk = lay.forward(h_in, cur, kvs_k[i], dest, idx, cos, sin, layer=i, advance=False)
            torch.cuda.synchronize()
            got = lay.intermediates()
            kw = dict(states[i])
            if c.compress_ratio:
                kw |= dict(cos_c=cos, sin_c=sin)
            ref = golden_layer(Ws[i], h_in, [pos] * S, kvs_r[i], dest, idx, cos, sin, lambda z: z, moe_mode=mode, **kw)
            if _routing_flipped(got, ref, Ws[i], c, S):
                ref = _rebase_on_own_routing(got, ref, Ws[i], mode)
            rel, tol = _rel_l2(hk, ref["x_out"]), _tol(OUT_REL_L2, c.hc_mult, 1)
            assert rel < tol, f"pos={pos} layer {i} (ratio {c.compress_ratio}): rel_l2 {rel:.4f} >= {tol}"
        for v in variants.values():
            v.advance_step()
    if kv_fp8:
        for i, c in enumerate(cfgs):
            rel = _rel_l2(decode_kv_fp8(*kvs_k[i]), kvs_r[i])
            assert rel < 1e-2, f"layer {i} (ratio {c.compress_ratio}): fp8 KV rows off the golden's, rel_l2 {rel:.4f}"
    for v in variants.values():
        v.close()
