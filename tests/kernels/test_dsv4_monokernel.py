# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the fused DeepSeek-V4 layer kernel against its torch golden, stage by stage.

Covers the sliding-window, HCA and CSA layers of :mod:`kernels.monokernel.dsv4.kernel`.
The reduced shard keeps ``head_dim`` 512 and ``head_dim - rope_dim`` a multiple of 64,
which the kernel's mappings depend on.

Multi-GPU and benchmark runs::

    python3 tests/kernels/test_dsv4_monokernel.py --npes 8
    python3 tests/kernels/test_dsv4_monokernel.py --bench --npes 8 --real
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


def _tol(base, hc_mult, npes):
    """Double the bar for mHC mixing and for multi-rank bf16 partial sums, as measured."""
    return base * (2 if hc_mult > 1 else 1) * (2 if npes > 1 else 1)


def _cfg(hc_mult=1):
    return V4Config(
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
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        if name in L2_STAGES:
            rel = ((a - b).norm() / b.norm().clamp(min=1e-6)).item()
        else:
            rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"


@pytest.mark.parametrize("moe_mode", [MoeMode.A8W4, MoeMode.W8A8])
@pytest.mark.parametrize("hc_mult", [1, 4])
@pytest.mark.parametrize("S", [1, 2, 4, 8])
def test_dsv4_layer_matches_golden(S, moe_mode, hc_mult):
    """Every stage vs the golden; S > 1 is independent sequences, hc_mult=4 the hyper-connection stream."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    cfg = _cfg(hc_mult)
    cfg.validate()
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=moe_mode)

    hshape = (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden)
    h = (0.5 * torch.randn(*hshape, device=dev)).bfloat16()
    pos = cfg.window  # ring already wrapped once
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()

    kv_ref = kv0.clone()
    ref = golden_layer(W, h, [pos] * S, kv_ref, dest, idx, cos, sin, lambda z: z, moe_mode=moe_mode)

    if _routing_flipped(got, ref, W, cfg, S):
        ref = _rebase_on_own_routing(got, ref, W, moe_mode)
    _compare_stages(got, ref, cfg, 1)
    # relative L2: a lone FP8 rounding flip is one element, a clipped block max (E8M0 rounded down) ~5%
    xq_rel = ((got["xq"].float() - ref["xq"].float()).norm() / ref["xq"].float().norm()).item()
    assert xq_rel < 0.01, f"quantized expert input diverged: rel_l2 {xq_rel:.5f}"
    assert out.shape == h.shape, f"the layer must preserve its input shape, got {out.shape}"

    # end to end by relative L2: one upstream rounding flip moves an element far, hc_post spreads it
    a_out, b_out = out.float(), ref["x_out"].float()
    rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol} (rel_max {rel_max:.5f})"


@pytest.mark.large_shape
@pytest.mark.parametrize("S,hc_mult", [(1, 1), (8, 1), (1, 4), (8, 4)])
def test_dsv4_layer_matches_golden_at_real_dims(S, hc_mult):
    """V4-Pro TP8 dims: mappings only real sizes exercise (384 expert ids, two ug tiles per CTA at S=8)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    cfg = V4Config(hc_mult=hc_mult)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.validate()
    dev, mode = "cuda", MoeMode.A8W4
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)

    hshape = (S, cfg.hidden) if hc_mult == 1 else (S, hc_mult, cfg.hidden)
    h = (0.5 * torch.randn(*hshape, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, lambda z: z, moe_mode=mode)

    if _routing_flipped(got, ref, W, cfg, S):
        ref = _rebase_on_own_routing(got, ref, W, mode)
    _compare_stages(got, ref, cfg, 1)
    # an id above 255 needs more than an 8-bit key id field
    assert max(got["sel"].reshape(-1).tolist()[1:]) > 255 or cfg.n_experts <= 256

    a_out, b_out = out.float(), ref["x_out"].float()
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol}"


def test_dsv4_csa_shape_is_the_selected_one():
    """CSA's gather is sized for index_topk selected slots, not the whole compressed half."""
    from kernels.monokernel.dsv4.config import COMPRESS_CSA, validate_shard

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


def test_dsv4_rejects_a_bf16_router_bias():
    """A bf16 bias (from a bf16 default dtype) would make the kernel read past its end: a memory fault."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    cfg = _cfg(hc_mult=1)
    W = make_weights(rank=0, cfg=cfg, device="cuda", seed=3, moe_mode=MoeMode.A8W4)
    W.t["bias"] = W.t["bias"].bfloat16()
    with pytest.raises(ValueError, match="router bias must be float32"):
        Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=MoeMode.A8W4)


def test_dsv4_layout_sizes_moe_mailboxes_from_the_build_dims():
    """scores / sel / prob / mid follow the build's n_experts, top_k and inter, not the module defaults."""
    from kernels.monokernel.dsv4.kernel import layout, stage_tasks

    S, ne, k, inter = 2, 1024, 8, 768
    sc, _ = layout(S, 16, 1, n_experts=ne, top_k=k, inter=inter)
    names = list(sc)
    size = {n: sc[names[i + 1]] - sc[n] for i, n in enumerate(names[:-1])}
    assert size["scores"] >= S * ne * 8
    assert size["sel"] >= S * (1 + k) * 8 and size["prob"] >= S * (1 + k) * 8
    assert size["mid"] >= S * (1 + k) * inter * 8
    assert dict(stage_tasks(S, 16, n_experts=ne))["router"] * 8 * 2 >= S * ne  # 8 experts, <= 2 samples a task


def test_dsv4_rejects_int64_index_tensors():
    """Index tensors are read as int32 by raw loads: an int64 one would address the wrong rows."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    cfg = _cfg(hc_mult=1)
    W = make_weights(rank=0, cfg=cfg, device="cuda", seed=3, moe_mode=MoeMode.A8W4)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=MoeMode.A8W4)
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


def test_dsv4_rejects_head_dim_that_would_deadlock():
    """A head_dim below the PV MFMA's per-wave grouping would hang on an unfillable mailbox."""
    from kernels.monokernel.dsv4.kernel import build_dsv4_kernel

    with pytest.raises(AssertionError, match="head_dim"):
        build_dsv4_kernel(S=1, heads=8, npes=1, head_dim=128)


def test_dsv4_bounded_poll_flags_instead_of_hanging():
    """Timeout 0 flags the launch and still terminates; the default timeout never fires on a healthy launch."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel, Dsv4Variant

    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=MoeMode.A8W4)
    h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
    cur = torch.tensor([cfg.window], dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool([cfg.window], cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    kv = (0.3 * torch.randn(cfg.window, cfg.head_dim, device=dev)).bfloat16()

    from kernels.monokernel.dsv4.kernel import POLL_TIMEOUT_US

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
    from kernels.monokernel.dsv4.kernel import LAYER_SLOTS, scrub_period
    from kernels.monokernel.dsv4.op import Dsv4Variant

    cfg = _cfg(hc_mult=1)
    variant = Dsv4Variant(cfg, 1, rank=0, npes=1, moe_mode=MoeMode.A8W4)
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
        stale = (w[:, 1].unsqueeze(1) == kinds[: len(old)]).any(1)
        w[stale] = 0
    variant.step.fill_(_i32(s0))
    for _ in range(period):
        variant.advance_step()
    torch.cuda.synchronize()
    assert variant.step.item() == _i32(s0 + period)
    for name, pr, w in zip(["scratch", "sym"], pairs, want):
        bad = (pr != w).any(1).nonzero()
        assert (
            bad.numel() == 0
        ), f"{name}: {bad.numel()} pairs wrong, first {pr[bad[0, 0]].tolist()} want {w[bad[0, 0]].tolist()}"
    variant.close()


def test_dsv4_layer_output_is_the_same_across_the_tag_wrap():
    """Launches whose tags straddle the 2**32 wrap give bit-identical outputs to steps 0 and 1."""
    from kernels.monokernel.dsv4.kernel import LAYER_SLOTS
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=MoeMode.A8W4)
    h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
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


def _tp_cfg(real: bool, hc_mult: int = 1, compress_ratio: int = 0, max_seq: int | None = None):
    # not a module global: mp.spawn re-imports this module in each child
    cfg = V4Config(hc_mult=hc_mult) if real else _cfg(hc_mult)
    if compress_ratio:
        # max_seq sets the compressed shape: a benchmark must pass the length it reports
        cfg.compress_ratio = compress_ratio
        cfg.max_seq = 256 if max_seq is None else max_seq
    return cfg


def run_rank(rank, npes, real=False, iters=2, group=None, moe_mode=MoeMode.A8W4, hc_mult=1, compress_ratio=0):
    import torch.distributed as dist

    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real, hc_mult, compress_ratio)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=rank, npes=npes, group=group, moe_mode=moe_mode)
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
        ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
        ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    if npes == 1:
        allreduce = lambda x: x  # noqa: E731
    else:

        def allreduce(x):
            # model peer_reduce: bf16-rounded partials summed in rank order
            x = x.to(torch.bfloat16).float()
            parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
            dist.all_gather(parts, x.cpu().contiguous(), group=group)
            return sum(parts[1:], parts[0]).to(x.device)

    ok = True
    boundaries = 0
    for it in range(iters):
        hshape = (1, cfg.hidden) if cfg.hc_mult == 1 else (1, cfg.hc_mult, cfg.hidden)
        h = torch.randn(*hshape, generator=gen, device=dev).to(torch.bfloat16)
        if compress_ratio:
            pos = it
            idx, dest = contiguous_pool([pos], cfg, dev)
            boundaries += (pos + 1) % compress_ratio == 0
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        kv_in = kv_k if compress_ratio else kv0.clone()
        out = layer.forward(h, cur, kv_in, dest, idx, cos, sin)
        torch.cuda.synchronize()
        got = layer.intermediates()

        if npes > 1:
            peers = [torch.empty_like(out.cpu()) for _ in range(npes)]
            dist.all_gather(peers, out.cpu().contiguous(), group=group)
            for other in peers[1:]:
                torch.testing.assert_close(other, peers[0], atol=0, rtol=0)
            sels = [torch.empty_like(got["sel"].cpu()) for _ in range(npes)]
            dist.all_gather(sels, got["sel"].cpu().contiguous(), group=group)
            for other in sels[1:]:
                torch.testing.assert_close(other, sels[0], atol=0, rtol=0)

        gkw = dict(kv_state=ks, score_state=ss, cos_c=cos, sin_c=sin) if compress_ratio else {}
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r if compress_ratio else kv0.clone(),
            dest,
            idx,
            cos,
            sin,
            allreduce,
            moe_mode=moe_mode,
            **gkw,
        )
        if compress_ratio and (pos + 1) % compress_ratio == 0:
            slot = cfg.window + pos // compress_ratio
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            c_rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            if c_rel >= 2e-2:
                print(f"rank {rank}: compressed row at {slot} rel {c_rel:.5f}", flush=True)
                ok = False
        # a near-tie can flip routing; then judge the expert math against the kernel's own routing
        flipped = got["sel"].tolist() != ref["sel"].tolist()
        for name, base in STAGE_TOL.items():
            tol = _tol(base, cfg.hc_mult, npes) if name in SCALES_WITH_CONFIG else base
            if flipped and name in SCALES_WITH_CONFIG:
                continue
            a = got[name].float().reshape(-1)
            b = ref[name].float().reshape(-1)
            rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
            l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
            frac = ((a - b).abs() > 1e-3 * max(b.abs().max().item(), 1e-6)).float().mean().item()
            if rel >= tol:
                print(
                    f"rank {rank}: stage {name} rel_max {rel:.5f} rel_l2 {l2:.5f} " f"elems_off {frac * 100:.2f}%",
                    flush=True,
                )
                ok = False
        if flipped:
            print(f"rank {rank}: routing flipped on a near-tie (expected; judging by own routing)", flush=True)
        down = golden_moe(
            W,
            got["a"],
            allreduce,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=moe_mode,
        )
        d_out = down["x_out"].float()
        d_rel = ((out.float() - d_out).norm() / d_out.norm()).item()
        out_tol = _tol(OUT_REL_L2, cfg.hc_mult, npes)
        if d_rel >= out_tol:
            print(f"rank {rank}: x_out vs own-routing golden rel_l2 {d_rel:.5f}", flush=True)
            ok = False
        a_out, b_out = out.float(), ref["x_out"].float()
        rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
        rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
        if rank == 0:  # pytest captures this; it is what makes a near-miss legible
            print(f"rank {rank}: x_out rel_max {rel_max:.5f}  rel_l2 {rel_l2:.5f}", flush=True)
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


def _worker(rank, npes, real, iters, hc_mult, compress_ratio, port, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = run_rank(rank, npes, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    finally:
        dist.destroy_process_group()


def run_tp(npes, real=False, iters=2, hc_mult=1, compress_ratio=0):
    if npes == 1:
        return run_rank(0, 1, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(npes, real, iters, hc_mult, compress_ratio, _free_port(), results),
        nprocs=npes,
    )
    return all(results[r] for r in range(npes))


@pytest.mark.multi_gpu
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_dsv4_layer_tp8(hc_mult):
    """Both residual widths; the tolerances are calibrated per configuration."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    assert run_tp(8, hc_mult=hc_mult)


@pytest.mark.multi_gpu
def test_dsv4_hca_layer_tp8():
    """HCA across ranks over two compression boundaries; the replicated compressor must agree on every rank."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    assert run_tp(8, iters=20, compress_ratio=8)


# Benchmark: HIP-graph replay of BENCH_LAYERS launches, so the number excludes launch overhead.

BENCH_LAYERS = 16


def bench_rank(
    rank,
    npes,
    real=True,
    iters=320,
    group=None,
    timeline=False,
    moe_mode=MoeMode.A8W4,
    hc_mult=1,
    compress_ratio=0,
    max_seq=None,
    samples=1,
):
    """Returns (us per layer, cfg); ``compress_ratio`` 0/128/4 is sliding-window/HCA/CSA."""
    import torch.distributed as dist

    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real, hc_mult, compress_ratio, max_seq)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    cos, sin = rope_table(max(4096, cfg.max_seq), theta=cfg.rope_base, device=dev)
    kv = torch.randn(samples * cfg.cache_rows, cfg.head_dim, device=dev).to(torch.bfloat16)
    # a full compressed cache: the steady state of a decode
    pos = (cfg.max_seq - 1) if compress_ratio else cfg.window
    idx, dest = contiguous_pool([pos] * samples, cfg, dev)
    cur = torch.tensor([pos] * samples, dtype=torch.int32, device=dev)
    op = Dsv4MonoKernel(W, samples, rank=rank, npes=npes, group=group, moe_mode=moe_mode)
    hshape = (samples, cfg.hidden) if cfg.hc_mult == 1 else (samples, cfg.hc_mult, cfg.hidden)
    h = torch.randn(*hshape, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    for _ in range(10):
        op.forward(h, cur, kv, dest, idx, cos, sin, x_out=x)
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()

    if timeline:
        top = Dsv4MonoKernel(
            W,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            timeline=True,
            moe_mode=moe_mode,
        )
        for _ in range(3):
            top.forward(h, cur, kv, dest, idx, cos, sin, x_out=x)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
        top.close()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(BENCH_LAYERS):
            op.forward(h, cur, kv, dest, idx, cos, sin, x_out=x, layer=layer, advance=False)
        op.advance_step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // BENCH_LAYERS):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / (iters // BENCH_LAYERS * BENCH_LAYERS)
    op.close()
    return us, dict(n_keys=cfg.n_keys, n_comp=cfg.n_compressed, max_seq=cfg.max_seq, samples=samples)


def _bench_worker(rank, npes, real, timeline, moe_mode, hc_mult, compress_ratio, max_seq, samples, port, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = bench_rank(
            rank,
            npes,
            real=real,
            timeline=timeline,
            moe_mode=moe_mode,
            hc_mult=hc_mult,
            compress_ratio=compress_ratio,
            max_seq=max_seq,
            samples=samples,
        )
    finally:
        dist.destroy_process_group()


def run_bench(
    npes, real=True, timeline=False, moe_mode=MoeMode.A8W4, hc_mult=1, compress_ratio=0, max_seq=None, samples=1
):
    kw = dict(
        real=real,
        timeline=timeline,
        moe_mode=moe_mode,
        hc_mult=hc_mult,
        compress_ratio=compress_ratio,
        max_seq=max_seq,
        samples=samples,
    )
    if npes == 1:
        return {0: bench_rank(0, 1, **kw)}
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(
        _bench_worker,
        args=(npes, real, timeline, moe_mode, hc_mult, compress_ratio, max_seq, samples, _free_port(), results),
        nprocs=npes,
    )
    return dict(results)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=8)
    ap.add_argument("--real", action="store_true", help="use the real V4-Pro TP8 shard")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--moe-mode", default=MoeMode.A8W4.value, choices=tuple(m.value for m in MoeMode))
    ap.add_argument("--hc-mult", type=int, default=1, help="1 = plain residual, 4 = hyper-connections")
    a = ap.parse_args()
    if a.bench:
        res = run_bench(a.npes, a.real, a.timeline, MoeMode(a.moe_mode), a.hc_mult)
        us = [res[r][0] for r in sorted(res)]
        tag = "real V4-Pro" if a.real else "reduced"
        print(
            f"{tag} shard, {a.moe_mode}, hc={a.hc_mult}, npes={a.npes}: {max(us):7.1f} us/layer  (per rank: "
            + " ".join(f"{v:.1f}" for v in us)
            + ")"
        )
    else:
        print("PASS" if run_tp(a.npes, a.real, a.iters, a.hc_mult) else "FAIL")


@pytest.mark.parametrize("moe_mode", [MoeMode.W8A8, MoeMode.A8W4])
def test_dsv4_hca_layer_matches_golden(moe_mode):
    """The HCA layer over several compression boundaries; ratio 16 is the same code path as V4's 128."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio = 16
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio = ratio
    cfg.max_seq = 512
    cfg.validate()
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=moe_mode)

    # one rope table per layer (cfg.rope_base): window q/kv and compressed rows share it
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    # -inf like the layer's own state: unwritten rows must drop out of the softmax
    ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    boundaries = 0
    for pos in range(3 * ratio + 2):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        out = layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=moe_mode,
            kv_state=ks,
            score_state=ss,
            cos_c=cos,
            sin_c=sin,
        )
        if (pos + 1) % ratio == 0:
            boundaries += 1
            slot = cfg.window + pos // ratio
            # not bit-exact: online softmax on hardware exp2 vs a batch softmax, rounded to bf16
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            assert rel < 2e-2, f"compressed row at slot {slot} differs by rel {rel:.5f}"

        # the x_out check below uses the kernel's own `a`, so check attention here
        got = layer.intermediates()
        rel_o = (got["o"].float() - ref["o"].float()).abs().max().item() / max(
            ref["o"].float().abs().max().item(), 1e-6
        )
        assert rel_o < STAGE_TOL["o"], f"pos={pos} stage o diverged: rel {rel_o:.5f}"

        # a long run eventually flips a near-tied expert: judge against the kernel's own routing
        own = golden_moe(
            W,
            got["a"],
            lambda z: z,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=moe_mode,
        )
        b_out = own["x_out"].float()
        rel_l2 = ((out.float() - b_out).norm() / b_out.norm()).item()
        tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
        assert rel_l2 < tol, f"pos={pos} x_out rel_l2 {rel_l2:.5f} >= {tol}"

    assert boundaries >= 3, "must cross several compression boundaries"


def test_dsv4_hca_merges_only_each_samples_live_splits():
    """Two samples far apart: each merge must read only its own live splits (a wrong count moves `o`)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio, S = 16, 2
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 4096
    cfg.validate()
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    pos = [300 + ratio // 2, 3000 + ratio // 2]  # off the compression boundaries
    n_split = cfg.n_keys // 64
    live = [(cfg.window + (p + 1) // ratio + 63) // 64 for p in pos]
    assert live[0] < live[1] < n_split, f"live splits {live} of {n_split}: the shape no longer has dead ones"
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    cur = torch.tensor(pos, dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool(pos, cfg, dev)
    ks = torch.zeros(S, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((S, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(
        W,
        h,
        pos,
        kv0.clone(),
        dest,
        idx,
        cos,
        sin,
        lambda z: z,
        moe_mode=mode,
        kv_state=ks,
        score_state=ss,
        cos_c=cos,
        sin_c=sin,
    )
    for s_ in range(S):
        a, b = got["o"][s_].float(), ref["o"][s_].float()
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < STAGE_TOL["o"], f"sample {s_} (pos {pos[s_]}, {live[s_]} live splits): o rel {rel:.5f}"
    layer.close()


def test_dsv4_hca_folds_tile_groups_at_long_context():
    """Past one grid round (S * live tiles > 256) a split task folds several 64-key tiles; checks `o` per sample."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio, S = 16, 8
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 65536
    cfg.validate()
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    pos = [30000 + 3700 * i + ratio // 2 for i in range(S)]  # off the compression boundaries
    live = [(cfg.window + (p + 1) // ratio + 63) // 64 for p in pos]
    assert S * max(live) > 256, f"live tiles {live}: one round of the grid, so no task folds tiles"
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    cur = torch.tensor(pos, dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool(pos, cfg, dev)
    ks = torch.zeros(S, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((S, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(
        W,
        h,
        pos,
        kv0.clone(),
        dest,
        idx,
        cos,
        sin,
        lambda z: z,
        moe_mode=mode,
        kv_state=ks,
        score_state=ss,
        cos_c=cos,
        sin_c=sin,
    )
    for s_ in range(S):
        a, b = got["o"][s_].float(), ref["o"][s_].float()
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < STAGE_TOL["o"], f"sample {s_} (pos {pos[s_]}, {live[s_]} live tiles): o rel {rel:.5f}"
    layer.close()


@pytest.mark.large_shape
def test_dsv4_hca_layer_at_real_dims():
    """HCA at V4-Pro's real dims: ratio 128 pooling, the full ``ape`` table and the real ``n_keys``."""
    from kernels.monokernel.dsv4.config import COMPRESS_HCA
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    cfg = V4Config(hc_mult=1)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.compress_ratio = COMPRESS_HCA
    cfg.validate()
    ratio, dev, mode = cfg.compress_ratio, "cuda", MoeMode.W8A8
    assert cfg.n_keys > cfg.window, "the gather must reach past the window"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    # two boundaries: the first anchors at pos 0, where rope cannot tell the compressor's base apart
    boundaries, checked, top_id = 0, 0, 0
    for pos in range(2 * ratio + 4):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        out = layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=mode,
            kv_state=ks,
            score_state=ss,
            cos_c=cos,
            sin_c=sin,
        )
        if (pos + 1) % ratio == 0:
            boundaries += 1
            slot = cfg.window + pos // ratio
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            assert rel < 2e-2, f"compressed row at slot {slot} differs by rel {rel:.5f}"

        got = layer.intermediates()
        # stage-by-stage only around the boundaries
        if any(abs(pos - (b * ratio - 1)) <= 2 for b in (1, 2)):
            checked += 1
            # skip only `mid` on a routing flip
            same_route = got["sel"].tolist() == ref["sel"].tolist()
            for name, base in STAGE_TOL.items():
                if name == "mid" and not same_route:
                    continue
                tol = _tol(base, cfg.hc_mult, 1) if name in SCALES_WITH_CONFIG else base
                a = got[name].float().reshape(-1)
                b = ref[name].float().reshape(-1)
                rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
                assert rel < tol, f"pos={pos} stage {name} diverged: rel {rel:.5f} >= {tol}"
        top_id = max(top_id, *got["sel"].reshape(-1).tolist()[1:])

        # flip-immune: judge against the kernel's own routing
        own = golden_moe(
            W,
            got["a"],
            lambda z: z,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=mode,
        )
        b_out = own["x_out"].float()
        rel_l2 = ((out.float() - b_out).norm() / b_out.norm()).item()
        tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
        assert rel_l2 < tol, f"pos={pos} x_out rel_l2 {rel_l2:.5f} >= {tol}"

    assert boundaries == 2 and checked == 10
    # an id above 255 needs more than an 8-bit key id field
    assert top_id > 255 or cfg.n_experts <= 256


def test_dsv4_csa_compressor_in_kernel():
    """CSA's overlapping compressor in the kernel: each compressed row vs ``compress_step``."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import compress_step, qkv_a_split

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    assert cfg.overlap and cfg.c_coff == 2, "ratio 4 is the overlapping form"
    dev, mode = "cuda", MoeMode.W8A8
    W, t = None, None
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)
    cache_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq = qkv_a_matrix(t)
    cut = qkv_a_split(cfg)

    emitted = 0
    for pos in range(5 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq.float().T
        ref = compress_step(
            proj[0, slice(*cut["c_kv"])],
            proj[0, slice(*cut["c_gate"])],
            pos,
            cfg,
            t,
            ks,
            ss,
            cache_r,
            cos,
            sin,
            dest_row=cfg.window + pos // ratio,
        )
        if (pos + 1) % ratio:
            assert ref is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        slot = cfg.window + pos // ratio
        a, b = kv_k[slot].float(), cache_r[slot].float()
        # a last-bit fp32 difference can push one element across an FP8 code (~9%):
        # bound the count and the bulk; a wrong overlap moves most of the row
        n_diff = int((a != b).sum())
        rel_l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert n_diff <= 4, f"pos={pos}: {n_diff} elements differ, not a quantization tie"
        assert rel_l2 < 5e-3, f"pos={pos} compressed row rel_l2 {rel_l2:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


@pytest.mark.parametrize("indexer_hadamard", [True, False])
def test_dsv4_indexer_compressor_in_kernel(indexer_hadamard):
    """The indexer's compressor (Hadamard + FP4) in the kernel, against ``compress_step(rotate=True)``."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import compress_step, qkv_a_split

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.indexer_hadamard = indexer_hadamard  # ATOM's indexer rotates neither side
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    assert cfg.indexed, "only CSA runs an indexer"
    ihd, dev, mode = cfg.index_head_dim, "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
    dq = qkv_a_matrix(t)
    cut = qkv_a_split(cfg)

    emitted = 0
    for pos in range(5 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq.float().T
        ref = compress_step(
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
            head_dim=ihd,
            ape=t["i_ape"],
            gamma=t["g_ickv"],
            rotate=True,
        )
        if (pos + 1) % ratio:
            assert ref is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        slot = pos // ratio
        a, b = unpack_fp4(_icache_rows(layer, 0, slot + 1)[slot]), unpack_fp4(i_ref[slot])
        # coarse FP4 levels: bound the count and the bulk, not the max
        n_diff = int((a != b).sum())
        rel_l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert n_diff <= 4, f"pos={pos}: {n_diff}/{ihd} elements differ, not a quantization tie"
        assert rel_l2 < 1e-2, f"pos={pos} indexer compressed row rel_l2 {rel_l2:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


@pytest.mark.parametrize("indexer_hadamard", [True, False])
def test_dsv4_indexer_query_in_kernel(indexer_hadamard):
    """The indexer's query path in the kernel: projection, RoPE, Hadamard, FP4 (no per-head RMS)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import (
        hadamard,
        quant_dequant_fp4,
        rope,
    )

    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)
    cfg.indexer_hadamard = indexer_hadamard  # ATOM's indexer rotates neither side
    cfg.compress_ratio, cfg.max_seq = COMPRESS_CSA, 256
    ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq_qkv = qkv_a_matrix(t)
    dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)

    for pos in range(3):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        q_a = (x @ dq_qkv.float().T)[:, : cfg.q_lora]
        q_an = bf(rmsnorm(q_a, t["g_q"], cfg.eps))
        q = (q_an @ dq_iqb.T).view(ih, ihd)
        q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
        ref = quant_dequant_fp4(bf(hadamard(bf(q))) if cfg.indexer_hadamard else bf(q))

        got = layer.debug("i_q", (1, ih, ihd))[0]
        # one FP4 step is ~0.5 here: bound the rate of differing elements, a broken rotation moves most
        n_diff = int((got != ref).sum())
        rel_l2 = ((got - ref).norm() / max(ref.norm().item(), 1e-6)).item()
        assert n_diff <= ih * ihd // 50, f"pos={pos}: {n_diff}/{ih * ihd} elements differ"
        assert rel_l2 < 6e-2, f"pos={pos} indexer query rel_l2 {rel_l2:.5f}"


@pytest.mark.parametrize("indexer_hadamard", [True, False])
def test_dsv4_indexer_scoring_in_kernel(indexer_hadamard):
    """The indexer score sum_h relu(q[h] . k[c]) * w[h] per written entry; unwritten entries score NEG."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import (
        compress_step,
        hadamard,
        qkv_a_split,
        quant_dequant_fp4,
        rope,
    )

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.indexer_hadamard = indexer_hadamard  # ATOM's indexer rotates neither side
    # up to index_topk live entries nothing is scored, so a small k scores from the third entry on
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 2
    ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
    dq_qkv = qkv_a_matrix(t)
    dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)
    cut = qkv_a_split(cfg)
    scale = ihd**-0.5 * cfg.index_heads**-0.5

    scored = 0
    for pos in range(4 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        compress_step(
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
            head_dim=ihd,
            ape=t["i_ape"],
            gamma=t["g_ickv"],
            rotate=True,
        )
        q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
        q = (q_an @ dq_iqb.T).view(ih, ihd)
        q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
        q = quant_dequant_fp4(bf(hadamard(bf(q))) if cfg.indexer_hadamard else bf(q))
        w = bf(x @ t["i_w"].float().T)[0] * scale

        n = (pos + 1) // ratio
        got = layer.debug("i_score", (1, cfg.n_compressed))[0]
        if n <= cfg.index_topk:  # every live entry kept, nothing scored
            continue
        scored += 1
        ref = (torch.einsum("hd,td->ht", q, unpack_fp4(i_ref[:n])).relu() * w.view(ih, 1)).sum(0)
        a, b = got[:n], ref
        rel = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert rel < 2e-2, f"pos={pos} score rel_l2 {rel:.5f}"
        assert bool((got[n:] < 0).all()), f"pos={pos}: unwritten entries are scorable"

    assert scored >= 3, f"expected several scored steps, got {scored}"


def _indexer_score_rank(rank, npes, port, results):
    """One rank of the replicated indexer's scoring; see the test below."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        from kernels.monokernel.dsv4.op import Dsv4MonoKernel
        from kernels.monokernel.dsv4.reference import (
            dequant,
            hadamard,
            quant_dequant_fp4,
            rope,
        )

        dev = torch.device("cuda", rank)
        torch.cuda.set_device(dev)
        ratio = COMPRESS_CSA
        cfg = _cfg(hc_mult=1)
        cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 2
        ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
        mode = MoeMode.W8A8
        W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=mode)
        t = W.t
        layer = Dsv4MonoKernel(W, samples=1, rank=rank, npes=npes, moe_mode=mode)
        cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
        kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        dq_qkv = qkv_a_matrix(t)
        dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)
        scale = ihd**-0.5 * cfg.index_heads**-0.5
        gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical everywhere

        ok, scored = True, 0
        for pos in range(4 * ratio):  # entries 3 and 4 pass index_topk and are scored
            h = torch.randn(1, cfg.hidden, generator=gen, device=dev).to(torch.bfloat16)
            cur = torch.tensor([pos], dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool([pos], cfg, dev)
            layer.forward(h, cur, kv_k, dest, idx, cos, sin)
            torch.cuda.synchronize()
            got = layer.debug("i_score", (1, cfg.n_compressed))[0]

            # bit-identical across ranks, or the top-k would pick different keys per rank
            peers = [torch.empty_like(got.cpu()) for _ in range(npes)]
            dist.all_gather(peers, got.cpu().contiguous())
            for other in peers[1:]:
                torch.testing.assert_close(other, peers[0], atol=0, rtol=0)

            x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
            proj = x @ dq_qkv.float().T
            q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
            q = (q_an @ dq_iqb.T).view(ih, ihd)
            q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
            q_ref = quant_dequant_fp4(bf(hadamard(bf(q))) if cfg.indexer_hadamard else bf(q))
            w_ref = bf(x @ t["i_w"].float().T)[0] * scale
            # score from the kernel's own q, w and key cache (each checked on its own bar),
            # so only the dot product and head sum are under test
            q = layer.debug("i_q", (1, ih, ihd))[0]
            w = layer.debug("i_wp", (1, ih))[0]
            dq_n = int((q != q_ref).sum())
            dq_l2 = ((q - q_ref).norm() / max(q_ref.norm().item(), 1e-6)).item()
            dw = (w - w_ref).abs().max().item() / max(w_ref.abs().max().item(), 1e-6)
            if dq_n > ih * ihd // 50 or dq_l2 >= 6e-2 or dw >= 1e-2:
                print(
                    f"rank {rank} pos={pos} q differs on {dq_n} (l2 {dq_l2:.5f}), " f"w rel {dw:.5f}",
                    flush=True,
                )
                ok = False
            n = (pos + 1) // ratio
            if n <= cfg.index_topk:  # every live entry kept, nothing scored
                continue
            scored += 1
            kcache = unpack_fp4(_icache_rows(layer, 0, n))
            ref = (torch.einsum("hd,td->ht", q, kcache).relu() * w.view(ih, 1)).sum(0)
            rel = ((got[:n] - ref).norm() / max(ref.norm().item(), 1e-6)).item()
            if rel >= 1e-3:
                print(f"rank {rank} pos={pos} score rel_l2 {rel:.5f}", flush=True)
                ok = False
        layer.close()
        results[rank] = ok and scored >= 2
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_dsv4_indexer_scores_match_on_every_rank_tp8():
    """Every rank scores with all index heads: the scores match the golden and are bit-identical across ranks."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_indexer_score_rank, args=(8, _free_port(), results), nprocs=8)
    assert all(results[r] for r in range(8))


def test_dsv4_indexer_scores_the_new_entry_past_the_first_tile():
    """The entry written this launch is scored from the mailbox even past the first SCORE_TILE (poisoned cache)."""
    from kernels.monokernel.dsv4.kernel import SCORE_TILE
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 4096, 64
    ih, ihd = cfg.index_heads, cfg.index_head_dim
    assert cfg.n_compressed > SCORE_TILE + 4, "the shape must reach the second score tile"
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)
    layer.i_cache.copy_(torch.randint(0, 256, layer.i_cache.shape, device=dev))
    layer.i_cache_s.copy_(torch.randint(0, 256, layer.i_cache_s.shape, device=dev))
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    # the last entry of tile 0 as a control, then the first four of tile 1
    checks = [ratio * (SCORE_TILE + j) - 1 for j in range(5)]
    for pos in range(checks[-1] + 1):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        new = pos // ratio
        q = layer.debug("i_q", (1, ih, ihd))[0]
        w = layer.debug("i_wp", (1, ih))[0]
        got = layer.debug("i_score", (1, cfg.n_compressed))[0][new].item()
        ref = ((q @ unpack_fp4(_icache_rows(layer, 0, new + 1)[new])).relu() * w).sum().item()
        assert abs(got - ref) <= 1e-3 * max(abs(ref), 1e-3), f"pos={pos} entry {new}: score {got} vs {ref}"
    layer.close()


def test_dsv4_indexer_topk_in_kernel():
    """The indexer's selected set is exactly the top-k of its scores (index_topk reduced so it discards)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import indexer_step

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 4
    ihd, dev, mode = cfg.index_head_dim, "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev)
    dq_qkv = qkv_a_matrix(t)
    cut = qkv_a_split(cfg)

    chose, discarded = 0, 0
    for pos in range(8 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
        ref = indexer_step(
            x[0],
            q_an[0],
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
        )
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]

        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        assert int((got >= 0).sum()) == k, f"pos={pos}: picked {int((got >= 0).sum())}, want {k}"
        if not k:
            continue
        chose += 1
        discarded += n > cfg.index_topk
        # judge against the kernel's own scores: golden FP4 ties can move a borderline entry
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        want = set((cfg.window + sc.topk(k).indices).tolist())
        a = set(got[got >= 0].tolist())
        margin = (sc.sort(descending=True).values[k - 1] - sc.sort(descending=True).values[k]).item() if n > k else 1.0
        if a != want and margin > 1e-6:
            raise AssertionError(f"pos={pos} picked {sorted(a)} vs {sorted(want)} (margin {margin:.3e})")
        assert len(set(ref[ref >= 0].tolist())) == k

    assert chose >= 6 and discarded >= 3, f"chose {chose}, discarded on {discarded}"


def test_dsv4_indexer_topk_compaction_spans_waves():
    """The top-k compaction writes one slot per pick when picks span all eight waves (cross-wave scan)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 2048, 200
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)
    cos, sin = rope_table(4096, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    checks = [400, 700, 1100]
    waves_hit = 0
    for pos in range(checks[-1] + 1):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
        sel = got[got >= 0].tolist()
        assert len(sel) == k, f"pos={pos}: wrote {len(sel)} slots, want {k}"
        assert len(set(sel)) == k, f"pos={pos}: {k - len(set(sel))} picks collided on a slot"
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        srt = sc.sort(descending=True).values
        margin = (srt[k - 1] - srt[k]).item() if n > k else 1.0
        want = set((cfg.window + sc.topk(k).indices).tolist())
        if set(sel) != want and margin > 1e-6:
            raise AssertionError(f"pos={pos}: picked {len(set(sel) - want)} entries the scores do not rank")
        waves_hit = max(waves_hit, (max(s - cfg.window for s in sel) // 64) + 1)
    # the point of the shape: without this the cross-wave term is never read
    assert waves_hit >= 5, f"picks only reached wave {waves_hit}, so the scan is still untested"
    layer.close()


def test_dsv4_indexer_topk_spans_many_candidates_per_thread():
    """The top-k holds when each thread owns several candidates (the radix's strided per-thread walk)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 8192, 300
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)
    cos, sin = rope_table(8192, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    last = 5999  # 1500 live candidates: rounds 0, 1 and part of 2
    checks = [2500, 4000, last]
    reach = 0
    for pos in range(last + 1):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
        sel = got[got >= 0].tolist()
        assert len(sel) == k, f"pos={pos}: wrote {len(sel)} slots, want {k}"
        assert len(set(sel)) == k, f"pos={pos}: {k - len(set(sel))} picks collided on a slot"
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        srt = sc.sort(descending=True).values
        margin = (srt[k - 1] - srt[k]).item() if n > k else 1.0
        want = set((cfg.window + sc.topk(k).indices).tolist())
        if set(sel) != want and margin > 1e-6:
            raise AssertionError(f"pos={pos}: picked {len(set(sel) - want)} entries the scores do not rank")
        reach = max(reach, max(s - cfg.window for s in sel))
    # the point of the shape: without this only the first round is ever read
    assert reach >= 2 * 512, f"picks stopped at candidate {reach}, so the later rounds are untested"
    layer.close()


@pytest.mark.parametrize("n_live", [3000, 6000])
def test_dsv4_indexer_topk_spans_several_ctas(n_live):
    """The top-k with candidates in one CTA part (3000) and split across parts, one of them empty (6000)."""
    from kernels.monokernel.dsv4.kernel import THREADS, n_topk_parts
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import pack_fp4

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 65536, 300
    parts = n_topk_parts(cfg.max_seq, ratio, cfg.index_head_dim)
    one_part = cfg.n_compressed // parts
    assert parts == 4 and one_part == 4096, "shape no longer splits as the cases assume"
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)
    rows = pack_fp4(torch.randn(cfg.n_compressed, cfg.index_head_dim, device=dev))
    fp4_pool_store(layer.i_cache[0], layer.i_cache_s[0], layer.block_tables[0], rows)
    cos, sin = rope_table(65536, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    pos = n_live * ratio - 1
    h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos], cfg, dev)
    layer.forward(h, torch.tensor([pos], dtype=torch.int32, device=dev), kv_k, dest, idx, cos, sin)
    torch.cuda.synchronize()
    n = (pos + 1) // ratio
    k = min(cfg.index_topk, n)
    got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
    sel = got[got >= 0].tolist()
    assert len(sel) == k, f"wrote {len(sel)} slots, want {k}"
    assert len(set(sel)) == k, f"{k - len(set(sel))} picks collided on a slot"
    sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
    srt = sc.sort(descending=True).values
    want = set((cfg.window + sc.topk(k).indices).tolist())
    if set(sel) != want and (srt[k - 1] - srt[k]).item() > 1e-6:
        raise AssertionError(f"picked {len(set(sel) - want)} entries the scores do not rank")
    reach = max(s_ - cfg.window for s_ in sel)
    if n_live > one_part:
        assert reach >= 2 * THREADS * 4, f"picks stopped at candidate {reach}, so part 2 was not tested"
    layer.close()


def test_dsv4_indexer_topk_at_a_full_1m_context():
    """The top-k at a full 1M context: eight register-held trips per thread, all live (cache filled directly)."""
    from kernels.monokernel.dsv4.kernel import THREADS, n_topk_parts
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel
    from kernels.monokernel.dsv4.reference import pack_fp4

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 1 << 20
    n, ihd = cfg.n_compressed, cfg.index_head_dim
    trips = n // (THREADS * 4 * n_topk_parts(cfg.max_seq, ratio, ihd))
    assert trips >= 8, f"only {trips} trips a thread; the point is several"
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=1, rank=0, npes=1, moe_mode=mode)
    rows = pack_fp4(torch.randn(n, ihd, device=dev))
    fp4_pool_store(layer.i_cache[0], layer.i_cache_s[0], layer.block_tables[0], rows)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    pos = cfg.max_seq - 1
    h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos], cfg, dev)
    layer.forward(h, torch.tensor([pos], dtype=torch.int32, device=dev), kv_k, dest, idx, cos, sin)
    torch.cuda.synchronize()
    k = min(cfg.index_topk, n)
    got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
    sel = got[got >= 0].tolist()
    assert len(sel) == k, f"wrote {len(sel)} slots, want {k}"
    assert len(set(sel)) == k, f"{k - len(set(sel))} picks collided on a slot"
    sc = layer.debug("i_score", (1, n))[0]
    srt = sc.sort(descending=True).values
    want = set((cfg.window + sc.topk(k).indices).tolist())
    if set(sel) != want and (srt[k - 1] - srt[k]).item() > 1e-6:
        raise AssertionError(f"picked {len(set(sel) - want)} entries the scores do not rank")
    layer.close()


def test_dsv4_fp8_kv_reads_each_groups_own_scale():
    """The fp8 KV read takes each 64-wide group's own scale (group g scaled 4**g, so a wrong one is >= 4x off)."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg(hc_mult=1)
    cfg.kv_fp8 = True
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    nope = cfg.head_dim - cfg.rope_dim
    gain = torch.ones(cfg.head_dim, device=dev)
    gain[:nope] = 4.0 ** (torch.arange(nope, device=dev) // 64 - 3).float()
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev) * gain).bfloat16()
    planes = encode_kv_fp8(kv0)
    kv_ref = decode_kv_fp8(*planes).bfloat16()  # the same values, as the golden's bf16 plane
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    layer.forward(h, cur, planes, dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(W, h, [pos] * S, kv_ref, dest, idx, cos, sin, lambda z: z, moe_mode=mode)
    a, b = got["o"].float(), ref["o"].float()
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 2e-2, f"attention output off the golden by rel_l2 {rel:.4f}"
    layer.close()


def test_dsv4_fp8_kv_leaves_out_unwritten_and_nan_rows():
    """Rows ATOM never wrote (all 0xFF) or wrote with a NaN RoPE half are left out, as if the key were -1."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg(hc_mult=1)
    cfg.kv_fp8 = True
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    nope_u8, rope = encode_kv_fp8(kv0)
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    unwritten, nan_rope = int(idx[0, 3]), int(idx[0, 10])
    nope_u8[unwritten] = 0xFF
    rope[unwritten].view(torch.int16).fill_(-1)
    rope[nan_rope] = float("nan")
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    layer.forward(h, cur, (nope_u8, rope), dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    kv_ref = torch.nan_to_num(decode_kv_fp8(nope_u8, rope)).bfloat16()
    idx_ref = idx.clone()
    idx_ref[0, 3] = idx_ref[0, 10] = -1
    ref = golden_layer(W, h, [pos] * S, kv_ref, dest, idx_ref, cos, sin, lambda z: z, moe_mode=mode)
    a, b = got["o"].float(), ref["o"].float()
    assert not torch.isnan(a).any(), "a poisoned row reached the attention output"
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 2e-2, f"attention output off the golden (poisoned keys left out) by rel_l2 {rel:.4f}"
    layer.close()


@pytest.mark.parametrize("S", [1, 2])
def test_dsv4_hash_routing_takes_the_table(S):
    """Hash-routed layers pick exactly tid2eid[token] per sample, not the scored top-k."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode = "cuda", MoeMode.W8A8
    cfg = _cfg(hc_mult=1)
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    vocab = 1000
    W.t["tid2eid"] = torch.stack([torch.randperm(cfg.n_experts, device=dev)[: cfg.top_k] for _ in range(vocab)]).to(
        torch.int32
    )
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    tokens = torch.randint(0, vocab, (S,), dtype=torch.int32, device=dev)
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin, tokens=tokens)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, lambda z: z, moe_mode=mode, tokens=tokens)
    for s in range(S):
        want = set(W.t["tid2eid"][tokens[s].long()].tolist())
        picked = set(got["sel"][s].tolist()[1:])  # slot 0 is the shared expert
        assert picked == want, f"sample {s}: routed to {sorted(picked)}, table says {sorted(want)}"
    a, b = out.float(), ref["x_out"].float()
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < _tol(OUT_REL_L2, cfg.hc_mult, 1), f"x_out rel_l2 {rel:.5f}"
    layer.close()


@pytest.mark.parametrize("ratio", [0, COMPRESS_CSA])
def test_dsv4_batching_is_independent_sequences(ratio):
    """S=2 equals two S=1 runs exactly: catches a stage reading sample 0's state or position for all."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 2
    cfg = _cfg(hc_mult=4)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    if ratio:
        cfg.index_topk = 4
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
            cur = torch.tensor(ps, dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool(ps, cfg, dev)
            h = torch.cat([hs[step][r : r + 1] for r in rows])
            out = layer.forward(h, cur, kv, dest, idx, cos, sin).clone()
            torch.cuda.synchronize()
            seq.append((out, {k: v.clone() for k, v in layer.intermediates().items()}))
        layer.close()
        return seq

    batched = run(list(range(S)))
    for s in range(S):
        alone = run([s])
        for pos in range(steps):
            for name in ("q_a", "kv", "q", "o", "o_lora", "a", "scores"):
                d = (batched[pos][1][name][s].float() - alone[pos][1][name][0].float()).abs().max().item()
                assert d == 0.0, f"pos={pos} sample {s}: stage {name} moved by {d:.3e} when batched"
            assert batched[pos][1]["sel"][s].tolist() == alone[pos][1]["sel"][0].tolist()
            # x_out is not bit-exact: batching regroups `down`'s f32 summation tree
            a, b = batched[pos][0][s].float(), alone[pos][0][0].float()
            rel = ((a - b).norm() / b.norm().clamp(min=1e-6)).item()
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
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 512
    if ratio == COMPRESS_CSA:
        cfg.index_topk = 4
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
            ref[(r, ps[r])] = (
                out[r],
                {n: got[n][r].clone() for n in ("q_a", "kv", "q", "o", "o_lora", "a", "scores", "sel")},
            )
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
        out = lay.forward(
            torch.cat(hs), torch.tensor(ps, dtype=torch.int32, device=dev), kv, dest, idx, cos, sin
        ).clone()
        torch.cuda.synchronize()
        got = lay.intermediates()
        for r in range(NSEQ):
            for j in range(acc[r]):
                i, pos = r * TOK + j, nxt[r] + j
                if (r, pos) not in ref:  # past the reference's history (the other sequence lagged)
                    continue
                r_out, r_st = ref[(r, pos)]
                for name in ("q_a", "kv", "q", "o", "o_lora", "a", "scores"):
                    d = (got[name][i].float() - r_st[name].float()).abs().max().item()
                    assert d == 0.0, f"seq {r} pos {pos} (token {j}): stage {name} moved by {d:.3e} in the MTP launch"
                assert got["sel"][i].tolist() == r_st["sel"].tolist(), f"seq {r} pos {pos}: routing differs"
                rel = ((out[i].float() - r_out.float()).norm() / r_out.float().norm().clamp(min=1e-6)).item()
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
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 1024
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
        h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
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
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = COMPRESS_HCA, 512
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
        h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos] * S, cfg, dev)
        a = lay_a.forward(h, cur, kv_a, dest, idx, cos, sin)
        b = lay_b.forward(
            h, cur, kv_b, dest + far_rows, torch.where(idx >= 0, idx + far_rows, idx), cos, sin, state=far
        )
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
    cfg = _cfg(hc_mult=4)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 4
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


def test_dsv4_compress_schedule_is_the_checkpoints():
    """The per-layer compress-ratio schedule matches V4-Pro's (and its config.json when present)."""
    import json
    import os

    from kernels.monokernel.dsv4.config import COMPRESS_CSA as CSA
    from kernels.monokernel.dsv4.config import COMPRESS_HCA as HCA
    from kernels.monokernel.dsv4.config import compress_ratios

    r = compress_ratios()
    assert len(r) == 62, f"61 layers plus one MTP entry, got {len(r)}"
    main, mtp = r[:61], r[61:]
    assert main.count(HCA) == 31 and main.count(CSA) == 30, f"31 HCA + 30 CSA, got {main}"
    assert 0 not in main, "V4-Pro has no sliding-window-only main layer"
    assert mtp == (0,), "the MTP block is the ratio-0 entry"
    assert main[0] == main[1] == HCA, "layers 0 and 1 are the one break in the alternation"
    for i in range(2, 61):
        want = CSA if i % 2 == 0 else HCA
        assert main[i] == want, f"layer {i} should be {want}, schedule says {main[i]}"

    path = "/tmp/dsv4_ref/config.json"
    if os.path.exists(path):
        assert list(r) == json.load(open(path))["compress_ratios"], "schedule differs from the config"


def test_dsv4_split_merge_spans_the_block():
    """More than 64 key splits takes the block-wide merge path; overrunning it is silently wrong, not a crash."""
    from kernels.monokernel.dsv4.kernel import SPLIT_KEYS
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg()
    cfg.compress_ratio, cfg.max_seq = COMPRESS_HCA, 655360
    cfg.validate()
    splits = cfg.n_keys // SPLIT_KEYS
    assert splits > 64, f"this shape must exercise the wide path, got {splits} splits"

    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MonoKernel(W, samples=S, rank=0, npes=1, moe_mode=mode)
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.max_seq - 1  # every compressed entry live
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_theta, device=dev)

    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    ref = golden_layer(
        W,
        h,
        [pos] * S,
        kv0.clone(),
        dest,
        idx,
        cos,
        sin,
        lambda z: z,
        moe_mode=mode,
        kv_state=torch.zeros(S, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev),
        score_state=torch.full((S, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev),
        cos_c=cos,
        sin_c=sin,
    )

    a_out, b_out = out.float(), ref["x_out"].float()
    assert torch.isfinite(a_out).all(), "the wide merge produced non-finite output"
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    assert rel_l2 < _tol(OUT_REL_L2, cfg.hc_mult, 1), f"x_out diverged: rel_l2 {rel_l2:.5f}"


@pytest.mark.parametrize("kv_fp8", [False, True])
@pytest.mark.parametrize("n_layers", [4])
def test_dsv4_alternating_stack_matches_golden(n_layers, kv_fp8):
    """V4-Pro's [HCA, HCA, CSA, HCA] prefix: layers of one variant share a kernel, not weights or state."""
    from kernels.monokernel.dsv4.op import Dsv4MonoKernel, Dsv4Variant

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    base = _cfg(hc_mult=4)
    base.max_seq = 256
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
        layers.append(
            Dsv4MonoKernel(
                Ws[i],
                S,
                rank=0,
                npes=1,
                moe_mode=mode,
                variant=variants[c.compress_ratio],
            )
        )
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
            coff, ihd = c.c_coff, c.index_head_dim
            st = dict(
                kv_state=torch.zeros(S, c.c_rows, coff * c.head_dim, device=dev),
                score_state=torch.full((S, c.c_rows, coff * c.head_dim), float("-inf"), device=dev),
            )
            if c.indexed:
                st |= dict(
                    i_state=torch.zeros(S, c.c_rows, coff * ihd, device=dev),
                    i_score_state=torch.full((S, c.c_rows, coff * ihd), float("-inf"), device=dev),
                    i_cache=torch.zeros(S, c.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=dev),
                )
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
            a, b = hk.float(), ref["x_out"].float()
            rel = ((a - b).norm() / b.norm()).item()
            tol = _tol(OUT_REL_L2, c.hc_mult, 1)
            assert rel < tol, f"pos={pos} layer {i} (ratio {c.compress_ratio}): rel_l2 {rel:.4f} >= {tol}"
        for v in variants.values():
            v.advance_step()
    if kv_fp8:
        for i, c in enumerate(cfgs):
            a, b = decode_kv_fp8(*kvs_k[i]), kvs_r[i].float()
            rel = ((a - b).norm() / b.norm()).item()
            assert rel < 1e-2, f"layer {i} (ratio {c.compress_ratio}): fp8 KV rows off the golden's, rel_l2 {rel:.4f}"
    for v in variants.values():
        v.close()
