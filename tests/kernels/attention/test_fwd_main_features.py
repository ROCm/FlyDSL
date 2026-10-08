# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FWD-36: the optional inputs main's forward carried (ALiBi, sink, split-K, paged, XCD swizzle), against the new forward.

They are all off by default and AOTriton never sets them, so the default ABI does not move (ABI-07). Each is gated the
same way as the rest of the forward: the fp64 reference with the rounding floor, NaN-poisoned slack, an effect check
(an input dropped on the way in must not pass), and the refusals of the host.
"""

import math

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
    VarlenCase,
    alloc,
    fwd_check,
    lse_alloc,
    meta_of,
    randn,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
BF16 = DTYPES["bf16"]

# ---------------------------------------------------------------------------
# ALiBi (metadata `alibi`): bottom-right aligned `-slope * |i + Sk - Sq - j|`, added after the scale and before the mask
# ---------------------------------------------------------------------------


def slopes_of(b, h, two_d, seed=7):
    """fp32 slopes in (0.05, 0.5): `(H,)` shared, or `(B, H)` with a different table per sequence."""
    gen = seeded(seed)
    shape = (b, h) if two_d else (h,)
    return (torch.rand(*shape, device="cuda", generator=gen) * 0.45 + 0.05).float()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("two_d", [False, True], ids=["shared", "per_batch"])
@pytest.mark.parametrize("shape", [(2, 8, 8, 512, 128), (1, 8, 4, 384, 64)], ids=["mha_d128", "gqa_d64"])
def test_alibi_dense(fwd_build, causal, two_d, shape):
    """ALiBi, dense, with and without a causal window, MHA and GQA, a shared `[H]` and a per-sequence `[B, H]` table.
    A dropped ALiBi must not pass: the same build with zero slopes gives a clearly different answer."""
    b, hq, hk, s, d = shape
    fn = fwd_build(meta_of(head_dim=d, window=causal, alibi=True))
    slopes = slopes_of(b, hq, two_d)
    window = BR if causal else None
    out = fwd_check(
        fn, b=b, hq=hq, hk=hk, sq=s, d=d, dtype=BF16, window=window, alibi_slopes=slopes, ctx=f"{shape} {causal}"
    )
    flat = alloc(b, hq, s, d, BF16)
    run_fwd(fn, out["q"], out["k"], out["v"], flat, window=window, alibi_slopes=torch.zeros_like(slopes))
    assert (flat.float() - out["o"].float()).abs().max().item() > 1e-2, "the slopes had no effect"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqs", [(130, 300), (300, 130)], ids=["sq_lt_sk", "sq_gt_sk"])
def test_alibi_cross_length(fwd_build, causal, seqs):
    """The bias is measured from the bottom-right corner (`Sk - Sq`), so a call whose lengths differ shifts it."""
    sq, sk = seqs
    fn = fwd_build(meta_of(head_dim=64, window=causal, alibi=True))
    fwd_check(
        fn,
        b=1,
        hq=4,
        sq=sq,
        sk=sk,
        d=64,
        dtype=BF16,
        window=BR if causal else None,
        alibi_slopes=slopes_of(1, 4, False),
        ctx=f"{seqs} {causal}",
    )


@pytest.mark.parametrize("hdim", [32, 100, 256, 384])
def test_alibi_across_rungs(fwd_build, hdim):
    """Every kernel body carries it: a narrow rung, a padded head, the 8-wave rung and the wide body (384)."""
    fn = fwd_build(meta_of(head_dim=hdim, window=True, alibi=True))
    fwd_check(
        fn,
        b=1,
        hq=4,
        sq=130,
        d=hdim,
        dtype=BF16,
        window=BR,
        alibi_slopes=slopes_of(1, 4, False),
        ctx=f"d{hdim}",
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("two_d", [False, True], ids=["shared", "per_sequence"])
def test_alibi_varlen(fwd_build, causal, two_d):
    """Under varlen the slope row is the *sequence's* (a `[B, H]` table is indexed by sequence, not by batch slice) and
    the offset `Sk - Sq` is per sequence."""
    fn = fwd_build(meta_of(head_dim=64, window=causal, alibi=True))
    case = VarlenCase("0x0B0B", [100, 200, 64], [150, 200, 300], 8, 4, 64, BF16)
    case.check(
        fn,
        window=BR if causal else None,
        alibi_slopes=slopes_of(3, 8, two_d),
        ctx=f"varlen {causal} {two_d}",
    )


def test_alibi_per_sequence_table_as_the_first_call(backend, arch):
    """The slope index is linear (`sequence * alibi_stride_b + head`), so the table reaches the kernel flat. A `[B, H]`
    tensor passed as-is keys the compiled signature at rank 2 and its layout decomposes the index column-major: wrong
    slopes for every sequence but the first, and only when that call is the *first* of a build (a later one reuses the
    first call's compile). A fresh builder, so no earlier test has made that call."""
    meta = meta_of(head_dim=64, alibi=True)
    fn = backend.build_fwd(meta, backend.fwd_knobs(arch).resolve(meta))
    fwd_check(
        fn, b=2, hq=4, sq=130, sk=256, d=64, dtype=BF16, alibi_slopes=slopes_of(2, 4, True), ctx="first call [B, H]"
    )


def test_alibi_and_bias_combine(fwd_build):
    """ALiBi and a bias are independent additive terms and may be combined."""
    b, h, s = 1, 4, 256
    gen = seeded(3)
    bias = (randn(b, h, s, s, BF16, gen=gen)).contiguous()
    fn = fwd_build(meta_of(head_dim=64, bias=True, alibi=True))
    fwd_check(fn, b=b, hq=h, sq=s, d=64, dtype=BF16, bias=bias, alibi_slopes=slopes_of(b, h, False), ctx="alibi+bias")


def test_alibi_with_bias_and_a_mask_is_rejected():
    """Bias with a causal or window mask has no defined meaning (plan 4.10), whatever else is on."""
    from kernels.attention import flash_attn_gfx950_config as cfg

    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.fwd_knobs("gfx950").resolve(meta_of(head_dim=64, window=True, bias=True, alibi=True))


def test_alibi_host_checks(fwd_build):
    """A build with ALiBi needs slopes (fp32, `[H]` or `[B, H]`, matching the heads and sequences); a build without it
    must not be handed any, because ignoring them would return the right shape and the wrong answer."""
    q, k, v = (randn(2, 4, 64, 64, BF16) for _ in range(3))
    o = alloc(2, 4, 64, 64, BF16)
    on = fwd_build(meta_of(head_dim=64, alibi=True))
    off = fwd_build(meta_of(head_dim=64))
    bad = {
        "missing": (on, None, "requires an fp32"),
        "fp16": (on, torch.ones(4, device="cuda", dtype=torch.float16), "fp32"),
        "3d": (on, torch.ones(2, 4, 1, device="cuda"), "fp32"),
        "heads": (on, torch.ones(3, device="cuda"), "3 heads"),
        "rows": (on, torch.ones(5, 4, device="cuda"), "5 rows"),
        "unwanted": (off, torch.ones(4, device="cuda"), "not compiled for ALiBi"),
    }
    for name, (fn, slopes, msg) in bad.items():
        with pytest.raises(ValueError, match=msg):
            run_fwd(fn, q, k, v, o, **({} if slopes is None else dict(alibi_slopes=slopes)))


# ---------------------------------------------------------------------------
# Attention sink (metadata `sink`): one extra denominator logit per head, no value row
# ---------------------------------------------------------------------------


def calibrated_sink(q, k, hq, window, share=0.5):
    """An fp32 `[H]` sink that takes about `share` of each head's softmax mass, so a kernel that drops it fails.

    The mean log-sum-exp of the head's live rows plus `log(share / (1 - share))`: the sink logit is that much above the
    typical row, so it weighs `share / (1 - share)` against the keys. An uncalibrated sink near zero would be invisible
    under unit-variance logits.
    """
    from tests.kernels.attention.attn_testlib import reference, sdpa_scale, window_mask

    mask = window_mask(q.shape[2], k.shape[2], *window) if window is not None else None
    _, lse = reference(q, k, k[..., : q.shape[-1]], sdpa_scale(q.shape[-1]), mask=mask)
    live = torch.isfinite(lse)
    mean = torch.where(live, lse, torch.zeros_like(lse)).sum(dim=(0, 2)) / live.sum(dim=(0, 2)).clamp(min=1)
    return (mean + torch.log(torch.tensor(share / (1.0 - share), dtype=torch.float64, device=mean.device))).float()


def sink_inputs(b, hq, hk, s, d, dtype, window, share, seed=11, sk=None):
    gen = seeded(seed)
    sk = s if sk is None else sk
    q = randn(b, hq, s, d, dtype, gen=gen)
    k = randn(b, hk, sk, d, dtype, gen=gen)
    v = randn(b, hk, sk, d, dtype, gen=gen)
    return (q, k, v), calibrated_sink(q, k.repeat_interleave(hq // hk, dim=1), hq, window, share)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("share", [0.25, 0.9])
@pytest.mark.parametrize("shape", [(1, 8, 8, 512, 128), (2, 8, 4, 384, 64)], ids=["mha_d128", "gqa_d64"])
def test_sink_dense(fwd_build, causal, share, shape):
    """The sink takes weight from the keys: O matches the sink-aware reference, the LSE includes the sink, and with the
    sink pushed to -inf-ish the answer visibly changes (so a dropped sink cannot pass)."""
    b, hq, hk, s, d = shape
    window = BR if causal else None
    qkv, sink = sink_inputs(b, hq, hk, s, d, BF16, window, share)
    fn = fwd_build(meta_of(head_dim=d, window=causal, sink=True))
    out = fwd_check(fn, b=b, hq=hq, hk=hk, sq=s, d=d, dtype=BF16, window=window, qkv=qkv, sink=sink, ctx=str(shape))
    off = alloc(b, hq, s, d, BF16)
    run_fwd(fn, *qkv, off, window=window, sink=torch.full_like(sink, -1e4))
    assert (off.float() - out["o"].float()).abs().max().item() > 1e-2, "the sink had no effect"


@pytest.mark.parametrize("hdim", [32, 100, 256, 384])
def test_sink_across_rungs(fwd_build, hdim):
    """Every kernel body folds it: a narrow rung, a padded head, the 8-wave rung and the wide body (384)."""
    qkv, sink = sink_inputs(1, 4, 4, 130, hdim, BF16, BR, 0.5)
    fn = fwd_build(meta_of(head_dim=hdim, window=True, sink=True))
    fwd_check(fn, b=1, hq=4, sq=130, d=hdim, dtype=BF16, window=BR, qkv=qkv, sink=sink, ctx=f"d{hdim}")


@pytest.mark.parametrize("causal", [False, True])
def test_sink_varlen(fwd_build, causal):
    """One `[H]` table shared by every sequence; each sequence is gated against its own sink-aware reference."""
    fn = fwd_build(meta_of(head_dim=128, window=causal, sink=True))
    case = VarlenCase("0x0B0B", [512, 256, 384], [512, 256, 384], 8, 4, 128, BF16)
    sink = calibrated_sink(case.seq(0)[0], case.seq(0)[1].repeat_interleave(2, dim=1), 8, BR if causal else None, 0.5)
    case.check(fn, window=BR if causal else None, sink=sink, ctx=f"varlen {causal}")


@pytest.mark.parametrize("seqs", [(512, 128), (512, 160)], ids=["whole_blocks_skipped", "rows_in_a_live_block"])
def test_sink_lse_cross_attn_skipped_blocks(fwd_build, seqs):
    """Causal cross-attention with `Sk < Sq`: the first `Sq - Sk` rows see no key. Their denominator is the sink alone,
    so the LSE is the sink (not -inf, not `+inf`) and O is exactly zero. A skipped q block never reaches the main body,
    so its LSE is written by the skip path; `Sk = 160` also puts some all-masked rows inside a live block."""
    sq, sk = seqs
    gen = seeded(5)
    q = randn(2, 8, sq, 128, BF16, gen=gen)
    k = randn(2, 8, sk, 128, BF16, gen=gen)
    v = randn(2, 8, sk, 128, BF16, gen=gen)
    sink = (torch.arange(8, device="cuda", dtype=torch.float32) - 3.5) * 0.4
    fn = fwd_build(meta_of(head_dim=128, window=True, sink=True))
    out = fwd_check(fn, b=2, hq=8, sq=sq, sk=sk, d=128, dtype=BF16, window=BR, qkv=(q, k, v), sink=sink, ctx=str(seqs))
    dead = sq - sk
    assert torch.allclose(out["lse"][:, :, :dead], sink.view(1, 8, 1).expand(2, 8, dead), atol=1e-4)
    assert out["o"][:, :, :dead].abs().max().item() == 0.0


def test_sink_and_alibi_combine(fwd_build):
    """The two additive-logit inputs are independent."""
    qkv, sink = sink_inputs(1, 4, 4, 256, 64, BF16, BR, 0.5)
    fn = fwd_build(meta_of(head_dim=64, window=True, alibi=True, sink=True))
    fwd_check(
        fn,
        b=1,
        hq=4,
        sq=256,
        d=64,
        dtype=BF16,
        window=BR,
        qkv=qkv,
        sink=sink,
        alibi_slopes=slopes_of(1, 4, False),
        ctx="sink+alibi",
    )


def test_sink_host_checks(fwd_build):
    q, k, v = (randn(2, 4, 64, 64, BF16) for _ in range(3))
    o = alloc(2, 4, 64, 64, BF16)
    on = fwd_build(meta_of(head_dim=64, sink=True))
    off = fwd_build(meta_of(head_dim=64))
    bad = {
        "missing": (on, None, "requires an fp32"),
        "fp16": (on, torch.ones(4, device="cuda", dtype=torch.float16), "fp32"),
        "heads": (on, torch.ones(3, device="cuda"), r"\(4,\)"),
        "unwanted": (off, torch.ones(4, device="cuda"), "not compiled for a sink"),
    }
    for name, (fn, sink, msg) in bad.items():
        with pytest.raises(ValueError, match=msg):
            run_fwd(fn, q, k, v, o, **({} if sink is None else dict(sink=sink)))


# ---------------------------------------------------------------------------
# Paged KV (metadata `paged`, `kv_cache_layout`): K/V are a page pool of 64-token pages, a block table maps (sequence,
# page) to a physical page. Dense: every sequence has the same `seqlen_k`.
# ---------------------------------------------------------------------------

LAYOUTS = ["linear", "vectorized"]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [(2, 8, 4, 512, 512, 128), (1, 4, 4, 130, 300, 64), (2, 4, 2, 300, 130, 128)],
    ids=["gqa_d128", "cross_sq_lt_sk", "cross_sq_gt_sk"],
)
def test_paged_dense(fwd_build, layout, causal, shape):
    """Dense attention over a randomly placed page pool equals the same attention over the logical K/V (fp64 floor),
    in both cache layouts, MHA/GQA, with a ragged last page and `Sq != Sk`."""
    b, hq, hk, sq, sk, d = shape
    fn = fwd_build(meta_of(head_dim=d, window=causal, paged=True, kv_cache_layout=layout))
    fwd_check(
        fn,
        b=b,
        hq=hq,
        hk=hk,
        sq=sq,
        sk=sk,
        d=d,
        dtype=BF16,
        window=BR if causal else None,
        paged=layout,
        ctx=f"{layout} {shape} {causal}",
    )


@pytest.mark.parametrize(
    "layout,hdim",
    [("linear", d) for d in (32, 64, 192, 256)] + [("vectorized", d) for d in (64, 128)],
)
def test_paged_across_rungs(fwd_build, layout, hdim):
    """Every dual-wave rung that can be paged: all of them in the linear layout, 64 and 128 in aiter's vectorized one
    (the wide body above 256 refuses)."""
    fn = fwd_build(meta_of(head_dim=hdim, window=True, paged=True, kv_cache_layout=layout))
    fwd_check(fn, b=1, hq=4, hk=2, sq=130, sk=200, d=hdim, dtype=BF16, window=BR, paged=layout, ctx=f"{layout} d{hdim}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_paged_long_context_and_unowned_pages(fwd_build, layout):
    """A long sequence (128 pages, randomly placed among unowned ones), non-causal: a wrong page id anywhere changes the
    softmax."""
    fn = fwd_build(meta_of(head_dim=128, paged=True, kv_cache_layout=layout))
    fwd_check(fn, b=1, hq=4, hk=4, sq=128, sk=8192, d=128, dtype=BF16, paged=layout, ctx=layout)


@pytest.mark.parametrize("table", [[[0, 1]], [[4, 1], [2, 6]], [[0, 1], [2, 3], [4, 5]]], ids=["b1", "b2", "b3"])
def test_paged_block_table_is_read_row_major(fwd_build, table):
    """Which pages does each sequence read? V page `p` holds the constant `p` and K is zero, so attention is uniform and
    sequence `b` returns the mean of its table row. (The kernel indexes the table as a flat array; a 2D tensor handed to
    it would be decomposed column-major and every row but the first would read the wrong pages.)"""
    b = len(table)
    q = alloc(b, 1, 128, 64, BF16, fill=torch.zeros(b, 1, 128, 64))
    pool_k = torch.zeros(8, 64, 1, 64, device="cuda", dtype=BF16)
    pool_v = torch.zeros(8, 64, 1, 64, device="cuda", dtype=BF16)
    for page in range(8):
        pool_v[page] = float(page)
    o = alloc(b, 1, 128, 64, BF16)
    fn = fwd_build(meta_of(head_dim=64, paged=True))
    run_fwd(fn, q, pool_k, pool_v, o, seqlen_k=128, block_table=torch.tensor(table, dtype=torch.int32, device="cuda"))
    got = [o[i, 0].float().mean().item() for i in range(b)]
    assert got == [sum(row) / len(row) for row in table]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_paged_ignores_the_stale_tail_of_the_last_page(fwd_build, layout):
    """The slots of the last page past `seqlen_k` hold stale keys; they must not take softmax weight."""
    fn = fwd_build(meta_of(head_dim=64, paged=True, kv_cache_layout=layout))
    fwd_check(fn, b=2, hq=4, sq=64, sk=70, d=64, dtype=BF16, paged=layout, ctx=f"{layout} tail", input_scale=3.0)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_paged_with_bias_alibi_and_sink(fwd_build, layout):
    """Bias, ALiBi and a sink are all indexed by the logical KV position, which the block table does not move."""
    b, h, sq, sk, d = 2, 4, 130, 256, 64
    gen = seeded(9)
    bias = randn(b, h, sq, sk, BF16, gen=gen).contiguous()
    qkv, sink = sink_inputs(b, h, h, sq, d, BF16, None, 0.5, sk=sk)
    fn = fwd_build(meta_of(head_dim=d, paged=True, kv_cache_layout=layout, bias=True, alibi=True, sink=True))
    fwd_check(
        fn,
        b=b,
        hq=h,
        sq=sq,
        sk=sk,
        d=d,
        dtype=BF16,
        bias=bias,
        alibi_slopes=slopes_of(b, h, True),
        sink=sink,
        paged=layout,
        ctx=f"{layout} bias+alibi+sink",
    )


def test_paged_host_checks_and_refusals(fwd_build):
    from tests.kernels.attention.attn_testlib import PagedCache

    q = randn(2, 4, 64, 64, BF16)
    o = alloc(2, 4, 64, 64, BF16)
    cache = PagedCache.random(2, 4, 128, 64, BF16, "linear")
    fn = fwd_build(meta_of(head_dim=64, paged=True))

    def call(fn=fn, **over):
        args = dict(k=cache.k, v=cache.v, seqlen_k=128, block_table=cache.block_table)
        args.update(over)
        k, v = args.pop("k"), args.pop("v")
        run_fwd(fn, q, k, v, o, **args)

    call()  # the control: each case below is wrong in exactly one way
    cases = {
        "no_table": (dict(block_table=None), "requires an int32"),
        "no_seqlen": (dict(seqlen_k=None), "needs `seqlen_k`"),
        "dtype": (dict(block_table=cache.block_table.long()), "int32"),
        "rows": (dict(block_table=cache.block_table[:1]), "a row per sequence"),
        "short": (dict(block_table=cache.block_table[:, :1].contiguous()), "pages for seqlen_k"),
        "pool_page": (dict(k=cache.k[:, :32], v=cache.v[:, :32]), "64"),
        "pool_mismatch": (dict(v=cache.v[:1].contiguous()), "share shape"),
    }
    for name, (over, msg) in cases.items():
        with pytest.raises(ValueError, match=msg):
            call(**over)
    # a table handed to a build that is not paged
    with pytest.raises(ValueError, match="not compiled for a paged"):
        run_fwd(
            fwd_build(meta_of(head_dim=64)),
            q,
            cache.k_logical,
            cache.v_logical,
            o,
            seqlen_k=128,
            block_table=cache.block_table,
        )


def test_paged_builder_refusals():
    """Where the paged path stops: a padded head (the page layout fixes the head stride at the rung), the wide body, and
    the vectorized layout off the two rungs it is staged for."""
    from kernels.attention import flash_attn_gfx950 as fwd_module
    from kernels.attention import flash_attn_gfx950_config as cfg

    for hdim, layout, match in (
        (100, "linear", "exact head dims"),
        (384, "linear", "wide body"),
        (96, "vectorized", "head_dim 64 and 128"),
        (256, "vectorized", "head_dim 64 and 128"),
    ):
        meta = meta_of(head_dim=hdim, paged=True, kv_cache_layout=layout)
        knobs = cfg.fwd_knobs("gfx950").resolve(meta)
        with pytest.raises(NotImplementedError, match=match):
            fwd_module.build_flash_attn_gfx950_fwd(meta, knobs)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("causal", [False, True])
def test_paged_varlen_one_block_table_row_per_sequence(fwd_build, layout, causal):
    """Packed (stacked) varlen Q over a shared page pool: sequences of different lengths, each with its own row of the block
    table. The row is the *sequence* (the grid's z), not the batch slice, which is 0 for every stacked sequence: with that
    mistake every sequence would read the first one's pages."""
    from tests.kernels.attention.attn_testlib import (
        PagedCache,
        check_floor,
        floor_rel,
        reference,
        sdpa_scale,
        window_mask,
    )

    d, h, hk = 64, 4, 2
    lens_q, lens_k = [100, 200, 64], [150, 70, 64]
    case = VarlenCase("0x0B0B", lens_q, lens_k, h, hk, d, BF16)
    ks = [case.seq(z)[1] for z in range(case.n)]
    vs = [case.seq(z)[2] for z in range(case.n)]
    cache = PagedCache.from_sequences(ks, vs, layout, gen=seeded(8))
    fn = fwd_build(meta_of(head_dim=d, window=causal, paged=True, kv_cache_layout=layout))
    window = BR if causal else None
    k0, v0 = case.k, case.v
    case.k, case.v = cache.k, cache.v  # the kernel reads the pool; the reference reads the logical keys (`ks`, `vs`)
    try:
        case.launch(fn, window=window, block_table=cache.block_table)
    finally:
        case.k, case.v = k0, v0
    for z in range(case.n):
        q = case.seq(z)[0]
        mask = window_mask(q.shape[2], ks[z].shape[2], *window) if window is not None else None
        sm = sdpa_scale(d)

        def ref(**kw):
            return reference(q, ks[z], vs[z], sm, mask=mask, **kw)

        exact_o, exact_lse = ref()
        check_floor("O", case.o_of(z), exact_o, floor_rel(ref, exact_o, BF16), f"paged varlen seq {z}")


# ---------------------------------------------------------------------------
# Split-K (knob `NUM_KV_SPLITS`): the KV range of one q block is cut into `splits` chunks run by separate workgroups, each
# writes a normalised partial O plus (m, l) into a workspace, and a **second kernel (its own builder)** combines them.
# Dense self-attention only, over a BSHD-flat O.
# ---------------------------------------------------------------------------

BSHD = (0, 2, 1)  # physical order of the (B, H, S) axes of a BSHD-flat allocation
SPLITS = [2, 3, 4]


@pytest.mark.parametrize("splits", SPLITS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hdim", [64, 128])
def test_splitk_matches_reference(fwd_build, splits, causal, hdim):
    """O and the LSE (written by the combine) against the fp64 floor, at 2, 3 and 4 splits, with and without a causal
    window (the causal case is where the splits' work is unequal)."""
    fn = fwd_build(meta_of(head_dim=hdim, window=causal), NUM_KV_SPLITS=splits)
    fwd_check(
        fn,
        b=1,
        hq=8,
        sq=2048,
        d=hdim,
        dtype=BF16,
        window=BR if causal else None,
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx=f"{splits} splits d{hdim} {causal}",
    )


@pytest.mark.parametrize("causal", [False, True])
def test_splitk_gqa_batched_and_ragged_tiles(fwd_build, causal):
    """GQA (the head remap and the workspace are indexed by the q head), a batch of two, and a length that is not a tile
    multiple (the last split's last tile is partial)."""
    fn = fwd_build(meta_of(head_dim=64, window=causal), NUM_KV_SPLITS=3)
    fwd_check(
        fn,
        b=2,
        hq=8,
        hk=2,
        sq=1000,
        d=64,
        dtype=BF16,
        window=BR if causal else None,
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx=f"gqa {causal}",
    )


@pytest.mark.parametrize("seq_len", [64, 130, 512])
@pytest.mark.parametrize("hdim", [64, 128])
def test_split_kv_writes_every_row(fwd_build, seq_len, hdim):
    """Short sequences leave splits with no tiles: they write an empty partial (m = -1e30, l = 0) the combine ignores.
    O is NaN-poisoned, so a row nobody wrote shows up; the answer still matches the reference."""
    fn = fwd_build(meta_of(head_dim=hdim, window=True), NUM_KV_SPLITS=4)
    out = fwd_check(
        fn,
        b=1,
        hq=4,
        sq=seq_len,
        d=hdim,
        dtype=BF16,
        window=BR,
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx=f"S={seq_len}",
    )
    assert not torch.isnan(out["o"]).any()


@pytest.mark.parametrize("splits", SPLITS)
def test_sink_splitk_counted_once(fwd_build, splits):
    """Split-K writes sink-free partials and folds the sink in once, in the combine. The LSE is the sharp signal: it is
    the log denominator, so a sink counted once per split (or not at all) shows up there directly, where O normalises it
    away."""
    b, h, s, d = 1, 8, 2048, 128
    qkv, sink = sink_inputs(b, h, h, s, d, BF16, BR, 0.5)
    split = fwd_build(meta_of(head_dim=d, window=True, sink=True), NUM_KV_SPLITS=splits)
    whole = fwd_build(meta_of(head_dim=d, window=True, sink=True))
    got = fwd_check(
        split,
        b=b,
        hq=h,
        sq=s,
        d=d,
        dtype=BF16,
        window=BR,
        qkv=qkv,
        sink=sink,
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx=f"{splits} splits + sink",
    )
    ref = alloc(b, h, s, d, BF16)
    ref_lse = lse_alloc(b, h, s)
    run_fwd(whole, *qkv, ref, lse=ref_lse, window=BR, sink=sink)
    assert (got["lse"] - ref_lse).abs().max().item() < 0.5 * math.log(splits)  # a double count shifts it by ~ln(splits)
    assert (got["lse"] - ref_lse).abs().max().item() < 2e-2


def test_splitk_with_alibi_bias_and_dropout_off(fwd_build):
    """The additive logit inputs are position-indexed, which the split's tile range does not move."""
    b, h, s, d = 1, 4, 1024, 64
    bias = randn(b, h, s, s, BF16, gen=seeded(2)).contiguous()
    fn = fwd_build(meta_of(head_dim=d, bias=True, alibi=True), NUM_KV_SPLITS=2)
    fwd_check(
        fn,
        b=b,
        hq=h,
        sq=s,
        d=d,
        dtype=BF16,
        bias=bias,
        alibi_slopes=slopes_of(b, h, False),
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx="split+bias+alibi",
    )


@pytest.mark.parametrize("seqs", [(512, 2048), (2048, 512)])
@pytest.mark.parametrize("causal", [False, True])
def test_splitk_rejects_cross_length_kv(fwd_build, seqs, causal):
    """Dense split-K is self-attention only, and it used to fail without saying so."""
    sq, sk = seqs
    fn = fwd_build(meta_of(head_dim=64, window=causal), NUM_KV_SPLITS=4)
    q, k, v = randn(1, 4, sq, 64, BF16), randn(1, 4, sk, 64, BF16), randn(1, 4, sk, 64, BF16)
    o = alloc(1, 4, sq, 64, BF16, BSHD)
    with pytest.raises(ValueError, match="self-attention only"):
        run_fwd(fn, q, k, v, o, window=BR if causal else None)


def test_splitk_host_checks(fwd_build):
    fn = fwd_build(meta_of(head_dim=64), NUM_KV_SPLITS=2)
    q, k, v = (randn(1, 4, 256, 64, BF16) for _ in range(3))
    good = alloc(1, 4, 256, 64, BF16, BSHD)
    run_fwd(fn, q, k, v, good)  # the control
    with pytest.raises(ValueError, match="contiguous in \\(batch, seq, head, dim\\)"):
        run_fwd(fn, q, k, v, alloc(1, 4, 256, 64, BF16))  # BHSD-flat O
    small = torch.empty(16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="fp32 workspace of at least"):
        run_fwd(fn, q, k, v, good, workspace=small)
    case = VarlenCase("0x0B0B", [64, 64], [64, 64], 4, 4, 64, BF16)
    with pytest.raises(ValueError, match="dense-only"):
        case.launch(fn)


def test_splitk_builder_refusals():
    from kernels.attention import flash_attn_gfx950 as fwd_module
    from kernels.attention import flash_attn_gfx950_config as cfg

    for hdim, match in ((100, "exact head dims"), (384, "wide body")):
        meta = meta_of(head_dim=hdim)
        knobs = cfg.fwd_knobs("gfx950", NUM_KV_SPLITS=2).resolve(meta)
        with pytest.raises(NotImplementedError, match=match):
            fwd_module.build_flash_attn_gfx950_fwd(meta, knobs)


def test_splitk_combine_is_its_own_builder(fwd_build):
    """The forward builder keeps exactly one kernel (AOTriton locates its kernel by uniqueness); the combine is a second
    builder that the forward's launcher drives."""
    fn = fwd_build(meta_of(head_dim=64), NUM_KV_SPLITS=2)
    assert fn.combine is not None and fn.combine.launcher is not fn.launcher
    assert fwd_build(meta_of(head_dim=64)).combine is None


@pytest.mark.parametrize("mode", ["runtime", "always", "never"])
def test_splitk_return_lse_modes(fwd_build, mode):
    """`RETURN_LSE` follows through the combine: "never" writes no LSE, "runtime" skips a null one, "always" needs one."""
    fn = fwd_build(meta_of(head_dim=64), NUM_KV_SPLITS=2, RETURN_LSE=mode)
    q, k, v = (randn(1, 4, 512, 64, BF16, gen=seeded(1)) for _ in range(3))
    o = alloc(1, 4, 512, 64, BF16, BSHD)
    lse = lse_alloc(1, 4, 512)
    if mode == "never":
        run_fwd(fn, q, k, v, o)
        assert not torch.isnan(o).any()
    else:
        run_fwd(fn, q, k, v, o, lse=lse)
        assert torch.isfinite(lse).all()
        if mode == "runtime":
            o2 = alloc(1, 4, 512, 64, BF16, BSHD)
            run_fwd(fn, q, k, v, o2)  # a null LSE must not fault
            assert torch.equal(o, o2)


@pytest.mark.parametrize("what", ["paged", "paged_causal", "band", "top_left"])
def test_splitk_composes_with_paged_and_windows(fwd_build, what):
    """The split's tile range is the thing paging (`split_tile`) and windows (`_skip_dead_leading_tiles`) were written
    against, so they compose with it: a paged pool, a causal paged pool, a banded window and a top-left causal window.
    """
    meta = dict(head_dim=64, paged=what.startswith("paged"), window=what != "paged")
    window = {"paged": None, "paged_causal": BR, "band": (300, 0), "top_left": (WINDOW_TOPLEFT, WINDOW_TOPLEFT)}[what]
    kw = dict(paged="linear") if meta["paged"] else {}
    fn = fwd_build(meta_of(**meta), NUM_KV_SPLITS=2)
    fwd_check(
        fn,
        b=1,
        hq=4,
        sq=1024,
        d=64,
        dtype=BF16,
        window=window,
        perms=((0, 1, 2), (0, 1, 2), (0, 1, 2), BSHD),
        ctx=what,
        **kw,
    )


def test_splitk_dropout_draws_the_same_mask_as_one_split(fwd_build):
    """Dropout is keyed by (sequence, head, row, column), not by the workgroup, so a split-K run keeps and drops exactly
    the elements the unsplit one does, and the combine's `l`-weighted sum reproduces its output (to rounding)."""
    b, h, s, d = 1, 4, 1024, 64
    gen = seeded(21)
    q, k, v = (randn(b, h, s, d, BF16, gen=gen) for _ in range(3))
    meta = meta_of(head_dim=d, dropout=True)
    whole = alloc(b, h, s, d, BF16)
    split = alloc(b, h, s, d, BF16, BSHD)
    run_fwd(fwd_build(meta), q, k, v, whole, p_drop=0.25, seed=5, offset=0)
    run_fwd(fwd_build(meta, NUM_KV_SPLITS=3), q, k, v, split, p_drop=0.25, seed=5, offset=0)
    assert (whole.float() - split.float()).abs().max().item() < 2e-2
    # ... and it did drop something: the dropout-free output differs
    plain = alloc(b, h, s, d, BF16)
    run_fwd(fwd_build(meta), q, k, v, plain, p_drop=0.0, seed=5, offset=0)
    assert (plain.float() - split.float()).abs().max().item() > 1e-2


# ---------------------------------------------------------------------------
# XCD swizzle (knob `XCD_SWIZZLE`): head-slow workgroup mapping, non-causal only, a bijection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heads", [8, 16, 64, 12, 4], ids=["h8", "h16", "h64", "h12_not_xcd", "h4_not_xcd"])
def test_xcd_swizzle_is_bit_identical(fwd_build, heads):
    """The head-slow remap re-derives (head, q block) from the same linear workgroup id, so it is a bijection and must
    not change one bit of the output. A mistake in the derivation shows as a permuted or half-recomputed output, not as
    an error, so this pins it. Head counts that do not divide into the eight XCDs fall back to the plain mapping at run
    time (a runtime head count makes that a runtime condition), however the knob is set."""
    s = 8 * 128  # several q blocks per head
    gen = seeded(heads)
    q, k, v = (randn(1, heads, s, 128, BF16, gen=gen) for _ in range(3))
    meta = meta_of(head_dim=128)
    off, on = alloc(1, heads, s, 128, BF16), alloc(1, heads, s, 128, BF16)
    run_fwd(fwd_build(meta), q, k, v, off)
    run_fwd(fwd_build(meta, XCD_SWIZZLE=True), q, k, v, on)
    assert torch.equal(off, on)
