# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FWD-36: the optional inputs main's forward carried (ALiBi, sink, split-K, paged, XCD swizzle), against the new forward.

They are all off by default and AOTriton never sets them, so the default ABI does not move (ABI-07). Each is gated the
same way as the rest of the forward: the fp64 reference with the rounding floor, NaN-poisoned slack, an effect check
(an input dropped on the way in must not pass), and the refusals of the host.
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    fwd_check,
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
    fwd_check(fn, b=2, hq=4, sq=130, sk=256, d=64, dtype=BF16, alibi_slopes=slopes_of(2, 4, True), ctx="first call [B, H]")


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


def sink_inputs(b, hq, hk, s, d, dtype, window, share, seed=11):
    gen = seeded(seed)
    q = randn(b, hq, s, d, dtype, gen=gen)
    k = randn(b, hk, s, d, dtype, gen=gen)
    v = randn(b, hk, s, d, dtype, gen=gen)
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
