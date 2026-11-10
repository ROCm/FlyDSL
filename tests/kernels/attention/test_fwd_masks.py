# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward masks: the window suite (FWD-10), STATIC_WINDOW (FWD-11) and per-sequence bottom-right under varlen (FWD-25)."""

import pytest
import torch

from kernels.attention.flash_attn_gfx950_config import LADDER
from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
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
TL = (WINDOW_TOPLEFT, WINDOW_TOPLEFT)

# ---------------------------------------------------------------------------
# FWD-10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", LADDER)
def test_window_sentinel_is_bitwise_the_explicit_causal_band(fwd_build, hdim):
    """The causal sentinels resolve on the device to `(seqlen_q, seqlen_k - seqlen_q)`, so a build fed them must equal
    the same build fed the explicit band, bit for bit, at every rung (both kernel bodies)."""
    sq = 512
    fn = fwd_build(meta_of(head_dim=hdim, window=True))
    dtype = DTYPES["bf16"]
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, sq, hdim, dtype, gen=gen) for _ in range(3))
    got = alloc(1, 4, sq, hdim, dtype)
    run_fwd(fn, q, k, v, got, window=BR)
    want = alloc(1, 4, sq, hdim, dtype)
    run_fwd(fn, q, k, v, want, window=(sq, 0))
    assert torch.equal(got, want)
    # Top-left alignment is the same band when sq == sk.
    tl = alloc(1, 4, sq, hdim, dtype)
    run_fwd(fn, q, k, v, tl, window=TL)
    assert torch.equal(tl, want)


@pytest.mark.parametrize("shape", [(2, 4, 512), (1, 4, 1024)], ids=["s512", "s1024"])
@pytest.mark.parametrize("band", [(31, 0), (127, 0), (511, 0), (63, 63), (0, 0), (255, 32)])
def test_window_matches_masked_reference(fwd_build, shape, band):
    """Real bands against the fp64 reference with the same band as a mask. `(0, 0)` is the degenerate diagonal (one key
    per row) and `(63, 63)` a symmetric band that is not causal at all."""
    b, h, s = shape
    fn = fwd_build(meta_of(head_dim=64, window=True))
    fwd_check(fn, b=b, hq=h, sq=s, d=64, dtype=DTYPES["bf16"], window=band, ctx=f"{shape} {band}")


@pytest.mark.parametrize(
    "band",
    [(-32, 128), (-1, 64), (512, -32), (512, -1), (256, -128), (-32, 0), (-64, 32), (-32, -16)],
)
def test_window_negative_bounds(fwd_build, band):
    """Either bound may be negative, and then some rows keep no key at all: those rows must come back exactly 0 with
    LSE +inf, never NaN (the floor-seeded `reduce_max` keeps every lane off -inf)."""
    fn = fwd_build(meta_of(head_dim=64, window=True))
    r = fwd_check(fn, b=1, hq=4, sq=512, d=64, dtype=DTYPES["bf16"], window=band, ctx=str(band))
    assert not torch.isfinite(r["exact_lse"]).all(), "this case is meant to exercise rows with no live key"


@pytest.mark.parametrize("hdim", [64, 128])
@pytest.mark.parametrize("sq,sk", [(256, 128), (512, 128), (256, 64), (384, 256)])
def test_window_masks_the_kv_tail_when_q_overhangs_k(fwd_build, sq, sk, hdim):
    """K09/K10: `seqlen_q > seqlen_k` under a top-left window. A window re-points `delta` at the resolved right bound,
    so a row at or past `seqlen_k` reaches columns the K buffer does not hold; they read back 0 (a logit, not -inf)
    and take softmax weight. Wrong by 1e-1..3e-1 before the tail mask."""
    fn = fwd_build(meta_of(head_dim=hdim, window=True))
    fwd_check(
        fn,
        b=1,
        hq=2,
        sq=sq,
        sk=sk,
        d=hdim,
        dtype=DTYPES["bf16"],
        window=(WINDOW_BOTRIGHT, 0),
        seed=sq + sk,
        ctx=f"{sq}x{sk}",
    )


# ---------------------------------------------------------------------------
# FWD-11
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", [(-1, 0), (127, 0), (64, 32)])
@pytest.mark.parametrize("hdim", [64, 128, 512])
def test_static_window_is_bitwise_the_runtime_window(fwd_build, hdim, band):
    """STATIC_WINDOW bakes the pair (a cache entry per value); the output is bitwise the runtime window's."""
    dtype = DTYPES["bf16"]
    sq = 300
    gen = seeded(hdim)
    q, k, v = (randn(1, 2, sq, hdim, dtype, gen=gen) for _ in range(3))
    meta = meta_of(head_dim=hdim, window=True)
    want = alloc(1, 2, sq, hdim, dtype)
    run_fwd(fwd_build(meta), q, k, v, want, window=band)
    got = alloc(1, 2, sq, hdim, dtype)
    run_fwd(fwd_build(meta, STATIC_WINDOW=True), q, k, v, got, window=band)
    assert torch.equal(got, want)


def test_static_window_with_sentinels_takes_the_unbounded_left_path(fwd_build):
    """A baked sentinel left edge is vacuous: the build compiles the left bound out and is correct for causal."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, window=True), STATIC_WINDOW=True)
    fwd_check(fn, b=2, hq=4, sq=257, sk=400, d=64, dtype=dtype, window=BR, ctx="static causal")
    fwd_check(fn, b=2, hq=4, sq=257, sk=257, d=64, dtype=dtype, window=TL, ctx="static top-left")


@pytest.mark.parametrize("sk_multiple", [True, False], ids=["tail_free", "ragged"])
@pytest.mark.parametrize("hdim", [64, 128])
def test_static_seqlen_is_bitwise_the_runtime_build(fwd_build, hdim, sk_multiple):
    """STATIC_SEQLEN bakes the lengths (a compile per pair). A build that compiles nothing away (non-causal, ragged
    `sk`) is bit-identical to the runtime build; once the `MASK_ALL_TILES` walk (baked causal window, `sq == sk`) or the tail
    mask (`sk % BLOCK_N == 0`) is gone it is a different program and matches to one bf16 ulp."""
    dtype = DTYPES["bf16"]
    s = 512 if sk_multiple else 300
    gen = seeded(hdim + s)
    q, k, v = (randn(2, 4, s, hdim, dtype, gen=gen) for _ in range(3))
    for window, feats in ((BR, dict(window=True)), (None, {})):
        meta = meta_of(head_dim=hdim, **feats)
        want, got = alloc(2, 4, s, hdim, dtype), alloc(2, 4, s, hdim, dtype)
        run_fwd(fwd_build(meta), q, k, v, want, window=window)
        pins = dict(STATIC_SEQLEN=True, **(dict(STATIC_WINDOW=True) if window else {}))
        run_fwd(fwd_build(meta, **pins), q, k, v, got, window=window)
        if window is None and not sk_multiple:
            assert torch.equal(got, want), (window, s)  # nothing compiled away: the same program
        else:
            # A compiled-away mask (the `MASK_ALL_TILES` walk, or the tail mask) makes it a different program, and
            # `contract|reassoc` fast-math lets a row's sum round differently: measured up to 70 of 262144 elements, one
            # bf16 ulp. Not bit-exact by construction.
            assert (got != want).sum().item() <= 0.001 * got.numel(), (window, s)
            assert torch.allclose(got.float(), want.float(), rtol=2**-7, atol=2**-10), (window, s)


def test_static_seqlen_refuses_varlen_and_cross_lengths_stay_correct(fwd_build):
    """A baked length cannot serve per-sequence lengths: the host refuses a varlen call. With `sq != sk` the cross-length
    machinery stays on, and the answer is still right."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, window=True), STATIC_SEQLEN=True, STATIC_WINDOW=True)
    case = VarlenCase("0x0B0B", [64, 64], [64, 64], 2, 2, 64, dtype)
    with pytest.raises(ValueError, match="STATIC_SEQLEN"):
        case.launch(fn, window=BR)
    fwd_check(fn, b=1, hq=2, sq=130, sk=300, d=64, dtype=dtype, window=BR, ctx="static sq != sk")
    fwd_check(fn, b=1, hq=2, sq=256, sk=256, d=64, dtype=dtype, window=BR, ctx="static sq == sk")


# ---------------------------------------------------------------------------
# FWD-25
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["0x0B0B", "0x0202"])
def test_varlen_bottom_right_per_sequence(fwd_build, mode):
    """K34: bottom-right causal under varlen resolves per sequence, with `k - q` differing per sequence (including
    sq > sk), not against the batch maximum."""
    fn = fwd_build(meta_of(head_dim=64, window=True))
    case = VarlenCase(mode, [96, 40, 130, 17], [64, 40, 200, 90], 4, 4, 64, DTYPES["bf16"], seed=2)
    case.check(fn, window=BR, ctx=mode)
