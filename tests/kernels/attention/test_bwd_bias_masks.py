# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward bias, dB, masks and bounded reads: BWD-09..12, 16 and 29.

Bias with a causal or window mask is rejected everywhere (plan 4.10), so the bias cases here are unmasked and the mask
cases carry no bias.
"""

import pytest
import torch

import flydsl.expr as fx
from kernels.attention.flash_attn_gfx950_helpers import wire_ptr
from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
    VarlenCase,
    alloc,
    bwd_check,
    bwd_floor,
    bwd_reference,
    check_floor,
    dkdv_family_pins,
    meta_of,
    randn,
    row_inputs,
    run_bwd,
    sdpa_scale,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
TL = (WINDOW_TOPLEFT, WINDOW_TOPLEFT)
BF16 = DTYPES["bf16"]

# ---------------------------------------------------------------------------
# BWD-09: the dB zero-stride gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [64, 128])
@pytest.mark.parametrize("rows", [32, 16])
def test_db_zero_stride_gate(bwd_build, hdim, rows):
    """AOTriton passes a null dB and an all-zero stride triple when the bias needs no gradient, and the kernel must treat
    that as "do not store": (a) a null dB with zero strides: no fault, dQ correct; (b) a real canary buffer with zero
    strides: left untouched; (c) a real dB with real strides: dB correct. Launched through the compiled launcher with the
    arguments rewritten, the way a C++ caller's are."""
    b, h, s = 1, 2, 130
    meta = meta_of(head_dim=hdim, bias=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=dkdv_family_pins(hdim, rows))
    gen = seeded(3)
    q, k, v, do = (randn(b, h, s, hdim, BF16, gen=gen) for _ in range(4))
    bias = randn(b, h, s, s, BF16, gen=gen).contiguous()

    # (c) the control: a real dB
    full = bwd_check(
        builds, b=b, hq=h, sq=s, d=hdim, dtype=BF16, bias=bias, want_db=True, qkv_do=(q, k, v, do), ctx="dB"
    )
    lse2, delta2 = row_inputs(full["o"], do, full["lse"])
    want_dq = full["dq"]

    def launch(db, zero_strides, null_pointer):
        dq = alloc(b, h, s, hdim, BF16)
        packed, extras = builds.dq.host_args(q, k, v, do, dq, lse2, delta2, b, s, seqlen_k=s, bias=bias, db=db)
        packed = list(packed)
        if zero_strides:
            packed[-3:] = [0, 0, 0]
        if null_pointer:
            packed[6] = wire_ptr(None, fx.BFloat16)
        builds.dq.launcher(*packed, fx.Stream(extras[0]))
        torch.cuda.synchronize()
        return dq

    canary = torch.full((b, h, s, s), 3.0, device="cuda", dtype=BF16)
    dq_null = launch(canary.clone(), zero_strides=True, null_pointer=True)  # (a)
    assert torch.equal(dq_null, want_dq), "(a) a null dB with zero strides changed dQ"
    held = canary.clone()
    dq_held = launch(held, zero_strides=True, null_pointer=False)  # (b)
    assert torch.equal(held, canary), "(b) a real dB buffer with zero strides must be left untouched"
    assert torch.equal(dq_held, want_dq)


# ---------------------------------------------------------------------------
# BWD-10: dB and the bias path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sk", [130, 131, 301])
def test_db_matches_fp64_ragged_kv(bwd_build, sk):
    """dB = dS against the fp64 one, including an odd `seqlen_k` (the last column straddles a dword in the store)."""
    b, h, sq, d = 1, 2, 100, 64
    bias = randn(b, h, sq, sk, BF16, gen=seeded(4)).contiguous()
    out = bwd_check(
        bwd_build(meta_of(head_dim=d, bias=True)),
        b=b,
        hq=h,
        sq=sq,
        sk=sk,
        d=d,
        dtype=BF16,
        bias=bias,
        want_db=True,
        ctx=f"dB sk={sk}",
    )
    assert out["db"].shape == (b, h, sq, sk)


def test_db_under_dropout_is_the_dropped_ds(bwd_build):
    """dB = dS where dS is built from `dP` *after* the dropout mask: getting that backwards is a silently wrong bias
    gradient that no shape check sees. The mask is recovered from the forward (V = I), so the reference drops the very
    same elements."""
    b, h, s, d, p = 1, 2, 128, 128, 0.3
    meta = meta_of(head_dim=d, bias=True, dropout=True)
    builds = bwd_build(meta)
    gen = seeded(6)
    q, k, do = (randn(b, h, s, d, BF16, gen=gen) for _ in range(3))
    bias = randn(b, h, s, s, BF16, gen=gen).contiguous()
    v = torch.eye(s, device="cuda", dtype=BF16).expand(b, h, s, s).contiguous()
    out = run_bwd(builds, q, k, v, do, bias=bias, p_drop=p, seed=17, want_db=True)
    keep = out["o"] != 0
    assert abs(keep.float().mean().item() - (1 - p)) < 0.03
    sm = sdpa_scale(d)
    ref = bwd_reference(q, k, v, do, sm, bias=bias, keep=keep, p_drop=p)
    floors = bwd_floor(lambda **kw: bwd_reference(q, k, v, do, sm, bias=bias, keep=keep, p_drop=p, **kw), ref, BF16)
    check_floor("db", out["db"], ref[3], floors[3], "dB under dropout")
    check_floor("dq", out["dq"], ref[0], floors[0], "dQ under dropout")


def test_single_key_dv_equals_do(bwd_build):
    """One query, one key: P is exactly 1, so `dV = P^T dO = dO` bit for bit. A `log2e` applied in low precision (AOTriton
    lesson 4) breaks exactly this."""
    dtype = BF16
    b, h, d = 1, 2, 64
    gen = seeded(12)
    q, k, v, do = (randn(b, h, 1, d, dtype, gen=gen) for _ in range(4))
    out = run_bwd(bwd_build(meta_of(head_dim=d)), q, k, v, do)
    assert torch.equal(out["dv"], do), "single-key dV must equal dO exactly"


def test_bias_free_build_equals_a_zero_bias_build(bwd_build):
    """A build without bias and one with a zero bias compute the same gradients (dQ/dK/dV), bit for bit."""
    b, h, s, d = 1, 2, 130, 64
    gen = seeded(14)
    q, k, v, do = (randn(b, h, s, d, BF16, gen=gen) for _ in range(4))
    plain = run_bwd(bwd_build(meta_of(head_dim=d)), q, k, v, do)
    zero = torch.zeros(b, h, s, s, device="cuda", dtype=BF16)
    biased = run_bwd(bwd_build(meta_of(head_dim=d, bias=True)), q, k, v, do, bias=zero, want_db=True)
    for name in ("dq", "dk", "dv"):
        assert torch.equal(plain[name], biased[name]), name


# ---------------------------------------------------------------------------
# BWD-11: -inf in the bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [64, 384])
def test_bias_minus_inf(bwd_build, hdim):
    """`-inf` entries and a whole `-inf` row ("never attend here"): the dead row's dQ and dB are exactly 0, dK/dV stay
    finite and correct for the live ones, and nothing is NaN. `-inf` is real arithmetic here (the first place in the kernel
    that is), so this is where a stray `ninf` flag would show."""
    b, h, s = 1, 2, 130
    gen = seeded(15)
    bias = randn(b, h, s, s, BF16, gen=gen).contiguous()
    bias[:, :, :, 5] = float("-inf")  # one key nobody attends
    bias[:, :, 7, :] = float("-inf")  # one query attending nothing
    out = bwd_check(
        bwd_build(meta_of(head_dim=hdim, bias=True)),
        b=b,
        hq=h,
        sq=s,
        d=hdim,
        dtype=BF16,
        bias=bias,
        want_db=True,
        ctx=f"-inf bias d{hdim}",
    )
    assert (out["dq"][:, :, 7] == 0).all() and (out["db"][:, :, 7] == 0).all()
    assert (out["dk"][:, :, 5] == 0).all() and (out["dv"][:, :, 5] == 0).all()
    for name in ("dq", "dk", "dv", "db"):
        assert not torch.isnan(out[name]).any(), name


# ---------------------------------------------------------------------------
# BWD-12: fully masked rows, from the forward's own LSE (+inf)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["bf16", "f16"])
@pytest.mark.parametrize("hdim", [64, 512])
@pytest.mark.parametrize("seqs", [(301, 190), (1024, 128)], ids=["tile_and_rows", "whole_tiles"])
def test_fully_masked_rows_bwd(bwd_build, seqs, hdim, dt):
    """Bottom-right causal with `Sq > Sk`: the first `Sq - Sk` rows attend nothing. Their LSE is `+inf` (the forward writes
    it, the backward needs it) so `exp(s - lse)` is 0: dQ of those rows is exactly 0, in a whole masked tile (1024 x 128)
    and for rows sharing a tile with live ones (301 x 190); no NaN anywhere."""
    sq, sk = seqs
    dtype = DTYPES[dt]
    out = bwd_check(
        bwd_build(meta_of(head_dim=hdim, dtype_str=dt, window=True)),
        b=1,
        hq=2,
        sq=sq,
        sk=sk,
        d=hdim,
        dtype=dtype,
        window=BR,
        ctx=f"{seqs} d{hdim} {dt}",
    )
    dead = sq - sk
    assert (out["dq"][:, :, :dead] == 0).all(), "dQ of fully masked rows must be exactly 0"
    assert torch.isinf(out["lse"][:, :, :dead]).all() and (out["lse"][:, :, :dead] > 0).all()
    for name in ("dq", "dk", "dv"):
        assert not torch.isnan(out[name]).any(), name


def test_unattended_keys_have_exactly_zero_dk_dv(bwd_build):
    """Top-left causal with `Sq < Sk`: the last `Sk - Sq` keys are attended by no row, so their dK and dV are exactly 0."""
    sq, sk = 130, 301
    out = bwd_check(
        bwd_build(meta_of(head_dim=64, window=True)),
        b=1,
        hq=2,
        sq=sq,
        sk=sk,
        d=64,
        dtype=BF16,
        window=TL,
        ctx="top-left",
    )
    assert (out["dk"][:, :, sq:] == 0).all() and (out["dv"][:, :, sq:] == 0).all()


# ---------------------------------------------------------------------------
# BWD-16: rows past seqlen_q read a bounded LSE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [64, 256])
def test_rows_past_seqlen_read_a_bounded_lse(bwd_build, hdim):
    """In a packed batch the q rows past one sequence's end are the next sequence's rows, and a `seqlen_q % BLOCK_M != 0`
    tail block reads their LSE. Set the next sequence's LSE to -1e30 and `+inf` alternately: the first sequence's dV is
    still finite and still matches fp64, because the kernel masks those rows' probabilities rather than trusting what it
    read."""
    meta = meta_of(head_dim=hdim)
    builds = bwd_build(meta)
    case = VarlenCase("0x0B0B", [70, 64], [70, 64], 2, 2, hdim, BF16)
    do = case.new_do()
    case.launch(builds.fwd)
    lse, delta = case.row_inputs(do)
    lse = lse.clone()
    seg = lse[:, 70:]
    seg[:, 0::2] = -1e30
    seg[:, 1::2] = float("inf")
    case.run_bwd(builds, do, lse_delta=(lse, delta), which=("dkdv",))
    q, k, v = case.seq(0)
    sm = sdpa_scale(hdim)
    qb, qr = case._where_q(0)
    ref = bwd_reference(q, k, v, do[qb : qb + 1, :, qr], sm)
    floors = bwd_floor(lambda **kw: bwd_reference(q, k, v, do[qb : qb + 1, :, qr], sm, **kw), ref, BF16)
    got = case.grad_of("dv", 0)
    assert torch.isfinite(got).all()
    check_floor("dv", got, ref[2], floors[2], f"seq 0 d{hdim}")


# ---------------------------------------------------------------------------
# BWD-29: windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sq,sk", [(256, 256), (150, 300), (300, 150)])
def test_sentinel_window_is_bitwise_the_explicit_causal_band(bwd_build, sq, sk):
    """The bottom-right sentinel resolves on the device to `(sq, sk - sq)` (and the top-left one to `(sq, 0)`): a build fed
    the sentinels equals the same build fed the explicit band, bit for bit, in all three gradients."""
    meta = meta_of(head_dim=64, window=True)
    builds = bwd_build(meta)
    gen = seeded(sq + sk)
    q, k, v, do = (
        randn(1, 2, sq, 64, BF16, gen=gen),
        randn(1, 2, sk, 64, BF16, gen=gen),
        randn(1, 2, sk, 64, BF16, gen=gen),
        randn(1, 2, sq, 64, BF16, gen=gen),
    )
    got = run_bwd(builds, q, k, v, do, window=BR)
    # the forward's LSE/O are the same by construction (same window): reuse them so only the backward differs
    want = run_bwd(builds, q, k, v, do, window=(sq, sk - sq), o_lse=(got["o"], got["lse"]))
    for name in ("dq", "dk", "dv"):
        assert torch.equal(got[name], want[name]), name
    tl = run_bwd(builds, q, k, v, do, window=TL, o_lse=(got["o"], got["lse"])) if sq == sk else None
    if tl is not None:
        for name in ("dq", "dk", "dv"):
            assert torch.equal(got[name], tl[name]), f"top-left == bottom-right at Sq == Sk: {name}"


@pytest.mark.parametrize("band", [(31, 0), (127, 0), (63, 63), (0, 0), (255, 32)])
@pytest.mark.parametrize("rows", [32, 16])
def test_band_windows(bwd_build, band, rows):
    """Real bands against the fp64 reference with the same band as a mask, in both families: `(0, 0)` is one key per row
    and `(63, 63)` a symmetric band that is not causal at all. The tile cut (skipping tiles the band cannot reach) is not
    inert for a narrow band, which is what makes this a test of it."""
    meta = meta_of(head_dim=64, window=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=dkdv_family_pins(64, rows))
    # `(0, 0)` is one key per row: P is exactly 1, so dS is exactly 0 in the reference and fp32 noise in the kernel, where a
    # relative floor of 0 has nothing to say (the same degenerate case as a single key). dV keeps the floor gate.
    degenerate = band == (0, 0)
    out = bwd_check(
        builds,
        b=1,
        hq=2,
        sq=300,
        d=64,
        dtype=BF16,
        window=band,
        ctx=f"band {band} rows{rows}",
        outputs=("dv",) if degenerate else ("dq", "dk", "dv"),
    )
    if degenerate:
        assert out["dq"].abs().max().item() < 1e-4 and out["dk"].abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# STATIC_WINDOW / STATIC_SEQLEN: the Leading_upper_snake_case parameters, baked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "static", [("STATIC_WINDOW",), ("STATIC_SEQLEN",), ("STATIC_WINDOW", "STATIC_SEQLEN")], ids="+".join
)
def test_static_bwd_builds_are_bitwise_the_runtime_build(bwd_build, static):
    """`Window_left/Window_right` and `Max_seqlen_q/Max_seqlen_k` are one parameter each whose annotation is `Constexpr` when
    the knob is on and `Int32` otherwise (AOTriton's `constexpr_or_i32`): a JIT build that bakes them computes exactly what the
    runtime build does, in dQ and dK/dV alike."""
    meta = meta_of(head_dim=64, window=True)
    pins = {name: True for name in static}
    runtime = bwd_build(meta)
    baked = bwd_build(meta, dq=pins, dkdv=pins)
    assert baked.dq is not runtime.dq and baked.dkdv is not runtime.dkdv
    gen = seeded(33)
    q, k, v, do = (randn(1, 2, 200, 64, BF16, gen=gen) for _ in range(4))
    want = run_bwd(runtime, q, k, v, do, window=BR)
    got = run_bwd(baked, q, k, v, do, window=BR, o_lse=(want["o"], want["lse"]))
    for name in ("dq", "dk", "dv"):
        assert torch.equal(got[name], want[name]), name


def test_static_seqlen_bwd_refuses_varlen(bwd_build):
    """A baked length cannot serve per-sequence lengths: a varlen call on such a build is refused by the host."""
    meta = meta_of(head_dim=64)
    baked = bwd_build(meta, dq=dict(STATIC_SEQLEN=True), dkdv=dict(STATIC_SEQLEN=True))
    case = VarlenCase("0x0B0B", [64, 64], [64, 64], 2, 2, 64, BF16)
    do = case.new_do()
    with pytest.raises(ValueError, match="STATIC_SEQLEN"):
        case.run_bwd(baked, do)
