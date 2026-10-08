# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward head dims, layouts, store bounds and refusals: BWD-05..08, 27 and 28.

Every output is NaN-prefilled, so a gradient element the kernel never wrote (or wrote to the wrong place) shows up; every
input has NaN slack on the D axis, so a read past the granted `ceil8(head_dim)` shows up as NaN in the gradient.
"""

import pytest
import torch

from kernels.attention import flash_attn_gfx950_config as cfg
from tests.kernels.attention.attn_testlib import (
    DTYPES,
    PERMS,
    WINDOW_BOTRIGHT,
    bwd_check,
    ceil8,
    dkdv_family_pins,
    fwd_o_lse,
    meta_of,
    randn,
    row_inputs,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
BF16 = DTYPES["bf16"]

# ---------------------------------------------------------------------------
# BWD-05: asymmetric head dims, each output on its own width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [32, 16])
@pytest.mark.parametrize("dims", [(120, 8), (24, 152), (64, 32), (128, 64), (192, 128), (100, 40)], ids=str)
def test_asymmetric_hdim_own_output_views(bwd_build, dims, rows):
    """`hdim_qk != hdim_vo`: dQ and dK carry the qk extent, dV the vo one, in one compiled tile of the wider rung. NaN
    prefilled outputs and the fp64 floor: a store that used the other tensor's extent would leave NaN or overwrite a
    neighbour. (The 16-row family needs a tile that is a multiple of 64, so it is skipped where the rung is not.)"""
    d, dv = dims
    block = cfg.tile_width_for(max(d, dv))
    if rows == 16 and block % 64:
        pytest.skip("the 16-row family serves rungs that are multiples of 64")
    pins = dict(BLOCK_DMODEL=block)
    pins_dkdv = {**pins, **(dkdv_family_pins(block, rows) or {})} if rows == 16 else dict(pins)
    meta = meta_of(head_dim=d, head_dim_v=dv)
    builds = bwd_build(meta, fwd=pins, dq={**pins, "MFMA_ROWS": rows}, dkdv=pins_dkdv)
    bwd_check(builds, b=1, hq=2, sq=130, d=d, dv=dv, dtype=BF16, ctx=f"{dims} rows{rows}")


# ---------------------------------------------------------------------------
# BWD-06: prime head dims, and every tensor on its own layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feat", ["dense", "bias", "causal"])
@pytest.mark.parametrize("hdim", [53, 179])
def test_prime_hdim_and_layouts_bwd(bwd_build, hdim, feat):
    """A prime head dim (8xD slack, not a multiple of anything) with Q, K, V, dO, O, dQ, dK, dV and (for a bias build) the
    bias and dB each on a *different* physical axis order. Every tensor has its own strides; a kernel that reuses one
    tensor's for another passes the symmetric layout and fails here."""
    window = BR if feat == "causal" else None
    meta = meta_of(head_dim=hdim, window=window is not None, bias=feat == "bias")
    b, h, s = 2, 2, 100
    perms = [PERMS[(i + hdim) % 6] for i in range(10)]
    kw = {}
    if feat == "bias":
        kw["bias"] = randn(b, h, s, s, BF16, perms[8], gen=seeded(1))
        kw["want_db"] = True
        kw["perm_db"] = perms[9]
    bwd_check(
        bwd_build(meta),
        b=b,
        hq=h,
        sq=s,
        d=hdim,
        dtype=BF16,
        window=window,
        in_perms=tuple(perms[:4]),
        perm_o=perms[4],
        perms=tuple(perms[5:8]),
        ctx=f"d{hdim} {feat}",
        **kw,
    )


# ---------------------------------------------------------------------------
# BWD-07: nothing is written past a tensor's extent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [64, 100])
@pytest.mark.parametrize("sq,sk", [(130, 130), (70, 200)])
def test_store_bounds(bwd_build, hdim, sq, sk):
    """dQ, dK and dV are written into views of larger allocations whose rows past `seqlen` hold a canary: after the launch
    every canary is intact (`seqlen_q` for dQ, `seqlen_k` for dK/dV, with a ragged last block), and the live rows are
    finite."""
    meta = meta_of(head_dim=hdim)
    builds = bwd_build(meta)
    b, h = 1, 2
    gen = seeded(5)
    q, do = (randn(b, h, sq, hdim, BF16, gen=gen) for _ in range(2))
    k, v = (randn(b, h, sk, hdim, BF16, gen=gen) for _ in range(2))
    o, lse = fwd_o_lse(builds.fwd, q, k, v)
    lse2, delta2 = row_inputs(o, do, lse)
    canary = 7.0

    def out_view(rows):
        big = torch.full((b, h, rows + 70, ceil8(hdim)), canary, device="cuda", dtype=BF16)
        return big, big[:, :, :rows, :hdim]

    big_dq, dq = out_view(sq)
    big_dk, dk = out_view(sk)
    big_dv, dv = out_view(sk)
    for t in (dq, dk, dv):
        t.fill_(float("nan"))
    builds.dq(q, k, v, do, dq, lse2, delta2, b, sq, seqlen_k=sk)
    builds.dkdv(q, k, v, do, dk, dv, lse2, delta2, b, sq, seqlen_k=sk)
    torch.cuda.synchronize()
    for name, big, t, rows in (("dQ", big_dq, dq, sq), ("dK", big_dk, dk, sk), ("dV", big_dv, dv, sk)):
        assert torch.isfinite(t).all(), f"{name}: a live element was never written"
        assert (big[:, :, rows:] == canary).all(), f"{name} wrote past its last row"


# ---------------------------------------------------------------------------
# BWD-08: the 8xD contract, refused by both kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [20, 73])
@pytest.mark.parametrize("layout", ["tight_bhsd", "compact_bshd"])
def test_8xd_refusals(bwd_build, hdim, layout):
    """The kernels touch `ceil8(head_dim)` columns per row, so an odd head dim needs the caller's slack. A tight
    `(B, H, S, D)` allocation, or a BSHD-compact one (consecutive heads of a token adjacent: a pitch check alone waves it
    through while the store corrupts the next head), is refused by **both** the dQ and the dK/dV host paths."""
    builds = bwd_build(meta_of(head_dim=hdim))
    b, h, s = 1, 2, 64

    def tight():
        if layout == "tight_bhsd":
            return torch.zeros(b, h, s, hdim, device="cuda", dtype=BF16)
        return torch.zeros(b, s, h, hdim, device="cuda", dtype=BF16).transpose(1, 2)

    q, k, v, do, dq, dk, dv = (tight() for _ in range(7))
    lse2 = torch.zeros(b * h, s, device="cuda")
    delta2 = torch.zeros(b * h, s, device="cuda")
    with pytest.raises(ValueError, match="8-wide"):
        builds.dq(q, k, v, do, dq, lse2, delta2, b, s)
    with pytest.raises(ValueError, match="8-wide"):
        builds.dkdv(q, k, v, do, dk, dv, lse2, delta2, b, s)


# ---------------------------------------------------------------------------
# BWD-27: slab offsets past 2 GiB
# ---------------------------------------------------------------------------


@pytest.mark.large_shape
@pytest.mark.parametrize("rows", [32, 16])
def test_offsets_past_2gi(bwd_build, rows):
    """A (batch, head) slab whose base byte offset is above 2^31 (a late head of a large tensor): the buffer descriptors take
    an i64 base, and an `i32` byte offset anywhere in the addressing wraps there. A narrow band keeps the work small while
    the tensors are ~2 GiB each; the *last* slab's last rows (and the keys they alone cover) are gated against the fp64
    reference computed on just those rows and keys."""
    from tests.kernels.attention.attn_testlib import (
        bwd_floor,
        bwd_reference,
        check_floor,
        run_bwd,
        sdpa_scale,
        window_mask,
    )

    b, h, s, d, band = 2, 64, 65536, 128, 255
    assert (b * h - 1) * s * d * 2 > 2**31
    gen = seeded(27)
    meta = meta_of(head_dim=d, window=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=dkdv_family_pins(d, rows))
    q, k, v, do = (randn(b, h, s, d, BF16, gen=gen) for _ in range(4))
    out = run_bwd(builds, q, k, v, do, window=(band, 0))
    rows_n, keys_n = 512, 512 + band
    sl = (slice(b - 1, b), slice(h - 1, h))
    qs, dos = q[sl][:, :, s - rows_n :], do[sl][:, :, s - rows_n :]
    ks, vs = k[sl][:, :, s - keys_n :], v[sl][:, :, s - keys_n :]
    mask = window_mask(rows_n, keys_n, 0, band)  # local frame: key - row in [0, band]
    sm = sdpa_scale(d)

    def ref(**kw):
        return bwd_reference(qs, ks, vs, dos, sm, mask=mask, **kw)

    exact = ref()
    floors = bwd_floor(ref, exact, BF16)
    got_dq = out["dq"][sl][:, :, s - rows_n :]
    check_floor("dq", got_dq, exact[0], floors[0], f"2Gi rows{rows}")
    # keys [S - 512, S - 256] are fed only by rows inside the slice (local 255..511 of the keys)
    lo, hi = band, keys_n - band
    for i, name in ((1, "dk"), (2, "dv")):
        got = out[name][sl][:, :, s - keys_n :][:, :, lo:hi]
        check_floor(name, got, exact[i][:, :, lo:hi], floors[i], f"2Gi {name} rows{rows}")


# ---------------------------------------------------------------------------
# BWD-28: refusals
# ---------------------------------------------------------------------------


def test_builder_refusals():
    """What the backward does not serve is refused where it is decided, by name: paged KV and ALiBi (NotImplementedError;
    the backward serves the forward's training inputs), a head dim past the widest rung, a D split (not a dQ or dK/dV knob),
    and bias with a mask (undefined, everywhere)."""
    base = dict(dtype_str="bf16", head_dim=64)
    for make in (cfg.dq_knobs, cfg.dkdv_knobs):
        with pytest.raises(NotImplementedError, match="paged"):
            make("gfx950").resolve(cfg.FmhaInputMetadata(**base, paged=True))
        with pytest.raises(NotImplementedError, match="alibi"):
            make("gfx950").resolve(cfg.FmhaInputMetadata(**base, alibi=True))
        with pytest.raises(NotImplementedError, match="sink"):
            make("gfx950").resolve(cfg.FmhaInputMetadata(**base, sink=True))
        with pytest.raises(ValueError, match="exceeds the widest tile"):
            make("gfx950").resolve(cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=513))
        with pytest.raises(ValueError, match="mutually exclusive"):
            make("gfx950").resolve(cfg.FmhaInputMetadata(**base, bias=True, window=True))
    with pytest.raises(TypeError, match="D_STAGES"):
        cfg.dq_knobs("gfx950", D_STAGES=2)
    with pytest.raises(ValueError, match="does not shard"):
        cfg.dkdv_knobs("gfx950", MFMA_ROWS=16, DKV_SHARDS=2).resolve(cfg.FmhaInputMetadata(**base))


def test_host_refusals(bwd_build):
    """A padded call into an unpadded build, mismatched row tensors, and a missing logsumexp are refused by the host."""
    builds = bwd_build(meta_of(head_dim=64))
    b, h, s = 1, 2, 64
    q, k, v, do, dq = (randn(b, h, s, 64, BF16) for _ in range(5))
    narrow = randn(b, h, s, 32, BF16)
    lse2 = torch.zeros(b * h, s, device="cuda")
    delta2 = torch.zeros(b * h, s, device="cuda")
    with pytest.raises(ValueError, match="not compiled for a padded head"):
        builds.dq(narrow, narrow, narrow, narrow, narrow, lse2, delta2, b, s)
    with pytest.raises(ValueError, match="logsumexp is required"):
        builds.dq(q, k, v, do, dq, None, delta2, b, s)
    with pytest.raises(ValueError, match=r"delta with LSE_LAYOUT_HT wants"):
        builds.dq(q, k, v, do, dq, lse2, delta2[:, :32].contiguous(), b, s)
    with pytest.raises(ValueError, match="logsumexp must be"):
        builds.dq(q, k, v, do, dq, lse2[:1].contiguous(), delta2, b, s)
