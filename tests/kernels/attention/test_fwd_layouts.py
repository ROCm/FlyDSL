# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward addressing: head-dim padding and the 8xD contract, memory layouts, bias slabs (FWD-12..18, FWD-39).

Shape is the ABI and layout is not: any permutation of the outer axes with D innermost is accepted, the strides are
read rather than derived, and the D axis is granted `ceil8(head_dim)` contiguous elements on every row and nothing else.
Every tensor here carries NaN slack so a read or write past a row is visible.
"""

import itertools

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    PERMS,
    WINDOW_BOTRIGHT,
    alloc,
    ceil8,
    check_floor,
    fwd_check,
    layouts,
    meta_of,
    randn,
    reference,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)

# ---------------------------------------------------------------------------
# FWD-12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("poison", [0.0, 1e4, float("inf"), float("nan")], ids=["zero", "big", "inf", "nan"])
@pytest.mark.parametrize("hdim", [17, 33, 80, 100, 113, 129, 193, 225, 241])
def test_padded_head_ignores_pad_contents(fwd_build, hdim, poison):
    """K11: the answer must not depend on what sits in the D-axis padding. Masking Q alone is enough for a finite pad
    (`0 * x`), not for NaN or Inf, so K is masked too. The wide widths sit at `floor + 1`, so any off-by-one in
    HDIM_QK_FLOOR lets poison straight through."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim))
    gen = seeded(hdim)
    q, k, v = (randn(2, 4, 256, hdim, dtype, gen=gen, poison=poison) for _ in range(3))
    fwd_check(fn, b=2, hq=4, sq=256, d=hdim, dtype=dtype, qkv=(q, k, v), ctx=f"{hdim} poison={poison}")


def test_hdim_at_or_below_the_floor_is_refused(fwd_build):
    """The host rejects `hdim <= floor` rather than computing it: the kernel skips masking the columns at or below."""
    fn = fwd_build(meta_of(head_dim=200))  # 224 rung, floor 192
    assert fn.traits.HDIM_QK_FLOOR == 192
    dtype = DTYPES["bf16"]
    q, k, v = (randn(1, 2, 64, 120, dtype) for _ in range(3))
    o = alloc(1, 2, 64, 120, dtype)
    with pytest.raises(ValueError, match="serves hdim_qk in"):
        run_fwd(fn, q, k, v, o)


def test_unpadded_build_refuses_a_narrower_call(fwd_build):
    """A build that is not padded serves exactly its width: a narrower call would reduce over the caller's padding."""
    fn = fwd_build(meta_of(head_dim=128))
    assert not fn.knobs.PADDED_HEAD
    dtype = DTYPES["bf16"]
    q, k, v = (randn(1, 2, 64, 120, dtype) for _ in range(3))
    o = alloc(1, 2, 64, 120, dtype)
    with pytest.raises(ValueError, match="not padded"):
        run_fwd(fn, q, k, v, o)


@pytest.mark.parametrize("hdim,hdim_v", [(128, 64), (96, 48), (64, 32), (128, 96), (192, 128), (256, 128), (384, 256)])
def test_asymmetric_hdim(fwd_build, hdim, hdim_v):
    """`hdim_qk != hdim_vo`: each output is bounded by its own head dim and never written past it."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim, head_dim_v=hdim_v))
    fwd_check(fn, b=2, hq=4, sq=300, d=hdim, dv=hdim_v, dtype=dtype, ctx=f"{hdim}/{hdim_v}")


# ---------------------------------------------------------------------------
# FWD-13
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("hdim", [7, 73, 241])
def test_prime_hdim_8xd_contract(fwd_build, hdim, causal, dtype_str):
    """K13/K12: `ceil8` allocation with NaN slack, NaN-prefilled O/LSE. The descriptor must cover the last row's tail
    chunk (gfx950 range-checks a multi-dword buffer op per dword), so `O[.., sq-1, hdim-1]` is finite."""
    dtype = DTYPES[dtype_str]
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=hdim, window=causal))
    r = fwd_check(
        fn, b=3, hq=5, sq=257, sk=571, d=hdim, dtype=dtype, window=BR if causal else None, ctx=f"{hdim} {dtype_str}"
    )
    assert torch.isfinite(r["o"][:, :, -1, hdim - 1]).all()


# ---------------------------------------------------------------------------
# FWD-14
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feat", ["dense", "bias", "causal"])
@pytest.mark.parametrize("hdim", [53, 179])
@pytest.mark.parametrize("case", range(6))
def test_memory_layouts_round_robin(fwd_build, case, hdim, feat):
    """K12/K13/K15, lesson 10, G11: Q, K, V, O and the bias each get a different outer-axis permutation per case,
    round-robin over the six permutations, with NaN slack on the D axes."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim, bias=feat == "bias", window=feat == "causal"))
    pq, pk, pv, po, pb = layouts(5, case)
    b, h, sq, sk = 3, 5, 257, 571
    gen = seeded(case)
    q = randn(b, h, sq, hdim, dtype, pq, gen=gen)
    k = randn(b, h, sk, hdim, dtype, pk, gen=gen)
    v = randn(b, h, sk, hdim, dtype, pv, gen=gen)
    bias = None
    if feat == "bias":
        bias = alloc(b, h, sq, sk, dtype, pb, fill=torch.randn(b, h, sq, sk, device="cuda", generator=gen).to(dtype))
    fwd_check(
        fn,
        b=b,
        hq=h,
        sq=sq,
        sk=sk,
        d=hdim,
        dtype=dtype,
        qkv=(q, k, v),
        perms=(pq, pk, pv, po),
        bias=bias,
        window=BR if feat == "causal" else None,
        ctx=f"layouts case {case}",
    )


# ---------------------------------------------------------------------------
# FWD-15
# ---------------------------------------------------------------------------

_KINDS = ("bhsd", "bshd")


def _kind_layout(kind, b, h, s, d, dtype, gap=0, gen=None):
    """A `(B, H, S, D)`-shaped tensor whose memory is laid out as `kind`; `gap` over-allocates the sequence axis and
    slices it back, so the strides stop following from the shape (the case a kernel deriving strides gets wrong)."""
    perm = (0, 1, 2) if kind == "bhsd" else (0, 2, 1)
    big = randn(b, h, s + gap, d, dtype, perm, gen=gen)
    return big[:, :, :s, :]


@pytest.mark.parametrize("ql,kl,vl,ol", list(itertools.product(_KINDS, repeat=4)))
def test_every_qkvo_layout_combination(fwd_build, ql, kl, vl, ol):
    """All 16 layouts of the four tensors, chosen independently (the strides are runtime arguments: one build)."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64))
    gen = seeded(1)
    b, h, s, d = 2, 4, 256, 64
    q, k, v = (_kind_layout(kd, b, h, s, d, dtype, gen=gen) for kd in (ql, kl, vl))
    po = (0, 1, 2) if ol == "bhsd" else (0, 2, 1)
    fwd_check(
        fn, b=b, hq=h, sq=s, d=d, dtype=dtype, qkv=(q, k, v), perms=(None, None, None, po), ctx=f"{ql}{kl}{vl}{ol}"
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_layout_with_gapped_outer_strides(fwd_build, kind, causal):
    """Strides that do not follow from the shape must still be honoured: slicing an over-allocated sequence axis
    breaks the coincidence that a head stride is `seq * head_dim` on all four tensors at once."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, window=causal))
    gen = seeded(2)
    b, h, s, d = 2, 4, 256, 64
    q, k, v = (_kind_layout(kind, b, h, s, d, dtype, gap=37, gen=gen) for _ in range(3))
    assert q.stride(1) not in (s * d, d) or q.stride(0) != h * s * d
    fwd_check(fn, b=b, hq=h, sq=s, d=d, dtype=dtype, qkv=(q, k, v), window=BR if causal else None, ctx=f"gapped {kind}")


@pytest.mark.parametrize("ql,kl", list(itertools.product(_KINDS, repeat=2)))
def test_layout_combinations_under_gqa(fwd_build, ql, kl):
    """K/V carry `num_kv_heads`, so their head stride differs from Q's by shape: a kernel that reused Q's head stride
    for K would pass every MHA layout test and fail only under GQA."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, window=True))
    gen = seeded(3)
    b, hq, hk, s, d = 2, 8, 2, 256, 64
    q = _kind_layout(ql, b, hq, s, d, dtype, gen=gen)
    k, v = (_kind_layout(kl, b, hk, s, d, dtype, gen=gen) for _ in range(2))
    fwd_check(fn, b=b, hq=hq, hk=hk, sq=s, d=d, dtype=dtype, qkv=(q, k, v), window=BR, ctx=f"gqa {ql}{kl}")


def test_stride_slots_are_batch_head_seq(fwd_build):
    """The three stride slots mean (batch, head, seq), in that order: swapping head with seq gives finite garbage and
    never faults, and a square case where the two coincide would hide it. Here `h != s`."""
    fn = fwd_build(meta_of(head_dim=64))
    b, h, s, d = 2, 4, 256, 64
    assert h != s
    fwd_check(fn, b=b, hq=h, sq=s, d=d, dtype=DTYPES["bf16"], perms=(PERMS[3],) * 4, ctx="stride slots")


# ---------------------------------------------------------------------------
# FWD-16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [8, 24, 72, 136, 504])
def test_grid8_contiguous_is_exact_and_writes_nothing_past_o(fwd_build, hdim):
    """A plainly contiguous 8xD tensor (tight pitch, no padded view) must just work, and the canary row contiguous
    with the last real O row stays untouched."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim))
    gen = seeded(hdim)
    b, h, s = 1, 4, 256
    q, k, v = (torch.randn(b, h, s, hdim, device="cuda", dtype=dtype, generator=gen) for _ in range(3))
    assert q.stride(2) == hdim
    sentinel = -12345.0
    obuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=dtype)
    o = obuf[:, :, :s, :]
    run_fwd(fn, q, k, v, o)
    assert torch.all(obuf[:, :, s, :] == sentinel), "a store ran past the last O row"
    exact = reference(q, k, v, hdim**-0.5)[0]
    check_floor("O", o, exact, 2e-3, f"grid8 {hdim}", mult=4.0)


def test_tight_odd_hdim_is_refused_not_corrupted(fwd_build):
    """An odd head_dim in a tight allocation has nowhere to put the tail chunk: refusing is the contract."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=100))
    q, k, v = (torch.randn(1, 4, 256, 100, device="cuda", dtype=dtype) for _ in range(3))
    o = torch.empty(1, 4, 256, 100, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="multiple of 8"):
        run_fwd(fn, q, k, v, o)


def test_odd_hdim_bshd_without_slack_is_refused(fwd_build):
    """BSHD hides the overrun from a pitch check: heads of one token are adjacent, so there is no slack at all while
    `stride(2)` is a tidy multiple of 8. `check_8xd` looks at the smallest outer stride, not the pitch."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=100))
    h = 4
    assert (h * 100) % 8 == 0
    q, k, v = (torch.randn(1, 256, h, 100, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(3))
    o = torch.empty(1, 256, h, 100, device="cuda", dtype=dtype).transpose(1, 2)
    with pytest.raises(ValueError, match="multiple of 8"):
        run_fwd(fn, q, k, v, o)


def test_padded_head_never_writes_past_hdim_vo(fwd_build):
    """O's D-tail chunk may spill into the caller's pad, but not past it: columns from `ceil8(hdim)` on were never in any
    store chunk."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=100))
    gen = seeded(100)
    q, k, v = (randn(1, 4, 256, 100, dtype, gen=gen) for _ in range(3))
    full = torch.full((1, 4, 256, 128), -7.0, device="cuda", dtype=dtype)
    run_fwd(fn, q, k, v, full[..., :100])
    assert torch.all(full[..., ceil8(100) :] == -7.0)


# ---------------------------------------------------------------------------
# FWD-17, FWD-18
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sk", [63, 64, 65])
@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_bias_last_column_at_odd_seqlen_k(fwd_build, hdim, sk):
    """K16: a wide bias read straddles a dword with the out-of-range column, so for odd `seqlen_k` the last bias column
    read back as 0 for every (b, h). A one-hot bias at `[sq-1, sk-1]` must move `O[sq-1]` and the LSE by the
    fp64-predicted amount; zero sensitivity fails."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim, bias=True))
    b, h, sq = 2, 3, 70
    bias = torch.zeros(b, h, sq, sk, device="cuda", dtype=dtype)
    bias[:, :, sq - 1, sk - 1] = 6.0
    r = fwd_check(fn, b=b, hq=h, sq=sq, sk=sk, d=hdim, dtype=dtype, bias=bias, ctx=f"sk={sk} {hdim}")
    plain = reference(r["q"], r["k"], r["v"], hdim**-0.5)[0]
    moved = (r["exact_o"][:, :, sq - 1] - plain[:, :, sq - 1]).abs().max().item()
    assert moved > 1e-3, "the bias must be visible to the test"
    assert (r["o"][:, :, sq - 1].double() - plain[:, :, sq - 1]).abs().max().item() > moved / 2


def test_bias_bshd_view_last_slab(fwd_build):
    """K15: a bias passed as a transposed `(B, Sq, H, Sk)` view, with a canary right after it, is correct (the last
    slab's bound must not run off the tensor) and leaves the canary untouched."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, bias=True))
    b, h, sq, sk = 2, 5, 33, 51
    gen = seeded(4)
    flat = torch.randn(b * sq * h * sk + 4096, device="cuda", generator=gen).to(dtype)
    flat[b * sq * h * sk :] = 777.0
    bias = flat[: b * sq * h * sk].view(b, sq, h, sk).transpose(1, 2)
    fwd_check(fn, b=b, hq=h, sq=sq, sk=sk, d=64, dtype=dtype, bias=bias, ctx="bshd bias")
    assert torch.all(flat[b * sq * h * sk :] == 777.0)


# ---------------------------------------------------------------------------
# FWD-39
# ---------------------------------------------------------------------------


@pytest.mark.large_shape
def test_offsets_past_2gi_fwd(fwd_build):
    """K18/G02: a slab whose byte base exceeds 2^31 (a large batch stride over one allocation) is correct."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64))
    b, h, s, d = 3, 8, 4096, 64
    # 3 x (8 x 4096 x 64) x 2 B = 12 MiB per tensor is far below 2 GiB, so the batch stride is inflated instead.
    big = torch.empty(b, 1 << 30, device="cuda", dtype=dtype)[:, : h * s * d]  # stride(0) = 2^30 elements = 2 GiB
    q = big.view(b, h, s, d)
    q.copy_(torch.randn(b, h, s, d, device="cuda", dtype=dtype))
    k = randn(b, h, s, d, dtype)
    v = randn(b, h, s, d, dtype)
    fwd_check(fn, b=b, hq=h, sq=s, d=d, dtype=dtype, qkv=(q, k, v), ctx="2 GiB batch stride")
