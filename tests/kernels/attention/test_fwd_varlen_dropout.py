# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward varlen and dropout (FWD-23, 24, 26, 27).

Varlen is decoded at runtime from `varlen_bits` (dense is bits == 0), so one build serves all five modes. Dropout masks
are a function of element coordinates only: never of the tiling, never of the batch slice a packed sequence lives in.
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    VARLEN_MODES,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    meta_of,
    randn,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)

# ---------------------------------------------------------------------------
# FWD-23
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("mode", VARLEN_MODES)
def test_varlen_modes(fwd_build, mode, causal):
    """K30/K31/K32: all five `VarlenBits` modes, with and without bottom-right causal, against a per-sequence fp64
    reference (O to the floor, LSE to fp32)."""
    fn = fwd_build(meta_of(head_dim=64, window=causal))
    lens_q, lens_k = [100, 37, 64, 129], [100, 37, 64, 129]
    case = VarlenCase(mode, lens_q, lens_k, 4, 4, 64, DTYPES["bf16"], seed=11)
    case.check(fn, window=BR if causal else None, ctx=f"{mode} causal={causal}")


@pytest.mark.parametrize("mode", ["0x0B0B", "0x1313"])
def test_varlen_cross_lengths_and_gqa(fwd_build, mode):
    """Q and K lengths differ per sequence (including `seqlen_q > seqlen_k` under causal) with GQA."""
    fn = fwd_build(meta_of(head_dim=64, window=True))
    case = VarlenCase(mode, [96, 40, 130], [200, 40, 50], 8, 2, 64, DTYPES["bf16"], seed=12)
    case.check(fn, window=BR, ctx=mode)


# ---------------------------------------------------------------------------
# FWD-24
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["0x0B0B", "0x0202"])
def test_varlen_zero_length_sequences(fwd_build, mode):
    """K33/G03: an empty-q sequence writes nothing; an empty-k sequence gets O = 0 and LSE = +inf; no fault on a
    **trailing** empty sequence under a packed layout (its `row_off` is one past the last row of the tensor)."""
    fn = fwd_build(meta_of(head_dim=64, window=True))
    lens_q, lens_k = [0, 96, 0, 40, 0], [64, 0, 32, 40, 0]
    case = VarlenCase(mode, lens_q, lens_k, 4, 4, 64, DTYPES["bf16"], seed=13)
    case.launch(fn, window=BR)
    covered = torch.zeros(case.o.shape[0], case.o.shape[2], dtype=torch.bool, device="cuda")
    for z in range(case.n):
        if case.lens_q[z] == 0:
            continue
        qb, qr = case._where_q(z)
        covered[qb, qr] = True
        o = case.o_of(z)
        if case.lens_k[z] == 0:
            assert torch.count_nonzero(o) == 0 and not torch.isnan(o).any(), f"empty-k sequence {z}: O must be 0"
            assert (case.lse_of(z) == float("inf")).all(), f"empty-k sequence {z}: LSE must be +inf"
        else:
            case_check_one(case, z)
    # Nothing outside a live sequence was written.
    outside = case.o.permute(0, 2, 1, 3)[~covered]
    assert torch.isnan(outside).all(), "an empty or padded row was written"


def case_check_one(case, z):
    """A live sequence of the zero-length case: finite and in the right range (the numerics are FWD-23's)."""
    o = case.o_of(z)
    assert torch.isfinite(o).all() and o.abs().max() < 10


# ---------------------------------------------------------------------------
# FWD-26
# ---------------------------------------------------------------------------


def _drop(fn, q, k, v, p, seed=1234, offset=0, **kw):
    o = alloc(q.shape[0], q.shape[1], q.shape[2], v.shape[3], q.dtype)
    run_fwd(fn, q, k, v, o, p_drop=p, seed=seed, offset=offset, **kw)
    return o


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_dropout_suite(fwd_build, hdim):
    """Lesson 9, K37: p = 0 is bitwise the no-dropout build; the mask is a function of the seed (and of nothing else);
    causal composes with dropout; a null offset pointer with an immediate offset equals the immediate-only run, and a
    null seed pointer reads 0."""
    dtype = DTYPES["bf16"]
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, 512, hdim, dtype, gen=gen) for _ in range(3))
    plain = fwd_build(meta_of(head_dim=hdim))
    fn = fwd_build(meta_of(head_dim=hdim, dropout=True))
    ref = alloc(1, 4, 512, hdim, dtype)
    run_fwd(plain, q, k, v, ref)
    assert torch.equal(_drop(fn, q, k, v, 0.0), ref), "p = 0 must leave the answer alone, exactly"
    a, again, other = (_drop(fn, q, k, v, 0.5, seed=s) for s in (1234, 1234, 999))
    assert torch.equal(a, again) and not torch.equal(a, other) and not torch.isnan(a).any()
    # seed=None is a null seed pointer, which reads as 0.
    assert torch.equal(_drop(fn, q, k, v, 0.5, seed=None), _drop(fn, q, k, v, 0.5, seed=0))
    # A null offset pointer plus an immediate equals the immediate alone; a non-null pointer adds its value.
    base = _drop(fn, q, k, v, 0.5, offset=7)
    ptr = torch.tensor([5], dtype=torch.int64, device="cuda")
    o = alloc(1, 4, 512, hdim, dtype)
    run_fwd(fn, q, k, v, o, p_drop=0.5, seed=1234, offset=2, philox_offset1=ptr)
    assert torch.equal(o, base), "offset1 (pointer) + offset2 (immediate) must equal the pre-summed immediate"
    causal = fwd_build(meta_of(head_dim=hdim, dropout=True, window=True))
    c1, c2 = (_drop(causal, q, k, v, 0.5, window=BR) for _ in range(2))
    assert torch.equal(c1, c2) and not torch.isnan(c1).any()


def test_dropout_mask_does_not_depend_on_the_tiling(fwd_build):
    """**The reproducibility contract**: a mask generated here is regenerated by the backward and the debug mask kernel,
    so it is a function of element coordinates alone (`grid_plane` takes `max_seqlen`, never `BLOCK_M`/`BLOCK_N`). The
    same problem under two different, both-supported wave geometries gives bit-identical output; the no-dropout
    control shows the two geometries agree anyway."""
    dtype = DTYPES["bf16"]
    gen = seeded(5)
    q, k, v = (randn(1, 4, 512, 64, dtype, gen=gen) for _ in range(3))
    fam_a = dict(num_warps=8, BLOCK_M=256, BLOCK_N=64, HEAD_DIM_GRANULE=64)
    fam_b = dict(num_warps=4, BLOCK_M=128, BLOCK_N=64, HEAD_DIM_GRANULE=64)
    for drop in (False, True):
        meta = meta_of(head_dim=64, dropout=drop)
        a, b = (fwd_build(meta, **fam) for fam in (fam_a, fam_b))
        assert a.knobs.num_warps != b.knobs.num_warps
        oa = _drop(a, q, k, v, 0.5) if drop else alloc(1, 4, 512, 64, dtype)
        ob = _drop(b, q, k, v, 0.5) if drop else alloc(1, 4, 512, 64, dtype)
        if not drop:
            run_fwd(a, q, k, v, oa)
            run_fwd(b, q, k, v, ob)
        assert torch.equal(oa, ob), "dropout" if drop else "control"


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dropout_keep_rate_matches_p(fwd_build, p):
    """The keep rate read straight out of O: `Q = K = 0` makes the softmax uniform at `1/S`, `V[..., 0] = 1` makes
    column 0 equal `(kept / S) / (1 - p)`, whose expectation is 1; times `(1 - p)` is the keep rate."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, dropout=True))
    z = torch.zeros(1, 4, 2048, 64)
    q, k = alloc(1, 4, 2048, 64, dtype, fill=z), alloc(1, 4, 2048, 64, dtype, fill=z)
    vv = z.clone()
    vv[..., 0] = 1.0
    v = alloc(1, 4, 2048, 64, dtype, fill=vv)
    got = _drop(fn, q, k, v, p, seed=7)
    keep = (got[..., 0].float() * (1.0 - p)).mean().item()
    assert abs(keep - (1.0 - p)) < 0.02, f"keep rate {keep:.4f}, want {1 - p:.4f}"


def test_dropout_expectation_is_unbiased(fwd_build):
    """Averaging over seeds converges to the undropped answer: the only test that catches the `1/(1-p)` scale being
    dropped or applied twice (every other dropout test passes with any constant scale)."""
    dtype = DTYPES["bf16"]
    gen = seeded(6)
    q, k, v = (randn(1, 4, 512, 64, dtype, gen=gen) for _ in range(3))
    fn = fwd_build(meta_of(head_dim=64, dropout=True))
    plain = alloc(1, 4, 512, 64, dtype)
    run_fwd(fwd_build(meta_of(head_dim=64)), q, k, v, plain)
    n = 32
    acc = sum(_drop(fn, q, k, v, 0.5, seed=1000 + s).float() for s in range(n)) / n
    rel = ((acc - plain.float()).abs().mean() / plain.float().abs().mean()).item()
    assert rel < 3.0 / n**0.5, f"mean over {n} seeds deviates {rel:.4f}"


def test_dropout_tensor_requirements_are_checked(fwd_build):
    dtype = DTYPES["bf16"]
    q, k, v = (randn(1, 4, 256, 64, dtype) for _ in range(3))
    o = alloc(1, 4, 256, 64, dtype)
    with pytest.raises(ValueError, match="requires dropout_p"):
        run_fwd(fwd_build(meta_of(head_dim=64, dropout=True)), q, k, v, o)
    with pytest.raises(ValueError, match="not compiled for dropout"):
        run_fwd(fwd_build(meta_of(head_dim=64)), q, k, v, o, p_drop=0.5)


# ---------------------------------------------------------------------------
# FWD-27
# ---------------------------------------------------------------------------


def test_packed_varlen_draws_a_plane_per_sequence(fwd_build):
    """K35: every sequence of a packed batch gets its own dropout plane (`seq * H + head`, from the grid's `z`, not from
    the post-decode batch index, which is 0 for stacked layouts). The oracle is a dense call of batch N, sliced: a
    batch-1 call collapses to plane 0 the same way and would report agreement while both are wrong, so that degenerate
    reference is asserted to *disagree*."""
    h, d, n, length = 4, 64, 3, 128
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=d, dropout=True))
    case = VarlenCase("0x0B0B", [length] * n, [length] * n, h, h, d, dtype, seed=7)

    def to_b(t):
        return t.view(1, h, n, length, d).permute(2, 0, 1, 3, 4).reshape(n, h, length, d).contiguous()

    qb, kb, vb = to_b(case.q), to_b(case.k), to_b(case.v)
    p, seed = 0.5, 1234
    o_dense = alloc(n, h, length, d, dtype)
    run_fwd(fn, qb, kb, vb, o_dense, p_drop=p, seed=seed)
    case.launch(fn, dropout_p=p, philox_seed=seed, philox_offset2=0)
    for i in range(n):
        assert torch.equal(case.o_of(i)[0], o_dense[i]), (
            f"sequence {i} of a packed batch does not match dense row-group {i}; if sequence 0 passes and the rest fail, "
            "the plane has collapsed to the batch index and every sequence shares one mask"
        )
    o_b1 = torch.stack([_drop(fn, qb[i : i + 1], kb[i : i + 1], vb[i : i + 1], p, seed=seed)[0] for i in range(n)])
    assert torch.equal(o_b1[0], o_dense[0]), "sequence 0 shares plane 0 with the batch-1 reference by construction"
    assert not torch.equal(o_b1[1], o_dense[1]), "the oracle would be vacuous: batch_idx does not distinguish sequences"


@pytest.mark.parametrize("mode", VARLEN_MODES)
def test_varlen_bias_in_every_mode(fwd_build, mode):
    """A bias under varlen follows Q's batch and row layout (a sequence's rows are its Q rows) and is as wide as the longest
    sequence's keys: K is packed, so its token axis is the batch total and must not be mistaken for the bias width. Each
    sequence is gated against its own reference with its own slice of the bias."""
    h, d = 4, 64
    case = VarlenCase(mode, [100, 64, 37], [100, 64, 37], h, h, d, DTYPES["bf16"])
    bias = randn(case.batch, h, case.tq, case.maxk + 8, DTYPES["bf16"], gen=seeded(3)).contiguous()
    case.check(fwd_build(meta_of(head_dim=d, bias=True)), bias=bias, ctx=f"varlen bias {mode}")
