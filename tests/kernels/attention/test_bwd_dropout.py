# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward dropout: BWD-20 and BWD-21.

The backward must draw **the forward's mask**, not "a mask with the right statistics": a different one gives quietly wrong
gradients and no shape check notices. The oracle is the forward itself: with `V = I` (`head_dim == seqlen_k`) the forward's
`O = P * keep / (1 - p)` element for element, so `O != 0` *is* the mask, and the fp64 reference is then computed with it.
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    VarlenCase,
    bwd_floor,
    bwd_reference,
    check_floor,
    dkdv_family_pins,
    meta_of,
    randn,
    run_bwd,
    sdpa_scale,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BF16 = DTYPES["bf16"]


def identity_problem(b=1, h=2, s=128, seed=40):
    """`(q, k, v, do)` with `V = I`, so the forward's output reveals the dropout mask it used."""
    gen = seeded(seed)
    q, k, do = (randn(b, h, s, s, BF16, gen=gen) for _ in range(3))
    v = torch.eye(s, device="cuda", dtype=BF16).expand(b, h, s, s).contiguous()
    return q, k, v, do


def gate(out, q, k, v, do, keep, p, ctx):
    sm = sdpa_scale(q.shape[-1])

    def ref(**kw):
        return bwd_reference(q, k, v, do, sm, keep=keep, p_drop=p, **kw)

    exact = ref()
    floors = bwd_floor(ref, exact, BF16)
    for i, name in enumerate(("dq", "dk", "dv")):
        check_floor(name, out[name], exact[i], floors[i], ctx)


@pytest.mark.parametrize("rows", [32, 16])
@pytest.mark.parametrize("p", [0.0, 0.25, 0.5])
def test_dropout_regenerates_the_forwards_own_mask(bwd_build, p, rows):
    """**The cross-kernel contract**: dQ, dK and dV against the fp64 reference built from the mask the forward actually
    used (recovered through `V = I`), in both families. A backward that drew a different mask fails by O(1), not by a
    tolerance. Also pins the keep rate."""
    q, k, v, do = identity_problem()
    meta = meta_of(head_dim=128, dropout=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=dkdv_family_pins(128, rows))
    out = run_bwd(builds, q, k, v, do, p_drop=p, seed=1234)
    keep = out["o"] != 0
    assert abs(keep.float().mean().item() - (1.0 - p)) < 0.02, "keep rate is not 1 - p"
    gate(out, q, k, v, do, keep, p, f"p={p} rows{rows}")


def test_p_zero_is_bitwise_a_no_dropout_build(bwd_build):
    """`p = 0` must reproduce a build with no dropout at all, bit for bit: the dropout arm perturbs nothing it should not."""
    gen = seeded(2)
    q, k, v, do = (randn(1, 2, 130, 64, BF16, gen=gen) for _ in range(4))
    plain = run_bwd(bwd_build(meta_of(head_dim=64)), q, k, v, do)
    dropped = run_bwd(bwd_build(meta_of(head_dim=64, dropout=True)), q, k, v, do, p_drop=0.0, seed=7)
    for name in ("dq", "dk", "dv"):
        assert torch.equal(plain[name], dropped[name]), name


def test_dropout_is_deterministic_per_seed(bwd_build):
    """Same seed, same gradients bit for bit; a different seed, different ones."""
    gen = seeded(3)
    q, k, v, do = (randn(1, 2, 130, 64, BF16, gen=gen) for _ in range(4))
    builds = bwd_build(meta_of(head_dim=64, dropout=True))
    a = run_bwd(builds, q, k, v, do, p_drop=0.3, seed=11)
    b = run_bwd(builds, q, k, v, do, p_drop=0.3, seed=11)
    c = run_bwd(builds, q, k, v, do, p_drop=0.3, seed=12)
    for name in ("dq", "dk", "dv"):
        assert torch.equal(a[name], b[name]), f"{name} is not deterministic"
    assert not torch.equal(a["dq"], c["dq"])


def test_dropout_mask_is_independent_of_tiling(bwd_build):
    """The mask is a function of (sequence, head, row, column), never of the tile geometry: the 32-row and 16-row
    families, with different tiles, both match the reference built from the one forward mask."""
    q, k, v, do = identity_problem(seed=41)
    meta = meta_of(head_dim=128, dropout=True)
    keeps = []
    for rows in (32, 16):
        builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=dkdv_family_pins(128, rows))
        out = run_bwd(builds, q, k, v, do, p_drop=0.4, seed=99)
        keeps.append(out["o"] != 0)
        gate(out, q, k, v, do, keeps[-1], 0.4, f"tiling rows{rows}")
    # the forward is the same build both times, so the masks agree trivially; the gates above are the real claim.
    assert torch.equal(keeps[0], keeps[1])


# ---------------------------------------------------------------------------
# BWD-21: a packed varlen batch draws one plane per sequence
# ---------------------------------------------------------------------------


def test_packed_varlen_draws_a_dropout_plane_per_sequence(bwd_build):
    """N sequences packed into one batch slot must not share one mask: the plane is indexed by the grid's *sequence*, not by
    the batch slice (which is 0 for every stacked sequence). Equal lengths with `V = I` per sequence make each sequence's
    forward output reveal its own mask; the backward, run through the same packed descriptor, must reproduce every one.
    """
    meta = meta_of(head_dim=64, dropout=True)
    builds = bwd_build(meta)
    case = VarlenCase("0x0B0B", [64, 64, 64], [64, 64, 64], 2, 2, 64, BF16)
    eye = torch.eye(64, device="cuda", dtype=BF16)
    for z in range(case.n):
        kb, kr = case._where_k(z)
        case.v[kb, :, kr] = eye
    do = case.new_do()
    p, seed = 0.5, 1234
    case.run_bwd(builds, do, p_drop=p, seed=seed)
    sm = sdpa_scale(64)
    keeps = [(case.o_of(z) != 0) for z in range(case.n)]
    masks = [m[0].float().flatten() for m in keeps]
    assert not torch.equal(
        keeps[0], keeps[1]
    ), "two sequences drew the same mask: the plane collapsed to the batch slice"
    for z in range(case.n):
        q, k, v = case.seq(z)
        qb, qr = case._where_q(z)
        doz = do[qb : qb + 1, :, qr]

        def ref(**kw):
            return bwd_reference(q, k, v, doz, sm, keep=keeps[z], p_drop=p, **kw)

        exact = ref()
        floors = bwd_floor(ref, exact, BF16)
        for i, name in enumerate(("dq", "dk", "dv")):
            check_floor(name, case.grad_of(name, z), exact[i], floors[i], f"seq {z}")
    assert len(masks) == case.n
