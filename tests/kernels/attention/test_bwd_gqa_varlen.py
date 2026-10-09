# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward GQA and varlen: BWD-13..15 and 17..19.

One build serves every head count (the counts arrive as kernargs, so the group loop's trip count is a runtime value) and
every varlen layout (`VarlenBits` is decoded on the device), which is what these tests exercise: the builds below are made
once per rung and reused across every case.
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    VARLEN_MODES,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    bwd_check,
    bwd_floor,
    bwd_reference,
    check_floor,
    dkdv_family_pins,
    meta_of,
    randn,
    sdpa_scale,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
BF16 = DTYPES["bf16"]

# ---------------------------------------------------------------------------
# BWD-13: GQA, with the head counts only at run time
# ---------------------------------------------------------------------------

GQA_HEADS = [(8, 8), (8, 4), (8, 2), (8, 1), (6, 3), (10, 2)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rows", [32, 16])
@pytest.mark.parametrize("heads", GQA_HEADS, ids=lambda h: f"h{h[0]}_kv{h[1]}")
def test_gqa_runtime_trip_count(bwd_build, heads, rows, causal):
    """dK/dV sum over every query head of a KV head's group (`hq / hk` of them, a loop whose trip count is the `num_head_q /
    num_head_k` kernargs): an AOT-style launch at default knobs, one build for all six head configurations. A dK/dV that
    compiled the group size in would be right for one of them and quietly wrong for the rest."""
    hq, hk = heads
    pins = dkdv_family_pins(64, rows)
    meta = meta_of(head_dim=64, window=causal)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=pins)
    bwd_check(
        builds,
        b=1,
        hq=hq,
        hk=hk,
        sq=130,
        sk=150,
        d=64,
        dtype=BF16,
        window=BR if causal else None,
        ctx=f"{heads} rows{rows} causal={causal}",
    )


def test_gqa_one_build_serves_every_head_count(bwd_build):
    """The builds are keyed by traits and knobs only: two head configurations get the very same launcher."""
    meta = meta_of(head_dim=64)
    assert bwd_build(meta).dkdv is bwd_build(meta).dkdv


# ---------------------------------------------------------------------------
# BWD-14: GQA composes with varlen, a bias and a mask per query head
# ---------------------------------------------------------------------------


def test_gqa_composes_with_a_bias(bwd_build):
    """The bias (and its gradient) is per *query* head even under GQA: there is one score matrix per q head."""
    b, hq, hk, s, d = 1, 10, 2, 130, 64
    bias = randn(b, hq, s, s, BF16, gen=seeded(8)).contiguous()
    out = bwd_check(
        bwd_build(meta_of(head_dim=d, bias=True)),
        b=b,
        hq=hq,
        hk=hk,
        sq=s,
        d=d,
        dtype=BF16,
        bias=bias,
        want_db=True,
        ctx="gqa (10,2) + bias",
    )
    assert out["db"].shape == (b, hq, s, s)


@pytest.mark.parametrize("causal", [False, True])
def test_gqa_composes_with_varlen(bwd_build, causal):
    meta = meta_of(head_dim=64, window=causal)
    case = VarlenCase("0x0B0B", [100, 200, 64], [150, 200, 90], 8, 2, 64, BF16)
    case.check_bwd(bwd_build(meta), window=BR if causal else None, ctx=f"gqa varlen {causal}")


# ---------------------------------------------------------------------------
# BWD-15: the LSE layout is decoded at run time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hq", [1, 4])
@pytest.mark.parametrize("layout", ["HT", "TH"])
def test_lse_layout_decoded_at_runtime(bwd_build, layout, hq):
    """One build, both LSE layouts (`(H, T)` AOTriton's and `(T, H)` Transformer Engine's), chosen by the bits of the
    descriptor: a build specialised to one layout would read the wrong elements for the other (and AOTriton, which launches
    from C++ and never calls a host wrapper, got every `TH` backward wrong that way)."""
    meta = meta_of(head_dim=64)
    case = VarlenCase("0x0B0B", [100, 64], [100, 64], hq, hq, 64, BF16, lse_layout=layout)
    case.check_bwd(bwd_build(meta), ctx=f"{layout} hq={hq}")


# ---------------------------------------------------------------------------
# BWD-17: the five varlen modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", VARLEN_MODES)
def test_varlen_modes(bwd_build, mode, causal):
    """Every `VarlenBits` mode (compact, padded, strided positions, `seqused_k`, and packed Q against a batched KV cache),
    each sequence against its own fp64 reference; and dQ is **bitwise** the dQ of N dense calls on the same sequences.
    """
    meta = meta_of(head_dim=64, window=causal)
    lens_q, lens_k = [100, 200, 64], [100, 200, 64]
    builds = bwd_build(meta)
    case = VarlenCase(mode, lens_q, lens_k, 4, 2, 64, BF16)
    do = case.new_do()
    case.check_bwd(builds, window=BR if causal else None, do=do, ctx=f"mode {mode} causal={causal}")
    # dQ per sequence, dense: the same Q rows are computed by the same instructions whatever the layout around them.
    for z in range(case.n):
        q, k, v = case.seq(z)
        qb, qr = case._where_q(z)
        s_q = q.shape[2]
        lse_z = case.lse_of(z).contiguous()  # (H, S)
        delta_z = torch.nan_to_num((do[qb : qb + 1, :, qr].float() * case.o_of(z).float()).sum(-1), nan=0.0).reshape(
            -1, s_q
        )
        dq = alloc(1, case.hq, s_q, 64, BF16)
        builds.dq(
            q,
            k,
            v,
            do[qb : qb + 1, :, qr],
            dq,
            lse_z,
            delta_z.contiguous(),
            1,
            s_q,
            seqlen_k=k.shape[2],
            window=BR if causal else None,
        )
        torch.cuda.synchronize()
        assert torch.equal(dq, case.grad_of("dq", z)), f"{mode} seq {z}: varlen dQ differs from the dense call"


# ---------------------------------------------------------------------------
# BWD-18: empty sequences
# ---------------------------------------------------------------------------


def test_varlen_zero_length_and_empty_kv(bwd_build):
    """A sequence with no keys has dQ exactly 0 (its rows attend nothing); one with no queries contributes nothing and has
    dK = dV = 0; a trailing empty sequence does not fault. The sequences that are not empty stay correct."""
    meta = meta_of(head_dim=64)
    case = VarlenCase("0x0B0B", [64, 40, 0, 32, 0], [64, 0, 50, 32, 0], 4, 2, 64, BF16)
    do = case.new_do()
    case.run_bwd(bwd_build(meta), do)
    for z in range(case.n):
        if case.lens_q[z] == 0 or case.lens_k[z] == 0:
            if case.lens_q[z]:
                assert (case.grad_of("dq", z) == 0).all(), f"seq {z} (no keys): dQ must be exactly 0"
            if case.lens_k[z]:
                assert (case.grad_of("dk", z) == 0).all() and (
                    case.grad_of("dv", z) == 0
                ).all(), f"seq {z} (no queries): dK/dV must be exactly 0"
            continue
        q, k, v = case.seq(z)
        qb, qr = case._where_q(z)
        ref = bwd_reference(q, k, v, do[qb : qb + 1, :, qr], sdpa_scale(64))
        floors = bwd_floor(lambda **kw: bwd_reference(q, k, v, do[qb : qb + 1, :, qr], sdpa_scale(64), **kw), ref, BF16)
        for i, name in enumerate(("dq", "dk", "dv")):
            check_floor(name, case.grad_of(name, z), ref[i], floors[i], f"seq {z}")


# ---------------------------------------------------------------------------
# BWD-19: bottom-right alignment is per sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["0x0B0B", "0x0202"])
def test_varlen_bottom_right_per_sequence(bwd_build, mode):
    """Causal bottom-right resolves against *each sequence's* own `(seqlen_q, seqlen_k)`: sequences with `Sq < Sk`, `Sq >
    Sk` (whole q rows masked) and `Sq == Sk` in one batch, each against its own reference."""
    meta = meta_of(head_dim=64, window=True)
    case = VarlenCase(mode, [64, 150, 100], [150, 64, 100], 4, 2, 64, BF16)
    case.check_bwd(bwd_build(meta), window=BR, ctx=f"bottom-right {mode}")
