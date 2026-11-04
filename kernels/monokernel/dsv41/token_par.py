# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Token parallel (MI350's ``tok_par``, ATOM ``kernels/token_par.py``): rank r owns
tokens r S / TP .. of an S-row step. The per-token stages between the attention
and the MoE (the TP reduce, the seam, the norm, the gates) and the MoE's final
reduce run on a rank's own tokens only, through ``own_view``; the wo_b and FFN
partials go to a token's owner alone (a reduce-scatter instead of an all-reduce)
and the normed rows reach every rank through XG (an all-gather) before the MoE."""

import flydsl.expr as fx

from kernels.monokernel.dsv41.dims import HC, HIDDEN

# a token's row in each per-token kernel argument, in bytes
TOKEN_ROW_BYTES = {
    "res_in": HC * HIDDEN * 2,
    "res_out": HC * HIDDEN * 2,
    "normed": HIDDEN * 2,
    "pend": HIDDEN * 2,
    "out": HIDDEN * 2,
    "pre_in": HC * 4,
    "post_in": HC * 4,
    "comb_in": HC * HC * 4,
    "pre_out": HC * 4,
    "post_out": HC * 4,
    "comb_out": HC * HC * 4,
    # K1's (``attn_pre.K1_OWN``)
    "pos": 8,  # (position, 0) int32
    "ring_rows": 4,
    "qr_out": 1280,  # the q latent's FP8 bytes (Q_RANK)
    "qrs_out": 40,  # its E8M0 codes (Q_RANK / 32)
}

# K2a's per-token arguments of the stages from the TP reduce on
K2A_OWN = ("res_in", "res_out", "pre_in", "post_in", "comb_in", "pre_out", "post_out", "comb_out", "normed")


def tile_view(c, names, t0, n: int):
    """``c`` seen from a token tile t0 .. t0 + n (n compile-time) of the step: its
    per-token arguments in ``names`` moved to row t0, S = n for the stage's loops
    and LDS, and ``tt0`` / ``s_all`` (the tile's first token of the step, the
    step's tokens) for the mailbox and peer regions, laid out over the whole step
    (``attn_pre.slice_*``, ``attn_post.reduce_attn_pair``)."""
    args = dict(c["args"])
    lo = fx.Int64(t0)
    for name in names:
        args[name] = args[name] + lo * TOKEN_ROW_BYTES[name]
    base = c.get("tt0", 0)
    tt0 = t0 if isinstance(base, int) and base == 0 else base + t0
    return {**c, "S": n, "args": args, "tt0": tt0, "s_all": c.get("s_all", c["S"])}


def own_count(s: int, tp: int) -> int:
    """The tokens a rank owns in a token-parallel step."""
    assert s % tp == 0, f"{s} rows do not split over {tp} ranks"
    return s // tp


def own_view(c, names):
    """``c`` seen from this rank's tokens: S their count, ``t0`` the first one's
    row of the step, each per-token argument in ``names`` moved to it. The
    mailbox regions are then indexed by the own token (0 .. S / TP)."""
    n = own_count(c["S"], c["d"].tp)
    t0 = c["rank"] * n
    lo = fx.Int64(t0)
    args = dict(c["args"])
    for name in names:
        args[name] = args[name] + lo * TOKEN_ROW_BYTES[name]
    return {**c, "S": n, "args": args, "t0": t0}
