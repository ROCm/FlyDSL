# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Mailbox regions of the V4.1 kernels laid out in a scratch or peer buffer, and
the kernels' mailbox (bounded, recording waits in a debug build)."""

import flydsl.expr as fx

from kernels.monokernel.sync_gfx1250 import Mailbox

PAIR_BYTES = 8  # a (value, tag) pair


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) in order: name -> (byte offset, bytes)."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


HIDDEN = 5120
Q_RANK, KV_DIM = 1280, 512  # the q latent, the window row (``dims``)
IDX_HEADS = 32  # the indexer's heads (``index_query.HEADS``)


def peer_layout(tokens: int, tp: int) -> dict:
    """The symmetric peer buffer: region -> (byte offset, bytes). ATTN / FFN: the
    attention / FFN partials, [source rank][token][hidden / 2] bf16 pairs; XG the
    normed rows (K2's MoE input) of a token-parallel step. Then K1's token-parallel
    gathers, [token][..] each: QG / QGS the q latent's MXFP8 words / codes, KG the
    window row, NBG the normed rows (the compressor's) and IWG the head weights
    (an indexer layer's K1). Appended: the regions before keep their offsets."""
    row = tokens * HIDDEN // 2
    return pair_layout(
        [
            ("attn", tp * row), ("ffn", tp * row), ("xg", row),
            ("qg", tokens * Q_RANK // 4), ("qgs", tokens * Q_RANK // 32), ("kg", tokens * KV_DIM // 2),
            ("nbg", row), ("iwg", tokens * IDX_HEADS),
        ]
    )  # fmt: skip


def peer_offsets(tokens: int, tp: int) -> dict:
    return {region: off for region, (off, _) in peer_layout(tokens, tp).items()}


def peer_bytes(tokens: int, tp: int) -> int:
    return max(off + n for off, n in peer_layout(tokens, tp).values())


def region_ids(regions) -> dict:
    return {name: i for i, name in enumerate(sorted(regions))}


POST_TAG0 = 64  # K2a's tags: layer + 1 + 64, apart from K1's in shared regions
MOE_TAG0 = 128  # K2b's
# the layers a step can run: K1's tags (layer + 1) stay below K2a's (POST_TAG0 + 1) and
# K2a's below K2b's (MOE_TAG0 + 1) only for layer < 64 -- past it a layer's mailbox
# takes another's tag in the regions they share and a wait passes early
MAX_LAYERS = POST_TAG0


def check_layers(n: int) -> None:
    """A model of ``n`` layers fits the tag spaces (``MAX_LAYERS``)."""
    if not 0 < n <= MAX_LAYERS:
        raise ValueError(f"{n} layers: the mailbox tags fit {MAX_LAYERS} (layout.POST_TAG0)")


def mailbox(layer, scratch, diag_off, regions, tag0=0) -> Mailbox:
    """The kernel's mailbox for tag ``layer + tag0 + 1``; ``diag_off`` >= 0: a debug
    build whose waits give up and record at that scratch offset."""
    if diag_off < 0:
        return Mailbox(layer + tag0)
    return Mailbox(layer + tag0, scratch + fx.Int64(diag_off), region_ids(regions))
