# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests of :mod:`kernels.monokernel.dsv4.reference` itself: its quantizers and its batching."""

from __future__ import annotations

import pytest
import torch

from kernels.monokernel.dsv4.config import MoeMode
from kernels.monokernel.dsv4.reference import (
    V4Config,
    contiguous_pool,
    fp4_row_bytes,
    golden_layer,
    make_weights,
    pack_fp4,
    quant_dequant_fp4,
    rope_table,
    unpack_fp4,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _cfg(hc_mult=4, compress_ratio=0, max_seq=1024):
    # small but structurally faithful: nope_dim a multiple of 64, V4's 64 rope lanes
    return V4Config(
        heads=8,
        hidden=256,
        q_lora=128,
        head_dim=128,
        rope_dim=64,
        o_groups=2,
        o_lora=64,
        n_experts=8,
        top_k=2,
        inter=128,  # must be a multiple of the 128 FP8 block
        window=32,
        hc_mult=hc_mult,
        compress_ratio=compress_ratio,
        max_seq=max_seq,
    )


# DeepSeek's whole Block: also covers hc_pre / hc_post / Sinkhorn and the [S, hc, d] residual.


def test_mxfp8_ceil_scale_never_clips():
    """Ceil-rounded E8M0 scales never clip a block max (nearest rounding does); error <= one E4M3 half-ulp."""
    from kernels.monokernel.dsv4.reference import quant_dequant_mxfp8
    from kernels.monokernel.formats import quant_dequant_mxfp8 as nearest_mxfp8

    torch.manual_seed(0)
    x = torch.randn(64, 1024) * torch.exp2(torch.randint(-8, 8, (64, 1)).float())
    x[:, ::32] *= 20.0  # an outlier per block, as real activations have
    q = quant_dequant_mxfp8(x)
    blocks = x.reshape(64, -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    err = (q.reshape(64, -1, 32) - blocks).abs()
    assert (err <= 2**-4 * blocks.abs() + amax * 2**-17).all(), "ceil-scaled MXFP8 clipped or over-rounded"
    assert (q.abs().reshape(64, -1, 32).amax(-1, keepdim=True) >= amax * (1 - 2**-4)).all(), "a block max clipped"
    # the nearest-rounded scale does clip on this input
    assert (nearest_mxfp8(x) - x).norm() > 2 * (q - x).norm()


def test_v4_indexer_cache_packs_fp4_losslessly():
    """The FP4 indexer cache round-trips exactly to ``quant_dequant_fp4`` and has the byte layout the kernel decodes."""
    torch.manual_seed(0)
    n = 128
    x = torch.randn(64, n) * torch.logspace(-20, 12, 64, base=2.0)[:, None]
    x[5] = 0  # a zero row: the clamp's smallest scale
    x[6, 32:64] = 0  # one zero block among live ones
    p = pack_fp4(x)
    assert p.dtype == torch.uint8 and p.shape == (64, fp4_row_bytes(n)) == (64, 68)
    assert torch.equal(unpack_fp4(p), quant_dequant_fp4(x))

    # element i in nibble i % 2 of byte i // 2, sign in bit 3; one e8m0 per 32
    y = torch.zeros(n)
    y[0], y[1], y[2], y[33] = 6.0, -0.5, -6.0, 3.0  # block 0 scale 1, block 1 scale 0.5
    q = pack_fp4(y)
    assert q[0].item() == 0x7 | (0x9 << 4), f"byte 0 {q[0].item():#x}"
    assert q[1].item() == 0xF, f"byte 1 {q[1].item():#x}"
    assert q[16].item() == 0x7 << 4, f"byte 16 {q[16].item():#x}"
    assert q[64:].tolist() == [127, 126, 1, 1], q[64:].tolist()


@pytest.mark.parametrize("ratio", [0, 4])
def test_v4_golden_batches_independent_sequences(ratio):
    """Batched sequences equal their own S=1 runs (catches a stage reading sample 0's state for all)."""
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    if ratio:
        cfg.index_topk = 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    ihd, coff, S = cfg.index_head_dim, cfg.c_coff, 2

    def state(n):
        # ONE plane; sample s owns rows [s * cache_rows, (s+1) * cache_rows)
        d = dict(kv_cache=torch.zeros(n * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device))
        if not ratio:
            return d
        d |= dict(
            kv_state=torch.zeros(n, cfg.c_rows, coff * cfg.head_dim, device=device),
            score_state=torch.full((n, cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device),
            cos_c=cos,
            sin_c=sin,
        )
        if cfg.indexed:
            d |= dict(
                i_state=torch.zeros(n, cfg.c_rows, coff * ihd, device=device),
                i_score_state=torch.full((n, cfg.c_rows, coff * ihd), float("-inf"), device=device),
                i_cache=torch.zeros(n, cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device),
            )
        return d

    def run(st, h, pos):
        n = h.shape[0]
        kw = dict(st)
        idx, dest = contiguous_pool([pos] * n, cfg, device)
        return golden_layer(
            W,
            h,
            [pos] * n,
            kw.pop("kv_cache"),
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            **kw,
        )

    batched, alone = state(S), [state(1) for _ in range(S)]
    steps = 4 * cfg.window // 3
    for pos in range(steps):
        h = (0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        got = run(batched, h, pos)["x_out"]
        for s in range(S):
            want = run(alone[s], h[s : s + 1], pos)["x_out"]
            # norm-relative: batch 2 and batch 1 may take GEMMs that round a few bf16 ulps apart;
            # reading another sample's state is a ~0.65 difference
            rel = ((got[s].float() - want[0].float()).norm() / want[0].float().norm()).item()
            assert rel < 2e-2, f"pos={pos} sample {s}: batched differs from its own run (rel {rel:.3e})"
    # the run has to be deep enough that the compressor and its state actually ran
    if ratio:
        assert steps > 2 * ratio, "too shallow to have crossed a compression boundary"
