# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward schedule knobs and the build cache (FWD-28, FWD-33).

LPT (longest-processing-time-first) is a bijection over the q-block index set: it moves the schedule, never a bit of the
output. `test_abi.test_lpt_tile_order_changes_the_isa` is its other half: it proves the knob is *live*, which the first
port's version of this test never did (the ISA was byte-identical with LPT on and off, so bit-identical outputs proved
nothing).
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    lse_alloc,
    meta_of,
    randn,
    reference,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)


@pytest.mark.parametrize("hdim", [64, 256])
def test_lpt_is_bit_identical(fwd_build, hdim):
    """K49: outputs and LSE are bitwise equal with LPT on and off, for dense causal and for varlen causal (the varlen form
    reverses only *active* q-blocks, so a padded workgroup stays inactive)."""
    dtype = DTYPES["bf16"]
    meta = meta_of(head_dim=hdim, window=True)
    on, off = fwd_build(meta, LPT_TILE_ORDER=True), fwd_build(meta, LPT_TILE_ORDER=False)
    assert on.knobs.LPT_TILE_ORDER and not off.knobs.LPT_TILE_ORDER
    gen = seeded(hdim)
    q, k, v = (randn(2, 4, 1024, hdim, dtype, gen=gen) for _ in range(3))
    outs = []
    for fn in (on, off):
        o, lse = alloc(2, 4, 1024, hdim, dtype), lse_alloc(2, 4, 1024)
        run_fwd(fn, q, k, v, o, lse=lse, window=BR)
        outs.append((o, lse))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    # Varlen: ragged lengths, so some workgroups of the padded grid are inactive.
    results = []
    for fn in (on, off):
        case = VarlenCase("0x0B0B", [900, 130, 513, 64], [900, 130, 513, 64], 4, 4, hdim, dtype, seed=4)
        case.launch(fn, window=BR)
        results.append((case.o.clone(), case.lse.clone()))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("scale", [0.0, 1.0], ids=["ScaleZero", "ScaleOne"])
def test_bias_build_after_bias_free_build(fwd_build, scale):
    """K53: in one process the bias-free build comes first. The bias build must still apply its bias (the old cache tag
    omitted `BIAS_TYPE`, so the second build silently received the first one's binary: bit-identical output for a zero,
    constant, per-row and key-varying bias alike). Scale 0 rules out a bias expressed as `bias / sm_scale`."""
    dtype = DTYPES["bf16"]
    hdim, s = 64, 289
    gen = seeded(7)
    q, k, v = (randn(1, 4, s, hdim, dtype, gen=gen) for _ in range(3))
    bias = torch.randn(1, 4, s, s, device="cuda", generator=gen).to(dtype)
    plain = alloc(1, 4, s, hdim, dtype)
    run_fwd(fwd_build(meta_of(head_dim=hdim)), q, k, v, plain, scale=scale)
    biased = alloc(1, 4, s, hdim, dtype)
    run_fwd(fwd_build(meta_of(head_dim=hdim, bias=True)), q, k, v, biased, scale=scale, bias=bias)
    assert not torch.equal(plain, biased), "the bias build returned the bias-free binary's output"
    exact = reference(q, k, v, scale, bias=bias)[0]
    assert (biased.double() - exact).abs().max().item() <= 0.035
