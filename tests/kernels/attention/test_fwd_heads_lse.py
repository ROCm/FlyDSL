# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward head counts and the LSE output (FWD-19..22).

The head counts are runtime kernargs and the metadata carries none, so one build serves every count; these tests
launch one build with several. The LSE pointer may be null at runtime (`RETURN_LSE="runtime"`).
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    fwd_check,
    lse_alloc,
    meta_of,
    randn,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)


@pytest.mark.parametrize("hdim", [64, 256, 512])
def test_null_lse_pointer_at_runtime(fwd_build, hdim):
    """K24/A12: one binary serves a null and a non-null `L`. A null LSE does not fault and O is unchanged; a non-null LSE
    is fully written."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=hdim))
    assert fn.knobs.RETURN_LSE == "runtime"
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, 200, hdim, dtype, gen=gen) for _ in range(3))
    with_lse, without = alloc(1, 4, 200, hdim, dtype), alloc(1, 4, 200, hdim, dtype)
    lse = lse_alloc(1, 4, 200)
    run_fwd(fn, q, k, v, with_lse, lse=lse)
    run_fwd(fn, q, k, v, without)
    assert torch.equal(with_lse, without)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("layout", ["HT", "TH"])
@pytest.mark.parametrize("heads", [(5, 5), (10, 2), (8, 1)])
def test_lse_written_for_every_head(fwd_build, heads, layout):
    """K25: with runtime head counts at the default knobs the NaN-prefilled LSE is finite and correct for **every** head
    (a compile-time head count sized the slice for one head), in both `VarlenBits` LSE layouts."""
    hq, hk = heads
    fn = fwd_build(meta_of(head_dim=64, window=True))
    case = VarlenCase("0x0B0B", [100, 37, 64], [100, 37, 64], hq, hk, 64, DTYPES["bf16"], lse_layout=layout, seed=hq)
    case.check(fn, window=BR, ctx=f"{heads} {layout}")


def test_lse_store_survives_the_stagger_barrier(fwd_build):
    """K43: fp16 + PADDED_HEAD (100 -> 128, an 8-wave rung) with STAGGER on and a non-null LSE: the LSE is fully written.
    The stagger barrier's asm clobbers SCC, and the epilogue's null-LSE compare was hoisted above it."""
    for hdim in (100, 64):
        fn = fwd_build(meta_of(dtype_str="f16", head_dim=hdim), STAGGER=True)
        assert fn.knobs.STAGGER and fn.knobs.num_warps == 8
        fwd_check(fn, b=2, hq=4, sq=300, d=hdim, dtype=DTYPES["f16"], ctx=f"stagger f16 {hdim}")


@pytest.mark.parametrize("heads", [(8, 8), (8, 4), (8, 2), (8, 1), (10, 2)])
def test_gqa_mqa_runtime_heads(fwd_build, heads):
    """K25, lesson 12: MHA, GQA and MQA group sizes (and (10, 2) with a matrix bias) from one build."""
    hq, hk = heads
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, bias=heads == (10, 2)))
    bias = None
    if heads == (10, 2):
        bias = torch.randn(2, hq, 130, 130, device="cuda", dtype=torch.float32, generator=seeded(1)).to(dtype)
    fwd_check(fn, b=2, hq=hq, hk=hk, sq=130, d=64, dtype=dtype, bias=bias, ctx=f"gqa {heads}")
