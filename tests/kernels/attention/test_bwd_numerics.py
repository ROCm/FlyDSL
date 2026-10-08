# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward numerics: BWD-01..04, 22 and 30.

Every case runs the **new forward** (O and LSE as training would feed them), reduces delta in fp32 from the stored O, runs
dQ and dK/dV, and gates each gradient against AOTriton's fp64 reference with its rounding floor (G-floor): a correct
kernel lands at or below 1.0x the floor, the mistakes this suite exists for measured 2.6x to 17x.
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
    alloc,
    bwd_check,
    dkdv_family_pins,
    meta_of,
    randn,
    relrms,
    run_bwd,
    sdpa_scale,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
TL = (WINDOW_TOPLEFT, WINDOW_TOPLEFT)
DT_NAMES = ["bf16", "f16"]

# ---------------------------------------------------------------------------
# BWD-01
# ---------------------------------------------------------------------------

COMMON_MISTAKE_CASES = {
    "dense": dict(),
    "bottom_right_masked_rows": dict(sq=301, sk=190, window=BR),
    "sharp_softmax": dict(input_scale=6.0),
    "top_left_causal": dict(window=TL),
}


@pytest.mark.parametrize("dt", DT_NAMES)
@pytest.mark.parametrize("case", COMMON_MISTAKE_CASES)
def test_common_mistakes_bwd(bwd_build, case, dt):
    """The four classic backward mistakes, at head_dim 128: low-precision delta or scaled operands (the `sharp_softmax`
    case is where they show, at 2.6-2.8x the floor), masked rows that must give exactly zero gradients, and both causal
    alignments."""
    kw = dict(COMMON_MISTAKE_CASES[case])
    meta = meta_of(head_dim=128, dtype_str=dt, window="window" in kw)
    out = bwd_check(bwd_build(meta), b=1, hq=2, d=128, dtype=DTYPES[dt], ctx=f"{case} {dt}", **{"sq": 200, **kw})
    if case == "bottom_right_masked_rows":
        # Rows with no key (the first sq - sk of them) get exactly zero dQ; keys no row attends get exactly zero dK/dV.
        dead_q = out["dq"][:, :, : 301 - 190]
        assert (dead_q == 0).all(), "dQ of fully masked rows must be exactly 0"


@pytest.mark.parametrize("dt", DT_NAMES)
def test_zero_sm_scale_gives_exactly_zero_dq_dk(bwd_build, dt):
    """`sm_scale = 0`: the scores are all zero, so dQ and dK are exactly zero (the gradient flows through `sm_scale`), while
    dV is the uniform-attention one. Scaling the *masked accumulator* instead of the dot gives `-inf * 0 = NaN`."""
    meta = meta_of(head_dim=128, dtype_str=dt, window=True)
    out = bwd_check(
        bwd_build(meta), b=1, hq=2, sq=200, d=128, dtype=DTYPES[dt], window=BR, scale=0.0, ctx=f"scale 0 {dt}"
    )
    assert (out["dq"] == 0).all() and (out["dk"] == 0).all()


# ---------------------------------------------------------------------------
# BWD-02: both MFMA families at the rungs, plus ragged shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", DT_NAMES)
@pytest.mark.parametrize("shape", [(1, 2, 256, 256), (1, 4, 128, 777), (1, 2, 1, 512), (1, 2, 512, 1)])
@pytest.mark.parametrize("hdim", [64, 128])
def test_error_ratio_vs_math_shapes(bwd_build, shape, hdim, dt):
    """Square, wide-KV, and the degenerate ragged cases `(Sq, Sk)` = `(1, 512)` and `(512, 1)`."""
    b, h, sq, sk = shape
    meta = meta_of(head_dim=hdim, dtype_str=dt)
    # One key: P is exactly 1, so dS = P * (dP - delta) is exactly 0 in the reference and is fp32 noise (~1e-5) in the
    # kernel, where a *relative* floor of 0 has nothing to say. dV, which does not collapse, keeps the floor gate.
    outputs = ("dv",) if sk == 1 else ("dq", "dk", "dv")
    out = bwd_check(
        bwd_build(meta), b=b, hq=h, sq=sq, sk=sk, d=hdim, dtype=DTYPES[dt], ctx=f"{shape} d{hdim} {dt}", outputs=outputs
    )
    if sk == 1:
        assert out["dq"].abs().max().item() < 1e-4 and out["dk"].abs().max().item() < 1e-4


@pytest.mark.parametrize("mode", ["dense", "window"])
@pytest.mark.parametrize("rows", [32, 16])
@pytest.mark.parametrize("hdim", [64, 128, 256, 512])
def test_both_families_at_every_wide_rung(bwd_build, hdim, rows, mode):
    """The 32-row and the 16-row bodies, whatever the tuning table happens to pick: the *coverage* must not depend on the
    *policy*, or the next tuning change silently moves which family is tested where. dQ pins `MFMA_ROWS`; dK/dV pins the
    whole geometry the family needs. With a window and `Sq != Sk`, so each family's mask (keyed on its own lane to
    (row, col) map) is exercised rather than the one case where the alignments coincide."""
    pins = dkdv_family_pins(hdim, rows)
    if pins is None:
        pytest.skip("no legal dK/dV geometry for this family at this width")
    meta = meta_of(head_dim=hdim, window=mode == "window")
    try:
        builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=pins)
    except (ValueError, NotImplementedError) as exc:
        pytest.skip(f"this family does not fit at this width: {str(exc)[:100]}")
    bwd_check(
        builds,
        b=1,
        hq=2,
        sq=150,
        sk=130,
        d=hdim,
        dtype=DTYPES["bf16"],
        window=BR if mode == "window" else None,
        ctx=f"d{hdim} rows{rows} {mode}",
    )


# ---------------------------------------------------------------------------
# BWD-03: the sm_scale sweep, non-positive scales included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.0, -1.2, 0.02, 0.5, 1.0, 4.0])
@pytest.mark.parametrize("hdim", [64, 128, 256])
@pytest.mark.parametrize("window", [None, BR], ids=["dense", "causal"])
def test_scale_sweep_including_nonpositive(bwd_build, scale, hdim, window):
    """`-inf * scale` is NaN or `+inf` for a non-positive scale if the mask is applied after scaling (AOTriton #245): the
    backward scales the dot, never the masked accumulator. Gradients stay finite and inside the floor; with `Q = K = 0`
    the forward and backward are exactly symmetric."""
    meta = meta_of(head_dim=hdim, window=window is not None)
    bwd_check(
        bwd_build(meta),
        b=1,
        hq=2,
        sq=130,
        d=hdim,
        dtype=DTYPES["bf16"],
        window=window,
        scale=scale,
        ctx=f"scale {scale} d{hdim}",
    )


@pytest.mark.parametrize("scale", [0.0, -1.2])
def test_zero_inputs_are_exactly_symmetric_at_nonpositive_scale(bwd_build, scale):
    """`Q = K = 0` and a non-positive scale: every score is exactly 0, so P is exactly uniform and the gradients are
    exactly the uniform-attention ones, with no NaN anywhere."""
    dtype = DTYPES["bf16"]
    b, h, s, d = 1, 2, 128, 64
    gen = seeded(9)
    q = alloc(b, h, s, d, dtype, fill=torch.zeros(b, h, s, d))
    k = alloc(b, h, s, d, dtype, fill=torch.zeros(b, h, s, d))
    v = randn(b, h, s, d, dtype, gen=gen)
    do = randn(b, h, s, d, dtype, gen=gen)
    out = bwd_check(
        bwd_build(meta_of(head_dim=d)),
        b=b,
        hq=h,
        sq=s,
        d=d,
        dtype=dtype,
        scale=scale,
        qkv_do=(q, k, v, do),
        ctx="zero q,k",
    )
    assert not torch.isnan(out["dq"]).any() and not torch.isnan(out["dk"]).any()


# ---------------------------------------------------------------------------
# BWD-04: the backward agrees with the forward it follows, and with one autograd call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [64, 128, 256, 512])
def test_consistent_with_forward_and_joint(bwd_build, hdim):
    """The LSE the forward writes is the one the backward needs (natural base, scale folded): the backward driven by it
    matches a *joint* fp64 autograd of the whole attention, which is what training sees. A forward that wrote LSE in the
    wrong base would pass each kernel's own test and fail this one."""
    dtype = DTYPES["bf16"]
    b, h, s = 1, 2, 130
    gen = seeded(hdim)
    q, k, v, do = (randn(b, h, s, hdim, dtype, gen=gen) for _ in range(4))
    out = run_bwd(bwd_build(meta_of(head_dim=hdim)), q, k, v, do)
    qd, kd, vd, dod = (t.double().requires_grad_() for t in (q, k, v, do))
    p = torch.softmax((qd @ kd.transpose(-1, -2)) * sdpa_scale(hdim), dim=-1)
    (p @ vd).backward(dod)
    for name, got, ref in (("dq", out["dq"], qd.grad), ("dk", out["dk"], kd.grad), ("dv", out["dv"], vd.grad)):
        floor = relrms(ref.to(dtype), ref)
        assert relrms(got, ref) <= 2.0 * floor, f"{name} d{hdim}: {relrms(got, ref):.2e} vs floor {floor:.2e}"


# ---------------------------------------------------------------------------
# BWD-22: f16
# ---------------------------------------------------------------------------


def test_fp16_is_more_accurate_than_bf16(bwd_build):
    """Ten mantissa bits against seven: the f16 gradients are several times closer to the exact ones."""
    errs = {}
    for dt in DT_NAMES:
        dtype = DTYPES[dt]
        out = bwd_check(bwd_build(meta_of(head_dim=64, dtype_str=dt)), b=1, hq=2, sq=130, d=64, dtype=dtype, ctx=dt)
        errs[dt] = relrms(out["dq"], out["exact"][0])
    assert errs["f16"] * 4 < errs["bf16"], errs


def test_dtype_must_match_the_build(bwd_build):
    """f16 tensors into a bf16 build read as bf16: finite, wrong by ~2^112, and nothing downstream notices. Refused."""
    bf16 = bwd_build(meta_of(head_dim=64, dtype_str="bf16"))
    dtype = DTYPES["f16"]
    b, h, s, d = 1, 2, 64, 64
    q, k, v, do = (randn(b, h, s, d, dtype) for _ in range(4))
    lse2 = torch.zeros(b * h, s, device="cuda")
    delta2 = torch.zeros(b * h, s, device="cuda")
    dk, dv = alloc(b, h, s, d, dtype), alloc(b, h, s, d, dtype)
    with pytest.raises(ValueError, match="dtype_str='bf16'"):
        bf16.dkdv(q, k, v, do, dk, dv, lse2, delta2, b, s)


def test_fp16_range(bwd_build):
    """Large f16 inputs (|x| ~ 60, scores in the thousands): the fp32 softmax path must not overflow where f16 would."""
    dtype = DTYPES["f16"]
    bwd_check(
        bwd_build(meta_of(head_dim=64, dtype_str="f16")),
        b=1,
        hq=2,
        sq=130,
        d=64,
        dtype=dtype,
        input_scale=4.0,
        scale=0.05,
        ctx="f16 range",
    )


# ---------------------------------------------------------------------------
# BWD-30: large logits (AOTriton's lesson 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [32, 64])
def test_large_logits_bwd(bwd_build, hdim):
    """Scores of magnitude ~133120 at `sm_scale` 0.25 (AOTriton's lesson 3: the max and the exponent must come from the same
    rounding): P is one-hot, and no gradient may be inf or NaN. And a sharp N(0, 64): the answer stays inside the floor.
    """
    dtype = DTYPES["bf16"]
    b, h, s = 1, 2, 130
    gen = seeded(31)
    sigma = (133120.0 / hdim**0.5) ** 0.5  # S = sum_d q_d k_d ~ sigma^2 * sqrt(d) * N(0, 1)
    q, k = (randn(b, h, s, hdim, dtype, gen=gen, scale=sigma) for _ in range(2))
    v, do = (randn(b, h, s, hdim, dtype, gen=gen) for _ in range(2))
    out = run_bwd(bwd_build(meta_of(head_dim=hdim)), q, k, v, do, scale=0.25)
    for name in ("dq", "dk", "dv"):
        assert torch.isfinite(out[name]).all(), f"{name} has inf/NaN"
    bwd_check(
        bwd_build(meta_of(head_dim=hdim)), b=b, hq=h, sq=s, d=hdim, dtype=dtype, input_scale=8.0, ctx=f"sharp d{hdim}"
    )
