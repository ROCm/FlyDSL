# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward numerics against the fp64 rounding floor (FWD-01..09, FWD-32).

Every gate is the floor gate of `attn_testlib` (relative RMS at most 2x the error a kernel that keeps everything else in
fp32 still gets), never a fudge factor against torch's low-precision SDPA, which cannot see a precision mistake.
"""

import math

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
    alloc,
    fwd_check,
    lse_alloc,
    meta_of,
    randn,
    reference,
    run_fwd,
    seeded,
    window_mask,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
TL = (WINDOW_TOPLEFT, WINDOW_TOPLEFT)

# ---------------------------------------------------------------------------
# FWD-01
# ---------------------------------------------------------------------------

_RUNGS = [32, 64, 96, 128, 192, 256, 384, 512]
_PADDED = [17, 100, 129, 300]


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("hdim", _RUNGS + _PADDED)
def test_fwd_vs_fp64_floor(fwd_build, hdim, causal):
    """K13/K19/K20/K42/K44/K46/C01: O and LSE at every rung and padded width, ragged sequence lengths (513 x 771).
    The dtype alternates per width so each is covered at both."""
    dtype_str = ("bf16", "f16")[(_RUNGS + _PADDED).index(hdim) % 2]
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=hdim, window=causal))
    fwd_check(
        fn,
        b=2,
        hq=4,
        sq=513,
        sk=771,
        d=hdim,
        dtype=DTYPES[dtype_str],
        window=BR if causal else None,
        ctx=f"hdim={hdim} {dtype_str} causal={causal}",
    )


# ---------------------------------------------------------------------------
# FWD-02
# ---------------------------------------------------------------------------

# AOTriton's `test_common_mistakes`, forward half. See docs/attention-kernel-numerical-error-lessons.
_COMMON_MISTAKES = {
    # Logits with a standard deviation of 16: a trained model's attention is this sharp, and the error of rounding a
    # scaled Q or S to the input dtype grows with |S| while the floor does not.
    "sharp_softmax": dict(sq=257, sk=519, input_scale=4.0, window=None, scale=128**-0.5),
    # Uniform attention over a causal prefix. Any kernel that multiplies a masked -inf by the scale computes NaN.
    "zero_sm_scale": dict(sq=257, sk=519, input_scale=1.0, window=TL, scale=0.0),
    # Nothing requires sm_scale > 0: -inf * negative is +inf, which exp2 turns into inf.
    "negative_sm_scale": dict(sq=257, sk=519, input_scale=1.0, window=TL, scale=-(128**-0.5)),
    # The first 111 query rows attend to nothing, and 111 is off the grid of every BLOCK_M, so fully masked rows share
    # a tile with live ones and cannot take the whole-tile early exit.
    "bottom_right_masked_rows": dict(sq=301, sk=190, input_scale=1.0, window=BR, scale=128**-0.5),
}


@pytest.mark.parametrize("dtype_str", ["f16", "bf16"])
@pytest.mark.parametrize("case", sorted(_COMMON_MISTAKES))
def test_common_mistakes_fwd(fwd_build, case, dtype_str):
    """K01/K05/K07 and lessons 1/2/3/5/7/8: rounding a scaled operand, a logit-domain value rounded to the input
    dtype, the LSE base, -inf arithmetic, fully masked rows."""
    kw = dict(_COMMON_MISTAKES[case])
    window = kw.pop("window")
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=128, window=window is not None))
    fwd_check(fn, b=2, hq=3, d=128, dtype=DTYPES[dtype_str], window=window, ctx=f"{case} {dtype_str}", **kw)


# ---------------------------------------------------------------------------
# FWD-03
# ---------------------------------------------------------------------------

LOG2E = 1.4426950408889634
MAX_ABS_ERR = 0.035  # 1.5x the worst error left once the Q rounding is gone (0.0233 at bf16 hdim 256 scale 1.0)


def _folded_q_model(q, k, v, sm_scale, causal=False):
    """fp64 attention over the scores the *folded* kernel formed: Q scaled in f32 and rounded back to its own dtype,
    then a base-2 softmax. The discriminator: a kernel that folds tracks this model to the noise floor at every scale,
    while its distance to the exact answer grows with `sm_scale * sqrt(head_dim)`."""
    qs = (q.float() * (sm_scale * LOG2E)).to(q.dtype)
    s = qs.double() @ k.double().transpose(-1, -2)
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        s = s.masked_fill(~window_mask(sq, sk, *BR), float("-inf"))
    p = torch.exp2(s - s.max(-1, keepdim=True).values)
    return (p / p.sum(-1, keepdim=True)) @ v.double()


def _max_err(got, want):
    return (got.double() - want).abs().max().item()


@pytest.mark.parametrize("scale", [None, 1.0, 1.2], ids=["rsqrt", "1.0", "1.2"])
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("hdim", [64, 256, 512])
def test_qk_scale_survives_a_large_sm_scale(fwd_build, hdim, dtype_str, scale):
    """K01 (#247): `qk_scale` belongs on the f32 scores, not on Q. Folding it into bf16 Q gave errors up to 0.29 at
    hdim 512; the folded-Q model discriminates."""
    sm = hdim**-0.5 if scale is None else scale
    dtype = DTYPES[dtype_str]
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=hdim))
    r = fwd_check(fn, b=1, hq=4, sq=289, d=hdim, dtype=dtype, scale=sm, seed=hdim, ctx=f"{hdim} {dtype_str} {sm}")
    exact = reference(r["q"], r["k"], r["v"], sm)[0]
    err = _max_err(r["o"], exact)
    err_folded = _max_err(_folded_q_model(r["q"], r["k"], r["v"], sm), exact)
    assert err <= MAX_ABS_ERR, f"max error {err:.4f}; the folded-Q model is {err_folded:.4f} off"
    if err_folded > 2 * MAX_ABS_ERR:
        assert err < err_folded / 2, f"error {err:.4f} is no better than the folded-Q model's {err_folded:.4f}"


@pytest.mark.parametrize("scale", [None, 1.0])
@pytest.mark.parametrize("hdim", [64, 256])
def test_qk_scale_under_causal(fwd_build, hdim, scale):
    """The masked path, where -inf scores meet the scale multiply: `scale_scores` runs before every mask."""
    sm = hdim**-0.5 if scale is None else scale
    fn = fwd_build(meta_of(head_dim=hdim, window=True))
    r = fwd_check(fn, b=1, hq=4, sq=289, d=hdim, dtype=DTYPES["bf16"], scale=sm, window=BR, seed=hdim + 1, ctx="causal")
    assert _max_err(r["o"], r["exact_o"]) <= MAX_ABS_ERR


# ---------------------------------------------------------------------------
# FWD-04
# ---------------------------------------------------------------------------

_SCALES = [0.0, -(64**-0.5), -1.2, 0.05, 0.5, 1.2]


@pytest.mark.parametrize("scale", _SCALES)
@pytest.mark.parametrize("feat", ["dense", "causal", "window", "bias"])
def test_runtime_sm_scale_including_nonpositive(fwd_build, feat, scale):
    """A15/K04: `sm_scale` is a runtime kernarg and may be zero or negative; scaling a masked -inf must not make a NaN.
    At scale 0 the output is the mean of V over the live keys exactly."""
    window = {"causal": TL, "window": (16, 3)}.get(feat)
    meta = meta_of(head_dim=64, window=window is not None, bias=feat == "bias")
    fn = fwd_build(meta)
    gen = seeded(5)
    bias = torch.randn(1, 4, 100, 100, device="cuda", dtype=torch.bfloat16, generator=gen) if feat == "bias" else None
    r = fwd_check(
        fn, b=1, hq=4, sq=100, d=64, dtype=DTYPES["bf16"], scale=scale, window=window, bias=bias, ctx=f"{feat} {scale}"
    )
    if scale == 0.0 and feat == "dense":
        mean = r["v"].double().mean(dim=2, keepdim=True).expand_as(r["o"])
        assert (r["o"].double() - mean).abs().max().item() < 5e-3


# ---------------------------------------------------------------------------
# FWD-05, FWD-06, FWD-07
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
def test_lse_is_natural_base(fwd_build, dtype_str):
    """Lesson 5: Q = K = V = I at scale 1/4 gives every LSE = ln(15 + e^0.25), in natural base, to 1e-4."""
    dtype = DTYPES[dtype_str]
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=16))
    eye = torch.eye(16, device="cuda", dtype=dtype).expand(1, 2, 16, 16)
    q, k, v = (alloc(1, 2, 16, 16, dtype, fill=eye) for _ in range(3))
    o = alloc(1, 2, 16, 16, dtype)
    lse = lse_alloc(1, 2, 16)
    run_fwd(fn, q, k, v, o, lse=lse, scale=0.25)
    want = math.log(15 + math.exp(0.25))
    assert torch.allclose(lse, torch.full_like(lse, want), rtol=1e-4)


@pytest.mark.parametrize("bias_value", [0.0, 4.0, 16.0, -16.0, 64.0])
def test_bias_log2e_symmetry(fwd_build, bias_value):
    """Lesson 4: with one key and zero scores, LSE == bias and O == V; bias x log2e is formed in fp32."""
    fn = fwd_build(meta_of(head_dim=64, bias=True))
    dtype = DTYPES["bf16"]
    q = alloc(1, 2, 8, 64, dtype, fill=torch.zeros(1, 2, 8, 64))
    k = alloc(1, 2, 1, 64, dtype, fill=torch.zeros(1, 2, 1, 64))
    v = randn(1, 2, 1, 64, dtype, gen=seeded(1))
    bias = torch.full((1, 2, 8, 1), bias_value, device="cuda", dtype=dtype)
    o = alloc(1, 2, 8, 64, dtype)
    lse = lse_alloc(1, 2, 8)
    run_fwd(fn, q, k, v, o, lse=lse, bias=bias)
    assert torch.allclose(lse, torch.full_like(lse, bias_value), atol=1e-5, rtol=1e-5)
    assert (o.float() - v.float().expand_as(o)).abs().max().item() < 1e-5


@pytest.mark.parametrize("hdim", [32, 64])
def test_large_logits_no_nan(fwd_build, hdim):
    """Lesson 3/G08: bf16 inputs of magnitude ~1.3e5 at scale 0.25 must not produce NaN or inf (no FMA of an unrounded
    score against a rounded max); O stays within the cap of fp64."""
    fn = fwd_build(meta_of(head_dim=hdim))
    dtype = DTYPES["bf16"]
    gen = seeded(3)
    q = randn(1, 2, 131, hdim, dtype, scale=133120.0, gen=gen)
    k = randn(1, 2, 131, hdim, dtype, scale=133120.0, gen=gen)
    v = randn(1, 2, 131, hdim, dtype, gen=gen)
    o = alloc(1, 2, 131, hdim, dtype)
    run_fwd(fn, q, k, v, o, scale=0.25)
    assert torch.isfinite(o).all()
    exact = reference(q, k, v, 0.25)[0]
    assert (o.double() - exact).abs().max().item() <= MAX_ABS_ERR


@pytest.mark.parametrize("seqlen", [1, 1024], ids=["aotriton", "hot_loop"])
@pytest.mark.parametrize("hdim", [16, 128, 512])
def test_large_bf16_nan_values(fwd_build, hdim, seqlen):
    """AOTriton's `test_large_bf16_nan_values` (aotriton issue 54): Q = K = V = 133120 in bf16 at `sm_scale` 0.125. At
    head_dim 16 every scaled score is about 5.1e10, where an f32 ulp is 4096, so `exp2(fma(s, c, -m))` against the
    rounded max `m` leaves a residue of up to 2048 on the maximal score itself: `exp2` of it is inf or 0 and the output
    NaN. The rounded `exp2(rn(c * s) - m)` is exactly `exp2(0)`. AOTriton's case is seqlen 1; 1024 reaches the
    dual-wave hot loop, where LLVM had fused the two (every seqlen up to 256 passed with the fusion in)."""
    fn = fwd_build(meta_of(head_dim=hdim))
    q, k, v = (torch.full((1, 1, seqlen, hdim), 133120.0, dtype=DTYPES["bf16"], device="cuda") for _ in range(3))
    o = alloc(1, 1, seqlen, hdim, DTYPES["bf16"])
    run_fwd(fn, q, k, v, o, scale=0.125)
    assert not torch.isnan(o).any(), "Output should not contain NaNs!"


# ---------------------------------------------------------------------------
# FWD-08, FWD-09
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("hdim", [64, 128, 512])
def test_fully_masked_rows_get_positive_inf_lse(fwd_build, hdim, dtype_str):
    """K07/K05/K59: a fully masked row stores LSE exactly +inf (never NaN or -inf) and O exactly 0, both on the
    whole-block early exit and in a block straddling live and masked rows (`seqlen_q = 1.5 * BLOCK_M + 64`)."""
    meta = meta_of(dtype_str=dtype_str, head_dim=hdim, window=True)
    fn = fwd_build(meta)
    sq = int(1.5 * fn.knobs.BLOCK_M) + 64
    fwd_check(fn, b=1, hq=2, sq=sq, sk=64, d=hdim, dtype=DTYPES[dtype_str], window=BR, ctx=f"{hdim} {dtype_str}")


def test_window_with_negative_bounds_has_no_nan(fwd_build):
    """Negative window bounds mask whole bands (rows whose live range is empty): +inf LSE, zero O, no NaN."""
    fn = fwd_build(meta_of(head_dim=64, window=True))
    fwd_check(fn, b=1, hq=2, sq=200, sk=300, d=64, dtype=DTYPES["bf16"], window=(-5, 7), ctx="negative bounds")


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("hdim", [64, 384])
def test_bias_minus_inf_rows(fwd_build, hdim, dtype_str):
    """K06/K59/G05: a -inf bias entry is how a caller spells "never attend": a whole -inf row gives O = 0 and LSE = +inf
    exactly, scattered -inf entries match fp64, and no NaN appears (the bias add must not carry `ninf`)."""
    fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=hdim, bias=True))
    dtype = DTYPES[dtype_str]
    b, h, sq, sk = 1, 2, 130, 200
    gen = seeded(9)
    bias = torch.randn(b, h, sq, sk, device="cuda", dtype=torch.float32, generator=gen).to(dtype)
    bias[:, :, 3, :] = float("-inf")  # a whole row
    scatter = torch.rand(b, h, sq, sk, device="cuda", generator=gen) < 0.3
    scatter[:, :, 3, :] = True
    bias = bias.masked_fill(scatter, float("-inf"))
    bias[:, :, 5, :] = 0.0  # rows with finite bias stay live
    fwd_check(fn, b=b, hq=h, sq=sq, sk=sk, d=hdim, dtype=dtype, bias=bias, ctx=f"{hdim} {dtype_str}")


# ---------------------------------------------------------------------------
# FWD-32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1.2])
@pytest.mark.parametrize("hdim", [64, 128, 256])
def test_lazy_rescale_hook_is_off_by_default_and_finite(backend, arch, monkeypatch, tmp_path, hdim, scale):
    """K03/K58: `lazy_rescale` is a trait fixed False; the module-level `_LAZY_RESCALE` is the one hook that flips it.

    The trait stays off as a **precision policy** (plan Q19), not because of NaN: lazy rescale lets P = exp2(S - stale m)
    exceed 1 before the PV MFMA, roughly doubling the error, so enabling it by default would trade the user's precision for
    speed without consent. AOTriton measured non-finite fp16 O with it; on FlyDSL 0.4.0 fp16 is finite. So this checks that
    the hook works (the default is off; flipping the module constant yields a *different* binary, with the ballot-gated
    skip) and that the lazy output is finite and within the floor, so the knob can be studied but is never on by default.
    """
    from kernels.attention import flash_attn_gfx950_config as cfg
    from tests.kernels.attention import isa_tools

    meta = meta_of(dtype_str="f16", head_dim=hdim)
    dtype = DTYPES["f16"]
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, 289, hdim, dtype, gen=gen) for _ in range(3))

    def run(fn):
        o = alloc(1, 4, 289, hdim, dtype)
        run_fwd(fn, q, k, v, o, scale=scale)
        return o

    default = backend.build_fwd(meta, backend.fwd_knobs(arch).resolve(meta))
    assert default.traits.DUALWAVE_SWP_LAZY_RESCALE is False
    isa_default = isa_tools.fresh_fwd_dump(backend, arch, meta, tmp_path / "default", monkeypatch).isa
    monkeypatch.setattr(cfg, "_LAZY_RESCALE", True)
    lazy = backend.build_fwd(meta, backend.fwd_knobs(arch).resolve(meta))
    assert lazy.traits.DUALWAVE_SWP_LAZY_RESCALE is True
    isa_lazy = isa_tools.fresh_fwd_dump(backend, arch, meta, tmp_path / "lazy", monkeypatch).isa
    assert isa_default != isa_lazy, "the lazy build must be a different binary"
    o_default, o_lazy = run(default), run(lazy)
    assert torch.isfinite(o_default).all() and torch.isfinite(o_lazy).all()
    exact = reference(q, k, v, scale)[0]
    assert (o_lazy.double() - exact).abs().max().item() < 0.05


def test_bias_slab_past_i32_is_refused(fwd_build):
    """A (batch, head) bias slab whose element offsets would wrap i32 is refused before launch, on the strides alone."""
    rows, cols = 32769, 65536  # 2**31 + 65536 elements
    fn = fwd_build(meta_of(head_dim=64, bias=True))
    dtype = DTYPES["bf16"]
    q = alloc(1, 1, rows, 64, dtype)
    k = alloc(1, 1, cols, 64, dtype)
    o = alloc(1, 1, rows, 64, dtype)
    bias = torch.empty(rows * cols, device="cuda", dtype=dtype).view(1, 1, rows, cols)
    with pytest.raises(ValueError, match="i32 bias element offsets"):
        run_fwd(fn, q, k, k, o, bias=bias)
