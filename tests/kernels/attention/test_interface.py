# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FWD-37 and FWD-38: the `flydsl_flash_attn_func` contract on the gfx950 bf16/f16 route.

The existing `tests/kernels/test_flash_attn_fwd.py` keeps exercising the legacy surface; these tests pin what the rewiring
promises: any head dim that is a multiple of 8 up to 512 (with a V width of its own), windows and dropout as new arguments,
build options through `knobs=` with the old kwargs deprecated, a bias refused together with a mask, and the LSE convention.
"""

import warnings

import pytest
import torch

from kernels.attention import flash_attn_interface as iface
from kernels.attention.flash_attn_interface import flydsl_flash_attn_func
from tests.kernels.attention.attn_testlib import (
    DTYPES,
    check_floor,
    floor_rel,
    reference,
    sdpa_scale,
    seeded,
    window_mask,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BF16 = DTYPES["bf16"]


@pytest.fixture(autouse=True)
def _require_gfx950(arch):
    if not arch.startswith("gfx950"):
        pytest.skip(f"the gfx950 route, current arch is {arch}")


def bshd(b, s, h, d, gen, dtype=BF16):
    return torch.randn(b, s, h, d, device="cuda", dtype=torch.float32, generator=gen).to(dtype)


def reference_bshd(q, k, v, *, causal=False, window=None, bias=None, p_drop=0.0, keep=None):
    """`(o BSHD, lse (B, H, Sq))` in fp64 from BSHD inputs; `window` is `(left, right)`, `causal` bottom-right."""
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    sq, sk = qt.shape[2], kt.shape[2]
    mask = None
    if causal:
        mask = window_mask(sq, sk, -2147483646, -2147483646)
    elif window is not None:
        mask = window_mask(sq, sk, *window)

    def ref(**kw):
        o, lse = reference(qt, kt, vt, sdpa_scale(q.shape[-1]), mask=mask, bias=bias, **kw)
        return o.transpose(1, 2), lse

    return ref


def gate(out, ref, ctx):
    exact_o, exact_lse = ref()
    floor = floor_rel(lambda **kw: (ref(**kw)[0].transpose(1, 2), None), exact_o.transpose(1, 2), BF16)
    check_floor("O", out, exact_o, floor, ctx)
    return exact_lse


# ---------------------------------------------------------------------------
# Head dims, GQA, cross lengths, LSE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hdim", [8, 40, 64, 96, 128, 136, 192, 264, 384, 512])
def test_any_head_dim_that_is_a_multiple_of_8(hdim, causal):
    """New capability: every multiple of 8 up to 512 (exact rungs, padded ones, and the wide body), dense, GQA, both masks,
    with the LSE next to O."""
    gen = seeded(hdim)
    b, s, h, hk = 1, 400, 4, 2
    q, k, v = bshd(b, s, h, hdim, gen), bshd(b, s, hk, hdim, gen), bshd(b, s, hk, hdim, gen)
    out, lse = flydsl_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    torch.cuda.synchronize()
    kr, vr = k.repeat_interleave(h // hk, dim=2), v.repeat_interleave(h // hk, dim=2)
    exact_lse = gate(out, reference_bshd(q, kr, vr, causal=causal), f"d{hdim} causal={causal}")
    live = torch.isfinite(exact_lse)
    assert (lse[live] - exact_lse[live].float()).abs().max().item() < 2**-16 * max(
        1.0, exact_lse[live].abs().max().item()
    )


def test_v_head_dim_of_its_own():
    """`head_dim_v != head_dim` on the gfx950 half-precision route: the output carries V's width."""
    gen = seeded(5)
    q, k = bshd(1, 130, 4, 120, gen), bshd(1, 130, 4, 120, gen)
    v = bshd(1, 130, 4, 64, gen)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    assert out.shape == (1, 130, 4, 64)
    gate(out, reference_bshd(q, k, v), "dv=64")


@pytest.mark.parametrize("seqs", [(400, 600), (600, 400)], ids=["sq_lt_sk", "sq_gt_sk"])
@pytest.mark.parametrize("causal", [False, True])
def test_cross_lengths_infer_from_the_shapes(seqs, causal):
    """Dense cross-length attention needs no `cross_seqlen`: it comes from the shapes, and causal is bottom-right."""
    sq, sk = seqs
    gen = seeded(sq + sk)
    q, k, v = bshd(1, sq, 4, 64, gen), bshd(1, sk, 4, 64, gen), bshd(1, sk, 4, 64, gen)
    out = flydsl_flash_attn_func(q, k, v, causal=causal)
    gate(out, reference_bshd(q, k, v, causal=causal), f"{seqs} causal={causal}")


def test_varlen_needs_no_cross_seqlen_and_pads_the_lse():
    """Packed varlen: `cross_seqlen` is a runtime matter now (deprecated, ignored), and the returned LSE is main's padded
    `(B, H, max_seqlen_q)` whose first `len_b` entries per batch are defined."""
    gen = seeded(9)
    lens = [300, 500, 260]
    cu = torch.tensor([0, 300, 800, 1060], dtype=torch.int32, device="cuda")
    h, d = 4, 128
    q, k, v = (torch.randn(1060, h, d, device="cuda", generator=gen).to(BF16) for _ in range(3))
    out, lse = flydsl_flash_attn_func(
        q, k, v, causal=True, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), return_lse=True
    )
    torch.cuda.synchronize()
    assert lse.shape == (3, h, 500)
    for z, (s0, s1) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        qz, kz, vz = (t[s0:s1].unsqueeze(0) for t in (q, k, v))
        exact_lse = gate(out[s0:s1].unsqueeze(0), reference_bshd(qz, kz, vz, causal=True), f"seq {z}")
        assert (lse[z, :, : s1 - s0] - exact_lse[0].float()).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# New arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [(31, 0), (63, 63), (0, 31)])
def test_window_replaces_causal(window):
    """`window=(left, right)` is a new argument and replaces `causal`: a banded mask against the same band as a reference
    mask."""
    gen = seeded(31)
    q, k, v = bshd(1, 300, 4, 64, gen), bshd(1, 300, 4, 64, gen), bshd(1, 300, 4, 64, gen)
    out = flydsl_flash_attn_func(q, k, v, window=window)
    gate(out, reference_bshd(q, k, v, window=window), f"window {window}")


def test_dropout_is_deterministic_per_seed_and_zero_is_off():
    gen = seeded(32)
    q, k, v = bshd(1, 256, 4, 64, gen), bshd(1, 256, 4, 64, gen), bshd(1, 256, 4, 64, gen)
    kw = dict(causal=False, dropout_p=0.3)
    a = flydsl_flash_attn_func(q, k, v, philox_seed=11, **kw)
    b = flydsl_flash_attn_func(q, k, v, philox_seed=11, **kw)
    c = flydsl_flash_attn_func(q, k, v, philox_seed=12, **kw)
    plain = flydsl_flash_attn_func(q, k, v, causal=False)
    assert torch.equal(a, b) and not torch.equal(a, c) and (a.float() - plain.float()).abs().max().item() > 1e-2
    with pytest.raises(ValueError, match="philox_seed"):
        flydsl_flash_attn_func(q, k, v, **kw)


def test_bias_with_a_mask_raises():
    """Bias with `causal=True` or a window has no defined meaning (a bias already is an attention mask): refused with the
    reason and the way out. Main's kernel accepted it; its causal bias cases now expect this error."""
    gen = seeded(33)
    q, k, v = bshd(1, 128, 4, 64, gen), bshd(1, 128, 4, 64, gen), bshd(1, 128, 4, 64, gen)
    bias = torch.zeros(128, 128, device="cuda", dtype=BF16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        flydsl_flash_attn_func(q, k, v, causal=True, bias=bias)
    with pytest.raises(ValueError, match="mutually exclusive"):
        flydsl_flash_attn_func(q, k, v, causal=False, window=(8, 0), bias=bias)
    out = flydsl_flash_attn_func(q, k, v, causal=False, bias=bias)  # the legal spelling
    gate(out, reference_bshd(q, k, v), "zero bias")


# ---------------------------------------------------------------------------
# knobs=, and the deprecated build kwargs
# ---------------------------------------------------------------------------

DEPRECATED = [
    ("waves_per_eu", 1, "waves_per_eu"),
    ("daz", False, "daz"),
    ("dualwave_swp_setprio", False, "SETPRIO"),
    ("dualwave_swp_enable_stagger", False, "STAGGER"),
    ("dualwave_swp_xcd_swizzle", True, "XCD_SWIZZLE"),
]


@pytest.fixture
def seen_knobs(monkeypatch):
    """Record the knob overrides each gfx950 build is made with."""
    seen = []
    orig = iface._build_gfx950_fwd

    def spy(meta, items):
        seen.append(dict(items))
        return orig(meta, items)

    monkeypatch.setattr(iface, "_build_gfx950_fwd", spy)
    return seen


@pytest.mark.parametrize("old,value,knob", DEPRECATED, ids=[d[0] for d in DEPRECATED])
def test_deprecated_kwargs_warn_and_forward_to_their_knob(seen_knobs, old, value, knob):
    """Each old build kwarg, on a bf16/f16 call, emits a `DeprecationWarning` naming the knob and is forwarded to it."""
    gen = seeded(34)
    q, k, v = bshd(1, 256, 8, 64, gen), bshd(1, 256, 8, 64, gen), bshd(1, 256, 8, 64, gen)
    with pytest.warns(DeprecationWarning, match=knob):
        flydsl_flash_attn_func(q, k, v, causal=False, **{old: value})
    assert seen_knobs[-1][knob] == value


def test_num_kv_splits_is_deprecated_and_forwarded(seen_knobs):
    gen = seeded(35)
    q, k, v = (bshd(1, 512, 4, 64, gen) for _ in range(3))
    with pytest.warns(DeprecationWarning, match="NUM_KV_SPLITS"):
        out = flydsl_flash_attn_func(q, k, v, causal=False, num_kv_splits=2)
    assert seen_knobs[-1]["NUM_KV_SPLITS"] == 2
    gate(out, reference_bshd(q, k, v), "split-K via the deprecated kwarg")


@pytest.mark.parametrize("old", ["dualwave_swp_lazy_rescale", "cross_seqlen"])
def test_deprecated_kwargs_without_a_knob_warn_and_are_ignored(old):
    gen = seeded(36)
    q, k, v = bshd(1, 256, 4, 64, gen), bshd(1, 256, 4, 64, gen), bshd(1, 256, 4, 64, gen)
    plain = flydsl_flash_attn_func(q, k, v, causal=False, knobs={})
    with pytest.warns(DeprecationWarning, match="ignored"):
        out = flydsl_flash_attn_func(q, k, v, causal=False, **{old: True})
    assert torch.equal(out, plain)


def test_knobs_reach_the_build_and_are_validated(seen_knobs):
    gen = seeded(37)
    q, k, v = bshd(1, 256, 4, 128, gen), bshd(1, 256, 4, 128, gen), bshd(1, 256, 4, 128, gen)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # `knobs=` is the supported spelling: no warning
        flydsl_flash_attn_func(q, k, v, causal=False, knobs={"waves_per_eu": 1, "SETPRIO": False})
    assert seen_knobs[-1]["waves_per_eu"] == 1 and seen_knobs[-1]["SETPRIO"] is False
    with pytest.raises((TypeError, ValueError)):
        flydsl_flash_attn_func(q, k, v, knobs={"NOT_A_KNOB": 1})
    with pytest.raises((TypeError, ValueError)):
        flydsl_flash_attn_func(q, k, v, knobs={"BLOCK_DMODEL": 100})  # not a rung
    with pytest.raises(TypeError):
        flydsl_flash_attn_func(q, k, v, knobs=[("daz", False)])


@pytest.fixture
def gfx950_overrides(monkeypatch):
    """Record the knob overrides the interface hands the gfx950 route; nothing is built or launched."""
    seen = []

    def stub(q, k, v, **kw):
        seen.append(dict(kw["knob_overrides"]))
        return torch.empty_like(q)

    monkeypatch.setattr(iface, "_flydsl_flash_attn_gfx950", stub)
    return seen


# (B, S, H, call kwargs, expected XCD_SWIZZLE override: None = not set by the interface)
XCD_AUTO_CASES = [
    pytest.param(1, 16384, 8, {}, True, id="eligible"),
    pytest.param(2, 16129, 16, {}, True, id="eligible_64_blocks_at_the_edge"),
    pytest.param(1, 16128, 8, {}, None, id="63_q_blocks"),
    pytest.param(1, 16384, 12, {}, None, id="heads_not_a_multiple_of_8"),
    pytest.param(1, 16384, 8, {"causal": True}, None, id="causal"),
    pytest.param(1, 16384, 8, {"window": (127, 0)}, None, id="window"),
    pytest.param(1, 16384, 8, {"num_kv_splits": 2}, None, id="split_k"),
    pytest.param(1, 16384, 8, {"knobs": {"XCD_SWIZZLE": False}}, False, id="a_pin_to_off_is_kept"),
]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # the split-K case spells num_kv_splits the old way
@pytest.mark.parametrize("B,S,H,kw,expected", XCD_AUTO_CASES)
def test_xcd_swizzle_is_set_for_the_calls_it_applies_to(gfx950_overrides, B, S, H, kw, expected):
    """The knob is only ever set here: `XCD_SWIZZLE` defaults to off in the config, so an interface that stopped setting it
    would silently turn the head-slow mapping off for every call (main enables it for dense calls with `H % 8 == 0` and at
    least 64 q blocks). A pin from the caller wins."""
    q, k, v = (torch.empty(B, S, H, 128, device="cuda", dtype=BF16) for _ in range(3))
    flydsl_flash_attn_func(q, k, v, **{"causal": False, **kw})
    assert gfx950_overrides[-1].get("XCD_SWIZZLE") is expected


def test_fp8_calls_do_not_see_the_deprecation():
    """The fp8 kernels (out of scope) still read the old kwargs: no warning, and the legacy defaults are filled in."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values, overrides = iface._legacy_build_options(
            "fp8",
            waves_per_eu=1,
            daz=iface._UNSET,
            dualwave_swp_lazy_rescale=iface._UNSET,
            dualwave_swp_setprio=False,
            dualwave_swp_enable_stagger=iface._UNSET,
            dualwave_swp_xcd_swizzle=None,
            num_kv_splits=None,
            debug_counts=None,
            cross_seqlen=None,
        )
    assert (
        overrides == {}
        and values["waves_per_eu"] == 1
        and values["daz"] is True
        and values["dualwave_swp_setprio"] is False
    )


# ---------------------------------------------------------------------------
# FWD-38: the LSE of a row that attends nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seqs", [(512, 128), (601, 190)])
def test_lse_of_fully_masked_rows_is_exactly_plus_inf(seqs):
    """Bottom-right causal with `Sq > Sk`: the first `Sq - Sk` rows attend no key. Their LSE is `+inf` exactly (the backward
    needs it: `exp(s - lse)` is then 0), not `-inf` and not whatever the caller's buffer held; their O is 0."""
    sq, sk = seqs
    gen = seeded(sq)
    q, k, v = bshd(1, sq, 4, 64, gen), bshd(1, sk, 4, 64, gen), bshd(1, sk, 4, 64, gen)
    out, lse = flydsl_flash_attn_func(q, k, v, causal=True, return_lse=True)
    torch.cuda.synchronize()
    dead = sq - sk
    assert (lse[:, :, :dead] == float("inf")).all() and (out[:, :dead] == 0).all()
    assert torch.isfinite(lse[:, :, dead:]).all()
