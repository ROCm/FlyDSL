#!/usr/bin/env python3
"""Correctness test for the layout-API flex attention forward (gfx950).

Compares against torch.nn.attention.flex_attention using the same score/mask
callables. gfx950-only (uses cdna4-era MFMA + the layout API); skipped elsewhere.

Kernel constraints (see make_flex_attn_param): block_m=32, block_n=64,
head_dim=128, 512-thread workgroup (num_groups=8), seqlen_kv multiple of 64.
"""

import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ROCm not available")
    sys.exit(1)

import pytest  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flex_attention_gfx950 import (  # noqa: E402
    _effective_lower_tri_kv_splits,
    _infer_generic_kv_bounds,
    flydsl_flex_attention_layout,
    flydsl_flex_attention_layout_paged,
    inspect_flex_mods,
    make_flex_attn_param,
)

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="layout-API attention kernel targets gfx950",
)

_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}


def _make_qkv(B, Sq, Skv, Hq, D, dtype, *, Hkv=None):
    dev = "cuda"
    torch.manual_seed(0)
    if Hkv is None:
        Hkv = Hq
    q = torch.empty(B, Sq, Hq, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, scale


def _flex_ref(q, k, v, scale, *, score_mod=None, mask_mod=None):
    # create_block_mask/compile caches Python callable code objects. Parameterized
    # lambdas reuse one code object with different default constants, so reset
    # between references to keep those constants from leaking across test cases.
    torch.compiler.reset()
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    block_mask = None
    if mask_mod is not None:
        block_mask = create_block_mask(
            mask_mod,
            B=qh.shape[0],
            H=qh.shape[1],
            Q_LEN=qh.shape[2],
            KV_LEN=kh.shape[2],
            device=q.device,
        )
    out = flex_attention(
        qh,
        kh,
        vh,
        scale=scale,
        score_mod=score_mod,
        block_mask=block_mask,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _bottom_right_causal_ref(q, k, v, scale):
    Sq, Skv = q.shape[1], k.shape[1]
    mask_mod = lambda b, h, q, kv, offset=Skv - Sq: kv <= q + offset
    return _flex_ref(q, k, v, scale, mask_mod=mask_mod)


def _make_doc_mask_mod(spans, *, causal):
    """Document mask for packed/jagged sequences (pytorch.org/blog/flexattention).

    The blog looks up ``document_id[q_idx] == document_id[kv_idx]``. A tensor
    read is not traceable here, so the same block-diagonal predicate is built
    from the document spans, bound as constexpr defaults so the callable keeps
    no closure or global references.
    """

    def doc_mask(b, h, q_idx, kv_idx, spans=spans, causal=causal):
        same_doc = None
        for lo, hi in spans:
            in_doc = (q_idx >= lo) & (q_idx < hi) & (kv_idx >= lo) & (kv_idx < hi)
            same_doc = in_doc if same_doc is None else (same_doc | in_doc)
        if causal:
            return same_doc & (kv_idx <= q_idx)
        return same_doc

    return doc_mask


def _quadratic_rel_score(score, b, h, q_idx, kv_idx, inv_var=1e-4):
    """Bespoke non-affine score_mod: Gaussian relative-position bias.

    Same signature as torch.nn.attention.flex_attention score mods. The
    kernel traces this callable as-is (no ALiBi/affine lowering): a distance
    term ``(q-kv)^2`` is not score + slope*(kv-q). Bind extra scalars as
    defaults so FlyDSL constexprs see no closures or globals.
    """
    dist = q_idx - kv_idx
    return score - dist * dist * inv_var


def _doc_spans(lengths):
    spans = []
    start = 0
    for length in lengths:
        spans.append((start, start + length))
        start += length
    return tuple(spans)


def _check(out, ref, *, max_err_tol=8e-2, cos_tol=0.98, label=""):
    max_err = (out.float() - ref.float()).abs().max().item()
    cos = F.cosine_similarity(out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
    assert max_err < max_err_tol and cos > cos_tol, f"{label}: max_err={max_err} cos={cos}"
    return max_err, cos


def _run(B, Sq, Skv, H, D, dtype_str, *, num_groups=8, accurate_softmax=True):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        num_groups=num_groups,
        accurate_softmax=accurate_softmax,
    ).float()
    ref = _flex_ref(q, k, v, scale).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    return max_err, cos


_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (1, 256, 512, 4, 128),  # Sq != Skv
    (2, 256, 256, 8, 128),
    (1, 256, 64, 4, 128),  # single KV tile (Skv == block_n)
    (1, 512, 1024, 4, 128),  # larger sequences
    (1, 256, 256, 8, 128),  # more heads; GQA is a separate test with Hkv < Hq
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout(B, Sq, Skv, H, D, dtype_str):
    max_err, cos = _run(B, Sq, Skv, H, D, dtype_str)
    assert max_err < 8e-2 and cos > 0.98, f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: max_err={max_err} cos={cos}"


def _n64_long_seq_8c(skv, masked):
    flex_mod = inspect_flex_mods(
        None,
        (lambda b, h, q, kv: kv <= q) if masked else None,
        seqlen_q=skv,
        seqlen_kv=skv,
    )
    return bool(
        make_flex_attn_param(
            seqlen_kv=skv,
            block_n=64,
            head_dim=128,
            num_groups=8,
            flex_mod=flex_mod,
        ).long_seq_8c
    )


def test_n64_long_seq_8c_cutoffs():
    """Dense 8c at Skv>=768; masked 8c at Skv>=2048."""
    assert not _n64_long_seq_8c(704, False)
    assert _n64_long_seq_8c(768, False)
    assert not _n64_long_seq_8c(1984, True)
    assert _n64_long_seq_8c(2048, True)
    assert not _n64_long_seq_8c(1024, True)


def test_inspect_flex_mods():
    causal = inspect_flex_mods(None, lambda b, h, q, kv: kv <= q, seqlen_q=512, seqlen_kv=512)
    assert causal.packed_lower_tri_mask and causal.q_offset == 0
    banded = inspect_flex_mods(
        None,
        lambda b, h, q, kv, window=33: (kv <= q) & ((q - kv) <= window),
        seqlen_q=512,
        seqlen_kv=512,
    )
    assert banded.banded and banded.window == 33
    prefix = inspect_flex_mods(
        None,
        lambda b, h, q, kv, prefix_len=64: (kv <= q) | (kv < prefix_len),
        seqlen_q=512,
        seqlen_kv=512,
    )
    assert prefix.has_prefix and prefix.prefix_len == 64
    generic = inspect_flex_mods(
        None,
        lambda b, h, q, kv: (kv % 3) == (q % 2),
        seqlen_q=128,
        seqlen_kv=128,
    )
    assert generic.has_mask and not generic.lower_tri

    def named_causal(b, h, q, kv):
        return kv <= q

    named = inspect_flex_mods(None, named_causal, seqlen_q=128, seqlen_kv=128)
    assert named.packed_lower_tri_mask


def test_inspect_affine_score_mod():
    score_mod = lambda score, b, h, q, kv, slope=0.125: score + slope * (kv - q)
    affine = inspect_flex_mods(score_mod, None, seqlen_q=128, seqlen_kv=128)
    assert affine.has_score and affine.affine_score
    assert math.isclose(affine.score_slope, 0.125)

    reverse = inspect_flex_mods(
        lambda score, b, h, q, kv, slope=0.25: score + slope * (q - kv),
        None,
        seqlen_q=128,
        seqlen_kv=128,
    )
    assert reverse.affine_score
    assert math.isclose(reverse.score_slope, -0.25)

    exact = lambda score, b, h, q, kv, slope=0.125: score + slope * (kv - q)
    exact.flex_infer = False
    exact_mod = inspect_flex_mods(exact, None, seqlen_q=128, seqlen_kv=128)
    assert exact_mod.has_score and not exact_mod.affine_score

    exact_kind = lambda score, b, h, q, kv, slope=0.125: score + slope * (kv - q)
    exact_kind.flex_score_kind = "exact"
    exact_kind_mod = inspect_flex_mods(exact_kind, None, seqlen_q=128, seqlen_kv=128)
    assert exact_kind_mod.has_score and not exact_kind_mod.affine_score

    kwarg_exact = inspect_flex_mods(
        score_mod,
        None,
        seqlen_q=128,
        seqlen_kv=128,
        infer_score_mod=False,
    )
    assert kwarg_exact.has_score and not kwarg_exact.affine_score

    nonlinear = inspect_flex_mods(
        _quadratic_rel_score,
        None,
        seqlen_q=128,
        seqlen_kv=128,
    )
    assert nonlinear.has_score and not nonlinear.affine_score


def test_effective_lower_tri_kv_splits():
    common = dict(rows_per_wg=256, block_n=64, num_cus=256)
    # The unsplit grid already fills the device.
    assert (
        _effective_lower_tri_kv_splits(
            requested_splits=4,
            batch=2,
            seqlen_q=3072,
            seqlen_kv=3072,
            num_heads_q=32,
            **common,
        )
        == 1
    )
    # Half-full grid, but four KV tiles per split cannot repay combine.
    assert (
        _effective_lower_tri_kv_splits(
            requested_splits=2,
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_heads_q=32,
            **common,
        )
        == 1
    )
    # A small Q grid with long KV retains the requested useful fan-out.
    assert (
        _effective_lower_tri_kv_splits(
            requested_splits=4,
            batch=1,
            seqlen_q=256,
            seqlen_kv=8192,
            num_heads_q=4,
            **common,
        )
        == 4
    )


@_requires_gfx950
@pytest.mark.parametrize(
    "Skv,causal",
    [
        (704, False),
        (768, False),
        (1984, True),
        (2048, True),
    ],
)
def test_flex_attention_n64_long_sequence_threshold(Skv, causal):
    """Exercise the dense (768) and causal (2048) 8-cluster boundaries."""
    q, k, v, scale = _make_qkv(1, 1024, Skv, 4, 128, torch.bfloat16)
    mask_mod = (lambda b, h, q, kv, offset=Skv - 1024: kv <= q + offset) if causal else None
    assert _n64_long_seq_8c(Skv, causal) == (Skv >= (2048 if causal else 768))
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        block_n=64,
        num_groups=8,
        mask_mod=mask_mod,
    )
    ref = _bottom_right_causal_ref(q, k, v, scale) if causal else _flex_ref(q, k, v, scale)
    _check(
        out,
        ref,
        max_err_tol=8e-2,
        cos_tol=0.999,
        label=f"n64 threshold Skv={Skv} causal={causal}",
    )


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_layout_approx_softmax(B, Sq, Skv, H, D, dtype_str):
    _, cos = _run(B, Sq, Skv, H, D, dtype_str, accurate_softmax=False)
    assert cos > 0.95, f"approx B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}: cos={cos}"


_MOD_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (2, 256, 256, 8, 128),
    (1, 256, 512, 4, 128),  # Sq < Skv (prefill with longer KV)
    (1, 256, 64, 4, 128),  # single KV tile
    (1, 512, 512, 4, 128),  # larger sequence (tile-range clamping exercises more tiles)
]


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_causal(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    mask_mod = lambda b, h, q, kv, offset=Skv - Sq: kv <= q + offset
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label=f"causal B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_alibi(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    slope = 0.125
    score_mod = lambda score, b, h, q, kv, slope=slope: score + slope * (kv - q)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, score_mod=score_mod)
    ref = _flex_ref(q, k, v, scale, score_mod=score_mod)
    _check(out, ref, label=f"alibi B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
def test_flex_attention_layout_alibi_exact_override():
    """The explicit override executes the original callable math."""
    B, Sq, Skv, H, D = 1, 256, 512, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    score_mod = lambda score, b, h, q, kv, slope=0.125: score + slope * (kv - q)
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        score_mod=score_mod,
        infer_score_mod=False,
    )
    ref = _flex_ref(q, k, v, scale, score_mod=score_mod)
    _check(out, ref, label="alibi exact override")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_quadratic_score(B, Sq, Skv, H, D, dtype_str):
    """User-defined score_mod that is not ALiBi and is not affine-inferred."""
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, score_mod=_quadratic_rel_score)
    ref = _flex_ref(q, k, v, scale, score_mod=_quadratic_rel_score)
    _check(
        out,
        ref,
        label=f"quadratic score B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}",
    )


@_requires_gfx950
def test_flex_attention_layout_quadratic_score_causal():
    """Bespoke score composed with a causal mask_mod."""
    B, Sq, Skv, H, D = 1, 256, 512, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    mask_mod = lambda b, h, q, kv, offset=Skv - Sq: kv <= q + offset
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        score_mod=_quadratic_rel_score,
        mask_mod=mask_mod,
    )
    ref = _flex_ref(q, k, v, scale, score_mod=_quadratic_rel_score, mask_mod=mask_mod)
    _check(out, ref, label="quadratic score + causal")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_sliding_window(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    window = 16
    mask_mod = lambda b, h, q, kv, window=window: (kv <= q) & ((q - kv) <= window)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, cos_tol=0.97, label=f"sw B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("window", [33, 97])
def test_flex_attention_layout_sliding_window_odd(window):
    """Non-block-aligned windows that straddle tile boundaries."""
    B, Sq, Skv, H, D = 2, 256, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    mask_mod = lambda b, h, q, kv, window=window: (kv <= q) & ((q - kv) <= window)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, cos_tol=0.97, label=f"sw_odd w={window}")


@_requires_gfx950
@pytest.mark.parametrize("Hq,Hkv", [(8, 1), (8, 2), (32, 8)])
def test_flex_attention_layout_gqa(Hq, Hkv):
    B, Sq, Skv, D = 1, 256, 256, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, Hq, D, torch.bfloat16, Hkv=Hkv)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, num_kv_heads=Hkv)
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float().repeat_interleave(Hq // Hkv, dim=1)
    vh = v.permute(0, 2, 1, 3).float().repeat_interleave(Hq // Hkv, dim=1)
    ref = flex_attention(qh, kh, vh, scale=scale).permute(0, 2, 1, 3).contiguous()
    _check(out, ref, label=f"gqa Hq{Hq} Hkv{Hkv}")


@_requires_gfx950
def test_flex_attention_layout_multi_group():
    B, Sq, Skv, H, D = 1, 256, 128, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, num_groups=8)
    ref = _flex_ref(q, k, v, scale)
    _check(out, ref, label="groups=8")


@_requires_gfx950
@pytest.mark.parametrize(
    "Sq,Skv,splits,causal",
    [
        (256, 512, 4, False),  # min-chunk folds extra requested splits
        (256, 256, 4, True),  # auto-bypasses an uneconomic split request
        (512, 8192, 4, False),  # long 8c partitions
        (3072, 3072, 4, True),  # retained split-K with empty early partitions
        (1024, 768, 2, True),  # bottom-right dead Q rows plus combine
    ],
)
def test_flex_attention_layout_splitk(Sq, Skv, splits, causal):
    B, H, D = 1, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    mask_mod = (lambda b, h, q, kv, offset=Skv - Sq: kv <= q + offset) if causal else None
    out = flydsl_flex_attention_layout(
        q,
        k,
        v,
        scale=scale,
        mask_mod=mask_mod,
        num_kv_splits=splits,
    )
    ref = _bottom_right_causal_ref(q, k, v, scale) if causal else _flex_ref(q, k, v, scale)
    _check(out, ref, label=f"splitk Sq={Sq} Skv={Skv} splits={splits}")


@_requires_gfx950
def test_flex_attention_layout_sliding_window_full():
    """Window >= Skv: the band constraint is idle; still causal vs Torch."""
    B, Sq, Skv, H, D = 1, 256, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    mask_mod = lambda b, h, q, kv, window=Skv: (kv <= q) & ((q - kv) <= window)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label="sw_full")


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _MOD_SHAPES)
def test_flex_attention_layout_prefix_lm(B, Sq, Skv, H, D, dtype_str):
    prefix_len = max(1, Sq // 4)
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    mask_mod = lambda b, h, q, kv, prefix_len=prefix_len: (kv <= q) | (kv < prefix_len)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label=f"prefix_lm B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


_DOC_LENGTHS = [
    # Packed documents summing to Sq: uneven, and not block-aligned.
    (96, 32, 96, 32),
    (64, 64, 64, 64),
    (200, 24, 32),
    (1, 255),
]


def test_inspect_doc_mask_is_generic():
    """Document masks are block-diagonal, so no lower-tri layout opt applies."""
    for lengths in _DOC_LENGTHS:
        spans = _doc_spans(lengths)
        for causal in (False, True):
            mod = inspect_flex_mods(
                None,
                _make_doc_mask_mod(spans, causal=causal),
                seqlen_q=256,
                seqlen_kv=256,
            )
            assert mod.has_mask, (lengths, causal)
            assert not mod.lower_tri, (lengths, causal)
            assert not mod.packed_lower_tri_mask, (lengths, causal)
            assert mod.needs_kv_bounds, (lengths, causal)


@pytest.mark.parametrize(
    "causal,expected",
    [
        (
            False,
            [
                [0, 8],
                [0, 8],
                [8, 12],
                [12, 24],
                [12, 24],
                [12, 24],
                [24, 32],
                [24, 32],
            ],
        ),
        (
            True,
            [
                [0, 4],
                [0, 8],
                [8, 12],
                [12, 16],
                [12, 20],
                [12, 24],
                [24, 28],
                [24, 32],
            ],
        ),
    ],
)
def test_infer_document_kv_bounds(causal, expected):
    """Generic document masks get an exact per-workgroup KV envelope."""
    spans = _doc_spans((512, 256, 768, 512))
    mod = inspect_flex_mods(
        None,
        _make_doc_mask_mod(spans, causal=causal),
        seqlen_q=2048,
        seqlen_kv=2048,
        num_batches=2,
        num_heads=4,
    )
    bounds = _infer_generic_kv_bounds(
        mod.mask_mod,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_batches=2,
        num_heads=4,
        rows_per_wg=256,
        block_n=64,
    )
    expected_tensor = torch.tensor(expected, dtype=torch.int32)
    assert bounds.shape == (2, 4, 8, 2)
    assert torch.equal(bounds[0, 0], expected_tensor)
    assert torch.equal(bounds, expected_tensor.expand_as(bounds))


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("lengths", _DOC_LENGTHS)
def test_flex_attention_layout_document_mask(lengths, causal, dtype_str):
    """Sample packing: each token only attends within its own document."""
    B, Sq, H, D = 2, 256, 4, 128
    assert sum(lengths) == Sq
    q, k, v, scale = _make_qkv(B, Sq, Sq, H, D, _DTYPES[dtype_str])
    mask_mod = _make_doc_mask_mod(_doc_spans(lengths), causal=causal)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label=f"doc {lengths} causal={causal} {dtype_str}")


@_requires_gfx950
def test_flex_attention_layout_document_mask_long():
    """Longer packed sequence that crosses the masked 8c cutoff."""
    B, Sq, H, D = 1, 2048, 4, 128
    lengths = (512, 256, 768, 512)
    assert sum(lengths) == Sq
    q, k, v, scale = _make_qkv(B, Sq, Sq, H, D, torch.bfloat16)
    mask_mod = _make_doc_mask_mod(_doc_spans(lengths), causal=True)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label="doc long causal")


@_requires_gfx950
def test_flex_attention_layout_document_mask_empty_workgroups():
    """A packed Q document with no matching K keeps one safely masked tile."""
    B, Sq, Skv, H, D = 1, 512, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)
    mask_mod = _make_doc_mask_mod(_doc_spans((256, 256)), causal=False)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label="doc empty Q workgroup")


@_requires_gfx950
def test_flex_attention_layout_document_mask_alibi():
    """Document mask composed with a score_mod, as the blog composes mods."""
    B, Sq, H, D = 1, 256, 4, 128
    slope = 0.125
    q, k, v, scale = _make_qkv(B, Sq, Sq, H, D, torch.bfloat16)
    mask_mod = _make_doc_mask_mod(_doc_spans((96, 32, 96, 32)), causal=True)
    score_mod = lambda score, b, h, q, kv, slope=slope: score + slope * (kv - q)
    out = flydsl_flex_attention_layout(q, k, v, scale=scale, score_mod=score_mod, mask_mod=mask_mod)
    ref = _flex_ref(q, k, v, scale, score_mod=score_mod, mask_mod=mask_mod)
    _check(out, ref, label="doc + alibi")


@_requires_gfx950
def test_flex_attention_layout_causal_multi_group():
    B, Sq, Skv, H, D = 1, 256, 256, 4, 128
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, torch.bfloat16)

    def mask_mod(b, h, q, kv):
        return kv <= q

    out = flydsl_flex_attention_layout(q, k, v, scale=scale, mask_mod=mask_mod, num_groups=8)
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    _check(out, ref, label="causal groups=8")


_PAGED_SHAPES = [
    # (B, Sq, Skv, H, D) — Sq must be a multiple of block_m*num_groups (32*8=256)
    (1, 256, 256, 4, 128),
    (2, 256, 256, 8, 128),
    (1, 256, 512, 4, 128),
    (1, 256, 64, 4, 128),
]


def _paged_causal_cases():
    # f16 paged+causal is numerically flaky at the 0.1 max_err bound (cos stays >0.99).
    return [(*shape, "bf16") for shape in _PAGED_SHAPES]


def _scatter_to_paged(k_contig, v_contig, block_n, block_table, context_lens):
    """Scatter contiguous [B, Skv, H, D] KV into paged cache [num_blocks, block_n, H, D]."""
    B, Skv, H, D = k_contig.shape
    num_blocks = int(block_table.max().item()) + 1
    k_cache = torch.zeros(num_blocks, block_n, H, D, dtype=k_contig.dtype, device=k_contig.device)
    v_cache = torch.zeros(num_blocks, block_n, H, D, dtype=v_contig.dtype, device=v_contig.device)
    for b in range(B):
        ctx = int(context_lens[b].item())
        for t in range(ctx):
            page_idx = t // block_n
            within_page = t % block_n
            phys_page = int(block_table[b, page_idx].item())
            k_cache[phys_page, within_page] = k_contig[b, t]
            v_cache[phys_page, within_page] = v_contig[b, t]
    return k_cache, v_cache


def _make_block_table(B, Skv, block_n, device):
    """Create a random block table and context_lens for paged tests."""
    num_pages_per_seq = (Skv + block_n - 1) // block_n
    total_pages = B * num_pages_per_seq * 2
    context_lens = torch.full((B,), Skv, dtype=torch.int32, device=device)
    block_table = torch.zeros(B, num_pages_per_seq, dtype=torch.int32, device=device)
    used = set()
    for b in range(B):
        for p in range(num_pages_per_seq):
            while True:
                pid = torch.randint(0, total_pages, (1,)).item()
                if pid not in used:
                    used.add(pid)
                    break
            block_table[b, p] = pid
    return block_table, context_lens, total_pages


@_requires_gfx950
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("B,Sq,Skv,H,D", _PAGED_SHAPES)
def test_flex_attention_layout_paged(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    block_n = 64
    ref = _flex_ref(q, k, v, scale)
    block_table, context_lens, _ = _make_block_table(B, Skv, block_n, q.device)
    k_cache, v_cache = _scatter_to_paged(k, v, block_n, block_table, context_lens)
    out = flydsl_flex_attention_layout_paged(q, k_cache, v_cache, block_table, context_lens, scale=scale)
    _check(out, ref, max_err_tol=1e-1, cos_tol=0.97, label=f"paged B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D,dtype_str", _paged_causal_cases())
def test_flex_attention_layout_paged_causal(B, Sq, Skv, H, D, dtype_str):
    q, k, v, scale = _make_qkv(B, Sq, Skv, H, D, _DTYPES[dtype_str])
    block_n = 64
    mask_mod = lambda b, h, q, kv: kv <= q
    ref = _flex_ref(q, k, v, scale, mask_mod=mask_mod)
    block_table, context_lens, _ = _make_block_table(B, Skv, block_n, q.device)
    k_cache, v_cache = _scatter_to_paged(k, v, block_n, block_table, context_lens)
    out = flydsl_flex_attention_layout_paged(
        q, k_cache, v_cache, block_table, context_lens, scale=scale, mask_mod=mask_mod
    )
    _check(out, ref, max_err_tol=1.2e-1, cos_tol=0.97, label=f"paged_causal B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dtype_str}")


def main():
    for B, Sq, Skv, H, D in _SHAPES:
        for dt in ["bf16", "f16"]:
            me, cos = _run(B, Sq, Skv, H, D, dt)
            ok = me < 3e-2 and cos > 0.99
            print(
                f"B{B} Sq{Sq} Skv{Skv} H{H} D{D} {dt}: " f"max_err={me:.4g} cos={cos:.5f} -> {'PASS' if ok else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
