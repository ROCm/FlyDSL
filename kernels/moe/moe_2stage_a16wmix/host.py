# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side launch glue for the fused a16w4 (bf16 A x mxfp4 W) MoE kernels.

Ported (self-contained) from aiter's mixed_moe kernels. bf16 A (no A-scale),
bf16 ``[sorted_size, inter]`` intermediate (no intermediate scale). Reuses the
standard sorting/cumsum/m_indices contract and the shuffle_weight+e8m0_shuffle W
layout. Kernel launch args are raw device pointers (``fx.Int64``); tensors are
passed as ``.data_ptr()``.
"""

import csv
import functools
import re

import torch

from kernels.common.tensor_shim import _run_compiled
from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid


@functools.cache
def _get_compiled_gemm1_a16w4(
    BM, D_HIDDEN, D_INTER, NE, topk, TILE_N, TILE_K, act, b_cache_mod, xcd_swizzle, waves_per_eu, w_dtype="mxfp4"
):
    return compile_gemm1_a16w4_port(
        BM=BM,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        act=act,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
    )


@functools.cache
def _get_compiled_gemm2_a16w4(
    BM, NE, N_OUT, D_INTER, TILE_N, TILE_K, b_cache_mod=2, xcd_swizzle=1, waves_per_eu=None, w_dtype="mxfp4"
):
    return compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=N_OUT,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
    )


def a16wi4_scale_to_kernel_layout(scale_ng):
    """Re-layout a logical groupwise int4 scale ``[E, N, num_groups]`` into the
    ``(E, N, num_groups//2, 2)`` bf16-pair layout the a16wi4 kernel expects.

    The kernel indexes the scale per-lane by N and reads bf16 pairs (two adjacent
    K-groups per dword): dword index ``= n*(G//2) + group//2``, even group -> low
    bf16, odd group -> high bf16. ``num_groups`` must be even. Input is already
    N-major (``[E, N, G]``); we just pack consecutive group pairs into the last dim.
    """
    E, N, G = scale_ng.shape
    assert G % 2 == 0, f"num_groups must be even for bf16-pair packing, got {G}"
    s = scale_ng.to(torch.bfloat16).contiguous().view(E, N, G // 2, 2).contiguous()
    return s


def flydsl_a16w4_gemm1(
    *,
    a_bf16,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=None,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=0,
    gate_mode="separated",
    act="silu",
    w_dtype="mxfp4",
    stream=None,
):
    """a16w4/a16wi4 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="mxfp4"`` (default): W1 is mxfp4 (``w1_scale_u8`` = shuffled e8m0).
    ``w_dtype="int4"`` (a16wi4): W1 is packed signed int4 (same preshuffle byte
    layout as mxfp4) and ``w1_scale_u8`` is the groupwise bf16 scale already in the
    ``(E, N_OUT, G//2, 2)`` kernel layout (see :func:`a16wi4_scale_to_kernel_layout`).

    ``a_bf16`` is the bf16 activation ``[n_tokens, D_HIDDEN]``. Writes the bf16
    intermediate ``[sorted_size, D_INTER]`` (by sorted position) into
    ``inter_sorted_bf16`` (pre-allocated). No A-scale, no intermediate scale.

    Tile-config interface mirrors aiter's ``compile_mixed_moe_gemm1_a16w4``:
    ``tile_m`` (block M -> BM), ``tile_n`` (N tile -> TILE_N), ``tile_k`` (K tile
    -> TILE_K), ``waves_per_eu`` (min-occupancy hint -> ``rocdl.waves_per_eu``),
    ``b_nt`` (W-load cache modifier -> b_cache_mod; 0=cached, 2=nt), ``xcd_swizzle``
    (bijective XCD/HBM-channel remap of the launch grid). ``k_batch`` (split-K) and
    ``gate_mode`` are accepted for interface parity but this kernel only supports
    ``k_batch=1`` / ``gate_mode="separated"``.

    ``tile_n=None`` picks the largest supported N tile that divides ``D_INTER``:
    256 when ``D_INTER % 256 == 0`` (fastest, matches aiter's tuned tile) else 128
    (``D_INTER`` is always a multiple of 128 given the ``2*D_INTER % 256`` rule).

    ``b_nt=None`` uses the measured per-M W-load U-shape (same mechanism as gemm2):
    non-temporal streaming (b_nt=2) wins the mid-band, where each expert's W tiles
    are reused across only a few M-blocks so caching pollutes L2; cached (b_nt=0)
    wins at the ends (tiny M -> few blocks per expert; large M >= 2048 -> high W
    L2 residency). Measured (median-of-3, 3584x512, dev7): b_nt=2 is -7..-11% s1 at
    tok 16/128/1024 but +11%/+43% *regression* at tok 4096/16384, so the switch is
    keyed on n_tokens. Caller may pin either mode via an explicit ``b_nt``.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}")
    BM = tile_m
    TILE_K = tile_k
    _m = int(n_tokens)
    b_cache_mod = (2 if (16 <= _m <= 1024) else 0) if b_nt is None else b_nt
    TILE_N = tile_n
    if TILE_N is None:
        TILE_N = 256 if D_INTER % 256 == 0 else 128
    if D_HIDDEN % TILE_K != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires D_HIDDEN (K) % {TILE_K} == 0, got H={D_HIDDEN}")
    if (2 * D_INTER) % 256 != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires 2*D_INTER % 256 == 0, got D_INTER={D_INTER}")
    if D_INTER % TILE_N != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires D_INTER % TILE_N({TILE_N}) == 0, got D_INTER={D_INTER}")

    launch = _get_compiled_gemm1_a16w4(
        BM, D_HIDDEN, D_INTER, NE, topk, TILE_N, TILE_K, act, b_cache_mod, xcd_swizzle, waves_per_eu, w_dtype
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    _run_compiled(
        launch,
        a_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def flydsl_a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=256,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=1,
    w_dtype="mxfp4",
    stream=None,
):
    """a16w4/a16wi4 fused stage2 (down-proj). Consumes the bf16 [sorted_size, D_INTER]
    intermediate; scatters routing-weighted bf16 into ``flat_out`` [tokens*model_dim].

    Tile-config interface mirrors aiter's ``compile_mixed_moe_gemm2_a16w4``:
    ``tile_m`` -> BM, ``tile_n`` (model_dim N tile) -> TILE_N, ``tile_k`` (inter K
    tile) -> TILE_K, ``waves_per_eu`` -> ``rocdl.waves_per_eu``, ``b_nt`` -> W-load
    cache modifier, ``xcd_swizzle`` -> XCD/HBM-channel grid remap. ``k_batch`` is
    accepted for parity (must be 1). ``b_nt=None`` keeps the measured per-M U-shape
    default (cached at both ends, nt in the middle band).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")
    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
    if D_INTER % TILE_K != 0:
        raise NotImplementedError(f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}")
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}")

    # B (mxfp4 weight) cache modifier, per-token. Measured (median-of-5, 3584x512)
    # is a U-shape: CACHED loads (cache_modifier=0) win at BOTH ends -- small M
    # (whole expert's B reused across few M-blocks) and large M (>=2048; high L2
    # residency, L2 hit ~65% -> streaming would bypass a reusable cache) -- while
    # non-temporal streaming (cache_modifier=2) wins the middle band (32..1024,
    # where streaming avoids L2 pollution). Crossovers: cached<->nt between
    # tok16/32, nt<->cached between tok1024/2048. Caller may override via b_nt
    # (0=cached, 2=nt).
    _m = int(M_logical)
    _b_cache_mod = (0 if (_m <= 16 or _m >= 2048) else 2) if b_nt is None else b_nt
    launch = _get_compiled_gemm2_a16w4(
        BM, NE, D_HIDDEN, D_INTER, TILE_N, TILE_K, _b_cache_mod, xcd_swizzle, waves_per_eu, w_dtype
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm2_a16w4_grid(BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out


# =============================================================================
# aiter tuned-CSV config loader for a16w4.
#
# Reads aiter's kimik3_fp4_tuned_fmoe.csv (or any file with the same schema),
# selects the ``flydsl_moe1/2_abf16_wfp4`` rows for a (model_dim, inter_dim,
# experts, topk) shape, and decodes each token's kernelName into a tile-config
# dict consumable by ``flydsl_a16w4_gemm{1,2}``. The CSV is used only as a SOURCE
# OF CANDIDATE tile/waves/xcd geometries -- aiter's gemm bodies differ from ours,
# so the *latency* columns are not comparable, but the tile geometry (tile_m/n/k,
# waves_per_eu, xcd_swizzle, b_nt, k_wave) is informative.
# =============================================================================

# kernelName tokens:  flydsl_moe{stage}_abf16_wfp4_bf16_t{m}x{n}x{k}
#   [_w{N}]=waves_per_eu  [_xcd{N}]=xcd_swizzle  [_bnt{N}]=b_nt  [_kw{N}]=k_wave
#   (no _sk => k_batch=1). Extra epilogue tokens (_reduce/_atomic/_persist/...)
#   are ignored for tile-config purposes.
_A16W4_TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")
_A16W4_W_RE = re.compile(r"_w(\d+)")
_A16W4_XCD_RE = re.compile(r"_xcd(\d+)")
_A16W4_BNT_RE = re.compile(r"_bnt(\d+)")
_A16W4_KW_RE = re.compile(r"_kw(\d+)")


def _decode_a16w4_kname(kname):
    """Decode an ``abf16_wfp4`` kernelName into a tile-config dict, or None."""
    m = _A16W4_TILE_RE.search(kname)
    if m is None:
        return None
    tile_m, tile_n, tile_k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    w = _A16W4_W_RE.search(kname)
    xcd = _A16W4_XCD_RE.search(kname)
    bnt = _A16W4_BNT_RE.search(kname)
    kw = _A16W4_KW_RE.search(kname)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        # b_nt default in aiter's namer is 2 when the token is absent (only _bnt0
        # / _bnt{!=2} are named); mirror that.
        "b_nt": int(bnt.group(1)) if bnt else 2,
        "waves_per_eu": int(w.group(1)) if w else None,
        "xcd_swizzle": int(xcd.group(1)) if xcd else 0,
        "k_wave": int(kw.group(1)) if kw else 1,
        "k_batch": 1,
    }


@functools.cache
def _load_a16w4_csv(csv_path):
    """Parse the tuned CSV into {(model_dim,inter,E,topk,stage,tokens): cfg}."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key_shape = (
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    int(row["expert"]),
                    int(row["topk"]),
                    int(row["token"]),
                )
            except (KeyError, ValueError):
                continue
            for stage, col in ((1, "kernelName1"), (2, "kernelName2")):
                kname = row.get(col, "")
                if "abf16_wfp4" not in kname:
                    continue
                cfg = _decode_a16w4_kname(kname)
                if cfg is not None:
                    table[key_shape + (stage,)] = cfg
    return table


def pick_a16w4_config(csv_path, *, model_dim, inter_dim, experts, topk, tokens, stage):
    """Return aiter's tuned tile-config for one (shape, tokens, stage), or None.

    Picks the exact ``tokens`` row if present, else the nearest tuned token
    (largest tuned token <= requested, or the smallest tuned token otherwise) for
    the shape+stage. ``stage`` is 1 (gemm1) or 2 (gemm2).
    """
    table = _load_a16w4_csv(csv_path)
    exact = table.get((model_dim, inter_dim, experts, topk, tokens, stage))
    if exact is not None:
        return exact
    cand = sorted(
        t for (md, i, e, k, t, s) in table if (md, i, e, k, s) == (model_dim, inter_dim, experts, topk, stage)
    )
    if not cand:
        return None
    le = [t for t in cand if t <= tokens]
    pick = le[-1] if le else cand[0]
    return table[(model_dim, inter_dim, experts, topk, pick, stage)]
