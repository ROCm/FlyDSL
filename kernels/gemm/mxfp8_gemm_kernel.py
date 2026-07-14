# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 (a8w8, per-1x32 E8M0 block-scaled fp8 x fp8) dense GEMM host entry, gfx950.

The device kernel lives in ``kernels/gemm/fp8_gemm_8wave.py`` as the block-scale
mode of the shared HipKittens FP8_8wave pipeline (``compile_mxfp8_gemm_8w``).
This module keeps the dense host wrapper: raw E8M0 -> broadcast-int32 scale
preshuffle fused into the GEMM launch, plus per-shape autotune and workspace
caching. Ported from Primus-Turbo PR AMD-AGI/Primus-Turbo#390.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# Shared fp8 GEMM primitives plus the merged mxfp8 kernel builder (block-scale mode
# of the shared 8-wave pipeline) from fp8_gemm_8wave.py.
from kernels.gemm.fp8_gemm_8wave import compile_mxfp8_gemm_8w as _build_mxfp8_nt_kernel
from kernels.gemm.fp8_gemm_utils import (
    _PRESHUF_KT,
    _robust_time,
    build_preshuffle_ab_kernel,
    ceildiv,
    make_value_attrs,
)

# ── Dense host wrapper ───────────────────────────────────────────────────────

_BLOCK_M = 256
_BLOCK_N = 256
_PRESHUF_BLK = 256  # preshuffle kernel block size (matches build_preshuffle_ab_kernel)

# (K, bm, gm, xcd, gn, cbsz, blgp, out_fp16) -> launch_mxfp8_fused (preshuffle+gemm jit)
_MXFP8_FUSED_CACHE: dict = {}
_MXFP8_AT_CACHE: dict = {}  # (M,N,K,bm,gm,xcd,gn,cbsz,blgp,out_fp16) -> [raw_launch, compiled_or_None]
# (M, N, K128, device, stream) -> (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp). Caller-owned
# scale workspace (turbo-style): the fused stub's preshuffle writes a_sp/b_sp then the
# gemm reads them, in stream order, so reuse across same-shape calls on one stream is safe.
_MXFP8_WS_CACHE: dict = {}

# Per-shape NT autotune candidates (BLOCK_M, GROUP_M, num_xcd); BLOCK_N fixed 256.
# BLOCK_M=128 doubles the tiles (fills the CUs on skinny/small shapes), 256 wins big
# square / B-streaming; GROUP_M is the per-XCD L2-reuse super-block depth.
# BLOCK_M fixed at 256 (n_tiles_a = 256//64 = 4): the A-scale preshuffle layout is
# bm-dependent and the fused stub preshuffles the raw E8M0 scales to int32 (one fixed
# config) before the gemm, so the gemm cannot re-pack per candidate -> BLOCK_M constant.
# autotune sweeps only GROUP_M / num_xcd / group_n (none of which change the layout).
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
]


_MXFP8_AUTOTUNE_CACHE: dict = {}  # (M,N,K,out_dtype,cbsz,blgp) -> (BLOCK_M, GROUP_M, num_xcd, group_n)


def _mx_nt_gn_cands(N):
    """NT 2D N-band candidate widths for the autotune stage-2 sweep. The seed band
    (gn=0, NT's 1D swizzle) is measured separately as the baseline, so the final
    pick can never regress. Only offer a band when there are >= 2*gn 256-col
    N-blocks (else the band can't create the cross-tile B reuse it exists for).
    Winners are shape-dependent (NT: 7B GateUp gn16, 70B QKV gn8/16), so the
    per-shape bench picks rather than a single heuristic. Set env MX_DISABLE_NT_GN
    to force the seed band (NT -> 1D swizzle)."""
    import os

    if os.environ.get("MX_DISABLE_NT_GN"):
        return []
    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    # gn=32 was probed and dropped: its only win (NT 7B_GateUp +1.7% over gn16) is
    # coupled to tile (256,4), but stage-1 picks the tile at the seed band and lands
    # on (256,8) there, so the gn=32 win isn't reliably reachable — not worth the
    # extra autotune compile on every N>=16384 shape. {4,8,16} captures the robust
    # wins. (A fuller tile x gn cross-sweep could reach it but costs far more.)
    return [g for g in (4, 8, 16) if n_blocks >= 2 * g]


def _compile_mxfp8_fused(K, bm, gm, xcd, gn=0, cbsz=0, blgp=0, out_fp16=False):
    """Build the turbo-style fused @flyc.jit ``launch_mxfp8_fused``: ONE host stub
    that enqueues the A+B scale preshuffle kernel and then the NT mxfp8 GEMM kernel
    on the same stream (single Python dispatch, no separate preshuffle launch, no
    CPU sync). The preshuffle repacks raw E8M0 (int32-viewed) into the caller-owned
    a_sp / b_sp workspace; the gemm reads it in stream order. gn = autotuned 2D
    N-band width (0 = 1D swizzle); cbsz/blgp = per-operand fp8 format (0=E4M3,1=E5M2).

    NOTE: never name this jit (or any module global) ``launch`` -- the ``.launch()``
    method attribute lands in co_names and the cache-key dependency walker would
    resolve a module-global ``launch`` back to this JitFunction (infinite recursion).
    """
    K128 = K // 128
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128)
    gemm_kern, BM, BN, wpe = _build_mxfp8_nt_kernel(
        K=K,
        BLOCK_M=bm,
        BLOCK_N=_BLOCK_N,
        GROUP_M=gm,
        group_n=gn,
        num_xcd=xcd,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
    )

    @flyc.jit
    def launch_mxfp8_fused(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
        b_ngrp: fx.Int32,
        stream: fx.Stream,
    ):
        # 1) scale preshuffle (raw E8M0 -> broadcast int32 in a_sp/b_sp)
        pre_kern(a_raw, b_raw, a_sp, b_sp, c_m, c_n, a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
        )
        # 2) NT GEMM (reads the just-written a_sp/b_sp; same stream => ordered)
        grid_x = ceildiv(c_m, BM) * ceildiv(c_n, BN)
        gemm_kern(
            a8,
            b8,
            C,
            a_sp,
            b_sp,
            c_m,
            c_n,
            value_attrs=make_value_attrs(wpe, 0, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_mxfp8_fused


def _get_mxfp8_fused_launch(K, bm, gm, xcd, gn=0, cbsz=0, blgp=0, out_fp16=False):
    fk = (K, bm, gm, xcd, gn, cbsz, blgp, out_fp16)
    launch = _MXFP8_FUSED_CACHE.get(fk)
    if launch is None:
        launch = _compile_mxfp8_fused(K, bm, gm, xcd, gn, cbsz, blgp, out_fp16)
        _MXFP8_FUSED_CACHE[fk] = launch
    return launch


def _get_mx_workspace(M, N, K128, device, stream):
    """Caller-owned scale workspace (a_sp/b_sp) + the preshuffle launch dims, cached
    by (M, N, K128, device, stream). a_sp/b_sp are flat int32 buffers the fused stub's
    preshuffle writes and the gemm reads; sizing mirrors the A (layout-1) / combined-B
    (layout-3) repack (A: cdiv(M,64) groups; B: cdiv(N,256)*4 groups; each group*K128*256
    i32). cdiv for A (not M//64): a partial last 64-row group covers general (non-64) M,
    the preshuffle masks its OOB rows to 0 and the gemm StoreC drops their output."""
    key = (M, N, K128, device, stream)
    e = _MXFP8_WS_CACHE.get(key)
    if e is None:
        a_ngrp = ceildiv(M, 64)
        b_ngrp = ((N + 255) // 256) * 4
        a_blocks = a_ngrp * ceildiv(K128, _PRESHUF_KT)
        a_sp = torch.empty(a_ngrp * K128 * 256, dtype=torch.int32, device=device)
        b_sp = torch.empty(b_ngrp * K128 * 256, dtype=torch.int32, device=device)
        e = (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp)
        _MXFP8_WS_CACHE[key] = e
    return e


def _autotune_mxfp8(
    a8, b8, out_view, a_raw, b_raw, a_sp, b_sp, M, N, K, a_blocks, a_ngrp, b_ngrp, out_dtype, cbsz=0, blgp=0
):
    """First-call micro-bench of the candidates for (M,N,K); cache the fastest cfg by
    shape. Returns (BLOCK_M, GROUP_M, num_xcd, group_n). Each candidate times the FUSED
    stub (preshuffle + gemm): the preshuffle config is fixed (same K128) so it is a
    constant offset across candidates and the gemm-config ranking is preserved.

    Two-stage for NT: stage 1 picks (BM,GM,XCD) at group_n=0 (1D swizzle); stage 2
    fixes that tile and sweeps the 2D N-band width group_n. gn=0 is measured in
    stage 1, so the staged pick can never regress vs the gn-less NT path — it only
    captures the big-/mid-N L2-reuse win on shapes where a band helps."""
    key = (M, N, K, out_dtype, cbsz, blgp)
    _ofp16 = out_dtype == torch.float16
    cached = _MXFP8_AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached
    cands = _MXFP8_NT_CANDIDATES
    stream = torch.cuda.current_stream()

    def _time_cfg(bm, gm, xcd, gn):
        try:
            launch = _get_mxfp8_fused_launch(K, bm, gm, xcd, gn, cbsz=cbsz, blgp=blgp, out_fp16=_ofp16)
            args = (a8, b8, out_view, a_raw, b_raw, a_sp, b_sp, M, N, a_blocks, a_ngrp, b_ngrp, stream)
            launch(*args)
            torch.cuda.synchronize()
            if not torch.isfinite(out_view.reshape(-1)[:1024].float()).all().item():
                return float("inf")
            return _robust_time(launch, args, warmup=2, reps=3, iters=20)
        except Exception:
            return float("inf")

    # Stage 1: best (BLOCK_M, GROUP_M, num_xcd) at the SEED band width gn=0 (NT's
    # native 1D swizzle); stage 2 sweeps the 2D N-band width on top.
    seed_gn = 0
    best_us = float("inf")
    best = None
    for bm, gm, xcd in cands:
        us = _time_cfg(bm, gm, xcd, seed_gn)
        if us < best_us:
            best_us = us
            best = (bm, gm, xcd, seed_gn)
    if best is None:
        raise RuntimeError(
            f"mxfp8 autotune: all candidates failed for M={M} N={N} K={K}. "
            f"Check FlyDSL compilation (MLIR dialect, gfx target, OOM)."
        )
    # Stage 2: sweep 2D N-band width; adopt only if it beats re-measured seed by >1.5%.
    # gn=0 wins by default (bgn=seed_gn); sweep only visits non-seed bands.
    gn_cands = _mx_nt_gn_cands(N)
    if gn_cands:
        bm, gm, xcd, _ = best

        def _robust(gn):
            return min(_time_cfg(bm, gm, xcd, gn) for _ in range(4))

        seed_us = _robust(seed_gn)  # re-measured seed baseline (same estimator as the bands)
        bgn, bus = seed_gn, seed_us
        for gn in sorted(set([0] + gn_cands) - {seed_gn}):
            us = _robust(gn)
            if us < bus and us < seed_us * 0.985:
                bgn, bus = gn, us
        best = (bm, gm, xcd, bgn)
    _MXFP8_AUTOTUNE_CACHE[key] = best
    return best


def gemm_mxfp8_flydsl_kernel(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """MXFP8 (per-1x32 E8M0 block-scaled) dense GEMM, gfx950. Returns C [M,N].

    NT only (trans_a=False, trans_b=True): A [M,K], B [N,K], C = a @ b^T.

    ``a_scale`` / ``b_scale`` are the RAW E8M0 block scales ([M, K//32] / [N, K//32],
    uint8/e8m0): this kernel preshuffles them to the broadcast int32 layout itself,
    fused into the GEMM launch (turbo-style single dispatch -- one @flyc.jit host stub
    enqueues the A+B preshuffle kernel then the gemm kernel on the same stream, with a
    caller-owned int32 workspace, so there is no separate preshuffle launch and no CPU
    sync). The quant emits only raw E8M0 scales.

    Constraints: K % 128 == 0 and K >= 256; M >= 1; N >= 1 (general M/N: the A-scale
    preshuffle / gemm size A by cdiv(M,64) and bound the partial tail by the StoreC clamp).
    """
    assert a.dim() == 2 and b.dim() == 2, "a, b must be 2D"
    assert out_dtype in (torch.bfloat16, torch.float16), "mxfp8 FlyDSL store emits bf16/fp16"
    # Per-operand fp8 format -> MFMA cbsz(srcA)/blgp(srcB): 0=E4M3, 1=E5M2.
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    out_fp16 = out_dtype == torch.float16

    if (not trans_a) and trans_b:
        M, K = a.shape
        N, Kb = b.shape
    else:
        raise NotImplementedError(
            "mxfp8 FlyDSL GEMM is NT only (trans_a=False, trans_b=True); " f"got trans_a={trans_a}, trans_b={trans_b}."
        )
    assert K == Kb, f"K mismatch: a {a.shape}, b {b.shape}"
    assert K % 128 == 0 and K >= 256, f"K must be a multiple of 128 and >= 256, got {K}"
    assert M >= 1, f"M must be >= 1, got {M}"
    assert N >= 1, f"N must be >= 1, got {N}"
    assert a_scale.shape[0] == M and b_scale.shape[0] == N, "scale rows must match a/b rows"
    assert a_scale.shape[1] == K // 32 and b_scale.shape[1] == K // 32, "raw E8M0 scales are [dim, K//32]"

    K128 = K // 128
    # Raw E8M0 byte scales viewed little-endian as flat int32 [dim*K128] (the preshuffle
    # kernel reads grow*K128+gk); contiguous straight out of quant, copy only if a view broke it.
    a_raw = (a_scale if a_scale.is_contiguous() else a_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (b_scale if b_scale.is_contiguous() else b_scale.contiguous()).view(torch.int32).reshape(-1)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    # Keep 2D int8 views (NOT flat 1D): FlyDSL marshals each shape dim as int32, so a
    # 1D [M*K] view overflows when M*K > 2^31. The kernel addresses via i64 SRD re-base
    # (extract_base_index reads only the base ptr), so the 2D shape is just metadata.
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp, b_ngrp = _get_mx_workspace(M, N, K128, a.device, stream)
    # Per-shape cfg: first call benches GROUP_M/num_xcd/group_n (BLOCK_M fixed 256) on
    # the fused stub, caches the winner by (M,N,K).
    bm, gm, xcd, gn = _autotune_mxfp8(
        a8, b8, out, a_raw, b_raw, a_sp, b_sp, M, N, K, a_blocks, a_ngrp, b_ngrp, out_dtype, cbsz, blgp
    )
    launch = _get_mxfp8_fused_launch(K, bm, gm, xcd, gn, cbsz=cbsz, blgp=blgp, out_fp16=out_fp16)
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, M, N, a_blocks, a_ngrp, b_ngrp, stream)
    # [raw, compiled]: raw for CUDA-graph capture (flyc.compile regresses under capture);
    # compiled for eager (skips per-call jit dispatch overhead). Mirrors _GROUPED_AT_CACHE.
    at_key = (M, N, K, bm, gm, xcd, gn, cbsz, blgp, out_fp16)
    entry = _MXFP8_AT_CACHE.get(at_key)
    if entry is None:
        entry = [launch, None]
        _MXFP8_AT_CACHE[at_key] = entry
    raw, compiled = entry
    if torch.cuda.is_current_stream_capturing():
        raw(*args)
    else:
        if compiled is None:
            compiled = flyc.compile(raw, *args)
            entry[1] = compiled
        compiled(*args)
    return out
