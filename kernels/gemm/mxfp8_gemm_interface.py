# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 (a8w8, per-1x32 E8M0 block-scaled fp8 x fp8) dense GEMM host entry, gfx950."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.tensor_shim import _run_compiled

# Shared fp8 GEMM primitives plus the merged mxfp8 kernel builder (block-scale mode
# of the shared 8-wave pipeline) from fp8_gemm_8wave.py.
from kernels.gemm.fp8_gemm_8wave import compile_mxfp8_gemm_8w as _build_mxfp8_nt_kernel
from kernels.gemm.fp8_gemm_utils import (
    _PRESHUF_KT,
    build_preshuffle_ab_kernel,
    ceildiv,
    make_value_attrs,
)

# ── Dense host wrapper ───────────────────────────────────────────────────────


def _robust_time(launch, args, warmup=250, reps=5, iters=50):
    """Median-of-`reps` timing of launch(*args) after `warmup` iters.
    The long warmup reaches boost clock; short-K kernels mis-pick configs otherwise."""
    for _ in range(warmup):
        launch(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            launch(*args)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    ts.sort()
    return ts[len(ts) // 2]


_BLOCK_M = 256
_BLOCK_N = 256
_PRESHUF_BLK = 256  # preshuffle kernel block size (matches build_preshuffle_ab_kernel)

# (K, bm, gm, xcd, gn, cbsz, blgp, out_fp16) -> launch_mxfp8_fused (preshuffle+gemm jit)
_MXFP8_FUSED_CACHE: dict = {}
# (M, N, K128, device, stream) -> (a_sp, b_sp, a_blocks, a_ngrp, b_ngrp). Caller-owned
# scale workspace (turbo-style): the fused stub's preshuffle writes a_sp/b_sp then the
# gemm reads them, in stream order, so reuse across same-shape calls on one stream is safe.
_MXFP8_WS_CACHE: dict = {}

# Per-shape NT autotune candidates (BLOCK_M, GROUP_M, num_xcd);
_MXFP8_NT_CANDIDATES = [
    (256, 4, 8),
    (256, 8, 8),
]


_MXFP8_AUTOTUNE_CACHE: dict = {}  # (M,N,K,out_dtype,cbsz,blgp) -> (BLOCK_M, GROUP_M, num_xcd, group_n)


def _mxfp8_nt_gn_cands(N):
    import os

    if os.environ.get("MX_DISABLE_NT_GN"):
        return []
    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    return [g for g in (4, 8, 16) if n_blocks >= 2 * g]


def _compile_mxfp8_fused(K, bm, gm, xcd, gn=0, cbsz=0, blgp=0, out_fp16=False):
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


def _get_mxfp8_workspace(M, N, K128, device, stream):
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
    gn_cands = _mxfp8_nt_gn_cands(N)
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


def gemm_mxfp8(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """MXFP8 (per-1x32 E8M0 block-scaled) dense GEMM, gfx950. Returns C [M,N]."""
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
    a_raw = (a_scale if a_scale.is_contiguous() else a_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (b_scale if b_scale.is_contiguous() else b_scale.contiguous()).view(torch.int32).reshape(-1)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)
    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp, b_ngrp = _get_mxfp8_workspace(M, N, K128, a.device, stream)
    bm, gm, xcd, gn = _autotune_mxfp8(
        a8, b8, out, a_raw, b_raw, a_sp, b_sp, M, N, K, a_blocks, a_ngrp, b_ngrp, out_dtype, cbsz, blgp
    )
    launch = _get_mxfp8_fused_launch(K, bm, gm, xcd, gn, cbsz=cbsz, blgp=blgp, out_fp16=out_fp16)
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, M, N, a_blocks, a_ngrp, b_ngrp, stream)
    if torch.cuda.is_current_stream_capturing():
        launch(*args)
    else:
        _run_compiled(launch, *args)
    return out
