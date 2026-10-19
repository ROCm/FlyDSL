# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""SageAttention2 G2 independent quant launches (INT4 QK + gfx120 V-fp8)."""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import get_warp_size

EPS = 1e-7
INT4_QMAX = 7
E4M3_MAX = 448.0

_WAVES_PER_BLOCK = 8
_ALLOWED_D = (64, 128)
_ALLOWED_DTYPES = (torch.float16, torch.bfloat16)


def _require_cuda_4d(name: str, t: torch.Tensor) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dim() != 4:
        raise ValueError(f"{name} expected [B, H, S, D], got shape {tuple(t.shape)}")
    if t.dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"{name} dtype must be fp16 or bf16, got {t.dtype}")


def _require_d(d: int) -> None:
    if d not in _ALLOWED_D:
        raise ValueError(f"D must be 64 or 128, got {d}")


def _require_gfx120(what: str) -> None:
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        raise RuntimeError(f"{what} requires gfx120x, got {arch!r}")


def _maybe_contiguous(t: torch.Tensor) -> torch.Tensor:
    if t.stride(-1) != 1:
        return t.contiguous()
    return t


def _as_stream(stream: torch.cuda.Stream | None) -> torch.cuda.Stream:
    if stream is None:
        return torch.cuda.current_stream()
    return stream


@flyc.kernel
def quant_int4_rows_kernel(
    X: fx.Tensor,
    X_i4: fx.Tensor,
    Scale: fx.Tensor,
    R: fx.Int32,
    D: fx.Constexpr[int],
    WAVES_PER_BLOCK: fx.Constexpr[int],
    WARP: fx.Constexpr[int],
):
    tid = fx.Int32(gpu.thread_id("x"))
    bid = fx.Int32(gpu.block_id("x"))
    lane = tid % fx.Int32(WARP)
    wave = tid // fx.Int32(WARP)
    row = bid * fx.Int32(WAVES_PER_BLOCK) + wave
    n_words = D // 8
    owns = lane < fx.Int32(n_words)
    row_valid = row < fx.Int32(R)
    row_safe = row_valid.select(row, fx.Int32(0))
    col0 = owns.select(lane * fx.Int32(8), fx.Int32(0))

    x2d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(X)),
        fx.make_layout((R, D), (D, 1)),
    )
    i4_2d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(X_i4)),
        fx.make_layout((R, n_words), (n_words, 1)),
    )
    scale1d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(Scale)),
        fx.make_layout((R,), (1,)),
    )

    local_amax = fx.Float32(0.0)
    xs = []
    for i in range_constexpr(8):
        xv = x2d[row_safe, col0 + fx.Int32(i)].to(fx.Float32)
        xs.append(xv)
        local_amax = fx.max(local_amax, fx.absf(xv))
    local_amax = owns.select(local_amax, fx.Float32(0.0))

    amax = local_amax
    width = fx.Int32(WARP)
    for off in range_constexpr(5):
        peer = amax.shuffle_xor(fx.Int32(16 >> off), width)
        amax = fx.max(amax, peer)

    scale = amax / fx.Float32(float(INT4_QMAX)) + fx.Float32(float(EPS))
    half = fx.Float32(0.5)
    word = fx.Int32(0)
    for i in range_constexpr(8):
        y = xs[i] / scale
        q_f = fx.trunc(y + fx.copysign(half, y))
        q = q_f.to(fx.Int32)
        q = fx.max(fx.min(q, fx.Int32(7)), fx.Int32(-8))
        nibble = q & fx.Int32(0xF)
        word = word | (nibble << fx.Int32(4 * i))

    if owns:
        if row_valid:
            i4_2d[row, lane] = word
    if lane == fx.Int32(0):
        if row_valid:
            scale1d[row] = scale


@flyc.jit
def _quant_int4_rows(
    X: fx.Tensor,
    X_i4: fx.Tensor,
    Scale: fx.Tensor,
    R: fx.Int32,
    D: fx.Constexpr[int],
    WAVES_PER_BLOCK: fx.Constexpr[int],
    WARP: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    quant_int4_rows_kernel(X, X_i4, Scale, R, D, WAVES_PER_BLOCK, WARP).launch(
        grid=((R + WAVES_PER_BLOCK - 1) // WAVES_PER_BLOCK,),
        block=(WAVES_PER_BLOCK * WARP,),
        stream=stream,
    )


def _launch_int4_rows(x2d: torch.Tensor, stream: torch.cuda.Stream) -> tuple[torch.Tensor, torch.Tensor]:
    r, d = int(x2d.shape[0]), int(x2d.shape[1])
    warp = get_warp_size()
    x_i4 = torch.empty(r, d // 8, dtype=torch.int32, device=x2d.device)
    scale = torch.empty(r, dtype=torch.float32, device=x2d.device)
    _quant_int4_rows(
        x2d,
        x_i4,
        scale,
        r,
        d,
        _WAVES_PER_BLOCK,
        warp,
        stream=fx.Stream(stream),
    )
    return x_i4, scale


def _quant_qk_int4(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    km: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K → packed INT4 + per-token f32 scales.

    q:  [B, Hq,  Sq, D]  fp16 or bf16
    k:  [B, Hkv, Sk, D]  same dtype as q
    km: None, or broadcastable to [B, Hkv, 1, D] (fp32 or q.dtype)

    Returns:
      Q_i4    [B, Hq,  Sq, D/8] int32
      K_i4    [B, Hkv, Sk, D/8] int32
      q_scale [B, Hq,  Sq]      float32
      k_scale [B, Hkv, Sk]      float32
    """
    _require_gfx120("_quant_qk_int4")
    _require_cuda_4d("q", q)
    _require_cuda_4d("k", k)
    if q.dtype != k.dtype:
        raise ValueError("q, k must share dtype")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)} vs k {tuple(k.shape)}")
    d = int(q.shape[-1])
    _require_d(d)
    stream = _as_stream(stream)
    with torch.cuda.stream(stream):
        q = _maybe_contiguous(q)
        k = _maybe_contiguous(k)
        k_in = k
        if km is not None:
            k_in = (k.float() - km.float().to(device=k.device)).to(k.dtype)
        bq, hq, sq, _ = q.shape
        bk, hkv, sk, _ = k_in.shape
        q2 = q.reshape(bq * hq * sq, d)
        k2 = k_in.reshape(bk * hkv * sk, d)
        q_i4_2, q_scale_1 = _launch_int4_rows(q2, stream)
        k_i4_2, k_scale_1 = _launch_int4_rows(k2, stream)
        q_i4 = q_i4_2.view(bq, hq, sq, d // 8)
        k_i4 = k_i4_2.view(bk, hkv, sk, d // 8)
        q_scale = q_scale_1.view(bq, hq, sq)
        k_scale = k_scale_1.view(bk, hkv, sk)
        return q_i4, k_i4, q_scale, k_scale


@flyc.kernel
def quant_v_fp8_kernel(
    V: fx.Tensor,
    V_e4m3: fx.Tensor,
    Scale: fx.Tensor,
    R: fx.Int32,
    Sk: fx.Int32,
    D: fx.Constexpr[int],
    WARP: fx.Constexpr[int],
):
    lane = fx.Int32(gpu.thread_id("x"))
    bh = fx.Int32(gpu.block_id("x"))
    n0 = D // 16
    cols_per_lane = D // WARP
    k0 = Sk // fx.Int32(16)
    nbytes = fx.Int32(n0) * k0 * fx.Int32(2 * 16 * 8)

    v3d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(V)),
        fx.make_layout((R, Sk, D), (Sk * D, D, 1)),
    )
    scale2d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(Scale)),
        fx.make_layout((R, D), (D, 1)),
    )
    out2d = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(V_e4m3)),
        fx.make_layout((R, nbytes), (nbytes, 1)),
    )

    bh_i = bh
    sk_i = Sk
    for c in range_constexpr(cols_per_lane):
        d = fx.Int32(c) * fx.Int32(WARP) + lane
        amax = fx.Float32(0.0)
        for s in range(fx.Int32(0), sk_i, fx.Int32(1)):
            val = v3d[bh_i, s, d].to(fx.Float32)
            amax = fx.max(amax, fx.absf(val))
        sc = amax / fx.Float32(float(E4M3_MAX)) + fx.Float32(float(EPS))
        scale2d[bh_i, d] = sc

    gpu.barrier()

    ngroups = fx.Int32(n0) * k0 * fx.Int32(2) * fx.Int32(16)
    n_iter = (ngroups + fx.Int32(WARP) - fx.Int32(1)) // fx.Int32(WARP)
    for it in range(fx.Int32(0), n_iter, fx.Int32(1)):
        gidx = it * fx.Int32(WARP) + lane
        in_range = gidx < ngroups
        g_safe = in_range.select(gidx, fx.Int32(0))
        v = g_safe % fx.Int32(16)
        g = (g_safe // fx.Int32(16)) % fx.Int32(2)
        kb = (g_safe // fx.Int32(32)) % k0
        db = g_safe // (fx.Int32(32) * k0)
        s0 = kb * fx.Int32(16) + g * fx.Int32(8)
        dcol = db * fx.Int32(16) + v
        sc = scale2d[bh_i, dcol]
        ys = []
        for val_i in range_constexpr(8):
            ys.append(v3d[bh_i, s0 + fx.Int32(val_i), dcol].to(fx.Float32) / sc)
        p0 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=ys[0], src_b=ys[1], old=fx.Int32(0), word_sel=False))
        p0 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=ys[2], src_b=ys[3], old=p0, word_sel=True))
        p1 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=ys[4], src_b=ys[5], old=fx.Int32(0), word_sel=False))
        p1 = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(res=T.i32, src_a=ys[6], src_b=ys[7], old=p1, word_sel=True))
        base = (((db * k0 + kb) * fx.Int32(2) + g) * fx.Int32(16) + v) * fx.Int32(8)
        if in_range:
            for bi in range_constexpr(4):
                out2d[bh_i, base + fx.Int32(bi)] = ((p0 >> fx.Int32(8 * bi)) & fx.Int32(0xFF)).to(fx.Uint8)
            for bi in range_constexpr(4):
                out2d[bh_i, base + fx.Int32(4 + bi)] = ((p1 >> fx.Int32(8 * bi)) & fx.Int32(0xFF)).to(fx.Uint8)


@flyc.jit
def _launch_quant_v_fp8(
    V: fx.Tensor,
    V_e4m3: fx.Tensor,
    Scale: fx.Tensor,
    R: fx.Int32,
    Sk: fx.Int32,
    D: fx.Constexpr[int],
    WARP: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    quant_v_fp8_kernel(V, V_e4m3, Scale, R, Sk, D, WARP).launch(
        grid=(R,),
        block=(WARP,),
        stream=stream,
    )


def _quant_v_fp8(
    v: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V → G-I preshuffled e4m3fn + per-channel f32 scale. gfx120* only.

    v: [B, Hkv, Sk, D] fp16 or bf16
       bf16 is cast to fp16 first (G0 §2 / §4.4).

    Returns:
      V_e4m3  [B, Hkv, D/16, Sk/16, 2, 16, 8] uint8
      v_scale [B, Hkv, D] float32
    """
    _require_gfx120("_quant_v_fp8")
    _require_cuda_4d("v", v)
    d = int(v.shape[-1])
    _require_d(d)
    sk = int(v.shape[2])
    if sk % 16 != 0:
        raise ValueError(f"Sk must be a multiple of 16, got {sk}")
    stream = _as_stream(stream)
    with torch.cuda.stream(stream):
        v = _maybe_contiguous(v)
        if v.dtype == torch.bfloat16:
            v = v.to(torch.float16)
        b, h, sk, d = (int(x) for x in v.shape)
        r = b * h
        n0, k0 = d // 16, sk // 16
        v3 = v.reshape(r, sk, d)
        v_e4 = torch.empty(r, n0 * k0 * 2 * 16 * 8, dtype=torch.uint8, device=v.device)
        v_scale = torch.empty(r, d, dtype=torch.float32, device=v.device)
        warp = get_warp_size()
        _launch_quant_v_fp8(v3, v_e4, v_scale, r, sk, d, warp, stream=fx.Stream(stream))
        return v_e4.view(b, h, n0, k0, 2, 16, 8), v_scale.view(b, h, d)
