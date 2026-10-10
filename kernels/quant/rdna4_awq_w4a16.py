# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X AWQ W4A16 dequant + fused GEMV (packed uint4, in-reg dequant).

Wire format (``TensorCoreAWQW4A16Layout`` / eager ``gemv_awq_w4a16``):

* ``qweight[N, K//2]`` int8 — **unsigned** nibble pack: low = even col, high = odd
  (``(u0 & 0xF) | ((u1 & 0xF) << 4)``); indices in ``[0, 15]``.
* ``wscales[K//G, N]`` bf16/fp16 — per-group scales (G default 64).
* ``wzeros[K//G, N]`` same dtype — per-group fp zero points.

Dequant (per element)::

    W[n, k] = (qweight_u4[n, k] - 8) * wscales[k // G, n] + wzeros[k // G, n]

Hot reference path is ``gemv_awq_w4a16`` (modulation linears, small batch).
This module ships:

* FlyDSL ``dequant_awq_w4a16_weight`` — device unpack → bf16/fp16 W
* FlyDSL ``build_awq_gemv_module`` / fused ``gemv_awq_w4a16`` — keeps uint4
  packed; applies ``(q-8)*s+z`` in-register while accumulating ``x @ W.T``.
  Tiled path: one block per ``n_tile`` output cols; stage X in LDS; decode
  **8 weights / i32** with one scale+zero (HIP contract); register
  block over ``max_m`` in ``{1,4}`` for gemv shapes ``[1|4, N, K]``


"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, gpu, range_constexpr
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_DEFAULT_GROUP = 64
WARP = 32
KERNEL_NAME = "awq_dequant_w4a16_gfx120x"
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)
_LDS_BUDGET = 65536


def _kernel_signature(**params: object) -> str:
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


def _block_threads(k: int) -> int:
    packed = k // 2
    if packed <= 64:
        return 64
    if packed <= 128:
        return 128
    if packed <= 256:
        return 256
    return 256


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


def _stream_kw(stream: torch.cuda.Stream | None) -> dict:
    return {} if stream is None else {"stream": stream}


# Measured gfx1201 tile+BT pick (2026-09-30).
# M=1: n_tile=1/4 as before. M in (1,4]: n_tile=1 + block_threads=64 (mid_m4 faster).
def pick_awq_gemv_tiles(m: int, n: int, k: int = 0) -> tuple[int, int, int]:
    """Return ``(max_m, n_tile, block_threads)`` for fused AWQ GEMV (measured).

    gfx1201 host-API measured (2026-09-30, mid_m4):
    * M=1, N<128 → n_tile=1, default BT (faster than HIP)
    * M=1, 128≤N<384 → n_tile=4 (faster mid_m1)
    * M=1, N≥384 → n_tile=1 (faster large)
    * M∈(1,4] → n_tile=1 + **block_threads=64** (mid_m4 ~×1.48; prior
      n_tile=8 + BT=256 was ~×0.74 — oversubscribed waves)
    ``k`` reserved for future K-aware gates; unused when BT forced.
    """
    if m <= 1:
        if 128 <= n < 384:
            return 1, 4, _block_threads(k) if k > 0 else 256
        bt = _block_threads(k) if k > 0 else 256
        return 1, 1, bt
    # M in (1, 4]: skinny BT + n_tile=1 beats fat N-tile on measured mid shapes.
    return 4, 1, 64


_ZERO_BIAS_CACHE: dict = {}


def _zero_bias_buf(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Reuse a device zero vector so bias=None does not allocate every call."""
    key = (int(n), str(device), dtype)
    buf = _ZERO_BIAS_CACHE.get(key)
    if buf is None or buf.device != device or buf.dtype != dtype or buf.numel() != n:
        import torch

        buf = torch.zeros((n,), device=device, dtype=dtype)
        _ZERO_BIAS_CACHE[key] = buf
    return buf


def unpack_uint4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """(..., K//2) int8 → (..., K) int8 in [0, 15].

    Offline/oracle helper (gemv path unpacks in-register). Delegates to the
    shared codec so CUDA callers share one implementation.
    """
    from kernels.quant.rdna4_int4_codec import unpack_uint4_row_major as _codec_unpack

    return _codec_unpack(packed)


@lru_cache(maxsize=64)
def build_awq_dequant_w4a16_module(
    k: int,
    group_size: int = _DEFAULT_GROUP,
    out_dtype: str = "bfloat16",
    block_threads: Optional[int] = None,
) -> Callable[..., None]:
    """Unpack AWQ uint4 + apply group scales/zeros → bf16/fp16 weight row."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 8 or group_size % 2 != 0:
        raise ValueError(f"group_size={group_size} must be even and >=8")
    if out_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"out_dtype must be bfloat16|float16, got {out_dtype}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")

    packed_k = k // 2
    groups = (k + group_size - 1) // group_size
    pack_steps = (packed_k + block_threads - 1) // block_threads
    OutTy = fx.BFloat16 if out_dtype == "bfloat16" else fx.Float16
    out_bytes = 2
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=out_dtype,
        op="awq_dequant",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Tensor,
        Scales: fx.Tensor,  # (groups, N) ActTy API layout
        Zeros: fx.Tensor,  # (groups, N) ActTy API layout
        Out: fx.Tensor,
        n_rows: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        active_row = bid < n_rows

        q_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            Scales,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(n_rows) * fx.Int64(out_bytes),
        )
        z_buf = ptr_buf_tensor(
            Zeros,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(n_rows) * fx.Int64(out_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(k) * fx.Int64(out_bytes),
        )

        for step in range_constexpr(pack_steps):
            pc = tid + fx.Int32(step * block_threads)
            inb = active_row & (pc < fx.Int32(packed_k))
            gidx = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(pc)
            safe_g = inb.select(gidx, fx.Int64(0))
            raw = buf_copy_load(q_buf, safe_g, elem=fx.Int8, unit_elems=1)
            packed = fx.Int32(raw) & fx.Int32(0xFF)
            lo = packed & fx.Int32(0xF)
            hi = (packed >> fx.Int32(4)) & fx.Int32(0xF)

            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            # API layout (groups, N): s[g, n] = g * N + n
            sidx = fx.Int64(g) * fx.Int64(n_rows) + fx.Int64(bid)
            safe_s = inb.select(sidx, fx.Int64(0))
            s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=OutTy, unit_elems=1))
            z = fx.Float32(buf_copy_load(z_buf, safe_s, elem=OutTy, unit_elems=1))

            # W = (u4 - 8) * scale + zero
            v0 = (fx.Float32(lo) - fx.Float32(8.0)) * s + z
            v1 = (fx.Float32(hi) - fx.Float32(8.0)) * s + z

            o0 = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            # Match rdna4_asym_w4a8: inactive lanes skip the store (index-0
            # redirect is a real in-bounds write and races Out[0]).
            if inb:
                buf_copy_store(o_buf, o0, OutTy(v0), elem=OutTy, unit_elems=1)
                buf_copy_store(o_buf, o1, OutTy(v1), elem=OutTy, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Tensor,
        Scales: fx.Tensor,
        Zeros: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        dequant_kernel(Qdata, Scales, Zeros, Out, n_rows).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@contextmanager
def _validate_dequant(sig_args):
    import torch

    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def awq_dequant_direct(
    Qdata: fx.Tensor,
    Scales: fx.Tensor,
    Zeros: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    K: fx.Constexpr[int],
    group_size: fx.Constexpr[int],
    out_dtype: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_awq_dequant_w4a16_module(K, group_size, out_dtype, block_threads=BLOCK_THREADS)
    launch(Qdata, Scales, Zeros, Out, n_rows, stream)


_awq_dequant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["K", "group_size", "out_dtype", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_awq_dequant",
    validate_hook=_validate_dequant,
)(awq_dequant_direct)


def dequant_awq_w4a16_weight(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor | None,
    group_size: int = _DEFAULT_GROUP,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """FlyDSL AWQ W4A16 weight dequant → bf16/fp16 ``(N, K)``.

    ``wscales`` / ``wzeros`` keep layout ``(K//G, N)`` and are indexed in the kernel.
    """
    require_gfx120x(qweight.device, what="dequant_awq_w4a16_weight (gfx120x)")

    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda":
        raise ValueError("qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    k = k_half * 2
    groups = (k + group_size - 1) // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(
            f"wscales must be {(groups, n)} (ceil K to group_size={group_size}), got {tuple(wscales.shape)}"
        )
    if tuple(wzeros.shape) != (groups, n):
        raise ValueError(f"wzeros must be {(groups, n)}, got {tuple(wzeros.shape)}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")
    if wzeros.dtype != wscales.dtype:
        wzeros = wzeros.to(wscales.dtype)

    out_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    from kernels.common.gfx120x_pad import ensure_contiguous

    scales_gn = ensure_contiguous(wscales, stream=stream)
    zeros_gn = ensure_contiguous(wzeros, stream=stream)
    qw = ensure_contiguous(qweight, stream=stream)
    out = torch.empty((n, k), device=qweight.device, dtype=wscales.dtype)
    _awq_dequant_tuned(
        qw,
        scales_gn,
        zeros_gn,
        out,
        n,
        K=k,
        group_size=group_size,
        out_dtype=out_dtype,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    return out


@lru_cache(maxsize=64)
def build_awq_gemv_module(
    k: int,
    group_size: int = _DEFAULT_GROUP,
    act_dtype: str = "bfloat16",
    max_m: int = 1,
    n_tile: int = 1,
    block_threads: Optional[int] = None,
) -> Callable[..., None]:
    """Fused AWQ W4A16 GEMV — i32×8 decode, optional LDS-X + N-tile.

    Grid ``ceil(N / n_tile)``. HIP contract: decode eight weights per
    i32 chunk, one scale/zero (``group_size % 8 == 0``). Scales/zeros are
    ``(groups, N)`` ActTy (API layout). When ``n_tile > 1``, stage
    ``X[0:max_m, :]`` in LDS and reuse across the N-tile (cuts X traffic).
    When ``n_tile == 1``, read X from GMEM (LDS fill would be pure overhead).
    """
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size={group_size} must be a positive multiple of 8, got {group_size}")
    if act_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"act_dtype must be bfloat16|float16, got {act_dtype}")
    if max_m not in (1, 4):
        raise ValueError(f"max_m must be 1 or 4, got {max_m}")
    if n_tile not in (1, 4, 8):
        raise ValueError(f"n_tile must be 1, 4, or 8, got {n_tile}")
    packed_k = k // 2

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")

    groups = (k + group_size - 1) // group_size
    # Eight nibbles per i32. K % 8 == 0 keeps that load. A shorter tail is bytes.
    vec_k = packed_k // 4 if k % 8 == 0 else (packed_k + 3) // 4
    vec_steps = (vec_k + block_threads - 1) // block_threads
    ActTy = fx.BFloat16 if act_dtype == "bfloat16" else fx.Float16
    act_bytes = 2
    reduction_slots = block_threads // WARP
    n_acc = max_m * n_tile
    red_elems = max(reduction_slots * n_acc, n_acc)
    # X tile is max_m*K bf16. Past the LDS budget, read X from GMEM (same math).
    lds_bytes = max_m * k * act_bytes + red_elems * 4
    use_lds = n_tile > 1 and lds_bytes <= _LDS_BUDGET
    x_fill_steps = (max_m * k + block_threads - 1) // block_threads if use_lds else 0
    RedTy = fx.Array[fx.Float32, red_elems, 16]
    # Always declare X LDS (size 1 if unused) so struct shape is valid.
    x_lds_elems = max_m * k if use_lds else 1
    XLdsTy = fx.Array[ActTy, x_lds_elems, 16]
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=act_dtype,
        max_m=max_m,
        n_tile=n_tile,
        op="awq_gemv_tile_gn_act",  # (G,N) ActTy scales — measured host-prep 2026-09-30
    )

    @fx.struct
    class SharedStorage:
        x_tile: XLdsTy
        reduction_buffer: RedTy

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def gemv_kernel(
        X: fx.Tensor,
        Qdata: fx.Tensor,
        Scales: fx.Tensor,
        Zeros: fx.Tensor,
        Bias: fx.Tensor,
        Out: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        has_bias: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        n0 = bid * fx.Int32(n_tile)
        c0 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        x_lds = lds.x_tile.view(fx.make_layout(x_lds_elems, 1))
        red = lds.reduction_buffer.view(fx.make_layout(red_elems, 1))

        def wave_reduce_sum(w0: object) -> fx.Float32:
            w = w0
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = w + gpu.shuffle_xor(w, off, WARP)
            return w

        x_buf = ptr_buf_tensor(
            X,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(k) * fx.Int64(act_bytes),
        )
        q_i32_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(packed_k),
        )
        if const_expr(k % 8 != 0):
            # Row stride is packed_k bytes, not a multiple of 4, so an i32 index
            # would walk into the next row.
            q_byte_buf = ptr_buf_tensor(
                Qdata,
                elem=fx.Int8,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(N) * fx.Int64(packed_k),
            )
        # Scales/zeros: API layout (groups, N) in ActTy — host skips transpose+.float().
        s_buf = ptr_buf_tensor(
            Scales,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(N) * fx.Int64(act_bytes),
        )
        z_buf = ptr_buf_tensor(
            Zeros,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(N) * fx.Int64(act_bytes),
        )
        b_buf = ptr_buf_tensor(
            Bias,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(act_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(N) * fx.Int64(act_bytes),
        )

        # Optional LDS stage of X (only when n_tile>1 so X is reused).
        if use_lds:
            for step in range_constexpr(x_fill_steps):
                idx = tid + fx.Int32(step * block_threads)
                in_x = idx < fx.Int32(max_m * k)
                mi = idx // fx.Int32(k)
                ki = idx - mi * fx.Int32(k)
                row_ok = mi < M
                gidx = fx.Int64(mi) * fx.Int64(k) + fx.Int64(ki)
                safe_g = (in_x & row_ok).select(gidx, fx.Int64(0))
                xv = fx.Float32(buf_copy_load(x_buf, safe_g, elem=ActTy, unit_elems=1))
                store_v = (in_x & row_ok).select(ActTy(xv), ActTy(c0))
                if in_x:
                    fx.memref_store(store_v, x_lds, idx)
            gpu.barrier()

        partials = [c0 for _ in range(n_acc)]

        for step in range_constexpr(vec_steps):
            vi = tid + fx.Int32(step * block_threads)
            in_k = vi < fx.Int32(vec_k)
            col0 = vi * fx.Int32(8)
            g = col0 // fx.Int32(group_size)

            # Gather 8 X values per m-row (LDS or GMEM).
            x_m = []
            for mi in range_constexpr(max_m):
                xs = []
                for t in range_constexpr(8):
                    if const_expr(k % 8 != 0):
                        col_ok = (col0 + fx.Int32(t)) < fx.Int32(k)
                        if use_lds:
                            base = fx.Int32(mi * k) + col0
                            xi = (in_k & col_ok).select(base + fx.Int32(t), fx.Int32(0))
                            xv = fx.Float32(x_lds[xi])
                            xs.append((in_k & col_ok).select(xv, c0))
                        else:
                            gidx = fx.Int64(mi) * fx.Int64(k) + fx.Int64(col0 + fx.Int32(t))
                            m_ok = fx.Int32(mi) < M
                            take = in_k & m_ok & col_ok
                            safe = take.select(gidx, fx.Int64(0))
                            loaded = fx.Float32(buf_copy_load(x_buf, safe, elem=ActTy, unit_elems=1))
                            xs.append(take.select(loaded, c0))
                    elif use_lds:
                        base = fx.Int32(mi * k) + col0
                        xi = in_k.select(base + fx.Int32(t), fx.Int32(0))
                        xs.append(fx.Float32(x_lds[xi]))
                    else:
                        gidx = fx.Int64(mi) * fx.Int64(k) + fx.Int64(col0 + fx.Int32(t))
                        m_ok = fx.Int32(mi) < M
                        safe = (in_k & m_ok).select(gidx, fx.Int64(0))
                        xs.append(fx.Float32(buf_copy_load(x_buf, safe, elem=ActTy, unit_elems=1)))
                x_m.append(xs)

            for nt in range_constexpr(n_tile):
                n = n0 + fx.Int32(nt)
                active = in_k & (n < N)
                if const_expr(k % 8 != 0):
                    word = fx.Int32(0)
                    byte0 = vi * fx.Int32(4)
                    for b in range_constexpr(4):
                        bp = byte0 + fx.Int32(b)
                        one = fx.Int32(0)
                        if active & (bp < fx.Int32(packed_k)):
                            bidx = fx.Int64(n) * fx.Int64(packed_k) + fx.Int64(bp)
                            one = fx.Int32(buf_copy_load(q_byte_buf, bidx, elem=fx.Int8, unit_elems=1)) & fx.Int32(255)
                        word = word | (one << fx.Int32(8 * b))
                    raw = word
                else:
                    qidx = fx.Int64(n) * fx.Int64(vec_k) + fx.Int64(vi)
                    safe_q = active.select(qidx, fx.Int64(0))
                    raw = fx.Int32(buf_copy_load(q_i32_buf, safe_q, elem=fx.Int32, unit_elems=1))
                sidx = fx.Int64(g) * fx.Int64(N) + fx.Int64(n)
                safe_s = active.select(sidx, fx.Int64(0))
                s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=ActTy, unit_elems=1))
                z = fx.Float32(buf_copy_load(z_buf, safe_s, elem=ActTy, unit_elems=1))
                for mi in range_constexpr(max_m):
                    m_ok = fx.Int32(mi) < M
                    acc_i = mi * n_tile + nt
                    dot = c0
                    for t in range_constexpr(8):
                        nib = (raw >> fx.Int32(4 * t)) & fx.Int32(0xF)
                        w = (fx.Float32(nib) - fx.Float32(8.0)) * s + z
                        dot = dot + x_m[mi][t] * w
                    contrib = (active & m_ok).select(dot, c0)
                    partials[acc_i] = partials[acc_i] + contrib

        # Batched multi-acc reduce (2 barriers total).
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        wave_sums = []
        for acc_i in range_constexpr(n_acc):
            wave_sums.append(wave_reduce_sum(partials[acc_i]))
        if lane == 0:
            for acc_i in range_constexpr(n_acc):
                fx.memref_store(wave_sums[acc_i], red, wave * fx.Int32(n_acc) + fx.Int32(acc_i))
        gpu.barrier()
        if wave == 0:
            for acc_i in range_constexpr(n_acc):
                lane_ok = lane < fx.Int32(reduction_slots)
                lane_safe = lane_ok.select(lane, fx.Int32(0))
                vv = red[lane_safe * fx.Int32(n_acc) + fx.Int32(acc_i)]
                ww = lane_ok.select(vv, c0)
                ww = wave_reduce_sum(ww)
                if lane == 0:
                    fx.memref_store(ww, red, fx.Int32(acc_i))
        gpu.barrier()
        if tid == 0:
            for mi in range_constexpr(max_m):
                for nt in range_constexpr(n_tile):
                    acc_i = mi * n_tile + nt
                    acc = fx.Float32(red[fx.Int32(acc_i)])
                    n = n0 + fx.Int32(nt)
                    row_ok = fx.Int32(mi) < M
                    col_ok = n < N
                    if has_bias != 0:
                        b = fx.Float32(
                            buf_copy_load(
                                b_buf,
                                (row_ok & col_ok).select(fx.Int64(n), fx.Int64(0)),
                                elem=ActTy,
                                unit_elems=1,
                            )
                        )
                        acc = acc + (row_ok & col_ok).select(b, c0)
                    oidx = fx.Int64(mi) * fx.Int64(N) + fx.Int64(n)
                    safe_o = (row_ok & col_ok).select(oidx, fx.Int64(0))
                    if row_ok & col_ok:
                        buf_copy_store(o_buf, safe_o, ActTy(acc), elem=ActTy, unit_elems=1)

    gemv_kernel.__name__ = f"{KERNEL_NAME}_gemv_{sig}"

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Qdata: fx.Tensor,
        Scales: fx.Tensor,
        Zeros: fx.Tensor,
        Bias: fx.Tensor,
        Out: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        has_bias: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_n = (N + n_tile - 1) // n_tile
        gemv_kernel(X, Qdata, Scales, Zeros, Bias, Out, M, N, has_bias).launch(
            grid=(grid_n, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_gemv_{sig}"
    return launch


@contextmanager
def _validate_gemv(sig_args):
    import torch

    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def awq_gemv_direct(
    X: fx.Tensor,
    Qdata: fx.Tensor,
    Scales: fx.Tensor,
    Zeros: fx.Tensor,
    Bias: fx.Tensor,
    Out: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    has_bias: fx.Int32,
    K: fx.Constexpr[int],
    group_size: fx.Constexpr[int],
    act_dtype: fx.Constexpr[str],
    max_m: fx.Constexpr[int],
    n_tile: fx.Constexpr[int],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_awq_gemv_module(K, group_size, act_dtype, max_m, n_tile, block_threads=BLOCK_THREADS)
    launch(X, Qdata, Scales, Zeros, Bias, Out, M, N, has_bias, stream)


_awq_gemv_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["K", "group_size", "act_dtype", "max_m", "n_tile", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_awq_gemv",
    validate_hook=_validate_gemv,
)(awq_gemv_direct)


def gemv_awq_w4a16(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    group_size: int = _DEFAULT_GROUP,
    *,
    stream: torch.cuda.Stream | None = None,
    force_n_tile: Optional[int] = None,
) -> torch.Tensor:
    """Host-API ``gemv_awq_w4a16``: fused FlyDSL GEMV (packed W, in-reg dequant).

    Layout matches reference: ``qweight[N,K//2]`` uint4, ``wscales/wzeros[K/G,N]``.
    Kernel consumes scales/zeros in the same ``(G,N)`` ActTy layout (no host
    transpose / ``.float()`` — measured 2026-09-30; host prep was ~14 µs).
    Tile pick: :func:`pick_awq_gemv_tiles` (measured). ``force_n_tile`` overrides
    the N-tile for experiments. M>4 host-chunks in steps of 4.
    """
    require_gfx120x(x.device, what="gemv_awq_w4a16 (gfx120x)")

    if x.dim() < 1:
        raise ValueError("x must be at least 1D")
    orig_shape = x.shape
    from kernels.common.gfx120x_pad import ensure_contiguous

    x2d = ensure_contiguous(x.reshape(-1, orig_shape[-1]), stream=stream)
    m, k = x2d.shape
    if m == 0:
        return torch.empty(*orig_shape[:-1], qweight.shape[0], device=x.device, dtype=x.dtype)
    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda" or x2d.device.type != "cuda":
        raise ValueError("x/qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    if k_half * 2 != k:
        raise ValueError(f"qweight K//2={k_half} inconsistent with x K={k}")
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size must be a positive multiple of 8, got {group_size}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")
    if x2d.dtype != wscales.dtype:
        x2d = x2d.to(wscales.dtype)
    if wzeros.dtype != wscales.dtype:
        wzeros = wzeros.to(wscales.dtype)
    # A partial last group stays in the kernel. Scales are one column per started group.
    if wscales.dim() != 2 or int(wscales.shape[1]) != n:
        raise ValueError(f"wscales must be (K//G, N), got {tuple(wscales.shape)}")
    if tuple(wzeros.shape) != tuple(wscales.shape):
        raise ValueError(f"wzeros must match wscales {tuple(wscales.shape)}, got {tuple(wzeros.shape)}")
    groups = (int(k) + group_size - 1) // group_size
    groups_in = int(wscales.shape[0])
    if groups_in != groups:
        raise ValueError(
            f"wscales must be {(groups, n)} (ceil K to group_size={group_size}), got {tuple(wscales.shape)}"
        )
    qw = ensure_contiguous(qweight, stream=stream)
    scales_gn = ensure_contiguous(wscales, stream=stream)
    zeros_gn = ensure_contiguous(wzeros, stream=stream)

    act_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    # Pass API layout through; kernel indexes g*N+n in ActTy.
    out = torch.empty((m, n), device=x2d.device, dtype=wscales.dtype)
    if bias is None:
        bias_buf = _zero_bias_buf(n, x2d.device, wscales.dtype)
        has_bias = 0
    else:
        if tuple(bias.shape) != (n,):
            raise ValueError(f"bias must be {(n,)}, got {tuple(bias.shape)}")
        bias_buf = ensure_contiguous(bias.to(device=x2d.device, dtype=wscales.dtype), stream=stream)
        has_bias = 1

    # qw / scales_gn / zeros_gn are already contiguous. K is the caller's K.

    def _launch_chunk(x_chunk: object, out_chunk: object, mm: int) -> None:
        max_m, n_tile, _bt = pick_awq_gemv_tiles(mm, n, k)
        if force_n_tile is not None:
            if force_n_tile not in (1, 4, 8):
                raise ValueError(f"force_n_tile must be 1|4|8, got {force_n_tile}")
            n_tile = force_n_tile
        # The kernel masks rows past M. X stays at the caller's row count.
        _awq_gemv_tuned(
            x_chunk,
            qw,
            scales_gn,
            zeros_gn,
            bias_buf,
            out_chunk,
            mm,
            n,
            has_bias,
            K=k,
            group_size=group_size,
            act_dtype=act_dtype,
            max_m=max_m,
            n_tile=n_tile,
            tuning_schema=TUNING_SCHEMA,
            **_stream_kw(stream),
        )

    if m <= 4:
        _launch_chunk(x2d, out, m)
    else:
        for m0 in range(0, m, 4):
            mm = min(4, m - m0)
            _launch_chunk(x2d[m0 : m0 + mm], out[m0 : m0 + mm], mm)

    return out.reshape(*orig_shape[:-1], n)
