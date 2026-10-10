# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X SVDQuant W4A4 (residual LoRA + smooth, G=64, emit [-7,7]).

Wire format (``TensorCoreSVDQuantW4A4Layout`` / eager ``quantize_svdquant_w4a4``
+ ``scaled_mm_svdquant_w4a4``)::

  qweight:       (N, K // 2)  int8        packed signed W4 residual
  wscales:       (K // 64, N) bf16/fp16   per-group weight scales
  proj_down:     (K, R)       bf16/fp16   SVD down (V^T)
  proj_up:       (N, R)       bf16/fp16   SVD up (U)
  smooth_factor: (K,)         bf16/fp16   input-side smoothing

Forward (SmoothQuant: act DIVIDE by smooth)::

  out = (x / smooth)_int4 @ (W_int4 * wscales).T
      + x @ proj_down @ proj_up.T
      + bias

Quantizer emission: signed ``[-7, 7]`` with ``scale = absmax/7`` (nunchaku
contract; ``-8`` representable but not emitted). Unsigned act path
``[0, 15]`` / ``scale = absmax/15`` for post-GELU+shift fc2
(``act_unsigned=True``).

This module ships:

* FlyDSL ``dequant_svdquant_w4a4_weight`` — device signed unpack → bf16/fp16 W
* Host ``quantize_svdquant_w4a4`` — reference-compatible act quant + LoRA-down
* FlyDSL ``build_svdquant_scaled_mm_fused_module`` / fused ``scaled_mm_svdquant_w4a4``
  — keeps packed INT4 A and W in GMEM; signed/unsigned nibble unpack + group
  scales **in-register**; **i32×8** K-chunk decode (one scale pair / 8 cols);
  **N-tile** grid ``M * ceil(N / n_tile)`` (A reused across tile cols);
  batched multi-acc reduce. LoRA-up residual is **host bf16/fp16**
  (host bf16/fp16 LoRA residual for any adapter count); in-kernel LoRA epilogue intentionally not fused.
* ``svdquant_w4a4_linear`` — quantize + fused scaled_mm on **real M only**
  (slice away pad rows before launch — reference still pads for API compat)

Native ``iu4`` WMMA GEMM is available as ``kernels.gemm.rdna4_iu4_gemm``
(nibble pack → gfx12 scalar-i32 atom). This SVDQuant module keeps the fused
in-reg unpack+scale path as default (groupwise scales).
"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_INT4_GROUP_SIZE = 64
_INT4_MAX = 7  # signed emission [-7, 7]
_UINT4_MAX = 15
_GELU_UNSIGNED_SHIFT = 0.171875
_DEFAULT_PAD = 256
WARP = 32
KERNEL_NAME = "svdquant_dequant_w4_gfx120x"
KERNEL_NAME_FUSED = "svdquant_scaled_mm_fused_gfx120x"
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def pick_svdquant_n_tile(m: int, n: int) -> int:
    """Measured gfx1201 N-tile for fused SVDQuant (2026-09-30).

    Mid/large M keep n_tile=8 (established faster path). M=1 is launch-bound: prefer
    smaller tiles on tiny N; M=1 may still be slower than HIP — optimize, do not drop.
    """
    if m <= 1:
        if n < 128:
            return 1
        return 4 if n < 512 else 8
    if n >= 8:
        return 8
    if n >= 4:
        return 4
    return 1


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


@lru_cache(maxsize=64)
def build_svdquant_dequant_w4_module(
    k: int,
    group_size: int = _INT4_GROUP_SIZE,
    out_dtype: str = "bfloat16",
    block_threads: int | None = None,
    unsigned: bool = False,
) -> Callable[..., None]:
    """Unpack signed INT4 + apply group scales → bf16/fp16 weight row."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 8 or group_size % 8 != 0:
        raise ValueError(f"group_size={group_size} must be a multiple of 8 and >=8")
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
        op="svdquant_dequant",
        unsigned=unsigned,
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Tensor,
        Scales: fx.Tensor,  # (groups, N) ActTy API layout
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
            lo_u = packed & fx.Int32(0xF)
            hi_u = (packed >> fx.Int32(4)) & fx.Int32(0xF)
            if const_expr(unsigned):
                lo = lo_u
                hi = hi_u
            else:
                # Sign-extend nibble → [-8, 7]
                lo = (lo_u >= fx.Int32(8)).select(lo_u - fx.Int32(16), lo_u)
                hi = (hi_u >= fx.Int32(8)).select(hi_u - fx.Int32(16), hi_u)

            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            # API layout (groups, N): s[g, n] = g * N + n
            sidx = fx.Int64(g) * fx.Int64(n_rows) + fx.Int64(bid)
            safe_s = inb.select(sidx, fx.Int64(0))
            s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=OutTy, unit_elems=1))

            v0 = fx.Float32(lo) * s
            v1 = fx.Float32(hi) * s

            o0 = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            # Guard stores: select(...,0) alone still writes Out[0] when BLOCK > packed_k.
            if inb:
                buf_copy_store(o_buf, o0, OutTy(v0), elem=OutTy, unit_elems=1)
                buf_copy_store(o_buf, o1, OutTy(v1), elem=OutTy, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Tensor,
        Scales: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        dequant_kernel(Qdata, Scales, Out, n_rows).launch(
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
def svdquant_dequant_direct(
    Qdata: fx.Tensor,
    Scales: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    K: fx.Constexpr[int],
    group_size: fx.Constexpr[int],
    out_dtype: fx.Constexpr[str],
    unsigned: fx.Constexpr[bool],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_svdquant_dequant_w4_module(
        K, group_size, out_dtype, block_threads=BLOCK_THREADS, unsigned=bool(unsigned)
    )
    launch(Qdata, Scales, Out, n_rows, stream)


_svdquant_dequant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["K", "group_size", "out_dtype", "unsigned", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_svdquant_dequant",
    validate_hook=_validate_dequant,
)(svdquant_dequant_direct)


def dequant_svdquant_w4a4_weight(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    group_size: int = _INT4_GROUP_SIZE,
    *,
    unsigned: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """FlyDSL SVDQuant W4 weight dequant → bf16/fp16 ``(N, K)``.

    ``wscales`` keep layout ``(K//G, N)`` and are indexed in the kernel.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(qweight.device, what="dequant_svdquant_w4a4_weight (gfx120x)")

    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda":
        raise ValueError("qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    k = k_half * 2
    groups = (k + group_size - 1) // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")

    out_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    scales_gn = ensure_contiguous(wscales, stream=stream)
    qw = ensure_contiguous(qweight, stream=stream)
    out = torch.empty((n, k), device=qweight.device, dtype=wscales.dtype)
    _svdquant_dequant_tuned(
        qw,
        scales_gn,
        out,
        n,
        K=k,
        group_size=group_size,
        out_dtype=out_dtype,
        unsigned=unsigned,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    return out


@lru_cache(maxsize=32)
def build_svdquant_act_quant_module(
    dtype: str, k: int, act_unsigned: bool, scale_dtype: str | None = None
) -> Callable[..., None]:
    """One row per block: smooth, group absmax, nibble pack, and LoRA-down dot."""
    if k <= 0:
        raise ValueError(f"K={k} must be positive")
    _tys = {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }
    if scale_dtype is None:
        scale_dtype = dtype
    InTy, in_bytes = _tys[dtype]
    ScaleTy, scale_bytes = _tys[scale_dtype]
    block = _INT4_GROUP_SIZE
    n_groups = k // _INT4_GROUP_SIZE
    tail = k % _INT4_GROUP_SIZE
    packed_cols = (k + 1) // 2
    scale_rows = n_groups if tail == 0 else n_groups + 1
    qmax = _UINT4_MAX if act_unsigned else _INT4_MAX
    qmin = 0 if act_unsigned else -_INT4_MAX
    waves = block // WARP

    @fx.struct
    class SharedStorage:
        tile: fx.Array[fx.Float32, block, 16]
        red: fx.Array[fx.Float32, waves, 16]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def act_quant_kernel(
        X: fx.Tensor,
        Smooth: fx.Tensor,
        LoraX: fx.Tensor,
        LoraDown: fx.Tensor,
        Q: fx.Tensor,
        Scales: fx.Tensor,
        LoraAct: fx.Tensor,
        m_real: fx.Int32,
        m_stride: fx.Int32,
        R: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        c0 = fx.Float32(0.0)
        active = bid < m_real
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile_v = lds.tile.view(fx.make_layout(block, 1))
        red_v = lds.red.view(fx.make_layout(waves, 1))

        def wave_reduce(val: object, do_max: bool) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                other = gpu.shuffle_xor(w, off, WARP)
                w = fx.max(w, other) if do_max else (w + other)
            return w

        def block_reduce(val: object, do_max: bool) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce(val, do_max)
            if lane == 0:
                fx.memref_store(w, red_v, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < fx.Int32(waves)
                lane_safe = in_range.select(lane, fx.Int32(0))
                vv = red_v[lane_safe]
                ww = in_range.select(fx.Float32(vv), c0)
                ww = wave_reduce(ww, do_max)
                if lane == 0:
                    fx.memref_store(ww, red_v, fx.Int32(0))
            gpu.barrier()
            return fx.Float32(red_v[fx.Int32(0)])

        x_buf = ptr_buf_tensor(
            X,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(m_real) * fx.Int64(k) * fx.Int64(in_bytes),
        )
        sm_buf = ptr_buf_tensor(
            Smooth, elem=InTy, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(k) * fx.Int64(in_bytes)
        )
        lx_buf = ptr_buf_tensor(
            LoraX,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(m_real) * fx.Int64(k) * fx.Int64(in_bytes),
        )
        ld_buf = ptr_buf_tensor(
            LoraDown,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(k) * fx.Int64(R) * fx.Int64(in_bytes),
        )
        q_buf = ptr_buf_tensor(
            Q, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(m_real) * fx.Int64(k)
        )
        sc_buf = ptr_buf_tensor(
            Scales,
            elem=ScaleTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(scale_rows) * fx.Int64(m_stride) * fx.Int64(scale_bytes),
        )

        def scale_for_codes(scale_f: fx.Float32) -> fx.Float32:
            """Round the group scale through its stored dtype before the codes."""
            if const_expr(scale_dtype == "float32"):
                return scale_f
            return fx.Float32(fx.Float32(scale_f).to(ScaleTy))

        la_buf = ptr_buf_tensor(
            LoraAct,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(m_real) * fx.Int64(R) * fx.Int64(4),
        )

        def load_f(buf: object, idx: object) -> fx.Float32:
            raw = buf_copy_load(buf, idx, elem=InTy, unit_elems=1)
            if const_expr(dtype != "float32"):
                return fx.Float32(InTy(raw).to(fx.Float32))
            return fx.Float32(raw)

        for r in range(R):
            acc = c0
            for g in range_constexpr(n_groups):
                kidx = fx.Int32(g * block) + tid
                xidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(kidx)
                didx = fx.Int64(kidx) * fx.Int64(R) + fx.Int64(r)
                xv = load_f(lx_buf, active.select(xidx, fx.Int64(0)))
                dv = load_f(ld_buf, didx)
                acc = acc + active.select(xv * dv, c0)
            if const_expr(tail != 0):
                kidx = fx.Int32(n_groups * block) + tid
                live = active & (tid < fx.Int32(tail))
                xidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(kidx)
                didx = fx.Int64(kidx) * fx.Int64(R) + fx.Int64(r)
                xv = load_f(lx_buf, live.select(xidx, fx.Int64(0)))
                dv = load_f(ld_buf, live.select(didx, fx.Int64(0)))
                acc = acc + live.select(xv * dv, c0)
            total = block_reduce(acc, False)
            if tid == 0:
                buf_copy_store(
                    la_buf,
                    fx.Int64(bid) * fx.Int64(R) + fx.Int64(r),
                    active.select(total, c0),
                    elem=fx.Float32,
                    unit_elems=1,
                )
            gpu.barrier()

        for g in range_constexpr(n_groups):
            kidx = fx.Int32(g * block) + tid
            xidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(kidx)
            xv = load_f(x_buf, active.select(xidx, fx.Int64(0)))
            sm = load_f(sm_buf, fx.Int64(kidx))
            sm = fx.max(fmath.absf(sm), fx.Float32(1e-10))
            val = active.select(xv * fx.Float32(fx.rocdl.rcp(T.f32, sm)), c0)
            fx.memref_store(val, tile_v, tid)
            amax = block_reduce(fmath.absf(val), True)
            scale = fx.max(amax * fx.Float32(1.0 / qmax), fx.Float32(1e-10))
            scale = scale_for_codes(scale)
            inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))
            if tid == 0:
                stored = scale if const_expr(scale_dtype == "float32") else fx.Float32(scale).to(ScaleTy)
                sc_idx = fx.Int64(g) * fx.Int64(m_stride) + fx.Int64(bid)
                buf_copy_store(
                    sc_buf,
                    sc_idx,
                    active.select(stored, fx.Float32(0).to(ScaleTy) if const_expr(scale_dtype != "float32") else c0),
                    elem=ScaleTy,
                    unit_elems=1,
                )
            gpu.barrier()
            qf = fmath.roundeven(val * inv)
            qf = fx.max(fx.min(qf, fx.Float32(float(qmax))), fx.Float32(float(qmin)))
            fx.memref_store(qf, tile_v, tid)
            gpu.barrier()
            if tid < fx.Int32(block // 2):
                lo_i = tid * fx.Int32(2)
                hi_i = lo_i + fx.Int32(1)
                lo_b = fx.Int32(fx.Float32(tile_v[lo_i]).to(fx.Int32)) & fx.Int32(15)
                hi_b = fx.Int32(fx.Float32(tile_v[hi_i]).to(fx.Int32)) & fx.Int32(15)
                packed = (lo_b | (hi_b << fx.Int32(4))).to(fx.Int8)
                qidx = fx.Int64(bid) * fx.Int64(packed_cols) + fx.Int64(g * (block // 2)) + fx.Int64(tid)
                if active:
                    buf_copy_store(q_buf, qidx, packed, elem=fx.Int8, unit_elems=1)
            gpu.barrier()

        if const_expr(tail != 0):
            kidx = fx.Int32(n_groups * block) + tid
            live = active & (tid < fx.Int32(tail))
            xidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(kidx)
            xv = load_f(x_buf, live.select(xidx, fx.Int64(0)))
            sm = load_f(sm_buf, live.select(fx.Int64(kidx), fx.Int64(0)))
            sm = fx.max(fmath.absf(sm), fx.Float32(1e-10))
            val = live.select(xv * fx.Float32(fx.rocdl.rcp(T.f32, sm)), c0)
            fx.memref_store(val, tile_v, tid)
            amax = block_reduce(fmath.absf(val), True)
            scale = fx.max(amax * fx.Float32(1.0 / qmax), fx.Float32(1e-10))
            scale = scale_for_codes(scale)
            inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))
            if tid == 0:
                stored = scale if const_expr(scale_dtype == "float32") else fx.Float32(scale).to(ScaleTy)
                sc_idx = fx.Int64(n_groups) * fx.Int64(m_stride) + fx.Int64(bid)
                buf_copy_store(
                    sc_buf,
                    sc_idx,
                    active.select(stored, fx.Float32(0).to(ScaleTy) if const_expr(scale_dtype != "float32") else c0),
                    elem=ScaleTy,
                    unit_elems=1,
                )
            gpu.barrier()
            qf = fmath.roundeven(val * inv)
            qf = fx.max(fx.min(qf, fx.Float32(float(qmax))), fx.Float32(float(qmin)))
            fx.memref_store(live.select(qf, c0), tile_v, tid)
            gpu.barrier()
            if tid < fx.Int32((tail + 1) // 2):
                lo_i = tid * fx.Int32(2)
                hi_i = lo_i + fx.Int32(1)
                lo_b = fx.Int32(fx.Float32(tile_v[lo_i]).to(fx.Int32)) & fx.Int32(15)
                hi_b = fx.Int32(fx.Float32(tile_v[hi_i]).to(fx.Int32)) & fx.Int32(15)
                packed = (lo_b | (hi_b << fx.Int32(4))).to(fx.Int8)
                qidx = fx.Int64(bid) * fx.Int64(packed_cols) + fx.Int64(n_groups * (block // 2)) + fx.Int64(tid)
                if active:
                    buf_copy_store(q_buf, qidx, packed, elem=fx.Int8, unit_elems=1)
            gpu.barrier()

    act_quant_kernel.__name__ = f"svdquant_act_quant_{dtype}_{scale_dtype}_{k}_{int(act_unsigned)}"

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Smooth: fx.Tensor,
        LoraX: fx.Tensor,
        LoraDown: fx.Tensor,
        Q: fx.Tensor,
        Scales: fx.Tensor,
        LoraAct: fx.Tensor,
        m_real: fx.Int32,
        m_stride: fx.Int32,
        R: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        act_quant_kernel(X, Smooth, LoraX, LoraDown, Q, Scales, LoraAct, m_real, m_stride, R).launch(
            grid=(m_real, 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_svdquant_act_quant_{dtype}_{k}_{int(act_unsigned)}"
    return launch


@flyc.jit
def svdquant_act_quant_direct(
    X: fx.Tensor,
    Smooth: fx.Tensor,
    LoraX: fx.Tensor,
    LoraDown: fx.Tensor,
    Q: fx.Tensor,
    Scales: fx.Tensor,
    LoraAct: fx.Tensor,
    m_real: fx.Int32,
    m_stride: fx.Int32,
    R: fx.Int32,
    dtype: fx.Constexpr[str],
    K: fx.Constexpr[int],
    act_unsigned: fx.Constexpr[bool],
    tuning_schema: fx.Constexpr[int],
    scale_dtype: fx.Constexpr[str],
    stream: fx.Stream = fx.Stream(None),
):
    """Block is the group (64). No thread-block choice to autotune."""
    launch = build_svdquant_act_quant_module(dtype, K, bool(act_unsigned), str(scale_dtype))
    launch(X, Smooth, LoraX, LoraDown, Q, Scales, LoraAct, m_real, m_stride, R, stream)


def quantize_svdquant_w4a4(
    x: object,
    smooth: object,
    lora_down: object,
    pad_size: int = _DEFAULT_PAD,
    act_unsigned: bool = False,
    lora_x: object = None,
    *,
    scale_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object, object]:
    """Device INT4 act quant: smooth, group scale, nibble pack, LoRA-down dot.

    Returns:
        q_x: ``(M_pad, K//2)`` int8 packed
        ascales: ``(K//64, M_pad)`` in ``scale_dtype`` (``x.dtype`` when omitted)
        lora_act: ``(M_pad, R)`` float32

    Codes are rounded with the scale after it has been stored in that dtype.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(x.device, what="quantize_svdquant_w4a4 (gfx120x)")

    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    m, k = x.shape
    group = _INT4_GROUP_SIZE
    # A partial last group stays in the kernel. Output K is the caller's K.
    if int(k) <= 0:
        raise ValueError(f"K={int(k)} must be positive")
    if smooth.numel() != k and (smooth.dim() == 0 or smooth.shape[-1] != k):
        raise ValueError(f"smooth must have K={k} elems, got {tuple(smooth.shape)}")
    if lora_down.shape[0] != k:
        raise ValueError(f"lora_down must be (K, R), got {tuple(lora_down.shape)}")
    m_pad = _ceil_div(m, pad_size) * pad_size
    _names = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    dtype = _names[x.dtype]
    if scale_dtype is None:
        scale_dtype = x.dtype
    if scale_dtype not in _names:
        raise ValueError(f"scale_dtype must be f32/f16/bf16, got {scale_dtype}")
    scale_name = _names[scale_dtype]
    lora_src = x if lora_x is None else lora_x
    if lora_src.shape != x.shape:
        raise ValueError(f"lora_x shape {tuple(lora_src.shape)} must match x {tuple(x.shape)}")
    sm = ensure_contiguous(smooth.reshape(-1).to(device=x.device, dtype=x.dtype), stream=stream)
    down = ensure_contiguous(lora_down.to(device=x.device, dtype=x.dtype), stream=stream)
    r = int(down.shape[1])
    x2 = ensure_contiguous(x, stream=stream)
    lx = ensure_contiguous(lora_src.to(device=x.device, dtype=x.dtype), stream=stream)
    packed_cols = (int(k) + 1) // 2
    scale_rows = (int(k) + group - 1) // group
    q = torch.zeros((m_pad, packed_cols), device=x.device, dtype=torch.int8)
    scales = torch.zeros((scale_rows, m_pad), device=x.device, dtype=scale_dtype)
    lora_act = torch.zeros((m_pad, r), device=x.device, dtype=torch.float32)
    kw = dict(
        dtype=dtype,
        K=int(k),
        act_unsigned=act_unsigned,
        tuning_schema=TUNING_SCHEMA,
    )
    if stream is not None:
        kw["stream"] = stream
    svdquant_act_quant_direct(
        x2,
        sm,
        lx,
        down,
        q,
        scales,
        lora_act,
        m,
        m_pad,
        r,
        scale_dtype=scale_name,
        **kw,
    )
    return q, scales, lora_act


@lru_cache(maxsize=64)
def build_svdquant_scaled_mm_fused_module(
    k: int,
    group_size: int = _INT4_GROUP_SIZE,
    out_dtype: str = "bfloat16",
    act_unsigned: bool = False,
    n_tile: int = 1,
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Fused SVDQuant W4A4 scaled mm: packed A×W, in-reg unpack+scale, N-tile.

    Grid ``M * ceil(N / n_tile)``. Each block owns one M-row × ``n_tile``
    output cols; packed A (i32×8) is loaded once per K-step and reused across
    the N-tile (cuts A traffic + launch count vs one-block-per-(m,n)).
    Batched multi-acc reduce (2 barriers). Callers that pad M for host API
    should slice to real M before launch (see ``svdquant_w4a4_linear``).

    Scales use API layout ``(groups, M)`` / ``(groups, N)`` in ActTy
    (host skips transpose + ``.float()`` — measured 2026-09-30).
    """
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 8 or group_size % 8 != 0:
        raise ValueError(f"group_size={group_size} must be a multiple of 8 and >=8, got {group_size}")
    if out_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"out_dtype must be bfloat16|float16, got {out_dtype}")
    if n_tile not in (1, 4, 8):
        raise ValueError(f"n_tile must be 1, 4, or 8, got {n_tile}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")

    packed_k = k // 2
    groups = (k + group_size - 1) // group_size
    vec_k = packed_k // 4 if k % 8 == 0 else (packed_k + 3) // 4
    vec_steps = (vec_k + block_threads - 1) // block_threads
    OutTy = fx.BFloat16 if out_dtype == "bfloat16" else fx.Float16
    out_bytes = 2
    reduction_slots = block_threads // WARP
    n_acc = n_tile
    red_elems = max(reduction_slots * n_acc, n_acc)
    RedTy = fx.Array[fx.Float32, red_elems, 16]
    unsigned = 1 if act_unsigned else 0
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=out_dtype,
        unsigned=unsigned,
        n_tile=n_tile,
        op="svdquant_fused_mm_gn_act",  # (G,M)/(G,N) ActTy — host-prep 2026-09-30
    )

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def fused_mm_kernel(
        ActQ: fx.Tensor,  # (M, K//2) int8
        WgtQ: fx.Tensor,  # (N, K//2) int8
        AScales: fx.Tensor,  # (groups, M) ActTy API layout
        WScales: fx.Tensor,  # (groups, N) ActTy API layout
        Out: fx.Tensor,  # (M, N) out
        M: fx.Int32,
        N: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        grid_n = (N + fx.Int32(n_tile) - fx.Int32(1)) // fx.Int32(n_tile)
        m = bid // grid_n
        n0 = (bid - m * grid_n) * fx.Int32(n_tile)
        row_ok = m < M
        c0 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(red_elems, 1))

        def wave_reduce_sum(w0: object) -> fx.Float32:
            w = w0
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = w + gpu.shuffle_xor(w, off, WARP)
            return w

        a_i32_buf = ptr_buf_tensor(
            ActQ,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(packed_k),
        )
        w_i32_buf = ptr_buf_tensor(
            WgtQ,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(packed_k),
        )
        if const_expr(k % 8 != 0):
            a_byte_buf = ptr_buf_tensor(
                ActQ, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(M) * fx.Int64(packed_k)
            )
            w_byte_buf = ptr_buf_tensor(
                WgtQ, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(N) * fx.Int64(packed_k)
            )

            def _load_packed_i32(buf, row, vi, take):
                word = fx.Int32(0)
                byte0 = vi * fx.Int32(4)
                for b in range_constexpr(4):
                    bp = byte0 + fx.Int32(b)
                    one = fx.Int32(0)
                    if take & (bp < fx.Int32(packed_k)):
                        bidx = fx.Int64(row) * fx.Int64(packed_k) + fx.Int64(bp)
                        one = fx.Int32(buf_copy_load(buf, bidx, elem=fx.Int8, unit_elems=1)) & fx.Int32(255)
                    word = word | (one << fx.Int32(8 * b))
                return word

        as_buf = ptr_buf_tensor(
            AScales,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(M) * fx.Int64(out_bytes),
        )
        ws_buf = ptr_buf_tensor(
            WScales,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(groups) * fx.Int64(N) * fx.Int64(out_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(N) * fx.Int64(out_bytes),
        )

        partials = [c0 for _ in range(n_acc)]
        uflag = fx.Int32(1 if act_unsigned else 0)

        for step in range_constexpr(vec_steps):
            vi = tid + fx.Int32(step * block_threads)
            in_k = row_ok & (vi < fx.Int32(vec_k))
            # i32 chunk = 4 packed bytes = 8 K columns; group_size%8==0 ⇒ one group.
            col0 = vi * fx.Int32(8)
            g = col0 // fx.Int32(group_size)

            # Load A once; reuse across N-tile.
            if const_expr(k % 8 != 0):
                a_raw = _load_packed_i32(a_byte_buf, m, vi, in_k)
            else:
                aidx = fx.Int64(m) * fx.Int64(vec_k) + fx.Int64(vi)
                safe_a = in_k.select(aidx, fx.Int64(0))
                a_raw = fx.Int32(buf_copy_load(a_i32_buf, safe_a, elem=fx.Int32, unit_elems=1))
            as_idx = fx.Int64(g) * fx.Int64(M) + fx.Int64(m)
            safe_as = in_k.select(as_idx, fx.Int64(0))
            sa = fx.Float32(buf_copy_load(as_buf, safe_as, elem=OutTy, unit_elems=1))

            # Decode A nibbles once.
            a_vals = []
            for t in range_constexpr(8):
                a_u = (a_raw >> fx.Int32(4 * t)) & fx.Int32(0xF)
                a_s = (a_u >= fx.Int32(8)).select(a_u - fx.Int32(16), a_u)
                aval = (uflag != fx.Int32(0)).select(a_u, a_s)
                if const_expr(k % 8 != 0):
                    aval = ((col0 + fx.Int32(t)) < fx.Int32(k)).select(aval, c0)
                a_vals.append(aval)

            for nt in range_constexpr(n_tile):
                n = n0 + fx.Int32(nt)
                active = in_k & (n < N)
                if const_expr(k % 8 != 0):
                    w_raw = _load_packed_i32(w_byte_buf, n, vi, active)
                else:
                    widx = fx.Int64(n) * fx.Int64(vec_k) + fx.Int64(vi)
                    safe_w = active.select(widx, fx.Int64(0))
                    w_raw = fx.Int32(buf_copy_load(w_i32_buf, safe_w, elem=fx.Int32, unit_elems=1))
                ws_idx = fx.Int64(g) * fx.Int64(N) + fx.Int64(n)
                safe_ws = active.select(ws_idx, fx.Int64(0))
                sw = fx.Float32(buf_copy_load(ws_buf, safe_ws, elem=OutTy, unit_elems=1))
                scale = sa * sw
                dot = c0
                for t in range_constexpr(8):
                    w_u = (w_raw >> fx.Int32(4 * t)) & fx.Int32(0xF)
                    w_s = (w_u >= fx.Int32(8)).select(w_u - fx.Int32(16), w_u)
                    dot = dot + fx.Float32(a_vals[t]) * fx.Float32(w_s)
                contrib = dot * scale
                partials[nt] = partials[nt] + active.select(contrib, c0)

        # Batched multi-acc reduce (2 barriers total) — same pattern as AWQ tile.
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
            for nt in range_constexpr(n_tile):
                n = n0 + fx.Int32(nt)
                col_ok = n < N
                acc = fx.Float32(red[fx.Int32(nt)])
                oidx = fx.Int64(m) * fx.Int64(N) + fx.Int64(n)
                safe_o = (row_ok & col_ok).select(oidx, fx.Int64(0))
                if row_ok & col_ok:
                    buf_copy_store(o_buf, safe_o, OutTy(acc), elem=OutTy, unit_elems=1)

    fused_mm_kernel.__name__ = f"{KERNEL_NAME_FUSED}_{sig}"

    @flyc.jit
    def launch(
        ActQ: fx.Tensor,
        WgtQ: fx.Tensor,
        AScales: fx.Tensor,
        WScales: fx.Tensor,
        Out: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_n = (N + n_tile - 1) // n_tile
        fused_mm_kernel(ActQ, WgtQ, AScales, WScales, Out, M, N).launch(
            grid=(M * grid_n, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME_FUSED}_{sig}"
    return launch


@contextmanager
def _validate_fused(sig_args):
    import torch

    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def svdquant_fused_direct(
    ActQ: fx.Tensor,
    WgtQ: fx.Tensor,
    AScales: fx.Tensor,
    WScales: fx.Tensor,
    Out: fx.Tensor,
    M: fx.Int32,
    N: fx.Int32,
    K: fx.Constexpr[int],
    group_size: fx.Constexpr[int],
    out_dtype: fx.Constexpr[str],
    act_unsigned: fx.Constexpr[bool],
    n_tile: fx.Constexpr[int],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_svdquant_scaled_mm_fused_module(
        K, group_size, out_dtype, bool(act_unsigned), n_tile, block_threads=BLOCK_THREADS
    )
    launch(ActQ, WgtQ, AScales, WScales, Out, M, N, stream)


_svdquant_fused_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["K", "group_size", "out_dtype", "act_unsigned", "n_tile", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_svdquant_fused",
    validate_hook=_validate_fused,
)(svdquant_fused_direct)


def _host_lora_residual(lora_act_in: object, lora_up: object, out_dtype: torch.dtype | None) -> torch.Tensor:
    """LoRA-up residual in activation dtype. The matmul is the gfx120x GEMM."""
    from collections.abc import Sequence

    import torch

    from kernels.gemm.rdna4_fused_mlp_nmajor import gemm_bf16_nmajor_lds

    def _one(act: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        compute = out_dtype if out_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        delta = gemm_bf16_nmajor_lds(act.to(compute), up.to(compute))
        return delta if delta.dtype == out_dtype else delta.to(out_dtype)

    if isinstance(lora_act_in, Sequence) and not isinstance(lora_act_in, torch.Tensor):
        acts = list(lora_act_in)
        ups = list(lora_up) if isinstance(lora_up, Sequence) and not isinstance(lora_up, torch.Tensor) else [lora_up]
        if len(acts) != len(ups):
            raise ValueError(f"lora_act_in/lora_up length mismatch: {len(acts)} vs {len(ups)}")
        if not acts:
            raise ValueError("lora_act_in/lora_up sequences must be non-empty")
        acc = _one(acts[0], ups[0])
        for a_i, u_i in zip(acts[1:], ups[1:]):
            from kernels.common.gfx120x_row_bias import add_same

            acc = add_same(acc, _one(a_i, u_i))
        return acc
    return _one(lora_act_in, lora_up)


def scaled_mm_svdquant_w4a4(
    act: object,
    wgt: object,
    ascales: object,
    wscales: torch.Tensor,
    lora_act_in: object,
    lora_up: object,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
    group_size: int = _INT4_GROUP_SIZE,
    *,
    stream: torch.cuda.Stream | None = None,
    fused: bool = True,
    force_n_tile: int | None = None,
) -> torch.Tensor:
    """SVDQuant W4A4 scaled mm (host API).

    **Default (``fused=True``, CUDA bf16/fp16):** FlyDSL fused path keeps
    packed INT4 A and W; unpack + ``ascale*wscale`` in-register; host bf16/fp16
    LoRA-up residual (any adapter count). Does not materialize full W/A
    bf16 matrices.

    **Fallback (``fused=False``):** FlyDSL weight/act dequant + ``gemm_bf16_nmajor_lds``
    + host LoRA residual (same device GEMM as other gfx120x hosts; no torch matmul).

    ``fused`` / ``group_size`` / ``stream`` are fork-only.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(act.device, what="scaled_mm_svdquant_w4a4 (gfx120x)")

    m, k_half = act.shape
    n = wgt.shape[0]
    if int(wgt.shape[1]) != int(k_half):
        raise ValueError(
            f"wgt packed K {int(wgt.shape[1])} must match act packed K {int(k_half)} "
            f"(pad qweight/wscales to the same K as soft-padded activations, or keep K%64==0)"
        )
    k = k_half * 2
    compute_dtype = wscales.dtype

    use_fused = (
        fused
        and act.device.type == "cuda"
        and wgt.device.type == "cuda"
        and compute_dtype in (torch.bfloat16, torch.float16)
    )

    if use_fused:
        if ascales.dtype != wscales.dtype:
            raise ValueError(
                f"fused SVDQuant requires ascales.dtype == wscales.dtype "
                f"(got ascales={ascales.dtype}, wscales={wscales.dtype}); "
                f"quantize float32 x then cast ascales, or use matching bf16/fp16 scales"
            )
        if ascales.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"fused SVDQuant scales must be bf16/fp16, got ascales={ascales.dtype}")
        groups = (k + group_size - 1) // group_size
        if tuple(ascales.shape) != (groups, m):
            raise ValueError(f"ascales must be {(groups, m)}, got {tuple(ascales.shape)}")
        if tuple(wscales.shape) != (groups, n):
            raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
        out_dtype = "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
        # API layout (G,M)/(G,N) ActTy — kernel indexes g*M+m / g*N+n.
        ascales_gm = ensure_contiguous(ascales, stream=stream)
        wscales_gn = ensure_contiguous(wscales, stream=stream)
        out = torch.empty((m, n), device=act.device, dtype=compute_dtype)
        n_tile = pick_svdquant_n_tile(m, n)
        if force_n_tile is not None:
            if force_n_tile not in (1, 4, 8):
                raise ValueError(f"force_n_tile must be 1|4|8, got {force_n_tile}")
            n_tile = int(force_n_tile)
        act_c = ensure_contiguous(act, stream=stream)
        wgt_c = ensure_contiguous(wgt, stream=stream)
        _svdquant_fused_tuned(
            act_c,
            wgt_c,
            ascales_gm,
            wscales_gn,
            out,
            m,
            n,
            K=k,
            group_size=group_size,
            out_dtype=out_dtype,
            act_unsigned=act_unsigned,
            n_tile=n_tile,
            tuning_schema=TUNING_SCHEMA,
            **_stream_kw(stream),
        )
        from kernels.common.gfx120x_row_bias import add_row_bias, add_same

        out = add_same(out, _host_lora_residual(lora_act_in, lora_up, compute_dtype))
        if bias is not None:
            out = add_row_bias(out, bias, stream=stream)
        return out

    # Same device dequant and GEMM as the fused path. No torch matmul.
    from kernels.gemm.rdna4_fused_mlp_nmajor import gemm_bf16_nmajor_lds

    wgt_fp = dequant_svdquant_w4a4_weight(wgt, wscales, group_size=group_size, stream=stream)
    act_fp = dequant_svdquant_w4a4_weight(act, ascales, group_size=group_size, unsigned=act_unsigned, stream=stream)
    out = gemm_bf16_nmajor_lds(act_fp, wgt_fp)
    from kernels.common.gfx120x_row_bias import add_same

    out = add_same(out, _host_lora_residual(lora_act_in, lora_up, out.dtype))
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out = add_row_bias(out, bias, stream=stream)
    return out


def svdquant_w4a4_linear(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    proj_down: object,
    proj_up: object,
    smooth: object,
    bias: torch.Tensor | None = None,
    *,
    act_unsigned: bool = False,
    pad_size: int = _DEFAULT_PAD,
    group_size: int = _INT4_GROUP_SIZE,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Layout linear: quantize + scaled_mm (unpad M).

    Quantize still pads to ``pad_size`` for host API compat, but fused
    scaled_mm + host LoRA run on the real ``M`` rows only (slice pad away).
    For ``act_unsigned`` (nunchaku post-GELU fc2), applies ``+0.171875`` shift
    to the main-path activation only; LoRA always sees raw ``x``.
    """
    require_gfx120x(x.device, what="svdquant_w4a4_linear (gfx120x)")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    m = x2d.shape[0]
    if m == 0:
        n = int(qweight.shape[0])
        return torch.empty(*orig_shape[:-1], n, device=x.device, dtype=x.dtype)

    if act_unsigned:
        x_main = x2d + _GELU_UNSIGNED_SHIFT
        lora_x = x2d
    else:
        x_main = x2d
        lora_x = None

    q_x, ascales, lora_act = quantize_svdquant_w4a4(
        x_main,
        smooth=smooth,
        lora_down=proj_down,
        pad_size=pad_size,
        act_unsigned=act_unsigned,
        lora_x=lora_x,
        scale_dtype=wscales.dtype,
        stream=stream,
    )
    k_half_act = int(q_x.shape[1])
    k_half_w = int(qweight.shape[1])
    if k_half_act != k_half_w:
        raise ValueError(f"act packed K {k_half_act} != qweight packed K {k_half_w}")
    if ascales.dtype != wscales.dtype:
        raise ValueError(f"act scales {ascales.dtype} must match weight scales {wscales.dtype}")
    # Unpad-aware: reference pads M for API/WMMA align, but our fused grid is
    # M * ceil(N/n_tile) — do not launch pad rows or host-LoRA them.
    out = scaled_mm_svdquant_w4a4(
        act=q_x[:m],
        wgt=qweight,
        ascales=ascales[:, :m],
        wscales=wscales,
        lora_act_in=lora_act[:m],
        lora_up=proj_up,
        bias=bias,
        act_unsigned=act_unsigned,
        group_size=group_size,
        stream=stream,
    )
    n = qweight.shape[0]
    return out.reshape(*orig_shape[:-1], n)
