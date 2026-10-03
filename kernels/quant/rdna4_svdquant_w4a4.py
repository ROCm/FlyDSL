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
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import gpu, range_constexpr
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

from .rdna4_int4_codec import (
    dequant_int4_groupwise_signed,
    pack_int4_row_major,
    unpack_int4_row_major,
    unpack_uint4_row_major,
)

_INT4_GROUP_SIZE = 64
_INT4_MAX = 7  # signed emission [-7, 7]
_UINT4_MAX = 15
_GELU_UNSIGNED_SHIFT = 0.171875
_DEFAULT_PAD = 256
_MAX_K = 16384
WARP = 32
KERNEL_NAME = "svdquant_dequant_w4_gfx120x"
KERNEL_NAME_FUSED = "svdquant_scaled_mm_fused_gfx120x"


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


def _ptr(t: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def reference_dequant_svdquant_w4a4_weight(
    qweight: torch.Tensor, wscales: torch.Tensor, group_size: int = _INT4_GROUP_SIZE
) -> torch.Tensor:
    """Torch reference: signed INT4 × group scales (eager reference weight path)."""
    return dequant_int4_groupwise_signed(qweight, wscales, group_size=group_size)


@lru_cache(maxsize=64)
def build_svdquant_dequant_w4_module(
    k: int,
    group_size: int = _INT4_GROUP_SIZE,
    out_dtype: str = "bfloat16",
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Unpack signed INT4 + apply group scales → bf16/fp16 weight row."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 8 or group_size % 2 != 0 or k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide K={k}, be even, and >=8")
    if k > _MAX_K:
        raise ValueError(f"K={k} exceeds budget {_MAX_K}")
    if out_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"out_dtype must be bfloat16|float16, got {out_dtype}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    packed_k = k // 2
    groups = k // group_size
    pack_steps = (packed_k + block_threads - 1) // block_threads
    OutTy = fx.BFloat16 if out_dtype == "bfloat16" else fx.Float16
    out_bytes = 2
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=out_dtype,
        op="svdquant_dequant",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Pointer,
        Scales: fx.Pointer,  # (N, groups) row-major f32
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
    ) -> None:
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
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
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(groups) * fx.Int64(4),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K) * fx.Int64(out_bytes),
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
            # Sign-extend nibble → [-8, 7]
            lo = (lo_u >= fx.Int32(8)).select(lo_u - fx.Int32(16), lo_u)
            hi = (hi_u >= fx.Int32(8)).select(hi_u - fx.Int32(16), hi_u)

            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            sidx = fx.Int64(bid) * fx.Int64(groups) + fx.Int64(g)
            safe_s = inb.select(sidx, fx.Int64(0))
            s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=fx.Float32, unit_elems=1))

            v0 = fx.Float32(lo) * s
            v1 = fx.Float32(hi) * s

            o0 = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            out_idx0 = inb.select(o0, fx.Int64(0))
            out_idx1 = inb.select(o1, fx.Int64(0))
            buf_copy_store(o_buf, out_idx0, OutTy(v0), elem=OutTy, unit_elems=1)
            buf_copy_store(o_buf, out_idx1, OutTy(v1), elem=OutTy, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Pointer,
        Scales: fx.Pointer,
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        dequant_kernel(Qdata, Scales, Out, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def dequant_svdquant_w4a4_weight(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    group_size: int = _INT4_GROUP_SIZE,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """FlyDSL SVDQuant W4 weight dequant → bf16/fp16 ``(N, K)``.

    ``wscales`` keep layout ``(K//G, N)``; converted to row-major
    ``(N, G)`` f32 for the kernel.
    """
    require_gfx120x(qweight.device, what="dequant_svdquant_w4a4_weight (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda":
        raise ValueError("qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    groups = k // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")

    out_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    scales_ng = wscales.t().contiguous().float()
    out = torch.empty((n, k), device=qweight.device, dtype=wscales.dtype)
    launch = build_svdquant_dequant_w4_module(k=k, group_size=group_size, out_dtype=out_dtype)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        _ptr(qweight.contiguous()),
        _ptr(scales_ng),
        _ptr(out),
        n,
        k,
        stream,
    )
    return out


def quantize_svdquant_w4a4(
    x: object,
    smooth: object,
    lora_down: object,
    pad_size: int = _DEFAULT_PAD,
    act_unsigned: bool = False,
    lora_x: object = None,
) -> tuple[object, object, object]:
    """Quantize activations to INT4 with smoothing + LoRA-down (host API).

    Host/torch reference path matching common eager SVDQuant pack layout —
    host path for correct numerics; fused device act-quant is a follow-up.

    Returns:
        q_x: ``(M_pad, K//2)`` int8 packed
        ascales: ``(K//64, M_pad)`` same dtype as x
        lora_act: ``(M_pad, R)`` float32
    """
    require_gfx120x(x.device, what="quantize_svdquant_w4a4 (gfx120x)")
    import torch
    import torch.nn.functional as F

    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    m, k = x.shape
    group = _INT4_GROUP_SIZE
    if k % group != 0:
        raise ValueError(f"K={k} not divisible by group_size={group}")
    if smooth.shape[-1] != k and smooth.numel() != k:
        raise ValueError(f"smooth must have K={k} elems, got {tuple(smooth.shape)}")
    if lora_down.shape[0] != k:
        raise ValueError(f"lora_down must be (K, R), got {tuple(lora_down.shape)}")
    m_pad = _ceil_div(m, pad_size) * pad_size

    lora_src = lora_x if lora_x is not None else x
    lora_act = lora_src.float() @ lora_down.float()

    x_smooth = x / smooth.reshape(-1).to(dtype=x.dtype, device=x.device)
    groups = x_smooth.view(m, k // group, group)
    absmax = groups.abs().amax(dim=-1).clamp(min=1e-10)
    qmax = _UINT4_MAX if act_unsigned else _INT4_MAX
    qmin = 0 if act_unsigned else -_INT4_MAX
    scales = absmax / qmax
    q_vals = (groups / scales.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
    q_vals = q_vals.reshape(m, k)
    q_packed = pack_int4_row_major(q_vals)

    if m_pad > m:
        pad = m_pad - m
        q_packed = F.pad(q_packed, (0, 0, 0, pad))
        scales = F.pad(scales, (0, 0, 0, pad))
        lora_act = F.pad(lora_act, (0, 0, 0, pad))

    ascales = scales.t().contiguous().to(x.dtype)
    return q_packed, ascales, lora_act


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
    if group_size < 8 or group_size % 2 != 0 or k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide K={k}, be even, and >=8")
    if k > _MAX_K:
        raise ValueError(f"K={k} exceeds budget {_MAX_K}")
    if out_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"out_dtype must be bfloat16|float16, got {out_dtype}")
    if n_tile not in (1, 4, 8):
        raise ValueError(f"n_tile must be 1, 4, or 8, got {n_tile}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    packed_k = k // 2
    if packed_k % 4 != 0:
        raise ValueError(f"packed_k={packed_k} must be multiple of 4 for i32 decode")
    groups = k // group_size
    vec_k = packed_k // 4
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
        ActQ: fx.Pointer,  # (M, K//2) int8
        WgtQ: fx.Pointer,  # (N, K//2) int8
        AScales: fx.Pointer,  # (groups, M) ActTy API layout
        WScales: fx.Pointer,  # (groups, N) ActTy API layout
        Out: fx.Pointer,  # (M, N) out
        M: fx.Int32,
        N: fx.Int32,
    ) -> None:
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
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
                a_vals.append((uflag != fx.Int32(0)).select(a_u, a_s))

            for nt in range_constexpr(n_tile):
                n = n0 + fx.Int32(nt)
                active = in_k & (n < N)
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
        ActQ: fx.Pointer,
        WgtQ: fx.Pointer,
        AScales: fx.Pointer,
        WScales: fx.Pointer,
        Out: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_n = (N + n_tile - 1) // n_tile
        fused_mm_kernel(ActQ, WgtQ, AScales, WScales, Out, M, N).launch(
            grid=(M * grid_n, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME_FUSED}_{sig}"
    return launch


def _host_lora_residual(lora_act_in: object, lora_up: object, out_dtype: torch.dtype | None) -> torch.Tensor:
    """Host LoRA-up residual(s) in activation dtype (unbounded adapter count).

    Matches FP8/int8 fused policy: avoid heavy in-kernel LoRA epilogue on gfx120x.
    **Unbounded N:** ``lora_act_in`` / ``lora_up`` may be a single tensor or a
    sequence of equal length; residuals ``act_i @ up_i.T`` are summed in load
    order (no fixed max adapters).
    """
    from collections.abc import Sequence

    import torch

    if isinstance(lora_act_in, Sequence) and not isinstance(lora_act_in, torch.Tensor):
        acts = list(lora_act_in)
        ups = list(lora_up) if isinstance(lora_up, Sequence) and not isinstance(lora_up, torch.Tensor) else [lora_up]
        if len(acts) != len(ups):
            raise ValueError(f"lora_act_in/lora_up length mismatch: {len(acts)} vs {len(ups)}")
        if not acts:
            raise ValueError("lora_act_in/lora_up sequences must be non-empty")
        acc = None
        for a_i, u_i in zip(acts, ups):
            delta = a_i.to(dtype=out_dtype) @ u_i.to(dtype=out_dtype).t()
            acc = delta if acc is None else acc + delta
        return acc
    a = lora_act_in.to(dtype=out_dtype)
    b = lora_up.to(dtype=out_dtype)
    return a @ b.t()


def reference_scaled_mm_svdquant_w4a4(
    act: object,
    wgt: object,
    ascales: object,
    wscales: torch.Tensor,
    lora_act_in: object,
    lora_up: object,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
    group_size: int = _INT4_GROUP_SIZE,
) -> torch.Tensor:
    """Pure-torch SVDQuant W4A4 GEMM + LoRA-up (eager reference semantic mirror)."""

    m, k_half = act.shape
    k = k_half * 2
    compute_dtype = wscales.dtype

    wgt_fp = dequant_int4_groupwise_signed(wgt, wscales, group_size=group_size)

    unpack_act = unpack_uint4_row_major if act_unsigned else unpack_int4_row_major
    act_int = unpack_act(act).to(compute_dtype).view(m, k // group_size, group_size)
    ascales_mng = ascales.t().unsqueeze(-1)
    act_fp = (act_int * ascales_mng).view(m, k)

    out = act_fp @ wgt_fp.t()
    lora_contribution = lora_act_in.float() @ lora_up.float().t()
    out = out + lora_contribution.to(out.dtype)
    if bias is not None:
        out = out + bias
    return out


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
    use_flydsl_weight_dequant: bool = True,
    force_n_tile: int | None = None,
) -> torch.Tensor:
    """SVDQuant W4A4 scaled mm (host API).

    **Default (``fused=True``, CUDA bf16/fp16):** FlyDSL fused path keeps
    packed INT4 A and W; unpack + ``ascale*wscale`` in-register; host bf16/fp16
    LoRA-up residual (host, any adapter count). Does not materialize full W/A
    bf16 matrices.

    **Fallback (``fused=False`` or non-CUDA):** weight dequant (FlyDSL or host)
    + host act dequant + torch mm + host LoRA (legacy host path).

    Args mirror reference ``scaled_mm_svdquant_w4a4``; ``fused`` /
    ``use_flydsl_weight_dequant`` / ``group_size`` / ``stream`` are fork-only.
    """
    require_gfx120x(act.device, what="scaled_mm_svdquant_w4a4 (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled

    m, k_half = act.shape
    n = wgt.shape[0]
    k = k_half * 2
    compute_dtype = wscales.dtype

    use_fused = (
        fused
        and act.device.type == "cuda"
        and wgt.device.type == "cuda"
        and compute_dtype in (torch.bfloat16, torch.float16)
        and k % group_size == 0
    )

    if use_fused:
        groups = k // group_size
        if tuple(ascales.shape) != (groups, m):
            raise ValueError(f"ascales must be {(groups, m)}, got {tuple(ascales.shape)}")
        if tuple(wscales.shape) != (groups, n):
            raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
        out_dtype = "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
        # API layout (G,M)/(G,N) ActTy — kernel indexes g*M+m / g*N+n.
        ascales_gm = ascales if ascales.is_contiguous() else ascales.contiguous()
        wscales_gn = wscales if wscales.is_contiguous() else wscales.contiguous()
        out = torch.empty((m, n), device=act.device, dtype=compute_dtype)
        n_tile = pick_svdquant_n_tile(m, n)
        if force_n_tile is not None:
            if force_n_tile not in (1, 4, 8):
                raise ValueError(f"force_n_tile must be 1|4|8, got {force_n_tile}")
            n_tile = int(force_n_tile)
        launch = build_svdquant_scaled_mm_fused_module(
            k=k,
            group_size=group_size,
            out_dtype=out_dtype,
            act_unsigned=act_unsigned,
            n_tile=n_tile,
        )
        if stream is None:
            stream = torch.cuda.current_stream()
        act_c = act if act.is_contiguous() else act.contiguous()
        wgt_c = wgt if wgt.is_contiguous() else wgt.contiguous()
        _run_compiled(
            launch,
            _ptr(act_c),
            _ptr(wgt_c),
            _ptr(ascales_gm),
            _ptr(wscales_gn),
            _ptr(out),
            m,
            n,
            stream,
        )
        # Host LoRA residual (bf16/fp16) — not fused into the int4 kernel.
        out = out + _host_lora_residual(lora_act_in, lora_up, compute_dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    # --- legacy host path: dequant -> host mm -> host LoRA ---
    if use_flydsl_weight_dequant and wgt.device.type == "cuda" and wscales.dtype in (torch.bfloat16, torch.float16):
        wgt_fp = dequant_svdquant_w4a4_weight(wgt, wscales, group_size=group_size, stream=stream)
    else:
        wgt_fp = dequant_int4_groupwise_signed(wgt, wscales, group_size=group_size)

    unpack_act = unpack_uint4_row_major if act_unsigned else unpack_int4_row_major
    act_int = unpack_act(act).to(compute_dtype).view(m, k // group_size, group_size)
    ascales_mng = ascales.t().unsqueeze(-1)
    act_fp = (act_int * ascales_mng).view(m, k)

    out = act_fp @ wgt_fp.t()
    out = out + _host_lora_residual(lora_act_in, lora_up, out.dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
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
    )
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
