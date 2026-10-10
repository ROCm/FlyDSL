# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X Asym W4A8 (grouped INT4 weights + INT8 acts via ConvRot).

Wire format (``AsymW4A8Int8Layout`` / HIP ``w4a8_int8_*``):

* ``qdata[N, K//2]`` int8 -- **unsigned** nibble pack: low = even col, high = odd
  (``(u0 & 0xF) | ((u1 & 0xF) << 4)``); indices in ``[0, 15]``.
* ``s_rel[N, K//group_size]`` -- per-group relative scale (often fp8 e4m3fn).
* ``s_channel[N]`` float32 -- per-row channel scale (``amax/127`` of shifted W).
* Optional ``codebook[16]`` float32 Lloyd-Max levels; when present, decode
  ``values = codebook[idx]``, else ``values = idx - 8``.
* Optional ``correction[groups, N]`` for asymmetric (``symmetric=False``).
  That term does not fit the int8 WMMA product. The linear path runs one
  reconstruct kernel, then ``gemm_bf16_nmajor_lds``. Reconstruct decodes the
  packed int4, applies ``s_rel``, the int8-grid round, ``s_channel``, and the
  correction in registers, and un-rotates one ConvRot group in LDS. It does
  not allocate an int8 grid or an f32 weight.
* Weights are ConvRot-rotated offline (``convrot_groupsize`` in {16,64,256});
  symmetric activations use online ConvRot + INT8 before iu8 GEMM.

Device pack is the frozen 16-level LUT plus two ALS updates, or a symmetric /
asymmetric round. Symmetric inference: ``dequant_int4_grouped_to_int8`` then
``int8_linear_convrot``.

"""

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

# Default Lloyd-Max table for group-normalized Gaussian (eager w4a8).
_FIXED_LUT = (
    -0.980602,
    -0.794529,
    -0.638165,
    -0.500986,
    -0.377321,
    -0.263187,
    -0.155210,
    -0.050720,
    0.052541,
    0.156985,
    0.265284,
    0.379533,
    0.502636,
    0.638953,
    0.794876,
    0.980671,
)
_W4A8_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19
_ALS_ITERS = 2
_SUPPORTED_CONVROT = (16, 64, 256)
KERNEL_NAME = "w4a8_dequant_int4_to_int8_gfx120x"
WARP = 32
# Bump to invalidate dequant / pack autotune winner caches after a codegen change.
_W4A8_DEQUANT_TUNING_SCHEMA = 1
_W4A8_PACK_TUNING_SCHEMA = 1
_W4A8_BLOCKS = (32, 64, 128, 256, 512, 1024)


def _kernel_signature(**params: object) -> str:
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


def _block_threads(k: int) -> int:
    from kernels.common.gfx120x_autotune_tables import pick_asym_w4a8_block_threads

    return pick_asym_w4a8_block_threads(k)


def _legal_block(block_threads: int) -> int:
    """Wave32 block from the shared autotune set."""
    block = int(block_threads)
    if block not in _W4A8_BLOCKS:
        raise ValueError(f"block_threads={block} must be one of {_W4A8_BLOCKS}")
    return block


def _weight_elem(weight_dtype: str) -> tuple[object, int]:
    table = {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }
    if weight_dtype not in table:
        raise ValueError(f"weight must be float32, float16, or bfloat16, got {weight_dtype}")
    return table[weight_dtype]


@lru_cache(maxsize=64)
def build_w4a8_dequant_int4_to_int8_module(
    k: int,
    group_size: int = 16,
    use_codebook: bool = True,
    srel_fp8: bool = False,
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Decode packed unsigned INT4 (+ optional codebook) × s_rel → INT8 grid."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 4 or k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size} (>=4)")

    if block_threads is None:
        block_threads = _block_threads(k)
    block_threads = _legal_block(block_threads)

    packed_k = k // 2
    groups = k // group_size
    pack_steps = (packed_k + block_threads - 1) // block_threads
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        codebook=use_codebook,
        srel_fp8=srel_fp8,
        op="w4a8_dequant",
    )
    srel_elem = fx.Uint8 if srel_fp8 else fx.Float32
    srel_bytes = 1 if srel_fp8 else 4

    @fx.struct
    class DequantStorage:
        codebook: fx.Array[fx.Float32, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Tensor,
        SRel: fx.Tensor,
        Codebook: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        K: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        active_row = bid < n_rows

        lds = fx.SharedAllocator().allocate(DequantStorage).peek()
        cb_view = lds.codebook.view(fx.make_layout(16, 1))

        if const_expr(use_codebook):
            if tid < fx.Int32(16):
                cb_buf = ptr_buf_tensor(
                    Codebook,
                    elem=fx.Float32,
                    n=0x3FFFFFFF,
                    unit_elems=1,
                    num_records_bytes=fx.Int64(16 * 4),
                )
                v = buf_copy_load(cb_buf, fx.Int64(tid), elem=fx.Float32, unit_elems=1)
                fx.memref_store(fx.Float32(v), cb_view, tid)
            gpu.barrier()

        q_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            SRel,
            elem=srel_elem,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(groups) * fx.Int64(srel_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K),
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

            if const_expr(use_codebook):
                lo_v = fx.Float32(cb_view[lo])
                hi_v = fx.Float32(cb_view[hi])
            else:
                lo_v = fx.Float32(lo) - fx.Float32(8.0)
                hi_v = fx.Float32(hi) - fx.Float32(8.0)

            # Both nibbles share a group when group_size >= 2 (layout constraint).
            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            sidx = fx.Int64(bid) * fx.Int64(groups) + fx.Int64(g)
            safe_s = inb.select(sidx, fx.Int64(0))
            raw_s = buf_copy_load(s_buf, safe_s, elem=srel_elem, unit_elems=1)
            if const_expr(srel_fp8):
                bits = fx.Int32(raw_s) & fx.Int32(0xFF)
                s = fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, bits.ir_value(), 0))
            else:
                s = fx.Float32(raw_s)

            q0f = fmath.roundeven(lo_v * s)
            q1f = fmath.roundeven(hi_v * s)
            q0f = fx.max(fx.min(q0f, fx.Float32(127.0)), fx.Float32(-127.0))
            q1f = fx.max(fx.min(q1f, fx.Float32(127.0)), fx.Float32(-127.0))
            q0 = q0f.to(fx.Int8)
            q1 = q1f.to(fx.Int8)

            o0 = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            if inb:
                buf_copy_store(o_buf, o0, q0, elem=fx.Int8, unit_elems=1)
                buf_copy_store(o_buf, o1, q1, elem=fx.Int8, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Tensor,
        SRel: fx.Tensor,
        Codebook: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        dequant_kernel(Qdata, SRel, Codebook, Out, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def rotate_convrot_weight(weight: torch.Tensor, convrot_groupsize: int = 256) -> torch.Tensor:
    """``W @ H`` per ConvRot group, on the gfx120x FWHT kernel."""
    from kernels.quant.rdna4_int8_convrot import convrot_fwht

    return convrot_fwht(weight, convrot_groupsize)


def validate_w4a8_weight_shape(weight: torch.Tensor, group_size: int, convrot_groupsize: int) -> None:
    """Raise unless the packed W4A8 weight matches N, K, and group size."""
    if weight.dim() != 2:
        raise ValueError(f"W4A8 weight must be 2D, got shape {tuple(weight.shape)}")
    k = weight.shape[1]
    if (
        k % 16 != 0
        or k % group_size != 0
        or k % convrot_groupsize != 0
        or group_size < 4
        or (16 % group_size != 0 and group_size % 16 != 0)
    ):
        raise ValueError(
            f"K={k} must be divisible by 16, group_size={group_size}, and "
            f"convrot_groupsize={convrot_groupsize}; group_size must be >=4 "
            f"and divide 16 or be a multiple of 16"
        )


def _pick_level(levels, index):
    """levels[index] when index is a traced integer, not a Python int."""
    chosen = levels[0]
    for j in range_constexpr(16):
        hit = index == fx.Int32(j)
        chosen = hit.select(levels[j], chosen)
    return chosen


def _nearest_level(value, levels) -> fx.Int32:
    """Index of the closest of 16 reconstruction levels. Inlined while tracing."""
    best_i = fx.Int32(0)
    best_d = fmath.absf(value - levels[0])
    for j in range_constexpr(1, 16):
        dist = fmath.absf(value - levels[j])
        closer = dist < best_d
        best_d = closer.select(dist, best_d)
        best_i = closer.select(fx.Int32(j), best_i)
    return best_i


@lru_cache(maxsize=16)
def build_w4a8_pack_module(
    group_size: int, mode: int, weight_name: str = "float32", srel_fp8: bool = False
) -> Callable[..., None]:
    """Device W4A8 pack.

    mode 0: frozen 16-level codebook, two ALS scale updates, then int8-grid reassign.
    mode 1: symmetric round, scale = amax/7, stored nibble is signed+8.
    mode 2: asymmetric round onto [0, 15], plus the per-group correction term.

    ``weight_name`` is the weight's wire dtype. Values are widened to f32 in the kernel.
    """
    if group_size < 4:
        raise ValueError(f"group_size={group_size} must be >= 4")
    w_ty, w_bytes = _weight_elem(weight_name)
    gs = group_size
    block = 256

    def _load_w(buf, idx):
        raw = buf_copy_load(buf, idx, elem=w_ty, unit_elems=1)
        if const_expr(weight_name == "float32"):
            return fx.Float32(raw)
        return raw.to(fx.Float32)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def group_stats(
        Weight: fx.Tensor,
        Scale: fx.Tensor,
        Gmax: fx.Tensor,
        Amin: fx.Tensor,
        Codebook: fx.Tensor,
        n_rows: fx.Int32,
        n_groups: fx.Int32,
    ) -> None:
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(block) + fx.Int32(fx.thread_idx.x)
        n_tot = n_rows * n_groups
        inb = gid < n_tot
        safe_g = inb.select(gid, fx.Int32(0))
        row = safe_g // n_groups
        group = safe_g - row * n_groups
        k = n_groups * fx.Int32(gs)
        wbuf = ptr_buf_tensor(
            Weight,
            elem=w_ty,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(k) * fx.Int64(w_bytes),
        )
        cbuf = ptr_buf_tensor(Codebook, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(64))
        lut = []
        for j in range_constexpr(16):
            lut.append(fx.Float32(buf_copy_load(cbuf, fx.Int64(j), elem=fx.Float32, unit_elems=1)))
        vals = []
        for i in range_constexpr(gs):
            col = group * fx.Int32(gs) + fx.Int32(i)
            idx = fx.Int64(row) * fx.Int64(k) + fx.Int64(col)
            vals.append(_load_w(wbuf, idx))
        amax = fmath.absf(vals[0])
        amin = vals[0]
        vmax = vals[0]
        for i in range_constexpr(1, gs):
            amax = fx.max(amax, fmath.absf(vals[i]))
            amin = fx.min(amin, vals[i])
            vmax = fx.max(vmax, vals[i])
        if const_expr(mode == 0):
            scale = fx.max(amax, fx.Float32(1e-8))
            codes = []
            for i in range_constexpr(gs):
                codes.append(_nearest_level(vals[i] / scale, lut))
            for _it in range_constexpr(_ALS_ITERS):
                num = fx.Float32(0.0)
                den = fx.Float32(0.0)
                for i in range_constexpr(gs):
                    qc = _pick_level(lut, codes[i])
                    num = num + vals[i] * qc
                    den = den + qc * qc
                scale = fx.max(num / fx.max(den, fx.Float32(1e-8)), fx.Float32(1e-8))
                codes = []
                for i in range_constexpr(gs):
                    codes.append(_nearest_level(vals[i] / scale, lut))
            gmax = fx.Float32(0.0)
            for i in range_constexpr(gs):
                gmax = fx.max(gmax, fmath.absf(_pick_level(lut, codes[i]) * scale))
        elif const_expr(mode == 1):
            scale = fx.max(amax / fx.Float32(7.0), fx.Float32(1e-8))
            gmax = fx.Float32(0.0)
            for i in range_constexpr(gs):
                signed = fmath.roundeven(vals[i] / scale)
                signed = fx.max(fx.min(signed, fx.Float32(7.0)), fx.Float32(-8.0))
                gmax = fx.max(gmax, fmath.absf(signed * scale))
        else:
            # Asymmetric range is max(v) - min(v). abs-max widens the scale
            # whenever the negative peak is larger than the positive peak.
            scale = fx.max((vmax - amin) / fx.Float32(15.0), fx.Float32(1e-8))
            gmax = fx.Float32(0.0)
            for i in range_constexpr(gs):
                code = fmath.roundeven((vals[i] - amin) / scale)
                code = fx.max(fx.min(code, fx.Float32(15.0)), fx.Float32(0.0))
                gmax = fx.max(gmax, fmath.absf((code - fx.Float32(8.0)) * scale))
        if inb:
            sbuf = ptr_buf_tensor(
                Scale, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
            )
            mbuf = ptr_buf_tensor(
                Gmax, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
            )
            abuf = ptr_buf_tensor(
                Amin, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
            )
            buf_copy_store(sbuf, fx.Int64(safe_g), scale, elem=fx.Float32, unit_elems=1)
            buf_copy_store(mbuf, fx.Int64(safe_g), gmax, elem=fx.Float32, unit_elems=1)
            buf_copy_store(abuf, fx.Int64(safe_g), amin, elem=fx.Float32, unit_elems=1)

    @fx.struct
    class RowMaxStorage:
        red: fx.Array[fx.Float32, block, 16]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def row_max(Gmax: fx.Tensor, SChan: fx.Tensor, n_rows: fx.Int32, n_groups: fx.Int32) -> None:
        row = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        in_row = row < n_rows
        mbuf = ptr_buf_tensor(
            Gmax,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(n_groups) * fx.Int64(4),
        )
        local = fx.Float32(0.0)
        for g in range(tid, n_groups, fx.Int32(block)):
            take = in_row
            idx = fx.Int64(row) * fx.Int64(n_groups) + fx.Int64(g)
            val = fx.Float32(buf_copy_load(mbuf, take.select(idx, fx.Int64(0)), elem=fx.Float32, unit_elems=1))
            local = fx.max(local, take.select(val, fx.Float32(0.0)))
        lds = fx.SharedAllocator().allocate(RowMaxStorage).peek()
        red = lds.red.view(fx.make_layout(block, 1))
        fx.memref_store(local, red, tid)
        gpu.barrier()
        if tid == fx.Int32(0):
            mx = fx.Float32(0.0)
            for t in range_constexpr(block):
                piece = fx.Float32(red[fx.Int32(t)])
                mx = fx.max(mx, piece)
            sch = fx.max(mx / fx.Float32(127.0), fx.Float32(1e-8))
            if in_row:
                ob = ptr_buf_tensor(
                    SChan, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_rows) * fx.Int64(4)
                )
                buf_copy_store(ob, fx.Int64(row), sch, elem=fx.Float32, unit_elems=1)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def finalize(
        Weight: fx.Tensor,
        Scale: fx.Tensor,
        SChan: fx.Tensor,
        Amin: fx.Tensor,
        Codebook: fx.Tensor,
        Packed: fx.Tensor,
        SRel: fx.Tensor,
        Corr: fx.Tensor,
        n_rows: fx.Int32,
        n_groups: fx.Int32,
    ) -> None:
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(block) + fx.Int32(fx.thread_idx.x)
        n_tot = n_rows * n_groups
        inb = gid < n_tot
        safe_g = inb.select(gid, fx.Int32(0))
        row = safe_g // n_groups
        group = safe_g - row * n_groups
        k = n_groups * fx.Int32(gs)
        wbuf = ptr_buf_tensor(
            Weight,
            elem=w_ty,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(k) * fx.Int64(w_bytes),
        )
        sbuf = ptr_buf_tensor(
            Scale, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
        )
        cbuf = ptr_buf_tensor(
            SChan, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_rows) * fx.Int64(4)
        )
        abuf = ptr_buf_tensor(
            Amin, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
        )
        lutb = ptr_buf_tensor(Codebook, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(64))
        scale = fx.Float32(buf_copy_load(sbuf, fx.Int64(safe_g), elem=fx.Float32, unit_elems=1))
        sch = fx.Float32(buf_copy_load(cbuf, fx.Int64(row), elem=fx.Float32, unit_elems=1))
        amin = fx.Float32(buf_copy_load(abuf, fx.Int64(safe_g), elem=fx.Float32, unit_elems=1))
        srel = scale / sch
        # Dequant reloads s_rel as e4m3. Codes and the correction must use that
        # stored value, not the wider f32 ratio.
        if const_expr(srel_fp8):
            pk = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                srel.ir_value(),
                fx.Float32(0.0).ir_value(),
                fx.Int32(0).ir_value(),
                0,
            )
            bits = fx.Int32(pk) & fx.Int32(0xFF)
            srel = fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, bits.ir_value(), 0))
            scale_eff = srel * sch
        else:
            scale_eff = scale
        lut = []
        for j in range_constexpr(16):
            lut.append(fx.Float32(buf_copy_load(lutb, fx.Int64(j), elem=fx.Float32, unit_elems=1)))
        codes = []
        if const_expr(mode == 0):
            levels = []
            for j in range_constexpr(16):
                lvl = fmath.roundeven(lut[j] * srel)
                levels.append(fx.max(fx.min(lvl, fx.Float32(127.0)), fx.Float32(-127.0)))
            for i in range_constexpr(gs):
                col = group * fx.Int32(gs) + fx.Int32(i)
                idx = fx.Int64(row) * fx.Int64(k) + fx.Int64(col)
                v = _load_w(wbuf, idx)
                codes.append(_nearest_level(v / sch, levels))
        elif const_expr(mode == 1):
            for i in range_constexpr(gs):
                col = group * fx.Int32(gs) + fx.Int32(i)
                idx = fx.Int64(row) * fx.Int64(k) + fx.Int64(col)
                v = _load_w(wbuf, idx)
                signed = fmath.roundeven(v / scale_eff)
                signed = fx.max(fx.min(signed, fx.Float32(7.0)), fx.Float32(-8.0))
                codes.append((signed + fx.Float32(8.0)).to(fx.Int32))
        else:
            for i in range_constexpr(gs):
                col = group * fx.Int32(gs) + fx.Int32(i)
                idx = fx.Int64(row) * fx.Int64(k) + fx.Int64(col)
                v = _load_w(wbuf, idx)
                code = fmath.roundeven((v - amin) / scale_eff)
                codes.append(fx.max(fx.min(code, fx.Float32(15.0)), fx.Float32(0.0)).to(fx.Int32))
        if inb:
            rbuf = ptr_buf_tensor(
                SRel, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_tot) * fx.Int64(4)
            )
            pbuf = ptr_buf_tensor(
                Packed,
                elem=fx.Int8,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(n_rows) * fx.Int64(k) // fx.Int64(2),
            )
            buf_copy_store(rbuf, fx.Int64(safe_g), srel, elem=fx.Float32, unit_elems=1)
            half = gs // 2
            base_byte = fx.Int64(row) * fx.Int64(k) // fx.Int64(2) + fx.Int64(group) * fx.Int64(half)
            for i in range_constexpr(half):
                lo = codes[i * 2] & fx.Int32(15)
                hi = codes[i * 2 + 1] & fx.Int32(15)
                byte = (lo | (hi << fx.Int32(4))).to(fx.Int8)
                buf_copy_store(pbuf, base_byte + fx.Int64(i), byte, elem=fx.Int8, unit_elems=1)
            if const_expr(mode == 2):
                corr = fx.Float32(8.0) * scale_eff + amin
                # correction layout is [groups, N]
                cidx = fx.Int64(group) * fx.Int64(n_rows) + fx.Int64(row)
                ob = ptr_buf_tensor(
                    Corr,
                    elem=fx.Float32,
                    n=0x3FFFFFFF,
                    unit_elems=1,
                    num_records_bytes=fx.Int64(n_groups) * fx.Int64(n_rows) * fx.Int64(4),
                )
                buf_copy_store(ob, cidx, corr, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Weight: fx.Tensor,
        Scale: fx.Tensor,
        Gmax: fx.Tensor,
        Amin: fx.Tensor,
        Codebook: fx.Tensor,
        Packed: fx.Tensor,
        SRel: fx.Tensor,
        SChan: fx.Tensor,
        Corr: fx.Tensor,
        n_rows: fx.Int32,
        n_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n_tot = fx.Int64(n_rows) * fx.Int64(n_groups)
        grid = (n_tot + fx.Int64(block - 1)) // fx.Int64(block)
        group_stats(Weight, Scale, Gmax, Amin, Codebook, n_rows, n_groups).launch(
            grid=(grid, 1, 1), block=(block, 1, 1), stream=stream
        )
        row_max(Gmax, SChan, n_rows, n_groups).launch(grid=(n_rows, 1, 1), block=(block, 1, 1), stream=stream)
        finalize(Weight, Scale, SChan, Amin, Codebook, Packed, SRel, Corr, n_rows, n_groups).launch(
            grid=(grid, 1, 1), block=(block, 1, 1), stream=stream
        )

    return launch


def _quantize_rotated_w4a8(
    weight: torch.Tensor,
    group_size: int = 16,
    symmetric: bool = True,
    scale_dtype: object = None,
    codebook: bool = True,
    codebook_override: object = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device W4A8 pack. Codebook path uses the frozen 16-level LUT plus ALS."""
    import torch

    if scale_dtype is None:
        scale_dtype = torch.float8_e4m3fn
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")

    weight_name = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(weight.dtype)
    if weight_name is None:
        raise ValueError(f"weight must be f32/f16/bf16, got {weight.dtype}")
    n, k = weight.shape
    groups = k // group_size
    if symmetric and codebook:
        mode = 0
    elif symmetric:
        mode = 1
    else:
        mode = 2
    w = weight if weight.is_contiguous() else weight.contiguous()
    if codebook_override is not None:
        if codebook_override.dtype != torch.float32:
            raise ValueError(f"codebook must be float32, got {codebook_override.dtype}")
        if codebook_override.device != w.device:
            raise ValueError("codebook device does not match the weight")
        lut = codebook_override.reshape(16)
        if not lut.is_contiguous():
            lut = lut.contiguous()
        if lut.numel() != 16:
            raise ValueError(f"codebook must have 16 entries, got {lut.numel()}")
    else:
        lut = torch.tensor(_FIXED_LUT, device=w.device, dtype=torch.float32)
    packed = torch.empty((n, k // 2), device=w.device, dtype=torch.int8)
    s_rel = torch.empty((n, groups), device=w.device, dtype=torch.float32)
    s_channel = torch.empty((n,), device=w.device, dtype=torch.float32)
    scale = torch.empty((n, groups), device=w.device, dtype=torch.float32)
    gmax = torch.empty((n, groups), device=w.device, dtype=torch.float32)
    amin = torch.empty((n, groups), device=w.device, dtype=torch.float32)
    corr = torch.empty((groups, n), device=w.device, dtype=torch.float32)
    launch = build_w4a8_pack_module(group_size, mode, weight_name, srel_fp8=scale_dtype != torch.float32)
    launch(w, scale, gmax, amin, lut, packed, s_rel, s_channel, corr, n, groups)
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    correction = corr.to(weight.dtype) if mode == 2 else None
    codebook_tensor = lut if mode == 0 else None
    return packed, s_rel, s_channel, correction, codebook_tensor


def quantize_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: object = None,
    codebook: bool = True,
    codebook_tensor: object = None,
    stochastic_rounding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate on device, then pack W4A8 with the device quant kernel.

    The codebook path uses the frozen Gaussian LUT and two ALS updates.
    Stochastic rounding is rejected.
    """
    import torch

    require_gfx120x(weight.device, what="quantize_w4a8_int8_weight (gfx120x)")
    if scale_dtype is None:
        scale_dtype = torch.float8_e4m3fn
    if stochastic_rounding:
        raise ValueError("quantize_w4a8_int8_weight FlyDSL host path: no stochastic_rounding")
    if convrot_groupsize not in _SUPPORTED_CONVROT:
        raise ValueError(f"convrot_groupsize must be one of {_SUPPORTED_CONVROT}")
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize)
    rotated = rotate_convrot_weight(weight.contiguous(), convrot_groupsize)
    return _quantize_rotated_w4a8(
        rotated,
        group_size=group_size,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_override=codebook_tensor,
    )


def dequant_int4_grouped_to_int8(
    qdata: object,
    s_rel: object,
    codebook: object = None,
    group_size: int = 16,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """FlyDSL decode of packed W4A8 → INT8 GEMM grid."""
    require_gfx120x(qdata.device, what="dequant_int4_grouped_to_int8 (gfx120x)")
    import torch

    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("qdata must be 2D int8")
    if qdata.device.type != "cuda":
        raise ValueError("qdata must be on CUDA/ROCm")
    n, k_half = qdata.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    groups = k // group_size
    if s_rel.dtype == torch.float32:
        srel_fp8 = False
    elif s_rel.dtype == torch.float8_e4m3fn:
        srel_fp8 = True
    else:
        raise ValueError(f"s_rel must be float32 or float8_e4m3fn, got {s_rel.dtype}")
    if s_rel.device != qdata.device:
        raise ValueError("s_rel device does not match qdata")
    if tuple(s_rel.shape) != (n, groups):
        raise ValueError(f"s_rel must have shape {(n, groups)}, got {tuple(s_rel.shape)}")
    s = s_rel if s_rel.is_contiguous() else s_rel.contiguous()
    use_cb = codebook is not None
    if use_cb:
        if codebook.dtype != torch.float32:
            raise ValueError(f"codebook must be float32, got {codebook.dtype}")
        if codebook.device != qdata.device:
            raise ValueError("codebook device does not match qdata")
        cb = codebook.reshape(16)
        if not cb.is_contiguous():
            cb = cb.contiguous()
        if cb.numel() != 16:
            raise ValueError(f"codebook must have 16 entries, got {cb.numel()}")
    else:
        cb = torch.zeros(16, device=qdata.device, dtype=torch.float32)

    out = torch.empty((n, k), device=qdata.device, dtype=torch.int8)
    q = qdata if qdata.is_contiguous() else qdata.contiguous()
    launch = build_w4a8_dequant_int4_to_int8_module(k=k, group_size=group_size, use_codebook=use_cb, srel_fp8=srel_fp8)
    if stream is None:
        launch(q, s, cb, out, n, k)
    else:
        launch(q, s, cb, out, n, k, stream)
    return out


# Bump to invalidate reconstruct autotune winner caches after a codegen change.
_W4A8_RECONSTRUCT_TUNING_SCHEMA = 1
_W4A8_RECONSTRUCT_BLOCKS = (32, 64, 128, 256, 512, 1024)


def _default_reconstruct_block(convrot_groupsize: int) -> int:
    """Smallest power of two >= 32 that covers the ConvRot group, capped at 256.

    Group 16 -> 32, 64 -> 64, 256 -> 256. Legal for every supported tile.
    """
    block = 32
    width = int(convrot_groupsize)
    while block < width and block < 256:
        block *= 2
    return block


def _check_w4a8_reconstruct_shape(k: int, group_size: int, convrot_groupsize: int) -> None:
    """Legal when K splits into both groups and the quant group nests in 16."""
    group_ok = group_size >= 4 and (16 % group_size == 0 or group_size % 16 == 0)
    splits = group_ok and convrot_groupsize in _SUPPORTED_CONVROT and k % group_size == 0 and k % convrot_groupsize == 0
    if k <= 0 or k % 2 != 0 or not splits:
        raise ValueError(
            f"K={k} must be divisible by group_size={group_size} and "
            f"convrot_groupsize={convrot_groupsize} in {_SUPPORTED_CONVROT}; "
            "group_size must be >= 4 and (16 % group_size == 0 or group_size % 16 == 0)"
        )


def _layout_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Layout copy only. Does not change dtype."""
    return tensor if tensor.is_contiguous() else tensor.contiguous()


@lru_cache(maxsize=32)
def build_w4a8_reconstruct_module(
    k: int,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    use_codebook: bool = True,
    has_correction: bool = False,
    out_dtype: str = "bfloat16",
    srel_dtype: str = "float32",
    corr_dtype: str = "none",
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Packed int4 → original-basis weight. Direct JIT, autotuned BLOCK.

    No host cast of scales: ``s_rel`` is loaded as ``float32`` or
    ``float8_e4m3fn``, and ``correction`` as ``float32`` / ``float16`` /
    ``bfloat16`` (or not at all). Legal for every K divisible by
    ``group_size`` and ``convrot_groupsize`` in {16, 64, 256}, with
    ``group_size >= 4`` and (``16 % group_size == 0`` or
    ``group_size % 16 == 0``).

    The int8-grid rounding stays in registers (``round(centered * s_rel)``
    clamped to [-127, 127], times ``s_channel``, plus the per-group
    correction). One ConvRot group is staged in LDS as f32 for the un-rotate.
    Nothing writes an int8 matrix or an f32 matrix.
    """
    import math

    _check_w4a8_reconstruct_shape(k, group_size, convrot_groupsize)
    out_table = {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }
    if out_dtype not in out_table:
        raise ValueError(f"unsupported out dtype {out_dtype}")
    if srel_dtype not in ("float32", "float8_e4m3fn"):
        raise ValueError(f"unsupported s_rel dtype {srel_dtype}")
    corr_table = {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }
    if has_correction:
        if corr_dtype not in corr_table:
            raise ValueError(f"unsupported correction dtype {corr_dtype}")
        corr_ty, corr_bytes = corr_table[corr_dtype]
    else:
        if corr_dtype != "none":
            raise ValueError(f"corr_dtype must be 'none' when correction is absent, got {corr_dtype}")
        corr_ty, corr_bytes = fx.Float32, 4
    out_ty, out_bytes = out_table[out_dtype]
    if block_threads is None:
        block_threads = _default_reconstruct_block(convrot_groupsize)
    block = int(block_threads)
    if block < WARP or block > 1024 or block % WARP != 0:
        raise ValueError(f"block_threads={block} must be a wave32 block in [32, 1024]")

    srel_is_fp8 = srel_dtype == "float8_e4m3fn"
    srel_elem = fx.Uint8 if srel_is_fp8 else fx.Float32
    srel_bytes = 1 if srel_is_fp8 else 4
    packed_k = k // 2
    quant_groups = k // group_size
    n_cgroups = k // convrot_groupsize
    n_stages = int(math.log(convrot_groupsize, 4))
    g_butterflies = convrot_groupsize // 4
    g_bf_steps = (g_butterflies + block - 1) // block
    g_load_steps = (convrot_groupsize + block - 1) // block
    inv_sqrt_g = 1.0 / math.sqrt(float(convrot_groupsize))
    sig = _kernel_signature(
        block=block,
        group_size=group_size,
        convrot=convrot_groupsize,
        k=k,
        codebook=use_codebook,
        correction=has_correction,
        dtype=out_dtype,
        srel=srel_dtype,
        corr=corr_dtype,
        op="w4a8_reconstruct",
    )

    @fx.struct
    class SharedStorage:
        codebook: fx.Array[fx.Float32, 16]
        tile: fx.Array[fx.Float32, convrot_groupsize]

    @flyc.kernel(known_block_size=[block, 1, 1])
    def reconstruct_kernel(
        Qdata: fx.Tensor,
        SRel: fx.Tensor,
        SChan: fx.Tensor,
        Codebook: fx.Tensor,
        Corr: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        active_row = bid < n_rows

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        cb_view = lds.codebook.view(fx.make_layout(16, 1))
        row_view = lds.tile.view(fx.make_layout(convrot_groupsize, 1))

        q_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            SRel,
            elem=srel_elem,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(quant_groups) * fx.Int64(srel_bytes),
        )
        sc_buf = ptr_buf_tensor(
            SChan,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(4),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=out_ty,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(k) * fx.Int64(out_bytes),
        )
        if const_expr(use_codebook):
            cb_buf = ptr_buf_tensor(
                Codebook,
                elem=fx.Float32,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(64),
            )
            if tid < fx.Int32(16):
                cb_val = fx.Float32(buf_copy_load(cb_buf, fx.Int64(tid), elem=fx.Float32, unit_elems=1))
                fx.memref_store(cb_val, cb_view, tid)
            gpu.barrier()
        if const_expr(has_correction):
            cbuf = ptr_buf_tensor(
                Corr,
                elem=corr_ty,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(quant_groups) * fx.Int64(n_rows) * fx.Int64(corr_bytes),
            )

        row_i = active_row.select(bid, fx.Int32(0))
        sch = fx.Float32(buf_copy_load(sc_buf, fx.Int64(row_i), elem=fx.Float32, unit_elems=1))
        sch = active_row.select(sch, c0)

        def lds_load(idx: object) -> object:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(convrot_groupsize))
            off = safe.select(idx, fx.Int32(0))
            loaded = fx.Float32(row_view[off])
            return safe.select(loaded, c0)

        def lds_store(idx: object, val: object) -> None:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(convrot_groupsize))
            off = safe.select(idx, fx.Int32(0))
            fx.memref_store(fx.Float32(val), row_view, off)

        def fwht_tile() -> None:
            # Radix-4 un-rotate. Same signs as the previous kernel (H4 kronecker):
            # (a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d). Scale is 1/sqrt(group).
            for stage in range_constexpr(n_stages):
                stride = 4**stage
                span = 4 * stride
                for step in range_constexpr(g_bf_steps):
                    bf = tid + fx.Int32(step * block)
                    inb = active_row & (bf < fx.Int32(g_butterflies))
                    base_idx = (bf // fx.Int32(stride)) * fx.Int32(span)
                    i = bf % fx.Int32(stride)
                    idx0 = base_idx + i
                    idx1 = idx0 + fx.Int32(stride)
                    idx2 = idx0 + fx.Int32(2 * stride)
                    idx3 = idx0 + fx.Int32(3 * stride)
                    a = lds_load(idx0)
                    b = lds_load(idx1)
                    c = lds_load(idx2)
                    d = lds_load(idx3)
                    if inb:
                        lds_store(idx0, a + b + c - d)
                        lds_store(idx1, a + b - c + d)
                        lds_store(idx2, a - b + c + d)
                        lds_store(idx3, -a + b + c + d)
                gpu.barrier()

        # Rotated basis, one ConvRot group at a time. Threads stride by BLOCK.
        for cgroup in range_constexpr(n_cgroups):
            base = fx.Int32(cgroup) * fx.Int32(convrot_groupsize)
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block)
                col = base + local
                inb = active_row & (local < fx.Int32(convrot_groupsize))
                pc = col // fx.Int32(2)
                byte_i = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(pc)
                safe_b = inb.select(byte_i, fx.Int64(0))
                raw = buf_copy_load(q_buf, safe_b, elem=fx.Int8, unit_elems=1)
                packed = fx.Int32(raw) & fx.Int32(0xFF)
                lo = packed & fx.Int32(0xF)
                hi = (packed >> fx.Int32(4)) & fx.Int32(0xF)
                is_hi = (col & fx.Int32(1)) == fx.Int32(1)
                # Unsigned index in [0, 15]. Non-codebook centering is nibble-8
                # (stored signed+8), not a 4-bit sign-extend (that would leave 0 as 0).
                nibble = is_hi.select(hi, lo)
                if const_expr(use_codebook):
                    centered = fx.Float32(cb_view[nibble])
                else:
                    centered = fx.Float32(nibble) - fx.Float32(8.0)
                qg = col // fx.Int32(group_size)
                sidx = fx.Int64(bid) * fx.Int64(quant_groups) + fx.Int64(qg)
                safe_s = inb.select(sidx, fx.Int64(0))
                raw_s = buf_copy_load(s_buf, safe_s, elem=srel_elem, unit_elems=1)
                if const_expr(srel_is_fp8):
                    bits = fx.Int32(raw_s) & fx.Int32(0xFF)
                    # Same ROCDL cvt as rdna4_fp8_quant / rdna4_w8a16_linear.
                    # Byte sits in bits[7:0]; selector 0 reads that byte.
                    srel = fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, bits.ir_value(), 0))
                else:
                    srel = fx.Float32(raw_s)
                q = fmath.roundeven(centered * srel)
                q = fx.max(fx.min(q, fx.Float32(127.0)), fx.Float32(-127.0))
                val = q * sch
                if const_expr(has_correction):
                    cidx = fx.Int64(qg) * fx.Int64(n_rows) + fx.Int64(bid)
                    raw_c = buf_copy_load(cbuf, inb.select(cidx, fx.Int64(0)), elem=corr_ty, unit_elems=1)
                    corr = corr_ty(raw_c).to(fx.Float32)
                    val = val + inb.select(corr, c0)
                val = inb.select(val, c0)
                if local < fx.Int32(convrot_groupsize):
                    fx.memref_store(val, row_view, local)
            gpu.barrier()
            fwht_tile()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block)
                if local < fx.Int32(convrot_groupsize):
                    scaled = lds_load(local) * fx.Float32(inv_sqrt_g)
                    fx.memref_store(scaled, row_view, local)
            gpu.barrier()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block)
                col = base + local
                inb = active_row & (local < fx.Int32(convrot_groupsize))
                v = lds_load(local)
                if const_expr(out_dtype != "float32"):
                    stored = fx.Float32(v).to(out_ty)
                else:
                    stored = v
                gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                if inb:
                    buf_copy_store(out_buf, gidx, stored, elem=out_ty, unit_elems=1)
            gpu.barrier()

    reconstruct_kernel.__name__ = f"w4a8_reconstruct_gfx120x_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Tensor,
        SRel: fx.Tensor,
        SChan: fx.Tensor,
        Codebook: fx.Tensor,
        Corr: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        reconstruct_kernel(Qdata, SRel, SChan, Codebook, Corr, Out, n_rows).launch(
            grid=(n_rows, 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_w4a8_reconstruct_gfx120x_{sig}"
    return launch


@flyc.jit
def w4a8_reconstruct_direct(
    Qdata: fx.Tensor,
    SRel: fx.Tensor,
    SChan: fx.Tensor,
    Codebook: fx.Tensor,
    Corr: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    K: fx.Constexpr[int],
    group_size: fx.Constexpr[int],
    convrot_groupsize: fx.Constexpr[int],
    use_codebook: fx.Constexpr[int],
    has_correction: fx.Constexpr[int],
    out_dtype: fx.Constexpr[str],
    srel_dtype: fx.Constexpr[str],
    corr_dtype: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Specialize reconstruct through JIT Constexpr inputs.

    ``tuning_schema`` is not read by the kernel. It is a declared autotune key
    axis, so bumping it partitions the winner cache and forces a fresh search.
    """
    launch = build_w4a8_reconstruct_module(
        k=int(K),
        group_size=int(group_size),
        convrot_groupsize=int(convrot_groupsize),
        use_codebook=bool(int(use_codebook)),
        has_correction=bool(int(has_correction)),
        out_dtype=str(out_dtype),
        srel_dtype=str(srel_dtype),
        corr_dtype=str(corr_dtype),
        block_threads=int(BLOCK),
    )
    launch(Qdata, SRel, SChan, Codebook, Corr, Out, n_rows, stream)


@contextmanager
def _validate_w4a8_reconstruct(sig_args):
    """Poison the output, then reject a candidate that left a non-finite value."""
    import torch

    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("w4a8 reconstruct candidate left a non-finite output")


def _default_w4a8_reconstruct_config(*args, **kwargs):
    group = kwargs.get("convrot_groupsize")
    if group is None:
        group = args[9]
    return Config(BLOCK=_default_reconstruct_block(int(group)))


_w4a8_reconstruct = autotune(
    configs=[Config(BLOCK=block) for block in _W4A8_RECONSTRUCT_BLOCKS],
    key=[
        "K",
        "group_size",
        "convrot_groupsize",
        "use_codebook",
        "has_correction",
        "out_dtype",
        "srel_dtype",
        "corr_dtype",
        "tuning_schema",
    ],
    default=_default_w4a8_reconstruct_config,
    artifact_name="w4a8_reconstruct_gfx120x",
    validate_hook=_validate_w4a8_reconstruct,
)(w4a8_reconstruct_direct)


def dequantize_w4a8_int8_weight(
    qdata: object,
    s_rel: object,
    s_channel: object,
    codebook: object = None,
    correction: object = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: object = None,
) -> torch.Tensor:
    """Original-basis W4A8 weight. Direct JIT, autotuned BLOCK.

    No host cast of scales (``s_rel``, ``s_channel``, ``codebook``, and
    ``correction`` keep their wire dtypes). Legal for every K divisible by
    ``group_size`` and ``convrot_groupsize`` in {16, 64, 256}, with
    ``group_size >= 4`` and (``16 % group_size == 0`` or
    ``group_size % 16 == 0``).
    """
    require_gfx120x(qdata.device, what="dequantize_w4a8_int8_weight (gfx120x)")

    if output_dtype is None:
        output_dtype = torch.bfloat16
    out_name = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(output_dtype)
    if out_name is None:
        raise ValueError(f"unsupported output_dtype {output_dtype}")
    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("qdata must be 2D int8")
    if qdata.device.type != "cuda":
        raise ValueError("qdata must be on CUDA/ROCm")
    n, k_half = qdata.shape
    k = k_half * 2
    _check_w4a8_reconstruct_shape(k, group_size, convrot_groupsize)
    groups = k // group_size
    q = _layout_contiguous(qdata)

    if s_rel.dtype == torch.float32:
        srel_dtype = "float32"
    elif s_rel.dtype == torch.float8_e4m3fn:
        srel_dtype = "float8_e4m3fn"
    else:
        raise ValueError(f"s_rel must be float32 or float8_e4m3fn, got {s_rel.dtype}")
    if s_rel.device != qdata.device:
        raise ValueError(f"s_rel device {s_rel.device} does not match qdata device {qdata.device}")
    if tuple(s_rel.shape) != (n, groups):
        raise ValueError(f"s_rel must have shape {(n, groups)}, got {tuple(s_rel.shape)}")
    s = _layout_contiguous(s_rel)

    if s_channel.dtype != torch.float32:
        raise ValueError(f"s_channel must be float32 (wire format), got {s_channel.dtype}")
    if s_channel.device != qdata.device:
        raise ValueError(f"s_channel device {s_channel.device} does not match qdata device {qdata.device}")
    sc = s_channel.reshape(-1)
    if sc.numel() != n:
        raise ValueError(f"s_channel must have {n} values, got {sc.numel()}")
    sc = _layout_contiguous(sc)

    use_cb = 0 if codebook is None else 1
    if use_cb:
        if codebook.dtype != torch.float32:
            raise ValueError(f"codebook must be float32 (wire format), got {codebook.dtype}")
        if codebook.device != qdata.device:
            raise ValueError(f"codebook device {codebook.device} does not match qdata device {qdata.device}")
        if codebook.numel() != 16:
            raise ValueError(f"codebook must have 16 entries, got {codebook.numel()}")
        cb = _layout_contiguous(codebook.reshape(16))
    else:
        cb = torch.zeros(16, device=qdata.device, dtype=torch.float32)

    has_corr = 0 if correction is None else 1
    if has_corr:
        corr_dtype = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(
            correction.dtype
        )
        if corr_dtype is None:
            raise ValueError(f"correction must be float32, float16, or bfloat16, got {correction.dtype}")
        if correction.device != qdata.device:
            raise ValueError(f"correction device {correction.device} does not match qdata device {qdata.device}")
        if tuple(correction.shape) != (groups, n):
            raise ValueError(f"correction must have shape {(groups, n)}, got {tuple(correction.shape)}")
        corr = _layout_contiguous(correction)
    else:
        corr_dtype = "none"
        corr = torch.empty(1, device=qdata.device, dtype=torch.float32)

    out = torch.empty((n, k), device=qdata.device, dtype=output_dtype)
    _w4a8_reconstruct(
        q,
        s,
        sc,
        cb,
        corr,
        out,
        n,
        K=k,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        use_codebook=use_cb,
        has_correction=has_corr,
        out_dtype=out_name,
        srel_dtype=srel_dtype,
        corr_dtype=corr_dtype,
        tuning_schema=_W4A8_RECONSTRUCT_TUNING_SCHEMA,
    )
    return out


def w4a8_int8_linear(
    x: object,
    qdata: object,
    s_rel: object,
    s_channel: object,
    codebook: object = None,
    correction: object = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """``x @ W.T + bias`` via INT4→INT8 decode + ConvRot INT8 linear.

    Requires gfx120x iu8 WMMA (``rdna4_int8_linear``). Asymmetric ``correction``
    is not an int8 product: ``dequantize_w4a8_int8_weight`` reconstructs the
    original-basis weight in one kernel and ``gemm_bf16_nmajor_lds`` multiplies.
    """
    require_gfx120x(x.device, what="w4a8_int8_linear (gfx120x)")
    import torch

    from kernels.quant.rdna4_int8_convrot import int8_linear_convrot

    if out_dtype is None:
        out_dtype = x.dtype
    if x.shape[-1] != qdata.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={qdata.shape[-1] * 2}")
    if correction is not None:
        from kernels.gemm.rdna4_fused_mlp_nmajor import gemm_bf16_nmajor_lds

        weight = dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=torch.bfloat16,
        )
        x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
        y = gemm_bf16_nmajor_lds(x2, weight.contiguous(), out_dtype=torch.float32)
        if bias is not None:
            from kernels.common.gfx120x_row_bias import add_row_bias

            y = add_row_bias(y, bias, out_dtype=torch.float32, stream=stream)
        return y.reshape(*x.shape[:-1], weight.shape[0]).to(out_dtype)

    int8_w = dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size=group_size, stream=stream)
    return int8_linear_convrot(
        x,
        int8_w,
        s_channel,
        group_size=convrot_groupsize,
        bias=bias,
        out_dtype=out_dtype,
        stream=stream,
    )
