# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Per-1x32 MXFP8 E4M3 quant and dequant for gfx120x.

Scale bytes are E8M0 with the same round-to-nearest-even exponent rule as
``kernels.monokernel.formats.float_to_e8m0``. Values are packed with the
hardware ``cvt_pk_fp8_f32`` so the bytes match a torch fp8 cast. Launches are
direct ``fx.Tensor`` JIT calls. Block size is autotuned; a grid-stride loop
with a tail mask covers every group count. A K that is not a multiple of 32
is zero-filled inside the quant kernel and is not cloned up to 32.
"""

from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)
_DEFAULT_BLOCK = 256
TUNING_SCHEMA = 1
_FP8_MAX = 448.0
_TINY = 1.1754943508222875e-38
_FP8_NAN = 0x7F


def _stride_blocks(n_items: int) -> tuple[int, ...]:
    """Block sizes whose grid-stride loop can cover ``n_items``.

    Partial last tiles stay masked: the loop stops at ``n_items`` and stores
    are predicated with ``item < n_items``.
    """
    if n_items < 0:
        return ()
    return tuple(block for block in _BLOCK_CHOICES if 0 < block <= 1024 and block % 32 == 0)


def _e8m0_bits(value):
    """Positive f32 -> E8M0 exponent byte. NaN/Inf stay 0xFF (no wrap)."""
    bits = value.bitcast(fx.Int32)
    exponent = (bits >> fx.Int32(23)) & fx.Int32(0xFF)
    is_special = exponent == fx.Int32(0xFF)
    half = (bits & fx.Int32(0x400000)) != fx.Int32(0)
    sticky = ((bits & fx.Int32(0x200000)) != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
    low = ((bits & fx.Int32(0x1FFFFF)) != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
    nonzero = (exponent != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
    sticky = ((sticky | low) | nonzero) != fx.Int32(0)
    bump = half.select(sticky.select(fx.Int32(1), fx.Int32(0)), fx.Int32(0))
    bumped = exponent + bump
    saturated = (bumped > fx.Int32(0xFF)).select(fx.Int32(0xFF), bumped)
    return is_special.select(fx.Int32(0xFF), saturated)


def _e8m0_to_f32(exponent):
    bits = exponent << fx.Int32(23)
    bits = (exponent == fx.Int32(0)).select(fx.Int32(0x00400000), bits)
    bits = (exponent == fx.Int32(0xFF)).select(fx.Int32(0x7F800001), bits)
    return bits.bitcast(fx.Float32)


@lru_cache(maxsize=16)
def build_mxfp8_quant_module(in_name: str, block: int = _DEFAULT_BLOCK):
    in_ty, in_bytes = {"float32": (fx.Float32, 4), "bfloat16": (fx.BFloat16, 2), "float16": (fx.Float16, 2)}[in_name]
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP8 quant block {block} is not executable by the stride loop")
    cvt_pk = fx.rocdl.cvt_pk_fp8_f32

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quant_kernel(X: fx.Tensor, Q: fx.Tensor, Scale: fx.Tensor, n_blocks: fx.Int32) -> None:
        xbuf = ptr_buf_tensor(
            X, elem=in_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(32 * in_bytes)
        )
        qbuf = ptr_buf_tensor(
            Q, elem=fx.Int32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(32)
        )
        sbuf = ptr_buf_tensor(Scale, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks))
        start = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        step = fx.grid_dim.x * fx.Int32(block)
        for bid in range(start, n_blocks, step):
            inb = bid < n_blocks
            safe = inb.select(bid, fx.Int32(0))
            vals = []
            amax = fx.Float32(0.0)
            for i in range_constexpr(32):
                raw = buf_copy_load(xbuf, fx.Int64(safe) * fx.Int64(32) + fx.Int64(i), elem=in_ty, unit_elems=1)
                if const_expr(in_name == "float32"):
                    v = fx.Float32(raw)
                else:
                    v = raw.to(fx.Float32)
                vals.append(v)
                amax = fx.max(amax, fmath.absf(v))
            exp = _e8m0_bits(amax / fx.Float32(_FP8_MAX))
            scale = fx.max(_e8m0_to_f32(exp), fx.Float32(_TINY))
            inv = fx.Float32(1.0) / scale
            words = []
            for w in range_constexpr(8):
                i0 = w * 4
                a = fx.max(fx.min(vals[i0] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                b = fx.max(fx.min(vals[i0 + 1] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                c = fx.max(fx.min(vals[i0 + 2] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                d = fx.max(fx.min(vals[i0 + 3] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                acc = fx.Int32(0).ir_value()
                acc = cvt_pk(T.i32, a.ir_value(), b.ir_value(), acc, 0)
                acc = cvt_pk(T.i32, c.ir_value(), d.ir_value(), acc, 1)
                words.append(fx.Int32(acc))
            if inb:
                buf_copy_store(sbuf, fx.Int64(safe), exp.to(fx.Uint8), elem=fx.Uint8, unit_elems=1)
                base = fx.Int64(safe) * fx.Int64(8)
                for w in range_constexpr(8):
                    buf_copy_store(qbuf, base + fx.Int64(w), words[w], elem=fx.Int32, unit_elems=1)

    @flyc.jit
    def launch(
        X: fx.Tensor, Q: fx.Tensor, Scale: fx.Tensor, n_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        quant_kernel(X, Q, Scale, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=8)
def build_mxfp8_dequant_module(clamp_tiny: bool, block: int = _DEFAULT_BLOCK):
    """One thread per 32-block. ``clamp_tiny`` matches ``quant_dequant_mxfp8``."""
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP8 dequant block {block} is not executable by the stride loop")
    cvt = fx.rocdl.cvt_f32_fp8

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dequant_kernel(Q: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_blocks: fx.Int32) -> None:
        qbuf = ptr_buf_tensor(
            Q, elem=fx.Int32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(32)
        )
        sbuf = ptr_buf_tensor(Scale, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks))
        obuf = ptr_buf_tensor(
            Out, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(128)
        )
        start = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        step = fx.grid_dim.x * fx.Int32(block)
        for bid in range(start, n_blocks, step):
            inb = bid < n_blocks
            safe = inb.select(bid, fx.Int32(0))
            exp = fx.Int32(buf_copy_load(sbuf, fx.Int64(safe), elem=fx.Uint8, unit_elems=1))
            scale = _e8m0_to_f32(exp)
            if const_expr(clamp_tiny):
                scale = fx.max(scale, fx.Float32(_TINY))
            base_w = fx.Int64(safe) * fx.Int64(8)
            base_o = fx.Int64(safe) * fx.Int64(32)
            for w in range_constexpr(8):
                packed = buf_copy_load(qbuf, base_w + fx.Int64(w), elem=fx.Int32, unit_elems=1)
                for bi in range_constexpr(4):
                    value = fx.Float32(cvt(T.f32, packed.ir_value(), bi)) * scale
                    if inb:
                        buf_copy_store(obuf, base_o + fx.Int64(w * 4 + bi), value, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Q: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        dequant_kernel(Q, Scale, Out, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=16)
def build_mxfp8_quant_tail_module(in_name: str, block: int, logical_k: int):
    """Same quant as the aligned builder, for a K that is not a multiple of 32.

    Each group still owns 32 lanes. Lanes past the caller's K are zero and are
    not loaded. Packed FP8 bytes are stored one at a time so a short row does
    not spill into the next row. The aligned builder is a separate kernel.
    ``logical_k`` is baked in so the launch arguments stay the aligned ones.
    """
    in_ty, in_bytes = {"float32": (fx.Float32, 4), "bfloat16": (fx.BFloat16, 2), "float16": (fx.Float16, 2)}[in_name]
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP8 quant block {block} is not executable by the stride loop")
    if logical_k <= 0 or logical_k % 32 == 0:
        raise ValueError(f"MXFP8 quant tail logical_k={logical_k} must be a positive non-multiple of 32")
    groups = (logical_k + 31) // 32
    cvt_pk = fx.rocdl.cvt_pk_fp8_f32

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quant_tail_kernel(X: fx.Tensor, Q: fx.Tensor, Scale: fx.Tensor, n_blocks: fx.Int32) -> None:
        rows64 = fx.Int64(n_blocks) // fx.Int64(groups)
        xbuf = ptr_buf_tensor(
            X,
            elem=in_ty,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=rows64 * fx.Int64(logical_k) * fx.Int64(in_bytes),
        )
        qbuf = ptr_buf_tensor(
            Q, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=rows64 * fx.Int64(logical_k)
        )
        sbuf = ptr_buf_tensor(Scale, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks))
        start = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        step = fx.grid_dim.x * fx.Int32(block)
        for bid in range(start, n_blocks, step):
            inb = bid < n_blocks
            safe = inb.select(bid, fx.Int32(0))
            row = safe // groups
            grp = safe - row * groups
            col0 = grp * fx.Int32(32)
            vals = []
            amax = fx.Float32(0.0)
            for i in range_constexpr(32):
                col = col0 + fx.Int32(i)
                v = fx.Float32(0.0)
                if col < logical_k:
                    idx = fx.Int64(row) * fx.Int64(logical_k) + fx.Int64(col)
                    raw = buf_copy_load(xbuf, idx, elem=in_ty, unit_elems=1)
                    if const_expr(in_name == "float32"):
                        v = fx.Float32(raw)
                    else:
                        v = raw.to(fx.Float32)
                vals.append(v)
                amax = fx.max(amax, fmath.absf(v))
            exp = _e8m0_bits(amax / fx.Float32(_FP8_MAX))
            scale = fx.max(_e8m0_to_f32(exp), fx.Float32(_TINY))
            inv = fx.Float32(1.0) / scale
            words = []
            for w in range_constexpr(8):
                i0 = w * 4
                a = fx.max(fx.min(vals[i0] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                b = fx.max(fx.min(vals[i0 + 1] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                c = fx.max(fx.min(vals[i0 + 2] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                d = fx.max(fx.min(vals[i0 + 3] * inv, fx.Float32(_FP8_MAX)), fx.Float32(-_FP8_MAX))
                acc = fx.Int32(0).ir_value()
                acc = cvt_pk(T.i32, a.ir_value(), b.ir_value(), acc, 0)
                acc = cvt_pk(T.i32, c.ir_value(), d.ir_value(), acc, 1)
                words.append(fx.Int32(acc))
            if inb:
                buf_copy_store(sbuf, fx.Int64(safe), exp.to(fx.Uint8), elem=fx.Uint8, unit_elems=1)
                for w in range_constexpr(8):
                    word = words[w]
                    for b in range_constexpr(4):
                        col = col0 + fx.Int32(w * 4 + b)
                        if col < logical_k:
                            byte = (word >> fx.Int32(8 * b)) & fx.Int32(255)
                            idx = fx.Int64(row) * fx.Int64(logical_k) + fx.Int64(col)
                            buf_copy_store(qbuf, idx, byte.to(fx.Uint8), elem=fx.Uint8, unit_elems=1)

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Q: fx.Tensor,
        Scale: fx.Tensor,
        n_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        quant_tail_kernel(X, Q, Scale, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=8)
def build_mxfp8_dequant_tail_module(clamp_tiny: bool, block: int, logical_k: int):
    """Decode a short last MX group without reading past the caller's K."""
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP8 dequant block {block} is not executable by the stride loop")
    if logical_k <= 0 or logical_k % 32 == 0:
        raise ValueError(f"MXFP8 dequant tail logical_k={logical_k} must be a positive non-multiple of 32")
    groups = (logical_k + 31) // 32
    cvt = fx.rocdl.cvt_f32_fp8

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dequant_tail_kernel(Q: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_blocks: fx.Int32) -> None:
        rows64 = fx.Int64(n_blocks) // fx.Int64(groups)
        nbytes = rows64 * fx.Int64(logical_k)
        qbuf = ptr_buf_tensor(Q, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=nbytes)
        sbuf = ptr_buf_tensor(Scale, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks))
        obuf = ptr_buf_tensor(Out, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=nbytes * fx.Int64(4))
        start = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        step = fx.grid_dim.x * fx.Int32(block)
        for bid in range(start, n_blocks, step):
            inb = bid < n_blocks
            safe = inb.select(bid, fx.Int32(0))
            row = safe // groups
            grp = safe - row * groups
            col0 = grp * fx.Int32(32)
            exp = fx.Int32(buf_copy_load(sbuf, fx.Int64(safe), elem=fx.Uint8, unit_elems=1))
            scale = _e8m0_to_f32(exp)
            if const_expr(clamp_tiny):
                scale = fx.max(scale, fx.Float32(_TINY))
            for i in range_constexpr(32):
                col = col0 + fx.Int32(i)
                if col < logical_k:
                    idx = fx.Int64(row) * fx.Int64(logical_k) + fx.Int64(col)
                    byte = fx.Int32(buf_copy_load(qbuf, idx, elem=fx.Uint8, unit_elems=1)) & fx.Int32(255)
                    value = fx.Float32(cvt(T.f32, byte.ir_value(), 0)) * scale
                    if inb:
                        buf_copy_store(obuf, idx, value, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        Scale: fx.Tensor,
        Out: fx.Tensor,
        n_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        dequant_tail_kernel(Q, Scale, Out, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def _check_axes(K: int, groups: int, tuning_schema: int, block: int, what: str) -> None:
    expect = 0 if K == 0 else (K + 31) // 32
    if K < 0 or groups != expect or tuning_schema != TUNING_SCHEMA:
        raise ValueError(f"{what} tuning axes are invalid")
    if block not in _stride_blocks(0):
        raise ValueError(f"{what} block {block} is not executable by the stride loop")


@flyc.jit
def _mxfp8_quant_direct(
    X: fx.Tensor,
    Q: fx.Tensor,
    Scale: fx.Tensor,
    n_blocks: fx.Int32,
    K: fx.Constexpr[int],
    groups: fx.Constexpr[int],
    dtype_name: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
) -> None:
    _check_axes(K, groups, tuning_schema, BLOCK, "MXFP8 quant")
    if const_expr(K % 32 == 0):
        build_mxfp8_quant_module(dtype_name, BLOCK)(X, Q, Scale, n_blocks, stream)
    else:
        build_mxfp8_quant_tail_module(dtype_name, BLOCK, int(K))(X, Q, Scale, n_blocks, stream)


@flyc.jit
def _mxfp8_dequant_direct(
    Q: fx.Tensor,
    Scale: fx.Tensor,
    Out: fx.Tensor,
    n_blocks: fx.Int32,
    K: fx.Constexpr[int],
    groups: fx.Constexpr[int],
    dtype_name: fx.Constexpr[str],
    clamp: fx.Constexpr[int],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
) -> None:
    _check_axes(K, groups, tuning_schema, BLOCK, "MXFP8 dequant")
    if dtype_name != "mxfp8" or clamp not in (0, 1):
        raise ValueError(f"MXFP8 dequant axes are invalid: dtype={dtype_name} clamp={clamp}")
    if const_expr(K % 32 == 0):
        build_mxfp8_dequant_module(bool(clamp), BLOCK)(Q, Scale, Out, n_blocks, stream)
    else:
        build_mxfp8_dequant_tail_module(bool(clamp), BLOCK, int(K))(Q, Scale, Out, n_blocks, stream)


def _ref_e8m0_bits(value: torch.Tensor) -> torch.Tensor:
    """E8M0 oracle for this kernel. NaN and Inf stay exponent 0xFF."""

    bits = value.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF).to(torch.uint8)
    is_special = exponent == 0xFF
    round_up = ((bits & 0x400000) != 0) & (((bits & 0x200000) != 0) | ((bits & 0x1FFFFF) != 0) | (exponent != 0))
    bumped = exponent.to(torch.int16) + round_up.to(torch.int16)
    exponent = torch.clamp(bumped, max=0xFF).to(torch.uint8)
    return torch.where(is_special, torch.full_like(exponent, 0xFF), exponent)


def _ref_e8m0_f32(exponent: torch.Tensor) -> torch.Tensor:
    exp = exponent.to(torch.int32)
    bits = exp << 23
    bits = torch.where(exp == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exp == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _ref_mxfp8_dequant(q: torch.Tensor, scale: torch.Tensor, clamp: int, k: int) -> torch.Tensor:
    rows = q.reshape(-1, k)
    groups = (k + 31) // 32
    decoded = _ref_e8m0_f32(scale.reshape(-1, groups))
    if clamp:
        decoded = decoded.clamp_min(_TINY)
    group_of_col = torch.arange(k, device=rows.device) // 32
    return (rows.float() * decoded[:, group_of_col]).reshape(-1)


def _poison_fp8(q: torch.Tensor) -> None:
    q.view(torch.uint8).fill_(_FP8_NAN)


@contextmanager
def _validate_mxfp8_quant(sig_args):
    """NaN-poison FP8 values and reject a scale that is not the E8M0 byte.

    Also compare packed FP8 bytes to a torch e4m3 cast of the scaled block so a
    candidate cannot pass with correct scales and wrong Q.
    """
    q = sig_args["Q"]
    scale = sig_args["Scale"]
    _poison_fp8(q)
    scale.fill_(0xFF)
    yield
    if int(sig_args["n_blocks"]) == 0:
        return
    if bool(torch.isnan(q.float()).any()):
        raise ValueError("MXFP8 quant candidate left NaN values")
    k = int(sig_args["K"])
    rows = sig_args["X"].float().reshape(-1, k)
    pad = (32 - (k % 32)) % 32
    blocks = rows if pad == 0 else torch.nn.functional.pad(rows, (0, pad))
    blocks = blocks.reshape(-1, 32)
    ref_s = _ref_e8m0_bits(blocks.abs().amax(dim=-1) / _FP8_MAX)
    if not torch.equal(scale.reshape(-1), ref_s.reshape(-1)):
        raise ValueError("MXFP8 quant candidate did not match the E8M0 scale reference")
    decoded = _ref_e8m0_f32(ref_s).clamp_min(_TINY).unsqueeze(-1)
    scaled = (blocks / decoded).clamp(-_FP8_MAX, _FP8_MAX)
    ref_q = scaled.to(torch.float8_e4m3fn).reshape(rows.shape[0], -1)[:, :k]
    if not torch.equal(q.view(torch.uint8).reshape(-1), ref_q.view(torch.uint8).reshape(-1)):
        raise ValueError("MXFP8 quant candidate did not match the packed FP8 reference")


@contextmanager
def _validate_mxfp8_dequant(sig_args):
    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if int(sig_args["n_blocks"]) == 0:
        return
    if not bool(torch.isfinite(out).all()):
        raise ValueError("MXFP8 dequant candidate left non-finite values")
    ref = _ref_mxfp8_dequant(sig_args["Q"], sig_args["Scale"], int(sig_args["clamp"]), int(sig_args["K"]))
    if not torch.allclose(out.reshape(-1), ref, rtol=1e-5, atol=1e-6, equal_nan=False):
        raise ValueError("MXFP8 dequant candidate did not match the E8M0 reference")


def _default_block(*_args, **_kwargs):
    return Config(BLOCK=_DEFAULT_BLOCK)


def _quant_configs(*args, **kwargs):
    n_blocks = int(args[3] if len(args) > 3 else kwargs["n_blocks"])
    return [Config(BLOCK=block) for block in _stride_blocks(n_blocks)]


def _dequant_configs(*args, **kwargs):
    n_blocks = int(args[3] if len(args) > 3 else kwargs["n_blocks"])
    return [Config(BLOCK=block) for block in _stride_blocks(n_blocks)]


_mxfp8_quant_tuned = autotune(
    configs=_quant_configs,
    key=["K", "groups", "dtype_name", "tuning_schema"],
    default=_default_block,
    artifact_name="mxfp8_e8m0_quant",
    validate_hook=_validate_mxfp8_quant,
)(_mxfp8_quant_direct)

_mxfp8_dequant_tuned = autotune(
    configs=_dequant_configs,
    key=["K", "groups", "dtype_name", "clamp", "tuning_schema"],
    default=_default_block,
    artifact_name="mxfp8_e8m0_dequant",
    validate_hook=_validate_mxfp8_dequant,
)(_mxfp8_dequant_direct)


def quantize_mxfp8_device(
    x: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to FP8 plus one E8M0 scale per started group of 32.

    A K that is not a multiple of 32 stays at the caller's shape. The kernel
    zero-fills lanes past K in the last group.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(x.device, what="quantize_mxfp8 (gfx120x)")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"MXFP8 quant input must be f32/f16/bf16, got {x.dtype}")
    orig_k = int(x.shape[-1])
    x2 = ensure_contiguous(x, stream=stream)
    groups = 0 if orig_k == 0 else (orig_k + 31) // 32
    q = torch.empty(x2.shape, device=x2.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((*x2.shape[:-1], groups), device=x2.device, dtype=torch.uint8)
    if orig_k == 0 or x2.numel() == 0:
        return q, scale
    flat = x2.reshape(-1)
    n_blocks = (int(flat.numel()) // orig_k) * groups
    name = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}[x.dtype]
    kw = dict(K=orig_k, groups=groups, dtype_name=name, tuning_schema=TUNING_SCHEMA)
    if stream is not None:
        kw["stream"] = stream
    _mxfp8_quant_tuned(flat, q.reshape(-1), scale.reshape(-1), n_blocks, **kw)
    return q, scale


def dequantize_mxfp8_device(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    clamp_tiny: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Decode row-major MXFP8 and per-32 E8M0 scales to fp32.

    A K that is not a multiple of 32 is read at the caller's width. Lanes past
    K are not loaded.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(q.device, what="dequantize_mxfp8 (gfx120x)")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 values must be float8_e4m3fn, got {q.dtype}")
    if scale.dtype != torch.uint8:
        raise ValueError(f"MXFP8 scales must be uint8, got {scale.dtype}")
    orig_k = int(q.shape[-1])
    need_g = 0 if orig_k == 0 else (orig_k + 31) // 32
    if tuple(scale.shape) != (*q.shape[:-1], need_g):
        raise ValueError(f"MXFP8 scale shape must be {(*q.shape[:-1], need_g)}, got {tuple(scale.shape)}")
    q2 = ensure_contiguous(q, stream=stream)
    flat_s = ensure_contiguous(scale, stream=stream).reshape(-1)
    out = torch.empty(q2.shape, device=q2.device, dtype=torch.float32)
    n_blocks = int(flat_s.numel())
    if n_blocks == 0 or orig_k == 0:
        return out
    kw = dict(
        K=orig_k,
        groups=need_g,
        dtype_name="mxfp8",
        clamp=int(bool(clamp_tiny)),
        tuning_schema=TUNING_SCHEMA,
    )
    if stream is not None:
        kw["stream"] = stream
    _mxfp8_dequant_tuned(q2.reshape(-1), flat_s, out.reshape(-1), n_blocks, **kw)
    return out


def quant_dequant_mxfp8_device(
    x: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Per-1x32 MXFP8 round trip. The dequant scale is clamped the way the torch helper clamps it."""
    q, scale = quantize_mxfp8_device(x, stream=stream)
    return dequantize_mxfp8_device(q, scale, clamp_tiny=True, stream=stream)
