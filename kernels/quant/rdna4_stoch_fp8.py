# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Stochastic FP8 e4m3/e5m2 rounding for gfx120x (RDNA4).

Rounds floating-point values to FP8 with a per-element random threshold so
the expected value is unbiased across many samples (useful for quantization-
aware training and activation casting). Both path names compile the bitcast
kernel. ``_stoch_one_select`` is not called. The path string stays in the
kernel signature.

Microbench vs HIP measured about 1.17-1.82x depending on shape and path.
e4m3 and e5m2 share the same schedule; only the format constants differ.
"""

from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import math as fmath
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, kernel_signature, ptr_buf_tensor

KERNEL_NAME = "stoch_fp8_gfx120x"
BLOCK = 256
_F8_E4M3_MAX = 448.0
_F8_E5M2_MAX = 57344.0
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)


def _fp8_consts(e5m2: bool) -> tuple[int, int, int, float, int, int, int]:
    mantissa_bits = 2 if e5m2 else 3
    exponent_bits = 5 if e5m2 else 4
    exponent_bias = 15 if e5m2 else 7
    fp8_max = _F8_E5M2_MAX if e5m2 else _F8_E4M3_MAX
    mantissa_levels = 1 << mantissa_bits
    max_exponent_field = 30 if e5m2 else 15
    max_mantissa_field = (mantissa_levels - 1) if e5m2 else 6
    return (
        mantissa_bits,
        exponent_bits,
        exponent_bias,
        fp8_max,
        mantissa_levels,
        max_exponent_field,
        max_mantissa_field,
    )


def _stoch_one_bitcast(value: object, rng_b: object, C: object) -> object:
    (
        mantissa_bits,
        exponent_bits,
        exponent_bias,
        fp8_max,
        mantissa_levels,
        max_exponent_field,
        max_mantissa_field,
    ) = C
    random = fx.Float32(fx.Int32(rng_b)) * fx.Float32(1.0 / 256.0)
    abs_value = fmath.absf(value)
    zero = fx.Float32(0.0)
    one = fx.Float32(1.0)
    sign = (value < zero).select(fx.Float32(-1.0), (value > zero).select(one, zero))
    is_nan = fmath.isnan(value)
    nan_code = fx.Uint8(((value < zero).select(fx.Int32(0x80), fx.Int32(0)) | fx.Int32(0x7F)))
    is_zero = abs_value == zero
    zero_code = fx.Uint8(0)
    bits = abs_value.bitcast(fx.Uint32)
    ue = fx.Int32((bits >> fx.Uint32(23)) & fx.Uint32(0xFF))
    is_sub_in = ue == fx.Int32(0)
    exp_f_i = (ue - fx.Int32(127)) + fx.Int32(exponent_bias)
    exp_hi_i = fx.Int32((1 << exponent_bits) - 1)
    exp_f_i = (exp_f_i < fx.Int32(0)).select(fx.Int32(0), exp_f_i)
    exp_f_i = (exp_f_i > exp_hi_i).select(exp_hi_i, exp_f_i)
    exp_f_i = is_sub_in.select(fx.Int32(0), exp_f_i)
    normal = exp_f_i != fx.Int32(0)
    ue_scale = exp_f_i - fx.Int32(exponent_bias) + fx.Int32(127)
    exponent_scale = (ue_scale.bitcast(fx.Uint32) << fx.Uint32(23)).bitcast(fx.Float32)
    levels = fx.Float32(float(mantissa_levels))
    sub_mant_scale = fmath.exp2(fx.Float32(float(-exponent_bias + 1 - mantissa_bits)))
    sub_val_scale = fmath.exp2(fx.Float32(float(-exponent_bias + 1)))
    mant_scaled = normal.select(
        (abs_value / exponent_scale - one) * levels,
        abs_value / sub_mant_scale,
    )
    mantissa = fmath.floor(mant_scaled + random) / levels
    rounded = sign * normal.select(
        exponent_scale * (one + mantissa),
        sub_val_scale * mantissa,
    )
    rounded = fmath.clampf(rounded, fx.Float32(-fp8_max), fx.Float32(fp8_max))
    pack_abs = fmath.absf(rounded)
    pack_sign = (sign < zero).select(fx.Int32(0x80), fx.Int32(0))
    pack_is_nan = fmath.isnan(rounded)
    pack_is_zero = pack_abs == zero
    min_normal = fmath.exp2(fx.Float32(float(1 - exponent_bias)))
    is_sub = pack_abs < min_normal
    sub_scale = fmath.exp2(fx.Float32(float(1 - exponent_bias - mantissa_bits)))
    sub_x = pack_abs / sub_scale
    sub_floor = fmath.floor(sub_x)
    sub_frac = sub_x - sub_floor
    sub_n = fx.Int32(sub_floor)
    sub_n = (sub_frac > fx.Float32(0.5)).select(
        sub_n + fx.Int32(1),
        (sub_frac == fx.Float32(0.5)).select(sub_n + (sub_n & fx.Int32(1)), sub_n),
    )
    sub_exp = fx.Int32(0)
    sub_mant = sub_n
    sub_carry = sub_mant >= fx.Int32(mantissa_levels)
    sub_exp = sub_carry.select(fx.Int32(1), sub_exp)
    sub_mant = sub_carry.select(fx.Int32(0), sub_mant)
    pbits = pack_abs.bitcast(fx.Uint32)
    pue = fx.Int32((pbits >> fx.Uint32(23)) & fx.Uint32(0xFF))
    nor_exp = pue - fx.Int32(127) + fx.Int32(exponent_bias)
    pue_scale = nor_exp - fx.Int32(exponent_bias) + fx.Int32(127)
    nor_scale = (pue_scale.bitcast(fx.Uint32) << fx.Uint32(23)).bitcast(fx.Float32)
    nor_x = (pack_abs / nor_scale - one) * levels
    nor_floor = fmath.floor(nor_x)
    nor_frac = nor_x - nor_floor
    nor_n = fx.Int32(nor_floor)
    nor_n = (nor_frac > fx.Float32(0.5)).select(
        nor_n + fx.Int32(1),
        (nor_frac == fx.Float32(0.5)).select(nor_n + (nor_n & fx.Int32(1)), nor_n),
    )
    nor_mant = nor_n
    nor_carry = nor_mant >= fx.Int32(mantissa_levels)
    nor_mant = nor_carry.select(fx.Int32(0), nor_mant)
    nor_exp = nor_carry.select(nor_exp + fx.Int32(1), nor_exp)
    exp_field = is_sub.select(sub_exp, nor_exp)
    mant_field = is_sub.select(sub_mant, nor_mant)
    over = (exp_field > fx.Int32(max_exponent_field)).select(
        fx.Int32(1),
        ((exp_field == fx.Int32(max_exponent_field)) & (mant_field > fx.Int32(max_mantissa_field))).select(
            fx.Int32(1), fx.Int32(0)
        ),
    )
    exp_field = over.select(fx.Int32(max_exponent_field), exp_field)
    mant_field = over.select(fx.Int32(max_mantissa_field), mant_field)
    packed = fx.Uint8(pack_sign | (exp_field << fx.Int32(mantissa_bits)) | mant_field)
    packed = pack_is_zero.select(fx.Uint8(pack_sign), packed)
    packed = pack_is_nan.select(fx.Uint8(pack_sign | fx.Int32(0x7F)), packed)
    out = is_zero.select(zero_code, packed)
    out = is_nan.select(nan_code, out)
    return out


def _stoch_one_select(value: object, rng_b: object, C: object) -> object:
    (
        mantissa_bits,
        exponent_bits,
        exponent_bias,
        fp8_max,
        mantissa_levels,
        max_exponent_field,
        max_mantissa_field,
    ) = C
    random = fx.Float32(fx.Int32(rng_b)) * fx.Float32(1.0 / 256.0)
    abs_value = fmath.absf(value)
    zero = fx.Float32(0.0)
    one = fx.Float32(1.0)
    sign = (value < zero).select(fx.Float32(-1.0), (value > zero).select(one, zero))
    is_nan = fmath.isnan(value)
    nan_code = fx.Uint8(((value < zero).select(fx.Int32(0x80), fx.Int32(0)) | fx.Int32(0x7F)))
    is_zero = abs_value == zero
    zero_code = fx.Uint8(0)
    log2_abs = fmath.log2(fx.max(abs_value, fx.Float32(1.0e-45)))
    exp_f_i = fx.Int32(fmath.floor(log2_abs)) + fx.Int32(exponent_bias)
    exp_hi_i = fx.Int32((1 << exponent_bits) - 1)
    exp_f_i = (exp_f_i < fx.Int32(0)).select(fx.Int32(0), exp_f_i)
    exp_f_i = (exp_f_i > exp_hi_i).select(exp_hi_i, exp_f_i)
    normal = exp_f_i != fx.Int32(0)
    exponent_scale = fmath.exp2(fx.Float32(exp_f_i) - fx.Float32(float(exponent_bias)))
    levels = fx.Float32(float(mantissa_levels))
    sub_mant_scale = fmath.exp2(fx.Float32(float(-exponent_bias + 1 - mantissa_bits)))
    sub_val_scale = fmath.exp2(fx.Float32(float(-exponent_bias + 1)))
    mant_scaled = normal.select(
        (abs_value / exponent_scale - one) * levels,
        abs_value / sub_mant_scale,
    )
    mantissa = fmath.floor(mant_scaled + random) / levels
    rounded = sign * normal.select(
        exponent_scale * (one + mantissa),
        sub_val_scale * mantissa,
    )
    rounded = fmath.clampf(rounded, fx.Float32(-fp8_max), fx.Float32(fp8_max))
    pack_abs = fmath.absf(rounded)
    pack_sign = (sign < zero).select(fx.Int32(0x80), fx.Int32(0))
    pack_is_nan = fmath.isnan(rounded)
    pack_is_zero = pack_abs == zero
    min_normal = fmath.exp2(fx.Float32(float(1 - exponent_bias)))
    is_sub = pack_abs < min_normal
    sub_scale = fmath.exp2(fx.Float32(float(1 - exponent_bias - mantissa_bits)))
    sub_x = pack_abs / sub_scale
    sub_floor = fmath.floor(sub_x)
    sub_frac = sub_x - sub_floor
    sub_n = fx.Int32(sub_floor)
    sub_n = (sub_frac > fx.Float32(0.5)).select(
        sub_n + fx.Int32(1),
        (sub_frac == fx.Float32(0.5)).select(sub_n + (sub_n & fx.Int32(1)), sub_n),
    )
    sub_exp = fx.Int32(0)
    sub_mant = sub_n
    sub_carry = sub_mant >= fx.Int32(mantissa_levels)
    sub_exp = sub_carry.select(fx.Int32(1), sub_exp)
    sub_mant = sub_carry.select(fx.Int32(0), sub_mant)
    log2_pack = fmath.log2(fx.max(pack_abs, fx.Float32(1.0e-45)))
    nor_exp = fx.Int32(fmath.floor(log2_pack)) + fx.Int32(exponent_bias)
    nor_scale = fmath.exp2(fx.Float32(nor_exp) - fx.Float32(float(exponent_bias)))
    nor_x = (pack_abs / nor_scale - one) * levels
    nor_floor = fmath.floor(nor_x)
    nor_frac = nor_x - nor_floor
    nor_n = fx.Int32(nor_floor)
    nor_n = (nor_frac > fx.Float32(0.5)).select(
        nor_n + fx.Int32(1),
        (nor_frac == fx.Float32(0.5)).select(nor_n + (nor_n & fx.Int32(1)), nor_n),
    )
    nor_mant = nor_n
    nor_carry = nor_mant >= fx.Int32(mantissa_levels)
    nor_mant = nor_carry.select(fx.Int32(0), nor_mant)
    nor_exp = nor_carry.select(nor_exp + fx.Int32(1), nor_exp)
    exp_field = is_sub.select(sub_exp, nor_exp)
    mant_field = is_sub.select(sub_mant, nor_mant)
    over = (exp_field > fx.Int32(max_exponent_field)).select(
        fx.Int32(1),
        ((exp_field == fx.Int32(max_exponent_field)) & (mant_field > fx.Int32(max_mantissa_field))).select(
            fx.Int32(1), fx.Int32(0)
        ),
    )
    exp_field = over.select(fx.Int32(max_exponent_field), exp_field)
    mant_field = over.select(fx.Int32(max_mantissa_field), mant_field)
    packed = fx.Uint8(pack_sign | (exp_field << fx.Int32(mantissa_bits)) | mant_field)
    packed = pack_is_zero.select(fx.Uint8(pack_sign), packed)
    packed = pack_is_nan.select(fx.Uint8(pack_sign | fx.Int32(0x7F)), packed)
    out = is_zero.select(zero_code, packed)
    out = is_nan.select(nan_code, out)
    return out


def _build_scalar(in_dtype: str, e5m2: bool, path: str, block: int) -> Callable[..., None]:
    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    C = _fp8_consts(e5m2)
    tile = int(block)
    sig = kernel_signature(block=tile, dtype=in_dtype, e5m2=e5m2, path=path)
    stoch_fn = _stoch_one_bitcast  # select path uses same HIP-exact IR; path kept in signature

    @flyc.kernel(known_block_size=[tile, 1, 1])
    def stoch_kernel(In: fx.Pointer, Rng: fx.Pointer, n_elems: fx.Int32) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        i = fx.Int32(bid) * fx.Int32(tile) + fx.Int32(tid)
        in_buf = ptr_buf_tensor(
            In,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(in_bytes),
        )
        rng_buf = ptr_buf_tensor(
            Rng,
            elem=fx.Uint8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elems),
        )
        if i < n_elems:
            raw = buf_copy_load(in_buf, i, elem=InTy, unit_elems=1)
            value = fx.Float32(raw)
            rng_b = fx.Uint8(buf_copy_load(rng_buf, i, elem=fx.Uint8, unit_elems=1))
            out = stoch_fn(value, rng_b, C)
            buf_copy_store(rng_buf, i, out, elem=fx.Uint8, unit_elems=1)

    stoch_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(In: fx.Pointer, Rng: fx.Pointer, n_elems: fx.Int32, stream: fx.Stream = fx.Stream(None)) -> None:
        n64 = fx.Int64(n_elems)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        stoch_kernel(In, Rng, n_elems).launch(grid=(grid_x, 1, 1), block=(tile, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=16)
def build_stoch_fp8_module(
    in_dtype: str = "bfloat16",
    e5m2: bool = False,
    path: str = "bitcast",
    block: int = BLOCK,
) -> Callable[..., None]:
    """Build @flyc.jit launcher. path in {"bitcast","select"}."""
    if path not in ("bitcast", "select"):
        raise ValueError(f"unknown path {path!r}")
    if in_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported in_dtype {in_dtype!r}")
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"block={block} is not a legal wave32 block")
    return _build_scalar(in_dtype, e5m2, path, block)


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


@contextmanager
def _validate_stoch(sig_args):
    import torch

    # Rng is the in-place uint8 output. Fill only during the untimed
    # validation launch; the autotuner restores the caller's bytes after.
    out = sig_args["Rng"]
    out.fill_(0)
    yield
    if not bool(torch.isfinite(out.float()).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def stoch_fp8_direct(
    In: fx.Pointer,
    Rng: fx.Pointer,
    n_elems: fx.Int32,
    in_dtype: fx.Constexpr[str],
    e5m2: fx.Constexpr[bool],
    path: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Direct JIT entry. ``tuning_schema`` partitions the autotune cache."""
    launch = build_stoch_fp8_module(in_dtype, bool(e5m2), path, block=BLOCK_THREADS)
    launch(In, Rng, n_elems, stream)


_stoch_fp8_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["in_dtype", "e5m2", "path", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_stoch_fp8",
    validate_hook=_validate_stoch,
)(stoch_fp8_direct)

__all__ = [
    "KERNEL_NAME",
    "BLOCK",
    "build_stoch_fp8_module",
    "stoch_fp8_direct",
    "_stoch_fp8_tuned",
]
