# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Per-1x32 MXFP4 E2M1 quant and dequant for gfx120x.

The code matches ``kernels.monokernel.formats``: E8M0 scales, ties to even
at 0.75 / 1.75 / 3.5, low nibble = even element. Launches are direct
``fx.Tensor`` JIT calls. Block size is autotuned; a grid-stride loop with a
tail mask covers every group count.
"""

from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor
from kernels.quant.rdna4_mxfp8_e8m0 import _TINY, _e8m0_bits, _e8m0_to_f32

_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)
_DEFAULT_BLOCK = 256
TUNING_SCHEMA = 1
_FP4_MAX = 4.0
_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_FP4_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _stride_blocks(n_items: int) -> tuple[int, ...]:
    """Block sizes whose grid-stride loop can cover ``n_items``.

    ``for item in range(block_idx * BLOCK + thread_idx, n_items, grid * BLOCK)``
    runs for every positive wave-multiple up to the workgroup limit. A partial
    last tile is masked by the loop bound (and by ``item < n_items`` before stores).
    """
    if n_items < 0:
        return ()
    return tuple(block for block in _BLOCK_CHOICES if 0 < block <= 1024 and block % 32 == 0)


def _fp4_code(value):
    mag = fx.min(fmath.absf(value), fx.Float32(6.0))
    code = fx.Int32(0)
    for bound in _FP4_BOUNDS:
        code = code + (mag > fx.Float32(bound)).select(fx.Int32(1), fx.Int32(0))
    code = code + (mag == fx.Float32(0.75)).select(fx.Int32(1), fx.Int32(0))
    code = code + (mag == fx.Float32(1.75)).select(fx.Int32(1), fx.Int32(0))
    code = code + (mag == fx.Float32(3.5)).select(fx.Int32(1), fx.Int32(0))
    return code | (value < fx.Float32(0)).select(fx.Int32(8), fx.Int32(0))


def _fp4_value(code):
    value = fx.Float32(_LUT[0])
    for index, entry in enumerate(_LUT):
        value = (code == fx.Int32(index)).select(fx.Float32(entry), value)
    return value


@lru_cache(maxsize=16)
def build_mxfp4_quant_module(in_name: str, block: int = _DEFAULT_BLOCK):
    in_ty, in_bytes = {"float32": (fx.Float32, 4), "bfloat16": (fx.BFloat16, 2), "float16": (fx.Float16, 2)}[in_name]
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP4 quant block {block} is not executable by the stride loop")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def quant_kernel(X: fx.Tensor, Q: fx.Tensor, Scale: fx.Tensor, n_blocks: fx.Int32) -> None:
        xbuf = ptr_buf_tensor(
            X, elem=in_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(32 * in_bytes)
        )
        qbuf = ptr_buf_tensor(
            Q, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(16)
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
                    value = fx.Float32(raw)
                else:
                    value = raw.to(fx.Float32)
                vals.append(value)
                amax = fx.max(amax, fmath.absf(value))
            exp = _e8m0_bits(amax / fx.Float32(_FP4_MAX))
            scale = fx.max(_e8m0_to_f32(exp), fx.Float32(_TINY))
            inv = fx.Float32(1.0) / scale
            codes = []
            for i in range_constexpr(32):
                codes.append(_fp4_code(vals[i] * inv))
            if inb:
                buf_copy_store(sbuf, fx.Int64(safe), exp.to(fx.Uint8), elem=fx.Uint8, unit_elems=1)
                base = fx.Int64(safe) * fx.Int64(16)
                for i in range_constexpr(16):
                    byte = (codes[2 * i] | (codes[2 * i + 1] << fx.Int32(4))).to(fx.Uint8)
                    buf_copy_store(qbuf, base + fx.Int64(i), byte, elem=fx.Uint8, unit_elems=1)

    @flyc.jit
    def launch(
        X: fx.Tensor, Q: fx.Tensor, Scale: fx.Tensor, n_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        quant_kernel(X, Q, Scale, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@lru_cache(maxsize=8)
def build_mxfp4_dequant_module(block: int = _DEFAULT_BLOCK):
    if block not in _BLOCK_CHOICES:
        raise ValueError(f"MXFP4 dequant block {block} is not executable by the stride loop")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dequant_kernel(Q: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_blocks: fx.Int32) -> None:
        qbuf = ptr_buf_tensor(
            Q, elem=fx.Uint8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_blocks) * fx.Int64(16)
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
            base_q = fx.Int64(safe) * fx.Int64(16)
            base_o = fx.Int64(safe) * fx.Int64(32)
            for i in range_constexpr(16):
                byte = fx.Int32(buf_copy_load(qbuf, base_q + fx.Int64(i), elem=fx.Uint8, unit_elems=1))
                lo = _fp4_value(byte & fx.Int32(0xF)) * scale
                hi = _fp4_value((byte >> fx.Int32(4)) & fx.Int32(0xF)) * scale
                if inb:
                    buf_copy_store(obuf, base_o + fx.Int64(2 * i), lo, elem=fx.Float32, unit_elems=1)
                    buf_copy_store(obuf, base_o + fx.Int64(2 * i + 1), hi, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Q: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_blocks: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_blocks) + fx.Int64(block - 1)) // fx.Int64(block)
        dequant_kernel(Q, Scale, Out, n_blocks).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def _check_axes(K: int, groups: int, tuning_schema: int, block: int, what: str) -> None:
    if groups * 32 != K or K < 0 or tuning_schema != TUNING_SCHEMA:
        raise ValueError(f"{what} tuning axes are invalid")
    if block not in _stride_blocks(0):
        raise ValueError(f"{what} block {block} is not executable by the stride loop")


@flyc.jit
def _mxfp4_quant_direct(
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
    _check_axes(K, groups, tuning_schema, BLOCK, "MXFP4 quant")
    build_mxfp4_quant_module(dtype_name, BLOCK)(X, Q, Scale, n_blocks, stream)


@flyc.jit
def _mxfp4_dequant_direct(
    Q: fx.Tensor,
    Scale: fx.Tensor,
    Out: fx.Tensor,
    n_blocks: fx.Int32,
    K: fx.Constexpr[int],
    groups: fx.Constexpr[int],
    dtype_name: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
) -> None:
    _check_axes(K, groups, tuning_schema, BLOCK, "MXFP4 dequant")
    if dtype_name != "mxfp4":
        raise ValueError(f"MXFP4 dequant dtype axis must be 'mxfp4', got {dtype_name}")
    build_mxfp4_dequant_module(BLOCK)(Q, Scale, Out, n_blocks, stream)


def _ref_e8m0_bits(value: torch.Tensor) -> torch.Tensor:
    bits = value.float().contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    half = (bits & 0x400000) != 0
    sticky = ((bits & 0x200000) != 0) | ((bits & 0x1FFFFF) != 0) | (exponent != 0)
    return ((exponent + (half & sticky).to(torch.int32)) & 0xFF).to(torch.uint8)


def _ref_e8m0_f32(exponent: torch.Tensor) -> torch.Tensor:
    exp = exponent.to(torch.int32)
    bits = exp << 23
    bits = torch.where(exp == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(exp == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _ref_fp4_codes(scaled: torch.Tensor) -> torch.Tensor:
    mag = scaled.float().abs().clamp(max=6.0)
    code = torch.zeros_like(mag, dtype=torch.int32)
    for bound in _FP4_BOUNDS:
        code = code + (mag > bound).to(torch.int32)
    code = code + (mag == 0.75).to(torch.int32)
    code = code + (mag == 1.75).to(torch.int32)
    code = code + (mag == 3.5).to(torch.int32)
    return code | (scaled < 0).to(torch.int32) << 3


def _ref_mxfp4_quant(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = flat.float().reshape(-1, 32)
    exp = _ref_e8m0_bits(blocks.abs().amax(dim=-1) / _FP4_MAX)
    scale = _ref_e8m0_f32(exp).clamp_min(_TINY)
    codes = _ref_fp4_codes(blocks / scale.unsqueeze(-1))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).reshape(-1)
    return packed, exp


def _ref_mxfp4_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_LUT, dtype=torch.float32, device=q.device)
    byte = q.reshape(-1, 16).to(torch.int32)
    lo = lut[byte & 0xF]
    hi = lut[(byte >> 4) & 0xF]
    values = torch.stack((lo, hi), dim=-1).reshape(-1, 32)
    return (values * _ref_e8m0_f32(scale.reshape(-1)).unsqueeze(-1)).reshape(-1)


@contextmanager
def _validate_mxfp4_quant(sig_args):
    """Poison outputs, then reject a candidate that missed a store or changed the codes."""
    q = sig_args["Q"]
    scale = sig_args["Scale"]
    q.fill_(0xFF)
    scale.fill_(0xFF)
    yield
    if int(sig_args["n_blocks"]) == 0:
        return
    ref_q, ref_s = _ref_mxfp4_quant(sig_args["X"])
    if not torch.equal(q, ref_q) or not torch.equal(scale, ref_s):
        raise ValueError("MXFP4 quant candidate did not match the E2M1/E8M0 reference")


@contextmanager
def _validate_mxfp4_dequant(sig_args):
    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if int(sig_args["n_blocks"]) == 0:
        return
    if not bool(torch.isfinite(out).all()):
        raise ValueError("MXFP4 dequant candidate left non-finite values")
    ref = _ref_mxfp4_dequant(sig_args["Q"], sig_args["Scale"])
    if not torch.equal(out, ref):
        raise ValueError("MXFP4 dequant candidate did not match the E2M1/E8M0 reference")


def _default_block(*_args, **_kwargs):
    return Config(BLOCK=_DEFAULT_BLOCK)


def _quant_configs(*args, **kwargs):
    n_blocks = int(args[3] if len(args) > 3 else kwargs["n_blocks"])
    return [Config(BLOCK=block) for block in _stride_blocks(n_blocks)]


def _dequant_configs(*args, **kwargs):
    n_blocks = int(args[3] if len(args) > 3 else kwargs["n_blocks"])
    return [Config(BLOCK=block) for block in _stride_blocks(n_blocks)]


_mxfp4_quant_tuned = autotune(
    configs=_quant_configs,
    key=["K", "groups", "dtype_name", "tuning_schema"],
    default=_default_block,
    artifact_name="mxfp4_e2m1_quant",
    validate_hook=_validate_mxfp4_quant,
)(_mxfp4_quant_direct)

_mxfp4_dequant_tuned = autotune(
    configs=_dequant_configs,
    key=["K", "groups", "dtype_name", "tuning_schema"],
    default=_default_block,
    artifact_name="mxfp4_e2m1_dequant",
    validate_hook=_validate_mxfp4_dequant,
)(_mxfp4_dequant_direct)


def quantize_mxfp4_device(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to packed MXFP4 plus per-32 E8M0 scales."""
    require_gfx120x(w.device, what="quantize_mxfp4 (gfx120x)")
    if w.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"MXFP4 quant input must be f32/f16/bf16, got {w.dtype}")
    if w.shape[-1] % 32:
        raise ValueError(f"MXFP4 K dimension must be divisible by 32, got {w.shape[-1]}")
    flat = w.contiguous().reshape(-1)
    n_blocks = int(flat.numel() // 32)
    packed_shape = (*w.shape[:-1], w.shape[-1] // 2)
    packed = torch.empty(packed_shape, device=w.device, dtype=torch.uint8)
    scale_shape = (*w.shape[:-1], w.shape[-1] // 32)
    scale = torch.empty(scale_shape, device=w.device, dtype=torch.uint8)
    name = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}[w.dtype]
    if n_blocks:
        _mxfp4_quant_tuned(
            flat,
            packed.reshape(-1),
            scale.reshape(-1),
            n_blocks,
            K=int(w.shape[-1]),
            groups=int(w.shape[-1] // 32),
            dtype_name=name,
            tuning_schema=TUNING_SCHEMA,
        )
    return packed, scale


def dequantize_mxfp4_device(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode packed MXFP4 and per-32 E8M0 scales to fp32."""
    require_gfx120x(q.device, what="dequantize_mxfp4 (gfx120x)")
    q = q.view(torch.uint8)
    if scale.dtype != torch.uint8:
        raise ValueError(f"MXFP4 scales must be uint8, got {scale.dtype}")
    expected = (*q.shape[:-1], q.shape[-1] // 16)
    if q.shape[-1] % 16:
        raise ValueError(f"packed MXFP4 last dim must be a multiple of 16, got {q.shape[-1]}")
    if tuple(scale.shape) != expected:
        raise ValueError(f"MXFP4 scale shape must be {expected}, got {tuple(scale.shape)}")
    out_shape = (*q.shape[:-1], q.shape[-1] * 2)
    out = torch.empty(out_shape, device=q.device, dtype=torch.float32)
    flat_q = q.contiguous().reshape(-1)
    flat_s = scale.contiguous().reshape(-1)
    n_blocks = int(flat_s.numel())
    k = int(q.shape[-1] * 2)
    if n_blocks:
        _mxfp4_dequant_tuned(
            flat_q,
            flat_s,
            out.reshape(-1),
            n_blocks,
            K=k,
            groups=k // 32,
            dtype_name="mxfp4",
            tuning_schema=TUNING_SCHEMA,
        )
    return out
