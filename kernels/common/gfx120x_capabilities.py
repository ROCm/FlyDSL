# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host-facing list of gfx120x product calls.

``available_for_arch`` is empty for every other arch, including gfx1250,
gfx950, and gfx942. ``resolve`` returns the existing function. It does not
launch, and it does not change that function's own arch check.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from flydsl.runtime.device import is_gfx120x, is_gfx120x_arch


@dataclass(frozen=True)
class Gfx120xOp:
    """One callable a gfx120x host can import."""

    name: str
    group: str
    module: str
    attr: str
    summary: str


def _op(name: str, group: str, module: str, attr: str, summary: str) -> Gfx120xOp:
    return Gfx120xOp(name, group, module, attr, summary)


# Names are stable. Each attr is a function that already exists.
_OPS: tuple[Gfx120xOp, ...] = (
    _op("f16_gemm", "gemm", "kernels.gemm.rdna_f16_gemm", "create_wmma_gemm_module", "f16/bf16 WMMA GEMM"),
    _op(
        "fp8_preshuffle_gemm",
        "gemm",
        "kernels.gemm.rdna_fp8_preshuffle_gemm",
        "compile_fp8_gemm",
        "e4m3 preshuffle GEMM",
    ),
    _op("scaled_mm_fp8", "gemm", "kernels.gemm.rdna4_scaled_mm_fp8", "scaled_mm_fp8", "tensorwise scaled FP8 GEMM"),
    _op(
        "scaled_mm_fp8_fused",
        "gemm",
        "kernels.gemm.rdna4_scaled_mm_fp8_fused",
        "scaled_mm_fp8_fused",
        "fused FP8 quant GEMM",
    ),
    _op("w8a16_linear", "gemm", "kernels.gemm.rdna4_w8a16_linear", "w8a16_linear", "float activation, 8-bit weight"),
    _op("int8_linear", "gemm", "kernels.gemm.rdna4_int8_linear", "int8_linear", "int8 WMMA linear"),
    _op(
        "int8_linear_fused",
        "gemm",
        "kernels.gemm.rdna4_int8_linear_fused",
        "int8_linear_fused",
        "fused int8 quant linear",
    ),
    _op("iu4_gemm", "gemm", "kernels.gemm.rdna4_iu4_gemm", "iu4_gemm", "packed int4 GEMM"),
    _op("mxfp8_block_gemm", "gemm", "kernels.gemm.rdna4_mxfp8_block_gemm", "mxfp8_block_gemm", "MXFP8 block GEMM"),
    _op("mxfp4_block_gemm", "gemm", "kernels.gemm.rdna4_mxfp4_block_gemm", "mxfp4_block_gemm", "MXFP4 block GEMM"),
    _op(
        "fused_swiglu_mlp",
        "gemm",
        "kernels.gemm.rdna4_fused_mlp_nmajor",
        "fused_swiglu_mlp_nmajor",
        "fp16/bf16 SwiGLU MLP",
    ),
    _op(
        "flash_attn",
        "attention",
        "kernels.attention.flash_attn_interface",
        "flydsl_flash_attn_func",
        "shared FlashAttention entry; gfx120x returns first",
    ),
    _op(
        "flash_attn_varlen",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_varlen_func",
        "packed varlen bf16/fp16",
    ),
    _op(
        "flash_attn_paged",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_paged_func",
        "paged bf16/fp16",
    ),
    _op(
        "flash_attn_varlen_paged",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_varlen_paged_func",
        "varlen paged bf16/fp16",
    ),
    _op(
        "flash_attn_fp8",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_fp8_func",
        "fp8 e4m3/e5m2 attention",
    ),
    _op(
        "flash_attn_int8",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_int8_func",
        "int8 attention",
    ),
    _op(
        "flash_attn_fp8_varlen",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_fp8_varlen_func",
        "packed varlen fp8",
    ),
    _op(
        "flash_attn_int8_varlen",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_int8_varlen_func",
        "packed varlen int8",
    ),
    _op(
        "flash_attn_quant_paged",
        "attention",
        "kernels.attention.flash_attn_gfx120x_host",
        "flydsl_flash_attn_quant_paged_func",
        "fp8/int8 paged attention",
    ),
    _op("rope", "norm", "kernels.norm.rope_gfx120x", "build_rope_module", "RoPE launcher"),
    _op("rms_rope", "norm", "kernels.norm.rms_rope_gfx120x", "build_rms_rope_module", "RMSNorm + RoPE launcher"),
    _op("adaln", "norm", "kernels.norm.adaln_gfx120x", "build_adaln_module", "AdaLN launcher"),
    _op("fp8_quant", "quant", "kernels.quant.rdna4_fp8_quant", "fp8_quant_direct", "per-tensor FP8 quant"),
    _op("fp8_dequant", "quant", "kernels.quant.rdna4_fp8_quant", "dequantize_fp8", "per-tensor FP8 dequant"),
    _op("stoch_fp8", "quant", "kernels.quant.rdna4_stoch_fp8", "stoch_fp8_direct", "stochastic FP8 cast"),
    _op("mxfp8_quant", "quant", "kernels.quant.rdna4_mxfp8_e8m0", "quantize_mxfp8_device", "MXFP8 quant"),
    _op("mxfp8_dequant", "quant", "kernels.quant.rdna4_mxfp8_e8m0", "dequantize_mxfp8_device", "MXFP8 dequant"),
    _op("mxfp4_quant", "quant", "kernels.quant.rdna4_mxfp4_e2m1", "quantize_mxfp4_device", "MXFP4 quant"),
    _op("mxfp4_dequant", "quant", "kernels.quant.rdna4_mxfp4_e2m1", "dequantize_mxfp4_device", "MXFP4 dequant"),
    _op(
        "int8_rowwise_quant",
        "quant",
        "kernels.quant.rdna4_quantize_int8_rowwise",
        "quantize_int8_rowwise",
        "per-row int8 quant",
    ),
    _op(
        "int8_rowwise_dequant",
        "quant",
        "kernels.quant.rdna4_quantize_int8_rowwise",
        "dequantize_int8_rowwise",
        "per-row int8 dequant",
    ),
    _op(
        "int8_tensorwise_quant",
        "quant",
        "kernels.quant.rdna4_quantize_int8_tensorwise",
        "quantize_int8_tensorwise",
        "per-tensor int8 quant",
    ),
    _op(
        "int8_tensorwise_dequant",
        "quant",
        "kernels.quant.rdna4_quantize_int8_tensorwise",
        "dequantize_int8_tensorwise",
        "per-tensor int8 dequant",
    ),
    _op(
        "int8_convrot_quant",
        "quant",
        "kernels.quant.rdna4_int8_convrot",
        "quantize_int8_convrot_weight",
        "int8 ConvRot quant",
    ),
    _op(
        "int8_convrot_dequant",
        "quant",
        "kernels.quant.rdna4_int8_convrot",
        "dequantize_int8_convrot_weight",
        "int8 ConvRot dequant",
    ),
    _op(
        "int8_convrot_linear", "quant", "kernels.quant.rdna4_int8_convrot", "int8_linear_convrot", "int8 ConvRot linear"
    ),
    _op(
        "convrot_w4a4_quant",
        "quant",
        "kernels.quant.rdna4_convrot_w4a4",
        "quantize_convrot_w4a4_weight",
        "W4A4 ConvRot quant",
    ),
    _op(
        "convrot_w4a4_dequant",
        "quant",
        "kernels.quant.rdna4_convrot_w4a4",
        "dequantize_convrot_w4a4_weight",
        "W4A4 ConvRot dequant",
    ),
    _op(
        "convrot_w4a4_linear", "quant", "kernels.quant.rdna4_convrot_w4a4", "convrot_w4a4_linear", "W4A4 ConvRot linear"
    ),
    _op("w4a8_quant", "quant", "kernels.quant.rdna4_asym_w4a8", "quantize_w4a8_int8_weight", "asymmetric W4A8 quant"),
    _op(
        "w4a8_dequant",
        "quant",
        "kernels.quant.rdna4_asym_w4a8",
        "dequant_int4_grouped_to_int8",
        "asymmetric W4A8 dequant",
    ),
    _op("w4a8_linear", "quant", "kernels.quant.rdna4_asym_w4a8", "w4a8_int8_linear", "asymmetric W4A8 linear"),
    _op("awq_dequant", "quant", "kernels.quant.rdna4_awq_w4a16", "dequant_awq_w4a16_weight", "AWQ weight dequant"),
    _op("awq_gemv", "quant", "kernels.quant.rdna4_awq_w4a16", "gemv_awq_w4a16", "AWQ fused GEMV"),
    _op("svdquant_quant", "quant", "kernels.quant.rdna4_svdquant_w4a4", "quantize_svdquant_w4a4", "SVDQuant quant"),
    _op(
        "svdquant_dequant",
        "quant",
        "kernels.quant.rdna4_svdquant_w4a4",
        "dequant_svdquant_w4a4_weight",
        "SVDQuant dequant",
    ),
    _op("svdquant_linear", "quant", "kernels.quant.rdna4_svdquant_w4a4", "svdquant_w4a4_linear", "SVDQuant linear"),
    _op(
        "int4_dequant_signed",
        "quant",
        "kernels.quant.rdna4_int4_codec",
        "dequant_int4_groupwise_signed",
        "signed int4 group dequant",
    ),
    _op(
        "int4_dequant_awq",
        "quant",
        "kernels.quant.rdna4_int4_codec",
        "dequant_uint4_groupwise_awq",
        "AWQ uint4 group dequant",
    ),
    _op("silu_mul", "activation", "kernels.common.gfx120x_swiglu", "silu_mul", "SiLU multiply"),
    _op("swiglu_chunk", "activation", "kernels.common.gfx120x_swiglu", "swiglu_chunk", "chunk SwiGLU"),
    _op("alibi_bias", "attention", "kernels.attention.gfx120x_alibi_bias", "fill_alibi_bias", "ALiBi bias fill"),
)

_BY_NAME = {op.name: op for op in _OPS}
if len(_BY_NAME) != len(_OPS):
    raise RuntimeError("gfx120x capability names must be unique")


def catalog() -> tuple[Gfx120xOp, ...]:
    """Every gfx120x product call in this checkout. Does not read the device."""
    return _OPS


def available_for_arch(arch: str | None) -> tuple[Gfx120xOp, ...]:
    """Catalog when ``arch`` is gfx120x, otherwise empty. Never raises."""
    if not is_gfx120x_arch(arch):
        return ()
    return _OPS


def available(device=None) -> tuple[Gfx120xOp, ...]:
    """Catalog when ``device`` is gfx120x, otherwise empty. Never raises."""
    if not is_gfx120x(device):
        return ()
    return _OPS


def resolve(name: str):
    """Return the callable for ``name``. Does not launch and does not check the device."""
    op = _BY_NAME.get(name)
    if op is None:
        known = ", ".join(sorted(_BY_NAME))
        raise KeyError(f"unknown gfx120x capability {name!r}. Known: {known}")
    module = importlib.import_module(op.module)
    fn = getattr(module, op.attr, None)
    if not callable(fn):
        raise AttributeError(f"{op.module}.{op.attr} is not callable")
    return fn


__all__ = [
    "Gfx120xOp",
    "available",
    "available_for_arch",
    "catalog",
    "resolve",
]
