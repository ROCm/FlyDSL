# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""RDNA4 quantization and elementwise kernels (gfx120x).

| Module | Role |
|---|---|
| ``rdna4_fp8_quant`` | Per-tensor FP8 quant/dequant (e4m3 and e5m2 via ``e5m2=``) |
| ``rdna4_stoch_fp8`` | Stochastic FP8 rounding (same format flag) |
| ``kernels.common.gfx120x_swiglu`` | SiLU×mul and chunk-2 SwiGLU (not a quantizer) |
| ``rdna4_quantize_int8_rowwise`` | Rowwise absmax INT8 (rcp scale; no iu8 atom) |
| ``rdna4_quantize_int8_tensorwise`` | Tensorwise absmax INT8; host returns ``(q, scale)`` |
| ``rdna4_int8_convrot`` | Hadamard G in {16,64,256} + rowwise INT8 |
| ``rdna4_convrot_w4a4`` | Signed W4 ConvRot; default native iu4, or int8 unpack |
| ``rdna4_asym_w4a8`` | Grouped unsigned W4 + s_rel/s_channel (+ optional codebook) |
| ``rdna4_awq_w4a16`` | AWQ W4A16 dequant + fused ``gemv_awq_w4a16`` (group 64) |
| ``rdna4_svdquant_w4a4`` | SVDQuant W4A4 fused scaled_mm (host bf16 LoRA); defaults stay fused |
| ``rdna4_int4_codec`` | Shared signed/unsigned int4 pack/unpack + groupwise dequant |
| ``rdna4_mxfp8_e8m0`` | Per-32 MXFP8 E4M3 quant/dequant with E8M0 scales |
| ``rdna4_mxfp4_e2m1`` | Per-32 MXFP4 E2M1 quant/dequant, two values per byte |

Native iu4 GEMM lives in ``kernels.gemm.rdna4_iu4_gemm``. AWQ / SVDQuant
defaults stay on fused / unpack paths.
"""
