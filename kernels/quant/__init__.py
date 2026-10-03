# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""RDNA4 quantization and elementwise kernels (gfx120x).

| Module | Role |
|---|---|
| ``rdna4_fp8_quant`` | Per-tensor FP8 quant/dequant (e4m3 and e5m2 via ``e5m2=``) |
| ``rdna4_stoch_fp8`` | Stochastic FP8 rounding (same format flag) |
| ``rdna4_swiglu`` | Re-export of ``kernels.common.gfx120x_swiglu`` |
| ``rdna4_quantize_int8_rowwise`` | Rowwise absmax INT8 (rcp scale; no iu8 atom) |
| ``rdna4_quantize_int8_tensorwise`` | Tensorwise absmax INT8; host returns ``(q, scale)`` |
| ``rdna4_int8_convrot`` | Hadamard G in {16,64,256} + rowwise INT8; host ``int8_linear_convrot`` |
| ``rdna4_convrot_w4a4`` | Signed W4 ConvRot; **default** ``linear_dtype='int4'`` (native iu4); ``'int8'`` forces unpack→iu8 |
| ``rdna4_asym_w4a8`` | Grouped unsigned W4 + s_rel/s_channel (+ optional codebook) |
| ``rdna4_awq_w4a16`` | AWQ W4A16 dequant + fused ``gemv_awq_w4a16`` (group 64) |
| ``rdna4_svdquant_w4a4`` | SVDQuant W4A4 fused scaled_mm (host bf16 LoRA); defaults stay fused |
| ``rdna4_int4_codec`` | Shared signed/unsigned int4 pack/unpack + groupwise dequant |

Native iu4 GEMM lives in ``kernels.gemm.rdna4_iu4_gemm``. AWQ / SVDQuant
defaults stay on fused / unpack paths.
"""
