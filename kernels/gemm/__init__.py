# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""FlyDSL GEMM kernels.

gfx120x / RDNA4 entries in this tree:

* rdna4_scaled_mm_fp8 -- tensorwise FP8 e4m3fn / e5m2 scaled matrix multiply
* rdna4_scaled_mm_fp8_auto -- DEFAULT tile picker for the same kernel (EXPERIMENTAL
  measured breakpoints; prefer this over plain scaled_mm_fp8 on unknown shapes)
* rdna4_scaled_mm_fp8_fused -- fused activation quant + scaled_mm; multi-LoRA
  adapters stay as host bf16 residuals
* rdna4_int8_linear -- iu8 WMMA: int8 activations x int8 weights -> bf16/fp16
* rdna4_int8_linear_fused -- fused act-quant + iu8; multi-LoRA as host residuals
* rdna4_w8a16_linear -- float WMMA: int8 / FP8 e4m3fn/e5m2 weights, bf16/fp16 acts
* rdna4_int8_linear_dispatch / rdna4_int8_linear_auto -- DEFAULT EXPERIMENTAL
  size dispatcher (W8A16 vs iu8: prefer iu8 when K >= 256; legacy M/MK overrides default off)
* rdna4_iu4_gemm -- native gfx120x iu4 WMMA GEMM (scalar i32 A/B); prefer native
  when K % 16 == 0; prefer_native=False forces unpack->iu8
* rdna4_fused_mlp_nmajor -- zero-LDS N-major / fused_gemm_TN + in-register SwiGLU

Measured size/tile defaults: kernels/common/gfx120x_autotune_tables.py.
Call docs: docs/prebuilt_kernels_guide.md (gfx120x sections).
"""
