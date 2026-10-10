Pre-built kernels
=================

The FlyDSL repository includes a collection of pre-built GPU kernels in the
``kernels/`` directory, organized into subpackages (``gemm/``, ``norm/``,
``attention/``, ``moe/``, ``mma/``, ``common/``, ``comm/``, ``conv/``, ``quant/``).
These serve as both ready-to-use components and reference implementations for
kernel development.

The ``kernels`` tree is used from a source checkout; it is not installed by the
``flydsl`` wheel and is not covered by the stable Python API policy. Pin the
repository revision when integrating one of these entry points.

GEMM kernels
-------------

- ``kernels.gemm.preshuffle_gemm`` -- MFMA-based GEMM with LDS pipeline and pre-shuffled weights (FP8, INT8, FP16, BF16)
- ``kernels.gemm.mxfp4_preshuffle`` -- MXFP4 / FP4 (and f8f4) preshuffle GEMM
- ``kernels.gemm.fp4_gemm_4wave`` -- 4-wave FP4 GEMM (gfx950)
- ``kernels.gemm.rdna4_scaled_mm_fp8`` -- gfx120x FP8 scaled GEMM (e4m3×e4m3 or e5m2×e5m2)
- ``kernels.gemm.rdna4_scaled_mm_fp8_fused`` -- gfx120x fused activation quant + scaled FP8 GEMM, with host LoRA residuals
- ``kernels.gemm.rdna4_w8a16_linear`` -- gfx120x bf16/fp16 activations × int8, e4m3, or e5m2 weights
- ``kernels.gemm.rdna4_int8_linear`` -- gfx120x int8×int8 linear (iu8 WMMA)
- ``kernels.gemm.rdna4_int8_linear_fused`` -- gfx120x fused activation quant + int8 linear, with host LoRA residuals
- ``kernels.gemm.rdna4_iu4_gemm`` -- gfx120x packed int4 GEMM (iu4 WMMA)
- ``kernels.gemm.rdna4_mxfp8_block_gemm`` -- gfx120x MXFP8 block-scaled GEMM (e4m3 + E8M0)
- ``kernels.gemm.rdna4_mxfp4_block_gemm`` -- gfx120x MXFP4 block-scaled GEMM (packed E2M1 + E8M0)
- ``kernels.gemm.rdna4_fused_mlp_nmajor`` -- gfx120x fp16/bf16 SwiGLU MLP

MoE (Mixture-of-Experts) kernels
----------------------------------

- ``kernels.moe.moe_gemm_2stage`` -- fp8 MoE GEMM with 2-stage pipeline (stage1 gate-up +
  stage2 down-projection), gfx94*/gfx95*. Also provides the MoE reduction (sum over the topk
  dimension, ``Y[t, d] = sum(X[t, :, d])``), compiled via ``compile_moe_reduction()``.
- ``kernels.moe.mxfp_moe`` -- Fused a4w4 / a8w4 MoE 2-stage GEMM (device-side fp4 re-quant)

Paged attention
----------------

- ``kernels.attention.pa_decode_fp8`` -- Paged attention decode kernel with FP8 support

FlashAttention
--------------

- ``kernels.attention.flash_attn_gfx120x`` -- gfx120x bf16/fp16 FlashAttention (dense, varlen, paged, including vectorized)
- ``kernels.attention.flash_attn_fp8_gfx120x`` -- gfx120x fp8 and int8 FlashAttention. Int8 is ``flash_attn_int8_gfx120x``, which calls this kernel
- ``kernels.attention.flash_attn_gfx120x_host`` -- gfx120x host. ``flydsl_flash_attn_func`` routes a gfx120 device here
- ``kernels.attention.flash_attn_gfx120x_splitk`` -- gfx120x split-K combine (dense only)
- ``kernels.attention.gfx120x_alibi_bias`` -- ALiBi bias for the gfx120x host
- ``kernels.attention.gfx120x_attn_mask`` -- bool mask to additive bias for the gfx120x host
- ``kernels.attention.flash_attn_gfx120x_host`` -- gfx120x score mask, online softmax, and LSE, imported by the kernels

Normalization
-------------

- ``kernels.norm.layernorm_kernel`` -- Layer normalization
- ``kernels.norm.rmsnorm_kernel`` -- RMS normalization
- ``kernels.norm.rope_gfx120x`` -- gfx120x RoPE
- ``kernels.norm.rms_rope_gfx120x`` -- gfx120x RMSNorm + RoPE
- ``kernels.norm.adaln_gfx120x`` -- gfx120x AdaLN

Quantization
------------

- ``kernels.quant.rdna4_fp8_quant`` -- gfx120x per-tensor FP8 quant / ``dequantize_fp8`` (e4m3 and e5m2)
- ``kernels.quant.rdna4_stoch_fp8`` -- gfx120x stochastic FP8 cast (one-way, no stored scale)
- ``kernels.quant.rdna4_mxfp8_e8m0`` -- gfx120x MXFP8 quant / ``dequantize_mxfp8_device``
- ``kernels.quant.rdna4_mxfp4_e2m1`` -- gfx120x MXFP4 quant / ``dequantize_mxfp4_device``
- ``kernels.quant.rdna4_quantize_int8_rowwise`` -- gfx120x per-row int8 quant / ``dequantize_int8_rowwise``
- ``kernels.quant.rdna4_quantize_int8_tensorwise`` -- gfx120x per-tensor int8 quant / ``dequantize_int8_tensorwise``
- ``kernels.quant.rdna4_int8_convrot`` -- gfx120x int8 ConvRot quant / ``dequantize_int8_convrot_weight`` and the linear
- ``kernels.quant.rdna4_convrot_w4a4`` -- gfx120x ConvRot W4A4 quant / ``dequantize_convrot_w4a4_weight``. Default linear is packed int4
- ``kernels.quant.rdna4_asym_w4a8`` -- gfx120x asymmetric W4A8 quant / ``dequant_int4_grouped_to_int8``
- ``kernels.quant.rdna4_awq_w4a16`` -- gfx120x AWQ W4A16 ``dequant_awq_w4a16_weight`` and fused GEMV
- ``kernels.quant.rdna4_svdquant_w4a4`` -- gfx120x SVDQuant W4A4 quant / ``dequant_svdquant_w4a4_weight``
- ``kernels.quant.rdna4_int4_codec`` -- gfx120x int4 pack / ``dequant_int4_groupwise_signed`` and ``dequant_uint4_groupwise_awq``

Softmax
-------

- ``kernels.norm.softmax_kernel`` -- Numerically stable softmax

Utilities
---------

- ``kernels.common.kernels_common`` -- Shared constants and helper functions
- ``kernels.common.layout_utils`` -- Layout utility functions
- ``kernels.common.mma.mfma_preshuffle_pipeline`` -- B layout builder and XCD block remapping used by preshuffle GEMM and MoE kernels
- ``kernels.common.gfx120x_arch`` -- gfx120x arch check (``is_gfx120x``, ``require_gfx120x``)
- ``kernels.common.gfx120x_swiglu`` -- gfx120x SiLU and SwiGLU (``silu_mul``, ``swiglu_chunk``)
- ``kernels.common.gfx120x_autotune_tables`` -- gfx120x measured block-size tables

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
