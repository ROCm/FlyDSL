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
- ``kernels.gemm.rdna4_scaled_mm_fp8`` -- gfx120x FP8 scaled GEMM (``scaled_mm_fp8``)
- ``kernels.gemm.rdna4_scaled_mm_fp8_fused`` -- gfx120x fused FP8 GEMM with LoRA (``scaled_mm_fp8_fused``)
- ``kernels.gemm.rdna4_w8a16_linear`` -- gfx120x wide activation, narrow weight (``w8a16_linear``)
- ``kernels.gemm.rdna4_int8_linear`` -- gfx120x int8 linear (``int8_linear``)
- ``kernels.gemm.rdna4_int8_linear_fused`` -- gfx120x fused int8 linear with LoRA (``int8_linear_fused``)
- ``kernels.gemm.rdna4_iu4_gemm`` -- gfx120x packed int4 GEMM (``iu4_gemm``)
- ``kernels.gemm.rdna4_mxfp8_block_gemm`` -- gfx120x MXFP8 block GEMM (``mxfp8_block_gemm``)
- ``kernels.gemm.rdna4_mxfp4_block_gemm`` -- gfx120x MXFP4 block GEMM (``mxfp4_block_gemm``)
- ``kernels.gemm.rdna4_fused_mlp_nmajor`` -- gfx120x bf16/fp16 GEMM and SwiGLU MLP (``fused_swiglu_mlp_nmajor``)

MoE (Mixture-of-Experts) kernels
----------------------------------

- ``kernels.moe.moe_gemm_2stage`` -- fp8 MoE GEMM with 2-stage pipeline (stage1 gate-up +
  stage2 down-projection), gfx94*/gfx95*. Also provides the MoE reduction (sum over the topk
  dimension, ``Y[t, d] = sum(X[t, :, d])``), compiled via ``compile_moe_reduction()``.
- ``kernels.moe.mxfp_moe`` -- Fused a4w4 / a8w4 MoE 2-stage GEMM (device-side fp4 re-quant)

Paged attention
----------------

- ``kernels.attention.pa_decode_fp8`` -- Paged attention decode kernel with FP8 support
- ``kernels.attention.flash_attn_gfx120x_host`` -- gfx120x FlashAttention launches. ``flydsl_flash_attn_func`` enters here when the device arch starts with ``gfx120``

Normalization
-------------

- ``kernels.norm.layernorm_kernel`` -- Layer normalization
- ``kernels.norm.rmsnorm_kernel`` -- RMS normalization
- ``kernels.norm.rope_gfx120x`` -- gfx120x RoPE (``build_rope_module``)
- ``kernels.norm.rms_rope_gfx120x`` -- gfx120x RMSNorm + RoPE (``build_rms_rope_module``)
- ``kernels.norm.adaln_gfx120x`` -- gfx120x AdaLN (``build_adaln_module``)

Softmax
-------

- ``kernels.norm.softmax_kernel`` -- Numerically stable softmax

Quantization (gfx120x)
----------------------

- ``kernels.quant.rdna4_fp8_quant`` -- per-tensor FP8 quant and ``dequantize_fp8``
- ``kernels.quant.rdna4_stoch_fp8`` -- stochastic FP8 cast (``stoch_fp8_direct``)
- ``kernels.quant.rdna4_mxfp8_e8m0`` -- MXFP8 quant and dequant
- ``kernels.quant.rdna4_mxfp4_e2m1`` -- MXFP4 quant and dequant
- ``kernels.quant.rdna4_quantize_int8_rowwise`` -- per-row int8 quant and dequant
- ``kernels.quant.rdna4_quantize_int8_tensorwise`` -- per-tensor int8 quant and dequant
- ``kernels.quant.rdna4_int8_convrot`` -- int8 ConvRot quant, dequant, and linear
- ``kernels.quant.rdna4_convrot_w4a4`` -- ConvRot W4A4 quant, dequant, and linear
- ``kernels.quant.rdna4_asym_w4a8`` -- asymmetric W4A8 quant, dequant, and linear
- ``kernels.quant.rdna4_awq_w4a16`` -- AWQ W4A16 dequant and GEMV
- ``kernels.quant.rdna4_svdquant_w4a4`` -- SVDQuant W4A4 quant, dequant, and linear
- ``kernels.quant.rdna4_int4_codec`` -- int4 pack and groupwise dequant (no GPU launch)

Utilities
---------

- ``kernels.common.kernels_common`` -- Shared constants and helper functions
- ``kernels.common.layout_utils`` -- Layout utility functions
- ``kernels.common.mma.mfma_preshuffle_pipeline`` -- B layout builder and XCD block remapping used by preshuffle GEMM and MoE kernels
- ``kernels.common.gfx120x_swiglu`` -- gfx120x SiLU and SwiGLU (``silu_mul``, ``swiglu_chunk``)
- ``kernels.common.gfx120x_pad`` -- gfx120x pad (``device_pad``)
- ``kernels.common.gfx120x_row_bias`` -- gfx120x row bias (``add_row_bias``, ``add_same``, ``mul_by_scale1``)

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
