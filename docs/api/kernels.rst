Pre-built Kernels
=================

FlyDSL ships with a collection of pre-built GPU kernels in the ``kernels/``
directory. These serve as both ready-to-use components and reference
implementations for kernel development.

Flash Attention
---------------

Multiple versions of Flash Attention with progressive optimizations:

- ``kernels.flash_attention`` -- V1: baseline tiled attention
- ``kernels.flash_attention_v2`` -- V2: Q-in-registers, LDS optimizations
- ``kernels.flash_attention_v3`` -- V3: cooperative loads, vector operations
- ``kernels.flash_attention_v5`` -- V5: advanced optimizations

GEMM Kernels
-------------

- ``kernels.preshuffle_gemm`` -- MFMA-based GEMM with LDS pipeline and pre-shuffled weights
- ``kernels.mixed_preshuffle_gemm`` -- Mixed-precision GEMM with pre-shuffled layouts
- ``kernels.moe_gemm_2stage`` -- Mixture-of-Experts GEMM with 2-stage pipeline (stage1 + stage2)
- ``kernels.mixed_moe_gemm_2stage`` -- Mixed-precision MoE GEMM

MoE Reduction
--------------

- ``kernels.moe_reduce`` -- MoE reduction kernel: sums over the topk dimension
  (``Y[t, d] = sum(X[t, :, d])``). Supports optional masking, f16/bf16/f32,
  and is compiled via ``compile_moe_reduction()``.

Normalization
-------------

- ``kernels.layernorm_kernel`` -- Layer normalization
- ``kernels.rmsnorm_kernel`` -- RMS normalization

Softmax
-------

- ``kernels.softmax_kernel`` -- Numerically stable softmax

Reduction
---------

- ``kernels.reduce`` -- Warp-level reduction utilities (``warp_reduce_sum``, ``warp_reduce_max``)

Utilities
---------

- ``kernels.kernels_common`` -- Shared constants and helper functions
- ``kernels.mfma_epilogues`` -- MFMA epilogue patterns (store, accumulate, scale)
- ``kernels.mfma_preshuffle_pipeline`` -- Shared MFMA preshuffle helpers (B layout builder, K32 pack loads) used by preshuffle GEMM and MoE kernels

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
