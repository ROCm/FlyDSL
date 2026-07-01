Pre-built Kernels
=================

FlyDSL ships with a collection of pre-built GPU kernels in the ``kernels/``
directory. These serve as both ready-to-use components and reference
implementations for kernel development.

GEMM Kernels
-------------

- ``kernels.preshuffle_gemm`` -- MFMA-based GEMM with LDS pipeline and pre-shuffled weights (FP8, INT8, BF16)
- ``kernels.preshuffle_gemm_flyc`` -- Preshuffle GEMM using the new ``@flyc.kernel`` API
- ``kernels.mixed_preshuffle_gemm`` -- Mixed-precision GEMM with pre-shuffled layouts
- ``kernels.blockscale_preshuffle_gemm`` -- Block-scale (MXFP4) preshuffle GEMM

MoE (Mixture-of-Experts) Kernels
----------------------------------

- ``kernels.moe_gemm_2stage`` -- MoE GEMM with 2-stage pipeline (stage1 + stage2)
- ``kernels.mixed_moe_gemm_2stage`` -- Mixed-precision MoE GEMM
- ``kernels.moe_blockscale_2stage`` -- MoE with block-scale quantization (MXFP4)
- ``kernels.moe_reduce`` -- MoE reduction kernel: sums over the topk dimension
  (``Y[t, d] = sum(X[t, :, d])``). Supports optional masking, f16/bf16/f32,
  and is compiled via ``compile_moe_reduction()``.

Paged Attention
----------------

- ``kernels.pa_decode_fp8`` -- Paged attention decode kernel with FP8 support

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
- ``kernels.layout_utils`` -- Layout utility functions
- ``kernels.mfma_epilogues`` -- MFMA epilogue patterns (store, accumulate, scale)
- ``kernels.mfma_preshuffle_pipeline`` -- Shared MFMA preshuffle helpers (B layout builder, K32 pack loads) used by preshuffle GEMM and MoE kernels

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
