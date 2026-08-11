Pre-built kernels
=================

FlyDSL ships with a collection of pre-built GPU kernels in the ``kernels/``
directory, organized into subpackages (``gemm/``, ``norm/``, ``attention/``,
``moe/``, ``mma/``, ``common/``, ``comm/``, ``conv/``). These serve as both
ready-to-use components and reference implementations for kernel development.

GEMM kernels
-------------

- ``kernels.gemm.preshuffle_gemm`` -- MFMA-based GEMM with LDS pipeline and pre-shuffled weights (FP8, INT8, FP16, BF16)
- ``kernels.gemm.mxfp4_preshuffle`` -- MXFP4 / FP4 (and f8f4) preshuffle GEMM
- ``kernels.gemm.fp4_gemm_4wave`` -- 4-wave FP4 GEMM (gfx950)
- ``kernels.gemm.hgemm_splitk`` -- FP16 split-K GEMM

MoE (Mixture-of-Experts) kernels
----------------------------------

- ``kernels.moe.moe_gemm_2stage`` -- fp8 MoE GEMM with 2-stage pipeline (stage1 gate-up +
  stage2 down-projection), gfx94*/gfx95*. Also provides the MoE reduction (sum over the topk
  dimension, ``Y[t, d] = sum(X[t, :, d])``), compiled via ``compile_moe_reduction()``.
- ``kernels.moe.mxfp_moe`` -- Fused a4w4 / a8w4 MoE 2-stage GEMM (device-side fp4 re-quant)

Paged attention
----------------

- ``kernels.attention.pa_decode_fp8`` -- Paged attention decode kernel with FP8 support

Normalization
-------------

- ``kernels.norm.layernorm_kernel`` -- Layer normalization
- ``kernels.norm.rmsnorm_kernel`` -- RMS normalization

Softmax
-------

- ``kernels.norm.softmax_kernel`` -- Numerically stable softmax

Utilities
---------

- ``kernels.common.kernels_common`` -- Shared constants and helper functions
- ``kernels.common.layout_utils`` -- Layout utility functions
- ``kernels.mma.mfma_epilogues`` -- MFMA epilogue patterns (store, accumulate, scale)
- ``kernels.mma.mfma_preshuffle_pipeline`` -- Shared MFMA preshuffle helpers (B layout builder, K32 pack loads) used by preshuffle GEMM and MoE kernels

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
