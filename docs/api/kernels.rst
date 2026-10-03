Pre-built kernels
=================

The FlyDSL repository includes a collection of pre-built GPU kernels in the
``kernels/`` directory, organized into subpackages (``gemm/``, ``norm/``,
``attention/``, ``moe/``, ``mma/``, ``common/``, ``comm/``, ``conv/``).
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
- ``kernels.common.mma.mfma_preshuffle_pipeline`` -- B layout builder and XCD block remapping used by preshuffle GEMM and MoE kernels

gfx120x (RDNA4, wave32)
-----------------------

Not installed by the ``flydsl`` wheel. Same source-checkout rule as the rest of
``kernels/``. The shared FlashAttention router imports
``kernels.common.gfx120x_arch`` on every call, including gfx950 and gfx942,
and returns before those paths when the device is not gfx120x. It does not
import the gfx120x kernel modules on those arches.

- ``kernels.attention.flash_attn_gfx120x_host`` -- bf16/fp16, fp8, int8, and iu4 FlashAttention hosts
- ``kernels.gemm.rdna4_int8_linear_dispatch`` -- ``int8_linear_auto`` (default int8 linear)
- ``kernels.gemm.rdna4_scaled_mm_fp8_auto`` -- ``scaled_mm_fp8_auto``
- ``kernels.gemm.rdna4_scaled_mm_fp8`` -- ``build_scaled_mm_fp8_module`` (there is no ``scaled_mm_fp8()`` host)
- ``kernels.gemm.rdna4_iu4_gemm`` -- ``iu4_gemm``
- ``kernels.quant.rdna4_convrot_w4a4`` -- ConvRot W4A4, default ``linear_dtype="int4"``
- ``kernels.norm.rope_gfx120x`` / ``rms_rope_gfx120x`` / ``adaln_gfx120x`` -- RoPE, RMS+RoPE, AdaLN builders
- ``kernels.common.gfx120x_arch`` -- ``is_gfx120x`` (never raises) and ``require_gfx120x`` (gfx120x-only entries)

Who calls whom: :doc:`../gfx120x_call_graph`.

.. seealso:: :doc:`../prebuilt_kernels_guide` for detailed usage and configuration of each kernel.
