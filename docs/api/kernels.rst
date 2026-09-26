Repository kernel catalog
=========================

The ``kernels`` tree contains optimized operators, compiler-building blocks,
and integration shims used by the repository's tests and benchmarks. It is a
source-tree package, not part of the published ``flydsl`` wheel, and its module
paths are not covered by the ``flydsl`` API-stability policy.

Use these implementations in one of three ways:

1. run their matching tests/benchmarks as executable examples;
2. import a documented builder or high-level wrapper from a source checkout;
3. study the implementation when authoring a new kernel with the stable
   ``flydsl`` APIs.

Kernel families
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Family
     - Main modules
     - Capabilities
   * - GEMM
     - ``preshuffle_gemm``, ``fp8_gemm_4wave``, ``fp8_gemm_8wave``,
       ``mxfp8_gemm_8wave``, ``fp4_gemm_4wave``, ``mxfp4_preshuffle``,
       ``gemm_a16w16_gfx950``, ``gemm_*_gfx1250``, ``rdna*_gemm``
     - CDNA MFMA, RDNA WMMA, preshuffled weights, row/block scaling, split-K,
       and gfx1250 TDM/cluster pipelines.
   * - Normalization
     - ``layernorm_kernel``, ``rmsnorm_kernel``, ``rmsnorm_bwd_kernel``,
       ``softmax_kernel``, ``softmax_bwd_kernel``
     - Forward/backward LayerNorm, RMSNorm and softmax, fused residual paths,
       quantizing epilogues, and opt-in autotune adapters.
   * - Attention
     - ``flash_attn_interface``, ``flash_attn_*``, ``pa_decode_*``,
       ``pa_metadata``, ``mla_fwd_decode``, ``swa_gfx950``,
       ``fused_rope_cache_kernel``, ``qk_norm_rope_quant``
     - Dense/variable/paged FlashAttention, paged decode and work scheduling,
       MLA decode, sliding-window attention, RoPE/cache, and fused Q/K prep.
   * - MoE
     - ``moe_gemm_2stage``, ``moe_2stage_a16wmix``, ``mxfp_moe``,
       ``moe_sorting_kernel``, ``topk_gating_softmax_kernel``,
       ``moe_a8w4_mxscale_gfx1250``, ``mega_moe``
     - Routing/sorting, top-k gating, two-stage expert GEMMs, mixed low-precision
       formats, and fused multi-stage operators.
   * - Convolution
     - ``conv3d_implicit``, ``conv3d_implicit_fp8``, ``conv3d_autotune``
     - BF16 and FP8 implicit-GEMM 3D convolution with layout conversion and
       manual tile tuning.
   * - Communication
     - ``custom_all_reduce``, ``custom_all_reduce_kernel``,
       ``flydsl_dispatch_combine_intranode_*``
     - Multi-GPU all-reduce and intranode MoE dispatch/combine integration.
   * - Common building blocks
     - ``kernels.common`` and ``kernels.common.mma``
     - Tensor shims, buffer/memory helpers, DPP utilities, gfx1250 cluster
       helpers, activation functions, and shared MFMA pipeline code.

Entry-point conventions
-----------------------

The tree deliberately supports several integration layers:

- ``build_*`` and ``compile_*`` functions create a specialized launcher or
  module for explicit shapes/dtypes/configuration.
- ``launch_*`` functions are usually ``@flyc.jit`` launchers with constexpr
  tuning arguments.
- high-level functions such as ``conv3d_implicit``,
  ``flydsl_flash_attn_func``, and ``flydsl_qk_norm_rope_quant`` accept PyTorch
  tensors and manage specialization/caching for a concrete operator.
- utility modules and underscore-prefixed helpers are implementation details;
  do not depend on them without owning the corresponding integration.

The authoritative signature is always the implementation in the same
revision. Kernel interfaces change more frequently than the stable DSL because
they encode model layouts, architecture constraints, and active tuning work.

Testing and architecture support
--------------------------------

Use the matching file under ``tests/kernels`` to learn tensor layouts,
constraints, reference comparisons, supported dtypes, and launch arguments.
Architecture filtering is centralized in ``tests/arch_compat.py`` and the
kernel tests also perform finer-grained skips/fail-fast validation.

Examples:

.. code-block:: bash

   python -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
   python -m pytest tests/kernels/test_flash_attn_fwd.py -m "not large_shape"
   python -m pytest tests/kernels/test_conv3d_implicit.py -m "not large_shape"
   python -m pytest tests/kernels/test_allreduce.py -m multi_gpu

See :doc:`../prebuilt_kernels_guide` for the detailed module/entry-point/test
matrix and :doc:`../testing_benchmarking_guide` for the supported runners.
