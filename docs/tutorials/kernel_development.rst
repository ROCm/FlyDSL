Kernel Development
==================

This tutorial covers advanced kernel development techniques in FlyDSL,
including tiled data movement, MFMA instructions, shared memory, and
performance optimization.

Tiled Copies
-------------

FlyDSL uses a hierarchical tiling model to partition data across blocks,
warps, and threads:

.. code-block:: python

   # Define how threads are organized
   thr_layout = flir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
   val_layout = flir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

   # Create a copy atom with vectorized loads
   copy_atom = flir.make_copy_atom(T.f32(), vector_size=8)

   # Build the tiled copy descriptor
   tiled = flir.make_tiled_copy_tv(
       copy_atom, thr_layout, val_layout,
       thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N)
   )

   # Partition a tensor for this thread
   thr_tile = tiled.get_slice(tid).partition_S(blk_tile)

MFMA Instructions
-----------------

For matrix operations, FlyDSL supports AMD's Matrix Fused Multiply-Add (MFMA)
instructions:

- ``mfma_f32_16x16x16_f16`` -- 16x16x16 FP16 input, FP32 accumulate
- ``mfma_f32_32x32x8_f16`` -- 32x32x8 FP16 input, FP32 accumulate

See ``kernels/preshuffle_gemm.py`` for a complete GEMM implementation using
MFMA with an LDS pipeline.

Shared Memory (LDS)
--------------------

FlyDSL provides explicit control over Local Data Share (LDS) allocation and
data movement:

1. Allocate LDS buffers with appropriate padding to avoid bank conflicts
2. Use cooperative loads to fill LDS from global memory
3. Synchronize with barriers before consuming LDS data

See ``kernels/flash_attention_v3.py`` for cooperative vectorized tile loading
patterns.

Performance Optimization
------------------------

Key optimization techniques demonstrated in the pre-built kernels:

- **Q-in-registers**: Keep frequently accessed data in VGPRs (``flash_attention_v2``)
- **Causal early-exit**: Skip unnecessary computation blocks (``flash_attention_v2``)
- **Cooperative loads**: Vectorized tile loading across threads (``flash_attention_v3``)
- **LDS double-buffering**: Overlap compute with data movement (``preshuffle_gemm``)
- **Software pipelining**: Hide memory latency with multi-stage pipelines

Reference Implementations
-------------------------

Study these kernels for real-world patterns:

- ``kernels/preshuffle_gemm.py`` -- MFMA + LDS pipeline GEMM
- ``kernels/flash_attention_v3.py`` -- cooperative loads + vector operations
- ``kernels/softmax_kernel.py`` -- online numerically stable softmax
- ``kernels/layernorm_kernel.py`` -- fused normalization

.. seealso::

   - :doc:`../kernel_authoring_guide` -- comprehensive kernel authoring reference
   - :doc:`../prebuilt_kernels_guide` -- all pre-built kernels with configuration details
   - :doc:`../testing_benchmarking_guide` -- how to test and benchmark kernels
