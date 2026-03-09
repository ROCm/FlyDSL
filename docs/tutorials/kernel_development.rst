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

   import flydsl.expr as fx

   # Define thread and value layouts
   thr_layout = fx.make_layout((4, 1), (1, 1))
   val_layout = fx.make_layout((1, 8), (1, 1))

   # Create a copy atom (e.g., 128-bit buffer copy)
   copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

   # Build the tiled copy descriptor via raked product
   layout_thr_val = fx.raked_product(thr_layout, val_layout)
   tile_mn = fx.make_tile(4, 8)
   tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

   # Partition a tensor for this thread
   thr_copy = tiled_copy.get_slice(tid)
   partition_src = thr_copy.partition_S(block_tile)
   partition_dst = thr_copy.partition_D(fragment)

   # Execute copy
   fx.copy(copy_atom, partition_src, partition_dst)

See ``examples/02-tiledCopy.py`` for a complete working example.

MFMA Instructions
-----------------

For matrix operations, FlyDSL supports AMD's Matrix Fused Multiply-Add (MFMA)
instructions via ``make_mma_atom`` and ``make_tiled_mma``:

.. code-block:: python

   import flydsl.expr as fx

   # Create an MFMA atom (16x16x4 FP32)
   mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
   tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))

   # Partition A, B, C for this thread
   thr_mma = tiled_mma.thr_slice(tid)
   frag_A = thr_mma.make_fragment_A(partition_A)
   frag_B = thr_mma.make_fragment_B(partition_B)
   frag_C = thr_mma.make_fragment_C(partition_C)

   # Execute GEMM
   fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

See ``examples/03-tiledMma.py`` for a complete GEMM example and
``kernels/preshuffle_gemm.py`` for a production GEMM implementation with
LDS pipeline.

Shared Memory (LDS)
--------------------

FlyDSL provides explicit control over Local Data Share (LDS) allocation and
data movement:

1. Allocate LDS buffers with appropriate padding to avoid bank conflicts
2. Use cooperative loads to fill LDS from global memory
3. Synchronize with barriers before consuming LDS data

See ``kernels/preshuffle_gemm.py`` for LDS double-buffering patterns.

Performance Optimization
------------------------

Key optimization techniques demonstrated in the pre-built kernels:

- **LDS double-buffering**: Overlap compute with data movement (``preshuffle_gemm``)
- **Buffer tensor operations**: Hardware bounds-checked memory access (``rocdl.make_buffer_tensor``)
- **Software pipelining**: Hide memory latency with multi-stage pipelines
- **Pre-shuffled weights**: Avoid runtime layout transformations for MFMA

Reference Implementations
-------------------------

Study these kernels for real-world patterns:

- ``kernels/preshuffle_gemm.py`` -- MFMA + LDS pipeline GEMM
- ``kernels/preshuffle_gemm_flyc.py`` -- GEMM using new ``@flyc.kernel`` API
- ``kernels/softmax_kernel.py`` -- online numerically stable softmax
- ``kernels/layernorm_kernel.py`` -- fused normalization
- ``kernels/pa_decode_fp8.py`` -- paged attention decode with FP8

.. seealso::

   - :doc:`../kernel_authoring_guide` -- comprehensive kernel authoring reference
   - :doc:`../prebuilt_kernels_guide` -- all pre-built kernels with configuration details
   - :doc:`../testing_benchmarking_guide` -- how to test and benchmark kernels
