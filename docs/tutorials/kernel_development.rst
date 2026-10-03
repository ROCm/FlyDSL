Kernel development
==================

This tutorial covers advanced kernel development techniques in FlyDSL,
including tiled data movement, MFMA instructions, shared memory, and
performance optimization. The examples below are the CDNA / MFMA path
(``examples/03-tiledMma.py``). gfx120x is wave32 WMMA, not that path.
Who calls whom, and which example to copy, is :doc:`../gfx120x_call_graph`.

Tiled copies
-------------

FlyDSL uses a hierarchical tiling model to partition data across blocks,
warps, and threads:

.. literalinclude:: ../../examples/02-tiledCopy.py
   :language: python
   :start-at: import flydsl.compiler as flyc
   :end-before: @flyc.jit

See ``examples/02-tiledCopy.py`` for a complete working example.

MFMA instructions
-----------------

For matrix operations, FlyDSL supports AMD's Matrix Fused Multiply-Add (MFMA)
instructions via ``make_mma_atom`` and ``make_tiled_mma``:

.. literalinclude:: ../../examples/03-tiledMma.py
   :language: python
   :start-at: import flydsl.compiler as flyc
   :end-before: @flyc.jit

See ``examples/03-tiledMma.py`` for a complete GEMM example and
``kernels/gemm/preshuffle_gemm.py`` for a production GEMM implementation with
LDS pipeline.

Shared memory (LDS)
--------------------

FlyDSL provides explicit control over Local Data Share (LDS) allocation and
data movement:

1. Allocate LDS buffers with appropriate padding to avoid bank conflicts.
2. Use cooperative loads to fill LDS from global memory.
3. Synchronize with barriers before consuming LDS data.

See ``kernels/gemm/preshuffle_gemm.py`` for LDS double-buffering patterns.

Performance optimization
------------------------

Key optimization techniques demonstrated in the pre-built kernels:

- **LDS double-buffering**: Overlap compute with data movement (``preshuffle_gemm``)
- **Buffer tensor operations**: Hardware bounds-checked memory access
  (``fx.rocdl.make_buffer_tensor``)
- **Software pipelining**: Hide memory latency with multi-stage pipelines
- **Pre-shuffled weights**: Avoid runtime layout transformations for MFMA

Reference implementations
-------------------------

Study these kernels for real-world patterns:

- ``kernels/gemm/preshuffle_gemm.py`` -- MFMA + LDS pipeline GEMM
- ``kernels/norm/softmax_kernel.py`` -- online numerically stable softmax
- ``kernels/norm/layernorm_kernel.py`` -- fused normalization
- ``kernels/attention/pa_decode_fp8.py`` -- paged attention decode with FP8

.. seealso::

   - :doc:`../kernel_authoring_guide` -- comprehensive kernel authoring reference
   - :doc:`../prebuilt_kernels_guide` -- all pre-built kernels with configuration details
   - :doc:`../testing_benchmarking_guide` -- how to test and benchmark kernels
