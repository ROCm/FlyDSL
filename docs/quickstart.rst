Quick Start
===========

This guide walks through writing, compiling, and running a simple GPU kernel
with FlyDSL.

A Minimal Vector Add Kernel
----------------------------

The following example demonstrates the core FlyDSL workflow: define a kernel
class, use tiled copies and layout algebra, then compile and execute on the GPU.

.. code-block:: python

   import flydsl
   from flydsl.dialects.ext import flir
   import _mlir.extras.types as T

   THREADS, TILE, VEC = 256, 8, 4

   class VecAddKernel(flir.MlirModule):
       GPU_MODULE_NAME = "vec_kernels"
       GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

       @flir.kernel
       def vec_add(self: flir.T.i64,
                   A: lambda: T.memref(T.dynamic(), T.f32()),
                   B: lambda: T.memref(T.dynamic(), T.f32()),
                   C: lambda: T.memref(T.dynamic(), T.f32()),
                   n: lambda: T.index()):
           tid = flir.thread_idx("x")
           bid = flir.block_idx("x")

           # Define thread/value layouts for tiled copy
           thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
           val_layout = flir.make_ordered_layout((TILE,), order=(0,))
           copy_atom = flir.make_copy_atom(T.f32(), vector_size=VEC)
           tiled = flir.make_tiled_copy_tv(
               copy_atom, thr_layout, val_layout,
               thr_shape=(THREADS,), val_shape=(TILE,)
           )

           # Partition tensors across blocks and threads
           tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
           tiles_A = flir.zipped_divide(tensor_A, (THREADS * TILE,))
           blkA = tiles_A[(bid,)]
           thrA = tiled.get_slice(tid).partition_S(blkA)

           # Load to registers, compute, store
           frgA = flir.make_fragment_like(thrA, T.f32())
           flir.copy(tiled, thrA, frgA)
           # ... repeat for B, add, store back to C

   # Compile and run
   module = VecAddKernel().module
   exe = flydsl.compile(module)
   exe(a_dev, b_dev, c_dev, size)

See ``tests/kernels/test_vec_add.py`` for the complete implementation with
benchmarking.

Key Concepts
------------

1. **MlirModule**: Base class for kernel definitions. Sets GPU target and module name.
2. **@flir.kernel**: Decorator that compiles a Python method into GPU IR.
3. **Layout algebra**: ``make_ordered_layout``, ``make_tiled_copy_tv``,
   ``zipped_divide`` -- express data partitioning across the GPU hierarchy.
4. **Pipeline**: Composable MLIR pass pipeline for lowering to GPU binary.
5. **compile/Executor**: JIT-compile and launch kernels on AMD GPUs.

Compilation Pipeline
--------------------

FlyDSL provides a composable pipeline API for lowering kernels:

.. code-block:: python

   from flydsl.compiler.pipeline import Pipeline

   pipeline = (
       Pipeline()
       .flir_to_standard()
       .canonicalize()
       .cse()
       .rocdl_attach_target(chip="gfx942")
       .Gpu(Pipeline().convert_gpu_to_rocdl(runtime="HIP"))
       .gpu_to_llvm()
       .lower_to_llvm()
       .gpu_module_to_binary(format="bin")
   )

   binary_module = pipeline.run(module)

AOT Pre-compilation
--------------------

FlyDSL supports ahead-of-time (AOT) compilation of kernels for deployment
without JIT overhead. The ``tests/python/examples/aot_example.py`` script
demonstrates pre-compiling MoE kernels into a cache directory:

.. code-block:: bash

   # Pre-compile with default configurations (auto-detect GPU arch)
   python tests/python/examples/aot_example.py

   # Cross-compile for a specific arch (no GPU needed)
   FLYDSL_COMPILE_ONLY=1 FLYDSL_TARGET_ARCH=gfx942 python tests/python/examples/aot_example.py

   # Custom cache directory
   FLIR_CACHE_DIR=/my/cache python tests/python/examples/aot_example.py

At runtime, compiled kernels are loaded from the cache automatically when
``FLIR_CACHE_DIR`` is set.

Next Steps
----------

- :doc:`kernel_authoring_guide` -- detailed guide to writing GPU kernels
- :doc:`layout_system_guide` -- deep dive into the FLIR layout algebra
- :doc:`prebuilt_kernels_guide` -- available pre-built kernels (GEMM, Flash Attention, etc.)
- :doc:`architecture_guide` -- compilation pipeline and project architecture
