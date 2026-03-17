Quick Start
===========

This guide walks through writing, compiling, and running a simple GPU kernel
with FlyDSL.

A Minimal Vector Add Kernel
----------------------------

The following example demonstrates the core FlyDSL workflow: define a kernel
with ``@flyc.kernel``, use layout algebra to partition data, then launch with
``@flyc.jit``.

.. code-block:: python

   import torch
   import flydsl.compiler as flyc
   import flydsl.expr as fx

   @flyc.kernel
   def vectorAddKernel(
       A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
       block_dim: fx.Constexpr[int],
   ):
       bid = fx.block_idx.x
       tid = fx.thread_idx.x

       # Partition tensors by block using layout algebra
       tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
       tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
       tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

       tA = fx.slice(tA, (None, bid))
       tB = fx.slice(tB, (None, bid))
       tC = fx.slice(tC, (None, bid))

       # Allocate register fragments, load, compute, store
       RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1),
                                 fx.AddressSpace.Register)
       copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
       rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
       rB = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
       rC = fx.memref_alloca(RABTy, fx.make_layout(1, 1))

       fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
       fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

       vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
       fx.memref_store_vec(vC, rC)
       fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

   @flyc.jit
   def vectorAdd(
       A: fx.Tensor, B: fx.Tensor, C,
       n: fx.Int32,
       const_n: fx.Constexpr[int],
       stream: fx.Stream = fx.Stream(None),
   ):
       block_dim = 64
       grid_x = (n + block_dim - 1) // block_dim
       vectorAddKernel(A, B, C, block_dim).launch(
           grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream,
       )

   # Usage
   n = 128
   A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
   B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
   C = torch.zeros(n, dtype=torch.float32).cuda()
   vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())
   torch.cuda.synchronize()
   print("Result correct:", torch.allclose(C, A + B))

See ``examples/01-vectorAdd.py`` for the complete implementation with
CUDA Graph capture support.

Key Concepts
------------

1. **@flyc.kernel**: Decorator that compiles a Python function into GPU IR.
2. **@flyc.jit**: Decorator for host-side JIT functions that launch kernels.
3. **fx.Tensor / fx.Constexpr**: Type annotations for kernel/JIT arguments.
4. **Layout algebra**: ``make_layout``, ``logical_divide``, ``zipped_divide``,
   ``raked_product`` -- express data partitioning across the GPU hierarchy.
5. **Copy atoms**: ``make_copy_atom``, ``make_tiled_copy`` -- vectorized data
   movement with layout-aware partitioning.
6. **Automatic caching**: Compiled kernels are cached to disk
   (``~/.flydsl/cache/``) and reused on subsequent calls.

Compilation Pipeline
--------------------

On first call, ``@flyc.jit`` traces the Python function into an MLIR module,
then compiles it through the Fly MLIR pipeline:

.. code-block:: text

   Python Function (@flyc.kernel / @flyc.jit)
           │
           ▼  AST Rewriting + Tracing
      MLIR Module (gpu, arith, scf, memref dialects)
           │
           ▼  MlirCompiler.compile()
      ┌────────────────────────────────────────────────┐
      │  gpu-kernel-outlining                          │
      │  fly-canonicalize                              │
      │  fly-layout-lowering                           │
      │  convert-fly-to-rocdl                          │
      │  canonicalize + cse                            │
      │  gpu.module(convert-gpu-to-rocdl{...})         │
      │  rocdl-attach-target{chip=gfxNNN}              │
      │  gpu-to-llvm → convert-arith/func-to-llvm      │
      │  gpu-module-to-binary{format=fatbin}           │
      └────────────────────────────────────────────────┘
           │
           ▼
      Cached Compiled Artifact (ExecutionEngine)

AOT Pre-compilation
--------------------

FlyDSL supports ahead-of-time (AOT) compilation of kernels for deployment
without JIT overhead. The ``tests/python/examples/aot_example.py`` script
demonstrates pre-compiling preshuffle GEMM kernels into a cache directory:

.. code-block:: bash

   # Pre-compile with default configurations (auto-detect GPU arch)
   python tests/python/examples/aot_example.py

   # Pre-compile and verify by running kernels on GPU
   python tests/python/examples/aot_example.py --run_kernel

   # Custom cache directory
   FLYDSL_RUNTIME_CACHE_DIR=/my/cache python tests/python/examples/aot_example.py

At runtime, compiled kernels are loaded from the cache automatically when
``FLYDSL_RUNTIME_CACHE_DIR`` is set.

Next Steps
----------

- :doc:`kernel_authoring_guide` -- detailed guide to writing GPU kernels
- :doc:`layout_system_guide` -- deep dive into the Fly layout algebra
- :doc:`prebuilt_kernels_guide` -- available pre-built kernels (GEMM, MoE, etc.)
- :doc:`architecture_guide` -- compilation pipeline and project architecture
