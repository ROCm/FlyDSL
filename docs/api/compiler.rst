Compiler & Pipeline
===================

FlyDSL includes a JIT compiler that traces Python kernel functions into MLIR
and lowers them through the Fly dialect pipeline to GPU binaries.

``@flyc.kernel`` and ``@flyc.jit``
------------------------------------

The primary API for defining and compiling kernels:

.. code-block:: python

   import flydsl.compiler as flyc
   import flydsl.expr as fx

   @flyc.kernel
   def my_kernel(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int]):
       tid = fx.thread_idx.x
       bid = fx.block_idx.x
       # ... kernel body using layout ops ...

   @flyc.jit
   def launch(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int],
              stream: fx.Stream = fx.Stream(None)):
       my_kernel(A, B, n).launch(
           grid=(grid_x, 1, 1),
           block=(256, 1, 1),
           stream=stream,
       )

- ``@flyc.kernel`` compiles the function body into a ``gpu.func`` inside a
  ``gpu.module``. It uses AST rewriting to trace Python code into MLIR IR.
- ``@flyc.jit`` wraps a host-side function that constructs and launches kernels.
  On first call it triggers JIT compilation; subsequent calls with the same type
  signature use a cached compiled artifact.

Compilation Flow
-----------------

On first call, ``@flyc.jit`` runs the following pipeline:

1. **AST Rewriting**: The Python source is parsed and rewritten to emit MLIR ops.
2. **MLIR Module Construction**: Kernel body is traced into ``gpu``, ``arith``,
   ``scf``, ``memref``, and ``fly`` dialect ops.
3. **Fly Pass Pipeline**: The module is lowered through a series of MLIR passes:

   - ``gpu-kernel-outlining``
   - ``fly-canonicalize``
   - ``fly-layout-lowering``
   - ``convert-fly-to-rocdl``
   - ``canonicalize`` + ``cse``
   - ``gpu.module(convert-gpu-to-rocdl{...})``
   - ``rocdl-attach-target{chip=gfxNNN}``
   - ``gpu-to-llvm``, ``convert-arith/func-to-llvm``
   - ``gpu-module-to-binary{format=fatbin}``

4. **Cached Artifact**: The compiled binary is cached to disk
   (``~/.flydsl/cache/``) keyed by the compiler toolchain hash and kernel
   type signature.

Tensor Arguments
-----------------

Use ``flyc.from_dlpack`` to convert PyTorch tensors into FlyDSL tensor
descriptors with layout metadata:

.. code-block:: python

   import flydsl.compiler as flyc

   tA = flyc.from_dlpack(torch_tensor).mark_layout_dynamic(
       leading_dim=0, divisibility=4
   )
   launch(tA, B, n, stream=torch.cuda.Stream())

Buffer Operations
-----------------

The ``flydsl.expr.buffer_ops`` module provides high-level Python wrappers for
AMD CDNA3/CDNA4 buffer load/store operations. Buffer operations use a scalar
base pointer (SGPRs) and per-thread offsets for efficient global memory access
with hardware bounds checking.

ROCDL Operations
-----------------

The ``flydsl.expr.rocdl`` module provides AMD-specific operations:

- **fx.rocdl.make_buffer_tensor** -- create buffer resource descriptor from tensor
- **fx.rocdl.BufferCopy32b** / **BufferCopy128b** -- buffer copy atoms
- **fx.rocdl.MFMA** -- MFMA instruction atoms (e.g., ``MFMA(16, 16, 4, fx.Float32)``)

fly-opt CLI
------------

The ``fly-opt`` tool is a command-line interface for running MLIR passes on
``.mlir`` files:

.. code-block:: bash

   fly-opt --fly-canonicalize input.mlir
   fly-opt --fly-layout-lowering input.mlir
   fly-opt --help
