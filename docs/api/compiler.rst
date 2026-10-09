Compiler and pipeline
=====================

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
       grid_x = (n + 255) // 256
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

Compilation flow
-----------------

On first call, ``@flyc.jit`` runs the following pipeline:

1. **AST rewriting**: The Python source is parsed and rewritten to emit MLIR ops.
2. **MLIR module construction**: The kernel body is traced into ``fly``, ``gpu``,
   ``arith``, ``scf``, ``memref``, and ``vector`` dialect ops.
3. **Fly pass pipeline**: The module is lowered through three pass stages,
   defined in ``RocmBackend._pipeline_parts()``
   (``python/flydsl/compiler/backends/rocm.py``). See
   :doc:`../architecture_guide` §3 for the per-pass table.

   A. ``pre_binary_fragments`` (Fly → ROCDL):

      - ``fly-rewrite-func-signature``
      - ``fly-canonicalize``
      - ``fly-layout-lowering``
      - ``fly-int-swizzle-simplify``
      - ``canonicalize``
      - ``fly-convert-atom-call-to-ssa-form``
      - ``fly-promote-regmem-to-vectorssa``
      - ``convert-fly-to-rocdl``
      - ``canonicalize``
      - ``gpu.module(convert-scf-to-cf, cse, convert-rocdl-fastmath-ops, convert-gpu-to-rocdl{chipset=gfxNNN ...}, fly-rocdl-cluster-attr)``

   B. ``binary_prep_fragments`` (→ LLVM):

      - ``rocdl-attach-target{chip=gfxNNN ...}``
      - ``convert-scf-to-cf``
      - ``convert-cf-to-llvm``
      - ``gpu-to-llvm{use-bare-pointers-...=true}``
      - ``convert-vector-to-llvm``
      - ``convert-arith-to-llvm``
      - ``convert-func-to-llvm``
      - ``reconcile-unrealized-casts``
      - ``ensure-debug-info-scope-on-llvm-func`` (optional, gated by ``FLYDSL_DEBUG_ENABLE_DEBUG_INFO``)

   C. ``binary_fragment``:

      - ``gpu-module-to-binary{format=fatbin opts="..."}``

4. **Cached artifact**: The compiled binary is cached to disk
   (``~/.flydsl/cache/``) keyed by the compiler toolchain hash and kernel
   type signature.

Ahead-of-time C export
-----------------------

``CompiledFunction.export_to_c(file_path, file_name, function_prefix="")``
writes a target-specific object file and a C/C++ header. The object contains
the GPU binary and a small backend adapter, so the deployed executable or
shared library does not need FlyDSL or Python. The generated header records the
backend and system link flags; for ROCm these are
``-lamdhip64 -pthread -ldl``. The linker must also be able to find the HIP SDK
library. For a nonstandard ROCm installation,
add its library directory, for example::

   cc -shared -o libkernel.so kernel.o -L/path/to/rocm/lib -lamdhip64 -pthread -ldl

The metadata records ``libamdhip64.so`` as a link-time name. The versioned
runtime dependency (the ELF ``DT_NEEDED`` name) is determined by the library
selected when the final executable or shared library is linked. The deployed
system's dynamic loader must be able to find that versioned HIP library.

The header exposes two calling styles:

* ``<symbol>__module_init``, ``<symbol>__module_load`` and
  ``<symbol>__module_unload`` provide explicit lifecycle and device control.
  After loading, ``<symbol>_call`` provides a typed wrapper around the packed
  entry point.
* ``<symbol>_call_auto`` is the simple path. It idempotently initializes the
  module and loads it on the current device before each call. It intentionally
  does not unload automatically, so asynchronous launches remain safe.

The current exporter targets 64-bit little-endian Linux ELF hosts and uses
GNU-compatible relocatable linking and ``objcopy`` tools. The ROCm adapter
requires HIP and pthread. Exported objects are tied to the host ABI and GPU
target used during compilation. They can be moved and linked independently,
but must run on a compatible backend and GPU architecture.

Tensor arguments
-----------------

Use ``flyc.from_dlpack`` to convert PyTorch tensors into FlyDSL tensor
descriptors with layout metadata:

.. code-block:: python

   import torch

   import flydsl.compiler as flyc

   A = torch.randn(1024, device="cuda", dtype=torch.float32)
   B = torch.empty_like(A)
   tA = flyc.from_dlpack(A).mark_layout_dynamic(
       leading_dim=0, divisibility=4
   )
   tB = flyc.from_dlpack(B)
   launch(tA, tB, A.numel(), stream=torch.cuda.Stream())

Precompilation and C export
---------------------------

``flyc.compile(launcher, *specialization_args, **specialization_kwargs)``
compiles a ``@flyc.jit`` launcher for one argument signature and returns a
``CompiledFunction``. The result is callable with new runtime values of the
same signature, and ``Constexpr`` values remain fixed to the specialization
used at compile time.

The same compiled specialization can be exported as a position-independent
host object with a generated C header:

.. code-block:: python

   from pathlib import Path

   output = Path("build/aot")
   output.mkdir(parents=True, exist_ok=True)

   compiled = flyc.compile(launch, tA, tB, A.numel())
   compiled.export_to_c(
       file_path=output,
       file_name="vector_add",
       function_prefix="flydsl_vector_add",
   )

``export_to_c`` writes ``vector_add.o`` and ``vector_add.h`` into the existing
output directory. The object embeds the FlyDSL backend adapter; link the final
binary against the backend system libraries listed in the header. The header
declares the packed entry point, typed inline call helpers, module
initialization/load/unload functions, and embedded ABI metadata. When
``function_prefix`` is omitted, ``file_name`` is also used as the exported C
symbol.

Passing live device arguments to ``flyc.compile`` preserves its normal initial
launch. For an offline or CPU-only build host, use null pointer wrappers created
with ``flyc.from_c_void_p`` (or compatible non-device tensor placeholders) and
set the target architecture explicitly, for example with ``ARCH=gfx950``.
Export currently supports tensor, pointer, scalar, structure, and stream
arguments whose lowered types have a supported C ABI. Unsupported launchers
report the export error when ``export_to_c`` is called.

ROCDL operations
-----------------

The ``flydsl.expr.rocdl`` module provides AMD-specific operations:

The universal exported entries follow :doc:`../api_stability`. The gfx1250
``WMMAScale`` and TDM helpers below are target-specific implementation APIs and
remain unstable.

- **fx.rocdl.make_buffer_tensor** -- create buffer resource descriptor from tensor (CDNA buffer copy)
- **fx.rocdl.BufferCopy32b** / **BufferCopy128b** -- buffer copy atoms
- **fx.rocdl.MFMA** -- MFMA instruction atoms (CDNA3/CDNA4; for example, ``MFMA(16, 16, 4, fx.Float32)``)
- **fx.rocdl.WMMA** / **fx.rocdl.WMMAScale** -- wave32 WMMA MMA atoms; ``WMMA`` is arch-dispatched (gfx11 / gfx120x RDNA4 / gfx1250), ``WMMAScale`` is the gfx1250 E8M0 MX-scaled form
- **fx.rocdl.cdna5.make_tiled_tdm_atom** / **fx.rocdl.cdna5.tdm_partition** -- gfx1250 TDM tiled copy atom (preferred for new kernels); built over the whole tensor, cut per warp, K-loop advanced by an i32 tile index on the coordinate rest
- **fx.rocdl.make_tdm_atom** / **fx.rocdl.TDM** -- gfx1250 TDM async Global↔LDS whole-tile copy atom (1–5D; base from the copy operand, per-dim extent/stride/imm_offset/mask as atom state)

fly-opt CLI
------------

The ``fly-opt`` tool is a command-line interface for running MLIR passes on
``.mlir`` files:

.. code-block:: bash

   fly-opt --fly-canonicalize input.mlir
   fly-opt --fly-layout-lowering input.mlir
   fly-opt --help
