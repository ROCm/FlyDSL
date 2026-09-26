Compiler API
============

``flydsl.compiler`` is conventionally imported as ``flyc``. It turns traced
Python launchers and kernels into target binaries, manages specialization and
caching, and provides the supported host-boundary adapters.

.. code-block:: python

   import flydsl.compiler as flyc
   import flydsl.expr as fx

Top-level API
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - API
     - Returns
     - Purpose
   * - ``kernel(func=None, *, name=None, known_block_size=None)``
     - ``KernelFunction``
     - Decorate a GPU kernel. Calling it inside a JIT launcher produces a
       pending ``KernelLauncher``; ``.launch(...)`` emits the kernel and launch.
   * - ``jit(func=None)``
     - ``JitFunction``
     - Decorate a host launcher that is traced, specialized, compiled, cached,
       and executed on demand.
   * - ``compile(launcher, *example_args, **example_kwargs)``
     - ``CompiledFunction`` or ``None`` in compile-only mode
     - Compile one specialization eagerly and return a low-overhead
       positional-only callable for repeated launches.
   * - ``compile_aot(launcher, *example_args, **example_kwargs)``
     - ``AOTCompiledFunction``
     - Trace and lower without loading or launching the GPU module; the result
       can be exported as a linkable host object and C header.
   * - ``from_dlpack(tensor, *, assumed_align=None, use_32bit_stride=False)``
     - DLPack tensor adapter
     - Preserve shape/stride metadata for any DLPack-compatible tensor.
   * - ``from_torch_tensor(tensor, *, assumed_align=None, use_32bit_stride=False)``
     - PyTorch tensor adapter
     - Explicit form of the built-in PyTorch argument conversion.
   * - ``from_c_void_p(element_type, pointer, *, address_space=..., assumed_align=None)``
     - Typed pointer adapter
     - Pass an integer or ``ctypes.c_void_p`` as ``fx.Pointer``.
   * - ``JitArgumentRegistry``
     - Registry class
     - Register host Python types and already-adapted argument classes.
   * - ``GPUTarget`` / ``BaseBackend``
     - Target description / backend base class
     - Extension interfaces for compile backends.
   * - ``get_backend``, ``register_backend``, ``compile_backend_name``
     - Backend registry helpers
     - Resolve or extend the selected compiler backend.

Kernel and launch definitions
-----------------------------

.. code-block:: python

   @flyc.kernel(name="copy_f32", known_block_size=(256, 1, 1))
   def copy_kernel(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32):
       i = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
       if i < n:
           dst[i] = src[i]

   @flyc.jit
   def copy(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32,
            stream: fx.Stream = fx.Stream(None)):
       copy_kernel(src, dst, n).launch(
           grid=((n + 255) // 256, 1, 1),
           block=(256, 1, 1),
           stream=stream,
       )

``name`` controls the profiler-visible kernel name. ``known_block_size`` is a
contract for dynamic launch dimensions; static integer block dimensions are
normally inferred automatically.

``KernelLauncher.launch`` accepts ``grid``, ``block``, and ``stream`` plus
backend compile/launch controls such as dynamic shared memory and unit/value
attributes. The exact accepted keywords are validated by the launcher; use the
ordinary ``grid=(x, y, z)`` and ``block=(x, y, z)`` form unless an architecture
feature requires more.

Specialization and ``compile``
------------------------------

A JIT specialization is determined by the launcher source/dependencies,
backend/toolchain, compile-affecting environment, argument DSL types/layout
metadata, and ``Constexpr`` values. Runtime scalar values such as ``fx.Int32``
do not create a new specialization merely because their value changes.

``flyc.compile`` performs the first compile up front:

.. code-block:: python

   fast_copy = flyc.compile(copy, x, y, x.numel(), stream)

   # The returned callable is positional-only and must keep the same runtime
   # argument count/order. Constexpr values remain baked into the specialization.
   fast_copy(x2, y2, x2.numel(), stream)

Compile hints can be attached with subscription syntax:

.. code-block:: python

   compiled = flyc.compile[{"fastmath": "contract", "waves_per_eu": 2}](
       copy, x, y, x.numel(), stream
   )

``flyc.compile[hints](launcher)`` returns the hinted launcher without compiling
yet, which is useful when another layer owns the example arguments.

Argument conversion
-------------------

Plain Python/PyTorch arguments are converted through ``JitArgumentRegistry``:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Host value
     - DSL annotation
     - Notes
   * - ``bool`` / ``int`` / ``float``
     - ``Boolean`` / ``Int32`` / ``Float32``
     - Default scalar mappings; use explicit DSL annotations/adapters when a
       different width is required.
   * - ``torch.Tensor``
     - ``Tensor``
     - Uses data pointer, shape, stride, dtype, and alignment metadata.
   * - ``torch.cuda.Stream``
     - ``Stream``
     - PyTorch uses its CUDA-named API for both CUDA and ROCm builds.
   * - ``from_dlpack(...)``
     - ``Tensor``
     - Works with other DLPack producers and supports dynamic-layout marking.
   * - ``from_c_void_p(...)``
     - ``Pointer``
     - Carries element type, address space, and optional alignment.
   * - ``Constexpr[T]``
     - Compile-time Python value
     - Included in the specialization/cache key and not passed at runtime.

Custom host types can register either a DSL type conversion or a complete
``JitArgument`` adapter. The public protocol helpers live in
``flydsl.compiler.protocol``; see :doc:`../language/dsl_protocols` for
``DslType``, ``JitArgument``, and ``Storable`` contracts.

Backends and targets
--------------------

``GPUTarget`` contains ``backend``, ``arch``, and ``warp_size``. The built-in
backend id is ``rocm``. ``get_backend(name=None, arch="")`` resolves a backend
using explicit arguments, then ``FLYDSL_COMPILE_BACKEND`` and ``ARCH``/device
detection.

Third-party backends can either call ``register_backend(name, backend_cls)`` or
publish an entry point in the ``flydsl.backends`` group. A backend supplies
target detection, MLIR pipeline fragments, target attributes, runtime library
names, cache fingerprint inputs, and optional AOT hooks. If it uses a custom
device runtime, register the matching runtime mapping described in
:doc:`runtime`.

AOT and C export
----------------

``compile_aot`` compiles without creating an execution engine or launching a
kernel:

.. code-block:: python

   compiled = flyc.compile_aot(copy, x, y, x.numel(), stream)
   result = compiled.export_to_c(
       file_path="build",
       file_name="copy",
       function_prefix="my_copy",
   )

``CompiledFunction.export_to_c`` exports the same already-compiled
specialization. The result describes the C ABI and generated object/header;
runtime libraries and link flags are available through
``flydsl.runtime.find_runtime_libraries`` and:

.. code-block:: bash

   python -m flydsl.compiler.aot_config --ldflags --libs

See :doc:`../aot_export_guide` for object naming, ABI slots, lifecycle,
status codes, CPU-only cross-compilation, and limitations.

Compilation flow
----------------

On a cache miss, a JIT call performs:

1. signature binding and host-argument conversion;
2. Python AST rewriting for dynamic control flow;
3. tracing into Fly and upstream MLIR dialects;
4. Fly-to-ROCDL and LLVM lowering through the selected backend;
5. GPU binary generation and cache persistence;
6. execution-engine/module initialization and launch.

The exact pass fragments are defined in
``python/flydsl/compiler/backends/rocm.py`` and included directly in
:doc:`../architecture_guide`, which is the source for pipeline details.

Command-line tools
------------------

``fly-opt`` runs Fly/MLIR passes on textual modules:

.. code-block:: bash

   fly-opt --fly-canonicalize input.mlir
   fly-opt --fly-layout-lowering input.mlir
   fly-opt --help

``flydsl-lsp-server`` is an MLIR language server with the Fly and selected
backend dialects registered. It supports ``.mlir`` files; it is not a Python
language server for the DSL.

Environment and diagnostics
---------------------------

Compilation, cache, debug, target, and runtime environment variables are
listed in :doc:`runtime`. DSL compile failures preserve user source locations
and show a filtered Python-style diagnostic by default; set
``FLYDSL_DEBUG_SHOW_STACKTRACE=1`` when the internal traceback is needed.
