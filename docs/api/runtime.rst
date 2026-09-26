Runtime, autotuning, and configuration
======================================

This page covers host-side runtime integration, target selection, JIT cache
controls, and autotuning. These APIs execute as ordinary Python; they are not
expression builders unless explicitly passed into a traced launcher.

Runtime API
-----------

``flydsl.runtime`` exports the device-runtime interfaces and AOT library
resolver:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Purpose
   * - ``get_device_runtime()``
     - Return the process-wide runtime implementation selected by
       ``FLYDSL_RUNTIME_KIND``.
   * - ``DeviceRuntime`` / ``RocmDeviceRuntime``
     - Abstract runtime contract and built-in HIP/ROCm implementation.
   * - ``register_device_runtime(cls, *, kind=None, force=False)``
     - Register a process-wide runtime implementation before first use.
   * - ``register_compile_runtime_mapping(compile_backend, runtime_kind)``
     - Associate a custom compiler backend id with a runtime kind.
   * - ``ensure_compile_runtime_compatible`` /
       ``ensure_compile_runtime_pairing_from_env``
     - Validate that compiler and runtime selections agree.
   * - ``find_runtime_libraries(backend=None)``
     - Return ``RuntimeLibrary(path, soname)`` records for the FlyDSL-shipped
       libraries required by an AOT-exported host object, including shipped
       transitive dependencies.

Only the paths explicitly identified by :doc:`../api_stability` carry a stable
compatibility guarantee. Runtime registration interfaces are primarily for
backend integrators and can evolve with the backend protocol.

Device helpers
--------------

``flydsl.runtime.device`` contains architecture helpers used by kernels and
build tooling:

- ``get_rocm_arch()`` returns a lower-case ``gfx*`` target. It checks ``ARCH``,
  ``FLYDSL_GPU_ARCH``, and ``HSA_OVERRIDE_GFX_VERSION`` before hardware
  detection.
- ``is_rdna_arch(arch=None)`` classifies gfx10, gfx11, and gfx120 targets as
  RDNA. It intentionally does not classify the CDNA5 ``gfx1250`` target as
  RDNA.
- ``get_warp_size(arch=None)`` returns 64 for gfx9 CDNA targets and 32 for
  gfx10/gfx11/gfx12 targets, including wave32 ``gfx1250``.
- ``get_rocm_device_count()`` performs best-effort visible-device discovery.
- ``check_smem_capacity(bytes, arch)`` validates known per-target LDS limits.

The first two paths above are explicitly stable; the remaining helpers are
useful implementation utilities and should not be treated as a release
contract unless added to the stability policy.

Environment variables
---------------------

The typed definitions in ``python/flydsl/utils/env.py`` are the source of truth
for core configuration.

.. list-table:: Compilation
   :header-rows: 1
   :widths: 34 16 50

   * - Variable
     - Default
     - Meaning
   * - ``FLYDSL_COMPILE_BACKEND``
     - ``rocm``
     - Compiler backend id.
   * - ``FLYDSL_COMPILE_OPT_LEVEL``
     - ``2``
     - Optimization level from 0 through 3.
   * - ``FLYDSL_COMPILE_LLVM_DIR``
     - empty
     - External LLVM/MLIR prefix used for final device code generation.
   * - ``ARCH``
     - detected
     - Explicit target architecture such as ``gfx942`` or ``gfx950``.
   * - ``COMPILE_ONLY``
     - false
     - Compile and populate caches without creating an execution engine or
       launching work.

.. list-table:: Debugging
   :header-rows: 1
   :widths: 38 16 46

   * - Variable
     - Default
     - Meaning
   * - ``FLYDSL_DUMP_IR``
     - false
     - Write numbered IR stages for each compilation.
   * - ``FLYDSL_DUMP_DIR``
     - ``~/.flydsl/debug``
     - Root directory for IR/ISA dumps.
   * - ``FLYDSL_DEBUG_DUMP_ASM``
     - false
     - Write generated assembly.
   * - ``FLYDSL_DEBUG_AST_DIFF``
     - false
     - Print the Python AST rewrite diff.
   * - ``FLYDSL_DEBUG_PRINT_ORIGIN_IR``
     - false
     - Print the module before lowering.
   * - ``FLYDSL_DEBUG_PRINT_AFTER_ALL``
     - false
     - Print IR after every MLIR pass.
   * - ``FLYDSL_DEBUG_ENABLE_DEBUG_INFO``
     - false
     - Request line-table debug information in lowered LLVM functions.
   * - ``FLYDSL_DEBUG_ENABLE_VERIFIER``
     - true
     - Verify IR between compilation stages.
   * - ``FLYDSL_DEBUG_SHOW_STACKTRACE``
     - false
     - Show the unfiltered internal traceback for DSL compile errors.
   * - ``FLYDSL_DEBUG_MAX_LOC_DEPTH``
     - ``5``
     - Maximum user call-site depth stored in nested source locations.
   * - ``FLYDSL_DEBUG_LOG_LEVEL``
     - ``WARNING``
     - ``DEBUG``, ``INFO``, ``WARNING``, or ``ERROR``.
   * - ``FLYDSL_DEBUG_LOG_TO_FILE`` /
       ``FLYDSL_DEBUG_LOG_TO_CONSOLE``
     - empty / false
     - Optional log destination controls.

.. list-table:: Runtime and cache
   :header-rows: 1
   :widths: 38 20 42

   * - Variable
     - Default
     - Meaning
   * - ``FLYDSL_RUNTIME_KIND``
     - ``rocm``
     - Device-runtime kind; it must match the compiler backend mapping.
   * - ``FLYDSL_RUNTIME_CACHE_DIR``
     - ``~/.flydsl/cache``
     - Persistent JIT/AOT cache directory.
   * - ``FLYDSL_RUNTIME_ENABLE_CACHE``
     - true
     - Enable persistent cache reads/writes. In-process compiled artifacts are
       still retained when this is false.
   * - ``FLYDSL_RUNTIME_RUN_ONLY``
     - false
     - Refuse compilation and load only an existing AOT/JIT cache entry.
   * - ``FLYDSL_EXTRA_SOURCE_DIRS``
     - empty
     - Additional source trees included in the compiler fingerprint when used
       by integration code.

Architecture helper overrides ``FLYDSL_GPU_ARCH`` and
``HSA_OVERRIDE_GFX_VERSION`` are fallback inputs to ``get_rocm_arch``; ``ARCH``
is the preferred explicit compile target.

Cache behavior
--------------

The JIT cache key includes the function source and referenced dependencies,
argument specialization, compiler/backend fingerprint, target, compile hints,
and cache-invalidating environment values. Python-only edits to dependencies
normally invalidate their specializations automatically. C++ pass changes or
helpers that are intentionally outside dependency discovery may require:

.. code-block:: bash

   FLYDSL_RUNTIME_ENABLE_CACHE=0 python my_test.py
   # or remove a deliberately selected cache directory:
   rm -rf ~/.flydsl/cache

Do not disable the cache by default in benchmarks: compilation and runtime
dispatch are different costs and should be measured separately.

Autotuning API
--------------

The autotuner lives in ``flydsl.autotune``. Import it explicitly:

.. code-block:: python

   from flydsl.autotune import Config, autotune

   @autotune(
       configs=[Config(BLOCK=128), Config(BLOCK=256, waves_per_eu=2)],
       key=["n"],
   )
   @flyc.jit
   def launch(..., BLOCK: fx.Constexpr[int], ...):
       ...

``Config`` separates kernel constexpr arguments in ``kwargs`` from compiler
options such as ``waves_per_eu``. ``num_warps`` is injected as a kernel
argument when provided. ``maxnreg`` is no longer accepted because it never
reached LLVM.

``autotune`` accepts:

- ``configs``: a sequence or callable returning candidate ``Config`` objects;
- ``key``: launcher argument names that define the workload cache key;
- ``warmup`` and ``rep``: timing counts;
- ``prune_configs_by``: candidate filtering/early estimation hooks;
- ``reset_to_zero`` and ``restore_value``: repeatability controls for mutated
  tensors;
- ``pre_hook`` and ``post_hook``: per-candidate hooks;
- ``validate_hook``: an untimed correctness gate around a candidate launch;
- ``default``: a heuristic configuration used when a search is not forced;
- ``artifact_name``: stable name for portable offline configuration lookup;
- ``select_config``: policy over successful ``(config, elapsed_ms)`` pairs.

Set ``FLYDSL_AUTOTUNE=1`` to force a fresh search. The scratch winner cache is
stored under ``FLYDSL_AUTOTUNE_CACHE_DIR`` (default
``~/.flydsl/autotune``). Setting ``FLYDSL_AUTOTUNE_CONFIG_DIR`` enables
portable offline artifacts: normal runs may consume a matching artifact, while
a forced run validates/times candidates and emits one.

``do_bench(fn, warmup=5, rep=25, quantiles=None)`` measures asynchronous GPU
work with batched event windows. The callable must enqueue work on the current
stream and must not synchronize internally.

See :doc:`../autotune_guide` for artifact identity, validation, failure
handling, timing rules, and current adopters.

AOT runtime discovery
---------------------

An exported object needs the FlyDSL runtime libraries plus system ROCm
libraries. Resolve the packaged part programmatically:

.. code-block:: python

   from flydsl.runtime import find_runtime_libraries

   for library in find_runtime_libraries():
       print(library.path, library.soname)

Or obtain shell-oriented flags:

.. code-block:: bash

   python -m flydsl.compiler.aot_config --libdir
   python -m flydsl.compiler.aot_config --ldflags --libs

See :doc:`../aot_export_guide` for the full build and deployment contract.
