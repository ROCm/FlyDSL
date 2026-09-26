API overview
============

FlyDSL separates kernel authoring, compilation, reusable algorithms, and
runtime integration into distinct namespaces. This page defines those
boundaries and points to the detailed references.

Primary namespaces
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Namespace
     - Use it for
     - Reference
   * - ``flydsl.expr`` (normally ``fx``)
     - DSL scalar and aggregate values, layouts, tensors, memory operations,
       tiled copy/MMA, control flow helpers, GPU intrinsics, and target-specific
       operations.
     - :doc:`dsl`
   * - ``flydsl.compiler`` (normally ``flyc``)
     - Kernel and launcher decorators, specialization, argument conversion,
       backend registration, ahead-of-time compilation, and C export.
     - :doc:`compiler`
   * - ``flydsl.extension``
     - Libraries implemented on top of the expression language, currently
       cooperative algorithms and random-number generation. They are also
       lazily available as ``fx.coop`` and ``fx.random``.
     - :doc:`extensions`
   * - ``flydsl.runtime``
     - Device-runtime selection and discovery of libraries needed by exported
       host objects.
     - :doc:`runtime`
   * - ``flydsl.autotune``
     - Config search, timing, validation, caching, and deployable offline
       configuration artifacts.
     - :doc:`runtime`
   * - ``kernels``
     - Optimized implementations and integration references in a source
       checkout. This package is not installed by the ``flydsl`` wheel.
     - :doc:`kernels`

Recommended imports
-------------------

.. code-block:: python

   import flydsl
   import flydsl.compiler as flyc
   import flydsl.expr as fx

   # Extension aliases are loaded only when first accessed.
   reduce_fn = fx.coop.warp_reduce
   random_fn = fx.random.rand4x

``flydsl.__version__`` reports the source/package version. Most kernel code
needs only ``flyc`` and ``fx``.

Where DSL operations may run
----------------------------

Expression operations create MLIR and therefore run while a ``@flyc.jit`` or
``@flyc.kernel`` function is being traced. They are not eager numerical
operations on ordinary Python values. Compile-time values such as
``fx.Constexpr`` and static layouts may be folded during tracing; dynamic DSL
values emit IR.

Host-side helpers are different. Argument adapters such as
``flyc.from_dlpack`` and runtime discovery such as
``flydsl.runtime.find_runtime_libraries`` execute as ordinary Python code.

Package boundary and stability
------------------------------

The wheel contains the ``flydsl`` package and its embedded MLIR runtime. It
does not install the repository's ``kernels`` or ``tests`` packages. Run
kernel-library examples from a source checkout with the repository root on
``PYTHONPATH``.

Public-looking does not always mean stable. :doc:`../api_stability` defines the
compatibility contract for exported expression modules, compiler entry points,
extension libraries, and explicitly listed runtime helpers. Backend-specific
and upstream MLIR operations can be useful without carrying the same promise.

Print the exact stable catalog for the current checkout without importing the
GPU runtime:

.. code-block:: bash

   python3 scripts/list_stable_apis.py
   python3 scripts/list_stable_apis.py --format json

Use the catalog when checking a release boundary; use the pages in this
section when choosing an API and understanding its behavior.
