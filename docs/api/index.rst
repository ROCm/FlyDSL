API overview
============

This section documents the installable FlyDSL Python API and the source-tree
kernel library. Inclusion in the documentation does not by itself make an API
stable: :doc:`../api_stability` is the authoritative compatibility policy.

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
       C object export, and backend registration.
     - :doc:`compiler`
   * - ``flydsl.extension``
     - Libraries implemented on top of the expression language, currently
       cooperative algorithms and random-number generation. They are also
       lazily available as ``fx.coop`` and ``fx.random``.
     - :doc:`extensions`
   * - ``flydsl.runtime.device``
     - Query the target ROCm architecture and distinguish RDNA from CDNA.
     - :doc:`runtime`
   * - Repository ``kernels`` tree
     - Pre-built and reference kernels used from a source checkout. These
       interfaces are not installed by the wheel and are not stable APIs.
     - :doc:`kernels`

Recommended imports
-------------------

.. code-block:: python

   import flydsl.compiler as flyc
   import flydsl.expr as fx
   from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

   # Extension aliases are loaded only when first accessed.
   reduce_fn = fx.coop.warp_reduce
   random_fn = fx.random.rand4x

Most kernel code needs only ``flyc`` and ``fx``. Import the runtime helpers only
when host code must select an architecture-specific path.

Where DSL operations may run
----------------------------

Expression operations create MLIR and therefore run while a ``@flyc.jit`` or
``@flyc.kernel`` function is being traced. They are not eager numerical
operations on ordinary Python values. Compile-time values such as
``fx.Constexpr`` and static layouts may be folded during tracing; dynamic DSL
values emit IR.

Host-side helpers are different. Argument adapters such as
``flyc.from_dlpack`` and the architecture helpers in
``flydsl.runtime.device`` execute as ordinary Python code.

Package boundary and stability
------------------------------

The wheel contains the ``flydsl`` package and its embedded MLIR runtime. It
does not install the repository's ``kernels`` or ``tests`` packages. Those
source-tree modules are covered by the Guides section, not this API reference.

:doc:`../api_stability` defines the compatibility contract for exported
expression modules, compiler entry points, extension libraries, and the two
runtime helpers listed above.

Print the exact stable catalog for the current checkout without importing the
GPU runtime:

.. code-block:: bash

   python3 scripts/list_stable_apis.py
   python3 scripts/list_stable_apis.py --format json

Use the catalog when checking a release boundary. The reference pages provide
usage context, but only paths that satisfy :doc:`../api_stability` carry a
compatibility commitment. The source-tree kernel page is explicitly outside
that commitment.
