FlyDSL |FLYDSL_VERSION| documentation
=====================================

**FlyDSL** is a Python DSL and MLIR compiler stack for authoring high-performance
GPU kernels with explicit layout algebra, targeting AMD ROCm/HIP GPUs.

FlyDSL is the Python front-end (*Flexible Layout Python DSL*) powered by the
**Fly dialect**: an MLIR-native compiler stack with first-class layout IR
(``!fly.int_tuple``, ``!fly.layout``, ``!fly.coord_tensor``, ``!fly.memref``),
explicit algebra and coordinate mapping, plus a composable lowering pipeline
to GPU/ROCDL.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting started

      * :doc:`Install FlyDSL <installation>`

      * :doc:`Quick start <quickstart>`

      * :doc:`API stability <api_stability>`

   .. grid-item-card:: Guides

      * :doc:`Arithmetic types <language/arithmetic_types>`

      * :doc:`Composite types <language/composite_types>`

      * :doc:`Storage and allocator <language/storage_and_allocator>`

      * :doc:`Architecture and compilation pipeline <architecture_guide>`

      * :doc:`Layout algebra <layout_system_guide>`

      * :doc:`Kernel authoring <kernel_authoring_guide>`

      * :doc:`Kernel tuning <kernel_tuning_guide>`

      * :doc:`Offline autotune configs <autotune_guide>`

      * :doc:`Prebuilt kernel library <prebuilt_kernels_guide>`

      * :doc:`Testing and benchmarking <testing_benchmarking_guide>`

      * :doc:`CuTe layout algebra <cute_layout_algebra_guide>`

      * :doc:`External bitcode integration <extern_integration_guide>`

   .. grid-item-card:: API reference

      * :doc:`FlyDSL Python DSL <api/dsl>`

      * :doc:`Compiler and pipeline <api/compiler>`

      * :doc:`Prebuilt kernels <api/kernels>`

   .. grid-item-card:: Tutorials

      * :doc:`Basic usage <tutorials/basic_usage>`

      * :doc:`Kernel development <tutorials/kernel_development>`

