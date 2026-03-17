FlyDSL Documentation
=====================

**FlyDSL** is a Python DSL and MLIR compiler stack for authoring high-performance
GPU kernels with explicit layout algebra, targeting AMD ROCm/HIP GPUs.

FlyDSL is the Python front-end (*Flexible Layout Python DSL*) powered by the
**Fly dialect**: an MLIR-native compiler stack with first-class layout IR
(``!fly.int_tuple``, ``!fly.layout``, ``!fly.coord_tensor``, ``!fly.memref``),
explicit algebra and coordinate mapping, plus a composable lowering pipeline
to GPU/ROCDL.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   architecture_guide
   layout_system_guide
   kernel_authoring_guide
   prebuilt_kernels_guide
   testing_benchmarking_guide
   cute_layout_algebra_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/dsl
   api/compiler
   api/kernels

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
