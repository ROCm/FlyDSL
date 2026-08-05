FlyDSL |FLYDSL_VERSION| documentation
=====================================

**FlyDSL** is a Python DSL and MLIR compiler stack for authoring high-performance
GPU kernels with explicit layout algebra, targeting AMD ROCm/HIP GPUs.

FlyDSL is the Python front-end (*Flexible Layout Python DSL*) powered by the
**Fly dialect**: an MLIR-native compiler stack with first-class layout IR
(``!fly.int_tuple``, ``!fly.layout``, ``!fly.coord_tensor``, ``!fly.memref``),
explicit algebra and coordinate mapping, plus a composable lowering pipeline
to GPU/ROCDL.
