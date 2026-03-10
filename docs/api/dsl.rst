FlyDSL Python DSL
=================

The ``flydsl`` package provides the Python front-end for authoring GPU kernels
with explicit layout algebra.

Core Module
-----------

.. automodule:: flydsl
   :members:
   :undoc-members:
   :show-inheritance:

Expression API (``flydsl.expr``)
---------------------------------

The ``flydsl.expr`` module (imported as ``fx``) provides the high-level Python
API for constructing Fly IR, including layout construction, tiled copies, tensor
operations, and kernel definitions.

.. code-block:: python

   import flydsl.expr as fx

Key functions and types:

- **fx.make_layout**, **fx.make_shape**, **fx.make_stride** -- layout construction
- **fx.make_tile** -- tile construction from layouts
- **fx.make_copy_atom**, **fx.make_tiled_copy** -- tiled copy setup
- **fx.make_mma_atom**, **fx.make_tiled_mma** -- MFMA instruction setup
- **fx.logical_divide**, **fx.zipped_divide** -- tensor partitioning
- **fx.logical_product**, **fx.raked_product** -- layout product operations
- **fx.make_fragment_like**, **fx.copy** -- register fragment operations
- **fx.slice** -- tensor slicing
- **fx.memref_alloca**, **fx.memref_load_vec**, **fx.memref_store_vec** -- register operations
- **fx.copy_atom_call** -- invoke copy atom
- **fx.gemm** -- invoke MFMA GEMM
- **fx.thread_idx**, **fx.block_idx** -- GPU indexing (via ``fx.gpu``)
- **fx.size**, **fx.rank**, **fx.cosize** -- layout inspection
- **fx.crd2idx**, **fx.idx2crd** -- coordinate-to-index mapping

Type annotations:

- **fx.Tensor** -- GPU tensor argument
- **fx.Constexpr[int]** -- compile-time constant
- **fx.Int32** -- dynamic int32 argument
- **fx.Stream** -- GPU stream argument

Submodules:

- **fx.arith** -- arithmetic operations (``addf``, ``mulf``, ``subf``, etc.)
- **fx.gpu** -- GPU operations (``thread_idx``, ``block_idx``, ``barrier``)
- **fx.vector** -- vector operations
- **fx.buffer_ops** -- AMD buffer load/store instructions
- **fx.rocdl** -- ROCDL-specific operations (buffer tensors, copy atoms)

Compiler API (``flydsl.compiler``)
-----------------------------------

.. code-block:: python

   import flydsl.compiler as flyc

- **@flyc.kernel** -- decorator for GPU kernel functions
- **@flyc.jit** -- decorator for host-side JIT launch functions
- **flyc.from_dlpack** -- convert PyTorch tensors to FlyDSL tensors

.. seealso:: :doc:`compiler` for the full compilation and pipeline API.
