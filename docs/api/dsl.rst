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

FLIR Dialect Extensions
-----------------------

The ``flydsl.dialects.ext.flir`` module provides the high-level Python API for
constructing FLIR IR, including layout construction, tiled copies, tensor
operations, and kernel definitions.

Key classes and functions:

- **MlirModule** -- base class for GPU kernel modules
- **@kernel** / **@jit** -- decorators for GPU kernel and JIT function definitions
- **make_shape**, **make_stride**, **make_layout** -- layout construction
- **make_ordered_layout** -- ordered layout construction
- **make_copy_atom**, **make_tiled_copy_tv** -- tiled copy setup
- **make_tensor**, **zipped_divide** -- tensor partitioning
- **make_fragment_like**, **copy** -- register fragment operations
- **thread_idx**, **block_idx** -- GPU indexing
- **size**, **rank**, **cosize** -- layout inspection
- **crd2idx**, **idx2crd** -- coordinate-to-index mapping

Arithmetic Extensions
---------------------

The ``flydsl.dialects.ext.arith`` module wraps MLIR arithmetic operations with
Python operator overloading via ``ArithValue``.

SCF Extensions
--------------

The ``flydsl.dialects.ext.scf`` module provides structured control flow
(``for_``, ``if_``, ``while_``) with Python context manager syntax.

Buffer Operations
-----------------

The ``flydsl.dialects.ext.buffer_ops`` module provides high-level wrappers for
AMD buffer load/store instructions (CDNA3/CDNA4). These use scalar base pointers
with per-thread offsets for efficient global memory access with hardware bounds
checking. Key APIs: ``create_buffer_resource``, ``buffer_load``, ``buffer_store``,
``buffer_load_2d``, ``buffer_store_2d``, ``BufferResourceDescriptor``.

Block Reduce Operations
-----------------------

The ``flydsl.dialects.ext.block_reduce_ops`` module provides block-level and
warp-level collective operations (reductions, broadcasts) that combine shuffle,
shared memory, and barrier primitives into reusable patterns.

Compiler Pipeline
-----------------

.. seealso:: :doc:`compiler` for the compilation and pipeline API.
