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

Layout Construction
~~~~~~~~~~~~~~~~~~~~

- **fx.make_layout(shape, stride)** -- create a layout from shape and stride tuples
- **fx.make_shape(\*dims)** -- create a shape tuple
- **fx.make_stride(\*strides)** -- create a stride tuple
- **fx.make_coord(\*coords)** -- create a coordinate tuple
- **fx.make_ordered_layout(shape, order)** -- layout with explicit mode ordering
- **fx.make_identity_layout(shape)** -- identity layout (strides = prefix products)
- **fx.make_identity_tensor(shape)** -- identity coordinate tensor

Layout Inspection
~~~~~~~~~~~~~~~~~~

- **fx.size(layout)** -- total number of elements
- **fx.cosize(layout)** -- codomain size
- **fx.rank(layout)** -- number of modes
- **fx.depth(layout)** -- nesting depth
- **fx.get_shape(layout)** -- extract shape tuple
- **fx.get_stride(layout)** -- extract stride tuple
- **fx.get_scalar(int_tuple, idx)** -- extract scalar from nested int tuple

Layout Algebra
~~~~~~~~~~~~~~~

- **fx.composition(a, b)** -- compose two layouts
- **fx.complement(layout, codomain_size)** -- complementary layout
- **fx.right_inverse(layout)** -- right inverse
- **fx.coalesce(layout)** -- coalesce contiguous modes
- **fx.recast_layout(layout, old_type, new_type)** -- recast layout for type change

Layout Products & Divides
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **fx.logical_divide(tensor, tiler)** -- partition tensor by tiler layout
- **fx.zipped_divide**, **fx.tiled_divide**, **fx.flat_divide** -- divide variants
- **fx.logical_product(a, b)** -- layout product
- **fx.zipped_product**, **fx.tiled_product**, **fx.flat_product** -- product variants
- **fx.raked_product(thr_layout, val_layout)** -- interleaved (raked) product
- **fx.block_product(a, b)** -- blocked product

Coordinate Mapping
~~~~~~~~~~~~~~~~~~~

- **fx.crd2idx(coord, shape, stride)** -- coordinate to linear index
- **fx.idx2crd(idx, shape)** -- linear index to coordinate
- **fx.slice(tensor, slices)** -- slice a tensor by coordinates or ``None``
- **fx.get(layout, idx)** -- access element at index

Memory Operations
~~~~~~~~~~~~~~~~~~

- **fx.memref_alloca(type, layout)** -- allocate register-file memory
- **fx.memref_load(memref, indices)** -- scalar load from memref
- **fx.memref_store(value, memref, indices)** -- scalar store to memref
- **fx.memref_load_vec(memref)** -- load entire register as a vector
- **fx.memref_store_vec(vec, memref)** -- store vector to register memref
- **fx.make_fragment_layout_like(layout_like)** -- compute the corresponding fragment layout
- **fx.make_fragment_like(tensor)** -- allocate register fragment with same layout

Copy & GEMM
~~~~~~~~~~~~~

- **fx.make_copy_atom(instr, dtype)** -- create a CopyAtom from instruction descriptor
- **fx.make_mma_atom(instr, dtype)** -- create an MmaAtom from MFMA descriptor
- **fx.make_tile(layouts)** -- build a tile from a list of layouts
- **fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)** -- build a TiledCopy
- **fx.make_tiled_mma(mma_atom, ...)** -- build a TiledMma
- **fx.copy(tiled_copy, src, dst, pred=None)** -- execute a tiled copy (with optional predicate mask)
- **fx.gemm(tiled_mma, accum, A, B)** -- execute tiled matrix multiply-accumulate
- **fx.copy_atom_call(atom, src, dst)** -- invoke a single copy atom
- **fx.mma_atom_call(atom, accum, A, B)** -- invoke a single MMA atom

Derived Tiled Operations (``flydsl.expr.derived``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level classes for tiled copy and MMA partitioning:

- **CopyAtom** -- single hardware copy instruction descriptor
- **MmaAtom** -- single MMA instruction descriptor (MFMA)
- **TiledCopy** -- multi-thread tiled copy; use ``get_slice(tid)`` → ``ThrCopy``
- **TiledMma** -- multi-thread tiled MMA; use ``get_slice(tid)`` → ``ThrMma``
- **ThrCopy** -- per-thread copy view: ``partition_S(src)``, ``partition_D(dst)``, ``retile(t)``
- **ThrMma** -- per-thread MMA view: ``partition_A(a)``, ``partition_B(b)``, ``partition_C(c)``
- **make_layout_tv(thr, val)** -- build thread-value layout
- **make_tiled_copy_A/B/C(copy_atom, tiled_mma)** -- create TiledCopy matched to MMA operands

Type Annotations
~~~~~~~~~~~~~~~~~

- **fx.Tensor** -- GPU tensor argument
- **fx.Constexpr[int]** -- compile-time constant
- **fx.Int32** -- dynamic int32 argument
- **fx.Float32**, **fx.Float16**, **fx.BFloat16**, **fx.Float8** -- scalar types
- **fx.Stream** -- GPU stream argument
- **fx.T** -- type namespace (``T.f32``, ``T.f16``, ``T.bf16``, ``T.i8``, ``T.index``, etc.)

GPU Intrinsics (``flydsl.expr.gpu``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **fx.thread_idx** -- thread index (``Tuple3D`` with ``.x``, ``.y``, ``.z``)
- **fx.block_idx** -- block index
- **fx.block_dim** -- block dimensions
- **fx.grid_dim** -- grid dimensions
- **fx.gpu.barrier()** -- workgroup barrier synchronization
- **fx.gpu.smem_space()** -- shared memory (LDS) address space attribute

Arithmetic (``flydsl.expr.arith``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operator-overloaded arithmetic via ``ArithValue``:

.. code-block:: python

   from flydsl.expr import arith

   c = arith.constant(42, index=True)
   v = arith.index_cast(T.index, val)
   r = arith.select(cond, a, b)

Key functions:

- **arith.constant(value, type=None, index=False)** -- create constant
- **arith.constant_vector(value, vec_type)** -- create splat vector constant
- **arith.index_cast(target_type, value)** -- cast to/from index type
- **arith.select(cond, true_val, false_val)** -- ternary select
- **arith.cmpi(predicate, lhs, rhs)** -- integer comparison
- **arith.cmpf(predicate, lhs, rhs)** -- float comparison
- **arith.sitofp(type, value)** -- signed int to float
- **arith.trunc_f(type, value)** -- truncate float precision
- **ArithValue** -- wrapper class enabling ``+``, ``-``, ``*``, ``/``, ``%``, ``<<``, ``>>`` operators

Vector Operations (``flydsl.expr.vector``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **vector.from_elements(type, elements)** -- construct vector from scalars (auto-unwraps ArithValue)
- **vector.store(value, memref, indices)** -- store vector to memref
- **vector.extract(vector, position)** -- extract element from vector
- **vector.load_op(result_type, memref, indices)** -- load vector from memref
- **vector.bitcast(result_type, source)** -- bitcast vector type
- **vector.broadcast**, **vector.reduction** -- broadcast and reduce operations (from MLIR upstream)

Buffer Operations (``flydsl.expr.buffer_ops``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD CDNA3/CDNA4 buffer load/store with hardware bounds checking:

.. code-block:: python

   from flydsl.expr import buffer_ops

   rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)
   data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=T.i32)
   buffer_ops.buffer_store(data, rsrc, offset, mask=is_valid)

- **create_buffer_resource(tensor, num_records=None, max_size=False)** -- create buffer descriptor
- **buffer_load(rsrc, offset, vec_width, dtype, soffset_bytes, mask)** -- vector buffer load
- **buffer_store(data, rsrc, offset, soffset_bytes, mask)** -- buffer store
- **BufferResourceDescriptor** -- descriptor dataclass

ROCDL Operations (``flydsl.expr.rocdl``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD-specific operations for ROCm:

- **fx.rocdl.make_buffer_tensor(tensor)** -- create buffer resource from tensor
- **fx.rocdl.BufferCopy32b** / **BufferCopy128b** -- buffer copy instruction atoms
- **fx.rocdl.MFMA(M, N, K, acc_type)** -- MFMA instruction atom constructor
- **fx.rocdl.sched_mfma(cnt)** -- insert MFMA scheduling barrier
- **fx.rocdl.sched_vmem(cnt)** -- insert VMEM scheduling barrier
- **fx.rocdl.sched_dsrd(cnt)** -- insert DS read scheduling barrier
- **fx.rocdl.sched_dswr(cnt)** -- insert DS write scheduling barrier
- **mfma_f32_16x16x16f16**, **mfma_f32_16x16x16bf16_1k**, etc. -- direct MFMA intrinsics

Compiler API (``flydsl.compiler``)
-----------------------------------

.. code-block:: python

   import flydsl.compiler as flyc

- **@flyc.kernel** -- decorator for GPU kernel functions
- **@flyc.jit** -- decorator for host-side JIT launch functions
- **flyc.from_dlpack(tensor)** -- convert DLPack-compatible tensors (PyTorch, etc.) to FlyDSL
- **JitArgumentRegistry** -- registry for custom argument type adapters
- **CompilationContext** -- context object available during kernel compilation

.. seealso:: :doc:`compiler` for the full compilation pipeline and pass details.
