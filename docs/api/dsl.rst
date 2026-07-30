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

Layout Inspection
~~~~~~~~~~~~~~~~~~

- **fx.size(layout)** -- total number of elements
- **fx.cosize(layout)** -- codomain size
- **fx.rank(layout)** -- number of modes
- **fx.depth(layout)** -- nesting depth
- **fx.get_shape(layout)** -- extract shape tuple
- **fx.get_stride(layout)** -- extract stride tuple
- **fx.get_scalar(int_tuple)** -- extract the scalar from a single-leaf int tuple (per-mode access is ``fx.get``)

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
- **fx.blocked_product(a, b)** -- blocked product

Coordinate Mapping
~~~~~~~~~~~~~~~~~~~

- **fx.crd2idx(coord, layout)** -- coordinate to linear index
- **fx.idx2crd(idx, layout)** -- linear index to coordinate
- **fx.slice(tensor, slices)** -- slice a tensor by coordinates or ``None``
- **fx.get(layout, idx)** -- access element at index

Memory Operations
~~~~~~~~~~~~~~~~~~

- **fx.make_rmem_tensor(shape_or_layout, dtype)** -- allocate register-file memory
- **fx.memref_load(memref, indices)** -- scalar load from memref
- **fx.memref_store(value, memref, indices)** -- scalar store to memref
- **fx.memref_load_vec(memref)** -- load entire register as a vector
- **fx.memref_store_vec(vec, memref)** -- store vector to register memref
- **fx.make_fragment_layout_like(tensor)** -- compute the corresponding fragment layout
- **fx.make_fragment_like(tensor)** -- allocate register fragment with same layout

Copy & GEMM
~~~~~~~~~~~~~

- **fx.make_copy_atom(instr, dtype)** -- create a CopyAtom from instruction descriptor
- **fx.make_mma_atom(instr)** -- create an MmaAtom from an MMA op type (the op type carries the dtype, e.g. ``fx.rocdl.MFMA(16, 16, 4, fx.Float32)``)
- **fx.make_tile(\*layouts)** -- build a tile from layouts (variadic)
- **fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)** -- build a TiledCopy
- **fx.make_tiled_mma(mma_atom, ...)** -- build a TiledMma
- **fx.copy(copy_atom, src, dst, pred=None)** -- execute a copy (with optional predicate mask)
- **fx.gemm(mma_atom, d, a, b, c)** -- execute matrix multiply-accumulate (accumulator passed as both ``d`` and ``c``)
- **fx.copy_atom_call(atom, src, dst)** -- invoke a single copy atom
- **fx.mma_atom_call(atom, d, a, b, c)** -- invoke a single MMA atom

Derived Tiled Operations (``flydsl.expr.derived``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level classes for tiled copy and MMA partitioning:

- **CopyAtom** (``flydsl.expr.typing``) -- single hardware copy instruction descriptor
- **MmaAtom** (``flydsl.expr.typing``) -- single MMA instruction descriptor (MFMA)
- **CopyAtomType**, **MmaAtomType** -- atom type wrappers exposed by ``flydsl.expr.derived``
- **TiledCopy** -- multi-thread tiled copy; use ``get_slice(tid)`` → ``ThrCopy``
- **TiledMma** -- multi-thread tiled MMA; use ``get_slice(tid)`` → ``ThrMma``
- **ThrCopy** -- per-thread copy view: ``partition_S(src)``, ``partition_D(dst)``, ``retile(t)``
- **ThrMma** -- per-thread MMA view: ``partition_A(a)``, ``partition_B(b)``, ``partition_C(c)``
- **make_layout_tv(thr, val)** -- build thread-value layout
- **make_tiled_copy_A/B/C(copy_atom, tiled_mma)** -- create TiledCopy matched to MMA operands
- **fx.gather(copy_atom, base_iter, offset_tensor, dst_tensor, \*, pred=None)** -- indexed load ``dst = base[offset]`` via a copy atom (offset tensor is ``(TV, Rest...)``)
- **fx.scatter(copy_atom, src_tensor, base_iter, offset_tensor, \*, pred=None)** -- indexed store ``base[offset] = src`` (see ``examples/05-gather_scatter.py``)

Type Annotations
~~~~~~~~~~~~~~~~~

- **fx.Tensor** -- GPU tensor argument
- **fx.Constexpr[int]** -- compile-time constant
- **fx.Int32** -- dynamic int32 argument
- **fx.Float32**, **fx.Float16**, **fx.BFloat16** -- scalar types
- **fx.Float8E4M3FN**, **fx.Float8E4M3FNUZ**, **fx.Float8E5M2** -- FP8 scalar types
- **fx.Stream** -- GPU stream argument
- **fx.T** -- type namespace (``T.f32``, ``T.f16``, ``T.bf16``, ``T.i8``, ``T.index``, etc.)
- **fx.Basis(value, modes)** / **fx.E(\*modes)** -- basis-stride leaves for by-mode layout construction (``E(0)`` → ``1E0``)
- **fx.SyncScope** -- target-neutral LLVM sync scopes (``SyncScope.System``, ``SyncScope.SingleThread``); AMDGPU scopes live in ``flydsl.expr.rocdl.enum.SyncScope``

GPU Intrinsics (``flydsl.expr.gpu``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **fx.thread_idx** -- thread index (``Tuple3D`` with ``.x``, ``.y``, ``.z``)
- **fx.block_idx** -- block index
- **fx.block_dim** -- block dimensions
- **fx.grid_dim** -- grid dimensions
- **fx.gpu.barrier()** -- workgroup barrier synchronization
- **fx.gpu.smem_space()** -- shared memory (LDS) address space attribute

Arithmetic and Numeric Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer typed DSL values and operator-overloaded arithmetic:

.. code-block:: python

   import flydsl.expr as fx
   from flydsl.expr.typing import Vector as Vec

   c = fx.Int64(42)
   v = fx.Int32(idx)
   f = fx.Float32(1.0)
   r = cond.select(a, b)
   y = (x + 1) * scale

Preferred APIs:

- **fx.Int32(value)**, **fx.Int64(value)**, **fx.Float32(value)** -- typed constants and casts (use **fx.Int64** for index/offset values and loop bounds; **fx.Index** is deprecated)
- **ArithValue / Numeric operators** -- ``+``, ``-``, ``*``, ``/``, ``%``, ``<<``, ``>>``
- **cond.select(true_val, false_val)** -- ternary select when ``cond`` is an ``ArithValue``
- **arith.cmpi(predicate, lhs, rhs)** -- integer comparison
- **arith.cmpf(predicate, lhs, rhs)** -- float comparison
- **arith.maxnumf(a, b)** -- float maximum returning the non-NaN operand (libm ``fmax``); preserves the DSL type of ``a``
- **Chained comparisons** (``lo <= x < hi``) are supported inside traced kernels and lower to combined ``cmp`` + ``and``.

Fastmath flags can be applied ambiently to a block or per-op:

.. code-block:: python

   with fx.fastmath(fx.FastMathFlags.fast):
       y = a * b + c          # float operators/math funcs inherit the flags
       z = fx.exp(a, fastmath="contract")   # explicit arg overrides the ambient scope

- **fx.fastmath(flags)** -- context manager applying ``fastmath`` to float ops built in the block; nests and restores on exit
- **fx.FastMathFlags** -- flag enum (``fast``, ``contract``, ``reassoc``, …; combine with ``|``)
- **Direct ``arith.addf(..., fastmath=...)`` / ``arith.AddFOp(..., fastmath=...)``** -- per-op flags where an ambient scope is not desired

Vector Values (``flydsl.expr.typing.Vector``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Vec.from_elements(elements, dtype)** -- construct vector from scalars
- **Vec.filled(shape, value, dtype)** -- splat vector
- **Vec(value)[i]** -- extract element
- **Vec(value).bitcast(dtype)** -- bitcast vector element type
- **Vec(value).to(dtype)** -- convert vector element type
- **Vec(value).store(memref, indices)** -- store vector to memref

ROCDL Operations (``flydsl.expr.rocdl``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD-specific operations for ROCm:

- **fx.rocdl.make_buffer_tensor(tensor)** -- create buffer resource from tensor (CDNA buffer copy)
- **fx.rocdl.BufferCopy32b** / **BufferCopy128b** -- buffer copy instruction atoms
- **fx.rocdl.MFMA(m, n, k, elem_ty_ab, elem_ty_acc=None)** -- MFMA instruction atom constructor (CDNA3/CDNA4; 4th arg is the A/B element type; accumulator defaults to f32)
- **fx.rocdl.WMMA(m, n, k, elem_ty_ab, elem_ty_acc=None, \*\*kwargs)** -- WMMA MMA atom constructor (arch-dispatched: gfx11 / gfx12 / gfx1250). gfx1250 supports f32(K4), f16/bf16(K32), fp8/bf8(K64/128), i8(K64), i4(K32); integer paths take ``sign_a`` / ``sign_b`` / ``clamp``
- **fx.rocdl.WMMAScale(m, n, k, elem_ty_a, elem_ty_b=None, elem_ty_acc=None, \*, opsel_a=0, opsel_b=0, mod_c=0, reuse_a=False, reuse_b=False, block_size=32)** -- gfx1250 MX-scaled WMMA (E8M0 block scale, f8/f6/f4; ``16x16x128`` or ``32x16x128`` fp4-only). Per-operand scales are atom state (``scale_a`` / ``scale_b``)
- **fx.rocdl.make_tdm_atom(tensor, tensor_extents, strides=None, \*, num_warps, ...)** -- build a gfx1250 TDM (Tensor Data Mover) async Global↔LDS whole-tile copy atom (rank 1-5); the global base comes from the ``copy_atom_call`` operand pointer, while the per-dim extent (OOB), stride, ``imm_offset``, and MCAST ``workgroup_mask`` are atom state. ``fx.rocdl.TDM(rank, num_warps, ...)`` builds the atom type only. Advance the K-loop tile with ``fx.copy(atom, gt, dst, imm_offset=...)``
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
- **flydsl.compiler.kernel_function.CompilationContext** -- context object available during kernel compilation (not a top-level ``flydsl.compiler`` symbol)

.. seealso:: :doc:`compiler` for the full compilation pipeline and pass details.
