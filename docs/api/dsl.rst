Expression language
===================

``flydsl.expr`` is the kernel-authoring surface and is conventionally imported
as ``fx``:

.. code-block:: python

   import flydsl.expr as fx

Its direct-child modules export names into the ``fx`` namespace. For example,
``fx.make_layout``, ``fx.gpu.barrier``, and
``flydsl.expr.primitive.make_layout`` refer to the same public operation.
Backend and extension packages are loaded lazily as ``fx.rocdl``, ``fx.coop``,
and ``fx.random``.

Execution model
---------------

Most ``fx`` calls build MLIR and must execute while FlyDSL traces a
``@flyc.jit`` or ``@flyc.kernel`` function. A Python value wrapped in a DSL
numeric type can remain compile-time when all of its inputs are known; once an
operand is dynamic, the same expression emits IR. See
:doc:`../language/arithmetic_types` for the exact folding and promotion rules.

Values and annotations
----------------------

.. list-table:: Core value families
   :header-rows: 1
   :widths: 24 34 42

   * - Family
     - Important names
     - Purpose
   * - Scalar values
     - ``Boolean``, ``Int4/8/16/32/64/128``, ``Uint8/16/32/64/128``,
       ``Float16``, ``BFloat16``, ``Float32/64``, FP8/FP6/FP4 types
     - Typed constants, casts, and dynamic scalar results. ``Index`` remains
       compatible but is deprecated; prefer ``Int64`` for offsets and bounds.
   * - Function annotations
     - ``Tensor``, ``Pointer``, ``Stream``, ``Constexpr[T]``
     - Describe host arguments at the JIT boundary and kernel operands.
   * - Layout values
     - ``IntTuple``, ``Layout``, ``ComposedLayout``, ``Swizzle``, ``Tile``,
       ``Basis``, ``E``
     - Represent shapes, strides, coordinates, swizzles, and mappings.
   * - Tiled operations
     - ``CopyAtom``, ``TiledCopy``, ``TiledMma``, ``ThrCopy``, ``ThrMma``
     - Describe an instruction and how its work is partitioned across threads.
   * - Vectors
     - ``Vector``, aliases such as ``Int32x4`` and ``Float16x8``
     - SSA vectors with element access, conversion, bitcast, load, and store.
   * - Composite/storage types
     - ``struct``, ``union``, ``Struct``, ``Union``, ``Array``, ``Storage``,
       ``Arena``, ``Align``, ``Empty``
     - Define aggregate DSL values and their typed memory representations.

``fx.T`` exposes MLIR type constructors such as ``T.f32()``, ``T.i32()``, and
``T.index()``. Prefer the DSL numeric classes for arithmetic values; use
``fx.T`` when an API explicitly asks for an MLIR type.

Layout construction and inspection
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - API
     - Meaning
   * - ``make_int_tuple``, ``make_shape``, ``make_stride``, ``make_coord``
     - Construct scalar or nested integer tuples with the requested semantic
       role.
   * - ``make_layout(shape, stride)``
     - Construct a layout from shape and stride tuples.
   * - ``make_ordered_layout(shape, order)``
     - Construct a compact layout with an explicit slow-to-fast mode order.
   * - ``make_identity_layout(shape)``
     - Construct the identity coordinate layout for a shape.
   * - ``make_composed_layout(swizzle, offset, layout)``
     - Combine a base layout with an offset and coordinate swizzle.
   * - ``make_layout_like``, ``make_fragment_layout_like``
     - Derive layouts matching an existing tensor or layout.
   * - ``rank``, ``depth``, ``size``, ``cosize``, ``coshape``
     - Query logical rank/nesting and domain/codomain size.
   * - ``get_shape``, ``get_stride``, ``get_scalar``, ``get_leaves``
     - Inspect a layout or integer tuple. Use ``get_``/``unpack`` for element
       access; the old ``get`` spelling is deprecated.
   * - ``crd2idx``, ``idx2crd``
     - Map coordinates to indices or indices back to coordinates.

Layout algebra
--------------

The core transforms are ``composition``, ``complement``, ``coalesce``,
``right_inverse``, ``left_inverse``, ``recast_layout``, ``apply_swizzle``, and
``tile_to_shape``. Product and divide operations provide several result
organizations:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Operation
     - Variants
     - Typical use
   * - Product
     - ``logical_product``, ``zipped_product``, ``tiled_product``,
       ``flat_product``, ``raked_product``, ``blocked_product``
     - Combine thread/value or tile/remainder mappings.
   * - Divide
     - ``logical_divide``, ``zipped_divide``, ``tiled_divide``,
       ``flat_divide``
     - Partition a tensor or layout by a tiler.
   * - Structural editing
     - ``append``, ``prepend``, ``group``, ``take``, ``dice``, ``slice``
     - Reshape or select modes without reverting to manual byte arithmetic.

See :doc:`../layout_system_guide` for semantics and worked examples, and
:doc:`../cute_layout_algebra_guide` for the mathematical vocabulary shared
with CuTe-style layout algebra.

Tensors, pointers, and memory
-----------------------------

``Tensor`` combines an iterator/pointer with a layout. ``Pointer`` is a typed
pointer value. Both expose convenience indexing, but the underlying operations
are also available explicitly:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - API
     - Purpose
   * - ``make_ptr``, ``make_view``, ``add_offset``, ``recast_iter``
     - Construct or transform typed pointers and tensor views.
   * - ``ptr_load``, ``ptr_store``
     - Typed pointer access, including vector types and masking supported by the
       pointer API.
   * - ``memref_alloca``, ``memref_load``, ``memref_store``
     - Allocate and access register/shared memrefs.
   * - ``memref_load_vec``, ``memref_store_vec``
     - Move an entire register memref as a vector SSA value.
   * - ``make_rmem_tensor``
     - Allocate a register-memory tensor from a shape or layout.
   * - ``make_fragment_like``
     - Allocate a register fragment with the derived layout and element type.
   * - ``inttoptr``, ``ptrtoint``, ``to_llvm_ptr``
     - Low-level pointer conversions for integration code.
   * - ``printf``
     - Emit device-side formatted debug output.

For composite storage, typed address views, static/dynamic shared memory, and
alignment rules, see :doc:`../language/composite_types` and
:doc:`../language/storage_and_allocator`.

Copy and matrix operations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - API
     - Purpose
   * - ``make_copy_atom(op, dtype)``
     - Wrap a universal or backend-specific copy instruction.
   * - ``make_mma_atom(op)``
     - Wrap an FMA/MFMA/WMMA instruction descriptor.
   * - ``make_tile``
     - Construct an operand tile from one or more layouts.
   * - ``make_layout_tv``, ``make_tiled_copy_tv``
     - Build thread-value layouts and tiled-copy descriptors.
   * - ``make_tiled_copy``, ``make_tiled_copy_A/B/C``
     - Build an independent copy or one matched to a tiled MMA operand.
   * - ``make_tiled_mma``
     - Partition an MMA atom across threads and values.
   * - ``TiledCopy.get_slice(tid)``
     - Return ``ThrCopy`` with ``partition_S``, ``partition_D``, and ``retile``.
   * - ``TiledMma.get_slice(tid)``
     - Return ``ThrMma`` with ``partition_A/B/C``.
   * - ``copy(atom, src, dst, pred=None)``
     - Execute a tiled or single-atom copy with an optional predicate tensor.
   * - ``gemm(atom, d, a, b, c)``
     - Execute tiled or atomic matrix multiply-accumulate.
   * - ``gather`` / ``scatter``
     - Indexed copy-atom loads and stores using an offset tensor.

``copy_atom_call`` and ``mma_atom_call`` remain available for direct atom calls,
but new code should normally use ``copy`` and ``gemm`` so the same spelling
works for both atomic and tiled operations.

Control flow and compile-time helpers
-------------------------------------

Python ``if``, ``for``, ``while``, boolean expressions, and chained comparisons
inside traced functions are rewritten to the corresponding MLIR control flow
when their conditions are dynamic. These helpers make intent explicit:

- ``const_expr(value)`` requires a trace-time value.
- ``range_constexpr(...)`` unrolls a compile-time range.
- ``static(value)`` materializes a static Fly value.
- ``assume(condition)`` records an optimization assumption.
- ``select(condition, true_value, false_value)`` is the function form of a
  typed select; an arithmetic condition also exposes ``condition.select(...)``.

Arithmetic and math
-------------------

DSL numeric and vector values overload ordinary arithmetic, comparisons,
bitwise operators, and casts. Additional exported helpers include:

- ``ceildiv``/``ceil_div``, ``min``, ``max``, ``cmpi``, ``cmpf``, and
  ``shrui``;
- IEEE/NaN-sensitive ``minimumf``, ``maximumf``, ``minnumf``, and ``maxnumf``;
- ``fastmath(flags)`` and ``FastMathFlags`` for ambient or per-operation
  floating-point flags;
- elementary/transcendental functions including ``absf``, ``exp``, ``exp2``,
  ``log``, ``sqrt``, ``rsqrt``, trigonometric/hyperbolic functions, ``erf``,
  ``powf``, ``fma``, ``clampf``, and floating-point classification;
- integer helpers ``absi``, ``ctlz``, ``cttz``, ``ctpop``, and ``ipowi``.

The complete type, folding, overflow, conversion, vector, and fast-math
contract is in :doc:`../language/arithmetic_types`.

GPU and LLVM operations
-----------------------

``fx.thread_idx``, ``fx.block_idx``, ``fx.block_dim``, and ``fx.grid_dim`` are
``Tuple3D`` values with ``.x``, ``.y``, and ``.z`` members. Other public GPU
helpers include ``lane_id``, ``known_block_size``, ``num_warp_threads``,
``barrier``, the ``shuffle_xor/up/down/idx`` family, and ``SharedAllocator``.

``flydsl.expr.llvm`` provides target-neutral atomics and generic pointer access:
``atomic_add/sub/and/or/xor/xchg/min/max/fmin/fmax/cas``, ``generic_load``,
``generic_store``, and ``memory_fence``. Use ``AtomicOrdering`` and
``SyncScope`` to make ordering and scope explicit.

ROCm-specific operations
------------------------

``fx.rocdl`` is the AMD backend package. Prefer its higher-level descriptors
and helpers over direct calls to generated upstream MLIR builders:

- buffer access: ``make_buffer_tensor``, ``make_buffer_ptr``,
  ``BufferCopy8b/16b/32b/64b/128b``, ``BufferCopyLDS*``, and ``BufferAtomic*``;
- matrix instructions: ``MFMA``, architecture-dispatched ``WMMA``, and
  gfx1250 ``WMMAScale``;
- async and scheduling control: ``asyncmark``, ``wait_asyncmark``,
  ``s_waitcnt``, ``sched_mfma``, ``sched_vmem``, ``sched_dsrd``, and
  ``sched_dswr``;
- gfx1250 TDM and cluster operations under ``fx.rocdl.tdm_ops`` and
  ``fx.rocdl.cluster``.

Architecture-specific child modules (for example ``cdna4`` and ``rdna4``)
expose operations that are not portable to every target. Direct upstream ROCDL
operation builders may change with the bundled MLIR revision; consult
:doc:`../api_stability` before treating one as a compatibility contract.

Complete export catalog
-----------------------

This page groups the usable surface rather than repeating hundreds of vector
aliases and generated instruction names. The authoritative, mechanically
derived list of stable paths for this checkout is:

.. code-block:: bash

   python3 scripts/list_stable_apis.py

Use ``--format json`` for tooling and ``--include-deprecated`` during release
review.

.. seealso::

   - :doc:`compiler` for decorators, specialization, and argument adapters
   - :doc:`extensions` for cooperative algorithms and random generation
   - :doc:`../kernel_authoring_guide` for end-to-end kernel patterns
