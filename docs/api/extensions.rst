Extension libraries
===================

FlyDSL extensions are libraries implemented with the core expression API. They
have canonical imports under ``flydsl.extension`` and lazy aliases under
``flydsl.expr``:

.. code-block:: python

   from flydsl.extension import coop, quant, random

   # Equivalent inside traced code:
   import flydsl.expr as fx
   fx.coop.warp_reduce(...)
   fx.quant.pack_e4m3x4(...)
   fx.random.rand4x(...)

The dispatcher selects a target-specific implementation when one exists while
preserving the exported operation's contract.

Cooperative algorithms
----------------------

``flydsl.extension.coop`` provides warp- and block-scope collectives. Every
participating lane/thread must execute a collective with uniform control
options. ``width=None`` uses the target's native warp width (64 on gfx9 CDNA
targets; 32 on gfx10/gfx11/gfx12 targets, including CDNA5 ``gfx1250``); specify
``width`` when an algorithm requires the same logical group size across
targets.

Warp functions
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Behavior
   * - ``warp_reduce(value, op, *, width=None, valid_items=None)``
     - Reduce one value or a per-lane item range. The aggregate is returned to
       every participating lane.
   * - ``warp_head_segmented_reduce(value, head_flag, op, *, width=None)``
     - Reduce ascending-lane segments identified by a nonzero head flag;
       results are valid at segment heads.
   * - ``warp_tail_segmented_reduce(value, tail_flag, op, *, width=None)``
     - Tail-flag form of segmented reduction; results are valid at segment
       tails.
   * - ``warp_broadcast(value, src_lane, *, width=None)``
     - Broadcast a lane's scalar or aggregate value within each logical warp.
   * - ``warp_inclusive_scan`` / ``warp_exclusive_scan``
     - Ordered prefix scans with optional ``init`` and a leading
       ``valid_items`` count for scalar inputs.
   * - ``warp_scan``
     - Return ``(inclusive, exclusive)`` from one pass.
   * - ``warp_scan_with_aggregate``
     - Return ``(inclusive, exclusive, aggregate)``; the aggregate excludes
       ``init``.
   * - ``warp_reduce_batched``
     - Reduce batched/thread-packed inputs with an explicit output layout.
   * - ``warp_reduce_batched_to_blocked`` /
       ``warp_reduce_batched_to_striped``
     - Convenience result-layout variants for batched reduction.
   * - ``warp_bitonic_sort`` / ``warp_merge_sort``
     - Sort key or key/value inputs across a logical warp.

``op`` can be a built-in ``fx.ReductionOp`` or an associative callable. Some
algorithms can reorder operands only when the operation is declared
commutative. Floating-point reassociation can therefore change the last bits
even when the mathematical result is equivalent.

Warp primitive classes
~~~~~~~~~~~~~~~~~~~~~~

Class forms make data movement and reusable specializations explicit:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Purpose
   * - ``WarpReduce`` / ``WarpScan`` / ``WarpReduceBatched``
     - Reusable reduction and scan specializations corresponding to the
       function APIs.
   * - ``WarpLoad`` / ``WarpStore``
     - Move blocked/striped/transpose-oriented per-lane item arrays between
       memory and registers.
   * - ``WarpExchange``
     - Rearrange per-lane items between blocked and striped organizations.
   * - ``WarpBitonicSort`` / ``WarpMergeSort``
     - Reusable sorting-network specializations.
   * - ``WarpLoadAlgorithm``, ``WarpStoreAlgorithm``,
       ``WarpExchangeAlgorithm``
     - Algorithm selectors for the corresponding primitive classes.

Block primitives
~~~~~~~~~~~~~~~~

Block algorithms are specialized with subscription syntax. Their
``SharedStorage`` type must be allocated once and shared by all block threads:

.. code-block:: python

   block_scan = fx.coop.BlockScan[
       fx.Int32,
       fx.known_block_size(),
       fx.coop.BlockScanAlgorithm.WARP_SCANS,
   ]
   storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
   prefix = block_scan.exclusive(
       value,
       fx.ReductionOp.ADD,
       init=0,
       storage=storage,
   )

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - API
     - Contract
   * - ``BlockReduce[dtype, block_size, algorithm]``
     - Reduce scalar or per-thread item ranges in flattened blocked order.
       Every thread receives the aggregate. ``valid_items`` counts elements,
       not threads.
   * - ``BlockScan[dtype, block_size, algorithm]``
     - Provides ``inclusive``, ``exclusive``, and their
       ``*_with_aggregate`` variants, with optional initial value, valid-item
       count, identity, and prefix callback.
   * - ``BlockReduceAlgorithm``
     - Selects warp-reduction, raking, or commutative policies.
   * - ``BlockScanAlgorithm``
     - Selects warp-scan, raking, or memoized-raking policies.

Synchronize with ``fx.barrier()`` before reusing a collective's shared storage.
The block size must agree with the launched block and the specialization.

Runnable examples:

- ``examples/extension/coop/01-warp_collectives.py`` demonstrates warp
  reduction and exclusive scan without shared memory.
- ``examples/extension/coop/02-block_scan.py`` demonstrates a vector-valued
  block scan used for stream compaction.

Quantization codecs
-------------------

``flydsl.extension.quant`` provides register-scope micro-scaled and sub-byte
quantization codecs with portable bit-exact implementations under
``fx.quant.universal`` and target-specific hardware overrides under
``fx.quant.rocdl``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Behavior
   * - ``f32_to_e8m0`` / ``e8m0_to_f32``
     - Encode and decode OCP MX ``E8M0`` shared-exponent block scales.
   * - ``f32_to_ue4m3`` / ``ue4m3_to_f32``
     - Encode and decode non-negative ``UE4M3`` micro-block scales.
   * - ``pack_bf16x2`` / ``unpack_bf16x2``
     - Round-to-nearest-even ``Float32`` pair pack/unpack for ``bf16x2`` words.
   * - ``pack_e4m3x2`` / ``pack_e4m3x4`` / ``unpack_e4m3x4`` /
       ``unpack_e4m3x4_bf16x2``
     - OCP ``E4M3FN`` pack/unpack with optional block scale and lossless
       ``bf16x2`` register bridge.
   * - ``pack_e5m2x2`` / ``pack_e5m2x4`` / ``unpack_e5m2x4`` /
       ``unpack_e5m2x4_bf16x2``
     - OCP ``E5M2`` pack/unpack with optional block scale and lossless
       ``bf16x2`` register bridge.
   * - ``stoch_round_e4m3x4`` / ``stoch_round_e5m2x4``
     - Stochastic rounding from four ``Float32`` values and 32-bit entropy into
       packed OCP ``E4M3FN`` or ``E5M2``.
   * - ``pack_e2m1x8`` / ``unpack_e2m1x4`` / ``unpack_e2m1x8``
     - 8-way 4-bit ``E2M1`` (``MXFP4`` / ``NVFP4``) pack and ``Float32`` unpack.
   * - ``unpack_e2m1x4_bf16x2`` / ``unpack_e2m1x4_e4m3x4`` /
       ``unpack_e2m1x8_int4_pair``
     - Lossless register bridges from ``E2M1`` nibbles to ``bf16x2``, OCP
       ``E4M3FN``, and signed ``INT4`` ``(p, q)`` pairs (``p + q == 2 * w``).
   * - ``pack_int4x8`` / ``unpack_int4x8`` / ``unpack_uint4x8``
     - 8-way signed and unsigned 4-bit integer register pack/unpack.

See :doc:`../extension/quant` for architecture dispatch rules, True16 notes, and
``examples/extension/quant/01-microscaled_quant.py`` for a runnable ``MXFP4`` /
``NVFP4`` example.

Random-number generation
------------------------

``flydsl.extension.random`` implements stateless Philox-based generation. A
``seed`` selects a stream; ``offset`` selects a deterministic position in that
stream. Callers are responsible for assigning non-overlapping offsets when
threads produce more than one value.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - API
     - Result
   * - ``randint(seed, offset, n_rounds=10)``
     - One deterministic unsigned integer word.
   * - ``randint4x(seed, offset, n_rounds=10)``
     - Four consecutive integer words.
   * - ``rand(seed, offset, n_rounds=10)``
     - One uniform floating-point value.
   * - ``rand4x(seed, offset, n_rounds=10)``
     - Four uniform floating-point values.
   * - ``randn(seed, offset, n_rounds=10)``
     - One normally distributed value.
   * - ``randn4x(seed, offset, n_rounds=10)``
     - Four normally distributed values.

See :doc:`../extension/rand` for offset rules, vector generation, and a complete
kernel example.

Stability and dispatch
----------------------

An extension entry point, exported child namespace, and exported symbol form a
stable path according to :doc:`../api_stability`. ``fx.coop``/``fx.quant``/``fx.random``
aliases have the same stability as their canonical ``flydsl.extension`` paths.
Target-specific implementations must preserve the same exported contract.
