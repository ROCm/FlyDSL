# The Fly Dialect II — Tiling and Partitioning

Chapter 6 gave you layouts and the divide/product operators. This chapter is
about the step every kernel performs next: taking a block's tile and **handing
each thread its slice** — the FlyDSL equivalent of CK-Tile's `tile_distribution`
and CuTe's *thread-value (TV) layout*.

The mental picture: a tile is a 2-D array of elements; a wavefront is 64 threads;
each thread must own a specific subset of those elements, laid out in its
registers. The object that encodes "which thread owns which elements, and in what
register order" is the **TV layout**. Everything else — `partition_S`,
`partition_D`, `make_fragment_*` — is applying that TV layout to a concrete
tensor.

## The thread-value (TV) layout

A TV layout is a single Layout whose domain is a coordinate `(tid, vid)`:

```
TV(tid, vid) = the tile-local index of value vid owned by thread tid
```

- `tid` ranges over threads (0..63 for a wave, or 0..255 for a 256-thread block).
- `vid` ranges over the values (registers) that thread owns.

Read one way it says *"thread `tid`, give me your `vid`-th element → here is where
it lives in the tile."* Read the other way (its inverse) it answers *"element `e`
of the tile → which thread owns it and at which register."* That duality is the
whole point: it simultaneously defines the load pattern (coalescing) and the
register fragment (what feeds the MMA).

You build one from a **thread layout** and a **value layout**:

```python
thr_layout = fx.make_layout((16, 16), (16, 1))  # 256 threads as 16x16
val_layout = fx.make_layout((1, 1), (1, 1))     # 1 value per thread
tiler_mn, layout_tv = fx.make_layout_tv(thr_layout, val_layout)
```

`make_layout_tv` returns the tile shape it covers (`tiler_mn`) and the TV layout.
Choosing `thr_layout` is choosing your access pattern: a thread layout whose
fastest mode matches the tile's contiguous axis gives **coalesced** loads (this
is the `raked_product` from §6.3.3 happening under the hood).

> **HIP/CK-Tile → FlyDSL.** The TV layout *is* a `tile_distribution`: it is the
> single object encoding the H/Y/P/R mapping from `(warp, lane, register)` to
> tensor coordinates. Where CK-Tile makes you assemble it from
> `tile_distribution_encoding<>` template parameters, FlyDSL builds it from two
> ordinary Layouts via `make_layout_tv`. It is CuTe's TV layout exactly.

## TiledCopy and TiledMma: TV layouts with an atom attached

A bare TV layout says *how* to distribute elements. A **TiledCopy** or
**TiledMma** binds that distribution to a hardware **atom** (Chapters 9–9) so the
compiler knows *what instruction* moves/computes each thread's slice.

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tiler_mn)

mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
```

- `make_tiled_copy(atom, tv, tile)` — replicate a copy atom over the TV layout to
  cover `tile` elements.
- `make_tiled_mma(atom, atom_layout[, perm])` — replicate an MMA atom. The
  `atom_layout` says how many atoms tile the M/N/K space (here a 2×2 grid of MFMA
  atoms), which sets the warp-level tile size.

A crucial convenience for GEMM: you can derive matching copies *from* a TiledMma
so the copied fragments land exactly where the MMA expects them:

```python
tiled_copy_a = fx.make_tiled_copy_A(copy_atom, tiled_mma)
tiled_copy_b = fx.make_tiled_copy_B(copy_atom, tiled_mma)
tiled_copy_c = fx.make_tiled_copy_C(copy_atom, tiled_mma)
```

This is the FlyDSL answer to CK-Tile's requirement that your global→LDS→register
copies produce exactly the fragment layout the MFMA consumes. Here it is derived,
not hand-matched.

## Per-thread slicing: from tile to registers

With a TiledCopy/TiledMma in hand, each thread narrows the distribution to *its*
slice using its thread id.

```python
tid = fx.thread_idx.x

thr_copy = tiled_copy.get_slice(tid)     # a ThrCopy for this thread
thr_mma  = tiled_mma.thr_slice(tid)      # a ThrMma  for this thread
```

Then you *partition* concrete tensors through the slice:

```python
src_part = thr_copy.partition_S(gmem_tile)   # this thread's source elements
dst_part = thr_copy.partition_D(smem_tile)   # this thread's destination slots
```

`partition_S`/`partition_D` apply the TV layout to a tensor, returning a small
per-thread view whose shape is "values this thread owns." That view is what you
copy or compute on — you never index raw `tid` arithmetic yourself.

> **HIP/CK-Tile → FlyDSL.** `partition_S(tile)` ≡ taking a `tile_window` and
> `load_tile`-ing the current thread's distributed slice; the returned view plays
> the role of the CK `static_distributed_tensor` you'd read into. The difference
> is you don't write the index math — the TV layout already contains it.

## Register fragments

The register-resident operands of an MMA are **fragments**. From a ThrMma you
allocate them shaped to match the MMA atom:

```python
frag_A = thr_mma.make_fragment_A(bA_tile)   # A operand fragment
frag_B = thr_mma.make_fragment_B(bB_tile)   # B operand fragment
frag_C = thr_mma.make_fragment_C(bC_tile)   # accumulator fragment
frag_C.fill(0)                              # zero the accumulator
```

`make_fragment_*` returns register tensors whose layout is exactly what
`fx.gemm` will feed to `rocdl.mfma.*`. When you load into them via a TiledCopy,
you often need to *retile* the copied data to the fragment's view:

```python
frag_A_copy = thr_copy_a.retile(frag_A)     # match copy layout to fragment layout
fx.copy(copy_atom_a, thr_copy_a.partition_S(bA_tile), frag_A_copy)
```

`retile` reconciles the copy's TV layout with the fragment's TV layout so a
`buffer_load` lands the bytes exactly where the matrix core reads them. This is
the single most error-prone step when hand-writing MFMA kernels in C++; in
FlyDSL it is one call whose correctness is derived from the atoms.

> **HIP/CK-Tile → FlyDSL.** `make_fragment_A/B/C` ≡ declaring the
> `vgpr`-resident operand/accumulator arrays whose element order matches the
> `__builtin_amdgcn_mfma_*` ABI. `frag_C.fill(0)` is your accumulator
> zero-init. `retile` is the "shuffle the loaded tile into MFMA operand order"
> you normally do with careful indexing or an LDS round-trip.

## Putting the partition pipeline together

The canonical block-level flow, distilled from `examples/03-tiledMma.py`:

```python
# 1. whole-tensor layouts (buffer tensors, §5)
A = fx.rocdl.make_buffer_tensor(A_ptr, ...)      # (M,K)
B = fx.rocdl.make_buffer_tensor(B_ptr, ...)      # (N,K)  -> computes A @ B.T
C = fx.rocdl.make_buffer_tensor(C_ptr, ...)

# 2. tile + slice this block (Ch. 6)
bA = fx.slice(fx.zipped_divide(A, (block_m, block_k)), (None, bid_m))
bB = fx.slice(fx.zipped_divide(B, (block_n, block_k)), (None, bid_n))
bC = fx.slice(fx.zipped_divide(C, (block_m, block_n)), (None, bid))

# 3. build the tiled MMA and matching copies
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
thr_mma   = tiled_mma.thr_slice(tid)

# 4. fragments + compute
frag_A = thr_mma.make_fragment_A(bA)
frag_B = thr_mma.make_fragment_B(bB)
frag_C = thr_mma.make_fragment_C(bC)
frag_C.fill(0)
# ... copy into frag_A/frag_B (Ch. 8) ...
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)   # D = A*B + C
# ... store frag_C to bC (Ch. 8) ...
```

Steps 1–2 are Chapter 6 (layout algebra), step 3 is this chapter (TV
layouts/atoms), step 4 previews Chapters 9–9 (copy and MMA atoms). The rest of
the book fills in the "copy into fragments" and "store" ellipses, and then scales
this single-block kernel into a pipelined GEMM.
