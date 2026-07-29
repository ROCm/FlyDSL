# The Fly Dialect I — Layout Algebra

This is the chapter that pays off the promise of Chapter 2: the tiling and
fragment math you do *by hand* in CK-Tile is, in FlyDSL, a small algebra of
first-class objects. Master four of them — **IntTuple**, **Layout**,
**coordinate mapping**, and **divide/product** — and the rest of the book is
applying them.

If you have read the CuTe / CUTLASS 3.x layout algebra, this will feel familiar:
FlyDSL's layout algebra is deliberately the same algebra. If you have not, this
chapter is self-contained. Every operation shown here has a Python surface in
`python/flydsl/expr/primitive.py` and lowers to a `fly.*` MLIR op (§2.5, Stage
A). The authoritative API table is `docs/layout_system_guide.md`; this chapter
teaches the *why*.

## IntTuple: structure as a value

An **IntTuple** is a (possibly nested) tuple of integers. That is the whole
definition, but it is load-bearing: shapes, strides, and coordinates are *all*
IntTuples, and their nesting encodes tiling hierarchy.

```python
fx.make_shape(8, 16)          # (8, 16)          rank-2, depth-1
fx.make_shape(9, (4, 8))      # (9, (4, 8))      rank-2, depth-2
```

Two queries you will use constantly:

- `size(t)` — product of all leaves. `size((9,(4,8))) = 9*4*8 = 288`.
- `rank(t)` — number of *top-level* modes. `rank((9,(4,8))) = 2`.
- `depth(t)` — nesting depth. `depth((9,(4,8))) = 2`.

> **HIP/CK-Tile → FlyDSL.** An IntTuple is CK-Tile's `Sequence<...>` /
> `Tuple<...>` of compile-time extents, except a leaf may also be a *runtime*
> SSA value (a dynamic dimension). Static leaves are baked into the IR (like
> `constexpr`); dynamic leaves become operands. The nesting is exactly the
> hierarchical descriptor structure you build with `make_naive_tensor_descriptor`
> — made explicit and manipulable.

## Layout: a function from coordinates to indices

A **Layout** is a pair `(Shape, Stride)` where both are IntTuples of matching
structure. Its meaning is a *function*: given a logical coordinate, it returns a
linear memory index.

```
Layout = Shape : Stride            written  (8,16):(1,8)
index(coord) = Σ  coord_i * stride_i
```

That single formula is the entire semantics. Example — an 8×16 **column-major**
tile:

```python
L = fx.make_layout((8, 16), (1, 8))     # (8,16):(1,8)
```

- `crd2idx((3, 5), L) = 3*1 + 5*8 = 43`
- `idx2crd(43, L)     = (43 % 8, 43 // 8) = (3, 5)`

Row-major is just a different stride:

```python
Lrow = fx.make_layout((8, 16), (16, 1))  # (8,16):(16,1)  row-major
# crd2idx((3,5)) = 3*16 + 5*1 = 53
```

Two derived sizes matter:

- `size(L)` — number of logical elements (`8*16 = 128`).
- `cosize(L)` — largest index reachable, plus one (the memory footprint). For a
  dense layout `size == cosize`; for a *strided view* into a larger buffer,
  `cosize > size`. This is how a layout describes a sub-tile of a bigger tensor
  without copying.

> **HIP/CK-Tile → FlyDSL.** A Layout *is* a tensor descriptor:
> `Shape:Stride` ≡ `(lengths, strides)` in a CK naive tensor descriptor, and
> `crd2idx` is exactly the offset computation `descriptor.CalculateOffset(idx)`.
> The difference is that in CK you evaluate that offset in device C++; in FlyDSL
> `crd2idx` is an *op* the compiler can fold, CSE, and vectorize (the
> `fly-layout-lowering` pass, §2.5).

### Nested layouts express hierarchy

Because Shape and Stride can nest, one Layout can describe a multi-level tiling.
A layout `((4,2),(8,16)) : ((1,32),(4,64))` reads as "mode 0 is a (4,2) group,
mode 1 is an (8,16) group," each with its own strides. You rarely write these by
hand — the *divide* and *product* operators below build them for you.

## The two operators: divide (tiling) and product (replication)

Everything in Chapters 4–6 is built from two families of operators.

- **Divide** takes a big layout and a **tiler** and splits it into
  `(tile, rest)`: "carve this tensor into tiles of this shape." This is how you
  go from a whole matrix to per-block / per-thread tiles.
- **Product** takes a small layout and replicates it by a **tiler**: "stamp this
  atom out across this many threads/iterations." This is how you go from a
  single copy/MMA atom to a full tiled operation.

They are (loosely) inverses: divide *partitions*, product *assembles*.

### logical_divide — the canonical tiling

`logical_divide(L, tiler)` reshapes each mode of `L` into `(tile, rest)`.
Concretely, take a 1-D vector of 24 elements and tile it by 4:

```python
V = fx.make_layout(24, 1)                 # 24:1
D = fx.logical_divide(V, fx.make_layout(4, 1))
# D has shape (4, 6):(1, 4)  →  (tile=4, rest=6)
```

Now `D[i, t]` addresses element `i` within tile `t`: coordinate `(2, 3)` is
element 2 of tile 3 = index `2*1 + 3*4 = 14`. The first mode is *within a tile*,
the second mode *selects the tile*. That split — inner = intra-tile, outer =
inter-tile — is the workhorse.

For a 2-D tensor you tile each mode with a 2-D tiler. Given a `(M,N)` matrix and
a `(BM,BN)` block tiler, `logical_divide` yields a 4-mode layout you can index
as `[(bm, bn), (tile_m, tile_n)]`.

### zipped_divide — tiles you can slice by block id

In real kernels you do not want "inner mode, outer mode" interleaved per axis;
you want **all the intra-tile modes grouped together** and **all the
inter-tile (block) modes grouped together**, so you can pick block `bid` with a
single slice. That is `zipped_divide`:

```python
# examples/03-tiledMma.py pattern
bA = fx.zipped_divide(A, (block_m, block_k))   # -> ((block_m,block_k), (num_m,num_k))
tile_for_block = fx.slice(bA, (None, bid))     # keep tile modes, fix block index
```

The result's mode 0 is the tile `(block_m, block_k)`; mode 1 enumerates the
blocks. `fx.slice(bA, (None, bid))` says "give me all of mode 0, at block
coordinate `bid` in mode 1" — i.e. *this block's tile*. This one line is FlyDSL's
equivalent of computing a block's base pointer and building a tile descriptor in
CK-Tile.

> **HIP/CK-Tile → FlyDSL.** `zipped_divide` + `slice` ≡ the CK-Tile idiom of
> `make_tile_window` / `pad_tensor_view` then advancing the window to the block's
> origin. `zipped_divide` is also CuTe's `zipped_divide`; `slice` with `None`
> placeholders is CuTe's `layout(_, bid)` mode-slicing.

The other divides are conveniences over the same idea:

| Divide | Grouping of result |
|--------|--------------------|
| `logical_divide` | per-mode `(tile, rest)` interleaved |
| `zipped_divide` | `((all tile modes), (all rest/block modes))` — sliceable by block |
| `tiled_divide` | like zipped but keeps the tile modes hierarchical |
| `flat_divide` | fully flattened result |

### products — stamping an atom across threads

Products go the other way. Given a small **value layout** (what one thread owns)
and a **thread layout** (how threads are arranged), a product builds the full
tile layout. The two you will meet are:

- `blocked_product` — threads take **contiguous** blocks (thread 0 owns the first
  chunk, thread 1 the next). Good spatial locality per thread.
- `raked_product` — threads are **interleaved** (thread 0 owns elements
  `0, T, 2T, …`). This is how you get coalesced global loads: consecutive threads
  touch consecutive addresses.

```python
# 4 threads over 12 elements, each owning 3
val = fx.make_layout(3, 1)
thr = fx.make_layout(4, 1)
blocked = fx.blocked_product(val, thr)   # thr t owns elems [3t, 3t+3)
raked   = fx.raked_product(val, thr)     # thr t owns elems t, t+4, t+8
```

You will almost never call these directly. Instead you describe the
thread/value split with `make_layout_tv` (§4) and let `make_tiled_copy` /
`make_tiled_mma` apply the right product internally. But knowing *blocked vs
raked* explains why one copy is coalesced and another is not.

> **HIP/CK-Tile → FlyDSL.** `raked_product` is the "coalesced" thread mapping you
> hand-write as `thread_id + i * blockDim` indexing; `blocked_product` is the
> `thread_id * ItemsPerThread + i` mapping. In CK-Tile these are choices baked
> into a `tile_distribution`; here they are named layout operators.

## Composition: chaining coordinate maps

`composition(A, B)` builds the layout whose map is `A ∘ B`: apply `B`, then `A`.
`result(x) = A(B(x))`. Use it to *re-index* one layout through another — e.g. to
apply a permutation, or to view a tile in a different order.

```python
result = fx.composition(memory_layout, access_pattern)
# result(x) = memory_layout(access_pattern(x))
```

Two helpers support it:

- `coalesce(L)` — simplify: flatten nesting and merge adjacent modes where the
  strides line up, *preserving the map and size*. Always safe; use it to make IR
  and prints readable.
- `complement(tiler, N)` — the "modes not covered by the tiler," up to `N`
  elements. It is the internal engine of `logical_divide` (it computes the
  *rest* part). You rarely call it directly, but seeing it in dumped IR is a sign
  a divide is being lowered.

## A complete worked example

Tile a `(64, 32)` row-major matrix into `(16, 16)` blocks and address block
`(bm, bn) = (1, 1)`, thread-local element `(2, 3)`.

```python
A  = fx.make_layout((64, 32), (32, 1))          # row-major
bA = fx.zipped_divide(A, (16, 16))              # ((16,16),(4,2))
blk = fx.slice(bA, (None, fx.make_coord(1, 1))) # this block's (16,16) tile
idx = fx.crd2idx(fx.make_coord(2, 3), blk)      # element within the block
```

By hand: block `(1,1)` starts at row 16, col 16 → base index `16*32 + 16 = 528`.
Element `(2,3)` within it adds `2*32 + 3*1 = 67`. So `idx = 595`. The algebra
computes exactly this, but as ops the compiler can fold to a constant when the
shapes are static (they are here), or vectorize when they are not.

That is the loop you will repeat for every kernel: **make a layout for the whole
tensor → divide into tiles → slice the block's tile → partition across threads →
operate.** The next chapter formalizes the "partition across threads" half.
