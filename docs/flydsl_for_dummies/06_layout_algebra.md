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

Everything in Chapters 8–9 is built from two families of operators.

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

## Layout constructors: `make_ordered_layout` and `make_view`

### `make_ordered_layout` — compact layout from shape and stride order

`make_layout(shape, stride)` requires you to supply exact stride values.
`make_ordered_layout(shape, order)` is a convenience that computes compact
(densely packed, no gaps) strides automatically, given only the *order* in which
the modes should iterate:

```python
fx.make_ordered_layout((M, N), (0, 1))   # column-major: M is fastest (stride 1)
fx.make_ordered_layout((M, N), (1, 0))   # row-major:    N is fastest (stride 1)
```

`order[i]` is the *rank* of mode `i` among all modes, where 0 means "innermost /
fastest / stride-1 candidate." The resulting strides are the sorted prefix
products of the shape in that order:

```
make_ordered_layout((tile_m, tile_k), (1, 0))
  order: tile_k is innermost (order 0), tile_m is outermost (order 1)
  stride of tile_k = 1
  stride of tile_m = tile_k
  result: (tile_m, tile_k) : (tile_k, 1)   ← row-major
```

This is the standard way to describe an LDS buffer whose rows are contiguous —
the pattern used by the preshuffle GEMM's LDS A tile:

```python
fx.make_ordered_layout((tile_m, tile_k), (1, 0))  # tile_k is contiguous in LDS
```

### `make_view` — reinterpret a pointer through any layout

`fx.make_view(ptr, layout)` attaches a layout to a raw pointer and produces an
`fx.Tensor`. No data is moved; it is purely a type-level reinterpretation.

```python
smem_ptr = fx.get_dyn_shared(fx.Float16)       # raw LDS pointer
sA = fx.make_view(smem_ptr, fx.make_ordered_layout((tile_m, tile_k), (1, 0)))
# sA is now a (tile_m × tile_k) f16 tensor living in shared memory
```

Typical uses:

- Give structure to a flat LDS allocation (`get_dyn_shared` returns a 1-D
  pointer; `make_view` imposes the 2-D tile shape).
- Reinterpret typed LDS — e.g. after `recast_iter` to a narrower dtype, attach
  the wider shape back.
- Construct a tensor from an arbitrary `inttoptr`'d address for scale tensors or
  opaque pointer arithmetic.

The layout you pass becomes the tensor's `layout` property; all subsequent
`zipped_divide`, `partition_S`, `retile`, etc. operate on it normally.

> **HIP/CK-Tile → FlyDSL.** `make_view(ptr, layout)` ≡ constructing a CK-Tile
> `tensor_view` or `naive_tensor_view` from a pointer + descriptor. It is also the
> FlyDSL equivalent of `CuTe::make_tensor(ptr, layout)`.

## Composed layouts and XOR swizzle

A plain `Layout` maps a logical coordinate to a flat integer offset: it is a
linear function. Bank-conflict elimination needs a *non-linear* rearrangement of
that offset — the XOR swizzle. FlyDSL represents this as a **composed layout**:
an outer layout followed by a non-linear inner transform.

### `make_composed_layout` — stacking a transform on top of a layout

```python
fx.make_composed_layout(inner, outer)
fx.make_composed_layout(inner, offset, outer)
```

`outer` is a plain `Layout`; `inner` is a `SwizzleType` (or another layout).
Evaluation is right-to-left in three steps:

1. **`outer` maps the logical coordinate to a flat integer offset.**
2. **`offset` is added to that result.**
3. **`inner` transforms the sum** — for a `SwizzleType`, this is an XOR; for a
   plain `Layout`, it is another `crd2idx` call.

The optional `offset` is a plain integer constant (an `IntTuple` leaf, stored
as `fly.int_tuple<N>`). You almost never supply it by hand — the two-argument
form `make_composed_layout(inner, outer)` sets it automatically to the
*coprofile* of `outer`, which is **zero in every mode**:

```python
make_composed_layout(swz, layout)       # offset = coprofile(layout) = 0
make_composed_layout(swz, 16, layout)   # offset = 16 (manually chosen)
```

`coprofile(layout)` produces a zero tuple whose structure matches `outer`'s
stride profile, so when you add it to the coordinate-mapped offset the result
is unchanged — the offset is a no-op placeholder that keeps the IR structure
uniform. You will see `o 0 o` in MLIR dumps:

```
!fly.composed_layout<S<3,3,3> o 0 o (tile_m, tile_k):(tile_k, 1)>
```

**When does the offset become non-zero?** The compiler uses it internally
when you *slice* a composed-layout tensor. Slicing fixes one mode of the
`outer` layout to a constant coordinate; the compiler folds that constant
into the `offset` field so the inner transform always receives the correct
starting position, even after the outer layout has been simplified away
(`sliceComposedLayoutAttr` in `FlyOps.cpp:53`). You never write these
non-zero offsets yourself — they are an IR-level detail generated by the
lowering passes.

The one case where you *might* set a non-zero offset manually is when
composing across a static sub-tile boundary. For example, if two LDS buffers
share one allocation and the second starts at element 4096:

```python
fx.make_composed_layout(swz, 4096, layout_b)
# coordinate c -> offset  layout_b(c) + 4096  -> swz(result)
```

This is rare; in practice `SharedAllocator` and `SharedStorage` struct fields
handle sub-allocation automatically.

```python
swz = fx.static(fx.SwizzleType.get(3, 3, 3))
layout = fx.make_ordered_layout((tile_m, tile_k), (1, 0))
sA = fx.make_view(smem_ptr, fx.make_composed_layout(swz, layout))
```

Reading this: a logical `(m, k)` coordinate is first mapped to a flat LDS offset
by `layout`, then that offset is XOR'd by `swz` to produce the swizzled address.
The copy and partition machinery in `make_tiled_copy` sees the composed layout as
opaque — it passes coordinates through both layers automatically.

> **HIP/CK-Tile → FlyDSL.** A composed layout is CK-Tile's `ComposedTensorView`
> or the `xor_swizzle` + `TensorView` idiom. In CUTLASS 3 it is CuTe's
> `ComposedLayout(swizzle, offset, layout)`. The argument order in FlyDSL is
> `(inner, [offset,] outer)` — `outer` is applied first.

### `SwizzleType` — the XOR swizzle descriptor

`fx.SwizzleType.get(mask, base, shift)` constructs a swizzle descriptor with
three parameters, all in units of **bits**:

| Parameter | Role |
|-----------|------|
| `mask`    | Number of address bits XOR'd. The mask covers bits `base+shift .. base+shift+mask-1`. |
| `base`    | Lowest bit position of the source field (the bits that drive the XOR). |
| `shift`   | How far to shift the source field before XOR-ing. |

The formula applied to an offset `v` is (`IntUtils.cpp:237`).

First, extract the `mask`-bit field of `v` at bit position `base + shift`:

```
src = floor(v / 2^(base+shift))  mod  2^mask
```

Then XOR that field back into `v`, shifted down to bit position `base`:

```
swizzled(v) = v  XOR  (src × 2^base)
```

**What XOR does on integers.** Both `v` and `src` are non-negative integers;
XOR treats them as binary strings and flips each bit of `v` where the
corresponding bit of `src × 2^base` is 1, leaving all other bits of `v`
unchanged. In arithmetic terms, for each bit position `i`:

```
bit i of (a XOR b) = (bit i of a) + (bit i of b)  mod 2
```

So `a XOR b = a + b` when `a` and `b` share *no* set bits in common (no carry
ever propagates), and differs from addition wherever both have a 1-bit (addition
would carry; XOR does not). Here `v` and `src × 2^base` *are* guaranteed
non-overlapping: `src` occupies only bits `base … base+mask−1`, while the source
field in `v` sits at `base+shift … base+shift+mask−1` (shifted up by `shift ≥ 1`
bits). So inside the swizzled range `swizzled(v) = v + src × 2^base`, and
everywhere else `v` is untouched. Two useful properties follow directly:

- **Self-inverse:** `swizzled(swizzled(v)) = v`, because XOR-ing the same bits
  twice restores the original. The same transform un-swizzles as swizzles.
- **Bijective:** every `v` maps to a distinct `swizzled(v)` — no two inputs
  produce the same output — because the function is its own inverse.

`src` is a `mask`-bit integer (values 0 … 2^mask − 1), so `src × 2^base` occupies
bits `base … base+mask−1` with no overlap outside that range. Because those target
bits differ from the source bits (`base+shift … base+shift+mask−1`) by exactly
`shift` positions, the XOR mixes a higher-order part of the address (the "row")
into a lower-order part (the "column"), making threads that differ in row index
land in different bank groups.

The C source reads the same thing with bitwise operators:

```
bit_mask = ((1 << mask) - 1) << (base + shift)   # selects the source field
swizzled = v ^ ((v & bit_mask) >> shift)           # extract, descend, XOR
```

The **period** of the swizzle (the tile size over which the XOR pattern repeats)
is `2^(mask + base + shift)` elements (`FlyAttrDefs.td:146`).

#### Choosing parameters for LDS bank-conflict elimination

LDS on CDNA is 64 banks × 4 bytes, for a 256-byte bank period. Each contiguous
16 bytes (128-bit) of an LDS line maps to one bank group. The goal of swizzling
is to make threads that would collide in the same bank group address different
banks instead, by XOR-ing the row index into the column address.

For **f32 data, `tile_k` contiguous** (row-major LDS tile):

```python
swz = fx.SwizzleType.get(3, 3, 3)   # period = 2^(3+3+3) = 512 elements = 2048 B
```

This is the canonical f32/f16 GEMM swizzle in `preshuffle_gemm.py:274`. For
**8-bit data** the tile is narrower, so the swizzle must be narrower too —
`preshuffle_gemm.py:272` computes:

```python
k_blocks16 = (tile_k * elem_bytes) // 16   # number of 16-byte blocks per row
swz_bits   = (k_blocks16).bit_length() - 1  # log2
swz = fx.SwizzleType.get(swz_bits, 4, swz_bits)
```

#### What the swizzle does to addresses

For `tile_k = 16` f32 elements (64 bytes per row), `SwizzleType.get(3, 3, 3)`.

The table below shows **element offsets** stored in each LDS slot `(m, k)`.

**Bank formula** — identical on both CDNA generations, only `num_banks` differs:

```
bank = (byte_address / 4) % num_banks = element_offset % num_banks   (f32)

gfx942 (CDNA3, MI300X/MI325X):  32 banks × 4 B  →  128-byte period
gfx950 (CDNA4, MI350/MI355X):   64 banks × 4 B  →  256-byte period
```

For a row-major tile with `tile_k = 16` (stride 16 elements = 64 bytes/row):

| Arch | Banks | Bank period | Same bank repeats every |
|------|-------|-------------|-------------------------|
| gfx942 | 32 | 32 f32 = 128 B | **2 rows** (2×16 = 32) |
| gfx950 | 64 | 64 f32 = 256 B | **4 rows** (4×16 = 64) |

In the MFMA(16,16,4,f32) TV layout, 16 lanes read column k=0 simultaneously
(one lane per row, m=0..15). Without swizzle on **gfx942** (32 banks):

```
m=0  → offset   0  → bank  0
m=1  → offset  16  → bank 16
m=2  → offset  32  → bank  0  ← conflict with m=0 (repeats every 2 rows)
m=3  → offset  48  → bank 16  ← conflict with m=1
...                              (repeats for m=4..15)
```

16 lanes share only **2 banks** — an **8-way conflict** on gfx942. On gfx950
the same tile produces a **4-way conflict** (4-row period, 4 distinct banks).

**With swizzle(3,3,3).** The swizzle's source field starts at bit 6
(offset ≥ 64), so rows m=0..3 (offsets 0..63) are **unchanged** — the XOR is 0.
The rearrangement only kicks in at m=4 (offset 64 and above):

```
Without swizzle:                    With swizzle(3,3,3):

  k:  0  1  2 ... 7  8  9 ... 15     k:  0  1  2 ... 7  8  9 ... 15
m=0 [ 0  1  2 ... 7  8  9 ... 15]  m=0 [ 0  1  2 ... 7  8  9 ... 15]  (unchanged)
m=1 [16 17 18 ... 23 24 25 ... 31]  m=1 [16 17 18 ... 23 24 25 ... 31]  (unchanged)
m=2 [32 33 34 ... 39 40 41 ... 47]  m=2 [32 33 34 ... 39 40 41 ... 47]  (unchanged)
m=3 [48 49 50 ... 55 56 57 ... 63]  m=3 [48 49 50 ... 55 56 57 ... 63]  (unchanged)
m=4 [64 65 66 ... 71 72 73 ... 79]  m=4 [72 73 74 ... 79 64 65 ... 71]  ← XOR 8
m=5 [80 81 82 ... 87 88 89 ... 95]  m=5 [88 89 90 ... 95 80 81 ... 87]  ← XOR 8

slot (m=4, k=0): offset 64 → swizzled 72 → bank 72%32 = 8 ≠ 0 ✓
slot (m=5, k=0): offset 80 → swizzled 88 → bank 88%32 = 24 ≠ 16 ✓
```

Column k=0 banks across all 16 lanes with swizzle(3,3,3):

```
m:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
swz:  0  16  32  48  72  88 104 120 144 128 176 160 216 200 248 232
b42:  0  16   0  16   8  24   8  24  16   0  16   0  24   8  24   8
                                             (gfx942, 32 banks)
```

4 distinct banks {0, 8, 16, 24}, each hit 4 times: the 8-way conflict becomes
a **4-way conflict**. Swizzle(3,3,3) halves the conflict depth on gfx942.

On **gfx950** (64 banks), the same swizzle produces 8 distinct banks
{0, 8, 16, 24, 32, 40, 48, 56}, each hit twice — a **2-way conflict**.
Swizzle(3,3,3) was designed for the 64-bank gfx950 period (512-element swizzle
range covers two full 256-byte bank cycles); on gfx942 its reach only partially
covers the 128-byte bank period, leaving residual conflicts. A swizzle with a
larger `shift` value would be needed to fully eliminate them on gfx942.

In practice, `preshuffle_gemm.py` uses wider tiles (`tile_k ≥ 32` for fp16),
which bring the unswizzled conflict down to fewer ways and make swizzle(3,3,3)
sufficient for acceptable performance on both architectures.

#### Putting it all together

The complete LDS-A pattern from `preshuffle_gemm.py:277`:

```python
swz = fx.SwizzleType.get(3, 3, 3)

sA = fx.make_view(
    smem_ptr,
    fx.make_composed_layout(
        fx.static(swz),
        fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
    ),
)
```

Step by step:

1. `make_ordered_layout((tile_m, tile_k), (1, 0))` builds a row-major layout
   where `tile_k` is the contiguous (fastest) dimension with stride 1, and
   `tile_m` has stride `tile_k`. This is the *nominal* memory layout.
2. `fx.static(swz)` boxes the swizzle descriptor into a compile-time constant.
3. `make_composed_layout(swz, layout)` stacks the XOR transform on top: a
   logical `(m, k)` coordinate → flat offset `m*tile_k + k` → XOR'd address.
4. `make_view(smem_ptr, ...)` attaches the composed layout to the LDS pointer,
   producing an `fx.Tensor` with shape `(tile_m, tile_k)`.

After this, `thr_g2s.partition_D(sA)` computes per-thread destination addresses
using the composed layout, so every `ds_write` goes to a swizzled bank. The
matching `thr_s2r.partition_S(sA)` uses the same layout for the `ds_read` side,
so loads and stores stay coherent with no explicit address math.

**Why both sides must use the same swizzled view.** The swizzle remaps which
LDS offset `(m, k)` is stored to — it is a bijection, so no element is lost.
On the write side, element `(m, k)` lands at `swizzled(m*tile_k + k)`. On the
read side, the same expression is evaluated, so the read finds the element exactly
where the write left it. Using a plain layout on one side and a swizzled layout on
the other would silently read the wrong element — the kind of "plausible garbage"
bug that is hard to debug.

**The conflict-free guarantee is scoped to the copy atom's TV layout.** The
swizzle parameters are chosen so that the LDS addresses the `make_tiled_copy`
TV layout assigns to one wave span distinct banks. If you access the same `sA`
tensor with a *different* access pattern — for example a scalar `sA[m, k]` loop
— you get whatever bank mapping the swizzle happens to produce for that pattern,
with no bank-conflict guarantee. The swizzle is not a property of the memory; it
is a property of the (layout, access pattern) pair.

### `CoordSwizzleType` — coordinate-space swizzle

`SwizzleType` works on a **flat integer offset** after `crd2idx` has collapsed
the logical coordinate to one number. `CoordSwizzleType` works one step earlier —
on the **multi-dimensional logical coordinate itself**, before any stride
multiplication. This difference gives it more flexibility: it can mix bits between
two *independent* coordinate modes without having to know the stride between them.

#### Constructor

```python
fx.CoordSwizzleType.get(mask, base_row, mode_row, base_col, mode_col)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `mask` | int | number of bits XOR'd (same role as in `SwizzleType`) |
| `base_row` | int | bit position within the row-mode value to read from |
| `mode_row` | list[int] | index path into the coordinate tuple selecting the *row* mode |
| `base_col` | int | bit position within the col-mode value to write to |
| `mode_col` | list[int] | index path into the coordinate tuple selecting the *col* mode |

The MLIR assembly form is `CS<mask,base_row,[mode_row],base_col,[mode_col]>`.
`CS<3,0,[1],0,[2]>` reads from mode 1 and XORs into mode 2.

#### Formula

Given a logical coordinate `c = (c_0, c_1, …)`:

```
row_bits = floor(c[mode_row] / 2^base_row)  mod  2^mask
c[mode_col]  ←  c[mode_col]  XOR  (row_bits × 2^base_col)
```

In C:

```
maskBits = (1 << mask) - 1
rowBits  = (c[mode_row] >> base_row) & maskBits
c[mode_col] ^= rowBits << base_col
```

The result is a modified coordinate; all other modes are unchanged. The swizzled
coordinate is then passed to `crd2idx` with the outer layout to produce the final
address.

#### `SwizzleType` vs `CoordSwizzleType` — when to use which

| | `SwizzleType` | `CoordSwizzleType` |
|---|---|---|
| Operates on | flat offset (post-`crd2idx`) | logical coordinate (pre-`crd2idx`) |
| Knows about strides | implicitly (it XORs bit positions that align with stride bits) | no — XORs coordinate modes directly |
| Requires stride match | yes — shift/base must align with the tile's stride bits | no — mode paths are stride-independent |
| Typical use | row-major LDS tile, fixed-stride layouts | transposed LDS layouts where row/col are independent modes |

The practical consequence: `SwizzleType.get(3, 3, 3)` on a row-major `(tile_m,
tile_k):(tile_k, 1)` layout works because `tile_k` is a power of 2 and the row
bits land at the right bit positions in the flat offset. If you have a
**non-power-of-two** stride, or a layout whose row and column are genuinely
independent modes (e.g. after a `make_identity_layout` that keeps modes separate),
`SwizzleType` cannot express the XOR correctly — `CoordSwizzleType` can, because
it reads directly from the mode you name, regardless of stride.

#### Concrete example

Suppose you have a 3-mode LDS layout `(4, 8, 8):(1E0, 1E1, 1E2)` (identity-like,
basis strides), and you want to XOR 3 bits of mode 1 (the "row") into mode 2 (the
"column"):

```python
cs = fx.static(fx.CoordSwizzleType.get(
    mask=3,
    base_row=0, mode_row=[1],   # read bits 0..2 from mode 1
    base_col=0, mode_col=[2],   # write into bits 0..2 of mode 2
))
sA = fx.make_view(smem_ptr, fx.make_composed_layout(cs, layout_3d))
```

MLIR prints this as:
```
!fly.composed_layout<CS<3,0,[1],0,[2]> o (0,0,0) o (4,8,8):(1E0,1E1,1E2)>
```

For coordinate `(r, m, k)`: `k_swizzled = k XOR (m & 7)`. The flat address
depends on `crd2idx((r, m, k_swizzled), layout)`. Because the XOR happens in
coordinate space, it is independent of the strides — a different stride would
give a different address but the same XOR pattern.

#### Composing two swizzles

The `inner` slot of `make_composed_layout` accepts **any** inner type —
`SwizzleType`, `CoordSwizzleType`, or another `ComposedLayout`. Nesting a
`ComposedLayout` inside another gives a *chain* of swizzles; each outer layer
adds one more layout step before the innermost transform.

```python
# inner SwizzleType + outer layout = one swizzle level
cl1 = fx.make_composed_layout(swz1, layout1)

# nest cl1 as the outer of a second level
cl2 = fx.make_composed_layout(swz2, cl1)
```

The IR prints as a chain:
```
CS<...> o (0,0,0) o layout1  o (offset2) o layout2
         └── inner swizzle    └── second layout + offset
```

Evaluation during address calculation (`decomposeComposedLayoutValue` in
`LayoutLowering.cpp:2560`):

1. Start with the logical coordinate `c`.
2. Apply the **outermost** linear layout (`layout2`) → offset₂.
3. Add the **outer offset** (the `o (offset2)` field) → running total.
4. Apply the **next** layer's linear layout (`layout1`) → offset₁.
5. Add the **inner offset** → running total.
6. Apply the **innermost** transform (`swz1`) → final address.

In practice two stacked swizzles are rare; the compiler uses nesting internally
when slicing a composed-layout tensor (each slice folds a constant into the offset
field of the existing layer rather than rebuilding from scratch). You will see it
in IR dumps as increasingly long `o … o` chains, but you write only one
`make_composed_layout` call per deliberate swizzle level.

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
