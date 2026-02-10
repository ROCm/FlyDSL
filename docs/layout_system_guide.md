# FLIR Layout Algebra Guide

> Core types, construction, coordinate mapping, algebra operations, and swizzling in the FLIR layout system.

## Quick Reference

| Operation | Python API | MLIR Op | Description |
|---|---|---|---|
| **Construction** | `flir.make_shape(8, 16)` | `flir.make_shape` | Create shape |
| | `flir.make_stride(1, 8)` | `flir.make_stride` | Create stride |
| | `flir.make_layout(shape, stride)` | `flir.make_layout` | Create layout from (shape, stride) |
| | `flir.make_coord(i, j)` | `flir.make_coord` | Create coordinate |
| | `flir.make_ordered_layout((8, 16), order=(1, 0))` | -- | Create layout with dimension ordering |
| **Mapping** | `flir.crd2idx(coord, layout)` | `flir.crd2idx` | Coordinate → linear index |
| | `flir.idx2crd(idx, layout)` | `flir.idx2crd` | Linear index → coordinate |
| **Query** | `flir.size(layout)` | `flir.size` | Total element count |
| | `flir.cosize(layout)` | `flir.cosize` | Codomain span |
| | `flir.rank(layout)` | `flir.rank` | Number of dimensions |
| | `flir.get(shape, idx)` | `flir.get` | Extract element at index |
| **Algebra** | `flir.composition(A, B)` | `flir.composition` | Compose: A ∘ B |
| | `flir.complement(tiler, size)` | `flir.complement` | Complement of tiler |
| | `flir.coalesce(layout)` | `flir.coalesce` | Simplify layout |
| **Products** | `flir.logical_product(A, B)` | `flir.logical_product` | Basic product |
| | `flir.zipped_product(A, B)` | `flir.zipped_product` | Zipped product |
| | `flir.tiled_product(A, B)` | `flir.tiled_product` | Tiled product |
| | `flir.flat_product(A, B)` | `flir.flat_product` | Flat product |
| | `flir.raked_product(A, B)` | `flir.raked_product` | Raked product |
| | `flir.blocked_product(A, B)` | `flir.blocked_product` | Blocked product |
| **Divides** | `flir.logical_divide(A, B)` | `flir.logical_divide` | Basic divide |
| | `flir.zipped_divide(A, B)` | `flir.zipped_divide` | Zipped divide |
| | `flir.tiled_divide(A, B)` | `flir.tiled_divide` | Tiled divide |
| | `flir.flat_divide(A, B)` | `flir.flat_divide` | Flat divide |
| **Partition** | `flir.local_partition(layout, tile, idx)` | `flir.local_partition` | Thread-local partition |
| | `flir.local_tile(layout, tiler, coord)` | `flir.local_tile` | Extract tile at coordinate |

---

## 1. Core Types

FLIR defines five custom MLIR types:

| Type | MLIR Syntax | Description |
|---|---|---|
| `!flir.int` | `!flir.int` | Legacy integer type (compatibility) |
| `!flir.shape<pattern>` | `!flir.shape<(8, 16)>` | Dimension extents -- can be nested |
| `!flir.stride<pattern>` | `!flir.stride<(1, 8)>` | Memory strides -- can be nested |
| `!flir.layout<shape:stride>` | `!flir.layout<(8, 16):(1, 8)>` | Layout = (Shape, Stride) pair |
| `!flir.coord<pattern>` | `!flir.coord<(*, *)>` | Coordinate (flat, no nesting) |

### Pattern Attributes

Patterns encode the structure of shapes/strides/coords at the type level:

| Pattern | Meaning | Example |
|---|---|---|
| Integer literal | Static constant | `8` |
| `*` (`#flir.underscore`) | Wildcard (any value) | Used in rank-only types |
| `?` (`#flir.dync_i32`) | Dynamic value (runtime) | SSA operand provides the value |
| Nested tuple | Hierarchical mode | `(8, (4, 2))` |

**Type-mode ops**: `make_shape`, `make_stride`, `make_coord` encode structure in the result type. Only dynamic leaves appear as SSA operands.

---

## 2. Construction

### Python API

```python
from flydsl.dialects.ext import flir, arith

# Constants
c8 = arith.constant(8, index=True)
c16 = arith.constant(16, index=True)
c1 = arith.constant(1, index=True)

# Basic construction
shape = flir.make_shape(c8, c16)            # (8, 16)
stride = flir.make_stride(c1, c8)           # (1, 8) -- column-major
layout = flir.make_layout(shape, stride)    # ((8, 16), (1, 8))
coord = flir.make_coord(i, j)              # (i, j)

# Static constants (Python ints auto-materialized)
shape = flir.make_shape(8, 16)              # Same as above
layout = flir.make_layout((8, 16), (1, 8))  # Tuple shorthand

# Nested shapes
shape_nested = flir.make_shape(9, (4, 8))   # (9, (4, 8))

# Ordered layout (convenience)
layout_cm = flir.make_ordered_layout((8, 16), order=(1, 0))  # col-major
layout_rm = flir.make_ordered_layout((8, 16), order=(0, 1))  # row-major
```

### MLIR

```mlir
// Create 2D layout (8, 16) with column-major stride (1, 8)
%shape = flir.make_shape %c8, %c16 : (!flir.int, !flir.int) -> !flir.shape<2>
%stride = flir.make_stride %c1, %c8 : (!flir.int, !flir.int) -> !flir.stride<2>
%layout = flir.make_layout %shape, %stride
    : (!flir.shape<2>, !flir.stride<2>) -> !flir.layout<2>
%coord = flir.make_coord %i, %j : (index, index) -> !flir.coord<2>
```

---

## 3. Coordinate Mapping

The fundamental operation: mapping between logical coordinates and physical memory indices.

**Formula**: `Index = dot(Coord, Stride) = sum(coord_i * stride_i)`

### `crd2idx` -- Coordinate to Index

```python
# Python
idx = flir.crd2idx(coord, layout)
```

```mlir
// MLIR
%idx = flir.crd2idx %coord, %layout : (!flir.coord<2>, !flir.layout<2>) -> index
```

### `idx2crd` -- Index to Coordinate (inverse)

```python
# Python
coord = flir.idx2crd(idx, layout)
```

```mlir
// MLIR
%coord = flir.idx2crd %idx, %layout : (index, !flir.layout<2>) -> !flir.coord<2>
```

### Example

For layout `((8, 16), (1, 8))` (8x16, column-major):
- `crd2idx((3, 5), layout)` = `3*1 + 5*8` = `43`
- `idx2crd(43, layout)` = `(43 % 8, 43 / 8)` = `(3, 5)`

---

## 4. Query Operations

| Operation | Description | Example |
|---|---|---|
| `size(x)` | Product of all dimensions | `size((8, 16)) = 128` |
| `cosize(layout)` | Span of the layout mapping (max index + 1) | `cosize(((8, 16), (1, 8))) = 128` |
| `rank(x)` | Number of top-level dimensions | `rank((8, 16)) = 2` |
| `get(x, i)` | Extract i-th element | `get((8, 16), 0) = 8` |
| `get_shape(layout)` | Extract shape from layout | Returns `!flir.shape` |
| `get_stride(layout)` | Extract stride from layout | Returns `!flir.stride` |

```python
s = flir.size(layout)       # total elements
cs = flir.cosize(layout)    # codomain span
r = flir.rank(layout)       # number of modes
v = flir.get(shape, 0)      # first dimension

shape = flir.get_shape(layout)
stride = flir.get_stride(layout)
```

---

## 5. Layout Algebra

### 5.1 Composition: `composition(A, B)`

Composes two layouts: result maps through B first, then A.

**Semantics**: `result(x) = A(B(x))`

```python
composed = flir.composition(layout_a, layout_b)
```

```mlir
%composed = flir.composition %layoutA, %layoutB
    : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
```

**Use case**: Applying a permutation to a layout, or composing a tile coordinate mapping with a memory layout.

### 5.2 Complement: `complement(tiler, target_size)`

Computes the "remaining" modes not covered by the tiler, up to `target_size` elements.

```python
rest = flir.complement(tiler, target_size)
```

**Algorithm**:
1. Filter out stride-0 and size-1 modes from the tiler
2. Sort modes by stride (ascending)
3. Fold to compute rest modes

**Use case**: Internal building block for `logical_divide`. Also useful for computing the complementary iteration space when tiling.

### 5.3 Coalesce: `coalesce(layout)`

Simplifies a layout by flattening nested modes and combining adjacent modes when possible.

**Post-conditions**:
- `size(result) == size(layout)` (preserves total size)
- `result` has depth ≤ 1 (flattened)
- For all valid indices: `layout(i) == result(i)` (preserves mapping)

```python
simplified = flir.coalesce(layout)
```

---

## 6. Product Operations

Products combine two layouts to create a larger layout. All products take `(block, tiler)` and produce a result layout.

| Variant | Description |
|---|---|
| `logical_product` | Mode-wise concatenation (most basic). Scales tiler strides by block size. |
| `zipped_product` | Interleaves modes from block and tiler. |
| `tiled_product` | Creates hierarchical tiled structure. |
| `flat_product` | Produces a flattened result. |
| `raked_product` | Creates a raked (interleaved) access pattern. |
| `blocked_product` | Creates a blocked access pattern. |

```python
result = flir.logical_product(block_layout, tiler_layout)
result = flir.zipped_product(block_layout, tiler_layout)
result = flir.raked_product(block_layout, tiler_layout)
# ... etc
```

### Example

```python
# Thread layout: 4 threads along M, 32 along N
thr_layout = flir.make_ordered_layout((4, 32), order=(1, 0))

# Value layout: each thread handles 4x4 elements
val_layout = flir.make_ordered_layout((4, 4), order=(1, 0))

# Raked product: interleaved access across threads
raked = flir.raked_product(thr_layout, val_layout)
```

---

## 7. Divide Operations

Divides partition a layout by a tiler, creating a view that separates "tile" and "rest" dimensions.

| Variant | Description |
|---|---|
| `logical_divide` | Basic partitioning. Internally uses `complement`. |
| `zipped_divide` | Zipped division semantics. |
| `tiled_divide` | Hierarchical tiled division. |
| `flat_divide` | Flattened division. |

```python
result = flir.logical_divide(layout, tiler)
result = flir.zipped_divide(layout, tiler)
```

### `zipped_divide` with TensorView

`zipped_divide` has special support for `TensorView` objects, enabling block-level tiling:

```python
tensor = flir.make_tensor(A, shape=(M, N), strides=(N, 1))
tiles = flir.zipped_divide(tensor, (TILE_M, TILE_N))

# Index with block coordinates to get a tile
blk_tile = tiles[(flir.block_idx("y"), flir.block_idx("x"))]
```

---

## 8. Local Partition & Tile

### `local_partition(layout, tile, index)`

Partitions a layout for a specific thread/block index:

```python
# Partition layout by tile for thread `tid`
thr_portion = flir.local_partition(layout, tile_layout, tid)
```

### `local_tile(layout, tiler, coord)`

Extracts a tile from a layout at specific coordinates:

```python
tile = flir.local_tile(layout, tile_shape, block_coord)
```

---

## 9. Helper Functions

### `make_ordered_layout(shape, order, stride)`

Creates a layout with specified dimension ordering:

```python
# Column-major (N is the fast-changing dimension)
layout_cm = flir.make_ordered_layout((M, N), order=(1, 0))

# Row-major (M is the fast-changing dimension)
layout_rm = flir.make_ordered_layout((M, N), order=(0, 1))
```

When `order` is provided, strides are computed automatically. When `stride` is provided instead, it's used directly.

### `product_each(shape)`

Element-wise product of nested shape dimensions:

```python
# If shape is ((4, 8), (2, 16)):
# product_each returns (32, 32) -- products within each mode
result = flir.product_each(shape)
```

### `make_layout_tv(thr_layout, val_layout)`

Creates a tiler and TV (thread-value) layout from separate thread and value layouts. Used internally by `make_tiled_copy_tv`.

---

## 10. Swizzling

### `swizzle_xor16(row, col, kBlocks16)`

XOR-based swizzle for LDS bank-conflict avoidance at 16-byte granularity:

**Formula**: `result = col XOR ((row % kBlocks16) * 16)`

```python
col_swizzled = flir.swizzle_xor16(row, col, k_blocks16)
```

```mlir
%swizzled = flir.swizzle_xor16 %row, %col, %kBlocks16
    : (index, index, index) -> index
```

This matches the CK/ROCDSL-style LDS swizzle used for FP8/F16 tiles where the K dimension is permuted at 16-byte granularity while preserving intra-16B order.

**Usage in kernels**: Applied when storing tiles to LDS to avoid bank conflicts:

```python
# In preshuffle GEMM kernels:
col_bytes = col_i32 * arith.constant(4, index=True)
col_swz = flir.swizzle_xor16(row, col_bytes, k_blocks16)
coord = flir.make_coord(row, col_swz)
idx = flir.crd2idx(coord, lds_layout)
```

---

## 11. Nested / Hierarchical Layouts

FLIR supports nested layouts for representing multi-level tiling hierarchies:

```python
# Nested shape: 9 elements in first mode, (4, 8) = 32 elements in second
shape = flir.make_shape(9, (4, 8))

# This represents a 2-level hierarchy:
# - Outer: 9 x 4
# - Inner: 4 within each outer element, 8 within each inner group
```

Nested layouts are used in GEMM kernels for multi-level tiling (block → warp → thread → instruction).

---

## 12. Decision Tree

```
Which layout operation do I need?

├── Creating a layout?
│   ├── From explicit shape + stride → make_layout(shape, stride)
│   ├── With dimension ordering → make_ordered_layout(shape, order)
│   └── From existing layout components → make_layout(get_shape(l), new_stride)
│
├── Querying a layout?
│   ├── Total elements → size(layout)
│   ├── Memory span → cosize(layout)
│   ├── Number of modes → rank(layout)
│   └── Extract component → get(shape, i), get_shape(layout), get_stride(layout)
│
├── Coordinate mapping?
│   ├── Coord → memory index → crd2idx(coord, layout)
│   └── Memory index → coord → idx2crd(idx, layout)
│
├── Combining layouts?
│   ├── Sequential mapping (A then B) → composition(A, B)
│   ├── Extending to more threads → logical_product / raked_product / blocked_product
│   └── Simplifying → coalesce(layout)
│
├── Partitioning / tiling?
│   ├── Split layout into tiles → logical_divide / zipped_divide
│   ├── Get one thread's portion → local_partition(layout, tile, thread_id)
│   └── Get one block's tile → local_tile(layout, tile_shape, block_coord)
│
└── LDS bank-conflict avoidance?
    └── XOR swizzle → swizzle_xor16(row, col, k_blocks16)
```

---

## 13. Source Files

| File | Description |
|---|---|
| `flir/include/flir/FlirOps.td` | All FLIR op definitions (construction, query, algebra, divide, product, local ops) |
| `flir/include/flir/FlirTypeDefs.td` | Type definitions (`!flir.shape`, `!flir.stride`, `!flir.layout`, `!flir.coord`) |
| `flir/include/flir/FlirAttrDefs.td` | Attribute definitions (`#flir.underscore`, `#flir.dync_i32`) |
| `flir/lib/Dialect/Flir/FlirLayoutAlgebra.cpp` | Type inference for composition, logical_product, logical_divide |
| `flir/lib/Dialect/Flir/FlirOps.cpp` | Op verifiers and custom builders |
| `flydsl/src/flydsl/dialects/ext/flir.py` | Python API: all layout functions, TensorView, CopyAtom, TiledCopy |
| `tests/pyir/test_layout_algebra.py` | Layout algebra tests (composition, complement, coalesce) |
| `tests/pyir/test_product_divide.py` | Product and divide operation tests |
| `tests/pyir/test_nested_layouts.py` | Nested/hierarchical layout tests |
| `tests/pyir/test_local_ops.py` | local_partition and local_tile tests |
