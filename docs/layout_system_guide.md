# Layout Algebra Guide

> Core types, construction, coordinate mapping, algebra operations, and layout utilities in FlyDSL.

> **Important:** All `fx.*` layout operations generate MLIR IR and **must** be called inside a `@flyc.kernel` or `@flyc.jit` function body. Code snippets below show API usage patterns within that context, not standalone scripts.

## Quick Reference

| Operation | Python API | Fly Dialect Op | Description |
|---|---|---|---|
| **Construction** | `fx.make_shape(8, 16)` | `fly.make_shape` | Create shape (IntTuple) |
| | `fx.make_stride(1, 8)` | `fly.make_stride` | Create stride (IntTuple) |
| | `fx.make_layout(shape, stride)` | `fly.make_layout` | Create layout from (shape, stride) |
| | `fx.make_coord(i, j)` | `fly.make_coord` | Create coordinate |
| | `fx.make_int_tuple(elems)` | `fly.make_int_tuple` | Create generic IntTuple |
| | `fx.make_ordered_layout(shape, order)` | `fly.make_ordered_layout` | Create layout with mode ordering |
| **Mapping** | `fx.crd2idx(coord, layout)` | `fly.crd2idx` | Coordinate → linear index |
| | `fx.idx2crd(idx, layout)` | `fly.idx2crd` | Linear index → coordinate |
| **Query** | `fx.size(layout)` | `fly.size` | Total element count |
| | `fx.cosize(layout)` | `fly.cosize` | Codomain size (max index + 1) |
| | `fx.get_shape(layout)` | `fly.get_shape` | Extract shape from layout |
| | `fx.get_stride(layout)` | `fly.get_stride` | Extract stride from layout |
| | `fx.get(int_tuple, idx)` | `fly.select` + `fly.get_scalar` | Extract element at index |
| **Algebra** | `fx.composition(A, B)` | `fly.composition` | Compose: A ∘ B |
| | `fx.complement(tiler, size)` | `fly.complement` | Complement of tiler |
| | `fx.coalesce(layout)` | `fly.coalesce` | Simplify layout |
| | `fx.right_inverse(layout)` | `fly.right_inverse` | Right inverse of layout |
| **Products** | `fx.logical_product(A, B)` | `fly.logical_product` | Basic product |
| | `fx.zipped_product(A, B)` | `fly.zipped_product` | Zipped product |
| | `fx.tiled_product(A, B)` | `fly.tiled_product` | Tiled product |
| | `fx.flat_product(A, B)` | `fly.flat_product` | Flat product |
| | `fx.raked_product(A, B)` | `fly.raked_product` | Raked product |
| | `fx.block_product(A, B)` | `fly.block_product` | Blocked product |
| **Divides** | `fx.logical_divide(A, B)` | `fly.logical_divide` | Basic divide |
| | `fx.zipped_divide(A, B)` | `fly.zipped_divide` | Zipped divide |
| | `fx.tiled_divide(A, B)` | `fly.tiled_divide` | Tiled divide |
| | `fx.flat_divide(A, B)` | `fly.flat_divide` | Flat divide |
| **Structural** | `fx.select(it, indices)` | `fly.select` | Select modes by index |
| | `fx.group(it, begin, end)` | `fly.group` | Group modes into nested tuple |
| | `fx.append(base, elem)` | `fly.append` | Append mode to IntTuple |
| | `fx.prepend(base, elem)` | `fly.prepend` | Prepend mode to IntTuple |
| | `fx.zip(lhs, rhs)` | `fly.zip` | Zip two IntTuples |
| **Recast** | `fx.recast_layout(ly, old, new)` | `fly.recast_layout` | Recast layout for type width change |

---

## 1. Core Types

The Fly dialect defines several custom MLIR types for layout algebra:

| Type | MLIR Syntax | Description |
|---|---|---|
| `!fly.int_tuple` | `!fly.int_tuple<(8, 16)>` | Integer tuple — can be nested |
| `!fly.layout` | `!fly.layout<(8, 16):(1, 8)>` | Layout = (Shape, Stride) pair |
| `!fly.pointer` | `!fly.pointer<f16>` | Typed pointer |
| `!fly.memref` | `!fly.memref<...>` | Memory reference with layout |
| `!fly.swizzle` | `!fly.swizzle<...>` | Swizzle descriptor |
| `!fly.copy_atom` | `!fly.copy_atom_universal_copy<...>` | Copy atom type |
| `!fly.mma_atom` | `!fly.mma_atom_universal_fma<...>` | MMA atom type |

### IntTuple Patterns

IntTuples encode structure at the type level:

| Pattern | Meaning | Example |
|---|---|---|
| Integer literal | Static constant | `8` |
| Dynamic value | Runtime SSA value | Provided as operand |
| Nested tuple | Hierarchical mode | `(8, (4, 2))` |

---

## 2. Construction

### Python API (via `flydsl.expr`)

```python
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

# Shapes and strides (static constants auto-materialized)
shape = fx.make_shape(8, 16)              # !fly.int_tuple<(8, 16)>
stride = fx.make_stride(1, 8)             # !fly.int_tuple<(1, 8)>
layout = fx.make_layout(shape, stride)    # !fly.layout<(8, 16):(1, 8)>

# Shorthand — pass Python tuples directly
layout = fx.make_layout((8, 16), (1, 8))

# Coordinates
coord = fx.make_coord(i, j)

# Generic integer tuple
it = fx.make_int_tuple((4, 8, 2))

# Nested shapes
shape_nested = fx.make_shape(9, (4, 8))   # (9, (4, 8))

# Ordered layout — specify stride order (e.g., column-major vs row-major)
col_major = fx.make_ordered_layout((M, N), order=(0, 1))  # stride order: M-first
row_major = fx.make_ordered_layout((M, N), order=(1, 0))  # stride order: N-first

# Identity layout / tensor
identity = fx.make_identity_layout((M, N))
id_tensor = fx.make_identity_tensor((M, N))
```

---

## 3. Coordinate Mapping

The fundamental operation: mapping between logical coordinates and physical memory indices.

**Formula**: `Index = sum(coord_i * stride_i)`

### `crd2idx` — Coordinate to Index

```python
idx = fx.crd2idx(coord, layout)
```

### `idx2crd` — Index to Coordinate (inverse)

```python
coord = fx.idx2crd(idx, layout)
```

### Pure-Arith Helpers (`kernels/layout_utils.py`)

For static-stride layouts, `layout_utils` provides lightweight helpers that parse layout type strings and emit pure arith ops:

```python
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

# Parses '(4,64):(64,1)' from the type and emits arith ops
flat_idx = crd2idx([row, col], layout_value)
coords = idx2crd(flat_idx, layout_value)
dim_val = layout_get(int_tuple, 0)
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
| `cosize(layout)` | Max index + 1 (codomain size) | `cosize(((8,16),(1,8))) = 128` |
| `get_shape(layout)` | Extract shape from layout | Returns `!fly.int_tuple` |
| `get_stride(layout)` | Extract stride from layout | Returns `!fly.int_tuple` |
| `get(x, i)` | Extract i-th element | `get((8, 16), 0) = 8` |
| `get_scalar(x)` | Extract scalar from leaf IntTuple | Returns index value |
| `rank(x)` | Number of top-level modes | `rank((8, 16)) = 2` |
| `depth(x)` | Nesting depth | `depth((8, (4, 2))) = 2` |

```python
s = fx.size(layout)           # total elements (returns Int32 for static)
cs = fx.cosize(layout)        # codomain size (max index + 1)
shape = fx.get_shape(layout)
stride = fx.get_stride(layout)
v = fx.get(shape, 0)          # first dimension
r = fx.rank(shape)            # number of modes
```

---

## 5. Layout Algebra

### 5.1 Composition: `composition(A, B)`

Composes two layouts: result maps through B first, then A.

**Semantics**: `result(x) = A(B(x))`

```python
composed = fx.composition(layout_a, layout_b)
```

**Use case**: Applying a permutation or tile coordinate mapping to a memory layout.

### 5.2 Complement: `complement(tiler, target_size)`

Computes the "remaining" modes not covered by the tiler, up to `target_size` elements.

```python
rest = fx.complement(tiler, target_size)
```

**Use case**: Internal building block for `logical_divide`. Computing complementary iteration space when tiling.

### 5.3 Coalesce: `coalesce(layout)`

Simplifies a layout by flattening nested modes and combining adjacent modes when possible.

**Post-conditions**:
- `size(result) == size(layout)` (preserves total size)
- For all valid indices: `layout(i) == result(i)` (preserves mapping)

```python
simplified = fx.coalesce(layout)
```

### 5.4 Right Inverse: `right_inverse(layout)`

Computes the right inverse of a layout mapping.

```python
inv = fx.right_inverse(layout)
```

### 5.5 Recast Layout: `recast_layout(layout, old_bits, new_bits)`

Adjusts a layout for a type width change (e.g., FP16 → FP8):

```python
# Convert layout from 16-bit to 8-bit elements
recasted = fx.recast_layout(layout, old_type_bits=16, new_type_bits=8)
```

---

## 6. Product Operations

Products combine two layouts to create a larger layout. All products take `(layout, tiler)`.

| Variant | Description |
|---|---|
| `logical_product` | Mode-wise concatenation (most basic). Scales tiler strides by layout size. |
| `zipped_product` | Interleaves modes from layout and tiler. |
| `tiled_product` | Creates hierarchical tiled structure. |
| `flat_product` | Produces a flattened result. |
| `raked_product` | Creates a raked (interleaved) access pattern. |
| `block_product` | Creates a blocked access pattern. |

```python
result = fx.logical_product(layout, tiler)
result = fx.zipped_product(layout, tiler)
result = fx.raked_product(layout, tiler)
```

---

## 7. Divide Operations

Divides partition a layout by a divisor, creating a view that separates "tile" and "rest" dimensions.

| Variant | Description |
|---|---|
| `logical_divide` | Basic partitioning. Internally uses `complement`. |
| `zipped_divide` | Zipped division semantics. |
| `tiled_divide` | Hierarchical tiled division. |
| `flat_divide` | Flattened division. |

```python
result = fx.logical_divide(layout, divisor)
result = fx.zipped_divide(layout, divisor)
```

---

## 8. Structural Operations

### `select(int_tuple, indices)`

Select modes by index:

```python
selected = fx.select(int_tuple, indices=[0, 2])  # pick modes 0 and 2
```

### `group(int_tuple, begin, end)`

Group a range of modes into a nested tuple:

```python
grouped = fx.group(int_tuple, begin=1, end=3)
```

### `append(base, elem)` / `prepend(base, elem)`

Add a mode to the end/beginning:

```python
extended = fx.append(base_tuple, new_elem)
extended = fx.prepend(base_tuple, new_elem)
```

### `zip(lhs, rhs)`

Zip two IntTuples mode-wise:

```python
zipped = fx.zip(shapes_a, shapes_b)
```

### `slice(src, coord)`

Slice an IntTuple/layout at a coordinate:

```python
sliced = fx.slice(layout, coord)
```

---

## 9. MemRef / View / Copy Operations

### MemRef Operations

```python
# Allocate on-chip memory with layout
alloca = fx.memref_alloca(memref_type, layout)

# Load / store through layout
val = fx.memref_load(memref, indices)
fx.memref_store(value, memref, indices)

# Vector load / store
vec = fx.memref_load_vec(memref)
fx.memref_store_vec(vector, memref)

# Get layout from memref
ly = fx.get_layout(memref)

# Get iterator from memref
it = fx.get_iter(memref)
```

### View and Offset

```python
# Create a view from iterator + layout
view = fx.make_view(iterator, layout)

# Add offset to a pointer
ptr = fx.add_offset(ptr, offset)
```

### Copy Atoms and Tiled Copies

#### Copy Atom Types

| Type Factory | Description |
|---|---|
| `fx.UniversalCopy128b()` | Generic 128-bit copy |
| `fx.UniversalCopy64b()` | Generic 64-bit copy |
| `fx.UniversalCopy32b()` | Generic 32-bit copy |
| `fx.UniversalCopy(bits)` | Generic copy with custom bit width |
| `fx.rocdl.BufferCopy128b()` | AMD buffer-descriptor 128-bit copy |
| `fx.rocdl.BufferCopy64b()` | AMD buffer-descriptor 64-bit copy |
| `fx.rocdl.BufferCopy32b()` | AMD buffer-descriptor 32-bit copy |

#### Construction

```python
# Create copy atom (copy_op_type, elem_type)
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Create MMA atom
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

# Build thread-value layout from thread and value layouts
tiler_mn, layout_tv = fx.make_layout_tv(thr_layout, val_layout)

# Make tiled copy from copy atom + layout + tile
tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)

# Make tiled copy matched to a TiledMma's A/B/C partitioning
tiled_copy_a = fx.make_tiled_copy_A(copy_atom, tiled_mma)
tiled_copy_b = fx.make_tiled_copy_B(copy_atom, tiled_mma)
tiled_copy_c = fx.make_tiled_copy_C(copy_atom, tiled_mma)

# Make tiled MMA from MMA atom + atom layout + optional permutation
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout)
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout, permutation)
```

#### Thread Slicing and Partitioning

```python
# Get a per-thread view of a tiled copy
thr_copy = tiled_copy.get_slice(tid)   # returns ThrCopy
src_part = thr_copy.partition_S(src)   # partition source tensor
dst_part = thr_copy.partition_D(dst)   # partition destination tensor
retiled  = thr_copy.retile(tensor)     # retile tensor to match copy atom

# Get a per-thread view of a tiled MMA
thr_mma = tiled_mma.get_slice(tid)     # returns ThrMma
part_a = thr_mma.partition_A(tensor_a)
part_b = thr_mma.partition_B(tensor_b)
part_c = thr_mma.partition_C(tensor_c)

# Create register fragments
frag_a = tiled_mma.make_fragment_A(part_a)
frag_b = tiled_mma.make_fragment_B(part_b)
frag_c = tiled_mma.make_fragment_C(part_c)
```

#### Execution

```python
# Execute tiled copy
fx.copy(copy_atom, src_part, dst_part)

# Execute tiled copy with predicate mask (for boundary handling)
fx.copy(copy_atom, src_part, dst_part, pred=pred_tensor)

# Execute GEMM: D = A * B + C
fx.gemm(mma_atom, d, a, b, c)
```

#### Introspection

| Property | Class | Description |
|---|---|---|
| `copy_atom.thr_layout` | `CopyAtom` | Thread layout of copy atom |
| `copy_atom.tv_layout_src` | `CopyAtom` | Thread-value layout for source |
| `copy_atom.tv_layout_dst` | `CopyAtom` | Thread-value layout for destination |
| `mma_atom.thr_layout` | `MmaAtom` | Thread layout |
| `mma_atom.shape_mnk` | `MmaAtom` | M×N×K tile dimensions |
| `mma_atom.tv_layout_A/B/C` | `MmaAtom` | Thread-value layouts per operand |
| `tiled_copy.tiled_tv_layout_S` | `TiledCopy` | Full tiled source layout |
| `tiled_copy.tiled_tv_layout_D` | `TiledCopy` | Full tiled destination layout |
| `tiled_mma.tile_size_mnk` | `TiledMma` | Tiled MMA dimensions |
| `tiled_mma.thr_layout_vmnk` | `TiledMma` | Thread layout across V,M,N,K |
| `tiled_mma.tiled_tv_layout_A/B/C` | `TiledMma` | Full tiled layouts per operand |

---

## 10. Nested / Hierarchical Layouts

The Fly dialect supports nested layouts for representing multi-level tiling hierarchies:

```python
# Nested shape: 9 elements in first mode, (4, 8) = 32 elements in second
shape = fx.make_shape(9, (4, 8))
```

Nested layouts are used in GEMM kernels for multi-level tiling (block → warp → thread → instruction).

---

## 11. IntTuple Arithmetic

```python
# Element-wise operations on IntTuples
sum_it = fx.int_tuple_add(a, b)
diff_it = fx.int_tuple_sub(a, b)
prod_it = fx.int_tuple_mul(a, b)
quot_it = fx.int_tuple_div(a, b)

# Reduce to product
total = fx.int_tuple_product(int_tuple)

# Per-mode product (for nested tuples)
products = fx.int_tuple_product_each(int_tuple)
```

---

## 12. Printf Debugging

The Fly dialect provides a `printf` op for kernel debugging:

```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

Supports:
- `ir.Value` — dynamic values
- `int`, `float`, `bool` — auto-converted to constants
- `str`, `type` — embedded as static text
- DSL types with `__fly_values__` — auto-unwrapped

---

## 13. Decision Tree

```
Which layout operation do I need?

├── Creating a layout?
│   ├── From explicit shape + stride → make_layout(shape, stride)
│   ├── Identity layout → make_identity_layout(shape)
│   └── From existing components → make_layout(get_shape(l), new_stride)
│
├── Querying a layout?
│   ├── Total elements → size(layout)
│   ├── Extract component → get_shape(layout), get_stride(layout)
│   ├── Single mode → get(shape, i)
│   └── Number of modes → rank(layout)
│
├── Coordinate mapping?
│   ├── Coord → memory index → crd2idx(coord, layout)
│   ├── Memory index → coord → idx2crd(idx, layout)
│   └── Static-stride shortcut → layout_utils.crd2idx(crd, layout)
│
├── Combining layouts?
│   ├── Sequential mapping → composition(A, B)
│   ├── Extending threads → logical_product / raked_product / block_product
│   └── Simplifying → coalesce(layout)
│
├── Partitioning / tiling?
│   ├── Split layout → logical_divide / zipped_divide
│   └── Hierarchical tile → tiled_divide
│
├── Type width change?
│   └── recast_layout(layout, old_bits, new_bits)
│
└── Structural manipulation?
    ├── Select modes → select(it, indices)
    ├── Group modes → group(it, begin, end)
    └── Extend → append(it, elem) / prepend(it, elem)
```

---

## 14. Source Files

| File | Description |
|---|---|
| `python/flydsl/expr/primitive.py` | All layout functions: construction, query, algebra, divide, product, copy, gemm |
| `python/flydsl/expr/derived.py` | `CopyAtom`, `MmaAtom`, `TiledCopy` wrapper classes |
| `python/flydsl/expr/typing.py` | `IntTupleType`, `LayoutType`, type definitions |
| `kernels/layout_utils.py` | Pure-arith helpers: `crd2idx`, `idx2crd`, `get` for static layouts |
| `include/flydsl/Dialect/Fly/IR/FlyOps.td` | Fly dialect op definitions |
| `lib/Dialect/Fly/IR/FlyOps.cpp` | Type inference for composition, product, divide (Fly) |
| `include/flydsl/Dialect/Fly/Utils/LayoutUtils.h` | Layout algebra algorithms (composition, product, divide) |
| `tests/pyir/test_layout_algebra.py` | Layout algebra tests |
| `tests/pyir/test_product_divide.py` | Product and divide operation tests |
| `tests/pyir/test_nested_layouts.py` | Nested/hierarchical layout tests |
| `tests/pyir/test_local_ops.py` | Local partition and tile tests |
