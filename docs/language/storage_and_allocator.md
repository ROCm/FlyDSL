<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Storage and Allocator

Allocating memory does not hand you a value — it hands you an *address*, and the type is what says
how to read it. C++ writes that as `T*`; FlyDSL writes it as `Storage[T]`, and an allocator is
anything that produces one.

```text
allocator.allocate(T) --> Storage[T] --> .field  (another Storage)
                                     --> .peek()  (a T value)
                                     --> .poke(v) (write a T value)
```

`Storage[T]` is a *universal* wrapper, and it has to be. `fx.Pointer` cannot play this role: an MLIR
pointer's element type must be an MLIR type, so `PointerType.get(elem_ty=SomeStruct)` is a
`TypeError` — a `@fx.struct`, a `@fx.union`, and an `fx.Array` are trace-time types
with no MLIR counterpart. `Storage[T]` therefore keeps the address in whatever pointer the allocator
produced (typically an `i8` one) and carries `T` alongside it, in Python. Field offsets, variant
overlays, and typed loads/stores are all computed from that trace-time `T`, never from the MLIR
pointer type.

The layout rules a `Storage` navigates come from the
[`Storable` protocol](dsl_protocols.md#storable). Composites acquire them by
[closure over their fields](composite_types.md#closure-over-the-protocols) — grouping is a
composite's job, addressing is this page's.

## `fx.Storage[T]`: a typed address

The correspondence with C++ is close enough to use as a lookup table:

| C++ | FlyDSL | Note |
|---|---|---|
| `T* p` | `storage: Storage[T]` | the address plus the type to read it as |
| `*p` | `storage.peek()` | materialize a value |
| `*p = v` | `storage.poke(v)` | write a value |
| `&p->field` | `storage.⟨field⟩` | ⇒ `Storage[FieldType]` at that field's byte offset |
| reinterpreting a union member | `storage.⟨variant⟩` | ⇒ `Storage[VariantType]`, at offset zero |

Three things follow from `allocate` returning an address rather than a value:

- **allocation does not initialize memory** — `peek()` delegates to `T`'s `Storable` access
  contract to obtain a value; allocation itself does not perform that access;
- **not every `T` has a value form** — a `@fx.union` never does, so it exists only as
  `Storage[Union]` and is reached one variant at a time;
- **a composite is not one SSA value** — `Storage[T]` navigates its fields by offset, which is
  exactly what a `T` value could not do.

`peek` and `poke` compose recursively, so a nested struct reads and writes each leaf at `base +
outer_offset + inner_offset`. 

`peek` and `poke` are real members of this class, and attribute lookup finds a member before it
reaches the type's fields — which is why they, along with `replace` and any `_`-prefixed name, are
[reserved field names](composite_types.md#reserved-field-names).

## What a `Storage` can point at

`T` must implement the `Storable` contract: a static size and alignment, plus typed access through
a traced pointer. This applies equally to built-in and user-defined types. Builtin support includes:

| `T` | Size | Alignment |
|---|---|---|
| `fx.Empty` | 0 bytes | 1 byte |
| `Numeric` at least one byte wide (`fx.Int32`, `fx.Float32`, `fx.Int64`, …) | its byte width | its byte width |
| specialized `fx.Vector[E, Shape]` | `E.width × numel(Shape) / 8` bytes, no trailing padding | element byte width; 1 byte for packed sub-byte elements |
| specialized `fx.Pointer[E, Space]` / `fx.Pointer[E, Space, A]` | 8 bytes for Global; 4 for Shared | 8 bytes for Global; 4 for Shared |
| `fx.Array[E, N]` / `fx.Array[E, N, A]`, with `Storable` elements | see *Array* below | `A`, defaulting to the element's natural alignment |
| a composite whose non-`Constexpr` fields are all `Storable` | see *Byte layout* | see *Byte layout* |

Builtin types without this contract include sub-byte numerics such as `fx.Boolean` and `fx.Int4`,
plus unspecialized `fx.Vector` and `fx.Pointer`, and `fx.Tensor`; asking for their storage size is a `TypeError`.
One such field is enough to make the whole composite non-storable.

### `fx.Empty`

`fx.Empty` represents storage that requires no memory and carries no runtime IR values.
Allocating it returns a pointer-free `Storage[fx.Empty]` without allocating bytes or adding
alignment padding. `peek()` returns `fx.Empty()`, and `poke(fx.Empty())` is a no-op.

### `fx.Vector[E, Shape]`

A [specialized vector](arithmetic_types.md#vector) can be allocated directly or
stored in Struct fields. Its element type must be a concrete `Numeric` other than
`Index`, and its total bit width must be a multiple of eight. Packed vectors such
as `Vector[Int4, 2]` and `Vector[Boolean, 8]` are supported.

Alignment is the element byte width rounded up, at least one byte, regardless of lane count or
logical shape. For example, `Float32x4` occupies 16 bytes with 4-byte alignment. Use `Align[T, A]`
for stronger placement alignment and wider memory accesses where supported by the target.

```python
Vec4 = fx.Vector[fx.Float32, 4]

# Inside a kernel; works with static or dynamic SharedAllocator:
storage = fx.SharedAllocator().allocate(fx.Align[Vec4, 16])
storage.poke(Vec4(1.0) + 2.0)             # accepts a plain Vector result
value = storage.peek()                   # returns Vec4
```

### `fx.Pointer[E, Space, A]`

A specialized Pointer stores an address and returns a typed Pointer from `peek`.
Specify a fixed-width `Numeric` element type `E` and `AddressSpace.Global` or
`AddressSpace.Shared`. Optional pointee address alignment `A` defaults to the
element byte width rounded up. Swizzled pointers are unsupported.

```python
P = fx.Pointer[fx.Float32, fx.AddressSpace.Global]  # alignment 4

# Inside a kernel, with a compatible global pointer ptr:
slot = fx.SharedAllocator().allocate(P)
slot.poke(ptr)
loaded = slot.peek()                     # P, with pointee alignment 4
value = loaded[0]                        # Float32
```

### `fx.Array[E, N, A]`

The fixed-size storage view supports `Storable` element types `E`, a positive `int` count `N`, and
an optional positive byte alignment `A`. `A` aligns the array base without changing element stride.

Numeric arrays occupy `max(1, E.width * N // 8)` bytes, including packed sub-byte arrays.
Their default alignment is `max(1, E.width // 8)`; an explicit `A` may be any positive integer.

For other `Storable` elements, each element occupies `dsl_size_of(E)` bytes and indexing
delegates to `E`'s storage access hooks. `A` defaults to `dsl_align_of(E)` and must be a
power of two and a multiple of that alignment.

```python
Tile = fx.Array[fx.Float32, 32, 16]
Tile.size, Tile.align                      # ⇒ (32, 16)
dsl_size_of(Tile), dsl_align_of(Tile)      # ⇒ (128, 16)

# Inside a kernel:
Vec = fx.Vector[fx.Float32, 4]
vectors = fx.SharedAllocator().allocate(fx.Array[Vec, 64]).peek()
vectors[index] = Vec(1.0) + 2.0
value = vectors[index]                   # Vec

# Pointer arrays use the same indexing API:
Pointers = fx.Array[fx.Pointer[fx.Float32, fx.AddressSpace.Global], 64]
```

For example, an **array-of-structures (AoS)** layout includes each element's trailing padding:

```python
@fx.struct
class Item:
    key: fx.Int32
    weight: fx.Float64

Items = fx.Array[Item, 128]
dsl_size_of(Item), dsl_align_of(Item)      # ⇒ (16, 8)
dsl_size_of(Items), dsl_align_of(Items)    # ⇒ (2048, 8)

# Inside a kernel; valid for static or dynamic SharedAllocator placement:
items = fx.SharedAllocator().allocate(Items).peek()
index = fx.thread_idx.x                   # caller keeps indices in [0, 128)
items[index] = Item(index, 1.0)
item = items[index]                       # an Item value
```

### `fx.Align[T, A]`

`fx.Align[T, A]` sets the storage placement alignment to `A`, a positive power of two at least as
large as `T`'s natural alignment, while delegating size and access to `T`. When used as a Struct
field annotation, construction and replacement coerce through `T`, and field access returns a `T`
value directly. The annotation retains the alignment for storage layout; no intermediate Align
value is introduced.

```python
Weight = fx.Align[fx.Float64, 16]
Item = fx.Struct["weight": Weight]
item = Item(1.0)

assert type(item.weight) is fx.Float64
assert dsl_size_of(Weight) == dsl_size_of(fx.Float64) == 8
assert dsl_align_of(Weight) == 16
assert dsl_align_of(Item) == 16
assert dsl_size_of(Item) == 16  # 8-byte weight + 8-byte trailing padding
```

This field placement behavior is analogous to the member declaration
`alignas(A) T field;` in C++. The 16-byte-aligned example above can be compared with:

```cpp
#include <cstddef>
#include <type_traits>

struct Item {
    alignas(16) double weight;
};
Item item{1.0};  // item.weight is still a double

static_assert(std::is_same_v<decltype(item.weight), double>);
static_assert(offsetof(Item, weight) == 0);
static_assert(sizeof(double) == 8 && alignof(double) == 8);
static_assert(alignof(Item) == 16);
static_assert(sizeof(Item) == 16);
```


## Byte layout

The offsets `Storage` navigates. For a product type:

1. start at byte offset zero;
2. align each field's offset to that field's alignment;
3. place the field, then continue after its size;
4. round the total size up to the largest field alignment.

For a union, every field is at offset zero, the size is the largest field size, the alignment is the
largest field alignment, and the size is rounded up to that alignment. Nested composites apply both
rules recursively, and `Constexpr` fields are skipped entirely — they have no offset.

```python
@fx.struct
class Padded:
    head: fx.Int32                   # offset 0,  4 bytes
    payload: fx.Align[fx.Int32, 16]  # offset 16, 4 bytes, alignment 16

@fx.union
class Scratch:
    fp16: fx.Array[fx.Float16, 128]  # 256 bytes, align 2, offset 0
    fp32: fx.Array[fx.Float32, 64]   # 256 bytes, align 4, offset 0


dsl_align_of(Padded)                 # ⇒ 16
dsl_size_of(Padded)                  # ⇒ 32 — 20 bytes rounded up to the 16-byte alignment

dsl_size_of(Scratch)                 # ⇒ 256
```

Because both variants of `Scratch` name the same bytes, nothing validates that what one wrote is
meaningful when the other reads it — the program must establish that itself.

## Allocators

An allocator turns a `Storable` type into a `Storage` over real memory. `fx.Arena` is the
target-neutral bump allocator: it pads each request to the type's alignment, hands back a
`Storage[T]` over `base_ptr + offset`, and tracks the running total in `allocated_bytes`. It owns no
memory of its own — `base_ptr` raises `NotImplementedError` until a subclass supplies one.

| Call | Result |
|---|---|
| `allocate(T)` | `Storage[T]`, sized and aligned by the layout rules |
| `allocate(T, alignment=A)` | the same, with the start alignment raised to `max(A, dsl_align_of(T))` for that allocation only |
| `allocate(N)` | `Storage[Array[Uint8, N]]` — `N` raw bytes; a non-positive `N` is a `ValueError` |
| `allocated_bytes` | the bump cursor: everything allocated so far, including alignment padding |

Allocating a type that is not `Storable` is a `TypeError`.

### `fx.SharedAllocator` — the shared memory allocator

The concrete subclass to read as an example. It places the bytes in the shared memory, so it can
only be created while tracing a `@flyc.kernel`, and a kernel may register only one; both violations
are `RuntimeError`s.

```python
@fx.struct
class SharedStorage:
    a: fx.Array[fx.Float32, 128, 16]
    b: fx.Array[fx.Float32, 128, 16]

# Inside a @flyc.kernel body:
smem = fx.SharedAllocator().allocate(SharedStorage).peek()
a = smem.a.view(fx.make_layout(128, 1))
b = smem.b.view(fx.make_layout(128, 1))
```

Its two placement modes differ only in where the bytes come from:

| | `static=True` (default) | `static=False` |
|---|---|---|
| Shared source | one static allocation per struct leaf | one dynamic base pointer for every allocation |
| C analogue | `__shared__` | `extern __shared__` |
| Base pointer | none — `.base_ptr` raises `RuntimeError` | the shared dynamic base |
| Union | one allocation, sized to the widest variant, shared by every variant | one region, variants at offset zero |
| `kernel.launch(smem=...)` | left unset; the compiler sizes each allocation | inferred from `allocated_bytes` when `smem=None`; an explicit `smem` must be at least that size |

Both modes expose the same field-view API, and `allocated_bytes` follows the same logical layout
in both modes. In static mode a nested struct emits one allocation per leaf, which is why it has
no single contiguous base pointer.
