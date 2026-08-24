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

The layout rules a `Storage` navigates come from the `Storable` protocol. Composites acquire them by
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

- **the memory has no contents yet** — `peek()` is a load you ask for, not something allocation did
  for you;
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

`T` must be `Storable`: able to state a static size and alignment, and to be read from (and usually
written to) a traced pointer.

| `T` | Size | Alignment |
|---|---|---|
| `Numeric` at least one byte wide (`fx.Int32`, `fx.Float32`, `fx.Int64`, …) | its byte width | its byte width |
| `fx.Array[E, N]` / `fx.Array[E, N, A]` | `N` elements of `E` | `A`, defaulting to the element byte size |
| a composite whose non-`Constexpr` fields are all `Storable` | see *Byte layout* | see *Byte layout* |

Everything else is deliberately excluded, and asking for its size is a `TypeError`:
sub-byte numerics including `fx.Boolean` and `fx.Int4`, plus `fx.Vector`, `fx.Pointer`, and
`fx.Tensor`. One such field is enough to make the whole composite non-storable.

### `fx.Array[E, N, A]`

The fixed-size leaf: a `Numeric` subclass `E`, a positive `int` count `N`, and an optional positive
byte alignment `A`. Array types are cached, so the same parameters yield the same class. After
`peek` it behaves as a typed pointer view supporting indexing and `.view(layout)`.

```python
Tile = fx.Array[fx.Float32, 32, 16]
Tile.size, Tile.align                      # ⇒ (32, 16)
dsl_size_of(Tile), dsl_align_of(Tile)      # ⇒ (128, 16)
```

### `fx.Align[T, A]`

A *placement modifier*, not a composite form: it delegates size and access to `T` and overrides only
the alignment.

```python
Aligned = fx.Align[fx.Int32, 16]
dsl_size_of(Aligned), dsl_align_of(Aligned)   # ⇒ (4, 16)
```

`A` must be a positive power of two and at least `T`'s natural alignment; violations are
`ValueError`s, and a non-`int` `A` or a missing second parameter is a
`TypeError`.

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

In both modes the field-view API and `allocated_bytes` follow the same logical layout, so switching
modes does not change the addressing a kernel writes. In static mode a nested struct emits one
allocation per leaf, which is why it has no single contiguous base pointer.
