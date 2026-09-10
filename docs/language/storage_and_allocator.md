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

Storage handles implement the DSL value protocol, so they can participate in structured control
flow. Carrying a handle carries its pointer(s), not a copy of its contents. Field navigation also
works through `Align[Composite, A]`.

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
| `fx.Vector[E, Shape]` | product of `Shape` times the element byte width | element byte width |
| a composite whose non-`Constexpr` fields are all `Storable` | see *Byte layout* | see *Byte layout* |

Everything else is deliberately excluded, and asking for its size is a `TypeError`:
sub-byte numerics including `fx.Boolean` and `fx.Int4`, plus unspecialized `fx.Vector`, `fx.Pointer`, and
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

### `fx.Vector[E, Shape]`

A fixed-size value stored as a flat MLIR vector, retaining its logical shape after `peek()`.
`Shape` is a positive integer or a tuple of positive static dimensions, including nested tuples.
`E` must be a byte-sized or wider Storable numeric type. The layout is packed; it does not inherit
a target ABI's vector alignment. Use `Align` or the allocator's `alignment` argument to strengthen it.

```python
Tile = fx.Vector[fx.Float16, (2, 4)]
dsl_size_of(Tile), dsl_align_of(Tile)      # ⇒ (16, 2)
buf = allocator.allocate(fx.Align[Tile, 16])
buf.poke(fx.Vector.filled((2, 4), 1.0, fx.Float16))
value = buf.peek()                        # shape (2, 4), dtype Float16
```

`poke` checks the logical shape and dtype of a DSL vector. Existing `Vector(value, shape, dtype)`
construction and arithmetic remain available. Vector storage works with shared and register allocators.

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

### `fx.RegisterAllocator` — explicit physical register storage

`RegisterAllocator(register_class, start_offset=0)` uses the same `allocate(T)` and `Storage[T]`
interface. Each request creates independent register memory and a separate `fly.set_register`
declaration. Scalar numerics, fixed vectors, arrays, structs, unions, and custom Storable types use
their existing byte layouts and pointer access methods. Custom accesses must be expressible as
static register-memory loads and stores that the promotion pass can analyze.

```python
regs = fx.RegisterAllocator(fx.rocdl.VGPR, start_offset=65)
count = regs.allocate(fx.Int32)                         # v65
tile = regs.allocate(fx.Vector[fx.Float32, 4], alignment=16)  # v68-v71
count.poke(input_scalar)
tile.poke(input_vector)
result = count.peek() + tile.peek().reduce(fx.ReductionOp.ADD)

# Existing copy/MMA tensor APIs consume an Array's view:
acc = fx.RegisterAllocator(fx.rocdl.VGPR, 64)
fragment = acc.allocate(fx.Array[fx.Float32, 64]).peek().view(mma_layout)
```

`start_offset` indexes the ordered members of the LLVM register class. `RegisterClass(target, name)`
retains the symbolic LLVM name; `size_bits` and `member_count` query LLVM's MC register information.
The AMDGPU descriptors `rocdl.AGPR`, `rocdl.VGPR`, and `rocdl.SGPR` use `AGPR_32`, `VGPR_32`, and
`SGPR_32`. Their indices are the ISA register numbers. Class member counts are not a promise that
all those registers are available on a particular GPU or in its ABI.

Sizes and alignments stay in **bytes**. Alignment is applied to the absolute register byte offset,
and each independent allocation is padded to whole class members. For example, an Int8 allocation
uses one 32-bit AMDGPU register; two adjacent Int16 fields in a struct share a register.
`allocated_bytes` includes alignment and trailing register padding; `allocated_registers` is that
size divided by the class member size. There is no single `base_ptr` and no `free` operation.

The current code generation backend is AMDGPU, using optimized SelectionDAG. Promotion follows
static byte offsets through pointer recasts and carries storage across structured loops and branches.
When storage is read, it binds the covering 32-bit words; sub-word fields retain their position in
those words. A partial store updates the byte state, rather than forcing every historical partial
state into a separate physical register. This is SSA placement, not a promise that every source-level
`poke` emits an immediate physical write. Do not depend on uninitialized contents or padding.

Instructions producing explicitly placed values must directly accept the requested physical
registers, including their fixed inputs. LLVM's operand constraints determine compatibility.
An incompatible result fails compilation; the backend never repairs it by computing into a
temporary and copying back to fixed storage. This avoids turning loop accumulators into a backing
store with repeated class transfers. Tied read/write operands must also be supported directly.
Existing LLVM copies and constant initialization remain subject to normal code generation.

Consumers whose outputs are not explicitly placed may read through standard machine `COPY`s.
For example, a natively produced AGPR f32 value can be converted to f16 by reading it into a VGPR,
then performing LLVM's normal conversion. These copies preserve bits; numeric conversion belongs
to the original operation. Normal RA owns the read temporaries and the target expands the copies.
They do not inherit the fixed register numbers or grouping constraints. Read copies can increase
instruction count and register pressure, so inspect the final ISA for the intended access pattern.

The pass preserves LLVM's selected opcodes. It uses no instruction-name matching, encoding
substitution, hidden compiler switches, inline assembly, or LLVM source changes. Consequently,
class support depends on the instructions LLVM selects for the target, not just hardware capability:

| Current default LLVM selection | Supported explicit placement |
| --- | --- |
| gfx942/gfx950 example04 | A/B in AGPRs, C/D in VGPRs, directly used by MFMA |
| gfx942/gfx950 MFMA C/D in AGPRs | Rejected: selected MFMA requires VGPR C/D |
| gfx908 MFMA result in AGPRs | Supported when all other placement constraints hold |
| Ordinary VGPR arithmetic result in AGPRs | Rejected: no automatic write-back |
| AGPR load followed by an unplaced conversion | Supported with a read copy when needed |

`RegisterAllocator` is a layout convenience over `set_register`; both have the same backend rules.
Storable support describes byte layout and promotion, not arbitrary register-class legalization.
For a rejected class, choose a class compatible with LLVM's selected instruction or remove the
placement. In example04, all 256 MFMAs directly use the specified A/B/C ranges and the supported
configuration emits no AGPR/VGPR transfer instructions. AGPR C is deliberately rejected on
gfx942/gfx950 instead of silently inserting transfers into the mainloop.

This is not a general legalization of arbitrary class assignments. SGPR placement still requires a
compatible uniform LLVM class. Vector-to-scalar copies are rejected, including when an originally
uniform value placed in VGPR/AGPR storage is later required by a scalar instruction: the backend does
not infer a safe lane-selection operation. Instructions needing a bridge inside a bundle, at a
terminator, or in inline assembly are unsupported. Dynamic register-memory indexing,
O0/FastISel/GlobalISel, and forcing purely constant values into registers are also unsupported.
If instruction selection materializes a value only for a placement carrier while real instructions
use another value, placement is rejected instead of allowing the fixed copy to disappear later.
Incorrect class/range/alignment inputs produce Python errors where they can be checked early;
machine-level conflicts and unsupported transfers may still produce a native LLVM fatal diagnostic.
LLVM's full machine verifier runs by default before and after placement, and after standard post-RA
pseudo expansion. It checks virtual as well as physical operands. There is no instruction-specific
repair fallback when verification fails. `FLYDSL_REGISTER_DUMP_DIR` saves the MIR after placement.

The embedded ROCm pipeline runs `fly-serialize-register-kernels` before `gpu-module-to-binary`.
For modules containing placement carriers, this pass reuses the upstream ROCDL serializer and
assembles the resulting ISA with LLVM's MC parser for the target triple, chip, and features.
Parser or MC errors fail the MLIR pass before ISA is returned or an object is linked. These errors
reach Python as compilation errors; LLVM fatal errors in earlier codegen stages remain fatal.
Other GPU modules continue through the ordinary serializer. Custom pipelines should include this
pass with matching output options to obtain the same final assembly validation; invoking upstream
`gpu-module-to-binary` directly only gets the registered MIR placement checks.

Materialized explicit register ranges are reserved for the entire function against implicit allocation
and scavenging. Distinct explicit values can reuse a physical range if their live intervals do not
overlap. Allocators do not coordinate with each other, and keeping an old loaded SSA value alive
across a write can create an overlapping-placement error. Alignment gaps, unused fields, and
optimized-away values do not themselves reserve registers. Automatic lifetime-based release to the
implicit allocator remains future work.
