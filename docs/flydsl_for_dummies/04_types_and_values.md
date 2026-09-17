# Types and Values

Chapter 1 warned that "just use `float`" does not work (§1.3): a Python `float` is
a host constant, while an `fx.Float32` is *a node in the IR*. This chapter makes
that precise. It answers the type questions a C++ engineer actually has when
reading a FlyDSL kernel: what is `fx.Float32` really, how does `fp8` get chosen
per architecture, what is `fx.Tensor` versus a register fragment, and what does
any of it *lower to* at each IR level?

The authoritative reference for scalar arithmetic — promotion rules, rounding
modes, fast-math — is `docs/language/arithmetic_types.md`; the aggregate types are
in `docs/language/composite_types.md`, the protocols behind them in
`docs/language/dsl_protocols.md`, and the allocation rules in
`docs/language/storage_and_allocator.md`. This chapter gives the **hierarchy
across IR levels** and the C++ bridges; consult those docs for the full
elementwise operator tables and field-layout rules.

## Three things a C++ engineer calls "a value"

In C++ you rarely distinguish these; in FlyDSL the distinction is the whole game:

| Kind | FlyDSL form | Lives | C++ analogue |
|------|-------------|-------|--------------|
| Host constant | Python `int`/`float` you close over | baked into IR as a literal | a `constexpr` / macro |
| Specialization param | `X: fx.Constexpr` kernel arg | baked in, part of the cache key | non-type template param `<int X>` |
| Device value | `fx.Float32`, `fx.Vector`, `fx.Tensor`, … | an SSA `ir.Value` (or a struct of them) | a runtime variable in the `__global__` body |

The first two are known at trace time (Ch. 3's compile-time regime); the third is
the run-time regime. A single quantity can shift categories: a tile size passed as
`Constexpr` is a literal in the IR, whereas the same number read from a tensor is
a device value.

> **HIP/CK-Tile → FlyDSL.** `constexpr int BM = 128;` → a closed-over Python
> constant. `template <int BM>` → `BM: fx.Constexpr`. A plain runtime `int` in your
> device body → an `fx.Int32` SSA value.

## The scalar tower (`Numeric`)

Every scalar device value is a `Numeric` subclass
(`python/flydsl/expr/numeric.py`). The class *is* both the type and its
constructor/cast: `fx.Float32(x)` builds a `Float32` (or casts `x` to one).

- **Integers** — `Int4` `Int8` `Int16` `Int32` `Int64` `Int128`, and unsigned
  `Uint8` … `Uint128`. `Index` is a 64-bit index type (MLIR `index`), used for
  addressing. `Boolean` is `i1`, the result of comparisons and predicates.
- **Floats** — `Float16` `BFloat16` `Float32` `Float64`.
- **Narrow floats** — `Float8E4M3FN` `Float8E5M2` `Float8E4M3FNUZ` `Float8E4M3`,
  `Float6E2M3FN` `Float6E3M2FN`, `Float8E8M0FNU` (the MX scale type), and
  `Float4E2M1FN`. These are storage/transport types for MMA operands and scales,
  not general arithmetic; native support is architecture-restricted.

Each carries a fixed width and maps to exactly one MLIR element type:

| FlyDSL scalar | Width | MLIR element type |
|---------------|-------|-------------------|
| `Boolean` | 1 | `i1` |
| `Int32` / `Uint32` | 32 | `i32` |
| `Index` | 64 | `index` |
| `Float16` | 16 | `f16` |
| `BFloat16` | 16 | `bf16` |
| `Float32` | 32 | `f32` |
| `Float8E4M3FN` | 8 | `f8E4M3FN` |
| `Float4E2M1FN` | 4 | `f4E2M1FN` |

Converting between these — and reaching the raw `ir.Value` underneath — is its own
small topic, covered in §4.4.

> **HIP/CK-Tile → FlyDSL.** `fx.Float16` is `ck::half_t` / `_Float16`,
> `fx.BFloat16` is `ck::bhalf_t`, `fx.Boolean` is the `bool` a predicate returns.
> The difference: in C++ the type vanishes into the instruction stream; here it is
> an MLIR type you can print (§4.7).

## From a dtype string to an element type

Kernels usually take a dtype as a string and resolve it once. The canonical helper
is `dtype_to_elem_type` (`kernels/common/kernels_common.py:36`):

```python
from kernels.common.kernels_common import dtype_to_elem_type
elem = dtype_to_elem_type("bf16")   # -> fx.BFloat16
```

FP8 is architecture-dependent, so `default_f8_type()` selects the right one for the
target: **FNUZ** (`Float8E4M3FNUZ`) on gfx94\*, **OCP e4m3fn** (`Float8E4M3FN`) on
gfx95\* / gfx12\*. Let the helper choose rather than hard-coding a variant.

> **HIP/CK-Tile → FlyDSL.** This is the `__hip_fp8_e4m3` vs `__hip_fp8_e4m3_fnuz`
> selection you already do per-arch by hand; `default_f8_type()` centralizes it.

## Converting between values

There are three distinct conversions you will reach for, and it helps to keep them
apart — they answer different questions.

**1. Change the *type* of a value: `x.to(dtype)`.** The one method you use daily.
It is a value-preserving cast that emits the right `arith` conversion (int↔float,
widen/narrow, sign handling):

```python
a = fx.Int32(5)
f = a.to(fx.Float32)          # -> Float32(5.0)   (arith.sitofp)
h = f.to(fx.Float16)          # -> Float16        (arith.truncf)
```

Float-to-float narrowing accepts an explicit IEEE rounding mode (integers do not,
and it requires a run-time value):

```python
lo = x.to(fx.Float16, rounding_mode=fx.RoundingMode.downward)
hi = x.to(fx.Float16, rounding_mode=fx.RoundingMode.upward)
```

`x.to(...)` is also overloaded for the *degenerate* targets: `x.to(int)` /
`x.to(float)` / `x.to(bool)` materialize a **compile-time** `Numeric` back to a
host Python value (and raise on a run-time value — there is nothing to read yet),
and `x.to(ir.Value)` extracts the raw MLIR value (next point).

**2. Reinterpret the *bits*: `x.bitcast(dtype)`.** Same width, no numeric
conversion — the register contents are relabeled. This is how you move between a
float and its integer bit pattern (e.g. for a fast-math trick or a bit mask):

```python
bits = fx.Float32(1.0).bitcast(fx.Int32)   # -> Int32(0x3F800000)
```

Distinguish it sharply from `to`: `to` preserves the *number*, `bitcast` preserves
the *bits*.

**3. Reach the raw `ir.Value`: `x.ir_value()`.** A `Numeric` wraps either a Python
literal or an MLIR `ir.Value` (`is_static()` tells you which). When you call a raw
MLIR builder directly, you need the unwrapped `ir.Value`; `x.ir_value()` is the
public accessor (it is exactly `x.to(ir.Value)`, materializing a constant for a
static value). For arbitrary DSL values — vectors, structs, tuples, not just
scalars — the module-level `fx.as_ir_value(v)` is the canonical "DSL → `ir.Value`"
converter, and `fx.as_dsl_value(v, exemplar)` is its inverse ("`ir.Value` → DSL",
using `exemplar` as the type template). You rarely need these unless you are
dropping down to hand-written MLIR ops; the typed methods above cover normal
kernel code.

> **Gotcha — don't reach for the raw-unwrap helpers.** You will see `_to_raw(v)` in
> the codebase (`python/flydsl/expr/utils/arith.py`): it is an *internal* helper
> (leading underscore, re-exported only for legacy call sites) that unwraps a value
> to a raw `ir.Value`. Its public siblings `constant_vector` and `index_cast` carry
> an explicit *"Deprecated: will be removed in a future release"* marker in
> `python/flydsl/expr/arith.py:45`, and `unwrap` / `index` are the same vintage. Use
> `x.ir_value()` for a `Numeric`, `fx.as_ir_value(x)` for any DSL value, and
> `x.to(fx.Index)` instead of the index helpers.

> **HIP/CK-Tile → FlyDSL.** `x.to(dtype)` is `static_cast<T>(x)` /
> `type_convert<T>(x)`; `x.bitcast(dtype)` is `bit_cast<T>(x)` /
> `__builtin_bit_cast`; `x.ir_value()` / `fx.as_ir_value(x)` is "hand me the raw
> SSA operand" — the escape hatch for talking to the MLIR builder directly, with no
> C++ analogue because in C++ there is no IR object to reach for.

## Vectors, arrays, and structs

Scalars rarely travel alone on a GPU. FlyDSL has three aggregate forms:

- **`fx.Vector`** — a register `vector<N × Ty>`, the natural shape for
  per-thread fragments and vectorized loads/stores. Subscript it to name a
  concrete shape: `fx.Vector[fx.Float32, 4]` is `vector<4xf32>` (the alias
  `fx.Float32x4` is the same type; `BFloat16x8` and friends also exist), and
  nested shapes such as `fx.Vector[fx.Float32, (2, 2)]` are allowed. Operations
  are elementwise. (`python/flydsl/expr/typing.py`.)
- **`fx.Pointer[dtype, space]`** — a typed pointer, e.g.
  `fx.Pointer[fx.Uint32, fx.AddressSpace.Shared]`. An optional third parameter
  is the alignment.
- **`fx.Array[E, N]`** — a fixed-size array with a single backing pointer; used
  as a struct field for small per-thread tables. `E` may be any *storable* type:
  a scalar, a specialized `fx.Vector[...]` / `fx.Pointer[...]`, or an
  `@fx.struct` (an array of structs, laid out AoS at element-size strides). An
  optional third parameter is the alignment — `fx.Array[fx.Float32, 64, 16]`.
  The `.view(layout)` method exists only for numeric element types.
- **`fx.Align[T, N]`** — a field wrapper with `alignas`-like semantics: it
  forces `N`-byte alignment *and* behaves as an ordinary value field of type
  `T`, so an aligned field still round-trips through IR like any other.
- **`@fx.struct`** — a composite type (`python/flydsl/expr/struct.py`). Decorate a
  class with typed fields and it implements the DSL value protocols: it can flatten
  to / rebuild from a list of `ir.Value`s (so a whole struct can be a set of
  loop-carried values or function arguments), report its size/alignment, and
  peek/poke itself to a pointer. This is the abstraction behind shared-memory
  storage layouts (Ch. 8).

```python
@fx.struct
class ReduceState:
    m: fx.Float32                 # IR-carrying field
    l: fx.Float32
    SLOTS: fx.Constexpr[int]      # zero-IR: compile-time config, specializes the type
```

A `Constexpr[int]` field carries no IR value; it specializes the struct type at
trace time, so two different `SLOTS` produce two distinct compiled variants —
exactly like a template parameter on the struct.

### The three protocols a type may implement

"Implements the DSL value protocol" is not hand-waving: the DSL defines three
structural protocols (`docs/language/dsl_protocols.md`), and which ones a type
implements decides where you are allowed to put it.

| Protocol | Question it answers | Key methods |
|----------|---------------------|-------------|
| **JitArgument** | may this cross the host→kernel boundary? | `__get_ir_types__`, `__cache_signature__`, `__c_abi_spec__` |
| **DslType** | can this become / be rebuilt from SSA values? | `__extract_to_ir_values__`, `__construct_from_ir_values__` |
| **Storable** | does this have a byte layout you can peek/poke? | `__dsl_size_of__`, `__dsl_align_of__`, `__peek_from_ptr__`, `__poke_into_ptr__` |

Scalars implement all three. A bare `fx.Vector` / `fx.Pointer` is a DslType but
not Storable — only the *specialized* `fx.Vector[fx.Float32, 4]` /
`fx.Pointer[fx.Uint32, space]` forms know their size, which is exactly why those
are the forms you may use as a struct field or an `fx.Array` element type. A
struct is recursive: it is Storable iff every field is. That rule is what the
"does not implement the Storable protocol" allocation error (§15.7) is reporting.

> **HIP/CK-Tile → FlyDSL.** `fx.Vector` is your VGPR-resident
> `array<T, N>` / `ext_vector`; `@fx.struct` is a plain C++ `struct` used for an LDS
> storage block or a bundle of loop-carried registers, and a `Constexpr[int]` field
> is a template parameter on that struct.

## Alignment: what that trailing number means

The `16` in `fx.Array[fx.Float32, 64, 16]` is the single most common thing to
mis-read in a FlyDSL storage declaration, so it is worth being exact.

**Alignment is an address constraint, not a size.** It says: *the first byte of
this object must sit at an address that is a multiple of N*. It never changes how
many bytes the object itself occupies. It can change where the *next* object
lands, because the layout engine inserts padding to satisfy the constraint.

Two queries answer everything, and they are independent:

```python
from flydsl.compiler.protocol import dsl_size_of, dsl_align_of

dsl_size_of(fx.Array[fx.Float32, 32, 16])    # 128  — 32 × 4 bytes, ALIGN irrelevant
dsl_align_of(fx.Array[fx.Float32, 32, 16])   # 16   — the address constraint
```

(These live in `flydsl.compiler.protocol`, not in `fx.*`; import them explicitly.)

### Natural alignment of the primitive types

Every Storable type has a *natural* alignment it uses when you do not ask for one:

| Type | `dsl_size_of` | `dsl_align_of` | Rule |
|------|---------------|----------------|------|
| `fx.Int8` | 1 | 1 | scalar: align = byte width |
| `fx.Float16` / `fx.BFloat16` | 2 | 2 | ″ |
| `fx.Float32` / `fx.Int32` | 4 | 4 | ″ |
| `fx.Float64` / `fx.Int64` | 8 | 8 | ″ |
| `fx.Vector[fx.Float32, 4]` | 16 | **4** | vector: align = *element* width, not total |
| `fx.Vector[fx.Float16, 8]` | 16 | **2** | ″ |
| `fx.Pointer[T, Global]` | 8 | 8 | 64-bit address |
| `fx.Pointer[T, Shared]` | 4 | 4 | 32-bit LDS address |
| `fx.Array[E, N]` | `sizeof(E) × N` | `alignof(E)` | inherits the element's |

The vector row is the trap: `fx.Vector[fx.Float32, 4]` is 16 bytes but is only
*4-byte aligned* by default. A 16-byte object at a 4-byte-aligned address cannot
be moved with a single 128-bit instruction — which is the whole reason the
explicit form exists.

### The three ways to ask for more

```python
fx.Array[fx.Float32, 64, 16]      # 1. Array's third parameter
fx.Align[fx.Float32x4, 16]        # 2. Align wrapper — works on ANY storable type
fx.Pointer[fx.Float32, fx.AddressSpace.Shared, 16]   # 3. pointee alignment on a pointer
```

They are not interchangeable:

- **`fx.Array[E, N, ALIGN]`** sets the *array's* alignment. Size stays
  `sizeof(E) × N`; the element stride is unchanged (`fx.Array[Pair, 3, 32]` is
  still 24 bytes with 8-byte elements, just placed on a 32-byte boundary).
- **`fx.Align[T, N]`** wraps any storable `T` and raises its alignment to `N`.
  Size is unchanged. `N` must be a power of two and **must not be smaller than
  `T`'s natural alignment** — FlyDSL raises `ValueError` rather than silently
  ignoring the request the way C++ `alignas` does.
- **The alignment on `fx.Pointer[T, space, N]`** is a different animal: it is a
  promise about the addresses this pointer will hold, carried in the MLIR pointer
  type and attached to every `llvm.load`/`llvm.store` emitted through it. It is
  not the storage alignment of the pointer *value* (which is always 8 or 4 bytes,
  set by the address space).

### Struct layout follows the C rules exactly

Fields are laid out in declaration order; each field's offset is rounded up to
that field's alignment; the struct's alignment is the maximum over its fields;
and the struct's total size is rounded up to its own alignment (trailing padding),
so that an array of the struct keeps every element aligned. `Constexpr` fields
occupy no storage at all.

```python
@fx.struct
class Pair:                       # offsets {left: 0, right: 4}
    left: fx.Int32                #   size 8, align 4
    right: fx.Float32

@fx.struct
class Padded:                     # offsets {head: 0, payload: 16}
    head: fx.Int32                #   12 bytes of padding after head,
    payload: fx.Align[fx.Int32, 16]   #   then 12 more at the end:
                                  #   size 32, align 16
```

`Padded` is the instructive one: 4 bytes of `head`, a 12-byte gap so `payload`
lands on 16, 4 bytes of `payload` — 20 bytes of content, rounded up to **32**
because the struct's own alignment is now 16. Asking for alignment is not free;
you pay in LDS bytes.

Two array fields, with and without an explicit alignment:

```python
@fx.struct
class Tiles:                           # offsets {a: 0, b: 128}
    a: fx.Array[fx.Float32, 32]        #   size 256, align 4
    b: fx.Array[fx.Float32, 32]

@fx.struct
class Shared:                          # offsets {a: 0, b: 512}
    a: fx.Array[fx.Float32, 128, 16]   #   size 1024, align 16
    b: fx.Array[fx.Float32, 128, 16]
```

Here the explicit `16` costs nothing — 512 is already a multiple of 16 — which is
the common case, and the reason production kernels specify it by reflex.

### Why you actually care

Alignment is not bookkeeping; it decides which *instruction* the backend can use.
It flows end to end:

```
type declaration  fx.Array[fx.Float32, 128, 16]
   │  SharedAllocator places the object on a 16-byte boundary (§8.6)
Pointer.alignment = 16
   │  attached to every llvm.load / llvm.store through that pointer
LLVM              llvm.load ... {alignment = 16}
   │  AMDGPU instruction selection
ISA               ds_read_b128     ← needs a 16-byte-aligned address
```

Drop the `16` and the same source compiles to four `ds_read_b32`s instead of one
`ds_read_b128`: same bytes, four times the LDS instruction count. The repository's
own test asserts exactly this — allocating `fx.Align[fx.Float32x4, 16]` and then
grepping the generated ISA for `ds_read_b128`
(`tests/language/test_storage_and_allocator.py`, `test_explicit_alignment_enables_ds_read_b128`).

That is why essentially every LDS array in `kernels/` is declared with `16`: it is
the minimum alignment for the 128-bit transactions (`ds_read_b128`,
`ds_write_b128`, `buffer_load_dwordx4`) that the whole copy layer is built around.

```python
# kernels/gemm/rdna_f16_gemm.py:108 — double-buffered A/B staging, all 128-bit
@fx.struct
class SharedStorage:
    a0: fx.Array[elem_cls, LDS_A_ELEMS, 16]
    b0: fx.Array[elem_cls, LDS_B_ELEMS, 16]
    a1: fx.Array[elem_cls, LDS_A_ELEMS, 16]
    b1: fx.Array[elem_cls, LDS_B_ELEMS, 16]
```

Larger values appear when a *swizzle* needs them. Flash attention's bias tile
(`kernels/attention/flash_attn_gfx950.py:162`) asks for `128`: that tile is read
through an XOR swizzle (§6.6), and a swizzle is a permutation of the low address
bits relative to the tile base — so the base must be aligned to the swizzle's span
or the whole conflict-free pattern shifts:

```python
@fx.struct
class SharedStorage:
    kv:   fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]    # 128-bit copies
    bt:   fx.Array[fx.Int32, traits.PAGED_BT_LDS_SIZE, 16]
    bias: fx.Array[_lds_elem_dtype, _BIAS_LDS_ELEMS, 128]            # swizzle window
```

> **HIP/CK-Tile → FlyDSL.** `fx.Align[T, N]` is `alignas(N) T` and the `ALIGN` on
> `fx.Array` is `alignas(N)` on the array member; the struct layout rules are the
> ones your C++ compiler already applies to a `__shared__` struct. The difference
> is that here the numbers are queryable at trace time (`dsl_size_of` /
> `dsl_align_of`) instead of needing a `static_assert(sizeof(...) == ...)`, and
> under-aligning is a `ValueError` at trace time instead of a silently slower
> kernel.

## Tensors: the layout-carrying type

`fx.Tensor` is the principal type a kernel operates on. It is far more than a
pointer: it bundles an **element type**, an **address space**, an **alignment**,
and a **layout** (Ch. 6), lowering to `!fly.memref<elem, addrspace, layout, align>`.
Its address space is what distinguishes the memory tiers you already reason about:

| CK-Tile object you know | FlyDSL tensor | Address space |
|-------------------------|---------------|---------------|
| global tile window (HBM) | `fx.Tensor` (kernel arg) | Global |
| buffer-resource load (CDNA OOB-checked) | `fx.rocdl.make_buffer_tensor(t)` | BufferDesc |
| LDS staging tile | `fx.Tensor` in shared space (via `SharedAllocator`) | Shared |
| VGPR register fragment | `fx.Tensor` in register space (`make_fragment_*`) | Register |

The same `fx.Tensor` type spans all four tiers; only the address space (and layout)
changes. `make_buffer_tensor` wraps a Global pointer into a CDNA hardware buffer
descriptor so that copies get hardware bounds-checking (Ch. 8).

> **HIP/CK-Tile → FlyDSL.** A `fx.Tensor` is a CK naive tensor descriptor plus its
> address space rolled into one type: `Shape:Stride` is the `layout`, and Global /
> Shared / Register / BufferDesc are the memory spaces you switch between with
> `make_tile_window`, `__shared__`, VGPR fragments, and `buffer_load` in C++.

## One value, four IR levels

The payoff of tracking types explicitly: a single element type threads through the
whole pipeline (Ch. 2) as a *real, inspectable* MLIR type at each level. Take an
`f16` GEMM operand:

```
Python      fx.Float16
   │  .ir_type
MLIR elem   f16
   │  embedded in a tensor / atom
Fly         !fly.memref<f16, global, layout(...), align<16>>
            !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x32, (f16,f16)->f32>>
   │  convert-fly-to-rocdl  (Stage A, §2.4)
ROCDL       rocdl.mfma.f32.16x16x32.f16(...)
   │  convert-vector/arith-to-llvm  (Stage B)
LLVM        vector<4xf16>  (operand VGPRs)  /  vector<4xf32>  (accumulator VGPRs)
```

The function-signature and conversion passes (`fly-rewrite-func-signature`,
`convert-fly-to-rocdl`) are where these types are rewritten from Fly-level down to
ROCDL/LLVM. Because every level has a textual form, you can dump and read the type
at each stage with `FLYDSL_DUMP_IR=1` (§2.8) — the fastest way to confirm, say,
that your `fp8` operands really became `f8E4M3FN` and your accumulator is `f32`.

> **HIP/CK-Tile → FlyDSL.** In C++, once templates instantiate, the element type is
> gone — it lives only as register classes and intrinsic names in the ISA. In
> FlyDSL the type is a first-class MLIR type at four inspectable levels, which is
> what lets you *see* an operand/accumulator type mismatch instead of debugging it
> as wrong numerics.

## Constexpr versus runtime, one more time

The type of a value and its *availability* are independent. An `fx.Int32` can be a
literal folded at trace time or an SSA value computed on the device; a tile
dimension can arrive as a `Constexpr` (baked in, part of the compilation cache key)
or as a runtime argument. This is the same trace-time-vs-run-time axis from Ch. 3:
a bound that is `Constexpr` can drive a `range_constexpr` unroll (§3.2), while a
runtime bound forces an `scf.for`. Keeping "what type" and "known when" separate in
your head prevents most FlyDSL surprises.

With values and control flow in hand, Chapter 5 shows how to work with an
`fx.Tensor` directly — reading its shape, indexing individual elements, and
selecting sub-tiles. Chapter 6 then covers the layout algebra that describes how
those values are arranged in memory and across threads.
