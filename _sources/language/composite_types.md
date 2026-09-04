# Composite Types

A composite gives one name to several DSL values. There are exactly two forms — `@fx.struct`, a
**product** in which every declared field is present, and `@fx.union`, a **storage overlay** in
which every field starts at byte offset zero.

A composite adds no capability of its own. Its whole semantics is *inheritance by field*: a
composite supports exactly the protocols all of its fields support, recursively. Everything below is
a consequence of that one rule.

A composite is also a trace-time Python type, not a value in the generated MLIR program: grouping
fields emits no aggregate operation.


## Declaring a composite

`@fx.struct` declares an ordered product; the field annotations *are* the type, and class-level
defaults do not create optional constructor arguments. Construction takes exactly one value per
field, positionally or by name, and the result is frozen.

```python
@fx.struct
class Pair:
    left: fx.Int32
    right: fx.Float32

pair = Pair(1, 2.0)             # ⇒ Pair(left=Int32(1), right=Float32(2.0))
pair.left                       # ⇒ Int32(1)
pair.replace(left=3).left       # ⇒ Int32(3); `pair` is unchanged
```

`@fx.union` declares alternative views of the same bytes. It is for scratch memory whose
interpretation changes across phases, not for a tagged sum type: it carries no discriminant, tracks
no active variant, and has **no value form** at all — a union is only ever reached through [Storable
and Allocator](storage_and_allocator.md).

```python
@fx.union
class Scratch:
    fp16: fx.Array[fx.Float16, 128]
    fp32: fx.Array[fx.Float32, 64]

Scratch(fp16=...)               # ⇒ TypeError: no value form
```

`fx.Struct[...]` and `fx.Union[...]` are the inline forms of the same two declarations:

```python
Pair = fx.Struct["left": fx.Int32, "right": fx.Float32]
fx.Struct[fx.Int32, fx.Float32] # ⇒ generated field names `_0`, `_1`
```

| Form | Kind | Meaning | Example |
|---|---|---|---|
| `⟨Type⟩(...)` | constructor | one value per field, positional or by name; a field type may coerce its value | `Pair(1, 2.0)` |
| `value.⟨field⟩` | attribute access | read that field | `pair.left` ⇒ `Int32(1)` |
| `value.replace(⟨field⟩=new)` | method | a new value with that field replaced | `pair.replace(left=3)` |


### Reserved field names

A field is read as an ordinary attribute, and Python resolves a real class member before it ever
reaches the field lookup. A field named after an existing member would therefore be permanently
shadowed by it, so those names are rejected at declaration instead of failing mysteriously at use:

| Reserved | Why the name is taken |
|---|---|
| `replace` | the only public member of a `@fx.struct` value |
| `peek`, `poke` | the public members of the `Storage[T]` view a composite type is reached through (see [Storage and Allocator](storage_and_allocator.md#fxstoraget-a-typed-address)) |
| any name starting with `_` | the implementation's own namespace |

The rule covers `@fx.union` and the inline `fx.Struct[...]` / `fx.Union[...]` forms equally — a
union is reached through the same `Storage` view. Only the names generated for anonymous inline
fields (`_0`, `_1`, …) are exempt from the underscore rule; it applies to every name you write.


## What can be a field

**Any type that is a `DslType` can be a field** — that is the only requirement the composite itself
imposes. `fx.Constexpr[T]` is the one addition: it is not a run-time value at all, but is admitted
as trace-time configuration.

| Field type | As a value field | Contributes |
|---|---|---|
| `Numeric` (`fx.Int32`, `fx.Float32`, …) | yes | one SSA value; also a host argument and a storage leaf |
| `fx.Vector` | yes | one SSA value; neither a host argument nor a storage leaf |
| `fx.Pointer` | yes | one SSA value |
| `fx.Tensor` | yes, when built from a traced tensor | one SSA value |
| `fx.Array[E, N]` | yes | one SSA value |
| another `@fx.struct` | yes | its own fields, recursively |
| a `@fx.union` type | **no** — it has no value form | storage only |
| `fx.Constexpr[T]` | yes, as a Python value | nothing at run time |

One point is worth stating outright:

- a union-typed field gives a struct a byte layout but not a value form: such a struct can be
  allocated and viewed, never constructed.


## Nesting

A field whose type is itself a composite nests structurally, to any depth, in both directions: a
struct may hold structs and unions, and a union may hold structs. Nesting is not a special case —
the nested type is just a field whose capabilities are the ones inherited by the rule above.

```python
@fx.struct
class Inner:
    x: fx.Int32
    y: fx.Int32

@fx.struct
class Outer:
    head: fx.Int32
    inner: Inner
    tail: fx.Float32


Outer(head=1, inner=Inner(2, 3), tail=4.0).inner.y   # ⇒ Int32(3)
```

Everything a composite does to its fields, it does recursively: flattening, reconstruction, cache
signatures, ABI slots, byte offsets. A nested value is flattened only when a consumer needs its
underlying IR values or storage.


## Closure over the protocols

FlyDSL's three protocols — `DslType`, `JitArgument`, `Storable` describe what a value can do at a
boundary. Composites are **closed under each of them, independently**:

> a composite satisfies protocol `P` if and only if every non-`Constexpr` field satisfies `P`; its
> implementation of `P` is the concatenation of the fields' implementations, in declaration order.

Closure is per protocol, so a type can satisfy one and not another. A struct of `fx.Int32` and
`fx.Vector` is a perfectly good `DslType` — it flattens to two SSA values and rebuilds with the
vector's shape and dtype intact — but `fx.Vector` is neither a `JitArgument` nor `Storable`, so that
struct is neither a host launch argument nor a storable type:

```text
struct { scalar: Int32, vector: Vector }
  DslType      ⇒ [i32, vector<4xf32>]            # both fields qualify
  JitArgument  ⇒ TypeError (Vector)              # one field disqualifies it
  Storable     ⇒ TypeError (Vector)              # likewise
```


## Compile-time fields

`fx.Constexpr[T]` marks a field as part of the type's logical configuration rather than its
representation. Constructing the struct specializes the *type* to that value, so the value travels
in the type and is dropped from every protocol's per-field composition.

```python
@fx.struct
class Params:
    tile: fx.Constexpr[int]
    scale: fx.Float32

params = Params(tile=32, scale=1.0)
params.tile                     # ⇒ 32 (a Python int, usable in range_constexpr)
params.tile = 64                # ⇒ FrozenInstanceError, as for any field
type(params).__name__           # ⇒ 'Params[tile=32]'
dsl_size_of(Params)             # ⇒ 4 — only `scale` has bytes

# Changing it is a change of *type*, and the type is the JIT cache key:
type(params.replace(tile=64)).__name__      # ⇒ 'Params[tile=64]' — a new specialization
type(params.replace(scale=2.0)).__name__    # ⇒ 'Params[tile=32]' — reuses the artifact
```

`T` may be `int`, `bool`, `float`, `str`, `tuple`, or `Callable` — the value must match it, and a
capture-free lambda is the only accepted callable; a value of the wrong type is a **must-signal**
`TypeError` at construction. A run-time value is therefore never a `Constexpr` value:
`Params(tile=fx.Int32(4), …)` is that same `TypeError`. The same rules hold for a `Constexpr` field
nested inside another struct.


## JIT and kernel boundaries

Where a struct may be built decides what its fields may be:

- **inside `@flyc.jit`, passed to `@flyc.kernel`** — a struct built from traced DSL values crosses
  as one source-level kernel argument and arrives as the flattened field list. This is the way to
  group tensors: build `IOPair(a, b)` from the JIT function's own `fx.Tensor` parameters.
- **inside a kernel** — `value.field` is ordinary attribute access; the fields are reconstructed
  from the flattened argument list.
- **host → `@flyc.jit`** — every non-`Constexpr` field must already be a JIT-compatible leaf, which
  by the closure rule means every field must be a `JitArgument`. Scalar fields work directly,
  including from Python literals (`Pair(3, 4.0)`).

**Implementation restriction** — a raw framework tensor is not accepted as an `fx.Tensor` field:
`IOPair(torch_a, torch_b)` is a `TypeError` at construction, and wrapping the leaves with
`flyc.from_torch_tensor(...)` / `flyc.from_dlpack(...)` does not help either, because the resulting
host adapter is not an `fx.Tensor` value. Pass those tensors as top-level JIT arguments and build
the struct inside the JIT body.

A struct is still not a device aggregate ABI object: the non-`Constexpr` fields *are* the ABI slots,
in declaration order. Use a `Tensor`-field struct when the goal is to name and transport related
kernel arguments, and a storable struct when the goal is memory placement.
