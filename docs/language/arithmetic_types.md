# Arithmetic Types

The arithmetic layer is the set of scalar (`Numeric`) and SIMD (`Vector`) values
a kernel computes with: the types themselves, their public operations.

## The type tower

Every arithmetic value is either a scalar `Numeric` or a `Vector` of a `Numeric`
element type. A type class is also its constructor: `Float32(x)` builds (or
casts to) a `Float32`.

### Numeric scalars

`Numeric` is the base of every scalar that wraps an `ir.Value`.

- **Integers** — `Int4` `Int8` `Int16` `Int32` `Int64` `Int128`, `Uint8`
  `Uint16` `Uint32` `Uint64` `Uint128`. Signedness and width are fixed per type.
- **Floats** — `Float16` `BFloat16` `Float32` `Float64`.
- **Narrow floats** — `Float8E5M2` `Float8E4M3FN` `Float8E4M3FNUZ`
  `Float8E4M3B11FNUZ` `Float8E4M3`, `Float6E2M3FN` `Float6E3M2FN`,
  `Float8E8M0FNU`, `Float4E2M1FN`. These are storage/transport types, not meant
  for direct arithmetic, and native support is architecture-restricted.
- **`Boolean`** — the `i1` type; the result of comparisons and predicates.

### Public methods

Common to `Numeric` and `Vector`, elementwise for `Vector`:

| Method | Meaning | Example |
|--------|---------|---------|
| `Type(x)` | construct or cast — the type class is its own constructor. A `Vector` alias type (`Float32x4`, …) broadcasts a scalar across every lane | `Int32(5)`, `Float32(thread_idx.x)`; `Float32x4(1.0)` → all four lanes `1.0` |
| Arithmetic — `+` `-` `*` `/` `//` `%` `**`, unary `+x` `-x` `abs(x)`, `divmod(x, y)` | result type follows *Type interoperability*; elementwise with scalar broadcast for `Vector` | `Int32(3) + Int32(4)` → `Int32(7)`; `vec * 2.0` |
| Bitwise/shift — `&` `\|` `^` `<<` `>>`, unary `~x` | integer-only | `Int32(6) & Int32(3)` → `Int32(2)` |
| Comparison — `<` `<=` `>` `>=` `==` `!=` | result is `Boolean` | `a < b` |
| `x.bitcast(dtype)` | reinterpret the bits: `Numeric` equal width; `Vector` equal *total* width, recomputing the lane count | `Float32(1.0).bitcast(Int32)`; `Float32x4(0.0).bitcast(Int8)` → `Int8x16` |
| `x.dtype` | the scalar type (`Numeric`) / element type (`Vector`) | `Int32(5).dtype` / `Int32x4(0).dtype` → `Int32` |
| `x.ir_value()` | the underlying `ir.Value`, materializing a constant for a compile-time value | `Int32(5).ir_value()` |
| `cond.select(true_value, false_value)` | ternary select; a non-`Boolean` `cond` is converted by truthiness (nonzero) | `(a < b).select(a, b)` → min |
| `x.to(dtype)` | value-preserving conversion to another type | `Int32(5).to(Float32)` → `Float32(5.0)` |

`Numeric`-only methods:

| Method | Meaning | Example |
|--------|---------|---------|
| `x.is_static()` | whether the value is compile-time (Python) rather than run-time (`ir.Value`) | `Int32(5).is_static()` → `True` |
| `Numeric.width` / `Numeric.log_width` | the type's bit width, and `ceil(log2(width))` | `Int32.width` → `32`; `Int32.log_width` → `5` |
| `as_numeric` / `Numeric.from_python_value(value)` | build a `Numeric` from a Python value | `as_numeric(5)` → `Int32(5)` |
| `Numeric.from_ir_type(ir_type)` | the `Numeric` type for an MLIR type | `Numeric.from_ir_type(T.f32())` → `Float32` |

### Vector

`Vector` is a fixed-length sequence of `N` elements of a single `Numeric`
element type. It has value semantics and inherits the scalar operators, applied
elementwise; a scalar operand is auto-broadcast across the lanes.

- **Type aliases** — `Float32x4`, `BFloat16x8`, `Int32x4`, … name a
  `dtype`×`N` vector type directly (`<dtype>x<N>`).

## Compile-time and run-time values

A `Numeric` is *polymorphic in the value it holds* — the type is the same either
way, and `is_static()` reports which:

- a **compile-time value** — a Python `int` / `float` / `bool`, known while
  tracing; or
- a **run-time value** — an `ir.Value` from an MLIR op, known only at execution.

A `Vector` is always run-time — it is backed by an MLIR vector value, so it has
no compile-time (folded) form.

### Arithmetic preserves the compile-time property

If every operand is compile-time, the result is compile-time — the host folds it
and emits no MLIR (`Int32(3) + Int32(4)` ⇒ `Int32` holding `7`). As soon as one
operand is run-time, the result is run-time and an MLIR op is emitted. This
holds uniformly across arithmetic, comparison, bitwise, and shift operators.

### Python literals

A bare literal stays plain Python while it only meets other Python values
(`2 + 3` is ordinary Python). On contact with a `Numeric` it takes a DSL type
*by value* (see *Operand normalization*) as a compile-time value; on contact
with a `Vector` it broadcasts to the lanes and the result is run-time. After
this the interoperability rules apply. An explicit `Int32(5)` is likewise
compile-time until combined with a run-time value.

### Using a compile-time value as Python

Because it holds a real Python value, a compile-time `Numeric` works wherever
Python expects one — `int(x)`, `bool(x)`, indexing, a Python `if` — so a DSL
constant can still drive host-side control flow. A run-time `Numeric` raises if
forced to a Python value.

## Type interoperability

How a binary operation between two DSL numeric values determines the type its
operands are converted to (the *common type*) and the type it produces (the
*result type*). The rules are the same whether operands are scalar (`Numeric`),
`Vector`, or a mix of the two; `Vector ⊗ Vector` additionally broadcasts shapes,
which is independent of type and covered in the layout guides.

### Operand normalization

Before the rules below apply, operands are normalized:

- **Python literals** take a DSL type by value: an `int` becomes `Int32`, or
  `Int64` when it falls outside the `Int32` range; a `float` becomes `Float32`;
  a `bool` becomes `Boolean`.
- **`Boolean`** depends on the operation:
  - In arithmetic (`+ - * / // %`) it is converted to `Int32` and then follows
    the `Int32` rules.
  - In comparisons it is compared directly and the result is `Boolean`.
  - In bitwise (`& | ^`) and shift (`<< >>`) it stays `Boolean` and the result
    is `Boolean`.

### Common type

For two numeric operands (after normalization above; `Boolean` in arithmetic is
already `Int32` here), the common type is:

| lhs \ rhs | Int8 | Int16 | Int32 | Int64 | Uint32 | Float16 | BFloat16 | Float32 | Float64 |
|-----------|------|-------|-------|-------|--------|---------|----------|---------|---------|
| Int8      | Int8 | Int16 | Int32 | Int64 | Uint32 | Float16 | BFloat16 | Float32 | Float64 |
| Int16     | Int16 | Int16 | Int32 | Int64 | Uint32 | Float16 | BFloat16 | Float32 | Float64 |
| Int32     | Int32 | Int32 | Int32 | Int64 | Uint32 | Float32 | Float32  | Float32 | Float64 |
| Int64     | Int64 | Int64 | Int64 | Int64 | Int64  | Float64 | Float64  | Float64 | Float64 |
| Uint32    | Uint32 | Uint32 | Uint32 | Int64 | Uint32 | Float32 | Float32  | Float32 | Float64 |
| Float16   | Float16 | Float16 | Float32 | Float64 | Float32 | Float16 | Float32 | Float32 | Float64 |
| BFloat16  | BFloat16 | BFloat16 | Float32 | Float64 | Float32 | Float32 | BFloat16 | Float32 | Float64 |
| Float32   | Float32 | Float32 | Float32 | Float64 | Float32 | Float32 | Float32 | Float32 | Float64 |
| Float64   | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 |

The table follows these rules (other integer widths obey the same integer
rules):

- **Same type** → itself.
- **Two integers, same signedness** → the wider one (`Int8 + Int8` stays `Int8`;
  there is no promotion to a machine `int`).
- **Two integers, mixed signedness** → the unsigned type when it is at least as
  wide as the signed one, otherwise the signed type. So `Int32 + Uint32` is
  `Uint32`, and `Int64 + Uint32` is `Int64`.
- **One float, one integer** → the float, widened to cover the integer's width:
  `Float16 + Int32` is `Float32`, `Float32 + Int64` is `Float64`, and
  `Float16 + Int8` is `Float16`.
- **Two floats** → the wider one; at equal width the higher-precision one
  (`Float64 > Float32 > Float16`/`BFloat16`). `Float16` and `BFloat16` are
  equal width and neither converts to the other without loss, so they combine to
  `Float32`.

### Result type

Given the common type `C` from the table above:

| Operation | Result type |
|-----------|-------------|
| `+`  `-`  `*` `//`  `%` `**` | `C` |
| `/` | `C` if `C` is a `Float`; if `C` is an `Integer`, `Float32` when its width is at most 32 bits, otherwise `Float64` |
| `<`  `<=`  `>`  `>=`  `==`  `!=` | `Boolean` (operands are compared as `C`) |
| `&`  `\|`  `^`  `<<`  `>>` | `C`; operands must be `Integer` (a `Float` operand raises `TypeError`) |
