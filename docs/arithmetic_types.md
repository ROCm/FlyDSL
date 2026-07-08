# Arithmetic Types

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
| `+`  `-`  `*` | `C` |
| `//`  `%` | `C` |
| `/` | `C` if `C` is a `Float`; if `C` is an `Integer`, `Float32` when its width is at most 32 bits, otherwise `Float64` |
| `**` | `C` |
| `<`  `<=`  `>`  `>=`  `==`  `!=` | `Boolean` (operands are compared as `C`) |
| `&`  `\|`  `^`  `<<`  `>>` | `C`; operands must be `Integer` (a `Float` operand raises `TypeError`) |

### Types narrower than 16 bits

`Float8*`, `Float6*`, and `Float4*` are not meant for direct arithmetic. Convert
them to `Float16`, `BFloat16`, or `Float32` first, or use a backend-specific
operation.

### Operand kinds

There are two operand kinds, `Numeric` (scalar) and `Vector`; a Python literal
is just a `Numeric` (it is normalized to one, as above). For every operation the
result type is the same across these kinds:

| lhs \ rhs | `Numeric` | `Vector` |
|-----------|-----------|----------|
| `Numeric` | scalar `Numeric` result | `Vector` result (the scalar is broadcast into the vector) |
| `Vector`  | `Vector` result (the scalar is broadcast into the vector) | `Vector` result (shapes also broadcast) |
