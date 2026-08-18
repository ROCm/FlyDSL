# Working with Tensors

Chapter 4 established that `fx.Tensor` is a layout-carrying typed pointer — the
principal object a kernel operates on. This chapter is about *using* one: how to
read its shape, stride, and layout; how to access individual elements; how to select
a sub-tile with `None`-indexing; and how to construct views that reinterpret the same
memory through a different layout. The layout algebra that composes and divides these
objects is Chapter 6; this chapter focuses on the single-tensor operations every
kernel uses first.

## What a kernel receives

Every `fx.Tensor` argument to a `@flyc.kernel` carries three things baked into its
MLIR type:

- **Element type** — the `Numeric` dtype (`fx.Float32`, `fx.BFloat16`, …)
- **Layout** — a `(shape, stride)` pair describing how multi-dimensional coordinates
  map to linear memory
- **Address space** — Global (HBM), Shared (LDS), or Register (VGPRs)

All three are available at trace time as Python-level properties:

```python
@flyc.kernel
def inspect(A: fx.Tensor):
    M, N   = A.shape.unpack()        # (IntTuple, IntTuple) or plain Python ints if static
    elem   = A.element_type          # e.g. fx.Float32
    layout = A.layout                # Layout object (shape:stride)
```

> **HIP/CK-Tile → FlyDSL.** `A.shape.unpack()` is reading the tensor descriptor's
> extent fields — the same dimensions you pass to `make_naive_tensor_descriptor` in
> CK-Tile. The difference: in FlyDSL they are first-class layout objects you can pass
> to divide and product ops, not just integers in a template parameter.

## Reading shape and stride

`shape` and `stride` are both `IntTuple` values. Call `.unpack()` to destructure them
into Python scalars (for static dims) or DSL `Int32` values (for dynamic dims):

```python
M, N = A.shape.unpack()          # 2-D tensor: two values
K,   = A.shape.unpack()          # 1-D tensor: one value (trailing comma to unpack)
```

Index a specific mode with `shape[i]`:

```python
rows = A.shape[0]                 # first dim  (IntTuple element)
cols = A.shape[1]                 # second dim
row_stride = A.stride[0]         # stride along dim 0
```

For a statically-known single-element tuple (e.g. a rank-1 shape), `fx.get_scalar`
unwraps it to a plain scalar:

```python
n = fx.get_scalar(A.shape[0])   # Int32 scalar, usable in arithmetic
```

## Accessing individual elements: `T[coord]`

Index a `Tensor` with a coordinate to load or store a **single element**:

```python
# load
x = A[tid]                        # 1-D: coord is a scalar
x = A[r, c]                       # 2-D: coord is (row, col)

# store
C[tid] = x
C[r, c] = val
```

Under the hood `A[r, c]` calls `fly.memref_load(A, (r, c))`, which computes the
linear address as `crd2idx((r, c), layout(A))` and emits a load through the
tensor's pointer. For a global tensor this becomes `global_load`; for an LDS tensor
`ds_read`; for a buffer tensor `buffer_load` — the address space in the type chooses
the instruction automatically.

```python
# classic vectorAdd style: thread t handles element t of a 1-D tensor
@flyc.kernel
def add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    t = fx.thread_idx.x + fx.block_idx.x * fx.Int32(BLOCK)
    C[t] = A[t] + B[t]            # two loads, one store, one arith.addf
```

> **HIP/CK-Tile → FlyDSL.** `A[t]` is the device pointer dereference `A_ptr[t]` in
> HIP. The offset arithmetic (`t * sizeof(float)`) is computed from the layout's
> stride, not written by hand.

## Slicing: selecting a sub-tensor with `None`

A `None` in the index means "keep all of this dimension." Fixed integers select a
single slice and *drop* that dimension. The result is a lower-rank `Tensor` pointing
into the same memory:

```python
# A is (M, N, K): a 3-mode tensor (e.g. after zipped_divide)
col_block = A[None, bid, None]    # fix dim 1 to bid -> shape (M, K)
row       = A[r, None, None]      # fix dim 0 to r   -> shape (N, K)
```

This is the `fx.slice` function under the hood (`primitive.py:777`). The same
`None`-indexing applies to register tensors, LDS tensors, and partitioned fragments —
which is why tiling expressions like `bA[None, k]` ("the k-th K-tile of block A")
appear throughout the kernel examples.

```python
# common tiling idiom: divide the global tensor into blocks, slice this block
bA = fx.zipped_divide(A, (BM, BK))   # shape: ((BM, BK), (nM, nK))
tile = bA[None, bid_m, None, bid_k]  # shape: (BM, BK) — this block's A tile
```

The shape-and-stride of the sliced result follows CuTe's `slice` algebra; see
Chapter 6 for the full divide/product rules.

> **HIP/CK-Tile → FlyDSL.** `A[None, bid, None]` is selecting the `bid`-th tile
> window — equivalent to advancing a CK-Tile tile window with `Move_Window` or
> computing a pointer to the block's starting address. Here the layout carries the
> offset automatically; no manual `bid * BN * sizeof(T)` arithmetic.

## `make_view`: reinterpret the same memory with a different layout

Sometimes you need to look at the same bytes through a different shape or stride —
for example, to treat a 1-D LDS buffer as a 2-D matrix. `fx.make_view(ptr, layout)`
attaches a new layout to an existing pointer without moving data:

```python
smem = fx.get_dyn_shared(fx.Float32)              # raw LDS pointer
smem_2d = fx.make_view(smem, fx.make_layout((BM, BN), (BN, 1)))  # view as (BM x BN)
smem_2d[r, c] = val                               # write at logical (r, c)
```

`make_view` is also how you attach a layout to an arbitrary pointer (e.g. one
returned by `get_dyn_shared` or `get_iter`). The result is an `fx.Tensor` with that
layout; all the indexing, slicing, and copy operations then work as usual.
When the layout is a composed layout wrapping a swizzle, `make_view` is how you
build a bank-conflict-free LDS tensor — Chapter 6 covers this in full.

## `get_iter` and `get_layout`: decomposing a Tensor

`fx.get_iter(T)` strips the layout and returns a raw typed `Pointer` to the first
element — the address-space-aware pointer used by `ptr_load`/`ptr_store` (Chapter 10).
`fx.get_layout(T)` returns the `Layout` object standalone:

```python
ptr    = fx.get_iter(A)           # Pointer (same address space as A)
layout = fx.get_layout(A)         # Layout value: (shape:stride)
stride_k = A.stride[2]           # stride of the K dimension (after zipped_divide)
```

These are useful when you need the raw address (for `ptr_load`, buffer_ops, or
manual indexing) but want to keep the layout separate for size computations.

## Properties at a glance

| Property / method | Returns | Notes |
|---|---|---|
| `A.element_type` | `Numeric` class (e.g. `fx.Float32`) | dtype of one element |
| `A.dtype` | same | alias |
| `A.layout` | `Layout` | shape:stride pair |
| `A.shape` | `IntTuple` | extents; `.unpack()` to destructure |
| `A.stride` | `IntTuple` | strides; index with `[i]` |
| `A.address_space` | enum | Global / Shared / Register / BufferDesc |
| `A[coord]` | loaded value or sub-tensor | scalar coord → load; `None` coord → slice |
| `A[coord] = v` | — | scalar coord → store; `None` coord → `slice(A,…).store(v)` |
| `A.load()` | `Vector` | load the whole tensor as a flat SSA vector (register tensors) |
| `A.store(vec)` | — | store a flat SSA vector back (register tensors) |
| `A.fill(val)` | — | store a constant splat (register tensors) |
| `fx.get_iter(A)` | `Pointer` | raw pointer; use with `ptr_load`/`ptr_store` |
| `fx.get_layout(A)` | `Layout` | standalone layout object |
| `fx.make_view(ptr, layout)` | `Tensor` | attach a layout to a pointer |
| `fx.get_scalar(A.shape[i])` | scalar | unwrap a static single-element dim |

## Converting a tensor to a Vector

A `Tensor` and a `Vector` are two different views of the same registers:
the tensor carries a layout (shape and stride) while the vector is a flat
`vector<N×T>` SSA value. You cross between them with three operations.

### `tensor.load()` — the whole tensor as a flat Vector

`T.load()` reads every element of a register-space tensor into a single flat
`Vector`. For a 4-element `fx.Float32` register tensor this gives a
`Vector` of shape `(4,)` and dtype `fx.Float32`:

```python
frag = fx.make_rmem_tensor(4, fx.Float32)
frag.fill(0)                          # zero all four slots
# ... MFMA writes into frag ...
v = frag.load()                       # Vector<4×f32>  (the four accumulator values)
```

The result is a plain SSA vector you can do arithmetic on, pass to the raw
`rocdl.mfma_*` intrinsic, carry as a `scf.for` iter-arg, or store back:

```python
v2 = v * fx.Float32(2.0)             # elementwise scale (Vector op)
frag.store(v2)                        # write back into the four register slots
```

`T.store(vec)` is the inverse: it takes a flat `Vector` and writes its elements
into the tensor's register slots. The vector and tensor must have the same total
element count and element type.

### `tensor[i]` — a single element as a `Numeric`

Index a register tensor with a scalar integer to get (or set) one element:

```python
v = frag[0]          # first element -> Numeric (fx.Float32)
v = frag[3]          # fourth element

frag[2] = fx.Float32(1.0)   # write to the third slot
```

This emits a `vector.extractelement` / `vector.insertelement` (after the
register-promotion pass). Useful in epilogue code where you need to process
each accumulator value individually:

```python
# scale each accumulator lane by a per-lane factor
for i in fx.range_constexpr(4):
    frag[i] = frag[i] * scale[i]
```

### `fx.Vector.from_elements([e0, e1, …])` — build a Vector from individual values

When you have separate `Numeric` values (e.g. from a list of `ptr_load` results)
and need to pack them into a Vector to pass to a raw intrinsic:

```python
a_el = []
for i in fx.range_constexpr(4):
    a_el.append(fx.ptr_load(aptr + (m * fx.Int32(K) + k_base + fx.Int32(i))))
a_vec = fx.Vector.from_elements(a_el, dtype=fx.BFloat16)  # vector<4×bf16>
```

The dtype is inferred from the first element if omitted. This is the low-level
operand-packing step shown in the MFMA worked example (Chapter 11).

### `fx.Vector(raw_ir_value, shape, dtype)` — wrap an existing SSA vector

When a raw MLIR op (like a `rocdl.mfma_*` call) returns a plain `ir.Value` of
type `vector<4×f32>`, wrap it in a `Vector` to get back the FlyDSL surface:

```python
raw = rocdl.mfma_f32_16x16x16bf16_1k(Vec.make_type(4, fx.Float32), [a, b, c])
acc = fx.Vector(raw, (4,), fx.Float32)   # shape=(4,), dtype=Float32

# now use it like any other Vector
frag_C.store(acc)           # write into a register tensor
x = acc[2]                  # extract element 2 as a Numeric
```

### Summary: the three conversion paths

```
register Tensor  ──── .load()   ────►  Vector (flat SSA)
                 ◄─── .store() ────

individual Numerics ─ Vector.from_elements() ──► Vector
raw ir.Value ──────── Vector(val, shape, dtype) ► Vector
Vector ────────────── tensor[i]  ──────────────► Numeric (one element)
```

> **HIP/CK-Tile → FlyDSL.** `frag.load()` is reading the `float4` accumulator struct
> out of its register slots as a single value you can do math on; `frag.store(v)` is
> writing it back. `Vector.from_elements` is assembling the operand pack you would
> pass to `__builtin_amdgcn_mfma_*` as separate `float` arguments — here packed into
> one `vector<4×bf16>` that the intrinsic expects.

## A note on static vs. dynamic dimensions

Dimensions known at trace time (e.g. block-tile sizes passed as `Constexpr`) are
**static** — `A.shape.unpack()` returns Python ints and they are folded into the IR
as constants. Dimensions known only at runtime (e.g. `M` and `N` inferred from the
host-side tensor shape) are **dynamic** — they unpack to DSL `Int32` values and
remain as SSA operands in the IR.

The distinction matters for operations like `make_rmem_tensor(N, dtype)` (Chapter 10),
which requires a static element count: you cannot pass a dynamic shape there. Use
`fx.Constexpr` parameters for tile sizes that must be static, and keep problem
dimensions (M, N, K) dynamic.

With the Tensor API in hand, Chapter 6 covers the layout algebra — how `shape:stride`
layouts compose, divide, and product to build the tiling structures that distribute
tiles across threads.
