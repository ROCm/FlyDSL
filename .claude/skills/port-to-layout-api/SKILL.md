---
name: port-to-layout-api
description: >
  Port FlyDSL GPU kernels from raw buffer_ops (create_buffer_resource,
  buffer_load, buffer_store with manual byte-offset arithmetic) to the
  layout API (make_buffer_tensor + logical_divide + copy_atom_call with
  BufferCopy atoms). Use when a kernel uses raw buffer_ops and should
  be migrated to the higher-level layout algebra for consistency and
  readability.
  Usage: /port-to-layout-api <kernel_file>
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Port Kernel from Raw buffer_ops to Layout API

Refactor FlyDSL GPU kernels from manual `buffer_ops` to the buffer-backed
layout API (`make_buffer_tensor` + `copy_atom_call`).

## When to Use

- Kernel uses `buffer_ops.create_buffer_resource()` + `buffer_ops.buffer_load()` / `buffer_ops.buffer_store()`
- Manual byte-offset or dword-offset arithmetic (shrui, elem_bytes, etc.)
- Want to align with the layout algebra pattern used in norm/softmax kernels

## Step-by-Step Process

### Step 1: Identify the Memory Access Pattern

Read the kernel and classify each buffer_load/buffer_store:

| Pattern | Layout API Port | Example |
|---------|----------------|---------|
| Contiguous vec load along innermost dim | `make_buffer_tensor` + `BufferCopy128b` | Load 8xf16 from row |
| Scalar load (vec_width=1) | `make_buffer_tensor` + `BufferCopy32b`/`BufferCopy16b` | Scale/metadata loads |
| Scattered store (non-contiguous layout) | Keep as `buffer_ops.buffer_store` | Non-flash value_cache |
| Contiguous vec store along innermost dim | `make_buffer_tensor` + `BufferCopy` | Store 8xf16 to output |

### Step 2: Choose the Right BufferCopy Width

Match `BufferCopy<N>b` to element type and vector width:

```
Total bits = VEC_WIDTH * elem_bits
```

| elem_type | VEC_WIDTH | Total bits | Copy Atom |
|-----------|-----------|------------|-----------|
| f16/bf16  | 8         | 128        | `BufferCopy128b()` |
| f32       | 4         | 128        | `BufferCopy128b()` |
| f32       | 1         | 32         | `BufferCopy32b()` |
| i8        | 8         | 64         | `BufferCopy64b()` |
| f16/bf16  | 1         | 16         | `BufferCopy16b()` |

**Max supported**: 128 bits. No `BufferCopy256b` exists.
For f32 with VEC_WIDTH=8 (256 bits), fall back to scalar path or split into two 128b loads.

### Step 3: Replace buffer_ops with Layout API

**Before (raw buffer_ops):**
```python
from flydsl.expr import buffer_ops
from flydsl.expr.arith import ArithValue

rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)

elem_bytes = 2  # f16
row_soffset = ArithValue(bid) * (N * elem_bytes)
thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)
col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
dw = col_bytes.shrui(arith.constant(2, type=T.i32))

raw_data = buffer_ops.buffer_load(
    rsrc, dw, vec_width=vec_dwords, dtype=T.i32,
    soffset_bytes=row_soffset, mask=is_valid,
)
vec_f16 = vector.bitcast(vec_type_f16, raw_data)
```

**After (layout API):**
```python
# Wrap tensor in buffer descriptor
Input_buf = fx.rocdl.make_buffer_tensor(Input)

# Slice row for this block, tile by VEC_WIDTH
row_in = fx.slice(Input_buf, (bid, None))
in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))

# Copy atom: 8 x f16 = 128 bits
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
vec_reg_ty = fx.MemRefType.get(
    T.f16, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
)
vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

# Load
idx = tid + tile_i * BLOCK_THREADS
r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
fx.copy_atom_call(copy_atom, fx.slice(in_div, (None, idx)), r)
vec_f16 = ArithValue(fx.memref_load_vec(r))
```

### Step 4: Handle Multi-Dimensional Tensors

For 2D tensors `[M, N]`:
```python
buf = fx.rocdl.make_buffer_tensor(tensor)
row = fx.slice(buf, (row_idx, None))          # 1D: N elements
div = fx.logical_divide(row, fx.make_layout(VEC_WIDTH, 1))
```

For 3D tensors `[A, B, C]`:
```python
buf = fx.rocdl.make_buffer_tensor(tensor)
slice_bc = fx.slice(buf, (a_idx, b_idx, None))  # 1D: C elements
div = fx.logical_divide(slice_bc, fx.make_layout(VEC_WIDTH, 1))
```

For tiled access (grid tiles across a dimension):
```python
row = fx.slice(buf, (row_idx, None))
tiled = fx.logical_divide(row, fx.make_layout(tile_cols, 1))
tile = fx.slice(tiled, (None, tile_idx))
div = fx.logical_divide(tile, fx.make_layout(VEC_WIDTH, 1))
# Load: fx.copy_atom_call(copy_atom, fx.slice(div, (None, tid)), r)
```

### Step 5: Handle Masking / Bounds

The layout API's `copy_atom_call` does NOT accept a mask parameter.

**For loads**: With `make_buffer_tensor` using max_size (0xFFFFFFFF num_records),
out-of-bounds loads read adjacent memory or return 0 at allocation boundary.
Guard results with `arith.select(is_valid, value, zero)` as needed.

**For stores**: Wrap in a conditional to prevent OOB writes:
```python
if is_valid:
    _store_vec(val, out_div, idx)
```

### Step 6: Scalar Loads via Layout API

Scalar loads (vec_width=1) also work through the layout API:

```python
buf = fx.rocdl.make_buffer_tensor(tensor, max_size=True)
copy_atom_s = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)  # f32 scalar
scalar_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
scalar_reg_lay = fx.make_layout(1, 1)
div = fx.logical_divide(buf, fx.make_layout(1, 1))

def load_scalar(index):
    r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
    fx.copy_atom_call(copy_atom_s, fx.slice(div, (None, fx.Int32(index))), r)
    return Vec(fx.memref_load_vec(r))[0]  # extract scalar from vector<1xf32>
```

Scalar stores work the same way (reverse src/dst):
```python
def store_scalar(index, val):
    r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
    fx.memref_store_vec(Vec.filled(1, val, Float32), r)
    fx.copy_atom_call(copy_atom_s, r, fx.slice(div, (None, fx.Int32(index))))
```

Keep `buffer_ops` only for:
- Scattered stores where elements are truly non-contiguous in memory

### Step 7: Remove Dead Code

After porting, remove:
- `elem_bytes` / `vec_dwords` constants (no longer needed)
- `row_soffset` / `thr_col_bytes` byte-offset computations
- `shrui(..., 2)` dword conversion
- `vector.bitcast` from i32 to element type (loads produce the right type)

## Known Limitation: Dynamic Tensor Shapes & JIT Cache

`make_buffer_tensor` defaults to `max_size=True`, which sets `num_records` to
`0xFFFFFFFF`. This is necessary because the JIT cache reuses compiled kernels
for tensors of different shapes (the cache key includes dtype but not shape).
If `max_size=False` were used, the `num_records` from the first traced shape
would silently truncate larger tensors in subsequent calls.

Always use `fx.rocdl.make_buffer_tensor(tensor)` (default `max_size=True`).
Only use `make_buffer_tensor(tensor, max_size=False)` when you are certain the
tensor shape never changes across JIT-cached invocations.

## Known Limitation: soffset Regression

The layout API currently hardcodes `soffset=0` in `CopyAtomCallLowering`
(`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`). This means wave-uniform row
offsets (e.g., `bid * N`) are folded into voffset (VGPR) instead of soffset
(SGPR), causing:

1. Extra VGPR pressure
2. Extra VALU instructions
3. Wasted soffset hardware field

This is a known trade-off for API consistency. The fix (extending BufferFatPtr
to carry a separate soffset) is tracked separately.

## Validation Checklist

- [ ] Kernel compiles without errors
- [ ] Correctness matches reference (same tolerances as before)
- [ ] Performance is within acceptable range (may regress slightly due to soffset)
- [ ] gfx950 compile test passes (`COMPILE_ONLY=1 ARCH=gfx950`)
- [ ] No unused imports or dead variables left behind
