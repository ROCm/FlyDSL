---
name: raw-pointer-kernargs
description: >
  Replace fx.Tensor (memref descriptor) kernel arguments with bare i64 device
  addresses (fx.Int64 / fx.Pointer) to compact kernargs and enable a coalesced
  s_load prologue. Eliminates per-tensor shape/stride slots; biggest win at
  small-M / decode (prologue-bound), tapering to ~0 when compute dominates.
  Use when a FlyDSL kernel takes several global-buffer tensor arguments whose
  shape and strides are fully known at compile time and are never read from the
  incoming descriptor. Usage: /raw-pointer-kernargs
allowed-tools: Read Edit Bash Grep Glob
---

# Raw-Pointer Kernargs Optimization

Replace `fx.Tensor` (full memref descriptor) kernel arguments with bare `i64`
device addresses when the kernel only needs base pointers — shape and stride
are compile-time constants or supplied as separate scalar args. This compacts
the kernargs buffer and lets the GPU prologue fetch all pointers in one or two
`s_load_dwordx{8,16}` instructions instead of many scattered loads.

This is a pure **ABI change**: the arithmetic is untouched, so a correct
conversion is bit-identical to the original.

## Why This Matters

### `fx.Tensor` ABI — one memref descriptor per tensor

FlyDSL's `fx.Tensor` lowers to MLIR's **memref ABI**. Each tensor argument
in kernargs contains:

```
struct MemrefDescriptor {
    i64 base_ptr;
    i64 aligned_ptr;
    i64 offset;
    i64 shape[rank];    // rank × i64
    i64 stride[rank];   // rank × i64
};
```

For a rank-2 tensor that is `5 + 2×2 = 9` i64 words (72 bytes). Ten tensor
arguments → 720 bytes of kernargs, spread across non-contiguous descriptor
slots. The GPU prologue emits one `s_load_dwordx4` or `s_load_dwordx8` **per
descriptor field group**, serialising the reads.

### Raw `i64` / `fx.Pointer` ABI — one word per tensor

When the descriptor fields are never read, the kernel only uses `base_ptr`.
Replacing the argument with `fx.Int64` (or `fx.Pointer`) shrinks each slot from
72 bytes to **8 bytes**. Ten tensors → 80 bytes, packed contiguously → a single
`s_load_dwordx16` reads all ten pointers at once.

### Measured impact (it varies — measure your kernel)

The win equals the prologue's share of wall-clock, so it scales **inversely
with per-CTA arithmetic intensity** and is largest when the kernel is short
(small M / decode):

| Kernel | Arch | Regime | Δ |
|---|---|---|---|
| mxfp4 MoE gemm1/gemm2 (9 ptrs) | gfx950 | M=4–8 decode | **−7%** |
| " | | M=16–32 | −2–4% |
| " | | M≥64 (MFMA-bound) | ≈ 0% |
| PA decode `pa_decode_ps` + reduce (10 + 4 ptrs) | gfx942 | batch 1 decode | **≈ −3%** |

Two takeaways: (1) decode/small-M is where it pays; (2) kernels that already do
substantial per-CTA work (e.g. paged attention) see a smaller relative gain
than thin gemm/elementwise kernels. **Always A/B with per-kernel timing**
(`rocprofv3 --kernel-trace`, min/median of several captures) rather than
assuming a fixed number.

## When to Apply

Apply when **all** of the following hold:

1. The argument is a global (HBM) buffer — not LDS or register.
2. The kernel never reads the incoming descriptor's `.shape` / `.strides` /
   `.offset`. Sizes and strides are either compile-time constants or passed
   as separate scalar (`fx.Int32` / `fx.Int64`) arguments.
3. The body turns the pointer into something usable via one of the
   raw-pointer-friendly APIs (see Step 2) — `GTensor(addr, shape=…)`,
   `buffer_ops.create_buffer_resource_from_addr(addr)`, or an `inttoptr` to a
   global pointer.
4. You can edit the host launcher (`@flyc.jit` wrapper) and its call site to
   pass `tensor.data_ptr()`.

Do **not** apply when:
- Shape or strides are runtime values read from the descriptor (dynamic shapes).
- The argument feeds a layout API that reads the memref descriptor at runtime.
- You cannot modify the host launcher / call site.

## Step-by-Step

### 1. Identify eligible arguments

Find `fx.Tensor` parameters whose only use is to obtain a base pointer. In
practice the body does one of:

- `GTensor(arg, shape=…, strides=…)` with explicit compile-time sizes;
- `buffer_ops.create_buffer_resource(arg, …)` → `buffer_load`/`buffer_store`;
- `extract_global_ptr(arg)` → direct `llvm` / global loads.

If it only calls those (never `arg.shape` etc.), it is eligible.

### 2. Rewrite the kernel signature and body

Change each eligible `fx.Tensor` → `fx.Int64` (or `fx.Pointer`), then update
the consumption call to its raw-address counterpart:

| Memref form (before) | Raw-pointer form (after) |
|---|---|
| `GTensor(arg, shape=…)` | `GTensor(arg, shape=…)` *(accepts an i64 base too)* |
| `buffer_ops.create_buffer_resource(arg, max_size=True)` | `buffer_ops.create_buffer_resource_from_addr(arg)` |
| `buffer_ops.create_buffer_resource(arg, max_size=False)` | `create_buffer_resource_from_addr(arg, num_records_bytes=…)` — **see Pitfall B** |
| `extract_global_ptr(arg)` → `!llvm.ptr<1>` | `inttoptr(arg)` → `!llvm.ptr<1>` (a one-line helper, e.g. `llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr)`) |

```python
# BEFORE
@flyc.kernel(name="my_kernel")
def my_kernel(a: fx.Tensor, b: fx.Tensor, M: fx.Int32):
    a_rsrc = buffer_ops.create_buffer_resource(a, max_size=True)
    b_ptr  = extract_global_ptr(b)

# AFTER
@flyc.kernel(name="my_kernel")
def my_kernel(a: fx.Int64, b: fx.Int64, M: fx.Int32):
    a_rsrc = buffer_ops.create_buffer_resource_from_addr(a)
    b_ptr  = global_ptr_from_addr(b)   # inttoptr → !llvm.ptr<1>
```

### 3. Type the `@flyc.jit` launcher's pointer args as `fx.Int64`

This step is **mandatory, not cosmetic** — see Pitfall A. Every pointer
parameter of the launcher must be annotated `fx.Int64`. Untyped launcher
parameters are inferred as `i32` from a Python int and will **truncate the
64-bit address**.

```python
# BEFORE
@flyc.jit
def launch(a: fx.Tensor, b: fx.Tensor, M: fx.Int32):
    my_kernel(a, b, M)

# AFTER
@flyc.jit
def launch(a: fx.Int64, b: fx.Int64, M: fx.Int32):   # NOT bare `a, b`
    my_kernel(a, b, M)
```

### 4. Update the Python call site

```python
# BEFORE
launch(tensor_a, tensor_b, M)
# AFTER — pass raw device pointers
launch(tensor_a.data_ptr(), tensor_b.data_ptr(), M)
```

`data_ptr()` returns the buffer's *base* address. If the kernel expects a view
offset baked in, pass `t.data_ptr()` and keep the offset in your stride/offset
scalar args, or use the view's own `data_ptr()`. With FlyDSL ≥ 0.2.0 you can
also use `flyc.from_c_void_p` when bridging from a ctypes/cffi address.

### 5. Verify correctness, then measure

Correctness first — output must be bit-identical (ABI-only change):

```bash
python op_tests/test_<your_kernel>.py     # cosine / allclose vs reference
```

Then measure per-kernel GPU time at your target regime (decode/small-M):

```bash
rocprofv3 --kernel-trace -f csv -d /tmp/prof -- python bench.py
# compare min/median of `<kernel>` duration, raw vs memref, over several runs
```

## Common Pitfalls

### A. Untyped launcher → i32 pointer truncation (causes GPU faults)

The single most common mistake. A `@flyc.jit` launcher parameter with **no type
annotation** receives a Python int (`data_ptr()`) and is inferred as **i32**,
silently dropping the high 32 bits of the address. The kernel then dereferences
a truncated pointer → *Memory access fault by GPU* (often at a suspiciously
small address like `0x655ff000`).

Fix: annotate every launcher pointer parameter `fx.Int64`. Scalar stride/size
args can stay untyped (i32 is correct for them). Symptom-to-cause: if the
memref version worked but the raw version faults at a low address, check the
launcher annotations first.

### B. Losing the hardware OOB bound (`max_size=False`)

A buffer resource built from a memref with `max_size=False` derives its
`num_records` (OOB bound) from the **memref size**, so out-of-range
`buffer_load`s return 0 via the hardware bounds check. A raw i64 has no size, so
`create_buffer_resource_from_addr` defaults to an unbounded descriptor
(`0xFFFFFFFF`) — any OOB access now reads garbage or faults.

If the kernel *relies* on that bounds check (e.g. block-table reads past the end
on empty/short tiles), pass the size explicitly:

```python
# e.g. block_tables is [num_seqs, stride] i32, grid.x == num_seqs:
n_bytes = fx.Int32(gpu.grid_dim.x) * stride_bt * fx.Int32(4)
rsrc = buffer_ops.create_buffer_resource_from_addr(bt_addr, num_records_bytes=n_bytes)
```

Buffers used with `max_size=True` (no real bound) map directly to
`create_buffer_resource_from_addr(addr)` with no extra work.

### C. `data_ptr()` validity and CUDA-graph capture

`data_ptr()` is read at launch time; the buffer must be alive and resident.
Under CUDA-graph capture the captured address must remain stable across replays,
so pass **preallocated** tensors (the same requirement as the memref ABI, just
made explicit). Don't pass `.data_ptr()` of a temporary that may be freed.

### D. Convert the whole launch, not just one kernel

A decode op often fires several kernels (e.g. a partition kernel **and** a
reduce kernel). The kernarg/prologue saving compounds across them, so convert
every kernel in the hot path. Conversely, one-shot setup kernels (a worklist
scheduler that runs once per op, not per token) gain nothing — skip them to
limit churn.

## Reference Implementations

- **mxfp4 MoE gemm1/gemm2** — `aiter/ops/flydsl/kernels/mxfp4_gemm1.py`,
  commit `d2d28f031` (aiter #3861). 9 pointers → all `fx.Int64`, packed
  contiguously; ~7% at decode. The original worked example.
- **`chunk_gated_delta_h.py`** — commit `e7c5a31c3`, using `fx.Pointer`
  (equivalent to `fx.Int64`).
- **Paged-attention decode** — `pa_decode_ps_kernel` (10 ptrs),
  `pa_decode_sw_reduce_kernel` (4), `pa_decode_sw_kernel` (11),
  `pa_decode_metadata_kenrel` (14). Exercises the mixed-API case (both
  `create_buffer_resource_from_addr` and `inttoptr`/global-load), the
  `max_size=False` OOB bound (Pitfall B, block_tables), and the multi-kernel
  launch (Pitfall D). The worklist scheduler kernel was intentionally left on
  the memref ABI (one-shot, no decode benefit).

## Interaction With Other Optimizations

- **Prologue vmcnt relax**: after compacting kernargs the s_load fence can
  often be relaxed from `s_waitcnt vmcnt(0)` to a lower count, further hiding
  latency.
- **AOT / `flyc.compile`**: fully compatible — the low-overhead `flyc.compile`
  launch passes addresses the same way as JIT.
- **`fx.Pointer` vs `fx.Int64`**: semantically identical here. `fx.Pointer` is
  the newer name (FlyDSL ≥ 0.2.0); `fx.Int64` works on all versions.
