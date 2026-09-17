# The Fly Dialect III — Data Movement (Copy Atoms)

Getting bytes from global memory, through LDS, into registers — and back — is
half of a fast kernel. In HIP you write this as `buffer_load`/`ds_write`/
`ds_read` intrinsics with careful address math; in CK-Tile it is `load_tile` /
`store_tile` over tile windows. FlyDSL expresses it as a **copy atom** driving a
**TiledCopy** (Chapter 7), and a single `fx.copy` / `fx.copy_atom_call` per
transfer.

## What a copy atom is

A **copy atom** names *one hardware move instruction* plus the element type it
moves. It is the copy analogue of an MMA atom. FlyDSL ships a target-neutral
family and several AMD-specific ones:

| Atom factory | Lowers to | Use |
|--------------|-----------|-----|
| `fx.UniversalCopy128b()` / `64b` / `32b` / `UniversalCopy(bits)` | plain vectorized load/store | target-neutral; portable |
| `fx.rocdl.BufferCopy128b()` / `64b` / `32b` | `rocdl.buffer_load` / `buffer_store` | AMD buffer-descriptor path with hardware bounds checking |
| `fx.rocdl.BufferCopyLDS128b()` / `32b` | `buffer_load … lds` | one-instruction Global→LDS; the compiler inserts the `vmcnt` wait |
| `fx.rocdl.cdna4.BufferLoadAsyncLDS128b()` / `96b` / `32b` | async LDS DMA | gfx950 only; *you* wait (see below) |
| `fx.rocdl.make_tdm_atom(...)` | `rocdl` TDM async copy | gfx1250 whole-tile Global↔LDS (out of scope on CDNA) |

> **Gotcha — there is no 64-bit LDS DMA.** `BufferCopyLDS64b()` is deprecated and
> now raises: the hardware has no 8-byte `buffer_load … lds`. Use `32b` or `128b`.

You wrap the instruction in an atom with the element type:

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
```

The `128b` is the **access width**: one instruction moves 128 bits = 4×fp32 =
8×fp16. Wider is fewer instructions and better coalescing, but requires the tile
and TV layout to be divisible by that width. Choosing the widest width your
alignment allows is the first, cheapest copy optimization.

> **HIP/CK-Tile → FlyDSL.** `BufferCopy128b` ≡ a `buffer_load_dwordx4` through an
> AMD buffer resource descriptor; the descriptor gives you *free hardware
> out-of-bounds handling* (loads past the end return 0, stores are dropped) —
> the same reason CK-Tile prefers buffer addressing. `UniversalCopy` ≡ a plain
> `global_load`/`flat_load` with no descriptor. Access width is your
> `dwordx{1,2,4}` choice.

## Buffer tensors: the modern global-memory handle

New kernels do **not** compute byte offsets by hand. They wrap a global tensor in
a **buffer tensor**, which keeps the shape/stride (a Layout, §6.2) *and* adds the
AMD buffer descriptor that `BufferCopy` needs:

```python
A = fx.rocdl.make_buffer_tensor(A_tensor)
```

### What it actually changes

`make_buffer_tensor` takes a tensor whose iterator is a **global** pointer and
returns a tensor with the **same element type and the same layout**, whose
iterator is a pointer in the `BufferDesc` address space. Only the address space
changes; the algebra does not notice:

```
fx.Tensor  !fly.memref<f16, global,     layout(...), align<16>>
   │  fx.rocdl.make_buffer_tensor(t)
fx.Tensor  !fly.memref<f16, bufferdesc, layout(...), align<16>>
```

Because the layout rides along, everything in Chapters 6–7 works unchanged:

```python
A   = fx.rocdl.make_buffer_tensor(A_tensor)
bA  = fx.slice(fx.zipped_divide(A, (BM, BK)), (None, bid))   # this block's tile
tA  = thr_copy.partition_S(bA)                               # this thread's values
fx.copy(buffer_atom, tA, frag_A)                             # buffer_load_dwordx4
```

That address space is the whole point: it is what makes a copy atom lower to
`rocdl.raw.ptr.buffer.load` (→ `buffer_load_*`) instead of `llvm.load`
(→ `global_load_*`). What you buy is described in §10.2 — the descriptor gives
hardware bounds checking and a free SGPR offset path. What you give up is
generality: `UniversalCopy` wants a *plain* pointer and aborts in lowering if you
hand it a buffer one (§15.6), and the gfx1250 TDM engine needs a raw virtual
address, so it takes `fx.rocdl.make_tdm_atom(...)` instead of a buffer tensor.
(`fx.ptr_load` / `fx.ptr_store` do accept a `BufferDesc` pointer — they lower to
`buffer_load_*` in that case, §10.4.)

### The bound is a parameter, not a given

The descriptor carries a `num_records` field, and that is what the hardware
compares against. FlyDSL gives you three ways to set it:

```python
A = fx.rocdl.make_buffer_tensor(A_t)                        # default: max_size=True
A = fx.rocdl.make_buffer_tensor(A_t, max_size=False)        # bound = cosize(layout)*elem_bytes
A = fx.rocdl.make_buffer_tensor(A_t, num_records_bytes=nb)  # bound = your byte count
```

The **default is unbounded**: `num_records = 0xFFFFFFFF`, so no address a kernel
can realistically form is out of range. You still get the `soffset` path and the
buffer instruction encoding, but the "OOB loads return 0" guarantee is not armed.
`max_size=False` derives the byte count at run time from the tensor's own layout
footprint (`cosize(layout) × elem_bytes`), which is the right answer whenever the
tensor's layout describes the whole allocation. `num_records_bytes=` takes a
compile-time constant when you know it and want it folded into the IR. On RDNA
the distinction is doubly important: supplying a bound also switches the
descriptor's `OOB_SELECT` field into the checked mode (§10.2), so an unbounded
RDNA descriptor checks nothing at all.

### Per-atom `soffset`: stepping a K-loop for free

A CDNA buffer copy atom carries a **`soffset`** state field — a scalar (SGPR)
byte offset added by the hardware on top of the per-thread address. That is how a
K-loop advances its tile without recomputing any per-lane address:

```python
# examples/04-preshuffle_gemm.py:80 — the global K offset rides in the SGPR
fx.copy(
    buffer_copy_atom,
    thr_gA_k[None, None, None, 0],
    copy_frag_A,
    soffset=next_k * gA_k_stride,
)
```

Each lane's `voffset` stays fixed across the whole loop; only one scalar register
changes per iteration. In HIP you would be re-deriving `v_add` chains per lane.

> **HIP/CK-Tile → FlyDSL.** A buffer tensor is CK-Tile's `make_buffer_view` +
> `make_tile_window` fused into one value: the descriptor *and* the tensor
> descriptor in a single object. `max_size=False` is the equivalent of passing
> the real element count to `make_buffer_view` rather than a "big enough" one,
> and `soffset=` is the SGPR offset argument of `buffer_load`.

Raw `create_buffer_resource()` and manual byte offsets still exist in
`kernels/common/buffer_ops.py` for un-migrated kernels, but are discouraged for
new code (`CLAUDE.md` → Kernel Authoring Conventions): they lose the layout, so
none of the partitioning algebra applies.

## Executing a copy

Once a thread has a source partition and a destination partition (Chapter 7),
one call moves the data:

```python
fx.copy(copy_atom, src_part, dst_part)                # tiled copy
fx.copy_atom_call(copy_atom, src_part, dst_part)      # single-atom form
```

`fx.copy` drives a whole TiledCopy (every atom in the distribution);
`fx.copy_atom_call` issues one atom, used when you manage the loop yourself (e.g.
the softmax kernel copies a row slice at a time). Both lower, in Stage A
(`convert-fly-to-rocdl`, §2.5), to the `rocdl.buffer_load/store` or vector
load/store the atom names.

### The three copies of a GEMM inner step

A tiled MMA kernel performs, per K-step, exactly three copy patterns:

```python
# global -> registers (A and B operands), retiled to fragment order (§7.4)
fx.copy(copy_atom_a, thr_copy_a.partition_S(bA), thr_copy_a.retile(frag_A))
fx.copy(copy_atom_b, thr_copy_b.partition_S(bB), thr_copy_b.retile(frag_B))
# ... fx.gemm(...) ...
# registers -> global (accumulator store)
fx.copy(copy_atom_c, thr_copy_c.retile(frag_C), thr_copy_c.partition_D(bC))
```

Add an LDS staging tier (§10.5 / Chapter 14 / GEMM puzzles) and it becomes
global→LDS then LDS→registers, but the call shape is identical.

## Predication: safe boundaries without branches

When a tile does not evenly divide the tensor (the last block), some threads
would read/write out of bounds. Two defenses:

1. **Buffer atoms** give you hardware bounds checking for free — an OOB
   `buffer_load` returns 0, an OOB `buffer_store` is dropped. Often that is all
   you need. But the bound is opt-in: `make_buffer_tensor` defaults to an
   unbounded descriptor, so pass `num_records_bytes=` or `max_size=False` when you
   are relying on the check (§10.2).
2. **Explicit predicates** for the general/universal path: build a boolean
   *predicate tensor* marking which of a thread's values are in range, and pass
   it to `fx.copy`:

```python
# from examples/01-vectorAdd.py, conceptually
pred = fx.elem_less(coord_tensor, fx.make_coord(M, N))   # in-bounds mask
fx.copy(copy_atom, src_part, dst_part, pred=pred)
```

The predicate is itself a partitioned tensor (same TV layout as the data), so
`fx.copy` masks each thread's values individually. This is FlyDSL's version of
the `if (idx < N)` guard you scatter through a HIP kernel — computed once as a
layout, applied by the copy.

> **HIP/CK-Tile → FlyDSL.** Predicated `fx.copy` ≡ CK-Tile's `pad_tensor_view` +
> masked `load_tile`. Prefer buffer atoms' hardware bounds check where possible
> (it costs no VGPRs for a mask); reach for explicit predicates when you must
> also guard compute, or on the universal (non-buffer) path.

## A complete tiled-copy example

`examples/02-tiledCopy.py` copies a 2-D tile with a buffer atom. The skeleton:

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dtype)

src = fx.rocdl.make_buffer_tensor(src_ptr)
dst = fx.rocdl.make_buffer_tensor(dst_ptr)

# tile both by the block shape and slice this block (Ch. 6)
bsrc = fx.slice(fx.zipped_divide(src, (BM, BN)), (None, bid))
bdst = fx.slice(fx.zipped_divide(dst, (BM, BN)), (None, bid))

# thread/value layout -> TiledCopy (Ch. 7)
tiler_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy = fx.make_tiled_copy(copy_atom, tv, tiler_mn)
thr_copy = tiled_copy.get_slice(tid)

# move it
fx.copy(copy_atom, thr_copy.partition_S(bsrc), thr_copy.partition_D(bdst))
```

Everything before the last line is *description* (layouts, distribution); the
last line is the only *action*. That ratio — lots of layout algebra, one copy —
is normal and is exactly what makes the data movement inspectable and
optimizable. Chapter 9 does the same for the matrix multiply itself.

## LDS (shared memory): declaring and allocating it

In HIP you write `__shared__ float tile[128][32];` and the compiler reserves the
bytes. FlyDSL splits that into two explicit steps: you **declare a storage type**,
then you **allocate it** from an allocator that hands back typed pointers.

### Step 1 — declare the layout as a type

The storage layout is an `@fx.struct` whose fields are `fx.Array`s. This is the
same struct machinery from §4.5, used purely as a byte-layout description — the
field offsets, padding, and total size are computed by the C layout rules, and the
trailing `16` asks for the 128-bit alignment the copy atoms want:

```python
# kernels/gemm/rdna_f16_gemm.py:108 — A/B staging, double-buffered
@fx.struct
class SharedStorage:
    a0: fx.Array[elem_cls, LDS_A_ELEMS, 16]
    b0: fx.Array[elem_cls, LDS_B_ELEMS, 16]
    a1: fx.Array[elem_cls, LDS_A_ELEMS, 16]
    b1: fx.Array[elem_cls, LDS_B_ELEMS, 16]
```

`fx.Array` fields are **flat, one-dimensional, and element-counted** — not shaped.
Shape comes later, from a layout. Declare `fx.Array[dtype, M * N, 16]`, not a
two-dimensional array.

### Step 2 — allocate, peek, and view

```python
lds  = fx.SharedAllocator().allocate(SharedStorage).peek()
sA   = lds.a0.view(fx.make_layout((BM, BK), (BK, 1)))   # now it has a shape
```

Three calls, three distinct jobs:

- **`.allocate(T)`** reserves the bytes and returns a `Storage[T]` — a handle, not
  a value.
- **`.peek()`** materializes the Python-side object: for a struct, one attribute
  per field, each an `fx.Array` value carrying its typed LDS `.ptr`.
- **`.view(layout)`** attaches a layout to a field's pointer, producing an
  `fx.Tensor` in the Shared address space that `fx.copy` and the partitioning
  algebra accept. (`fx.make_view(field.ptr, layout)` is the same thing spelled
  out.) This is where the swizzle goes, if you want one:

```python
# kernels/gemm/preshuffle_gemm.py:303 — the layout carries the swizzle, not the copy
swz = fx.SwizzleType.get(3, 3, 3)
sA = fx.make_view(
    lds.a0.ptr,
    fx.make_composed_layout(
        fx.static(swz),
        fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
    ),
)
```

Then the actual staging is the three-line pattern:

```python
fx.copy(copy_atom, thr_copy.partition_S(bA), thr_copy.partition_D(sA_part))
fx.gpu.barrier()                       # all threads: LDS filled
fx.copy(copy_atom, sA_part_s2r, frag_A)
```

### Several independent buffers

Two idioms, both real, and they compose:

**(a) one struct, many fields** — the dominant pattern. Every buffer is a field;
the struct layout places them back to back with the padding each one asks for:

```python
# kernels/gemm/fp8_gemm_4wave.py:161 — 8 disjoint staging buffers
@fx.struct
class SharedStorage:
    A_lds_cur_0:  fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
    A_lds_cur_1:  fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
    ...
    B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

lds = fx.SharedAllocator().allocate(SharedStorage).peek()
a_cur0 = lds.A_lds_cur_0
```

**(b) repeated `allocate()` on the same allocator.** The allocator is a bump
allocator: each call rounds the cursor up to the requested alignment and advances
it, so successive allocations are **disjoint**, in call order:

```python
# kernels/moe/moe_sorting_kernel.py:221 — kernel's own storage, then a library's
allocator = fx.SharedAllocator()
lds = allocator.allocate(SharedStorage).peek()
scan_storage = allocator.allocate(block_scan.SharedStorage).peek()
```

Form (b) is what you need when a *library* owns part of the layout — a
`fx.coop.BlockScan` asks you to allocate its `SharedStorage` and hand it back
(§13.3), and it has no business being a field of your struct.

`allocator.allocated_bytes` is the running total, readable at trace time:

```python
allocator.allocate(Shared);  allocator.allocated_bytes   # 1024
allocator.allocate(Scratch); allocator.allocated_bytes   # 1280
```

### Deliberately overlapping buffers: `@fx.union`

Disjoint is the default. When two buffers are alive in *different phases* of the
kernel and you want them to share the same bytes, say so with `@fx.union`: all
variants sit at offset 0 and the union's size is the largest of them.

```python
# kernels/gemm/gemm_a16w16_gfx950.py:504 — the epilogue reuses the A/B staging bytes
@fx.struct
class SharedABStorage:
    a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
    b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

@fx.union
class SharedStorage:
    ab: SharedABStorage                                        # main loop
    c:  fx.Array[elem_dtype, k_waves * block_m * block_n, 16]  # epilogue

storage = fx.SharedAllocator().allocate(SharedStorage)
smem_a = storage.ab.a.peek().ptr
smem_c = storage.c.peek().ptr        # same LDS address as smem_a
```

This is a real aliasing contract, and the compiler will not police it: if the
epilogue starts writing `c` while another wave still reads `a`, you need a barrier
between the phases (Chapter 12) — the union guarantees the overlap, not the ordering.

### How many allocators may a kernel have? Exactly one.

`fx.SharedAllocator()` registers itself with the enclosing kernel, and a second
construction is a hard error:

```
RuntimeError: Only one SharedAllocator is allowed per kernel;
kernel 'my_kernel' already has one
```

It must also be constructed **inside** a `@flyc.kernel` body (constructing one at
module scope raises too). So the rule is simple: one allocator, created once near
the top of the kernel, threaded to whatever needs storage. If a helper needs LDS,
pass it the allocator or the already-peeked storage — do not let it make its own.

### Static versus dynamic placement

The allocator has two modes, and the default is the one you want:

| | `static=True` (default) | `static=False` |
|---|---|---|
| C analogue | `__shared__ float buf[N];` | `extern __shared__ char buf[];` |
| IR | one LDS global **per leaf field** | one `fly.get_dyn_shared` base, fields at fixed offsets |
| Who sizes it | the compiler, from the globals | you (or the launcher) |
| `launch(smem=)` | leave unset | inferred from `allocated_bytes` if `smem=None` |

In static mode each leaf of the storage tree becomes its own `@__shared_alloc_*`
LDS global, and the linker lays them out — which is why `.launch(...)` needs no
`smem` argument at all. In dynamic mode every field is an offset off a single
base, the launch wrapper fills in `smem` from `allocated_bytes` when you pass
`smem=None`, and an explicit `smem` that is smaller than the allocation is
rejected:

```
ValueError: launch smem=4096 is less than the 8192 bytes allocated by
SharedAllocator in kernel 'my_kernel'
```

> **Gotcha — nothing checks the LDS budget.** Neither mode validates the total
> against the target's LDS capacity (64KB on gfx942/gfx11\*/gfx120\*, 160KB on
> gfx950, 320KB on gfx1250). Over-allocate and you find out at kernel-launch time
> from the driver, not at trace time. A few kernels call
> `check_smem_capacity(nbytes, arch)` from `flydsl/utils/smem_allocator.py` by
> hand; it is not wired into `SharedAllocator`.

> **HIP/CK-Tile → FlyDSL.** `SharedAllocator(static=True)` is a set of
> `__shared__` declarations; `static=False` is `extern __shared__` plus the
> manual offset arithmetic you would otherwise write, done for you. `@fx.union`
> is the `union`/placement-new trick you use to reuse an LDS block between the
> main loop and the epilogue. The one-allocator rule has no C++ analogue — it
> exists because two independent bump cursors over the same dynamic base would
> silently alias.

The legacy `utils.smem_allocator.SmemAllocator` / `SmemPtr` path is still present
for un-migrated kernels. It works on raw `memref.global` + `memref.view` with
manual byte offsets and a `finalize()` step, and it carries the
`SmemPtr._view_cache = None` gotcha after an `scf.for`. `SharedAllocator` has no
view cache and no `finalize()`; new kernels should use it.

Bank-conflict avoidance is a *layout* concern: you pick a swizzled LDS layout so
lanes hit distinct banks (Chapter 14 and the GEMM puzzles). The copy calls do not
change — only the LDS layout does. Full LDS mechanics, double-buffering, and
swizzle live in `docs/kernel_tuning_guide.md`; the GEMM puzzles build them up
step by step.

### Async Global→LDS on CDNA4

`BufferCopyLDS*` is *synchronous* in the sense that matters: the compiler emits the
`vmcnt` wait for you before the staged LDS data is read, so you cannot overlap the
copy with anything. gfx950 adds atoms that skip that wait and hand you the
bookkeeping instead — `fx.rocdl.cdna4.BufferLoadAsyncLDS{32,96,128}b()` (from a
buffer tensor) and `GlobalLoadAsyncLDS{32,96,128}b()` (from a plain global
pointer). You bracket a group of them yourself:

```python
fx.copy(async_atom, thr_gA, thr_sA[..., stage])   # issue, no wait emitted
fx.rocdl.asyncmark()                              # close this group
# ... compute on the previous stage while the DMA runs ...
fx.rocdl.wait_asyncmark(0)                        # drain: 0 groups outstanding
fx.gpu.barrier()
```

`wait_asyncmark(n)` waits until at most `n` groups remain, which is how a
multi-stage pipeline keeps `n` copies in flight. The count must be compile-time.

> **HIP/CK-Tile → FlyDSL.** This is the AMD counterpart of `cp.async` +
> `cp.async.wait_group N`: the atom is the issue, `asyncmark` is the commit, and
> `wait_asyncmark` is the wait. Forget the wait and you read stale LDS — the
> compiler will not save you, which is exactly the trade you asked for.

A special case — reading a tile **transposed** out of LDS so it lands in
matrix-core operand order — is covered in Chapter 9 (§"Transposed operand
loads"), since it only makes sense in the context of feeding an MFMA.
