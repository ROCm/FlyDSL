# The Fly Dialect III — Data Movement (Copy Atoms)

Getting bytes from global memory, through LDS, into registers — and back — is
half of a fast kernel. In HIP you write this as `buffer_load`/`ds_write`/
`ds_read` intrinsics with careful address math; in CK-Tile it is `load_tile` /
`store_tile` over tile windows. FlyDSL expresses it as a **copy atom** driving a
**TiledCopy** (Chapter 7), and a single `fx.copy` / `fx.copy_atom_call` per
transfer.

## What a copy atom is

A **copy atom** names *one hardware move instruction* plus the element type it
moves. It is the copy analogue of an MMA atom. FlyDSL ships two families:

| Atom factory | Lowers to | Use |
|--------------|-----------|-----|
| `fx.UniversalCopy128b()` / `64b` / `32b` / `UniversalCopy(bits)` | plain vectorized load/store | target-neutral; portable |
| `fx.rocdl.BufferCopy128b()` / `64b` / `32b` | `rocdl.buffer_load` / `buffer_store` | AMD buffer-descriptor path with hardware bounds checking |
| `fx.rocdl.make_tdm_atom(...)` | `rocdl` TDM async copy | gfx1250 whole-tile Global↔LDS (out of scope on CDNA) |

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

New kernels do **not** compute byte offsets by hand. They wrap a raw pointer in a
**buffer tensor**, which carries the shape/stride (a Layout, §6.2) *and* the
buffer descriptor needed for `BufferCopy`:

```python
A = fx.rocdl.make_buffer_tensor(A_ptr)      # + shape/stride from the traced args
```

From there you use the layout algebra of Chapters 7–7 (`zipped_divide`, `slice`,
`partition_S`) directly on `A`. Raw `buffer_ops.create_buffer_resource()` and
manual offsets still exist for legacy kernels but are discouraged
(`CLAUDE.md` → Kernel Authoring Conventions).

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

Add an LDS staging tier (§10.5 / Chapter 13 / GEMM puzzles) and it becomes
global→LDS then LDS→registers, but the call shape is identical.

## Predication: safe boundaries without branches

When a tile does not evenly divide the tensor (the last block), some threads
would read/write out of bounds. Two defenses:

1. **Buffer atoms** give you hardware bounds checking for free — an OOB
   `buffer_load` returns 0, an OOB `buffer_store` is dropped. Often that is all
   you need.
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

## LDS (shared memory) in one paragraph

To stage through LDS, allocate it with a `SharedAllocator` over a `@fx.struct`
storage layout and copy global→LDS then LDS→registers, with a `gpu.barrier()`
between producers and consumers:

```python
smem = fx.SharedAllocator().allocate(SharedStorage).peek()
fx.copy(copy_atom, thr_copy.partition_S(bA), thr_copy.partition_D(smem_A))
fx.gpu.barrier()                       # all threads: LDS filled
fx.copy(copy_atom, smem_A_part, frag_A)
```

Bank-conflict avoidance is a *layout* concern: you pick a swizzled LDS layout so
lanes hit distinct banks (Chapter 13 and the GEMM puzzles). The copy calls do not
change — only the LDS layout does. Full LDS mechanics, double-buffering, and
swizzle live in `docs/kernel_tuning_guide.md`; the GEMM puzzles build them up
step by step.

A special case — reading a tile **transposed** out of LDS so it lands in
matrix-core operand order — is covered in Chapter 9 (§"Transposed operand
loads"), since it only makes sense in the context of feeding an MFMA.
