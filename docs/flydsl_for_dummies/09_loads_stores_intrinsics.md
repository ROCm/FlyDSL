# Loads & Stores, Close to the Metal

Chapter 7 drove data movement with a copy atom over a TiledCopy and left the
instruction implicit — a copy "lowers to `rocdl.buffer_load`." This chapter opens
the atom. It is the level you reach for when you read a dumped `rocdl.*` module, or
when you write `buffer_load_dwordx4` / `ds_read_b128` by hand in HIP and want to
know which FlyDSL construct produces each. Everything here is a deep-dive companion
to Chapter 7; the layout algebra that *feeds* these instructions is unchanged.

## The copy-op zoo: what an atom actually names

A copy atom is `make_copy_atom(op_type, element_type)`. The *op type* is the
hardware move; here are the ones FlyDSL ships and the single instruction each
becomes.

| Op-type constructor | Address spaces | Lowers to (ISA) |
|---------------------|----------------|-----------------|
| `fx.UniversalCopy128b()/64b/32b` | any → any | `global_load` / `ds_read` / `ds_write` (LLVM picks by space) |
| `fx.rocdl.BufferCopy128b()/64b/32b` | BufferDesc ↔ reg | `buffer_load/store_dwordx{4,2,1}` |
| `fx.rocdl.BufferCopyLDS128b()` | BufferDesc → LDS | `buffer_load_dwordx4 … lds` (direct, no VGPR) |
| `fx.rocdl.LDSReadTrans*` (CDNA4) | LDS → reg | `ds_read_tr*` (transpose load) |
| `fx.rocdl.TDM(...)` (gfx1250) | Global ↔ LDS | `tensor_load_to_lds` (N-D descriptor DMA) |

- **`UniversalCopy`** (`python/flydsl/expr/primitive.py:202`) is target-neutral: it
  emits a plain `llvm.load`/`llvm.store`, and the AMDGPU backend selects
  `global_load_dwordx4` (from global), `ds_read_b128` (from LDS), or `ds_write_b128`
  (to LDS) purely from the pointer's address space. One op type, three possible
  instructions.
- **`BufferCopy`** (`python/flydsl/expr/rocdl/universal.py:57`) is the CDNA
  buffer-descriptor path: `rocdl.raw.ptr.buffer.load/store` →
  `buffer_load/store_dwordx{1,2,4}`. It carries a per-atom `soffset` SGPR state field
  (used to fold a K-loop byte offset into an SGPR) and a `cache_modifier` (0 =
  cached, 2 = non-temporal). Load vs. store is inferred from which side is the buffer
  descriptor.
- **`BufferCopyLDS`** (`rocdl/universal.py:73`) is the direct DRAM→LDS DMA
  (`buffer_load_dwordx4 … lds`) that skips the VGPR round-trip; CDNA3/4 only.
- **`LDSReadTrans*`** (`rocdl/cdna4.py:8`) are the gfx950 `ds_read_tr*`
  transpose-on-load instructions.
- **`TDM`** (`rocdl/cdna5.py:64`) is the gfx1250 tensor-descriptor async DMA; it is a
  whole different mechanism (out of scope on CDNA), noted here only for completeness.

The width in the name is the **access width** of one instruction: `128b` = `dwordx4`
= 4×fp32 = 8×fp16. Wider means fewer instructions and better coalescing, if your
tile and TV layout divide evenly by it.

> **HIP/CK-Tile → FlyDSL.** `BufferCopy128b` ≡ `buffer_load_dwordx4` through a buffer
> resource (V#), with free hardware out-of-bounds handling — the reason CK-Tile
> prefers buffer addressing. `UniversalCopy128b` ≡ a plain `global_load_dwordx4` /
> `flat_load` (no descriptor), or `ds_read_b128`/`ds_write_b128` against LDS.
> `BufferCopyLDS128b` ≡ `buffer_load_dwordx4 … lds`, the DRAM→LDS DMA you would emit
> with `__builtin_amdgcn_raw_buffer_load_lds`.

## Buffer tensors are the V# descriptor

`fx.rocdl.make_buffer_tensor(t)` (`rocdl/universal.py:203`) is what turns an ordinary
global pointer into the hardware buffer resource that `BufferCopy` needs. It wraps
the pointer in a `!fly.ptr<…, BufferDesc>` (LLVM address space 8) carrying the base,
a record count, and format flags; that lowers via `rocdl.make.buffer.rsrc` to the
128-bit SGPR buffer descriptor (base / stride / num-records / flags) — the V#.

Contrast the two global-memory handles:

- A **plain global tensor** (a kernel `fx.Tensor` argument) is an address-space-1
  pointer. Loads go through `llvm.load` → `global_load_dwordx4`. No bounds checking.
- A **buffer tensor** carries the V#. Loads go through `rocdl.raw.ptr.buffer.load` →
  `buffer_load_dwordx4`, which gives hardware OOB (loads past `num_records` return 0,
  stores are dropped) and an SGPR `soffset` path for cheap K-loop stepping.

The legacy `buffer_ops.create_buffer_resource()` builds the same V# by hand and is
discouraged for new kernels (`CLAUDE.md` → Kernel Authoring Conventions); prefer
`make_buffer_tensor`, which keeps the layout attached so the Chapters 5–6 algebra
still applies.

> **HIP/CK-Tile → FlyDSL.** `make_buffer_tensor(t)` is
> `__builtin_amdgcn_make_buffer_rsrc` / CK-Tile's `make_buffer_view` — assembling the
> `s[0:3]` buffer descriptor you otherwise fill field by field.

## One atom at a time: `fx.copy_atom_call`

`fx.copy` loops one atom over an entire TiledCopy distribution. When you manage the
per-thread pointer yourself, `fx.copy_atom_call(atom, src, dst, pred=None)`
(`primitive.py:1052`) issues **exactly one** hardware instruction — no loop, no
thread distribution. This is the primitive `fx.copy` is built from.

```python
# kernels/attention/fused_rope_cache_kernel.py:142 — one load per slice, caller-driven
fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
```

Reach for it when the access pattern is not a clean tiled distribution — a single row
slice, a gather, a hand-indexed fragment — which is common in attention and
normalization kernels.

> **HIP/CK-Tile → FlyDSL.** `copy_atom_call` is writing the single `buffer_load`
> yourself; `fx.copy` is the CK-Tile `load_tile` that loops it over a tile window.

## Registers are memory, too

The destination of a load is usually a **register-space tensor** — a fragment. FlyDSL
models register files as memrefs in address space 5:

- `fx.make_rmem_tensor(shape_or_layout, dtype)` (`derived.py:89`) allocates a raw
  register tensor.
- `fx.make_fragment_like(t[, dtype])` (`primitive.py:536`) allocates one shaped like
  an existing partition — the usual way to get an operand/accumulator fragment.
- `.load()` / `.store(vec)` read and write the fragment as an SSA vector.

These are not real stack allocations. The `fly-promote-regmem-to-vectorssa` pass
(Chapter 2 pipeline) rewrites every register-space memref and its load/stores into
pure `vector<N×T>` SSA values — i.e. VGPRs. After that pass there are no address-5
allocas left.

> **HIP/CK-Tile → FlyDSL.** `make_fragment_like` / `make_rmem_tensor` is declaring the
> VGPR array (`float4 a_frag;`) that a `buffer_load` fills; `.load()`/`.store()` are
> reads and writes of those registers.

## The three-tier flow, instruction by instruction

Staging a tile from DRAM through LDS into registers is three copies. On CDNA3 the
global→LDS leg is **two** instructions (there is no single DRAM→LDS path unless you
use `BufferCopyLDS`): a `buffer_load` into VGPRs, then a `ds_write` into LDS.

```python
# g2s: global -> registers -> LDS   (from examples/04-preshuffle_gemm.py)
fx.copy(buffer_copy_128b, thr_gA, copy_frag_A)          # buffer_load_dwordx4 -> VGPRs
fx.copy(uni_copy_128b, copy_frag_A, thr_sA[..., stage]) # ds_write_b128 -> LDS
fx.gpu.barrier()                                        # s_barrier: LDS is filled

# s2r: LDS -> registers
fx.copy(uni_copy_128b, thr_sA_s2r[..., stage], mma_frag_A)   # ds_read_b128 -> VGPRs
```

`fx.gpu.barrier()` lowers to `s_barrier`; it sits between the producers (writes to
LDS) and consumers (reads from LDS). Bank-conflict avoidance is a *layout* concern —
you pick a swizzled LDS layout so lanes hit distinct banks — and does not change any
of these calls; see `docs/kernel_tuning_guide.md` and the GEMM puzzles for the LDS
double-buffer/swizzle mechanics.

> **HIP/CK-Tile → FlyDSL.** This is the `buffer_load` → `ds_write` → `s_barrier` →
> `ds_read` sequence of a hand-written GEMM prologue. `BufferCopyLDS128b` collapses
> the first two into one `buffer_load … lds`, exactly as `__builtin_amdgcn_raw_buffer_
> load_lds` does.

## Atomics and waits, briefly

For reductions into global memory, FlyDSL exposes buffer atomics as copy-atom
factories: `fx.rocdl.BufferAtomicAdd/Max/Min(dtype)` (`rocdl/universal.py:90`) lower
to `buffer_atomic_add_f32` / `_max` / `_min`; `fx.rocdl.UniversalAtomic(op, dtype)`
lowers to a target-neutral `flat`/`ds` atomic. Ordering and visibility come from
`fx.gpu.barrier()` (`s_barrier`) and the `s_waitcnt` machinery the compiler inserts.

> **HIP/CK-Tile → FlyDSL.** `BufferAtomicAdd(fx.Float32)` ≡ `buffer_atomic_add_f32` /
> `atomicAdd` on a buffer resource.

## The lowering ladder

Putting it together, each copy op type is one step from Python to ISA:

```
fx.copy(BufferCopy128b(), gA_buf, frag)
  → fly.copy_atom_call        (CopyAtomCallLowering, FlyToROCDL.cpp:541)
  → rocdl.raw.ptr.buffer.load (CDNA3/CopyAtom.cpp:68)
  → buffer_load_dwordx4 v[0:3], v_off, s[0:3], s_soff offen

fx.copy(UniversalCopy128b(), frag, lds_view)   → llvm.store → ds_write_b128
fx.copy(UniversalCopy128b(), lds_view, frag)   → llvm.load  → ds_read_b128
fx.copy(BufferCopyLDS128b(), gA_buf, lds_view) → rocdl.raw.ptr.buffer.load.lds
                                                → buffer_load_dwordx4 … lds
```

To *see* it for any kernel, dump the IR after Stage A: `FLYDSL_DUMP_IR=1` (§2.8)
gives you the module right after `convert-fly-to-rocdl`, where these `rocdl.*` ops
are explicit. Diffing the pre/post-Stage-A modules is the fastest way to confirm your
copy became the instruction and width you intended.

The matrix multiply gets the same treatment next: Chapter 10 opens the MMA atom down
to the `rocdl.mfma.*` instruction and the operand/accumulator VGPRs.
