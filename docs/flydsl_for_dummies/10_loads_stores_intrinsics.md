# Loads & Stores, Close to the Metal

Chapter 8 drove data movement with a copy atom over a TiledCopy and left the
instruction implicit — a copy "lowers to `rocdl.buffer_load`." This chapter opens
the atom. It is the level you reach for when you read a dumped `rocdl.*` module, or
when you write `buffer_load_dwordx4` / `ds_read_b128` by hand in HIP and want to
know which FlyDSL construct produces each. Everything here is a deep-dive companion
to Chapter 8; the layout algebra that *feeds* these instructions is unchanged.

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
`make_buffer_tensor`, which keeps the layout attached so the Chapters 7–7 algebra
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

## `fx.ptr_load` / `fx.ptr_store`: the single-element primitive

All of the copy paths above work on tiles and fragments. `fx.ptr_load` and
`fx.ptr_store` work on **individual elements** through a raw typed pointer
(`Pointer`). They are not a distinct hardware path — they are the one-element
primitive that everything else compiles through.

### The pointer type and its origin

A `Pointer` (`python/flydsl/expr/typing.py:877`) is a layout-free typed pointer to
any address space (global, shared, register). You get one from:

- `fx.get_iter(tensor)` — strip the layout from a `Tensor`, returning a pointer to
  its first element (used internally; also reachable from
  `tensor.ptr + offset` patterns).
- `fx.get_dyn_shared(dtype)` — the base of the kernel's dynamic LDS allocation
  (`primitive.py:1144`).
- `fx.recast_iter(dtype, ptr)` — reinterpret a pointer's element type (like
  `reinterpret_cast<T*>`).
- `ptr + offset` — element-count pointer arithmetic, emitting `fly.add_offset`
  (`primitive.py:1188`).

### `ptr_load` and `ptr_store`

```python
# primitive.py:1207
v = fx.ptr_load(ptr)                       # load ptr's element type
v = fx.ptr_load(ptr, result_type=fx.Int64) # load as a specific type/width

fx.ptr_store(value, ptr)                   # store value into ptr
```

`Pointer` objects also expose these as methods: `ptr.load()`, `ptr.store(v)`, and
`ptr[offset]` (load after arithmetic shift). They lower to `fly.ptr_load` /
`fly.ptr_store`, which the `PtrLoadOpLowering` and `PtrStoreOpLowering` patterns in
`FlyToROCDL.cpp:351` turn into:

- `LLVM::LoadOp` / `LLVM::StoreOp` for global (→ `global_load_*`) and shared (→
  `ds_read_*` / `ds_write_*`) address spaces.
- `ROCDL::RawPtrBufferLoadOp` when the address space is `BufferDesc` (→
  `buffer_load_*`).

### ptr_load/ptr_store are how high-level ops compile internally

This is the important structural point. `MemrefLowering.td` rewrites high-level
tensor indexing through `ptr_load`/`ptr_store`:

```
T[coord]  →  fly.memref_load(T, coord)
          →  fly.ptr_load(fly.add_offset(fly.get_iter(T), fly.crd2idx(coord, layout(T))))
          →  llvm.load (→ ISA)
```

So `ptr_load`/`ptr_store` sit at level 2 in the stack, between the layout algebra
(which computes the address) and LLVM:

```
Level 3: fx.copy / fx.copy_atom_call / T[coord]   — tile/fragment or indexed element
Level 2: fly.ptr_load / fly.ptr_store             — pointer + load/store
Level 1: llvm.load / ROCDL buffer load            — address-space-aware instruction
```

### When to use them directly

Reach for `ptr_load`/`ptr_store` when you have a pointer with a hand-computed
byte/element offset and no layout to index through — typically LDS staging in
attention kernels where per-thread addressing is non-standard:

```python
# kernels/attention/pa_decode_swa.py — write packed fp8 into LDS at a lane-computed offset
v01 = fx.Vector.from_elements([q_w0, q_w1], dtype=fx.Int32)
fx.ptr_store(v01, logits_base + lds_q_base)     # logits_base is a Pointer, + shifts it

# read back as a different element type (recast)
q_v1 = fx.ptr_load(
    fx.recast_iter(fx.Int64, logits_base) + lds_rd,
    result_type=fx.Vector.make_type(1, fx.Int64),
)
```

### Worked example: replacing ptr_load/ptr_store with the high-level path

When the access *does* fit a layout — a global tile, an LDS staging buffer with a
regular shape — the copy-atom + TiledCopy path is simpler, portable, and gives the
compiler more to optimize. Here is the *same* kernel written both ways, staging a
tile through LDS (global → LDS → registers → global). Both compile, run, and produce
`B == A`; the complete runnable file is
`examples/06-lds_staging_lowlevel_vs_highlevel.py`.

The problem: one block owns an `8×8 = 64`-element tile, one thread per element. Each
thread copies its element global → LDS, the block barriers, then reads it back
LDS → global.

**Low-level — raw `ptr_load`/`ptr_store` with hand-computed offsets:**

```python
BM, BN = 8, 8   # 64-element tile, 64 threads

@flyc.kernel
def stage_lowlevel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr):
    tid = fx.thread_idx.x                    # 0..63 within the block
    bid = fx.block_idx.x                     # which 8-column block
    r = tid // fx.Int32(BN)                  # row inside the tile
    c = tid % fx.Int32(BN)                   # column inside the tile
    g = r * fx.Int32(N) + (bid * fx.Int32(BN) + c)   # global row-major index

    x = fx.ptr_load(fx.get_iter(A) + g)      # global -> reg  (global_load)
    smem = fx.get_dyn_shared(fx.Float32)     # raw LDS base pointer
    fx.ptr_store(x, smem + tid)              # reg -> LDS     (ds_write_b32)
    fx.gpu.barrier()
    y = fx.ptr_load(smem + tid)              # LDS -> reg     (ds_read_b32)
    fx.ptr_store(y, fx.get_iter(B) + g)      # reg -> global  (global_store)
```

Every address is arithmetic you compute and must keep correct: the `r * N + …`
global index, the `smem + tid` LDS slot. Nothing records that this is a tile.

**High-level — a layouted LDS view driven by `fx.copy`:**

```python
@flyc.kernel
def stage_highlevel(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    bA = fx.slice(fx.zipped_divide(A, (BM, BN)), (None, bid))   # this block's tile
    bB = fx.slice(fx.zipped_divide(B, (BM, BN)), (None, bid))

    smem = fx.get_dyn_shared(fx.Float32)
    sT = fx.make_view(smem, fx.make_layout((BM, BN), (BN, 1)))  # LDS tile, given a layout

    thr_layout = fx.make_layout((BM, BN), (BN, 1))              # tid -> (row, col)
    val_layout = fx.make_layout((1, 1), (1, 1))                 # one element per thread
    tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)

    gcopy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)  # global path
    scopy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)     # LDS path
    tc_g = fx.make_tiled_copy(gcopy, tv, tile_mn).get_slice(tid)
    tc_s = fx.make_tiled_copy(scopy, tv, tile_mn).get_slice(tid)

    frag = fx.make_fragment_like(tc_g.partition_S(bA))
    fx.copy(gcopy, tc_g.partition_S(bA), frag)      # global -> reg
    fx.copy(scopy, frag, tc_s.partition_D(sT))      # reg -> LDS
    fx.gpu.barrier()
    fx.copy(scopy, tc_s.partition_S(sT), frag)      # LDS -> reg
    fx.copy(gcopy, frag, tc_g.partition_D(bB))      # reg -> global
```

The two kernels emit the same `ds_write_b32`/`ds_read_b32` (and the LDS legs go
through `ptr_store`/`ptr_load` internally — the high-level version *is* the low-level
one after lowering). What changed is where the addressing lives:

| | Low-level | High-level |
|---|-----------|------------|
| Global index | manual `r*N + bid*BN + c` | `zipped_divide` + `slice` |
| LDS slot | manual `smem + tid` | `partition_S/D` over `sT`'s layout |
| Thread→element map | implicit in your arithmetic | the `tv` layout object |
| OOB handling | you check it | free via the buffer tensor |
| Bank conflicts | you swizzle by hand | add a swizzle to `sT`'s layout, calls unchanged |
| Change the tile shape | rewrite the index math | change `(BM, BN)` and the layouts |

The high-level version costs more lines here because the tile is trivial; the payoff
grows with the kernel — the layout objects compose with tiling, MMA fragments
(Chapter 11), and swizzle without touching the copy calls.

> **HIP/CK-Tile → FlyDSL.** `fx.ptr_load(ptr)` / `fx.ptr_store(v, ptr)` is the typed
> `*ptr` dereference — the plain load/store you write in HIP against a `__shared__`
> or `__device__` pointer with a hand-computed index. `fx.copy` is CK-Tile's
> `load_tile` / `store_tile`: the same instruction wrapped in a distribution + layout.
> Reach for `ptr_load`/`ptr_store` when the addressing is genuinely irregular; use
> `fx.copy` when it is a regular tile or fragment.

## Registers are memory, too

The destination of a load — and the output of an MFMA — is a **register-space
tensor**. FlyDSL models register files as memrefs in LLVM address space 5. There
are two ways to allocate one; which you use depends on whether you have a tile
layout to derive from.

### Declaring a register tensor for a single MFMA output

When you drive the MFMA yourself (as in the low-level example in Chapter 11), you
need to declare the accumulator VGPR array directly. Use `make_rmem_tensor`:

```python
frag = fx.make_rmem_tensor(N_ELEMENTS, dtype)
```

`N_ELEMENTS` is the number of values **this lane** owns — set by the MFMA operand
ABI. For the shapes this book uses on CDNA (wave64):

| MFMA shape | Acc dtype | Acc elements per lane | Declare as |
|---|---|---|---|
| 16×16×\* | f32 | **4** | `make_rmem_tensor(4, fx.Float32)` |
| 32×32×\* | f32 | **16** | `make_rmem_tensor(16, fx.Float32)` |
| 16×16×\* | i32 | **4** | `make_rmem_tensor(4, fx.Int32)` |

These numbers come from `getMfmaAccVecSize` in `CDNA3/MmaAtom.cpp`: a 16×16 MFMA
has `GroupM = 64/16 = 4` lane-groups each covering 4 rows, so each lane owns
`ValM0 = 4` accumulators; a 32×32 MFMA has `GroupM = 2` × `ValM1 = 4`, giving 16.

Similarly for operands — for `MFMA(16, 16, 16, bf16)` each lane holds 4 bf16
elements of one A or B row, stored as `i16` per the hardware ABI:

```python
# A/B operand fragment (4 bf16 elements per lane, passed as i16 to the intrinsic)
a_frag = fx.make_rmem_tensor(4, fx.BFloat16)  # or Int16 if you bitcast immediately
# Accumulator (4 f32 per lane)
c_frag = fx.make_rmem_tensor(4, fx.Float32)
```

### The lifecycle: zero → gemm → load → use

Allocating a register tensor gives you a name for the VGPR slots; you must fill
them before reading:

```python
c_frag = fx.make_rmem_tensor(4, fx.Float32)
c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))   # zero the accumulators

# ... fill a_frag, b_frag by loading from global/LDS ...
fx.gemm(mma_atom, c_frag, a_frag, b_frag, c_frag)    # result written back to c_frag

result = c_frag.load()    # pull the vector<4xf32> out as an SSA value for use
```

`frag.store(vec)` / `frag.load()` are the register-level equivalents of writing to
and reading from a named `float4` variable in HIP. After the
`fly-promote-regmem-to-vectorssa` pass (Chapter 2 pipeline), every register-space
memref is rewritten into a pure `vector<N×T>` SSA value — the allocas disappear and
what remains is exactly the VGPR you would have declared by hand.

### The high-level alternative: `make_fragment_C/A/B` and `make_fragment_like`

When you use `fx.gemm` over a `TiledMma`, you do not need to know the accumulator
size: `thr_mma.make_fragment_C(bC)` reads the size from the atom's thread-value
layout and allocates the right register tensor automatically. This is what the
high-level MFMA code does:

```python
frag_C = thr_mma.make_fragment_C(bC)   # size inferred from the atom (4 or 16 f32)
frag_C.fill(0)                          # fills the underlying register tensor to 0
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

`make_fragment_like(partition, dtype=None)` allocates a register tensor shaped like
*an existing partition* — used when you have a copy partition and want a matching
register staging buffer. `make_rmem_tensor` is the manual version: reach for it
when you drive the MFMA yourself and need to name the accumulator explicitly.

### N separate MFMA atoms → a list of N register tensors

When you tile MFMA atoms in M and N (a `TiledMma` with atom-layout `(2, 2, 1)`
covers a 32×32 output with four 16×16 atoms), each atom's accumulator is an
*independent* set of 4 VGPRs. The pattern is a Python list of register tensors —
one per atom — built at trace time:

```python
N_TILES_M, N_TILES_N = 2, 2      # 2x2 grid of 16x16 atoms
N_ATOMS = N_TILES_M * N_TILES_N   # = 4

# one accumulator tensor per atom, all zeroed
c_frags = [fx.make_rmem_tensor(4, fx.Float32) for _ in fx.range_constexpr(N_ATOMS)]
for frag in c_frags:
    frag.store(fx.Vector.filled(4, 0.0, fx.Float32))

# issue four MFMAs
for i in fx.range_constexpr(N_TILES_M):
    for j in fx.range_constexpr(N_TILES_N):
        frag_idx = i * N_TILES_N + j
        fx.gemm(mma_atom, c_frags[frag_idx],
                a_frags[i], b_frags[j], c_frags[frag_idx])

# extract all results
results = [c_frags[k].load().ir_value() for k in fx.range_constexpr(N_ATOMS)]
```

This is the exact pattern used in `kernels/gemm/fp8_gemm_utils.py:211`
(`Mfma16x16x128.call`), which manages `n_tiles_a × n_tiles_b` accumulators as a
flat list. The loop body is unrolled at trace time by `range_constexpr`, so the
compiler sees all four `fx.gemm` calls simultaneously and can schedule them with
their loads freely.

> **Gotcha — the accumulator count is lane-local.** You declare `4` (not `16×16`)
> because each of the 64 lanes holds only 4 of the 256 output values. The full
> output tile only exists once all 64 lanes' registers are combined — which is why
> you never index into `c_frag` by row/col; you index the result tensor by matrix
> coordinate *after* storing through the fragment and copying back to global memory.

> **HIP/CK-Tile → FlyDSL.** `make_rmem_tensor(4, fx.Float32)` is `float4 c_frag = {0};`
> — the per-lane accumulator array the ISA manual specifies. A list of N such
> tensors is `float4 c_frag[N];`, and `frag.load()` is reading that array into an
> SSA value the next operation (another MFMA, a store, an epilogue cast) can
> consume.

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

T[coord]  (tensor element access)
  → fly.memref_load(T, coord)   (MemrefLowering.td)
  → fly.ptr_load(add_offset(get_iter(T), crd2idx(coord, layout(T))))
  → llvm.load / rocdl.raw.ptr.buffer.load

fx.ptr_store(v, ptr) / fx.ptr_load(ptr)   (raw pointer, no layout)
  → fly.ptr_store / fly.ptr_load          (PtrStoreOpLowering / PtrLoadOpLowering)
  → llvm.store / llvm.load  →  ds_write_* / ds_read_* / global_store / global_load
```

To *see* it for any kernel, dump the IR after Stage A: `FLYDSL_DUMP_IR=1` (§2.8)
gives you the module right after `convert-fly-to-rocdl`, where these `rocdl.*` ops
are explicit. Diffing the pre/post-Stage-A modules is the fastest way to confirm your
copy became the instruction and width you intended.

The matrix multiply gets the same treatment next: Chapter 11 opens the MMA atom down
to the `rocdl.mfma.*` instruction and the operand/accumulator VGPRs.
