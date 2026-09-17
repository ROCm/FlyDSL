# Reference

A working reference to keep open while you write kernels and solve the puzzles.
Authoritative sources are `docs/layout_system_guide.md` (full layout API) and the
env-var registry `python/flydsl/utils/env.py`; this appendix is the curated
subset a C++ GPU programmer reaches for most.

## Environment variables

Use these exact names (from `python/flydsl/utils/env.py`); do not invent
alternate spellings.

| Purpose | Variable | Notes |
|---------|----------|-------|
| Compile backend | `FLYDSL_COMPILE_BACKEND` | default `rocm` |
| Override compile arch | `ARCH` | e.g. `gfx942` |
| Compile without running | `COMPILE_ONLY` | trace + compile, skip launch |
| Dump IR at each stage | `FLYDSL_DUMP_IR=1` | writes the traced/Stage-A/B/final modules |
| IR dump directory | `FLYDSL_DUMP_DIR=/tmp/ir` | pairs with `FLYDSL_DUMP_IR` |
| Print IR after every pass | `FLYDSL_DEBUG_PRINT_AFTER_ALL` | very verbose; pinpoints a failing pass |
| Show AST rewrite (§2.3) | `FLYDSL_DEBUG_AST_DIFF` | see `if/for/while` → dispatch rewrite |
| Line info for profiler | `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` | excludes AOT-only mode |
| JIT cache directory | `FLYDSL_RUNTIME_CACHE_DIR` | on-disk artifact location |
| Enable/disable disk cache | `FLYDSL_RUNTIME_ENABLE_CACHE` | `0`/`false` → in-memory only |
| Disk-cache-only (AOT) | `FLYDSL_RUNTIME_RUN_ONLY=1` | load from disk, error on miss; no JIT |
| External LLVM/MLIR codegen | `FLYDSL_COMPILE_LLVM_DIR` | install prefix; part of cache key |
| GPU arch hints | `FLYDSL_GPU_ARCH`, `HSA_OVERRIDE_GFX_VERSION` | |

**The two you will use constantly while learning:**

```bash
# See what your Python actually built, uncached:
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/ir FLYDSL_RUNTIME_ENABLE_CACHE=0 python kernel.py

# Force a recompile after editing C++ passes / helpers not in the traced closure:
FLYDSL_RUNTIME_ENABLE_CACHE=0 python kernel.py
```

> **Gotcha (§2.7).** The disk cache invalidates on *kernel source and closure*
> changes, but **not** on edits to C++ passes or helper code outside the traced
> closure. When results look stale after such edits, set
> `FLYDSL_RUNTIME_ENABLE_CACHE=0`.

## API cheat-sheet

All `fx.*` calls emit IR and must run inside a `@flyc.kernel` / `@flyc.jit` body.

### Decorators and launch

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel          # device function  -> gpu.func
def k(...): ...

@flyc.jit             # host launcher
def run(..., stream=fx.Stream(None)):
    k(...).launch(grid=(gx,gy,gz), block=(bx,by,bz), stream=stream, smem=nbytes)
```

### Indices

| FlyDSL | HIP |
|--------|-----|
| `fx.thread_idx.x/y/z` | `threadIdx.x/y/z` |
| `fx.block_idx.x/y/z` | `blockIdx.x/y/z` |
| `fx.block_dim.x/...` | `blockDim.x/...` |

### Layout construction / query (Ch. 6)

```python
fx.make_shape(8, 16); fx.make_stride(1, 8)
fx.make_layout((8,16), (1,8)); fx.make_ordered_layout((M,N), order=(1,0))
fx.make_identity_layout((M,N))
fx.size(L); fx.cosize(L); fx.rank(L); fx.get_shape(L); fx.get_stride(L)
fx.crd2idx(coord, L); fx.idx2crd(idx, L); fx.coalesce(L); fx.composition(A,B)
```

### Divide / product (Ch. 6)

```python
fx.zipped_divide(A, (BM,BN)); fx.logical_divide(A, tiler)
fx.tiled_divide(...); fx.flat_divide(...)
fx.slice(bA, (None, bid))            # pick a block's tile
fx.blocked_product(v,t); fx.raked_product(v,t)
```

### Tiling / partitioning (Ch. 7)

```python
tiler_mn, tv = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy = fx.make_tiled_copy(copy_atom, tv, tiler_mn)
tiled_copy = fx.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
tiled_mma  = fx.make_tiled_mma(mma_atom, atom_layout[, perm])
tc_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)   # _B / _C too
thr_copy = tiled_copy.get_slice(tid); thr_mma = tiled_mma.thr_slice(tid)
s = thr_copy.partition_S(t); d = thr_copy.partition_D(t); r = thr_copy.retile(frag)
fA = thr_mma.make_fragment_A(t)  # _B / _C ; frag.fill(0)
fx.make_fragment_like(part[, dtype=fx.Boolean])
```

### Copy atoms and copy (Ch. 8)

```python
fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)          # portable
fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)       # AMD buffer path
A = fx.rocdl.make_buffer_tensor(A_ptr)                         # buffer tensor
fx.copy(copy_atom, src_part, dst_part[, pred=pred])            # tiled copy
fx.copy_atom_call(copy_atom, src, dst)                         # single atom
fx.elem_less(coord_tensor, (M,N))                              # in-bounds mask

fx.rocdl.BufferCopyLDS128b()                        # Global->LDS, compiler waits
fx.rocdl.cdna4.BufferLoadAsyncLDS128b()             # gfx950 async Global->LDS
fx.rocdl.asyncmark(); fx.rocdl.wait_asyncmark(0)    # ... you wait (§8.6)
```

### MMA atoms and gemm (Ch. 9)

```python
fx.make_mma_atom(fx.rocdl.MFMA(16,16,4, fx.Float32))          # CDNA
fx.make_mma_atom(fx.rocdl.WMMA(16,16,128, fx.Float8E4M3FN))   # RDNA/gfx1250
fx.gemm(mma_atom, D, A, B, C)                                 # D = A*B + C
```

### Control flow (Ch. 3)

```python
for i in fx.range_constexpr(N):        ...   # unrolled at trace time
for iv, state in range(0, K, step, init=[acc0]):    # scf.for, loop-carried
    acc = state[0] + work(iv)
    results = yield [acc]              # plain builtin range + plain yield
if fx.const_expr(FLAG): ...            # compile-time branch (no scf.if)
if dyn_cond: ...                       # runtime branch -> scf.if
```

### LDS / shared memory (§8.6)

```python
@fx.struct                                   # one storage type per kernel
class SharedStorage:
    a: fx.Array[fx.Float16, LDS_A_ELEMS, 16]   # flat, element-counted, 16B-aligned
    b: fx.Array[fx.Float16, LDS_B_ELEMS, 16]

alloc = fx.SharedAllocator()                 # exactly ONE per @flyc.kernel
lds   = alloc.allocate(SharedStorage).peek()
sA    = lds.a.view(fx.make_layout((BM, BK), (BK, 1)))     # give it a shape
alloc.allocated_bytes                        # running total (trace time)
# fx.SharedAllocator(static=False)           # dynamic: launch(smem=) auto-inferred
# @fx.union                                  # deliberately overlap two phases
```

### Synchronization (Ch. 12)

```python
fx.barrier()                           # == fx.gpu.barrier(); s_barrier. NOT divergent!
fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0) # hand-written completion wait
fx.rocdl.asyncmark(); fx.rocdl.wait_asyncmark(0)   # async LDS DMA group (§8.6)
fx.memory_fence(ordering=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.AgentOneAs)
# scopes:  fx.SyncScope.System / SingleThread
#          fx.rocdl.SyncScope.Agent / Workgroup / Wavefront (+ ...OneAs variants)
fx.rocdl.sched_barrier(0)              # scheduling hint — NOT synchronization
```

### Reductions (softmax/norm pattern)

```python
x.reduce(fx.ReductionOp.MAX)           # in-fragment (per-thread) reduce
fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=32)     # warp collective (Ch. 13)
fx.coop.warp_scan(x, fx.ReductionOp.ADD)                 # -> (inclusive, exclusive)
br = fx.coop.BlockReduce[fx.Float32, fx.known_block_size()]
st = fx.SharedAllocator().allocate(br.SharedStorage).peek()
total = br(x, fx.ReductionOp.ADD, storage=st)            # block collective
w.shuffle_xor(off, WARP_SIZE)          # raw butterfly step, when coop has no fit
fx.lane_id()                           # lane index within the warp
fmath.exp2(x, fastmath=...)            # from flydsl.expr import math as fmath
```

### Atomics and the memory model (§10.7)

```python
fx.atomic_add(ptr, v)                  # _sub/_min/_max/_and/_or/_xor/_xchg too
fx.atomic_fmax(ptr, x); fx.atomic_fmin(ptr, x)           # float, via int atomics
old, ok = fx.atomic_cas(ptr, expected, desired)          # integers only
fx.memory_fence(ordering=fx.AtomicOrdering.Release)
v = fx.generic_load(ptr, dtype=fx.Float32, count=4)      # ordering-aware load
fx.generic_store(ptr, v, memory_order=fx.AtomicOrdering.Release,
                 syncscope=fx.rocdl.SyncScope.Agent)
fx.copy(fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.Float32), fx.Float32), s, d)
```

### Debug

```python
fx.printf("tid={} val={}", tid, value)
```

## HIP / CK-Tile ↔ FlyDSL glossary

### Kernels, launch, and compile-time parameters

| You know (HIP / CK-Tile / CuTe) | FlyDSL |
|---------------------------------|--------|
| `__global__` function | `@flyc.kernel` (→ `gpu.func`) |
| host launch + `hipLaunchKernelGGL` | `@flyc.jit` + `.launch(grid, block)` |
| `dim3 grid/block` | `grid=(x,y,z)`, `block=(x,y,z)` tuples |
| dynamic LDS bytes arg | `.launch(smem=nbytes)` |
| `threadIdx` / `blockIdx` | `fx.thread_idx` / `fx.block_idx` |
| non-type template param `<int A>` | `A: fx.Constexpr` |
| type template param `<typename T>` | `T: type[fx.Float16]` |
| `#pragma unroll` fixed loop | `fx.range_constexpr(N)` loop |
| runtime `for` loop | `for iv, state in range(..., init=[...])` |
| `hipModuleLoadData` / `LaunchKernel` | `mgpuModuleLoad` / `mgpuLaunchKernel` |
| fat binary (`.hsaco`) | `gpu.binary` blob in the module |

### Layouts, tiling, and distribution

| You know (HIP / CK-Tile / CuTe) | FlyDSL |
|---------------------------------|--------|
| tensor descriptor `(lengths,strides)` | `fx.make_layout(shape, stride)` |
| `descriptor.CalculateOffset` | `fx.crd2idx(coord, layout)` |
| `Sequence<>` / `Tuple<>` extents | IntTuple (`fx.make_shape`, nested) |
| `make_tile_window` + advance | `fx.zipped_divide` + `fx.slice(_, (None,bid))` |
| `tile_distribution` / TV layout | `fx.make_layout_tv` → TV layout |
| coalesced thread mapping | `raked_product` (via TiledCopy) |
| contiguous per-thread mapping | `blocked_product` |
| `load_tile` distributed slice | `thr_copy.partition_S(tile)` |

### Copies and matrix multiply

| You know (HIP / CK-Tile / CuTe) | FlyDSL |
|---------------------------------|--------|
| `buffer_load_dwordx4` (buffer desc) | `rocdl.BufferCopy128b()` atom |
| `global_load` (no descriptor) | `fx.UniversalCopy128b()` atom |
| `buffer_load … lds` | `rocdl.BufferCopyLDS128b()` atom |
| `cp.async` + `wait_group N` | `cdna4.BufferLoadAsyncLDS*` + `asyncmark` |
| `pad_tensor_view` / masked load | `fx.copy(..., pred=...)` |
| operand/accumulator VGPR arrays | `make_fragment_A/B/C` |
| pack VGPRs into MFMA operand order | `thr_copy.retile(frag)` |
| `__builtin_amdgcn_mfma_*` | `fx.rocdl.MFMA(...)` atom + `fx.gemm` |
| `WarpGemmAttribute` / warp tiling | `make_tiled_mma(atom, atom_layout)` |

### Shared memory and synchronization

| You know (HIP / CK-Tile / CuTe) | FlyDSL |
|---------------------------------|--------|
| `__shared__ float buf[N];` | `@fx.struct` field + `SharedAllocator()` |
| `extern __shared__ char buf[];` | `SharedAllocator(static=False)` allocator |
| `alignas(16)` on an LDS member | `fx.Array[dtype, N, 16]` / `fx.Align[T, 16]` |
| `__syncthreads()` | `fx.barrier()` |
| `__builtin_amdgcn_s_waitcnt` | `fx.rocdl.s_waitcnt(vmcnt=, lgkmcnt=)` |
| `__threadfence()` (+ scope) | `fx.memory_fence(ordering=, syncscope=)` |
| `atomicAdd` / `atomicCAS` | `fx.atomic_add` / `fx.atomic_cas` |
| `__shfl_xor` | `w.shuffle_xor(off, WARP_SIZE)` |
| `__lane_id()` | `fx.lane_id()` |
| `cub::BlockReduce` / rocPRIM | `fx.coop.BlockReduce[dtype, block]` |

## Where to go next

- **Solve the puzzles** in `puzzles/` — they reference the sections above and are
  validated on CDNA (gfx942/gfx950). Start at P01 and work forward; the GEMM
  puzzles (P09–P13) grow `examples/03-tiledMma.py` into a pipelined kernel, and
  the application puzzles (P14–P15) build flash attention and a 2-D convolution.
- **Read production kernels** in `kernels/` — `norm/softmax_kernel.py` for the
  reduction pattern, the GEMM/MoE families for the full pipeline.
- **Deepen the algebra** with `docs/cute_layout_algebra_guide.md` (mathematical
  background) and `docs/layout_system_guide.md` (complete API).
- **Learn the type rules** from `docs/language/dsl_protocols.md` (JitArgument /
  DslType / Storable, §4.5) and `docs/language/storage_and_allocator.md`.
- **Use the extension libraries** in `python/flydsl/extension/` — `fx.coop` for
  warp/block collectives (runnable examples in `examples/extension/coop/`).
- **Tune** with `docs/kernel_tuning_guide.md` (LDS swizzle, double-buffering,
  MFMA scheduling, occupancy, ATT/PMC profiling).
