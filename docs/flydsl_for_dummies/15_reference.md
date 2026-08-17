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
| Line info for profiler | `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` | incompatible with `FLYDSL_RUNTIME_RUN_ONLY` |
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
smem = fx.SharedAllocator().allocate(SharedStorage).peek()
fx.gpu.barrier()
```

### Reductions (softmax/norm pattern)

```python
w.shuffle_xor(off, WARP_SIZE)          # warp-level reduce step
x.reduce(ReductionOp.MAX)              # in-fragment reduce
fmath.exp2(x, fastmath=...)            # from flydsl.expr import math as fmath
```

### Debug

```python
fx.printf("tid={} val={}", tid, value)
```

## HIP / CK-Tile ↔ FlyDSL glossary

| You know (HIP / CK-Tile / CuTe) | FlyDSL |
|---------------------------------|--------|
| `__global__` function | `@flyc.kernel` (→ `gpu.func`) |
| host launch + `hipLaunchKernelGGL` | `@flyc.jit` + `.launch(grid,block,stream)` |
| `dim3 grid/block` | `grid=(x,y,z)`, `block=(x,y,z)` tuples |
| dynamic LDS bytes arg | `.launch(smem=nbytes)` |
| `threadIdx` / `blockIdx` | `fx.thread_idx` / `fx.block_idx` |
| non-type template param `<int A>` | `A: fx.Constexpr` |
| type template param `<typename T>` | `T: type[fx.Float16]` |
| `#pragma unroll` fixed loop | `for i in fx.range_constexpr(N)` |
| runtime `for` loop | `for iv, state in range(..., init=[...])` (`scf.for`) |
| tensor descriptor `(lengths,strides)` | `fx.make_layout(shape, stride)` |
| `descriptor.CalculateOffset` | `fx.crd2idx(coord, layout)` |
| `Sequence<>` / `Tuple<>` extents | IntTuple (`fx.make_shape`, nested) |
| `make_tile_window` + advance to block | `fx.zipped_divide` + `fx.slice(_, (None,bid))` |
| `tile_distribution` / TV layout | `fx.make_layout_tv` → TV layout |
| coalesced thread mapping | `raked_product` (via TiledCopy) |
| contiguous per-thread mapping | `blocked_product` |
| `load_tile` distributed slice | `thr_copy.partition_S(tile)` |
| `buffer_load_dwordx4` (buffer desc) | `fx.rocdl.BufferCopy128b()` atom |
| `global_load` (no descriptor) | `fx.UniversalCopy128b()` atom |
| `pad_tensor_view` / masked load | `fx.copy(..., pred=...)` |
| operand/accumulator VGPR arrays | `make_fragment_A/B/C` |
| pack VGPRs into MFMA operand order | `thr_copy.retile(frag)` |
| `__builtin_amdgcn_mfma_*` | `fx.rocdl.MFMA(...)` atom + `fx.gemm` |
| `WarpGemmAttribute` / warp tiling | `make_tiled_mma(atom, atom_layout)` |
| `__syncthreads()` | `fx.gpu.barrier()` |
| `__shfl_xor` | `w.shuffle_xor(off, WARP_SIZE)` |
| `hipModuleLoadData` / `LaunchKernel` | `mgpuModuleLoad` / `mgpuLaunchKernel` (auto) |
| fat binary (`.hsaco`) | `gpu.binary` blob in the compiled module |

## Where to go next

- **Solve the puzzles** in `puzzles/` — they reference the sections above and are
  validated on CDNA (gfx942/gfx950). Start at P01 and work forward; the GEMM
  puzzles (P09–P13) grow `examples/03-tiledMma.py` into a pipelined kernel, and
  the application puzzles (P14–P15) build flash attention and a 2-D convolution.
- **Read production kernels** in `kernels/` — `norm/softmax_kernel.py` for the
  reduction pattern, the GEMM/MoE families for the full pipeline.
- **Deepen the algebra** with `docs/cute_layout_algebra_guide.md` (mathematical
  background) and `docs/layout_system_guide.md` (complete API).
- **Tune** with `docs/kernel_tuning_guide.md` (LDS swizzle, double-buffering,
  MFMA scheduling, occupancy, ATT/PMC profiling).
