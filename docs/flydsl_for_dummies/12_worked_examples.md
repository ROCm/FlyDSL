# Worked Examples

Three complete kernels from `examples/`, read line by line. They escalate from a
target-neutral elementwise add, through a 2-D tiled copy, to a single-block MFMA
GEMM — exercising every concept in Chapters 5–8. Run any of them directly:

```bash
python examples/01-vectorAdd.py     # prints PASS
python examples/02-tiledCopy.py     # prints Result correct: True
python examples/03-tiledMma.py      # prints Result correct: True
```

## Vectorized, predicated add (`01-vectorAdd.py`)

Target-neutral (`UniversalCopy`, no `rocdl.*`), so it runs on any backend. It adds
two `M×N` fp32 matrices with float4 vectorization and boundary predication.

### The launcher

```python
@flyc.jit
def vector_add(A, B, C, stream=fx.Stream(None)):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),   # 8x16 = 128 threads
        fx.make_ordered_layout((1, 4), order=(0, 1)),    # 1x4 values/thread -> float4
    )
    tile_m, tile_n = tiled_copy.tile_mn.unpack()          # (8, 64)
    M, N = A.shape.unpack()
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    vector_add_kernel(A, B, C, tiled_copy).launch(
        grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream)
```

- `make_tiled_copy_tv(atom, thr_layout, val_layout)` builds a TiledCopy (§6.2)
  directly from a thread layout (128 threads arranged 8×16) and a value layout
  (each thread owns 1×4 = a `float4`). Because the value layout's fast axis is N
  and the width is 4×fp32 = 128 bits, each thread issues one coalesced float4 —
  matching `UniversalCopy128b`.
- `tile_mn` is the tile one block covers: 8 rows × (16×4)=64 cols.
- The grid is a **ceil-div**, so `M,N` need not be multiples of the tile — the
  last blocks over-hang and rely on predication (below). Here `M,N = 100,1000`.
- The whole TiledCopy object is *passed as a kernel argument*. It is a
  compile-time (trace-time) value, so it specializes the kernel like a template
  parameter.

### The kernel

```python
@flyc.kernel
def vector_add_kernel(A, B, C, tiled_copy: fx.TiledCopy):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))   # coord tensor
    TileMN = tiled_copy.tile_mn
```

`idC` is an **identity (coordinate) tensor**: reading position `(m,n)` yields the
value `(m,n)`. It costs no memory; it exists so we can compute an in-bounds mask
by comparing coordinates against `(M,N)`.

```python
    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]
```

`flat_divide(A, TileMN)` tiles the whole tensor (§5.3); indexing
`[None, None, bid_x, bid_y]` keeps both intra-tile modes (`None`) and fixes the
block coordinate — this block's tile, exactly like `zipped_divide`+`slice` but
spelled with fancy indexing. `cC` is the same tiling applied to the coordinate
tensor, so `cC` holds the global `(m,n)` of every element in this block's tile.

```python
    thr_copy = tiled_copy.get_slice(tid)
    thr_gA = thr_copy.partition_S(gA)          # this thread's source elems (A)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)          # this thread's dest slots (C)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]
```

`partition_S/D` (§6.3) hand each thread its float4 of A, B, C, and its four
coordinates (`thr_cC`).

```python
    thr_rA = fx.make_fragment_like(thr_gA)     # register tensors matching the
    thr_rB = fx.make_fragment_like(thr_gB)     #   per-thread partition shape
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)   # predicate regs

    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))            # in-bounds mask
```

`make_fragment_like` allocates registers shaped like a partition (§6.4). The
unrolled `range_constexpr` loop (§1.3) fills a boolean predicate: value `a` is
kept iff its coordinate is `< (M, N)`.

```python
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)    # gmem -> regs, masked
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)
    thr_rC.store(thr_rA.load() + thr_rB.load())         # the actual add, in regs
    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)    # regs -> gmem, masked
```

The three `fx.copy` calls (§7.3) are gated by `pred=thr_pC` (§7.4) so overhang
threads never touch OOB memory. The arithmetic is `.load()`/`.store()` on
register fragments — `thr_rA.load()` yields a vector value, `+` emits
`arith.addf`, `.store()` writes it back. **This is the entire "compute": load
regs, add, store regs.** Everything else is layout and safety.

## Two-dimensional tiled copy (`02-tiledCopy.py`)

Copies an `M×N = 24×120` fp32 matrix tile-by-tile using the **buffer** copy path.
It is the minimal `make_layout_tv` → `make_tiled_copy` → `partition` → `copy`
skeleton (§7.5).

```python
@flyc.kernel
def copy_kernel(A, B):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    block_m, block_n = 8, 24

    A = fx.rocdl.make_buffer_tensor(A)          # buffer tensors (§7.2)
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.slice(fx.zipped_divide(A, (block_m, block_n)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (block_m, block_n)), (None, bid))
```

`zipped_divide` + `slice(_, (None, bid))` (§5.3.2) selects this block's 8×24 tile
from A and B.

```python
    thr_layout = fx.make_layout((4, 1), (1, 1))     # 4 threads down M
    val_layout = fx.make_layout((1, 8), (1, 1))     # each owns 1x8 along N
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    partition_src = thr_copy.partition_S(bA)
    partition_dst = thr_copy.partition_D(bB)
    frag = fx.make_fragment_like(partition_src)

    fx.copy(copy_atom, partition_src, frag)          # gmem A -> regs
    fx.copy(copy_atom, frag, partition_dst)          # regs -> gmem B
```

Note the launch is `block=(4,1,1)`, `grid=(15,1,1)` — 4 threads per block (the
thread layout has 4 threads), 15 blocks tiling the 24×120 matrix into 8×24 tiles
(3×5 = 15). The copy goes global→registers→global; add an LDS tier and a barrier
and this becomes the staging half of a GEMM (§7.6). No predication is needed
because the shape divides evenly and the buffer atom would bounds-check anyway.

## Single-block MFMA GEMM (`03-tiledMma.py`)

Computes `C = A @ B.T` for one `64×64` output tile with `K=8`, on one 256-thread
block. This is the seed the GEMM puzzles grow into a full pipeline.

```python
block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A, B, C):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)      # (M,K)
    B = fx.rocdl.make_buffer_tensor(B)      # (N,K)  -> B.T inside gemm
    C = fx.rocdl.make_buffer_tensor(C)      # (M,N)

    bA = fx.slice(fx.zipped_divide(A, (block_m, block_k)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, (block_n, block_k)), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, (block_m, block_n)), (None, bid))
```

B is stored as `(N, K)`, so partitioning it by `(block_n, block_k)` and feeding it
as the B operand yields `A @ B.T` — the common weight-matrix convention.

```python
    mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma   = tiled_mma.thr_slice(tid)
```

One `16×16×4` MFMA atom (§8.1), tiled `2×2×1` (§8.2) → a 32×32 warp tile; with 256
threads (4 waves) this covers the 64×64 block. `thr_slice(tid)` narrows to this
thread's MMA role.

```python
    copy_atom    = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)   # derive copies
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)   #   from the MMA
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)   #   (§6.2)
    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)
```

`make_tiled_copy_A/B/C` derive copies whose distribution matches what the MFMA
consumes/produces — so loaded data lands in operand order without hand-tuning.

```python
    copy_src_A = thr_copy_A.partition_S(bA)      # gmem source partitions
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)         # operand/accumulator fragments
    frag_B = thr_mma.make_fragment_B(bB)         #   (§6.4, §8.3)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)      # reconcile copy <-> MFMA order
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)
```

`retile` (§6.4) is the key correctness step: it maps the copy's TV layout onto the
fragment's MFMA-operand TV layout so `buffer_load` writes each byte where the
matrix core reads it.

```python
    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)   # gmem -> A frag
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)   # gmem -> B frag

    frag_C.fill(0)                                            # zero accumulator
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)         # D = A*B + C
    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)    # C frag -> gmem
```

The compute is one `fx.gemm` (§8.3), lowering to `rocdl.mfma.*`. Because `K=8`
here fits in a single load, there is no K-loop — the GEMM puzzles add the
accumulate-over-K loop (§8.4), LDS staging, double-buffering, and swizzle to turn
this into an efficient kernel.

Read these three side by side and the pattern is unmistakable: **describe layouts
and distributions, then issue one copy/gemm per action.** With that internalized,
Chapter 13 (debugging), Chapter 14's reference tables, and the puzzles are all you
need.
