# The Fly Dialect IV — MMA Atoms

This chapter is about the matrix cores. On CDNA (MI300X/MI350) that means the
**MFMA** instructions; you know them as `__builtin_amdgcn_mfma_*`. In FlyDSL an
MFMA is an **MMA atom**, replicated by a **TiledMma** (Chapter 7) and executed by
`fx.gemm`. This chapter defines the atom, the operand fragment contract, and how
a K-loop accumulates.

## The MMA atom

An **MMA atom** names one matrix instruction and its accumulation type:

```python
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
```

`MFMA(M, N, K, acc_type)` selects a specific hardware instruction shape:

- `(16, 16, 4, f32)` — a `16×16×4` MFMA accumulating in fp32. One instruction, one
  wavefront (64 lanes), computes a 16×16 output tile from a 16×4 (A) and 4×16 (B)
  input over K=4.
- Larger shapes exist (`32×32×8`, `16×16×32` for fp8, …); the shape you pick
  trades register pressure against instruction count. The input element type is
  inferred from the fragments you feed it; the number is the fp32/fp16/fp8 the
  hardware supports.

The atom knows three thread-value layouts — `layout_A_tv`, `layout_B_tv`,
`layout_C_tv` — which define exactly how the 64 lanes' registers map to the A, B,
and C matrix elements. **This is the ABI of the matrix core**, and it is why you
cannot just load A and B naively: the bytes have to be in the lane/register
positions the MFMA reads.

> **HIP/CK-Tile → FlyDSL.** `MFMA(16,16,4,f32)` ≡ picking
> `__builtin_amdgcn_mfma_f32_16x16x4f32`. Its `layout_A_tv/B_tv/C_tv` are the
> register-to-matrix mappings you normally learn from the ISA docs and encode by
> hand when packing operand VGPRs. In CK-Tile these live inside the
> `WarpGemmAttribute` / `mfma_instr` traits; here they are queryable properties of
> the atom.

## TiledMma: many waves make a block tile

One atom computes a 16×16 output — one wavefront, 64 lanes, 4 f32 accumulators per
lane. To cover a larger tile you replicate the atom with `make_tiled_mma` and an
**atom layout** describing the grid of atoms in the MNK dimensions. Because one atom
occupies one whole wave, that grid is simultaneously a grid of **waves**:

```python
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
```

`(2, 2, 1)` is the shape of the atom grid: **2 atoms along M, 2 along N, 1 along K**.
Four atoms in total, so the tile this TiledMma covers is **32×32×4**.

### Where the four atoms execute — one atom per wave

An MFMA is a *wavefront* instruction: all 64 lanes of one wave participate in a single
atom, and no lane can be in two atoms at once. So **a grid of four atoms is a grid of
four waves.** The atom layout is not an unroll factor for one wave — it is the rule
that cuts the block's threads into waves and hands each wave its `(m, n, k)` atom.

That rule is one line in the compiler. FlyDSL takes the atom's *own* thread layout —
`64:1` for a CDNA MFMA (`CDNA3/MmaAtom.cpp:44`) — and `tiled_product`s it with your
atom layout to get the **VMNK thread layout** (`TiledOpTraits.cpp:202`,
`LayoutLowering.cpp:1987`):

```
thr_layout_vmnk = tiled_product( 64:1 , (2,2,1):(1,2,0) )
                = (64, 2, 2, 1) : (1, 64, 128, 0)
                    │   │  │  │
                    │   │  │  └── K atom index
                    │   │  └───── N atom index
                    │   └──────── M atom index
                    └──────────── lane within the atom
```

Its size *is* the block's thread count: **64 × 2 × 2 × 1 = 256 threads = 4 waves**.
Launch this TiledMma with fewer threads and the missing waves simply never compute
their atoms. A thread id is decomposed back through that layout (`idx2crd`,
`LayoutLowering.cpp:1997`) into `(v, m, n, k)`:

```
 tid ──idx2crd──▶ (v, m, n, k)     shape (64,2,2,1)  stride (1,64,128,0)

   v = tid  %  64        lane inside the wave   (0..63)
   m = (tid / 64)  % 2   atom index along M     (0..1)
   n = (tid / 128) % 2   atom index along N     (0..1)
   k = 0                 atom index along K     (only one here)

   tid   0 .. 63    wave 0    atom (m=0, n=0)
   tid  64 ..127    wave 1    atom (m=1, n=0)
   tid 128 ..191    wave 2    atom (m=0, n=1)
   tid 192 ..255    wave 3    atom (m=1, n=1)
```

The stride `(1, 2, 0)` is what produced that assignment: the wave id is
`m*1 + n*2`, so **M is the fastest-varying axis** — consecutive waves step along M.
The stride is a wave↔atom *placement* decision, not an instruction-ordering one.

Laying the four waves over the 32×32 output tile:

```
  stride (1, 2, 0) — M fastest

                          N = 32
              0              16              32
            0 ┌───────────────┬───────────────┐
              │    wave 0     │    wave 2     │
              │  (m=0, n=0)   │  (m=0, n=1)   │
              │ C[ 0:16, 0:16]│ C[ 0:16,16:32]│
     M = 32  16├───────────────┼───────────────┤
              │    wave 1     │    wave 3     │
              │  (m=1, n=0)   │  (m=1, n=1)   │
              │ C[16:32, 0:16]│ C[16:32,16:32]│
           32 └───────────────┴───────────────┘
```

All four atoms are issued *at the same time* on four different waves. What each one
computes against the single K-column (`K_atom = 4`):

```
  stride (1, 2, 0) — M fastest

  wave 0 / atom (m=0,n=0): A[ 0:16, 0:4] × B[ 0:16, 0:4]ᵀ → C[ 0:16,  0:16]
  wave 1 / atom (m=1,n=0): A[16:32, 0:4] × B[ 0:16, 0:4]ᵀ → C[16:32,  0:16]
  wave 2 / atom (m=0,n=1): A[ 0:16, 0:4] × B[16:32, 0:4]ᵀ → C[ 0:16, 16:32]
  wave 3 / atom (m=1,n=1): A[16:32, 0:4] × B[16:32, 0:4]ᵀ → C[16:32, 16:32]
```

A is **32×K** (M-rows × K-cols), B is **32×K** stored **N-first** (so the MFMA sees
`A × Bᵀ`), and C is **32×32**.

#### Effect of changing the stride to `(2, 1, 0)`

Swapping the two non-zero stride values — `(1, 2, 0)` → `(2, 1, 0)` — makes **N the
fastest-varying axis**. The VMNK thread layout becomes:

```
  thr_layout_vmnk = (64, 2, 2, 1) : (1, 128, 64, 0)
                                          ↑    ↑
                                     M stride  N stride swapped vs (1,2,0)

    v = tid % 64           lane within the wave     (unchanged)
    m = (tid / 128) % 2    atom index along M        ← was tid/64
    n = (tid / 64)  % 2    atom index along N        ← was tid/128
```

Waves 1 and 2 swap their atom assignments; waves 0 and 3 are unchanged (they sit on
the diagonal where both formulas agree):

```
  stride (1, 2, 0)               stride (2, 1, 0)
  ─────────────────────────────  ─────────────────────────────
  tid   0.. 63  wave 0 (m=0,n=0) tid   0.. 63  wave 0 (m=0,n=0)
  tid  64..127  wave 1 (m=1,n=0) tid  64..127  wave 1 (m=0,n=1) ← swapped
  tid 128..191  wave 2 (m=0,n=1) tid 128..191  wave 2 (m=1,n=0) ← swapped
  tid 192..255  wave 3 (m=1,n=1) tid 192..255  wave 3 (m=1,n=1)
```

The C grid and the A × B → C table for `(2, 1, 0)`:

```
  stride (2, 1, 0) — N fastest

                          N = 32
              0              16              32
            0 ┌───────────────┬───────────────┐
              │    wave 0     │    wave 1     │
              │  (m=0, n=0)   │  (m=0, n=1)   │
              │ C[ 0:16, 0:16]│ C[ 0:16,16:32]│
     M = 32  16├───────────────┼───────────────┤
              │    wave 2     │    wave 3     │
              │  (m=1, n=0)   │  (m=1, n=1)   │
              │ C[16:32, 0:16]│ C[16:32,16:32]│
           32 └───────────────┴───────────────┘

  wave 0 / atom (m=0,n=0): A[ 0:16, 0:4] × B[ 0:16, 0:4]ᵀ → C[ 0:16,  0:16]
  wave 1 / atom (m=0,n=1): A[ 0:16, 0:4] × B[16:32, 0:4]ᵀ → C[ 0:16, 16:32]
  wave 2 / atom (m=1,n=0): A[16:32, 0:4] × B[ 0:16, 0:4]ᵀ → C[16:32,  0:16]
  wave 3 / atom (m=1,n=1): A[16:32, 0:4] × B[16:32, 0:4]ᵀ → C[16:32, 16:32]
```

**What changes:** which pairs of waves share an operand fragment.

```
                   A shared by            B shared by
  stride (1,2,0)   {w0,w2}, {w1,w3}       {w0,w1}, {w2,w3}
  stride (2,1,0)   {w0,w1}, {w2,w3}       {w0,w2}, {w1,w3}
```

With `(1, 2, 0)` (M fastest), waves in the same **column** of the grid share A, and
waves in the same **row** share B. With `(2, 1, 0)` (N fastest) those roles swap:
same-**row** waves share A, same-**column** waves share B.

**What does not change:** the kernel output, the thread count (still 256), the four
`(atom, A-slice, B-slice, C-slice)` tuples, and how `make_tiled_copy_A/B/C` derives
its own layout. Because the copy layouts are built from the same `tiled_mma` object,
they follow whatever stride you chose automatically — no retile or load pattern needs
to be hand-adjusted.

> **HIP/CK-Tile → FlyDSL.** The atom layout is CK-Tile's *warp tiling* — how many
> waves the workgroup spends along M, N and K. `(2, 2, 1)` is a 2×2 wave grid, the
> `MWarp=2, NWarp=2` of a `BlockGemmShape`, not `MRepeat`/`NRepeat`. Per-wave
> repetition is a separate thing in FlyDSL too, and it comes from the fragment's
> *rest* modes — see "When the block tile is larger than the wave tile" below.

### How A, B and C are distributed over the waves

The wave grid above tells you where C goes. A and B follow a different — and more
interesting — rule: **each operand only looks at the atom coordinates it actually
depends on, and is broadcast along the one it does not.**

`fx.gemm`'s A operand does not depend on N, its B operand does not depend on M, and
its C operand does not depend on K. FlyDSL encodes that literally: when it slices the
thread's operand view it picks two of the four VMNK coordinates and drops the third
(`LayoutLowering.cpp:2008-2024`), and it builds the M/N thread mode with a **stride of
0** on the unused axis (`TiledOpTraits.cpp:126-144`). A stride of 0 is the layout
algebra's spelling of "broadcast".

| Operand | Wave coords used | Wave coord ignored | Consequence |
|---------|------------------|--------------------|-------------|
| A       | `(m, k)`         | `n` (stride 0)     | Waves in the same atom-*row* get the **same** A fragment |
| B       | `(n, k)`         | `m` (stride 0)     | Waves in the same atom-*column* get the **same** B fragment |
| C / D   | `(m, n)`         | `k` (stride 0)     | Every wave owns a **disjoint** slice of C |

Drawn over the 32×32×4 wave tile of `make_layout((2, 2, 1), (1, 2, 0))`:

```
                         B  (N=32, K=4)  — stored N-first, MFMA sees Bᵀ
                    ┌────────────────┬────────────────┐
                    │ B[ 0:16, 0:4]  │ B[16:32, 0:4]  │
                    │  n = 0         │  n = 1         │
                    └───────┬────────┴───────┬────────┘
                            │                │
                    broadcast over m   broadcast over m
                            ▼                ▼
  A  (M=32, K=4)     ┌────────────────┬────────────────┐
 ┌────────────────┐  │                │                │
 │ A[ 0:16, 0:4]  │─▶│     wave 0     │     wave 2     │
 │  m = 0         │  │ C[ 0:16, 0:16] │ C[ 0:16,16:32] │
 ├────────────────┤  ├────────────────┼────────────────┤
 │ A[16:32, 0:4]  │─▶│     wave 1     │     wave 3     │
 │  m = 1         │  │ C[16:32, 0:16] │ C[16:32,16:32] │
 └────────────────┘  └────────────────┴────────────────┘
   broadcast over n            C  (M=32, N=32)
                        disjoint — no cross-wave reduction
```

Reading it off:

- **A** is selected by `m` only. Waves 0 and 2 (both `m=0`) hold *identical* A
  registers; waves 1 and 3 (both `m=1`) hold identical A registers. So the whole A
  tile is fetched **`N_atoms` = 2 times** by the block.
- **B** is selected by `n` only. Waves 0 and 1 (both `n=0`) share B; waves 2 and 3
  (both `n=1`) share B. The whole B tile is fetched **`M_atoms` = 2 times**.
- **C** is selected by `(m, n)`, which is a bijection onto the wave grid. Each wave
  accumulates into registers no other wave touches, so there is **no reduction and no
  barrier** at the end of the MMA — `frag_C` goes straight to the epilogue.

That duplicated A/B traffic is not a bug, it is the reuse trade the wave grid buys: 2×
the operand reads for 4× the output area. It is also exactly why real GEMMs stage A
and B through LDS — the duplicate reader hits shared memory, not HBM (Chapter 13).

The K axis of the atom layout is the one case that *would* need a reduction: a `K`
extent above 1 splits the contraction across waves, leaving each with a partial sum.
Every atom layout in this book, the examples, and the production kernels uses `K=1`
(`(2,2,1)` or `(1,4,1)`), so C stays disjoint.

### Inside one wave: which lane holds which element

Zoom into a single atom. For `MFMA(16,16,4,f32)` the atom's thread-value layouts
(`CDNA3/MmaAtom.cpp:17` for A/B, `:63` for C) work out to one f32 per lane for each
operand and four f32 per lane for the accumulator:

```
A operand — the atom's 16x4 tile, 1 f32 per lane      m = l % 16,  k = l / 16

          k=0      k=1      k=2      k=3
   m= 0 │ lane 0   lane16   lane32   lane48
   m= 1 │ lane 1   lane17   lane33   lane49
    ... │  ...      ...      ...      ...
   m=15 │ lane15   lane31   lane47   lane63

B operand — identical map with n in place of m        n = l % 16,  k = l / 16
```

```
C accumulator — the atom's 16x16 tile, 4 f32 per lane (v = 0..3)
                                        n = l % 16,  m = 4*(l/16) + v

          n=0      n=1      n=2    ...   n=15
   m= 0 │ l0  v0   l1  v0   l2  v0        l15 v0
   m= 1 │ l0  v1   l1  v1   l2  v1        l15 v1
   m= 2 │ l0  v2   l1  v2   l2  v2        l15 v2
   m= 3 │ l0  v3   l1  v3   l2  v3        l15 v3
   m= 4 │ l16 v0   l17 v0   l18 v0        l31 v0
    ... │  ...
   m=15 │ l48 v3   l49 v3   l50 v3        l63 v3
```

`l` here is the *lane* — the `v` coordinate of the VMNK decomposition, i.e.
`tid % 64` — not the block-wide thread id. This is the layer `make_fragment_A/B/C`
and `retile` implement for you; Chapter 11 opens it up further.

### When the block tile is larger than the wave tile

The atom layout sets the **wave tile**, and only that. Its size is
`atom_shape * atom_layout_shape` per mode (`TiledOpTraits.cpp:174`):

```
MFMA(16,16,4) × atom layout (2,2,1)  ->  tile_MNK = (32, 32, 4)
```

`examples/03-tiledMma.py` uses this TiledMma on a `64×64×8` block tile. The extra
factor does *not* come from the atom layout — it appears as **rest modes** on the
fragments. Partitioning a `(64, 64)` C tile with a `(32, 32)` wave tile leaves
`(2, 2)`, so `frag_C` becomes rank-3 `(val, rest_m, rest_n)` and `fx.gemm` expands
into one MFMA per rest coordinate (`ExpandGemmOpLowering`,
`LayoutLowering.cpp:2275`). *This* is the per-wave sequential repetition:

```
block tile 64x64  /  wave tile 32x32  ->  each wave owns 4 interleaved 16x16 blocks

          N: 0    16    32    48    64
        M: 0 ┌─────┬─────┬─────┬─────┐
             │ w0  │ w2  │ w0  │ w2  │
          16 ├─────┼─────┼─────┼─────┤
             │ w1  │ w3  │ w1  │ w3  │
          32 ├─────┼─────┼─────┼─────┤
             │ w0  │ w2  │ w0  │ w2  │
          48 ├─────┼─────┼─────┼─────┤
             │ w1  │ w3  │ w1  │ w3  │
          64 └─────┴─────┴─────┴─────┘

   The pattern is interleaved, not four contiguous quadrants: the rest mode
   steps by the *wave tile* (32), the atom coordinate offsets by 16 inside it.
```

So for `examples/03-tiledMma.py`: 4 waves × (2 rest_m × 2 rest_n × 2 rest_k) = 32 MFMA
instructions, which is exactly `(64/16)·(64/16)·(8/4)`. Per lane, `frag_C` holds
`4 × 2 × 2 = 16` f32 — the 4096 elements of the 64×64 tile spread over 256 threads.

**Two knobs, two different effects.** Growing the atom layout adds *waves*; growing
the block tile relative to the wave tile adds *instructions per wave*. Mixing them up
is the usual source of "my kernel computes a quarter of the output" bugs.

### The production pattern: split N, share A

Real CDNA GEMMs rarely use a square wave grid. `kernels/gemm/preshuffle_gemm.py:699`
and `examples/04-preshuffle_gemm.py:176` both use:

```python
tiled_mma = fx.make_tiled_mma(
    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16)),
    fx.make_layout((1, 4, 1), (0, 1, 0)),          # 4 waves, all along N
    fx.make_tile(None, None, k_perm),              # optional: re-tile the K mode
)
```

`thr_layout_vmnk = (64, 1, 4, 1) : (1, 0, 64, 0)` — still 256 threads, but now the
`m` coordinate has extent 1, so it is 0 for every wave:

```
                         N = 64
        ┌─────────┬─────────┬─────────┬─────────┐
  M=16  │  wave 0 │  wave 1 │  wave 2 │  wave 3 │
        └─────────┴─────────┴─────────┴─────────┘
                       ▲
        all four waves read the SAME A rows [0:16]
        each reads its own 16 N-rows of B
```

Every wave holds the same A fragment (maximum A reuse, one A read per block) and a
private quarter of B. C is still disjoint. When you see `(1, W, 1)` in a production
kernel, that is "W waves split along N" — the default warp tiling on CDNA.

The optional third argument, `permutation`, does not touch the wave count. It replaces
the tile extent in one mode with a layout of your choosing, re-ordering which elements
land in which atom step — `preshuffle_gemm` uses it on K to match a pre-shuffled
operand in memory.

> **Seeing it for yourself.** `fx.utils.print_typst(tiled_mma, file=...)` dumps a
> Typst source file that renders the real thread/value maps for a TiledMma, TiledCopy,
> MMA atom, or layout — `examples/utils/print_typst.py` is a ready-made driver. When a
> layout question is not answered by this chapter, render it.

### Worked example: 32×32 output, K=128, accumulated over 32 K-tiles

With `MFMA(16,16,4,f32)` the K-depth per atom call is 4. To compute a full
`A(32×128) × B(32×128)^T → C(32×32)` in one block, the K-loop runs **32 iterations**
(128/4). The block tile here is `32×32×4` — *exactly* the wave tile of the 2×2 atom
grid — so there are no rest modes: each iteration loads a `(32, 4)` slice of A and B
and calls `fx.gemm` once, which issues **one** MFMA per wave (four in the block, one
each on waves 0–3, in parallel), accumulating into the *same* `frag_C`. Launch it with
`block=(256, 1, 1)`; the atom layout fixed that thread count.

```python
BM, BN, BK_atom = 32, 32, 4   # tile sizes; BK_atom = K-depth of one atom call
K = 128                         # total K dimension
num_k_tiles = K // BK_atom     # = 32 K-tiles to accumulate

@flyc.kernel
def gemm_k128(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)  # (32, 128)  row-major
    B = fx.rocdl.make_buffer_tensor(B)  # (32, 128)  N-first, so B^T is (128, 32)
    C = fx.rocdl.make_buffer_tensor(C)  # (32, 32)

    # Divide into K-tiles of depth BK_atom=4, then select this block's MN tile.
    # After zipped_divide, bA has shape ((BM, BK_atom), num_k_tiles).
    bA_tiles = fx.slice(fx.zipped_divide(A, (BM, BK_atom)), (None, None))
    bB_tiles = fx.slice(fx.zipped_divide(B, (BN, BK_atom)), (None, None))
    bC       = fx.slice(fx.zipped_divide(C, (BM, BN)),      (None, 0))

    mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma   = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcA = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
    tcC = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(tid)

    # Allocate one (32×32) accumulator fragment and zero it once outside the loop.
    # 32*32 f32 over 256 threads = 4 f32 per lane (one atom's worth, since the
    # block tile equals the wave tile here).
    frag_C = thr_mma.make_fragment_C(bC)
    frag_C.fill(0)

    # K-loop: accumulate 32 K-tiles of depth 4 into the same frag_C.
    for k in fx.range_constexpr(num_k_tiles):
        kA = fx.slice(bA_tiles, (None, k))  # (BM, BK_atom) = (32, 4)
        kB = fx.slice(bB_tiles, (None, k))  # (BN, BK_atom) = (32, 4)

        frag_A = thr_mma.make_fragment_A(kA)
        frag_B = thr_mma.make_fragment_B(kB)

        fx.copy(copy_atom, tcA.partition_S(kA), tcA.retile(frag_A))  # load A k-tile
        fx.copy(copy_atom, tcB.partition_S(kB), tcB.retile(frag_B))  # load B k-tile

        # One fx.gemm call issues one MFMA per wave (4 in the block, concurrent),
        # all accumulating into the same frag_C across all 32 iterations.
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # Store the accumulated 32×32 result to global memory.
    fx.copy(copy_atom, tcC.retile(frag_C), tcC.partition_S(bC))
```

A few things to notice:

- **`frag_C` is allocated and zeroed once**, before the loop — the accumulator VGPRs
  persist across all 32 K-tiles. After the loop each lane holds its final 4 f32, its
  private slice of the 32×32 C tile, with nothing to reduce against the other waves.
- **`fx.range_constexpr(num_k_tiles)`** unrolls the 32-iteration loop at trace time.
  Each wave sees 32 consecutive MFMA instructions (128 across the block) and the
  compiler can schedule loads against them for latency hiding. For a runtime K, use
  `range(0, K, BK_atom, init=[frag_C.load()])` with a `yield` to carry the accumulator
  as an `scf.for` iter-arg (§3.2).
- **A is (BM, BK_atom) = (32, 4)** per K-tile, and **B is (BN, BK_atom) = (32, 4)**
  (N-first). Each wave takes the half of each that its atom coordinate selects, and the
  broadcast rule above means the halves are shared: waves 0 and 2 both load
  `kA[0:16,:]`, waves 1 and 3 both load `kA[16:32,:]`; waves 0 and 1 both load
  `kB[0:16,:]`, waves 2 and 3 both load `kB[16:32,:]`. The TV layout inside
  `make_tiled_copy_A/B` routes each thread's load to the right register slot
  automatically.

The no-loop version is `examples/03-tiledMma.py`. Note it uses a `64×64×8` block tile
against the same 2×2 atom layout, so it exercises the rest modes: still 4 waves, but
8 MFMA instructions and 16 accumulator f32 per lane.

## The fragment contract and `fx.gemm`

`fx.gemm` computes `D = A * B + C` on fragments allocated from the thread's MMA
slice (§7.4):

```python
thr_mma = tiled_mma.thr_slice(tid)
frag_A = thr_mma.make_fragment_A(bA)     # A-operand registers, MFMA order
frag_B = thr_mma.make_fragment_B(bB)     # B-operand registers, MFMA order
frag_C = thr_mma.make_fragment_C(bC)     # accumulator registers
frag_C.fill(0)

fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)   # D=C, A, B, C  (accumulate)
```

The contract that makes this correct:

1. `make_fragment_A/B/C` allocate register tensors whose layout **is**
   `layout_A_tv/B_tv/C_tv` of the atom — so the elements are already in
   MFMA-operand order.
2. When you *load* into them, you `retile` the copy to that same order (§7.4). Get
   this wrong and you get numerically plausible garbage — the classic MFMA
   debugging trap. In FlyDSL both sides derive from the *same atom*, so they match
   by construction.
3. `fx.gemm` lowers (Stage A, `convert-fly-to-rocdl`) to `rocdl.mfma.*` — the same
   intrinsic your CK-Tile kernel emits.

> **HIP/CK-Tile → FlyDSL.** `fx.gemm(atom, C, A, B, C)` ≡ the inner
> `c_frag = __builtin_amdgcn_mfma_...(a_frag, b_frag, c_frag, ...)` call.
> `make_fragment_*` ≡ declaring the operand/accumulator VGPR arrays with the
> ISA-mandated element order. `frag_C.fill(0)` ≡ zeroing the accumulator before
> the K-loop.

## Accumulating over K

A real GEMM sums many K-tiles into one accumulator. The accumulator fragment is
allocated and zeroed **once, outside** the K-loop; each iteration loads fresh A/B
tiles and MFMAs into the *same* `frag_C`:

```python
frag_C = thr_mma.make_fragment_C(bC)
frag_C.fill(0)

for k in fx.range_constexpr(num_k_tiles):     # unrolled tiling loop (§1.3)
    kA = fx.slice(bA_tiles, (None, k))
    kB = fx.slice(bB_tiles, (None, k))
    fx.copy(copy_atom_a, thr_copy_a.partition_S(kA), thr_copy_a.retile(frag_A))
    fx.copy(copy_atom_b, thr_copy_b.partition_S(kB), thr_copy_b.retile(frag_B))
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

# store frag_C once, after the loop
fx.copy(copy_atom_c, thr_copy_c.retile(frag_C), thr_copy_c.partition_D(bC))
```

Use `range_constexpr` when `num_k_tiles` is known at trace time (it usually is,
from a `Constexpr` block-K); the loop unrolls and the compiler schedules the
loads and MFMAs. When K is dynamic, use `range(..., init=[frag_C.load()])` and
carry the accumulator as a loop-carried value (§3.2) so the SSA form stays
well-defined.

> **HIP/CK-Tile → FlyDSL.** This is the classic accumulate-over-K main loop. The
> `range_constexpr` unroll is your `#pragma unroll` K-loop; the loop-carried
> `range(..., init=[...])` form (§3.2) is the runtime K-loop with the accumulator
> kept in registers across iterations. Software pipelining (prefetch next K while
> MFMA-ing current) is layered on top in Chapter 13 and the GEMM puzzles — the atom
> and fragment contract does not change.

## Where subtargets differ

The *atom* is architecture-independent in your Python; the *instruction it lowers
to* is chosen per subtarget in `lib/Dialect/FlyROCDL/`:

| Arch | Wave | MMA op | Atom factory |
|------|------|--------|--------------|
| gfx942 (MI300X, CDNA3) | 64 | MFMA | `fx.rocdl.MFMA(...)` |
| gfx950 (MI350, CDNA4) | 64 | MFMA (+ fp4, MFMA-scale) | `fx.rocdl.MFMA(...)` |
| gfx11xx (RDNA3) | 32 | WMMA | `fx.rocdl.WMMA(...)` |
| gfx1250 | 32 | WMMA / MX-scaled WMMA | `fx.rocdl.WMMA(...)` / `WMMAScale(...)` |

This book targets **CDNA MFMA (gfx942/gfx950)**, so every example uses
`fx.rocdl.MFMA`. On RDNA you would swap the atom for `fx.rocdl.WMMA`; the layout
algebra, tiling, partitioning, and copy machinery are identical. The one number that
changes is the atom's own thread layout — 32 instead of 64 — so the same atom layout
implies half the threads: `(2, 2, 1)` over a WMMA atom is 4 waves of 32, i.e. a
128-thread block. `tests/kernels/test_gfx1250_atoms_device.py:141` is the
one-wave case: atom layout `(1, 1, 1)` launched with `block=(32, 1, 1)`. That
portability — same layout algebra, swap the atom — is the payoff of making the atom a
first-class object.

With layouts (Ch. 6), tiling/partitioning (Ch. 7), copy atoms (Ch. 8), and MMA
atoms (Ch. 9) defined, you have the full high-level vocabulary. Chapters 10–11 open
the copy and MFMA atoms down to the actual `rocdl.*` instructions and VGPR layouts,
and Chapter 12 covers the escape hatches for when the atom layer is not enough.
Chapter 13 then reads three complete kernels line by line; Chapter 14 is how you
debug them when they break; Chapter 15 is the reference you keep open while working
the puzzles.
