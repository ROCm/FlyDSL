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

## TiledMma: many atoms make a warp/block tile

One atom computes a 16×16 output — one wavefront, 64 lanes, 4 f32 accumulators per
lane. To cover a larger block tile you replicate the atom with `make_tiled_mma` and
an **atom layout** describing the grid of atoms in the MNK dimensions:

```python
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
```

`(2, 2, 1)` is the shape of the atom grid: **2 atoms along M, 2 along N, 1 along K**.
Four atoms in total, so the output tile is **32×32 f32**.

### Where the four atoms execute — same wave, sequential calls

A crucial point: an MFMA instruction is a *wavefront* operation — all 64 lanes in the
wave participate together. The four atoms in the 2×2 grid are **four sequential MFMA
instructions issued by the same 64 lanes**, not four separate waves. The stride layout
`(1, 2, 0)` controls only the *logical ordering* of those four calls (how atom indices
map to instruction order), not which hardware lanes execute them.

What the four atoms compute against a single K-column (K_atom=4):

```
atom (m=0, n=0): A[0:16,  0:4]  ×  B[0:16,  0:4]^T  →  C[0:16,  0:16]
atom (m=1, n=0): A[16:32, 0:4]  ×  B[0:16,  0:4]^T  →  C[16:32, 0:16]
atom (m=0, n=1): A[0:16,  0:4]  ×  B[16:32, 0:4]^T  →  C[0:16,  16:32]
atom (m=1, n=1): A[16:32, 0:4]  ×  B[16:32, 0:4]^T  →  C[16:32, 16:32]
```

A is **32×K** (M-rows × K-cols), B is **32×K** stored **N-first** (so the MFMA sees
`A × B^T`), and C is **32×32**. Each atom reuses the K-slice — only the M and N tile
selects which rows/columns of A and B it reads.

For a block with 256 threads (4 waves), each wave handles a *different* 32×32 tile of
a larger 64×64 block. Within each wave, the four atoms are still sequential.

> **HIP/CK-Tile → FlyDSL.** The atom layout is CK-Tile's *warp tiling*: `(2, 2, 1)`
> is `MRepeat=2, NRepeat=2, KRepeat=1` in a `WarpGemmAttribute`. In C++ you set those
> repeats as template parameters; here they are one `make_layout` argument. The four
> resulting MFMA calls correspond to the `#pragma unroll` loop body you would write
> over `mRepeat × nRepeat`.

### Worked example: 32×32 output, K=128, accumulated over 32 K-tiles

With `MFMA(16,16,4,f32)` the K-depth per atom call is 4. To compute a full
`A(32×128) × B(32×128)^T → C(32×32)` in one block, the K-loop runs **32 iterations**
(128/4). Each iteration loads a `(32, 4)` slice of A and B and calls `fx.gemm` once —
which internally issues the four MFMA instructions for the 2×2 atom grid, accumulating
into the *same* `frag_C`.

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
    # Each lane holds 16 f32 values (4 per atom × 4 atoms in the 2×2 grid).
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

        # One fx.gemm call issues 4 MFMA instructions (the 2×2 atom grid),
        # all accumulating into the same frag_C across all 32 iterations.
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # Store the accumulated 32×32 result to global memory.
    fx.copy(copy_atom, tcC.retile(frag_C), tcC.partition_S(bC))
```

A few things to notice:

- **`frag_C` is allocated and zeroed once**, before the loop — the accumulator VGPRs
  persist across all 32 K-tiles and 4 atoms inside each `fx.gemm` call. After the
  loop, each lane holds a final 16 f32 values representing its portion of the 32×32 C
  tile.
- **`fx.range_constexpr(num_k_tiles)`** unrolls the 32-iteration loop at trace time.
  The compiler sees 32 × 4 = 128 consecutive MFMA instructions and can schedule loads
  against them for latency hiding. For a runtime K, use
  `range(0, K, BK_atom, init=[frag_C.load()])` with a `yield` to carry the accumulator
  as an `scf.for` iter-arg (§3.2).
- **A is (BM, BK_atom) = (32, 4)** per K-tile, and **B is (BN, BK_atom) = (32, 4)**
  (N-first). The 4-column slice feeds all four atoms: atom (m=0,n=0) and atom (m=1,n=0)
  use the same `kA[:,0:4]`; atom (m=0,n=0) and atom (m=0,n=1) use the same `kB[0:16,:]`
  vs `kB[16:32,:]`. The TV layout inside `make_tiled_copy_A/B` routes each thread's
  load to the right register slot automatically.

The full single-block version (K=BK_atom, no loop) is `examples/03-tiledMma.py`.

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
`fx.rocdl.MFMA`. On RDNA you would swap the atom for `fx.rocdl.WMMA` and adjust
the wave size to 32; the layout algebra, tiling, partitioning, and copy machinery
are identical. That portability — same layout algebra, swap the atom — is the
payoff of making the atom a first-class object.

With layouts (Ch. 6), tiling/partitioning (Ch. 7), copy atoms (Ch. 8), and MMA
atoms (Ch. 9) defined, you have the full high-level vocabulary. Chapters 11–11 open
the copy and MFMA atoms down to the actual `rocdl.*` instructions and VGPR layouts,
and Chapter 12 covers the escape hatches for when the atom layer is not enough.
Chapter 13 then reads three complete kernels line by line; Chapter 14 is how you
debug them when they break; Chapter 15 is the reference you keep open while working
the puzzles.
