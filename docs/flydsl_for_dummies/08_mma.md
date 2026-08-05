# The Fly Dialect IV — MMA Atoms

This chapter is about the matrix cores. On CDNA (MI300X/MI350) that means the
**MFMA** instructions; you know them as `__builtin_amdgcn_mfma_*`. In FlyDSL an
MFMA is an **MMA atom**, replicated by a **TiledMma** (Chapter 6) and executed by
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

One atom computes a 16×16 output. To cover a larger block tile you replicate the
atom with `make_tiled_mma` and an **atom layout** describing the grid of atoms:

```python
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
```

The `(2, 2, 1)` means "2 atoms in M, 2 in N, 1 in K" → a 32×32 output computed by
4 MFMA atoms, mapped across the wave(s) by the given stride layout. An optional
third argument permutes/strides the atom placement for scheduling. The resulting
`tile_size_mnk` is the block-level MMA tile your copies must fill.

> **HIP/CK-Tile → FlyDSL.** The atom layout is CK-Tile's *warp tiling*: how many
> MFMA instructions per warp in M/N/K and how warps are arranged. In C++ you set
> `MWarp`, `NWarp`, `WarpGemm` repeats; here it is one Layout argument.

## The fragment contract and `fx.gemm`

`fx.gemm` computes `D = A * B + C` on fragments allocated from the thread's MMA
slice (§6.4):

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
2. When you *load* into them, you `retile` the copy to that same order (§6.4). Get
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
> MFMA-ing current) is layered on top in Chapter 9 and the GEMM puzzles — the atom
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

With layouts (Ch. 5), tiling/partitioning (Ch. 6), copy atoms (Ch. 7), and MMA
atoms (Ch. 8) defined, you have the full vocabulary. Chapter 9 reads three
complete kernels line by line; Chapter 10 is how you debug them when they break;
Chapter 11 is the reference you keep open while working the puzzles.
