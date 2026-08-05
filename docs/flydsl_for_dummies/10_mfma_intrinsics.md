# MFMA, Close to the Metal

Chapter 8 issued `fx.gemm` over an MMA atom and told you it "lowers to
`rocdl.mfma.*`." This chapter opens the atom: the exact instruction each shape picks,
how the operand and accumulator fragments map to lanes and VGPRs (the matrix-core
ABI), and how to issue a single MFMA. It is the deep-dive companion to Chapter 8 — if
you have ever called `__builtin_amdgcn_mfma_*` directly, this is the FlyDSL view of
the same instruction.

## The MFMA op type and which instruction it picks

`fx.rocdl.MFMA(M, N, K, ab_dtype, acc_dtype=f32)` (`rocdl/universal.py:106`) selects
one hardware MFMA shape; `M == N` is required. The shape plus the operand dtype maps
to exactly one `rocdl.mfma.*` op and one `__builtin_amdgcn_mfma_*` builtin (dispatch
table in `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:170`). The shapes this book uses:

| FlyDSL constructor | `rocdl` op | HIP builtin |
|--------------------|------------|-------------|
| `MFMA(16,16,4, f32)` | `mfma.f32.16x16x4f32` | `__builtin_amdgcn_mfma_f32_16x16x4f32` |
| `MFMA(16,16,16, f16)` | `mfma.f32.16x16x16f16` | `…_mfma_f32_16x16x16f16` |
| `MFMA(16,16,16, bf16)` | `mfma.f32.16x16x16bf16.1k` | `…_mfma_f32_16x16x16bf16_1k` |
| `MFMA(32,32,8, f16)` | `mfma.f32.32x32x8f16` | `…_mfma_f32_32x32x8f16` |
| `MFMA(16,16,32, fp8)` | `mfma.f32.16x16x32.fp8.fp8` | `…_mfma_f32_16x16x32_fp8_fp8` |
| `MFMA(16,16,32, i8, i32)` | `mfma.i32.16x16x32.i8` | `…_mfma_i32_16x16x32i8` |

(gfx950 adds wider-K shapes like `16x16x32 f16` and `32x32x16 f16`; the full list is
in the dispatch table.)

> **HIP/CK-Tile → FlyDSL.** Choosing `MFMA(M,N,K,dtype)` ≡ choosing which
> `__builtin_amdgcn_mfma_*` to call. The shape trades register pressure against
> instruction count exactly as it does in C++.

## The fragment contract is the matrix-core ABI

This is the crux of the whole chapter. An MFMA does not read matrices from memory —
it reads them from **specific VGPRs in specific lanes**. The A/B operand fragments and
the C/D accumulator fragment must have their elements in exactly the lane×register
slots the instruction expects. FlyDSL encodes that mapping as the atom's thread-value
layouts (`CDNA3/MmaAtom.cpp` `getThrValLayoutAB:17`, `getThrValLayoutC:63`).

Take `MFMA(16, 16, 16, f16)` on a wave of 64 lanes:

- **A and B operands.** `GroupK = 64/16 = 4`, so K=16 is split into 4 groups; each
  lane holds `KPerThread = 4` contiguous K-elements — one `vector<4xf16>`. The 16 rows
  (M for A, N for B) rotate across the 64 lanes in groups of 16.
- **C/D accumulator.** Each lane owns `vector<4xf32>` — 4 accumulators. The N
  coordinate is `lane % 16`; the M-group is `lane / 16` (4 groups of 4 rows).

VGPR counts for the common shapes:

| Shape | A/B per lane | C/D per lane |
|-------|--------------|--------------|
| `16x16x16 f16` | `vector<4xf16>` | 4 × f32 |
| `16x16x32 fp8` | `i64` (8 packed fp8) | 4 × f32 |
| `32x32x8 f16` | `vector<4xf16>` | 16 × f32 |

This is why `make_fragment_A/B/C` and `retile` (Chapter 8) are not bookkeeping: they
place bytes in precisely those slots. Load A/B with a copy whose TV layout does *not*
match the atom's operand layout and the MFMA reads the wrong lanes — you get
numerically plausible garbage, the classic MFMA bug. In FlyDSL both the fragment and
the retiled copy derive from the *same atom*, so they match by construction.

> **HIP/CK-Tile → FlyDSL.** These thread-value layouts are the register↔matrix mapping
> you learn from the ISA docs and encode by hand when packing operand VGPRs; in
> CK-Tile they live inside `WarpGemmAttribute` / the `mfma_instr` traits. Here they
> are queryable properties of the atom.

## One instruction: `fx.mma_atom_call`, and how `fx.gemm` decomposes

`fx.gemm` over a TiledMma unrolls into one MFMA per atom in the tile. The single-atom
primitive underneath is `fx.mma_atom_call(atom, d, a, b, c)` (`primitive.py:1058`),
which emits one `fly.mma_atom_call`:

```mlir
%atom = fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32,f32) -> f32>>
fly.mma_atom_call(%atom, %d, %a, %b, %c)
  : (!fly.mma_atom<…>, !fly.memref<f32, register, 4:1>, …) -> ()
```

which `MmaAtomCallLowering` (`FlyToROCDL.cpp:627`) turns into
`rocdl.mfma.f32.16x16x4f32`. The operand/result vector types you see there
(`vector<4xf16>` for f16 operands, `vector<4xf32>` for the accumulator) *are* the
builtin's argument types.

> **HIP/CK-Tile → FlyDSL.** `fx.gemm(atom, C, A, B, C)` ≡ the inner
> `c = __builtin_amdgcn_mfma_...(a, b, c, 0, 0, 0)` call, and `fx.mma_atom_call` is
> that single builtin when you drive the loop yourself. `make_fragment_*` ≡ declaring
> the operand/accumulator VGPR arrays in ISA-mandated order.

## Accumulating over K, at the register level

A GEMM sums many K-tiles into one accumulator. The accumulator VGPRs are allocated and
zeroed **once**, and every MFMA in the K-loop reads and writes those same registers:

```python
# puzzles/solutions/puzzle10_gemm_kloop.py:65
frag_C = thr_mma.make_fragment_C(bC)
frag_C.fill(0)                                  # zero the accumulator VGPRs
for ki in range_constexpr(num_k):               # unroll the MFMA sequence
    # ... load frag_A, frag_B for K-tile ki ...
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)   # C += A·B, same regs
```

`range_constexpr` unrolls the MFMA chain (the compiler schedules loads against MFMAs);
when K is dynamic, `range(..., init=[frag_C.load()])` carries the accumulator as
`scf.for` iter_args (§3.2, Chapter 8) so the accumulator registers stay a
well-defined SSA value across iterations.

## MFMA-scale and other subtargets

CDNA4 (gfx950) adds *scaled* MFMA for microscaled fp8/fp6/fp4:
`fx.rocdl.cdna4.MFMA_Scale(M, N, K, dtype)` (`rocdl/cdna4.py:19`) is a **stateful**
atom carrying E8M0 block scales, injected per call:

```python
fx.gemm(scale_atom, cf, av, bv, cf, scale_a=sa, scale_b=sb)   # -> rocdl.mfma.scale.*
```

`16x16x128` / `32x32x64` fp8/fp6/fp4 lower to
`rocdl.mfma.scale.f32.16x16x128.f8f6f4` (real use:
`kernels/gemm/fp8_gemm_utils.py:211`). The matrix core differs by subtarget:

| Arch | Wave | Op | C/D acc (16×16) |
|------|------|----|-----------------|
| gfx942 (CDNA3) | 64 | MFMA | `vector<4xf32>` |
| gfx950 (CDNA4) | 64 | MFMA + MFMA-scale | `vector<4xf32>` |
| gfx11xx (RDNA3) | 32 | WMMA | `vector<8xf32>` |
| gfx1250 | 32 | WMMA / WMMA-scale | `vector<8xf32>` |

The atom is architecture-independent in your Python; only the chosen instruction and
its fragment layout change per subtarget. This book targets CDNA MFMA.

With the copy and MMA instructions both opened up, Chapter 11 turns to the cases where
even this atom layer is not enough — small MFMAs outside a GEMM, cross-lane ops, and
inline assembly. Chapter 12 then reads three complete kernels end to end.
