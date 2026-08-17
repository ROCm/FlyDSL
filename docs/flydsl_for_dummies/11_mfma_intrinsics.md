# MFMA, Close to the Metal

Chapter 9 issued `fx.gemm` over an MMA atom and told you it "lowers to
`rocdl.mfma.*`." This chapter opens the atom: the exact instruction each shape picks,
how the operand and accumulator fragments map to lanes and VGPRs (the matrix-core
ABI), and how to issue a single MFMA. It is the deep-dive companion to Chapter 9 — if
you have ever called `__builtin_amdgcn_mfma_*` directly, this is the FlyDSL view of
the same instruction.

## The MFMA op type and which instruction it picks

`fx.rocdl.MFMA(M, N, K, ab_dtype, acc_dtype=f32)` (`rocdl/universal.py:106`) selects
one hardware MFMA shape; `M == N` is required. The shape plus the operand dtype maps
to exactly one `rocdl.mfma.*` op and one `__builtin_amdgcn_mfma_*` builtin (dispatch
table in `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:170`). The shapes this book uses:

| FlyDSL constructor | `rocdl` op |
|--------------------|------------|
| `MFMA(16,16,4, f32)` | `mfma.f32.16x16x4f32` |
| `MFMA(16,16,16, f16)` | `mfma.f32.16x16x16f16` |
| `MFMA(16,16,16, bf16)` | `mfma.f32.16x16x16bf16.1k` |
| `MFMA(32,32,8, f16)` | `mfma.f32.32x32x8f16` | 
| `MFMA(16,16,32, fp8)` | `mfma.f32.16x16x32.fp8.fp8` | 
| `MFMA(16,16,32, i8, i32)` | `mfma.i32.16x16x32.i8` | 

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

This is why `make_fragment_A/B/C` and `retile` (Chapter 9) are not bookkeeping: they
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

## Worked example: one 16×16×16 MFMA, high and low

Here is a single `16×16×16` bf16 MFMA — one wavefront computing `D = A · Bᵀ` on
16×16 tiles — written three ways that all produce the identical result. The complete
runnable file is `examples/07-single_mfma_lowlevel_vs_highlevel.py`; `A` is 16×16
row-major (M×K), `B` is 16×16 row-major (N×K, so the MFMA sees `A · Bᵀ`), `D` is
16×16 f32.

**High-level — the MMA atom hides the VGPR layout:**

```python
A = fx.rocdl.make_buffer_tensor(A)                        # M x K
B = fx.rocdl.make_buffer_tensor(B)                        # N x K
C = fx.rocdl.make_buffer_tensor(C)                        # M x N
# bA / bB / bC: this block's 16x16 tiles (one block here, so tile 0).
# zipped_divide + slice give them a *static* shape, which the fragments need.
bA = fx.slice(fx.zipped_divide(A, (M, K)), (None, 0))
bB = fx.slice(fx.zipped_divide(B, (N, K)), (None, 0))
bC = fx.slice(fx.zipped_divide(C, (M, N)), (None, 0))

mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
thr_mma   = tiled_mma.thr_slice(tid)

# copy atoms + this thread's slice of the A/B/C tiled copies, derived from the MMA
# so the load order matches the operand layout (that is what retile relies on).
acopy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)   # bf16 operands
ccopy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)    # f32 accumulator
tcA = fx.make_tiled_copy_A(acopy, tiled_mma).get_slice(tid)
tcB = fx.make_tiled_copy_B(acopy, tiled_mma).get_slice(tid)
tcC = fx.make_tiled_copy_C(ccopy, tiled_mma).get_slice(tid)

frag_A = thr_mma.make_fragment_A(bA)   # vector<4xbf16> per lane
frag_B = thr_mma.make_fragment_B(bB)   # vector<4xbf16> per lane
frag_C = thr_mma.make_fragment_C(bC)   # vector<4xf32>  per lane

fx.copy(acopy, tcA.partition_S(bA), tcA.retile(frag_A))   # load A operand
fx.copy(acopy, tcB.partition_S(bB), tcB.retile(frag_B))   # load B operand
frag_C.fill(0)
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)         # -> rocdl.mfma...bf16_1k
fx.copy(ccopy, tcC.retile(frag_C), tcC.partition_S(bC))   # store C
```

Here **`bA`/`bB`/`bC`** are the block's 16×16 input/output tiles (from
`zipped_divide` + `slice`, Chapter 7), and **`tcA`/`tcB`/`tcC`** are this thread's
slices of the A/B/C tiled copies — built from the *same* `tiled_mma` via
`make_tiled_copy_A/B/C` so the copy fills the fragments in the exact order the atom
expects. You never name a lane or a register: `make_fragment_A/B/C` allocate the
operand and accumulator VGPRs *in the atom's layout*, and `retile` re-expresses the
copy in that layout.

**Low-level — fill the VGPRs by hand and call the raw op.** Now you must reproduce
the operand ABI from §"The fragment contract" yourself. For `16×16×16` on 64 lanes:
lane `l` holds `A[m, kg*4+i]` with `m = l%16`, `kg = l//16`, `i∈0..3`; `B[n, kg*4+i]`
with `n = l%16`; and the accumulator lane owns `C[kg*4+i, n]`.

```python
lane = fx.thread_idx.x
m = lane % fx.Int32(16); n = lane % fx.Int32(16); kg = lane // fx.Int32(16)
aptr, bptr, cptr = fx.get_iter(A), fx.get_iter(B), fx.get_iter(C)

a_el, b_el = [], []
for i in fx.range_constexpr(4):                    # 4 K-elements per lane
    k = kg * fx.Int32(4) + fx.Int32(i)
    a_el.append(fx.ptr_load(aptr + (m * fx.Int32(K) + k)))   # A[m, k]
    b_el.append(fx.ptr_load(bptr + (n * fx.Int32(K) + k)))   # B[n, k]

# rocdl mfma...bf16_1k takes the bf16 operands as i16 lanes
a = fx.Vector.from_elements(a_el, dtype=fx.BFloat16).bitcast(fx.Int16)
b = fx.Vector.from_elements(b_el, dtype=fx.BFloat16).bitcast(fx.Int16)
c0 = fx.Vector.filled(4, 0.0, fx.Float32)

acc = rocdl.mfma_f32_16x16x16bf16_1k(                # the raw intrinsic
    fx.Vector.make_type(4, fx.Float32),
    [a.ir_value(), b.ir_value(), c0.ir_value()],
)
acc = fx.Vector(acc, (4,), fx.Float32)

for i in fx.range_constexpr(4):                     # scatter accumulator back
    mrow = kg * fx.Int32(4) + fx.Int32(i)
    fx.ptr_store(acc[i], cptr + (mrow * fx.Int32(N) + n))    # C[mrow, n]
```

The `rocdl.mfma_f32_16x16x16bf16_1k(result_type, [a, b, c])` call is the FlyDSL
spelling of `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`. Get the lane→element map
wrong and it still runs — it just computes the wrong matrix (the "plausible garbage"
trap). This is precisely the bookkeeping the atom does for you.

**Bridging the two — go from fragments to the raw op and back.** You do not have to
pick one level for the whole kernel. A fragment's registers *are* the MFMA operand
VGPRs, so `frag.load()` hands them straight to the raw intrinsic, and `frag.store()`
pushes a raw result back into fragment form. This reuses the same `bA`/`bB`/`bC`
tiles, `thr_mma`, and `tcA`/`tcB`/`tcC` copy slices set up in the high-level version
above:

```python
frag_A = thr_mma.make_fragment_A(bA)     # build operands the high-level way
frag_B = thr_mma.make_fragment_B(bB)
frag_C = thr_mma.make_fragment_C(bC)
fx.copy(acopy, tcA.partition_S(bA), tcA.retile(frag_A))
fx.copy(acopy, tcB.partition_S(bB), tcB.retile(frag_B))

# HIGH -> LOW: pull raw vectors out and call the intrinsic directly
a_vec = frag_A.load()                     # vector<4xbf16>
b_vec = frag_B.load()
acc = rocdl.mfma_f32_16x16x16bf16_1k(
    fx.Vector.make_type(4, fx.Float32),
    [a_vec.bitcast(fx.Int16).ir_value(),
     b_vec.bitcast(fx.Int16).ir_value(),
     fx.Vector.filled(4, 0.0, fx.Float32).ir_value()],
)
# LOW -> HIGH: push the raw accumulator back into the fragment
frag_C.store(fx.Vector(acc, (4, ), fx.Float32))
fx.copy(ccopy, tcC.retile(frag_C), tcC.partition_S(bC))
```

This is the useful escape hatch: keep the high-level fragment/copy machinery for
loading and storing (where the layout algebra earns its keep), and drop to the raw
`rocdl.mfma_*` only for the instruction itself — for example to pass an `op_sel`
modifier, pin the accumulator in an AGPR, or issue an instruction the atom does not
yet cover (Chapter 12).

> **HIP/CK-Tile → FlyDSL.** `frag.load()` / `frag.store()` around a raw
> `rocdl.mfma_*` is the CK-Tile move of reading a `WarpGemm`'s register spans as a
> plain `fp32x4` / `bf16x4`, calling `__builtin_amdgcn_mfma_*` yourself, and writing
> the result back into the accumulator span.

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
`scf.for` iter_args (§3.2, Chapter 9) so the accumulator registers stay a
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

With the copy and MMA instructions both opened up, Chapter 12 turns to the cases where
even this atom layer is not enough — small MFMAs outside a GEMM, cross-lane ops, and
inline assembly. Chapter 13 then reads three complete kernels end to end.
