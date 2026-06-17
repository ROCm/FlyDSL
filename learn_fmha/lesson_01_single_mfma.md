# Lesson 01 — A single MFMA instruction

**Prereq:** Lesson 00 (kernel/launch model). **Run:**
`HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_01_single_mfma.py` → `max abs err = 0.0000 PASS`.

## What we build
One 16×16 output tile `C = A @ Bᵀ` with a **single** `mfma_f32_16x16x16bf16_1k` issued by one
64-lane wavefront. No LDS, no loops. This is the atom of every matrix kernel.

## If you come from CK Tile
This is exactly one `WarpGemmImpl` / `WarpGemmAttribute` call. In CK you pick a `WarpGemm` and the
framework's `tile_distribution` hides which thread owns which fragment element. In FlyDSL (at the
level we teach here) **there is no distribution abstraction** — *you* compute, per lane, the global
indices to load A/B from and store C to. That sounds painful but it's the point: every optimization
later is "make the data land in the lane the MFMA wants," so you must see the mapping explicitly.

## The one thing to internalize: the lane↔element layout
A wavefront is 64 lanes. For `16x16x16` bf16 on CDNA3 (gfx942), view the 64 lanes as
`lane = k_outer*16 + mn`, where `k_outer ∈ 0..3` and `mn ∈ 0..15`. Then:

| fragment | per-lane data (4 elements) | meaning |
|---|---|---|
| **A** (input) | `A[mn, k_outer*4 + e]`, e=0..3 | lane holds 4 contiguous-K of row `mn` |
| **B** (input) | `B[mn, k_outer*4 + e]`, e=0..3 | B read as `[N,K]` → instruction does `A @ Bᵀ` |
| **C** (output) | `C[k_outer*4 + e, mn]`, e=0..3 | result **row** = `(lane//16)*4+e`, **col** = `lane%16` |

Two subtleties that bite everyone:
1. **B is row-major `[N,K]`**, so the hardware computes `A @ Bᵀ`. To get `A @ B` you'd transpose B
   in memory or swap operands. (In attention this is *why* GEMM1 is naturally `K @ Qᵀ`.)
2. **The output layout ≠ the input layout.** A lane that fed K-slices gets back M-rows. This
   mismatch is the seed of the "P-transpose problem" in attention (Lesson 05): GEMM1's C output has
   the contraction index scattered the wrong way for GEMM2's input.

## How we *verified* it (the method, not the trust)
The layout above is written as code, then checked against torch `A @ B.T` with an `assert`. If any
index were wrong, `err` would be ~O(1), not 0. **Never trust a layout comment — probe it.** The
production kernel was built exactly this way: a tiny kernel that writes known values and reads back
which lane got what. When you move to a new MFMA shape (e.g. `32x32x16` fp8 in Lesson 07) you
re-derive and re-verify the layout; do not assume it carries over.

## FlyDSL mechanics introduced
- `fx.buffer_ops.create_buffer_resource(T)` — wrap a tensor as a buffer descriptor (CK: a
  `BufferView`). `buffer_load(res, elem_index, vec_width, dtype)` / `buffer_store(val, res, index)`
  load/store by **element index** (not byte).
- `fx.Vector.from_elements([...], dtype)` builds a register vector; `.bitcast(dtype)` reinterprets.
  The `_1k` bf16 MFMA wants operands as `vector<4xi16>`, so we `.bitcast(fx.Int16)` — a gotcha you'd
  hit immediately otherwise (the verifier error names the expected type, which is how we found it).
- `fx.rocdl.mfma_f32_16x16x16bf16_1k_(result_type, a, b, c, cbsz, abid, blgp)` — the raw MFMA op.
  `c` is the accumulator (start at zero here; in a K-loop you chain it).

## Profiling note (we'll use this constantly from Lesson 08)
Even here you can dump the ISA and confirm exactly one MFMA was emitted:
```
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_01_single_mfma.py
grep -c v_mfma /tmp/isa/mfma_kernel_0/21_final_isa.s   # -> 1
```

## Next
Lesson 02: loop this MFMA over K to do a real `[M,K]@[K,N]` GEMM tile, add wide loads and the
first `do_bench` throughput measurement.
