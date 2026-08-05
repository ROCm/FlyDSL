# When the High-Level Dialect Isn't Enough

The layout/atom API of Chapters 5–10 covers the two shapes that dominate GPU kernels:
tile-shaped copies and tiled GEMM. Real kernels routinely hit work that does not fit
that mold — a single small MFMA feeding an online-softmax rescale, a cross-lane
reduction, a hand-scheduled software pipeline. FlyDSL does not wall these off; it
gives you supported lower-level APIs, layered so you drop *only* the piece that needs
it and keep the rest of the kernel high-level.

This chapter is the honest map of those escape hatches: what each is, the API to
reach it, and — most importantly — *why* the high-level layer cannot express it. The
guiding split (from `CLAUDE.md`) is that `python/flydsl/expr/` is target-neutral,
while the target-specific instructions live in the `python/flydsl/expr/rocdl/`
package — which is where most of these hatches are.

## A single MFMA outside a GEMM — the headline case

The archetype is **flash attention**. Its two matmuls — Q·Kᵀ (scores) and P·V
(output) — are not a tiled GEMM you can hand to `fx.gemm`. Between them sits the
online softmax: after each KV block you rescale the running output by a correction
factor (`o_acc = o_acc * corr`) and update running max/sum. The MFMA operands come
from on-chip state (the just-computed probabilities), not a clean global tile, and the
accumulator must stay live in registers *across* the rescale.

So attention kernels build the fragments by hand and issue a single MFMA via the SSA
form of the atom call, which keeps the accumulator as a plain `vector<…f32>` SSA value:

```python
# kernels/attention/flash_attn_utils.py:148
def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)

# :2738 — dispatch straight to a rocdl MFMA builtin for the chosen dtype
def mfma_acc(self, a, b, c):
    return self._mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)   # one instruction
```

The accumulator (`o_acc`, a `vector<16xf32>`) is an ordinary SSA value the kernel
multiplies, adds to, and feeds back into the next MFMA.

**Why the high-level API can't express this.** `fx.gemm` assumes one uniform tiled
accumulator layout and a pure `D = A·B + C` repeated over a tile. It offers no seam to
interleave a per-element rescale between MFMA steps, to carry differently-shaped
lo/hi sub-fragments, or to source operands from on-chip state. The moment the matmul
is not the whole loop body, you drop to the single-atom call.

> **HIP/CK-Tile → FlyDSL.** This is exactly calling `__builtin_amdgcn_mfma_*` directly
> on your own `float4` / `floatx16` fragments — how a CK-Tile flash-attention kernel is
> written too. `fx.gemm` is the convenience for the case where the matmul *is* the loop.

## Direct `rocdl.*` ops

`fx.rocdl` (`python/flydsl/expr/rocdl/__init__.py`) exposes thin wrappers that emit a
ROCDL op directly, bypassing the atom infrastructure. The ones you actually reach for:

- **MFMA builtins** — `rocdl.mfma_f32_32x32x8f16`, `rocdl.mfma_scale_f32_*` — when you
  want the instruction without a TiledMma (as above).
- **Pack / convert** — `rocdl.cvt_pk_fp8_f32`, `rocdl.cvt_scalef32_pk_fp4_f32` — per-lane
  narrow-float conversions used to *build* MFMA operands from f32.
- **Misc** — `rocdl.rcp` (fast reciprocal), `rocdl.readfirstlane`, `rocdl.ballot`.

**Why:** a per-lane type conversion or a reciprocal is a scalar/vector op on register
values; it has no copy or gemm analogue, so there is nothing at the atom layer to
express it.

> **HIP/CK-Tile → FlyDSL.** These are the loose `__builtin_amdgcn_*` intrinsics you
> sprinkle between the big tiled ops in a hand-written kernel.

## Cross-lane / warp primitives

Reductions (softmax max/sum, RMS norm, absmax for quantization) move values *between
lanes' registers*, which is not a memory movement and has no tile layout. FlyDSL
exposes three mechanisms:

- `x.shuffle_xor(offset, width)` (`expr/utils/arith.py:515`) → `ds_swizzle`, the
  butterfly-reduce step.
- `fx.rocdl.ds_bpermute(ty, byte_idx, v)` — an arbitrary cross-lane read (any lane
  reads any lane's VGPR by absolute address), which `shuffle_xor`'s mask cannot express.
- `fx.rocdl.permlane32_swap(...)`, and DPP via `kernels/common/dpp_utils.py`
  (`dpp_xor_f32`) for row/bank-masked cross-lane math.

```python
# kernels/attention/pa_decode_tile.py:631 — online-softmax max reduction
for sh in (32, 16, 8, 4, 2, 1):
    pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
```

**Why:** copy atoms move data to/from memory. A register-to-register lane exchange has
no address space and no tile — the layout algebra simply has no vocabulary for it.

> **HIP/CK-Tile → FlyDSL.** `__shfl_xor` / `__builtin_amdgcn_ds_bpermute` / DPP
> modifiers — the warp-reduce you already hand-roll in HIP.

## Inline assembly

`llvm.inline_asm(...)` is the bottom hatch: for an instruction MLIR has no ROCDL op
for, or where you must force a register class or instruction adjacency the compiler
would otherwise disturb. Real, well-commented cases in the tree:

- **AGPR pinning for fp8 MFMA** (`kernels/gemm/fp8_gemm_4wave.py:41`): the asm
  `"v_mfma_f32_16x16x128_f8f6f4 $0,$1,$2,$0"` with constraints `"=a,v,v,0"` pins the
  accumulator in an AGPR across iterations, eliminating the `v_accvgpr_mov`/`s_nop`
  shuffle that the SSA-lowered path emits — the dominant stall on that kernel.
- **`op_sel` MX-scaled MFMA** (`fp4_gemm_4wave.py:229`): the `op_sel`/`op_sel_hi`
  nibble-select immediates on `v_mfma_scale_*` are not surfaced by the ROCDL op.
- **`s_nop` scheduling spacers** (`flash_attn_utils.py:75`) and **adjacent
  `s_waitcnt`+`s_barrier`** (`fp8_gemm_utils.py:201`), where the two instructions must
  stay adjacent in the ISA stream.
- **Missing ISA ops** — `ds_read_b64_tr_b16` (gfx950), and the target-neutral converter
  wrappers in `expr/rocdl/inline_asm.py`, which carry an explicit *"TODO: remove once
  upstream MLIR adds the ROCDL op"* note — a reminder that inline asm here is a
  stopgap, not the intended long-term API.

**Why:** either no MLIR op exists yet, or you need to override the register allocator /
instruction scheduler, which operate below the dialect level.

> **HIP/CK-Tile → FlyDSL.** This is dropping to `asm volatile(...)` in a HIP kernel —
> same tool, same reasons (missing intrinsic, register pinning, scheduling control).

## Scheduling and low-level control

A software-pipelined main loop needs the instructions interleaved in a specific order
(load next tile while MFMA-ing the current one). The LLVM instruction scheduler runs
as an opaque pass *after* lowering, so FlyDSL exposes hints to steer it:

- `fx.rocdl.sched_mfma / sched_vmem / sched_dsrd / sched_dswr(cnt)`
  (`expr/rocdl/__init__.py`) — each emits a `sched_group_barrier` that forces exactly
  `cnt` instructions of one class into a scheduling group. These are the public
  scheduling knobs.
- Below those, `sched_group_barrier` / `s_waitcnt` / `s_setprio` (wave scheduler
  priority) are reachable directly; kernels typically wrap them in local helpers
  (e.g. attention's `_sched_barrier`, `_s_setprio`, `_s_waitcnt`) rather than a
  polished `fx.rocdl.` call.

```python
# kernels/gemm/preshuffle_gemm.py:362 — hot_loop_scheduler(): interleave the main loop
rocdl.sched_dsrd(2); rocdl.sched_mfma(1); rocdl.sched_vmem(1); rocdl.sched_mfma(1)
```

The dualwave attention pipeline (`kernels/attention/flash_attn_gfx950.py:356`) sets
wave priority `1` around MFMAs and `0` around memory ops so two wave groups
time-multiplex — one computes while the other moves data.

**Why:** these are hints to a post-lowering, post-tiling scheduler. The layout algebra
describes *what* to compute, not the *order* the final instructions issue in — there
is nothing at that level to attach a schedule to.

> **HIP/CK-Tile → FlyDSL.** `__builtin_amdgcn_sched_barrier` / `s_setprio` — the same
> scheduling knobs a hand-tuned CK-Tile GEMM reaches for.

## A decision guide

- **Use the atom API** (`fx.copy`, `fx.gemm`, `make_buffer_tensor`, `fx.rocdl.MFMA`) when
  the work is a tile-shaped copy or a tiled GEMM with one accumulator layout. This is
  the default and buys you portability (swap the atom for another subtarget) and
  inspectability (dump the layout ops).
- **Drop to `fx.rocdl.*`** for a single MFMA interleaved with other math, per-lane
  type conversion, cross-lane reductions, or scheduling hints.
- **Drop to `llvm.inline_asm`** only when there is no ROCDL op, or you must pin a
  register class (AGPR) or force instruction adjacency/scheduling.

Dropping lower is a *local* decision: it forfeits the portability and inspectability
the atom layer gives you, so confine it to the few lines that need it and keep the
surrounding kernel in the high-level dialect. When a lowered kernel misbehaves,
Chapter 13 (debugging) shows how to dump and read the `rocdl.*`/`llvm.*` your escape
hatch produced. Chapter 12 next reads three complete kernels that stay on the
high-level path from start to finish.
