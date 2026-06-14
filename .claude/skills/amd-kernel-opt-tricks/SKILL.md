---
name: amd-kernel-opt-tricks
description: >
  Catalogue of HipKittens-derived optimization tricks for AMD CDNA (gfx94x/gfx95x)
  kernels, mapped to what is/isn't expressible in FlyDSL, plus a discipline for
  deciding whether a hand-codegen trick is even worth applying on top of a DSL
  (often it's redundant because the compiler already does it). Use when optimizing
  a FlyDSL or HIP kernel on AMD GPUs and reaching for low-level scheduling/register
  tricks (setprio, readfirstlane SGPR pinning, MFMA scheduling, adaptive softmax,
  buffer_load_to_lds, chiplet/XCD scheduling). Source: HipKittens arXiv 2511.08083.
---

# AMD CDNA kernel optimization tricks (HipKittens catalogue + FlyDSL reality check)

HipKittens (HazyResearch, arXiv 2511.08083) is the reference for hand-tuned AMD gfx94x kernels.
This skill catalogues its tricks and — critically — records which actually help when you're writing
in **FlyDSL** (a DSL whose LLVM backend already does a lot of what HK does by hand). Measured on
MI308X gfx942 against `kernels/fmha_prefill_fp8_ck.py` (baseline VGPR 165, 61 TF @sq16384).

## ★ RULE ZERO: measure SQ_LDS_BANK_CONFLICT before calling any LDS-wait "latency"
The single biggest FMHA win (+66-73%) came from realizing a 61% LDS-wait was actually 68%
BANK CONFLICTS (un-swizzled K/V LDS rows at stride=128B aliasing all 32 banks), not latency.
FIX = pad each LDS row by 16 bytes (LDS addressing only; keep global strides). Or XOR-swizzle
(st_shape.cuh). A high LDS-wait + high SQ_LDS_BANK_CONFLICT = SWIZZLE/PAD, a cheap ~2x win — NOT a
ceiling. PMC: `pmc: SQ_BUSY_CU_CYCLES SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS`. If
bank_conflict/busy is high (>20%), this is your bottleneck — fix it before anything else below.

## THE GOLDEN RULE: probe before you build, check if the compiler already does it

HK's tricks are written in raw assembly where the dev controls register allocation and instruction
scheduling. In a DSL the compiler already does scalar promotion, MFMA scheduling, and latency
hiding — so **most hand-codegen tricks are redundant and come back neutral.** Before implementing
any trick below:

1. **Upper-bound probe.** Make the *maximally optimistic* (even incorrect) version first — e.g. to
   test "skip the rescale", DELETE the rescale entirely and time it. If the absolute upper bound is
   <2-3%, the real (partial, correct) version isn't worth the machinery. (This killed the adaptive
   softmax rescale: deleting it entirely was only +1.5%.)
2. **Check the compiler already does it.** Dump ISA + resource metadata
   (`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 python run.py`,
   read `/tmp/isa/<kernel>_0/19_gpu_module_to_binary.mlir` for vgpr_count/sgpr_count/spills and
   `21_final_isa.s` for the op mix). If wave-uniform values are already in SGPRs, readfirstlane
   pinning won't help.
3. **Classify the bottleneck (PMC) first** (see the gemm-optimization / kernel-trace skills):
   LDS-wait/busy, VALU:MFMA, GB/s vs peak, grid vs CU count. Only apply a trick that targets the
   *binding* counter.

## Trick catalogue

### A. MFMA scheduling
- **Per-MFMA `s_setprio(1)/(0)` bookends.** HK wraps EVERY mfma. In FlyDSL: `fx.rocdl.s_setprio(1)`
  /`(0)` around each `mfma_*` call. **MEASURED NEUTRAL** — one bookend per loop-iteration is enough;
  the scheduler handles the rest. Low value.
- **`sched_barrier` / `sched_group_barrier` / `sched_mfma` / `sched_dsrd` hints.** FlyDSL exposes
  `fx.rocdl.sched_mfma(N)`, `sched_dsrd(N)`, `sched_barrier`, `sched_group_barrier`. Use to cap
  post-RA reordering between micro-phases. We already use sched_mfma/sched_dsrd; marginal.
- HK extras NOT in FlyDSL: VALU=0x002 / TRANS=0x400 sched_group masks, templated mfma+valu
  interleaving helpers.

### B. Register pressure / occupancy
- **`readfirstlane` SGPR pinning** of wave-uniform addresses (tile bases, LDS offsets that depend
  only on loop-index/block, not lane). FlyDSL: `fx.Int32(fx.rocdl.readfirstlane(fx.typing.T.i32,
  v.ir_value()))`. **MEASURED NEUTRAL / slightly worse (VGPR 165->168)** — the LLVM backend already
  scalar-promotes wave-uniform values. The VGPR ceiling is genuinely-VECTOR live state (accumulators,
  per-lane operand packs), which pinning can't touch. To cut VGPR you need an ALGORITHM change
  (less live state) or external-LLVM regalloc, not readfirstlane.
- **`__launch_bounds__(N, waves_per_eu)`** occupancy hint. HK uses (NUM_THREADS, 2). In FlyDSL the
  0.2.0 wheel's compile_hints (waves_per_eu/maxnreg) DON'T plumb through to codegen — confirmed
  no-op. Needs external-LLVM (`FLYDSL_COMPILE_LLVM_DIR`).
- HK `pinned_register_tile<...,start_vgpr,start_agpr>` (names exact regs incl AGPRs as MFMA inputs
  to dodge HIPCC's AGPR-input refusal, ~19% on attn-bwd): NOT expressible in FlyDSL (no inline-asm
  register naming). This is a real HK lever that the DSL can't reach.

### C. Memory / addressing
- **`buffer_load_to_lds`** (async global->LDS DMA, HK's M0-cursor trick). FlyDSL exposes it but it
  needs a **WAVE-UNIFORM LDS destination pointer** (hw forces M0=readfirstlane(ptr)); a per-lane ptr
  corrupts it. Even correct, it freed only ~1 VGPR for us and KT=128 still busted occupancy ->
  net ~2x SLOWER. The M0-cursor bump-between-calls (s_add_u32 m0) is NOT controllable in FlyDSL.
- **Wide loads** (`buffer_load_dwordx4`): helps only if memory-bound. FMHA attention is NOT
  bandwidth-bound (~12 GB/s of 5300) -> neutral. Check GB/s first.
- HW buffer swizzle in rsrc DW3, SOFF scalar offset: not exposed in FlyDSL.

### D. Compute / softmax
- **`rocdl.exp2(x*log2e)`** instead of generic exp2: removes the v_ldexp range-reduction. FlyDSL:
  `fx.rocdl.exp2(f32t, _ar(x * LOG2E))`. Valid when input <= 0 (always true for softmax s-m). We
  already use it; real but small (VALU-bound only).
- **Adaptive online-softmax rescale**: skip `o_acc *= corr` when corr~1 (max didn't move).
  Upper-bound probe = +1.5% MAX (deleting it entirely) -> NOT worth the wave-vote machinery on FMHA.
- **Wave-vote `__all` / ballot**: `fx.rocdl.ballot` exists; useful for "all lanes agree" without a
  ds_bpermute ladder. Niche.

### E. Layout (the levers that ACTUALLY moved FMHA — see feedback-fmha-perf-lessons)
NOT from HK's micro-codegen list, but these are what actually won on FlyDSL: **column-major V**
(deletes the GEMM2 transpose, +20%), **diagonal-pair tiling** (causal load balance, +8-24%),
**BM=128/4-wave** (occupancy sweet spot, +17%). LAYOUT + ALGORITHM beat micro-codegen in a DSL.

### F. Chiplet scheduling (MI355X 8-XCD only; NOT MI308X)
HK Algorithm 1 (XCD grouping + hierarchical windowed traversal) for per-XCD L2 vs shared-LLC
locality. Only relevant on multi-XCD parts. aiter already implements it in
`aiter/ops/triton/utils/_triton/pid_preprocessing.py` (remap_xcd_chunked = Phase1, pid_grid
GROUP_SIZE_M>1 = Phase2). Tune (C, W) per shape; the "L2-greedy" max-chunk extreme is a TRAP
(starves LLC, slower than naive).

## Bottom line
On a DSL like FlyDSL, the HK micro-codegen tricks (setprio, readfirstlane, adaptive rescale) tend to
be **redundant** because the compiler already does them — verified neutral on our FMHA kernel. The
DSL-reachable wins are **layout + algorithm** (Part E). The genuinely-unreachable HK levers
(pinned_register_tile for AGPR inputs, M0-cursor DMA, launch_bounds occupancy) are exactly the
"codegen control" ceiling that separates a DSL kernel from hand-asm/CK — close that only via
external-LLVM or by dropping to assembly.
