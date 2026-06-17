# Learn FlyDSL by Building an FP8 Attention Kernel

A teaching series that reconstructs — step by step — how to build and then optimize a fused FP8
causal attention (FMHA prefill) kernel in **FlyDSL** on **AMD MI308X / gfx942**. Each lesson is a
standalone runnable kernel (`lesson_NN_*.py`) plus a writeup (`lesson_NN_*.md`).

**Audience:** comfortable with GPU matrix kernels (e.g. CK Tile), new to FlyDSL. Lessons cross-map
FlyDSL concepts to CK Tile where useful.

**Philosophy:** build bottom-up (single MFMA → GEMM → scores → softmax → fused attention), then
apply optimizations **one isolated change at a time**, and **profile every change** so you *see* the
bottleneck and *why* the change helped — or didn't. The negative results are lessons too.

## How to run
```bash
HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_NN_name.py     # GPU 2 (0,1 broken; 2-7 usable)
```
Each file self-checks correctness against a torch reference and prints timing.

## The profiling loop (the real skill — used from Lesson 08 on)
Every optimization lesson follows: **measure → classify the bottleneck → change ONE thing →
re-measure → explain the delta.**

1. **Throughput** — `from flydsl.autotune import do_bench`; `ms = do_bench(fn, warmup=10, rep=50)`
   returns median ms; `tflops = flop/1e9/ms`.
2. **Resources (VGPR / LDS / spills)** — dump and read metadata:
   ```bash
   FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 \
     HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_NN.py
   grep -oE 'vgpr_count = [0-9]+|sgpr_count = [0-9]+|group_segment_fixed_size = [0-9]+|vgpr_spill_count = [0-9]+' \
     /tmp/isa/<kernel>_0/19_gpu_module_to_binary.mlir
   grep -cE 'v_mfma|ds_read|ds_write|ds_bpermute|v_perm' /tmp/isa/<kernel>_0/21_final_isa.s
   ```
3. **HW counters (the "why")** — `pmc.txt`:
   `pmc: SQ_WAVES SQ_BUSY_CU_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAIT_INST_LDS SQ_INSTS_LDS`
   then `rocprofv3 -i pmc.txt -- python3 lesson_NN.py`, and query the sqlite `*_results.db`:
   `SELECT n.name, SUM(e.value) FROM rocpd_pmc_event e JOIN rocpd_info_pmc n ON e.pmc_id=n.id GROUP BY n.name;`
   Diagnostic ratios to read off:
   - **LDS-wait / busy** (`SQ_WAIT_INST_LDS / SQ_BUSY_CU_CYCLES`) → LDS-bound?
   - **VALU : MFMA** (`SQ_INSTS_VALU / SQ_INSTS_MFMA`) → drowning in scalar/vector ALU?
   - **achieved GB/s vs 5300** → memory-bound?
   - **grid (workgroups) vs 80 CU** → occupancy / CU-starvation?

### gfx942 / MI308X constants to keep in your head
80 CU · 4 SIMD/CU · **256 VGPR/thread** · **512 VGPR-banks/SIMD** (occupancy ≈ ⌊512 / vgpr_count⌋
waves/SIMD) · **64 KB LDS/SIMD** · ~5.3 TB/s HBM · ~1.3 PFLOPS fp8.

## Curriculum

### Part A — FlyDSL foundations (bf16)
- **00 — hello_flydsl** ✅ kernel/launch model, raw buffer API, CK→FlyDSL dictionary.
- **01 — single_mfma** ✅ one MFMA; the all-important lane↔element layout; verify-don't-trust.
- **02 — gemm_tiles** ✅ K-loop accumulator chaining; wide loads; first `do_bench` (and why tiny
  tiles measure launch latency, not math).

### Part B — Build attention step by step (bf16, single q-tile, 1 wave)
- **03 — qk_scores** ✅ GEMM1 `S = K @ Qᵀ`, scores `[kv, q]`; the layout decision.
- **04 — softmax** ✅ softmax over kv = the cross-lane (`shuffle_xor`) reduction.
- **05 — pv_fused** ✅ GEMM2 `O = P·V`; full fused attention; why the P-transpose *doesn't* bite at
  16×16×16.
- **06 — causal_seqloop** ✅ runtime kv-loop (`range`+`yield`), online softmax, causal mask; the
  GEMM2 orientation bug war-story.

### Part C — fp8 (isolated dtype step)
- **07 — fp8_quant** ✅ e4m3 FNUZ, `cvt_pk_fp8_f32`, descale, p_scale; the REAL P-transpose appears
  (16×16×32 fp8 MFMA); two GEMM1 sub-tiles; the `scf.for`+smem and one-shape-per-process gotchas.

### Part D — Optimizations (each ISOLATED + PROFILED, fp8)
Two molds: **A/B microkernel** (local tricks — the diff is a few lines, runs standalone) and
**driver** (structural tricks — runs the real validated production kernel and reads the bench).
- **08 — multiwave** ✅ driver: BM/waves; BM=128/4-wave sweet spot; occupancy vs starvation.
- **09 — wide_loads** ✅ A/B microkernel: 128-bit loads; the "not bandwidth-bound" neutral result.
- **10 — cooperative_lds** ✅ ISA reader: shared K/V→LDS once; barrier cost; don't stage non-hotspots.
- **11 — v_transpose_lds** ✅ ISA reader: V transposed in LDS → kill the byte-gather.
- **12 — register_p** ✅ ISA+PMC reader: ds_bpermute transpose; **the decisive LDS-wait/VALU reading**.
- **13 — fast_exp2** ✅ A/B microkernel + ISA diff: `rocdl.exp2` removes `v_ldexp`.
- **14 — causal_bound** ✅ explainer: skip masked tiles; why less work ≠ less time when under-occupied.
- **15 — pingpong_prefetch** ✅ ISA reader: LDS double-buffer + prefetch; +0.3 TF (wrong bottleneck).
- **16 — diagonal_pair** ✅ driver: the big structural win (+8–24%); sequential mirror, cheap VGPR.
- **17 — column_V** ✅ driver: the biggest win — column-major V *deletes* the transpose (+20%).

### Part E — Instructive failures (why good ideas regress)
- **18 — neg_128kv_softmax** ✅ amortized a non-bottleneck (VALU wasn't the wall).
- **19 — neg_2rep_pipeline** ✅ VGPR 164→215 → occupancy drop; compiler won't interleave.
- **20 — neg_vperm_transpose** ✅ crushed LDS-wait 102%→27% yet regressed — transpose is irreducible.
- **21 — neg_async_dma** ✅ copied CK's KT=128+DMA config, ~2× slower — codegen quality is the gap.

### Part F — Wrap-up
- **22 — capstone** ✅ full scoreboard vs CK-Tile (141/146) & PyISA (238/291); the remaining gap is
  instruction-scheduling + register-allocation — abstracted away by the DSL — a structural ceiling.
- **22b — all_tricks** ✅ the single self-contained "all wins in one file" kernel: multiwave +
  column-V (transpose deleted) + register-P + causal-bound + fast-exp2, end to end, correctness-
  checked (single-head, cleanest read of the kernel body).
- **22c — multihead** ✅ 22b + a head dimension so `grid = nq·ceil(sq/BM)` fills the 80 CUs. Lifts
  the starved single-head perf to ~38 TF (honest wall-clock); shows occupancy is a *grid-mapping*
  property, plus a do_bench-lied-to-me sanity-check gotcha.

## Status
**COMPLETE.** All 23 lessons (00–22) implemented and validated on GPU 2: a full bf16→fp8 fused causal
attention kernel built from scratch (00–07), 10 isolated+profiled optimizations (08–17, ending at the
61/70 TF column-V kernel), 4 instructive negative results (18–21), and the capstone scoreboard (22).
Each lesson = a runnable `lesson_NN_*.py` + a CK-Tile-framed `lesson_NN_*.md`. Structural lessons use
`_drive.py` to bench the real production kernels in `kernels/`.
