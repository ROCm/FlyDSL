# FP8 Paged FMHA Prefill — FlyDSL Port & Optimization Handoff

**Target:** AMD **MI308X** (gfx942 / CDNA3, 80 CU, 4 SIMD/CU, 256 VGPR/thread,
512 VGPR-banks/SIMD, 64 KB LDS/SIMD, ~5.3 TB/s HBM, ~1.3 PFLOPS fp8 peak).
**Ticket:** AITERKER-112 (HunyuanVideo 3.0, TP8). **Customer cares only about batch size = 1.**
**Date of this handoff:** 2026-06-13.

---

## 0. Repository map — where everything lives

Workspace root: **`/workspaces/amir/`**. The pieces you need:

| path | what it is |
|---|---|
| **`/workspaces/amir/FlyDSL/`** | **The project you work in.** FlyDSL = a Python/MLIR DSL + compiler for AMD GPU kernels. Our port lives in `kernels/`, tests/bench in `tests/kernels/`. This handoff + the optimize prompt are at the FlyDSL root. Read `FlyDSL/CLAUDE.md` for DSL conventions and `FlyDSL/docs/kernel_authoring_guide.md` for the authoring API. |
| **`/workspaces/amir/kernels/pyisa/gfx942/fmha_prefill/f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv/`** | **THE PyISA reference kernel we are porting** (the optimization target). Hand-written instruction-level gfx942 assembly expressed in Python. Files: `kernel.py` (entry + prologue/epilogue), `helpers.py` (the meat: `core_loop`, `gemm_QK`, `gemm_PV`, `process_current_work`, V-transpose, paging, diagonal-pair tiling), `constants.py` (tile sizes: kTileQ=256, kTileKV=128, kWaves=8, mfma 32×32×16), `regs.py`, `init.py`, `metadata.py`. **Study this to understand the structure you're chasing.** |
| **`/workspaces/amir/PyISA/`** | The PyISA *framework/toolchain* itself (the assembler that turns the kernel above into a `.co`). You normally don't edit this — just know the reference kernel is authored against it. |
| **`/workspaces/amir/asm/`** | **Compiled PyISA reference + its benchmark harness.** `fwd_fp8` (executable), `fwd_causal.co` / `fwd_causal_sched.co` (the kernel binaries it loads by relative path), `fwd_fp8.cpp` (harness source), `fwd_causal.s` (disassembly — useful to see the real instruction schedule). Run from inside `asm/`. |
| **`/workspaces/amir/aiter/`** | AMD's aiter library (editable-installed). Provides the **CK-Tile fp8 FMHA** we benchmark against, via `op_tests/test_batch_prefill.py`. |
| **`/workspaces/amir/mlse-tools-internal/performance/kernel_optimization/`** | Optimization toolkit + runbook + FlyDSL skills (`.claude/OPTIMIZATION_RUNBOOK.md`, `.claude/skills/*-flydsl`, `src/stageN_*`). `source setup_env.sh` first. Methodology reference. |
| **`/workspaces/amir/rocm-libraries/`** | ROCm libs checkout (CK-DSL experiments referenced for ideas; gfx950-oriented). |
| `/root/.claude/plans/proud-plotting-cloud.md` | The original multi-phase port plan (Phases 0–5). Historical context. |

**How the three implementations relate:** they all compute the *same* fp8 paged causal FMHA prefill.
**PyISA** (`asm/fwd_fp8` + the `kernels/pyisa/...` source) is the hand-tuned gfx942 target.
**CK-Tile** (via `aiter`) is the production library kernel (fastest at large seq, helped by gfx950
features). **Ours** (`FlyDSL/kernels/fmha_prefill_fp8_8wave.py`) is the FlyDSL port we're optimizing.
The torch reference (`FlyDSL/tests/kernels/fmha_prefill_fp8_ref.py`) defines numerical correctness.

---

## 1. What the task is

Port the hand-written **PyISA** assembly-level FP8 causal FMHA *prefill* kernel
`f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv` to **FlyDSL** (a Python/MLIR DSL),
preserving **all** features, then optimize to match/beat the reference.

**Required features (all implemented & validated):**
- HD=128 (QK and V), FP8 **e4m3 FNUZ** inputs, **bf16** output, causal mask.
- **GQA** (`nq = gqa * nk`; q head `h` uses kv head `h // gqa`). Customer config: nq=8, nk=1.
- **Per-token-per-head Q/K descale** (applied to S after the QK MFMA), **per-head V descale**
  (folded into output normalization).
- **Paged KV**, `vec_k_col_v` layout: K pool `[pages, nk, hd/16, page_size, 16]`,
  V pool **row-major** `[pages, page_size, nk, hd]`. Addressed via flat page-id table `LTD`
  + per-batch `kv_indptr` `LTP`. page_size=16 default (any power-of-2).
- **`p_scale`** per-(batch, q_head): rescales fp8 P for e4m3 range, cancels in O/L (precision-only knob).

---

## 2. Current status (TL;DR)

**Correctness: DONE.** All 9 pytest cases pass (err ≤ 0.055, fp8 noise floor).
**Performance: best = `fmha_prefill_fp8_v8.py` (per-shape dispatch: baseline ≤sq1024, diagonal-pair
`v7` for larger).** Two wins landed: (1) the CU-starvation fix (BM=128/4-wave, +17-23%), then
(2) **diagonal-pair tiling (v7), a further +8-24% at sq≥2048**. Remaining gap to CK is structural (§6).

### Measured performance, bs=1, nq=8, nk=1, causal, fp8 (TFLOPS)

| seq   | base BM=256 | v4 BM=128 | **v7 diag-pair** | **v8 dispatch (best)** | CK-Tile fp8 | PyISA asm |
|-------|-------------|-----------|------------------|------------------------|-------------|-----------|
| 1024  | 4.3         | 5.1       | 4.8 *(loss)*     | **5.1**                | 30          | **36**    |
| 2048  | —           | 12.3      | 15.3            | **15.0**               | 62          | **83**    |
| 16384 | 39.7        | 45.8      | 50.5            | **50.5**               | **141**     | — *(1)*   |
| 32768 | ~45         | 52.9      | 56.9            | **56.9**               | **146**     | —         |

*(1)* The PyISA harness (`asm/fwd_fp8`) always runs a slow CPU reference, so it is only practical
to time up to ~sq2048. `validate=0` just skips the whole run (not a perf-only mode).

**Interpretation:** PyISA wins at small seq; CK-Tile is the production bar at large seq (~145 TF).
The best FlyDSL is now ~2.6× behind CK at large seq (was ~3×). The remaining gap is structural (§6).
**Diagonal-pair (v7) is a clean +8-24% at sq≥2048 but a LOSS at sq1024** (halving grid 64→32 starves
CUs) — hence v8's per-shape dispatch (sq≤1024 → baseline, else → v7). VGPR 164→168 only, LDS
unchanged: cheap, because the mirror tile runs sequentially in one kernel (shared live state).

---

## 3. Files (all under `/workspaces/amir/FlyDSL/`)

### Kernels (`kernels/`)
| file | what it is | status |
|---|---|---|
| `fmha_prefill_fp8.py` | **Correctness reference.** BM=32, 1 wave, simplest. P transposed through LDS. | keep |
| `fmha_prefill_fp8_8wave.py` | **★ CANONICAL BEST ("v4").** BM=128/4-wave default (sweepable `FMHA_NWAVES`). Register-resident P (4× ds_bpermute), cooperative K/V→LDS, V stored transposed, ping-pong double-buffer, register prefetch (OPT3), causal kv-bound, fast `rocdl.exp2`, mask fold (OPT2), 128-bit wide global loads, s_setprio. 16 KB LDS, 164 VGPR. | **use this** |
| `fmha_prefill_fp8_v2.py` | Historical: register-P + ping-pong + fast-exp2 at BM=256. Superseded. | archive |
| `fmha_prefill_fp8_v3.py` | v2 + causal kv-bound at BM=256. Superseded by v4. | archive |
| `fmha_prefill_fp8_v4.py` | Snapshot == older 8wave (BM=256). The live BM=128 work lives in `_8wave.py`. | archive |
| `fmha_prefill_fp8_v5.py` | **Negative result.** 128-KV/softmax cross-GEMM (one softmax per 128 kv). Correct, ~slightly slower. See §5. | keep as doc |
| `fmha_prefill_fp8_v6.py` | **Negative result.** 2-rep software pipeline (PyISA-style). Correct, regressed (VGPR 164→215). See §5. | keep as doc |

> NOTE: "v4" in prose = the BM=128 content now living in `fmha_prefill_fp8_8wave.py` (the file the
> tests and bench import). `_v4.py` on disk is an older BM=256 snapshot.

### Tests & benchmarks (`tests/kernels/`)
| file | purpose |
|---|---|
| `fmha_prefill_fp8_ref.py` | Torch reference: `quantize_per_token_head`, `quantize_per_head`, `pack_paged_cache` (round-trips bit-exact), `gather_from_paged`, `fmha_prefill_reference`. |
| `test_fmha_prefill_fp8.py` | 9 pytest cases, **subprocess-per-shape** (required — the JIT smem global cannot be re-finalized across shapes in one process). Imports `fmha_prefill_fp8` (the BM=32 ref); to test v4/v5/v6 edit the import or use the standalone scripts below. |
| `bench_fmha_compare.py` | **Unified benchmark.** Runs FlyDSL versions + PyISA + CK-Tile in one table. |

### External references
- **PyISA source** (read this to understand the target): `/workspaces/amir/kernels/pyisa/gfx942/fmha_prefill/f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv/{kernel,helpers,constants,regs,init}.py`
- **PyISA executable:** `/workspaces/amir/asm/fwd_fp8` (symbol `_ZN5aiter36fmha_fwd_hd128_fp8_causal_qkptph_vphE`, loads `fwd_causal.co` by relative path → must run with `cwd=asm/`).
- **CK-Tile fp8** (production bar): via aiter `op_tests.test_batch_prefill.run_batch_prefill_per_token_head`.

---

## 4. How to run things

```bash
# Environment: GPU 2 (GPU 0 is broken). flydsl 0.2.0 via pip (do NOT build from source).
# aiter installed editable: cd /workspaces/amir/aiter && pip install -e . --no-build-isolation
#   WARNING: aiter pins flydsl==0.1.9 and DOWNGRADES it. We call flydsl directly, so after
#   installing aiter: pip install --upgrade flydsl==0.2.0

# --- Correctness (all 9 cases, subprocess per shape) ---
cd /workspaces/amir/FlyDSL
HIP_VISIBLE_DEVICES=2 python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py -q

# --- Unified benchmark (FlyDSL versions + CK + PyISA) ---
# Small seqs (PyISA can be timed):
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 \
  python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_8wave --ck --seqs 1024 2048
# Large seqs (skip PyISA — its CPU ref is too slow):
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 \
  python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_8wave --ck --no-pyisa --seqs 16384 32768

# --- PyISA reference directly ---
cd /workspaces/amir/asm && HIP_VISIBLE_DEVICES=2 ./fwd_fp8 causal=1 nheads=8 nheads_k=1 seq_len=2048

# --- Extract VGPR/SGPR/LDS from a compiled kernel ---
rm -rf /tmp/isa
FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  HIP_VISIBLE_DEVICES=2 FMHA_NWAVES=4 python3 <your_runner>.py
grep -o "vgpr_count = [0-9]*\|sgpr_count = [0-9]*\|group_segment_fixed_size = [0-9]*\|vgpr_spill_count = [0-9]*" \
  /tmp/isa/attn_kernel_0/19_gpu_module_to_binary.mlir
# Final ISA assembly is at /tmp/isa/attn_kernel_0/21_final_isa.s

# --- PMC hardware counters (rocprofv3) ---
# pmc.txt:  pmc: SQ_WAVES SQ_BUSY_CU_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAIT_INST_LDS
HIP_VISIBLE_DEVICES=2 rocprofv3 -i pmc.txt -- python3 <bench>.py
# Output: pmc_1/<hash>/<pid>_results.db (sqlite). Query:
#   SELECT n.name, SUM(e.value) FROM rocpd_pmc_event e
#   JOIN rocpd_info_pmc n ON e.pmc_id=n.id GROUP BY n.name;
# (rocprof-compute's `analyze` has a v3->v2 csv bug; read the raw db instead.)
```

`FMHA_NWAVES` env sweeps workgroup size: 4 (BM=128, **default/best**), 8 (BM=256), 2 (BM=64).

---

## 5. Everything we tried (with evidence)

### 5.1 The WIN: BM=128 / 4-wave (CU-starvation fix) — SHIPPED
Made `NWAVES` sweepable; cooperative K/V load loops over `NPASS=ceil(256/NTHREADS)` passes when
threads < 256. BM=128/4-wave is **uniformly fastest**:

| seq | 8w/BM256 (old) | 4w/BM128 (new) | gain |
|---|---|---|---|
| 1024 | 4.3 | 5.0 | +16% |
| 4096 | 18.8 | 23.2 | +23% |
| 16384 | 39.7 | 46.6 | +17% |
| 32768 | ~45 | 52.8 | +17% |

NWAVES=2 (BM=64) falls off at large seq (41.9 @32768). Why 4 wins: smaller workgroup → better
occupancy/latency-hiding, shorter per-wg latency chain. **Note:** at sq1024 the starvation itself
was minor (grid32→64 only +16%) — confirming the small-seq bottleneck is the per-wg MFMA
latency-chain, not grid size.

### 5.2 LDS shrink (removed dead P-LDS scratch) — NEUTRAL, kept as cleanup
v4 still allocated an 8KB (4KB at NWAVES=4) P-LDS scratch that is unused (P is register-resident).
Removed it: LDS 20KB→16KB. **Zero perf change** → confirms LDS is NOT the occupancy limiter. Kept
anyway (free, all 9 tests pass).

### 5.3 v5 — 128-KV per softmax (cross-GEMM) — REGRESSED
Process 4× BN=32 subtiles (128 kv) under ONE softmax: 4 GEMM1 → 1 softmax over 64 S-vals → 4
register-P transpose → 4 GEMM2, one barrier per 128-kv group. **Correct** (err identical to v4).
**Perf: 44.5 @16384 / 21.5 @4096 vs v4's 46.6 / 23.2 — slightly slower at every wave count.**
Lesson: amortizing softmax VALU 4× does nothing because VALU is not the binding wall (see PMC §6);
the coarse one-barrier-per-128kv also hurt load/compute overlap.

### 5.4 waves_per_eu / maxnreg occupancy hints — NO-OP in this wheel
`CompilationContext.compile_hints({"waves_per_eu": N})` and `{"maxnreg": N}` **do nothing** in the
flydsl 0.2.0 wheel — VGPR stayed pinned at 164 for waves_per_eu ∈ {2,3,4} and maxnreg=128. The
backend builds `--amdgpu-waves-per-eu` / `--amdgpu-num-vgpr` into `bin_cli_opts` but those only
reach the EXTERNAL-LLVM codegen path (`FLYDSL_COMPILE_LLVM_DIR`), not the default in-process
`gpu-module-to-binary`. **So VGPR-cap occupancy tuning is unavailable without an external LLVM build.**

### 5.5 v6 — 2-rep software pipeline (the real PyISA structure) — REGRESSED
Factored compute into `compute_tile()` (GEMM1→softmax→regP→GEMM2, no barrier), ran TWO per outer
iteration (tiles in pairs), issued next-pair global loads during the pair's compute, one barrier
per pair. **Correct** (err ≤0.055 incl odd tile counts, sk≠sq, non-causal, batch2, GQA, pscale).
**Perf: 43.6 @16384 / 49 @32768 vs v4's 46.6 / 53 — slower.** Root cause from binary:
- **VGPR 164 → 215** (no spill, but 512/215 = 2 waves/SIMD vs 3) — two tiles' K/V packs + two
  next-pair load sets live at once. Occupancy loss > barrier savings.
- The two reps **serialize** on the online-softmax dependency (rep1 depends on rep0's m/l/o). FlyDSL
  does NOT reorder rep1's independent K-reads ahead across the call boundary, so the intended
  MFMA/LDS overlap never happens. PyISA achieves it via **manual instruction scheduling**
  (issue rep1 K-read early + interleaved `s_waitcnt lgkmcnt(N)`) — a granularity FlyDSL doesn't
  expose, and doing it by hand needs the extra live regs that cost occupancy. The two goals conflict.

### 5.6 Earlier plateau experiments (from prior sessions, BM=256 era)
All plateaued ~11 TF @grid64 / ~21 @grid8192:
- 8 waves vs 1 wave: no change (not occupancy-bound at those shapes).
- Cooperative K→LDS: regressed (added barriers for a non-bottleneck).
- Cooperative V→LDS transposed: killed 64 byte-gathers/tile (clean ISA) but perf flat.
- BN=128 wide tile (BM=256 era): regressed (40KB LDS + ~200 VGPR → 1 wg/CU).
- LDS ping-pong: +0.3 TF. Register prefetch: negligible. Vectorized k_descale: no change.
- s_setprio / barrier reduction / wide global loads: small or neutral. **Wide global loads were
  NEUTRAL here** (kernel is not bandwidth-bound — only ~12.5 GB/s of 5300).

---

## 6. The decisive evidence — why the gap is structural

### 6.1 PMC hardware counters (rocprofv3) — measured on v4 BM=128, sq4096
```
SQ_BUSY_CU_CYCLES = 389.8M    SQ_INSTS_MFMA = 5.41M    SQ_INSTS_VALU = 126.9M
SQ_WAIT_INST_LDS  = 210.4M    SQ_WAVES = 9260
=> LDS-wait / busy = 54%      VALU:MFMA = 23.5:1
```
(Was 56% / 24.5:1 at BM=256 — essentially unchanged.) **The two walls are: (1) LDS-wait = 54% of
busy cycles, (2) VALU:MFMA = 23.5:1.** NOT memory-bound (12.5 GB/s of 5300), NOT compute-bound
(~1–2% fp8 peak), NOT occupancy-starved at large grids.

### 6.2 Why the 54% LDS-wait is largely IRREDUCIBLE on gfx942
GEMM2 (O = Vᵀ@P) needs the kv (contraction) dim **contiguous per lane**, but V is stored
kv-outer (row-major), so kv is strided. **A transpose is unavoidable for either MFMA orientation**
(checked both O[d,q] and O[q,d]: P or V always needs transposing). On gfx942 **every** transpose
mechanism uses the DS (LDS) hardware unit and counts toward `SQ_WAIT_INST_LDS`:
- LDS scatter-write (the V-transpose store: 16 `ds_write_b8`/subtile), AND
- `ds_bpermute` / `ds_swizzle` (the register-P transpose path) — these are DS-unit ops too.

**gfx950+ has `ds_read_tr` (true hardware transpose-load, no extra DS traffic)** — a major reason
CK-Tile reaches 141 TF there. On gfx942 there is NO transpose that avoids the DS unit. So the
54% LDS-wait can only be **hidden by overlap**, not removed.

### 6.3 Resources (from compiled binary, v4 BM=128)
`vgpr_count=164, sgpr_count=71, group_segment=16384, 0 spills, block=256`.
→ VGPR (164) is the occupancy limiter: 512/164 = **3 waves/SIMD** (~3 wg/CU). NOT LDS.
Live state dominated by 64 `o_acc` f32 (4×16) + 32 `q_i64` held across the whole kv loop.

### 6.4 How PyISA actually wins (corrected understanding — IMPORTANT)
**Earlier notes claimed PyISA wave-specializes QK vs PV. That was a MISREAD of the source.** Ground
truth from `helpers.py`/`kernel.py`:
- `core_loop(is_wave47)` runs the **full** gemm_QK → softmax → gemm_PV chain for **both** wave
  groups. Every wave computes both GEMMs. All 8 waves are **q-partitioned** (32 q-rows each) —
  same as our v4.
- The `s_cmp_lt_i32(s_wave_id, 4)` branch only changes which **page-IDs** each group cooperatively
  loads in the prologue (waves 0-3 fetch K pages, 4-7 fetch V pages) — load coalescing, NOT compute
  specialization.

PyISA's real speed levers (which FlyDSL cannot currently match):
1. **2-rep software pipeline** with rep N+1's loads issued during rep N's MFMAs (we tried → v6 →
   regressed due to VGPR blowup + no compiler interleave).
2. **Partial `s_waitcnt lgkmcnt(N)`** interleaved *inside* the MFMA loops (keeps LDS reads in flight
   during MFMA) instead of drain-then-compute. FlyDSL doesn't expose per-instruction waitcnt
   scheduling at that granularity.
3. The **diagonal-pair tg_div=2 tile** (each wg does a tile and its causal mirror) for load balance.

**Conclusion:** the ~3× gap is a consequence of (a) hand-scheduled instruction-level overlap and
(b) gfx942 lacking a HW transpose — neither reachable with FlyDSL's current scheduling control.
This is a real ceiling, not a missing tweak.

---

## 7. Untried / speculative levers (for the next instance)

Ranked by estimated ROI given the evidence above:

1. **Diagonal-pair tiling (tg_div=2).** Each workgroup processes q-tile `t` AND its causal mirror
   `num_tiles-1-t`. Balances the causal triangle (early q-tiles do little KV work, late ones do
   lots) → fuller machine at low grid + 2× work/wg. PyISA uses it; we never ported it. **This is the
   most promising untried algorithmic lever** and is independent of the LDS/scheduling walls.
   Risk: doubles per-wg register/LDS live state — watch occupancy.
2. **External LLVM codegen** (`FLYDSL_COMPILE_LLVM_DIR`) to unlock `--amdgpu-waves-per-eu` /
   `--amdgpu-num-vgpr`. Would let you trade VGPR for occupancy (3→4 waves/SIMD) to hide the DS-wait.
   Requires building/pointing at an external LLVM. Unknown payoff (occupancy may not be the binding
   constraint at large grids — but worth a controlled test).
3. **Reduce VGPR live state** to raise occupancy without external LLVM: e.g. recompute instead of
   holding `q_i64` (32 regs) across the loop, or narrow `o_acc`. Hard without an algorithm change;
   likely small.
4. **Per-shape autotuning**: small seq (grid < 80) wants even smaller workgroups; large seq wants
   max MFMA density. Could ship a tiny dispatch table keyed on seqlen.
5. **gfx950 path** (if hardware available): `ds_read_tr` HW transpose-load removes the DS-unit
   transpose entirely — this is the single biggest structural win, but it's a different target.

What NOT to re-try (proven dead-ends): waves_per_eu/maxnreg hints (no-op in 0.2.0 wheel),
128-KV/softmax grouping (v5), 2-rep pipeline as a plain DSL restructure (v6), wide global loads
(neutral — not bandwidth-bound), cooperative-K-LDS without pipelining.

---

## 8. Prompt for the next optimizing instance

> You are continuing optimization of an **FP8 paged causal FMHA *prefill* kernel** written in
> **FlyDSL** for **AMD MI308X (gfx942 / CDNA3, 80 CU)**. The full history, evidence, and file map
> are in `/workspaces/amir/FlyDSL/FMHA_FP8_OPTIMIZATION_HANDOFF.md` — **read it first**, end to end.
>
> **Current best:** `kernels/fmha_prefill_fp8_8wave.py` (BM=128/4-wave, `FMHA_NWAVES=4`). All 9
> pytest cases pass (`HIP_VISIBLE_DEVICES=2 python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py`).
> Perf bs=1: 5 TF @sq1024, 47 @sq16384, 53 @sq32768. Reference bars: CK-Tile fp8 = 30/141/146 TF,
> PyISA asm = 36 @sq1024 / 83 @sq2048.
>
> **Hard evidence (do not re-derive):** PMC says **LDS-wait = 54% of busy cycles, VALU:MFMA = 23.5:1**;
> NOT memory- or compute- or (at large grid) occupancy-bound. VGPR=164 (occupancy limiter, 3
> waves/SIMD); LDS=16KB (not limiting). On gfx942 **every** transpose (LDS scatter-write,
> ds_bpermute, ds_swizzle) hits the DS unit, so the GEMM2 V/P transpose is the irreducible source of
> the 54% LDS-wait — it can only be HIDDEN by overlap, not removed (gfx950+ `ds_read_tr` would fix
> it but that's a different target). PyISA does NOT wave-specialize (common misconception — it
> q-partitions like us); its speed is hand-scheduled 2-rep pipelining + interleaved partial
> `s_waitcnt lgkmcnt(N)` + diagonal-pair tiling.
>
> **Proven dead-ends (do NOT repeat):** waves_per_eu/maxnreg compile-hints (no-op in flydsl 0.2.0 —
> only work via external LLVM); 128-KV/softmax grouping (`v5`, regressed); 2-rep software pipeline as
> a plain restructure (`v6`, regressed — VGPR 164→215, compiler won't interleave across the
> online-softmax dependency); wide global loads (neutral); cooperative-K-LDS without pipelining.
>
> **Most promising untried lever: diagonal-pair tiling (tg_div=2)** — each workgroup does q-tile `t`
> and its causal mirror `num_tiles-1-t`, balancing the causal triangle. It's the main PyISA trick we
> never ported and is independent of the LDS/scheduling walls. Validate correctness against
> `tests/kernels/fmha_prefill_fp8_ref.py` (use subprocess-per-shape; the JIT smem global can't
> re-finalize across shapes in one process). Watch VGPR/occupancy — wider per-wg work can blow the
> 3-waves/SIMD budget (that's what killed earlier wide-tile attempts).
>
> **Also worth a controlled test:** building/pointing at an external LLVM
> (`FLYDSL_COMPILE_LLVM_DIR`) to unlock `--amdgpu-num-vgpr`, then trade VGPR for occupancy (3→4
> waves/SIMD) to hide the DS-wait.
>
> **Workflow:** lock GPU 2 (`HIP_VISIBLE_DEVICES=2`; GPU 0 is broken). Keep `_8wave.py` as the safe
> baseline; do experiments in a new `_v7.py` (give it a unique `global_sym_name` smem symbol).
> Benchmark with `tests/kernels/bench_fmha_compare.py` (FlyDSL + `--ck` + PyISA). After each change:
> verify correctness (err < 0.06), extract VGPR/LDS from
> `/tmp/isa/attn_kernel_0/19_gpu_module_to_binary.mlir`, and re-take PMC if perf moves. Record every
> variant (hypothesis / correct? / TFLOPS / VGPR / LDS / why) — neutral and negative results are as
> valuable as wins here, because the search space is mostly walls.

---

## 9. FlyDSL authoring gotchas (so the next instance doesn't rediscover them)

- `.to(fp8)` emits a non-lowerable `arith.truncf` → use `rocdl.cvt_pk_fp8_f32(i32, a, b, prev, hi_bool)`
  (4 calls → i64). `.to(bf16)` is fine.
- Load fp8 as i32/i64 dwords, NOT `v8i8` (backend crashes). Wide loads: `vec_width=4` Int32 → bitcast.
- `range(N)` with a python int unrolls/errors → use `fx.range_constexpr(N)` for compile-time loops;
  runtime loops use `range(fx.Index(0), fx.Index(stop), fx.Index(1), init=[...])` + `st = yield ...`.
- LDS store/load indices need `fx.Index`, not `fx.Int32`.
- Causal mask needs the `kv <= qrow + (sk - sq)` offset (fails when sk>sq without it).
- All-masked softmax history → guard: `m_is_neg = m_new < -1e38; safe_m = sel(m_is_neg,0,m_new); corr = sel(m_is_neg,0,...)`.
- Divide-by-zero in epilogue (l_run=0) → `inv_l = sel(l_run<1e-30, 0, 1/l_run)`.
- Do NOT name the jit wrapper `launch` (infinite recursion) — use `run_*`.
- The module-global `SmemAllocator` is `finalize()`d on first launch and cannot re-finalize for a
  different constexpr config in the same process → **test each shape in its own subprocess**.
- `nohup` does NOT survive in this env (exits 143/144) → use tracked background or foreground.

---

## 10. Exact setup & invocations for the reference implementations

### 10.1 Environment (one-time)
```bash
# GPU: always GPU 2. GPU 0 is broken. Box has multiple MI308X (gfx942, 80 CU).
export HIP_VISIBLE_DEVICES=2

# FlyDSL: prebuilt wheel, do NOT build from source.
pip install --upgrade flydsl==0.2.0
python3 -c "import flydsl; from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())"  # gfx942

# torch: 2.10.0+rocm7.2 (has torch.float8_e4m3fnuz). Already installed.

# aiter (provides CK-Tile fp8 FMHA). Editable install REQUIRES --no-build-isolation:
cd /workspaces/amir/aiter && pip install -e . --no-build-isolation
pip install psutil einops pybind11           # aiter deps
# !!! aiter pins flydsl==0.1.9 and DOWNGRADES it on install. We call flydsl DIRECTLY, so AFTER
#     installing aiter, re-upgrade and verify the kernel still works:
pip install --upgrade flydsl==0.2.0

# rocprof-compute extras (only if using its analyze; raw rocprofv3 db needs none of this):
pip install plotext astunparse==1.6.2 plotille plotly pymongo textual textual_plotext \
  textual-fspicker dash dash-bootstrap-components dash-svg kaleido==0.2.1 colorlover \
  tabulate "sqlalchemy>=2.0.42"
```

### 10.2 Run our FlyDSL kernel (correctness + perf)
```bash
cd /workspaces/amir/FlyDSL
# Correctness — all 9 cases, subprocess per shape (REQUIRED, see §9 smem note):
HIP_VISIBLE_DEVICES=2 python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py -q
# (test imports fmha_prefill_fp8 = BM=32 ref; to correctness-check v4/v5/v6 use a standalone
#  runner that imports the desired module — see §10.6.)
```

### 10.3 Run the unified benchmark (FlyDSL + CK-Tile + PyISA, one table)
```bash
cd /workspaces/amir/FlyDSL
# Small seqs — includes PyISA (its CPU ref is tolerable up to ~sq2048):
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 \
  python3 tests/kernels/bench_fmha_compare.py \
    --kernels fmha_prefill_fp8_8wave --ck --seqs 1024 2048

# Large seqs — drop PyISA (CPU ref too slow); keep CK:
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/workspaces/amir/aiter FMHA_NWAVES=4 \
  python3 tests/kernels/bench_fmha_compare.py \
    --kernels fmha_prefill_fp8_8wave --ck --no-pyisa --seqs 16384 32768

# Compare multiple FlyDSL versions head-to-head:
HIP_VISIBLE_DEVICES=2 PYTHONPATH=/workspaces/amir/aiter \
  python3 tests/kernels/bench_fmha_compare.py \
    --kernels fmha_prefill_fp8_8wave fmha_prefill_fp8_v5 fmha_prefill_fp8_v6 --no-pyisa --seqs 16384
```
Flags: `--kernels <mods...>` (any FlyDSL kernel module in `kernels/`), `--ck` (CK-Tile fp8),
`--no-pyisa` (skip PyISA), `--seqs <ints>` (bs=1 seqlens; default 1024/16384/32768).
**First `--ck` run JIT-compiles aiter kernels (~5 min); subsequent runs are fast.**

### 10.4 Run CK-Tile fp8 directly (without our bench wrapper)
CK-Tile is reached through aiter's op-test helper. The exact call our bench uses:
```python
import sys; sys.path.insert(0, "/workspaces/amir/aiter")
from op_tests.test_batch_prefill import run_batch_prefill_per_token_head
r = run_batch_prefill_per_token_head(
    kvcache_layout="vec_k_col_v",   # matches our layout
    table_layout="sglang",          # NOT "vllm"
    batch_size=1, qo_len=SQ, kv_len=SK,
    page_size=64,                   # CK uses ps=64 internally
    num_qo_heads=8, num_kv_heads=1, head_dim=128,
    causal=True, logits_soft_cap=0.0,
    dtype=torch.bfloat16,           # fp8 quant is INTERNAL; uniform-init fp8 errors, so pass bf16
    contiguous_kv=True, seed=42, profile=True, skip_reference=True,
)
print(r["time_us"], r["tflops"])   # returns a dict, not a tuple
```
Gotchas: `dtype=torch.float8_e4m3fnuz` errors ("check_uniform_bounds not implemented") — use
`bf16` (fp8 conversion happens inside). `table_layout` must be `"sglang"`. Run with
`PYTHONPATH=/workspaces/amir/aiter` (or `python3 -m op_tests...`).

### 10.5 Run the PyISA reference directly
```bash
# The executable hipModuleLoads "fwd_causal.co" by RELATIVE path → must run from asm/.
cd /workspaces/amir/asm
HIP_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/opt/rocm/lib \
  ./fwd_fp8 causal=1 nheads=8 nheads_k=1 seq_len=2048
# Parses a line like:  time: 0.103 ms ... gflops: 83353 ... PASSED
# Symbol: _ZN5aiter36fmha_fwd_hd128_fp8_causal_qkptph_vphE ; co files: fwd_causal.co
#         (also fwd_causal_sched.co = snake-schedule variant).
# CAVEAT: ALWAYS runs a slow CPU reference → only practical up to ~seq_len=2048.
#         validate=0 just prints "Skip CHECKING" and skips the WHOLE run (not a perf-only mode).
```

### 10.6 Standalone correctness runner for ANY FlyDSL version (template)
```python
# /tmp/check.py  —  run:  HIP_VISIBLE_DEVICES=2 FMHA_NWAVES=4 python3 /tmp/check.py
import sys; sys.path.insert(0, "tests/kernels"); sys.path.insert(0, "kernels")
import torch, fmha_prefill_fp8_ref as R
import fmha_prefill_fp8_v6 as K          # <-- pick the module to test
HD = K.HD
for (b, sq, sk, nk, gqa, causal, ps, pscale) in [
    (1, 256, 256, 1, 8, 1, 16, 1.0), (1, 1024, 1024, 1, 8, 1, 16, 1.0),
    (1, 512, 768, 1, 8, 1, 16, 1.0), (1, 256, 256, 1, 8, 0, 16, 1.0),
    (2, 128, 128, 2, 4, 1, 16, 1.5),
]:
    torch.manual_seed(0); nq = nk * gqa; sm = 1.0 / HD**0.5
    q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True)
    args = [qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
            c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"), qd.to("cuda"), kd.to("cuda"),
            vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"),
            torch.full((b * nq,), pscale, device="cuda")]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    torch.cuda.synchronize()
    ref = R.fmha_prefill_reference(qf, kf, vf, qd, kd, vd, sm, causal=bool(causal))
    err = (Og.float().cpu() - ref.float()).abs().max().item()
    print(f"  sq{sq} sk{sk} c{causal} -> ERR {err:.4f} {'OK' if err < 6e-2 else 'FAIL'}")
```
> Run multi-shape via the bench (which forks) OR one shape per process — in-process multi-shape
> trips the smem-finalize issue (§9). The 5 shapes above cover odd tile counts, sk≠sq, non-causal,
> batch>1, GQA, p_scale. `run_attn` signature:
> `run_attn(Q, K, V, Qd, Kd, Vd, LTD, LTP, Ps, O, sq, sk, nq, nk, page_size, k_page_stride, v_page_stride, sm_scale, causal, grid_blocks)`.

### 10.7 Standalone perf runner (template)
```python
# /tmp/bench.py  —  HIP_VISIBLE_DEVICES=2 FMHA_NWAVES=4 python3 /tmp/bench.py 1024 16384 32768
import sys; sys.path.insert(0, "tests/kernels"); sys.path.insert(0, "kernels")
import torch, fmha_prefill_fp8_ref as R
import fmha_prefill_fp8_8wave as K
from flydsl.autotune import do_bench
HD = K.HD
def tf(b, sq, sk, nq, ms): return b * nq * (4.0 * sq * sk * HD) / 2.0 / 1e9 / ms
for sq in [int(x) for x in sys.argv[1:]] or [16384]:
    b, sk, nk, gqa, causal, ps = 1, sq, 1, 8, 1, 16; nq = nk * gqa; sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True)
    args = [qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
            c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"), qd.to("cuda"), kd.to("cuda"),
            vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"), torch.full((b*nq,), 1.0, device="cuda")]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    fn = lambda: K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, causal, grid)
    fn(); torch.cuda.synchronize()
    ms = do_bench(fn, warmup=10, rep=50)
    print(f"NWAVES={K.NWAVES} BM={K.BM} sq{sq} grid={grid}  {ms:.3f}ms  {tf(b,sq,sk,nq,ms):.1f}TF")
```
FLOP convention (matches `asm/fwd_fp8` and the bench): causal FMHA = `b*nq*(2*sq*sk*hd + 2*sq*sk*hd)/2`.
