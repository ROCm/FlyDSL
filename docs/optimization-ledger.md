# fp8 FlashAttention Forward — Optimization Ledger (gfx950)

Candidate ledger for the fp8 no-round-trip V parity loop (`rlcr/fa-fp8-parity`).
Every performance claim must name: GPU id+model, branch+commit, exact command,
shape set, dtype (fp8 e4m3fn / bf16 / fp16) + causal mode, warmup/iters, and the
CSV/profile artifact path. Failed candidates are recorded here, not discarded.

## Candidate Entry Format

```
### CAND-<n>: <short title>
- Date / round:
- Hypothesis:
- Change (one candidate):
- Mode(s): HIPREC | FROMBF16 | NATIVE | bf16 | fp16
- GPU / model:
- Branch / commit:
- Command:
- Shapes:
- Warmup / iters:
- Correctness: max_err / min_cos vs SDPA ref (gate: fp8 max_err<5e-2 min_cos>0.98; bf16/fp16 max_err<1e-2 min_cos>0.99)
- Throughput: TFLOPS (and vs aiter_asm / vs HIPREC / vs FROMBF16)
- Artifact path (CSV / profile):
- Outcome: KEEP | REVERT | INCONCLUSIVE
- Notes:
```

## Baseline (Round 0)

Full detail: `results/loop-05-flashattn-fp8-gfx950-baseline.md`.
GPU: MI350X (gfx950) id 0, idle. Branch `rlcr/fa-fp8-parity` @ e921a550. warmup10/iters100.
Headline shape `B=1 S=2048 H=64 D=128 nocausal`, fp8 e4m3fn (except bf16 row).
aiter_asm confirmed native (`fmha_v3_fwd`, `fwd_hd128_fp8.co`, `how_v3_bf16_cvt=0`).

| Mode | TFLOPS | MaxErr | MinCos | Gate | vs asm |
|---|---|---|---|---|---|
| bf16 (ref) | 1079.7 | 4.88e-04 | 0.99999 | PASS | — |
| HIPREC (default) | 979.2 | 2.73e-04 | 1.0000 | PASS | 74.0% |
| FROMBF16 (oracle) | 636.2 | 2.72e-03 | 0.9986 | PASS | 47.2% |
| NATIVE (target) | 811.1 | 2.70e-02 | 0.8863 | FAIL | 60.5% |
| aiter_asm | 1324–1347 | 2.03e-03 | 0.9993 | PASS | 100% |

ATT trace: latency-bound, barrier-dominated. Total stall 57.1% (barrier 22.92%, vmcnt 11.46%).
Artifacts: `.humanize/kernel-agent/baseline/` + `.humanize/kernel-agent/profile/.../flyprof/`.

Key read: NATIVE is already fast (60.5% of asm, > FROMBF16) but FAILS on min_cos 0.886 —
an element-order/permutation defect (layout, not precision), since FROMBF16 passes the same
fp8×fp8 PV math. This is the AC-2/AC-3 target for the next round.

## Candidates

### CAND-1: NATIVE fp8 PV read -> proven B-operand order
- Date / round: 2026-06-22 / round 3
- Hypothesis: NATIVE's min_cos 0.886 failure is a B-operand element-order defect
  (layout, not precision); reading the proven mfma B(V) order will pass the gate.
- Change (one candidate): rewrite `_read_vtf_packs_fp8` from one contiguous 8-key
  i64 load to two 4-fp8 loads 8 keys apart (key = 16*k_substep + 8*(n//4) +
  4*(lane//32) + (n%4); d = 32*dc + lane%32), assembled into the i64 pack.
- Mode(s): NATIVE (FLYDSL_FP8_HIPREC=0 FLYDSL_FP8_PV_NATIVE=1)
- GPU / model: MI350X (gfx950) id 0, idle
- Branch / commit: rlcr/fa-fp8-parity @ 1785951e
- Command: `tests/kernels/test_flash_attn_fwd.py --dtype fp8 [--causal|--no-causal]`
- Shapes: full PR683 DEFAULT_CONFIGS dense fp8; headline B1 S2048 H64 D128
- Warmup / iters: 2/3 (correctness sweep), 3/5 (headline)
- Correctness: headline nocausal max_err 2.03e-03 min_cos 0.99930 (was 0.886);
  causal min_cos 0.99927; full sweep 0 FAIL / 80 PASS / 22 expected split-K ERROR
  (both causal and nocausal). Gate (5e-2 / 0.98) PASS.
- Throughput: headline ~700 TF nocausal (down from the broken 811 TF: two 32-bit
  LDS loads replace one 64-bit). vs HIPREC 953, FROMBF16 626, aiter_asm ~1335.
- Artifact: `.humanize/kernel-agent/operand-oracle/permutation_check.md`. The order is
  proven 3 ways and the checker gates exit 0 on all: (1) a MECHANICAL bf16-layout
  compose (`bf16_layout_model.py`, 8192-entry table, 0 mismatch vs closed form);
  (2) the same-run on-device FROMBF16 dump matches the table over the full 1024 packs
  (dequantized); (3) the post-fix NATIVE on-device read matches the table over all
  1024 packs (raw fp8). Same-run S=512, fail-loud driver.
- Outcome: KEEP (correctness gate AC-3 met). Perf is below HIPREC -> AC-5/AC-6
  optimization (the two-load read is the obvious next target).
- Notes: definitively confirms "layout, not precision". v_descale folded once
  (O-scale fold unchanged). Non-regression: HIPREC/FROMBF16/bf16/fp16 unchanged.

## AC-5 Benchmark (Round 5–6)

MI350X (gfx950) id 0, idle. Branch `rlcr/fa-fp8-parity`. **Measured kernel SHA 6babd86f**
(the NATIVE proven-order read); causal HIPREC/FROMBF16 taken at dd9fa975 which is a
comment/doc-only delta → identical fp8 codegen. warmup 10 / iters 100 (headline) and 50
(coverage). Per-mode subprocess isolation. fp8 e4m3fn. aiter_asm confirmed true native
asm: `fmha_v3_fwd`, hsaco `fwd_hd128_fp8.co` (nocausal) / `fwd_hd128_fp8_causal.co`
(causal), `how_v3_bf16_cvt=0` — NOT CK fallback.
Artifacts: `.humanize/kernel-agent/round-5/*.{log,csv}` + `.humanize/kernel-agent/round-6/*.{log,csv}`.
**Full command/env provenance: `.humanize/kernel-agent/round-6/commands.md`.**

### Headline `B=1 S=2048 H=64 D=128` — TFLOPS (MinCos), gate PASS

| mode | nocausal TF | vs aiter_asm | causal TF | vs aiter_asm |
|---|---|---|---|---|
| FlyDSL NATIVE (fp8) | 726.5 (0.9993) | **53.9%** | 431.6 (0.9993) | **80.4%** |
| FlyDSL HIPREC (fp8, default) | 995.9 (1.0000) | 74.0% | 712.7 (1.0000) | 123.2% |
| FlyDSL FROMBF16 (fp8) | 634.9 (0.9986) | 47.9% | 466.3 (0.9986) | 81.4% |
| aiter_ck (fp8) | 941–948 | ~70% | 718–725 | ~125% |
| aiter_asm (fp8, native) | 1345–1347 | 100% | 572–578 | 100% |

Note: causal aiter_asm (~575 TF) is much slower than nocausal (~1346 TF) on this GPU,
so all FlyDSL causal numbers look relatively closer; HIPREC even exceeds aiter_asm
causal (123%). NATIVE causal (431.6) is below HIPREC (712.7) -> still below the DEC-1
"at least HIPREC" bar. (Denominator note: each "vs aiter_asm" % uses that row's OWN
same-run aiter_asm value — NATIVE causal 431.6/537.0=80.4%, HIPREC causal 712.7/578.3=
123.2%, FROMBF16 causal 466.3/572.5=81.4%; the per-run aiter_asm causal value drifts
537-578 TF across the three subprocess runs, which does not change the gate since every
FlyDSL mode's NATIVE row is below HIPREC regardless.)

### FlyDSL fp8(NATIVE) vs FlyDSL bf16 (dtype speedup), headline nocausal
- bf16 baseline: 1097.2 TF (min_cos 0.99999). NATIVE fp8: 726.5 TF → **0.66× of bf16**.
  (fp8 is currently SLOWER than bf16 here: the correct NATIVE path uses two 32-bit LDS
  loads; the bandwidth win is not yet realized. This is the AC-6 optimization target.)

### Reported coverage (non-gating, DEC-2 conservative), NATIVE nocausal
| shape | NATIVE TF | aiter_asm TF | vs asm | gate |
|---|---|---|---|---|
| B1 S4096 H64 D128 | 770.7 | 1618.7 | 47.6% | PASS (0.9993) |
| B1 S8192 H64 D128 | 784.0 | 1708.9 | 45.9% | PASS (0.9993) |
| B2 S2048 H32 D128 | 728.1 | 1295.1 | 56.2% | PASS (0.9992) |

### Read
NATIVE is correct everywhere (min_cos ~0.999) but ~46–56% of aiter_asm nocausal and
below HIPREC. The two-32-bit-load read (the AC-3 correctness fix) costs throughput vs
the original (incorrect) single-64-bit load. Causal is notably closer (80.4%) because
aiter_asm causal is itself much slower (~575 TF).

## AC-6 Post-fix profile + parity escalation (Round 6)

Named question: *is corrected NATIVE below asm/HIPREC because of LDS/read dependency,
barrier/vmcnt stalls, MFMA issue underutilization, O-store, or grid underfill?*

flyprof on the **actual NATIVE fp8 headline workload** (`--dtype fp8 --no-causal
--batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 --head_dim 128`,
`FLYDSL_FP8_HIPREC=0 FLYDSL_FP8_PV_NATIVE=1`), captured via
`flyprof run ... --invocation "<exact fp8 H=64 command>"`. Workload + mode proven by
`capture.json` (invocation = the fp8 H=64 command) and the LDS bubble's top instruction
being `ds_read2_b32` (the two-32-bit V read that only exists on the NATIVE path).
(`report.json`/`counters.json`/`REPORT.md` print `dtype bf16` — that is flyprof's
roofline-model DEFAULT, invariant to workload, NOT the workload dtype; the round-6 bf16
capture shows the same label. Workload dtype is established by `capture.json`, not these
labels.) Artifact + provenance:
`.humanize/kernel-agent/round-7/flyprof-native-fp8-headline/` (`flyprof/`, `PROVENANCE.md`).

> Supersedes the round-6 profile (`round-6/flyprof-native-20260622-native/`), which the
> round-6 review correctly flagged as INVALID: the flyprof `flash_attn_fwd` recipe ran
> its own default invocation (H=16, no `--dtype fp8`), so it profiled the bf16/default
> path, not NATIVE fp8. The `--invocation` override fixes this.

- Total stall **76.4%** — kernel is **latency/stall-bound**, NOT MFMA-issue-bound.
- barrier **33.74%** (top bubble; `s_barrier` @ flash_attn_gfx950.py:1959)
- LDS **26.95%** (`ds_read2_b32` @ :636 — the two-32-bit V read)
- vmcnt (global-load wait) **17.26%** (`s_waitcnt vmcnt(0)` @ :1236)

**Bottleneck hypothesis (one), now evidence-backed:** corrected NATIVE is below
HIPREC/aiter_asm because the kernel is dominated by cross-wave **barrier (33.74%)** +
**LDS-read (26.95%)** + **global-load (17.26%)** stalls, not by PV MMA throughput. The
LDS bubble is led by `ds_read2_b32` — exactly the two-32-bit V read the AC-3 correctness
fix introduced — so NATIVE's read adds LDS-read dependency without relieving the dominant
barrier bubble, making it slower than HIPREC's single `ds_read_b64_tr_b16` vt read. fp8
and bf16 32x32x16 MMA have equal CDNA4 throughput, so closing the gap requires removing
barriers (33.74%) + a wider single-instruction proven-order V read (to cut the 26.95% LDS
bubble) — well beyond a read-order change.

**Parity escalation gate (DEC-1 conservative default: parity = ≥90% of same-GPU
aiter_asm AND ≥ HIPREC):** NATIVE nocausal 53.9% of asm and below HIPREC (726.5 vs
995.9); causal below HIPREC (431.6 vs 712.7). **Below threshold on both.**

### Formal outcome: BLOCKED (correct, below parity; binding constraint = stall-bound kernel)
Per the plan's binding-constraint escalation gate, STOP here with profile evidence
rather than open-ended tuning. The correct NATIVE path is independently mergeable as
opt-in experimental infrastructure (defaults unchanged); parity is a documented
follow-up. Pause for a human go/no-go on whether to invest in the barrier/global-load
pipeline rework (the only path to parity per the trace).

### CAND-2 (not taken): widen NATIVE V read to one 64-bit load
- Hypothesis: replace the two 32-bit LDS loads (the `ds_read2_b32` driving the 26.95%
  LDS bubble) with one 64-bit load to cut the LDS-read stall.
- Why not taken this round: the proven order is NOT 8 contiguous keys (it is
  [0..3]+[8..11]), so a single 64-bit contiguous load reintroduces the wrong element
  order (the exact defect that capped NATIVE at 0.886). A correct single-instruction
  read needs a strided/transposed 64-bit LDS read (e.g. an 8-bit transpose load whose
  lane permutation matches the proven order) — a real design item, not a quick swap.
  Recorded as the primary parity follow-up behind the escalation gate.

<!-- Append CAND-2, ... as the loop runs. -->

# ===================================================================
# PERF OPTIMIZATION LOOP (max 24 rounds) — plan: refined-plan-perf.md
# ===================================================================

## Perf Loop Baseline (Round 0)

MI350X (gfx950) id 0, idle. Branch rlcr/fa-fp8-parity @ 5862ae42. warmup10/iters100,
per-mode subprocess isolation. Headline B1 S2048 H64 D128. aiter_asm = true fmha_v3_fwd.
Artifacts: .humanize/kernel-agent/perf-round-0/.

| path | nocausal TF | min_cos | vs aiter_asm |
|---|---|---|---|
| NATIVE fp8 | 728.0 | 0.9993 | 55.0% |
| HIPREC fp8 (default, best) | 993.7 | 1.0000 | 74.6% |
| aiter_asm fp8 | ~1322-1332 | 0.9993 | 100% |

Tiers (DEC-P1 default, vs ~1330 asm): T1>993.7 (beat HIPREC); T2 ~1130 (85%);
T3 ~1197 (90%, parity target); T4 ~1330 (100%).

### Valid profiles (flyprof --invocation, fp8 H=64; capture.json proves --dtype fp8)
| bubble | NATIVE (76.3% stall) | HIPREC (64.7% stall) |
|---|---|---|
| barrier (s_barrier) | 33.64% | 33.21% |
| LDS (ds_read2_b32) | 27.18% | not top |
| vmcnt (global load) | 17.07% | 19.42% |
Artifacts: perf-round-0/flyprof-{native,hiprec}/.

### Ranked candidate menu (expected value x confidence)
1. **Barrier reduction (~33%, BOTH modes).** #1 target: shared dual-wave SWP
   s_barrier/sched_barrier/stagger. A win lifts HIPREC (already 74.6%) toward T2/T3 AND
   NATIVE. Highest value; highest regression risk (bf16/fp16 share it) -> strict gate.
2. **Single-instruction proven-order fp8 V read (LDS 27%, NATIVE-only).** Replace
   ds_read2_b32 (two 32-bit) with one strided/transposed read; closes NATIVE->HIPREC gap.
3. **Deeper VMEM prefetch / double-buffer K/V (vmcnt 17-19%, both).**
4. **Raise occupancy / cut register footprint.**

Strategy: HIPREC is closest to parity and barrier-dominated -> attack barrier first
(helps the best path directly), then prefetch; pursue NATIVE LDS read in parallel as the
no-round-trip route. Re-rank after each kept candidate.

## Perf Loop Round 1

### Baseline (corrected, full): see .humanize/kernel-agent/perf-round-1/PROVENANCE.md
Headline B1 S2048 H64 D128, all PASS. nocausal / causal TFLOPS:
NATIVE 721.3/485.1, HIPREC 969.2/696.4, FROMBF16 642.5/460.9, bf16 1112.9/779.2,
fp16 983.8/738.1, aiter_asm ~1300-1351/~570-580. Best fp8 = HIPREC 969.2 (T1 bar).

### CAND-perf-1: disable dual-wave phase stagger (--no-stagger) — REVERT
- Date/round: 2026-06-22 / perf r1
- Hypothesis: barrier is the top stall (~33%); fewer s_barriers (drop the stagger
  open/close extra barrier) might cut it.
- Change: harness toggle --no-stagger (DUALWAVE_SWP_ENABLE_STAGGER=False); no code edit.
- GPU/branch: MI350X id0 / rlcr/fa-fp8-parity @ 5862ae42. warmup10/iters100, headline.
- Result (nocausal, vs baseline): HIPREC 969.2 -> 800.0 TF (-17.5%); NATIVE 721.3 ->
  542.7 TF (-24.8%). Both PASS correctness. Artifacts: perf-round-1/cand1_*.{log,csv}.
- Outcome: **REVERT** (default stagger stays ON).
- Negative-result lesson: the stagger barriers are NET-BENEFICIAL — they overlap the two
  wave groups and hide more latency than they cost. Naive barrier REMOVAL is the wrong
  direction. The ~33% barrier bubble must be attacked by SMARTER scheduling (reduce
  redundant sched_barrier fences, tighten the s_waitcnt that forces the barrier wait,
  raise occupancy so more waves hide the barrier) rather than deleting synchronization.

### Re-ranked menu after CAND-perf-1
1. Single-instruction proven-order fp8 V read (NATIVE LDS 27%) — now promoted to #1:
   NATIVE-specific, lower regression risk, directly cuts ds_read2_b32; the barrier is
   confirmed not-naively-removable.
2. Deeper VMEM prefetch / double-buffer K/V (vmcnt 17-19%, both modes).
3. Raise occupancy / cut register footprint (helps hide the barrier indirectly).
4. Barrier: only via reduced-redundant-fence / waitcnt tightening, not removal.

## Perf Loop Round 2

Strategy pivot (user decision): both fp8 paths are bounded by the SHARED ~33% barrier
stall, not by V-read. A NATIVE V-read fix is ceiling-bounded by HIPREC (969) and cannot
reach parity, so attack the shared barrier/occupancy lever first. (CAND-perf-2 was
re-scoped from "NATIVE single-instruction V read" to the occupancy probe.)

Provenance repaired this round: perf-round-0/flyprof-{native,hiprec}/PROVENANCE.md +
CAND-perf-1 command block in perf-round-1/PROVENANCE.md.

### CAND-perf-2: raise occupancy via waves_per_eu {3,4} — REVERT (both fp8 paths)
- Date/round: 2026-06-22 / perf r2-r3. Headline nocausal, warmup10/iters100, GPU0,
  branch rlcr/fa-fp8-parity @ 5862ae42. Full provenance: perf-round-2/PROVENANCE.md.
- Exact commands: the six literal commands (hiprec/native x wpe{2,3,4}, full shape +
  warmup/iters spelled out, no placeholders) are in
  `.humanize/kernel-agent/perf-round-2/PROVENANCE.md` under "Six literal commands".
- Hypothesis: profile flags occupancy at 4 waves/CU (low); raising waves_per_eu could
  hide the exposed barrier/LDS/VMEM latency.
- Result (nocausal TF / gate):
  - HIPREC: wpe2 983.1 PASS; wpe3 71.2 FAIL(nan); wpe4 28.9 FAIL(nan).
  - NATIVE: wpe2 728.0 PASS; wpe3 67.8 PASS(0.9993); wpe4 30.5 PASS(0.9993).
- Outcome: **REVERT both paths** (default wpe=2 unchanged).
- Negative-result lesson (observation + labeled inference). OBSERVED: raising
  waves_per_eu above the autotune default 2 collapses throughput on both fp8 paths
  (~10-30x); HIPREC additionally produces nan at wpe3/4 while NATIVE stays numerically
  correct (PASS). INFERENCE (not separately proven): the collapse is most likely
  register/resource pressure — the profiler recommends "Raise occupancy by cutting
  register footprint" and warns "Do NOT force maxnreg ... spills ~4.5x". NATIVE staying
  PASS while HIPREC goes nan narrows the likely problem to HIPREC-side resource pressure,
  but no VGPR-count/code-object dump was taken, so a datapath bug at high wpe is not
  separately ruled out. Six literal commands + results: perf-round-2/PROVENANCE.md.
  Takeaway: the wpe hint is exhausted; the remaining occupancy route is REDUCING register
  footprint (stage long-lived operands through LDS to free VGPRs) so the compiler
  naturally fits more waves — a real kernel-logic change (CAND-perf-3).

### Status: two shared-pipeline knobs exhausted (barrier removal r1, occupancy hint r2)
Both easy/low-risk shared-pipeline knobs are negative. Remaining parity levers are all
deeper kernel-logic changes:
1. Reduce register footprint via async-copy A/B tile staging through LDS (frees VGPRs ->
   compiler fits more waves -> hides the shared barrier). Highest expected value now.
2. Remove genuinely-redundant sched_barrier fences (audit, not bulk removal).
3. Deeper VMEM prefetch / double-buffer K/V (vmcnt 17-19%).
4. NATIVE single-instruction V read (LDS 27%) — ceiling = HIPREC; do only as a NATIVE
   local win, not for parity.

## Perf Loop Round 4

### CAND-perf-2 docs: closed to literal-exact standard (review r3)
Six literal commands + narrowed nan wording in perf-round-2/PROVENANCE.md, ledger, and
bitlesson. nan-at-high-wpe is now stated as observed-on-HIPREC + labeled inference
(register/resource pressure), not as proven "not a datapath bug".

### CAND-perf-3: reduce fp8 long-lived VGPR pressure (stage Q operand via LDS)
- Feasibility (from perf-round-0/flyprof-hiprec report.json — occupancy limiter check):
  vgpr_alloc=112, accum_vgpr=0, waves_per_simd_by_vgpr=4, occupancy_pct=25, waves_per_cu=8;
  lds_bytes=65536 of 163840 (lds_wgs_per_cu=2). => Occupancy is VGPR-LIMITED (4 waves/SIMD
  at 112 VGPR; next tier 5 waves needs <=96 VGPR, ~16 VGPR cut). LDS is NOT the limiter
  (huge headroom). The fp8 Q operand = 8 i64 packs = 16 VGPR, closure-captured live across
  the ENTIRE inner loop (q_all_scaled_bf16 defined in prologue, read by _mma0 every tile).
  => Moving Q to LDS trades abundant LDS for the exact ~16 scarce VGPR needed to gain a wave.
  Well-targeted, not speculative. Risk: re-reading loop-invariant Q from LDS each tile adds
  LDS traffic (already a bottleneck); net effect must be measured.
- Status: feasibility established; implementation in progress this round.

### CAND-perf-3 result: stage fp8 Q operand via LDS — REVERT (env-gated off)
- Implemented FLYDSL_FP8_QLDS=1 (fp8-only, const_expr-gated, default OFF): adds a 32 KiB
  Q LDS region; the prologue stores the 8 scaled Q i64 packs per lane; _mma0 re-reads
  them per tile instead of carrying the closure-captured q_all (16 VGPR) live all loop.
- GPU/branch: MI350X id0 / rlcr/fa-fp8-parity. headline nocausal, warmup10/iters100.
  Artifacts: perf-round-4/cand3_hiprec_qlds.{log,csv} + flyprof-hiprec-qlds/.
- Correctness: PASS (min_cos 1.0000, identical to baseline) — LDS staging is numerically
  exact.
- Throughput: 948.7 TF vs ~969 baseline (-2%, within noise/slightly worse).
- PROFILE (the decisive evidence): vgpr_alloc UNCHANGED at 112; waves_per_simd_by_vgpr
  still 4; occupancy 25%. Barrier bubble shrank 33.6%->29.2% but total stall ~flat
  (65.9% vs 64.7%) — the LDS round-trip added latency offsetting it.
- Outcome: **REVERT** (env flag stays OFF; default codegen unchanged).
- KEY STRUCTURAL FINDING: occupancy is VGPR-bound (112) but the Q operand is NOT the
  binding consumer. Freeing Q's ~16 VGPR did not drop vgpr_alloc below the 96-VGPR
  threshold for the next wave tier — the 112 ceiling is set by OTHER live values (K
  packs, the 4x v16 PV accumulators, softmax running state). So register-footprint
  occupancy work must target the ACCUMULATORS / K live-set, not Q. Naive "move a big
  operand to LDS" does not lower vgpr_alloc if that operand wasn't on the critical
  live-range peak.

### Re-ranked menu after CAND-perf-3
1. Reduce the binding VGPR consumer: the 4x v16f32 PV accumulators + K live set (not Q).
   Hard / high-risk; may not be DSL-addressable without restructuring the MMA loop.
2. Deeper VMEM prefetch / double-buffer K/V (vmcnt 17-19%) — independent of occupancy.
3. NATIVE single-instruction V read (LDS 27%) — NATIVE-local win only (ceiling=HIPREC).
Assessment: three of the profile-ranked levers (barrier removal, occupancy hint, Q-LDS
footprint) are now negative. The shared barrier appears structurally bound by VGPR
occupancy that is NOT cheaply reducible. Approaching a likely BLOCKED outcome on parity
unless prefetch (vmcnt) yields a real gain.

## Perf Loop Round 5

Cleanup: removed the rejected CAND-perf-3 QLDS prototype entirely from the kernel (it was
mode-unsafe: _FP8_QLDS enabled for any fp8 mode but the qlds LDS field existed only in the
_PV_USE_VT storage variant, so NATIVE+QLDS would fail to build). Kernel restored to the
clean baseline; all modes (HIPREC/NATIVE/bf16/fp16) build + PASS. Closed the residual
CAND-perf-2 doc nits (ledger now points to the 6 literal commands in
perf-round-2/PROVENANCE.md; bitlesson nan wording fully qualified).

### CAND-perf-4: deeper VMEM prefetch / double-buffer K/V — REVERT (measured, r6)
- Hypothesis: vmcnt (global-load wait) is 17-19%; deeper prefetch could hide it.
- Investigation (no perf claim; structural assessment):
  - The kernel is ALREADY double-buffered: NUM_PREFETCH_K=2, prefetches K(+1) and V ahead
    of compute, with the buffer parity HARDCODED as tile%2 and fully-unrolled
    prologue/epilogue clusters + stagger barriers tied to even/odd wave groups.
  - The s_waitcnt vmcnt values are already tuned to the exact DMA counts
    (_waitcnt_vm_n(NUM_DMA_K+NUM_DMA_V)), i.e. no over-conservative waiting to remove —
    the vmcnt stall is genuine HBM latency, not slack in the waitcnt.
  - Going deeper (NUM_PREFETCH_K=3, triple-buffer) is NOT a contained candidate: it
    requires rewriting the whole software-pipeline schedule (buf cycling, LDS sizing for
    all 3 modes, the unrolled cluster sequence, the stagger barrier parity), touches the
    SHARED bf16/fp16 path (high regression risk), and the profile gives no reason to
    expect it reaches parity (already double-buffered + VGPR-bound occupancy caps the
    overlap available).
- r5 recorded a structural assessment only; r6 MEASURED a contained probe (see below),
  so this lever is now falsified by measurement, not assertion.

### Convergence assessment -> route to formal outcome (AC-4)
All four profile-ranked levers are now exhausted with evidence:
1. barrier removal (r1): REVERT — stagger barriers are net-beneficial.
2. occupancy hint wpe (r2-r3): REVERT — wpe=2 is the sweet spot; higher spills/collapses.
3. register footprint via Q-LDS (r4): REVERT — Q is not the binding VGPR consumer
   (vgpr_alloc unchanged at 112; accumulators/K set the ceiling).
4. deeper VMEM prefetch (r6): REVERT (measured) -- the contained K-hoist probe broke
   correctness (min_cos 0.9626) and didn't improve TF; deeper prefetch needs the
   triple-buffer rewrite (deferred).
Best fp8 remains HIPREC ~969-983 TF (74% of same-GPU aiter asm); no kept speedup.
The binding constraint is VGPR-limited occupancy on an already-double-buffered,
barrier-overlapped, exactly-waitcnt'd pipeline — the ~26-pt gap to hand-written aiter asm
appears structural to the DSL-generated kernel.

> SINGLE CONVERGENCE DIRECTION (supersedes any earlier "route task11 next" wording in
> this section): the next lever is CAND-perf-5 (tile/block-shape, user-requested,
> independent of the fixed-tile ranking). task11 formal AC-4 outcome (BLOCKED) follows
> ONLY IF CAND-perf-5 is negative or explicitly declined. The triple-buffer rewrite is a
> queued high-risk theoretical lever for task11/user decision, neither completed nor
> silently deferred.

### Direction update (user decision r5): pivot to tile/block-shape autotune before BLOCKED
The four profile-ranked levers assumed the FIXED BLOCK_M=256 / BLOCK_N=64 tile. A
different tile/block config is an INDEPENDENT lever (changes the occupancy/barrier/LDS
balance wholesale), not a retry of an exhausted one. CAND-perf-5 (next round): explore
tile/block-shape variants (BLOCK_M alt path, BLOCK_N, head_dim tiling) and any @autotune
Config dimensions the builder exposes, measured for the best fp8 path. KEEP only via the
full gate; this is the last independent hypothesis before routing the formal BLOCKED
outcome (the triple-buffer rewrite is explicitly deferred as too invasive for the value).

## Perf Loop Round 6

### CAND-perf-4 (K-prefetch hoist) — REVERT (measured)
Implemented a default-off FLYDSL_FP8_KPREFETCH_HOIST gate that issued the next-K async
load one cluster earlier in the double-buffer schedule (after the K-read barrier instead
of in the following memory cluster), per the review's contained-probe spec. Measured,
then removed.
- Baseline (this SHA): HIPREC nocausal 967.2 PASS (1.0000), causal 699.2 PASS.
- Candidate: HIPREC nocausal 972.7 FAIL (min_cos 0.9626); throughput within noise.
- Mechanism: the hoisted load overwrites a K buffer still read by later compute clusters
  (double-buffer parity assumes the load lands in the LATER cluster). Cannot move the
  prefetch earlier without a third buffer.
- Outcome: REVERT, candidate code removed; default codegen byte-identical. Provenance:
  perf-round-5/PROVENANCE.md + baseline_*/cand4_*.{log,csv}.
- This is the MEASURED falsification the prior round lacked: "deeper prefetch within the
  existing double buffer" is impossible (correctness), so the only deeper-prefetch route
  is the triple-buffer rewrite (deferred as too invasive).

### Convergence status (now fully measured)
All four profile-ranked levers are falsified BY MEASUREMENT (barrier removal r1,
occupancy hint r2-3, register footprint r4, K-prefetch hoist r6). Best fp8 = HIPREC
~967-983 TF (~72-74% of same-GPU aiter asm); zero kept speedup. Remaining hypotheses:
CAND-perf-5 tile/block-shape autotune (user-requested, independent of the fixed-tile
ranking) and the triple-buffer rewrite (deferred). Per AC-4, route to formal outcome only
after CAND-perf-5 is tried or explicitly declined.

## Perf Loop Round 7

CAND-perf-4 evidence closed: perf-round-5/PROVENANCE.md literal commands (no <shape>);
perf-round-5/cand4_khoist.patch snapshot saved; single convergence direction stated.

### CAND-perf-5: tile/block-shape — feasibility + measured shape sweep — NEGATIVE
- Feasibility finding: the fp8 path is built ONLY by build_flash_attn_dualwave_swp_module
  with a HARDCODED BLOCK_M=256 / NUM_WAVES=8 (BLOCK_SIZE=512). The dual-wave SWP schedule
  (2 groups of 4 waves, stagger parity, LDS sizing, q-row map) derives from NUM_WAVES=8
  across 46 structural references. The BLOCK_M=128 path exists ONLY in the generic
  bf16/f16 launcher (flash_attn_generic.py:366), NOT wired for fp8's raw-i64 MMA operands.
  => A BLOCK_M=128 fp8 tile is a MAJOR restructure (comparable to the triple-buffer
  rewrite), not a contained candidate.
- Measured probe (so this is falsified by measurement, not assertion): is the fp8-vs-asm
  ratio shape-dependent / is there a better operating point? HIPREC fp8 vs aiter_asm,
  nocausal, warmup10/iters50 (perf-round-7/):
  | shape | FlyDSL TF | vs aiter_asm |
  | B1 S2048 H64 | 969.6 | 72.2% |
  | B1 S8192 H64 | 998.2 | 58.5% |
  | B1 S16384 H64 | 896.6 | 51.2% |
  The ratio only gets WORSE at longer sequences (aiter asm scales better); the headline
  S2048 is FlyDSL's BEST relative regime. No operating point reaches parity.
- Outcome: **NEGATIVE** — no contained tile knob for fp8, and no shape reaches a better
  ratio. The tile-shape lever cannot close the gap without a major NUM_WAVES restructure.

### CONVERGENCE -> formal outcome candidate: BLOCKED (correct, below parity)
Levers tested and result:
1. barrier removal (r1): REVERT (measured) — barriers net-beneficial.
2. occupancy hint wpe (r2-3): REVERT (measured) — wpe=2 sweet spot.
3. register footprint Q->LDS (r4): REVERT (measured) — Q not the binding VGPR consumer.
4. K-prefetch hoist (r6): REVERT (measured) — breaks double-buffer parity.
5. tile/block shape (r7): NEGATIVE (measured shape sweep) — no better operating point;
   BLOCK_M=128 fp8 needs a major restructure.
Remaining theoretical levers, both MAJOR restructures (not contained candidates):
(a) triple-buffer (NUM_PREFETCH_K=3); (b) BLOCK_M=128 fp8 port (NUM_WAVES=4).
Best fp8 = HIPREC ~969 TF, 72.2% of same-GPU aiter asm at the headline (FlyDSL's best
regime); correctness fully holds across all modes. The ~28-pt gap to hand-written aiter
asm is structural to the DSL-generated dual-wave kernel given VGPR-limited occupancy on an
already barrier-overlapped / exactly-waitcnt'd / double-buffered pipeline.
RECOMMENDATION (for task11 / user): declare BLOCKED with this evidence; the two remaining
levers are multi-round high-risk rewrites of shared pipeline code with no profile evidence
they would reach parity. Pursue only on explicit user opt-in.

### Direction (user decision r7): attempt BOTH remaining major rewrites
User opted to spend the remaining rounds attempting both high-risk levers rather than
declaring BLOCKED now: (1) triple-buffer (NUM_PREFETCH_K=3) FIRST (targets the measured
17-19% vmcnt; the K-hoist proof showed deeper prefetch needs a 3rd buffer), then
(2) BLOCK_M=128 fp8 port (NUM_WAVES=4 occupancy point). Accepted as maximum-effort
exploration with high revert risk; each rewrite is multi-round and is developed behind a
default-off gate / on a branch-safe path so the default kernel stays correct+unchanged at
every round boundary. BLOCKED is deferred until both are attempted or the round budget
(24) is reached. CAND-perf-6 = triple-buffer (next, multi-round).

## Perf Loop Round 8

Evidence cleanup: cand4_khoist.patch relabeled "reconstructed sketch" (never committed,
so no exact git diff); perf-round-7/PROVENANCE.md added (shape sweep is operating-point
evidence at iters50, not the tile candidate; CSVs/labeling clarified).

### CAND-perf-5/7: BLOCK_M=128 fp8 tile — ATTEMPTED, MEASURED nan -> needs multi-round rewrite
- Implemented a default-off FLYDSL_FP8_BLOCK_M128 gate parameterizing BLOCK_M (128 ->
  NUM_WAVES=4, BLOCK_SIZE=256; grid + flat_work_group_size auto-scale). Default BLOCK_M=256
  path byte-identical (verified PASS 969.9).
- Outcome (perf-round-8/cand5_m128_build.log): gate-on BUILDS + RUNS but FAIL (nan).
- MEASURED FACT: gate-on builds+runs but fails the first nocausal fixed fp8 gate with
  nan. SOURCE DIAGNOSIS (consistent with the nan, not separately measured): the dual-wave
  SWP scheme assumes exactly 8 waves = 2 groups of 4
  -- the stagger split is hardcoded `wave_id_uni / 4` (line ~449), and n_in_tile =
  n_in_warp*NUM_WAVES+wave_id, the q-row map, LDS line strides, and the manually-unrolled
  8-cluster schedule all bake in the 8-wave layout. At 4 waves the two-group multiplexing
  degenerates (group 1 empty) -> nan. Parameterizing the tile constants is necessary but
  NOT sufficient.
- Decision: the gate was REMOVED (a default-off path that produces nan is not worth
  shipping); a +5-line NOTE comment records the finding in the kernel. A correct M128 fp8
  port = re-derive the wave-group/stagger/q-row/cluster logic for 4 waves = a multi-round
  rewrite (the user-approved "attempt both" track).

### Status: both remaining levers now MEASURED to require multi-round rewrites
- triple-buffer (r7): LDS-feasible (~154/160 KB) but would halve WG occupancy (2->1/CU).
- BLOCK_M=128 fp8 (r8): builds but nan; needs the wave-group scheme re-derived for 4 waves.
Five contained levers measured-negative; both major levers now have measured
feasibility/failure evidence. Best fp8 = HIPREC ~969 TF (72% of aiter asm). Per user
decision, next rounds attempt the actual rewrites (triple-buffer first, then M128),
each multi-round behind a default-off path; BLOCKED deferred until both are attempted or
the round budget is reached.

## Perf Loop Round 10

Minor: perf-round-8/PROVENANCE.md "Exact diff" -> "Reconstructed rerun diff" (the M128
gate was reconstructed+reran in r9).

### CAND-perf-6 risk probe: NUM_PREFETCH_K=3 (3rd KV buffer alloc only) — ENCOURAGING
- Default-off FLYDSL_FP8_TRIPLE_KV gate sets NUM_PREFETCH_K=3 (LDS_KV_TOTAL 34048->51072
  bf16-equiv; build-verified), schedule unchanged so the 3rd buffer is allocated-but-unused
  (correctness unaffected, isolates the occupancy cost).
- Result (headline nocausal, warmup10/iters100): baseline 970.4 PASS -> probe 977.4 PASS
  (within noise). NO regression. Artifacts: perf-round-10/{baseline,cand6_triplekv}_nocausal.*.
- FINDING (r11 authoritative metadata): kernel group_segment_size grew 68864 -> 103296 B
  (exactly one extra KV buffer; from big_out_results.json .kd, NOT the stale report.json
  lds_bytes), SGPR/VGPR unchanged (112/120), and headline throughput held 970 -> 977 TF
  (PASS). At 103296 B/WG only 1 WG fits per CU (vs 2 at baseline), yet throughput did not
  regress -> the binding occupancy factor is wave/VGPR, not WG-per-CU LDS. So the round-7
  fear that a 3rd buffer would crater throughput via WG-occupancy is not borne out here.
- IMPLICATION: the occupancy objection to triple-buffering is REMOVED. The full
  triple-buffer SCHEDULE rewrite (prologue fills 3 buffers; steady-state cycles 0->1->2 to
  add prefetch distance against the 17% vmcnt bubble) can net-win and is the next candidate
  (CAND-perf-6 full). Probe gate removed; default NUM_PREFETCH_K=2 unchanged.

## Perf Loop Round 12

CAND-perf-6 probe provenance consolidated into one coherent record
(perf-round-10/PROVENANCE.md: literal split baseline/probe benchmark + flyprof commands,
authoritative group_segment_size metadata, fact+inference conclusion; stale overclaim
lines removed).

### CAND-perf-6 (triple-buffer) pre-rewrite analysis -> TARGETING CAUTION (not a falsification)
Before the schedule rewrite, two analyses:
1. Codex (analyze-routed) on tractability: the full step=3/12-cluster K+V triple-buffer is
   multi-round, high blast radius. A K-ONLY triple-buffer is the minimal variant, BUT
   "limited upside if the exposed vmcnt wait is mostly V-side."
2. vmcnt bubble source (baseline flyprof report.json): the 19.93% vmcnt s_waitcnt is at
   flash_attn_gfx950.py:1224; its wait SET includes BOTH the K/V DMA atom (line 659,
   buffer_load_dwordx4 ... lds) AND the HIPREC V-dequant load (line 1217, _stage_vt_dequant_fp8).

TARGETING CAUTION (corrected r13; the earlier "REDIRECT/NOT pursued" overclaimed): a
K-ONLY triple-buffer is poorly targeted because the wait set is not purely K. It does NOT
follow that the full K+V rewrite is dead -- a full K+V/V-dequant-aware triple-buffer can
move BOTH the line-659 DMA and the line-1217 V-dequant staging earlier. So the full
CAND-perf-6 K+V rewrite remains ACTIVE and is attempted in r13; only a K-only variant is
ruled out. (The de-risk probe r10-11 stands: the 3rd buffer's LDS is free.)

## Perf Loop Round 13

Round-12 ledger overclaim corrected above (REDIRECT -> TARGETING CAUTION; full K+V rewrite
remains active).

### CAND-perf-6 full triple-buffer: INFRASTRUCTURE landed (schedule rotation = remaining)
- Added default-off gate FLYDSL_FP8_TRIPLE_SCHED. Under it: NUM_PREFETCH_K=3;
  DUALWAVE_SWP_K_BUF_BASE / V_BUF_BASE generalized to NUM_PREFETCH_K entries (K0/K1/K2,
  V0/V1/V2); lds_scope_names generalized to add lds_k2/lds_v2 alias scopes. Gate-off path
  byte-identical (verified: default fp8 PASS 955).
- Gate-on with the EXISTING 2-buffer schedule (buf2 allocated-but-unused): builds + PASS
  (961 TF, min_cos 0.99999) -- confirms the infra (bases/scopes/sizing) is correct and the
  3rd buffer integrates cleanly.
- REMAINING (the hard part, next round): the steady-state SCHEDULE rotation to actually USE
  buf2 for deeper K+V prefetch. The main loop is manually unrolled into 8 clusters that
  process 2 tiles/iter with buf ids hardcoded 0/1 (tile%2 parity); the prologue fills buf0,
  buf1 + prefetches buf0-tile2; the epilogue drains in 4 stages (e3/e7/e11/e13). A correct
  3-buffer rotation needs either a 6-tile static super-iteration (LCM(2-tile-step, 3-buf))
  or explicit compile-time 3-phase clusters, with the prologue filling 3 and a matching
  epilogue, prefetching K AND V one tile deeper (to overlap both the line-659 DMA and the
  line-1217 V-dequant in the vmcnt wait set). High blast radius; correctness-first.
- Status: infrastructure committed (default-off, correct); schedule rotation is the active
  next-round work. No throughput claim yet (the unused 3rd buffer adds no prefetch depth).

## Perf Loop Round 14

### CAND-perf-6 full triple-buffer rotation: SUPERSEDED (closed by profile)
Re-reading perf-round-10/flyprof-baseline/flyprof/report.json this round changed the
mechanism. The vmcnt bubble (19.93%, rank 2) is `s_waitcnt vmcnt(1)` at
flash_attn_gfx950.py:1224 INSIDE _stage_vt_dequant_fp8, and its waits_on set is line 659
(K/V DMA atom) AND line 1217 -- the dequant's OWN buffer_load_dwordx2. The steady-state
schedule already stages bf16 V ~6 clusters before it is read (V prefetch is ALREADY deep),
so the bubble is the dequant's load->convert adjacency at STAGING time, not shallow
prefetch. This is exactly why the r10 alloc-only buf2 probe measured +0 TF: a deeper LDS
buffer cannot decouple a staging-internal self-wait. -> The deeper-buffer ROTATION is the
wrong lever for this bubble (per BL target-the-actual-bubble); the FLYDSL_FP8_TRIPLE_SCHED
infrastructure (r13) stays in the tree but is not the vmcnt fix. Mechanism refined to the
correctly-targeted contained probe below.

### CAND-perf-7 dequant load-hoist: REVERT (measured neutral; compiler already does it)
- Hypothesis: in _stage_vt_dequant_fp8, issue BOTH d-iter buffer_load_dwordx2 before the
  cvt_pk_f32_fp8 converts so the loads overlap and the cvt-phase vmcnt wait resolves once.
- Change: env-gated default-off FLYDSL_FP8_DEQ_HOIST (HIPREC fp8 only); split the per-d
  loop into load-all-then-convert-all. Numerically identical. Snapshot:
  .humanize/kernel-agent/perf-round-14/cand7_deq_hoist.patch.
- Correctness gate-on: HIPREC nocausal PASS min_cos 1.0000 (965.0 TF); causal PASS
  min_cos 1.0000 (693.6 TF). (perf-round-14/gateon_{nocausal,causal}.{log,csv})
- Throughput: paired nocausal warmup10/iters100, gate-off ~983 TF mean vs gate-on ~983 TF
  mean; bands overlap -> within run-to-run noise, NO T1 improvement.
- DECISIVE ISA evidence (perf-round-14/isa_structural_diff.txt): the gate-on final ISA is
  instruction-order-identical to the default; register/imm-normalized diff is EMPTY except
  scalar reg allocation (s[24:27]->s[28:31]). The two buffer_load_dwordx2 are ALREADY
  grouped ahead of the cvt_pk_f32_fp8 in both. The compiler already performs this hoist, so
  the source reorder is a no-op at ISA level -> measured-neutral by construction.
- Default codegen byte-identical: pre-r14 (3d5ccb87) default 21_final_isa.s == r14 gate-off
  21_final_isa.s (diff -q BYTE-IDENTICAL). Committed change is comment-only; gate removed.
- Non-regression: bf16 nocausal PASS 1110.1 TF (>= 1097 baseline). fp8 default unchanged.
- Outcome: REVERT. Valid AC-4 negative evidence: the profile-named vmcnt bubble's
  intra-staging load ordering is already optimal in codegen, so a source-level hoist cannot
  move it. The residual vmcnt is load latency + shared K/V DMA under VGPR-limited occupancy
  -- consistent with the structural BLOCKED case.

## Perf Loop Round 15 — task11 FORMAL OUTCOME: BLOCKED (AC-4)

Analyze task (routed to Codex gpt-5.5:xhigh via /humanize:ask-codex). Verdict + the exact
evidence package submitted are saved durably at
.humanize/kernel-agent/perf-round-15/{task11_codex_verdict.md, task11_question.md}.

### Formal outcome: BLOCKED (not NO-GO)
The fp8 throughput-parity objective is meaningful, but this incremental RLCR loop has
exhausted the cheap, falsifiable levers; what remains is a structural redesign, not another
ranked candidate. Best fp8 stays HIPREC 996 TF (~74% of same-GPU aiter asm); no kept
speedup beats T1 after 14 candidate rounds.

### Named binding constraint
Structural latency on a VGPR-limited dual-wave fp8 HIPREC pipeline. The kernel is
latency/stall-bound (flyprof bound_type=latency), capped at vgpr_alloc=112 / 4 waves/CU /
25% occupancy, with the exposed waits being the _stage_vt_dequant_fp8 s_waitcnt vmcnt(1)
(fp8 V load->dequant adjacency) plus shared K/V DMA pressure. The largest bubble (barrier
33%) is NOT the fix target: the dual-wave stagger barriers are proven net-beneficial.

### Why this is BLOCKED, backed by the falsification probes (all measured, all in this ledger)
- barrier (top bubble): CAND-perf-1 --no-stagger REVERT (-17..25%; barriers net-beneficial).
- vmcnt: CAND-perf-4 K-hoist REVERT (parity break, needs 3rd buffer); CAND-perf-6
  deeper-buffer +0 TF (bubble is staging-internal, not prefetch-depth); CAND-perf-7 source
  load-hoist REVERT (compiler already hoists — ISA-proven no-op).
- occupancy/VGPR: CAND-perf-2 wpe hint cliff (983->71->29, nan); CAND-perf-3 Q-LDS no
  occupancy change (Q is not the live-range peak); CAND-perf-5 M128 constant-flip nan.

### Remaining structural routes — NOT funded in this loop (task11 ruling)
- (A) VGPR-peak-reducing accumulator/softmax/PV dataflow redesign to reach vgpr_alloc<=96
  (5 waves/SIMD): the only route attacking a real constraint, but a large-blast-radius
  rewrite with no evidence that higher occupancy removes a structural latency wall.
- (B) BLOCK_M=128 / 4-wave full schedule rewrite: a separate rewrite project (constant-flip
  already nan'd), not an optimization continuation.
- (C) NATIVE single-instruction fp8 V read: not the throughput-max path (NATIVE 726 TF <<
  HIPREC 996 TF; cannot reach the best-fp8 parity target).

### Disposition
Stop the AC-4 candidate loop at round 15 with BLOCKED recorded. Remaining mainline work is
task12 (final report + docs/comment reconciliation + DEC-P3 + placement check). The
opt-in/experimental fp8 modes remain correct and mergeable; defaults unchanged throughout
(every candidate was gated default-off and byte-identical when off). Routes (A)/(B) are the
documented follow-ups if the fp8 throughput effort is resumed as a funded rewrite.

## Wide 32x32x64 MFMA Loop — Pre-loop state (handoff from the exploratory worktree)

Root cause (proven, PR #711 gist): aiter fp8 ASM uses v_mfma_f32_32x32x64_f8f6f4 (65536
MACs/op); FlyDSL used mfma_f32_32x32x16_fp8_fp8 (16384) -> 4x more MFMA instructions. Fix =
adopt the wide atom.

### CAND-wide-1: wide 32x32x64 fp8 PV (LDS P gather) — CORRECT, throughput-flat (KEEP as baseline)
- Gated FLYDSL_FP8_WIDE_MMA (NATIVE). Wide mfma_scale_f32_32x32x64_f8f6f4 (unit scale
  0x7F7F7F7F = plain fp8xfp8). A=V wide-read from vtf (32 contiguous keys/lane); P cross-lane
  gather via LDS round-trip (_stage_p_fp8_wide -> s_barrier -> _read_p_fp8_wide).
- Correctness: min_cos 0.9993 nocausal AND causal (== narrow). rocprofv3: PV SQ_INSTS_MFMA
  4.19e6 -> 2.62e6 (~4x on PV; QK still narrow).
- Throughput: ~765 TF (vs HIPREC default ~953, narrow NATIVE ~728-913, aiter ~1300). FLAT:
  the per-cluster P-stage s_barrier serializes the dual-wave pipeline (GRBM_GUI_ACTIVE 1.42x
  up). Barrier eats the MFMA win. Default-off byte-identical. Commit 61c26a62.

### CAND-wide-2: in-register P shuffle (permlane32, NO barrier) — WIP, min_cos 0.939 (not yet kept)
- Gated FLYDSL_FP8_WIDE_PSHUF. Replaces the LDS P round-trip + barrier with a permlane32
  cross-lane gather (_read_p_fp8_wide_shuffle). ~810 TF (+18% over narrow) but min_cos 0.939
  FAIL -- one dword subset of the permutation still wrong (0.876 -> 0.939 after lo/hi-separate
  permute + own/other flip-by-dest-half). Operand K-layout fully decoded
  (k=(lane//32)*32+byte_p). NEXT: shuffle-vs-LDS per-dword diff probe to finish it. Commits
  6c9170f6, a8ebbb2b, fa0d61ac. LDS path remains the correct default.

### Operand-oracle method (reusable)
On-device probe tests/kernels/probe_wide_layout.py decodes the wide MFMA operand layout by
the MMA identity (all-ones sanity -> 64; single-byte A/B K-match). Definitive operand-layout
proof is the end-to-end fp8 correctness gate. See LAYOUT_FINDINGS.md.

### CAND-wide-3: in-register P shuffle FIXED (no-barrier) — CORRECT + net win (KEEP)
- Date / round: 2026-06-25 / RLCR wide-mfma round 0
- Hypothesis: the wide-LDS PV path is throughput-flat because the per-cluster P-stage
  s_barrier serializes the dual-wave pipeline; an in-register permlane32 P gather removes
  the barrier and converts the proven ~4x PV MFMA reduction into wall-clock. The existing
  shuffle (FLYDSL_FP8_WIDE_PSHUF) was min_cos 0.939 FAIL due to one wrong cross-lane step.
- Root cause of the 0.939 bug: `_permlane32_swap_i32` extracted the permlane32_swap pair as
  result[1] UNCONDITIONALLY. permlane32_swap(a,a) returns the +/-32 partner value in
  result[1] for low-half lanes but result[0] for high-half lanes (the same convention the
  proven O-store `_swap_halves` uses). High-half lanes (L+32, which supply K=32..63 of every
  wide-P output column) thus received their OWN value instead of the partner's, corrupting
  the K=32..63 half of every column -> partial correlation (min_cos 0.939).
- Change (one candidate): select result[0] on high-half lanes / result[1] on low-half lanes
  in `_permlane32_swap_i32` (lane//32 == 1 ? result0 : result1). Refreshed the stale
  `_WIDE_PSHUF` comment to match (correct, opt-in, default-off).
- Mode(s): NATIVE fp8 (FLYDSL_FP8_WIDE_MMA=1 FLYDSL_FP8_WIDE_PSHUF=1). Default-off; bf16/fp16
  and gate-off fp8 codegen unchanged (shuffle code is behind _WIDE_PSHUF const_expr).
- GPU / model: GPU4, AMD Instinct MI350X (gfx950).
- Branch / commit: feat/fp8-fa-wide-mfma / base e83d996d (pre-commit), see git log (this candidate).
- Command:
    HIP_VISIBLE_DEVICES=4 FLYDSL_FP8_HIPREC=0 FLYDSL_FP8_PV_NATIVE=1 FLYDSL_FP8_WIDE_MMA=1 \
      FLYDSL_FP8_WIDE_PSHUF=1 python3 tests/kernels/test_flash_attn_fwd.py --dtype fp8 \
      --compare --no-causal --batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 \
      --head_dim 128 --warmup 10 --iters 50
- Shapes: headline B1 S2048 H64 D128 (nocausal + causal) + full DEFAULT_CONFIGS fp8 sweep.
- Warmup / iters: 10/50 (headline), 2/3 (sweep).
- Correctness:
    - headline nocausal: min_cos 0.9993 PASS (max_err 2.03e-03); was 0.939 FAIL.
    - headline causal:   min_cos 0.9993 PASS (max_err 2.29e-02).
    - fp8 DEFAULT sweep nocausal: 80 PASS / 0 FAIL (lowest min_cos 0.99914); all ERRORs are
      the expected fp8 split-K rejections (split-K stays rejected).
- Throughput (headline nocausal, warmup10/iters50):
    - wide-shuffle (this): 789.6 TF  (run2: 759.5->789.6 over wide-LDS)
    - wide-LDS baseline:   759.5 TF
    - aiter_asm:           1288-1342 TF (native fmha_v3_fwd, fwd_hd128_fp8.co)
    - => +4.0% over wide-LDS; ~61% of aiter_asm. Still below HIPREC ~953 (T1 not yet reached
      as the single best-fp8 number; this lifts the NATIVE-wide branch only).
- Re-profile (rocprofv3, GPU4, warmup2/iters5, per-dispatch):
    - SQ_INSTS_MFMA: 2.62e6 (unchanged vs wide-LDS — same wide atom)
    - SQ_INSTS_LDS:  4.98e6 -> 4.19e6 (0.842x — P LDS round-trip removed)
    - GRBM_GUI_ACTIVE: 3.32e6 -> 3.19e6 (0.961x — barrier bubble reduced)
    - artifacts: /tmp/prof_wide_lds, /tmp/prof_shuffle (counter DBs).
- Outcome: KEEP. The targeted barrier bubble moved (LDS + GRBM down) and wall-clock improved
  with correctness equal to the LDS path. This is the no-barrier route-A fix. The shuffle is
  now the faster wide-PV variant; further gains (toward HIPREC/parity) need wide QK (route C)
  and/or making wide the default fp8 path.

### CAND-wide-3 provenance refresh + task2 probe (RLCR wide-mfma round 1)
- Round-0 cited the mutable shared CSV `fmha_perf_compare_MI350X.csv`, which a later gate-off
  verification run overwrote (it then showed the NATIVE-narrow 698 TF row, not the wide
  numbers). The TFLOPS claims were real (in stdout) but the artifact was not immutable. Fixed
  by re-running the headline matrix with per-mode runtime caches and immutable per-mode files.
- Isolated headline matrix (GPU4 MI350X gfx950, B1 S2048 H64 D128 nocausal, warmup10/iters50,
  per-mode FLYDSL_RUNTIME_CACHE_DIR; artifacts under
  .humanize/rlcr/2026-06-25_07-12-31/artifacts/round-1/bench_<mode>.{log,csv}):
  | mode | FlyDSL TF | min_cos | aiter_asm TF | Fly/asm |
  |---|---:|---|---:|---|
  | HIPREC (best default fp8) | 969.4 | 1.0000 | 1304.1 | 74.3% |
  | NATIVE narrow | 697.9 | 0.9993 | 1289.8 | 54.1% |
  | NATIVE wide-LDS | 761.7 | 0.9993 | 1279.9 | 59.5% |
  | NATIVE wide-shuffle (route A, KEPT) | 797.5 | 0.9993 | 1294.2 | 61.6% |
  - Route A reproduced: wide-shuffle 797.5 > wide-LDS 761.7 (+4.7%) at equal correctness;
    aiter_asm native fmha_v3_fwd. Supersedes the round-0 759.5/789.6 figures (same conclusion,
    immutable provenance). T1 (beat HIPREC 969) NOT reached by the NATIVE-wide branch alone.
- task2 probe (tests/kernels/probe_wide_layout.py, artifact probe_shuffle_vs_lds.log):
  (1) on-device permlane32 lane-half check -> out[L]==L^32 for all 64 lanes PASS;
  (2) analytical shuffle-vs-LDS wide-P per-dword diff -> 0 mismatches PASS (the pre-fix
  high-half model yields 128 dword mismatches, so the probe is non-vacuous). This is the
  planned shuffle-vs-LDS diff evidence for route A.

### CAND-wide-4: wide 32x32x64 QK (route C) — CORRECT + net win (KEEP)
- Date / round: 2026-06-25 / RLCR wide-mfma round 1
- Hypothesis: QK still used 8 narrow 32x32x16 MFMAs per N-strip; moving QK to the wide atom
  (like V/PV) removes more MFMA instructions. Q/K are both already in registers/LDS (the easy
  operands), so the wide read mirrors the V wide read. Independent lever from route A.
- Change (one candidate): new default-off gate FLYDSL_FP8_WIDE_QK (fp8 NATIVE + wide MMA only).
  _load_q_all_wide(): each lane reads 32 contiguous head-dim fp8 (256b = 2x128b load) for its
  query row, per head-dim half (ws=0 D0..63, ws=1 D64..127) -> 2 i32x8. _read_k_packs_fp8_wide():
  reads K from the K LDS tile with the derived+probe-checked address
  addr(key,d)=(key%8)*SMEM_K_LINE_STRIDE+(key//8)*D_128B_SIZE+d, 32 contiguous D/lane for its
  key column n (lo strip n=lane%32, hi strip n+32), per ws -> (k_lo,k_hi). _mma0_wide():
  2 wide MFMAs per N-strip (A=K,B=Q, matching narrow operand order) + the same
  c_logit_scale*log2e fp32-logit scaling. Gate-off _mma0 byte-identical.
- Mode(s): NATIVE fp8 (with FLYDSL_FP8_WIDE_MMA=1 FLYDSL_FP8_WIDE_PSHUF=1 FLYDSL_FP8_WIDE_QK=1).
  Default-off; bf16/fp16 untouched (fp8-only const_expr gate).
- GPU / model: GPU4 (bench) / GPU5 (profile), AMD Instinct MI350X gfx950.
- Branch / commit: feat/fp8-fa-wide-mfma (this candidate, see git log).
- Command:
    HIP_VISIBLE_DEVICES=4 FLYDSL_FP8_HIPREC=0 FLYDSL_FP8_PV_NATIVE=1 FLYDSL_FP8_WIDE_MMA=1 \
      FLYDSL_FP8_WIDE_PSHUF=1 FLYDSL_FP8_WIDE_QK=1 python3 tests/kernels/test_flash_attn_fwd.py \
      --dtype fp8 --compare {--no-causal,--causal} --batch 1 --seq_len 2048 --num_heads 64 \
      --num_kv_heads 64 --head_dim 128 --warmup 10 --iters 50
- Shapes: headline B1 S2048 H64 D128 nocausal + causal; full DEFAULT_CONFIGS fp8 sweep.
- Warmup / iters: 10/50 (headline), 2/3 (sweep).
- Correctness:
    - headline nocausal min_cos 0.9993 PASS (max_err 2.03e-03).
    - headline causal   min_cos 0.9993 PASS (max_err 2.29e-02).
    - fp8 DEFAULT sweep nocausal: 80 PASS / 0 FAIL (lowest min_cos 0.99914); split-K ERROR only.
      Artifact artifacts/round-1/fp8_sweep_wide_qk.log.
- Throughput (headline, isolated per-mode artifacts under artifacts/round-1/):
    - wide-QK+shuffle (this): 841.2 TF nocausal (bench_native_wide_qk.{log,csv}); 580.2 TF causal.
    - wide-shuffle (route A only): 797.5 TF nocausal.
    - wide-LDS: 761.7; NATIVE narrow: 697.9; HIPREC: 969.4.
    - aiter_asm: ~1282-1294 nocausal / 570.8 causal (native fmha_v3_fwd).
    - => +4.9% over shuffle, +20% over NATIVE narrow; 65.6% of aiter_asm nocausal; BEATS
      aiter_asm causal (101.7%). T1 (beat HIPREC 969) NOT reached by the NATIVE-wide branch.
- Re-profile (rocprofv3, GPU5, per-dispatch): SQ_INSTS_MFMA 2.62e6 -> 1.05e6 (0.400x of narrow;
  both QK+PV now wide, matches the predicted QKn+PVn/4 = 1.05e6); GRBM_GUI_ACTIVE 3.19e6 ->
  3.00e6 (0.942x of shuffle); SQ_INSTS_LDS unchanged. Targeted MFMA bubble moved AND converted
  to wall-clock this time. Artifact /tmp/prof_wide_qk.
- Outcome: KEEP. Note vs the round-0 profile prediction: I predicted wide QK would NOT convert
  (PV's MFMA drop moved GRBM ~0); it DID convert here (+4.9%), so the kernel is not purely
  barrier-bound -- there is a real instruction-issue component QK width reduces. Recorded as a
  correction to the round-0 hypothesis.

### CAND-wide-4 fixups (RLCR wide-mfma round 2): hot-path Q load + causal provenance
- B3: gate-on wide QK still ran the narrow _load_q_all + _scale_q_all (unused on the wide
  path). Fixed: the prologue now branches on _WIDE_QK and loads ONLY q_all_wide for wide QK.
  Re-measured headline nocausal NATIVE wide-QK: 841.2 -> 855.6 TF (+1.7%), min_cos 0.9993 PASS.
  Artifact artifacts/round-1/bench_native_wide_qk_nocausal_after_qload_fix.{log,csv}.
- B4: durable causal provenance for NATIVE wide-QK: 582.0 TF, min_cos 0.9993 PASS, beats
  aiter_asm causal 576.5 (101.0%). Artifact artifacts/round-1/bench_native_wide_qk_causal.{log,csv}.

### CAND-wide-5: wide QK extended to HIPREC fp8 (route 4, default-mode lever) — CORRECT + T1 CROSSED (KEEP)
- Date / round: 2026-06-25 / RLCR wide-mfma round 2
- Hypothesis: QK is native fp8 in EVERY fp8 PV mode (HIPREC/FROMBF16/NATIVE all run
  _mfma_acc_fp8_i64 for QK and differ only in PV), so wide QK is independent of the PV mode.
  Enabling wide QK on HIPREC -- the shipping best-fp8 default -- should beat HIPREC and cross T1.
- Change (one candidate): decouple FLYDSL_FP8_WIDE_QK from _WIDE_VREAD/NATIVE; gate is now
  `dtype_str=="fp8" and env`. HIPREC PV (bf16-P/V dequant path) is UNCHANGED; only QK switches
  to _load_q_all_wide + _read_k_packs_fp8_wide + _mma0_wide. Gate-off byte-identical.
- Mode(s): fp8 HIPREC (FLYDSL_FP8_HIPREC=1 FLYDSL_FP8_WIDE_QK=1). bf16/fp16 untouched.
- GPU / model: GPU4 (bench) / GPU5 (profile), AMD Instinct MI350X gfx950.
- Branch / commit: feat/fp8-fa-wide-mfma (this candidate, see git log).
- Command:
    HIP_VISIBLE_DEVICES=4 FLYDSL_FP8_HIPREC=1 FLYDSL_FP8_WIDE_QK=1 python3
      tests/kernels/test_flash_attn_fwd.py --dtype fp8 --compare {--no-causal,--causal}
      --batch 1 --seq_len 2048 --num_heads 64 --num_kv_heads 64 --head_dim 128 --warmup 10 --iters 50
- Shapes: headline B1 S2048 H64 D128 nocausal + causal; full DEFAULT_CONFIGS fp8 sweep.
- Correctness:
    - headline nocausal min_cos 1.0000 PASS; causal min_cos 1.0000 PASS (HIPREC is exact-P).
    - fp8 DEFAULT sweep nocausal: 80 PASS / 0 FAIL (all min_cos >= 0.99999); split-K ERROR only.
      Artifact artifacts/round-2/fp8_sweep_hiprec_wideqk.log.
- Throughput (isolated per-mode artifacts under artifacts/round-2/, warmup10/iters50):
    | path | nocausal TF | causal TF | min_cos |
    |---|---:|---:|---|
    | HIPREC baseline | 947.7 | 696.7 | 1.0000 |
    | HIPREC + wide QK (this) | 1058.1 | 759.4 | 1.0000 |
    | aiter_asm | 1295.9 | 565.8 | 0.9993 |
    - nocausal +11.6% over HIPREC baseline; 81.7% of aiter_asm (T1 CROSSED, approaching T2 85%).
    - causal +9% over baseline; 134% of aiter_asm causal.
- Re-profile (rocprofv3, per-dispatch): SQ_INSTS_MFMA 4.19e6 -> 2.62e6 (0.625x; QK wide, PV
  bf16-narrow in HIPREC); GRBM_GUI_ACTIVE 0.959x; converted to wall-clock. Artifacts
  /tmp/prof_hiprec_base, /tmp/prof_hiprec_qk.
- Outcome: KEEP. This is the new best fp8 path and the first to cross T1 (beat HIPREC ~969).
  Per DEC-2 it strictly dominates HIPREC at equal correctness (min_cos 1.0000), so it is a
  default-promotion candidate; kept opt-in this round pending the formal default-flip decision
  (task9/DEC-2) + the full bf16/fp16 base-vs-head non-regression sign-off (bf16/fp16 are
  unaffected by an fp8-only gate, but a default flip warrants the explicit gate).

### task8: full non-regression evidence for HIPREC+wideQK (RLCR wide-mfma round 3)
- Gate surface (B5): wide QK is correct in ALL THREE fp8 PV modes -> the broad
  `dtype_str=="fp8"` gate is justified (no narrowing needed). Headline B1 S2048 H64 D128:
    - HIPREC+wideQK: nocausal min_cos 1.0000 / causal 1.0000 (artifacts/round-2 + round-3).
    - NATIVE+wideQK: nocausal 0.9993 / causal 0.9993 (artifacts/round-1).
    - FROMBF16+wideQK: nocausal 0.9986 / causal 0.9986 PASS (artifacts/round-3/frombf16_wideqk_*).
- fp8 dense DEFAULT_CONFIGS sweep, HIPREC+wideQK: nocausal 80P/0F (round-2), causal 80P/0F
  (round-3, lowest real cos 0.99999); all ERRORs are the expected fp8 split-K rejections.
- Gate-off byte-identity (B6): final ISA (21_final_isa.s) AND final LLVM IR (20_llvm_ir.ll) are
  byte-identical (same md5) between HEAD and base e83d996d for the fp8 HIPREC default (no wide
  gates). Artifact artifacts/round-3/gate_off_isa_identity.txt. The "default codegen
  byte-identical when gates off" claim is now ISA-proven, not asserted.

### task9 + DEC-2 (RLCR wide-mfma round 3): convergence outcome + default promotion
- AC-4 OUTCOME (Codex analyze, gpt-5.5:xhigh): IMPROVEMENT (T1-only). Kept route A + route C +
  HIPREC wide-QK; +11.6% over the HIPREC fp8 baseline at identical correctness; T1 crossed
  (beat HIPREC), T2 (85%/~1100) not reached nocausal (81.7% of aiter_asm). Causal beats aiter.
  Clean correctness/non-regression; gate-off ISA byte-identical. Candidate search stopped here;
  T2/T3 levers (HIPREC wide PV, schedule rebalance) deferred to a new optimization phase.
- DEC-2 DECISION (binding): PROMOTE HIPREC+wideQK to the DEFAULT fp8 path, with an opt-out.
  Implemented: FLYDSL_FP8_WIDE_QK now defaults ON for fp8 (set =0 to opt out to narrow QK).
- Post-flip validation (artifacts/round-3/):
    - No-env default fp8 headline: 1052.3 TF nocausal / 746.7 causal, min_cos 1.0000
      (default_fp8_after_flip_{nocausal,causal}.{log,csv}). Reproduces the env-on band.
    - No-env fp8 dense sweep nocausal: 80P/0F (fp8_sweep_default_after_flip.log); split-K ERROR
      expected.
    - Opt-out FLYDSL_FP8_WIDE_QK=0: final ISA byte-identical to base e83d996d narrow-QK path
      (gate_off_isa_identity.txt) -> the escape hatch exactly restores prior behavior.
    - bf16 headline unchanged (fp8-only gate).

### CAND-wide-6: T2/T3 falsification — already-wired wide-PV combos vs default (TESTED-NEGATIVE)
- Date / round: 2026-06-25 / RLCR wide-mfma round 4
- Hypothesis: with wide QK now default, do the already-implemented wide-PV combinations beat the
  default HIPREC+wideQK (1058.1/759.4) and push toward T2 (>=85% aiter ~1100)?
- Candidates (headline B1 S2048 H64 D128, GPU4 MI350X, warmup10/iters50, isolated; artifacts
  artifacts/round-4/falsify_*):
  | combo | nocausal TF | causal TF | min_cos |
  |---|---:|---:|---|
  | NATIVE wide-PV-shuffle + wideQK (WIDE_MMA+PSHUF) | 852.2 | 577.6 | 0.9993 |
  | FROMBF16 wide-PV + wideQK (WIDE_MMA)             | 685.0 | 464.4 | 0.9986 |
  | DEFAULT HIPREC + wideQK (reference)              | 1058.1| 759.4 | 1.0000 |
- Outcome: TESTED-NEGATIVE. Both wide-PV combos are CORRECT but materially SLOWER than the
  default HIPREC+wideQK (NATIVE -19% nocausal, FROMBF16 -35%). The wide-PV path's PV precision
  modes (NATIVE raw-fp8 / FROMBF16 requant) are inherently below HIPREC's bf16 PV throughput on
  this shape; making PV wide does not overcome that. => the remaining ranked levers that are
  already wired do NOT beat the default. Reaching T2/T3 requires NEW work outside this loop's
  budget: a wide fp8 PV that runs in the HIPREC precision regime (not NATIVE/FROMBF16), or a
  dual-wave schedule rewrite. AC-4 stands at IMPROVEMENT (T1-only) AFTER testing the remaining
  ranked levers; T2/T3 is queued WITH this falsification evidence, not merely deferred.
