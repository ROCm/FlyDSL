# Session Journal — FlyDSL FMHA Setup, Benchmark & Autoresearch

**Host:** Linux 5.15, AMD Instinct MI308X (gfx942), ROCm 7.13 · **flydsl** 0.2.0 (PyPI wheel)
**Kernel:** `kernels/fmha_prefill_fp8_ck_hk5.py` · **Branch:** `anguyenh/prefill_batch_attn`
**Dates:** setup/bench 2026-06-16; VALU-fold loop 2026-06-15/16

---

## 1. Setup (one-time)

- **GitHub SSH:** key `~/.ssh/anguyenh`; `~/.ssh/config` maps `github.com` → that identity.
- **Knowledge base:** `claude-knowledge-base` cloned to `~/workspace/`, symlinked `~/knowledge-base`; skills copied to `~/.claude/skills/`.
- **FlyDSL repo:** cloned `git@github.com:ROCm/FlyDSL.git`; `.devcontainer/` added (base `therock-main` ROCm 7.13 gfx94X image).
- **Install:** use **`pip install flydsl` (PyPI 0.2.0)** + bind-mount the repo — kernel code lives in `kernels/`, not the package. Avoid `pip install -e .` (triggers a full MLIR source build) unless developing the compiler itself.
- **aiter (CK-Tile baseline):** cloned `git@github.com:ROCm/aiter.git`; `git submodule update --init --recursive` (pulls `composable_kernel`); `AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop` in the GPU container; `git config --global --add safe.directory /workspaces/aiter`. First FMHA JIT build ~40s/variant.

Full step-by-step in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (loop repro in §11).

### Decisions / gotchas

1. Branch ref is `opt/fmha-batch-prefill-fp8` (the `.../learn_fmha` URL suffix is a folder, not a ref).
2. GPU select: `HIP_VISIBLE_DEVICES=2` (or 2–7); GPUs 0–1 noted broken in `learn_fmha/README.md`.
3. `--no-pyisa` because `/workspaces/amir/asm/fwd_fp8` is absent on this host.

### Benchmark command

```bash
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 --ck --no-pyisa
```

---

## 2. Kernel design (CK-Tile port + hk5 padding)

Canonical FlyDSL FMHA kernel: a fresh port of AMD CK-Tile's production fp8
`BlockFmhaBatchPrefillPipelineQRKSVSAsync` (reached via aiter), with full feature parity to the
8wave/v8 baselines (drop-in `run_attn` signature + tensor layouts). GEMMs use
`mfma_f32_32x32x16_fp8_fp8`: GEMM1 = K@Qᵀ (S as [kv,q]); GEMM2 = Vᵀ@P (O as [d,q]). Online-softmax
recurrence reused from 8wave (register-resident P transposed via `ds_bpermute`, `rocdl.exp2`,
per-token-head Q/K descale, per-head V descale, `p_scale`).

CK structural choices adopted:

- **Large outer K-tile** `KT` (CK `kN0`): one cooperative load + one `gpu.barrier()` + one prefetch per `KT` keys, amortised over `NSUB` MFMA subtiles.
- **Q loaded once** into registers, reused over the whole KV loop (CK `kQLoadOnce`).
- **Diagonal-pair tiling** (CK causal load-balancer): each CTA does q-tile `t` AND its mirror `num_q_tiles-1-t`, so light/heavy causal tiles share a workgroup (+8–24% at sq≥2048).
- **Masked/unmasked loop split:** interior tiles skip the per-element causal-mask VALU (VALU:MFMA 24→19, +13%).

**Tunables (env):** `FMHA_NWAVES` (default 4 → TILE_BM=128), `FMHA_KT` (32), `FMHA_VCOL` (1),
`FMHA_DIAG` (1), `FMHA_BUFK` (0, broken in this wheel).

### Win — COLUMN-V (`VCOL`)

CK's true `vec_k_col_v` stores V column-major so the GEMM2 contraction dim (kv) is contiguous ⇒
**no transpose**. We match it (`pack_paged_cache(v_col=True)`); the V→LDS copy is one 128-bit
store/slot instead of the 16× `ds_write_b8` scatter the row-major path needs. PMC: LDS-wait 54% →
18% of busy. Disproves the handoff claim that the gfx942 transpose DS-wait is irreducible / needs
gfx950 `ds_read_tr` — CK avoids it purely by V layout, and so do we.

### Win — LDS row PADDING (the "hk5" in the name)

Baseline `SQ_LDS_BANK_CONFLICT` = 68% of busy: K LDS rows stride HD=128B = 32 banks×4B, so
consecutive kv rows alias the same bank (up to 32-way conflict on `ds_read`). Fix: pad each LDS row
so the stride is coprime-ish with the 32-bank period. Swept 2026-06-14 @ sq16384: **K_PAD=V_PAD=8 is
optimum** (108.5 TF, LDS 18944) — beats 16/16 (106.8, LDS 21504) with *less* LDS. Response is
bank-period-sensitive & non-monotonic (KPAD=0→63% back; VPAD=4→59%; VPAD=32→75%). Env-sweepable via
`FMHA_KPAD`/`FMHA_VPAD`. XOR-swizzle was tried but lost (+27 VGPR for address math, and we're
VGPR-bound).

---

## 3. Autoresearch optimization loop (`fmha-hk5-autoresearch`)

Autonomous commit→measure→keep/reset loop (branch `anguyenh/prefill_batch_attn`). Goal: close the
FlyDSL→CK-Tile gap at large seq. Every kept lever is a VGPR-neutral softmax-VALU op-count cut.
Loop repro + the CK-disassembly method are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §11–12.

### Progression (MI308X, bs=1 nq8 nk1 causal; TFLOPS)

| Shape | Start | R1 | **HEAD** | CK-Tile | CK/FlyDSL now |
|---|---|---|---|---|---|
| sq=1024  | 4–6 | 6 | 6 | 37 | 6.2× (grid-starved, untouched) |
| sq=16384 | 103 | 108 | **117** | 170 | **1.45×** |
| sq=32768 | 122 | 128 | **140** | 177 | **1.26×** |

HEAD kernel: VGPR 165, LDS 18944, 0 spills. Correctness gate (max-abs err < 6e-2) green on all
shapes incl. GQA, p_scale∈{16,64}, non-causal, partial tiles, b=2. 9/9 `test_fmha_prefill_fp8` pass.

**R6 (2026-06-16)** — 4+2+3 parallel subagent experiments (own worktree + dedicated GPU each). One
VGPR-neutral win promoted (XCD remap, 115→117 / 137→140); 7 decisive discards closing every remaining
kernel-side large-seq route (dead-end list below). Branch `fmha-hk5-fold`.

### Kept levers (commits)

| Commit | Lever | Effect |
|---|---|---|
| `5e3af84f` | Fold LOG2E into `qs` descale — scores enter log2-space, drops a per-element fmul | +2% |
| `15d8d1cc` | Drop `sched_mfma`/`sched_dsrd` hints — over-constrained the scheduler; gained perf *and* deleted code | +1–2 TF |
| `2f7e2484` | Fold `p_scale` shift into softmax pivot (`safe_m_p = safe_m − log2_pscale`) — removes one add/element | +2 TF |
| `250138ff` | **maxnum reduction → `v_max3_f32`** (see below) | **+7%** |
| `00357cca` | **XCD/chiplet block-ID remap** (qhead-fast L2 grouping; see below) | **+1–2 TF** |

**The XCD-remap win (R6):** MI308X has **4 XCDs** (`num_xcc 4`). The flat grid scatters the 8 GQA
q-heads of one q-tile — which share an *identical* causal K/V range — across XCDs, so each XCD's
private L2 re-fetches the same K/V. An inverse-round-robin block-ID remap with qhead-fast decode
(env `FMHA_XCD=1`, `FMHA_XCD_C=4`) groups those 8 q-heads onto one XCD's L2. Pure kernel-top scalar
math: **VGPR 165→165 (neutral)**, 0 spills, LDS unchanged. It's exactly the VGPR-neutral class of win
that sticks. **Rigorously re-confirmed (interleaved warm-vs-warm, 10 reps, rocprofv3 `TCC_HIT/MISS`):
at sq32768 flat 138 → C=4 140 TF (zero spread, all 10 reps = 140), L2 miss 13.86M→2.85M (−79%), hit
81%→96%.** At sq16384 the TF delta (116→117) is within noise but PMC-positive (831K→687K miss,
95.7%→96.5%) — the remap's value scales with KV-vs-L2 pressure, growing with seq length. **C=4
qhead-fast is provably optimal:** C∈{5,6,8} tie, C∈{2,3} = 139; a diag-aware/qtile-fast decode LOST
decisively (139 TF, 7.08M miss = 2.5× more) because it groups overlapping-but-non-identical KV instead
of the identical-KV GQA q-heads. No per-seq C dispatch needed (C=4 never regresses). sq1024 untouched
(grid-starved).

**The maxnum win:** the softmax max-reduction used `maximumf` → `llvm.maximum.f32` (NaN-propagating),
which the AMDGPU backend **will not fuse** into the 3-input `v_max3_f32`. Switching to `arith.maxnumf`
(non-NaN-propagating — correct here, scores are finite or −inf) fuses: `v_max_f32` 68→4 (replaced by
32 `v_max3`), **VGPR unchanged at 165**. Found by diffing our ISA op-histogram against CK's. (An older
journal "maxnumf dead-end" note was wrong.)

### Diagnosis — the gap is MFMA/VALU non-overlap, quantified and arch-walled

VALU-bound (VALU:MFMA 13.6:1 after maxnum, LDSBankConflict ~1.5%, MemUnitStalled ~0). Every change
that cut VALU op-count *without adding VGPR* won; everything that added VGPR regressed.

rocprofv3 on HEAD (sq16384) gives the closing number:

| Counter | Value | Reading |
|---|---|---|
| `MfmaUtil` | **26.6%** | MFMA unit idle 73% of the time |
| `VALUBusy` | **57.6%** | softmax VALU is the busy pipe |
| `VALUUtilization` | 99.99% | VALU fully packed *when it runs* |
| `MeanOccupancyPerActiveCU` | 2.69 | the VGPR-pinned 3-wave ceiling (minus tail) |
| SQ VALU:MFMA | 230.5M : 16.9M = **13.6:1** | matches the ISA op-count ratio |

**The two pipes run serialized:** busy ≈ VALUBusy + MfmaUtil = 84%. Perfect overlap → busy →
max(57.6, 26.6) ≈ 58% ⇒ **84/58 = 1.46× — exactly the CK gap.** The entire large-seq gap is MFMA/VALU
non-overlap; CK's win is overlapping the pipes, not leaner softmax.

**It is not occupancy.** CK's gfx942 fp8 kernel (disassembled from the aiter `.so` fatbin) runs
VGPR=214 → 2 waves/SIMD — *lower* occupancy than our 3 waves — yet 1.4–1.6× faster. Its symbol decodes
as `fmha_batch_prefill_d128_fp8bf16 … b128x128x32x128 … qr_async_vr` ⇒ **kN0=128 KV-tile + async
K→LDS**: softmax max/exp/rescale runs once per 128 kv (4× our KT=32), amortising per-tile VALU over 4×
the MFMA, with K loads hidden via async.

**Accumulators stay in VGPR — AGPR is not the lever.** CK keeps the MFMA C-accumulators in VGPR
(`"+v"`) in every `WGAttrCtlEnum` preset; only A/B *operands* ever go to AGPR
(`warp_gemm_attribute_mfma_impl.hpp:106,170`). 0.2.0 likewise places all accumulators in VGPR
(`v_mfma … v[66:81]`, `num_agpr=0`). Forcing accumulators to AGPR is not how CK wins and is a dead-end.

**Decomposition ceiling probes** (all INCORRECT, used only to bound headroom; reverted):

| Probe (sq16384 / sq32768) | TF | Δ vs HEAD |
|---|---|---|
| HEAD baseline | 115 / 137 | — |
| strip max/corr/`o_acc*=corr` rescale only | 122 / 148 | +6% / +8% |
| strip per-token K-descale (`kdv`) only | 124 / 147 | +8% / +7% |
| strip BOTH (GEMM + exp + P-transpose only) | **138 / 165** | **+20%** |

Two firm conclusions: (1) **all softmax-VALU op-count levers share a hard ~+8% ceiling** — removing
the entire max/corr/rescale block buys only +8%, so the op-count hunt is retired. (2) **Even the
GEMM-only ceiling (138/165) is BELOW CK's 170/177** — CK's raw MFMA scheduling wins too, not just its
softmax overhead. The per-token K-descale's ~8% is **not removable**: the reference quantizes K
per-token-head (`fmha_prefill_fp8_ref.quantize_per_token_head`), so the per-element `kdv` multiply is
mathematically required; CK's per-tensor descale is a looser numerics contract.

**CK's gfx942 `qr_async_vr` path overlaps via kN0=128 + async-K, and they are coupled.** kN0=128 gives
4× independent MFMAs per softmax pass — enough within-wave MFMA slots for the scheduler to fill the VALU
gap — but only helps *if* K is async-staged global→LDS so it doesn't occupy VGPRs. The **0.2.0 wheel**
delivers neither: KT>32 regresses (VGPR-staged loads cap occupancy, KT=64 PMC occ 2.69→1.0), and
async-K (`buffer_load_to_lds`) is slow (wheel emits 4B/lane, serialized, 32–64 TF) and broken in hk5.
The coupling — kN0=128 needs fast async-K to avoid blowing VGPR, which *this wheel's codegen* cannot
deliver — is a wheel wall, **not** a gfx942 hardware wall (CK does exactly this on gfx942).

**In-wave overlap is VGPR-walled.** Two independent cross-tile pipelines that overlap GEMM-MFMA into
softmax-VALU both work *mechanically* but both cross the 3-wave cliff:

| | HEAD | L1 (sv-carry) | defer-GEMM2 (`FMHA_DEFER`) |
|---|---|---|---|
| TF (sq16384/32768) | 115 / 137 | −16/−20% pre-maxnum | 101 / 114 |
| VGPR | 165 | 213 | 210 |
| occupancy | 2.69 | 2 waves | 1.89 |
| overlap happened? | — | MFMA runs 23→48, M↔V 47→97 | VALUBusy 57.6→50.4% |

The defer route is the cleaner one: defer tile j-1's `V^T@P` into tile j's softmax, fold deferred
`A(j-1)` into `o_acc` *before* the `corr(j)` rescale (bit-exact; carries only transposed P = 4 i64/lane,
V triple-buffered in LDS not registers). Correctness green on all 8 shapes (err 0.0391 @ sq16384).
Overlapping forces both operand sets live → VGPR 165→210 → the lost wave eats the overlap. At NSUB=1
there is no independent MFMA *within a wave* (GEMM1→softmax→GEMM2 is a hard chain), so an in-wave
`sched_group_barrier` is inert; the independence CK exploits comes from a second warp group.

**Cross-warp-group ping-pong is gfx942-PORTABLE and remains OPEN (a prior "needs gfx1250" claim here
was WRONG — corrected 2026-06-16).** CK's *fastest* fp8 path (the V3 pipeline
`block_fmha_fwd_v3_pipeline.hpp`, tag `qr_async_trload_v3`) runs 8 waves = 2 warp groups on the same
4-phase loop phase-shifted by 3, so while WG0 is MFMA-heavy WG1 is VALU-heavy → both SIMD pipes busy,
accumulators in VGPR. **Verified from source** (`grep` of the V3 pipeline): it uses **only plain
`__builtin_amdgcn_s_barrier()`** (14 call sites) and **zero** split/named barriers — no
`s_barrier_signal`/`s_barrier_wait`/`barrier_arrive` anywhere. The phase-offset is pure code structure
(`CoreLoopSchedulerDefaultBase::schedule`, lines 97–115):

```
WG0: compute0(P0), load(P1), compute1(P2), load(P3)
WG1: load(P0),     compute0(P1), load(P2), compute1(P3)   // effective = (Phase+3)%4
```

Both warp groups rendezvous at the **same** plain workgroup `s_barrier` between phases; the overlap
comes from running *different phase bodies* in each inter-barrier region (one group's `schedule_gemm*_compute`
MFMAs cover the other's `schedule_load_phase`/softmax VALU), with `sched_group_barrier(MFMA,1)/(TRANS,2)/(VALU,2)`
interleaving inside each compute phase. **Every primitive here — plain `s_barrier`, `sched_group_barrier` —
exists on gfx942.** Nothing requires gfx1250.

The first gfx942 attempt (42/49 TF) built same-code-both-groups (no phase offset) → lockstep, no
overlap. The "needs gfx1250 named barriers" inference from it was wrong (V3 uses only plain
`s_barrier`).

**The phase-offset ping-pong was then BUILT and MEASURED (2026-06-16) — it DISCARDS, and the reason
settles the whole question.** 8 waves = 2 warp groups, each an independent q-tile with private LDS
(37888 B), CTA-uniform KV trip count for barrier parity. The 1-WG-vs-2-WG isolation (same file,
`FMHA_PP` flag), rocprofv3-confirmed:

| metric @ sq16384 | 1-WG (baseline) | 2-WG ping-pong |
|---|---|---|
| TF (sq16384/32768) | 115 / 137 | **60 / 64** |
| VGPR/wave | 165 | 188 |
| occupancy | **3 waves/SIMD** | **2 waves/SIMD** |
| MfmaUtil | 26.2% | **19.7%** |

Overlap did NOT materialize — MfmaUtil went DOWN (26.2→19.7%), TF halved. **Root cause (the real
lesson): an 8-wave workgroup is capped at ~2 waves/SIMD on gfx942 (512 VGPR/SIMD ÷ 165–188 VGPR), but
the baseline 4-wave workgroup runs at 3 waves/SIMD = three INDEPENDENT co-resident workgroups.** Those
3 independent workgroups are *already* how the hardware overlaps MFMA with VALU — when one workgroup's
wave is in softmax-VALU, another's is in MFMA. That is why baseline MfmaUtil is 26%, not lower. A
2-warp-group ping-pong gives each SIMD only 2 phase-coupled streams — FEWER than the 3 independent ones
multi-workgroup occupancy already provides — so it cannot win. **Ping-pong is the wrong tool for THIS
occupancy regime: CK needs explicit ping-pong because its kernel runs ONE big workgroup at 2 waves/SIMD
(VGPR 214) with no spare occupancy; hk5 already gets the same overlap for free via 3-way workgroup
occupancy.** The 1.46× gap is therefore not "missing ping-pong" — it is that CK's 3 independent
*things* per SIMD are kN0=128 tiles (4× the MFMA work per softmax) while ours are KT=32 tiles, and
KT>32 is VGPR-walled in this wheel. The lever is kN0, not warp-group structure.

### Dead-ends measured (do not re-try)

- **async-K** (`buffer_load_to_lds`, `FMHA_BUFK`): `fmha_prefill_fp8_ck_async.py` has a *correct*
  async path (err 0.041, KT=128) but runs 32–64 TF — in the **0.2.0 wheel** async-to-LDS emits 4B/lane
  with ~160 `s_waitcnt vmcnt(0)`, serialized and slower than VGPR-staging. The hk5 `FMHA_BUFK=1` path is
  separately broken (per-lane LDS ptr; M0 is wave-uniform on gfx942). NOTE: this is a **wheel codegen
  limit, not a gfx942 hardware limit** — CK uses `buffer_load_to_lds` on gfx942 at full width; a wheel
  that emits wide async-to-LDS would unblock the kN0=128 path.
- **+VGPR levers all regress** (hard VGPR-binding): FMA-contract fastmath (VGPR→172, 97/114);
  vectorized descale (VGPR→196, 97); kdv-hoist; KT>32 even with maxnum (77–87 TF — VGPR-staged loads
  cap occupancy, KT=64 PMC: occupancy 2.69→1.0, MfmaUtil 27→17%); NBUF=3/4 (idle LDS, no pipeline).
- **`iglp_opt` v0/1/2** — all regress (101–106 TF). **No-transpose probe**: only +3 TF (P-transpose
  already hidden). **Halving the bpermute** — wrong (err 3.0+; the 4 permutes gather distinct
  source-lane halves). **`sched_group_barrier`** on the L1 pipeline — worse (occupancy, not
  scheduling, is the limit).
- **`waves_per_eu`/`maxnreg` hints** — plumbed through both embedded `gpu-module-to-binary` and
  external-LLVM paths, but `--amdgpu-num-vgpr` is advisory; VGPR stays 166. **AGPR accumulation** —
  not exposed in 0.2.0. Manual max/sum trees — compiler already optimal. `s_setprio` removal —
  regresses (load-bearing).
- **Small-seq knobs** (DIAG=0, NWAVES=2): 6→7 TF; sq1024 has only 32 grid blocks for ~80 CUs — needs
  split-K/persistent (rewrite), not a knob.
- Older: KT>32, NWAVES≠4, batched-within-tile overlap, readfirstlane, XOR-swizzle (+27 VGPR),
  split-K (wash at large seq).
- **R6 dead-ends (2026-06-16, parallel fan-out; all confirm "every VGPR-relief raises VGPR" + the
  4-wave wall):**
  - **In-wave MFMA-into-softmax overlap** (`FMHA_PROBE_FILL`, inject 8/16 GEMM1 MFMAs): 115→88 TF,
    VGPR 165→166 — softmax window has no free MFMA slots at occ 3 (wave latency-bound). Negative *even
    VGPR-neutral*.
  - **16×16×32 fp8 atom** — 2× instructions for same work, o_acc still 64 VGPR. Negative ceiling.
  - **K-staging restructure** (fuse per-pass load→store) to unlock KT>32 — *inflated* VGPR to 254 at
    KT=64 (vs 200 plain; no-prefetch diag also 254). Existing prefetch already packs staging into
    compute's dead regs; the KT>32 wall is the COMPUTE working set (NSUB≥2 ≈ 200 VGPR), not staging.
  - **S-in-LDS roundtrip** (CK V3 `async_trload`) — spilling S to per-lane LDS *inflated* VGPR +19 even
    at NSUB=1 (165→184) → 246 at KT=64 → 58 TF (~2× slower). Third confirmation memory-spill raises VGPR.
  - **Compute-set VGPR cut → 4th wave is UNREACHABLE.** Probe (`FMHA_PROBE_DT`, incorrect) gutting o_acc
    DT=4→1 drops total VGPR only 165→135, occupancy stays 3 waves — compiler refills freed low regs from
    the high-water softmax set. Two correctness-preserving reorders (GEMM2 v_packs interleave; fuse
    P-transpose into exp) were VGPR-neutral — 0.2.0 compiler is already liveness-optimal. Aggregate
    resident set ~37 VGPR over the 128 (4-wave) threshold. Needs AGPR (not in 0.2.0) or an algorithm change.
  - **Conditional online-softmax rescale (FA4-style)** — *correct* ballot-gated `scf.if` skip (VGPR
    165→164, err green) **dissolves into DVFS noise** (12-rep interleaved); skip-ceiling only +3%, eaten
    by the branch overhead. Softmax restructuring exhausted.
  - **Per-seq small-q-tile dispatch** (NWAVES=2, sq≤4096) — the apparent 1.4–1.5× was a **DVFS/clock
    artifact** (same variant swings 0.32↔0.52 ms; warm-vs-warm indistinguishable). Small-seq is
    launch-floored. **Methodology: sub-3% large-seq + all small-seq deltas need interleaved warm-vs-warm
    on a dedicated GPU.**

### Net

**HEAD is now 117/140** (R6 XCD remap; up from 115/137), the best measured config for the 0.2.0 wheel
+ per-token-descale numerics. CK gap **1.45×/1.26×**. The gap is quantified as pure MFMA/VALU
non-overlap (84%/58%). **All overlap and occupancy routes are now tried and understood:** in-wave
cross-tile pipeline → VGPR-walled; in-wave MFMA-into-softmax injection → negative even VGPR-neutral
(latency-bound at occ 3, R6); in-wave sched_group_barrier → inert (no independent MFMA at NSUB=1);
cross-warp-group ping-pong (BOTH naive and the real phase-offset design) → discards because an 8-wave
workgroup drops to 2 waves/SIMD while the baseline 4-wave workgroup runs 3 INDEPENDENT workgroups/SIMD
that already overlap MFMA/VALU for free (MfmaUtil 26%); **the 4th wave is structurally unreachable**
(R6: gutting o_acc to DT=1 still gives 3 waves — compiler is liveness-optimal, resident set ~37 VGPR
over the 128 threshold); **every memory-spill VGPR-relief raises VGPR** (R6: S-in-LDS +19, K-staging
+54 — three independent confirmations). **The one true open path is a wheel with wide async-to-LDS
(unlocking kN0=128) or AGPR accumulation;** both are *compiler/wheel* features, not kernel-authorable
in 0.2.0. Everything kernel-side (overlap, occupancy, softmax op-count, atom choice, staging,
ping-pong, named barriers) is now ruled out with measured evidence. Small-seq is launch/prologue-floored
(R6: per-seq small-q-tile dispatch was a DVFS artifact), not split-K-fixable (probe ceiling 1.17×).
**Net for this wheel: 117/140 is at/near the ceiling; further gains need an upstream FlyDSL codegen
feature (wide async-to-LDS / AGPR), not another kernel lever.**

**Config caveat:** FlyDSL uses `page_size=16`; CK path uses aiter `vectorized` layout with
`page_size=1024` (`CK_PAGE_SIZE`). Quant: CK aiter **per-tensor** fp8 descale; FlyDSL per-token-head
Q/K + per-head V — not identical numerics, but same operator class (fp8 causal batch prefill).
