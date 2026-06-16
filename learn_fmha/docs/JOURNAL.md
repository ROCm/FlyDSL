# Session Journal — FlyDSL FMHA Setup, Benchmark & Autoresearch

**Host:** Linux 5.15, AMD Instinct MI308X (gfx942), ROCm 7.13 · **flydsl** 0.2.0 (PyPI wheel)
**Kernel:** `kernels/fmha_prefill_fp8_ck_hk5.py` · **Branch:** `opt/fmha-hk5-loop`
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

Autonomous commit→measure→keep/reset loop on `opt/fmha-hk5-loop`. Goal: close the FlyDSL→CK-Tile
gap at large seq.

### Result (MI308X, bs=1 nq8 nk1 causal)

| Shape | Baseline | **After** | CK-Tile | Ratio CK/FlyDSL before → after |
|---|---|---|---|---|
| sq=1024  | 4–6 TF | 6 TF (0.344 ms)  | 37 TF (0.059 ms) | 6.2× (unchanged, grid-starved) |
| sq=16384 | 103 TF | **108 TF** (5.32 ms) | 170 TF (3.23 ms) | 1.65× → **1.57×** |
| sq=32768 | 122 TF | **128 TF** (18.0 ms) | 177 TF (12.46 ms) | 1.45× → **1.38×** |

Final kernel: VGPR 166, LDS 18944, 0 spills. Net +5/+6 TF at large seq, all from cutting per-element
softmax VALU op-count. Correctness gate (max-abs err < 6e-2) green on all shapes incl. GQA,
p_scale∈{16,64}, non-causal, partial tiles. 9/9 `test_fmha_prefill_fp8` pass.

### Kept levers (4 commits)

| Commit | Lever | Effect |
|---|---|---|
| `5e3af84f` | Fold LOG2E into `qs` descale — scores enter log2-space, drops a per-element fmul | +2% @ sq16384/32768 |
| `15d8d1cc` | Drop `sched_mfma`/`sched_dsrd` hints — over-constrained the scheduler; removing gained perf *and* deleted code | +1–2 TF |
| `2f7e2484` | Fold `p_scale` shift into softmax pivot (`safe_m_p = safe_m - log2_pscale`) — removes one VALU add per element | +2 TF |
| `c1e3247b` | Docstring PERF table + lever ledger refresh | (doc) |

### Diagnosis (the durable finding)

- **VALU-bound, not memory/transpose.** PMC: VALUBusy 62%, MfmaUtil 24%, LDSBankConflict 1.5%, MemUnitStalled 0.16%. Every change that cut VALU op-count without adding register state won; everything else regressed.
- **The residual gap is occupancy, not VALU.** Upper-bound probe (gut exp2 + max-reduction → VALU-free) reaches only ~129/156 TF, *below* CK's 170/177. So even a perfect softmax can't close the gap from inside the loop — the wall is the 3-waves/SIMD occupancy cliff (VGPR pinned at 166; a 4th wave needs ≤128). o_acc accumulators alone are 64 VGPR, allocated to VGPR not AGPR (`agpr_count=0`).

### Dead-ends measured (do not re-try)

- `waves_per_eu` / `maxnreg` compile hints — plumbed through **both** the embedded `gpu-module-to-binary` and external-LLVM (`FLYDSL_COMPILE_LLVM_DIR`) paths, but `--amdgpu-num-vgpr=128` is **advisory, not enforced**: VGPR stays 166, no 4th wave. Closes lever L4 in the 0.2.0 wheel.
- +VGPR levers (kdv-hoist, vectorized descale, KT=64) all regress (86/93/69 TF) — confirms hard VGPR-binding.
- Halving the P-transpose `ds_bpermute` — **wrong** (err 3.0+): the 4 bpermutes gather from distinct source-lane halves, not redundant.
- `buffer_load_to_lds` (async-K, `FMHA_BUFK=1`) — still broken in 0.2.0 (err 3.0).
- AGPR accumulation — not exposed in 0.2.0. Manual max/sum reduction trees — compiler already schedules optimally. `s_setprio` removal — regresses (existing setprio is load-bearing).
- Older dead-ends: KT>32, NWAVES≠4, maxnumf, batched-within-tile overlap, readfirstlane, XOR-swizzle, split-K (wash at small seq), DIAG=0/smaller-tile (only 7 TF at sq1024).

### Remaining gap is structural / out of scope for this loop

Closing the last 1.38–1.57× needs a compiler-level lever (enforce VGPR cap → 4 waves, or AGPR
accumulation) or a major rewrite (persistent/finer-grained scheme for small-seq grid starvation,
fixed async-K, gfx950 `ds_read_tr`). None reachable from kernel source in the 0.2.0 wheel.

**Config caveat:** FlyDSL uses `page_size=16`; CK path uses aiter `vectorized` layout with
`page_size=1024` (env `CK_PAGE_SIZE`). Quantization: CK uses aiter **per-tensor** fp8 descale;
FlyDSL uses per-token-head Q/K + per-head V descale — not identical numerics, but same operator
class (fp8 causal batch prefill, vec_k_col_v-style layout).
