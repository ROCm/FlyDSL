# Session Journal — FlyDSL FMHA Setup & Benchmark

**Date:** 2026-06-16  
**Host:** Linux 5.15, AMD Instinct MI308X (gfx942), ROCm 7.13  
**Operator:** automated setup via Cursor agent

---

## Objective

1. Clone and install the personal knowledge base (`claude-knowledge-base`).
2. Clone FlyDSL branch `opt/fmha-batch-prefill-fp8`, create a devcontainer, and benchmark `kernels/fmha_prefill_fp8_ck_hk5.py`.
3. Document setup and results for reproducibility.

---

## What Was Done

### 1. SSH & GitHub access

- Found SSH private key at `~/.ssh/anguyenh` (no `~/.ssh/config` existed).
- Created `~/.ssh/config` pointing `github.com` → `IdentityFile ~/.ssh/anguyenh`.
- Added GitHub to `known_hosts` via `ssh-keyscan`.
- Verified: `Hi anhminhnguyenhoang! You've successfully authenticated`.

### 2. Knowledge base

| Step | Action |
|------|--------|
| Clone | `git clone git@github.com:anhminhnguyenhoang/claude-knowledge-base.git` → `~/workspace/claude-knowledge-base` |
| Symlink | `ln -sfn ~/workspace/claude-knowledge-base ~/knowledge-base` |
| Claude awareness | Created `~/.claude/CLAUDE.md` with Knowledge Base section (per `SETUP.md`) |
| Skills | Copied all skills from `~/knowledge-base/.claude/skills/` → `~/.claude/skills/` |

Topics available: `hipkitten/`, `ck-dsl-runbook/`, `flash-attention-v4/`, `flydsl-jdbba/`.

### 3. FlyDSL work repo

| Step | Action |
|------|--------|
| Clone | `git clone -b opt/fmha-batch-prefill-fp8 git@github.com:ROCm/FlyDSL.git` → `~/workspace/FlyDSL` |
| Branch note | URL path `.../tree/opt/fmha-batch-prefill-fp8/learn_fmha` means branch `opt/fmha-batch-prefill-fp8`, folder `learn_fmha/` |
| Devcontainer | Added `.devcontainer/Dockerfile` + `.devcontainer/devcontainer.json` |
| Base image | `registry-sc-harbor.amd.com/framework/therock-main:1192_gfx94X_7.13.0.dev0-...` (ROCm 7.13, PyTorch 2.10, gfx94X) |
| Image tag | `flydsl-fmha-dev:local` |

**Install note:** `pip install -e .` triggers a full MLIR source build via `scripts/build.sh` and failed in the container (~4 min, build error). For benchmarking branch kernels, **`pip install flydsl` (PyPI 0.2.0)** + bind-mount the repo is sufficient — kernel code lives in `kernels/`, not in the package.

### 4. Benchmark

**Command:**

```bash
HIP_VISIBLE_DEVICES=2 python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 --no-pyisa
```

**Environment:** Docker container with `/dev/kfd`, `/dev/dri`, GPU 2 (GPUs 0–1 noted as broken in `learn_fmha/README.md`).

**Results (2026-06-16):**

| Shape (bs=1, nq=8, nk=1, causal) | Latency | TFLOPS |
|-------------------------------------|---------|--------|
| sq=1024 | 0.345 ms | **6 TF** |
| sq=16384 | 5.321 ms | **103 TF** |
| sq=32768 | 18.002 ms | **122 TF** |

Kernel docstring reference (MI308X, flydsl 0.2.0): 5 / 16 / 61 / 69 TF at sq 1024/2048/16384/32768 — our large-seq numbers (103–122 TF) exceed the docstring baseline, consistent with hk5 LDS padding optimizations on this hardware.

### 5. Artifacts created

| Path | Purpose |
|------|---------|
| `~/workspace/FlyDSL/.devcontainer/` | Devcontainer for FMHA work |
| `learn_fmha/docs/REPRODUCIBILITY.md` | Step-by-step setup guide |
| `learn_fmha/docs/JOURNAL.md` | This file |
| `~/.ssh/config` | GitHub SSH identity |
| `~/.claude/CLAUDE.md` | KB awareness for Claude Code |
| `~/knowledge-base` → workspace clone | KB symlink |

---

## Issues & Decisions

1. **Branch name:** `opt/fmha-batch-prefill-fp8/learn_fmha` is not a valid git ref; use branch `opt/fmha-batch-prefill-fp8`.
2. **Editable install:** Avoid `pip install -e .` unless developing the FlyDSL compiler itself; use PyPI wheel for kernel work.
3. **GPU selection:** Use `HIP_VISIBLE_DEVICES=2` (or 2–7) per project docs.
4. **PyISA baseline:** `--no-pyisa` used because `/workspaces/amir/asm/fwd_fp8` is not present on this host.

---

## Next Steps (optional)

- Run correctness: `pytest tests/kernels/test_fmha_prefill_fp8.py` (may need import swap to hk5).
- Work through `learn_fmha/lesson_*.py` tutorials inside the devcontainer.
- For compiler changes: `bash scripts/build_llvm.sh && bash scripts/build.sh && pip install -e .` inside a longer-lived container.

---

## 6. aiter install + CK-Tile comparison (2026-06-16, continued)

### aiter setup

| Step | Action |
|------|--------|
| Clone | `git clone git@github.com:ROCm/aiter.git` → `~/workspace/aiter` |
| Submodules | `git submodule sync && git submodule update --init --recursive` (~6 min; pulls `composable_kernel`) |
| Install (devcontainer) | `AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop` inside GPU container |
| Git safe.dir | `git config --global --add safe.directory /workspaces/aiter` (required for bind-mount) |

First FMHA kernel JIT build takes ~40s per variant after submodules are present.

### FlyDSL hk5 vs CK-Tile fp8

**Command:**

```bash
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 --ck --no-pyisa
```

**Results (MI308X, GPU 2):**

| Shape | FlyDSL hk5 | CK-Tile (aiter) | CK / FlyDSL |
|-------|------------|-----------------|-------------|
| sq=1024 | 6 TF (0.344 ms) | 37 TF (0.059 ms) | **6.2×** |
| sq=16384 | 103 TF (5.319 ms) | 170 TF (3.232 ms) | **1.65×** |
| sq=32768 | 122 TF (18.014 ms) | 177 TF (12.456 ms) | **1.45×** |

FlyDSL closes most of the gap at large seq (~69% of CK at sq=32768), but CK wins at all shapes. Small-seq FlyDSL is grid-starved (6 TF vs CK 37 TF).

---

## 7. Autoresearch optimization loop — hk5 VALU folds (2026-06-16)

Ran the `fmha-hk5-autoresearch` skill (autonomous commit→measure→keep/reset loop)
on branch `opt/fmha-hk5-loop` (off `opt/fmha-batch-prefill-fp8`). Goal: close the
FlyDSL→CK-Tile gap at large seq. Repro of the loop itself is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §11.

### Result

| Shape (bs=1, nq8, nk1, causal) | Baseline | **After** | CK-Tile | Ratio before → after |
|---|---|---|---|---|
| sq=1024  | 4–6 TF | 6 TF  | 37 TF  | 6.2× (unchanged, grid-starved) |
| sq=16384 | 103 TF | **108 TF** | 170 TF | 1.65× → **1.57×** |
| sq=32768 | 122 TF | **128 TF** | 177 TF | 1.45× → **1.38×** |

Net: +5/+6 TF at large seq, all from cutting per-element softmax VALU op-count
(the binding resource — PMC shows VALUBusy 62% vs MfmaUtil 24%). Correctness gate
(max-abs err < 6e-2) green on all shapes incl. GQA, p_scale∈{16,64}, non-causal,
partial tiles. 9/9 `test_fmha_prefill_fp8` pass.

### Kept levers (4 commits)

| Commit | Lever | Effect |
|---|---|---|
| `5e3af84f` | Fold LOG2E into `qs` descale — scores enter log2-space, drops a per-element fmul | +2% @ sq16384/32768 |
| `15d8d1cc` | Drop `sched_mfma`/`sched_dsrd` hints — they over-constrained the scheduler; removing them gained perf *and* deleted code | +1–2 TF |
| `2f7e2484` | Fold `p_scale` shift into the softmax pivot (`safe_m_p = safe_m - log2_pscale`) — removes one VALU add per element | +2 TF |
| `c1e3247b` | Docstring PERF table + lever ledger refresh | (doc) |

### Diagnosis (the durable finding)

- **VALU-bound, not memory/transpose.** PMC: VALUBusy 62%, MfmaUtil 24%,
  LDSBankConflict 1.5%, MemUnitStalled 0.16%. Every change that cut VALU op-count
  without adding register state won; everything else regressed.
- **The residual gap is occupancy, not VALU.** Upper-bound probe (gut exp2 +
  max-reduction → VALU-free) reaches only ~156 TF, *below* CK's 177. So even a
  perfect softmax can't close the gap from inside the loop — the wall is the
  3-waves/SIMD occupancy cliff (VGPR pinned at 166; a 4th wave needs ≤128).

### Dead-ends measured this session (do not re-try)

- `waves_per_eu` / `maxnreg` compile hints — plumbed through **both** the embedded
  `gpu-module-to-binary` and the external-LLVM (`FLYDSL_COMPILE_LLVM_DIR`) codegen
  paths, but `--amdgpu-num-vgpr=128` is **advisory, not enforced**: VGPR stays 166,
  no 4th wave. This closes lever L4 in the 0.2.0 wheel.
- +VGPR levers (kdv-hoist, vectorized descale, KT=64) all regress (86/93/69 TF) —
  confirms hard VGPR-binding; any lever that grows live state loses.
- Halving the P-transpose `ds_bpermute` — **wrong** (err 3.0+): the 4 bpermutes
  gather from distinct source-lane halves, not redundant.
- `buffer_load_to_lds` (async-K, `FMHA_BUFK=1`) — still broken in 0.2.0 (err 3.0).
- AGPR accumulation (`agpr_count=0`; o_acc lives in VGPR) — not exposed in 0.2.0.

### Remaining gap is structural / out of scope for this loop

Closing the last 1.38–1.57× needs a compiler-level lever (enforce VGPR cap →
4 waves, or AGPR accumulation) or a major rewrite (persistent/finer-grained
scheme for small-seq grid starvation, fixed async-K). None reachable from kernel
source in the 0.2.0 wheel.

**Config caveat:** FlyDSL uses `page_size=16`; CK path uses aiter `vectorized` layout with `page_size=1024` (env `CK_PAGE_SIZE`). Quantization: CK uses aiter **per-tensor** fp8 descale; FlyDSL uses per-token-head Q/K + per-head V descale — not identical numerics/setup, but same operator class (fp8 causal batch prefill, vec_k_col_v-style layout).
