# Reproducibility Guide — Knowledge Base + FlyDSL FMHA Benchmark

Step-by-step commands to reproduce the setup and benchmark on an AMD MI308X-class host with ROCm 7.x, Docker, and GitHub SSH access.

---

## Prerequisites

- Linux host with AMD GPU (`/dev/kfd`, `/dev/dri`)
- Docker with GPU passthrough (`--device=/dev/kfd --device=/dev/dri --group-add=video`)
- GitHub SSH key authorized for `anhminhnguyenhoang` and `ROCm` org repos
- Network access to `registry-sc-harbor.amd.com` (AMD internal ROCm image) or substitute a public ROCm image (see §6)

---

## 1. Configure SSH for GitHub

```bash
# If you don't already have ~/.ssh/config for GitHub:
cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/anguyenh
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
ssh -T git@github.com
# Expected: "Hi anhminhnguyenhoang! You've successfully authenticated..."
```

Replace `IdentityFile` with your key path if different.

---

## 2. Clone & install the knowledge base

```bash
mkdir -p ~/workspace
cd ~/workspace

git clone git@github.com:anhminhnguyenhoang/claude-knowledge-base.git
ln -sfn ~/workspace/claude-knowledge-base ~/knowledge-base

# Wire into Claude Code (once per machine)
mkdir -p ~/.claude/skills
cp -r ~/knowledge-base/.claude/skills/* ~/.claude/skills/

# Create or append to ~/.claude/CLAUDE.md — see ~/knowledge-base/SETUP.md §1
```

Verify:

```bash
ls ~/knowledge-base/README.md
ls ~/.claude/skills/kb-add-topic/SKILL.md
```

---

## 3. Clone FlyDSL (FMHA branch)

```bash
cd ~/workspace

# Branch name is opt/fmha-batch-prefill-fp8 (learn_fmha/ is a folder inside it)
git clone -b opt/fmha-batch-prefill-fp8 git@github.com:ROCm/FlyDSL.git
cd FlyDSL
git branch --show-current
# Expected: opt/fmha-batch-prefill-fp8
```

---

## 4. Devcontainer setup

The repo includes:

- `.devcontainer/Dockerfile` — extends ROCm 7.13 + PyTorch 2.10 gfx94X image
- `.devcontainer/devcontainer.json` — GPU passthrough, `HIP_VISIBLE_DEVICES=2`, post-create installs `flydsl`

### Option A — VS Code / Cursor Dev Containers

1. Open `~/workspace/FlyDSL` in Cursor/VS Code.
2. **Reopen in Container** (uses `.devcontainer/devcontainer.json`).
3. Wait for `postCreateCommand` (`pip install flydsl`).

### Option B — Manual Docker (no IDE)

```bash
cd ~/workspace/FlyDSL

# Build dev image (once)
docker build -t flydsl-fmha-dev:local -f .devcontainer/Dockerfile .

# Shell into environment
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --ipc=host --shm-size=64g \
  -e HIP_VISIBLE_DEVICES=2 \
  -v "$PWD:/workspaces/FlyDSL" \
  -w /workspaces/FlyDSL \
  flydsl-fmha-dev:local \
  bash
```

Inside the container:

```bash
pip install flydsl
python -c "import flydsl, torch; print(torch.cuda.get_device_name(0))"
# Expected: AMD Instinct MI308X (or your gfx942 GPU)
```

---

## 5b. Install aiter (for CK-Tile baseline)

```bash
cd ~/workspace
git clone git@github.com:ROCm/aiter.git
cd aiter
git submodule sync && git submodule update --init --recursive   # required (~6 min)
```

Inside the devcontainer (GPU required for import/JIT):

```bash
git config --global --add safe.directory /workspaces/aiter
pip install einops
cd /workspaces/aiter
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
python -c "import aiter; print('aiter OK')"
```

Mount aiter alongside FlyDSL:

```bash
-v ~/workspace/aiter:/workspaces/aiter \
-e AITER_ROOT=/workspaces/aiter
```

---

## 5c. Compare FlyDSL hk5 vs CK-Tile

```bash
cd /workspaces/FlyDSL
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 \
  --ck \
  --no-pyisa
```

**One-liner from host:**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --ipc=host --shm-size=64g \
  -e HIP_VISIBLE_DEVICES=2 \
  -e AITER_ROOT=/workspaces/aiter \
  -v ~/workspace/FlyDSL:/workspaces/FlyDSL \
  -v ~/workspace/aiter:/workspaces/aiter \
  flydsl-fmha-dev:local \
  bash -lc 'git config --global --add safe.directory /workspaces/aiter && pip install -q flydsl einops && cd /workspaces/aiter && AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop >/dev/null 2>&1 && cd /workspaces/FlyDSL && python3 tests/kernels/bench_fmha_compare.py --kernels fmha_prefill_fp8_ck_hk5 --ck --no-pyisa'
```

Reference run (2026-06-16, MI308X, GPU 2):

| Shape | FlyDSL hk5 | CK-Tile |
|-------|------------|---------|
| sq=1024 | 6 TF | 37 TF |
| sq=16384 | 103 TF | 170 TF |
| sq=32768 | 122 TF | 177 TF |

First `--ck` run JIT-compiles CK FMHA kernels (~1–2 min). Override CK page size: `CK_PAGE_SIZE=1024` (default).

---

## 5. Run the FMHA hk5 benchmark

From inside the devcontainer (or the manual `docker run` shell above):

```bash
cd /workspaces/FlyDSL

# Unified benchmark (customer shapes: bs=1, nq=8, nk=1, causal)
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 \
  --no-pyisa
```

**One-liner from host** (no interactive shell):

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --ipc=host --shm-size=64g \
  -e HIP_VISIBLE_DEVICES=2 \
  -v ~/workspace/FlyDSL:/workspaces/FlyDSL \
  -w /workspaces/FlyDSL \
  flydsl-fmha-dev:local \
  bash -lc 'pip install -q flydsl && python3 tests/kernels/bench_fmha_compare.py --kernels fmha_prefill_fp8_ck_hk5 --no-pyisa'
```

### Expected output format

```
shape                     fmha_prefill_fp8_ck_hk5
-------------------------------------------------
b1 sq1024 nq8 nk1                    X.XXXms/YTF
b1 sq16384 nq8 nk1                   X.XXXms/YTF
b1 sq32768 nq8 nk1                   X.XXXms/YTF
```

Reference run (2026-06-16, MI308X, GPU 2): **6 / 103 / 122 TF** at sq 1024 / 16384 / 32768.

### Benchmark variants

```bash
# Additional sequence lengths
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 \
  --seqs 1024 2048 16384 32768 \
  --no-pyisa

# Compare multiple FlyDSL kernel variants
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 fmha_prefill_fp8_8wave \
  --no-pyisa
```

### Correctness tests

```bash
pip install pytest pandas
python -m pytest tests/kernels/test_fmha_prefill_fp8.py -v
# Note: default test imports fmha_prefill_fp8 (reference kernel); edit import for hk5-specific tests
```

---

## 6. GPU & environment notes

| Variable | Value | Why |
|----------|-------|-----|
| `HIP_VISIBLE_DEVICES` | `2` (or 2–7) | GPUs 0–1 documented as broken on this cluster |
| `--shm-size` | `64g` | Large tensor allocations in FMHA |
| `--ipc=host` | required | ROCm / PyTorch multi-process stability |

**Do not use `pip install -e .`** for benchmark-only work — it triggers a full MLIR build (`scripts/build.sh`). Use **`pip install flydsl`** and keep the branch checkout mounted for `kernels/`.

For **compiler development**, build from source:

```bash
bash scripts/build_llvm.sh -j$(nproc)   # ~30 min first time
bash scripts/build.sh -j$(nproc)
pip install -e .
```

---

## 7. Alternative base image (no AMD registry)

If `registry-sc-harbor.amd.com/...` is unavailable, edit `.devcontainer/Dockerfile`:

```dockerfile
FROM rocm/dev-ubuntu-24.04:7.2-complete
RUN pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
RUN pip install flydsl pytest pandas ninja cmake pybind11 nanobind
WORKDIR /workspaces/FlyDSL
```

Then rebuild and re-run §4–§5.

---

## 8. File locations summary

| Item | Path |
|------|------|
| Knowledge base | `~/knowledge-base` (symlink → `~/workspace/claude-knowledge-base`) |
| FlyDSL repo | `~/workspace/FlyDSL` |
| aiter repo | `~/workspace/aiter` |
| Target kernel | `~/workspace/FlyDSL/kernels/fmha_prefill_fp8_ck_hk5.py` |
| Benchmark script | `~/workspace/FlyDSL/tests/kernels/bench_fmha_compare.py` |
| FMHA tutorials | `~/workspace/FlyDSL/learn_fmha/` |
| Devcontainer | `~/workspace/FlyDSL/.devcontainer/` |
| This guide | `learn_fmha/docs/REPRODUCIBILITY.md` |
| Session journal | `learn_fmha/docs/JOURNAL.md` |

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied (publickey)` on git clone | Check `~/.ssh/config` and `ssh -T git@github.com` |
| `No module named flydsl` | `pip install flydsl` inside container |
| `CUDA not available` | Add `--device=/dev/kfd --device=/dev/dri --group-add=video` |
| Wrong GPU / hang | Set `HIP_VISIBLE_DEVICES=2` |
| `pip install -e .` fails | Use PyPI `flydsl` wheel; only `-e .` when building compiler |
| CK column shows `n/a` | Run `git submodule update --init --recursive` in aiter; rebuild inside GPU container |
| `fmha_fwd.hpp` not found | Same — composable_kernel submodule missing |
| PyISA column missing | Expected without `/workspaces/amir/asm/fwd_fp8`; use `--no-pyisa` |

---

## 10. Related KB material

Before optimizing FMHA further, consult:

- `~/knowledge-base/ck-dsl-runbook/` — optimization runbook & methodology
- `~/knowledge-base/flydsl-jdbba/` — FlyDSL measurement discipline
- `~/workspace/FlyDSL/FMHA_FP8_OPTIMIZATION_HANDOFF.md` — full port context
- `~/workspace/FlyDSL/learn_fmha/README.md` — profiling loop & lesson index

---

## 11. Reproduce the hk5 autoresearch optimization loop (2026-06-16)

Branch `opt/fmha-hk5-loop` carries commits that lift large-seq from 103/122 →
**115/137 TF** (CK ratio 1.65/1.45 → 1.48/1.29). See [`JOURNAL.md`](JOURNAL.md)
§3 for the full result table, kept levers, and dead-end ledger. To reproduce:

### Setup (inside the devcontainer, aiter built per §5b)

```bash
cd /workspaces/FlyDSL
git checkout opt/fmha-hk5-loop      # off opt/fmha-batch-prefill-fp8
git log --oneline -4                # 5e3af84f, 15d8d1cc, 2f7e2484, c1e3247b
```

The 4 commits are independent VALU-op-count cuts in
`kernels/fmha_prefill_fp8_ck_hk5.py`; no toolchain change is needed (kernel code
is read from the mounted tree, not the wheel).

### Verify correctness (one shape per process — see §5)

```bash
cd /workspaces/FlyDSL
python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck_hk5  1 256 256 1 8 1 16 1.0   # base
python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck_hk5  1 64  64  2 2 1 16 64.0  # p_scale cancel
python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck_hk5  1 64  64  1 1 0 16 1.0   # non-causal
# each prints "... -> ERR <x> OK|FAIL"; OK iff err < 6e-2
python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py -v   # 9 passed
```

### Measure vs CK-Tile

```bash
python3 tests/kernels/bench_fmha_compare.py \
  --kernels fmha_prefill_fp8_ck_hk5 --ck --no-pyisa > /tmp/bench.log 2>&1
grep -E 'sq(16384|32768)' /tmp/bench.log
# expect ~5.10ms/108TF (sq16384), ~17.2ms/128TF (sq32768) vs CK 170/177
```

### The loop discipline (for continuing the search)

Per the `fmha-hk5-autoresearch` skill: commit each attempt FIRST, run correctness
gate (err<6e-2 all shapes), measure all 3 seq lengths, confirm structural changes
in ISA/PMC — then keep (advance branch) or `git reset` (log as dead-end). Levers
already proven dead this session are listed in JOURNAL §3; do not re-try them.

---

## 12. Reproduce round-2 (maxnum win + CK disassembly), 2026-06-16

Round-2 added commit `250138ff` (maxnum → `v_max3_f32`), lifting large-seq to
**115/137 TF** (CK ratio 1.48/1.29). Full findings in [`JOURNAL.md`](JOURNAL.md) §3.

```bash
cd /workspaces/FlyDSL && git checkout opt/fmha-hk5-loop
# perf (expect ~115 TF @ sq16384, ~137 @ sq32768 vs CK 170/177):
python3 tests/kernels/bench_fmha_compare.py --kernels fmha_prefill_fp8_ck_hk5 --ck --no-pyisa \
  > /tmp/b.log 2>&1; grep -E 'sq(16384|32768)' /tmp/b.log
python3 -m pytest tests/kernels/test_fmha_prefill_fp8.py -q   # 9 passed
```

### The reusable diagnostic — disassemble CK and diff the VALU histogram

The most productive method this round: compare our ISA op-mix against CK's actual
kernel (not guesses). This is how the maxnum win (`v_max3` gap) was found and how
the "CK runs at 2 waves" reframing was established.

```bash
# 1. Our ISA op histogram:
rm -rf /tmp/isa
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python3 tests/kernels/ck_check.py fmha_prefill_fp8_ck_hk5 1 16384 16384 1 8 1 16 1.0 >/dev/null 2>&1
D=$(find /tmp/isa -name 21_final_isa.s|head -1)
grep -iE 'vgpr_count|agpr_count|group_segment_fixed' $D | sort -u       # resources
for op in v_mfma v_exp v_max_f32 v_max3_f32 v_fma_f32 v_mul_f32 v_pk_mul v_cndmask ds_bpermute; do
  echo "$op: $(grep -cE "$op" $D)"; done

# 2. Carve CK's gfx942 code object from the aiter fatbin and disassemble:
SO=/workspaces/aiter/aiter/jit/mha_batch_prefill_fp8bf16_*_pertensor_nsink.so
python3 - "$SO" <<'PY'
import sys,struct
so=__import__('glob').glob(sys.argv[1])[0]
data=open(so,'rb').read()
# .hip_fatbin section offset is in `llvm-readelf -S`; here we scan for the AMDGPU ELF (e_machine 0xE0)
i=data.find(b'\x7fELF')
while i>=0:
    if struct.unpack_from('<H',data,i+18)[0]==0xE0:
        open('/tmp/ck.elf','wb').write(data[i:]); print('wrote /tmp/ck.elf @',hex(i)); break
    i=data.find(b'\x7fELF',i+4)
PY
# metadata (VGPR/AGPR/LDS per kernel — CK fp8 prefill is VGPR=214 -> 2 waves):
/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/ck.elf | grep -iE 'vgpr_count|sgpr_count|group_segment|\.name' | head -20
# op histogram to diff against ours:
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942 /tmp/ck.elf > /tmp/ck.s 2>/dev/null
for op in v_mfma v_exp v_max3_f32 v_fma_f32 v_mul_f32 v_pk_mul v_cndmask; do
  echo "CK $op: $(grep -cE "$op" /tmp/ck.s)"; done
```

Findings that drove round-2: CK VGPR=214 (2 waves, lower than our 3) yet 1.4–1.6×
faster ⇒ the gap is MFMA/VALU **overlap**, not occupancy. CK uses 33 `v_max3` vs
our (then) 68 `v_max` ⇒ switch the softmax reduction to `arith.maxnumf` (the DSL
`maximumf` lowers to NaN-propagating `llvm.maximum.f32`, which the backend will
NOT fuse into `v_max3`). Dead-ends this round (do NOT re-try): FMA-contract
fastmath (VGPR+reassoc regress), `iglp_opt` (worse than flydsl scheduler),
vectorized descale (VGPR 196), KT>32 even with maxnum, async-K via
`buffer_load_to_lds` (correct in `fmha_prefill_fp8_ck_async.py` but 32–64 TF —
gfx942 async-to-LDS is 4B/lane serialized), and the L1 cross-tile pipeline (works
+ improves interleaving but VGPR 165→213 → 2 waves → net loss; gated on a ~40-VGPR
cut to keep the 3rd wave).
