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
| This guide | `~/workspace/docs/REPRODUCIBILITY.md` |
| Session journal | `~/workspace/docs/JOURNAL.md` |

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
