# FlyDSL Flash Attention Forward Kernel (bf16) Optimization Log

Target: Align FlyDSL `flash_attn_func` kernel bf16 performance with CK FMHA forward kernel on MI350X (gfx950)

- **Benchmark config**: `--batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100` (causal)
- **Target kernel data type**: bf16
- **Reference kernel data type**: bf16 (CK `fmha_fwd_d128_bf16_batch_...psddv...gfx9`)
- **Reference kernel baseline**: 910 TFLOPS
- **Target kernel starting point**: 691.0 TFLOPS (committed at `f9aff08`, FlyDSL branch `hyg_mha_new_api_MI350X`)
- **Target kernel final result**: **~700 TFLOPS** (76.9% of reference, LLVM default opts change uncommitted)

---

## 1. Performance Progression

| # | Optimization | TFLOPS | Delta | MaxErr | Commit |
|---|---|---|---|---|---|
| 0 | Baseline (bf16 support + DMA double-buffer + causal mask) | 691.0 | -- | 3.91e-03 | FlyDSL `f9aff08`, LLVM `4604179967de` |
| 1 | LLVM default opts: `-enable-post-misched=0 --lsr-drop-solution=1` | ~700 | +~9 (+1.3%) | 3.91e-03 | **UNCOMMITTED** (llvm-project `ModuleToObject.cpp`) |

> **Note**: Row 1 is based on an uncommitted change in `llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp`. All other optimization attempts (15+) regressed performance and were reverted. The commit hash will be added once the change is committed and verified.

---

## 2. Effective Optimizations (Detailed)

### 2.1 LLVM Default Backend Flags (691.0 → ~700 TFLOPS)

**Problem**: LLVM's post-RA machine scheduler (`PostMachineScheduler`) and Loop Strength Reduction (LSR) were producing suboptimal instruction orderings for the FMHA kernel. Post-RA scheduling disrupted MFMA placement, and LSR introduced unnecessary address computation overhead.

**Changes**:
- Modified `llvm-project/mlir/lib/Target/LLVM/ModuleToObject.cpp` to embed `-enable-post-misched=0 --lsr-drop-solution=1` as default LLVM backend flags
- `-enable-post-misched=0`: disables the post-RA machine scheduler, preserving pre-RA instruction ordering through register allocation
- `--lsr-drop-solution=1`: allows LSR to drop solutions that increase register pressure, reducing unnecessary address recomputation
- Flags are applied automatically via `std::call_once` at `TargetMachine` creation, with `FLYDSL_LLVM_OPTS` env var available as an override

**Register impact**: VGPR 218, SGPR 91, spill 0 (unchanged from baseline)

---

## 3. Ineffective / Cancelled Optimizations

### 3.1 Tree Max Reduction — REVERTED

**Problem**: FlyDSL's softmax uses a sequential max-reduction across score chunks, serializing VALU operations.

**Investigation**:
- Attempted tree-based reduction (pairwise max → reduce depth from O(n) to O(log n))
- Benchmarked at 685 TFLOPS, a 6 TFLOPS regression from baseline

**Why reverted**: Tree reduction requires additional temporary VGPRs and introduces more VALU instructions. On MI350X where MFMA-VALU can run in parallel, the sequential pattern was already being hidden behind MFMA execution.

### 3.2 Inline O Rescale After GEMM2 — REVERTED

**Problem**: Output rescaling (`O *= correction_factor`) after GEMM2 creates a serial dependency chain.

**Investigation**:
- Moved rescale operations inline immediately after GEMM2 MFMA completion
- Benchmarked at 692 TFLOPS, marginal regression

**Why reverted**: The rescale VALU operations were already overlapping with subsequent memory operations. Inlining them disrupted the compiler's scheduling of memory prefetches.

### 3.3 P Pack Before V Barrier — REVERTED

**Problem**: The `v_perm_b32` packing of P (attention scores) to bf16 occurs after the V data barrier, adding latency to the GEMM2 critical path.

**Investigation**:
- Moved P packing before the V barrier to overlap with barrier wait time
- Benchmarked at 698 TFLOPS, slight regression from baseline

**Why reverted**: Moving the pack earlier increased register pressure (live P values across barrier) and the compiler couldn't schedule MFMA chains as efficiently.

### 3.4 O Rescale Fused Into GEMM2 Accumulator — REVERTED

**Problem**: Separate rescale pass (`O *= corr`) after GEMM2 adds VALU cycles.

**Investigation**:
- Attempted to fold the correction factor directly into GEMM2's accumulator initialization
- Produced numerical errors exceeding the 8e-03 threshold

**Why reverted**: Numerical error. The rescale must be applied to the previous iteration's O before accumulating new GEMM2 results; fusing it into the accumulator changes mathematical semantics.

### 3.5 N_SUBTILES=1 and N_SUBTILES=4 Tile Changes — REVERTED

**Problem**: FlyDSL uses `BLOCK_N=64` with `N_SUBTILES=2` (processing 128 KV columns as 2×64), incurring 2× barriers and 2× softmax per 128 columns vs CK's single-pass.

**Investigation**:
- `N_SUBTILES=1` (`BLOCK_N=128`): Would require restructuring the entire MFMA tiling and LDS layout. Not a simple parameter change — the MFMA instruction `v_mfma_f32_32x32x16_bf16` processes 32×32×16 tiles, and K=16 steps with `BLOCK_N=128` requires different accumulator management.
- `N_SUBTILES=4` (`BLOCK_N=32`): Benchmarked, produced significant regression due to 4× barrier overhead.

**Why reverted**: `N_SUBTILES=1` requires major architectural changes (not a parameter tweak). `N_SUBTILES=4` increased synchronization overhead by 4×.

### 3.6 Deeper V Read Pipeline — REVERTED

**Problem**: V data reads from LDS may not be sufficiently pipelined with GEMM2 MFMAs.

**Investigation**:
- Increased V read pipeline depth to prefetch more V tiles ahead of GEMM2
- Benchmarked at 697 TFLOPS, marginal regression

**Why reverted**: Deeper pipeline increased VGPR pressure for holding prefetched V data, and the additional LDS reads created port contention with K reads.

### 3.7 LLVM Backend Flag Sweep (12 Combinations) — NO IMPROVEMENT

**Problem**: Searching for additional LLVM backend flags beyond `-enable-post-misched=0 --lsr-drop-solution=1`.

**Investigation**:
- Tested 12 flag combinations including:
  - `-amdgpu-sched-strategy=max-ilp`
  - `-amdgpu-enable-merge-m0`
  - `-amdgpu-dpp-combine`
  - `-amdgpu-atomic-optimizations`
  - `-machine-sink-split-probability-threshold`
  - `-amdgpu-use-divergent-register-indexing`
  - Various combinations of the above
- None produced improvement beyond the baseline `-enable-post-misched=0 --lsr-drop-solution=1`

**Why cancelled**: Exhaustive flag search with no positive results. The scheduler and register allocator are already near-optimal for this kernel given its structural constraints.

### 3.8 V Double-Buffering with Early DMA Launch — NO IMPROVEMENT

**Problem**: V DMA barrier stall (~630 cycles) is the single largest barrier cost. Double-buffering V data could eliminate this wait.

**Investigation**:
- Launched V DMA for the next subtile during current subtile's GEMM1 phase
- Used separate LDS buffers for ping-pong V data
- Benchmarked at 700.8 TFLOPS, within noise of baseline

**Why cancelled**: The V DMA and GEMM1 compute already overlap to some degree. Explicit double-buffering added LDS management complexity but the DMA hardware was already saturated by K data transfers during GEMM1.

### 3.9 Pre-Read All K Packs — REVERTED

**Problem**: K data reads interleaved with MFMA pairs break MFMA chains (FlyDSL has all length-1 MFMA chains vs CK's better grouping).

**Investigation**:
- Pre-read all 8 K packs (for K_STEPS_QK=8) before starting MFMA chain
- Benchmarked at 683 TFLOPS, 8 TFLOPS regression

**Why reverted**: Pre-reading all K packs requires 8×2=16 extra VGPRs to hold the data simultaneously. This pushed VGPR usage past the occupancy threshold, and the compiler inserted additional spill/fill code. The interleaved pattern, while fragmenting MFMA chains, has lower peak register pressure.

### 3.10 sched_group_barrier MFMA Grouping — REVERTED

**Problem**: MFMA chains are all length-1 (fragmented by LDS reads and waitcnts).

**Investigation**:
- Used `rocdl.sched_group_barrier` with `mask_mfma`, `mask_dsrd`, `mask_vmem_rd` to enforce MFMA grouping
- Attempted various group sizes (4, 8, 16 MFMAs per group)
- Benchmarked at 693 TFLOPS, regression

**Why reverted**: Forced MFMA grouping delayed K data reads, causing later `lgkmcnt` waits to stall longer. The compiler's natural interleaving, while producing length-1 chains, actually hides LDS latency better than forced grouping.

### 3.11 PREFETCH_K=4 — REVERTED

**Problem**: Default `PREFETCH_K=2` may not provide enough K data lookahead to hide LDS read latency.

**Investigation**:
- Increased K prefetch depth from 2 to 4
- Benchmarked at 684 TFLOPS, 7 TFLOPS regression

**Why reverted**: Deeper K prefetch requires holding 4× K packs in VGPRs simultaneously, increasing register pressure significantly. The VGPR budget (218 used out of ~256 limit for occupancy) cannot accommodate the extra live values.

### 3.12 Branchless Causal Masking — REVERTED

**Problem**: FlyDSL uses conditional branches (`v_cmp` + `v_cndmask` with VCC) for causal masking, while CK uses SGPR-pair comparisons (`v_cmp_gt_i32_e64` + `v_cndmask_b32_e64`) avoiding VCC write-after-write hazards.

**Investigation**:
- Implemented CK-style branchless masking using SGPR-pair comparisons
- Benchmarked at 632 TFLOPS, 59 TFLOPS regression

**Why reverted**: The branchless approach required more SGPR pairs and additional VALU instructions for the comparison chain. The tile-level causal skip (`scf.IfOp`) already eliminates most masked tiles, so the per-element masking overhead is small. The regression was due to increased instruction count and register pressure.

### 3.13 s_setprio Around Barriers — REVERTED

**Problem**: High barrier stall cycles (1422 cycles total across 4 barriers) suggest wave desynchronization.

**Investigation**:
- Added `s_setprio(1)` before and `s_setprio(0)` after all `gpu.barrier()` calls (4 locations)
- Benchmarked at 683 TFLOPS, 8 TFLOPS regression

**Why reverted**: On MI350X with `waves_per_eu=2`, the wave scheduling is handled differently from MI308X's 8-wave time-multiplexing. The priority switching added overhead without reducing barrier stalls because the 8-wave workgroup's natural scheduling already provides reasonable overlap.

### 3.14 Pre-Scale Q to Eliminate v_fmamk_f32 — CANCELLED

**Problem**: FlyDSL has 64 `v_fmamk_f32` instructions for softmax scaling (vs CK's 1), contributing ~62 extra VALU instructions per iteration.

**Investigation**:
- Pre-scaled Q by `1/sqrt(d)` before the main loop to eliminate per-iteration scaling
- Analysis showed the 62 `v_fmamk_f32` instructions account for ~0.01% of total cycles because they overlap with MFMA execution on MI350X (MFMA-VALU parallelism)

**Why cancelled**: Negligible cycle savings (~0.01%). On MI350X, MFMA and VALU execute in parallel, so the `v_fmamk_f32` instructions are effectively free when scheduled during MFMA pipeline gaps.

### 3.15 V Double-Buffer: Launch Both V DMAs During Subtile 0 — REVERTED

**Problem**: Two subtiles require two V DMA + barrier sequences. Launching both DMAs during the first subtile could eliminate the second barrier.

**Investigation**:
- Modified V DMA launch to issue both `coop_dma_v(kv_start, 0)` and `coop_dma_v(kv_start + BLOCK_N, 1)` during subtile 0's GEMM1 phase
- Conditionally skipped the barrier for subtile 1
- Benchmarked at 693.6 TFLOPS, regression

**Why reverted**: Launching two DMAs simultaneously created HBM bandwidth contention. The DMA engine couldn't sustain two concurrent V tile fetches at full bandwidth, causing both DMAs to complete later than if they were sequential. The saved barrier cycles were offset by longer DMA latency.

### 3.16 MLIR Compilation O=3 — NO IMPROVEMENT

**Problem**: Default MLIR compilation uses `-O2`. Higher optimization level might produce better code.

**Investigation**:
- Changed `rocdl-attach-target` optimization level from `O=2` to `O=3`
- Benchmarked at 698 TFLOPS, same as O=2

**Why cancelled**: O=3 enables aggressive inlining and loop transformations that are not beneficial for this already-structured kernel. The MFMA-centric computation pattern is not amenable to standard O=3 transforms.

### 3.17 rocdl.sched_barrier(0) in GEMM1 Loop — NO IMPROVEMENT

**Problem**: Attempting to use scheduling barriers to force MFMA grouping and prevent interleaving with K reads.

**Investigation**:
- Inserted `rocdl.sched_barrier(0)` between MFMA and subsequent K loads in the GEMM1 inner loop
- Benchmarked at 691.2 TFLOPS, same as baseline

**Why cancelled**: `SCHED_BARRIER` is a zero-size pseudo-instruction. While it survives backend passes (after the `isMeta=1` removal in LLVM), the pre-RA scheduler was already producing the same ordering. The barrier didn't change the final ISA schedule.

---

## 4. ISA Comparison (Target vs Reference)

### 4.1 Register & Resource Usage

| Resource | FlyDSL | CK | Notes |
|---|---|---|---|
| VGPR | 218 | 256 | FlyDSL has headroom; CK is at max |
| SGPR | 91 | 91 | Identical |
| LDS (bytes) | 49,152 (48 KB) | 26,112 (25.5 KB) | FlyDSL uses 1.88× more LDS for DMA double-buffering |
| VGPR spills | 0 | 18 | CK has spills; FlyDSL is clean |
| SGPR spills | 0 | 0 | Both clean |
| Private segment | 0 | 76 | CK uses stack for spills |
| Workgroup size | 512 (8 waves) | 256 (4 waves) | FlyDSL uses 2× waves per workgroup |
| Accum offset | 220 | 256 | FlyDSL has more non-accum VGPRs available |

### 4.2 Instruction Count Comparison

| Instruction | FlyDSL | CK | Ratio | Notes |
|---|---|---|---|---|
| `v_mfma` | 64 | 64 | 1.00× | Same MFMA count (8 K-steps × 4 D-chunks × 2 GEMMs / subtiles) |
| `v_pk_mul` | 79 | 57 | 1.39× | FlyDSL has more packed multiplies for softmax rescaling |
| `v_mul_f32` | 41 | 20 | 2.05× | FlyDSL has 2× scalar multiplies |
| `v_fmamk_f32` | 64 | 1 | 64.0× | FlyDSL scales per-iteration; CK pre-scales Q |
| `v_fma_f32` | 0 | 69 | 0.00× | CK uses FMA; FlyDSL uses separate mul+add |
| `v_cmp` | 70 | 184 | 0.38× | CK has more comparisons (SGPR-pair style, branchless) |
| `v_cndmask` | 68 | 93 | 0.73× | CK has more conditional moves |
| `ds_read_b128` | 32 | 64 | 0.50× | CK reads all data via wide 128-bit LDS reads |
| `ds_read_b64_tr_b16` | 64 | 0 | ∞ | FlyDSL uses 64-bit transpose reads for K; CK uses none |
| `ds_read` (total) | 96 | 64 | 1.50× | FlyDSL issues 50% more LDS reads |
| `ds_write` | 0 | 16 | 0.00× | CK writes V to LDS pre-transposed; FlyDSL uses DMA |
| `buffer_load` | 10 | 26 | 0.38× | CK does more explicit VMEM loads |
| `buffer_store` | 0 | 18 | 0.00× | CK stores to buffer; FlyDSL uses global_store |
| `global_store` | 16 | 0 | ∞ | FlyDSL uses global stores for output |
| `v_exp` | 66 | 65 | 1.02× | Nearly identical softmax exp count |
| `v_cvt` | 70 | 72 | 0.97× | Nearly identical type conversions |
| `v_perm_b32` | 0 | 33 | 0.00× | CK uses perm for byte rearrangement |
| `s_nop` | 78 | 13 | 6.00× | FlyDSL has 6× more hazard NOPs |
| `s_waitcnt` | 63 | 64 | 0.98× | Nearly identical wait count |
| `s_barrier` | 4 | 8 | 0.50× | FlyDSL has fewer barriers but higher per-barrier stall |
| **Total ISA lines** | **1588** | **2145** | **0.74×** | FlyDSL has shorter ISA but worse performance |

### 4.3 Key Structural Differences

The fundamental architectural gap between FlyDSL and CK stems from three interrelated design choices:

1. **Tiling and subtile strategy**: FlyDSL uses `BLOCK_N=64` with `N_SUBTILES=2`, processing 128 KV columns as two separate 64-column passes. Each subtile incurs its own barrier, softmax, and DMA sequence. CK processes `BLOCK_N=128` in a single pass with a more sophisticated pipeline (`psddv` — pre-softmax double-buffered DMA with V prefetch), halving the per-128-column overhead.

2. **LDS read patterns**: FlyDSL reads K data via 64 `ds_read_b64_tr_b16` (64-bit transpose reads at 8 bytes each), while CK reads K and V via 64 `ds_read_b128` (128-bit reads at 16 bytes each). CK pre-transposes V data in LDS via `ds_write` before reading it back with wide loads. This means CK moves 2× more data per LDS instruction, reducing total LDS instruction count by 33% and LDS port contention.

3. **Wave configuration and occupancy**: FlyDSL launches 512 threads (8 waves) per workgroup with 218 VGPRs, effectively running at 1 wave/SIMD due to VGPR limits. CK launches 256 threads (4 waves) per workgroup with 256 VGPRs and 18 spills, also at 1 wave/SIMD but with a more aggressive register allocation that enables wider reads and fewer instructions per data element.

The 6× `s_nop` count (78 vs 13) is a symptom, not a cause: FlyDSL's fragmented MFMA chains (all length-1) create more MFMA→VALU transitions, each requiring hazard NOPs. CK's better instruction scheduling produces fewer transitions.

---

## 5. Remaining Performance Gap Analysis

| Factor | Estimated Impact | Feasibility to Fix |
|---|---|---|
| Subtile overhead (2× barriers, 2× softmax per 128 cols) | ~8-10% | Hard. Requires `BLOCK_N=128` single-pass architecture, which means restructuring MFMA tiling, LDS layout, and accumulator management. |
| Narrow LDS reads (64-bit transpose vs 128-bit wide) | ~5-7% | Hard. Requires pre-transposing V data in LDS (adding `ds_write` phase) and changing K read strategy to use wider loads. |
| Excessive s_nop from fragmented MFMA chains | ~3-4% | Medium. Requires better MFMA grouping in LLVM scheduler or manual scheduling barriers, but all attempts so far regressed. |
| 64 v_fmamk_f32 (per-iteration softmax scale) | ~0-1% | Easy but negligible. MFMA-VALU parallelism on MI350X hides these instructions. |
| 8-wave workgroup overhead (scheduling, barriers) | ~3-5% | Hard. Reducing to 4 waves requires halving the workgroup and restructuring the cooperative DMA model. |
| VGPR headroom not exploited (218 vs 256 available) | ~2-3% | Medium. 38 unused VGPRs could enable wider reads or deeper pipelines, but prior attempts (PREFETCH_K=4, pre-read all K) regressed due to scheduling disruption. |

**Root cause**: The 23% performance gap (700 vs 910 TFLOPS) is dominated by two structural factors: (1) the 2-subtile architecture doubling per-128-column overhead, and (2) the narrower LDS read pattern issuing 50% more LDS instructions. These are not tunable parameters — they are baked into the kernel's fundamental data movement and tiling strategy. Closing the gap requires a major architectural rewrite: moving to `BLOCK_N=128` single-pass with pre-transposed V reads and unified 128-bit LDS access patterns, essentially rebuilding the kernel to match CK's `psddv` pipeline.

---

## 6. Test Configurations & Verification

| Config | batch | seq_len | num_heads | head_dim | dtype | iters | Purpose |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 8192 | 64 | 128 | bf16 | 100 | Primary benchmark (target: 910 TFLOPS) |
| 2 | 1 | 8192 | 64 | 128 | fp16 | 100 | fp16 comparison (higher baseline) |
| 3 | 1 | 512 | 64 | 128 | bf16 | 100 | Small-scale quick verification |
| 4 | 1 | 256 | 64 | 128 | bf16 | 100 | Minimal-scale edge case |
| 5 | 8 | 512 | 64 | 128 | bf16 | 100 | Multi-batch throughput |

**Acceptance criteria**: MaxErr < 8e-03, MinCos > 0.999.

All causal attention. `waves_per_eu=2` for benchmarking, `waves_per_eu=1` for ATT profiling.

---

## 7. How to Run Performance Tests

### 7.1 Environment

- **Hardware**: AMD MI350X (gfx950, CDNA4), 256 CUs, 2200 MHz max clock
- **Docker container**: `hyg_trn_rocm7.1`
- **FlyDSL branch**: `hyg_mha_new_api_MI350X`
- **LLVM branch**: `flydsl_new_api_MI350X`
- **Workspace path**: `/home/yanguahe/code/my_asm_code`
- **FlyDSL path**: `/home/yanguahe/code/my_asm_code/FlyDSL`
- **LLVM path**: `/home/yanguahe/code/my_asm_code/llvm-project`

### 7.2 Check GPU Availability

```bash
docker exec hyg_trn_rocm7.1 bash -c "rocm-smi --showuse | egrep '0    |Device'"
```

### 7.3 Build (Incremental)

```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/llvm-project/buildmlir &&
  ninja -j128 &&
  cd /home/yanguahe/code/my_asm_code/llvm-project &&
  rm -rf mlir_install &&
  mkdir -p mlir_install &&
  cmake --install buildmlir --prefix mlir_install &&
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  rm -rf build-fly &&
  export MLIR_PATH=/home/yanguahe/code/my_asm_code/llvm-project/mlir_install &&
  bash scripts/build.sh -j128 &&
  pip install -e .
"
```

### 7.4 Build (Full — Before Commit)

```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/llvm-project &&
  rm -rf buildmlir && mkdir -p buildmlir && cd buildmlir &&
  cmake -G Ninja \
    -S ../llvm \
    -DLLVM_ENABLE_PROJECTS='mlir;clang' \
    -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;AMDGPU' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=\$(which python3) \
    -Dnanobind_DIR=\$(python3 -c \"import nanobind,os; print(os.path.dirname(nanobind.__file__)+'/cmake')\") \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF &&
  ninja -j128 &&
  cd /home/yanguahe/code/my_asm_code/llvm-project &&
  rm -rf mlir_install && mkdir -p mlir_install &&
  cmake --install buildmlir --prefix mlir_install &&
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  rm -rf build-fly &&
  export MLIR_PATH=/home/yanguahe/code/my_asm_code/llvm-project/mlir_install &&
  bash scripts/build.sh -j128 &&
  pip install -e .
"
```

### 7.5 Run Target Kernel Benchmark

**Quick run (via run.sh)**:
```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  HIP_VISIBLE_DEVICES=0 bash run.sh
"
```

**Direct command (primary benchmark)**:
```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  HIP_VISIBLE_DEVICES=0 \
  python3 tests/kernels/test_flash_attn_func.py \
    --dtype bf16 --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100
"
```

**Multi-config verification**:
```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  for SL in 8192 512 256; do
    HIP_VISIBLE_DEVICES=0 \
    python3 tests/kernels/test_flash_attn_func.py \
      --dtype bf16 --batch 1 --num_heads 64 --seq_len \$SL --head_dim 128 --iters 100
  done
"
```

### 7.6 Dump Target Kernel ISA

```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  rm -rf ~/.flydsl/cache/* &&
  HIP_VISIBLE_DEVICES=0 FLIR_DUMP_ISA=1 FLIR_NO_CACHE=1 FLIR_REBUILD=1 \
  python3 tests/kernels/test_flash_attn_func.py \
    --dtype bf16 --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --iters 5
"
```

ISA output: `/root/.flydsl/debug/flash_attn_func_kernel_0/15_final_isa.s`

### 7.7 Run Reference Kernel (CK) Benchmark

```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/aiter &&
  HIP_VISIBLE_DEVICES=0 python op_tests/test_mha_ck_vs_torch.py \
    -b 1 -n 64 -q 8192 -k 8192 -d_qk_v 128,128 \
    -d bf16 -c --no-local -bt no -m mha -det --bf16-fwd-backend ck
"
```

### 7.8 Dump Reference Kernel (CK) ISA

```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/aiter &&
  GPU_ARCH=\$(rocminfo | grep -oP 'gfx\w+' | head -1) &&
  AITER_ROOT=. &&
  CK_DIR=\${AITER_ROOT}/3rdparty/composable_kernel &&
  MODULE=mha_fwd_bf16_nbias_mask_lse_ndropout_nqscale &&
  BLOB_DIR=\${AITER_ROOT}/aiter/jit/build/\${MODULE}/blob &&
  SRC=fmha_fwd_d128_bf16_batch_b128x128x32x128x32x128_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psddv_nlogits_nbias_mask_lse_ndropout_nskip_nqscale_ntrload_nsink_gfx9 &&
  /opt/rocm/bin/hipcc \
    -DWITH_HIP -D_GLIBCXX_USE_CXX11_ABI=1 -DFAV2_ON=1 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DCUDA_HAS_FP16=1 \
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
    -I\${AITER_ROOT}/3rdparty/ck_helper \
    -I\${CK_DIR}/include -I\${CK_DIR}/library/include \
    -I\${AITER_ROOT}/csrc/include -I\${BLOB_DIR} \
    -I\${CK_DIR}/example/ck_tile/01_fmha \
    \$(python3 -c \"import torch; print(f'-isystem {torch.__path__[0]}/include')\") \
    \$(python3 -c \"import torch; print(f'-isystem {torch.__path__[0]}/include/torch/csrc/api/include')\") \
    -isystem /opt/rocm/include \
    -fPIC -std=c++20 -O3 -mcmodel=large -fno-unique-section-names \
    -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__ \
    -Wno-macro-redefined -Wno-missing-template-arg-list-after-template-kw \
    -Wno-switch-bool -Wno-undefined-func-template -Wno-unused-result \
    -Wno-vla-cxx-extension -fgpu-flush-denormals-to-zero \
    -fno-offload-uniform-block \
    -mllvm --amdgpu-kernarg-preload-count=16 \
    -mllvm --lsr-drop-solution=1 \
    -mllvm -amdgpu-coerce-illegal-types=1 \
    -mllvm -amdgpu-early-inline-all=true \
    -mllvm -amdgpu-function-calls=false \
    -mllvm -enable-post-misched=0 \
    -S --cuda-device-only --offload-arch=\${GPU_ARCH} \
    \${BLOB_DIR}/\${SRC}.cpp \
    -o /home/yanguahe/code/my_asm_code/aiter/ck_fmha_fwd_d128_bf16_causal_psddv_\${GPU_ARCH}.s
"
```

### 7.9 rocprofv3 Profiling

**input.yaml** (save as `FlyDSL/input.yaml`):
```yaml
jobs:
  -
    kernel_include_regex: (flash_attn_func_kernel)
    kernel_exclude_regex:
    kernel_iteration_range: "[1]"
    output_file: out
    output_directory: thread_trace/rpf_v3
    output_format: [csv]
    truncate_kernels: false
    sys_trace: false
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0xf"
    att_simd_select: "0xf"
    att_buffer_size: "0x6000000"
```

**Execution command** (use `waves_per_eu=1` for profiling):
```bash
docker exec hyg_trn_rocm7.1 bash -c "
  cd /home/yanguahe/code/my_asm_code/FlyDSL &&
  rm -rf ~/.flydsl/cache/* &&
  HIP_VISIBLE_DEVICES=0 FLYDSL_WAVES_PER_EU=1 \
  rocprofv3 -i ./input.yaml -- \
    python3 tests/kernels/test_flash_attn_func.py \
    --dtype bf16 --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --iters 5
"
```

**Output location**: `FlyDSL/thread_trace/rpf_v3/`

### 7.10 Key Environment Variables

| Variable | Purpose | Values |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | Select GPU | `0`-`7` (check availability first) |
| `FLYDSL_WAVES_PER_EU` | Override waves_per_eu | `1` (profiling), `2` (benchmark, default) |
| `FLIR_DUMP_ISA` | Dump final ISA | `1` to enable |
| `FLIR_DUMP_IR` | Dump MLIR IR | `1` to enable |
| `FLIR_NO_CACHE` | Bypass kernel cache | `1` to force recompile |
| `FLIR_REBUILD` | Force rebuild | `1` to force |
| `FLYDSL_LLVM_OPTS` | Override LLVM backend flags | Space-separated flags (default: `-enable-post-misched=0 --lsr-drop-solution=1`) |

### 7.11 Output Format

```
FlyDSL flash_attn_func (causal, bf16)
                                             Config/Path | Status |   MaxErr   MinCos |   Time(us)   TFLOPS
           B=1 S=8192 H=64 D=128 bf16 / ck_n128_fastpath |   PASS | 3.91e-03  0.99999 |     1591.1   691.019
```

| Field | Meaning |
|---|---|
| `Config/Path` | B=batch, S=seq_len, H=num_heads, D=head_dim, dtype, code path |
| `Status` | PASS (MaxErr < 8e-03) or FAIL |
| `MaxErr` | Maximum absolute error vs PyTorch reference |
| `MinCos` | Minimum cosine similarity vs reference |
| `Time(us)` | Average kernel time in microseconds |
| `TFLOPS` | Measured throughput in TFLOPS |
