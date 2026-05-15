# Flash Attention OPUS Kernel Detailed Analysis
# BF16 Flash Multi-Head Attention on AMD MI355X / GFX950 (CDNA4)

**Kernel file:** `FlyDSL/kernels/flash_attn_opus.py`
**Reference:** `FlyDSL/opus_attn/gqa_d128_kernel_template.hpp` (faithfully ported after P1–P7)
**Config:** BF16; FMHA; FWD; D=128; 1 thread-group; 8 waves; BLOCK_M=256 / BLOCK_N=64; MFMA 32×32×16 bf16; causal & non-causal; gfx950+
**Entry point:** `build_flash_attn_opus_module(...)` → `_launch(Q, K, V, O, batch_size, seq_len)`
**Target GPU:** AMD Instinct MI355X-class GFX950 / CDNA4 ISA — required for `ds_read_tr16_b64`
**Activation:** opt-in via `FLYDSL_ENABLE_OPUS_PATH=1`; runtime dispatch when `seq_len % 256 == 0 && seq_len >= 384`
**Correctness:** MaxErr 3.91 × 10⁻³ causal (numerical parity with OPUS C++ at B=16 S=8192 H=64 D=128 bf16)
**Companion doc:** [`FLASH_ATTN_OPUS_vs_CPP_Differences.md`](FLASH_ATTN_OPUS_vs_CPP_Differences.md) — side-by-side comparison with `gqa_d128_kernel_template.hpp`
**Latest change:** commit `2d07de2` — atomic port of OPUS `u_gk/u_sk/u_rk/u_gv/u_sv/u_rv` LDS layouts (P7). All eight OPUS layouts (`u_q`, `u_o`, `u_gk`, `u_sk`, `u_rk`, `u_gv`, `u_sv`, `u_rv`) are now identical to `gqa_d128_kernel_template.hpp` §5.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Algorithm](#11-algorithm)
   - [1.2 Key Parameters](#12-key-parameters)
   - [1.3 MFMA Instruction Details](#13-mfma-instruction-details)
   - [1.4 Relationship to OPUS C++ Reference](#14-relationship-to-opus-c-reference)
2. [Workgroup and Thread Organization](#2-workgroup-and-thread-organization)
   - [2.1 3D Grid Launch](#21-3d-grid-launch)
   - [2.2 Block Structure (8 Waves)](#22-block-structure-8-waves)
   - [2.3 Wave Division of Labor](#23-wave-division-of-labor)
   - [2.4 GQA Head Mapping](#24-gqa-head-mapping)
3. [Kernel Signature](#3-kernel-signature)
4. [LDS Layout (OPUS interleaved K0/V0/K1/V1, 68 096 B)](#4-lds-layout-opus-interleaved-k0v0k1v1-68-096-b)
   - [4.1 Overview](#41-overview)
   - [4.2 K Tile (OPUS `u_sk`, line-padded)](#42-k-tile-opus-u_sk-line-padded)
   - [4.3 V Tile (OPUS `u_sv`, wider line padding)](#43-v-tile-opus-u_sv-wider-line-padding)
   - [4.4 DMA Cooperative Loading (OPUS `u_gk` / `u_gv`)](#44-dma-cooperative-loading-opus-u_gk--u_gv)
5. [Q Preload (Register-Resident, With In-Flight Scaling)](#5-q-preload-register-resident-with-in-flight-scaling)
6. [GEMM0: Q × K^T Score Computation](#6-gemm0-q--kt-score-computation)
   - [6.1 Tile Decomposition](#61-tile-decomposition)
   - [6.2 K LDS Read (OPUS `u_rk`, N-permuted)](#62-k-lds-read-opus-u_rk-n-permuted)
   - [6.3 MFMA Chain with 2-Step Prefetch](#63-mfma-chain-with-2-step-prefetch)
7. [Causal Mask](#7-causal-mask)
   - [7.1 Masking Condition (OPUS-permuted N)](#71-masking-condition-opus-permuted-n)
   - [7.2 P4: Inline-asm `v_cmp_lt_i32 + v_cndmask_b32` With Immediate Thresholds](#72-p4-inline-asm-v_cmp_lt_i32--v_cndmask_b32-with-immediate-thresholds)
8. [Online Softmax with Lazy Rescaling](#8-online-softmax-with-lazy-rescaling)
   - [8.1 Algorithm](#81-algorithm)
   - [8.2 Row-Max Reduction](#82-row-max-reduction)
   - [8.3 Lazy Rescale Check (ballot)](#83-lazy-rescale-check-ballot)
   - [8.4 Fused Scale + Subtract + Base Conversion](#84-fused-scale--subtract--base-conversion)
   - [8.5 P-Value Packing to BF16](#85-p-value-packing-to-bf16)
9. [GEMM2: P × V Output Accumulation](#9-gemm2-p--v-output-accumulation)
   - [9.1 Tile Decomposition](#91-tile-decomposition)
   - [9.2 V LDS Read (OPUS `u_rv` via `ds_read_tr16_b64`)](#92-v-lds-read-opus-u_rv-via-ds_read_tr16_b64)
   - [9.3 V Hoist Across Cluster Boundary (P5/P6 correctness)](#93-v-hoist-across-cluster-boundary-p5p6-correctness)
10. [Output Finalization](#10-output-finalization)
    - [10.1 Normalize O](#101-normalize-o)
    - [10.2 Convert FP32 to BF16](#102-convert-fp32-to-bf16)
    - [10.3 Store O to Global Memory](#103-store-o-to-global-memory)
11. [Main Loop Pipeline (Post P1–P7)](#11-main-loop-pipeline-post-p1p7)
    - [11.1 Pipeline Structure](#111-pipeline-structure)
    - [11.2 Cluster-by-Cluster Breakdown](#112-cluster-by-cluster-breakdown)
    - [11.3 Synchronization Points](#113-synchronization-points)
    - [11.4 OPUS Optimizations Layered In](#114-opus-optimizations-layered-in)
12. [Performance Status and Known Gaps](#12-performance-status-and-known-gaps)
13. [P1–P7 Evolution History](#13-p1p7-evolution-history)
14. [Q/K/V/S/O Layout Atlas](#14-qkvso-layout-atlas)

---

## 1. Overview

### 1.1 Algorithm

This kernel implements the **Flash Attention Forward Pass** with online softmax for BF16 inputs on GFX950 / CDNA4. It is a **complete structural port** of `opus_attn/gqa_d128_kernel_template.hpp` (D=128 only) into FlyDSL Python, achieving numerical parity with the C++ reference (MaxErr 3.91 × 10⁻³ causal, 2.44 × 10⁻⁴ nocausal).

Post-P1–P7 (commits `ad1fcaf` … `2d07de2`), the kernel runs an OPUS-style 2-tile-per-iter **`j += 2` ping-pong pipeline** with 8 clusters per main-loop iteration and a 14-cluster (0..13) epilogue. Q is held register-resident and pre-multiplied by `(1/√D) · log2(e)` in the prologue (in-flight scaling, P2) so all softmax math runs in log2 space. K and V live in shared memory under the OPUS-defined `u_sk`/`u_sv` layouts (interleaved K0/V0/K1/V1 double-buffer, line-strided with 16 B / 64 B padding for bank-conflict-free `ds_read` and `ds_read_tr16_b64`), and are pulled into MFMA operand registers via OPUS `u_rk` / `u_rv` (P7).

```
For each (batch, head, q_block) in (H, ceil(S/256), B):           # 3D grid (OPUS-style)
  PROLOGUE (P1–P13 in source comments, ~10 functional stages, C++ lines 397-436)
    async_load K[0] → s_k[0]; s_waitcnt(0); s_barrier
    Q = load Q tile (256 × 128); Q *= (1/√D) · log2e            # P2 in-flight scale
    async_load K[1] → s_k[1]; async_load V[0] → s_v[0]
    v_k = ds_read s_k[0]; lgkm(0); vmcnt(2)
    if (stagger) s_barrier                                       # P5 dual-group open
    v_s[0] = mma0(v_q, v_k)
    causal_mask v_s[0] (inline-asm v_cmp + v_cndmask)            # P4
    m_row = row_max(v_s[0]); v_s[0] -= m_row
    exp2(v_s[0][0..15])  (first half only)
    s_barrier; async_load K[2] → s_k[0]

  MAIN LOOP (for j = 3; j < max-1; j += 2)
    # Cluster 0: async_load V[j-2] → s_v[1]; ds_read s_k[1]; waits + s_barrier
    # Cluster 1: v_s[1] = mma0(Q, K[j-2]);  exp2(v_s[0][16..31]); l += sum; P = cast<bf16>(v_s[0])
    # Cluster 2: async_load K[j] → s_k[1]; tr_load V[j-3] from s_v[0] (HOISTED, P5/P6)
    # Cluster 3: setprio(1); step_k(0..3) [16 MFMAs]; lazy rescale on v_s[1];
    #            sub_row + exp2(v_s[1][0..15]); setprio(0); s_barrier
    # Cluster 4: async_load V[j-1] → s_v[0]; ds_read s_k[0]; waits + s_barrier
    # Cluster 5: v_s[0] = mma0(Q, K[j-1]); finish softmax(v_s[1]); P = cast<bf16>(v_s[1])
    # Cluster 6: async_load K[j+1] → s_k[0]; tr_load V[j-2] from s_v[1] (HOISTED);
    #            causal_mask v_s[0]; waits + s_barrier
    # Cluster 7: setprio(1); step_k(0..3) on v_s[0]; lazy rescale; sub_row + exp2;
    #            setprio(0); s_barrier

  EPILOGUE (14 stages, drains 4 carried tiles)
    # Cluster 0–13: mirrors the same ping-pong but with explicit per-iteration
    #               rescale (no lazy gate) and unrolled across all four trailing tiles

  CLOSE-OUT
    v_o /= l_row
    if (!stagger) s_barrier                                      # P5 dual-group close
    store v_o to gmem
```

The lazy-rescale gate skips both `m_row` and `v_o` rescale when every lane satisfies `tile_row_max - m_row ≤ 8.0` in the already-scaled log2 domain. In that case the kernel clamps `m_new = m_row`, so `corr = exp2(0) = 1` and the running output remains in the same normalization frame. The dual-group stagger lets warps 0–3 and warps 4–7 occupy opposite cluster phases for resource time-sharing.

### 1.2 Key Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Data type | `D_ATTN` | bf16 | 2 bytes per element |
| Accumulator | `D_ACC` | f32 | softmax + GEMM accum |
| Head dimension | `HEAD_DIM` | 128 | fixed, both QK and V |
| Q tile (rows/workgroup) | `BLOCK_M` | 256 | 8 waves × 32 rows |
| KV tile (cols/iter) | `BLOCK_N` | 64 | inner loop step |
| Outer step | `BLOCK_N_OUT` | 64 | one sub-tile per outer iter |
| MFMA tile | `W_M×W_N×W_K` | 32×32×16 | gfx950 bf16 |
| GEMM0 K-steps | `K_STEPS_QK` | 8 | `HEAD_DIM / W_K` |
| GEMM0 N-strips | (implicit) | 2 | `BLOCK_N / W_N` |
| GEMM2 D-chunks | `D_CHUNKS` | 4 | `HEAD_DIM / D_CHUNK` |
| GEMM2 PV-steps | `PV_K_STEPS` | 2 | `K_SUB_N / PV_K_STEP` |
| BF16 per MFMA pack | `MFMA_LANE_K` | 8 | 8 bf16 per lane per pack |
| Waves per workgroup | `NUM_WAVES` | 8 | 512 threads / 64 |
| Threads per workgroup | `BLOCK_SIZE` | 512 | `NUM_WAVES × WARP_SIZE` |
| LDS size (OPUS layout, P7) | — | **68 096 B** | interleaved K0 / V0 / K1 / V1 with line padding |
| K LDS line stride | `SMEM_K_LINE_STRIDE` | 520 bf16 | `smem_linear_wave + smem_padding_16B` |
| V LDS line stride | `SMEM_V_LINE_STRIDE` | 544 bf16 | `smem_linear_wave + smem_padding_64B` |
| K LDS prefetch buffers | `NUM_PREFETCH_K` | 2 | double-buffered |
| V LDS prefetch buffers | `NUM_PREFETCH_V` | 2 | double-buffered (fully used since P1) |
| DMA per-lane vector | `DMA_BYTES` | 16 bytes (= 8 bf16) | `buffer_load_dwordx4_lds` |
| DMA calls per K tile | `NUM_DMA_K` | 2 = `SMEM_D_RPT` | one per d_rpt array |
| DMA calls per V tile | `NUM_DMA_V` | 2 = `SMEM_D_RPT` | one per d_rpt array |
| Softmax scale | `SM_SCALE` | `1/√128` | fused with `log2(e)` in `c_sm_scale_log2e` |
| Lazy-rescale threshold | `OPUS_RESCALE_THRESHOLD` | 8.0 | direct threshold in the same log2-domain units as `m_row` / tile row-max |

### 1.3 MFMA Instruction Details

The kernel uses **`v_mfma_f32_32x32x16_bf16`** (gfx950 / CDNA4 only):

- **Computation:** `D[32×32, FP32] = A[32×16, BF16] × B[16×32, BF16] + C[32×32, FP32]`
- **Pack width:** 8 BF16 per lane (`MFMA_LANE_K=8`) → 4 VGPRs per operand per lane
- **Accumulator:** 16 FP32 per lane → 16 VGPRs (held in an `llvm.vector<16 × f32>`)
- **K extent per MFMA:** 16 BF16 (twice the K of CDNA3's `v_mfma_f32_32x32x8_bf16`)
- **Reduction across lanes:** per-lane `shuffle_xor` by 32 merges the two halves of a 64-lane warp

A single `mfma_acc(a, b, c)` call in the Python source emits one such instruction.

### 1.4 Relationship to OPUS C++ Reference

This Python kernel is a **complete structural port** of `opus_attn/gqa_d128_kernel_template.hpp` after the P1–P7 evolution. Every cluster, scheduling hint, `s_barrier`, `s_setprio` call, register anchor, inline-asm causal-mask block, and LDS layout in the C++ template has a corresponding line in the Python source, and source-line cross-references appear as comments throughout the kernel (e.g. "C++ lines 461-468").

What is aligned:

- 8-cluster `j += 2` main loop + 14-cluster (0..13) epilogue (P1).
- In-flight Q scaling so softmax operates directly in log2 space (P2).
- `sched_group_barrier` MFMA/VALU/EXP pair patterns at every C++ call site (P3).
- `v_cmp_lt_i32_e64 + v_cndmask_b32_e64` immediate-threshold inline asm for the causal mask + 8 `asm volatile("" : "+v"(v))` register anchors (P4).
- Dual-group `if (warp_id/4) s_barrier` stagger in the prologue + `if (!stagger) s_barrier` close-out (P5).
- V LDS reads hoisted into the cluster **preceding** every GEMM2 consumer cluster so the asymmetric stagger barrier is race-free (P5/P6).
- **OPUS K/V LDS layout (P7)**: interleaved K0/V0/K1/V1 (68 096 B total) with `smem_linear_wave + smem_padding_{16B,64B}` line stride; OPUS `u_gk`/`u_sk` DMA writers, OPUS `u_rk` K-read with N-permutation π(m)=(m%8)·8+m/8, OPUS `u_gv`/`u_sv` DMA writers, OPUS `u_rv` V-read via `ds_read_tr16_b64` with the same lane → (grp_k, lane_hi, grp_n, lane_lo) mapping as the C++ template. The causal-mask thresholds are reordered to match the π-permuted N axis.

A detailed side-by-side analysis of the remaining (mostly mechanical) differences between this port and the C++ template is in [`FLASH_ATTN_OPUS_vs_CPP_Differences.md`](FLASH_ATTN_OPUS_vs_CPP_Differences.md).

---

## 2. Workgroup and Thread Organization

### 2.1 3D Grid Launch

```
Grid:  (gdx, gdy, gdz)
  gdx = NUM_HEADS_Q                       (Q-head index)
  gdy = ceil(seq_len / BLOCK_M)           (Q-block index along sequence)
  gdz = batch_size                        (batch index)

Block: (512, 1, 1)
  512 threads = 8 waves × 64 threads/wave
```

Mapping (from the kernel body, `gpu.block_idx.{x,y,z}`):

```python
h_idx       = block_idx.x   # Q-head id (after GQA un-grouping)
q_block_idx = block_idx.y   # Q-tile id along sequence (each tile = 256 rows)
batch_idx   = block_idx.z   # batch id
tid         = thread_idx.x  # 0..511
```

The 3D grid (vs. a 1D `H × num_q_blocks × B` flattened grid) is the first OPUS-style optimization: it lets the GPU's scheduler distribute the X dimension across CUs without per-thread integer division to recover the head index.

### 2.2 Block Structure (8 Waves)

Per-wave layout:

```
512 threads / 64 = 8 waves
  wave_id = tid // 64           (0..7)
  lane    = tid %  64           (0..63)
  lane_mod_32 = lane % 32       (0..31, row index within wave's score tile)
  lane_div_32 = lane // 32      (0..1, K-pack column index for MFMA B)

HW-transpose decomp (used by ds_read_tr16_b64 for V):
  tr_k_group  = (lane % 16) // 4    (0..3, 4-row groups within 16-row block)
  tr_col_sub  = lane % 4            (0..3, sub-column within a 4×4 transpose)
  tr_col_half = (lane % 32) // 16   (0..1, which half of the 32 rows)
```

### 2.3 Wave Division of Labor

The 256-row Q tile is split row-wise among the 8 waves; each wave owns 32 contiguous rows and processes them independently from softmax onward:

```
256 Q rows (BLOCK_M = 256):
+---------------------------------+
| Wave 0: rows   0- 31 (32 rows) |
| Wave 1: rows  32- 63 (32 rows) |
| Wave 2: rows  64- 95 (32 rows) |
| Wave 3: rows  96-127 (32 rows) |
| Wave 4: rows 128-159 (32 rows) |
| Wave 5: rows 160-191 (32 rows) |
| Wave 6: rows 192-223 (32 rows) |
| Wave 7: rows 224-255 (32 rows) |
+---------------------------------+
            ×                        ×
   K tile (shared LDS)       V tile (shared LDS)
   [64 KV × 128 bf16]        [64 KV × 128 bf16]
```

**Cooperative phases (all 8 waves participate):**
- K DMA into LDS: `coop_dma_k(...)` — each wave issues `WARP_SIZE × NUM_DMA_K` `buffer_load_dwordx4_lds`
- V DMA into LDS: `coop_dma_v(...)` — same pattern
- `gpu.barrier()` between DMA issue and LDS read

**Independent phases (each wave works alone):**
- Q preload from global to register (each wave loads its 32 rows × 128 cols)
- GEMM0 (Q × K^T): each wave computes its own `32 × 64` score tile
- Causal mask & softmax row-max/sum: each wave operates on its own 32 rows
- GEMM2 (P × V): each wave computes its own `32 × 128` O slice
- Output store: each wave stores its 32 rows to global

Softmax requires **no cross-wave communication** because different Q rows have independent statistics. The only intra-wave reduction is `shuffle_xor` by 32 (merges lane ↔ lane+32).

### 2.4 GQA Head Mapping

The kernel supports Grouped-Query Attention. The mapping replicates OPUS lines 310-312:

```python
h_kv_idx    = h_idx %  NUM_HEADS_KV
group_id    = h_idx // NUM_HEADS_KV
q_head_idx  = h_kv_idx * GQA_GROUP_SIZE + group_id   # un-interleaved Q head
kv_head_idx = h_kv_idx                                # shared K/V head
```

When `NUM_HEADS_KV == NUM_HEADS_Q` (vanilla MHA), `GQA_GROUP_SIZE == 1` and `q_head_idx == h_idx`.

---

## 3. Kernel Signature

```python
@flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
def flash_attn_opus_kernel(
    Q: fx.Tensor,    # [B, S, H,    D]  contiguous (BSHD layout)
    K: fx.Tensor,    # [B, S, H_KV, D]
    V: fx.Tensor,    # [B, S, H_KV, D]
    O: fx.Tensor,    # [B, S, H,    D]
    seq_len: fx.Int32,
):
    ...
```

**Strides (BSHD layout, computed in Python from `num_heads` / `num_kv_heads`):**

```
stride_token_q  = NUM_HEADS_Q  * HEAD_DIM    # bytes / 2  =  64 * 128 = 8192 bf16
stride_token_kv = NUM_HEADS_KV * HEAD_DIM    # bytes / 2  =  64 * 128 = 8192 bf16
```

Global-address helpers:

```python
def global_idx_q(token_idx, col):       # for Q and O
    token = batch_idx * seq_len + token_idx
    return token * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col
```

K/V global addresses are computed inline inside the DMA loaders using the same formula with `STRIDE_TOKEN_KV` and `kv_head_idx`.

---

## 4. LDS Layout (OPUS interleaved K0/V0/K1/V1, 68 096 B)

### 4.1 Overview

After P7 (commit `2d07de2`), the K/V LDS layout is **bit-for-bit identical** to the OPUS C++ template's `smem<D_ATTN> s_k[2]` and `smem<D_ATTN> s_v[2]` (`gqa_d128_kernel_template.hpp` lines 326-333). The four buffers are **interleaved** so each iteration's K and V live in adjacent address space, and every "row" within a wave's slab is padded so the `ds_read` / `ds_read_tr16_b64` patterns are bank-conflict-free.

```
LDS layout (bf16 elements; multiply by 2 for bytes):

  Offset (bf16)        Region                                Size (bf16) Size (B)
  ───────────────────  ────────────────────────────────────  ──────────  ────────
  lds_kv_offset+ 0     K buffer 0  (s_k[0], OPUS u_sk pad16)     8 320    16 640
  lds_kv_offset+ 8 320 V buffer 0  (s_v[0], OPUS u_sv pad64)     8 704    17 408
  lds_kv_offset+17 024 K buffer 1  (s_k[1], OPUS u_sk pad16)     8 320    16 640
  lds_kv_offset+25 344 V buffer 1  (s_v[1], OPUS u_sv pad64)     8 704    17 408
  ──────────────────────────────────────────────────────────────────────  ────────
                                                  Total:        34 048    68 096
```

This is the same `[K0][V0][K1][V1]` interleaving as the C++ kernel: each "pair" of `(K[i], V[i])` is `OPUS_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS = 8320 + 8704 = 17024 bf16` apart. The interleaving matters because the DMA loaders and `ds_read_*` consumers compute their base offsets from a single `smem_buf` pointer + per-buffer offset, mirroring how the C++ kernel constructs `s_k[i]` and `s_v[i]` from a single `smem_buf`.

Trait constants (all derived from the OPUS C++ template, see kernel source lines 149-196):

```
BF16_BYTES         = 2
D_128B_SIZE        = 64                # = 128 B / sizeof(bf16)
VEC_KV             = 8                 # bf16 per ds_read pack (also MFMA pack)
VEC_TR_V           = 4                 # bf16 per ds_read_tr16_b64

SMEM_LINEAR_WAVE   = WARP_SIZE * 16 // BF16_BYTES = 512   # bf16 per wave per "line"
SMEM_N_PER_WAVE    = SMEM_LINEAR_WAVE // D_128B_SIZE = 8  # KV rows per wave per line
SMEM_N_RPT         = BLOCK_N // SMEM_N_PER_WAVE = 8       # lines along N
SMEM_D_RPT         = HEAD_DIM // D_128B_SIZE = 2          # lines along D
SMEM_K_PAD         = 16 // BF16_BYTES = 8                 # 16 B padding = 8 bf16
SMEM_V_PAD         = 64 // BF16_BYTES = 32                # 64 B padding = 32 bf16

SMEM_K_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_K_PAD = 520  # bf16 per K line
SMEM_V_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_V_PAD = 544  # bf16 per V line

SMEM_K_TILE_ELEMS  = SMEM_N_RPT * SMEM_D_RPT * SMEM_K_LINE_STRIDE  # 8 * 2 * 520 = 8 320
SMEM_V_TILE_ELEMS  = SMEM_N_RPT * SMEM_D_RPT * SMEM_V_LINE_STRIDE  # 8 * 2 * 544 = 8 704
OPUS_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS = 17 024
LDS_KV_TOTAL_SIZE  = 2 * OPUS_KV_PER_BUFFER = 34 048 bf16 = 68 096 B
```

`SmemAllocator` aligns the start to 16 bytes (`lds_kv_offset = align(allocator.ptr, 16)`).

### 4.2 K Tile (OPUS `u_sk`, line-padded)

Each K buffer holds an `SMEM_N_RPT × SMEM_D_RPT` array of "lines"; each line is owned by one wave and stretches `SMEM_LINEAR_WAVE = 512` bf16 of K data followed by `SMEM_K_PAD = 8 bf16 = 16 B` of pad. The 16 B padding shifts each wave's slab into a different bank set, so the eight lanes of a wave that perform `ds_read_b128` see eight distinct banks per access — without needing an XOR swizzle.

Geometry:

```
Per wave w  ∈ {0..7}:
  line stride = SMEM_K_LINE_STRIDE = 520 bf16
  Each line:  [ 8 N-rows × 64 D-bf16  ][ 8 bf16 pad ]
                                       ^—— 16 B padding (= one extra bank)
  Per-buffer per-wave slab: SMEM_D_RPT × line_stride = 2 × 520 = 1040 bf16
  Total per buffer: NUM_WAVES × 1040 + (wave-distributed across SMEM_N_RPT) = 8320 bf16

DMA u_sk write (per lane l in warp w, for d_rpt = d):
  lds_addr (bf16) = k_buf_base
                  + w * SMEM_K_LINE_STRIDE
                  + d * SMEM_N_RPT * SMEM_K_LINE_STRIDE
                  + (l // VEC_KV) * SMEM_K_LINE_STRIDE          (wait, no — see u_sk explained below)
```

The exact OPUS `u_sk` formula folds the warp axis as a `p_dim` (per-lane) coordinate and writes contiguous `VEC_KV = 8 bf16` packs from each lane into one line. The simpler way to read it: **each wave gets one slab of SMEM_D_RPT lines, and within the slab the `SMEM_LINEAR_WAVE = WARP_SIZE × VEC_KV` bf16 are laid out lane-by-lane.**

The full read formula (OPUS `u_rk`) is in §6.2.

### 4.3 V Tile (OPUS `u_sv`, wider line padding)

V uses the **same structural layout** as K but with `SMEM_V_PAD = 32 bf16 = 64 B` of padding per line instead of 16 B. The wider padding is what lets the hardware-transposing `ds_read_tr16_b64` (consumed in §9.2) pull a 4×4 transposed bf16 block without LDS bank contention — the C++ template documents this in the `make_layout_sk_sv<T, smem_padding>` helper (lines 102-118), where `smem_padding_64B` is passed for V and `smem_padding_16B` for K.

Geometry:

```
Per wave w  ∈ {0..7}:
  line stride = SMEM_V_LINE_STRIDE = 544 bf16
  Each line:  [ 8 N-rows × 64 D-bf16  ][ 32 bf16 pad ]
                                       ^—— 64 B padding (covers a 4×4 transpose's bank distance)
  Per-buffer per-wave slab: SMEM_D_RPT × line_stride = 2 × 544 = 1088 bf16
  Total per buffer: 8 waves × 1088 = 8704 bf16
```

The full read formula (OPUS `u_rv` via `ds_read_tr16_b64`) is in §9.2.

After P1 (cluster restructure) and P7 (atomic layout port), **V is fully double-buffered with the OPUS-defined line padding**: each main-loop iteration writes V[j-2] into `s_v[1]` and V[j-1] into `s_v[0]` while consuming V[j-3]/V[j-2] from the opposite buffer. The 2-buffer ping-pong matches the C++ template `s_v[2]` exactly, and the padding makes the `ds_read_tr16_b64` reads conflict-free.

### 4.4 DMA Cooperative Loading (OPUS `u_gk` / `u_gv`)

K and V are loaded directly from global to LDS via `buffer_load_dwordx4_lds` (`DMA_BYTES = 16` = 8 bf16 per lane per instruction). This bypasses VGPRs and is one of the major perf advantages of gfx950.

After P7 the loaders use the OPUS `u_gk` / `u_gv` (global) + `u_sk` / `u_sv` (LDS) layouts (kernel source `coop_dma_k` / `coop_dma_v`, lines 564-616). Each wave covers `SMEM_N_PER_WAVE = 8` rows of N and the full HEAD_DIM (128) bf16 of D per buffer, split across `NUM_DMA_K = NUM_DMA_V = SMEM_D_RPT = 2` DMA instructions (one per d_rpt array of 64 bf16 of D):

```python
# coop_dma_k(tile_start, buf_id):
n_in_warp = lane_in_warp // VEC_KV          # 0..7  → which of 8 N-rows in this wave
d_bucket  = lane_in_warp %  VEC_KV          # 0..7  → which 8-bf16 slice within a 64-bf16 array

for d in 0..SMEM_D_RPT-1:                   # 2 iterations: d=0 (D bytes 0..63), d=1 (D 64..127)
    # LDS destination
    lds_addr (bytes) = lds_kv_byte_base
                     + k_buf_base(buf_id) * BF16_BYTES
                     + wave_id * SMEM_K_LINE_STRIDE * BF16_BYTES         # which wave's slab
                     + d * SMEM_N_RPT * SMEM_K_LINE_STRIDE * BF16_BYTES  # which d_rpt array

    # Global source
    global_n   = wave_id * SMEM_N_PER_WAVE + n_in_warp                   # 0..63
    global_d   = d_bucket * VEC_KV + d * D_128B_SIZE                     # 0..127
    global_row = batch * seq_len + tile_start + global_n
    global_byte = global_row * (STRIDE_TOKEN_KV * BF16_BYTES)
                + kv_head_idx * (HEAD_DIM * BF16_BYTES)
                + global_d * BF16_BYTES

    # buffer_load_dwordx4_lds: 16 B / lane, lane-wide gather
    rocdl.raw_ptr_buffer_load_lds(k_rsrc, lds_ptr, 16, global_byte_i32, ...)
```

`coop_dma_v` is identical to `coop_dma_k` except for the buffer base (`v_buf_base(buf_id)` instead of `k_buf_base(buf_id)`) and the line stride (`SMEM_V_LINE_STRIDE = 544` vs `SMEM_K_LINE_STRIDE = 520`).

Per workgroup per tile this fires **2 `buffer_load_dwordx4_lds` instructions × 8 waves = 16 DMA issues**, covering `BLOCK_N × HEAD_DIM × 2 B = 16 384 B` of K (or V) data into LDS in one issue burst.

---

## 5. Q Preload (Register-Resident, With In-Flight Scaling)

Q is loaded **once per Q-block** at the top of the kernel and kept in VGPRs throughout all KV iterations. There is no LDS Q region in this kernel (unlike the MI308X ASM kernel which time-shares LDS for Q/V).

**P2 in-flight scaling.** Each Q pack is pre-multiplied by the fused constant `temperature_scale = (1/√D) · log2(e)` during the prologue load (matches OPUS C++ lines 404–406). This collapses the per-FMA `S * sm_scale_log2e` chain inside the softmax body into a plain `S - m_row` subtract, because both operands are already in log2 space:

```
Original (P0/P1):    P = exp2((S * sm_scale - m_row) * log2e)         # 1 FMA + 1 exp2 per element
After P2:            P = exp2(S' - m_row')                            # 1 subf + 1 exp2 per element
                     where S' = Q' @ K = (Q * sm_scale * log2e) @ K
                     and   m_row' = max(S')
```

Per-lane Q address (each lane owns 8 bf16 per K-step, total 8 K-steps × 8 = 64 bf16 = 32 VGPRs):

```python
q_row = q_start + wave_id * 32 + lane_mod_32       # 0..S-1
q_col(ks) = ks * 16 + lane_div_32 * 8              # 0..127, step 8
g_idx     = (batch * S + q_row) * STRIDE_TOKEN_Q
          + q_head_idx * HEAD_DIM + q_col

c_sm_scale_log2e = (1/√D) * log2(e)                # f32 const

for ks in 0..7:
    raw_bf16x8 = load_global_mfma_pack(q_ptr, g_idx(ks))
    raw_safe   = q_in_bounds.select(raw_bf16x8, zeros)
    f32x8      = Vec(raw_safe).extf(v8f32_type)
    scaled_f32 = [Vec(f32x8)[k] * c_sm_scale_log2e for k in 0..7]
    q_b_packs[ks] = Vec.from_elements(scaled_f32).truncf(v8f16_type)
```

The bf16 → f32 → bf16 round-trip is cheap (8 × `v_pk_mul_f32` + 8 × `v_cvt_pkrtz_bf16_f32` per Q pack) and pays for itself many times over by eliminating the per-FMA multiply on every softmax element across all KV iterations.

Out-of-range Q rows (when `q_row >= seq_len`) are masked to zero via an `ArithValue.select(q_in_bounds, raw, zero)` — this lets the kernel safely operate when `seq_len < q_start + BLOCK_M`.

The 8 packs (`q_b_packs[0..7]`) feed all 8 GEMM0 MFMAs along the K dimension.

---

## 6. GEMM0: Q × K^T Score Computation

### 6.1 Tile Decomposition

```
S[32 × 64] = Q[32 × 128] @ K^T[128 × 64]    (per wave's slice)

decomposed into:
  K-steps along D (HEAD_DIM):   8   (K_STEPS_QK)
  N-strips along KV (BLOCK_N):  2   (BLOCK_N / W_N = 64 / 32)

Total MFMAs per wave per outer iter: 8 × 2 = 16

S accumulator (per wave):
  s_acc_lo: <16 × f32>   covers KV rows  0-31  (N-strip 0)
  s_acc_hi: <16 × f32>   covers KV rows 32-63  (N-strip 1)
```

### 6.2 K LDS Read (OPUS `u_rk`, N-permuted)

After P7 the K read uses OPUS `u_rk` (`make_layout_rk`, `gqa_d128_kernel_template.hpp` lines 122-148). Each MFMA-A pack is `8 bf16` per lane, and the lane → element mapping applies a **4×8 transpose permutation** on the N axis:

```
π(m) = (m % 8) * 8 + (m / 8)        # where m = lane%32 is the MFMA-A "M-axis" lane
```

Concretely, lane `m` of an MFMA-A operand reads `K[N = π(m), D = (m/8)*8 + 0..7]` (for the lower 32 N-strip; the upper strip adds 32 to N via a `+256 bf16` LDS offset). The MFMA result therefore has its **N axis permuted**: `S[N = π(m), M = q_lane]` rather than `S[N = m, M = q_lane]`. Downstream code (causal mask in §7, softmax in §8) accounts for this permutation by using OPUS-permuted thresholds.

```python
# Per-lane base offset (bf16), kernel source lines 692-696:
urk_base_per_lane = (
      (lane_mod_32 %  8) * SMEM_K_LINE_STRIDE       # axis "m % 8" → which line in the wave's slab
    + (lane_mod_32 // 8) * D_128B_SIZE              # axis "m / 8" → which 64-bf16 array within the line
    +  lane_div_32       * VEC_KV                   # axis "lane / 32" → which 8-bf16 pack within the array
)

# K-step offsets along D (8 packs cover HEAD_DIM = 128):
#   ks // 4 selects d_rpt = 0 or 1   →   outer stride OPUS_URK_KSTEP_OUTER = 4 160 bf16
#   ks %  4 selects 16-bf16 step within a d_rpt   →   inner stride OPUS_URK_KSTEP_INNER = 16 bf16

# Lower N-strip (n=0..31) and upper N-strip (n=32..63):
idx_lo(ks) = k_buf_base(buf) + urk_base_per_lane + (ks // 4) * 4160 + (ks % 4) * 16
idx_hi(ks) = idx_lo(ks) + OPUS_URK_N_STRIP_STRIDE                          # = +256 bf16
```

The constants `OPUS_URK_N_STRIP_STRIDE = 256`, `OPUS_URK_KSTEP_OUTER = SMEM_N_RPT * SMEM_K_LINE_STRIDE = 4160`, `OPUS_URK_KSTEP_INNER = 16` are derived directly from the C++ template's `make_layout_rk` shape and stride tuples (kernel source lines 180-185 cross-reference these).

Eight `ds_read_b128` per lane per N-strip × 2 N-strips = 16 K packs are read per consumer cluster. All addresses are bank-conflict-free because of the 16 B padding (§4.2).

### 6.3 MFMA Chain and K Pack Lifetime

The current Python source does **not** contain a separate `_QK_PREFETCH_DEPTH` loop. Instead, `_read_k_packs_for_buf(buf)` first materializes all 16 K operand packs from LDS into VGPR SSA values, then `_gemm0(k_lo, k_hi)` consumes those packs in a compact MFMA loop:

```python
k_lo, k_hi = _read_k_packs_for_buf(buf)       # 8 lo + 8 hi ds_read_b128 packs

v_s_lo = zero
v_s_hi = zero
for ks in 0..7:
    v_s_lo = mfma_acc(k_lo[ks], q_b_packs[ks], v_s_lo)
    v_s_hi = mfma_acc(k_hi[ks], q_b_packs[ks], v_s_hi)
```

The LLVM scheduler may still interleave the emitted `ds_read_b128` instructions with nearby independent work inside each sched-barrier-bounded cluster, but that is a backend scheduling result rather than an explicit Python-level 2-step prefetch loop.

---

## 7. Causal Mask

### 7.1 Masking Condition (OPUS-permuted N)

For causal attention each Q row `q` may attend only to KV columns `k ≤ q`. After P7 the K LDS read uses OPUS `u_rk` (§6.2), which applies the N-axis permutation π(m) = (m%8)·8 + m/8 on the MFMA-A lane index. The MFMA output therefore has its N axis permuted, and the per-element K column derivation becomes:

```
For S_lo (lower N-strip, n_strip = 0):
  N_attention[r] = kv_block_start + lane_div_32 * 32 + π( (r//4) * 8 + (r%4) )
                 = kv_block_start + lane_div_32 * 32 + ((r%4)*8 + (r//4))

For S_hi (upper N-strip, n_strip = 1):
  N_attention[r] = N_attention_lo[r] + 4                   # u_rk adds +4 to N (smem_d_n_split)
```

Notable consequences of the OPUS layout:

- **`lane_div_32` (= 0 or 1) adds +32 to N** (not +4 as in the pre-P7 layout). This is because π(m_a + 4) = π(m_a) + 32 when m_a%8 < 4.
- **The v_s_hi strip adds +4 to N** (not +32). This matches `smem_d_n_split = 4` in the OPUS layout (`make_layout_rk` axis 6).
- **The per-r threshold pattern is reordered** to follow π:

```
r=0,1,2,3       → thresholds 0,8,16,24
r=4,5,6,7       → thresholds 1,9,17,25
r=8,9,10,11     → thresholds 2,10,18,26
r=12,13,14,15   → thresholds 3,11,19,27
```

Compared to the pre-P7 layout's `{0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27}` ordering, the new pattern is the OPUS π-image of the same 16 thresholds.

### 7.2 P4: Inline-asm `v_cmp_lt_i32 + v_cndmask_b32` With Immediate Thresholds

The current kernel uses the **same inline-asm sequence as OPUS C++** (`attn_mask_vec2_imm`, gqa_d128_kernel_template.hpp lines 233-249) — `v_cmp_lt_i32_e64 + v_cndmask_b32_e64` pairs with immediate-literal thresholds:

```python
def _attn_mask_imm_single(rel_i32, neg_inf_i32, thr, x_ref_i32):
    asm_str = (
        f"v_cmp_lt_i32_e64 $1, $2, {int(thr)}\n\t"
        "v_cndmask_b32_e64 $0, $3, $4, $1"
    )
    # $0 = new_x (=v), $1 = mask (=&s early-clobber), $2 = rel, $3 = x_ref, $4 = neg_inf
    return llvm.inline_asm(struct<(i32,i64)>, [rel, x_ref, neg_inf],
                            asm_str, "=v,=&s,v,v,v", has_side_effects=False)
```

The C++ template encodes both `x` and `y` outputs in a single 4-output `asm volatile` block; the FlyDSL port splits this into two consecutive single-output calls because MLIR's `llvm.inline_asm` with multiple `"=s"` SGPR-pair outputs proved brittle (silent corruption during register allocation). The emitted ISA is the same `v_cmp_lt + v_cndmask` pattern, with two more SGPR allocations.

`_causal_mask_inplace` iterates 8 `(thr_x, thr_y)` threshold pairs (in OPUS-permuted order, matching the C++ `static_for<c_rept>` × `static_for<c_pack/2>` nest after π is applied) and applies the mask in-place on the 16-element `s_lo` and `s_hi` Python lists:

```python
# Permuted relations (matches OPUS u_rk):
#   rel_lo = q_pos - kv_start - lane_div_32 * 32
#   rel_hi = rel_lo - 4
kv_start_i32 = fx.Int32(tile_idx * BLOCK_N)
lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(32)         # OPUS grpm offset (×32, not ×4)
rel_lo_i32   = fx.Int32(q_row_i32 - kv_start_i32 - lane_off_i32)
rel_hi_i32   = fx.Int32(rel_lo_i32 - fx.Int32(4))            # OPUS smem_d_n_split = 4

# OPUS-permuted thresholds (kernel source lines 801-806):
pair_thresholds = [
    (0,  8), (16, 24),     # r=0,1   r=2,3
    (1,  9), (17, 25),     # r=4,5   r=6,7
    (2, 10), (18, 26),     # r=8,9   r=10,11
    (3, 11), (19, 27),     # r=12,13 r=14,15
]
for thr_x, thr_y in pair_thresholds:
    s_lo[idx_x], s_lo[idx_y] = _attn_mask_vec2_imm(rel_lo_i32, neg_inf_i32,
                                                   thr_x, thr_y,
                                                   s_lo[idx_x], s_lo[idx_y])
    s_hi[idx_x], s_hi[idx_y] = _attn_mask_vec2_imm(rel_hi_i32, neg_inf_i32,
                                                   thr_x, thr_y,
                                                   s_hi[idx_x], s_hi[idx_y])
```

The mask is applied unconditionally when `CAUSAL=True` (no runtime `q_pos < kv_end_pos` early-exit). For tiles entirely below the diagonal the masks are no-ops (all `rel ≥ thr`) but the instructions are still issued — costing ~30 unused VALU cycles per such iteration. Adding a runtime gate is constrained by the FlyDSL AST rewriter's treatment of `if` blocks that mutate Python list state.

---

## 8. Online Softmax with Lazy Rescaling

### 8.1 Algorithm

The classical online-softmax recurrence keeps two scalars per Q row (`m`, `l`) and the running output `O`:

```
S_log2 = Q_scaled @ K^T          # Q_scaled = Q * (1/sqrt(D)) * log2(e)
m_new  = max(m_old, rowmax(S_log2))
corr   = exp2(m_old - m_new)
P      = exp2(S_log2 - m_new)
O     = O * corr + P @ V
l_new = l_old * corr + rowsum(P)
m_old, l_old := m_new, l_new
```

Final: `O := O / l_final`.

The OPUS lazy variant adds an optimization: when no lane's tile-max moved the running max by more than 8 in this log2 domain, it clamps `m_new := m_old`, which forces `corr ≡ 1` and lets the `O *= corr` multiply be elided.

### 8.2 Row-Max Reduction

Each wave reduces its `32 + 32 = 64` per-row score elements (across N-strips lo + hi) into one max value per Q row:

```python
m_raw = -inf
for r in range_constexpr(16):
    m_raw = max(m_raw, s_raw_lo[r])
    m_raw = max(m_raw, s_raw_hi[r])

# Cross-half reduction: lane i ↔ lane (i+32)
m_peer = m_raw.shuffle_xor(width=64, mask=32)
m_tile_max = max(m_raw, m_peer)
```

After the `shuffle_xor`, every lane in the wave holds the same value: the max over all 64 score elements of *that* Q row. This is identical in semantics to OPUS's `permlane32_swap + max` pair.

### 8.3 Lazy Rescale Check (ballot)

```python
m_diff     = m_tile_max - m_running
below_pred = (m_diff <= 8.0)
ballot_val      = rocdl.ballot(i64, below_pred)          # 64-bit mask
all_below       = (ballot_val == -1_i64)                  # all 64 lanes set

m_new_raw = all_below.select(m_running, max(m_running, m_tile_max))
```

When `all_below` is true:
- `m_new_raw = m_running` → `corr = exp2(0) = 1` → `l_new = l_running + tile_sum`
- The `O *= corr` multiply is replaced by `O *= 1.0` (LLVM constant-fold).

The math is **exact**: with `m_new := m_old`, the subsequent `S_log2 - m_new` is just `S_log2 - m_old`, and the running `O` already represents partial outputs in the `m_old` frame, so no rescale is needed. The threshold is compared directly to `m_tile_max - m_row` because the Q preload has already folded `sm_scale * log2e` into `Q`.

The scaling factor `sm_scale * log2e` is folded once as a single `f32` constant `c_sm_scale_log2e = (1/√D) * log2(e) ≈ 0.1275` for `D=128`, during Q preload.

### 8.4 Subtract + Base-2 Exponentiation

```python
diff = S_log2[r] - m_new
P[r] = exp2(diff)
```

There is no per-element softmax FMA in the current OPUS path. Q was already multiplied by `(1/√D) * log2(e)` during preload, so each softmax element is a plain `fsub` followed by `rocdl.exp2`. A running per-lane `local_sum` accumulates across the 32 elements. A final `shuffle_xor` by 32 + add gives the wave-wide row sum.

### 8.5 P-Value Packing to BF16

The fp32 `P` values produced by exp2 are packed back to bf16 for GEMM2:

```python
def bf16_trunc_pack_v8(f32_vals):
    # Pack 8 f32 → 8 bf16 by taking the upper 16 bits of each f32.
    pairs = []
    for j in range_constexpr(4):
        lo_i32 = bitcast<i32>(f32_vals[2*j])
        hi_i32 = bitcast<i32>(f32_vals[2*j + 1])
        pairs.append((hi_i32 & 0xFFFF_0000) | (lo_i32 >> 16))
    return bitcast<v8bf16>(Vec.from_elements(pairs, i32))
```

This is **round-to-zero (rtz) truncation**, matching the OPUS C++ `opus::cast<D_ATTN>(v_s)` semantics. The truncation produces a `v8bf16` (one MFMA A pack) for each of the 2 PV-steps × 2 N-strips = 4 packs total per wave.

---

## 9. GEMM2: P × V Output Accumulation

### 9.1 Tile Decomposition

```
O[32 × 128] += P[32 × 64] @ V[64 × 128]    (per wave's slice)

decomposed into:
  D-chunks (HEAD_DIM):    4   (D_CHUNKS, columns of O)
  PV K-steps (KV_TILE):   2   (PV_K_STEPS, sub-K within each MFMA chain)
  N-strips per PV-step:   2   (lo + hi V halves)

Total MFMAs per wave per outer iter: 4 × 2 × 2 = 16

O accumulator (per wave):
  o_accs[0..3]:  <16 × f32> each, one per D-chunk
                 = 4 × 16 = 64 f32 values per lane → 64 VGPRs
```

The outer iteration generates an `_steps` list `[(dc, pks) for dc in 0..3 for pks in 0..1]` of length 8. Each step contributes two MFMAs (`v_lo` * `p_lo`, `v_hi` * `p_hi`), accumulating into `o_accs[dc]`.

### 9.2 V LDS Read (OPUS `u_rv` via `ds_read_tr16_b64`)

After P7 the V read uses OPUS `u_rv` (`make_layout_rv`, `gqa_d128_kernel_template.hpp` lines 150-185). The helper `_read_v_packs_for_k_substep(buf_id, k_substep)` returns 4 D-chunk packs (`<8 × bf16>` each) for one of the four `step_k` substeps; the caller iterates `kss ∈ {0,1,2,3}` to gather all 16 V packs for a GEMM2 cluster (kernel source lines 731-754).

The lane decomposition follows `make_layout_rv`'s `grp_id / lane_in_grp` split:

```
lane_per_grp = 16   (lanes per ds_read_tr16_b64 row group)
lane_lo      = 4    (low 2 bits of lane_in_grp → which 4-bf16 slice)
lane_hi      = 4    (mid 2 bits of lane_in_grp → which D row within a 4-row block)
num_grps     = WARP_SIZE / lane_per_grp = 4
grp_n        = W_N / (lane_lo * VEC_TR_V) = 32 / (4*4) = 2
grp_k        = num_grps / grp_n = 2

# Per-lane base offset (bf16), kernel source lines 724-729:
urv_base_per_lane = (
      (lane // 32)        * OPUS_URV_GRPK         #  2 176 bf16 = 4 * 544  (grp_k axis 2)
    + ((lane % 16) // 4)  * OPUS_URV_LANE_HI      #    544 bf16  (lane_hi axis 3)
    + ((lane // 16) % 2)  * OPUS_URV_GRP_N        #     16 bf16  = lane_lo * VEC_TR_V (grp_n axis 6)
    + (lane % 4)          * OPUS_URV_LANE_LO      #      4 bf16  = VEC_TR_V (lane_lo axis 7)
)
```

Inside `_read_v_packs_for_k_substep`, two further axes select the substep and D-chunk:

```python
# OPUS_URV_STEP_K_STRIDE = 2 * D_128B_SIZE = 128 bf16 per step_k
step_k_off = k_substep * 128

for dc in range(D_CHUNKS):                # 4 D-chunks
    # axes 0 / 1 of u_rv:
    #   i_0 = dc // 2  →  selects d_rpt (D < 64 or D >= 64);  stride 4 352 bf16
    #   i_1 = dc %  2  →  selects half-D sub-row;             stride    32 bf16
    dc_off = i_0 * OPUS_URV_DC_AXIS0 + i_1 * OPUS_URV_DC_AXIS1

    # axis 5 of u_rv: stride D_128B_SIZE = 64 bf16 within the same step_k
    lds_off_a = v_buf_base(buf_id) + urv_base_per_lane + step_k_off + dc_off
    lds_off_b = lds_off_a + 64

    a = ds_read_tr_v4f16(lds_off_a)        # → 4 bf16 / lane (with 4x4 HW transpose)
    b = ds_read_tr_v4f16(lds_off_b)
    packs.append(Vec(a).shuffle(Vec(b), [0..7]))   # 8 bf16 / lane → MFMA-A pack
```

`ds_read_tr16_b64` is gfx950-only — it performs a 4×4 transpose during the LDS read, reorganizing the data layout from "row-major K × D" (natural for storing V via `u_sv`) to "column-major D × K" (needed as the MFMA A operand for `O += V^T @ P`). The 64 B padding in `u_sv` (§4.3) is what guarantees these tr-reads are bank-conflict-free. Without this instruction the kernel would need a software `ds_permute_b32 + v_perm_b32` chain (as in the MI308X ASM kernel).

Per consumer cluster `_read_v_packs_for_k_substep` is called 4 times (one per substep) and each call issues `2 × D_CHUNKS = 8` `ds_read_tr16_b64` instructions, totalling **32 tr-reads per cluster** (= 16 V packs of `<8 × bf16>`).

### 9.3 V Hoist Across Cluster Boundary (P5/P6 correctness)

After P5 the kernel uses an asymmetric `s_barrier` to phase-shift warps 0–3 and warps 4–7 by one cluster. To make this race-free, **all V LDS reads are hoisted out of the GEMM2 cluster into the immediately preceding cluster**, so V is held in VGPRs across the cluster-boundary `s_barrier`:

| GEMM2 cluster (consumer) | V hoist site (producer) | V from buffer |
|---|---|---|
| Main loop Cluster 3 | Main loop Cluster 2 | `s_v[0]` |
| Main loop Cluster 7 | Main loop Cluster 6 | `s_v[1]` |
| Epilogue Cluster 3 | Epilogue Cluster 2 | `s_v[0]` |
| Epilogue Cluster 7 | Epilogue Cluster 6 | `s_v[1]` |
| Epilogue Cluster 11 | Epilogue Cluster 10 | `s_v[0]` |
| Epilogue Cluster 13 | Epilogue Cluster 12 | `s_v[1]` |

The producer cluster issues `for kss in range(4): v_packs.append(_read_v_packs_for_k_substep(buf, kss))` — 16 packs materialized into VGPRs. The consumer cluster's MFMA chain then reads those VGPRs:

```python
# ─── Cluster 2 ───
coop_dma_k(j_idx * BLOCK_N, 1)               # K[j] async into s_k[1]
v_packs_a = []
for kss in range_constexpr(4):               # 16 V packs into VGPRs
    v_packs_a.append(_read_v_packs_for_k_substep(0, kss))
rocdl.s_waitcnt(_LGKMCNT_0_ONLY); _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
gpu.barrier()                                # ← stagger-aware s_barrier

# ─── Cluster 3 ───
rocdl.s_setprio(1)
for kss in range_constexpr(4):
    v_pk = v_packs_a[kss]                    # use pre-loaded VGPRs
    p_pk = (v_p_lo_a if kss < 2 else v_p_hi_a)[kss % 2]
    for dc in range_constexpr(D_CHUNKS):
        o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])
```

Without the hoist, warps 4-7 (one cluster behind warps 0-3 under stagger) would read `s_v[buf]` after warps 0-3 had already issued the next iteration's `async_load` into the same LDS buffer — a classic data race. With V already in registers, the peer overwrite of LDS is harmless.

This mirrors the C++ template's `tr_load<T::VEC_TR_V>(s_v[buf], u_rv)` placement before every cluster-boundary `s_barrier`.

---

## 10. Output Finalization

### 10.1 Normalize O

After the main loop, every lane holds the final running `l` and `O` values:

```python
l_final  = loop_results[1]               # scalar f32 per lane
o_finals = loop_results[2 : 2 + D_CHUNKS]  # 4 × <16 × f32>

inv_l    = rocdl.rcp(f32, l_final)        # 1 / l_final, one v_rcp_f32
inv_l_vec = broadcast(inv_l, 16)
```

`rocdl.rcp` emits `v_rcp_f32` — a 1-cycle hardware reciprocal.

### 10.2 Convert FP32 to BF16

Each O element is truncated f32 → bf16 (rtz) before storing:

```python
for dc in 0..3:
    o_norm = o_finals[dc] * inv_l_vec
    for r in 0..15:
        o_val = o_norm[r]
        o_f16 = f32.to(bf16)(o_val)
        ...
```

The Vec elementwise multiply produces a `v_pk_mul_f32` per pair, and `Vec → bf16` lowers to truncation. LLVM scheduler interleaves these with the upcoming stores.

### 10.3 Store O to Global Memory

```python
if q_in_bounds:
    for dc in 0..3:
        for r in 0..15:
            d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
            d_col     = dc * 32 + d_row_rel
            o_global  = (batch * S + q_row) * STRIDE_TOKEN_Q
                       + q_head * HEAD_DIM + d_col
            global_store(o_ptr, o_global, o_f16)
```

The per-element global address mirrors the MFMA C output layout (the same `{0..3, 8..11, 16..19, 24..27}` element ordering used in the causal mask). Total 4 × 16 = 64 stores per lane per Q row, packed by LLVM into `global_store_dword` / `global_store_dwordx2` as the scheduler sees fit.

No LSE output is produced (FlyDSL's flash_attn_func API does not emit LSE; OPUS C++ similarly does not in the GQA path).

---

## 11. Main Loop Pipeline (Post P1–P7)

### 11.1 Pipeline Structure

```
╔═══════════════════════════════════════════════════════════════════════════╗
║       Complete Pipeline: 1 Q-Block, OPUS-style 8-cluster `j += 2` loop     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  Q REGISTER STATE (held across all KV iterations):                          ║
║    q_b_packs[0..7]: 8 × <8 × bf16>   GEMM0 B operand                       ║
║                                       (pre-scaled by (1/√D)·log2e, P2)     ║
║                                                                             ║
║  LDS PARTITIONS (OPUS interleaved layout, 68 096 B total, P7):              ║
║    K buf 0  ([+    0])  ↔  K buf 1  ([+17024])  u_sk, line stride 520     ║
║    V buf 0  ([+ 8320])  ↔  V buf 1  ([+25344])  u_sv, line stride 544     ║
║    Order:   K0 / V0 / K1 / V1   (interleaved, matches s_k[i] / s_v[i])    ║
║                                                                             ║
║  LOOP-CARRIED STATE (yielded across KV iterations):                         ║
║    m_row:               scalar f32     running max (log2-space, P2)        ║
║    l_row:               scalar f32     running sum                          ║
║    o_accs[0..3]:        4 × <16 × f32> output accumulator (4 D-chunks)     ║
║    v_s_0_lo_partial:    <16 × f32>     carried half-softmax (low N-strip)  ║
║    v_s_0_hi_partial:    <16 × f32>     carried half-softmax (high N-strip) ║
║                                                                             ║
║  STAGGER STATE (P5):                                                        ║
║    wave_id_uni = readfirstlane(tid / 64);   stagger = wave_id_uni / 4      ║
║    Group A (waves 0-3, stagger=0) and Group B (waves 4-7, stagger=1)       ║
║    occupy opposite cluster phases via asymmetric s_barrier (prologue +     ║
║    close-out).                                                              ║
║                                                                             ║
║ ┌─────────────────────────────────────────────────────────────────────────┐ ║
║ │ PROLOGUE (C++ lines 397-436)                                             │ ║
║ │   coop_dma_k(0, 0)              async K[0] → s_k[0]                      │ ║
║ │   s_waitcnt(0); s_barrier                                                │ ║
║ │   Q = load + scale (P2 in-flight)                                        │ ║
║ │   coop_dma_k(BLOCK_N, 1)        async K[1] → s_k[1]                      │ ║
║ │   coop_dma_v(0, 0)              async V[0] → s_v[0]                      │ ║
║ │   v_k_pro = ds_read s_k[0]                                                │ ║
║ │   sched_barrier; lgkm(0); vmcnt(NUM_DMA_V)                                │ ║
║ │   if (stagger) s_barrier        ← P5 dual-group open                      │ ║
║ │   v_s[0] = mma0(v_q, v_k_pro)                                            │ ║
║ │   causal_mask v_s[0]            (P4 inline asm)                           │ ║
║ │   m_row = row_max(v_s[0]); v_s[0] -= m_row; exp2(v_s[0][0..15])          │ ║
║ │   _anchor_pair(v_s[0])          (P4 anchor #1)                            │ ║
║ │   sched_barrier; s_barrier                                                │ ║
║ │   coop_dma_k(2*BLOCK_N, 0)      async K[2] → s_k[0]                      │ ║
║ │   init_args = [m_row, 0, 0, 0, 0, 0, v_s_0_lo_partial, v_s_0_hi_partial] │ ║
║ └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                             ║
║ ┌─────────────────────────────────────────────────────────────────────────┐ ║
║ │ MAIN LOOP: for j = 3; j < max_num_tiles - 1; j += 2                      │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 0 (C++ 441-447) ════════════════════════════════════════    │ ║
║ │   coop_dma_v((j-2)*BLOCK_N, 1)      async V[j-2] → s_v[1]                │ ║
║ │   k_pl_a, k_ph_a = ds_read s_k[1]   load K[j-2] packs (16)                │ ║
║ │   lgkm(0); vmcnt(NUM_DMA_K+NUM_DMA_V)                                    │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 1 (C++ 449-459) ════════════════════════════════════════    │ ║
║ │   v_s[1] = mma0(v_q, K[j-2])                          (16 MFMAs)         │ ║
║ │   exp2(v_s[0][16..31]); l_row += sum(v_s[0])                              │ ║
║ │   v_p_a = cast<bf16>(v_s[0])     P[j-3] packs                             │ ║
║ │   _anchor_packs(v_p_a)           (P4 anchor #2)                           │ ║
║ │   sched_barrier_exp_pairs<6,3,1>; sched_barrier_pairs<10,5,1>  (P3)      │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 2 (C++ 461-468, with P5 V hoist) ═══════════════════════    │ ║
║ │   coop_dma_k(j*BLOCK_N, 1)           async K[j] → s_k[1]                 │ ║
║ │   v_packs_a[0..3] = tr_load V[j-3] from s_v[0]    (16 packs, HOISTED)    │ ║
║ │   lgkm(0); vmcnt(NUM_DMA_K+NUM_DMA_V)                                    │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 3 (C++ 470-496) ════════════════════════════════════════    │ ║
║ │   s_setprio(1)                                                            │ ║
║ │   step_k(0): 4 MFMAs (one per D-chunk)  o_accs += V[j-3]_0 @ P[j-3]_0   │ ║
║ │   row_max_a = wave_row_max(v_s[1])                                        │ ║
║ │   sched_barrier_pairs<4,5,2>     (P3)                                     │ ║
║ │   below_a = (row_max_a - m_row) ≤ 8.0                                    │ ║
║ │   all_below_a = ballot(below_a) == read_exec()                            │ ║
║ │   m_new_a = all_below_a ? m_row : max(m_row, row_max_a)                  │ ║
║ │   corr_a  = exp2(m_row - m_new_a)                                         │ ║
║ │   eff_corr_a = all_below_a ? 1.0 : corr_a    (LAZY RESCALE)              │ ║
║ │   _scale_o(o_accs, eff_corr_a); l_row *= corr_a; m_row = m_new_a         │ ║
║ │   step_k(1..3): 12 more MFMAs (V packs from v_packs_a[1..3])             │ ║
║ │   v_s[1] -= m_new_a; exp2(v_s[1][0..15])                                  │ ║
║ │   _anchor_pair(v_s[1])            (P4 anchor #3)                          │ ║
║ │   sched_barrier_pairs<6,5,2>; sched_barrier_exp_pairs<6,3,2>             │ ║
║ │   s_setprio(0); sched_barrier; s_barrier; sched_barrier                   │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 4 (C++ 498-505) ════════════════════════════════════════    │ ║
║ │   coop_dma_v((j-1)*BLOCK_N, 0)       async V[j-1] → s_v[0]               │ ║
║ │   k_pl_b, k_ph_b = ds_read s_k[0]    load K[j-1] packs                    │ ║
║ │   lgkm(0); vmcnt(NUM_DMA_K+NUM_DMA_V)                                    │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 5 (C++ 507-517) ════════════════════════════════════════    │ ║
║ │   v_s[0] = mma0(v_q, K[j-1])              (16 MFMAs)                     │ ║
║ │   exp2(v_s[1][16..31]); l_row += sum(v_s[1])                              │ ║
║ │   v_p_b = cast<bf16>(v_s[1])     P[j-2] packs                             │ ║
║ │   _anchor_packs(v_p_b)            (P4 anchor #4)                          │ ║
║ │   sched_barrier_exp_pairs<6,3,3>; sched_barrier_pairs<10,5,3>            │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 6 (C++ 519-532, with P5 V hoist + mask) ════════════════    │ ║
║ │   coop_dma_k((j+1)*BLOCK_N, 0)       async K[j+1] → s_k[0]               │ ║
║ │   v_packs_b[0..3] = tr_load V[j-2] from s_v[1]    (16 packs, HOISTED)    │ ║
║ │   causal_mask v_s[0]              (P4 inline asm, on S[j-1])              │ ║
║ │   lgkm(0); vmcnt(NUM_DMA_K+NUM_DMA_V)                                    │ ║
║ │   sched_barrier; s_barrier; sched_barrier                                 │ ║
║ │                                                                          │ ║
║ │ ═══ Cluster 7 (C++ 534-560) ════════════════════════════════════════    │ ║
║ │   s_setprio(1)                                                            │ ║
║ │   step_k(0): 4 MFMAs   o_accs += V[j-2]_0 @ P[j-2]_0                    │ ║
║ │   row_max_b = wave_row_max(v_s[0])                                        │ ║
║ │   sched_barrier_pairs<4,5,4>                                              │ ║
║ │   {LAZY RESCALE as Cluster 3, on row_max_b}                              │ ║
║ │   step_k(1..3): 12 more MFMAs                                             │ ║
║ │   v_s[0] -= m_new_b; exp2(v_s[0][0..15])                                  │ ║
║ │   _anchor_pair(v_s[0])            (P4 anchor #5)                          │ ║
║ │   sched_barrier_pairs<6,5,4>; sched_barrier_exp_pairs<6,3,4>             │ ║
║ │   s_setprio(0); sched_barrier; s_barrier; sched_barrier                   │ ║
║ │                                                                          │ ║
║ │   yield [m_row, l_row, o_accs[0..3], v_s_0_lo_partial, v_s_0_hi_partial]│ ║
║ └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                             ║
║ ┌─────────────────────────────────────────────────────────────────────────┐ ║
║ │ EPILOGUE (C++ lines 564-742) — drains the last 4 KV tiles                │ ║
║ │   Cluster 0-13 mirrors the main-loop ping-pong but                       │ ║
║ │   (a) no `j += 2` outer iteration,                                       │ ║
║ │   (b) every rescale is unconditional (no lazy gate),                     │ ║
║ │   (c) V hoist still in place at Clusters 2/6/10/12,                      │ ║
║ │   (d) Cluster 11 runs FULL exp (both halves) + final P cast,             │ ║
║ │   (e) Cluster 13 is the final mma1.                                      │ ║
║ └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                             ║
║ ┌─────────────────────────────────────────────────────────────────────────┐ ║
║ │ CLOSE-OUT (C++ lines 744-754)                                            │ ║
║ │   inv_l = rcp(l_row)                                                     │ ║
║ │   if (!stagger) s_barrier        ← P5 dual-group close                   │ ║
║ │   if q_in_bounds: for dc in 0..3, for r in 0..15:                        │ ║
║ │       o_bf16 = f32_to_bf16(o_accs[dc][r] * inv_l)                        │ ║
║ │       global_store(o_ptr, addr(dc, r), o_bf16)                           │ ║
║ └─────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 11.2 Cluster-by-Cluster Breakdown

Each KV iteration is split into 8 clusters with `rocdl.sched_barrier(0)` fences flanking every `gpu.barrier()`. Within each cluster, `_sched_barrier_pairs(N, V, G)` and `_sched_barrier_exp_pairs(N, E, G)` (P3) emit grouped scheduling hints that mirror the C++ template's `sched_barrier_pairs<N,V,G>` template instantiations exactly.

| Cluster | Dominant work | MFMAs | LDS reads | Sync |
|:-------:|---|:-:|:-:|---|
| Main 0 | wait + s_barrier; ds_read K[j-2] + async V[j-2] | 0 | 16 K packs | lgkm(0)+vmcnt(NUM_DMA_K+NUM_DMA_V); s_barrier |
| Main 1 | mma0 (S[j-2]) + finish softmax v_s[0] + cast P[j-3] | 16 | 0 | sched_group_barrier (P3); s_barrier |
| Main 2 | async K[j] + tr_load V[j-3] (HOIST) | 0 | 16 V packs | lgkm(0)+vmcnt; s_barrier |
| Main 3 | GEMM2 step_k(0..3) (16 MFMAs) + lazy rescale + softmax v_s[1] head | 16 | 0 | setprio(1/0); s_barrier |
| Main 4 | wait + s_barrier; ds_read K[j-1] + async V[j-1] | 0 | 16 K packs | lgkm(0)+vmcnt; s_barrier |
| Main 5 | mma0 (S[j-1]) + finish softmax v_s[1] + cast P[j-2] | 16 | 0 | sched_group_barrier; s_barrier |
| Main 6 | async K[j+1] + tr_load V[j-2] (HOIST) + causal mask v_s[0] | 0 | 16 V packs | lgkm(0)+vmcnt; s_barrier |
| Main 7 | GEMM2 step_k(0..3) (16 MFMAs) + lazy rescale + softmax v_s[0] head | 16 | 0 | setprio(1/0); s_barrier |

Total per iteration: **64 MFMAs** (32 GEMM0 + 32 GEMM2), 64 K packs + 64 V packs from LDS, 4 K async-load tiles + 4 V async-load tiles into LDS, 8 `s_barrier`s.

### 11.3 Synchronization Points

| Location | Sync type | What it guards |
|---|---|---|
| Cluster 0/2/4/6 (each) | `rocdl.s_waitcnt(_LGKMCNT_0_ONLY)` | Wait for K/V LDS reads from this cluster to retire |
| Cluster 0/2/4/6 (each) | `_waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)` | Wait for outstanding K/V VMEM loads to drop below the buffer-double-buffer limit |
| Every cluster boundary | `gpu.barrier()` | All 8 waves see fresh LDS contents before consuming |
| Prologue / close-out | `_stagger_extra_barrier_if_*` | P5 dual-group asymmetric barrier (only one group hits it) |
| Cluster 3/7 begin/end | `rocdl.s_setprio(1) / s_setprio(0)` | Raise priority during GEMM2 MFMA chain, lower after |
| Every cluster | `rocdl.sched_barrier(0)` (pair: before+after each `gpu.barrier()`) | Scheduler fence — bound LLVM reordering window to one cluster |
| Inside Cluster 1/3/5/7 | `_sched_barrier_pairs` / `_sched_barrier_exp_pairs` | Group-tagged hints telling the scheduler how many MFMA/VALU/EXP it should fit per slot (P3) |

The total `s_barrier` count per main-loop iteration is **8** (one per cluster). Under stagger, warps 0-3 and 4-7 are offset by one ordinal at the prologue and re-aligned at the close-out, so the total barrier count is identical for both groups.

### 11.4 OPUS Optimizations Layered In

After P1–P7 every OPUS optimization is present:

| OPUS feature | Status | FlyDSL site | C++ reference |
|---|---|---|---|
| 3D grid `(H, num_q_blocks, B)` | ✓ | `launch_flash_attn_opus` line 1593 (grid spec at line 1628) | `dim3 grid(H, num_q_blocks, B)` |
| `ds_read_tr16_b64` HW transpose for V | ✓ | `ds_read_tr_v4f16` line 368 | `tr_load<T::VEC_TR_V>` |
| Double-buffered K LDS | ✓ | `NUM_PREFETCH_K = 2` line 171 | `s_k[2]` |
| **Double-buffered V LDS (fully)** | ✓ (P1) | `NUM_PREFETCH_V = 2` line 172; used at every cluster | `s_v[2]` |
| `buffer_load_dwordx4_lds` async load | ✓ | `coop_dma_k/v` lines 564, 593 | `async_load<T::VEC_KV>` |
| Online softmax (m, l) | ✓ | loop-carried state line 989 | loop-carried locals |
| **Lazy rescale** (FlyDSL: `ballot == -1`; C++: `ballot == read_exec`) | ✓ (default ON) | `OPUS_LAZY_RESCALE` line 222; sites 1082-1095, 1195-1208 | lines 475-484 / 539-548 |
| `s_setprio(1) / s_setprio(0)` bracket | ✓ (default ON) | `OPUS_SETPRIO` line 223; sites 1062, 1122, etc. | lines 471, 493 |
| `s_nop 15 + s_nop 7` yield window | ✗ not emitted | `OPUS_YIELD_NOP` is parsed at line 237 but currently unused | no `s_nop` sequence in this C++ template |
| **In-flight Q scaling** | ✓ (P2) | Q load loop lines 631-655 | lines 403-406 |
| **8-cluster `j += 2` ping-pong** | ✓ (P1) | `for j, loop_args in range(3, max-1, 2)` line 997 | `for (int j = 3; j < max-1; j += 2)` |
| **`sched_group_barrier` discipline** | ✓ (P3) | `_sched_barrier_pairs/_exp_pairs` line 286-304 | template `sched_barrier_pairs<P,V,G>` line 18-30 |
| **Inline-asm causal mask** | ✓ (P4) | `_attn_mask_imm_single` line 413 | `attn_mask_vec2_imm` line 233-249 |
| **Register anchors (8 sites)** | ✓ (P4) | `_anchor_vec`/`_anchor_pair`/`_anchor_packs` line 447-461 | `asm volatile("" : "+v"(v))` at 8 C++ sites |
| **Dual-group stagger (`if warp/4 s_barrier`)** | ✓ (P5/P6, default ON) | `_stagger_extra_barrier_if_*` line 463-508 | lines 308, 415-418, 748-750 |
| **V LDS read hoisting** | ✓ (P5/P6 correctness) | 6 sites: 1051, 1165, 1293, 1384, 1469, 1541 | C++ tr_load before each cluster-boundary s_barrier |
| **OPUS K/V LDS layout (`u_gk`/`u_sk`/`u_rk`/`u_gv`/`u_sv`/`u_rv`)** | ✓ (P7, atomic) | constants line 149-196; `coop_dma_k/v` line 564/593; `_read_k_packs_for_buf` line 698; `_read_v_packs_for_k_substep` line 731 | `make_layout_gk_gv/sk_sv/rk/rv` lines 77-185 |
| **OPUS-permuted causal mask thresholds** | ✓ (P7) | `_causal_mask_inplace` line 769-830 | `attn_mask_vec2_imm` lines 233-249; wrapper `attn_mask_causal_tile` lines 251-289 |

Most runtime-toggled optimizations can be disabled for ablation:

- `FLYDSL_OPUS_LAZY_RESCALE=0` disables the ballot-clamp (always rescale)
- `FLYDSL_OPUS_SETPRIO=0` removes the `s_setprio(1/0)` calls
- `FLYDSL_OPUS_YIELD_NOP` is parsed by the module but currently unused; no `s_nop 15 + s_nop 7` sequence is emitted
- `FLYDSL_OPUS_STAGGER=0` falls back to symmetric `gpu.barrier()` (no dual-group phase shift)

The OPUS K/V LDS layout itself is **not** gated — there is no env var to revert to the pre-P7 XOR-swizzled layout. The atomic port replaces the layout end-to-end (allocator + DMA writers + readers + causal mask) because the consumers depend on the producer's layout for correctness.

---

## 12. Performance Status and Known Gaps

### Verified correctness (MaxErr threshold 1 × 10⁻², MinCos threshold 0.99)

Measured against the PyTorch f32 reference with `FLYDSL_ENABLE_OPUS_PATH=1` and a freshly cleared `~/.flydsl/cache/`:

| Config (B / S / H_Q / H_KV / D / dtype, OPUS path) | causal MaxErr / MinCos | nocausal MaxErr / MinCos |
|---|---|---|
| 1 / 256  / 64 / 64 / 128 / bf16  | 3.91 × 10⁻³ / 0.99999 | 9.77 × 10⁻⁴ / 0.99999 |
| 1 / 512  / 64 / 64 / 128 / bf16  | 3.91 × 10⁻³ / 0.99999 | 9.77 × 10⁻⁴ / 0.99999 |
| 1 / 8192 / 64 / 64 / 128 / bf16  | 3.91 × 10⁻³ / 0.99999 | 2.44 × 10⁻⁴ / 0.99999 |
| **16 / 8192 / 64 / 64 / 128 / bf16** | **3.91 × 10⁻³ / 0.99999** | **2.44 × 10⁻⁴ / 0.99999** |
| 16 / 8192 / 64 /  8 / 128 / bf16 (GQA) | 3.91 × 10⁻³ / 0.99999 | 2.44 × 10⁻⁴ / 0.99999 |

The documented configurations in `tests/kernels/test_flash_opus_attn.py` PASS under `--warmup 5 --iters 100`; the active `DEFAULT_CONFIGS` currently contains the two B=16/S=8192/H=64 bf16 MHA/GQA shapes.

### Performance (B=16 S=8192 H=64 D=128 bf16, MI355X, post P7)

| Path | causal TFLOPS | % of OPUS C++ | nocausal TFLOPS |
|---|---:|---:|---:|
| **FlyDSL OPUS path, default (P1–P7, stagger=1)** | **69.6** | 6.2 % | 66.3 |
| FlyDSL OPUS path, GQA 8-to-1 (16/8192/64/8/128 bf16) | 73.5 | 6.5 % | 69.1 |
| FlyDSL default `flash_attn_func` (non-OPUS, bf16) | 716 | 63.3 % | 640 |
| OPUS C++ (`mha_fwd` via aiter, reference) | 1131 | 100 % | 1165 |
| ASM kernel (`fmha_v3_fwd`) | 596 | 52.7 % | 1249 |

The OPUS path is the **structurally correct** port (matches C++ exactly post P1–P7) but is currently dominated by:

1. The P7 layout change introduces line-padding (16 B per K line, 64 B per V line) into the LDS DMA writers and readers. The line strides 520 bf16 / 544 bf16 require non-power-of-two address arithmetic that LLVM hoists into `v_mad_u64` chains, slightly inflating SALU work compared to the pre-P7 power-of-two-stride XOR-swizzled layout.
2. The OPUS `u_rk` N-permutation π and `u_rv` 4-axis decomposition emit more `ds_read` instructions per cluster than the pre-P7 layout (each cluster issues 32 `ds_read_tr16_b64` for V via the substep helper vs. one fused `tr_load` in C++).
3. The V hoist (P5/P6 correctness fix) still applies — every V LDS read sits in the cluster preceding its MFMA consumer, lengthening the V VGPR live range.

Net effect: P7 traded ~15 TFLOPS for full structural alignment with OPUS C++ (85.0 → 69.6 causal). The simpler XOR-swizzled layout was faster but did not match the C++ source line-by-line; the OPUS port is the correct foundation for further bit-for-bit ISA tuning. See [`FLASH_ATTN_OPUS_vs_CPP_Differences.md`](FLASH_ATTN_OPUS_vs_CPP_Differences.md) §4.3.1 and §7 for the perf-gap root-cause analysis.

### Activation

The OPUS path is opt-in via `FLYDSL_ENABLE_OPUS_PATH=1`. Once enabled, the P1–P7 structural optimizations are active; `FLYDSL_OPUS_LAZY_RESCALE`, `FLYDSL_OPUS_SETPRIO`, and `FLYDSL_OPUS_STAGGER` default ON. `FLYDSL_OPUS_YIELD_NOP` is parsed but currently unused. The dispatcher (`FlyDSL/kernels/flash_attn_func.py` → `_wrap_with_opus`) selects this path when `seq_len % 256 == 0 && seq_len >= 384 && head_dim == 128 && dtype == bf16 && gpu_arch.startswith("gfx950")`.

### Remaining work to close the perf gap

Now that the structural alignment is complete, the remaining work is at the **register-allocation / scheduling layer**:

1. **Coalesce V VGPR live range.** The current `_read_v_packs_for_k_substep` returns a Python list of 4 separate `<8 × bf16>` packs per substep × 4 substeps = 16 distinct SSA values. The C++ template's `tr_load` materializes one contiguous `vtype_b` that LLVM aliases against the GEMM2 accumulator. Fusing the 16 packs into one logical Vec.<32 × bf16> per cluster would let LLVM emit the same register layout.
2. **Anchor V VGPRs at the consumer.** Adding `_anchor_packs(v_packs_a)` between the V hoist (Cluster 2) and the consumer (Cluster 3) would force LLVM to keep the V VGPRs in their assigned banks across the cluster-boundary s_barrier, preventing spill.
3. **Vector-fused softmax tail.** The per-element `_fsub + exp2 + _fadd` chain in `_finish_softmax_cast_p` creates 32 separate SSA edges per cluster. Rewriting this with `Vec.exp2` + `Vec.add` + a single `bf16_trunc_pack_v8` would reduce IR size and let the scheduler fuse the operations into packed VALU instructions.
4. **Runtime causal-mask gate.** Adding the C++-style `if (q_start_pos < kv_end_pos)` early-exit around `_causal_mask_inplace` would save ~30 unused VALU cycles per below-diagonal tile (~10% of iterations on average causal workloads).
5. **Constant-fold the OPUS line strides.** The 520 / 544 bf16 line strides become 1 040 / 1 088 bytes in the DMA loaders; these multiplies presently hit `v_mad_u64` instead of constant-folded immediates. Hoisting the per-buffer base addresses into SGPR before the main loop (as the C++ template does via `s_k[i].ptr`) would let LLVM fold the multiplications away.

Each item is independently testable; item 1 is estimated to contribute the largest share of the remaining gap.

---

## 13. P1–P7 Evolution History

The kernel reached its current state through seven phases, each adding one alignment dimension with C++. See [`FLASH_ATTN_OPUS_vs_CPP_Differences.md`](FLASH_ATTN_OPUS_vs_CPP_Differences.md) §2 for full line-by-line mapping.

| Phase | Commit | Title | What was aligned |
|---|---|---|---|
| **P1** | `ad1fcaf` | align structure with gqa_d128_kernel_template.hpp | 8-cluster `j += 2` main loop + 14-cluster epilogue; V double-buffered; loop-carries `v_s_0_partial` |
| **P2** | `558b77c` | in-flight Q scaling (log2-space softmax) | Q pre-multiplied by `(1/√D)·log2e`; per-element softmax scaling collapses to subtract |
| **P3** | `39feb59` | sched_group_barrier_pairs/_exp_pairs discipline | `_sched_barrier_pairs(N,V,G)` / `_sched_barrier_exp_pairs(N,E,G)` helpers at 16 C++ call sites |
| **P4** | `e14cb5e` | inline-asm causal mask + 8 register anchors | `v_cmp_lt_i32 + v_cndmask_b32` imm-threshold asm; 8 `_anchor_*` sites mirroring C++ `asm volatile("" : "+v"(v))` |
| **P5** | `b62c600` | stagger mechanism (wired in, default OFF) | Asymmetric `if (stagger) s_barrier` (prologue) + `if (!stagger) s_barrier` (close-out); SGPR-resident `stagger_i32` via `readfirstlane` |
| **P5b** | `598981b` | hoist V LDS reads to fix P5 stagger correctness | All 6 V LDS read sites moved one cluster earlier so V is captured in VGPRs *before* each cluster-boundary barrier |
| **P6** | `55a135c` | flip OPUS_ENABLE_STAGGER default to ON | `FLYDSL_ENABLE_OPUS_PATH=1` selects the OPUS kernel; `FLYDSL_OPUS_*` flags still control lazy rescale / setprio / stagger defaults |
| **P7** | `2d07de2` | port OPUS permuted K/V LDS layout (atomic) | Interleaved `[K0][V0][K1][V1]` LDS (68 096 B); OPUS `u_gk` / `u_sk` DMA writers; OPUS `u_rk` K-read with N-permutation π(m)=(m%8)·8+m/8; OPUS `u_gv` / `u_sv` DMA writers; OPUS `u_rv` V-read via `ds_read_tr16_b64`; causal-mask thresholds reordered to π-image. Atomic switch — no env toggle. |

After P7, the test command

```bash
FLYDSL_ENABLE_OPUS_PATH=1 python tests/kernels/test_flash_opus_attn.py \
    --warmup 5 --iters 100
```

passes the documented test matrix with `MaxErr 3.91e-03 ≪ 1e-2` and `MinCos 0.99999 ≫ 0.99` — confirming numerical parity with the OPUS C++ reference while preserving the C++ template's clustering, synchronization, and LDS layout structure.

---

## 14. Q/K/V/S/O Layout Atlas

This section follows the same presentation style as `opus_attn/GQA_D128_KERNEL_Analysis_Detail.md` §5.6, but uses the **actual FlyDSL formulas** from `kernels/flash_attn_opus.py`. It expands every layout into concrete formulas and only uses tables for lane/axis expansion.

One important distinction: **Q and O never use LDS**. Q is loaded directly from GM to VGPR, K/V are double-buffered in LDS, S/P/O are VGPR-resident, and O is finally stored back to GM.

Notation:

```text
tid       = thread_id_x
lane_id   = tid % WARP_SIZE = tid % 64
wave_id   = tid / WARP_SIZE = tid / 64                   // 0..7 because BLOCK_SIZE / WARP_SIZE = 512 / 64 = 8

lane_mod_32 = lane_id % 32             // Q/O row inside one wave tile
lane_div_32 = lane_id / 32             // 0 or 1, half-wave selector

n_in_warp = lane_id / 8                // 0..7, which KV row inside a wave slab
d_bucket  = lane_id % 8                // 0..7, which 8-bf16 D vector inside a 64-bf16 half

q_block_size  = 8 * 32 = 256
q_block_start = q_start = q_block_idx * q_block_size
kv_tile(j)    = tile_start = kv_tile_start = j * 64

All element formulas below are in bf16 elements unless marked as byte addresses.
byte_addr = elem_addr * sizeof(bf16) = elem_addr * 2.
```

Notation range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `tid` | `0..511` | kernel launch block size is `BLOCK_SIZE = NUM_WAVES * WARP_SIZE = 8 * 64 = 512` |
| `lane_id` | `0..63` | `lane_id = tid % WARP_SIZE`, `WARP_SIZE = 64` |
| `wave_id` | `0..7` | `wave_id = tid // WARP_SIZE`, `tid=0..511` |
| `lane_mod_32` | `0..31` | `lane_mod_32 = lane_id % 32` |
| `lane_div_32` | `0, 1` | `lane_div_32 = lane_id // 32`, `lane_id=0..63` |
| `n_in_warp` | `0..7` | `n_in_warp = lane_id // VEC_KV`, `VEC_KV = 8` |
| `d_bucket` | `0..7` | `d_bucket = lane_id % VEC_KV`, `VEC_KV = 8` |
| `q_block_size` | `256` | `q_block_size = BLOCK_M = NUM_WAVES * ROWS_PER_WAVE = 8 * 32` |
| `q_block_start` / `q_start` | `q_block_idx * 256` | `q_start = q_block_idx * BLOCK_M` |
| `kv_tile(j)` / `tile_start` | `j * 64` | every KV tile has `BLOCK_N = 64` rows |

### 14.1 Q: GM -> VGPR

Q has no LDS residency. The FlyDSL Q preload mirrors OPUS `u_q`: each lane owns one Q row and reads two interleaved 8-bf16 D vectors for each 16-wide MFMA K step.

```text
GM Q base:
  Q[batch_idx, q_block_start, q_head_idx, 0]

Per workgroup Q rows:
  wave 0: rows q_block_start +   0..31
  wave 1: rows q_block_start +  32..63
  wave 2: rows q_block_start +  64..95
  wave 3: rows q_block_start +  96..127
  wave 4: rows q_block_start + 128..159
  wave 5: rows q_block_start + 160..191
  wave 6: rows q_block_start + 192..223
  wave 7: rows q_block_start + 224..255
```

Q GM address formula:

```text
ks  = 0..K_STEPS_QK-1 = 0..7              // K_STEPS_QK = HEAD_DIM / K_STEP_QK = 128 / 16 = 8
e   = 0..MFMA_LANE_K-1 = 0..7             // load_global_mfma_pack reads MFMA_LANE_K = 8 bf16

q_row = q_start + wave_id * 32 + lane_mod_32
q_in_bounds = q_row < seq_len_v
q_row_safe = q_in_bounds ? q_row : 0

q_col = ks * 16 + lane_div_32 * 8
q_dim = q_col + e

Q GM elem addr =
  (batch_idx * seq_len_v + q_row_safe) * STRIDE_TOKEN_Q
  + q_head_idx * HEAD_DIM
  + q_dim

If q_in_bounds is false, the loaded <8xbf16> pack is replaced with zero.
```

Q variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `ks` | `0..7` | loop `for ks in range_constexpr(K_STEPS_QK)`, `K_STEPS_QK = HEAD_DIM / K_STEP_QK = 128 / 16 = 8` |
| `e` | `0..7` | one `load_global_mfma_pack` returns `MFMA_LANE_K = 8` bf16 values |
| `q_row` | `q_start .. q_start+255` | `wave_id*32` covers `0,32,...,224`; `lane_mod_32` covers `0..31` |
| `q_in_bounds` | boolean | `q_row < seq_len_v` in Q preload |
| `q_row_safe` | `q_row` or `0` | selected by `q_in_bounds`; out-of-bound rows load from row 0 then are masked to zero |
| `q_col` | `0, 8, 16, 24, ..., 120` | `q_col = ks*K_STEP_QK + lane_div_32*MFMA_LANE_K` |
| `q_dim` | `0..127` | `ks*16` covers 8 groups of 16; `lane_div_32*8 + e` covers low/high 8 bf16 inside each group |
| `q_head_idx` | `(h_idx % NUM_HEADS_KV) * GQA_GROUP_SIZE + h_idx // NUM_HEADS_KV` | GQA head mapping in the kernel; `GQA_GROUP_SIZE = NUM_HEADS_Q / NUM_HEADS_KV` |

Per-lane VGPR expansion:

| lane_id | lane_mod_32 | lane_div_32 | `q_b_packs[0..7]` contents |
|---:|---:|---:|---|
| 0 | 0 | 0 | row 0, dim `0..7, 16..23, ..., 112..119` |
| 1 | 1 | 0 | row 1, dim `0..7, 16..23, ..., 112..119` |
| ... | ... | ... | ... |
| 31 | 31 | 0 | row 31, dim `0..7, 16..23, ..., 112..119` |
| 32 | 0 | 1 | row 0, dim `8..15, 24..31, ..., 120..127` |
| 33 | 1 | 1 | row 1, dim `8..15, 24..31, ..., 120..127` |
| ... | ... | ... | ... |
| 63 | 31 | 1 | row 31, dim `8..15, 24..31, ..., 120..127` |

VGPR shape:

```text
q_b_packs[ks][e] =
  Q[batch_idx,
    q_start + wave_id * 32 + lane_mod_32,
    q_head_idx,
    ks * 16 + lane_div_32 * 8 + e]

q_b_packs: 8 x <8xbf16> per lane = 64 bf16 values.
Each pack is pre-scaled by (1 / sqrt(128)) * log2e before GEMM0.
```

### 14.2 K: GM -> LDS -> VGPR

K is cooperatively loaded by all 512 threads. Each lane transfers one 16-byte vector (`VEC_KV=8 bf16`) for each of the two D half-lines.

K GM address formula:

```text
d_half = 0..SMEM_D_RPT-1 = 0..1          // SMEM_D_RPT = HEAD_DIM / D_128B_SIZE = 128 / 64 = 2
e      = 0..VEC_KV-1 = 0..7              // raw_ptr_buffer_load_lds copies 16 B = 8 bf16 per lane

n_in_warp = lane_id / 8
d_bucket  = lane_id % 8

global_n = n_in_warp * NUM_WAVES + wave_id
global_d = d_bucket * VEC_KV + d_half * D_128B_SIZE
global_row = batch_idx * seq_len_v + tile_start + global_n

K GM vector base elem addr =
  global_row * STRIDE_TOKEN_KV
  + kv_head_idx * HEAD_DIM
  + global_d

K GM elem addr for element e inside the 16B DMA vector =
  K GM vector base elem addr + e

K GM DMA byte offset passed to raw_ptr_buffer_load_lds =
  K GM vector base elem addr * BF16_BYTES
```

K GM variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `d_half` | `0, 1` | `coop_dma_k` loop `for d in range_constexpr(NUM_DMA_K)`, `NUM_DMA_K = SMEM_D_RPT = 2` |
| `e` | `0..7` | one DMA is `DMA_BYTES = 16`, i.e. `16 / sizeof(bf16) = 8` bf16 |
| `n_in_warp` | `0..7` | `lane_id // VEC_KV`, `lane_id=0..63`, `VEC_KV=8` |
| `d_bucket` | `0..7` | `lane_id % VEC_KV` |
| `global_n` | `0..63` | `n_in_warp * NUM_WAVES + wave_id`, `NUM_WAVES = 8` |
| `global_d + e` | `0..127` | `d_bucket*8 + d_half*64 + e` |
| `global_row` | `batch_idx * seq_len_v + tile_start + global_n` | `coop_dma_k` uses `seq_len_v = fx.Index(seq_len)` |
| `kv_head_idx` | `h_idx % NUM_HEADS_KV` | GQA head mapping in the kernel |

Per-wave GM expansion for `d_half=0`:

| wave | lane range | K row | D range |
|---:|---:|---:|---|
| 0 | `0..7` | 0 | `0..63` |
| 0 | `8..15` | 8 | `0..63` |
| 0 | `16..23` | 16 | `0..63` |
| ... | ... | ... | ... |
| 0 | `56..63` | 56 | `0..63` |
| 1 | `0..7` | 1 | `0..63` |
| 1 | `8..15` | 9 | `0..63` |
| ... | ... | ... | ... |
| 7 | `56..63` | 63 | `0..63` |

`d_half=1` repeats the same row mapping for `dim 64..127`.

K LDS buffer map:

```text
K_base[0] = OPUS_K_BUF_BASE[0] = 0
K_base[1] = OPUS_K_BUF_BASE[1] = OPUS_KV_PER_BUFFER = 17024

K line stride = SMEM_K_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_K_PAD = 512 + 8 = 520 bf16 = 1040 B
K tile size   = SMEM_K_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_K_LINE_STRIDE = 8 * 2 * 520 = 8320 bf16
```

K LDS destination formula:

```text
line = d_half * SMEM_N_RPT + wave_id             // d_half*8 + wave_id, range 0..15
in_line_elem = lane_id * VEC_KV + e              // lane_id*8 + e, range 0..511

K LDS elem addr =
  K_base[buf] + line * SMEM_K_LINE_STRIDE + in_line_elem

K LDS byte addr =
  lds_kv_base_idx
  + (K_base[buf] + line * SMEM_K_LINE_STRIDE + lane_id * VEC_KV + e) * BF16_BYTES
```

K LDS variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `buf` | `0` or `1` | all current `_read_k_packs_for_buf` / `coop_dma_k` call sites pass buffer id `0` or `1` |
| `line` | `0..15` | `d_half=0..1`, `SMEM_N_RPT=8`, `wave_id=0..7` |
| `in_line_elem` | `0..511` | `lane_id=0..63`, `VEC_KV=8`, `e=0..7` |
| padding | `elem 512..519` | K line stride is 520 bf16, so data `0..511` is followed by 8 bf16 padding |

K LDS address space for one buffer:

```text
K LDS Address Space:
Address increases downward; each line is 1040B = 1024B data + 16B pad.

D half 0: dim[0..63]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 0:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ┌────────────────────────────┐
Wave 1 line:        │ line 1:  rows 1,9,...,57  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 7:  rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘

D half 1: dim[64..127]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 8:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ┌────────────────────────────┐
Wave 1 line:        │ line 9:  rows 1,9,...,57  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 15: rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘
```

Inside one K LDS line:

```text
Byte[0:15]       <- lane 0:  one 16B vector, row n_in_warp=0, d_bucket=0
Byte[16:31]      <- lane 1:  one 16B vector, row n_in_warp=0, d_bucket=1
...
Byte[112:127]    <- lane 7:  one 16B vector, row n_in_warp=0, d_bucket=7
Byte[128:143]    <- lane 8:  one 16B vector, row n_in_warp=1, d_bucket=0
...
Byte[1008:1023]  <- lane 63: one 16B vector, row n_in_warp=7, d_bucket=7
Byte[1024:1039]  <- 8 bf16 padding
```

K VGPR read formula (`u_rk`) rearranges the padded LDS tile into GEMM0 operand order:

```text
lane_id_n = lane_id % K_SUB_N = lane_id % 32        // range 0..31

urk_base_per_lane =
  (lane_id_n % 8) * SMEM_K_LINE_STRIDE
  + (lane_id_n / 8) * D_128B_SIZE
  + lane_div_32 * VEC_KV

ks_offset =
  (ks / 4) * OPUS_URK_KSTEP_OUTER
  + (ks % 4) * OPUS_URK_KSTEP_INNER

idx_lo = K_base[buf] + urk_base_per_lane + ks_offset
idx_hi = idx_lo + OPUS_URK_N_STRIP_STRIDE
```

K VGPR base/range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `lane_id_n` | `0..31` | `lane_id % 32` |
| `lane_id_n % 8` | `0..7` | selects one of 8 K LDS lines within a D half |
| `lane_id_n / 8` | `0..3` | selects row offset `0..3` inside each 8-row line group |
| `lane_div_32` | `0, 1` | selects low/high 8 bf16 inside each 16-bf16 K step |
| `ks` | `0..7` | `_read_k_packs_for_buf` loop `for ks in range_constexpr(K_STEPS_QK)` |
| `ks_offset` | `(ks//4)*4160 + (ks%4)*16` | `OPUS_URK_KSTEP_OUTER = SMEM_N_RPT * SMEM_K_LINE_STRIDE = 4160`, `OPUS_URK_KSTEP_INNER = 16` |
| `idx_hi` | `idx_lo + 256` | `OPUS_URK_N_STRIP_STRIDE = 256`, matching the second N strip |

Expanded per-element form:

```text
p0 = lane_id_n % 8          // 0..7
p1 = lane_id_n / 8          // 0..3
p2 = lane_div_32            // 0 or 1

i_ngrp  = 0..1              // corresponds to lo/hi N strip selection in the expanded u_rk axes
i_dhalf = 0..1              // D half, derived from ks//4
i_k     = 0..3              // 16-bf16 step within one D half, derived from ks%4
e       = 0..7              // element inside one 8-bf16 K pack

K read elem addr =
    K_base[buf]
  + p0 * 520
  + (i_ngrp * 4 + p1) * 64
  + i_dhalf * (8 * 520)
  + ((i_k * 2 + p2) * 8 + e)
```

Reversing this address through the FlyDSL LDS write layout gives the logical K coordinate:

```text
K row = (i_ngrp * 4 + p1) * NUM_WAVES + p0
K dim = i_dhalf * 64 + (i_k * 2 + p2) * 8 + e
```

Per-lane K VGPR read expansion for `i_ngrp=0`, `i_dhalf=0`, `i_k=0`:

| lane_id | p0 | p1 | p2 | K row | D range |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 0 | `0..7` |
| 1 | 1 | 0 | 0 | 1 | `0..7` |
| 2 | 2 | 0 | 0 | 2 | `0..7` |
| ... | ... | ... | ... | ... | ... |
| 7 | 7 | 0 | 0 | 7 | `0..7` |
| 8 | 0 | 1 | 0 | 8 | `0..7` |
| 9 | 1 | 1 | 0 | 9 | `0..7` |
| ... | ... | ... | ... | ... | ... |
| 31 | 7 | 3 | 0 | 31 | `0..7` |
| 32 | 0 | 0 | 1 | 0 | `8..15` |
| 33 | 1 | 0 | 1 | 1 | `8..15` |
| ... | ... | ... | ... | ... | ... |
| 63 | 7 | 3 | 1 | 31 | `8..15` |

Thread/lane to LDS read address expansion for the same `i_ngrp=0`, `i_dhalf=0`, `i_k=0` slice:

| lane_id | thread_id_x | p0 | p1 | p2 | K LDS elem base | K LDS byte range | Logical K |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | `wave_id*64 + 0` | 0 | 0 | 0 | `K_base+0` | `0x000..0x00F` | row 0, dim `0..7` |
| 1 | `wave_id*64 + 1` | 1 | 0 | 0 | `K_base+520` | `0x410..0x41F` | row 1, dim `0..7` |
| 2 | `wave_id*64 + 2` | 2 | 0 | 0 | `K_base+1040` | `0x820..0x82F` | row 2, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | `wave_id*64 + 7` | 7 | 0 | 0 | `K_base+3640` | `0x1C70..0x1C7F` | row 7, dim `0..7` |
| 8 | `wave_id*64 + 8` | 0 | 1 | 0 | `K_base+64` | `0x080..0x08F` | row 8, dim `0..7` |
| 9 | `wave_id*64 + 9` | 1 | 1 | 0 | `K_base+584` | `0x490..0x49F` | row 9, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31 | `wave_id*64 + 31` | 7 | 3 | 0 | `K_base+3832` | `0x1DF0..0x1DFF` | row 31, dim `0..7` |
| 32 | `wave_id*64 + 32` | 0 | 0 | 1 | `K_base+8` | `0x010..0x01F` | row 0, dim `8..15` |
| 33 | `wave_id*64 + 33` | 1 | 0 | 1 | `K_base+528` | `0x420..0x42F` | row 1, dim `8..15` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 63 | `wave_id*64 + 63` | 7 | 3 | 1 | `K_base+3840` | `0x1E00..0x1E0F` | row 31, dim `8..15` |

The loop axes then cover the remaining K tile:

| Axis | Values | Effect |
|---|---|---|
| `i_ngrp` | `0, 1` | selects K rows `0..31` then `32..63` |
| `i_dhalf` | `0, 1` | selects D half `0..63` then `64..127` |
| `i_k` | `0..3` | advances within a D half by `16` bf16 per step |
| `p2` | `0, 1` | lane half reads the low/high 8 bf16 inside each 16-bf16 step |

VGPR shape:

```text
k_lo/k_hi packs in FlyDSL are slices of the OPUS u_rk stream.
Together they form the MFMA operand for:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

### 14.3 V: GM -> LDS -> VGPR

V uses the same GM coordinate layout as K and the same cooperative `buffer_load_dwordx4 ... lds`, but it uses a larger LDS padding and a transpose LDS read.

V GM address formula:

```text
d_half = 0..SMEM_D_RPT-1 = 0..1
e      = 0..VEC_KV-1 = 0..7

n_in_warp = lane_id / 8
d_bucket  = lane_id % 8

global_n = n_in_warp * NUM_WAVES + wave_id
global_d = d_bucket * VEC_KV + d_half * D_128B_SIZE
global_row = batch_idx * seq_len_v + tile_start + global_n

V GM vector base elem addr =
  global_row * STRIDE_TOKEN_KV
  + kv_head_idx * HEAD_DIM
  + global_d

V GM elem addr for element e inside the 16B DMA vector =
  V GM vector base elem addr + e

V GM DMA byte offset passed to raw_ptr_buffer_load_lds =
  V GM vector base elem addr * BF16_BYTES
```

V GM variable ranges are identical to K GM:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `d_half` | `0, 1` | `coop_dma_v` loop `for d in range_constexpr(NUM_DMA_V)`, `NUM_DMA_V = SMEM_D_RPT = 2` |
| `e` | `0..7` | one DMA copies `VEC_KV = 8` bf16 |
| `n_in_warp` | `0..7` | `lane_id // VEC_KV`, identical to K |
| `d_bucket` | `0..7` | `lane_id % VEC_KV`, identical to K |
| `global_n` | `0..63` | `n_in_warp * NUM_WAVES + wave_id`, identical to K |
| `global_d + e` | `0..127` | `d_bucket*8 + d_half*64 + e` |
| `global_row` | `batch_idx * seq_len_v + tile_start + global_n` | `coop_dma_v` uses `seq_len_v = fx.Index(seq_len)` |
| `kv_head_idx` | `h_idx % NUM_HEADS_KV` | same GQA head mapping as K |

V LDS buffer map:

```text
V_base[0] = OPUS_V_BUF_BASE[0] = SMEM_K_TILE_ELEMS = 8320
V_base[1] = OPUS_V_BUF_BASE[1] = SMEM_K_TILE_ELEMS + OPUS_KV_PER_BUFFER = 25344

V line stride = SMEM_V_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_V_PAD = 512 + 32 = 544 bf16 = 1088 B
V tile size   = SMEM_V_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_V_LINE_STRIDE = 8 * 2 * 544 = 8704 bf16
```

V LDS destination formula:

```text
line = d_half * SMEM_N_RPT + wave_id             // d_half*8 + wave_id, range 0..15
in_line_elem = lane_id * VEC_KV + e              // lane_id*8 + e, range 0..511

V LDS elem addr =
  V_base[buf] + line * SMEM_V_LINE_STRIDE + in_line_elem

V LDS byte addr =
  lds_kv_base_idx
  + (V_base[buf] + line * SMEM_V_LINE_STRIDE + lane_id * VEC_KV + e) * BF16_BYTES
```

V LDS variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `buf` | `0` or `1` | all current `coop_dma_v` / `_read_v_packs_for_k_substep` call sites pass buffer id `0` or `1` |
| `line` | `0..15` | `d_half=0..1`, `SMEM_N_RPT=8`, `wave_id=0..7` |
| `in_line_elem` | `0..511` | `lane_id=0..63`, `VEC_KV=8`, `e=0..7` |
| padding | `elem 512..543` | V line stride is 544 bf16, so data `0..511` is followed by 32 bf16 padding |

V LDS address space for one buffer:

```text
V LDS Address Space:
Address increases downward; each line is 1088B = 1024B data + 64B pad.

D half 0: dim[0..63]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 0:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 7:  rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘

D half 1: dim[64..127]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 8:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 15: rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘
```

Inside one V LDS line, the first 1024 bytes are lane-contiguous data exactly like K, followed by 64 bytes of padding:

```text
Byte[0:15]       <- lane 0
Byte[16:31]      <- lane 1
...
Byte[1008:1023]  <- lane 63
Byte[1024:1087]  <- 32 bf16 padding
```

V VGPR read formula (`u_rv`) is shaped for `ds_read_tr16_b64`:

```text
lane_per_grp = 16                       // fixed by ds_read_tr16_b64 16-lane transpose group
lane_lo      = 4                        // 4 lanes form the low transpose axis
lane_hi      = lane_per_grp / lane_lo = 4
num_grps     = WARP_SIZE / lane_per_grp = 64 / 16 = 4
grp_n        = K_SUB_N / (lane_lo * VEC_TR_V) = 32 / (4 * 4) = 2
grp_k        = num_grps / grp_n = 4 / 2 = 2

grp_id      = lane_id / 16
lane_in_grp = lane_id % 16

p0 = grp_id / grp_n
p1 = lane_in_grp / lane_lo
p2 = grp_id % grp_n
p3 = lane_in_grp % lane_lo
```

`u_rv` lane-variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `grp_id` | `0..3` | `lane_id // lane_per_grp`, `lane_per_grp=16` |
| `lane_in_grp` | `0..15` | `lane_id % 16` |
| `p0` | `0, 1` | `grp_id // grp_n`, `grp_n=2` |
| `p1` | `0..3` | `lane_in_grp // lane_lo`, `lane_lo=4` |
| `p2` | `0, 1` | `grp_id % grp_n` |
| `p3` | `0..3` | `lane_in_grp % lane_lo` |

Per-lane `u_rv` transpose-group expansion:

| lane_id range | grp_id | lane_in_grp | p0 | p1 | p2 | p3 | Meaning |
|---:|---:|---:|---:|---:|---:|---:|---|
| `0..3` | 0 | `0..3` | 0 | 0 | 0 | `0..3` | group 0, first 4-lane lo subgroup |
| `4..7` | 0 | `4..7` | 0 | 1 | 0 | `0..3` | group 0, second 4-lane lo subgroup |
| `8..11` | 0 | `8..11` | 0 | 2 | 0 | `0..3` | group 0, third 4-lane lo subgroup |
| `12..15` | 0 | `12..15` | 0 | 3 | 0 | `0..3` | group 0, fourth 4-lane lo subgroup |
| `16..19` | 1 | `0..3` | 0 | 0 | 1 | `0..3` | group 1, first 4-lane lo subgroup |
| `20..23` | 1 | `4..7` | 0 | 1 | 1 | `0..3` | group 1, second 4-lane lo subgroup |
| `24..27` | 1 | `8..11` | 0 | 2 | 1 | `0..3` | group 1, third 4-lane lo subgroup |
| `28..31` | 1 | `12..15` | 0 | 3 | 1 | `0..3` | group 1, fourth 4-lane lo subgroup |
| `32..35` | 2 | `0..3` | 1 | 0 | 0 | `0..3` | group 2, first 4-lane lo subgroup |
| `36..39` | 2 | `4..7` | 1 | 1 | 0 | `0..3` | group 2, second 4-lane lo subgroup |
| `40..43` | 2 | `8..11` | 1 | 2 | 0 | `0..3` | group 2, third 4-lane lo subgroup |
| `44..47` | 2 | `12..15` | 1 | 3 | 0 | `0..3` | group 2, fourth 4-lane lo subgroup |
| `48..51` | 3 | `0..3` | 1 | 0 | 1 | `0..3` | group 3, first 4-lane lo subgroup |
| `52..55` | 3 | `4..7` | 1 | 1 | 1 | `0..3` | group 3, second 4-lane lo subgroup |
| `56..59` | 3 | `8..11` | 1 | 2 | 1 | `0..3` | group 3, third 4-lane lo subgroup |
| `60..63` | 3 | `12..15` | 1 | 3 | 1 | `0..3` | group 3, fourth 4-lane lo subgroup |

`u_rv` LDS element address, written with the same variables as `_read_v_packs_for_k_substep`:

```text
buf = buf_id                         // passed into _read_v_packs_for_k_substep(buf_id, k_substep)
k_substep                            // passed into _read_v_packs_for_k_substep; callers pass kss from range_constexpr(4), so k_substep = 0..3
dc = 0..D_CHUNKS-1 = 0..3             // from: for dc in range_constexpr(D_CHUNKS), D_CHUNKS = HEAD_DIM / D_CHUNK = 128 / 32 = 4
read_pair = 0 or 1                    // two hard-coded reads: lds_off_lo and lds_off_lo + OPUS_URV_I5_STRIDE
e = 0..VEC_TR_V-1 = 0..3              // one ds_read_tr16_b64 returns VEC_TR_V = 4 bf16

i_0 = dc / 2
i_1 = dc % 2

urv_base_per_lane =
  lane_div_32 * OPUS_URV_GRPK
  + ((lane_id % 16) / 4) * OPUS_URV_LANE_HI
  + ((lane_id / 16) % 2) * OPUS_URV_GRP_N
  + (lane_id % 4) * OPUS_URV_LANE_LO

step_k_off = k_substep * OPUS_URV_STEP_K_STRIDE
dc_off = i_0 * OPUS_URV_DC_AXIS0 + i_1 * OPUS_URV_DC_AXIS1

lds_off_lo = V_base[buf] + urv_base_per_lane + step_k_off + dc_off
lds_off_hi = lds_off_lo + OPUS_URV_I5_STRIDE
```

Range derivation from the kernel code:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `buf` | caller parameter `buf_id` (`0` or `1` at all current call sites) | `_read_v_packs_for_k_substep(buf_id, k_substep)` calls use `0` or `1` |
| `k_substep` | `0..3` | every call site wraps the call in `for kss in range_constexpr(4)` |
| `dc` | `0..3` | `for dc in range_constexpr(D_CHUNKS)`, `D_CHUNKS = HEAD_DIM / D_CHUNK = 128 / 32 = 4` |
| `i_0` | `0, 1` | `i_0 = dc // 2`, so `dc=0,1 -> 0`; `dc=2,3 -> 1` |
| `i_1` | `0, 1` | `i_1 = dc % 2`, so it selects the low/high 32-bf16 group within each 64-bf16 D half |
| `read_pair` | `0, 1` | the code emits exactly two reads: `a = ds_read_tr_v4f16(lds_off_lo)` and `b = ds_read_tr_v4f16(lds_off_lo + OPUS_URV_I5_STRIDE)` |
| `e` | `0..3` | each `ds_read_tr16_b64` returns `v4f16`, i.e. `VEC_TR_V = 4` bf16 elements |

Expanded per-element form:

```text
i0  = dc / 2
i1  = dc % 2
i4a = k_substep
i4b = read_pair

V read elem addr =
    V_base[buf]
  + i0 * (8 * 544)
  + i1 * 32
  + (p0 * lane_hi + p1) * 544
  + (i4a * 2 + i4b) * 64
  + ((p2 * lane_lo + p3) * 4 + e)
```

Reversing this address through the FlyDSL V LDS write layout gives the logical V coordinate:

```text
V row = (i4a * 2 + i4b) * NUM_WAVES + (p0 * lane_hi + p1)
V dim = i0 * 64 + i1 * 32 + (p2 * lane_lo + p3) * 4 + e
```

Per-lane V LDS read expansion for `i0=0`, `i1=0`, `i4a=0`, `i4b=0`:

| lane_id range | p0 | p1 | p2 | p3 | V LDS elem base range | V LDS byte range | Logical V |
|---:|---:|---:|---:|---:|---|---|---|
| `0..3` | 0 | 0 | 0 | `0..3` | `V_base + {0,4,8,12}` | `0x000..0x01F` | row 0, dim `0..15` |
| `4..7` | 0 | 1 | 0 | `0..3` | `V_base + {544,548,552,556}` | `0x440..0x45F` | row 1, dim `0..15` |
| `8..11` | 0 | 2 | 0 | `0..3` | `V_base + {1088,1092,1096,1100}` | `0x880..0x89F` | row 2, dim `0..15` |
| `12..15` | 0 | 3 | 0 | `0..3` | `V_base + {1632,1636,1640,1644}` | `0xCC0..0xCDF` | row 3, dim `0..15` |
| `16..19` | 0 | 0 | 1 | `0..3` | `V_base + {16,20,24,28}` | `0x020..0x03F` | row 0, dim `16..31` |
| `20..23` | 0 | 1 | 1 | `0..3` | `V_base + {560,564,568,572}` | `0x460..0x47F` | row 1, dim `16..31` |
| `24..27` | 0 | 2 | 1 | `0..3` | `V_base + {1104,1108,1112,1116}` | `0x8A0..0x8BF` | row 2, dim `16..31` |
| `28..31` | 0 | 3 | 1 | `0..3` | `V_base + {1648,1652,1656,1660}` | `0xCE0..0xCFF` | row 3, dim `16..31` |
| `32..35` | 1 | 0 | 0 | `0..3` | `V_base + {2176,2180,2184,2188}` | `0x1100..0x111F` | row 4, dim `0..15` |
| `36..39` | 1 | 1 | 0 | `0..3` | `V_base + {2720,2724,2728,2732}` | `0x1540..0x155F` | row 5, dim `0..15` |
| `40..43` | 1 | 2 | 0 | `0..3` | `V_base + {3264,3268,3272,3276}` | `0x1980..0x199F` | row 6, dim `0..15` |
| `44..47` | 1 | 3 | 0 | `0..3` | `V_base + {3808,3812,3816,3820}` | `0x1DC0..0x1DDF` | row 7, dim `0..15` |
| `48..51` | 1 | 0 | 1 | `0..3` | `V_base + {2192,2196,2200,2204}` | `0x1120..0x113F` | row 4, dim `16..31` |
| `52..55` | 1 | 1 | 1 | `0..3` | `V_base + {2736,2740,2744,2748}` | `0x1560..0x157F` | row 5, dim `16..31` |
| `56..59` | 1 | 2 | 1 | `0..3` | `V_base + {3280,3284,3288,3292}` | `0x19A0..0x19BF` | row 6, dim `16..31` |
| `60..63` | 1 | 3 | 1 | `0..3` | `V_base + {3824,3828,3832,3836}` | `0x1DE0..0x1DFF` | row 7, dim `16..31` |

The remaining axes complete the full V tile:

| Axis | Values | Effect |
|---|---|---|
| `i0` | `0, 1` | selects D half `0..63` then `64..127` |
| `i1` | `0, 1` | selects the low/high 32 bf16 within that D half |
| `i4a` | `0..3` | advances V row group by `16` rows per step |
| `i4b` | `0, 1` | advances V row group by `8` rows within each `i4a` |
| `p0/p1` | lane-derived | selects rows `0..7` inside each 8-row group |
| `p2/p3` | lane-derived | selects 4-bf16 chunks inside each 32-bf16 D segment |

VGPR shape:

```text
_read_v_packs_for_k_substep(buf, k_substep)
  returns 4 x <8xbf16> per k_substep, one pack per O D chunk.

One GEMM2 cluster uses k_substep = 0..3:
  v_packs[4][4] = 16 x <8xbf16>
```

### 14.4 S and P: VGPR-only Score/Probability Layout

S never touches GM or LDS. It is the FP32 output of GEMM0:

```text
v_s_*_lo, v_s_*_hi: <16xf32> each

Logical tile per wave:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

S VGPR index formula:

```text
i_n  = 0..1        // 0 for lo score strip, 1 for hi score strip; hi adds +4 to the permuted N coordinate
rept = 0..3        // four 4-element groups per 16-lane score vector
j    = 0..3        // element inside each group

idx = i_n * 16 + rept * 4 + j

q_row = q_start + wave_id * 32 + lane_mod_32
k_col = kv_tile_start + i_n * 32 + lane_div_32 * 4 + rept * 8 + j
```

S/P variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `i_n` | `0, 1` | FlyDSL stores score as separate `lo` and `hi` vectors; `hi` is the second strip |
| `rept` | `0..3` | each `<16xf32>` score vector is described as 4 groups of 4 elements |
| `j` | `0..3` | element inside one 4-element group |
| `idx` | `0..31` | `i_n*16 + rept*4 + j` covers both `lo` and `hi` strips |
| `q_row` | `q_start .. q_start+255` | same Q row formula as Q load |
| `k_col` | `kv_tile_start .. kv_tile_start+63` | `i_n*32` selects `lo`/`hi` N strip; `lane_div_32*4` selects the lane group's 4-column offset; `rept*8+j` follows OPUS C++ `attn_mask_causal_tile` order |

Per-lane expansion for one wave:

| lane_id | q row inside wave | lane_div_32 | `v_s` K columns |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27` in `lo`; plus `+32` in `hi` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31` in `lo`; plus `+32` in `hi` |
| 1 | 1 | 0 | same K-column pattern as lane 0, for Q row 1 |
| 33 | 1 | 1 | same K-column pattern as lane 32, for Q row 1 |

After masking, max/subtract, and `exp2`, the probability fragment uses the same logical index layout:

```text
raw score:
  v_s_0_lo_raw / v_s_0_hi_raw
  v_s_1_lo_raw / v_s_1_hi_raw

after mask/max/sub/first-half-exp:
  v_s_*_lo_partial : <16xf32>
  v_s_*_hi_partial : <16xf32>

after finish:
  P packs: v_p_lo[0..1], v_p_hi[0..1], each <8xbf16>
```

Probability-pack range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `r` in score vectors | `0..15` | helpers `_wave_row_max`, `_sub_row_first_half_exp`, and `_finish_softmax_cast_p` iterate `for r in range_constexpr(16)` |
| `pks` | `0, 1` | `_finish_softmax_cast_p` iterates `for pks in range_constexpr(PV_K_STEPS)`, `PV_K_STEPS = K_SUB_N / PV_K_STEP = 32 / 16 = 2` |
| `p_base` | `0` or `8` | `p_base = pks * 8` |
| `v_p_lo[pks]` | covers `lo_partial[p_base .. p_base+7]` | each pack is built by `bf16_trunc_pack_v8(lo_slice)` |
| `v_p_hi[pks]` | covers `hi_full[p_base .. p_base+7]` | each pack is built by `bf16_trunc_pack_v8(hi_slice)` |

### 14.5 O: VGPR -> GM

O never uses LDS. GEMM2 accumulates output directly in VGPRs:

```text
o_accs[0..3] : 4 x <16xf32>

Logical tile per wave:
  O[32 x 128] += P[32 x 64] @ V[64 x 128]
```

`o_accs` index formula:

```text
dc   = 0..D_CHUNKS-1 = 0..3
rept = 0..3
j    = 0..3

idx = rept * 4 + j

o_row = q_row
d_row_rel = lane_div_32 * 4 + rept * 8 + j
o_dim = dc * D_CHUNK + d_row_rel
```

O variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `dc` | `0..3` | final store loop `for dc in range_constexpr(D_CHUNKS)`, `D_CHUNKS = HEAD_DIM / D_CHUNK = 128 / 32 = 4` |
| `rept` | `0..3` | `r // 4` for `r=0..15` |
| `j` | `0..3` | `r % 4` for `r=0..15` |
| `idx` / `r` | `0..15` | each `o_accs[dc]` is `<16xf32>` |
| `d_row_rel` | `0..31` | `lane_div_32*4 + rept*8 + j` spans two lane halves across a 32-wide D chunk |
| `o_dim` | `0..127` | `dc*32 + d_row_rel`, `dc=0..3` |

Final GM store:

```text
Before store:
o_norm = o_accs[dc] * (1 / l_row)
  o_bf16 = fp32_to_bf16(o_norm)

global_idx_q(o_row, o_dim) =
  (batch_idx * seq_len_v + o_row) * STRIDE_TOKEN_Q
  + q_head_idx * HEAD_DIM
  + o_dim
```

Final GM-store variable range derivation:

| Variable | Range / Formula | Derived from |
|---|---|---|
| `o_row` | `q_start .. q_start+255`, guarded by `q_in_bounds` | final store uses the same `q_row` computed during Q preload |
| `o_dim` | `0..127` | from `dc*D_CHUNK + d_row_rel`, see `O variable range derivation` above |
| `STRIDE_TOKEN_Q` | `NUM_HEADS_Q * HEAD_DIM` | kernel constant used by `global_idx_q(token_idx, col)` |
| `q_head_idx` | `(h_idx % NUM_HEADS_KV) * GQA_GROUP_SIZE + h_idx // NUM_HEADS_KV` | GQA mapping in the kernel before Q preload |

Per-lane store expansion for one fixed `dc`:

| lane_id | row inside wave | lane_div_32 | O dims stored inside one `dc` chunk |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31` |
| 1 | 1 | 0 | same D pattern as lane 0, for O row 1 |
| 33 | 1 | 1 | same D pattern as lane 32, for O row 1 |

The `dc` axis then covers the full D tile:

| Axis | Values | Effect |
|---|---|---|
| `dc` | `0..3` | selects D chunks `0..31`, `32..63`, `64..95`, `96..127` |
| `lane_div_32` | `0, 1` | selects low/high 4 bf16 inside each 8-wide D subgroup |
| `rept` | `0..3` | advances by `8` bf16 within one 32-wide D chunk |
| `j` | `0..3` | selects contiguous 4 bf16 inside the subgroup |

### 14.6 Summary: Where Each Tensor Lives

```text
Q:
  GM [B, S, H_Q, D]
    -> direct vector loads
    -> VGPR q_b_packs[0..7], pre-scaled
  No LDS.

K:
  GM [B, S, H_KV, D]
    -> buffer_load_dwordx4 ... lds
    -> LDS K0/K1, each tile 8320 bf16, 16B padding per line
    -> u_rk / ds_read_b128
    -> VGPR k operand packs for GEMM0

V:
  GM [B, S, H_KV, D]
    -> buffer_load_dwordx4 ... lds
    -> LDS V0/V1, each tile 8704 bf16, 64B padding per line
    -> u_rv / ds_read_tr16_b64
    -> VGPR v_packs[4][4] for GEMM2

S:
  VGPR only.
  v_s_0/v_s_1 lo+hi fp32 score accumulators.

P:
  VGPR only.
  v_p_lo/v_p_hi bf16 packs, same logical index layout as S after softmax.

O:
  VGPR o_accs[0..3] fp32 accumulator
    -> normalize by l_row
    -> cast bf16
    -> GM [B, S, H_Q, D]
  No LDS.
```

---

## Quick Reference

**VGPR data flow (per wave, after P1–P7):**

| Symbol | Source line | Type | Use |
|---|---|---|---|
| `q_b_packs[0..7]` | lines 631-655 | 8 × `<8 × bf16>` | GEMM0 B, pre-scaled by `(1/√D)·log2e` (P2), resident across all KV iters |
| `v_s_0_lo_raw`/`_raw_b`/`_e5`, `v_s_0_hi_*` | line 957 (prologue, `_raw`), 1140 (main loop, `_raw_b`), 1360 (epilogue, `_e5`) | `<16 × f32>` × 2 | GEMM0 D, rebuilt per `j-1` / `max-2` half-iter |
| `v_s_1_lo_raw`/`_e`/`_e9`, `v_s_1_hi_*` | line 1023 (main loop, `_raw`), 1269 (epilogue, `_e`), 1446 (epilogue, `_e9`) | `<16 × f32>` × 2 | GEMM0 D for `j-2` / epilogue half-iter |
| `s_lo_pro/a/b/e1/e5/e9[0..15]`, `s_hi_*[0..15]` | various clusters | 32 × scalar f32 | per-element extracted scores |
| `m_row`, `l_row` | loop-carried | scalar f32 × 2 | online softmax state (log2-space, P2) |
| `lo_partial[0..15]`, `hi_partial[0..15]` | lines 846-855 | 32 × scalar f32 | per-element softmax weights (lo = post-exp2, hi = pre-exp2 carry) |
| `v_p_lo_*[0..1]`, `v_p_hi_*[0..1]` | lines 862-883 plus cluster call sites | 4 × `<8 × bf16>` | GEMM2 B (bf16-packed P) |
| `v_packs_a/b/e3/e7/e11/e13[0..3][0..3]` | lines 1051, 1165, 1293, 1384, 1469, 1541 | 16 × `<8 × bf16>` per V hoist site | GEMM2 A, hoisted into VGPRs before consumer cluster (P5/P6) |
| `o_accs[0..3]` | loop-carried | 4 × `<16 × f32>` | output accumulator (4 D-chunks × 16 f32) |
| `v_s_0_lo_partial`, `v_s_0_hi_partial` | loop-carried (yield) | `<16 × f32>` × 2 | half-finished softmax carried across iter |

**LDS region quick reference (post P7, OPUS interleaved K0/V0/K1/V1 with line padding):**

| Region | Offset (bf16) | Size (bf16) | Line stride | Owner |
|---|---:|---:|---:|---|
| K buffer 0 (`s_k[0]`, `u_sk` pad16) |     0 | 8 320 | 520 | DMA via `u_gk`; GEMM0 reads via `u_rk` |
| V buffer 0 (`s_v[0]`, `u_sv` pad64) | 8 320 | 8 704 | 544 | DMA via `u_gv`; GEMM2 reads via `u_rv` (`ds_read_tr16_b64`) |
| K buffer 1 (`s_k[1]`, `u_sk` pad16) | 17 024 | 8 320 | 520 | DMA via `u_gk`; GEMM0 reads via `u_rk` |
| V buffer 1 (`s_v[1]`, `u_sv` pad64) | 25 344 | 8 704 | 544 | DMA via `u_gv`; GEMM2 reads via `u_rv` (`ds_read_tr16_b64`) |

Total LDS: **34 048 bf16 = 68 096 B**. The interleaved `[K0][V0][K1][V1]` ordering and per-line padding match the OPUS C++ template's `smem<D_ATTN> s_k[2]` / `smem<D_ATTN> s_v[2]` exactly (see `gqa_d128_kernel_template.hpp` lines 326-333 and `make_layout_sk_sv` lines 102-118).

**Stagger state (P5):**

| Symbol | Source line | Use |
|---|---|---|
| `_wave_id_uni_i32` | line 348 | SGPR-resident wave id, `readfirstlane(tid/64)` |
| `_stagger_i32` | line 352 | `_wave_id_uni_i32 / 4`, equals 0 for warps 0-3, 1 for warps 4-7 |
| `_stagger_extra_barrier_if_one` | line 463 | Emits `if (stagger == 1) s_barrier;` (prologue) |
| `_stagger_extra_barrier_if_zero` | line 489 | Emits `if (stagger == 0) s_barrier;` (close-out) |

**File locations:**

- Kernel source: [`FlyDSL/kernels/flash_attn_opus.py`](kernels/flash_attn_opus.py)
- C++ reference: [`FlyDSL/opus_attn/gqa_d128_kernel_template.hpp`](opus_attn/gqa_d128_kernel_template.hpp)
- Dispatcher: [`FlyDSL/kernels/flash_attn_func.py`](kernels/flash_attn_func.py) → `_wrap_with_opus`
- Differences doc: [`FLASH_ATTN_OPUS_vs_CPP_Differences.md`](FLASH_ATTN_OPUS_vs_CPP_Differences.md)
- Test: [`FlyDSL/tests/kernels/test_flash_opus_attn.py`](tests/kernels/test_flash_opus_attn.py)

**Commit history (P1–P7):**

```bash
git log --oneline kernels/flash_attn_opus.py
# 2d07de2 flash_attn_opus: port OPUS permuted K/V LDS layout (atomic)    ← P7
# 55a135c flash_attn_opus: flip OPUS_ENABLE_STAGGER default to ON
# 598981b flash_attn_opus: hoist V LDS reads to fix P5 stagger correctness
# b62c600 flash_attn_opus: P5 - stagger mechanism (wired in, default OFF)
# e14cb5e flash_attn_opus: P4 - inline-asm causal mask + 8 register anchors
# 39feb59 flash_attn_opus: P3 - sched_group_barrier_pairs/_exp_pairs discipline
# 558b77c flash_attn_opus: P2 - in-flight Q scaling (log2-space softmax)
# ad1fcaf flash_attn_opus: P1 - align structure with gqa_d128_kernel_template.hpp
# 93b5cd7 flash_attn: add OPUS-style fast path (opt-in)
```
