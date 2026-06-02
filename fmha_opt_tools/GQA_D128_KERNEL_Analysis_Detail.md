# OPUS GQA D=128 Causal Kernel Detailed Analysis
# BF16 Grouped-Query Flash Attention on AMD MI355X (GFX950 / CDNA4)

**Kernel file:** `FlyDSL/opus_attn/gqa_d128_kernel_causal.cc`
  → instantiates `gqa_d128_kernel<opus_gqa_traits<32, 64, 128, 8, true>>` from
  `FlyDSL/opus_attn/gqa_d128_kernel_template.hpp`
**Host launcher:** `FlyDSL/opus_attn/gqa_host.cc` → `gqa_launch<Traits>`
**Config:** BF16; GQA-FMHA; FWD; D=128; 8W; Q-tile=32/wave (M=256/workgroup); KV-tile=64; MFMA 32×32×16 bf16; CAUSAL=true; gfx950
**Target GPU:** AMD Instinct MI355X (GFX950, CDNA4 ISA — requires `ds_read_tr16_b64` / `__builtin_amdgcn_permlane32_swap`)
**Launch bounds:** `__launch_bounds__(BLOCK_SIZE=512, 2)` (2 workgroups per CU)

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Algorithm](#11-algorithm)
   - [1.2 Key Parameters](#12-key-parameters)
   - [1.3 MFMA Instruction Details](#13-mfma-instruction-details)
   - [1.4 File Layout](#14-file-layout)
2. [Workgroup and Thread Organization](#2-workgroup-and-thread-organization)
   - [2.1 3D Grid Launch](#21-3d-grid-launch)
   - [2.2 Block Structure (8 Warps)](#22-block-structure-8-warps)
   - [2.3 Per-Warp Q Tile Mapping](#23-per-warp-q-tile-mapping)
   - [2.4 Stagger: Dual-Group Phase Shift](#24-stagger-dual-group-phase-shift)
3. [Kernel Arguments](#3-kernel-arguments)
   - [3.1 `opus_gqa_kargs` Struct](#31-opus_gqa_kargs-struct)
   - [3.2 GQA Head Mapping](#32-gqa-head-mapping)
4. [LDS Layout (~68 KB)](#4-lds-layout-68-kb)
   - [4.1 Overview](#41-overview)
   - [4.2 Tile Element Counts](#42-tile-element-counts)
   - [4.3 K Tile (`smem_padding_16B`)](#43-k-tile-smem_padding_16b)
   - [4.4 V Tile (`smem_padding_64B`)](#44-v-tile-smem_padding_64b)
5. [Layout System (`opus::make_layout`)](#5-layout-system-opusmake_layout)
   - [5.1 Q Global Load (`u_q`)](#51-q-global-load-u_q)
   - [5.2 K/V Global Load (`u_gk`, `u_gv`)](#52-kv-global-load-u_gk-u_gv)
   - [5.3 K/V Shared Write (`u_sk`, `u_sv`)](#53-kv-shared-write-u_sk-u_sv)
   - [5.4 K Register Read (`u_rk`)](#54-k-register-read-u_rk)
   - [5.5 V Register Read via HW Transpose (`u_rv`)](#55-v-register-read-via-hw-transpose-u_rv)
   - [5.6 Full GM/LDS/VGPR Layout Maps](#56-full-gmldsvgpr-layout-maps)
6. [Q Preload with In-Flight Scaling](#6-q-preload-with-in-flight-scaling)
7. [GEMM0: S = Q @ K^T](#7-gemm0-s--q--kt)
   - [7.1 Per-Warp Tile Decomposition](#71-per-warp-tile-decomposition)
   - [7.2 `make_tiled_mma` + `mfma_adaptor_swap_ab`](#72-make_tiled_mma--mfma_adaptor_swap_ab)
   - [7.3 Score Output `v_s[0]`, `v_s[1]`](#73-score-output-v_s0-v_s1)
8. [Causal Mask via Inline ASM](#8-causal-mask-via-inline-asm)
   - [8.1 `attn_mask_vec2_imm` Primitive](#81-attn_mask_vec2_imm-primitive)
   - [8.2 `attn_mask_causal_tile` Driver](#82-attn_mask_causal_tile-driver)
9. [Online Softmax with Lazy Rescaling](#9-online-softmax-with-lazy-rescaling)
   - [9.1 Algorithm Recap (Pre-Scaled Q)](#91-algorithm-recap-pre-scaled-q)
   - [9.2 `attn_row_max` (`permlane32_swap`)](#92-attn_row_max-permlane32_swap)
   - [9.3 Lazy Rescale via `ballot_w64 == read_exec`](#93-lazy-rescale-via-ballot_w64--read_exec)
   - [9.4 Half-Slice `attn_exp2_slice` Strategy](#94-half-slice-attn_exp2_slice-strategy)
   - [9.5 P Conversion + Register Anchor](#95-p-conversion--register-anchor)
10. [GEMM2: O = P @ V via `tr_load`](#10-gemm2-o--p--v-via-tr_load)
    - [10.1 Per-Warp Tile Decomposition](#101-per-warp-tile-decomposition)
    - [10.2 `tr_load` Hardware Transpose](#102-tr_load-hardware-transpose)
    - [10.3 `mma1.step_k(K, ...)` Sub-Step Issuing](#103-mma1step_kk--sub-step-issuing)
11. [Output Finalization](#11-output-finalization)
12. [`sched_group_barrier` Discipline](#12-sched_group_barrier-discipline)
    - [12.1 Masks and Counts](#121-masks-and-counts)
    - [12.2 `sched_barrier_pairs` / `sched_barrier_exp_pairs` Helpers](#122-sched_barrier_pairs--sched_barrier_exp_pairs-helpers)
    - [12.3 Group Numbering (1..10)](#123-group-numbering-110)
13. [Main Loop Pipeline (8 Clusters × 2 KV-Tiles per Iter)](#13-main-loop-pipeline-8-clusters--2-kv-tiles-per-iter)
    - [13.1 High-Level Structure](#131-high-level-structure)
    - [13.2 Cluster-by-Cluster Breakdown](#132-cluster-by-cluster-breakdown)
    - [13.3 Synchronization Points](#133-synchronization-points)
    - [13.4 Buffer Ping-Pong Summary](#134-buffer-ping-pong-summary)
14. [Prologue (1 Tile + GEMM0 Setup)](#14-prologue-1-tile--gemm0-setup)
15. [Epilogue (14 Clusters Drain)](#15-epilogue-14-clusters-drain)
16. [Stagger: Wave-Group Phase Shift](#16-stagger-wave-group-phase-shift)
17. [Performance and Comparison](#17-performance-and-comparison)

---

## 1. Overview

### 1.1 Algorithm

This kernel implements **Grouped-Query Flash Attention forward** with online softmax for BF16 inputs on AMD MI355X (gfx950 / CDNA4). It is the **performance-target reference** for the FlyDSL `flash_attn_opus` port and achieves ~1131 TFLOPS (causal, B=16 S=8192 H=64 D=128).

```
For each (batch b, query head h, q_block_idx) in (B × H × ceil(N / 256)):
  Prologue:
    Load K[0]   → s_k[0]    (async)
    s_waitcnt(0), s_barrier
    Load Q (per-warp 32×128 bf16)
    Q.f32 *= (1/√D) × log2e               // in-flight scaling (key optimization)
    Q.bf16 = cast(Q.f32)
    Load K[1]   → s_k[1]    (async)
    Load V[0]   → s_v[0]    (async)
    v_k = ds_read(s_k[0])                  // first K → registers
    s_waitcnt lgkmcnt(0), vmcnt(v_buffer_load_insts)
    if (stagger): s_barrier                 // group B parks here
    v_s[0] = mma0(v_q, v_k)                 // GEMM0 tile 0
    if (CAUSAL && masked): attn_mask_causal_tile(v_s[0], ...)
    m_row    = attn_row_max(v_s[0])
    v_s[0]  -= m_row
    v_s[0][0:half]   = exp2(...)            // first half of exp
    s_barrier
    Load K[2]   → s_k[0]   (async)

  Main loop (for j = 3; j < max_num_tiles − 1; j += 2):
    // 8 clusters per iter, each separated by s_barrier; 2 KV tiles advanced per iter
    [Cluster 0] Load V[j-2]→s_v[1]; v_k=ds_read(s_k[1]=K[j-2]); wait+barrier
    [Cluster 1] v_s[1] = mma0(v_q, v_k);          // GEMM0 tile j-2
                v_s[0][half:] = exp2;             // finish exp of tile j-3
                l_row += sum(v_s[0])
                v_p = cast<bf16>(v_s[0])
                sched_barrier_pairs(10,5,1) etc.
                s_barrier
    [Cluster 2] Load K[j]→s_k[1]; v_v=tr_load(s_v[0]); wait+barrier
    [Cluster 3] s_setprio(1)
                v_o += step_k<0>(v_p=P[j-3], v_v=V[j-3])
                row_max = attn_row_max(v_s[1])
                LAZY-RESCALE check (ballot==read_exec)
                  if all_below: row_max = m_row     (no rescale)
                  else:         v_o *= exp2(m_row−row_max); l_row*=...; m_row←row_max
                v_o += step_k<1>(v_p, v_v)
                v_o += step_k<2>(v_p, v_v)
                v_o += step_k<3>(v_p, v_v)
                v_s[1] -= row_max
                v_s[1][0:half] = exp2
                s_setprio(0); s_barrier
    [Cluster 4-7] same pattern, swapped roles (v_s[0]↔v_s[1], s_k[0]↔s_k[1], s_v[0]↔s_v[1])

  Epilogue (14 clusters: 0..13):
    Drain the trailing KV tiles using GEMM0/GEMM1 alternation
    Final v_o += step_k<...>(v_p, v_v)
    l_inv = 1 / l_row
    v_o *= l_inv
    if (!stagger): s_barrier                       // resync group A with group B
    store(g_o, cast<bf16>(v_o))
```

### 1.2 Key Parameters

From the trait instantiation `opus_gqa_traits<32, 64, 128, 8, true>`:

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Q tile per warp | `Q_TILE_SIZE` | 32 | 32 Q rows per warp × 8 warps = 256 rows/workgroup |
| KV tile size | `KV_TILE_SIZE` | 64 | 64 KV columns per tile; one `j += 2` main-loop iter advances two tiles (=128 KV positions) |
| Head dimension | `D_TILE_SIZE` | 128 | fixed; no D slicing (`SLICE_D == D`) |
| Warps per workgroup | `NUM_WARPS` | 8 | M-direction wave parallelism |
| Causal mode | `CAUSAL` | true | enables causal mask |
| Wave size | `WARP_SIZE` | 64 | AMD wavefront |
| Block size | `BLOCK_SIZE` | 512 | `NUM_WARPS × WARP_SIZE` |
| Data type | `D_ATTN` | `bf16_t` | I/O bf16 |
| Accumulator | `D_ACC` | `float` | softmax + MFMA accum |
| MFMA tile | `W_M × W_N × W_K` | 32 × 32 × 16 | bf16 32x32x16 |
| Waves M-dir | `T_M` | 8 | along Q rows |
| Waves N-dir | `T_N` | 1 | along KV cols |
| Waves K-dir | `T_K` | 1 | along D |
| GEMM0 reps (M, N, K) | `GEMM0_E_*` | (1, 2, 8) | per-warp MFMA chain shape |
| GEMM1 reps (M, N, K) | `GEMM1_E_*` | (1, 4, 4) | per-warp MFMA chain shape |
| MFMAs per warp/KV tile | — | 16 (GEMM0) + 16 (GEMM2) | 32 total per KV tile; steady-state `j += 2` iter = 64 |
| Vec width (Q load) | `VEC_Q` | 8 | bf16 per `load` instruction |
| Vec width (K/V load) | `VEC_KV` | 8 | bf16 per `async_load` lane |
| Vec width (V transpose) | `VEC_TR_V` | 4 | bf16 per `ds_read_tr16_b64` |
| Vec width (O store) | `VEC_O` | 4 | bf16 per `store` |
| K shared padding | `smem_k_padding` | 16 B (8 bf16) | `smem_padding_16B` |
| V shared padding | `smem_v_padding` | 64 B (32 bf16) | `smem_padding_64B` |
| LDS K tile elements | `smem_k_tile_elems` | 8 320 bf16 | `smem_n_rpt × smem_d_rpt × (linear_wave + pad)` |
| LDS V tile elements | `smem_v_tile_elems` | 8 704 bf16 | (V uses larger padding) |
| LDS K+V (1 buf each) | `smem_buffer_elems` | 17 024 bf16 | |
| LDS total (×2 buffers) | `smem_size_bytes()` | 68 096 B (~66.5 KB) | 2 K-buffers + 2 V-buffers |
| Lazy-rescale threshold | `RESCALE_THRESHOLD` | 8.0 | in scaled (log2) space |
| Q in-flight scale | `temperature_scale` | `(1/√D) × log2e` | folded into Q at prologue |

### 1.3 MFMA Instruction Details

The kernel uses **`v_mfma_f32_32x32x16_bf16`** (gfx950 / CDNA4):

- **Computation:** `D[32×32, FP32] = A[32×16, BF16] × B[16×32, BF16] + C[32×32, FP32]`
- **Latency:** ~16 cycles issue, ~32 cycles total
- **Operand vec sizes (from `vector_traits`):**
  - `v_q`: `vector<bf16, q_len>` where `q_len = GEMM0_E_M × GEMM0_E_K × (W_M×W_K)/WARP_SIZE = 1×8×8 = 64`
  - `v_s`: `vector<f32, s_len>` where `s_len = GEMM0_E_M × GEMM0_E_N × (W_M×W_N)/WARP_SIZE = 1×2×16 = 32`
  - `v_p`: `vector<bf16, p_len>` (same layout as v_s after cast)
  - `v_o`: `vector<f32, o_len>` where `o_len = GEMM1_E_M × GEMM1_E_N × (W_M×W_N)/WARP_SIZE = 1×4×16 = 64`

The `make_tiled_mma<...>` factory with `mfma_adaptor_swap_ab{}` swaps A/B operands so that the row-major Q × column-major K^T contraction maps cleanly onto the MFMA's column-major A × row-major B layout. The `mma1.step_k(K_idx, ...)` method exposes one of the `GEMM1_E_K=4` sub-step MFMA groups so the lazy-rescale check can happen between sub-steps.

### 1.4 File Layout

```
FlyDSL/opus_attn/
  gqa_defs.h                        ← shared kargs + opus_gqa_traits<...>
  gqa_d128_kernel_template.hpp      ← <THIS DOC> 756 lines: full device kernel
  gqa_d128_kernel_causal.cc         ← instantiate <32,64,128,8,true>
  gqa_d128_kernel_noncausal.cc      ← instantiate <32,64,128,8,false>
  gqa_d512_kernel_template.hpp      ← D=512 variant (16x16x32 MFMA)
  gqa_d512_kernel_causal.cc / noncausal.cc
  gqa_host.cc                       ← benchmark harness, CPU reference, main()
  gqa_python_api.cc                 ← pybind11 wrapper for test_flash_opus_attn.py
```

The `_causal.cc` and `_noncausal.cc` files are thin wrappers that exist only to give the C++ build system separate translation units for each variant (so device-stub generation works across both compilation modes):

```cpp
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gqa_d128_kernel(opus_gqa_kargs kargs) {}
template __global__ void gqa_d128_kernel<opus_gqa_traits<32, 64, 128, 8, true>>(opus_gqa_kargs);
#else
#include "gqa_d128_kernel_template.hpp"
template __global__ void gqa_d128_kernel<opus_gqa_traits<32, 64, 128, 8, true>>(opus_gqa_kargs);
#endif
```

The host TU sees only an empty stub; the device TU includes the full kernel.

---

## 2. Workgroup and Thread Organization

### 2.1 3D Grid Launch

From `gqa_host.cc::run()`:

```cpp
const int num_q_tiles  = ceil_div(N, Q_TILE_SIZE);              // ceil(N / 32)
const int num_q_blocks = ceil_div(num_q_tiles, NUM_WARPS);      // ceil(num_q_tiles / 8)
dim3 grid (H, num_q_blocks, B);
dim3 block(BLOCK_SIZE);                                          // = 512
```

| Grid axis | Index in kernel | Mapping |
|---|---|---|
| x | `workgroup_x = block_id_x()` | Q-head id (after GQA de-interleave) |
| y | `q_block_idx = block_id_y()` | Q-tile id along sequence (each tile = 256 rows) |
| z | `b           = block_id_z()` | batch id |

### 2.2 Block Structure (8 Warps)

```cpp
const int warp_id   = __builtin_amdgcn_readfirstlane(thread_id_x() / WARP_SIZE);  // 0..7
const int lane_id   = thread_id_x() % WARP_SIZE;                                  // 0..63
const int stagger   = warp_id / 4;                                                // 0 or 1
```

The `readfirstlane` ensures `warp_id` is held in an SGPR (uniform across the warp), letting the compiler use scalar address arithmetic instead of per-lane vectorized arithmetic when the kernel does `warp_id * stride`.

### 2.3 Per-Warp Q Tile Mapping

Unlike kernels where every warp processes the same Q rows but different KV columns, OPUS GQA splits the Q-row dimension across warps:

```
256-row workgroup Q tile (q_block_size = NUM_WARPS × Q_TILE_SIZE = 8 × 32):

  warp 0:  rows q_block_start +   0..31    Q_TILE_SIZE = 32 rows
  warp 1:  rows q_block_start +  32..63
  warp 2:  rows q_block_start +  64..95
  warp 3:  rows q_block_start +  96..127
  warp 4:  rows q_block_start + 128..159
  warp 5:  rows q_block_start + 160..191
  warp 6:  rows q_block_start + 192..223
  warp 7:  rows q_block_start + 224..255

Per-warp:
  q_start_pos = q_block_start + warp_id × 32       // starting Q index for causal mask
  v_q         = load(g_q, u_q)                     // 64 bf16 per lane (32×128 / 64 lanes × 8 vec)
```

**Cooperative phases** (all 8 warps): K and V async loads from global to LDS.
**Independent phases** (each warp): Q load, GEMM0, causal mask, softmax, GEMM1, output store.

The independence of the softmax phase is what allows the per-warp Q-tile split — different Q rows have independent statistics.

### 2.4 Stagger: Dual-Group Phase Shift

```cpp
const int stagger = warp_id / 4;   // group A (warps 0-3) → 0,  group B (warps 4-7) → 1
```

The stagger flag selects whether a warp executes an **extra** `s_barrier` after the prologue (line 415-418) and **skips** the final pre-store `s_barrier` (line 748-750):

```cpp
// Prologue tail:
if (stagger) {                       // only group B
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
}

// ... main loop, identical for both groups, hits the same N s_barriers ...

// Just before store:
if (!stagger) {                      // only group A
    __builtin_amdgcn_s_barrier();
}
```

**Effect:** Because `s_barrier` is matched *globally by ordinal arrival count* across the workgroup (not by source location), this offset pushes group A exactly **one cluster ahead** of group B throughout the steady state of the main loop:
- When A is in cluster K's GEMM/VALU work, B is in cluster K−1's memory wait
- When A finishes cluster K and arrives at the cluster boundary `s_barrier`, B's earlier cluster K−1 boundary `s_barrier` is the *same* global barrier (ordinal match)
- Both proceed: A enters cluster K+1, B enters cluster K
- At kernel end the asymmetric pre-store barrier on group A only re-aligns the two groups before they both perform global stores in lockstep

This naturally **time-shares the MFMA and VMEM/LDS units** between the two groups, similarly to the MI308X ASM kernel's `label_03DA` / `label_07AA` split, but here both groups execute the *same instruction sequence* — the phase shift is purely via barrier offset.

---

## 3. Kernel Arguments

### 3.1 `opus_gqa_kargs` Struct

```cpp
struct opus_gqa_kargs {
    const void* __restrict__ ptr_q;   // [B, N, H,    D]   bf16
    const void* __restrict__ ptr_k;   // [B, N, H_KV, D]   bf16
    const void* __restrict__ ptr_v;   // [B, N, H_KV, D]   bf16
    void*       __restrict__ ptr_o;   // [B, N, H,    D]   bf16
    int B, N, H, H_KV, D;             // runtime shapes
    int stride_q_b, stride_q_n, stride_q_h;
    int stride_kv_b, stride_kv_n, stride_kv_h;
};
```

All strides are in **bf16 element count**, not bytes. The host fills them as:

```cpp
kargs.stride_q_b  = N * H    * D;     // bytes / sizeof(bf16)
kargs.stride_q_n  = H    * D;
kargs.stride_q_h  = D;
kargs.stride_kv_b = N * H_KV * D;
kargs.stride_kv_n = H_KV * D;
kargs.stride_kv_h = D;
```

This is contiguous BSHD layout, fastest-varying axis = D.

### 3.2 GQA Head Mapping

```cpp
const int group_size = kargs.H / kargs.H_KV;
const int h          = (workgroup_x % kargs.H_KV) * group_size + (workgroup_x / kargs.H_KV);
const int h_kv       = h / group_size;     // == workgroup_x % kargs.H_KV
const int qo_gmem_offset = b * kargs.stride_q_b
                         + q_block_start * kargs.stride_q_n
                         + h * kargs.stride_q_h;
const int kv_gmem_offset = b * kargs.stride_kv_b
                         + h_kv * kargs.stride_kv_h;
```

When `H == H_KV` (vanilla MHA) `group_size == 1` and `h == workgroup_x`. The re-shuffle `h = (workgroup_x % H_KV) * group_size + (workgroup_x / H_KV)` interleaves Q heads sharing the same KV head into **adjacent workgroups along the grid x axis**, which improves L2 reuse of K/V across the GQA group.

`make_gmem(...)` returns a Q/K/V/O tensor view base-pointer offset to this workgroup's slice; per-element addressing is then driven by the layout system.

---

## 4. LDS Layout (~68 KB)

### 4.1 Overview

```
__shared__ char smem_buf[T::smem_size_bytes()];          // = 68096 bytes

  s_k[0]:  smem_buf + 0                                   K buffer 0  (8 320 bf16)
  s_v[0]:  smem_buf + smem_k_tile_elems                   V buffer 0  (8 704 bf16)
  s_k[1]:  smem_buf + smem_buffer_elems                   K buffer 1  (8 320 bf16)
  s_v[1]:  smem_buf + smem_buffer_elems
                    + smem_k_tile_elems                   V buffer 1  (8 704 bf16)

  Total: 2 × (8320 + 8704) × 2 B = 68 096 B  (~66.5 KB)
```

K and V are **both double-buffered** (`s_k[2]`, `s_v[2]`). The current FlyDSL `flash_attn_opus.py` port now replicates this OPUS interleaved `[K0][V0][K1][V1]` LDS layout with two K buffers and two V buffers.

### 4.2 Tile Element Counts

The padded element count comes from the trait calculation:

```cpp
D_128B_SIZE       = 128 / sizeof(bf16) = 64
VEC_KV            = 16 / sizeof(bf16)  = 8
smem_linear_wave  = WARP_SIZE × 16 / sizeof(bf16) = 64 × 8 = 512    // bf16 per wave per "line"
smem_n_per_wave   = smem_linear_wave / D_128B_SIZE = 8              // 8 KV rows per wave per line
smem_n_rpt        = KV_TILE_SIZE / smem_n_per_wave = 64 / 8 = 8     // 8 lines along N
smem_d_rpt        = D_TILE_SIZE / D_128B_SIZE     = 128 / 64 = 2    // 2 lines along D

smem_k_padding    = smem_padding_16B = 16 / sizeof(bf16) =  8 bf16
smem_v_padding    = smem_padding_64B = 64 / sizeof(bf16) = 32 bf16

smem_k_tile_elems = smem_n_rpt × smem_d_rpt × (smem_linear_wave + smem_k_padding)
                  = 8 × 2 × (512 + 8)  = 16 × 520 = 8 320 bf16
smem_v_tile_elems = 8 × 2 × (512 + 32) = 16 × 544 = 8 704 bf16
```

### 4.3 K Tile (`smem_padding_16B`)

Each line is `512 + 8 = 520` bf16. There are 16 lines per K tile (`smem_n_rpt × smem_d_rpt = 8 × 2`). The **8 bf16 padding** (= 1 bank of 32 B / 4 banks of 4 B) breaks bank conflicts during MFMA `v_k = load<VEC_KV>(s_k[buf], u_rk)` reads — when 8 lanes simultaneously target the same row offset, the padding shifts each lane's bank by 1 step.

The layout produced by `make_layout_sk_sv<T, smem_padding_16B>` is non-trivial — it interleaves D-direction lines (`smem_d_rpt = 2`) with N-direction lines (`smem_n_rpt = 8`) per warp, producing a "diagonal stripe" pattern that matches the way `make_layout_gk_gv` distributes lanes across the K matrix's `[N=64, D=128]` plane.

### 4.4 V Tile (`smem_padding_64B`)

V uses a **larger padding** (32 bf16 vs 8 bf16). The reason: `tr_load` (which compiles to `ds_read_tr16_b64`) performs a hardware transpose across 16-lane subgroups during the LDS read, and the access stride from the V layout differs from K's. The 32-bf16 (= 64 B = 4 banks of 16 B) padding ensures all 64 lanes simultaneously hit distinct banks even with the transpose's swizzled access pattern.

This larger padding is why `smem_v_tile_elems > smem_k_tile_elems`.

---

## 5. Layout System (`opus::make_layout`)

The OPUS layout system is a compile-time CuTe-style descriptor: every layout encodes a **shape**, a **stride function**, and a **per-lane coordinate** as nested tuples of `y_dim` (compile-time / static) and `p_dim` (per-thread / dynamic).

For this kernel, **seven** layouts are constructed once at kernel entry (lines 349-355):

```cpp
auto u_q  = make_layout_q<T>      (warp_id, lane_id, kargs.stride_q_n);
auto u_gk = make_layout_gk_gv<T>  (warp_id, lane_id, kargs.stride_kv_n);
auto u_sk = make_layout_sk_sv<T, smem_padding_16B>(warp_id);
auto u_rk = make_layout_rk<T>     (lane_id);
auto u_gv = make_layout_gk_gv<T>  (warp_id, lane_id, kargs.stride_kv_n);   // identical to u_gk
auto u_sv = make_layout_sk_sv<T, smem_padding_64B>(warp_id);
auto u_rv = make_layout_rv<T>     (lane_id);
```

These are used as the second argument to `load<...>`, `async_load<...>`, `tr_load<...>`, and `store<...>`. The first three layouts define the **iteration axes** (`y_dim` → loop indices), the lane coord defines the **per-thread starting offset** (`p_dim`), and the kernel iterates implicitly through the y-axes when one of the high-level transfer functions is called.

### 5.1 Q Global Load (`u_q`)

`make_layout_q<T>` (lines 33-51) defines a `[GEMM0_E_M=1, T_M=8, W_M=32, GEMM0_E_K=8, WARP_SIZE/W_M=2, VEC_Q=8]` shape. The lane coord `(warp_id, lane_id % W_M, lane_id / W_M)` maps each lane to its starting Q row + column in the per-warp 32×128 tile. The kernel issues:

```cpp
v_q = load<T::VEC_Q>(g_q, u_q);     // → v_q: vector<bf16, 64> per lane
```

This unrolls to 8 `buffer_load` instructions per lane (`Q_TILE_SIZE × D_TILE_SIZE × NUM_WARPS / BLOCK_SIZE / VEC_Q = 32 × 128 × 8 / 512 / 8 = 8`), reading 8 bf16 each.

### 5.2 K/V Global Load (`u_gk`, `u_gv`)

`make_layout_gk_gv<T>` (lines 76-99) defines the cooperative tile load:

```cpp
threads_d            = D_128B_SIZE / VEC_KV         = 64 / 8 = 8     // lanes covering one D=128 row
threads_n_per_block  = BLOCK_SIZE / threads_d       = 512 / 8 = 64   // distinct N rows per block
threads_n_per_wave   = WARP_SIZE / threads_d        =  64 / 8 = 8    // N rows per wave

shape = [smem_d_rpt=2, KV_TILE_SIZE/threads_n_per_block=1, threads_n_per_wave=8,
         NUM_WARPS=8,  threads_d=8, VEC_KV=8]
```

Each `async_load<VEC_KV>` lowers to **`buffer_load_dwordx4_lds`** (16 B/lane direct to LDS, bypassing VGPRs). One call issues `2 × 1 = 2` VMEM instructions per wave (`smem_d_rpt × KV_TILE_SIZE/threads_n_per_block`). Across the 8 waves in the workgroup, that is 16 dynamic wave-instruction instances and `2 × 512 = 1024` per-lane 16-byte transfers for the full 64×128 K/V tile. The per-wave instruction count is exposed as `T::k_buffer_load_insts` and `T::v_buffer_load_insts`.

### 5.3 K/V Shared Write (`u_sk`, `u_sv`)

`make_layout_sk_sv` (lines 102-118) is parameterized by `smem_padding`:
- `u_sk = make_layout_sk_sv<T, smem_padding_16B>` (8-bf16 pad)
- `u_sv = make_layout_sk_sv<T, smem_padding_64B>` (32-bf16 pad)

The shape is `[smem_d_rpt, smem_n_rpt/NUM_WARPS, NUM_WARPS, VEC_KV]`, and the LDS line stride is `smem_linear_wave + smem_padding` (= 520 for K, 544 for V).

The async_load implicitly performs the (global-coord → smem-coord) translation by combining `u_gk` (read coord) with `u_sk` (write coord) — the result is a per-lane LDS write address that already includes the padding.

### 5.4 K Register Read (`u_rk`)

`make_layout_rk<T>` (lines 121-148) defines how `load<VEC_KV>(s_k[buf], u_rk)` rearranges the LDS layout into MFMA-compatible register packing:

```cpp
n_per_wave = WARP_SIZE / (D_128B_SIZE / VEC_KV) = 64 / 8 = 8
n_grp      = n_per_wave / (W_N / NUM_WARPS)     = 8 / (32/8) = 2

shape = [GEMM0_E_N/n_grp = 1, NUM_WARPS = 8, n_grp = 2,
         W_N/NUM_WARPS = 4, smem_d_rpt = 2, GEMM0_E_K/smem_d_rpt = 4,
         WARP_SIZE/W_N = 2, VEC_KV = 8]
```

After this read, `v_k` is a `vector<bf16, q_len = 64>` per lane organized as one MFMA A-operand fragment for GEMM0. The N-direction iteration of `GEMM0_E_N=2` is exposed as a `y_dim` so that `mma0(v_q, v_k)` consumes the full N-tile in 2 MFMA chains.

### 5.5 V Register Read via HW Transpose (`u_rv`)

`make_layout_rv<T>` (lines 150-185) is the most intricate layout — it must align with `ds_read_tr16_b64`'s built-in 16-lane subgroup transpose:

```cpp
lane_per_grp = 16            // each transpose group is 16 lanes
lane_lo      =  4            // 4 lanes form a "lo" sub-group (the transposed axis)
lane_hi      = 16 / 4 = 4    // 4 sub-groups per group

num_grps = WARP_SIZE / lane_per_grp = 4
grp_n    = W_N / (lane_lo × VEC_TR_V) = 32 / (4 × 4) = 2
grp_k    = num_grps / grp_n = 2
```

This is essentially saying: each wave's V-tile read is divided into `grp_n × grp_k = 4` sub-tiles, each handled by `lane_per_grp = 16` lanes. Within each 16-lane group, every lane reads `VEC_TR_V = 4` bf16 values and the hardware transpose presents the collective 16×4 bf16 tile in the orientation expected by the MFMA A operand.

The corresponding kernel call is `tr_load<VEC_TR_V>(s_v[buf], u_rv)` (lines 463, 521, 587, etc.), which the compiler lowers to a chain of `ds_read_tr16_b64` instructions.

### 5.6 Full GM/LDS/VGPR Layout Maps

This section expands the previous `make_layout_*` descriptions into concrete formulas, in the same style as an ISA address derivation. One important distinction from some persistent-attention kernels: **this OPUS D=128 kernel does not stage Q in LDS**. Q is loaded directly from GM to VGPR; only K and V are double-buffered in LDS.

Notation:

```text
lane_id = thread_id_x % 64
warp_id = thread_id_x / 64          // 0..7, readfirstlane-broadcast to SGPR

lane_m     = lane_id % 32           // Q/O row inside one warp tile
lane_group = lane_id / 32           // 0 or 1, half-wave selector

lane_n = lane_id >> 3               // 0..7, which KV row group inside a wave
lane_d = lane_id & 7                // 0..7, which 8-bf16 D vector inside a 64-bf16 half

q_block_size  = NUM_WARPS * Q_TILE_SIZE = 256
q_block_start = block_id_y * q_block_size
kv_tile(j)    = j * KV_TILE_SIZE * stride_kv_n

All element formulas below are in bf16 elements unless marked as byte addresses.
byte_addr = elem_addr * sizeof(bf16) = elem_addr * 2.
```

#### 5.6.0 MFMA A-Operand Convention and CDNA4 Transpose Load

When reading the layout maps below, distinguish the **mathematical left-hand matrix** from the **hardware MFMA A operand**. Both GEMMs in this kernel are built with `mfma_adaptor_swap_ab{}` (lines 336-346), so OPUS passes operands in mathematical order but the adaptor swaps them before calling the underlying MFMA:

```cpp
return base::operator()(b, a, c, ...);
```

Therefore the hardware `v_mfma_f32_32x32x16_bf16` source operands are:

| Source site | Mathematical GEMM | Source-level call | Mathematical left matrix | Hardware MFMA A operand (`src0`) | Hardware MFMA B operand (`src1`) |
|---|---|---|---|---|---|
| Cluster 1 | `S = Q @ K^T` | `mma0(v_q, v_k)` | `Q` / `v_q` | `K` / `v_k` | `Q` / `v_q` |
| Cluster 7 | `O += P @ V` | `mma1.step_k(..., v_p, v_v, v_o)` | `P` / `v_p` | `V` / `v_v` | `P` / `v_p` |

This is the main reason the V LDS read is special. In GEMM1/GEMM2, `v_v` is not just "the right-hand source operand" at the C++ call site; after `mfma_adaptor_swap_ab`, it becomes the **hardware MFMA A operand**. CDNA4 provides `DS_READ_B64_TR_B16` exactly for this kind of operand load.

CDNA4 ISA section **11.4 MFMA Transpose Load from LDS** defines the instruction family as LDS-to-VGPR loads that transpose matrix data while transferring 16-, 8-, 6-, or 4-bit elements. For this kernel the relevant mnemonic is:

```asm
ds_read_b64_tr_b16 v[dst:dst+1], vaddr offset:imm
```

Its per-lane interface is:

```text
Input:
  vaddr + offset  = LDS byte address
  EXEC            = all lanes active (ISA requirement)
  LDS data        = matrix fragment interpreted as 16-bit elements

Output:
  v[dst:dst+1]    = 64 bits per lane = 4 bf16 values
```

The `b64` part means one instruction returns `4 × bf16` per lane. The `tr_b16` part means the data is interpreted as 16-bit matrix elements and transposed on the LDS-to-VGPR path. A normal `ds_read_b64` would preserve each lane's contiguous 8-byte vector; `ds_read_b64_tr_b16` instead makes the lanes collectively read a matrix fragment and returns the MFMA-oriented transposed view.

ISA also states that a complete 16-bit transpose-load matrix is loaded by **two** instructions with different LDS addresses and VGPR destinations:

```text
Complete B16 transpose-load matrix:

  first  ds_read_b64_tr_b16: K slices 0..3  and 8..11
  second ds_read_b64_tr_b16: K slices 4..7  and 12..15

Here "K" is the MFMA reduction dimension, not the attention K tensor.
```

That is why the FlyDSL port's equivalent helper does:

```python
a = ds_read_tr_v4f16(lds_off_lo)
b = ds_read_tr_v4f16(lds_off_lo + OPUS_URV_I5_STRIDE)  # +64 bf16
pack = concat(a, b)                                    # 8 bf16 MFMA-A pack
```

The layout effect for one 16-lane V transpose group can be visualized as:

```text
Before ds_read_b64_tr_b16: LDS view written by DMA-friendly V layout

                 dim0..3    dim4..7    dim8..11   dim12..15
  V row 0        lane 0     lane 1     lane 2     lane 3
  V row 1        lane 4     lane 5     lane 6     lane 7
  V row 2        lane 8     lane 9     lane10     lane11
  V row 3        lane12     lane13     lane14     lane15

After hardware transpose: VGPR view consumed as MFMA A operand

  lane 0  -> [V(row0, dim0),  V(row1, dim0),  V(row2, dim0),  V(row3, dim0)]
  lane 1  -> [V(row0, dim1),  V(row1, dim1),  V(row2, dim1),  V(row3, dim1)]
  lane 2  -> [V(row0, dim2),  V(row1, dim2),  V(row2, dim2),  V(row3, dim2)]
  ...
  lane15  -> [V(row0, dim15), V(row1, dim15), V(row2, dim15), V(row3, dim15)]
```

In the dumped causal ISA, the first two V transpose-loads in the main-loop cluster appear as:

```asm
ds_read_b64_tr_b16 v[20:21], v222 offset:0
ds_read_b64_tr_b16 v[22:23], v222 offset:0x80
```

The following GEMM2 MFMA then uses the transpose-loaded V registers as `src0`, i.e. hardware A:

```asm
v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[112:115], v[80:95]
```

Here `v[20:23]` is V (`v_v`) and `v[112:115]` is P (`v_p`). This is why V uses the OPUS `u_rv` layout, `tr_load<VEC_TR_V>`, and the wider `smem_padding_64B`: the LDS layout is built so the hardware transpose load can turn the DMA-friendly stored V tile into an MFMA-A-friendly VGPR fragment without a software `ds_read + ds_permute/v_perm` transpose sequence.

#### 5.6.1 Q: GM -> VGPR

Q has no LDS residency. `u_q` maps each lane to one Q row and two interleaved 8-bf16 D vectors per 16-wide K step.

```text
GM Q base:
  g_q = ptr_q
      + b * stride_q_b
      + q_block_start * stride_q_n
      + h * stride_q_h

Per workgroup Q rows:
  warp 0: rows q_block_start +   0..31
  warp 1: rows q_block_start +  32..63
  warp 2: rows q_block_start +  64..95
  warp 3: rows q_block_start +  96..127
  warp 4: rows q_block_start + 128..159
  warp 5: rows q_block_start + 160..191
  warp 6: rows q_block_start + 192..223
  warp 7: rows q_block_start + 224..255
```

`make_layout_q` address formula:

```text
k_iter = 0..7
vec    = 0..7

q_row = warp_id * 32 + lane_m
q_dim = k_iter * 16 + lane_group * 8 + vec

GM elem addr = q_row * stride_q_n + q_dim
```

Per-lane VGPR expansion:

| lane_id | lane_m | lane_group | `v_q` contents |
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
v_q: vector<bf16, 64> per lane
  v_q[k_iter * 8 + vec] =
      Q[q_block_start + warp_id * 32 + lane_m,
        k_iter * 16 + lane_group * 8 + vec]

Then:
  v_q_f32 *= (1 / sqrt(D)) * log2e
  v_q = cast<bf16>(v_q_f32)
```

#### 5.6.2 K: GM -> LDS -> VGPR

K is cooperatively loaded by all 512 threads. Each lane transfers one 16-byte vector (`VEC_KV=8 bf16`) for each of the two D half-lines.

GM address formula (`u_gk`):

```text
d_half = 0..1
vec    = 0..7

k_row = lane_n * NUM_WARPS + warp_id      // interleaved rows: wave 0 gets 0,8,...; wave 1 gets 1,9,...
k_dim = d_half * 64 + lane_d * 8 + vec

GM elem addr = kv_tile(j) + k_row * stride_kv_n + k_dim
```

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
s_k[0] = smem_buf + 0
s_k[1] = smem_buf + smem_buffer_elems

K line stride = smem_linear_wave + smem_padding_16B
              = 512 + 8 = 520 bf16 = 1040 B

K tile size   = 16 lines * 520 bf16 = 8320 bf16
```

K LDS address formula (`u_sk` as the `async_load` destination):

```text
line = d_half * NUM_WARPS + warp_id       // 0..15
in_line_elem = lane_id * VEC_KV + vec     // 0..511, then 8 bf16 padding

K LDS elem addr = line * 520 + in_line_elem
K LDS byte addr = (line * 520 + lane_id * 8 + vec) * 2
```

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
Byte[0:15]       <- lane 0:  one 16B vector, row lane_n=0, dim lane_d=0
Byte[16:31]      <- lane 1:  one 16B vector, row lane_n=0, dim lane_d=1
...
Byte[112:127]    <- lane 7:  one 16B vector, row lane_n=0, dim lane_d=7
Byte[128:143]    <- lane 8:  one 16B vector, row lane_n=1, dim lane_d=0
...
Byte[1008:1023]  <- lane 63: one 16B vector, row lane_n=7, dim lane_d=7
Byte[1024:1039]  <- 8 bf16 padding
```

K VGPR read formula (`u_rk`) rearranges the padded LDS tile into GEMM0 operand order:

```text
lane_id_n = lane_id % W_N = lane_id % 32

p0 = lane_id_n % NUM_WARPS       // 0..7
p1 = lane_id_n / NUM_WARPS       // 0..3
p2 = lane_id / W_N               // 0 or 1

i_ngrp  = 0..1
i_dhalf = 0..1
i_k     = 0..3
vec     = 0..7

K read elem addr =
    p0 * 520
  + (i_ngrp * 4 + p1) * 64
  + i_dhalf * (8 * 520)
  + ((i_k * 2 + p2) * 8 + vec)
```

Reversing this address through the LDS write layout gives the logical K coordinate:

```text
K row = (i_ngrp * 4 + p1) * NUM_WARPS + p0
K dim = i_dhalf * 64 + (i_k * 2 + p2) * 8 + vec
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

```text
thread_id_x = warp_id * 64 + lane_id
K LDS byte range = [K read elem addr * 2, K read elem addr * 2 + 15]
```

| lane_id | thread_id_x | p0 | p1 | p2 | K LDS elem base | K LDS byte range | Logical K |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | `warp_id*64 + 0` | 0 | 0 | 0 | 0 | `0x000..0x00F` | row 0, dim `0..7` |
| 1 | `warp_id*64 + 1` | 1 | 0 | 0 | 520 | `0x410..0x41F` | row 1, dim `0..7` |
| 2 | `warp_id*64 + 2` | 2 | 0 | 0 | 1040 | `0x820..0x82F` | row 2, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | `warp_id*64 + 7` | 7 | 0 | 0 | 3640 | `0x1C70..0x1C7F` | row 7, dim `0..7` |
| 8 | `warp_id*64 + 8` | 0 | 1 | 0 | 64 | `0x080..0x08F` | row 8, dim `0..7` |
| 9 | `warp_id*64 + 9` | 1 | 1 | 0 | 584 | `0x490..0x49F` | row 9, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31 | `warp_id*64 + 31` | 7 | 3 | 0 | 3832 | `0x1DF0..0x1DFF` | row 31, dim `0..7` |
| 32 | `warp_id*64 + 32` | 0 | 0 | 1 | 8 | `0x010..0x01F` | row 0, dim `8..15` |
| 33 | `warp_id*64 + 33` | 1 | 0 | 1 | 528 | `0x420..0x42F` | row 1, dim `8..15` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 63 | `warp_id*64 + 63` | 7 | 3 | 1 | 3840 | `0x1E00..0x1E0F` | row 31, dim `8..15` |

Notice that the K LDS read is intentionally not lane-contiguous. For example, lane 1 reads from byte `0x410`, i.e. K line 1, while lane 8 returns to byte `0x080` inside line 0. This cross-line access is exactly what converts the cooperative GM->LDS stripe layout into the MFMA-friendly `v_k` operand layout.

The loop axes then cover the remaining K tile:

| Axis | Values | Effect |
|---|---|---|
| `i_ngrp` | `0, 1` | selects K rows `0..31` then `32..63` |
| `i_dhalf` | `0, 1` | selects D half `0..63` then `64..127` |
| `i_k` | `0..3` | advances within a D half by `16` bf16 per step |
| `p2` | `0, 1` | lane half reads the low/high 8 bf16 inside each 16-bf16 step |

VGPR shape:

```text
v_k: vector<bf16, 64> per lane

It is not a simple row-major K vector; it is already packed as the MFMA operand
needed by:
  v_s = mma0(v_q, v_k)

GEMM0 logical tile:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

#### 5.6.3 V: GM -> LDS -> VGPR

V uses the same GM coordinate layout as K and the same cooperative `async_load<VEC_KV>`, but it uses a larger LDS padding and a transpose LDS read.

GM address formula (`u_gv`):

```text
d_half = 0..1
vec    = 0..7

v_row = lane_n * NUM_WARPS + warp_id
v_dim = d_half * 64 + lane_d * 8 + vec

GM elem addr = kv_tile(j) + v_row * stride_kv_n + v_dim
```

V LDS buffer map:

```text
s_v[0] = smem_buf + smem_k_tile_elems
s_v[1] = smem_buf + smem_buffer_elems + smem_k_tile_elems

V line stride = smem_linear_wave + smem_padding_64B
              = 512 + 32 = 544 bf16 = 1088 B

V tile size   = 16 lines * 544 bf16 = 8704 bf16
```

V LDS address formula (`u_sv` as the `async_load` destination):

```text
line = d_half * NUM_WARPS + warp_id
in_line_elem = lane_id * VEC_KV + vec

V LDS elem addr = line * 544 + in_line_elem
V LDS byte addr = (line * 544 + lane_id * 8 + vec) * 2
```

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

V VGPR read formula (`u_rv`) is shaped for `ds_read_tr16_b64`.

### `ds_read_tr16_b64` / `ds_read_b64_tr_b16` instruction semantics

On MI355X / gfx950, OPUS lowers `tr_load<VEC_TR_V>` to the CDNA4 LDS hardware-transpose read instruction. The CDNA4 ISA documents this family in **11.4 MFMA Transpose Load from LDS**: these instructions transpose matrix data while transferring 16-, 8-, 6-, or 4-bit elements from LDS to VGPRs. In the C++ helper this appears as the ISA mnemonic:

```cpp
ds_read_b64_tr_b16 %dst, %addr, offset:imm
```

In FlyDSL / MLIR ROCDL the same operation is exposed as:

```python
rocdl.ds_read_tr16_b64(v4f16_type, ptr)
```

The names emphasize different parts of the same operation:

| Name part | Meaning |
|---|---|
| `ds_read` | Read from LDS / shared memory. |
| `b64` | Each lane loads 64 bits = 8 bytes. For bf16 this is `4 × bf16`. |
| `tr_b16` / `tr16` | Treat the 64-bit lane payload as 16-bit elements and apply the MFMA transpose-load mapping. |

Semantically, this is not a normal `ds_read_b64`. A normal LDS read would give each lane the four bf16 values that are contiguous at that lane's address. The transpose variant makes the participating lanes collectively read a matrix fragment from LDS and returns the transposed view in VGPRs. MLIR describes this family as: each lane reads a vector from a column-major matrix in LDS, and the lane result becomes a row of the transposed matrix.

ISA constraints that matter here:

| ISA rule | Kernel implication |
|---|---|
| `EXEC` must be all 1s before executing these instructions. | The OPUS V read is issued in the uniform GEMM2 path, not under a per-lane predicate. |
| LDS address must be aligned to the element size. | bf16 `tr_b16` addresses are at least 2-byte aligned; OPUS computes them in bf16 element units and converts to bytes. |
| DS ops reading/writing 64-bit or larger data require even-aligned VGPR destinations. | The compiler allocates the 64-bit return as an aligned VGPR pair / `vector<4xbf16>`. |
| A complete 16-bit transpose-load matrix is loaded by two instructions with different LDS addresses and VGPR destinations. | This is why the FlyDSL port emits two reads, `a = ds_read_tr_v4f16(lds_off_lo)` and `b = ds_read_tr_v4f16(lds_off_lo + 64)`, then concatenates them into one `<8xbf16>` MFMA pack. |

For this kernel:

```text
one DS_READ_B64_TR_B16 / rocdl.ds_read_tr16_b64:
  per-lane payload = 64 bits = 4 bf16 = VEC_TR_V
  result per lane  = vector<4xbf16>, already transposed for MFMA operand use

two DS_READ_B64_TR_B16 instructions:
  complete the 16-bit transpose-load matrix for one OPUS V pack
  first instruction covers K slices 0..3 and 8..11
  second instruction covers K slices 4..7 and 12..15
  combined result per lane = 8 bf16, i.e. one MFMA V operand pack
```

This is exactly what GEMM2 needs. V is stored in LDS in a DMA-friendly layout where rows and D slices are contiguous enough for `buffer_load_dwordx4_lds`. GEMM2, however, consumes V as the MFMA A operand for:

```text
O[32 × 128] += P[32 × 64] @ V[64 × 128]
```

so the operand seen by MFMA is effectively `V^T`: D-chunk fragments must be presented across the MFMA M direction while the KV dimension becomes the reduction direction. `ds_read_tr16_b64` performs that lane/data transpose during the LDS read, avoiding a software sequence such as:

```text
ds_read_b64 / ds_read_b128
  + ds_permute_b32 / v_perm_b32
  + extra VGPR temporaries
```

The cost is that the LDS layout must be built for the hardware transpose. That is why V uses `smem_padding_64B` (32 bf16 padding per line) instead of K's `smem_padding_16B`: the wider line stride keeps the 16-lane transpose groups bank-conflict-free while preserving the row-major DMA store pattern.

```text
lane_per_grp = 16
lane_lo      = 4
lane_hi      = 4
num_grps     = 4
grp_n        = 2
grp_k        = 2

grp_id      = lane_id / 16        // 0..3
lane_in_grp = lane_id % 16

p0 = grp_id / grp_n               // 0..1
p1 = lane_in_grp / lane_lo        // 0..3
p2 = grp_id % grp_n               // 0..1
p3 = lane_in_grp % lane_lo        // 0..3
```

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

The table shows the 64-lane wave as four `ds_read_tr16_b64` transpose groups. Within each 16-lane group, `p1` selects one of the four 4-lane high subgroups, while `p3` is the lane position inside the 4-lane low subgroup. `p0` and `p2` split the four groups into the two `grp_k` and two `grp_n` directions used by the V operand layout.

`u_rv` LDS element address:

```text
i0  = 0..1
i1  = 0..1
i4a = 0..3
i4b = 0..1
vec = 0..3

V read elem addr =
    i0 * (8 * 544)
  + i1 * (grp_n * lane_lo * VEC_TR_V)          // i1 * 32
  + (p0 * lane_hi + p1) * 544
  + (i4a * 2 + i4b) * 64
  + ((p2 * lane_lo + p3) * VEC_TR_V + vec)
```

Reversing this address through the V LDS write layout gives the logical V coordinate:

```text
V row = (i4a * 2 + i4b) * NUM_WARPS + (p0 * lane_hi + p1)
V dim = i0 * 64 + i1 * 32 + (p2 * lane_lo + p3) * VEC_TR_V + vec
```

Per-lane V LDS read expansion for `i0=0`, `i1=0`, `i4a=0`, `i4b=0`:

| lane_id range | p0 | p1 | p2 | p3 | V LDS elem base range | V LDS byte range | Logical V |
|---:|---:|---:|---:|---:|---|---|---|
| `0..3` | 0 | 0 | 0 | `0..3` | `0, 4, 8, 12` | `0x000..0x01F` | row 0, dim `0..15` |
| `4..7` | 0 | 1 | 0 | `0..3` | `544, 548, 552, 556` | `0x440..0x45F` | row 1, dim `0..15` |
| `8..11` | 0 | 2 | 0 | `0..3` | `1088, 1092, 1096, 1100` | `0x880..0x89F` | row 2, dim `0..15` |
| `12..15` | 0 | 3 | 0 | `0..3` | `1632, 1636, 1640, 1644` | `0xCC0..0xCDF` | row 3, dim `0..15` |
| `16..19` | 0 | 0 | 1 | `0..3` | `16, 20, 24, 28` | `0x020..0x03F` | row 0, dim `16..31` |
| `20..23` | 0 | 1 | 1 | `0..3` | `560, 564, 568, 572` | `0x460..0x47F` | row 1, dim `16..31` |
| `24..27` | 0 | 2 | 1 | `0..3` | `1104, 1108, 1112, 1116` | `0x8A0..0x8BF` | row 2, dim `16..31` |
| `28..31` | 0 | 3 | 1 | `0..3` | `1648, 1652, 1656, 1660` | `0xCE0..0xCFF` | row 3, dim `16..31` |
| `32..35` | 1 | 0 | 0 | `0..3` | `2176, 2180, 2184, 2188` | `0x1100..0x111F` | row 4, dim `0..15` |
| `36..39` | 1 | 1 | 0 | `0..3` | `2720, 2724, 2728, 2732` | `0x1540..0x155F` | row 5, dim `0..15` |
| `40..43` | 1 | 2 | 0 | `0..3` | `3264, 3268, 3272, 3276` | `0x1980..0x199F` | row 6, dim `0..15` |
| `44..47` | 1 | 3 | 0 | `0..3` | `3808, 3812, 3816, 3820` | `0x1DC0..0x1DDF` | row 7, dim `0..15` |
| `48..51` | 1 | 0 | 1 | `0..3` | `2192, 2196, 2200, 2204` | `0x1120..0x113F` | row 4, dim `16..31` |
| `52..55` | 1 | 1 | 1 | `0..3` | `2736, 2740, 2744, 2748` | `0x1560..0x157F` | row 5, dim `16..31` |
| `56..59` | 1 | 2 | 1 | `0..3` | `3280, 3284, 3288, 3292` | `0x19A0..0x19BF` | row 6, dim `16..31` |
| `60..63` | 1 | 3 | 1 | `0..3` | `3824, 3828, 3832, 3836` | `0x1DE0..0x1DFF` | row 7, dim `16..31` |

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
v_v = tr_load<VEC_TR_V>(s_v[buf], u_rv)

The compiler lowers this to ds_read_tr16_b64. Each 16-lane group reads 4 bf16
per lane and the hardware transpose presents the data in the orientation needed by:
  v_o = mma1(v_p, v_v, v_o)

GEMM1 logical tile:
  O[32 x 128] += P[32 x 64] @ V[64 x 128]
```

#### 5.6.4 S and P: VGPR-only Score/Probability Layout

S never touches GM or LDS. It is the FP32 output of GEMM0:

```text
v_s[0], v_s[1]: vector<float, 32> per lane

Logical tile per warp:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

S VGPR index formula:

```text
i_n  = 0..1        // 32-column N strip
rept = 0..3
j    = 0..3

idx = i_n * 16 + rept * 4 + j

q_row = warp_id * 32 + lane_m
k_col = i_n * 32 + lane_group * 4 + rept * 8 + j
```

Per-lane expansion for one warp:

| lane_id | q row inside warp | lane_group | `v_s` K columns |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27, 32..35, 40..43, 48..51, 56..59` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31, 36..39, 44..47, 52..55, 60..63` |
| 1 | 1 | 0 | same K-column pattern as lane 0, for Q row 1 |
| 33 | 1 | 1 | same K-column pattern as lane 32, for Q row 1 |

This is why `attn_row_max` and `attn_sum` use `permlane32_swap`: lanes `x` and `x+32` together hold the full 64-column score row for one Q row.

After masking, max/subtract, and `exp2`, the probability fragment uses the same index layout:

```text
v_p = cast<bf16>(v_s)
v_p: vector<bf16, 32> per lane
```

#### 5.6.5 O: VGPR -> GM

O is accumulated in VGPRs and only written to GM at the end. It has no LDS residency.

VGPR accumulator:

```text
v_o: vector<float, 64> per lane

Logical tile per warp:
  O[32 x 128] += P[32 x 64] @ V[64 x 128]

GEMM1_E_N = 4      // four 32-wide D strips
GEMM1_E_K = 4      // four 16-wide KV substeps
```

`v_o` index formula mirrors the S layout, replacing the S N-axis with O's D-axis:

```text
d_strip = 0..3
rept    = 0..3
j       = 0..3

idx = d_strip * 16 + rept * 4 + j

o_row = warp_id * 32 + lane_m
o_dim = d_strip * 32 + lane_group * 4 + rept * 8 + j
```

Final GM store (`u_o`):

```text
g_o = ptr_o
    + b * stride_q_b
    + q_block_start * stride_q_n
    + h * stride_q_h

Before store:
  l_inv = 1 / l_row
  v_o *= l_inv
  v_o_bf16 = cast<bf16>(v_o)

store<VEC_O=4>(g_o, v_o_bf16, u_o)
```

`make_layout_o` store address formula:

```text
d_strip = 0..3
pack    = 0..3
vec     = 0..3

o_row = warp_id * 32 + lane_m
o_dim = d_strip * 32 + pack * 8 + lane_group * 4 + vec

GM elem addr = o_row * stride_q_n + o_dim
```

Per-lane store expansion:

| lane_id | row inside warp | lane_group | O dims stored |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27, 32..35, ..., 120..123` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31, 36..39, ..., 124..127` |
| 1 | 1 | 0 | same D pattern as lane 0, for O row 1 |
| 33 | 1 | 1 | same D pattern as lane 32, for O row 1 |

#### 5.6.6 Summary: Where Each Tensor Lives

```text
Q:
  GM [B, N, H, D]
    -> u_q load
    -> VGPR v_q[64] bf16, pre-scaled
  No LDS.

K:
  GM [B, N, H_KV, D]
    -> u_gk async_load
    -> LDS s_k[2], each tile 8320 bf16, 16B padding per line
    -> u_rk load
    -> VGPR v_k[64] bf16, GEMM0 operand

V:
  GM [B, N, H_KV, D]
    -> u_gv async_load
    -> LDS s_v[2], each tile 8704 bf16, 64B padding per line
    -> u_rv tr_load / ds_read_tr16_b64
    -> VGPR v_v, GEMM1 operand

S:
  VGPR only.
  v_s[0/1][32] fp32.
  Lanes x and x+32 together hold one full 64-column score row.

P:
  VGPR only.
  v_p[32] bf16, same logical index layout as S after softmax.

O:
  VGPR v_o[64] fp32 accumulator
    -> normalize by l_row
    -> cast bf16
    -> u_o store
    -> GM [B, N, H, D]
  No LDS.
```

#### 5.6.7 Low-Level Call Chains and Issue Counts

The source-level calls in the cluster code are high-level OPUS layout operations. The table below records the final backend primitive and the number of dynamic wave-instructions emitted by one source statement in the D=128 kernel.

| Source statement | High-level role | Lowest-level primitive | ISA form | Count per wave |
|---|---|---|---|---:|
| `async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(j))` | GM K tile -> LDS K buffer | `__builtin_amdgcn_raw_ptr_buffer_load_lds(...)` | `buffer_load_dwordx4 ... lds` | 2 |
| `async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(j - 1))` | GM V tile -> LDS V buffer | `__builtin_amdgcn_raw_ptr_buffer_load_lds(...)` | `buffer_load_dwordx4 ... lds` | 2 |
| `v_k = load<T::VEC_KV>(s_k[buf], u_rk)` | LDS K -> VGPR MFMA operand | LDS address-space vector load (`smem::_load`) | `ds_read_b128` | 16 |
| `v_v = tr_load<T::VEC_TR_V>(s_v[buf], u_rv)` | LDS V -> VGPR MFMA-A operand | inline asm `ds_read_b64_tr_b16` (`smem::_tr_load`) | `ds_read_b64_tr_b16` | 32 |
| `v_s[x] = mma0(v_q, v_k)` | GEMM0 score tile | `__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)` | `v_mfma_f32_32x32x16_bf16` | 16 |
| `v_o = mma1.step_k(k, v_p, v_v, v_o)` | One GEMM1 reduction sub-step | `__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)` | `v_mfma_f32_32x32x16_bf16` | 4 |

The relevant call chains are:

```text
async_load<T::VEC_KV>(g_k/g_v, ...)
  -> opus::async_load(...)
  -> gmem::async_load(layoutG, layoutS)
  -> gmem::_async_load(...)
  -> __builtin_amdgcn_raw_ptr_buffer_load_lds(...)
  -> buffer_load_dwordx4 ... lds

load<T::VEC_KV>(s_k, u_rk)
  -> opus::load(...)
  -> smem::load(layout)
  -> smem::_load(offset)
  -> LDS address-space vector pointer dereference
  -> ds_read_b128

tr_load<T::VEC_TR_V>(s_v, u_rv)
  -> opus::tr_load(...)
  -> smem::tr_load(layout)
  -> smem::_tr_load<vec, imm_offset>(base)
  -> asm volatile("ds_read_b64_tr_b16 ...")

mma0(v_q, v_k)
  -> tiled_mma_adaptor::operator()
  -> mfma_adaptor_swap_ab::operator()
  -> mfma::operator()
  -> __builtin_amdgcn_mfma_f32_32x32x16_bf16(...)

mma1.step_k(k, v_p, v_v, v_o)
  -> tiled_mma_adaptor::step_k<k>()
  -> mfma_adaptor_swap_ab::operator()
  -> mfma::operator()
  -> __builtin_amdgcn_mfma_f32_32x32x16_bf16(...)
```

The `mma0` / `mma1` counts come from the tiled-MFMA expansion dimensions. The D=128 instantiation is `opus_gqa_traits<32, 64, 128, 8, causal>`, so:

```text
W_M = 32
W_N = 32
W_K = 16
SLICE_D = 128

GEMM0_E_M = Q_TILE_SIZE  / W_M = 32 / 32  = 1
GEMM0_E_N = KV_TILE_SIZE / W_N = 64 / 32  = 2
GEMM0_E_K = SLICE_D      / W_K = 128 / 16 = 8

GEMM1_E_M = Q_TILE_SIZE  / W_M = 32 / 32  = 1
GEMM1_E_N = SLICE_D      / W_N = 128 / 32 = 4
GEMM1_E_K = KV_TILE_SIZE / W_K = 64 / 16  = 4
```

Therefore:

```text
mma0(v_q, v_k):
  GEMM0_E_M * GEMM0_E_N * GEMM0_E_K
  = 1 * 2 * 8
  = 16 MFMA instructions

mma1.step_k(k, v_p, v_v, v_o):
  STEP_K is fixed, so only GEMM1_E_M * GEMM1_E_N remains
  = 1 * 4
  = 4 MFMA instructions per step_k

mma1.step_k(0..3) together:
  4 steps * 4 MFMA/step
  = 16 MFMA instructions
```

Because both `mma0` and `mma1` are built with `mfma_adaptor_swap_ab{}`, source operand order and hardware MFMA source order differ:

```text
mma0(v_q, v_k):
  mathematical left matrix = Q / v_q
  hardware MFMA A operand  = K / v_k

mma1.step_k(..., v_p, v_v, v_o):
  mathematical left matrix = P / v_p
  hardware MFMA A operand  = V / v_v
```

The `load` / `tr_load` counts come from `layout_load_traits<Layout, vec>::r_elem`. OPUS computes:

```text
issue_space      = all y_dim axes from the layout
issue_space_vec  = issue_space with the final y_dim divided by vec
r_elem           = product(issue_space_vec)
```

The bulk load functions then loop exactly `r_elem.value` times. For `u_rk`, the y-dim issue space is:

```text
u_rk issue_space =
  [GEMM0_E_N / n_grp,
   n_grp,
   smem_d_rpt,
   GEMM0_E_K / smem_d_rpt,
   VEC_KV]

load<T::VEC_KV> vectorizes the final axis:

r_elem(u_rk, VEC_KV)
  = (GEMM0_E_N / n_grp) * n_grp * smem_d_rpt * (GEMM0_E_K / smem_d_rpt)
  = GEMM0_E_N * GEMM0_E_K
  = 2 * 8
  = 16
```

This equals the explicit trait formula:

```text
k_ds_read_insts =
  (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV)
  = (2 * 8 * 32 * 16) / (64 * 8)
  = 16
```

For `u_rv`, the y-dim issue space is:

```text
u_rv issue_space =
  [GEMM1_E_N / (D_128B_SIZE / W_N),
   D_128B_SIZE / W_N,
   GEMM1_E_K,
   W_K / (lane_hi * grp_k),
   VEC_TR_V]

tr_load<T::VEC_TR_V> vectorizes the final axis:

r_elem(u_rv, VEC_TR_V)
  = GEMM1_E_N * GEMM1_E_K * W_K / (lane_hi * grp_k)
  = 4 * 4 * 16 / (4 * 2)
  = 32
```

This equals the explicit trait formula:

```text
v_ds_read_insts =
  (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V)
  = (4 * 4 * 32 * 16) / (64 * 4)
  = 32
```

Finally, `async_load<T::VEC_KV>` uses the global-to-LDS layout `u_gk/u_gv`; for D=128 each call issues:

```text
k_buffer_load_insts = v_buffer_load_insts =
  (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV)
  = (64 * 128) / (512 * 8)
  = 2
```

---

## 6. Q Preload with In-Flight Scaling

Lines 403-406:

```cpp
v_q = load<T::VEC_Q>(g_q, u_q);
auto v_q_f32 = opus::cast<float>(v_q);                              // bf16 → f32 (8 per lane × 8 = 64 elts)
static_for<q_len>([&](auto i) { v_q_f32[i.value] *= temperature_scale; });
v_q = opus::cast<D_ATTN>(v_q_f32);                                  // f32 → bf16 truncate
```

where `temperature_scale = (1.0 / sqrtf(D)) * log2e` (line 376).

**This is the "in-flight Q scaling" optimization.** By pre-multiplying Q with `(1/√D) × log2e` *once* at the prologue:

1. The GEMM0 score `v_s = Q_scaled @ K^T` already contains the softmax temperature **and** the log2-base conversion factor
2. The softmax kernel can use plain `attn_sub_row(v_s, m_row)` + `exp2(v_s)` **without** any per-element FMA to apply `c_sm_scale_log2e`
3. Saves 1 `v_pk_fma_f32` per S element (32 FMAs per warp per KV tile)

The cost is precision: Q is rounded to bf16 *after* scaling, but since `1/√128 × log2e ≈ 0.1275`, the multiply does not amplify the bf16 truncation error meaningfully (verified by the validator passing at MaxErr < 5e-2).

The current FlyDSL OPUS port matches this optimization: `flash_attn_opus.py` loads Q in lines 631-655, multiplies each f32 Q element by `c_sm_scale_log2e`, then truncates back to bf16. Its later softmax helper therefore uses plain subtract + `exp2`, not a per-element scale FMA.

---

## 7. GEMM0: S = Q @ K^T

### 7.1 Per-Warp Tile Decomposition

```
Per warp:
  S[32 × 64] = Q[32 × 128] @ K^T[128 × 64]

Decomposed via mma0:
  GEMM0_E_M = 1   (M / W_M = 32/32)
  GEMM0_E_N = 2   (N / W_N = 64/32)
  GEMM0_E_K = 8   (K / W_K = 128/16)

Total MFMAs per warp per outer GEMM0 call: 1 × 2 × 8 = 16

Score output:
  v_s[i]: vector<f32, 32>     (i ∈ {0, 1})    GEMM0 ping-pong buffers
  s_len      = 32
  s_half_len = 16             (first 16 elements = N-strip 0, last 16 = N-strip 1)
```

### 7.2 `make_tiled_mma` + `mfma_adaptor_swap_ab`

Lines 336-340:

```cpp
auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
    seq<GEMM0_E_M, GEMM0_E_N, GEMM0_E_K>{},   // {1, 2, 8}
    seq<T_M, T_N, T_K>{},                      // {8, 1, 1}
    seq<W_M, W_N, W_K>{},                      // {32, 32, 16}
    mfma_adaptor_swap_ab{});
```

`mfma_adaptor_swap_ab` instructs the factory to emit MFMA instructions with A/B operand roles swapped — required because `Q @ K^T` is conceptually (row-major Q) × (row-major K) → (row-major S), but `v_mfma_f32_32x32x16_bf16` natively computes (column-major A) × (row-major B). The swap collapses the conceptual transpose without any extra data movement.

A call like:

```cpp
v_s[0] = mma0(v_q, v_k);          // line 420 / 508 / 630
```

unrolls (at compile time) into **16 MFMA instructions** writing the f32 `v_s[0]` of size 32, with appropriate register packing chosen by the swap adaptor.

### 7.3 Score Output `v_s[0]`, `v_s[1]`

The kernel keeps **two score buffers** `v_s[0]` and `v_s[1]` (line 360) — these are the per-warp ping-pong score accumulators. The main loop's 8-cluster body alternates which buffer holds the freshly computed score versus the one being consumed by exp2/sum/cast:

```
Cluster 1: v_s[1] = mma0(v_q, v_k);    // compute tile j-2
           v_s[0][half:] = exp2(...);   // finish exp of tile j-3
Cluster 5: v_s[0] = mma0(v_q, v_k);    // compute tile j-1
           v_s[1][half:] = exp2(...);   // finish exp of tile j-2
```

This is the same alternation pattern as the MI308X ASM kernel's `v[64:79]` ↔ `v[80:95]` score ping-pong, but exposed cleanly through C++ index variables.

---

## 8. Causal Mask via Inline ASM

### 8.1 `attn_mask_vec2_imm` Primitive

Lines 233-249:

```cpp
template<int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(u32 rel_vgpr, u32 neg_inf_vgpr,
                                          u32& x_ref, u32& y_ref) {
    uint64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"               // x_mask = (rel < THR_X)
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"               // y_mask = (rel < THR_Y)
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"          // x_ref = x_mask ? -inf : x_ref
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"          // y_ref = y_mask ? -inf : y_ref
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
}
```

This handles **two score-vector elements at once** with **immediate threshold operands** `"n"(THR_X)` / `"n"(THR_Y)`. The two `v_cmp_lt_i32_e64` produce two `vcc`-style SGPR-pair masks, and the two `v_cndmask_b32_e64` apply those masks to overwrite each score element with `neg_inf_vgpr` when `rel < THR`.

Using immediate thresholds (compile-time constants) lets the compiler inline the comparison constant directly into the `v_cmp` encoding (saving a register and reducing the dependency chain).

### 8.2 `attn_mask_causal_tile` Driver

Lines 251-290 iterate the masking over the full `v_s` score vector:

```cpp
elems_per_wave_tile = (W_M × W_N) / WARP_SIZE = (32 × 32) / 64 = 16     // per N-strip
c_pack          = 4                                                       // 2 (cmp) × 2 (vec2_imm) interleave
c_rept          = elems_per_wave_tile / c_pack = 4
c_rept_stride   = (WARP_SIZE / W_M) × c_pack = 2 × 4 = 8

const int q_pos       = q_start_pos + (lane_id % W_M);                    // this lane's Q row
const int k_start_pos = kv_tile_idx × KV_TILE_SIZE;
const int lane_group  = lane_id / W_M;                                    // 0 or 1 (which K-pack)

for i_n in 0..GEMM0_E_N-1 (= 0..1):                                       // each N-strip
    base_idx = i_n × 16
    k_pos    = k_start_pos + i_n × W_N + lane_group × c_pack              // K col base for this lane
    rel      = q_pos - k_pos                                              // (signed) relative position
    for i_rept in 0..3:                                                   // 4 reps per N-strip
        for i_pair in 0..1:                                               // 2 pairs per rep
            idx       = base_idx + i_rept × c_pack + i_pair × 2
            thr_x     = i_rept × 8 + i_pair × 2                           // K-offset of element X
            thr_y     = thr_x + 1                                         // K-offset of element Y
            attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
```

The condition `rel < THR_X` is equivalent to `q_pos - k_pos < k_offset`, i.e. `k_pos + k_offset > q_pos`, which is the standard "K column is in the future of Q row" causal violation check.

Each `attn_mask_causal_tile` call emits **16 pairs** (`2 N-strips × 4 reps × 2 pairs`) of the `v_cmp + v_cndmask` instruction sequence — total 32 `v_cmp` + 32 `v_cndmask` per warp per masked tile. The C++ template applies the mask only under a tile-level guard `q_start_pos < kv_end_pos`: prologue lines 422-427, main-loop lines 522-527, and epilogue lines 588-593 / 644-649 / 699-704. This gates entire tile-skips when the warp's Q rows are all "below the diagonal" relative to the tile.

---

## 9. Online Softmax with Lazy Rescaling

### 9.1 Algorithm Recap (Pre-Scaled Q)

Because Q is pre-scaled by `(1/√D) × log2e`, the softmax recurrence simplifies to:

```
m_row, l_row : per-lane scalars                 (running max, running sum)
v_o          : output accumulator               (per-warp vector<f32, 64>)

Initially: m_row = -inf, l_row = 0, v_o = 0

For each KV tile:
  v_s = mma0(v_q_scaled, v_k)         // already includes (1/√D) × log2e
  v_s = causal_mask(v_s)              // optional
  row_max = attn_row_max(v_s)         // half-warp reduction

  // LAZY RESCALE CHECK (the OPUS optimization)
  below = (row_max - m_row) <= 8.0
  all_below = ballot_w64(below) == read_exec()
  if (all_below):
      row_max = m_row                 // clamp; corr = exp2(0) = 1
  else:
      rescale_m = exp2(m_row - row_max)
      v_o     *= rescale_m
      l_row   *= rescale_m
      m_row    = row_max

  v_s -= row_max                      // {-inf, 0, ..., 0}
  v_s  = exp2(v_s)                    // softmax weights (unnormalized)
  v_p  = cast<bf16>(v_s)
  v_o += v_p @ v_v
  l_row += sum(v_s)
```

Final: `v_o *= 1/l_row` (line 745-746).

### 9.2 `attn_row_max` (`permlane32_swap`)

Lines 187-197:

```cpp
template<typename T, typename V>
__device__ inline D_ACC attn_row_max(const V& v_s) {
    constexpr index_t s_len = vector_traits<V>::size();           // = 32
    D_ACC row_max = -1e30f;
    static_for<s_len>([&](auto i) {
        row_max = max(row_max, v_s[i.value]);                     // per-lane reduce of 32 elements
    });
    vector<u32, 2> res = __builtin_amdgcn_permlane32_swap(
        bit_cast<u32>(row_max), bit_cast<u32>(row_max),
        /*fi=*/false, /*bound_ctrl=*/true);
    return max(bit_cast<float>(res.x), bit_cast<float>(res.y));   // cross-half merge
}
```

`__builtin_amdgcn_permlane32_swap` (gfx950 ISA) exchanges values between lane *i* (in lanes 0-31) and lane *i+32* (in lanes 32-63) in **a single instruction**, returning a 2-element vector where `.x` holds the original lane's value and `.y` holds the swapped lane's value. One `v_max_f32` then gives the wave-wide row max.

This replaces the MI308X-style `ds_permute_b32` + `v_max_f32` chain (which costs ~20 cycles for `ds_permute` plus an LDS waitcnt) — `permlane32_swap` is a single VALU instruction, ~4 cycles latency.

`attn_sum` (lines 215-225) uses the same primitive for sum reduction.

### 9.3 Lazy Rescale via `ballot_w64 == read_exec`

Lines 475-484 (Cluster 3 body, first KV tile of the iter pair):

```cpp
bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);     // per lane
bool all_below    = (__builtin_amdgcn_ballot_w64(below_thresh)
                     == __builtin_amdgcn_read_exec());
if (__builtin_expect(all_below, 1)) {
    row_max = m_row;                          // clamp
} else {
    rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
    scale_output_tile<T>(v_o, rescale_m);     // 64 v_pk_mul_f32 (or interleaved with v_mul)
    l_row    *= rescale_m;
    m_row     = row_max;
}
```

- `ballot_w64(p)`: returns a 64-bit mask where bit *i* is set iff lane *i*'s `p` is non-zero
- `read_exec()`: returns the current EXEC mask (which lanes are active)
- `ballot == read_exec` ⇒ every active lane has `below_thresh = true`

When all 64 lanes see their tile-max within 8.0 of the running max (in log2 space, a `2^8 = 256` ratio in linear softmax weight space), the **entire `v_o` rescale and `l_row` rescale are elided**. The `__builtin_expect(all_below, 1)` hint tells the compiler to lay out the fast path inline and the slow path out-of-line.

**Numerical correctness:** When `all_below`, we clamp `row_max := m_row` so `rescale_m = exp2(0) = 1`. The subsequent `v_s -= row_max` is just `v_s -= m_row`, which is the *same* m_row used to scale prior tiles — so `v_o` (accumulated in m_row frame) and the new `exp2(v_s)` (also computed in m_row frame) are consistent. No rescale needed.

### 9.4 Half-Slice `attn_exp2_slice` Strategy

The exp2 computation is **split across two clusters** in the main loop:

```cpp
// Cluster N:   first half of exp2 (right after row_max + sub_row)
attn_sub_row<T>(v_s[X], row_max);
asm volatile("" : "+v"(v_s[X]) ::);                // register anchor
attn_exp2_slice<T, 0, s_half_len>(v_s[X]);          // exp2 elements [0..15]

// Cluster N+1: second half + sum + cast
attn_exp2_slice<T, s_half_len, s_half_len>(v_s[X]); // exp2 elements [16..31]
l_row += attn_sum<T>(v_s[X]);
v_p = cast<bf16>(v_s[X]);
```

Each `attn_exp2_slice` emits `Count` `v_exp_f32` instructions. Splitting them across clusters lets the LLVM scheduler interleave the second half's `v_exp_f32` with **GEMM0's MFMA chain** that runs concurrently in the next cluster — this is exactly what `sched_barrier_exp_pairs<6, 3, Group>` enforces (alternating 1 MFMA + 3 EXP).

### 9.5 P Conversion + Register Anchor

```cpp
v_p = opus::cast<D_ATTN>(v_s[X]);
asm volatile("" : "+v"(v_p) ::);
```

The `asm volatile("" : "+v"(v_p) ::)` is a **register anchor**: an empty inline-asm statement that constrains `v_p` to remain in a VGPR and prevents the compiler from re-materializing or reordering its definition past the anchor. This is critical because the subsequent `mma1.step_k(...)` calls expect `v_p` to be live across cluster boundaries — without the anchor, LLVM might rematerialize the `cast<bf16>` (which depends on `v_s[X]`, soon to be overwritten by the next GEMM0).

The epilogue also anchors sub-vectors of `v_o` after unconditional rescale sites (lines 614, 669, 730):

```cpp
auto* v_o_pin = reinterpret_cast<vector_t<fp32_t, 16>*>(&v_o);
asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]),
                  "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
```

This pins all four 16-element sub-vectors of `v_o` (one per D-chunk, total 64 f32) so that the `scale_output_tile<T>(v_o, rescale_m)` multiplication doesn't get sunk into the next cluster's MFMA chain.

---

## 10. GEMM2: O = P @ V via `tr_load`

### 10.1 Per-Warp Tile Decomposition

```
Per warp:
  O[32 × 128] += P[32 × 64] @ V[64 × 128]

Decomposed via mma1:
  GEMM1_E_M = 1   (M / W_M = 32/32)
  GEMM1_E_N = 4   (D / W_N = 128/32)
  GEMM1_E_K = 4   (KV / W_K = 64/16)

Total MFMAs per warp per outer GEMM1 call: 1 × 4 × 4 = 16

Output accumulator:
  v_o: vector<f32, 64> per lane = 4 × 16 elements (4 D-chunks)
  o_len = 64
```

### 10.2 `tr_load` Hardware Transpose

```cpp
v_v = tr_load<T::VEC_TR_V>(s_v[buf], u_rv);
```

`tr_load` lowers to a chain of `ds_read_tr16_b64` instructions (gfx950 only). Each instruction reads 4 bf16 from LDS per lane and **transposes** across a 16-lane subgroup, returning a `vector<bf16, VEC_TR_V=4>` per lane. Combined across the subgroup, this is the equivalent of reading a 16-lane × 4-bf16 LDS tile and presenting its transposed view to the MFMA A operand.

Without this instruction, the kernel would need an explicit `ds_read` + `v_perm_b32` / `ds_permute_b32` software transpose sequence instead of one hardware-transpose LDS read.

### 10.3 `mma1.step_k(K, ...)` Sub-Step Issuing

Unlike `mma0(v_q, v_k)` which emits all 16 MFMAs at once, `mma1` is invoked in **per-K-step** mode in the main loop (lines 472, 485-487, etc.):

```cpp
v_o = mma1.step_k(0_I, v_p, v_v, v_o);     // K=0 sub-step: 4 MFMAs (covers GEMM1_E_N=4)
... attn_row_max + lazy rescale check ...
v_o = mma1.step_k(1_I, v_p, v_v, v_o);     // K=1 sub-step
v_o = mma1.step_k(2_I, v_p, v_v, v_o);     // K=2 sub-step
v_o = mma1.step_k(3_I, v_p, v_v, v_o);     // K=3 sub-step
```

Each `step_k` emits 4 MFMAs (one per D-chunk of the 128-wide D). By placing the row-max + lazy-rescale check **between** `step_k(0)` and `step_k(1)`, the kernel gives the scheduler a window to place row-max, branch, and possible `scale_output_tile<T>(v_o, rescale_m)` work before the remaining 12 GEMM1 MFMAs, instead of waiting for the full 16-MFMA GEMM1 chain to finish first. The `sched_barrier_pairs` calls around this region are what make the intended MFMA/VALU interleave explicit.

In the epilogue's later clusters (lines 602, 658, 712, 742), `mma1(v_p, v_v, v_o)` is called *without* `step_k` because the lazy-rescale window is no longer interleaved — the full 16-MFMA chain runs back-to-back.

---

## 11. Output Finalization

Lines 744-754:

```cpp
D_ACC l_inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;             // guard against tile-fully-masked
static_for<o_len>([&](auto i) { v_o[i.value] *= l_inv; });        // O /= sum

if (!stagger) {                                                    // group A re-syncs with group B
    __builtin_amdgcn_s_barrier();
}

auto u_o = make_layout_o<T>(warp_id, lane_id, kargs.stride_q_n);
auto v_o_bf16 = opus::cast<D_ATTN>(v_o);
store<T::VEC_O>(g_o, v_o_bf16, u_o);                              // global store, 4 bf16 per lane per inst
```

- `l_inv`: one `v_rcp_f32` + one `v_cmp_lt_f32` + `v_cndmask` (the zero-guard branch)
- `v_o *= l_inv`: 64 `v_mul_f32` (or 32 `v_pk_mul_f32` if LLVM vectorizes)
- `cast<bf16>`: 32 `v_perm_b32` (truncate pair-wise)
- `store<VEC_O>`: emits `global_store_dwordx2` (8 B / 4 bf16 per instruction) per lane, total `o_len / VEC_O = 16` stores per lane

The asymmetric barrier re-aligns the two staggered wave groups before global store so both groups observe the same total barrier ordinal across the kernel.

`make_layout_o<T>` (lines 54-73) uses the same warp/lane layout style as `make_layout_q<T>`, but it is **not** a literal inverse with the same shape. `make_layout_q` is built from `GEMM0_E_M × GEMM0_E_K × VEC_Q`, while `make_layout_o` is built from `GEMM1_E_M × GEMM1_E_N × VEC_O` and stores the GEMM1 accumulator layout back to `g_o`.

---

## 12. `sched_group_barrier` Discipline

One of the kernel's distinguishing scheduling features is its **fine-grained scheduling discipline** using `__builtin_amdgcn_sched_group_barrier(mask, count, group)`. The current FlyDSL OPUS port mirrors these helper calls, but the C++ template form remains the source of truth for the intended grouping.

### 12.1 Masks and Counts

```cpp
constexpr int MFMA_MASK = 0x08;       // = SCHED_GROUP_BARRIER_INST_MFMA
constexpr int VALU_MASK = 0x02;       // = SCHED_GROUP_BARRIER_INST_VALU
constexpr int EXP_MASK  = 0x400;      // = SCHED_GROUP_BARRIER_INST_VALU_TRANS (transcendental, incl exp/log/rcp)
```

A `sched_group_barrier(MASK, N, GROUP)` instruction tells the LLVM machine scheduler:

> "Within scheduling group `GROUP`, schedule exactly `N` instructions that match `MASK` between this barrier and the next."

When alternated as `(MFMA × 1, VALU × N)` in tight sequence, this forces an exact "1 MFMA + N VALU" issuing pattern, fully predictable to the post-RA scheduler.

### 12.2 `sched_barrier_pairs` / `sched_barrier_exp_pairs` Helpers

Lines 18-30:

```cpp
template<int Pairs, int VALU_CNT, int Group>
__device__ inline void sched_barrier_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
    __builtin_amdgcn_sched_group_barrier(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ inline void sched_barrier_exp_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
    __builtin_amdgcn_sched_group_barrier(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}
```

A call like `sched_barrier_pairs<10, 5, 1>()` expands to 10 alternating `(MFMA × 1, VALU × 5)` group-1 barriers, forcing the scheduler to lay down exactly:

```
MFMA, VALU, VALU, VALU, VALU, VALU,
MFMA, VALU, VALU, VALU, VALU, VALU,
... × 10
```

Similarly, `sched_barrier_exp_pairs<6, 3, 1>()` enforces 6 pairs of `(MFMA × 1, EXP × 3)`.

### 12.3 Group Numbering (1..10)

Each cluster uses a distinct group ID so that barriers within one cluster don't interfere with barriers in another:

| Cluster | Group IDs used | Pattern |
|---|---|---|
| Main loop cluster 1 (after `mma0` + exp2 + sum + cast) | `1` | `exp_pairs<6,3,1>` + `pairs<10,5,1>` |
| Main loop cluster 3 (between `step_k` calls) | `2` | `pairs<4,5,2>` (pre-rescale) + `pairs<6,5,2>` (post) + `exp_pairs<6,3,2>` |
| Main loop cluster 5 (`mma0` + exp2 + sum + cast) | `3` | `exp_pairs<6,3,3>` + `pairs<10,5,3>` |
| Main loop cluster 7 (between `step_k` calls) | `4` | same as cluster 3 with group=4 |
| Epilogue cluster 1 (`mma0` + exp2 + sum + cast) | `5` | same as main cluster 1 with group=5 |
| Epilogue cluster 3 (`mma1` + row_max + exp2 + scale) | `6` | `pairs<10,5,6>` + `exp_pairs<6,3,6>` |
| Epilogue cluster 5 | `7` | group=7 |
| Epilogue cluster 7 | `8` | group=8 |
| Epilogue cluster 9 | `9` | group=9 |
| Epilogue cluster 11 | `10` | group=10 |

The integer group IDs are arbitrary — any pair of integers `g1 ≠ g2` would work for keeping clusters independent. The kernel just uses ascending IDs (1, 2, ..., 10) for readability.

The FlyDSL OPUS port currently uses only the generic `rocdl.sched_barrier(0)` (mask=0, count=0 = "no constraint, just a fence"), losing this fine-grained control. Adding `sched_group_barrier` support to FlyDSL is one of the listed future-work items.

---

## 13. Main Loop Pipeline (8 Clusters × 2 KV-Tiles per Iter)

### 13.1 High-Level Structure

```
For j = 3; j < max_num_tiles − 1; j += 2:            // 2 KV tiles per iter

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║ Each iter advances two KV tiles. It consumes P/V for tiles             ║
  ║ (j-3) and (j-2), computes score for tiles (j-2) and (j-1),             ║
  ║ and prefetches later K/V tiles for the next stages.                    ║
  ║                                                                       ║
  ║ Tile-to-cluster mapping (within one iter):                            ║
  ║   • Clusters 0-3:  consume tile (j-3) via tail-exp + GEMM2,            ║
  ║                    compute tile (j-2) via GEMM0 + start softmax       ║
  ║   • Clusters 4-7:  consume tile (j-2) via tail-exp + GEMM2,            ║
  ║                    compute tile (j-1) via GEMM0 + start softmax       ║
  ║                                                                       ║
  ║ Score buffer ping-pong:                                               ║
  ║   • Cluster 1: v_s[0] tail-exp + cast → P[j-3],                       ║
  ║                v_s[1] = mma0(v_q, v_k) → S[j-2]                       ║
  ║   • Cluster 3: v_o += P[j-3] @ V[j-3],                                ║
  ║                v_s[1] head-exp begun (for tile j-2)                   ║
  ║   • Cluster 5: v_s[1] tail-exp + cast → P[j-2],                       ║
  ║                v_s[0] = mma0(v_q, v_k) → S[j-1]                       ║
  ║   • Cluster 7: v_o += P[j-2] @ V[j-2],                                ║
  ║                v_s[0] head-exp begun (for tile j-1)                   ║
  ║                                                                       ║
  ║ K buffer ping-pong:                                                   ║
  ║   • Cluster 0: read s_k[1] = K[j-2]                                  ║
  ║   • Cluster 2: load s_k[1] ← K[j]                                    ║
  ║   • Cluster 4: read s_k[0] = K[j-1]                                  ║
  ║   • Cluster 6: load s_k[0] ← K[j+1]                                  ║
  ║                                                                       ║
  ║ V buffer ping-pong:                                                   ║
  ║   • Cluster 0: load s_v[1] ← V[j-2]                                  ║
  ║   • Cluster 2: tr_load s_v[0]   (consume V[j-3])                     ║
  ║   • Cluster 4: load s_v[0] ← V[j-1]                                  ║
  ║   • Cluster 6: tr_load s_v[1]   (consume V[j-2])                     ║
  ╚═══════════════════════════════════════════════════════════════════════╝
```

### 13.2 Cluster-by-Cluster Breakdown

```
══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 0 — V[j-2] async-load + K[j-2] ds-read + wait + barrier (memory cluster)
══════════════════════════════════════════════════════════════════════════════════
   async_load(g_v, s_v[1], u_gv, u_sv, kv_tile(j-2))         // V[j-2] → s_v[1]
   v_k = load<VEC_KV>(s_k[1], u_rk)                          // K[j-2] → registers
   s_waitcnt_lgkmcnt(0)                                       // K LDS read complete
   s_waitcnt_vmcnt(k_buffer_load_insts + v_buffer_load_insts)// keep K+V async loads outstanding
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 1 — GEMM0 + finish softmax of v_s[0] (compute cluster)
══════════════════════════════════════════════════════════════════════════════════
   v_s[1] = mma0(v_q, v_k)                                   // 16 MFMA → S[j-2]
   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0])        // finish exp of S[j-3][16:32]
   l_row += attn_sum<T>(v_s[0])                              // row sum of P[j-3]
   v_p = cast<bf16>(v_s[0])                                  // pack P[j-3] → bf16
   asm("" : "+v"(v_p))                                       // anchor
   sched_barrier_exp_pairs<6, 3, 1>()                        // 6 × (MFMA + 3 EXP) group=1
   sched_barrier_pairs<10, 5, 1>()                           // 10 × (MFMA + 5 VALU) group=1
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 2 — K[j] async-load + V[j-3] ds-read (memory cluster)
══════════════════════════════════════════════════════════════════════════════════
   async_load(g_k, s_k[1], u_gk, u_sk, kv_tile(j))            // K[j] → s_k[1]
   v_v = tr_load<VEC_TR_V>(s_v[0], u_rv)                      // V[j-3]→regs via tr16_b64
   s_waitcnt_lgkmcnt(0)                                       // V LDS read complete
   s_waitcnt_vmcnt(k+v_buffer_load_insts)
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 3 — GEMM2 for tile j-3 + lazy-rescale + start softmax of S[j-2]
══════════════════════════════════════════════════════════════════════════════════
   s_setprio(1)                                                // raise priority
   v_o = mma1.step_k(0, v_p, v_v, v_o)                         //  4 MFMA → O[D-chunk 0]
   row_max = attn_row_max<T>(v_s[1])                           // half-warp reduce
   sched_barrier_pairs<4, 5, 2>()                              // 4 × (MFMA + 5 VALU)
   
   ── LAZY RESCALE CHECK ──
   below = (row_max − m_row) ≤ 8.0
   all_below = ballot_w64(below) == read_exec()
   if (__builtin_expect(all_below, 1)):
       row_max = m_row                                         // clamp, skip rescale
   else:
       rescale_m = exp2(m_row − row_max)
       v_o *= rescale_m                                        // 64 v_mul_f32
       l_row *= rescale_m
       m_row = row_max
   
   v_o = mma1.step_k(1, v_p, v_v, v_o)                         //  4 MFMA → O[D-chunk 1]
   v_o = mma1.step_k(2, v_p, v_v, v_o)                         //  4 MFMA → O[D-chunk 2]
   v_o = mma1.step_k(3, v_p, v_v, v_o)                         //  4 MFMA → O[D-chunk 3]
   attn_sub_row<T>(v_s[1], row_max)                            // v_s[1] -= max
   asm("" : "+v"(v_s[1]))                                      // anchor
   attn_exp2_slice<T, 0, s_half_len>(v_s[1])                   // exp2 of v_s[1][0:16]
   sched_barrier_pairs<6, 5, 2>()                              // 6 × (MFMA + 5 VALU)
   sched_barrier_exp_pairs<6, 3, 2>()                          // 6 × (MFMA + 3 EXP)
   s_setprio(0)                                                // yield
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 4 — V[j-1] async-load + K[j-1] ds-read (mirror of Cluster 0, buffer 0)
══════════════════════════════════════════════════════════════════════════════════
   async_load(g_v, s_v[0], u_gv, u_sv, kv_tile(j-1))           // V[j-1] → s_v[0]
   v_k = load<VEC_KV>(s_k[0], u_rk)                            // K[j-1] → registers
   s_waitcnt_lgkmcnt(0)
   s_waitcnt_vmcnt(k+v_buffer_load_insts)
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 5 — GEMM0 + finish softmax of v_s[1]   (mirror of Cluster 1, buffer swap)
══════════════════════════════════════════════════════════════════════════════════
   v_s[0] = mma0(v_q, v_k)                                     // 16 MFMA → S[j-1]
   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1])          // finish exp of S[j-2][16:32]
   l_row += attn_sum<T>(v_s[1])
   v_p = cast<bf16>(v_s[1])
   asm("" : "+v"(v_p))
   sched_barrier_exp_pairs<6, 3, 3>(); sched_barrier_pairs<10, 5, 3>()
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 6 — K[j+1] async-load + V[j-2] ds-read + CAUSAL MASK
══════════════════════════════════════════════════════════════════════════════════
   async_load(g_k, s_k[0], u_gk, u_sk, kv_tile(j+1))           // K[j+1] → s_k[0]
   v_v = tr_load<VEC_TR_V>(s_v[1], u_rv)                       // V[j-2] → regs
   if constexpr (CAUSAL):
       kv_end_pos = j × KV_TILE_SIZE
       if (q_start_pos < kv_end_pos):
           attn_mask_causal_tile<T>(v_s[0], q_start_pos, j-1, neg_inf_v, lane_id)
   s_waitcnt_lgkmcnt(0)
   s_waitcnt_vmcnt(k+v_buffer_load_insts)
   sched_barrier(0); s_barrier(); sched_barrier(0)

══════════════════════════════════════════════════════════════════════════════════
 CLUSTER 7 — GEMM2 for tile j-2 + lazy-rescale + start softmax of S[j-1]
══════════════════════════════════════════════════════════════════════════════════
   s_setprio(1)
   v_o = mma1.step_k(0, v_p, v_v, v_o)                         // 4 MFMA
   row_max = attn_row_max<T>(v_s[0])
   sched_barrier_pairs<4, 5, 4>()
   
   ── LAZY RESCALE CHECK (same as Cluster 3) ──
   below = (row_max − m_row) ≤ 8.0
   all_below = ballot_w64(below) == read_exec()
   if (all_below): row_max = m_row
   else: rescale_m = exp2(m_row − row_max); v_o *= rescale_m; l_row *= ...; m_row ← row_max
   
   v_o = mma1.step_k(1, v_p, v_v, v_o)
   v_o = mma1.step_k(2, v_p, v_v, v_o)
   v_o = mma1.step_k(3, v_p, v_v, v_o)
   attn_sub_row<T>(v_s[0], row_max)
   asm("" : "+v"(v_s[0]))
   attn_exp2_slice<T, 0, s_half_len>(v_s[0])
   sched_barrier_pairs<6, 5, 4>(); sched_barrier_exp_pairs<6, 3, 4>()
   s_setprio(0)
   sched_barrier(0); s_barrier(); sched_barrier(0)

   (end of iter; loop back, j += 2)
```

### 13.3 Synchronization Points

| Cluster | `s_waitcnt` | `s_barrier` | Purpose |
|---|---|---|---|
| 0, 4 | `lgkmcnt(0)` + `vmcnt(k+v_buffer_load_insts)` | yes | wait K ds_read complete, keep K+V async loads outstanding |
| 1, 3, 5, 7 | none | yes | barriers only; loads in flight |
| 2, 6 | `lgkmcnt(0)` + `vmcnt(k+v_buffer_load_insts)` | yes | wait V ds_read complete |

The `vmcnt(N)` value (= `T::k_buffer_load_insts + T::v_buffer_load_insts`) is computed at compile time from the trait. It tells the hardware: "you may have up to N outstanding global-to-LDS loads; further loads must wait." Choosing N = total async loads issued so far in the iter lets the scheduler keep all of them in flight while still waiting on LDS reads.

### 13.4 Buffer Ping-Pong Summary

```
Within one iter:
  Cluster 0 reads  s_k[1] = K[j-2]     (loaded by the previous iter's Cluster 2,
                                        or by the prologue for the first iter)
  Cluster 2 writes s_k[1] ← K[j]

  Cluster 4 reads  s_k[0] = K[j-1]     (loaded by the previous iter's Cluster 6,
                                        or by the prologue for the first iter)
  Cluster 6 writes s_k[0] ← K[j+1]

  Cluster 0 writes s_v[1] ← V[j-2]
  Cluster 2 reads  s_v[0] = V[j-3]

  Cluster 4 writes s_v[0] ← V[j-1]
  Cluster 6 reads  s_v[1] = V[j-2]
```

The important ordering is **read-before-overwrite** for each LDS buffer. A K tile is loaded at least one half-iteration before it is consumed; V is loaded in one memory cluster and consumed two clusters later, giving the async global-to-LDS path enough independent work to hide latency.

---

## 14. Prologue (1 Tile + GEMM0 Setup)

Lines 397-436:

```
1. async_load K[0]   → s_k[0]
2. s_waitcnt(0)      // wait for K[0] in LDS
3. s_barrier         // sync 8 warps

4. v_q = load Q                          // 32 × 128 bf16 per warp
5. v_q_f32 = cast<float>(v_q)
   v_q_f32 *= temperature_scale          // <-- in-flight Q scaling
   v_q = cast<bf16>(v_q_f32)             // 64 scaled elements packed back to bf16

6. async_load K[1]   → s_k[1]
7. async_load V[0]   → s_v[0]
8. v_k = load(s_k[0], u_rk)              // K[0] → registers
9. s_waitcnt lgkmcnt(0)
   s_waitcnt vmcnt(v_buffer_load_insts)  // wait K loads done, keep V outstanding

10. if (stagger): s_barrier              // <-- group B parks here

11. v_s[0] = mma0(v_q, v_k)              // 16 MFMA, first GEMM0
12. CAUSAL: attn_mask_causal_tile(v_s[0], q_start_pos, 0, neg_inf, lane_id)

13. m_row = attn_row_max(v_s[0])         // permlane32_swap
14. attn_sub_row(v_s[0], m_row)
15. asm("" : "+v"(v_s[0]))
16. attn_exp2_slice<0, s_half_len>(v_s[0])     // first half of exp

17. s_barrier                            // <-- group A and group B re-align
18. async_load K[2]   → s_k[0]
```

The extra stagger barrier does **not** skip any work. It pairs group B's step-10 barrier with group A's step-17 barrier, so group A enters main-loop Cluster 0 while group B is still doing the prologue GEMM0 + softmax-start work. When group A reaches Cluster 0's barrier, group B reaches the prologue step-17 barrier; that establishes the one-cluster phase offset used throughout the steady-state loop. Both groups still execute the full algorithm.

---

## 15. Epilogue (14 Clusters Drain)

Lines 563-742. After the main loop exits, the ping-pong pipeline still has carried softmax/output state and trailing KV tiles to flush. The C++ template drains this state explicitly through **14 labelled clusters (0..13)**:

| Cluster | Code line | Operation |
|---|---|---|
| 0 (epi) | 565-571 | V[max-3] async + K[max-3] ds_read |
| 1 (epi) | 574-583 | mma0 tile (max-3), tail-exp v_s[0], cast v_p[max-4] |
| 2 (epi) | 586-598 | K[max-1] async + V[max-4] ds_read + CAUSAL mask v_s[1] |
| 3 (epi) | 601-618 | GEMM2 (max-4), row_max (no lazy here, full rescale always), exp v_s[1] |
| 4 (epi) | 621-627 | V[max-2] async + K[max-2] ds_read |
| 5 (epi) | 630-640 | mma0 tile (max-2), tail-exp v_s[1], cast v_p[max-3] |
| 6 (epi) | 643-654 | V[max-3] ds_read + CAUSAL mask v_s[0] |
| 7 (epi) | 657-673 | GEMM2 (max-3), row_max, full rescale, exp v_s[0] |
| 8 (epi) | 676-682 | V[max-1] async + K[max-1] ds_read |
| 9 (epi) | 685-695 | mma0 tile (max-1), tail-exp v_s[0], cast v_p[max-2] |
| 10 (epi) | 698-709 | V[max-2] ds_read + CAUSAL mask v_s[1] |
| 11 (epi) | 712-732 | GEMM2 (max-2), row_max, full rescale, finish v_s[1] head-exp + tail-exp + sum + cast |
| 12 (epi) | 735-739 | V[max-1] tr_load (no more async; final tile only) |
| 13 (epi) | 742 | Final GEMM2 (max-1) — `v_o += v_p @ v_v`, no rescale needed |

The epilogue still uses `s_setprio(1/0)` around earlier GEMM2 drain clusters (clusters 3 and 7). It uses **no `s_setprio`** before cluster 11 and the final cluster-13 `mma1` (line 742), because there is no following steady-state cluster boundary that needs a priority handoff; both groups are draining the same final state before the pre-store re-sync.

The lazy-rescale optimization is **disabled in the epilogue** (clusters 3, 7, 11) — every rescale cluster executes the full `rescale_m = exp2(...); v_o *= rescale_m; l_row *= rescale_m; m_row = row_max` block. The reason: the lazy gate trades a small probability of redundant rescale for the certainty of a rescale-free fast path, which is only worthwhile when there are many steady-state iterations to amortize the branch overhead. The epilogue is a fixed drain sequence, so the unconditional rescale is simpler and faster.

---

## 16. Stagger: Wave-Group Phase Shift

Recapping the stagger mechanism:

```cpp
const int stagger = warp_id / 4;   // 0 for warps 0-3, 1 for warps 4-7

// Prologue (after line 414):
if (stagger) { sched_barrier(0); s_barrier(); }      // group B parks here

// Just before final store (lines 748-750):
if (!stagger) { s_barrier(); }                       // group A re-syncs
```

**Counted barrier mapping** — let `S(g, i)` = the i-th `s_barrier` executed by warps in group g:

| Source location | Group A (warp 0-3) | Group B (warp 4-7) |
|---|---|---|
| Prologue, line 401 (after K[0] in LDS) | S(A, 1) | S(B, 1) |
| Prologue, line 417 (stagger gate, only B) | — | S(B, 2) |
| Prologue, line 433 (after softmax start) | S(A, 2) | S(B, 3) |
| Main-loop cluster 0 barrier, iter 1 | S(A, 3) | S(B, 4) |
| Main-loop cluster 1 barrier, iter 1 | S(A, 4) | S(B, 5) |
| ... | ... | ... |
| Main-loop cluster 7 barrier, iter K | S(A, 2+8K) | S(B, 3+8K) |
| Epilogue clusters 0-12 barriers | S(A, 3+8K+i)| S(B, 4+8K+i) |
| Final pre-store gate, line 749 (only A) | S(A, last) | — |

**Workgroup-wide barrier matching** is by ordinal arrival count, so global barriers fire when **every warp has reached its i-th `s_barrier`**:

```
Global Bar 1:  A's S(A, 1)   ←→   B's S(B, 1)   (both at prologue line 401)
Global Bar 2:  A's S(A, 2)   ←→   B's S(B, 2)   (B at stagger gate; A at line 433)
Global Bar 3:  A's S(A, 3)   ←→   B's S(B, 3)   (A at main cluster 0; B at line 433)
Global Bar 4:  A's S(A, 4)   ←→   B's S(B, 4)   (A at main cluster 1; B at main cluster 0)
...
```

The result: in steady state, while group A is finishing Cluster N, group B is finishing Cluster N−1. The two groups execute **the same code** but one cluster apart in time.

**Why this matters:** Memory-cluster (0, 2, 4, 6) operations (`async_load`, `s_waitcnt`, `tr_load`) and compute-cluster (1, 3, 5, 7) operations (MFMA, exp2, sum) use different hardware units:

| Cluster | Group A | Group B (one step behind) | Resource overlap |
|---|---|---|---|
| Time t   | Compute (MFMA + VALU + EXP) | Memory (VMEM + LDS) | None — fully parallel |
| Time t+1 | Memory (VMEM + LDS) | Compute (MFMA + VALU) | None — fully parallel |
| Time t+2 | Compute | Memory | None — fully parallel |
| ... | | | |

This achieves **continuous occupancy of both MFMA and VMEM/LDS hardware units** with only 4 active waves per group (since MFMA hardware can sustain ~4 waves before saturating). The same idea was applied in the MI308X ASM kernel via two distinct code paths (`label_03DA` / `label_07AA`) — here it's achieved with a single code path and an asymmetric barrier gate.

The final pre-store barrier (`if (!stagger) s_barrier()`) re-aligns the two groups so they issue global stores together, maximizing L2 write coalescing.

---

## 17. Performance and Comparison

### Default `gqa_host.cc` config (from main, line 274-278)

Compile-time benchmark default:
- B=16, H=64, H_KV=8 (GQA group size = 8), **N=1024**, D=128
- bf16 inputs, validated via `gqa_attention_ref` (CPU reference) with threshold 5e-2

### Performance numbers (collected on MI355X, gfx950, S=8192 — not the `gqa_host.cc` default of N=1024)

| Config | OPUS C++ (this kernel) | FlyDSL `flash_attn_opus` port | FlyDSL baseline | MI308X ASM kernel |
|---|---:|---:|---:|---:|
| B=16, S=8192, H=64, D=128, causal | **1131 TFLOPS** | 69.6 | 716 | 595 |
| B=16, S=8192, H=64, D=128, nocausal | **1165 TFLOPS** | 66.3 | 640 | 1249 |

The performance numbers are measured at S=8192 — a different scale than the compile-time default in `gqa_host.cc` (N=1024) — to stress the steady-state main-loop body.

### Why OPUS C++ outperforms the FlyDSL port (gap analysis)

The current FlyDSL port now replicates the major structural mechanisms of this C++ kernel: 3D grid, 8-cluster `j += 2` ping-pong loop, double-buffered K/V LDS, in-flight Q scaling, `sched_group_barrier` helpers, inline-asm causal-mask primitive, register anchors, V-hoist placement, and stagger barriers. The remaining gap is mostly lower-level codegen and register-allocation behavior:

1. **V `tr_load` representation.** C++ materializes one layout-aware `vtype_b`; FlyDSL materializes 16 independent Python SSA packs (`4 k_substeps × 4 D-chunks`), increasing VGPR live range across the hoist barrier.
2. **Line-stride address arithmetic.** OPUS C++ keeps `s_k[i].ptr` / `s_v[i].ptr` as pointer bases; FlyDSL recomputes 520 / 544-bf16 line-stride formulas in Python IR, producing extra address arithmetic.
3. **Causal tile gates.** C++ wraps `attn_mask_causal_tile` in `q_start_pos < kv_end_pos` guards at every masked site; FlyDSL currently applies the mask helper whenever `CAUSAL=True`, so below-diagonal tiles still pay mask instruction cost.
4. **Unused yield-NOP toggle.** `flash_attn_opus.py` parses `FLYDSL_OPUS_YIELD_NOP`, but neither the current Python source nor this C++ template emits an `s_nop 15; s_nop 7` yield window.

Items 1 and 2 are estimated to contribute the largest share of the current performance gap.

### Why the MI308X ASM kernel wins nocausal but loses causal

- **Causal**: MI308X kernel uses a two-pass causal scheme (`s36 = 0 → 1` with mirrored Q tile) to balance load across the diagonal. Its OPUS-equivalent (this C++ kernel) gates the loop bound with `causal_num_tiles` and uses the inline-asm mask, which is more efficient at large S. → OPUS wins causal at S=8192.
- **Nocausal**: MI308X ASM hand-tuned the 8-wave `label_03DA`/`label_07AA` split + the `15 MFMA + VALU + 1 MFMA` interleave pattern. Without the causal-skip overhead, this scheme reaches 1249 TFLOPS, slightly beating OPUS's 1165 TFLOPS. → ASM wins nocausal.

---

## Quick Reference

### MFMA chain summary (per warp, per KV tile)

| Phase | Source | MFMAs |
|---|---|---|
| GEMM0 prologue | line 420 | 16 |
| GEMM0 main cluster 1 | line 450 | 16 |
| GEMM0 main cluster 5 | line 508 | 16 |
| GEMM2 main cluster 3 (`step_k 0-3`) | lines 472-487 | 16 |
| GEMM2 main cluster 7 (`step_k 0-3`) | lines 536-551 | 16 |
| GEMM0 epilogue clusters 1, 5, 9 | lines 574, 630, 685 | 16 × 3 = 48 |
| GEMM2 epilogue clusters 3, 7, 11, 13 | lines 602, 658, 712, 742 | 16 × 4 = 64 |

**Steady-state per pair-iter (2 KV tiles): 32 + 32 = 64 MFMAs per warp.**

### Buffer ping-pong summary

| Buffer | Sources | Lifetime |
|---|---|---|
| `s_k[0]` | even K tiles: `kv_tile(0)`, `kv_tile(2)`, then `kv_tile(j+1)` in main cluster 6 | read before overwrite in cluster 4/6 pair |
| `s_k[1]` | odd K tiles: `kv_tile(1)`, then `kv_tile(j)` in main cluster 2 | read before overwrite in cluster 0/2 pair |
| `s_v[0]` | even V tiles: `kv_tile(0)`, then `kv_tile(j-1)` in main cluster 4 | loaded two clusters before `tr_load` |
| `s_v[1]` | odd V tiles: `kv_tile(j-2)` in main cluster 0, plus epilogue odd tiles | loaded two clusters before `tr_load` |
| `v_s[0]` (register) | even score buf | overwritten in cluster 5 |
| `v_s[1]` (register) | odd score buf | overwritten in cluster 1 |
| `v_p` (register) | bf16 of the score tile ready for GEMM2 | overwritten when a score tile finishes tail-exp/cast (main clusters 1 and 5; epilogue clusters 1, 5, 9, 11) |

### Key built-ins used

| Builtin / instruction | Purpose | Lines used |
|---|---|---|
| `__builtin_amdgcn_s_waitcnt(N)` | full memory wait | 399 |
| `s_waitcnt_lgkmcnt(0)` | wait LDS (and SMRD, GDS) | 412, 443, 464, ... |
| `s_waitcnt_vmcnt(N)` | wait VMEM, keep N outstanding | 413, 444, 465, ... |
| `__builtin_amdgcn_s_barrier()` | workgroup sync | 401, 417, 433, ... |
| `__builtin_amdgcn_sched_barrier(0)` | scheduler fence (any inst) | passim |
| `__builtin_amdgcn_sched_group_barrier(M, N, G)` | scheduler fence (N of type M, group G) | inside `sched_barrier_*pairs` |
| `__builtin_amdgcn_s_setprio(N)` | wave priority N=0..3 | 471, 493, 535, 557, 601, 615, 657, 670 |
| `__builtin_amdgcn_permlane32_swap(...)` | lane[i] ↔ lane[i+32] swap | 195, 223 |
| `__builtin_amdgcn_ballot_w64(p)` | 64-bit lane mask | 476, 540 |
| `__builtin_amdgcn_read_exec()` | read EXEC mask | 476, 540 |
| `__builtin_amdgcn_exp2f(x)` | hardware `v_exp_f32` | 211, 480, 544 |
| `__builtin_amdgcn_readfirstlane(x)` | broadcast lane 0 → SGPR | 306 |
| `async_load<VEC>` | → `buffer_load_dwordx4_lds` | 398, 408, ... |
| `tr_load<VEC_TR_V>` | → `ds_read_tr16_b64` | 463, 521, 587, ... |
| `asm volatile("" : "+v"(v))` | register pin / anchor | 8 sites in main loop |
| `asm("v_cmp_lt_i32_e64 ...; v_cndmask_b32_e64 ...")` | causal mask pair | `attn_mask_vec2_imm` |

### File locations

- **Device kernel template:** `FlyDSL/opus_attn/gqa_d128_kernel_template.hpp` (756 lines)
- **Causal instantiation:** `FlyDSL/opus_attn/gqa_d128_kernel_causal.cc` (13 lines)
- **Non-causal instantiation:** `FlyDSL/opus_attn/gqa_d128_kernel_noncausal.cc` (13 lines)
- **Traits + kargs:** `FlyDSL/opus_attn/gqa_defs.h`
- **Host launcher + benchmark:** `FlyDSL/opus_attn/gqa_host.cc`
- **Python bindings:** `FlyDSL/opus_attn/gqa_python_api.cc`
- **Build target:** `opus_gqa_d128` (see `CMakeLists.txt` in `FlyDSL/opus_attn/`)
- **Test:** `FlyDSL/tests/kernels/test_flash_opus_attn.py --compare`
