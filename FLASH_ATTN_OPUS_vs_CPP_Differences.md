# Differences Analysis: `flash_attn_opus.py` (FlyDSL) vs `gqa_d128_kernel_template.hpp` (OPUS C++)

**Status:** post P1-P7 alignment (commits `ad1fcaf` → `2d07de2`)
**Scope:** D=128 BF16 GQA forward on AMD MI355X / gfx950 (CDNA4)
**FlyDSL file:** [`FlyDSL/kernels/flash_attn_opus.py`](kernels/flash_attn_opus.py)
**OPUS C++ file:** [`FlyDSL/opus_attn/gqa_d128_kernel_template.hpp`](opus_attn/gqa_d128_kernel_template.hpp)
**Companion doc:** [`FLASH_ATTN_OPUS_Kernel_Analysis_Detail.md`](FLASH_ATTN_OPUS_Kernel_Analysis_Detail.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What P1–P7 Aligned](#2-what-p1p7-aligned)
3. [Current Structural Map (Cluster-by-Cluster)](#3-current-structural-map-cluster-by-cluster)
4. [Remaining Differences After P1–P7](#4-remaining-differences-after-p1p7)
   - [4.1 Mechanically Equivalent (different syntax, same intent)](#41-mechanically-equivalent-different-syntax-same-intent)
   - [4.2 Semantically Different but Functionally Correct](#42-semantically-different-but-functionally-correct)
   - [4.3 Known Functional Gaps](#43-known-functional-gaps)
5. [Side-by-Side Code Mapping](#5-side-by-side-code-mapping)
6. [Line-Level Cross-Reference Table](#6-line-level-cross-reference-table)
7. [Performance Gap Analysis](#7-performance-gap-analysis)
8. [Quick Reference](#8-quick-reference)

---

## 1. Executive Summary

After phases P1 → P7, the FlyDSL Python kernel is a **faithful structural port** of the OPUS C++ template. The two kernels share:

- The same 8-cluster `j += 2` main-loop pipeline and 14-cluster (0..13) epilogue.
- The same loop-carry state shape `[m_row, l_row, o_accs[0..3], v_s_partial_lo, v_s_partial_hi]`.
- The same **OPUS-defined K/V LDS layout** (interleaved `[K0][V0][K1][V1]`, 68 096 B total; line-padded with `smem_padding_16B` for K and `smem_padding_64B` for V — P7).
- The same OPUS layouts `u_q`, `u_o`, `u_gk`, `u_sk`, `u_rk`, `u_gv`, `u_sv`, `u_rv` for global ↔ LDS ↔ register movement (P7).
- The same N-axis permutation π(m) = (m%8)·8 + m/8 applied to K LDS reads and propagated through the causal mask (P7).
- The same in-flight Q scaling (Q pre-multiplied by `(1/√D) · log2e` in the prologue, log2-space softmax).
- The same lazy rescale gate semantics (C++ spells the full-wave check as `ballot == read_exec`; FlyDSL uses `ballot == -1` under the same wave64 full-active assumption) → clamp `m_new := m_old`.
- The same hand-written `s_setprio`, `sched_group_barrier`, `s_barrier`, register-anchor and inline-asm causal-mask primitives. C++ still has tile-level causal-mask skip guards that FlyDSL does not currently emit.
- The same dual-group wave stagger (asymmetric `s_barrier` opened in the prologue, closed before global-store).
- The same V LDS read placement (all `tr_load` ops in the cluster **before** the consumer cluster).

Numerical correctness (against PyTorch f32 reference, threshold `MaxErr < 1e-2 ∧ MinCos > 0.99`):

| Config (B / S / H_Q / H_KV / D / dtype) | causal MaxErr / MinCos | nocausal MaxErr / MinCos |
|---|---|---|
| 1  / 512  / 64 / 64 / 128 / bf16 | 3.91 × 10⁻³ / 0.99999 | 9.77 × 10⁻⁴ / 0.99999 |
| 1  / 8192 / 64 / 64 / 128 / bf16 | 3.91 × 10⁻³ / 0.99999 | 2.44 × 10⁻⁴ / 0.99999 |
| **16 / 8192 / 64 / 64 / 128 / bf16** | **3.91 × 10⁻³ / 0.99999** (numerical parity with OPUS C++) | **2.44 × 10⁻⁴ / 0.99999** |
| 16 / 8192 / 64 /  8 / 128 / bf16 (GQA 8:1) | 3.91 × 10⁻³ / 0.99999 | 2.44 × 10⁻⁴ / 0.99999 |

The documented shape configurations pass under `--warmup 5 --iters 100`; the active `DEFAULT_CONFIGS` in `tests/kernels/test_flash_opus_attn.py` currently contains the two B=16/S=8192/H=64 bf16 MHA/GQA shapes.

Activation: `FLYDSL_ENABLE_OPUS_PATH=1` selects the OPUS path. `FLYDSL_OPUS_LAZY_RESCALE`, `FLYDSL_OPUS_SETPRIO`, and `FLYDSL_OPUS_STAGGER` default ON. `FLYDSL_OPUS_YIELD_NOP` is parsed by `flash_attn_opus.py` but currently unused; no `s_nop 15; s_nop 7` sequence is emitted. The OPUS K/V LDS layout (P7) is not gated — it is the only layout the kernel supports after the atomic port.

The remaining gap is performance, not behavior. Differences fall into three buckets:

1. **Mechanically equivalent** (no impact on emitted ISA): C++ template metaprogramming vs FlyDSL imperative helpers.
2. **Semantically different but correct**: a few helpers are split differently (e.g. `attn_mask_vec2_imm` is two single-output asm calls instead of one 4-output asm call), or use slightly different MFMA decompositions (FlyDSL reads V in 4 substeps × 4 D-chunks; C++ uses a single `tr_load`).
3. **Known functional gaps**: register pressure (the V hoist required for stagger correctness costs more VGPRs than the C++ version) plus the non-power-of-two line strides 520 / 544 bf16 introducing `v_mad_u64` chains in DMA address arithmetic. These together explain why FlyDSL is currently at ~6 % of C++ throughput despite identical structure.

---

## 2. What P1–P7 Aligned

| Phase | Commit | OPUS feature ported | C++ lines | FlyDSL lines |
|---|---|---|---|---|
| **P1** | `ad1fcaf` | 8-cluster `j += 2` main loop + 14-cluster (0..13) epilogue; loop-carry of `v_s_0_partial`; prologue/epilogue ladder | 397–754 | 909–1559 |
| **P2** | `558b77c` | In-flight Q scaling (`v_q *= (1/√D)·log2e` in prologue) → log2-space softmax → per-element softmax scaling collapses to subtract | 404–406, 488, 552 | 631–655 (Q load), 846–855 (`_sub_row_first_half_exp`) |
| **P3** | `39feb59` | `sched_barrier_pairs<P,V,G>` and `sched_barrier_exp_pairs<P,E,G>` template helpers in 8 loop / 8 epilogue groups | 14–30, 455–720 | 286–304, 1034–1505 (call sites) |
| **P4** | `e14cb5e` | `v_cmp_lt_i32_e64 + v_cndmask_b32_e64` immediate-threshold inline asm for the causal mask; 8 `asm volatile("" : "+v"(...))` register anchors at the C++ sites | 233–249, 430, 454, 489, 512, 553, 578, 607, 635 | 408–440, 442–461 (anchor helpers), 769–830 (mask), 974, 1030, 1115, 1147, 1228, 1276, 1332, 1368 (8 anchor sites, `_lo` call line) |
| **P5** | `b62c600`, `598981b` | Wave-group stagger: `if (warp_id/4) s_barrier;` (prologue) + `if (!stagger) s_barrier;` (pre-store); V LDS reads hoisted out of clusters 3/7/11/13 into clusters 2/6/10/12 (this is what makes the stagger race-free) | 308, 415–418, 463, 521, 587, 643, 698, 735, 748–750 | 347–360 (stagger compute), 463–508 (helpers), 950–951 (prologue), 1576–1579 (close-out), 6 hoisted `_read_v_packs_for_k_substep` sites (1051–1053, 1165–1167, 1293–1295, 1384–1386, 1469–1471, 1541–1543) |
| **P6** | `55a135c` | Flip `OPUS_ENABLE_STAGGER` default to ON; `FLYDSL_ENABLE_OPUS_PATH=1` selects the kernel while `FLYDSL_OPUS_*` flags still control ablation toggles | – | 236 (default change) |
| **P7** | `2d07de2` | Atomic port of the OPUS K/V LDS layout: interleaved `[K0][V0][K1][V1]` (68 096 B); `u_gk`/`u_sk` DMA writers with line stride 520 bf16 (K) / 544 bf16 (V); `u_rk` K-read with N-permutation π(m)=(m%8)·8+m/8; `u_gv`/`u_sv` DMA writers; `u_rv` V-read via `ds_read_tr16_b64`; causal-mask thresholds reordered to the π-image | 77–185 (layout helpers), 326–333 (s_k/s_v), 398, 408–409, 436, 441, 461–463, 499–500, 519–521, etc. | 149–196 (trait constants), 524–534 (buffer bases), 564–587 (`coop_dma_k`), 593–616 (`coop_dma_v`), 692–709 (`_read_k_packs_for_buf`), 724–754 (`_read_v_packs_for_k_substep`), 769–830 (`_causal_mask_inplace`) |

After P7 the kernel reaches **MaxErr 3.91 × 10⁻³ numerical parity with OPUS C++** on the user-canonical config `B=16 S=8192 H=64 D=128 bf16 causal`, and the documented configurations in `tests/kernels/test_flash_opus_attn.py` PASS (`MaxErr < 1e-2 ∧ MinCos > 0.99`).

---

## 3. Current Structural Map (Cluster-by-Cluster)

Both kernels follow the same cluster layout. Below shows the alignment after P1–P7. The K LDS reads use OPUS `u_rk` (with N-permutation π) and V LDS reads use OPUS `u_rv` via `ds_read_tr16_b64`, all on the interleaved `[K0][V0][K1][V1]` buffer arrangement (P7).

```
                       OPUS C++ template lines     FlyDSL Python lines
Prologue (P1–P13 in source, 11 mapped stages):
  async_load K[0] via u_gk      398                  912    (P7 coop_dma_k)
  s_waitcnt(0) + s_barrier      399-401              915-917
  v_q = load + scale            403-406              631-655  (P2 in-flight scale)
  async_load K[1] + V[0]        408-409              926-927  (P7 coop_dma_k/v)
  v_k = u_rk read s_k[0]        410                  930      (P7 _read_k_packs_for_buf)
  sched_barrier + waitcnt       411-413              934-936
  STAGGER if (stagger)          415-418              950-951  (P5)
  mma0 → v_s[0] + mask          420-427              957-964  (P7 permuted mask)
  m_row + sub_row + exp         428-431              967-974  (P4 anchor #1)
  s_barrier                     433                  978
  async_load K[2]               436                  982

Main loop (j = 3; j < max-1; j += 2, 8 clusters / iter):
  Cluster 0  async_load V[j-2] u_gv + u_rk K[j-2] + waits  441-447   1013-1019
  Cluster 1  mma0 v_s[1] + finish softmax v_s[0]           449-459   1023-1038
  Cluster 2  async_load K[j] + u_rv V[j-3] (HOISTED)       461-468   1050-1058  (P5 V hoist, P7 u_rv)
  Cluster 3  GEMM2 step_k(0..3) + lazy rescale             470-496   1062-1126  (P3 sched hints)
  Cluster 4  async_load V[j-1] u_gv + u_rk K[j-1] + waits  498-505   1130-1136
  Cluster 5  mma0 v_s[0] + finish softmax v_s[1]           507-517   1140-1154
  Cluster 6  async_load K[j+1] + u_rv V[j-2] + mask        519-532   1164-1176  (P5 V hoist, P7)
  Cluster 7  GEMM2 step_k(0..3) + lazy rescale             534-560   1180-1239  (P3 sched hints)

Epilogue (14 stages, 0..13):
  Cluster 0   async_load V[max-3] + u_rk K[max-3]          565-571   1259-1265
  Cluster 1   mma0 v_s[1] + finish softmax v_s[0]          573-583   1269-1283
  Cluster 2   async_load K[max-1] + u_rv V[max-4] + mask   585-598   1292-1304  (P5 V hoist, P7)
  Cluster 3   FULL GEMM2 + rescale + first-half exp        600-618   1308-1346  (P3, P4 anchor #7)
  Cluster 4   async_load V[max-2] + u_rk K[max-2]          620-627   1350-1356
  Cluster 5   mma0 v_s[0] + finish softmax v_s[1]          629-640   1360-1375
  Cluster 6   u_rv V[max-3] + mask                         642-654   1384-1395  (P5 V hoist, P7)
  Cluster 7   FULL GEMM2 + rescale                         656-673   1399-1432  (P3)
  Cluster 8   async_load V[max-1] + u_rk K[max-1]          675-682   1436-1442
  Cluster 9   mma0 v_s[1] + finish softmax v_s[0]          684-695   1446-1458
  Cluster 10  u_rv V[max-2] + mask                         697-709   1469-1480  (P5 V hoist, P7)
  Cluster 11  FULL GEMM2 + FULL exp + cast P[max-1]        711-732   1486-1534  (P3)
  Cluster 12  u_rv V[max-1] + s_waitcnt + barrier          735-739   1541-1547  (P5 V hoist, P7)
  Cluster 13  Final mma1                                   742       1552-1559

Close-out:
  Normalize O *= 1/l_row                                   745-746   1563-1564
  STAGGER if (!stagger) s_barrier                          748-750   1576-1579  (P5)
  store O bf16                                             752-754   1581-1590
```

---

## 4. Remaining Differences After P1–P7

### 4.1 Mechanically Equivalent (different syntax, same intent)

These are differences in how the same construct is expressed, with no impact on the emitted ISA:

| Aspect | OPUS C++ | FlyDSL Python |
|---|---|---|
| Main loop | `for (int j = 3; j < max_num_tiles - 1; j += 2)` | `for j, loop_args in range(fx.Index(3), max_num_tiles - fx.Index(1), fx.Index(2), init=init_args):` (yields loop-carried state) |
| Loop-carried state | Mutable local variables (`v_s[2]`, `v_o`, `m_row`, `l_row`) | Explicit `init_args` tuple → unpacked at iter start, `yield yield_args` at iter end |
| Compile-time loop unrolling | `opus::static_for<N>([&](auto i) { ... })` | `for i in range_constexpr(N): ...` |
| MFMA dispatch | `mma0(v_q, v_k)` (returns 16-lane vector) | `_gemm0(k_lo, k_hi)` Python helper returning two halves |
| Vector concatenation | `vector_t<float, 16>` | `Vec.from_elements(list, fx.Float32)` and `[Vec(v)[r] for r in range_constexpr(16)]` |
| LDS pointer construction | `__shared__ char smem_buf[...]; smem<bf16> s_k[2] = {smem, smem+OPUS_KV_PER_BUFFER}; smem<bf16> s_v[2] = {smem+SMEM_K_TILE_ELEMS, smem+OPUS_KV_PER_BUFFER+SMEM_K_TILE_ELEMS};` | `SmemAllocator` + `SmemPtr` produces a single LDS region; `k_buf_base(buf_id)` returns `OPUS_K_BUF_BASE[buf_id] ∈ {0, OPUS_KV_PER_BUFFER}` and `v_buf_base(buf_id)` returns `OPUS_V_BUF_BASE[buf_id] ∈ {SMEM_K_TILE_ELEMS, OPUS_KV_PER_BUFFER + SMEM_K_TILE_ELEMS}` (same offsets, different syntax). |
| LDS layout algebra | `opus::make_layout(shape, stride, coord)` compile-time tuple algebra (`make_layout_gk_gv`, `make_layout_sk_sv`, `make_layout_rk`, `make_layout_rv`) | Hand-rolled per-lane index computation that **uses the same trait constants** (`SMEM_K_LINE_STRIDE`, `SMEM_V_LINE_STRIDE`, `OPUS_URK_*`, `OPUS_URV_*`) derived from the OPUS shape/stride tuples (P7). |
| DMA loaders | `async_load<T::VEC_KV>(g_k, s_k[i].ptr, u_gk, u_sk, kv_tile(j))` | `coop_dma_k(tile_start, buf_id)` issues `NUM_DMA_K` × `raw_ptr_buffer_load_lds` per warp; per-lane offset math matches `u_gk`/`u_sk` exactly (P7). |
| Scheduler hint helpers | `template<int P, int V, int G> sched_barrier_pairs()` recursive C++ template | `_sched_barrier_pairs(pairs, valu_cnt, group)` Python for-loop emits `pairs` × `(rocdl.sched_group_barrier(MFMA_MASK,1,G), rocdl.sched_group_barrier(VALU_MASK,V,G))` |
| Builtins | `__builtin_amdgcn_s_barrier()`, `__builtin_amdgcn_s_setprio(1)`, `__builtin_amdgcn_exp2f(x)`, `__builtin_amdgcn_ballot_w64(p)`, `__builtin_amdgcn_read_exec()` | `gpu.barrier()`, `rocdl.s_setprio(1)`, `rocdl.exp2(T.f32, x)`, `rocdl.ballot(T.i64, p)`, comparison against `fx.Int64(-1)` (which is what `read_exec()` returns when every lane is active) |
| Bit-cast | `std::bit_cast<u32_t>(x)` | `_bitcast_i32(value) = fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))` |
| `permlane32_swap` reduction | `__builtin_amdgcn_permlane32_swap` returns `<u32×2>` | `fx.Float32(v).shuffle_xor(32, 64)` returns swapped peer value; `_fmax(local, peer)` gives the same answer |

**Result:** Identical LDS addressing at every load/store site (verified by manual derivation from the OPUS shape/stride/coord tuples). The only remaining ISA-visible difference is *how many separate SSA values* the FlyDSL form exposes vs the C++ vector form (see §4.2 and §4.3).

### 4.2 Semantically Different but Functionally Correct

These produce different ISA but mathematically equivalent results within the documented numerical tolerances:

#### 4.2.1 Causal mask: 4-output asm split into 2 × 2-output asm

**OPUS C++** (`attn_mask_vec2_imm`, lines 233–249) emits one asm block with 4 outputs (`x_mask`, `y_mask`, `x_ref`, `y_ref`):

```cpp
asm volatile(
    "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
    "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
    "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
    "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
    : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
    : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
      "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
    : "vcc"
);
```

**FlyDSL Python** splits this into two `_attn_mask_imm_single(...)` calls (one for `x`, one for `y`) because MLIR's `llvm.inline_asm` with **multiple `"=s"` SGPR-pair outputs** in a single struct return proved brittle (silently corrupted the SGPR class assignment during register allocation):

```python
def _attn_mask_imm_single(rel_i32, neg_inf_i32, thr, x_ref_i32):
    asm_str = (
        f"v_cmp_lt_i32_e64 $1, $2, {int(thr)}\n\t"
        "v_cndmask_b32_e64 $0, $3, $4, $1"
    )
    constraints = "=v,=&s,v,v,v"
    ...
```

**ISA impact:** Same `v_cmp_lt_i32_e64 + v_cndmask_b32_e64` pair pattern, but two more SGPR allocations and an extra anti-dep edge between the x-mask and y-mask reuse of the same SGPR pair (mitigated by `=&s` early-clobber).

#### 4.2.2 V LDS read decomposition (`tr_load` vs substep helper)

Both kernels now use the OPUS `u_rv` layout (P7), but they materialize it differently:

**OPUS C++** does one `tr_load<T::VEC_TR_V>(s_v[i], u_rv)` per GEMM2 cluster (lines 463 / 521 / 587 / 643 / 698 / 735). `tr_load` is a layout-aware wrapper that materializes the *entire* `vtype_b` (a 32-element bf16 vector covering all 4 substeps × 4 D-chunks) using 32 internal `ds_read_tr16_b64` issues + an internal shuffle, **producing one contiguous SSA value** that LLVM can pack into a contiguous VGPR run.

**FlyDSL Python** decomposes the same `u_rv` layout into a Python loop of 4 substeps × 4 D-chunks, each call emitting `2 × ds_read_tr_v4f16` + `Vec.shuffle`. The per-lane base offset, axis strides, and inner step strides are all derived from the OPUS `make_layout_rv` (kernel source lines 724-754):

```python
urv_base_per_lane = (
      (lane // 32)        * OPUS_URV_GRPK         # 2176 bf16 (4*544) — grp_k axis 2
    + ((lane % 16) // 4)  * OPUS_URV_LANE_HI      #  544 bf16        — lane_hi axis 3
    + ((lane // 16) % 2)  * OPUS_URV_GRP_N        #   16 bf16        — grp_n axis 6
    + (lane % 4)          * OPUS_URV_LANE_LO      #    4 bf16        — lane_lo axis 7
)

def _read_v_packs_for_k_substep(buf_id, k_substep):
    step_k_off = k_substep * OPUS_URV_STEP_K_STRIDE       # 128 bf16
    packs = []
    for dc in range_constexpr(D_CHUNKS):                  # 4 D-chunks
        i_0, i_1 = dc // 2, dc % 2
        dc_off = i_0 * OPUS_URV_DC_AXIS0 + i_1 * OPUS_URV_DC_AXIS1
        lds_off_a = v_base + urv_base_per_lane + step_k_off + dc_off
        lds_off_b = lds_off_a + OPUS_URV_I5_STRIDE        # 64 bf16
        a = ds_read_tr_v4f16(lds_off_a)                   # 4 bf16 / lane
        b = ds_read_tr_v4f16(lds_off_b)
        packs.append(Vec(a).shuffle(Vec(b), [0..7]).ir_value())  # 8-bf16 pack
    return packs

# Called 4 times per consumer cluster, 16 packs total
v_packs_a = [_read_v_packs_for_k_substep(0, kss) for kss in range_constexpr(4)]
```

**ISA impact:** Same number of `ds_read_tr16_b64` instructions (`4 substeps × 4 D-chunks × 2 = 32` per V hoist site) and same shuffle pattern, but the FlyDSL form exposes **16 separate Python-level packs** (`<8 × bf16>` each) where the C++ `tr_load` exposes one contiguous `vtype_b`. This gives LLVM:
- **More freedom to reschedule** individual reads against waitcnts (a small positive);
- **Less ability to alias the V scratch with the MFMA accumulator's VGPR run** (a significant negative — see §4.3.1).

#### 4.2.3 LDS pointer construction (interleaved K/V buffers)

After P7 both kernels use the **same interleaved buffer layout** `[K0][V0][K1][V1]`, but expressed slightly differently:

**OPUS C++** uses four separate `smem<D_ATTN>` objects (`gqa_d128_kernel_template.hpp` lines 326-333):

```cpp
__shared__ char smem_buf[T::smem_size_bytes()];
smem<D_ATTN> s_k[2] = {
    make_smem(reinterpret_cast<D_ATTN*>(smem_buf)),
    make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_buffer_elems)
};
smem<D_ATTN> s_v[2] = {
    make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_k_tile_elems),
    make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_buffer_elems
                                                  + T::smem_k_tile_elems)
};
```

**FlyDSL Python** uses a single `SmemPtr` to a contiguous 68 096 B LDS region and computes the four buffer bases from constants derived from the same OPUS layout (kernel source lines 173-179):

```python
OPUS_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS   # 17 024 bf16
OPUS_K_BUF_BASE    = (0, OPUS_KV_PER_BUFFER)                  # K0=0,   K1=17024
OPUS_V_BUF_BASE    = (SMEM_K_TILE_ELEMS,                       # V0=8320,
                      SMEM_K_TILE_ELEMS + OPUS_KV_PER_BUFFER)  # V1=25344

def k_buf_base(buf_id):  return fx.Index(OPUS_K_BUF_BASE[buf_id])
def v_buf_base(buf_id):  return fx.Index(OPUS_V_BUF_BASE[buf_id])
```

The four buffer addresses (`{0, 8320, 17024, 25344}` bf16) are **identical to the C++ layout** — that is the heart of the P7 atomic port. Both forms emit `ds_read_*` and `buffer_load_dwordx4_lds` with statically-known base offsets folded into the immediate field — zero runtime cost.

**ISA impact:** None. The only difference is whether the four buffer offsets live in four `smem<>` objects (C++) or in a Python tuple looked up by `buf_id` (FlyDSL).

#### 4.2.4 Lazy rescale: select-based vs branch-based

**OPUS C++**:

```cpp
if (__builtin_expect(all_below, 1)) {
    row_max = m_row;            // skip rescale
} else {
    rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
    scale_output_tile<T>(v_o, rescale_m);
    l_row *= rescale_m;
    m_row = row_max;
}
v_o = mma1.step_k(1_I, v_p, v_v, v_o);   // 3 more MFMAs continue regardless
```

The `__builtin_expect(all_below, 1)` annotation hints to LLVM that the "below" path is hot, biasing the layout so the skip path is the fall-through and the rescale path becomes a forward branch.

**FlyDSL Python** rewrites this as an unconditional FMA chain gated by `select`:

```python
if const_expr(OPUS_LAZY_RESCALE):
    below_a = ArithValue(fx.Float32(m_diff_a) <= c_eight_f)
    ballot_a = rocdl.ballot(T.i64, _raw(below_a))
    all_below_a = fx.Int64(ballot_a) == fx.Int64(-1)
    ab_a = ArithValue(all_below_a)
    m_new_a = ab_a.select(m_row, _fmax(m_row, m_tile_max_a))
    corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
    eff_corr_a = ab_a.select(c_one_f, corr_a)

_scale_o(o_accs, eff_corr_a)   # always executes; relies on eff_corr=1 → no-op when below
l_row = _fmul(l_row, corr_a)
m_row = m_new_a
```

**ISA impact:** Same exp2 and the same `o_accs *= eff_corr` MFMA-pipe loaded multiplications, but FlyDSL **always** runs the scale path (relying on `eff_corr = 1.0` to be a numerical no-op). This costs ~16 extra `v_pk_mul_f32` per skipped iteration but avoids branch divergence within a wave. On gfx950 where VALU and MFMA can dual-issue with very low branch-cost, this is normally a wash; on rare workloads the extra mul edges can occupy the wave's primary pipe and delay GEMM2.

### 4.3 Known Functional Gaps

These are differences that produce **measurable** behavior or perf delta:

#### 4.3.1 V VGPR live range under stagger

The P5 stagger fix (commit `598981b`) hoists 16 V packs into Cluster 2/6/10/12 of the corresponding GEMM2. Each V pack is `<8 × bf16>` (= 1 VGPR for the bf16x8 representation, but split into 2 VGPRs during MFMA prep). Across the cluster boundary every V pack stays live in VGPR space. The C++ template's `tr_load` is similarly hoisted, but because it returns a single `vtype_b` (typically packed into a contiguous VGPR run), LLVM is able to alias the V scratch back into the MFMA accumulator's VGPRs when scheduling.

In FlyDSL the 16 separate Python-level packs (4 substeps × 4 D-chunks) prevent that aliasing (LLVM sees 16 different SSA values rather than one), so the VGPR count is higher under stagger=ON. The P7 atomic layout port did *not* change the per-call decomposition of `_read_v_packs_for_k_substep` (still 4 D-chunk packs per substep), so this gap is unchanged from the pre-P7 baseline.

| Path | VGPR live across cluster bound | causal TFLOPS (B=16 S=8192 H=64 D=128 bf16, MI355X) |
|---|---|---|
| FlyDSL pre-P7, stagger=0 (XOR-swizzle, lockstep) | low | 108.5 |
| FlyDSL pre-P7, stagger=1 (XOR-swizzle, V hoist) | high | 85.0 |
| **FlyDSL post-P7, stagger=1 (OPUS layout, V hoist, default)** | **high** | **69.6** |
| OPUS C++ (single `tr_load` per GEMM2) | low | ~1131 |

P7 traded ~15 TFLOPS for full structural alignment with the OPUS C++ source (XOR-swizzle 85.0 → OPUS-layout 69.6 at stagger=1). The OPUS layout's non-power-of-two line strides 520 / 544 bf16 add `v_mad_u64` chains to the DMA writers (where the pre-P7 power-of-two strides 128 bf16 folded into `v_lshl_add`). This is the principal cost of the layout port; closing it requires:

- **(a)** Coalescing the 16 packs into a single fused `Vec` representation that LLVM treats as one SSA value (recovers the C++ `tr_load` aliasing); or
- **(b)** Splitting the V read further so the live range is interleaved with the GEMM2 MFMA chain instead of contiguous before it (reduces VGPR pressure); or
- **(c)** Constant-folding the line-stride multiplies by hoisting per-buffer base addresses into SGPR before the main loop (as `s_k[i].ptr` in C++).

#### 4.3.2 Scheduling hint granularity

OPUS C++ uses 16 `sched_group_barrier` calls per iteration spread between MFMA pairs. The FlyDSL kernel emits the exact same number at the same syntactic positions, but the LLVM scheduler has more freedom because every helper, MFMA pack, and reference-count is a separate SSA value (vs. C++ template-fused operands). Empirically this lets the scheduler **regroup** the MFMAs more aggressively than the C++ author intended — sometimes beneficially, sometimes not. This is partly responsible for run-to-run perf jitter on FlyDSL but not for the steady-state gap to C++.

#### 4.3.3 Q load: extra f32x8 staging

OPUS C++ does the P2 scaling on the raw `v_q` register vector:

```cpp
v_q = load<T::VEC_Q>(g_q, u_q);
auto v_q_f32 = opus::cast<float>(v_q);
static_for<q_len>([&](auto i) { v_q_f32[i.value] *= temperature_scale; });
v_q = opus::cast<D_ATTN>(v_q_f32);
```

FlyDSL emits an equivalent f32x8 stage per MFMA pack but in a Python loop:

```python
for ks in range_constexpr(K_STEPS_QK):     # 8 packs
    raw = load_global_mfma_pack(q_ptr, g_idx)
    raw_safe = ArithValue(q_in_bounds).select(raw, c_zero_mfma_pack)
    pack_f32 = Vec(raw_safe).extf(v8f32_type)
    scaled_elems = []
    for k in range_constexpr(MFMA_LANE_K):
        scaled_elems.append(_fmul(Vec(pack_f32)[k], c_sm_scale_log2e))
    pack_scaled_f32 = Vec.from_elements(scaled_elems, fx.Float32)
    pack_scaled_bf16 = pack_scaled_f32.truncf(v8f16_type)
    q_b_packs.append(pack_scaled_bf16.ir_value())
```

Same number of `v_pk_mul_f32` + `v_cvt_pkrtz_bf16_f32` operations, but the SSA chain is per-element (`scaled_elems[0..7]`) rather than vector-wide, which inflates the LLVM IR-size for this region. No observed ISA impact after `--enable-post-misched=0` (which FlyDSL sets by default for the OPUS path); the LLVM scheduler reconstitutes the vector form.

#### 4.3.4 Causal mask: unconditional

OPUS C++ guards the causal mask call with an early-exit when the entire tile is below the diagonal:

```cpp
if constexpr (T::CAUSAL) {
    const int kv_end_pos = j * T::KV_TILE_SIZE;
    if (q_start_pos < kv_end_pos) {                  // ← runtime gate
        attn_mask_causal_tile<T>(v_s[0], q_start_pos, j - 1, neg_inf_v, lane_id);
    }
}
```

FlyDSL applies the mask unconditionally for any iteration that *might* need it. The mask is a no-op (`rel < thr` evaluates false everywhere) when the entire tile is below the diagonal, but the 16 `v_cmp + v_cndmask` pairs are still issued. This costs ~30 unused VALU cycles per "deep below diagonal" iteration. The runtime gate is hard to express cleanly in FlyDSL because the if-condition contains the loop induction value `j_idx`, and the body assigns to a Python list (`s_lo`, `s_hi`) used afterward — which would force the FlyDSL AST rewriter into `scf.if` + nested function semantics that don't propagate the list mutation.

---

## 5. Side-by-Side Code Mapping

### 5.1 In-flight Q scaling (P2)

```
[OPUS C++  lines 403-406]                       [FlyDSL flash_attn_opus.py lines 631-655]
─────────────────────────────────                ──────────────────────────────────────────
v_q = load<T::VEC_Q>(g_q, u_q);                  q_row_safe = ArithValue(q_in_bounds).select(q_row, fx.Index(0))
auto v_q_f32 = opus::cast<float>(v_q);           q_b_packs = []
static_for<q_len>([&](auto i) {                  for ks in range_constexpr(K_STEPS_QK):
    v_q_f32[i.value] *= temperature_scale;           ...
});                                                  raw = load_global_mfma_pack(q_ptr, g_idx)
v_q = opus::cast<D_ATTN>(v_q_f32);                   raw_safe = ArithValue(q_in_bounds).select(raw, c_zero_mfma_pack)
                                                     pack_f32 = Vec(raw_safe).extf(v8f32_type)
                                                     scaled_elems = []
                                                     for k in range_constexpr(MFMA_LANE_K):
                                                         scaled_elems.append(_fmul(Vec(pack_f32)[k], c_sm_scale_log2e))
                                                     pack_scaled_f32 = Vec.from_elements(scaled_elems, fx.Float32)
                                                     pack_scaled_bf16 = pack_scaled_f32.truncf(v8f16_type)
                                                     q_b_packs.append(pack_scaled_bf16.ir_value())
```

Both produce the same ISA: 8× `flat_load_dword_x4` (Q) + 64× `v_pk_mul_f32` (scale) + 8× `v_cvt_pkrtz_bf16_f32` (cast back).

### 5.2 Lazy rescale (P1)

```
[OPUS C++ lines 475-484]                                [FlyDSL flash_attn_opus.py lines 1081-1097]
────────────────────────────────────                    ──────────────────────────────────────────────
bool below_thresh = ((row_max - m_row) <= 8.0f);        m_diff_a = _fsub(m_tile_max_a, m_row)
bool all_below = (__builtin_amdgcn_ballot_w64(           if const_expr(OPUS_LAZY_RESCALE):
                  below_thresh) ==                          below_a = ArithValue(fx.Float32(m_diff_a) <= c_eight_f)
                 __builtin_amdgcn_read_exec());             ballot_a = rocdl.ballot(T.i64, _raw(below_a))
                                                            all_below_a = fx.Int64(ballot_a) == fx.Int64(-1)
if (__builtin_expect(all_below, 1)) {                       ab_a = ArithValue(all_below_a)
    row_max = m_row;                                        m_new_a = ab_a.select(m_row, _fmax(m_row, m_tile_max_a))
} else {                                                    corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
    rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);    eff_corr_a = ab_a.select(c_one_f, corr_a)
    scale_output_tile<T>(v_o, rescale_m);                else:
    l_row *= rescale_m;                                     m_new_a = _fmax(m_row, m_tile_max_a)
    m_row = row_max;                                        corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
}                                                           eff_corr_a = corr_a
                                                        _scale_o(o_accs, eff_corr_a)
                                                        l_row = _fmul(l_row, corr_a)
                                                        m_row = m_new_a
```

Functional difference: C++ branches; FlyDSL uses select values and still calls `_scale_o(o_accs, eff_corr_a)`. With `FLYDSL_OPUS_LAZY_RESCALE=0`, FlyDSL drops the ballot/select path and always uses the non-lazy `m_new=max(...)` / `corr` path. See §4.2.4.

### 5.3 sched_group_barrier in main loop Cluster 1 (P3)

```
[OPUS C++ lines 455-456]                       [FlyDSL flash_attn_opus.py lines 1034-1035]
─────────────────────────────────              ───────────────────────────────────────────
sched_barrier_exp_pairs<6, 3, 1>();            _sched_barrier_exp_pairs(6, 3, 1)
sched_barrier_pairs<10, 5, 1>();               _sched_barrier_pairs(10, 5, 1)
```

The C++ recursive template `sched_barrier_pairs<P, V, G>()` unrolls into `P` calls of `(sched_group_barrier(MFMA,1,G); sched_group_barrier(VALU,V,G))`. The FlyDSL helper does the same via a Python for-loop — same ISA emitted.

### 5.4 Causal mask (P4 inline-asm + P7 permuted thresholds)

The inline-asm primitive `attn_mask_vec2_imm` (4-output asm in C++, split into two 2-output asm calls in FlyDSL) is the same as before P7. What changed in P7 is the **per-r threshold sequence** that `_causal_mask_inplace` iterates: it now follows the OPUS N-permutation π(m) = (m%8)·8 + m/8 applied to the MFMA-A lane index, and the strip-relative N offset for `v_s_hi` shifted from +32 to +4 (matching OPUS `smem_d_n_split = 4`).

C++ calls the wrapper only under tile-level `q_start_pos < kv_end_pos` guards (prologue, main loop, and epilogue). FlyDSL currently applies `_causal_mask_inplace` whenever `CAUSAL=True`, so below-diagonal tiles still execute no-op mask instructions.

```
[OPUS C++ lines 233-249 (asm primitive)]          [FlyDSL flash_attn_opus.py lines 408-440]
─────────────────────────────────────             ─────────────────────────────────────────────────────
template<int THR_X, int THR_Y>                    def _attn_mask_imm_single(rel_i32, neg_inf_i32, thr, x_ref_i32):
__device__ inline void                                asm_str = (
attn_mask_vec2_imm(opus::u32_t rel, ...) {            f"v_cmp_lt_i32_e64 $1, $2, {int(thr)}\n\t"
    asm volatile(                                     "v_cndmask_b32_e64 $0, $3, $4, $1"
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"           )
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"           # ...   # split into two single-output asm calls
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"      # See §4.2.1.
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), ...);
}

[OPUS C++ lines 251-290 (per-tile glue)]          [FlyDSL flash_attn_opus.py lines 769-830]
─────────────────────────────────────             ─────────────────────────────────────────────────────
template<typename T, typename V>                   def _causal_mask_inplace(s_lo, s_hi, tile_idx):
__device__ inline void                                 kv_tile_start = tile_idx * fx.Index(BLOCK_N)
attn_mask_causal_tile(V& v_s, ...) {                   kv_start_i32  = fx.Int32(kv_tile_start)
    const int q_pos = q_start_pos + (lane_id % T::W_M);
    const int k_start_pos = kv_tile_idx * T::KV_TILE_SIZE;     # OPUS grpm offset is +32 per lane_div_32
    const int lane_group = lane_id / T::W_M;                   lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(32)
                                                               rel_lo_i32 = fx.Int32(q_row_i32 - kv_start_i32 - lane_off_i32)
    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {             # OPUS smem_d_n_split = 4 → v_s_hi delta is -4
        ...                                                    rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(4))
        opus::static_for<c_rept>([&](auto i_rept) {
            ...                                                # OPUS-permuted thresholds (π-image of {0..27}):
            opus::static_for<c_pack/2>([&](auto i_pair) {      pair_thresholds = [
                constexpr int thr_x =                              (0, 8), (16, 24),
                  i_rept.value * c_rept_stride                     (1, 9), (17, 25),
                  + i_pair.value * 2;                              (2,10), (18, 26),
                constexpr int thr_y = thr_x + 1;                   (3,11), (19, 27),
                attn_mask_vec2_imm<thr_x, thr_y>(              ]
                  rel, neg_inf_v, x_ref, y_ref);
            });                                                for thr_x, thr_y in pair_thresholds:
        });                                                        s_lo[idx_x], s_lo[idx_y] = _attn_mask_vec2_imm(
    });                                                                rel_lo_i32, neg_inf_i32, thr_x, thr_y, ...)
}                                                                  s_hi[idx_x], s_hi[idx_y] = _attn_mask_vec2_imm(
                                                                       rel_hi_i32, neg_inf_i32, thr_x, thr_y, ...)
```

After P7 **both kernels mask the same set of (M, N) elements** because the C++ template's `attn_mask_causal_tile` (`gqa_d128_kernel_template.hpp` lines 251-290) implicitly applies the same π permutation through its `c_rept_stride = (WARP_SIZE / W_M) * c_pack = 8` and the `thr = i_rept * 8 + i_pair * 2` formula — which evaluates to exactly the OPUS-permuted `{0, 2, 8, 10, 16, 18, 24, 26}` set (then `+1` for thr_y), the same as the FlyDSL pair list when paired against the right `s_lo[idx]` positions.

Both materialize `(v_cmp_lt_i32_e64, v_cndmask_b32_e64)` with the threshold as an immediate. FlyDSL serializes the 4-op block into two 2-op blocks. See §4.2.1 for that asm-level split.

### 5.5 Stagger barrier (P5)

```
[OPUS C++ lines 415-418]                        [FlyDSL flash_attn_opus.py lines 463-487, 950-951]
────────────────────────────────                ───────────────────────────────────────────────────
if (stagger) {                                  def _stagger_extra_barrier_if_one():
    __builtin_amdgcn_sched_barrier(0);              rocdl.sched_barrier(0)
    __builtin_amdgcn_s_barrier();                   llvm.inline_asm(
}                                                       ir.Type.parse("!llvm.void"),
                                                        [_stagger_i32],
                                                        ("s_cmp_eq_u32 $0, 0\n\t"
                                                         "s_cbranch_scc1 1f\n\t"
                                                         "s_barrier\n\t"
                                                         "1:"),
                                                        "s",                  # SGPR-resident input
                                                        has_side_effects=True,
                                                    )

                                                # Call site
                                                if const_expr(OPUS_ENABLE_STAGGER):
                                                    _stagger_extra_barrier_if_one()
                                                else:
                                                    rocdl.sched_barrier(0)
                                                    gpu.barrier()
```

The FlyDSL form uses raw inline asm because MLIR's `scf.if` + `gpu.barrier()` was getting CSE'd into a single unconditional barrier during the GPU-to-ROCDL conversion (see commit `b62c600` for the diagnostic trail).

### 5.6.0 OPUS K LDS read `u_rk` (P7, with N-permutation π)

```
[OPUS C++ lines 122-148 (make_layout_rk)]      [FlyDSL flash_attn_opus.py lines 692-709]
─────────────────────────────────────────      ─────────────────────────────────────────────
template<typename T>                            urk_base_per_lane = (
__device__ inline auto make_layout_rk(            (lane_mod_32 %  8) * SMEM_K_LINE_STRIDE   # axis "m % 8"
    int lane_id) {                                + (lane_mod_32 // 8) * D_128B_SIZE        # axis "m / 8"
    constexpr auto rk_block_shape =               + lane_div_32        * VEC_KV             # lane / 32
      opus::make_tuple(                       )
        opus::number<GEMM0_E_N / n_grp>{},
        opus::number<NUM_WARPS>{},               def _read_k_packs_for_buf(buf_id):
        opus::number<n_grp>{},                       k_base = k_buf_base(buf_id)
        opus::number<W_N / NUM_WARPS>{},            k_lo = [None] * K_STEPS_QK
        opus::number<smem_d_rpt>{},                 k_hi = [None] * K_STEPS_QK
        opus::number<GEMM0_E_K / smem_d_rpt>{},     for ks in range_constexpr(K_STEPS_QK):
        opus::number<WARP_SIZE / W_N>{},                ks_offset = (
        opus::number<VEC_KV>{});                          (ks // 4) * OPUS_URK_KSTEP_OUTER      # 4160
    ...                                                  + (ks % 4) * OPUS_URK_KSTEP_INNER     # 16
    return opus::make_layout(rk_block_shape,         )
      ..., opus::tuple{lane_id_n % NUM_WARPS,         idx_lo = k_base + urk_base_per_lane + fx.Index(ks_offset)
                       lane_id_n / NUM_WARPS,         idx_hi = idx_lo + fx.Index(OPUS_URK_N_STRIP_STRIDE)  # +256
                       lane_id / W_N});               k_lo[ks] = Vec.load(mfma_pack_type, lds_kv, [idx_lo])
}                                                     k_hi[ks] = Vec.load(mfma_pack_type, lds_kv, [idx_hi])
                                                  return k_lo, k_hi

// At call site (line 410):
v_k = load<T::VEC_KV>(s_k[0], u_rk);
```

Both forms compute the same per-lane LDS offset:

```
offset(lane, ks, n_strip) = (lane%32%8) * 520
                          + (lane%32/8) * 64
                          + (lane/32)   * 8
                          + (ks//4)     * 4160
                          + (ks%4)      * 16
                          + n_strip     * 256
```

This produces the **N-permutation π(m) = (m%8)·8 + m/8** on the MFMA A operand: lane `m = lane%32` reads `K[N = π(m), D = ks*16 + (m/8)*8 + 0..7]` (n_strip=0). The C++ `make_layout_rk` derives the same permutation from its shape/stride/coord tuples; the FlyDSL form spells it out directly with the named constants `OPUS_URK_*` (kernel source lines 180-185).

### 5.6 V hoist (P5 correctness fix + P7 OPUS `u_rv`)

```
[OPUS C++ lines 461-468, 470-472]              [FlyDSL flash_attn_opus.py lines 1050-1069]
─────────────────────────────────              ────────────────────────────────────────────────────
// Cluster 2:                                  # ─── Cluster 2 ───
async_load<T::VEC_KV>(g_k, s_k[1].ptr,         coop_dma_k(j_idx * fx.Index(BLOCK_N), 1)
                     u_gk, u_sk, kv_tile(j));  v_packs_a = []
v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);      for kss in range_constexpr(4):
s_waitcnt_lgkmcnt(0_I);                            v_packs_a.append(_read_v_packs_for_k_substep(0, kss))
s_waitcnt_vmcnt(...);                          rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
__builtin_amdgcn_sched_barrier(0);             _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
__builtin_amdgcn_s_barrier();                  rocdl.sched_barrier(0)
                                               gpu.barrier()
// Cluster 3:                                  rocdl.sched_barrier(0)
__builtin_amdgcn_s_setprio(1);                 # ─── Cluster 3 ───
v_o = mma1.step_k(0_I, v_p, v_v, v_o);         if const_expr(OPUS_SETPRIO):
                                                   rocdl.s_setprio(1)
                                               v_pk = v_packs_a[0]   # ← uses pre-loaded OPUS u_rv V
                                               p_pk = v_p_lo_a[0]
                                               for dc in range_constexpr(D_CHUNKS):
                                                   o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])
```

Both kernels load V into registers in Cluster 2 (before the cluster-2 `s_barrier`) and consume the registers in Cluster 3 step_k(0..3). The FlyDSL form pre-loads all 4 substeps (16 packs) up front using `_read_v_packs_for_k_substep` (which encodes the same `make_layout_rv` shape/stride formula as the C++ template, P7); the C++ form materializes a single contiguous `vtype_b` via `tr_load`. The underlying `ds_read_tr16_b64` count is identical between the two. See §4.2.2 and §4.3.1 for register-level consequences.

---

## 6. Line-Level Cross-Reference Table

(Line numbers refer to `FlyDSL/kernels/flash_attn_opus.py` post-P7 commit `2d07de2`, total 1655 lines.)

| OPUS C++ lines | Feature | FlyDSL Python lines | Notes |
|---|---|---|---|
| 14–30 | `sched_barrier_pairs` / `sched_barrier_exp_pairs` template helpers | 286–304 (`_sched_barrier_pairs`, `_sched_barrier_exp_pairs`) | P3 — Python helpers emit the same `rocdl.sched_group_barrier` chain |
| 77–99 | `make_layout_gk_gv` (global K/V load layout) | 149–196 (trait constants) + 564–587 (`coop_dma_k`) + 593–616 (`coop_dma_v`) | P7 — same per-lane (warp_id, lane%threads_d, lane/threads_d, d) decomposition expressed as FlyDSL index math |
| 102–118 | `make_layout_sk_sv<T, smem_padding>` (LDS K/V layout) | 149–196 (trait constants) + 564–616 (writers) + 692–754 (readers) | P7 — same line-padded slab layout; constants `SMEM_K_LINE_STRIDE=520`, `SMEM_V_LINE_STRIDE=544` derived from OPUS `smem_padding_{16B,64B}` |
| 122–148 | `make_layout_rk` (K LDS → MFMA-A register layout) | 692–709 (`_read_k_packs_for_buf` + `urk_base_per_lane`) | P7 — applies the same N-permutation π(m)=(m%8)·8+m/8 via `urk_base_per_lane` + `OPUS_URK_*` strides |
| 150–185 | `make_layout_rv` (V LDS → MFMA-A register layout via `ds_read_tr16_b64`) | 724–754 (`_read_v_packs_for_k_substep` + `urv_base_per_lane`) | P7 — same (grp_id, lane_in_grp, dc, step_k) decomposition expressed via `urv_base_per_lane` + `OPUS_URV_*` strides |
| 233–249 | `attn_mask_vec2_imm<THR_X, THR_Y>` inline asm | 413–440 (`_attn_mask_imm_single`, `_attn_mask_vec2_imm`) | P4 — split into two `_attn_mask_imm_single` calls |
| 251–289 | `attn_mask_causal_tile<T>` (per-tile mask glue) | 769–830 (`_causal_mask_inplace`) | P4 + P7 — thresholds reordered to OPUS π-image; `v_s_hi` delta from +32 to +4 (smem_d_n_split=4) |
| 308 | `const int stagger = warp_id / 4;` | 347–360 (`_stagger_i32`) | P5 — materialized via `rocdl.readfirstlane` + `arith.divsi` to SGPR |
| 326–333 | `s_k[2]` / `s_v[2]` interleaved buffer pointer construction | 174–179 (`OPUS_KV_PER_BUFFER`, `OPUS_K_BUF_BASE`, `OPUS_V_BUF_BASE`) + 524–534 (`k_buf_base` / `v_buf_base`) | P7 — same {0, 8320, 17024, 25344} bf16 offsets via `OPUS_KV_PER_BUFFER`-based interleaving |
| 398–401 | Prologue async K[0] + waitcnt(0) + s_barrier | 912–917 | P7 `coop_dma_k` + `gpu.barrier()` |
| 403–406 | Prologue Q load + temperature scale | 631–655 (`q_b_packs` load + scale loop; `c_sm_scale_log2e` defined at 622) | P2 in-flight scaling |
| 408–409 | async K[1] + V[0] | 926–927 | P7 `coop_dma_k` / `coop_dma_v` |
| 410 | `v_k = load<T::VEC_KV>(s_k[0], u_rk)` | 930 (`_read_k_packs_for_buf(0)`) | P7 |
| 411–413 | sched_barrier + lgkm(0) + vmcnt(v_buffer_load_insts) | 934–936 (after `_read_k_packs_for_buf(0)`) | identical |
| 415–418 | `if (stagger) s_barrier` | 950–954 (`_stagger_extra_barrier_if_one` + else branch) | P5 |
| 420 | mma0 of tile 0 | ~957 | identical |
| 422–427 | Causal mask on v_s[0] | ~963–964 | P4 + P7 (permuted) |
| 428–431 | row_max + sub_row + first-half exp + register anchor | ~967–974 | P2 (log2-scale) + P4 anchor #1 |
| 433 | `s_barrier` | 978 (`gpu.barrier()`) | identical |
| 436 | async K[2] | 982 (`coop_dma_k`) | P7 |
| 441–447 | Main-loop Cluster 0 | 1013–1019 | P7 (`coop_dma_v`, `_read_k_packs_for_buf`) |
| 449–459 | Main-loop Cluster 1 | ~1023–1038 | P3, P4 anchor #2 |
| 461–468 | Main-loop Cluster 2 (with V hoist) | 1050–1058 | P5 V hoist, P7 `_read_v_packs_for_k_substep` |
| 470–496 | Main-loop Cluster 3 | ~1062–1126 (`s_setprio` brackets at 1063, 1123) | P1 lazy rescale, P3 sched hints, P4 anchor #3 |
| 498–505 | Main-loop Cluster 4 | 1130–1136 | P7 |
| 507–517 | Main-loop Cluster 5 | ~1140–1154 | P3, P4 anchor #4 |
| 519–532 | Main-loop Cluster 6 (with V hoist + mask) | 1164–1176 | P5 V hoist, P4+P7 mask |
| 534–560 | Main-loop Cluster 7 | ~1180–1239 (`s_setprio` brackets at 1181, 1236) | P1 lazy rescale, P3 sched hints, P4 anchor #5 |
| 564–571 | Epilogue Cluster 0 | 1259–1265 | P7 |
| 573–583 | Epilogue Cluster 1 | ~1269–1283 | P3, P4 anchor #6 |
| 585–598 | Epilogue Cluster 2 (with V hoist) | 1292–1304 | P5 V hoist, P7 |
| 600–618 | Epilogue Cluster 3 (full GEMM2 + rescale) | ~1308–1346 (`s_setprio` brackets at 1309, 1343) | P3, P4 anchor #7 |
| 620–627 | Epilogue Cluster 4 | 1350–1356 | P7 |
| 629–640 | Epilogue Cluster 5 | ~1360–1375 | P3, P4 anchor #8 |
| 642–654 | Epilogue Cluster 6 (with V hoist + mask) | 1384–1395 | P5 V hoist, P4+P7 mask |
| 656–673 | Epilogue Cluster 7 (full GEMM2 + rescale) | ~1399–1432 (`s_setprio` brackets at 1400, 1429) | P3 |
| 675–682 | Epilogue Cluster 8 | 1436–1442 | P7 |
| 684–695 | Epilogue Cluster 9 | ~1446–1458 | P3 |
| 697–709 | Epilogue Cluster 10 (with V hoist + mask) | ~1469–1480 | P5 V hoist, P4+P7 mask |
| 711–732 | Epilogue Cluster 11 (full GEMM2 + full exp) | ~1486–1534 | P3 |
| 735–739 | Epilogue Cluster 12 (with V hoist) | ~1541–1547 | P5 V hoist, P7 |
| 742 | Epilogue Cluster 13 (final mma1) | ~1552–1559 | identical |
| 745–746 | Normalize `v_o /= l_row` | ~1563–1564 | identical |
| 748–750 | `if (!stagger) s_barrier` | 1576–1579 (`_stagger_extra_barrier_if_zero`) | P5 close-out |
| 752–754 | Store O bf16 | 1581–1590 (`_store_global_half` in `for r in range_constexpr(16)`) | identical |

---

## 7. Performance Gap Analysis

Post-P7 performance numbers (B=16 S=8192 H=64 D=128 bf16, MI355X, `FLYDSL_ENABLE_OPUS_PATH=1`, stagger=1 default):

| Path | causal TFLOPS | % of OPUS C++ | nocausal TFLOPS | % of OPUS C++ |
|---|---:|---:|---:|---:|
| FlyDSL `flash_attn_opus` (pre-P7, XOR-swizzle, stagger=0) | 108.5 | 9.6 % | — | — |
| FlyDSL `flash_attn_opus` (pre-P7, XOR-swizzle, stagger=1) | 85.0 | 7.5 % | — | — |
| **FlyDSL `flash_attn_opus` (post-P7, OPUS layout, stagger=1, default)** | **69.6** | **6.2 %** | **66.3** | **5.7 %** |
| FlyDSL `flash_attn_opus` (post-P7, GQA 8:1, default) | 73.5 | 6.5 % | 69.1 | 5.9 % |
| FlyDSL default `flash_attn_func` (non-OPUS, bf16) | 716 | 63.3 % | 640 | 54.9 % |
| OPUS C++ (`mha_fwd` via aiter) | 1131 | 100 % | 1165 | 100 % |
| ASM kernel (`fmha_v3_fwd`) | 596 | 52.7 % | 1249 | 107 % |

Despite close source-structure alignment with OPUS C++ (same 8 clusters, same `sched_group_barrier` count and grouping, same hand-coded inline asm primitives, same stagger barrier pattern, same OPUS LDS layout after P7), the FlyDSL OPUS path is currently at **~6 %** of OPUS C++ throughput. The primary causes are:

1. **Higher VGPR live range during V hoist** (§4.3.1). FlyDSL materializes 16 separate Python-level `<8 × bf16>` packs that LLVM treats as 16 SSA values; OPUS C++ materializes them as one contiguous `vtype_b` register tuple via `tr_load` that LLVM can spill or alias more aggressively. Result: occupancy drops from 4 waves/SIMD to 2.
2. **Non-power-of-two line strides** (P7 cost). The OPUS `SMEM_K_LINE_STRIDE = 520` and `SMEM_V_LINE_STRIDE = 544` bf16 introduce `v_mad_u64` chains into the DMA-writer and LDS-reader address arithmetic, whereas the pre-P7 power-of-two-strided XOR layout folded into `v_lshl_add`. This is the principal reason post-P7 TFLOPS is ~15 lower than pre-P7 stagger=1.
3. **Scheduler less constrained**. The Python helper expansion is more verbose than the C++ template instantiation, so the LLVM scheduler has more freedom to interleave operations across the `sched_group_barrier` boundaries — sometimes choosing a worse schedule than the C++ author intended.
4. **Unconditional causal mask**. FlyDSL applies the mask on every causal iteration; C++ early-exits when `q_start_pos >= kv_end_pos`. For long sequences this is a ~30-cycle penalty per "deep below diagonal" tile.
5. **Per-element FMA chain in softmax**. FlyDSL emits 16 separate `_fsub + exp2 + _fadd` operations per row half, while C++ uses `attn_sub_row<T>(v_s, m_row)` which compiles to a packed vector form. Functionally identical but creates extra SSA edges.

Items 1 and 2 dominate. Closing the gap requires structural FlyDSL changes (vector-fused V hoist via a single `Vec<32 × bf16>` SSA value; hoisting per-buffer base addresses into SGPR before the main loop so the line-stride multiplies become immediate-folded), not algorithmic changes.

**Why P7 still landed in spite of the perf regression:** P7 was a correctness-first port. The OPUS LDS layout is the source of truth in `gqa_d128_kernel_template.hpp`; aligning to it (instead of FlyDSL's earlier XOR-swizzled layout) is the only way to make later optimizations like vector-fused `tr_load`, fused softmax tails, and constant-folded line strides legal and meaningful, because those optimizations directly assume the OPUS layout. With P7 done, the remaining gap is now purely about how LLVM lowers the same OPUS operations, not about whether the operations themselves are the OPUS ones.

---

## 8. Quick Reference

**Activation:** `export FLYDSL_ENABLE_OPUS_PATH=1` — selects the OPUS path; P1–P7 structural layout is always active in this kernel.

**Opt-out flags (for ablation):**

| Env var | Default | Effect / current status |
|---|---|---|
| `FLYDSL_OPUS_LAZY_RESCALE` | `1` | Always rescale (skip the ballot-clamp); MaxErr unchanged |
| `FLYDSL_OPUS_SETPRIO` | `1` | Skip `s_setprio(1)/(0)` brackets around GEMM2 |
| `FLYDSL_OPUS_STAGGER` | `1` | Use symmetric `gpu.barrier()` (no dual-group phase shift) |
| `FLYDSL_OPUS_YIELD_NOP` | `1` | Parsed but currently unused; no `s_nop 15; s_nop 7` sequence is emitted |

The P7 OPUS LDS layout (`SMEM_K_LINE_STRIDE`, `SMEM_V_LINE_STRIDE`, interleaved K0/V0/K1/V1, `_read_k_packs_for_buf`/`_read_v_packs_for_k_substep` shapes, N-permuted causal mask thresholds) has **no opt-out flag** — it is baked into the kernel because correctness depends on Q/O layouts being consistent with the K-read permutation.

**Where the OPUS_PATH gate lives:** `FlyDSL/kernels/flash_attn_func.py` → `_wrap_with_opus()` which dispatches to `build_flash_attn_opus_module(...)` when `seq_len % 256 == 0 && seq_len >= 384 && head_dim == 128 && dtype == bf16 && gpu_arch.startswith("gfx950")`.

**Test command** (full coverage of P1–P7):

```bash
FLYDSL_ENABLE_OPUS_PATH=1 \
python tests/kernels/test_flash_opus_attn.py --causal \
    --dtype bf16 --batch 16 --num_heads 64 --num_kv_heads 64 \
    --seq_len 8192 --head_dim 128 --iters 100 --compare
```

Expected output: `causal MaxErr ≤ 3.91e-03` (numerical parity with OPUS C++).

**Files:**

- FlyDSL kernel: [`FlyDSL/kernels/flash_attn_opus.py`](kernels/flash_attn_opus.py) (1655 lines, post P1-P7 commit `2d07de2`)
- OPUS C++ template: [`FlyDSL/opus_attn/gqa_d128_kernel_template.hpp`](opus_attn/gqa_d128_kernel_template.hpp) (755 lines)
- OPUS C++ instantiation (causal): [`FlyDSL/opus_attn/gqa_d128_kernel_causal.cc`](opus_attn/gqa_d128_kernel_causal.cc) (12 lines)
- Companion kernel description: [`FLASH_ATTN_OPUS_Kernel_Analysis_Detail.md`](FLASH_ATTN_OPUS_Kernel_Analysis_Detail.md) (post P1-P7)
