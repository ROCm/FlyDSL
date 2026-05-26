# DeepSeek-V4 Grouped Output Projection BMM — Status

**Updated**: 2026-05-20  
**Operation**: `einsum("sgd,grd->sgr", o, wo_a)` = BMM `[G, T, D] @ [G, D, R] → [G, T, R]`  
**V4 shape**: G=B=16, D=K=4096, R=N=1024  
**Files**: `~/flydsl/kernels/bmm_gfx1250.py` (bf16), `~/flydsl/kernels/bmm_a16w8_gfx1250.py` (a16w8)

---

## Phase 1a — bf16 BMM: COMPLETE ✓

All precision tests pass: `python -m pytest tests/kernels/test_bmm_gfx1250.py -v` → 8/8.

**Kernel structure**: TDM Mode 0 async copy + N-stage LDS pipeline + bf16 WMMA 16×16×32 + cluster multicast support + TDM store epilogue. Batch dimension mapped to grid-z (`block_id("z")`).

**Tile configs**:
- Decode (T≤64): tile_m=64, tile_n=128, tile_k=128, num_buffers=2
- Prefill (T≥128): tile_m=128, tile_n=128, tile_k=128, num_buffers=3

---

## Phase 1b — a16w8 BMM (fp8 weight + per-block scale): COMPLETE ✓

All precision tests pass: `python -m pytest tests/kernels/test_bmm_a16w8_gfx1250.py -v` → 8/8.

**Operation**:
- A: bf16 activations `[G, T, D]`
- B: fp8_e4m3fn weights `[G, D, R]` + scale `[G, D//128, R//128]` fp32
- C: bf16 output `[G, T, R]`

**Dequant path (optimized)**:
- `ds_load_tr8_b64` → `vec<2xi32>` (8 fp8 bytes)
- Decompose fp32 scale: E8M0_byte = fp32_exponent; residual = fp32 with exponent=127 (∈[1.0,2.0))
- `V_CVT_SCALE_PK8_BF16_FP8(vec<2xi32>, e8m0_i32, scaleSel=0)` → `vec<8xbf16>` (single instruction)
- `V_PK_MUL_BF16 × 4` to multiply by residual_bf16 broadcast
- Then bf16 WMMA 16×16×32

**Key bugs found and fixed**:
1. **fp8 format mismatch**: Test created B as `torch.float8_e4m3fnuz` (exponent bias=8, AMD CDNA-specific). The gfx1250 hardware instruction `V_CVT_PK_F32_FP8` uses **E4M3FN** (bias=7, OCP standard), causing a systematic 2× magnitude error for all N=1024 tests. Fix: test changed to `torch.float8_e4m3fn`.
2. **ISA clarification**: `rocdl.cvt.scalef32.pk.bf16.fp8` and `rocdl.cvt.scalef32.pk.fp8.bf16` are under `HasFP8ConversionScaleInsts` → **GFX950 (MI325X) only, NOT gfx1250**. gfx1250 uses `V_CVT_SCALE_PK8_BF16_FP8` (fp8→bf16 with E8M0 integer scale).

Reference: same fp8 format fix seen in MLA test (`DTYPE_QKV = torch.float8_e4m3fn`).

**Scale indexing**: `scale[bz, k_tile, n_block]` with flat offset `bz * n_k_blocks * n_n_blocks + k_tile * n_n_blocks + n_block`. One scale value per 128×128 B tile. Applied per WMMA fragment after fp8→bf16 conversion.

**fp8 LDS addressing** (ds_load_tr8_b64 layout, confirmed correct):
- Lane layout: `(2,4,4)` strides `(4,1,8)` on lane_id
- k_lane_off = `(lane_kgrp×8 + lane_half×4 + lane4) × lds_b_stride × ELEM_BYTES_B`
- n_lane_off = `nonK_grp × 8`

---

## Phase 2 — Performance: AM Benchmark Results

Performance measured on AM (cycle-accurate analytical model). Cycle counts = delta_sclk from `DumpDispatchEndTime` log.

### AM sclk cycle counts (gfx1250, V4 shape B=16 K=4096 N=1024)

| M | bf16 sclk | a16w8 nb=2 fp32 sclk | a16w8 nb=3 fp32 | a16w8 nb=3 E8M0 | nb=3 E8M0 vs bf16 |
|---|-----------|---------------------|-----------------|-----------------|-------------------|
| 1 | 200,071 | 220,932 (orig) | — | — | — |
| 64 | 199,749 | **214,886** (+7.6%) | **207,035** (+3.6%) | **170,392** | **−14.7%** ✓ |
| 256 | 448,777 | **491,579** (+9.5%) | TBD | **413,929** | **−7.8%** ✓ |

Notes:
- M=1 ≈ M=64 because both map to gx=1 (same 128 WGs; tile_m=64, ceil(M/64)=1)
- M=1024 bf16 was killed at 15% completion (25-min cap per run)
- **O2 E8M0 M=64** (2026-05-25): 170,392 sclk (clk1=45,203 clk2=215,595) — **−14.7% vs bf16, −17.7% vs nb=3 fp32**
  - a16w8 E8M0 is NOW FASTER than bf16 in memory-bound regime
  - Root cause of large gain: 4×V_PK_MUL_BF16 per 8 fp8 were in the critical dequant→WMMA path, not hidden by nb=3 pipeline. Removing them shortens the dependency chain and enables faster WMMA scheduling.
- **O2 E8M0 M=256** (2026-05-25): 413,929 sclk (clk1=52,143 clk2=466,072) — **−7.8% vs bf16, −15.7% vs nb=2 fp32**
  - a16w8 E8M0 is faster than bf16 even in compute-bound regime (M=256 crosses roofline ridge for a16w8)
  - Smaller gain than M=64 because compute cost (WMMA) dominates; E8M0 dequant VALU reduction still helps but HBM savings matter less
- **nb=3 M=64 fp32** (2026-05-25): 207,035 sclk — −3.6% vs nb=2. Extra stage hides TDM fetch latency.
- O1 (scale hoist) null result (2026-05-25): 217,535 sclk — MLIR CSE pre-empted; scale_f32 same SSA value → decompose ops CSE'd automatically. No VGPR/latency benefit.
- M=64 nb=2 fp32 dispatch: clk2=261,055 clk1=46,169 delta=214,886 (2026-05-20)
- M=256 nb=2 fp32 dispatch: clk2=544,450 clk1=52,871 delta=491,579 (2026-05-20)

### Dequant VALU optimization: `cvt_scale_pk8_bf16_fp8`

**Root cause of a16w8 overhead**: dequant was 4×`cvt_pk_f32_fp8` + 8×extract + `mulf(8×f32)` + `trunc_f` ≈ 14 VALU instructions per 8 fp8 bytes → kernel VALU-bound.

**Optimization**: Decompose fp32 scale into E8M0 × residual (exact, no approximation):
- E8M0 byte = `(bitcast_f32_to_i32(scale) >> 23) & 0xFF` (fp32 biased exponent)
- residual = fp32 with exponent replaced by 127 → value in [1.0, 2.0)
- `V_CVT_SCALE_PK8_BF16_FP8`: converts 8 fp8 → 8 bf16 in ONE instruction using E8M0 scale
- `V_PK_MUL_BF16 × 4`: multiply by residual broadcast
- Total: ~6 VALU instructions vs ~14 original → ~2.3× reduction in dequant VALU

**All 8/8 precision tests still pass** after optimization.

### ISA notes (gfx1250 fp8 conversion landscape)

| Op | gfx1250 | GFX950 | Description |
|----|---------|--------|-------------|
| `V_CVT_PK_F32_FP8` | ✓ | ✓ | 2 fp8 → 2 f32, no scale |
| `V_CVT_SCALE_PK8_BF16_FP8` | ✓ | ✗ | 8 fp8 → 8 bf16, E8M0 scale |
| `V_CVT_SCALEF32_PK_BF16_FP8` | ✗ | ✓ | 2 fp8 → 2 bf16, fp32 scale |

### Roofline analysis (theoretical, gfx1250: 4200 TFLOPS bf16, 19.6 TB/s HBM)

| T | a16w8 roofline µs | bf16 roofline µs | regime |
|---|-------------------|-------------------|--------|
| 1 | 3.43 | 6.86 | memory-bound |
| 8 | 3.49 | 6.91 | memory-bound |
| 32 | 3.69 | 7.12 | memory-bound |
| 64 | 3.96 | 7.38 | memory-bound |
| 128 | 4.49 | 7.92 | memory-bound |
| 256 | 8.18 | 8.99 | a16w8 compute-bound |
| 512 | 16.36 | 16.36 | both compute-bound |
| 1024 | 32.72 | 32.72 | both compute-bound |
| 4096 | 130.89 | 130.89 | both compute-bound |

### Pending optimizations (need silicon to measure)
1. **Tile sweep**: tile_m ∈ {32,64,128}, tile_n ∈ {64,128}, tile_k ∈ {64,128}
2. **GL2 weight hint**: B descriptor `cache_policy` for NV/HT — API undocumented for non-default values; no existing kernel uses it
3. **Decode L2 prefetch distance**: Currently `l2_prefetch_distance=2`; tune per regime
4. **Cluster multicast for A**: T=1 row is 8KB bf16 — worth broadcasting within cluster

### Benchmark commands (run on silicon)
```bash
# a16w8 sweep
cd ~/flydsl
for M in 1 8 32 64; do
  python tests/kernels/test_bmm_a16w8_gfx1250.py -B 16 -M $M -N 1024 -K 4096 \
    --tile-m 64 --tile-n 128 --tile-k 128 --num-buffers 2 --bench --bench-iters 200
done
for M in 128 256 512 1024 4096; do
  python tests/kernels/test_bmm_a16w8_gfx1250.py -B 16 -M $M -N 1024 -K 4096 \
    --tile-m 128 --tile-n 128 --tile-k 128 --num-buffers 3 --bench --bench-iters 200
done

# bf16 BMM sweep
python tests/kernels/test_bmm_gfx1250.py --sweep --bench --bench-iters 200
```

---

## Phase 3 — Optimization O1: Scale decomposition hoist (2026-05-25)

**Change**: `_dequant_scale_to_bf16(v_2xi32, scale_f32)` split into:
- `_decompose_scale(scale_f32) → (e8m0_byte, residual_bf16)` — called ONCE per `load_wmma_frag_b_fp8` call (not per k_half)
- `_load_frag_b_fp8_parts(lds_base_idx, b_lane_base, ks, e8m0_byte, residual_bf16)` — inner loop unchanged

**Precision test**: 8/8 pass (2026-05-25, ffmlite).

**AM result (2026-05-25)**: M=64 O1 delta = **217,535 sclk** (clk1=45,146 → clk2=262,681).
- Baseline (E8M0 before O1): 214,886 sclk → **O1 is +1.2% slower (no improvement, within noise)**.
- **Root cause of null result**: MLIR CSE already eliminated the redundant decompose ops.
  `scale_f32` is the same SSA value for both k_half=0 and k_half=1 iterations inside the original
  `_dequant_scale_to_bf16`. CSE hoists `scale_i32`, `e8m0_byte`, `residual_bf16` to before the loop
  automatically, so O1's explicit hoist produced identical MLIR IR. VGPR pressure unchanged.
- **Gap vs bf16 unchanged**: a16w8 still ~8.9% slower at M=64 (199,749 bf16 vs 217,535 a16w8).
  True cause is dequant VALU instructions (V_CVT_SCALE_PK8_BF16_FP8 + 4×V_PK_MUL_BF16) that cannot
  be hidden by WMMA execution at this occupancy level.

---

## Phase 4 — O3 tile sweep: nb=3 for decode (AM pending)

**Config**: tile_m=64, tile_n=128, tile_k=128, **num_buffers=3**, M=64.
- ffmlite result: dec_tn128_nb3 → PASS (2026-05-25)
- Rationale: fp8 B stage uses only 16KB/stage (vs 32KB for bf16). With group_k=group_n=128 constraint
  tile_n and tile_k are fixed. nb=3 is the only viable decode config change.
- AM run: pending.

---

## Deliverable checklist

- [x] bf16 BMM passes all shape precision tests (8/8)
- [x] a16w8 BMM passes all shape precision tests (8/8)
- [x] Dequant VALU optimized: 4×cvt_pk_f32_fp8 → 1×cvt_scale_pk8_bf16_fp8 + residual mulf
- [x] Scale decomposition hoisted: _decompose_scale called once per fragment (O1, no AM benefit — CSE pre-empted)
- [x] O1 AM validation: 217,535 sclk (+1.2% vs 214,886 baseline, no improvement)
- [x] O3 AM: dec_tn128_nb3 → **207,035 sclk** (−3.6% vs nb=2 fp32; gap vs bf16 7.6%→3.6%)
- [x] O2 kernel+test: E8M0 uint8 scale path (`use_e8m0_scale=True`), 3/3 precision tests pass
- [x] O2 AM: nb=3 E8M0 M=64 → **170,392 sclk** (−14.7% vs bf16; −17.7% vs nb=3 fp32) 🎉
- [x] Update default decode config: tile_m=64, tile_n=128, tile_k=128, num_buffers=3 (test updated)
- [x] O2 AM: M=256 E8M0 (nb=3, tile_m=128) → **413,929 sclk** (−7.8% vs bf16; −15.7% vs nb=2 fp32) 🎉
- [x] convert.py: `--wo-dtype a16w8_e8m0` flag — reshape fp8 `[G*R,D]→[G,D,R]`, round scale to uint8 E8M0 `[G,D//128,R//128]`
- [x] model.py: `_WoABuffer` + `_get_bmm_a16w8_launch_fn` + forward dispatch to BMM kernel when `wo_dtype=="a16w8_e8m0"`
- [ ] Decode T=1 latency ≤5µs (roofline: 3.43µs, needs silicon confirmation)
- [ ] Prefill T=4096 WMMA utilization >50% (roofline: 100%, needs silicon)
- [x] Status document written
