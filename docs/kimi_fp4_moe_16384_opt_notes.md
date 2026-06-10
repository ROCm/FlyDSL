# Kimi FP4 MoE 16384 Opt Notes

## Scope

This note tracks the migration experiment for the fixed Kimi M=16384 row:

- `TOKEN=16384`
- `EXPERTS=385`
- `MODEL_DIM=7168`
- `INTER_DIM=512`
- `TOPK=9`
- current FlyDSL kernels:
  - `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_xcd4`
  - `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_xcd4`
- mxfp4 row selected by aiter for this same shape:
  - `mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128`
  - `mxfp4_moe_g2_a4w4_NE385_H7168_E512_BM128_NONATOMIC_MXFP4OUT`

The new experiment file is `kimi_fp4_moe_16384_opt.py`.

## Current FlyDSL Path

`kimi_fp4_moe_16384.py` is already specialized to the single CSV-selected
FlyDSL row. Its end-to-end path is:

1. `moe_sorting_opus_fwd` with `BLOCK_M=64`.
   It produces `sorted_ids`, `sorted_weights`, `sorted_expert_ids`, and
   `num_valid_ids`.
2. `fused_dynamic_mxfp4_quant_moe_sort(hidden_states, ...)`.
   For M=16384 this is the large-M split path: per-token fp4 quant first,
   then scale sorting/swizzle through `mxfp4_moe_sort_hip`.
3. FlyDSL stage1 writes bf16 intermediate `[TOKEN, TOPK, INTER_DIM]`.
4. `fused_dynamic_mxfp4_quant_moe_sort(a2.view(-1, INTER_DIM), ...)`.
5. FlyDSL stage2 writes weighted bf16 per topk slot into a large
   `[TOKEN, TOPK, MODEL_DIM]` staging buffer, then `torch.sum(..., dim=1)`.

The important point: the current FlyDSL path already uses the mxfp4 scale-sort
helper for activation scales. It does not use the mxfp4 MoE sorter or the
mxfp4 output/scatter-reduce path.

## Aiter mxfp4 Path

The aiter mxfp4 route in `fused_moe.py` does a different pipeline:

1. `mxfp4_moe_sort` with `BM=128`.
   This emits `sorted_token_ids`, `sorted_expert_ids`, `cumsum_tensor`,
   `reverse_sorted`, `sorted_weights`, `masked_m`, and `m_indices`.
2. `mxfp4_moe_quant` quantizes the original hidden states once.
3. `mxfp4_moe_sort_scales` produces the GEMM-consumed scale layout.
4. `mxfp4_moe_gemm1_a4w4` consumes `m_indices` and the mxfp4 preshuffled
   weight/scale layout.
5. For this M=16384 row, stage2 uses `_MXFP4OUT`:
   `mxfp4_moe_gemm2_a4w4_mxfp4out` writes packed fp4 output plus e8m0 scales,
   then `mxfp4_moe_scatter_reduce_q` does weighted topk reduce.

This avoids the FlyDSL path's large bf16 `[TOKEN, TOPK, MODEL_DIM]` stage2
buffer and the final `torch.sum`.

## Experiments Added

`kimi_fp4_moe_16384_opt.py` has two entry points:

- `run_kimi_fp4_flydsl_mxfp4_sort_16384`
  - runs `mxfp4_moe_sort(BM=128)`;
  - expands `sorted_expert_ids` from BM128 to BM64 with `repeat_interleave`;
  - reuses the existing FlyDSL stage1/stage2 kernels.
- `run_kimi_fp4_flydsl_atomic_stage2_16384`
  - keeps the current FlyDSL stage1;
  - calls the generic FlyDSL stage2 in `mode="atomic"` to avoid the explicit
    `[TOKEN, TOPK, MODEL_DIM]` bf16 staging buffer plus final `torch.sum`;
  - this is a direct test of whether output accumulation, not mxfp4out, is
    enough to close the gap.
- `run_kimi_fp4_mxfp4_moe_16384_opt`
  - fixed-shape extraction of aiter's mxfp4 pipeline;
  - no CSV dispatch;
  - uses the mxfp4 sort, quant, sort_scales, gemm1, gemm2_mxfp4out, and
    scatter_reduce_q operations directly.

`bench_flydsl_16384.py` now compares:

- `aiter_moe`
- `aiter_mxfp4_moe`
- `local_kimi_fp4`
- `local_kimi_fp4_mxfp4_sort`
- `local_kimi_fp4_atomic_stage2`
- `local_mxfp4_opt`

## Validation

Command:

```bash
CUDA_VISIBLE_DEVICES=5 /opt/venv/bin/python bench_flydsl_16384.py --warmup 5 --graph-iters 5 --measure 5
```

Result:

```text
aiter_mxfp4_vs_aiter_cos=0.981046 aiter_mxfp4_vs_aiter_max_abs=1.828125
local_vs_aiter_cos=1.000000 local_vs_aiter_max_abs=0.000000
local_mxfp4_sort_vs_aiter_cos=1.000000 local_mxfp4_sort_vs_aiter_max_abs=0.000000
local_atomic_stage2_vs_aiter_cos=0.999991 local_atomic_stage2_vs_aiter_max_abs=0.093750
local_mxfp4_opt_vs_aiter_mxfp4_cos=1.000000 local_mxfp4_opt_vs_aiter_mxfp4_max_abs=0.000000
aiter_moe_us=2289.4
aiter_mxfp4_moe_us=1811.5
local_kimi_fp4_us=2311.2
local_kimi_fp4_mxfp4_sort_us=2375.4
local_kimi_fp4_atomic_stage2_us=2745.6
local_mxfp4_opt_us=1830.6
```

Syntax check:

```bash
/opt/venv/bin/python -m py_compile \
  kimi_fp4_moe_16384.py \
  kimi_fp4_moe_16384_opt.py \
  bench_flydsl_16384.py \
  profile_flydsl_16384.py
```

## Torch Profiler Snapshot

Command:

```bash
CUDA_VISIBLE_DEVICES=5 /opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_kimi_fp4,local_mxfp4_opt \
  --warmup 3 --iters 5 --trace-dir profiler_traces
```

Profiler traces:

- `profiler_traces/profile_16384_aiter_mxfp4_moe.json.gz`
- `profiler_traces/profile_16384_local_kimi_fp4.json.gz`
- `profiler_traces/profile_16384_local_mxfp4_opt.json.gz`

### `aiter_mxfp4_moe`

Total device time: `1920.5us/iter`, `8` kernels per iter.

| idx | avg us | note |
| --- | ---: | --- |
| 0 | 6.4 | mxfp4 sort count |
| 1 | 10.7 | mxfp4 sort cumsum |
| 2 | 33.1 | mxfp4 sort place/pad |
| 3 | 52.4 | `mxfp4_moe_quant` |
| 4 | 59.5 | `mxfp4_moe_sort_scales` |
| 5 | 722.0 | `mxfp4_moe_gemm1_a4w4` |
| 6 | 905.7 | `mxfp4_moe_gemm2_a4w4_mxfp4out` |
| 7 | 130.7 | `mxfp4_moe_scatter_reduce_q` |

### `local_kimi_fp4`

Total device time: `2351.9us/iter`, `11` kernels per iter.

| idx | avg us | note |
| --- | ---: | --- |
| 0 | 4.9 | Opus sort clear workspace |
| 1 | 4.8 | Opus sort phase 0 |
| 2 | 5.0 | Opus sort phase 1 |
| 3 | 65.0 | Opus sort phase 2/3 |
| 4 | 69.3 | quant hidden |
| 5 | 42.7 | scale sort hidden |
| 6 | 806.3 | FlyDSL stage1 |
| 7 | 31.4 | quant intermediate |
| 8 | 6.1 | scale sort intermediate |
| 9 | 928.9 | FlyDSL stage2 bf16 topk staging |
| 10 | 387.4 | PyTorch `torch.sum(TOPK)` reduce |

### `local_mxfp4_opt`

Total device time: `1872.4us/iter`, `8` kernels per iter.

| idx | avg us | note |
| --- | ---: | --- |
| 0 | 6.6 | mxfp4 sort count |
| 1 | 10.7 | mxfp4 sort cumsum |
| 2 | 33.0 | mxfp4 sort place/pad |
| 3 | 53.2 | `mxfp4_moe_quant` |
| 4 | 58.7 | `mxfp4_moe_sort_scales` |
| 5 | 709.8 | `mxfp4_moe_gemm1_a4w4` |
| 6 | 870.6 | `mxfp4_moe_gemm2_a4w4_mxfp4out` |
| 7 | 129.7 | `mxfp4_moe_scatter_reduce_q` |

### Delta Summary

Compared with `aiter_mxfp4_moe`, `local_kimi_fp4` spends roughly:

- `+84us` in stage1 (`806us` vs `722us`).
- `+23us` in stage2 GEMM itself (`929us` vs `906us`), but this is not the main
  issue.
- `+387us` in the separate PyTorch TOPK reduce after stage2.
- `+50us` more in sort/quant/scale-sort orchestration because the current path
  uses Opus sort plus two generic quant+scale-sort pairs instead of the mxfp4
  BM128 prologue.

The main gap is therefore the stage2 output/reduce design: current FlyDSL
materializes bf16 `[TOKEN, TOPK, MODEL_DIM]` and then reduces it, while mxfp4
materializes packed fp4 output plus scales and uses a specialized
scatter/reduce kernel.

## Conclusions

The "maybe sort should be written into the kernel" hypothesis is not enough by
itself. Replacing Opus sorting with `mxfp4_moe_sort` is correct after expanding
BM128 expert ids to BM64, but it does not improve performance for the current
FlyDSL GEMMs. It is slightly slower in the measured run because it pads to
BM128 and still executes the same FlyDSL stage2 bf16 staging plus `torch.sum`.

The speed gap is more likely in the stage2 output/reduce design:

- FlyDSL current path materializes bf16 topk output and reduces with
  `torch.sum`.
- mxfp4 uses `_MXFP4OUT` to materialize packed fp4 output and reduces with
  `mxfp4_moe_scatter_reduce_q`.

The atomic FlyDSL experiment confirms that plain atomic accumulation is not the
missing piece for this shape. It avoids the explicit `torch.sum`, but bf16
atomics are much slower here and introduce small order-dependent numeric drift:
`local_kimi_fp4_atomic_stage2_us=2745.0`, slower than both the current reduce
path and mxfp4.

The current performance-aligned optimized path is `local_mxfp4_opt`. It is a
fixed-shape extraction of the mxfp4 pipeline and lands within about 1% of
`aiter_mxfp4_moe` in the measured run (`1822.4us` vs `1804.1us`) while avoiding
the dynamic CSV/fused_moe dispatch.

The next useful migration step is therefore not just moving sort into FlyDSL.
It should be one of:

1. Add a FlyDSL stage2 mxfp4out epilogue and use `mxfp4_moe_scatter_reduce_q`.
2. Port the scatter-reduce-q logic to FlyDSL after stage2 can emit packed fp4
   output plus e8m0 scales.
3. Then revisit a native FlyDSL sorter or BM128 stage1/stage2 tiling if the
   remaining overhead is still sorting-related.

The fixed `local_mxfp4_opt` path is useful as the migration reference because
it strips away the CSV/dynamic fused_moe dispatch and leaves only the exact
ops needed for this Kimi row.

## Migration Plan

The migration is tracked as one kernel at a time, with a fixed-shape aiter
oracle kept next to the active FlyDSL candidate:

- `run_kimi_fp4_mxfp4_moe_16384_aiter_ref`
  keeps the extracted Kimi M=16384 mxfp4 pipeline on aiter kernels only.
- `run_kimi_fp4_mxfp4_moe_16384_opt`
  is the active migration candidate.
- `_run_mxfp4_pipeline_16384(...)`
  has explicit switches for `gemm1`, `gemm2`, and `scatter_reduce_q`.

The intended order remains:

1. Port `scatter_reduce_q`.
2. Port `gemm2_a4w4_mxfp4out`.
3. Port `gemm1_a4w4`.
4. Revisit the aux kernels only after the GEMM/reduce path is performance
   aligned.

## Migration Log

### Step 0: Fixed Aiter Oracle

`local_mxfp4_aiter_ref` was added to `bench_flydsl_16384.py`. It calls the
same fixed-shape pipeline as `local_mxfp4_opt`, but leaves all kernels on
aiter. This gives a local oracle that avoids CSV dispatch while preserving the
exact mxfp4 behavior.

Validation:

```text
local_mxfp4_aiter_ref_vs_aiter_mxfp4 cos=1.000000 max_abs=0.000000
```

### Step 1: FlyDSL `scatter_reduce_q`

`kimi_fp4_moe_16384_opt.py` now contains
`kimi_mxfp4_scatter_reduce_q_16384`, a fixed-shape FlyDSL port of aiter's
`scatter_reduce_mxfp4_kernel<7168, 9, 8, true>`.

The kernel shape is fixed to:

- `D_HIDDEN=7168`
- `TOPK=9`
- `THREADS_PER_CTA=128`
- `COLS_PER_THREAD=8`
- grid `x=7`, grid `y=16384`

The first correct version decoded each FP4 nibble with scalar select logic.
It was bitwise correct after using `math.fma` and `v_cvt_pk_bf16_f32`, but
the scatter kernel took about `514us`, far slower than aiter's `~132us`.

The optimized version uses the gfx950 intrinsic
`llvm.amdgcn.cvt.scalef32.pk.f32.fp4`, matching aiter's
`__builtin_amdgcn_cvt_scalef32_pk_f32_fp4` path. Each thread loads one packed
`u32`, calls the intrinsic four times to decode 8 FP4 values to f32 with the
E8M0 scale applied, accumulates with `math.fma`, then packs the final 8 values
to 4 bf16 words with `v_cvt_pk_bf16_f32`.

The attempted `vector<8xf4E2M1FN>` route does not currently lower cleanly in
FlyDSL/LLVM: it leaves an unrealized conversion cast from `vector<8xi4>` to
`vector<8xf4E2M1FN>`. Keep using the AMD intrinsic for this port.

Validation:

```text
local_mxfp4_opt_vs_aiter_mxfp4 cos=1.000000 max_abs=0.000000
```

Profiler snapshot after intrinsic decode:

```text
local_mxfp4_aiter_ref device_total_us_per_iter=1927.1
  scatter_reduce_mxfp4_kernel ~132.7us

local_mxfp4_opt device_total_us_per_iter=1958.8
  flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9 ~144.2us
```

CUDAGraph snapshot:

```text
aiter_mxfp4_moe_us=1798.2
local_mxfp4_aiter_ref_us=1794.3
local_mxfp4_opt_us=1825.1
```

Selected bench command after adding `--runners`:

```bash
CUDA_VISIBLE_DEVICES=5 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_aiter_ref,local_mxfp4_opt \
  --warmup 3 --graph-iters 5 --measure 3
```

Result:

```text
local_mxfp4_ref_vs_aiter_mxfp4_cos=1.000000 local_mxfp4_ref_vs_aiter_mxfp4_max_abs=0.000000
local_mxfp4_opt_vs_aiter_mxfp4_cos=1.000000 local_mxfp4_opt_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1822.3
local_mxfp4_aiter_ref_us=1820.4
local_mxfp4_opt_us=1829.0
```

Current status: the first real migrated kernel is bitwise correct and within
roughly `+11-12us` of the aiter scatter kernel in profiler, `+31us` end to end
in one CUDAGraph run and `+9us` in the selected bench run above. The next step
is `gemm2_a4w4_mxfp4out`, because
that is the largest remaining single kernel and it defines the packed FP4
output contract consumed by this scatter/reduce kernel.

### Step 2: FlyDSL `gemm2_a4w4_mxfp4out` Candidate

`kimi_fp4_moe_16384_opt.py` now contains
`kimi_mxfp4_gemm2_mxfp4out_16384`, an experimental fixed-shape FlyDSL port of
aiter's Kimi-only `mxfp4_moe_gemm2_a4w4_mxfp4out` path.

The implemented contract matches the aiter mxfp4-out path:

- input A is already sorted flat: `[max_sorted, 512 / 2]` packed FP4
- input A scale is the shuffled scale buffer produced by aiter GEMM1
- input B is the mxfp4 preshuffled W2 buffer
- output is sorted-row-major `flat_out_q` plus `flat_out_scale`
- output is consumed by `kimi_mxfp4_scatter_reduce_q_16384`

The first version intentionally prioritizes correctness over scheduling:

- BM=128, BN=256, BK=256, K=512
- 2D grid: N block by sorted-M block
- no persistent work loop yet
- no AGPR accumulator inline-asm path yet
- f32 CShuffle-style epilogue quantizes 8 contiguous f32 values per thread
- uses `llvm.amdgcn.cvt.scalef32.pk.fp4.f32` for FP4 packing

Validation:

```text
local_mxfp4_opt_gemm2_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm2_vs_aiter_mxfp4_max_abs=0.000000
```

CUDAGraph snapshot:

```text
aiter_mxfp4_moe_us=1831.7
local_mxfp4_aiter_ref_us=1822.5
local_mxfp4_opt_us=1819.1
local_mxfp4_opt_gemm2_us=2089.7
```

Profiler snapshot:

```text
local_mxfp4_aiter_ref:
  aiter gemm2 mxfp4out ~856.4us
  aiter scatter_reduce_q ~128.9us

local_mxfp4_opt_gemm2:
  flydsl_kimi_mxfp4_gemm2_mxfp4out_NE385_H7168_E512_BM128_v0 ~1172.2us
  flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9 ~139.9us
```

Conclusion: the GEMM2 port is bitwise correct, but not performance aligned.
The current gap is dominated by GEMM2 itself (`~+315us` in the profiler
snapshot). The most likely reasons are exactly the deliberate omissions:
aiter uses a persistent 1D grid capped around CU count and pins the BM=128
non-atomic path accumulators into AGPRs; this FlyDSL v0 uses many more CTAs and
regular MFMA accumulator values. The next optimization target is to change the
FlyDSL GEMM2 launch/body to a persistent work loop, then investigate AGPR MFMA
inline asm or compiler support if the remaining gap is still large.

### Step 3: FlyDSL `gemm1_a4w4` Candidate

`kimi_fp4_moe_16384_opt.py` now contains
`kimi_mxfp4_gemm1_16384`, a fixed-shape FlyDSL candidate for aiter's selected
Kimi mxfp4 GEMM1:

```text
mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128
```

The runner wired into `bench_flydsl_16384.py` is:

```text
local_mxfp4_opt_gemm1
```

That path uses aiter sort, quant, sort_scales, a FlyDSL GEMM1, aiter GEMM2
mxfp4out, and the existing FlyDSL scatter/reduce-q.

Correctness fixes made during the port:

- GEMM1 scale indexing must advance one `kBS_stride_k0_dw` per BK=256 tile.
  The first draft used `k_tile * k_unroll * kBS_stride_k0_dw`, which skipped
  every other scale block after tile 0.
- Invalid padded `m_indices` rows must read one-past the A buffer, matching
  aiter's MUBUF OOB-zero behavior. Falling back to token 0 made padded rows
  nonzero and broke the GEMM1 output.
- After those fixes, `inter_sorted_quant[:cumsum]` is bitwise exact versus
  aiter GEMM1. The used prefix of `inter_sorted_shuffled_scale` is also exact;
  unused tail bytes may differ because both outputs are allocated with
  `torch.empty`.

CUDAGraph snapshot on `CUDA_VISIBLE_DEVICES=6`:

```text
local_mxfp4_opt_vs_aiter_mxfp4_cos=1.000000 local_mxfp4_opt_vs_aiter_mxfp4_max_abs=0.023438
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000 local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.023438
aiter_mxfp4_moe_us=1850.6
local_mxfp4_opt_us=1840.2
local_mxfp4_opt_gemm1_us=2298.9
```

Profiler snapshot on `CUDA_VISIBLE_DEVICES=6`:

```text
local_mxfp4_aiter_ref:
  device_kernel_calls_per_iter=8
  aiter gemm1 ~727.3us
  aiter gemm2 mxfp4out ~916.3us
  aiter scatter_reduce_q ~134.3us

local_mxfp4_opt_gemm1:
  device_kernel_calls_per_iter=8
  flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v0 ~1228.1us
  aiter gemm2 mxfp4out ~861.4us
  flydsl scatter_reduce_q ~137.5us
```

Kernel count status: aligned. The aiter reference, `local_mxfp4_opt`, and
`local_mxfp4_opt_gemm1` all launch 8 kernels per iteration:

```text
sort_count
sort_cumsum
sort_place_pad
quant
sort_scales
gemm1
gemm2_mxfp4out
scatter_reduce_q
```

Performance status: GEMM1 is functionally aligned but not performance aligned.
The current gap is about `+500us` in profiler (`~1228us` FlyDSL vs `~727us`
aiter). The known missing pieces are aiter's overlapped A/B/scale pipeline and
BM=128 AGPR accumulator inline-asm path. A quick attempt to express ping-pong A
staging at the Python level was not kept because it broke correctness and did
not improve time; this needs a more literal port of aiter's physical LDS slots
and wait/barrier schedule.

### Step 4: GEMM1 Resource Inspection

The selected aiter HIP GEMM1 instance was compiled with:

```text
-Rpass-analysis=kernel-resource-usage
```

Selected source:

```text
/root/up-aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM128.cu
```

Compiler resource report:

```text
TotalSGPRs: 49
VGPRs: 165
AGPRs: 168
ScratchSize [bytes/lane]: 0
Dynamic Stack: False
Occupancy [waves/SIMD]: 1
SGPRs Spill: 0
VGPRs Spill: 0
LDS Size [bytes/block]: 131072
```

The FlyDSL GEMM1 hsaco metadata currently reports:

```text
agpr_count: 0
sgpr_count: 46
vgpr_count: 222
sgpr_spill_count: 0
vgpr_spill_count: 0
private_segment_fixed_size: 0
group_segment_fixed_size: 131072
max_flat_workgroup_size: 256
wavefront_size: 64
```

Disassembly confirms the structural difference. aiter's MFMA accumulators are
allocated in AGPRs:

```asm
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], ...
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], ...
```

The FlyDSL candidate currently accumulates into VGPRs:

```asm
v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], ...
v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], ...
```

Local artifacts:

```text
resource_inspect/aiter_gemm1_rpass/compile.stderr
resource_inspect/aiter_gemm1_rpass/aiter_gemm1_bm128.hsaco
resource_inspect/aiter_gemm1_rpass/aiter_gemm1_bm128.s
resource_inspect/flydsl_gemm1.hsaco
resource_inspect/flydsl_gemm1.s
```

Conclusion: this is not a spill problem. Both kernels have zero scratch and the
same 128KiB LDS allocation. The main confirmed resource gap is that aiter uses
168 AGPRs for accumulators while FlyDSL uses 0 AGPRs and raises VGPR pressure to
222. The next GEMM1 optimization should port the AGPR MFMA inline-asm path from
aiter before spending time on smaller schedule tweaks.

### Step 5: Grid and Waitcnt Comparison

Current grid status for the fixed M=16384 mxfp4 path:

```text
sort_count:       grid=[16, 1, 1]      block=[1024, 1, 1]
sort_cumsum:      grid=[1, 1, 1]       block=[1024, 1, 1]
sort_place_pad:   grid=[16, 1, 1]      block=[1024, 1, 1]
quant:            grid=[512, 1, 1]     block=[1024, 1, 1]
sort_scales:      grid=[512, 1, 1]     block=[1024, 1, 1]
aiter GEMM1:      grid=[6136, 1, 1]    block=[256, 1, 1]
FlyDSL GEMM1:     grid=[4, 1534, 1]    block=[256, 1, 1]
aiter GEMM2:      grid=[256, 1, 1]     block=[256, 1, 1]
FlyDSL GEMM2:     grid=[28, 1534, 1]   block=[256, 1, 1]
scatter_reduce_q: grid=[7, 16384, 1]   block=[128, 1, 1]
```

GEMM1 launches the same number of CTAs:

```text
1534 M blocks * 4 N blocks = 6136 CTAs
```

aiter expresses this as a 1D grid and maps `pid / 4` to M and `pid % 4` to N.
FlyDSL expresses the same logical work as a 2D grid with `x=N` and `y=M`.
This grid shape is not the main GEMM1 gap.

GEMM2 is different: aiter's nonatomic mxfp4out path is persistent and launches
only `NUM_CU=256` CTAs, while the current FlyDSL GEMM2 candidate launches the
full logical tile grid:

```text
28 N blocks * 1534 M blocks = 42952 CTAs
```

That explains why the GEMM2 candidate is also slower; it is not just a local
MFMA schedule issue.

Static GEMM1 ISA counts before the waitcnt tweak:

```text
                 aiter GEMM1    FlyDSL GEMM1 v0
mfma_scale       1792           1792
buffer_load      408            622
ds_read          536            480
ds_write         128            96
s_waitcnt        320            499
s_barrier        30             58
s_nop            123            455
```

The MFMA count is aligned, so the gap is in load/wait/barrier scheduling and
epilogue/control overhead. The original FlyDSL loop did:

```text
barrier
A raw load -> LDS
B load -> VGPR
A/B scale load -> VGPR
s_waitcnt(0)
barrier
compute
```

The first small optimization keeps the single-LDS-buffer structure but changes
the pre-compute wait from `s_waitcnt(0)` to `s_waitcnt(14)`. For this GEMM1
tile, each K tile issues:

```text
8 A raw LDS VMEM loads
8 B vector VMEM loads
6 scale VMEM loads
```

Waiting until `vmcnt <= 14` is enough for the older 8 A->LDS loads to complete
before the cross-wave barrier. The newer B and scale loads can continue until
their values are actually consumed by the MFMA path. This mirrors the aiter
idea of not draining all VMEM before the barrier.

Validated result for `flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v1`:

```text
CUDAGraph:
  local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
  local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
  aiter_mxfp4_moe_us=1804.5
  local_mxfp4_opt_gemm1_us=2248.6

Profiler:
  FlyDSL GEMM1 v1 ~1184.5us
  previous FlyDSL GEMM1 v0 snapshot ~1226.6us
  aiter GEMM1 snapshot ~725.0us
```

The v1 resource metadata remains:

```text
agpr_count=0
sgpr_count=46
vgpr_count=222
spill=0
LDS=131072
```

Because VGPR does not spill, AGPR is not the first required porting target. It
may still help occupancy or scheduling later, but the confirmed gap is currently
extra load/wait/barrier work in the FlyDSL GEMM1 body.

An attempted one-ahead LDS double-buffer/prefetch experiment reduced time a
little but produced incorrect output, so it was not kept. A correct version
needs to match aiter's physical LDS slot lifetime and explicit LDS read/write
ordering more closely instead of just moving Python-level loads.

### Step 6: GEMM1 Load Reduction Checkpoint

The next two changes were kept in `kimi_fp4_moe_16384_opt.py` after validating
correctness against aiter mxfp4.

First, the A raw LDS DMA byte count was fixed for packed FP4:

```python
bytes_per_thread_x = tile_m * tile_k // a_elem_vec_pack // total_threads
```

The old expression missed the FP4 pack ratio and loaded twice the required A
bytes per K tile. Static ISA counts changed from:

```text
FlyDSL v1 buffer_load total: 622
  buffer_load_dwordx4 lds,offen: 224
```

to:

```text
FlyDSL v2 buffer_load total: 510
  buffer_load_dwordx4 lds,offen: 112
```

That aligns the A raw vector load count with aiter, whose corresponding
`buffer_load_dwordx4 lds,offen` count is 116.

Second, A scales were moved from per-K-tile global scalar loads into a CTA-level
LDS preload that mirrors aiter's `issue_a_scale_load` / `issue_a_scale_ds_read`
structure:

```text
A scale global -> LDS once per CTA
MFMA loop reads A scale dwords from LDS
```

Validated CUDAGraph result for
`flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v3`:

```text
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1813.3
local_mxfp4_opt_gemm1_us=2148.7
```

This is a runnable checkpoint. It improves the earlier v1 CUDAGraph snapshot
(`2248.6us`) but GEMM1 is still not fully aligned with aiter.

### Step 7: GEMM1 Two-Slot A Pipeline

The next useful optimization is `flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v5`.
It changes the GEMM1 K loop from a single A LDS slot to two A LDS slots:

```text
slot 0: compute K tile i
slot 1: preload A for K tile i + 1
```

B and B-scale for the next tile are also issued before computing the current
tile. This mirrors the important part of aiter's two-stage GEMM1 pipeline
without changing FlyDSL compiler passes.

The launch grid is not the main GEMM1 issue:

```text
aiter GEMM1 CTA grid:  6136 x 1 x 1, block 256
FlyDSL GEMM1 CTA grid: 4 x 1534 x 1, block 256
```

Both are 6136 CTAs. `rocprofv3` reports global work-items, so the same launch
appears as:

```text
aiter:  Grid_Size=(1570816, 1, 1)
FlyDSL: Grid_Size=(1024, 1534, 1)
```

Static GEMM1 ISA counts after v5:

```text
                 aiter GEMM1    FlyDSL GEMM1 v3    FlyDSL GEMM1 v5
mfma_scale       1792           1792               1792
buffer_load      408            414                414
ds_read          536            543                543
ds_write         128            96                 96
s_waitcnt        320            599                492
s_barrier        30             58                 31
s_nop            123            388                416
```

Validated CUDAGraph result for v5:

```text
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1805.4
local_mxfp4_opt_gemm1_us=2128.0
```

Torch profiler snapshot for the GEMM1 kernel itself:

```text
aiter GEMM1:  733.9us
FlyDSL v5:   1070.8us
```

Resource metadata from the FlyDSL cached artifact:

```text
agpr_count=0
sgpr_count=42
vgpr_count=238
spill=0
LDS=131072
```

This confirms the current remaining GEMM1 gap is no longer grid or raw load
count. The biggest static differences are now the extra `s_waitcnt` count and
the much larger `s_nop` scheduler padding. A tried source-level reorder that
held A fixed while sweeping N groups stayed correct but regressed CUDAGraph
time to about `2286.8us`, so it was not kept.

### Step 8: v7 Load Count and ATT Checkpoint

The current committed checkpoint is:

```text
c4a9fbf4 Checkpoint runnable Kimi mxfp4 GEMM1 optimization
```

`kimi_fp4_moe_16384_opt.py` is back at module suffix `v7`. This version is
bit-exact against aiter mxfp4 for GEMM1:

```text
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
```

The pipeline kernel count is aligned with aiter mxfp4: both paths launch 8
kernels. GEMM1 launch shape is also equivalent in CTA count:

```text
aiter GEMM1:  wg=(256,1,1), grid=(1570816,1,1) threads = 6136 CTAs
FlyDSL v7:    wg=(256,1,1), grid=(1024,1534,1) threads = 6136 CTAs
```

Static ISA confirms the important global load widths/counts are now aligned:

```text
                 aiter GEMM1    FlyDSL GEMM1 v7
v_mfma           1792           1792
buffer_load      408            408
buffer_load_x4   340            340
buffer_load_dw   68             68
global_load      4              4
ds_read          536            543
ds_write         128            96
s_waitcnt        320            121
s_barrier        30             31
s_nop            123            441
```

So the remaining gap is not extra `buffer_load` instructions or a different
preshuffle load width. The remaining differences are scheduling/LDS details:
FlyDSL still has more compiler scheduler padding and slightly different LDS
traffic.

ATT decode for v7 shows the practical bottleneck more clearly:

```text
aiter GEMM1:
  total cycles 5.49M
  total stalls 3.62M (65.9%)
  MFMA/FMA     1.43M
  VMEM-load    740.9K
  VMEM-wait    644.9K
  LDS          386.6K
  barrier      166.8K

FlyDSL GEMM1 v7:
  total cycles 8.20M
  total stalls 6.61M (80.6%)
  VMEM-wait    2.54M
  MFMA/FMA     1.71M
  VMEM-load    1.28M
  barrier      656.8K
  LDS          147K
  LDS/SMEMwait 129.7K
```

The next optimization target is therefore source scheduling: preserve the
matched load counts but make the K loop closer to aiter's `run_one` sequence,
where B/B-scale loads are issued per J cluster around MFMA clusters instead of
creating long outstanding VMEM regions that later force large waits.

A v8 experiment moved A LDS reads before issuing the next tile's A/B VMEM
prefetch. It remained correct:

```text
cos=1.000000
max_abs=0.000000
```

but regressed CUDAGraph time from the v7 range (`~2059-2078us`) to
`~2105-2148us`, likely because it reduced useful B-load overlap. That source
change was removed and should not be used as the next baseline.

### Step 9: v9-v11 GEMM1 Scheduling Alignment

The next successful direction was to match aiter's K-loop schedule more closely
instead of changing load counts.

`v9` changed B from one-ahead to two-ahead:

```text
initial: load B0 and B1
loop K=i: compute with B_i, issue B_{i+2} by J cluster
```

This stayed bit-exact and improved GEMM1 from the v7 `~997-1042us` profiler
range to:

```text
FlyDSL GEMM1 v9: 964.3us
```

ATT showed this was directionally useful but still wrong for A scheduling:
issuing future A before reading current A caused many compiler-inserted
`s_waitcnt vmcnt(0)` waits at the current A LDS read.

`v10` fixed the order to the aiter shape:

```text
read current A from LDS
issue future A -> LDS
compute current J cluster
issue future B for that J
```

This removed the large pre-A-read VMEM wait cluster. Results:

```text
CUDAGraph short:
  aiter_mxfp4_moe_us=1824.7
  local_mxfp4_opt_gemm1_us=1851.2

Torch profiler:
  aiter GEMM1=729.8us
  FlyDSL GEMM1 v10=752.7us

ATT v10:
  total cycles 5.63M
  total stalls 3.89M (69.1%)
  MFMA/FMA   1.47M
  VMEM-wait  894.6K
  VMEM-load  576.0K
  LDS        413.2K
  barrier    150.5K
```

`v11` then made A two-ahead as well:

```text
initial: load A0/A1 and B0/B1
loop K=i: read A_i, issue A_{i+2}, compute B_i, issue B_{i+2}
barrier wait keeps the newest future A+B tile outstanding
```

This remains bit-exact:

```text
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
```

Longer CUDAGraph run:

```text
CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_opt_gemm1 \
  --warmup 5 --graph-iters 20 --measure 20

aiter_mxfp4_moe_us=1816.8
local_mxfp4_opt_gemm1_us=1845.2
```

Profiler snapshot from a separate run:

```text
aiter GEMM1=729.7us
FlyDSL GEMM1 v11=761.5us
```

ATT v11:

```text
instructions 4479
total cycles 5.83M
total stalls 4.01M (68.8%)
MFMA/FMA   1.53M
VMEM-wait  792.1K
VMEM-load  669.3K
LDS        430.2K
barrier    211.5K
```

Current conclusion: the main scheduling gap has been closed at the Python
level. Static load counts still match aiter (`buffer_load=408`, `MFMA=1792`).
The remaining GEMM1 difference is now small and is likely in lower-level
scheduling details, MFMA operand/register allocation, and the FlyDSL lowering
around waits/barriers rather than missing Python-level work.

## GEMM1 Grid / Resource Audit

The current saved code is commit `1af39d1f Optimize Kimi mxfp4 GEMM1 scheduling`.
The code state intentionally remains v11 after later experiments did not give a
stable win.

GEMM1 launch/resource comparison from rocprof kernel traces:

| kernel | workgroup | grid reported by rocprof | CTA count | LDS | VGPR | AGPR | SGPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aiter GEMM1 | 256 | `(1570816, 1, 1)` | `1570816 / 256 = 6136` | 131072 | 168 | 0 | 112 |
| FlyDSL GEMM1 v11 | 256 | `(1024, 1534, 1)` | `(1024 / 256) * 1534 = 6136` | 131072 | 168 | 0 | 112 |
| FlyDSL GEMM1 v18 experiment | 256 | `(1024, 1534, 1)` | `(1024 / 256) * 1534 = 6136` | 131072 | 168 | 0 | 112 |

So the grid shape differs (`aiter` flattens to 1D, FlyDSL uses `(N block, M
block)`), but the physical CTA count and resource numbers match. Since `N`
has four blocks, FlyDSL's 2D launch maps to the same logical tile order aiter
gets from `pid / 4` and `pid % 4`; changing to 1D would mainly add block-id
decode arithmetic unless the backend scheduler treats 1D/2D dispatch
differently.

Static ATT instruction counts also show that the suspected extra preshuffle
loads are not the remaining issue:

| instruction group | aiter | FlyDSL v11/v18 |
| --- | ---: | ---: |
| `v_mfma` | 1792 | 1792 |
| `buffer_load_dwordx4` | 340 | 340 |
| `buffer_load_dword` | 68 | 68 |
| `global_load` | 4 | 4 |

Remaining notable static differences:

| group | aiter | FlyDSL v11 |
| --- | ---: | ---: |
| total instructions | 4368 | 4479 |
| `s_waitcnt` | 320 | 375 |
| `s_barrier` | 30 | 31 |
| `ds_read` | 536 | 543 |
| `ds_write` | 128 | 96 |
| `v_add*` bucket | 143 | 255 |
| `v_or*` bucket | 55 | 133 |
| max ops | `v_max_f32/v_max3_f32` | `v_maximum3_f32` |

Negative experiments after v11:

- v18 changed MFMA operand packing from padded `i32x8` to direct `i32x4`.
  It stayed bit-exact, but static ISA stayed effectively unchanged and long
  CUDAGraph/profiler runs were not consistently faster. Not retained.
- Waitcnt bitfield encoder experiments with explicit `vmcnt=15/13` were
  bit-exact but slower than the existing raw wait values. Not retained.
- Epilogue `local_max = fabs(result[0])` matched aiter source more closely but
  slowed the measured GEMM1 path. Not retained.
- v20 forced volatile scalar LDS stores in the C-shuffle epilogue to avoid
  `ds_write2_b32`. Without an explicit wait it was numerically unstable; with
  the wait it was bit-exact but slower (`aiter_mxfp4_moe_us=1815.6`,
  `local_mxfp4_opt_gemm1_us=1838.0` in a short graph run). Not retained.

Next optimization targets:

1. Reduce FlyDSL's extra epilogue/address arithmetic (`v_add*`, `v_or*`) if it
   can be done without changing correctness.
2. Investigate why FlyDSL emits more `s_waitcnt` and one extra barrier in the
   generated ISA.
3. Treat the Python-level GEMM1 schedule as close to the current ceiling until
   a lowering/pass or targeted inline-asm change removes static instruction
   differences.

## Relaxed-Correctness GEMM1 v21

The correctness target was relaxed from bit-exact replacement to cosine-level
similarity. Under that target, v21 keeps the v11 schedule and changes only the
epilogue max reduction used for MXFP4 scale selection:

```python
arith.maximumf(...)
```

becomes:

```python
arith.MaxNumFOp(..., fastmath=arith.FastMathFlags.fast)
```

This aligns the generated max instructions more closely with aiter and removes
the `v_maximum3_f32` sequence from FlyDSL GEMM1.

ATT static comparison:

| group | aiter | FlyDSL v11 | FlyDSL v21 |
| --- | ---: | ---: | ---: |
| total instructions | 4368 | 4480 | 4480 |
| `v_mfma` | 1792 | 1792 | 1792 |
| `buffer_load_dwordx4` | 340 | 340 | 340 |
| `buffer_load_dword` | 68 | 68 | 68 |
| `global_load` | 4 | 4 | 4 |
| `v_maximum3` | 0 | 48 | 0 |
| `v_max3_f32` | 24 | 0 | 32 |
| `v_max_f32` | 40 | 0 | 16 |
| `s_waitcnt` | 320 | 375 | 375 |
| `s_barrier` | 30 | 31 | 31 |

The full buffer-load variant breakdown remains exactly matched in v21:

| buffer-load variant | count |
| --- | ---: |
| `buffer_load_dword lds,offen` | 12 |
| `buffer_load_dword offen` | 4 |
| `buffer_load_dword offen,offset` | 52 |
| `buffer_load_dwordx4 lds,offen` | 116 |
| `buffer_load_dwordx4 offen` | 56 |
| `buffer_load_dwordx4 offen,offset` | 168 |

Measurements on GPU 6:

```text
CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_opt_gemm1 \
  --warmup 8 --graph-iters 40 --measure 40

local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1811.1
local_mxfp4_opt_gemm1_us=1833.2
```

Sanity rerun:

```text
CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_opt_gemm1 \
  --warmup 5 --graph-iters 20 --measure 20

local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1814.4
local_mxfp4_opt_gemm1_us=1836.7
```

A direct global-store experiment (`v23`) was also tested to mimic aiter's
`global_store` epilogue instead of FlyDSL `buffer_store`. It was not retained:
it produced `cos=0.999992`, `max_abs=1.007812`, and `local_mxfp4_opt_gemm1_us=1841.0`
in a short graph run.

Current acceptance rule: the GEMM1 replacement no longer needs to be bit-exact
against aiter.  Candidate kernels are acceptable when their final MoE output is
finite and cosine-similar to the aiter mxfp4 path.  This keeps fastmath,
instruction-order, and scale/rounding variants in scope even when `max_abs`
changes.

Additional negative experiments after v21:

- v24 initialized the epilogue max from `fabs(result[0])` like the aiter C++
  source, then reduced the remaining seven values.  It stayed cosine-equivalent
  but regressed the short graph run to about `1839.3us`.  Not retained.
- v25 merged the initial A-scale wait/barrier with the A0/A1/B0/B1 prologue
  wait.  It stayed cosine-equivalent but regressed to about `1840.3us`.  Not
  retained.
- v26 replaced the main-loop `rocdl.s_waitcnt(14)` with inline asm
  `s_waitcnt vmcnt(10)`.  It was slower at about `1840.7us`.  Not retained.
- v27 moved the initial A-scale wait into a combined `vmcnt(20)` prologue wait.
  It stayed cosine-equivalent with small `max_abs` drift but measured about
  `1838.7us`.  Not retained.
- v28 changed B-q loads to use scalar `soffset` plus per-lane `voffset`, closer
  to aiter's source.  It regressed to about `1847.1us`.  Not retained.
- v29 simplified the B-q N address formula using lane-local constants.  It stayed
  cosine-equivalent but regressed to about `1851.0us`.  Not retained.
- v30 added fastmath to the SiLU add/mul chain.  It stayed cosine-equivalent but
  regressed to about `1851.1us`.  Not retained.

Follow-up relaxed-scale experiments:

- v31 forced GEMM1 output scale to exponent byte `123`, which is the dominant
  byte in the aiter output scale distribution for this synthetic benchmark.  It
  failed the relaxed correctness target (`cos=0.790916`) and did not improve
  speed (`local_mxfp4_opt_gemm1_us=1837.5` in a short graph run).  Not retained.
- v32 removed the cross-kk max reduction and broadcast the kk0 lane's scale
  across the 32-column group.  It also failed the relaxed correctness target
  (`cos=0.891668`) and regressed speed (`local_mxfp4_opt_gemm1_us=1841.8` in a
  short graph run).  Not retained.

The best retained GEMM1 was v21 until the later DPP bound-control cleanup below.

### GEMM1 v34 DPP Bound Control

The saved GEMM1 kernel is now:

```text
flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v34_dppbound
```

This keeps the v21 schedule and f32 amax reduction, but changes the explicit
DPP exchange used by the epilogue quad reduction to request `bound_ctrl:1`.
That matches the DPP form emitted by aiter GEMM1:

```text
v_mov_b32_dpp ... quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_mov_b32_dpp ... quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
```

The change is deliberately narrow: MFMA schedule, global/LDS load counts,
wait/barrier placement, and the scale math stay unchanged.  The DPP helper now
accepts an optional `bound_ctrl` argument; existing GEMM2/scatter uses keep the
old default.

Validation:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --graph-iters 20 --measure 20

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1853.5
local_mxfp4_all_flydsl_us=1872.0
```

Graph-profiler GEMM1 samples after the change landed around `705-706us`
(`705.9us` in the last valid ordered run).  For comparison, the previous v21
snapshots in the same graph-profiling setup were typically in the `714-723us`
range.  End-to-end graph replay still has several microseconds of run-to-run
noise, so this is treated as a small GEMM1-local win rather than a full pipeline
match.

Rejected follow-up checks:

- `v33_i32dppamax` replaced the two post-DPP f32 max ops with unsigned integer
  max on the nonnegative f32 bit patterns.  It sometimes measured similarly to
  v34, but it changed the scale-selection path more than necessary and was not
  more stable end-to-end.
- `v34_i32dpp_noexpguard` removed the expert-id validity check after the same
  i32 DPP change.  It stayed correct for the fixed benchmark but did not improve
  GEMM1 timing, so the guard was restored.

Latest CUDAGraph total timings on GPU 6:

```text
CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_opt,local_mxfp4_opt_gemm1 \
  --warmup 5 --graph-iters 20 --measure 20

local_mxfp4_opt_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_opt_gemm1_vs_aiter_mxfp4_cos=1.000000
aiter_mxfp4_moe_us=1813.2
local_mxfp4_opt_us=1833.8
local_mxfp4_opt_gemm1_us=1835.9
```

Latest non-CUDAGraph torch-profiler per-kernel breakdown on GPU 6
(`warmup=3`, `iters=5`):

| path / kernel | avg us |
| --- | ---: |
| aiter sort_count | 6.7 |
| aiter sort_cumsum | 10.7 |
| aiter sort_place_pad | 33.3 |
| aiter quant | 53.3 |
| aiter sort_scales | 60.2 |
| aiter GEMM1 | 718.9 |
| aiter GEMM2 | 907.0 |
| aiter scatter_reduce | 131.1 |
| aiter total | 1921.3 |
| local opt sort_count | 6.7 |
| local opt sort_cumsum | 11.9 |
| local opt sort_place_pad | 35.0 |
| local opt quant | 58.7 |
| local opt sort_scales | 65.2 |
| local opt aiter GEMM1 | 733.1 |
| local opt aiter GEMM2 | 906.4 |
| local opt FlyDSL scatter_reduce | 137.6 |
| local opt total | 1954.6 |
| local opt GEMM1 sort_count | 6.6 |
| local opt GEMM1 sort_cumsum | 10.9 |
| local opt GEMM1 sort_place_pad | 33.7 |
| local opt GEMM1 quant | 58.8 |
| local opt GEMM1 sort_scales | 61.4 |
| local opt GEMM1 FlyDSL GEMM1 v21 | 741.0 |
| local opt GEMM1 aiter GEMM2 | 906.8 |
| local opt GEMM1 FlyDSL scatter_reduce | 137.6 |
| local opt GEMM1 total | 1956.9 |

## Full FlyDSL Initial Migration

The first all-FlyDSL mxfp4 pipeline is now wired as
`local_mxfp4_all_flydsl`.  This path uses fixed-shape FlyDSL kernels for every
runtime stage:

1. `flydsl_kimi_mxfp4_sort_count_NE385_TOPK9_M16384_BM128_v0`
2. `flydsl_kimi_mxfp4_sort_cumsum_NE385_TOPK9_M16384_BM128_v0`
3. `flydsl_kimi_mxfp4_sort_place_pad_NE385_TOPK9_M16384_BM128_v0`
4. `flydsl_kimi_mxfp4_quant_NE385_TOPK9_H7168_M16384_BM128_v0`
5. `flydsl_kimi_mxfp4_sort_scales_NE385_H7168_E512_M16384_BM128_v0`
6. `flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v21`
7. `flydsl_kimi_mxfp4_gemm2_mxfp4out_NE385_H7168_E512_BM128_v0`
8. `flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9`

Correctness checks:

- FlyDSL sort now matches aiter's `cumsum`, `masked_m`, full
  `sorted_expert_ids` including the unused tail, `reverse_sorted` token/topk
  mapping, `m_indices`, and `sorted_weights`.
- FlyDSL quant is bitwise equal to aiter for both `a_quant` and `a_scale`.
- FlyDSL sort_scales is bitwise equal to aiter for the consumed shuffled scale
  byte range.
- End-to-end all-FlyDSL output matches aiter mxfp4:

```text
CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 0 --graph-iters 1 --measure 1

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=2245.7
local_mxfp4_all_flydsl_us=2738.0
```

Initial non-CUDAGraph profiler breakdown (`warmup=1`, `iters=2`):

| stage | kernel | avg us |
| --- | --- | ---: |
| sort_count | FlyDSL v0 | 189.2 |
| sort_cumsum | FlyDSL v0 | 227.1 |
| sort_place_pad | FlyDSL v0 | 194.8 |
| quant | FlyDSL v0 | 61.1 |
| sort_scales | FlyDSL v0 | 58.7 |
| GEMM1 | FlyDSL v21 | 738.6 |
| GEMM2 | FlyDSL v0 | 1233.4 |
| scatter_reduce_q | FlyDSL | 138.4 |
| total | all FlyDSL | 2841.3 |

The aux kernels intentionally prioritize algorithm/precision alignment over
performance.  The obvious next optimization targets are the three sort kernels
and GEMM2 v0.

## Sort Kernel v1 Optimization

The initial FlyDSL sort used global atomics directly in `sort_count` and
`sort_place_pad`, and did most of `sort_cumsum` on thread 0.  v1 ports the key
aiter structure:

- `sort_count`: per-CTA `local_count[NE]` in LDS with LDS atomic add, then one
  global write per expert.
- `sort_place_pad`: per-CTA `local_offsets[NE]` plus `row_starts[NE+1]` in LDS,
  with LDS atomic add for placement.
- `sort_cumsum`: each thread handles one expert's 16 CTA counts, thread 0 only
  performs the prefix scan over padded counts, then experts update
  `block_offsets` and `sorted_expert_ids` in parallel.

Correctness after v1:

- `cumsum`, `masked_m`, full `sorted_expert_ids`, `reverse_sorted` token/topk
  mapping, `m_indices`, and `sorted_weights` match the aiter three-stage sort.
- End-to-end all-FlyDSL output remains bitwise equal to aiter mxfp4:

```text
local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
```

Profiler comparison for the three FlyDSL sort kernels:

| kernel | v0 avg us | v1 avg us |
| --- | ---: | ---: |
| sort_count | 189.2 | 7.2 |
| sort_cumsum | 227.1 | 16.6 |
| sort_place_pad | 194.8 | 35.8 |
| sort total | 611.1 | 59.6 |

The remaining sort gap versus aiter is now small enough that GEMM2 v0 is again
the dominant all-FlyDSL bottleneck.

## Pure Aiter vs Pure FlyDSL Snapshot

`bench_flydsl_16384.py` and `profile_flydsl_16384.py` now auto-select a clean
GPU before importing torch.  The selector uses `rocm-smi`, chooses the physical
GPU with the lowest process VRAM, and exits if no GPU has process VRAM below
1KB.  It binds the process by setting `HIP_VISIBLE_DEVICES`,
and `CUDA_VISIBLE_DEVICES`.  It intentionally does not set
`ROCR_VISIBLE_DEVICES`, because combining ROCR and HIP filters can hide the
selected device when the physical GPU index is not zero.

Measured with the current all-FlyDSL path and no mixed local-opt middle variant:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 3 --iters 10 --top 30

[profile_flydsl_16384] Auto-selected GPU 0 (process_vram=0B, total_used=297766912B)
```

| stage | pure aiter | pure FlyDSL | delta | ratio |
| --- | ---: | ---: | ---: | ---: |
| sort_count | 6.4 us | 6.7 us | +0.3 us | 1.05x |
| sort_cumsum | 10.3 us | 16.2 us | +5.9 us | 1.57x |
| sort_place_pad | 32.4 us | 36.8 us | +4.4 us | 1.14x |
| quant | 58.6 us | 63.0 us | +4.4 us | 1.08x |
| sort_scales | 56.8 us | 57.6 us | +0.8 us | 1.01x |
| GEMM1 | 716.5 us | 735.3 us | +18.8 us | 1.03x |
| GEMM2 | 892.8 us | 1176.8 us | +284.0 us | 1.32x |
| scatter_reduce | 135.2 us | 148.4 us | +13.2 us | 1.10x |
| total | 1909.1 us | 2240.8 us | +331.7 us | 1.17x |

Both paths launch 8 device kernels.  On a clean GPU, sort_scales is essentially
matched; the remaining all-FlyDSL gap is dominated by GEMM2.

CUDAGraph end-to-end timing from the same snapshot:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --graph-iters 20 --measure 20

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1855.2
local_mxfp4_all_flydsl_us=2146.7
```

GEMM2 v1 persistent resource comparison:

| kernel | grid | LDS | SGPR | VGPR | AGPR | SGPR spill | VGPR spill | private segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aiter GEMM2 nonatomic mxfp4out | 256 CTAs | 133120 B | 66 | 288 | 128 | 0 | 0 | 0 B |
| FlyDSL GEMM2 v1 persistent | 256 CTAs | 133120 B | 53 | 272 | 16 | 0 | 0 | 0 B |

So the remaining GEMM2 gap is not a scratch-spill problem.  Both kernels use
the same LDS allocation and neither spills SGPR/VGPR.  The main remaining
resource/scheduling difference is accumulator placement and generated schedule:
aiter reserves 128 AGPRs for the nonatomic BM128 path, while the current FlyDSL
kernel reports only 16 AGPRs and relies mostly on VGPR/ordinary generated MFMA
values.

## GEMM2 v2 AGPR Hint and GPU Selection Audit

The GPU selector now checks both process VRAM and instantaneous GPU busy
percentage before binding torch.  It still uses the physical `rocm-smi`
`cardN` / `GPU[N]` index for `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES`, but now requires:

```text
process_vram < 1024 B
GPU use <= 0%
```

This avoids the ROCm-SMI `showpids` table ambiguity where the `GPU(s)` column
can show per-process visible ordinals; the selector uses the physical
`showpidgpus` DRM-device mapping plus total card memory and `showuse`.

GEMM2 v2 changed only codegen hints:

```python
value_attrs={"passthrough": [["amdgpu-agpr-alloc", "128,128"]]}
compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
```

Resource metadata for v2:

| kernel | grid | LDS | SGPR | VGPR | AGPR | SGPR spill | VGPR spill | private segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FlyDSL GEMM2 v2 agprhint | 256 CTAs | 133120 B | 53 | 348 | 128 | 0 | 0 | 0 B |

The AGPR count now matches aiter, but VGPR rises substantially.  A profiler run
on a clean GPU measured:

```text
aiter GEMM2: 902.3 us
FlyDSL GEMM2 v2 agprhint: 1072.3 us
```

An additional v3 experiment staged A scales through LDS, matching aiter's
nonatomic scale path more closely.  It stayed correct but regressed:

```text
aiter GEMM2: 911.4 us
FlyDSL GEMM2 v3 agpr_scalelds: 1090.9 us
```

The v3 scale-staging edit was rejected; the saved GEMM2 code is v2 agprhint.

## GEMM2 v4-v8 Cleanup and DPP Amax

The later GEMM2 experiments were kept in
`kimi_fp4_moe_16384_opt.py` only.  The current saved kernel is:

```text
flydsl_kimi_mxfp4_gemm2_mxfp4out_NE385_H7168_E512_BM128_v8_dppamax
```

Accumulated changes after v2:

| version | change | result |
| --- | --- | --- |
| v4_noguard | Removed redundant block/expert/row validity checks after switching to persistent work over valid blocks. | Correct, small speedup. |
| v5_xloadpacked | Fixed X DMA byte count to use packed fp4 bytes, reducing A `buffer_load_dwordx4` from 32 to 24. | Correct by cosine, large speedup. |
| v6_scalelds | Staged A scales through LDS, matching aiter's scale-sharing path more closely. | Correct, reduced scalar global loads. |
| v7_noendbar | Removed the end-of-work barrier and relied on the next persistent iteration's entry barrier. | Correct, small speedup. |
| v8_dppamax | Replaced epilogue `shuffle_xor` amax reduction with explicit DPP update intrinsic. | Correct, large speedup; GEMM2 nearly matches aiter. |

Profiler command:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 3 --iters 10 --top 30
```

Latest clean-GPU profiler snapshot:

| stage | pure aiter | pure FlyDSL v8 | delta |
| --- | ---: | ---: | ---: |
| sort_count | 6.7 us | 7.2 us | +0.5 us |
| sort_cumsum | 10.9 us | 16.4 us | +5.5 us |
| sort_place_pad | 33.1 us | 37.6 us | +4.5 us |
| quant | 60.0 us | 63.0 us | +3.0 us |
| sort_scales | 61.2 us | 57.8 us | -3.4 us |
| GEMM1 | 752.0 us | 748.4 us | -3.6 us |
| GEMM2 | 912.6 us | 917.2 us | +4.6 us |
| scatter_reduce | 138.0 us | 156.4 us | +18.4 us |
| total | 1974.5 us | 2004.0 us | +29.5 us |

Both paths still launch 8 device kernels per iteration.  GEMM2 is no longer the
dominant gap in this snapshot; the remaining all-FlyDSL gap is mostly
scatter_reduce plus small sort/cumsum overhead.

CUDAGraph end-to-end timing:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --graph-iters 20 --measure 20

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_us=1859.3
local_mxfp4_all_flydsl_us=1909.0
```

v8 ISA comparison against the aiter mxfp4out kernel:

| instruction/resource | aiter | FlyDSL v8 |
| --- | ---: | ---: |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 128 | 128 |
| `buffer_load_dwordx4` | 24 | 24 |
| `buffer_load_dword` | 6 | 8 |
| `ds_write_b32` | 128 | 128 |
| `ds_read_b128` | 64 | 64 |
| `s_barrier` | 4 | 4 |
| `s_waitcnt` | 54 | 46 |
| amax quad reduction | `v_max_u32_dpp` x32 | `v_mov_b32_dpp` x32 + `v_max_u32_e32` x32 |
| output stores | `global_store_*` x32 | `buffer_store_*` x32 |
| LDS | 133120 B | 133120 B |
| AGPR | 128 | 128 |
| VGPR | 384 | 364 |
| SGPR | 73 | 48 |
| spills/private segment | 0 / 0 B | 0 / 0 B |

The v8 DPP change removes the previous `ds_swizzle_b32` epilogue lowering.  The
remaining low-level differences worth checking next are the output store form
(`global_store_* nt` in aiter versus `buffer_store_*` in FlyDSL) and the
FlyDSL scatter_reduce kernel.

## Scatter Reduce v4 Route Vector

The saved scatter kernel in `kimi_fp4_moe_16384_opt.py` is now:

```text
flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9_v4_routevec
```

The v4 change keeps the v3 raw global I/O path and adds one vector preload for
the first eight `reverse_sorted[token, topk]` route ids:

- routes 0..7: one `vector<8xi32>` load, lowering to `s_load_dwordx8`
- route 8: one scalar `s_load_dword`

Correctness remains exact against aiter for the fixed test input:

```text
local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
```

Profiler snapshot:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --iters 30 --top 30
```

| stage | aiter | FlyDSL v4 | delta |
| --- | ---: | ---: | ---: |
| sort_count | 6.3 us | 6.7 us | +0.4 us |
| sort_cumsum | 10.4 us | 15.6 us | +5.2 us |
| sort_place_pad | 32.3 us | 35.9 us | +3.6 us |
| quant | 58.4 us | 58.9 us | +0.5 us |
| sort_scales | 55.8 us | 54.3 us | -1.5 us |
| GEMM1 | 706.8 us | 718.8 us | +12.0 us |
| GEMM2 | 875.5 us | 865.8 us | -9.7 us |
| scatter_reduce | 134.5 us | 140.0 us | +5.5 us |
| total | 1880.0 us | 1896.1 us | +16.1 us |

CUDAGraph end-to-end timing:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --graph-iters 20 --measure 20

aiter_mxfp4_moe_us=1859.5
local_mxfp4_all_flydsl_us=1883.3
```

Scatter ISA/resource comparison:

| instruction/resource | aiter NT | FlyDSL v3 globalio | FlyDSL v4 routevec |
| --- | ---: | ---: | ---: |
| `v_pk_fma_f32` | 36 | 36 | 36 |
| `v_cvt_scalef32_pk_f32_fp4` | 36 | 36 | 36 |
| `v_cvt_pk_bf16_f32` | 4 | 4 | 4 |
| `global_load_dword` | 18 | 18 | 18 |
| `s_load_dword` | 11 | 18 | 10 |
| `s_load_dwordx8` | 2 | 1 | 2 |
| `global_store_dwordx4` | 1 | 1 | 1 |
| `s_waitcnt` | 19 | 35 | 27 |
| SGPR | 34 | 36 | 38 |
| VGPR | 62 | 28 | 29 |
| spills/private segment | 0 / 0 B | 0 / 0 B | 0 / 0 B |

v4 reduces the remaining scatter gap from the earlier `~18us` snapshot to about
`5.5us` in the latest profiler run.  A v5 experiment staged all route
weight/scale/q values before decode to increase ILP, but it measured
`141.1us`, so it was rejected.  The remaining difference is likely mostly
instruction scheduling: aiter spends many more VGPRs to keep more memory results
live, while the current FlyDSL version is still more serialized around waits.

## Sort Cumsum v2-v4

After committing scatter v4, the sort-cumsum path was optimized in
`kimi_fp4_moe_16384_opt.py`.  The saved cumsum kernel is now:

```text
flydsl_kimi_mxfp4_sort_cumsum_NE385_TOPK9_M16384_BM128_v4_globalio
```

Changes:

- v2 removed the global `expert_starts` scratch buffer.  `sort_cumsum` now keeps
  expert starts only in LDS, and `sort_place_pad` reads the final row start from
  `cumsum_tensor[0]`, matching aiter's data flow more closely.
- v2 also removed the single-thread tail clear of `sorted_expert_ids`; aiter
  does not clear blocks after `cumsum_tensor[0]`.
- v3 changed the tid0 prefix scan over `NE=385` experts from a fully unrolled
  `range_constexpr` to an explicit dynamic `scf.for`, reducing the huge unrolled
  LDS read/write sequence.
- v4 switched `sort_cumsum` I/O from buffer resources to raw global pointer
  loads/stores.  The first pass over `block_offsets[e, 0:16]` uses
  `vector<4xi32>` loads, lowering closer to aiter's `global_load_dwordx4`.

Validation:

```text
local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
```

Profiler snapshot:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --iters 30 --top 30
```

| stage | aiter | FlyDSL after cumsum v4 | delta |
| --- | ---: | ---: | ---: |
| sort_count | 5.9 us | 6.8 us | +0.9 us |
| sort_cumsum | 10.2 us | 9.6 us | -0.6 us |
| sort_place_pad | 32.4 us | 35.7 us | +3.3 us |
| quant | 58.4 us | 59.5 us | +1.1 us |
| sort_scales | 55.9 us | 55.1 us | -0.8 us |
| GEMM1 | 702.5 us | 723.5 us | +21.0 us |
| GEMM2 | 863.4 us | 871.0 us | +7.6 us |
| scatter_reduce | 134.9 us | 140.6 us | +5.7 us |
| total | 1863.6 us | 1901.7 us | +38.1 us |

The cumsum kernel itself moved from the previous `~15.6us` range to `9.6us`,
slightly faster than aiter in this run.  v4 was kept; v2/v3 were intermediate.

CUDAGraph end-to-end timing after cumsum v4:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
  --warmup 5 --graph-iters 20 --measure 20

aiter_mxfp4_moe_us=1851.1
local_mxfp4_all_flydsl_us=1878.7
```

At this point `sort_cumsum` is no longer a remaining gap.  The still-visible
sort-side delta is mostly `sort_place_pad`.

## GPU Selection Note

`gpu_select.py` intentionally treats `rocm-smi cardN` / `GPU[N]` as the
physical GPU index.  It binds the process with:

```text
HIP_VISIBLE_DEVICES=N
CUDA_VISIBLE_DEVICES=N
```

Torch then sees the selected physical GPU as logical `cuda:0`.  The selector
does not set `ROCR_VISIBLE_DEVICES`, because setting both HIP and ROCR filters
can remap or hide devices unexpectedly.  The current safety gate requires
process VRAM below 1024 bytes and GPU use at 0 percent before selecting a card.

## Graph Replay Measurement Defaults

`bench_flydsl_16384.py` now defaults to a longer graph replay measurement:

```text
--warmup 20 --graph-iters 40 --measure 40 --repeat 3
```

Each repeat captures a fresh graph and reports the median elapsed time per
logical MoE invocation.  The printed `<runner>_us` value is the median across
the repeats, with `<runner>_samples_us` showing the individual sample medians.

`profile_flydsl_16384.py` now profiles graph replay by default with:

```text
--warmup 20 --iters 5 --graph-iters 20 --repeat 5 --expected-kernels 8
```

The profiler keeps the per-sample window at 100 logical iterations because
larger windows on this setup often drop device events.  The script rejects
samples whose device event stream cannot be split into exactly 8 kernels per
logical iteration, then reports median per-kernel timings and an ordered kernel
diff.

Latest graph replay end-to-end timing on auto-selected GPU/card 3:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl

aiter_mxfp4_moe_samples_us=1847.5,1841.6,1841.3 range_us=1841.3..1847.5
aiter_mxfp4_moe_us=1841.6
local_mxfp4_all_flydsl_samples_us=1864.0,1865.8,1863.5 range_us=1863.5..1865.8
local_mxfp4_all_flydsl_us=1864.0
```

Latest graph replay profiler median over 5 valid samples:

| stage | aiter | FlyDSL all | delta |
| --- | ---: | ---: | ---: |
| sort_count | 6.4 us | 6.4 us | +0.1 us |
| sort_cumsum | 10.3 us | 9.6 us | -0.7 us |
| sort_place_pad | 32.0 us | 35.0 us | +3.0 us |
| quant | 59.8 us | 59.8 us | -0.0 us |
| sort_scales | 55.2 us | 54.5 us | -0.7 us |
| GEMM1 | 698.9 us | 703.9 us | +5.0 us |
| GEMM2 | 855.0 us | 852.0 us | -3.0 us |
| scatter_reduce | 135.6 us | 139.7 us | +4.1 us |
| total kernel sum | 1853.2 us | 1861.2 us | +8.0 us |

The profiler kernel sum is useful for stage attribution, but the event-timed
bench graph replay remains the authoritative end-to-end number.  In this
snapshot the main remaining profiler-visible gaps are `GEMM1`, `scatter_reduce`,
and `sort_place_pad`; `GEMM2`, `sort_cumsum`, and `sort_scales` are roughly
matched or slightly ahead in FlyDSL.

## Sort Place/Scatter v4-v6 Alignment

The saved sort placement kernel is now:

```text
flydsl_kimi_mxfp4_sort_place_pad_NE385_TOPK9_M16384_BM128_v4_globalio
```

Changes:

- Match aiter's phase order: place real routed tokens first, then barrier, then
  fill the padded tail for each expert.
- Switch `sort_place_pad` from buffer resources to raw global loads/stores,
  matching the plain pointer access style used by aiter's HIP kernel.

The saved scatter kernel is now:

```text
flydsl_kimi_mxfp4_scatter_reduce_q_NE385_H7168_E512_M16384_TOPK9_v6_weightpreload
```

Change:

- Keep the v4 route-vector preload and raw global I/O, but preload only the 9
  route weights before entering the decode/FMA body.  Unlike the rejected v5
  experiment, this does not stage all q/scale payloads, so it adds latency
  hiding without substantially increasing vector-register pressure.

Validation:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl --warmup 5 \
  --graph-iters 20 --measure 10 --repeat 1

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
```

Profiler baseline at the start of this round, before the changes:

| stage | aiter | previous FlyDSL | delta |
| --- | ---: | ---: | ---: |
| sort_place_pad | 32.2 us | 35.2 us | +3.0 us |
| scatter_reduce | 135.0 us | 140.2 us | +5.2 us |

Profiler after the changes, run 1:

| stage | aiter | FlyDSL v4/v6 | delta |
| --- | ---: | ---: | ---: |
| sort_place_pad | 32.1 us | 34.1 us | +2.0 us |
| scatter_reduce | 135.5 us | 130.5 us | -5.0 us |
| total kernel sum | 1860.1 us | 1860.1 us | +0.0 us |

Profiler after the changes, run 2:

| stage | aiter | FlyDSL v4/v6 | delta |
| --- | ---: | ---: | ---: |
| sort_place_pad | 31.8 us | 34.3 us | +2.5 us |
| scatter_reduce | 134.6 us | 130.6 us | -4.0 us |
| total kernel sum | 1853.4 us | 1861.7 us | +8.3 us |

Default graph replay end-to-end timing after the changes:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl

aiter_mxfp4_moe_samples_us=1872.3,1876.8,1856.9 range_us=1856.9..1876.8
aiter_mxfp4_moe_us=1872.3
local_mxfp4_all_flydsl_samples_us=1864.4,1864.2,1862.8 range_us=1862.8..1864.4
local_mxfp4_all_flydsl_us=1864.2
```

Interpretation: `scatter_reduce` is now faster than aiter in the graph-profiler
breakdown.  `sort_place_pad` improved by about 1 us but is still roughly
2-2.5 us slower than aiter, so the next sort-side work should focus there.

## Sort Place v5 Dynamic Placement Loop

Current saved sort placement kernel:

```text
flydsl_kimi_mxfp4_sort_place_pad_NE385_TOPK9_M16384_BM128_v5_placeloop
```

Change from v4:

- Keep the raw global I/O and aiter-like phase order from v4.
- Replace the compile-time unrolled real-token placement loop with one dynamic
  `scf.ForOp(c_start + tx, end, 1024)`.
- Keep the padding loop compile-time unrolled.  A dynamic padding-loop
  experiment did not improve the graph-profiler timing.

Rejected follow-up:

- `v6_exactend` removed the `min(c_start + per_cta, total_pairs)` end bound
  because this shape has exactly `16 * 9216 == 16384 * 9` routed pairs.  It was
  slower in the graph profiler, so the saved version keeps the v5 bound form.

Graph replay profiler median over 5 valid samples:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl --max-retries 10
```

| stage | aiter | FlyDSL v5 | delta |
| --- | ---: | ---: | ---: |
| sort_count | 6.0 us | 6.4 us | +0.5 us |
| sort_cumsum | 10.2 us | 9.7 us | -0.5 us |
| sort_place_pad | 31.9 us | 32.9 us | +1.0 us |
| quant | 58.8 us | 59.4 us | +0.6 us |
| sort_scales | 55.1 us | 54.9 us | -0.2 us |
| GEMM1 | 698.9 us | 710.9 us | +12.0 us |
| GEMM2 | 859.9 us | 858.2 us | -1.6 us |
| scatter_reduce | 134.5 us | 130.5 us | -4.1 us |
| total kernel sum | 1855.0 us | 1863.3 us | +8.3 us |

Default graph replay end-to-end timing:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_samples_us=1864.9,1856.7,1853.7 range_us=1853.7..1864.9
aiter_mxfp4_moe_us=1856.7
local_mxfp4_all_flydsl_samples_us=1862.3,1863.3,1862.6 range_us=1862.3..1863.3
local_mxfp4_all_flydsl_us=1862.6
```

Interpretation: v5 recovers most of the previous `sort_place_pad` gap, from
roughly +2-3 us down to about +1 us in graph replay profiling.  The all-FlyDSL
path still launches the same 8 kernels as aiter; the remaining graph-profiler
gap is mostly `GEMM1`, with `sort_place_pad` now a smaller residual item.

Rejected v7 padding-loop follow-up:

- `v7_hippadloop` changed the padding expert loop to an explicit dynamic
  `scf.ForOp(e_lo, e_hi, 1)`, closer to the aiter HIP source.
- Static ISA became much smaller: v5 `sort_place_pad` was about 1079 assembly
  lines with 79 static `global_store_dword` and 79 static `s_cbranch`
  occurrences; v7 dropped to about 201 lines, 7 stores, and 8 branches.
- Measured graph performance did not improve.  A 3-sample graph profiler saw
  `sort_place_pad` around 33.0 us, and the default graph replay bench reported
  `local_mxfp4_all_flydsl_us=1864.1`, slightly slower than saved v5's
  `1862.6`.  The saved code therefore remains v5.

## Sort Place v14 Padding Lane Mask

Current saved sort placement kernel:

```text
flydsl_kimi_mxfp4_sort_place_pad_NE385_TOPK9_M16384_BM128_v14_padlanemask
```

Change from v5:

- Keep the v5 dynamic placement loop.
- Add a padding-phase guard `tx < BM` before the unrolled expert padding loop.
  Padding per expert is always in `[0, BM)`, because `sort_cumsum` rounds each
  expert's real count up to a multiple of `BM=128`.  Therefore lanes
  `tx>=128` cannot write padding and can skip the whole padding phase.

Rejected follow-ups while searching for this:

- `v8_udivrem`, `v9_udivmulsub`, `v10_divsimulsub`, and `v13_magicdiv9`
  tried to replace `token_id = idx / 9` and `topk_id = idx % 9` lowering.
  All were correct but slower in graph profiling.
- `v11_ldsorder` changed sort-place LDS layout to look more like aiter's
  compiler-chosen layout.  It was correct but slower in the short graph bench.
- `v12_i32while` changed the placement loop from index `scf.ForOp` to an i32
  `scf.WhileOp`.  It was correct but did not improve `sort_place_pad`.

Graph replay profiler median over 5 valid samples:

```text
/opt/venv/bin/python profile_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl --max-retries 15
```

| stage | aiter | FlyDSL v14 | delta |
| --- | ---: | ---: | ---: |
| sort_count | 6.3 us | 6.5 us | +0.2 us |
| sort_cumsum | 10.4 us | 9.7 us | -0.7 us |
| sort_place_pad | 32.0 us | 32.1 us | +0.1 us |
| quant | 59.5 us | 59.7 us | +0.1 us |
| sort_scales | 55.4 us | 55.0 us | -0.4 us |
| GEMM1 | 699.3 us | 711.8 us | +12.4 us |
| GEMM2 | 861.8 us | 855.4 us | -6.4 us |
| scatter_reduce | 135.3 us | 130.4 us | -4.8 us |
| total kernel sum | 1860.6 us | 1860.7 us | +0.1 us |

Default graph replay end-to-end timing:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_samples_us=1890.5,1852.8,1855.9 range_us=1852.8..1890.5
aiter_mxfp4_moe_us=1855.9
local_mxfp4_all_flydsl_samples_us=1879.8,1865.2,1864.9 range_us=1864.9..1879.8
local_mxfp4_all_flydsl_us=1865.2
```

Interpretation: `sort_place_pad` is now effectively aligned in graph replay
profiling, with the remaining measured gap around +0.1 us.  End-to-end graph
timing is still dominated by GEMM variance and should not be used to judge this
specific sort-place change in isolation.

## GEMM1 v36 AGPR Inline MFMA

Current saved GEMM1 kernel:

```text
flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM128_v36_agprasm
```

Motivation:

- aiter BM128 GEMM1 uses inline asm constraints to keep accumulators in AGPR:
  `=a` for the first zero-init MFMA and tied `+a` for accumulation.
- The previous FlyDSL `v34_dppbound` used the ROCDL MFMA intrinsic for the main
  loop.  Its final ISA had all MFMA destinations in VGPR, with resource usage
  around `VGPR=256, AGPR=80` and graph-profiler GEMM1 medians about
  `707-709 us`, roughly `+10-14 us` slower than aiter.

Change:

- Keep the v34 schedule, loads, waits, barriers, and epilogue.
- Replace only the GEMM1 main-loop MFMA calls in `compute_tile_jmajor*` with
  inline asm using the same instruction:
  `v_mfma_scale_f32_16x16x128_f8f6f4`.
- Use `=a,v,v,v,v` for the first MFMA of each accumulator and
  `=a,v,v,v,v,0` for subsequent tied accumulation.  The tied form maps to the
  same LLVM constraint shape emitted by clang for HIP `"+a"` inline asm.

Rejected checks while developing v36:

- `v35_agprinit` changed only the first MFMA to `=a` and left later MFMA calls
  as ROCDL intrinsics.  It compiled but failed correctness
  (`cos=0.960782`), because LLVM did not preserve the AGPR accumulator chain in
  the same way aiter's tied asm does.
- `v35_inlinevgprtest` used the same inline asm text with `=v` output.  It was
  bitwise correct, proving the operand packing and `op_sel` mapping were
  correct; the issue was specifically the incomplete AGPR constraint chain.

Resource / ISA checks:

| kernel | MFMA dest AGPR | MFMA src acc AGPR | VGPR / AGPR | spills |
| --- | ---: | ---: | ---: | ---: |
| aiter BM128 | 1792 / 1792 | 1760 / 1792 | 165 / 168 | 0 |
| FlyDSL v34 | 0 / 1792 | 0 / 1792 | 256 / 80 | 0 |
| FlyDSL v36 | 1792 / 1792 | 1760 / 1792 | 169 / 128 | 0 |

Static instruction counts relevant to scheduling:

| kernel | `s_waitcnt` | `s_barrier` | `buffer_load_dwordx4` | `buffer_load_lds` | `ds_read_b128` |
| --- | ---: | ---: | ---: | ---: | ---: |
| aiter BM128 | 320 | 30 | 340 | 128 | 480 |
| FlyDSL v34 | 375 | 31 | 340 | 128 | 480 |
| FlyDSL v36 | 287 | 31 | 340 | 128 | 480 |

Default graph-profiler validation (`warmup=20`, `iters=5`,
`graph_iters=20`, `repeat=5`; only `--max-retries 30` was changed to tolerate
profiler event drops):

| run | aiter GEMM1 | FlyDSL v36 GEMM1 | delta |
| --- | ---: | ---: | ---: |
| 1 | 695.5 us | 694.5 us | -1.1 us |
| 2 | 696.9 us | 693.5 us | -3.4 us |
| 3 | 695.1 us | 694.1 us | -1.0 us |

Full graph-profiler totals from the same runs:

| run | aiter total | FlyDSL all-FlyDSL total | delta |
| --- | ---: | ---: | ---: |
| 1 | 1856.8 us | 1843.5 us | -13.4 us |
| 2 | 1858.4 us | 1842.3 us | -16.1 us |
| 3 | 1851.5 us | 1842.7 us | -8.8 us |

Default graph replay correctness / timing:

```text
/opt/venv/bin/python bench_flydsl_16384.py \
  --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl

local_mxfp4_all_flydsl_vs_aiter_mxfp4_cos=1.000000
local_mxfp4_all_flydsl_vs_aiter_mxfp4_max_abs=0.000000
aiter_mxfp4_moe_samples_us=1859.5,1855.5,1850.2 range_us=1850.2..1859.5
aiter_mxfp4_moe_us=1855.5
local_mxfp4_all_flydsl_samples_us=1841.5,1839.9,1841.0 range_us=1839.9..1841.5
local_mxfp4_all_flydsl_us=1841.0
```

Interpretation: GEMM1 is now effectively aligned with aiter for the fixed
M=16384 Kimi mxfp4 path.  In the three default graph-profiler runs above,
FlyDSL v36 GEMM1 is slightly faster than aiter by about `1-3 us`, and the
all-FlyDSL pipeline remains bitwise equal to aiter for the benchmark output.

## Small-M BM16 GEMM1 Pipeline v1

Target:

```text
mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT
mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT
```

Change in `kimi_fp4_moe_small_bm16.py`:

- Switch GEMM1 inline quant packing from the f32 fp4 intrinsic to the bf16
  intrinsic used by aiter:
  `llvm.amdgcn.cvt.scalef32.pk.fp4.bf16`.
- Split inline quant into hidden load and finish/write phases.
- Add 3 physical A LDS slots and 2 B register stages.
- Preload the first two K tiles, then in the main K loop compute the current
  tile while issuing the next tile's B loads and finishing the next tile's
  hidden inline quant into a separate LDS slot.
- Keep the drained last two K tiles as pure compute, matching the aiter
  `kStages=2` structure.

Default graph replay measurement used the then-current longer `bench.py`
defaults:

```text
warmup=100 graph_iters=2000 measure=51 graph_warmup_replays=5
```

Stage-delta graph replay result after the pipeline change:

| M | aiter | FlyDSL sort + aiter GEMM | FlyDSL GEMM1 + aiter GEMM2 | all FlyDSL | sort delta | GEMM1 delta | GEMM2 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39.164 us | 40.300 us | 52.012 us | 53.232 us | +1.135 us | +11.712 us | +1.220 us |
| 8 | 58.727 us | 59.326 us | 66.738 us | 68.100 us | +0.599 us | +7.412 us | +1.362 us |
| 16 | 96.095 us | 97.186 us | 99.299 us | 100.989 us | +1.091 us | +2.113 us | +1.690 us |
| 32 | 146.344 us | 147.723 us | 152.987 us | 154.751 us | +1.380 us | +5.264 us | +1.764 us |
| 64 | 200.434 us | 202.720 us | 211.454 us | 214.212 us | +2.286 us | +8.734 us | +2.758 us |
| 128 | 251.650 us | 255.206 us | 261.301 us | 265.713 us | +3.556 us | +6.095 us | +4.412 us |

Correctness stayed close to aiter for every row above:

```text
cos(sort_aiter, aiter)       >= 0.999996
cos(gemm1fly_aiter, aiter)   >= 0.999997
cos(allfly, aiter)           >= 0.999997
```

Interpretation: the previous BM16 FlyDSL GEMM1 path serialized inline quant,
B loads, wait/barrier, and MFMA for every K tile.  The v1 pipeline removes a
large part of that bubble.  The largest remaining gaps are now `M=4`
(`+11.7 us` GEMM1) and `M=64` (`+8.7 us` GEMM1), while `M=16` is within about
`+2.1 us` for GEMM1.

## Bench.py Accurate Small-M Defaults

`bench.py` defaults were increased again for small-M decisions, where the
pipeline is only tens of microseconds and single replay/event overhead can hide
real differences:

```text
warmup=200 graph_iters=5000 measure=101 graph_warmup_replays=10
eager_iters=5000
```

Validation command:

```text
/opt/venv/bin/python bench.py -M 4,8,16,32,64,128 --ref-max-M 128
```

This run compares aiter's `mxfp4_moe` CSV path against the original untagged
`fused_moe` FlyDSL CSV path, not the local extracted BM16 file.

| M | mxfp4 | FlyDSL fused_moe | ratio | mx.fly | mx.ref | fly.ref |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39.2 us | 52.3 us | 1.34x | 0.9845 | 0.9533 | 0.9482 |
| 8 | 58.4 us | 71.9 us | 1.23x | 0.9868 | 0.9524 | 0.9485 |
| 16 | 95.9 us | 115.4 us | 1.20x | 0.9876 | 0.9538 | 0.9515 |
| 32 | 145.8 us | 160.7 us | 1.10x | 0.9868 | 0.9548 | 0.9520 |
| 64 | 200.3 us | 209.7 us | 1.05x | 0.9874 | 0.9555 | 0.9529 |
| 128 | 251.4 us | 262.8 us | 1.05x | 0.9875 | 0.9562 | 0.9536 |

## Small-M BM16 Stage Split

`bench_small_bm16.py` was added to make the BM16 stage split reproducible.  It
compares:

- `aiter`: aiter `mxfp4_moe` CSV path.
- `sort_aiter`: FlyDSL sort/zero-init plus aiter GEMM1/GEMM2.
- `gemm1fly_aiter`: FlyDSL sort/GEMM1 plus aiter GEMM2.
- `allfly`: FlyDSL sort/GEMM1/GEMM2.

Default command:

```text
/opt/venv/bin/python bench_small_bm16.py -M 4,8,16,32,64,128
```

Default graph timing was:

```text
warmup=200 graph_iters=5000 measure=101 graph_warmup_replays=10 repeat=1
```

Results:

| M | aiter | sort_aiter | gemm1fly_aiter | allfly | sort delta | GEMM1 delta | GEMM2 delta | cos all | max abs all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39.351 us | 42.560 us | 42.446 us | 42.535 us | +3.210 us | -0.114 us | +0.089 us | 0.999996305 | 0.031250 |
| 8 | 59.011 us | 59.411 us | 60.372 us | 61.925 us | +0.400 us | +0.962 us | +1.553 us | 0.999998152 | 0.015625 |
| 16 | 96.219 us | 97.519 us | 100.646 us | 102.193 us | +1.300 us | +3.128 us | +1.547 us | 0.999998629 | 0.015625 |
| 32 | 145.850 us | 147.566 us | 150.351 us | 153.022 us | +1.717 us | +2.785 us | +2.671 us | 0.999999046 | 0.015625 |
| 64 | 200.427 us | 202.655 us | 207.603 us | 210.420 us | +2.228 us | +4.948 us | +2.816 us | 0.999998689 | 0.031250 |
| 128 | 251.346 us | 254.723 us | 260.665 us | 264.844 us | +3.377 us | +5.942 us | +4.179 us | 0.999998510 | 0.031250 |

Interpretation: after the BM16 GEMM1 v4 alignment, the remaining small-M gap is
not one uniform issue.  M=4 is almost entirely sort/prologue overhead.  M=64 and
M=128 are dominated by GEMM1, with GEMM2 as the second largest term.  The next
kernel-level optimization should therefore start from GEMM1 for M=64/128, while
keeping sort/prologue in mind for the very small M=4 case.

Profiler sanity check for M=64 (`warmup=20`, `graph_iters=20`, `replays=5`)
confirmed the stage split is real kernel time, not extra graph kernels:

| runner | sort | GEMM1 | GEMM2 | total |
| --- | ---: | ---: | ---: | ---: |
| `aiter` | 5.391 us | 135.481 us | 66.707 us | 207.579 us |
| `sort_aiter` | 7.324 us | 135.804 us | 66.822 us | 209.950 us |
| `gemm1fly_aiter` | 7.275 us | 142.643 us | 66.625 us | 216.543 us |
| `allfly` | 7.233 us | 142.682 us | 69.813 us | 219.727 us |

Rejected follow-ups:

- GEMM2 barrier relaxation: replacing the pre-compute
  `rocdl.s_waitcnt(0); gpu.barrier()` with `_barrier(lgkmcnt=0)` was not
  correct.  M=8/16 cosine dropped to about `0.80`, so the current GEMM2 path
  still needs a stronger VMEM/LDS ordering point.
- GEMM1 preload v5: interleaving the initial two K-tile preload as
  half-quant/B-load/half-quant/B-load was correct (`min_cos=0.999997199`) but
  did not improve performance.  Short graph profiler changed M=64 GEMM1 only
  from about `142.643 us` to `142.564 us`, and long graph timing made the
  M=64 GEMM1 stage delta worse (`+6.861 us`).  The saved code stays at GEMM1
  v4.

## Small-M BM16 GEMM1 v7 Epilogue Cleanup

Change:

- Match aiter BM16 scale epilogue and write only the low scale byte.  The
  adjacent byte is padding for BM16 and is not read by GEMM2.
- Replace the epilogue f32 amax `shuffle_xor` reductions with DPP controls
  `0xB1` and `0x4E`, matching aiter's `apply_cshuffle_quant_epilog`.

Correctness sweep against aiter mxfp4:

```text
M=4   cos=0.999997199 max_abs=0.031250
M=8   cos=0.999997497 max_abs=0.031250
M=16  cos=0.999998033 max_abs=0.015625
M=32  cos=0.999997973 max_abs=0.031250
M=64  cos=0.999998271 max_abs=0.031250
M=128 cos=0.999999046 max_abs=0.031250
min_cos=0.999997199
```

Static ISA comparison:

| kernel | `s_waitcnt` | `buffer_store_byte` | `ds_swizzle_b32` | `v_mov_b32_dpp` |
| --- | ---: | ---: | ---: | ---: |
| FlyDSL GEMM1 v4 | 152 | 2 | 2 | 0 |
| FlyDSL GEMM1 v7 | 150 | 1 | 0 | 2 |

Short graph profiler (`warmup=20`, `graph_iters=20`, `replays=5`) shows the
GEMM1 kernel body moving in the right direction:

| M | aiter GEMM1 | FlyDSL v7 GEMM1 | delta |
| ---: | ---: | ---: | ---: |
| 64 | 135.123 us | 141.395 us | +6.272 us |
| 128 | 172.911 us | 170.789 us | -2.122 us |

The M=128 profiler sample is noisy on the aiter side, but the static diff is
strictly closer to aiter and correctness is unchanged.  A long graph sample
with `--no-check` was noisy (`M=64` sort/GEMM stages all inflated together), so
the retained evidence for this small cleanup is correctness plus direct kernel
profiler/ISA alignment.

## Small-M BM16 GEMM2 v2 LDS Load Alignment

Change:

- Replace GEMM2's A path from `buffer_load_dwordx4` into VGPR followed by
  `ds_write*` with `raw_ptr_buffer_load_lds`, matching aiter's direct
  global-to-LDS load for the two K tiles.
- Keep the stronger pre-compute `s_waitcnt(0); s_barrier` ordering.  The earlier
  `lgkmcnt=0` relaxation was not correct.

Correctness sweep against aiter mxfp4:

```text
M=4   cos=0.999997437 max_abs=0.031250
M=8   cos=0.999998689 max_abs=0.015625
M=16  cos=0.999999166 max_abs=0.015625
M=32  cos=0.999998748 max_abs=0.031250
M=64  cos=0.999998808 max_abs=0.023438
M=128 cos=0.999999225 max_abs=0.031250
min_cos=0.999997437
```

Static ISA comparison:

| kernel | `s_waitcnt` | `buffer_load ... lds` | `ds_write_b128` | `ds_write_b32` |
| --- | ---: | ---: | ---: | ---: |
| aiter GEMM2 | 34 | 2 | 0 | 0 |
| FlyDSL GEMM2 v0 | 19 | 0 | 1 | 4 |
| FlyDSL GEMM2 v2 | 17 | 2 | 0 | 0 |

Short graph profiler (`warmup=20`, `graph_iters=20`, `replays=5`) for the
kernel body:

| M | aiter GEMM2 | FlyDSL v2 GEMM2 | delta |
| ---: | ---: | ---: | ---: |
| 64 | 67.528 us | 69.296 us | +1.768 us |
| 128 | 84.206 us | 88.799 us | +4.593 us |

Interpretation: v2 removes the obvious extra LDS writes and makes the static
A-load sequence closer to aiter.  The measured M=64 gap is slightly smaller
than the earlier `~3 us` snapshot, while M=128 still has a meaningful gap.  The
remaining GEMM2 difference is now more likely in the atomic epilogue form
(`buffer_atomic_pk_add_bf16` vs aiter's `global_atomic_pk_add_bf16`) and/or the
exact wait scheduling around the epilogue, rather than the A-load staging.

## Small-M BM16 Stable Bench / Profiler Defaults

`bench_small_bm16.py` now defaults to `repeat=5` on top of the long graph
timing inherited from `bench.py`:

```text
warmup=200 graph_iters=5000 measure=101 graph_warmup_replays=10 repeat=5
```

This is intentionally slow. A repeat-3 run before the default was raised showed
why repeated captures are needed: most samples are tight, but individual graph
captures can still produce outliers.

```text
/opt/venv/bin/python bench_small_bm16.py -M 4,8,16,32,64,128
```

| M | aiter | sort_aiter | gemm1fly_aiter | allfly | sort delta | GEMM1 delta | GEMM2 delta | cos all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 39.250 us | 40.235 us | 41.192 us | 42.650 us | +0.985 us | +0.957 us | +1.459 us | 0.999996483 |
| 8 | 58.433 us | 59.186 us | 59.749 us | 61.833 us | +0.753 us | +0.563 us | +2.083 us | 0.999998450 |
| 16 | 97.599 us | 97.425 us | 99.425 us | 100.636 us | -0.174 us | +2.000 us | +1.210 us | 0.999998927 |
| 32 | 146.151 us | 147.739 us | 150.295 us | 152.101 us | +1.588 us | +2.556 us | +1.806 us | 0.999998868 |
| 64 | 200.347 us | 204.580 us | 210.103 us | 210.362 us | +4.234 us | +5.522 us | +0.259 us | 0.999998212 |
| 128 | 251.843 us | 254.975 us | 261.167 us | 265.062 us | +3.132 us | +6.192 us | +3.895 us | 0.999999166 |

The long graph bench is the end-to-end timing source, but per-stage deltas are
still sensitive to adjacent runner noise. `profile_small_bm16.py` was added to
profile the graph replay directly as three kernel bodies per logical iteration:

```text
/opt/venv/bin/python profile_small_bm16.py -M 64,128
```

The profiler default is deliberately shorter than the bench default
(`warmup=20 graph_iters=20 replays=5 repeat=5`). Longer profiler windows caused
ROCProfiler event drops for `M=128`; the shorter window keeps the event sequence
complete while still averaging 100 logical iterations per sample.

## Small-M BM16 Sort v2 and GEMM2 v3

Sort changes:

- `flydsl_kimi_mxfp4_sort_zero_init_NE385_TOPK9_H7168_BM16_v2`
- Count phase now matches aiter's `int4` route-id load path.
- Place phase carries `token_id/topk_id` through the loop, avoiding per-route
  division/modulo and matching aiter's `place_tokens` update pattern.

GEMM2 change:

- `flydsl_kimi_mxfp4_gemm2_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT_v3`
- Replace `raw_ptr_buffer_atomic_fadd` with LLVM `atomicrmw fadd` on the global
  pointer, which lowers to `global_atomic_pk_add_bf16`.
- ISA check:

```text
v2: buffer_atomic_pk_add_bf16
v3: global_atomic_pk_add_bf16
aiter: global_atomic_pk_add_bf16
```

Correctness sweep:

```text
M=4   cos=0.999997497 max_abs=0.031250
M=8   cos=0.999997735 max_abs=0.031250
M=16  cos=0.999998033 max_abs=0.023438
M=32  cos=0.999998331 max_abs=0.031250
M=64  cos=0.999998808 max_abs=0.015625
M=128 cos=0.999999166 max_abs=0.031250
min_cos=0.999997497
```

Latest graph-profiler median over 5 samples:

| M | stage | aiter | allfly | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 4.945 us | 7.064 us | +2.118 us |
| 64 | GEMM1 | 134.625 us | 135.204 us | +0.579 us |
| 64 | GEMM2 | 66.341 us | 67.899 us | +1.558 us |
| 64 | total | 205.865 us | 210.209 us | +4.344 us |
| 128 | sort_zero_init | 6.339 us | 8.491 us | +2.152 us |
| 128 | GEMM1 | 164.029 us | 169.704 us | +5.674 us |
| 128 | GEMM2 | 81.698 us | 86.086 us | +4.388 us |
| 128 | total | 252.123 us | 264.297 us | +12.173 us |

Interpretation: v2/v3 are correct and statically closer to aiter. The sort
changes only give small profiler movement, so the remaining sort gap is probably
not just scalar route counting or div/mod. The global-atomic GEMM2 form matches
aiter ISA, but the measured GEMM2 gap remains meaningful, especially at `M=128`.
The next work should look at GEMM1/GEMM2 resource scheduling and exact wait/DS
traffic rather than only the atomic opcode form.

## Accuracy-First BM16 Measurement Defaults

The small-M BM16 timing scripts now default to longer graph measurements because
the kernels are short and launch/event overhead can otherwise move the conclusion:

```text
bench.py shared defaults:
  warmup=5000
  eager_iters=200000
  graph_iters=512
  measure=101
  graph_warmup_replays=20
  repeat=15

bench_small_bm16.py defaults:
  warmup=20000
  eager_iters=500000
  graph_iters=2048
  measure=101
  graph_warmup_replays=20
  repeat=21

profile_small_bm16.py defaults:
  warmup=3000
  graph_iters=128
  replays=4
  logical_iters_per_sample=512
  repeat=101
  max_retries=2000
```

Use the default commands for final comparison:

```text
/opt/venv/bin/python bench_small_bm16.py -M 4,8,16,32,64,128
/opt/venv/bin/python profile_small_bm16.py -M 64,128
```

For smoke checks only, override these values explicitly with small numbers.
The defaults are intentionally slower than smoke tests: for BM16 the end-to-end
bench measures `2048 * 101` graph calls per sample and reports 21 independent
graph captures.  The profiler uses `128 * 4` logical graph iterations per
sample and `repeat=101`.  This is large enough to stabilize small-kernel
medians while still finishing during iterative kernel work.

`bench.py` now also has a `--repeat` loop and prints the per-backend timing
samples. This makes the root sweep usable for final conclusions without relying
on a single CUDA-event median.

Two larger `profile_small_bm16.py` windows were rejected for default use:
`graph_iters=512, replays=8` repeatedly captured incomplete CUDA event streams
(`~10.8k` kernel events instead of the expected `12.3k` for M=64 aiter), and
`graph_iters=128, replays=4` again had near-zero acceptance in this environment.
`graph_iters=64, replays=2` accepted complete 3-kernel/iteration samples
reliably in the spot check, while `64,4` already showed substantial drops.

## BM16 GEMM2 v4 Packed-Multiply Trial

Tried replacing GEMM2 v3's epilogue route-weight multiply from two scalar
`v_mul_f32` instructions per bf16 pair to one inline-asm `v_pk_mul_f32`.

Correctness against aiter remained within the accepted cosine band:

```text
M=4   cos=0.999997199 max_abs=0.031250
M=8   cos=0.999997437 max_abs=0.031250
M=16  cos=0.999997318 max_abs=0.015625
M=32  cos=0.999998748 max_abs=0.031250
M=64  cos=0.999998152 max_abs=0.031250
M=128 cos=0.999999046 max_abs=0.023438
min_cos=0.999997199
```

ISA moved closer to aiter:

```text
v3: v_pk_mul_f32=0, v_mul_f32=16, s_waitcnt=17, VGPR=98, SGPR=34, spills=0
v4: v_pk_mul_f32=8, v_mul_f32=0,  s_waitcnt=19, VGPR=98, SGPR=34, spills=0
```

Graph-profiler medians with the longer default profiler window:

| M | stage | aiter | allfly v4 | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 4.907 us | 7.026 us | +2.119 us |
| 64 | GEMM1 | 130.719 us | 135.476 us | +4.757 us |
| 64 | GEMM2 | 65.340 us | 67.746 us | +2.405 us |
| 64 | total | 200.713 us | 210.226 us | +9.513 us |
| 128 | sort_zero_init | 6.248 us | 8.420 us | +2.172 us |
| 128 | GEMM1 | 163.551 us | 169.731 us | +6.180 us |
| 128 | GEMM2 | 81.690 us | 86.114 us | +4.423 us |
| 128 | total | 251.468 us | 264.259 us | +12.791 us |

Conclusion: v4 is correct and statically closer, but does not produce a
stable runtime win over v3. The two extra `s_waitcnt` instructions appear to
erase the scalar-multiply reduction. Reverted to GEMM2 v3 as the current stable
performance baseline.

Previous v3 baseline collected before the accuracy-first profiler defaults were
raised (`warmup=100`, `graph_iters=64`, `replays=16`, `repeat=9`):

| M | stage | aiter | allfly v3 | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 4.953 us | 7.057 us | +2.103 us |
| 64 | GEMM1 | 135.529 us | 136.310 us | +0.781 us |
| 64 | GEMM2 | 65.772 us | 68.188 us | +2.417 us |
| 64 | total | 207.244 us | 211.560 us | +4.316 us |
| 128 | sort_zero_init | 6.273 us | 8.423 us | +2.150 us |
| 128 | GEMM1 | 164.476 us | 170.508 us | +6.032 us |
| 128 | GEMM2 | 82.186 us | 86.617 us | +4.431 us |
| 128 | total | 252.823 us | 265.521 us | +12.698 us |

## BM16 Sort DPP Scan Trials

The FlyDSL sort v2 cumsum used DPP row shifts plus two `ds_bpermute` steps for
the 16/32-lane carry. Aiter's `dpp_inclusive_scan_wave` uses pure DPP for the
whole wave:

```text
row_shr 1/2/4/8, then update_dpp 0x142 row_mask=0xA, 0x143 row_mask=0xC
```

Two FlyDSL variants were tested:

- Sort v3: exact aiter-style wave scan and wave-total scan. Static ISA removed
  `ds_bpermute` and reduced `s_waitcnt` from 22 to 15, but added one extra
  `s_barrier`.
- Sort v4: pure-DPP wave scan while keeping v2's single-barrier cross-wave LDS
  accumulation.

Both variants remained correct:

```text
v3 min_cos=0.999997258
v4 min_cos=0.999997020
```

Profiler medians:

| M | variant | sort | total | result |
| ---: | --- | ---: | ---: | --- |
| 64 | v2 baseline | 7.057 us | 211.560 us | baseline |
| 64 | v3 exact wave_totals | 6.869 us | 211.030 us | small sort win |
| 64 | v4 DPP + old cross-wave | 6.954 us | 213.828 us | noisy total regression |
| 128 | v2 baseline | 8.423 us | 265.521 us | baseline |
| 128 | v3 exact wave_totals | 8.917 us | 266.945 us | regression |
| 128 | v4 DPP + old cross-wave | 8.552 us | 267.231 us | regression |

Conclusion: reducing `s_waitcnt` alone did not improve the stable small-M sort
path. The extra barrier in v3 is harmful at M=128, and v4 did not beat the v2
baseline. Keep sort v2 for now; the remaining sort gap likely needs ATT/Pmc
analysis rather than scan-shape tweaks.

## Small-M BM16 GEMM2 ATT Waitcnt Trial

Analyzed the existing FlyDSL GEMM2 v3 ATT trace with
`.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`. The trace is
dominated by VMEM waits rather than MFMA issue:

```text
VMEM-wait      ~5.01M stall cycles, 50.0%
VMEM-load      ~3.75M stall cycles, 37.4%
barrier        ~0.47M stall cycles,  4.7%
LDS/SMEM-wait  ~0.31M stall cycles,  3.1%
MFMA/FMA       ~0.08M stall cycles,  0.8%
```

Largest source-level hotspots:

| rank | source | dominant issue |
| ---: | --- | --- |
| 1 | GEMM2 epilogue sorted-token wait (`line ~1672`) | `s_waitcnt vmcnt(0)` |
| 2 | B tile loads (`mfma_preshuffle_pipeline.py:83`) | `buffer_load_dwordx4` |
| 3 | GEMM2 precompute wait (`line ~1637`) | full `s_waitcnt` before barrier |
| 4 | GEMM2 epilogue weight/LDS wait (`line ~1704`) | `s_waitcnt vmcnt(0) lgkmcnt(0)` |

Two partial-wait experiments were tried and rejected:

- v5/v6: replace the precompute full `s_waitcnt(0); s_barrier` with partial
  waits around `vmcnt(23/22)` and `vmcnt(20/19)`. Both produced finite outputs
  but poor cosine, so the waits were not strong enough for the current FlyDSL
  instruction ordering.
- v7: add aiter-style per-MFMA descending `vmcnt` waits while keeping the current
  FlyDSL compute loop shape. This also failed correctness. The aiter schedule
  interleaves LDS reads, barriers, and MFMA in a different order; the wait
  counts cannot be transplanted without matching that order.

Keep GEMM2 v3 for now. A correct GEMM2 waitcnt optimization needs a full
schedule rewrite to mirror aiter's LDS-read/MFMA sequence, not only local
wait-count edits.

## Small-M BM16 GEMM1 v8 Epilogue Amax Cleanup

Changed GEMM1 requant epilogue to match aiter's local amax pattern: initialize
`local_max` from `fabs(result[0])` and reduce only results 1..7, instead of
starting at zero and doing eight `fmax` operations. This is a
semantics-preserving cleanup of the epilogue dependency chain.

Correctness against aiter remains in the accepted band:

```text
M=4   cos=0.999996305 max_abs=0.015625
M=8   cos=0.999998391 max_abs=0.015625
M=16  cos=0.999998331 max_abs=0.031250
M=32  cos=0.999998212 max_abs=0.031250
M=64  cos=0.999998927 max_abs=0.023438
M=128 cos=0.999998927 max_abs=0.031250
min_cos=0.999996305
```

Graph-profiler check (`warmup=300`, `graph_iters=128`, `replays=4`,
`repeat=5`) showed a small positive movement:

| M | stage | aiter | allfly v8 | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 4.899 us | 7.091 us | +2.191 us |
| 64 | GEMM1 | 130.791 us | 135.658 us | +4.868 us |
| 64 | GEMM2 | 65.285 us | 67.807 us | +2.521 us |
| 64 | total | 200.889 us | 210.606 us | +9.717 us |
| 128 | sort_zero_init | 6.275 us | 8.453 us | +2.178 us |
| 128 | GEMM1 | 163.757 us | 169.965 us | +6.208 us |
| 128 | GEMM2 | 81.784 us | 86.257 us | +4.473 us |
| 128 | total | 251.823 us | 264.641 us | +12.818 us |

An allfly-only repeat-21 check had noisy M=64 outliers but stable medians:

| M | sort | GEMM1 v8 | GEMM2 | total |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 7.229 us | 135.824 us | 67.880 us | 210.958 us |
| 128 | 8.652 us | 170.064 us | 86.302 us | 265.016 us |

The improvement is small, but the code is cleaner and statically closer to the
aiter epilogue, so keep GEMM1 v8.

## Small-M BM16 Stable Profile Window and Rejected Follow-ups

The default `profile_small_bm16.py` window was adjusted after testing larger
graph-replay samples.  `graph_iters=512, replays=8` and then
`graph_iters=128, replays=4` repeatedly produced incomplete torch-profiler CUDA
event streams for M=64 aiter.  The retained default is a bounded window with
more independent samples:

```text
warmup=1000
graph_iters=64
replays=2
repeat=101
max_retries=1500
```

The retained window was later kept at `graph_iters=64, replays=2`, but the
default was raised to `warmup=2000`, `repeat=201`, and `max_retries=4000` so
the small-kernel profiler conclusions use more accepted graph samples without
reintroducing incomplete event streams from a larger single profiler window.

With a development repeat of 21, the current v8/v3 all-FlyDSL baseline remains:

| M | stage | aiter | allfly | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 4.912 us | 7.044 us | +2.132 us |
| 64 | GEMM1 v8 | 130.852 us | 135.523 us | +4.672 us |
| 64 | GEMM2 v3 | 65.364 us | 67.740 us | +2.376 us |
| 64 | total | 200.929 us | 210.284 us | +9.355 us |
| 128 | sort_zero_init | 6.291 us | 8.473 us | +2.183 us |
| 128 | GEMM1 v8 | 163.787 us | 169.849 us | +6.062 us |
| 128 | GEMM2 v3 | 81.691 us | 86.111 us | +4.420 us |
| 128 | total | 251.759 us | 264.474 us | +12.715 us |

Rejected follow-ups:

- GEMM1 early expert-load guard: moving the `sorted_expert_ids` load behind the
  `cumsum`/block-valid check was correct but did not improve performance. M=64
  GEMM1 measured `135.695 us`; M=128 measured `170.001 us`, both slightly worse
  than v8.
- GEMM2 LDS-value preload epilogue: reading all four bf16-pair LDS values before
  issuing atomics was correct but did not improve performance. M=64 GEMM2
  measured `67.868 us`; M=128 measured `86.163 us`, effectively equal or worse
  than v3.
- GEMM1 1D-grid remap: changing from `(num_n_blocks, m_blocks)` grid to aiter's
  one-dimensional `pid -> (m_block,n_block)` mapping was correct but slower.
  M=64 GEMM1 measured `136.435 us`; M=128 measured `170.354 us`.

Keep GEMM1 v8 and GEMM2 v3 as the stable BM16 kernel baselines.  The next
useful GEMM1 work likely needs deeper schedule/resource changes rather than
small launch or epilogue reshaping.

## BM16 GEMM1 v9 A-Read Hoist

Moved the main-loop A LDS reads for the BM16 inline-quant GEMM1 out of
`compute_tile_prefetch_b()` and into the loop body immediately after the LDS
barrier, before the inline hidden-state quant loads.  This matches the aiter
ordering more closely: barrier, A ds-read, A-scale ds-read, hidden load/quant,
then MFMA cluster.

Correctness against aiter remained within the accepted cosine band:

```text
M=4   cos=0.999997437 max_abs=0.015625
M=8   cos=0.999998808 max_abs=0.015625
M=16  cos=0.999998629 max_abs=0.015625
M=32  cos=0.999998689 max_abs=0.023438
M=64  cos=0.999998569 max_abs=0.031250
M=128 cos=0.999999046 max_abs=0.031250
```

Static GEMM1 ISA counts moved in the intended direction:

```text
                 aiter   FlyDSL v8   FlyDSL v9
s_waitcnt          126        150         126
v_mfma             224        224         224
buffer_load        336        336         336
DS read             88         88          88
DS write            92         92          92
s_barrier           30         30          30
s_setprio          208        208         208
```

Long graph-profiler check using the current default profiler settings
(`warmup=2000`, `graph_iters=64`, `replays=2`, `repeat=201`) measured:

| M | stage | aiter | allfly v9 | delta |
| ---: | --- | ---: | ---: | ---: |
| 64 | sort_zero_init | 5.157 us | 7.272 us | +2.115 us |
| 64 | GEMM1 v9 | 130.773 us | 135.489 us | +4.716 us |
| 64 | GEMM2 v3 | 65.227 us | 67.720 us | +2.493 us |
| 64 | total | 201.152 us | 210.503 us | +9.352 us |
| 128 | sort_zero_init | 6.627 us | 8.662 us | +2.035 us |
| 128 | GEMM1 v9 | 163.527 us | 169.310 us | +5.784 us |
| 128 | GEMM2 v3 | 81.570 us | 85.976 us | +4.405 us |
| 128 | total | 251.730 us | 263.970 us | +12.240 us |

Conclusion: v9 is retained.  It removes the static extra waitcnts and gives a
small, stable M=128 GEMM1 improvement, but the remaining GEMM1 gap is not simply
static waitcnt count.  The next target is the residual hot-loop wait placement
and dataflow: FlyDSL still has an early `s_waitcnt vmcnt(2)` between the first
pair of MFMA instructions in the first cluster, while aiter waits differently
before entering that pair.

Rejected follow-up:

- v10 inserted an explicit `rocdl.s_waitcnt(2)` at each MFMA cluster entry to
  pull the pair-internal VMEM wait forward.  Static ISA moved the wait before the
  two MFMA instructions, but total `s_waitcnt` increased from 126 to 151, so it
  was not retained.

## BM16 GEMM2 v8 J-Major MFMA Trial

Tried changing the FlyDSL BM16 GEMM2 compute order to mirror aiter's local
`issue_a_ds_read()` / `issue_mfma_cluster()` shape more closely.  The trial
preloaded both A LDS packs for a K-stage first, then iterated by output J so
each J consumed the two K half-tiles back-to-back:

```text
old v3: for k in [0,1]: ds_read A[k]; for J in [0..3]: mfma(J,k)
trial:  ds_read A[0]; ds_read A[1]; for J in [0..3]: mfma(J,0); mfma(J,1)
```

Correctness was within the accepted cosine band:

```text
M=4   cos=0.999997795 max_abs=0.015625
M=8   cos=0.999998868 max_abs=0.015625
M=16  cos=0.999998808 max_abs=0.015625
M=32  cos=0.999997795 max_abs=0.031250
M=64  cos=0.999998808 max_abs=0.031250
M=128 cos=0.999998569 max_abs=0.031250
min_cos=0.999997795
```

However, the graph-mode profiler did not show a runtime win.  With
`profile_small_bm16.py -M 64 --runners aiter,gemm1fly_aiter,allfly` and the
current long profiler defaults, the M=64 allfly trial measured:

| stage | aiter | allfly v8 trial | delta |
| --- | ---: | ---: | ---: |
| sort_zero_init | 6.632 us | 7.267 us | +0.635 us |
| GEMM1 v9 | 130.987 us | 135.174 us | +4.187 us |
| GEMM2 v8 trial | 65.428 us | 67.808 us | +2.380 us |
| total | 203.381 us | 210.253 us | +6.872 us |

This is effectively equal to the retained GEMM2 v3 baseline
(`~67.7 us` on M=64), so the j-major compute order was reverted.  The next
GEMM2 work still needs to address the full VMEM wait/barrier schedule rather
than only the local MFMA loop order.

## BM16 GEMM1 v11 Tail J-Major Trial

Tried applying the same J-major idea only to GEMM1's final two drain/tail
K-tiles: preload both A LDS packs with `load_a128_tile()`, then iterate by
output J and issue the two K-half MFMA instructions back-to-back.  This matched
the aiter source shape more closely for the tail path while leaving the main
pipeline unchanged.

Correctness stayed in the accepted band:

```text
M=4   cos=0.999997139 max_abs=0.015625
M=8   cos=0.999998391 max_abs=0.015625
M=16  cos=0.999998689 max_abs=0.023438
M=32  cos=0.999998629 max_abs=0.015625
M=64  cos=0.999998689 max_abs=0.031250
M=128 cos=0.999998927 max_abs=0.031250
min_cos=0.999997139
```

Graph-mode profiler checks with `repeat=101` measured:

| M | aiter GEMM1 | FlyDSL trial GEMM1 | delta |
| ---: | ---: | ---: | ---: |
| 64 | 130.333 us | 135.311 us | +4.977 us |
| 128 | 163.324 us | 169.223 us | +5.899 us |

The trial did not give a meaningful runtime win versus the retained v9 numbers.
The static ISA diff between v9 and the trial contained only the kernel symbol
rename; instruction counts, VGPR/SGPR usage, waits, MFMA count, and barriers
were identical:

```text
                 aiter   FlyDSL v9   FlyDSL trial
s_waitcnt          126        126          126
v_mfma             224        224          224
ds_read             94         88           88
ds_write            92         92           92
s_barrier           30         30           30
s_setprio          208        208          208
VGPR               139        140          140
SGPR                41         34           34
spills               0          0            0
```

Because the compiler already emitted the same ISA for the retained v9 schedule,
the source-only tail reorder was reverted.

## BM16 Sort v5 Normal Zero-Store Trial

Tried changing the BM16 fused sort/zero-init kernel from non-temporal int4
stores back to normal int4 stores for the bf16 atomic output zero-fill.  This
was motivated by the aiter C++ source using plain `out_v[i] = zero` in
`zero_init_bf16_out_impl`.

The graph-mode profiler did not show a win.  With
`profile_small_bm16.py -M 64,128 --runners aiter,sort_aiter` after increasing
the default profiler graph window to 128 captured calls and 4 graph replays per
sample, the no-NT v5 sort measured:

| M | aiter sort | FlyDSL sort v5 | sort delta | total delta |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 5.068 us | 7.270 us | +2.202 us | +2.175 us |
| 128 | 6.602 us | 8.742 us | +2.140 us | +2.889 us |

The normal store variant did not improve the retained v2 behavior, so v5 was
reverted to the v2 kernel name and non-temporal zero-fill store.

## BM16 Sort v6 Seven-Wave Cross-Prefix Trial

Tried reducing the retained v2 cross-wave LDS prefix loop from all 16 CTA waves
to only the `ceil(385 / 64) = 7` waves that can contain valid expert rows.  This
kept the v2 DPP plus `ds_bpermute` wave-local scan shape and did not add the
extra wave-total scan barrier from the rejected v3 trial.

Correctness stayed in the accepted cosine band:

```text
M=4   cos=0.999997497 max_abs=0.015625
M=8   cos=0.999997973 max_abs=0.023438
M=16  cos=0.999997675 max_abs=0.023438
M=32  cos=0.999998569 max_abs=0.015625
M=64  cos=0.999998808 max_abs=0.031250
M=128 cos=0.999998927 max_abs=0.023438
min_cos=0.999997497
```

The graph profiler did not show a runtime win.  On an otherwise idle GPU 3,
`profile_small_bm16.py -M 64 --runners aiter,sort_aiter` measured:

| stage | aiter | FlyDSL sort v6 path | delta |
| --- | ---: | ---: | ---: |
| sort_zero_init | 4.970 us | 7.451 us | +2.481 us |
| total | 201.295 us | 207.391 us | +6.096 us |

The v6 sort median was worse than the retained v2/v5 measurements, so it was
reverted.  The remaining sort gap is not fixed by simply trimming unused
cross-wave LDS reads.

## BM16 GEMM2 v9 Split-Barrier Trial

Tried replacing the retained GEMM2 v3 precompute `s_waitcnt(0); s_barrier()`
with aiter-style split barriers around the two A LDS slots:

```text
wait vmcnt(23); barrier; compute slot 0
wait vmcnt(22); barrier; compute slot 1
```

This matched the high-level atomic BM16 source shape more closely than the
earlier partial-wait-only attempts, but it was still not correct without
aiter's full per-MFMA descending VMEM waits and exact instruction schedule.

Correctness failed the accepted cosine band:

```text
M=4   cos=0.993439555 max_abs=3.984375
M=8   cos=0.993382812 max_abs=5.243408
M=16  cos=0.999997556 max_abs=0.054688
M=32  cos=0.994739234 max_abs=6.148438
M=64  cos=0.998132169 max_abs=5.007812
M=128 cos=0.997795224 max_abs=5.382812
min_cos=0.993382812
```

The trial was reverted.  This reinforces the prior ATT conclusion: GEMM2 wait
optimization has to transplant the full aiter schedule, including the
per-MFMA `vmcnt` ladder, rather than only splitting the coarse barrier.

## BM16 GEMM2 v10 A-Load Schedule Barrier Trial

Tried adding `rocdl.sched_barrier(0)` immediately after the two GEMM2 A
direct-to-LDS loads, matching the source-level aiter atomic BM16 sequence before
issuing scale and B loads.  This does not relax the retained full
`s_waitcnt(0); s_barrier()` correctness fence; it only nudges compiler
scheduling.

Correctness stayed in the accepted cosine band:

```text
M=4   cos=0.999996245 max_abs=0.031250
M=8   cos=0.999998748 max_abs=0.031250
M=16  cos=0.999998868 max_abs=0.015625
M=32  cos=0.999998331 max_abs=0.015625
M=64  cos=0.999999106 max_abs=0.031250
M=128 cos=0.999999225 max_abs=0.031250
min_cos=0.999996245
```

M=64 graph-profiler comparison with
`profile_small_bm16.py -M 64,128 --runners gemm1fly_aiter,allfly` was enough to
reject it:

| stage | gemm1fly_aiter | allfly v10 | delta |
| --- | ---: | ---: | ---: |
| sort_zero_init | 7.085 us | 7.079 us | -0.006 us |
| GEMM1 | 135.224 us | 135.091 us | -0.133 us |
| GEMM2 | 64.880 us | 67.731 us | +2.852 us |
| total | 207.179 us | 209.894 us | +2.715 us |

The GEMM2 time is effectively unchanged from retained v3 (`~67.72 us`), so the
schedule barrier was reverted.

## BM16 GEMM2 v11 B-Scale Cache Modifier

Changed the retained BM16 GEMM2 B-scale scalar load from non-temporal
`cache_modifier=2` to normal cached `cache_modifier=0`:

```text
flydsl_kimi_mxfp4_gemm2_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT_v11
```

This matches the aiter GEMM2 ISA shape more closely for the scale path: aiter's
B-scale `buffer_load_dword` does not carry the non-temporal modifier. Correctness
remained within the accepted cosine band for `M=4,8,16,32,64,128`
(`min_cos=0.9999963045120239`).

The profiler was also updated with a `by-kernel` aggregation mode. Ordered graph
event slicing requires every captured replay event to be present; on this ROCm
stack torch profiler can drop some CUDA events in large replay windows. The
`by-kernel` mode still profiles graph replay, but aggregates each kernel name's
captured per-call average, making the default long window usable for `M=128`.

Default `profile_small_bm16.py` settings after the update:

```text
warmup=3000
graph_iters=128
replays=4
repeat=101
logical_iters_per_runner=51712
aggregation=by-kernel
```

Graph-profiler comparisons against pure aiter:

| M | stage | aiter | allfly v11 | delta |
| ---: | --- | ---: | ---: | ---: |
| 4 | sort_zero_init | 4.289 us | 4.140 us | -0.149 us |
| 4 | GEMM1 v9 | 24.504 us | 25.984 us | +1.480 us |
| 4 | GEMM2 v11 | 10.949 us | 12.351 us | +1.402 us |
| 4 | total | 39.720 us | 42.469 us | +2.749 us |
| 16 | sort_zero_init | 4.530 us | 4.604 us | +0.075 us |
| 16 | GEMM1 v9 | 60.845 us | 63.897 us | +3.052 us |
| 16 | GEMM2 v11 | 30.970 us | 30.967 us | -0.003 us |
| 16 | total | 96.359 us | 99.477 us | +3.118 us |
| 64 | sort_zero_init | 4.956 us | 7.213 us | +2.257 us |
| 64 | GEMM1 v9 | 130.204 us | 135.310 us | +5.106 us |
| 64 | GEMM2 v11 | 65.075 us | 65.078 us | +0.003 us |
| 64 | total | 200.233 us | 207.638 us | +7.405 us |
| 128 | sort_zero_init | 8.422 us | 8.430 us | +0.007 us |
| 128 | GEMM1 v9 | 169.249 us | 169.200 us | -0.049 us |
| 128 | GEMM2 v11 | 82.199 us | 82.440 us | +0.241 us |
| 128 | total | 259.858 us | 260.048 us | +0.189 us |

Conclusion: keep GEMM2 v11.  It effectively closes the retained GEMM2 gap for
`M=16,64` and keeps `M=128` within `+0.241 us` on GEMM2.  The remaining BM16
work should start with `M=64` GEMM1, then the `M=64` sort gap.  `M=4` still has
a GEMM2 fixed-cost gap, but it is no longer the main throughput bucket.

## BM16 GEMM1 v12 B-Scale Cache Modifier

Changed the retained BM16 GEMM1 B-scale scalar load from non-temporal
`cache_modifier=2` to normal cached `cache_modifier=0`:

```text
flydsl_kimi_mxfp4_gemm1_NE385_H7168_E512_BM16_INLINEQUANT_v12
```

This mirrors the successful GEMM2 v11 change and matches aiter's B-scale load
shape more closely: the B data loads remain non-temporal, but the scale dword
loads use the normal cached path.

Short correctness smoke for `M=64` stayed in the accepted cosine band:

```text
cos_sort=0.999998689
cos_gemm1=0.999998927
cos_all=0.999998927
max_abs_all=0.031250
```

Default graph-profiler comparison:

```text
/opt/venv/bin/python profile_small_bm16.py -M 64 --runners sort_aiter,gemm1fly_aiter
```

| runner | sort_zero_init | GEMM1 | GEMM2 | total |
| --- | ---: | ---: | ---: | ---: |
| `sort_aiter` | 7.171 us | 131.178 us | 65.568 us | 203.946 us |
| `gemm1fly_aiter` | 7.131 us | 131.829 us | 65.520 us | 204.557 us |
| delta | -0.040 us | +0.651 us | -0.048 us | +0.611 us |

Conclusion: keep v12.  The previous retained M=64 GEMM1 gap was about
`+5.106 us`; v12 reduces the isolated GEMM1 gap to about `+0.651 us` in the
graph-profiler run.  The remaining M=64 work is now small enough that further
GEMM1 changes should be validated with the full default profile window.
