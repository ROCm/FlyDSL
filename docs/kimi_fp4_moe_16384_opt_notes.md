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
