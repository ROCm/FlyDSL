# gfx950 / CDNA4 / MI355X — LLVM options

Load this when tuning specifically for gfx950. For the shared workflow, the six
hint keys, and the verification protocol, stay in `SKILL.md`.

**In one sentence:** gfx950 keeps gfx942's register file and occupancy model
unchanged and adds 2.5× LDS plus a large new instruction set — so port your
occupancy arithmetic from MI300 unchanged, but re-examine anything touching LDS
capacity or the new MFMA/conversion families.

Evidence: `[measured]` = ran `llc`; `[source]` = read the pinned LLVM
(`thirdparty/llvm-build-info.json`); `[doc]` = repeated from repo docs,
not re-measured.

## 1. What gfx950 does NOT change

This is the most common wrong assumption, so it comes first.

**The VGPR/AGPR register file is identical to gfx942.** [source] Both carry
`FeatureGFX90AInsts`, and every register-layout query keys off exactly that bit —
`getTotalNumVGPRs`, `getAddressableNumVGPRs`, `getAddressableNumArchVGPRs`,
`getVGPRAllocGranule`, `getVGPREncodingGranule` — **never** off
`FeatureGFX950Insts`.

| Property | gfx942 | gfx950 |
|---|---|---|
| Wave size | 64 | 64 |
| Vector file | unified 512 (256 arch + 256 AGPR) | **same** |
| VGPR alloc granule | 8 | **same** |
| MaxWavesPerEU | 8 | **same** |
| EUsPerCU | 4 | **same** |

So the `512 // (arch + accum)` occupancy rule carries over from MI300 unchanged.

## 2. The real capacity delta — LDS

[source] `FeatureAddressableLocalMemorySize163840`: **160 KB** per workgroup vs
gfx942's 64 KB. This is the one number that genuinely changes your tiling budget.
(gfx1250 has 320 KB, but in a wave32 world — see the other file.)

## 3. Occupancy and register knobs

| Attribute | Behaviour on gfx950 |
|---|---|
| `amdgpu-waves-per-eu="<min>[,<max>]"` | Works; max 8. [measured] `"8,8"` drove `num_vgpr` to 32. The second value is optional. |
| `amdgpu-flat-work-group-size="<min>,<max>"` | Works; **both values required**. Tune this *first* — its implied waves-per-EU bound wins over an explicit conflicting `waves-per-eu` (`AMDGPUSubtarget.cpp:193-201`). |
| `amdgpu-agpr-alloc="<min>[,<max>]"` | Splits the unified 512 budget. If unset and AGPRs are needed at all, defaults to a **256/256** split; an entry function with no calls and no AGPR use can take all 512 for VGPRs. |
| `amdgpu-num-vgpr="<N>"` | ⚠ **Silently doubled** — see below. Also marked **Deprecated** in `AMDGPUUsage.rst:2534` in favour of `waves-per-eu`. |

### ⚠ `amdgpu-num-vgpr` is silently doubled

[source] `GCNSubtarget.cpp:624-626`:
`if (Requested != Max && hasGFX90AInsts()) Requested *= 2;`

[measured] Same IR, `"amdgpu-num-vgpr"="64"`:

| arch | num_vgpr | num_agpr | total |
|---|---|---|---|
| gfx1030 | 64 | 0 | 64 |
| **gfx950** | 64 | **64** | **128** |
| gfx1250 | 64 | 0 | 64 |

The number you write is a **combined VGPR+AGPR budget**, not an arch-VGPR count.
[measured] that same run reported `Occupancy 4` with **74 VGPR spills** — the
mechanism behind the ~4.5× regression recorded in
`docs/kernel_tuning_guide.md:448-449` **[doc]**. Confirm against the emitted
`.vgpr_count` or a `kernel-resource-usage` remark, never against the number you
requested.

⚠ Invalid values **fail silently and differently per attribute**. [measured] a
`flat-work-group-size` of `"512,128"` or `"1,99999"` is discarded wholesale —
reverting to the default `1024` — with **no diagnostic**.

## 4. gfx950-relevant `-mllvm` flags

All are cl::opts usable through `llvm_options`. Check existence first with
`llvm_knob_check.py`.

| Flag | Default | Use |
|---|---|---|
| `-amdgpu-spill-vgpr-to-agpr` | true | Lets the allocator spill into idle AGPRs instead of scratch — meaningful precisely *because* of the unified file. Turning it off forces scratch. |
| `-amdgpu-mfma-vgpr-form` | true | Force VGPR (not AGPR) for MFMA Opc/Dest. |
| `-amdgpu-mfma-padding-ratio=<pct>` | 0 | Fills a percentage of inter-MFMA latency with `s_nop`s. A blunt probe for whether a kernel is MFMA-latency-bound. |
| `-amdgpu-snop-padding` | — | Related s_nop padding control. |

## 5. Hazards that cost cycles here

[source] Two predicates exist **only** on gfx950:

- `hasCvtScaleForwardingHazard()` — results of the new `v_cvt_scalef32_*`
  instructions cannot be forwarded without a wait/nop. Relevant to any mxfp
  quantisation epilogue.
- `hasLoopHeadInstSplitSensitivity()` — the source comment is explicit that this
  is a **gfx950 performance pathology**, not a correctness issue.

The hazard recognizer also encodes gfx950-specific XDL/DMFMA wait states,
generally **one more cycle** than gfx942 for most pass counts.

## 6. Instruction families worth targeting

| Family | Notes | Intrinsic / mnemonic |
|---|---|---|
| LDS transpose loads | ⚠ **wave64 only** | `llvm.amdgcn.ds.read.tr{4.b64,6.b96,8.b64,16.b64}` → `ds_read_b64_tr_b4/b8/b16`, `ds_read_b96_tr_b6` |
| Widened LDS-direct loads | 96/128-bit; gfx942 tops out at dword | `global_load_lds_dwordx3/x4`, `buffer_load_dwordx3/x4` with the lds bit; query `hasLDSLoadB96_B128()` |
| Dense MFMA, 2× K | 6 new shapes | `v_mfma_f32_16x16x32_f16`, `32x32x16_f16`, the bf16 pair, `i32_16x16x64_i8`, `i32_32x32x32_i8` |
| mxfp MFMA | 9 source-format combos under one mnemonic | `v_mfma_f32_16x16x128_f8f6f4`, `32x32x64_f8f6f4` |
| Scaled mxfp MFMA | block E8M0 scales, VOP3PX encoding | `v_mfma_scale_f32_*_f8f6f4`, fed by `v_mfma_ld_scale_b32` |
| Sparse MFMA | 14 new SMFMAC shapes | `v_smfmac_f32_16x16x64_f16`, … |
| Conversions | ~49 `v_cvt_scalef32_*` for MX types | gated by the six `*-cvt-scale-insts` features |
| Misc | `v_prng_b32`, `v_bitop3_b16/b32`, `v_ashr_pk_i8/u8_i32`, `v_permlane16/32_swap_b32`, `v_cvt_f32_bf16`, `v_cvt_pk_f16_f32` | |

⚠ Only the `_f8_f8` form of the F8F6F4 MFMAs is disassemblable; the other eight
are `isAsmParserOnly`. Mixed-format MFMAs will **not round-trip** through the
disassembler when you diff ISA.

## 7. ⚠ gfx950 is not a pure superset of gfx942

[source] Two removals:

- **`xf32-insts` is gone.** `v_mfma_f32_16x16x8xf32` and `v_mfma_f32_32x32x4xf32`
  do not exist on gfx950, enforced by an explicit `if (Kind != GK_GFX950)` guard
  in the TargetParser. A gfx942 kernel using XF32 MFMAs will not port.
- `cvt-fp8-vop1-bug` is absent — a gfx942 erratum that gfx950 **fixes**.

## 8. Target-ID and `-mattr` traps

- ⚠ **`-mattr=+sramecc` / `+xnack` is now a hard error** in this LLVM revision:
  *"xnack/sramecc should be specified via module flags. Use module flag
  amdgpu.sramecc instead of subtarget feature"* (`AMDGPUAsmPrinter.cpp:1199`).
  Older LLVM accepted it. [source]
- ⚠ **`llc -mcpu=gfx950:sramecc+` silently produces an empty arch name**
  (`.amdgcn_target "amdgcn-amd-amdhsa-unknown-"`) rather than erroring. More
  dangerous than a typo: `gfx9999` at least reports "not a recognized processor".
  Target-ID syntax belongs to **clang's** `-mcpu`/`--offload-arch`, not `llc`.
- Sub-features of `gfx950-insts` (permlane swaps, ashr-pk, the cvt-scale
  families) are **separate** features that the umbrella merely implies, so
  `-mattr=-gfx950-insts` does not necessarily remove them.

## 9. Empirical priors from this tree **[doc]**

`waves_per_eu` on gfx950 kernels: `2` is the house default. GEMMs key it to wave
count (4-wave/256-thread → 1, 8-wave/512-thread → 2) and emit it as a **matched
pair** with `rocdl.flat_work_group_size`. One documented outlier:
`waves_per_eu=8` in `qk_norm_rope_quant.py:651-658`, from a measured MI355X sweep
on a memory-bound kernel.

⚠ Three sibling gfx950 dualwave FMHA kernels **disagree** on `enable-post-misched`
(two `False`, one `True`, no comment). Do not generalise a family setting.

⚠ Benchmark noise on gfx950 reaches **±14%** from clock variation. Run isolated
(60–80 iterations), take a median-of-7, and treat smaller wins as noise.
