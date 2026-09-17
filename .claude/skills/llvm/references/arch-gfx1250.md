# gfx1250 — LLVM options

Load this when tuning specifically for gfx1250. For the shared workflow, the six
hint keys, and the verification protocol, stay in `SKILL.md`.

**In one sentence:** gfx1250 is wave32 with no AGPRs, 1024 addressable VGPRs and
twice CDNA's occupancy ceiling — so almost every occupancy number you have from
MI300/MI355 means something different here, and two of its failure modes are
silent.

Evidence: `[measured]` = ran `llc` built from a **newer** tree than the FlyDSL
pin; behaviours were cross-checked against the pin and agree, but flag *sets*
could differ. `[source]` = read the pinned LLVM.

## 1. ⚠⚠ `-mattr=+wavefrontsize64` produces an EMPTY object

The worst failure mode on this arch. [measured, reproduced] Same IR:

| arch | instructions emitted | exit code |
|---|---|---|
| gfx1200 | 945 | 0 |
| gfx950 | 878 | 0 |
| **gfx1250** | **0** | **0** |

`amdhsa.kernels: []`, every `amdgpu.max_num_*` zero, **no diagnostic at all**.
A build "succeeds" and ships nothing.

[source] `supportsWave64()` is false for gfx1250 (`GCNSubtarget.h:912`) and
`GCNSubtarget.cpp:115-123` unsets the default wave32, but the only wavesize
diagnostic (`:203-210`) covers just the both-set case. **Never pass
`+wavefrontsize64` here**, and if a build script sets it globally, special-case
gfx1250.

## 2. ⚠⚠ The legacy `s_waitcnt` intrinsic aborts the compiler

[measured, reproduced] `llvm.amdgcn.s.waitcnt(i32 0)`:

- gfx950 → `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)`, rc=0
- **gfx1250 → `LLVM ERROR: Cannot select`, rc=134, core dumped**

Same for `llvm.amdgcn.s.waitcnt.depctr`. The packed bitfield has no gfx12
encoding.

**This vindicates FlyDSL's design.** `fx.rocdl.s_waitcnt` deliberately raises a
Python error on gfx1250 (`universal.py:53-55`) instead of emitting the op.
Raising early is strictly better than an ISel crash — **do not "fix" it**. Use
the split counters in §6.

## 3. Structural differences from CDNA

| Property | gfx950 | gfx1250 |
|---|---|---|
| Wave size | 64 | **32** |
| AGPRs | 256 | **none** |
| Addressable VGPRs | 256 arch (512 unified) | **1024** |
| Total VGPRs | 512 | 1024 (no headroom) |
| VGPR alloc granule | 8 | **16** |
| MaxWavesPerEU | 8 | **16** |
| EUsPerCU | 4 | 4 |
| Max waves/CU | 32 | **64** |
| LDS | 160 KB | **320 KB** |
| `Feature1536VGPRs` | n/a | **not set** |

[source] There is no `FeatureMAIInsts`/`FeatureGFX90AInsts` in any gfx125x set, so
the `a` inline-asm constraint fails outright — [measured] *"could not allocate
output register for constraint 'a'"*, where gfx950 succeeds.

⚠ The CDNA `512 // (arch + accum)` occupancy rule **must not be reused here**.

## 4. ⚠ Porting: the same number means something different

**`amdgpu-num-vgpr` is NOT doubled here.** [source] the `Requested *= 2` at
`GCNSubtarget.cpp:625-626` is gated on `hasGFX90AInsts()`, which gfx1250 lacks.

[measured] side by side:

| request | gfx950 gets | gfx1250 gets |
|---|---|---|
| 16 | **32** | 16 |
| 24 | **36** | 24 |
| 64 | **128** | 64 |

⚠ **Do not halve your gfx942/gfx950 `amdgpu-num-vgpr` values when retargeting.**

**`waves_per_eu` means a different fraction of peak.** Max occupancy is 8 on CDNA
and **16** here, so `waves_per_eu=2` requests 1/4 of peak on CDNA but **1/8** on
gfx1250. Every `waves_per_eu` value in this repo was tuned for CDNA.

[measured] sweep on a register-hungry kernel: `1/4/8` → 101 VGPR, occ 9;
**`12` → 80 VGPR with spills, occ 12; `16` → 64 VGPR, occ 16.**

## 5. Knobs: what works, what is inert

**Work** [measured]: `amdgpu-num-vgpr` (a real clamp), `amdgpu-waves-per-eu`,
`amdgpu-flat-work-group-size`, `amdgpu-num-sgpr`.

**Silently do nothing:**

- ⚠ **`amdgpu-agpr-alloc` is inert.** [source] parsed only under
  `hasGFX90AInsts()`/`hasMAIInsts()` (`GCNSubtarget.cpp:657,699`), backstopped by
  `if (!ST.hasMAIInsts()) MaxNumAGPRs = 0;` (`SIRegisterInfo.cpp:715-717`).
  [measured] setting it leaves `max_num_agpr 0` and returns rc=0.
  **AMDGPUAttributor may still attach it to gfx1250 IR — never read it as
  evidence of AGPR usage.**
- ⚠ **Out-of-range `waves-per-eu` / `flat-work-group-size` are discarded
  wholesale**, not clamped (`AMDGPUSubtarget.cpp:156-176,179-203`). On gfx1250 any
  max > 16 is thrown away with no diagnostic.
- ⚠ **Target-ID is rejected with rc=0.** `gfx1250:xnack+`, `:xnack-`,
  `:sramecc+` all report *"not a recognized processor … (ignoring processor)"* but
  **exit 0** and compile against a default subtarget. gfx1250 has no xnack/sramecc
  variants. `gfx12-5-generic` is valid.
- `amdgpu-dynamic-vgpr-block-size` is accepted and shifts occupancy (9/9/8) but
  [measured] did **not** cut VGPRs, and its LLVM test coverage is amdpal/gfx1200
  only. **Unconfirmed on this hardware** — confirm against the ISA docs first.

## 6. Split wait counters

gfx12 replaced the packed bitfield with per-class counters. All [measured]
working: `llvm.amdgcn.s.wait.{loadcnt,storecnt,dscnt,kmcnt,expcnt,samplecnt,bvhcnt}`
(`IntrinsicsAMDGPU.td:397-403`, i16 argument) → `s_wait_loadcnt`,
`s_wait_dscnt`, and the fused `s_wait_loadcnt_dscnt`.

gfx1250-only additions: `llvm.amdgcn.s.wait.asynccnt` and `.s.wait.tensorcnt`
(`:3788-3790`), pairing with the async-copy and TDM families below.

`s_wait_xcnt` is **compiler-managed and has no user intrinsic** — [measured] it
was auto-emitted 24× in the probe kernel.

## 7. gfx1250-relevant flags

Of 163 AMDGPU `cl::opt`s, **exactly one** names gfx12 in its description.

| Flag | Notes |
|---|---|
| `-amdgpu-expert-scheduling-mode` | Gate is `getGeneration() >= GFX12` (`GCNSubtarget.h:684`) — **gfx12+ generally, not gfx1250-only**. Emits `s_setreg hwreg(HW_REG_WAVE_SCHED_MODE,0,2), 2`, models 3 extra counters, splits VALU into XDL/TRANS/DPMACC/CSMACC classes. ⚠ [measured] it is **additive**: waitcnt count rose 20→23 and it added `s_wait_alu depctr_va_vdst`. Not a free win — it genuinely puts the wave in hardware SCHED_MODE 2. The attribute form exists too; the flag overrides it module-wide. |
| `-amdgpu-sched-strategy=coexec` | Intended for gfx1250. ⚠ On other targets it emits a `DS_Warning` **and runs anyway** — [measured] on gfx950 it changed register allocation (`v[8:11]`→`v[60:63]`). `overrideSchedPolicy` forces top-down with **no target guard at all**. Never include it in a CDNA sweep. ⚠ The **attribute form wins over the flag** (`AMDGPUTargetMachine.cpp:602-611`). |
| `-amdgpu-wmma-vnop-hoisting` | default true; effectively gfx1250-only (reachable only via `fixWMMACoexecutionHazards`), inert elsewhere. |
| `-amdgpu-barrier-signal-wait-latency` | default 16; gfx12+, fires on split barriers and the TDM/`TENSOR_CNT` paths. |

[source] **No `cl::opt` exists at all** for dynamic VGPR, asynccnt, tensorcnt,
xcnt, TDM or clusters — those are feature/attribute-driven only.

## 8. Instruction families

⚠ **Arity reads exactly like absence.** Two "unsupported instruction" results
during this survey turned out to be malformed signatures. Check the declared
arity and overload types before concluding an intrinsic is missing.

| Family | Intrinsic | Notes |
|---|---|---|
| WMMA | `llvm.amdgcn.wmma.*` | f64/f32/f16/bf16/fp8/bf8/iu8 shapes incl. 16x16x128 |
| Scaled WMMA (mxfp) | `wmma.scale{,16}.f32.16x16x128.f8f6f4` | ⚠ **14 arguments** |
| SWMMAC | `swmmac.f32.16x16x64.{f16,bf16}` | |
| TDM | `llvm.amdgcn.tensor.load.to.lds` / `.tensor.store.from.lds` | ⚠ **6 arguments** (4 descriptor groups + a reserved `<8 x i32>` zeroinit + cachepolicy immarg). Pairs with `s_wait_tensorcnt` |
| Async global→LDS | `global.load.async.to.lds.{b8,b32,b64,b128}`, cluster forms | Pairs with `s_wait_asynccnt` |
| Prefetch | `llvm.amdgcn.global.prefetch` | → `global_prefetch_b8 ... scope:SCOPE_SE` |
| Cluster | `cluster.id`, `cluster.workgroup.id`, `s.cluster.barrier` | ⚠ `s.cluster.barrier` lowers to `s_barrier_signal_isfirst`/`s_barrier_wait`, not one named op |
| Transposes | ⚠ **`ds.load.tr{4.b64,6.b96,8.b64,16.b128}`** | gfx950 spells these **`ds.read.tr*`** — different intrinsics that hard-error on each other's arch. `tr16.b128` takes `v8i16/v8f16/v8bf16`, **not** `v4i32` |
| permlane | `llvm.amdgcn.permlane16.swap` | shared with gfx950 |

## 9. Further traps

- ⚠ **Silent instruction drop.** `rocdl.global_prefetch` lowers to
  `llvm.amdgcn.global.prefetch`, which needs the gfx1250 `global_prefetch_b8` ISel
  pattern. If the LLVM build lacks it, the instruction is **silently dropped with
  no diagnostic** (`tdm_ops.py:1263-1265`) — the same failure class that got
  `BufferCopyLDS64b` deprecated. Always assert the mnemonic in `*_final_isa.s`.
- ⚠ **No arch guards in FlyDSL's ROCDL layer.** `cdna5.py`, `cluster.py`,
  `inline_asm.py` and the wildcard-exported gfx1250 ops perform **no** arch check,
  so a wrong-arch call fails late at ISel or in the assembler, never as a clean
  Python error.
- ⚠ **`gfx1250-insts` is not "is gfx1250".** It is also set on gfx1251,
  gfx12-5-generic **and gfx13**; `isGFX1250()` excludes gfx13 explicitly
  (`AMDGPUBaseInfo.cpp:2591-2593`).
- `cdna3.py` serves **both** gfx942 and gfx950 for waitcnt encoding, while
  `cdna4.py` holds gfx950-only atoms and `cdna5.py` holds gfx1250 — the file's
  generation number does not track the target arch.
